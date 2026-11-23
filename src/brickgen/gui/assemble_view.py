'''
File:   assemble_view.py
Brief:  Assemble-tab 3D viewport: free piece placement, collision feedback, magnets.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

from collections.abc import Callable

import pyvista as pv
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QWidget

from brickgen.gui.preview import (
    RUBBER_BAND_THRESHOLD_PX,
    SELECTION_COLOR,
    PreviewWidget,
    assemble_variant_mesh,
)
from brickgen.services.assemble_service import (
    Rect,
    Support,
    assemble_pose_collides,
    assemble_snap_target,
    offset_rects,
    transform_footprint_rects,
)

# Ghost/outline tint while the current pose intersects another piece.
COLLISION_COLOR = "#d02020"

# Opacity of the cursor-attached ghost piece while placing from the panel.
GHOST_OPACITY = 0.6


class AssembleView(PreviewWidget):
    '''Assemble-tab viewport: free 3D piece placement with collision feedback.

    Pieces are ordinary scene blocks for the base :class:`PreviewWidget`
    (their meta carries ``pos_z`` and the piece orientation ``rot_z`` so
    actors stack above the floor plane). On top of the base navigation/
    selection machinery this subclass adds:

    - panel-driven free 3D placement: a translucent ghost piece attaches to
      the cursor (like the G grab), follows the working plane controlled by
      the mouse wheel, and every left click drops a copy (R or Space
      rotates 90 degrees, F flips, Esc/right-click exits the mode);
    - collision prohibition: ``assemble_pose_collides`` blocks drops that
      intersect existing pieces; outlines tint to :data:`COLLISION_COLOR`;
    - magnets from service candidates: stack on ``stack_top`` at any height,
      side-flush at support bottom/top planes;
    - Space during a move drag / G-grab rotates every moved piece 90
      degrees about Z around its own center (height stays as-is);
    - the Del hotkey deletes the selected pieces;
    - paste-ghost reuses :meth:`start_placement` to continue placement at
      the copied pose.

    ``blocks_moved`` payloads are ``{index: (x, y, z, rot_z)}`` here (the
    base planning/placement tabs keep ``(x, y)``).
    '''

    # Emitted on every ghost drop: dict with keys ``block`` (block index),
    # ``x``, ``y``, ``z`` (drop position of the footprint center / bottom),
    # ``rot_z`` (quarter turns) and ``flipped``.
    piece_placed = Signal(object)
    # Emitted when the user deletes pieces (Del hotkey).
    delete_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the assemble viewport (empty scene, floor grid).'''
        super().__init__(parent=parent)
        self._snap_parts = True
        # Supports query: window callback returning the world footprints of
        # the placed pieces, excluding the given piece uids.
        self._supports_provider: Callable[[set[str]], list[Support]] | None = None
        # Supports frozen at drag/placement start (must not follow the moved
        # pieces mid-drag).
        self._drag_supports: list[Support] = []
        # --- Panel-driven placement (ghost) state ---
        self._placing = False
        self._ghost_base: pv.PolyData | None = None
        self._ghost_actor: pv.Actor | None = None
        self._ghost_block = -1
        self._ghost_block_height = 0.0
        # Ghost material rectangles: base pose (rot 0, unflipped) and the
        # current pose (rebuilt on R/Space/F) in piece-relative coords.
        self._ghost_base_rects: tuple[Rect, ...] = ()
        self._ghost_rects: tuple[Rect, ...] = ()
        self._ghost_height = 0.0
        self._ghost_z = 0.0
        self._ghost_rot = 0
        self._ghost_flip = False
        self._ghost_color = "#ffffff"
        self._ghost_pos: tuple[float, float, float] | None = None
        # --- Drag state ---
        self._grab_orig_z: dict[int, float] = {}
        self._drag_dz = 0.0
        self._applied_dz = 0.0
        self._drag_valid_pose: dict[int, tuple[float, float, float, int]] = {}
        self._grab_start_pose: dict[int, tuple[float, float, float, int]] = {}
        self._rot_total = 0  # quarter turns applied during the active drag
        self._rot_undo = 0   # quarter turns applied after the last valid snapshot

    # ------------------------------------------------------------------ API

    def set_snap_parts(self, enabled: bool) -> None:
        '''Enable magnetic snapping to other pieces (stacking/side-flush).

        Args:
            enabled: True aligns drops and drags onto supports.
        '''
        self._snap_parts = bool(enabled)

    def set_supports_provider(
        self, provider: Callable[[set[str]], list[Support]]
    ) -> None:
        '''Install the callback returning the placed-piece footprints.

        Args:
            provider: Callable mapping a set of piece uids to exclude into
                the :class:`Support` list of the remaining pieces.
        '''
        self._supports_provider = provider

    def start_placement(
        self,
        mesh: pv.PolyData,
        block_index: int,
        rects: tuple[Rect, ...],
        height: float,
        color: str,
        block_height: float = 0.0,
        rot_z: int = 0,
        flipped: bool = False,
        z: float = 0.0,
    ) -> bool:
        '''Attach a ghost copy of *mesh* to the cursor (panel-driven).

        The ghost follows the working plane controlled by the mouse wheel
        with the current snapping until Esc or a right click cancels; every
        left click drops a copy (the ghost stays for the next drop).

        Args:
            mesh: Untransformed block mesh (centered at origin).
            block_index: Scene index of the source block (reported on drop).
            rects: Piece-relative material rectangles of the block in its
                base orientation (excluded cells leave voids that pieces
                may enter).
            height: Block total height (flip lift), mm.
            color: Ghost surface color.
            block_height: Block stud-plane offset (height from bottom to the
                top of the highest stud, used for collision checks).
            rot_z: Initial rotation in quarter-turns (0-3).
            flipped: Whether the piece is flipped (upside-down).
            z: Initial working-plane height (piece bottom z).

        Returns:
            True when the placement mode was entered.
        '''
        if self._placing:
            self.cancel_placement()
        if self._grabbing:
            self.cancel_grab()
        self._placing = True
        self._ghost_base = mesh
        self._ghost_block = block_index
        self._ghost_base_rects = tuple(rects)
        self._ghost_height = float(height)
        self._ghost_block_height = float(block_height)
        self._ghost_rot = int(rot_z) % 4
        self._ghost_flip = bool(flipped)
        self._ghost_z = max(0.0, float(z))
        self._ghost_color = color
        self._refresh_ghost_rects()
        self._drag_supports = self._query_supports(set())
        self._build_ghost_actor()
        self.setCursor(Qt.CursorShape.CrossCursor)
        # Placement starts from a press on a panel card, so the viewport
        # must take the keyboard focus itself: without it R/Space/F stay
        # dead until the first drop click focuses the view.
        self.setFocus()
        # Seed the ghost from the cursor only when it is inside the
        # viewport: the mode normally starts over the panel, where the
        # extrapolated floor point would park the ghost off-scene.
        cursor = self.mapFromGlobal(QCursor.pos())
        if self.rect().contains(cursor):
            self._update_ghost(cursor)
        return True

    def cancel_placement(self) -> None:
        '''Leave the ghost placement mode and remove the ghost actor.'''
        if not self._placing:
            return
        self._placing = False
        if self._ghost_actor is not None:
            try:
                self.remove_actor(self._ghost_actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._ghost_actor = None
        self._ghost_base = None
        self._ghost_base_rects = ()
        self._ghost_rects = ()
        self._drag_supports = []
        self._ghost_pos = None
        self.unsetCursor()
        self.render()

    def placing(self) -> bool:
        '''True while a ghost piece follows the cursor.'''
        return self._placing

    # ------------------------------------------------------- drag/ghost glue

    def _query_supports(self, exclude_uids: set[str]) -> list[Support]:
        '''Supports of the placed pieces via the window provider.'''
        if self._supports_provider is None or not self._snap_parts:
            return []
        return list(self._supports_provider(exclude_uids))

    def _refresh_ghost_rects(self) -> None:
        '''Material rectangles of the ghost for its current pose.

        Re-applies the base-orientation rectangles for the current R/Space
        rotation and F flip, mirroring the mesh variant transform order.
        '''
        self._ghost_rects = transform_footprint_rects(
            self._ghost_base_rects, self._ghost_rot, self._ghost_flip
        )

    def _build_ghost_actor(self) -> None:
        '''(Re)build the translucent ghost actor for the current pose.'''
        if self._ghost_actor is not None:
            try:
                self.remove_actor(self._ghost_actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
            self._ghost_actor = None
        if self._ghost_base is None:
            return
        variant = assemble_variant_mesh(
            self._ghost_base, self._ghost_rot, self._ghost_flip, self._ghost_height
        )
        self._ghost_actor = self.add_mesh(
            variant,
            color=self._ghost_color,
            opacity=GHOST_OPACITY,
            smooth_shading=False,
            lighting="three lights",  # type: ignore[arg-type]  # pyvista stub bug
            pickable=False,
            reset_camera=False,
            render=False,
        )
        self._ghost_actor.prop.color = self._ghost_color
        if self._ghost_pos is not None:
            self._ghost_actor.SetPosition(*self._ghost_pos)

    def _update_ghost(self, pos: QPoint) -> None:
        '''Move the ghost to the snapped drop position under the cursor.'''
        if not self._placing or self._ghost_actor is None:
            return
        world = self._cursor_on_plane(pos, self._ghost_z)
        if world is None:
            return
        x, y, z = assemble_snap_target(
            float(world[0]), float(world[1]), self._ghost_z,
            self._ghost_rects, self._ghost_height, self._ghost_block_height,
            self._ghost_flip, self._drag_supports, self.grid_step(),
            self._grid_snap, self._snap_parts,
        )
        self._ghost_pos = (x, y, z)
        self._ghost_actor.SetPosition(x, y, z)
        self._tint_ghost(not self._ghost_valid_at(x, y, z))
        self.render()

    def _drop_ghost(self) -> None:
        '''Drop one copy at the current ghost pose (ghost stays active).'''
        if not self._placing or self._ghost_actor is None:
            return
        if self._ghost_pos is not None:
            x, y, z = self._ghost_pos
        else:
            position = self._ghost_actor.GetPosition()
            x, y, z = float(position[0]), float(position[1]), float(position[2])
        if not self._ghost_valid_at(x, y, z):
            # Blocked drop: make the refusal visible even when the ghost
            # has not followed the cursor into this pose yet.
            self._tint_ghost(True)
            return
        self.piece_placed.emit(
            {
                'block': self._ghost_block,
                'x': x,
                'y': y,
                'z': z,
                'rot_z': self._ghost_rot,
                'flipped': self._ghost_flip,
            }
        )
        # The window committed the new piece (direct Qt connection): refresh
        # the support set so the next drop stacks on it.
        self._drag_supports = self._query_supports(set())

    # ------------------------------------------------------- ghost pose helpers

    def _ghost_stack_offset(self) -> float:
        '''Stud plane offset of the ghost in its current orientation.'''
        return self._ghost_height if self._ghost_flip else self._ghost_block_height

    def _ghost_valid_at(self, x: float, y: float, z: float) -> bool:
        '''False when the ghost pose at (x, y, z) intersects a piece.'''
        return not assemble_pose_collides(
            self._drag_supports,
            offset_rects(self._ghost_rects, x, y),
            z,
            self._ghost_height,
            self._ghost_stack_offset(),
        )

    def _tint_ghost(self, invalid: bool) -> None:
        '''Red ghost while blocked, part color again when legal.'''
        if self._ghost_actor is not None:
            color = COLLISION_COLOR if invalid else self._ghost_color
            self._ghost_actor.prop.color = color

    # --------------------------------------------------------- input routing

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt naming
        '''Wheel steps the working plane during placement and the drag height
        during a move; otherwise native zoom.'''
        if self._placing:
            delta = event.angleDelta().y()
            if delta > 0:
                self._ghost_z += self.grid_step()
            elif delta < 0:
                self._ghost_z = max(0.0, self._ghost_z - self.grid_step())
            self._update_ghost(event.position().toPoint())
            event.accept()
            return
        if self._grabbing and self._grab_orig:
            self._adjust_drag_height(event.angleDelta().y(), event.position().toPoint())
            event.accept()
            return
        super().wheelEvent(event)
        self._refresh_grid_labels()

    def _adjust_drag_height(self, delta: int, pos: QPoint) -> None:
        '''Wheel mid-drag: step the whole selection's height (z >= 0).'''
        if not self._grab_orig_z:
            return
        step = self.grid_step()
        if delta > 0:
            self._drag_dz += step
        elif delta < 0:
            self._drag_dz = max(self._drag_dz - step, -min(self._grab_orig_z.values()))
        else:
            return
        # Re-anchor on the new plane so the next XY move does not jump.
        self._grab_anchor = None
        self._update_grab(pos)

    def _update_grab(self, pos: QPoint) -> None:
        '''Move the selection with the cursor on the plane at its height.

        The cursor ray is intersected with the horizontal plane through the
        primary piece's current bottom height (the wheel raises/lowers it),
        so XY deltas stay true to the piece's own level.
        '''
        if not self._grabbing or not self._grab_orig:
            return
        primary = (
            self._primary if self._primary in self._grab_orig else min(self._grab_orig)
        )
        plane_z = float(self._scene_meta[primary].get('pos_z', 0.0))
        world = self._cursor_on_plane(pos, plane_z)
        if world is None:
            return
        if self._grab_anchor is None:
            self._grab_anchor = world[:2].copy()
            if self._drag_dz != self._applied_dz:
                self._apply_move(self._grab_orig, 0.0, 0.0, dz=self._drag_dz)
                self._applied_dz = self._drag_dz
            return
        original_x, original_y = self._grab_orig[primary]
        target_x = original_x + float(world[0] - self._grab_anchor[0])
        target_y = original_y + float(world[1] - self._grab_anchor[1])
        if self._grid_snap:
            meta = self._scene_meta[primary]
            target_x, target_y = self._snap_move_target(
                target_x, target_y,
                float(meta.get('half_x', 0.0)), float(meta.get('half_y', 0.0)),
            )
        dx, dy = target_x - original_x, target_y - original_y
        self._apply_move(self._grab_orig, dx, dy, dz=self._drag_dz)
        self._applied_dz = self._drag_dz

    def _begin_piece_drag(self) -> None:
        '''Freeze the support set for the starting move drag/grab.'''
        uids: set[str] = set()
        # Exclude grabbed pieces from supports — use selection if available,
        # otherwise fall back to _grab_orig (covers tests that set _grab_orig
        # directly without setting _selection).
        source = self._selection if self._selection else self._grab_orig
        for index in source:
            if 0 <= index < len(self._scene_meta) and 'uid' in self._scene_meta[index]:
                uids.add(str(self._scene_meta[index]['uid']))
        self._drag_supports = self._query_supports(uids)
        self._grab_orig_z = {
            index: float(self._scene_meta[index].get('pos_z', 0.0))
            for index in self._grab_orig
            if 0 <= index < len(self._scene_meta)
        }
        self._drag_dz = 0.0
        self._applied_dz = 0.0
        self._rot_total = 0
        self._rot_undo = 0
        self._grab_start_pose = self._snapshot_pose()
        self._drag_valid_pose = dict(self._grab_start_pose)
        self._tint_drag(False)

    def _snapshot_pose(self) -> dict[int, tuple[float, float, float, int]]:
        '''Return the current (x, y, z, rot_z) for every grabbed piece.'''
        result: dict[int, tuple[float, float, float, int]] = {}
        for index in self._grab_orig:
            if not 0 <= index < len(self._scene_meta):
                continue
            meta = self._scene_meta[index]
            result[index] = (
                float(meta['pos_x']),
                float(meta['pos_y']),
                float(meta.get('pos_z', 0.0)),
                int(meta.get('rot_z', 0)),
            )
        return result

    def _apply_move(
        self,
        originals: dict[int, tuple[float, float]],
        dx: float,
        dy: float,
        restore: bool = False,
        dz: float = 0.0,
    ) -> None:
        '''Shift (or restore) the moved pieces rigidly in 3D.

        Pieces keep their heights except for the drag's accumulated wheel
        offset ``dz`` (``restore`` writes the original XY/Z back verbatim);
        heights are NEVER auto-resolved. When the resulting group pose
        intersects a static piece the outlines turn red and the pose is not
        captured as the commit fallback.
        '''
        for index, (original_x, original_y) in originals.items():
            if not 0 <= index < len(self._scene_meta):
                continue
            meta = self._scene_meta[index]
            original_z = float(
                self._grab_orig_z.get(index, float(meta.get('pos_z', 0.0)))
            )
            if restore:
                x, y, z = original_x, original_y, original_z
            else:
                x, y = original_x + dx, original_y + dy
                z = original_z + dz
                if dx != 0.0 or dy != 0.0 or dz != 0.0:
                    self._grab_moved = True
            meta['pos_x'] = x
            meta['pos_y'] = y
            meta['pos_z'] = z
            self._position_actor(index, x, y)
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.SetPosition(x, y, z)
        if restore:
            self._tint_drag(False)
        else:
            self._capture_drag_pose()
        self.render()

    def _capture_drag_pose(self) -> None:
        '''Validate the current drag pose; capture it when legal.

        Legal poses become the commit fallback (`_drag_valid_pose`) and the
        outlines return to the selection color; an intersecting pose only
        tints the outlines red.
        '''
        invalid = self._drag_pose_invalid()
        if not invalid:
            self._drag_valid_pose = self._snapshot_pose()
            self._rot_undo = 0
        self._tint_drag(invalid)

    def _drag_pose_invalid(self) -> bool:
        '''True when ANY moved piece intersects a static one.

        The frozen `_drag_supports` set already excludes the moved pieces.
        '''
        for index in self._grab_orig:
            if not 0 <= index < len(self._scene_meta):
                continue
            meta = self._scene_meta[index]
            rects = meta.get('rects')
            if rects is None:
                rects = (Rect(0.0, 0.0, float(meta['half_x']), float(meta['half_y'])),)
            z = float(meta.get('pos_z', 0.0))
            height = float(meta.get('top_z', 0.0))
            flipped = bool(meta.get('flipped', False))
            block_h = float(meta.get('block_h', height))
            if assemble_pose_collides(
                self._drag_supports,
                offset_rects(rects, float(meta['pos_x']), float(meta['pos_y'])),
                z, height,
                height if flipped else block_h,
            ):
                return True
        return False

    def _tint_drag(self, invalid: bool) -> None:
        '''Red selection outlines while the dragged group intersects.'''
        color = COLLISION_COLOR if invalid else SELECTION_COLOR
        for index in self._grab_orig:
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.prop.color = color

    def _rotate_moved_pieces(self) -> None:
        '''Rotate every piece of the active move drag 90 degrees about Z.

        Space mid-drag: each moved piece turns around its own center, the
        footprint halves swap, the actors gain an accumulated 90-degree
        Z rotation (the committed rot_z in the meta keeps the canonical
        orientation for the next rebuild). Height stays as-is.
        '''
        for index in self._grab_orig:
            if not 0 <= index < len(self._scene_meta):
                continue
            meta = self._scene_meta[index]
            meta['rot_z'] = (int(meta.get('rot_z', 0)) + 1) % 4
            meta['half_x'], meta['half_y'] = meta['half_y'], meta['half_x']
            base_rects = meta.get('base_rects')
            if base_rects is not None:
                meta['rects'] = transform_footprint_rects(
                    base_rects,
                    int(meta['rot_z']),
                    bool(meta.get('flipped', False)),
                )
            x, y = float(meta['pos_x']), float(meta['pos_y'])
            if index < len(self._block_actors):
                for actor in self._block_actors[index]:
                    actor.RotateZ(90.0)
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.RotateZ(90.0)
                z = float(meta.get('pos_z', 0.0))
                outline.SetPosition(x, y, z)
            self._position_actor(index, x, y)
        self._rot_total += 1
        self._rot_undo += 1
        self._capture_drag_pose()

    def _restore_valid_pose(self) -> None:
        '''Snap the drag back to the last legal pose (commit fallback).

        Rewrites the meta from `_drag_valid_pose`, rolls the actors back by
        the rotations applied since that snapshot, and repositions the
        outlines.
        '''
        for index, (x, y, z, rot_z) in self._drag_valid_pose.items():
            if not 0 <= index < len(self._scene_meta):
                continue
            meta = self._scene_meta[index]
            meta['pos_x'], meta['pos_y'], meta['pos_z'], meta['rot_z'] = x, y, z, rot_z
            self._position_actor(index, x, y)
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.SetPosition(x, y, z)
        if self._rot_undo:
            for index in self._drag_valid_pose:
                if not 0 <= index < len(self._block_actors):
                    continue
                for actor in self._block_actors[index]:
                    actor.RotateZ(-90.0 * self._rot_undo)
                outline = self._outline_actors.get(index)
                if outline is not None:
                    outline.RotateZ(-90.0 * self._rot_undo)
        self.render()

    def _rollback_drag(self) -> None:
        '''Undo a sub-threshold drag completely: positions, heights, rotations.'''
        if self._rot_total:
            for index in self._grab_orig:
                if not 0 <= index < len(self._block_actors):
                    continue
                for actor in self._block_actors[index]:
                    actor.RotateZ(-90.0 * self._rot_total)
                outline = self._outline_actors.get(index)
                if outline is not None:
                    outline.RotateZ(-90.0 * self._rot_total)
        self._restore_rot_meta()
        self._apply_move(self._grab_orig, 0.0, 0.0, restore=True)

    def _restore_rot_meta(self) -> None:
        '''Roll the meta orientation back to the drag start.

        Space mid-drag rotated the canonical meta too (rot_z, footprint
        halves, material rects); the drag-start snapshot restores all of
        them so a follow-up drag never inherits the rolled-back turns.
        '''
        for index in self._grab_orig:
            if not 0 <= index < len(self._scene_meta):
                continue
            start = self._grab_start_pose.get(index)
            if start is None:
                continue
            meta = self._scene_meta[index]
            start_rot = int(start[3]) % 4
            if int(meta.get('rot_z', 0)) % 4 == start_rot:
                continue
            meta['rot_z'] = start_rot
            if self._rot_total % 2:
                meta['half_x'], meta['half_y'] = meta['half_y'], meta['half_x']
            base_rects = meta.get('base_rects')
            if base_rects is not None:
                meta['rects'] = transform_footprint_rects(
                    base_rects, start_rot, bool(meta.get('flipped', False))
                )

    def start_move_drag(self, pos: QPoint) -> bool:
        '''Start dragging the selection (supports frozen first).'''
        if not self._placing:
            self._begin_piece_drag()
        return super().start_move_drag(pos)

    def start_grab(self, pos: QPoint | None = None) -> bool:
        '''Start the G-grab move (supports frozen first).'''
        if not self._placing:
            self._begin_piece_drag()
        return super().start_grab(pos)

    def confirm_grab(self) -> None:
        '''Finish the grab move and commit the new (x, y, z, rot_z) states.

        A pose intersecting a static piece is committed at the last legal
        pose of the drag instead (the red outlines already warned).
        '''
        if not self._grabbing:
            return
        if self._drag_pose_invalid():
            self._restore_valid_pose()
            self._tint_drag(False)
        self._end_grab()
        positions = {
            index: (
                float(self._scene_meta[index]['pos_x']),
                float(self._scene_meta[index]['pos_y']),
                float(self._scene_meta[index].get('pos_z', 0.0)),
                int(self._scene_meta[index].get('rot_z', 0)),
            )
            for index in self._drag_valid_pose
        }
        self._grab_orig.clear()
        self._grab_anchor = None
        self.blocks_moved.emit(positions)

    def _finish_move_drag(self, event: QMouseEvent) -> None:
        '''Release of an LMB move drag: commit, undo, or apply the click.'''
        drag_delta = event.position().toPoint() - self._grab_press_pos
        dragged = drag_delta.manhattanLength() > RUBBER_BAND_THRESHOLD_PX
        moved = self._grab_moved
        if moved and dragged:
            if self._drag_pose_invalid():
                self._restore_valid_pose()
                self._tint_drag(False)
            self._end_grab()
            positions = {
                index: (
                    float(self._scene_meta[index]['pos_x']),
                    float(self._scene_meta[index]['pos_y']),
                    float(self._scene_meta[index].get('pos_z', 0.0)),
                    int(self._scene_meta[index].get('rot_z', 0)),
                )
                for index in self._drag_valid_pose
            }
            self._grab_orig.clear()
            self._grab_anchor = None
            self.blocks_moved.emit(positions)
            return
        if moved:
            self._rollback_drag()
        self._grab_orig.clear()
        self._grab_anchor = None
        self._finish_click(event)

    def cancel_grab(self) -> None:
        '''Finish the grab move restoring positions, heights and rotations.'''
        if not self._grabbing:
            return
        self._rollback_drag()
        self._tint_drag(False)
        self._end_grab()
        self._grab_orig.clear()
        self._grab_anchor = None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''Ghost mode: LMB drops, RMB cancels, MMB keeps the camera moves.'''
        pos = event.position().toPoint()
        button = event.button()
        if self._placing:
            self.setFocus()
            if button == Qt.MouseButton.LeftButton:
                # Drop exactly under the press cursor, never at the ghost's
                # last followed position.
                self._update_ghost(pos)
                self._drop_ghost()
            elif button == Qt.MouseButton.RightButton:
                self.cancel_placement()
            elif button == Qt.MouseButton.MiddleButton:
                self._last_mouse = pos
                self._panning = bool(
                    event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                )
                self._orbiting = not self._panning
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''Ghost mode: camera moves, else the ghost follows the cursor.'''
        pos = event.position().toPoint()
        if self._placing:
            if self._orbiting:
                self._orbit_camera(
                    pos.x() - self._last_mouse.x(), pos.y() - self._last_mouse.y()
                )
                self._last_mouse = pos
            elif self._panning:
                self._pan_camera(self._last_mouse, pos)
                self._last_mouse = pos
            else:
                self._update_ghost(pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''Ghost mode: release the camera-move buttons, swallow the rest.'''
        if self._placing:
            if event.button() == Qt.MouseButton.MiddleButton:
                self._orbiting = False
                self._panning = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt naming
        '''Placement keys (R/Space rotate, F flips, Esc cancels), Space
        rotates the pieces of an active move drag, Del deletes.'''
        key = event.key()
        if self._placing:
            if key in (Qt.Key.Key_R, Qt.Key.Key_Space):
                self._ghost_rot = (self._ghost_rot + 1) % 4
                self._refresh_ghost_rects()
                self._build_ghost_actor()
                self.render()
                event.accept()
                return
            if key == Qt.Key.Key_F:
                self._ghost_flip = not self._ghost_flip
                self._refresh_ghost_rects()
                self._build_ghost_actor()
                self.render()
                event.accept()
                return
            if key == Qt.Key.Key_Escape:
                self.cancel_placement()
                event.accept()
                return
            event.accept()
            return
        if self._grabbing and key == Qt.Key.Key_Space:
            self._rotate_moved_pieces()
            event.accept()
            return
        if key == Qt.Key.Key_Delete and self._selection:
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
