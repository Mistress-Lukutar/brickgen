'''
File:   assemble_view.py
Brief:  Assemble-tab 3D viewport: placed pieces, cursor ghost placement, stacking.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

from collections.abc import Callable

import pyvista as pv
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QWidget

from brickgen.gui.preview import (
    RUBBER_BAND_THRESHOLD_PX,
    PreviewWidget,
    assemble_variant_mesh,
)
from brickgen.services.assemble_service import (
    Rect,
    Support,
    assemble_snap_target,
    offset_rects,
    support_top_z,
    transform_footprint_rects,
)

# Opacity of the cursor-attached ghost piece while placing from the panel.
GHOST_OPACITY = 0.6


class AssembleView(PreviewWidget):
    '''Assemble-tab viewport: pieces placed in free 3D space.

    Pieces are ordinary scene blocks for the base :class:`PreviewWidget`
    (their meta carries ``pos_z`` and the piece orientation ``rot_z`` so
    actors stack above the floor plane). On top of the base navigation/
    selection machinery this subclass adds:

    - panel-driven placement: a translucent ghost piece attaches to the
      cursor (like the G grab), follows the floor point with grid and
      part snapping, and every left click drops a copy (R or Space
      rotates 90 degrees, F flips, Esc/right-click exits the mode);
    - height-aware moves: dragging keeps the piece on the topmost support
      under its footprint (part snapping) or at its original height;
    - Space during a move drag / G-grab rotates every moved piece 90
      degrees about Z around its own center;
    - the Del hotkey deletes the selected pieces.

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
        # Ghost material rectangles: base pose (rot 0, unflipped) and the
        # current pose (rebuilt on R/Space/F) in piece-relative coords.
        self._ghost_base_rects: tuple[Rect, ...] = ()
        self._ghost_rects: tuple[Rect, ...] = ()
        self._ghost_height = 0.0
        self._ghost_rot = 0
        self._ghost_flip = False
        self._ghost_color = "#ffffff"

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
    ) -> bool:
        '''Attach a ghost copy of *mesh* to the cursor (panel-driven).

        The ghost follows the floor point with the current snapping until
        Esc or a right click cancels; every left click drops a copy (the
        ghost stays for the next drop).

        Args:
            mesh: Untransformed block mesh (centered at origin).
            block_index: Scene index of the source block (reported on drop).
            rects: Piece-relative material rectangles of the block in its
                base orientation (excluded cells leave voids that pieces
                may enter).
            height: Block total height (flip lift), mm.
            color: Ghost surface color.

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
        self._ghost_rot = 0
        self._ghost_flip = False
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

    def _update_ghost(self, pos: QPoint) -> None:
        '''Move the ghost to the snapped drop position under the cursor.'''
        if not self._placing or self._ghost_actor is None:
            return
        world = self._cursor_on_floor(pos)
        if world is None:
            return
        x, y, z = assemble_snap_target(
            float(world[0]),
            float(world[1]),
            self._ghost_rects,
            self._drag_supports,
            self.grid_step(),
            self._grid_snap,
            self._snap_parts,
        )
        self._ghost_actor.SetPosition(x, y, z)
        self.render()

    def _drop_ghost(self) -> None:
        '''Drop one copy at the current ghost pose (ghost stays active).'''
        if not self._placing or self._ghost_actor is None:
            return
        position = self._ghost_actor.GetPosition()
        self.piece_placed.emit(
            {
                'block': self._ghost_block,
                'x': float(position[0]),
                'y': float(position[1]),
                'z': float(position[2]),
                'rot_z': self._ghost_rot,
                'flipped': self._ghost_flip,
            }
        )
        # The window committed the new piece (direct Qt connection): refresh
        # the support set so the next drop stacks on it.
        self._drag_supports = self._query_supports(set())

    # ------------------------------------------------------- height handling

    def _piece_drop_z(self, index: int, x: float, y: float) -> float:
        '''Drop height of piece *index* at (x, y) under the current snapping.

        With part snapping the piece lands on the topmost support whose
        MATERIAL overlaps its footprint (floor when free — excluded-cell
        voids let pieces drop into the empty space); without it the piece
        keeps its current height (free 3D drag, exact heights via the
        inspector).
        '''
        meta = self._scene_meta[index]
        if not self._snap_parts:
            return float(meta.get('pos_z', 0.0))
        rects = meta.get('rects')
        if rects is None:  # scene meta without material rects: full box
            rects = (Rect(0.0, 0.0, float(meta['half_x']), float(meta['half_y'])),)
        top = support_top_z(self._drag_supports, offset_rects(rects, x, y))
        return 0.0 if top is None else top

    def _rotate_moved_pieces(self) -> None:
        '''Rotate every piece of the active move drag 90 degrees about Z.

        Space mid-drag: each moved piece turns around its own center, the
        footprint halves swap, the drop height re-resolves and the actors
        gain an accumulated 90-degree Z rotation (the committed rot_z in
        the meta keeps the canonical orientation for the next rebuild).
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
            meta['pos_z'] = self._piece_drop_z(index, x, y)
            if index < len(self._block_actors):
                for actor in self._block_actors[index]:
                    actor.RotateZ(90.0)
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.RotateZ(90.0)
            z = float(meta['pos_z'])
            self._position_actor(index, x, y)
            if outline is not None:
                outline.SetPosition(x, y, z)
        self.render()

    def _begin_piece_drag(self) -> None:
        '''Freeze the support set for the starting move drag/grab.'''
        uids = {
            str(self._scene_meta[index]['uid'])
            for index in self._selection
            if 0 <= index < len(self._scene_meta) and 'uid' in self._scene_meta[index]
        }
        self._drag_supports = self._query_supports(uids)

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

    def _apply_move(
        self,
        originals: dict[int, tuple[float, float]],
        dx: float,
        dy: float,
        restore: bool = False,
    ) -> None:
        '''Shift (or restore) the moved pieces with height resolution.

        Every moved piece re-resolves its drop height at the shifted
        footprint (stack onto supports, or keep its height without part
        snapping); ``restore`` writes the original XY back verbatim and
        leaves the height untouched.

        Args:
            originals: Piece index -> (x, y) the move started from.
            dx: Delta applied to every X coordinate.
            dy: Delta applied to every Y coordinate.
            restore: True restores the original coordinates.
        '''
        for index, (original_x, original_y) in originals.items():
            if not 0 <= index < len(self._scene_meta):
                continue
            meta = self._scene_meta[index]
            if restore:
                x, y = original_x, original_y
            else:
                x, y = original_x + dx, original_y + dy
                meta['pos_z'] = self._piece_drop_z(index, x, y)
            meta['pos_x'] = x
            meta['pos_y'] = y
            self._position_actor(index, x, y)
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.SetPosition(x, y, float(meta.get('pos_z', 0.0)))
        self.render()

    def confirm_grab(self) -> None:
        '''Finish the grab move and commit the new (x, y, z, rot_z) states.'''
        if not self._grabbing:
            return
        self._end_grab()
        positions = {
            index: (
                float(self._scene_meta[index]['pos_x']),
                float(self._scene_meta[index]['pos_y']),
                float(self._scene_meta[index].get('pos_z', 0.0)),
                int(self._scene_meta[index].get('rot_z', 0)),
            )
            for index in self._grab_orig
        }
        self._grab_orig.clear()
        self._grab_anchor = None
        self.blocks_moved.emit(positions)

    def _finish_move_drag(self, event: QMouseEvent) -> None:
        '''Release of an LMB move drag: commit, undo, or apply the click.'''
        dragged = (
            event.position().toPoint() - self._grab_press_pos
        ).manhattanLength() > RUBBER_BAND_THRESHOLD_PX
        moved = self._grab_moved
        self._end_grab()
        if moved and dragged:
            positions = {
                index: (
                    float(self._scene_meta[index]['pos_x']),
                    float(self._scene_meta[index]['pos_y']),
                    float(self._scene_meta[index].get('pos_z', 0.0)),
                    int(self._scene_meta[index].get('rot_z', 0)),
                )
                for index in self._grab_orig
            }
            self._grab_orig.clear()
            self._grab_anchor = None
            self.blocks_moved.emit(positions)
            return
        if moved:
            self._apply_move(self._grab_orig, 0.0, 0.0, restore=True)
        self._grab_orig.clear()
        self._grab_anchor = None
        self._finish_click(event)

    # --------------------------------------------------------- input routing

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
