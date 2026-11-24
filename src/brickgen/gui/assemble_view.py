'''
File:   assemble_view.py
Brief:  Assemble-tab 3D viewport: free piece placement, collision feedback, magnets.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyvista as pv
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QWidget

from brickgen.gui.keys import event_key
from brickgen.gui.preview import (
    AXIS_VECTORS,
    RUBBER_BAND_THRESHOLD_PX,
    SELECTION_COLOR,
    PreviewWidget,
    assemble_variant_mesh,
)
from brickgen.services.assemble_service import (
    Rect,
    Support,
    _bbox_halves,
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
      the cursor (like the G grab), follows the cursor with axis-lock keys
      (X/Y/Z constrain to that axis, Shift+X/Y/Z constrains to the plane
      excluding that axis); every left click drops a copy (R or Space
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
    # Emitted when a multi-piece ghost group is dropped atomically: list of
    # per-piece dicts with keys ``block``, ``x``, ``y``, ``z``, ``rot_z``,
    # ``flipped`` in group order. Single-entry groups still use piece_placed.
    pieces_placed = Signal(list)
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
        # Free-path anchor for view-plane ghost movement (reset on pose changes).
        self._ghost_free_anchor: np.ndarray | None = None
        # Multi-piece ghost group (set by _paste_assembled). Each entry is a
        # dict with keys: block, base_rects, height, block_height, rot_z,
        # flipped, dx, dy, dz. Anchor entry has (0.0, 0.0, 0.0).
        self._ghost_group: list[dict] | None = None
        # --- Drag state ---
        self._grab_orig_z: dict[int, float] = {}
        self._drag_dz = 0.0
        self._applied_dz = 0.0
        self._drag_valid_pose: dict[int, tuple[float, float, float, int]] = {}
        self._grab_start_pose: dict[int, tuple[float, float, float, int]] = {}
        self._rot_total = 0  # quarter turns applied during the active drag
        self._rot_undo = 0  # quarter turns applied after the last valid snapshot

    def active_axes(self) -> frozenset[str]:
        '''Assemble tab allows Z movement (3D placement).'''
        return frozenset({"x", "y", "z"})

    def _z_snap_step(self) -> float:
        '''Z snap step: block height scaled by the grid scale factor.'''
        if self._placing:
            height = self._ghost_block_height
        elif self._grab_orig:
            primary = (
                self._primary
                if self._primary in self._grab_orig
                else min(self._grab_orig)
            )
            height = float(self._scene_meta[primary].get('block_h', 0.0))
        else:
            height = 0.0
        if height <= 1e-9:
            return super()._z_snap_step()
        return max(height * self._grid_scale, 1e-9)

    def _clamp_move_dz(self, dz: float) -> float:
        '''Clamp Z delta so the lowest grabbed piece never goes below z=0.'''
        if self._grab_orig_z:
            min_z = -min(self._grab_orig_z.values())
            return max(dz, min_z)
        return dz

    def _retained_move_dz(self) -> float:
        '''Plane locks keep the drag's accumulated Z offset (current height).'''
        return float(self._drag_dz)

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

        The ghost follows the cursor with axis-lock keys (X/Y/Z constrain
        to that axis, Shift+X/Y/Z constrains to the plane excluding that
        axis) until Esc or a right click cancels; every left click drops a
        copy (the ghost stays for the next drop).

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
        # Reset axis-lock state for a fresh placement.
        self._axis_lock = None
        self._axis_plane = False
        self._constraint_anchor = None
        self._constraint_origin = None
        self._constraint_zero = None
        self._last_grab_pos = None
        self._ghost_free_anchor = None
        self._ghost_group = None
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
        self._update_gizmo()
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
        '''Leave the ghost placement mode and remove the ghost actor.

        Also clears guide lines and resets axis-lock state.
        '''
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
        self._ghost_group = None
        self.unsetCursor()
        self._clear_guides()
        self._axis_lock = None
        self._axis_plane = False
        self._constraint_anchor = None
        self._constraint_origin = None
        self._constraint_zero = None
        self._last_grab_pos = None
        self._ghost_free_anchor = None
        self._update_gizmo()
        self.render()

    def set_ghost_group(self, entries: list[dict]) -> None:
        '''Store a copy of the multi-piece ghost group.

        Each entry is a dict with keys: ``block``, ``base_rects``,
        ``height``, ``block_height``, ``rot_z``, ``flipped``, ``dx``,
        ``dy``, ``dz``. The anchor entry (first) has ``(0.0, 0.0, 0.0)``
        offset; its ``rot_z`` / ``flipped`` mirror what
        :meth:`start_placement` received.

        Args:
            entries: List of per-piece metadata for the group drop.
        '''
        self._ghost_group = [dict(e) for e in entries]

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
        '''Move the ghost to the snapped drop position under the cursor.

        When an axis lock is active (X/Y/Z key), delegates to the constrained
        path; otherwise uses camera view-plane movement through the ghost anchor
        with magnet snapping.
        '''
        if not self._placing or self._ghost_actor is None:
            return
        self._last_grab_pos = QPoint(pos)
        if self._axis_lock is not None:
            self._update_ghost_constrained(pos)
            return
        if self._guide_actors:
            # A just-disabled lock leaves stale guide lines behind.
            self._clear_guides()
        hit = self._pick_scene_point(pos)
        pick_x, pick_y = 0.0, 0.0
        pick_point: np.ndarray | None = None
        cursor_z = self._ghost_z
        if hit is not None:
            pick_x, pick_y = float(hit[1][0]), float(hit[1][1])
            pick_point = hit[1]
        else:
            # Free path: camera view-plane through the ghost anchor.
            if self._ghost_free_anchor is None:
                gx, gy, gz = (
                    self._ghost_pos
                    if self._ghost_pos is not None
                    else (0.0, 0.0, self._ghost_z)
                )
                self._ghost_free_anchor = np.array([gx, gy, gz], dtype=np.float64)
            raw = self._cursor_on_view_plane(pos, self._ghost_free_anchor)
            if raw is None:
                return
            self._ghost_free_anchor = raw.copy()
            pick_x = float(raw[0])
            pick_y = float(raw[1])
            cursor_z = float(raw[2])
        x, y, z = assemble_snap_target(
            pick_x,
            pick_y,
            cursor_z,
            self._ghost_rects,
            self._ghost_height,
            self._ghost_block_height,
            self._ghost_flip,
            self._drag_supports,
            self.grid_step(),
            self._grid_snap,
            self._snap_parts,
            z_step=self._z_snap_step(),
            pick_point=pick_point,
        )
        z = max(z, 0.0)
        if pick_point is not None:
            # Handoff: re-anchor the free path at the magnet result, so
            # leaving the hovered piece continues from the snapped pose
            # instead of jumping back to a stale cursor anchor.
            self._ghost_free_anchor = np.array([x, y, z], dtype=np.float64)
        self._ghost_pos = (x, y, z)
        self._ghost_actor.SetPosition(x, y, z)
        self._tint_ghost(not self._ghost_valid_at(x, y, z))
        self.render()

    def _update_ghost_constrained(self, pos: QPoint) -> None:
        '''Constrained-path ghost placement via axis-lock intersection.

        Uses ``_constrained_delta`` to compute the total world displacement
        from the ghost anchor, then applies plain grid snapping (no magnet
        snapping).

        Args:
            pos: Cursor position in Qt widget coordinates.
        '''
        # Clear stale free-path anchor when entering constrained mode.
        self._ghost_free_anchor = None
        # Lazily engage: when constraint_anchor is None, set it from ghost pos.
        if self._constraint_anchor is None:
            gx, gy, gz = (
                self._ghost_pos
                if self._ghost_pos is not None
                else (0.0, 0.0, self._ghost_z)
            )
            self._constraint_anchor = np.array([gx, gy, gz], dtype=np.float64)
            self._constraint_zero = np.zeros(3, dtype=np.float64)
            # Compute _constraint_origin based on lock mode.
            if not self._axis_plane:
                # Single-axis: absorb the engagement cursor's projection onto
                # the axis line into the origin, so the first constrained
                # frame is a no-move (the lock never jumps the ghost).
                self._constraint_origin = self._constraint_anchor.copy()
                raw0 = self._constrained_raw_delta(pos)
                if raw0 is not None:
                    idx = {'x': 0, 'y': 1, 'z': 2}[str(self._axis_lock)]
                    self._constraint_origin[idx] += float(raw0[idx])
            else:
                # Plane mode: origin = cursor ray intersection with the plane
                # through the anchor (fallback: the anchor itself when the
                # ray is too parallel to the plane).
                lock = self._axis_lock
                assert lock is not None  # constrained path only when locked
                self._constraint_origin = self._constraint_anchor.copy()
                ray = self._cursor_ray(pos)
                if ray is not None:
                    point = self._ray_plane_point(
                        ray, self._constraint_anchor, AXIS_VECTORS[lock]
                    )
                    if point is not None:
                        self._constraint_origin = point

        # Compute total delta from constrained math.
        raw = self._constrained_raw_delta(pos)
        if raw is None:
            return

        anchor = self._constraint_anchor
        assert anchor is not None  # ensured by lazy engagement above
        tx = anchor[0] + float(raw[0])
        ty = anchor[1] + float(raw[1])
        tz = anchor[2] + float(raw[2])

        allowed = self._allowed_axes()
        assert allowed is not None  # constrained path only when lock is set
        if "x" not in allowed:
            tx = float(anchor[0])
        if "y" not in allowed:
            ty = float(anchor[1])
        if "z" not in allowed:
            tz = float(anchor[2])

        # Plain snapping (no magnet snapping under constraint): the footprint
        # minimum corner rounds to the grid step like the free path does.
        if self._grid_snap:
            step = max(self.grid_step(), 1e-9)
            z_step = self._z_snap_step()
            hx = hy = 0.0
            if self._ghost_rects:
                cx_vals = [r.cx + r.hx for r in self._ghost_rects]
                cx_vals += [r.cx - r.hx for r in self._ghost_rects]
                cy_vals = [r.cy + r.hy for r in self._ghost_rects]
                cy_vals += [r.cy - r.hy for r in self._ghost_rects]
                hx = (max(cx_vals) - min(cx_vals)) / 2.0
                hy = (max(cy_vals) - min(cy_vals)) / 2.0
            if "x" in allowed:
                tx = round((tx - hx) / step) * step + hx
            if "y" in allowed:
                ty = round((ty - hy) / step) * step + hy
            if "z" in allowed:
                tz = round(tz / z_step) * z_step

        # Clamp z >= 0.
        tz = max(tz, 0.0)

        self._ghost_pos = (tx, ty, tz)
        ghost_actor = self._ghost_actor
        assert ghost_actor is not None
        ghost_actor.SetPosition(tx, ty, tz)
        self._tint_ghost(not self._ghost_valid_at(tx, ty, tz))
        self._update_guides(
            np.array([tx, ty, tz + self._ghost_height / 2.0], dtype=np.float64)
        )
        self.render()

    def _reengage_constraint(self) -> None:
        '''Override to handle ghost re-engagement during placement.'''
        if self._placing:
            self._constraint_anchor = None
            self._constraint_origin = None
            self._constraint_zero = None
            if self._last_grab_pos is not None:
                self._update_ghost(QPoint(self._last_grab_pos))
        else:
            super()._reengage_constraint()

    def _drop_ghost(self) -> None:
        '''Drop one copy at the current ghost pose (ghost stays active).

        When a multi-piece ghost group is active, all pieces are dropped
        atomically via ``pieces_placed``; single-entry groups fall back to
        the existing ``piece_placed`` path.
        '''
        if not self._placing or self._ghost_actor is None:
            return
        if self._ghost_pos is not None:
            x, y, z = self._ghost_pos
        else:
            position = self._ghost_actor.GetPosition()
            x, y, z = float(position[0]), float(position[1]), float(position[2])

        # Multi-piece group drop.
        if (
            self._ghost_group is not None
            and len(self._ghost_group) >= 2
        ):
            self._drop_ghost_group(x, y, z)
            return

        # Single-piece path (unchanged).
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

    def _drop_ghost_group(
        self, x: float, y: float, z: float
    ) -> None:
        '''Drop a multi-piece ghost group atomically.

        All entries are validated against existing supports (and each other
        in group order). On success emits ``pieces_placed`` once; on failure
        tints the ghost red and returns without emission.

        Args:
            x, y, z: Current ghost pose (anchor piece position).
        '''
        assert self._ghost_group is not None  # only called when group exists
        anchor = self._ghost_group[0]
        anchor_rot = anchor['rot_z']
        anchor_flip = anchor['flipped']

        # Compute the delta between current ghost pose and what start_placement
        # originally received (from the anchor entry's stored values).
        d_rot = (self._ghost_rot - anchor_rot) % 4
        d_flip = self._ghost_flip != anchor_flip

        # Build all poses and validate atomically.
        supports_snapshot: list[Support] = list(self._drag_supports)
        infos: list[dict] = []

        for entry in self._ghost_group:
            rot = (entry['rot_z'] + d_rot) % 4
            flip = entry['flipped'] != d_flip

            # Rotate the offset by d_rot quarter turns CCW.
            dx_r, dy_r = float(entry['dx']), float(entry['dy'])
            for _ in range(d_rot):
                dx_r, dy_r = -dy_r, dx_r

            px = x + dx_r
            py = y + dy_r
            pz = max(z + entry['dz'], 0.0)

            rects_i = transform_footprint_rects(
                entry['base_rects'], rot, flip
            )
            stack_offset = (
                entry['height'] if flip else entry['block_height']
            )
            if not assemble_pose_collides(
                supports_snapshot,
                offset_rects(rects_i, px, py),
                pz,
                entry['height'],
                stack_offset,
            ):
                # Append this piece's support for subsequent pieces to check.
                hx, hy = _bbox_halves(rects_i)
                supports_snapshot.append(
                    Support(
                        x=px, y=py, z=pz,
                        hx=hx, hy=hy,
                        height=entry['height'],
                        stack_top=pz + stack_offset,
                        rects=offset_rects(rects_i, px, py),
                    )
                )
                infos.append({
                    'block': entry['block'],
                    'x': px,
                    'y': py,
                    'z': pz,
                    'rot_z': rot,
                    'flipped': flip,
                })
            else:
                # Any collision → refuse the whole group.
                self._tint_ghost(True)
                return

        # All pieces valid — emit once.
        self.pieces_placed.emit(infos)
        # The window committed all pieces (direct Qt connection): refresh
        # the support set so the next drop stacks on them.
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

    def _update_grab(self, pos: QPoint) -> None:
        '''Move the selection with the cursor via camera view-plane movement.

        When an axis lock is active (X/Y/Z key), delegates to the constrained
        path; otherwise uses camera view-plane delta tracking anchored at the
        primary piece's current position.
        '''
        if not self._grabbing or not self._grab_orig:
            return
        self._last_grab_pos = QPoint(pos)
        # Clear stale free-path anchor when entering constrained mode.
        if self._axis_lock is not None:
            self._grab_free_anchor = None
            self._update_grab_constrained(pos)
            self._sync_drag_dz_from_primary()
            return
        primary = (
            self._primary if self._primary in self._grab_orig else min(self._grab_orig)
        )
        if self._grab_free_anchor is None:
            meta = self._scene_meta[primary]
            plane_point = np.array(
                [
                    float(meta['pos_x']),
                    float(meta['pos_y']),
                    float(meta.get('pos_z', 0.0)),
                ],
                dtype=np.float64,
            )
            raw = self._cursor_on_view_plane(pos, plane_point)
            if raw is None:
                return
            self._grab_free_anchor = raw.copy()
            self._grab_anchor = raw[:2].copy()
            self._grab_free_dz = float(self._drag_dz)
            if self._drag_dz != self._applied_dz:
                self._apply_move(self._grab_orig, 0.0, 0.0, dz=self._drag_dz)
                self._applied_dz = self._drag_dz
            return
        raw = self._cursor_on_view_plane(pos, self._grab_free_anchor)
        if raw is None:
            return
        # The anchor stays FIXED after engagement: the plane through it is
        # camera-parallel, so delta is the TOTAL view-plane displacement
        # since the free path started (re-anchoring every frame would turn
        # it into a per-frame increment and the piece would jitter in place).
        delta = raw - self._grab_free_anchor
        original_x, original_y = self._grab_orig[primary]
        target_x = original_x + float(delta[0])
        target_y = original_y + float(delta[1])
        dz = self._grab_free_dz + float(delta[2])
        dz = self._clamp_move_dz(dz)
        if self._grid_snap:
            meta = self._scene_meta[primary]
            target_x, target_y = self._snap_move_target(
                target_x,
                target_y,
                float(meta.get('half_x', 0.0)),
                float(meta.get('half_y', 0.0)),
            )
            orig_z = float(self._grab_orig_z.get(primary, 0.0))
            z_step = self._z_snap_step()
            dz = round((orig_z + dz) / z_step) * z_step - orig_z
        dx, dy = target_x - original_x, target_y - original_y
        self._apply_move(self._grab_orig, dx, dy, dz=dz)
        self._sync_drag_dz_from_primary()
        self._update_guides(self._selection_center())

    def _sync_drag_dz_from_primary(self) -> None:
        '''Sync _drag_dz / _applied_dz from the primary piece's current Z.'''
        if not self._grab_orig:
            return
        primary = (
            self._primary if self._primary in self._grab_orig else min(self._grab_orig)
        )
        dz = float(self._scene_meta[primary].get('pos_z', 0.0)) - float(
            self._grab_orig_z.get(primary, 0.0)
        )
        self._drag_dz = dz
        self._applied_dz = dz

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
                z,
                height,
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

        Single-selection: pivot equals the piece's own center, so its
        position stays unchanged (rigid rotation about self-centre).
        Multi-selection: the group pivots rigidly about the XY centre of
        the selection bounding box; each piece follows a +90° CCW affine
        transform around that common pivot.

        Space mid-drag: each moved piece's ``rot_z`` increments (mod 4),
        the footprint halves swap, the actors gain an accumulated
        90-degree Z rotation (the committed ``rot_z`` in the meta keeps
        the canonical orientation for the next rebuild). Height stays as-is.
        '''
        # Compute group pivot: AABB centre of the selection (same as
        # _selection_center() used by MoveGizmo and axis-lock anchor).
        first = True
        for index in self._grab_orig:
            if not 0 <= index < len(self._scene_meta):
                continue
            meta = self._scene_meta[index]
            x0 = float(meta['pos_x']) - float(meta['half_x'])
            y0 = float(meta['pos_y']) - float(meta['half_y'])
            x1 = float(meta['pos_x']) + float(meta['half_x'])
            y1 = float(meta['pos_y']) + float(meta['half_y'])
            if first:
                min_x0, max_x1 = x0, x1
                min_y0, max_y1 = y0, y1
                first = False
            else:
                if x0 < min_x0:
                    min_x0 = x0
                if x1 > max_x1:
                    max_x1 = x1
                if y0 < min_y0:
                    min_y0 = y0
                if y1 > max_y1:
                    max_y1 = y1

        # If no valid pieces, nothing to do — return early without touching
        # the rotation counters so a double-Space on empty grab is idempotent.
        if first:
            return

        px = (min_x0 + max_x1) / 2.0
        py = (min_y0 + max_y1) / 2.0

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
            # Piece centre BEFORE rotation.
            x, y = float(meta['pos_x']), float(meta['pos_y'])
            dx = x - px
            dy = y - py
            # Rigid +90° CCW about pivot (px, py): (cx,cy) -> (px-dy, py+dx).
            new_x = px - dy
            new_y = py + dx
            meta['pos_x'] = new_x
            meta['pos_y'] = new_y
            if index < len(self._block_actors):
                for actor in self._block_actors[index]:
                    actor.RotateZ(90.0)
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.RotateZ(90.0)
                z = float(meta.get('pos_z', 0.0))
                outline.SetPosition(new_x, new_y, z)
            self._position_actor(index, new_x, new_y)
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
        started = super().start_move_drag(pos)
        if started and not self._placing:
            self._begin_piece_drag()
        return started

    def start_grab(self, pos: QPoint | None = None) -> bool:
        '''Start the G-grab move (supports frozen first).'''
        started = super().start_grab(pos)
        if started and not self._placing:
            self._begin_piece_drag()
        return started

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
        self._grab_free_anchor = None
        self._grab_free_dz = 0.0
        self.blocks_moved.emit(positions)

    def _finish_move_drag(self, event: QMouseEvent) -> None:
        '''Release of an LMB move drag: commit, undo, or apply the click.'''
        drag_delta = event.position().toPoint() - self._grab_press_pos
        dragged = drag_delta.manhattanLength() > RUBBER_BAND_THRESHOLD_PX
        moved = self._grab_moved
        from_axis = self._drag_from_axis
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
            self._grab_free_anchor = None
            self._grab_free_dz = 0.0
            self.blocks_moved.emit(positions)
            return
        if moved:
            self._rollback_drag()
        self._tint_drag(False)
        self._end_grab()
        self._grab_orig.clear()
        self._grab_anchor = None
        self._grab_free_anchor = None
        self._grab_free_dz = 0.0
        if not from_axis:
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
        self._grab_free_anchor = None
        self._grab_free_dz = 0.0

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
        '''Placement keys (R/Space rotate, F flips, Esc cancels), X/Y/Z lock
        movement to axis/plane during placement, Space rotates the pieces of
        an active move drag, Del deletes.'''
        key = event_key(event)
        if self._placing:
            lock_map: dict[int, str] = {
                Qt.Key.Key_X: "x",
                Qt.Key.Key_Y: "y",
                Qt.Key.Key_Z: "z",
            }
            if key in lock_map and self._set_axis_lock(
                lock_map[key],
                bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
            ):
                event.accept()
                return
            if key in (Qt.Key.Key_R, Qt.Key.Key_Space):
                self._ghost_rot = (self._ghost_rot + 1) % 4
                self._refresh_ghost_rects()
                self._build_ghost_actor()
                self._ghost_free_anchor = None
                self.render()
                event.accept()
                return
            if key == Qt.Key.Key_F:
                self._ghost_flip = not self._ghost_flip
                self._refresh_ghost_rects()
                self._build_ghost_actor()
                self._ghost_free_anchor = None
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
