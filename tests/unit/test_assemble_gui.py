'''
File:   test_assemble_gui.py
Brief:  Integration tests for the Assemble tab (MainWindow level).
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

import os
import time
from collections.abc import Iterator

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from brickgen.core.project import BlockInstance  # noqa: E402
from brickgen.gui.app import MainWindow  # noqa: E402

# vtk 9.6 numpy_support sets .shape on arrays, deprecated in numpy 2.5.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy Array:DeprecationWarning"
)


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


def _wait_applied(window: MainWindow, timeout_s: float = 120.0) -> None:
    '''Pump Qt events until the first generation lands.'''
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if window._applied_context is not None:
            return
        time.sleep(0.05)
    pytest.fail(f"Generation did not land within {timeout_s}s")


@pytest.fixture(scope='module')
def window() -> Iterator[MainWindow]:
    '''One MainWindow for the whole module (mirrors test_app_tabs).'''
    _ensure_app()
    win = MainWindow()
    win.resize(1200, 760)
    win.show()
    _wait_applied(win)
    yield win
    win._dirty = False
    win._jobs.close()
    win.close()


def _wait_meshes(win: MainWindow, timeout_s: float = 120.0) -> None:
    '''Pump Qt events until the block meshes are re-populated.

    Undo/redo restores the project wholesale and clears the mesh cache;
    the regeneration is async, so actor-level assertions must wait.
    '''
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if win._block_meshes and any(
            mesh is not None and mesh.n_points > 0 for mesh in win._block_meshes
        ):
            return
        time.sleep(0.05)
    pytest.fail(f"Meshes did not land within {timeout_s}s")


def _reset(win: MainWindow) -> None:
    '''Back to one unlinked default block, empty assembly and placement.'''
    # Cancel any active placement / grab mode so stale state doesn't leak.
    view = win._assemble_view
    if view.placing():
        view.cancel_placement()
    if view._grabbing:
        view.cancel_grab()
    win._project.blocks[:] = [win._project.blocks[0]]
    win._project.blocks[0].quantity = 1
    win._project.blocks[0].quantity_linked = False
    win._project.print_layout.pieces.clear()
    win._project.assemble.pieces.clear()
    win._aselection = set()
    win._aprimary = None
    win._clip_assembled = []
    win._assemble_view.set_selection(set(), None)
    win._sync_placement()
    win._sync_assemble()
    win._reset_edit_history()
    win._dirty = False


def _place(win: MainWindow, block_index: int = 0, **overrides) -> int:
    '''Drop one piece through the app handler; return the piece count.'''
    info = {'block': block_index, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    info.update(overrides)
    win._on_assemble_piece_placed(info)
    return len(win._project.assemble.pieces)


def _actor_color_hex(actor) -> str:
    '''Extract the hex color string from a VTK actor's prop.color.

    Args:
        actor: A pyvista/vtk actor with ``prop.color`` attribute.

    Returns:
        Hex color string like ``"#d02020"``.
    '''
    c = actor.prop.color
    if hasattr(c, '__iter__') and not isinstance(c, str):
        # VTK Color / tuple of RGB floats (0-1).
        r, g, b = float(c[0]), float(c[1]), float(c[2])
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    # Already a string.
    return str(c)


def _wheel(view, dy: int) -> QWheelEvent:
    '''Send a wheel delta to *view* and return the event.

    Args:
        view: AssembleView to receive the wheel event.
        dy: Vertical scroll delta (positive = up / increase height).

    Returns:
        The constructed ``QWheelEvent``.
    '''
    event = QWheelEvent(
        QPointF(10.0, 10.0), QPointF(10.0, 10.0), QPoint(0, 0), QPoint(0, dy),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    view.wheelEvent(event)
    return event


# ===========================================================================


def test_place_piece_updates_everything(window: MainWindow) -> None:
    '''Placing a copy adds a piece, refreshes the panel counts and dirties.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    assert _place(window) == 1
    piece = window._project.assemble.pieces[0]
    assert piece.block_uid == window._project.blocks[0].uid
    assert window._dirty is True

    # The planned-parts card mirrors the count.
    counts_label = window._parts_panel._cards[0]._counts.text()
    assert "Placed: 1" in counts_label
    # The scene shows one actor.
    assert len(window._assemble_view._block_actors) == 1

    _reset(window)


def test_placed_piece_keeps_height_through_sync(window: MainWindow) -> None:
    '''A piece placed at height renders at that height after scene syncs.

    Regression: full actor rebuilds (piece drop, tab switch, block add)
    hard-coded the surface actor z to 0, so stacked pieces sank into each
    other while the selection outline kept the correct height.
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _place(window, z=3.0)
    surface = window._assemble_view._block_actors[0][0]
    assert surface.GetPosition()[2] == pytest.approx(3.0)

    # Leaving and re-entering the tab, then a manual resync (the block-add
    # path) rebuild the scene: the height must survive every rebuild.
    window._tabs.setCurrentIndex(0)
    window._tabs.setCurrentIndex(1)
    window._sync_assemble()
    surface = window._assemble_view._block_actors[0][0]
    assert surface.GetPosition()[2] == pytest.approx(3.0)
    _reset(window)


def test_delete_piece_keeps_template(window: MainWindow) -> None:
    '''Deleting placed copies empties the scene; the part card stays.'''
    _reset(window)
    _place(window)
    _place(window)
    assert len(window._project.assemble.pieces) == 2

    window._aselection = {0, 1}
    window._aprimary = 0
    window._on_assemble_delete()

    assert window._project.assemble.pieces == []
    assert len(window._assemble_view._block_actors) == 0
    # The planned part is still listed (count 0).
    assert len(window._parts_panel._cards) == 1
    assert "Placed: 0" in window._parts_panel._cards[0]._counts.text()
    _reset(window)


def test_linked_quantity_follows_assemble(window: MainWindow) -> None:
    '''Linked quantity: place -> quantity 1, delete all -> quantity 0;
    the placement scene resyncs (one piece per placed copy).'''
    _reset(window)
    window._selection = {0}
    window._primary = 0
    window._on_qty_link_toggled(True)
    assert window._project.blocks[0].quantity_linked is True
    assert window._project.blocks[0].quantity == 0
    # The quantity spin is read-only while linked.
    assert not window._block_quantity.isEnabled()

    _place(window)
    assert window._project.blocks[0].quantity == 1
    assert len(window._project.print_layout.pieces) == 1

    _place(window)
    assert window._project.blocks[0].quantity == 2
    assert len(window._project.print_layout.pieces) == 2

    # Undo path: the snapshots restore the pre-edit states wholesale.
    window._on_undo()
    assert window._project.blocks[0].quantity == 1
    assert len(window._project.assemble.pieces) == 1
    window._on_redo()
    assert window._project.blocks[0].quantity == 2
    _wait_meshes(window)

    # Manual override stays impossible while linked: quantity edit skips it.
    window._block_quantity.setValue(9)
    assert window._project.blocks[0].quantity == 2

    # Unlinking keeps the current value as the manual quantity.
    window._cb_qty_linked.setChecked(False)
    assert window._project.blocks[0].quantity == 2
    assert window._block_quantity.isEnabled()
    _reset(window)


def test_move_pieces_commit_xyz(window: MainWindow) -> None:
    '''A drag commit writes x, y and z into the pieces (undoable).'''
    _reset(window)
    _place(window)
    window._reset_edit_history()
    window._dirty = False

    window._on_assemble_pieces_moved({0: (12.0, 8.0, 4.0, 0)})
    piece = window._project.assemble.pieces[0]
    assert (piece.position_x, piece.position_y, piece.position_z) == (
        pytest.approx(12.0),
        pytest.approx(8.0),
        pytest.approx(4.0),
    )
    assert window._dirty is True

    window._on_undo()
    piece = window._project.assemble.pieces[0]
    assert piece.position_x == pytest.approx(0.0)
    window._on_redo()
    _wait_meshes(window)
    window._reset_edit_history()
    _reset(window)


def test_selection_syncs_inspector(window: MainWindow) -> None:
    '''Selecting a piece enables the inspector and mirrors its state.'''
    _reset(window)
    _place(window, x=5.0, y=6.0, z=7.0, rot_z=2, flipped=True)

    window._on_assemble_selection((frozenset({0}), 0))
    assert window._aselection == {0}
    assert window._assemble_panel._inspector_group.isEnabled()
    assert window._assemble_panel._spin_piece_x.value() == pytest.approx(5.0)
    assert window._assemble_panel._spin_piece_z.value() == pytest.approx(7.0)
    assert window._assemble_panel._combo_rotation.currentIndex() == 2
    assert window._assemble_panel._cb_flip.isChecked()

    window._on_assemble_selection((frozenset(), None))
    assert not window._assemble_panel._inspector_group.isEnabled()
    _reset(window)


def test_inspector_edits_apply(window: MainWindow) -> None:
    '''Position / rotation / flip edits from the panel reach the pieces.'''
    _reset(window)
    _place(window)
    window._on_assemble_selection((frozenset({0}), 0))
    window._dirty = False

    window._on_assemble_position_changed(3.0, 4.0, 5.0)
    piece = window._project.assemble.pieces[0]
    assert (piece.position_x, piece.position_y, piece.position_z) == (
        pytest.approx(3.0),
        pytest.approx(4.0),
        pytest.approx(5.0),
    )

    window._on_assemble_rotation_changed(1)
    assert piece.rot_z == 1

    window._on_assemble_flip_changed(True)
    assert piece.flipped is True

    _reset(window)


def test_rotated_piece_uses_variant_mesh(window: MainWindow) -> None:
    '''A rotated/flipped piece renders from the orientation variant cache.'''
    _reset(window)
    _place(window, rot_z=1, flipped=True)
    window._sync_assemble()

    assert (0, 1, True) in window._assemble_variants
    assert window._assemble_variants[(0, 1, True)].n_points > 0
    assert len(window._assemble_view._block_actors) == 1
    _reset(window)


def test_prune_stale_pieces_on_block_removal(window: MainWindow) -> None:
    '''Removing the planning block drops its assembled copies on sync.'''
    _reset(window)
    window._project.blocks.append(
        BlockInstance(uid="blk2", cells_x=2, cells_y=2, quantity=1)
    )
    _place(window, block_index=1)
    assert len(window._project.assemble.pieces) == 1

    window._selection = {1}
    window._primary = 1
    window._on_remove_blocks()
    window._sync_assemble()

    assert window._project.assemble.pieces == []
    assert len(window._parts_panel._cards) == 1  # only block 0 remains
    _reset(window)


def test_snap_settings_reach_view_and_project(window: MainWindow) -> None:
    '''The panel snapping toggles store per-project and reach the view.'''
    _reset(window)
    window._on_assemble_snap_grid(False)
    assert window._project.assemble.view.snap_to_grid is False
    assert window._assemble_view._grid_snap is False

    window._on_assemble_snap_parts(False)
    assert window._project.assemble.view.snap_to_parts is False
    assert window._assemble_view._snap_parts is False
    assert window._dirty is True

    window._on_assemble_snap_grid(True)
    window._on_assemble_snap_parts(True)
    _reset(window)


def test_select_all_assemble_pieces(window: MainWindow) -> None:
    '''Ctrl+A on the assemble tab selects every placed copy.'''
    _reset(window)
    _place(window)
    _place(window)
    window._tabs.setCurrentIndex(1)
    window._on_select_all()
    assert window._aselection == {0, 1}
    assert window._aprimary == 0
    _reset(window)


def test_place_requested_starts_ghost(window: MainWindow) -> None:
    '''A card click enters the ghost placement mode (mesh ready).'''
    _reset(window)
    window._sync_assemble_panel()
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert window._assemble_view.placing() is True
    assert window._assemble_view._ghost_actor is not None

    # Leaving the tab cancels the modal placement.
    window._tabs.setCurrentIndex(0)
    assert window._assemble_view.placing() is False

    # A stale uid is ignored silently.
    window._on_parts_place_requested("no-such-uid")
    _reset(window)


def test_scene_meta_and_ghost_carry_material_rects(window: MainWindow) -> None:
    '''Excluded-cell pieces expose their voids to the drop logic.

    Regression for the T-notch bug: scene meta and the placement ghost
    must carry the material rectangles, otherwise the drop logic silently
    falls back to the bounding box and pieces can never enter the empty
    space of a notch. The scene meta and ghost rects are derived from the
    block settings, so no regeneration is needed (the stale 2x2 mesh is
    irrelevant to them).
    '''
    _reset(window)
    block = window._project.blocks[0]
    block.cells_x = 3
    block.cells_y = 2
    block.excluded_cells = [[2, 0]]
    window._tabs.setCurrentIndex(1)
    unit = window._project.settings.unit_size
    _place(window, x=0.0, y=0.0)

    view = window._assemble_view
    meta = view._scene_meta[0]
    assert len(meta['rects']) == 2
    # rot 0 / unflipped: the current rects are the base rects.
    assert meta['rects'] == meta['base_rects']
    # The support provider reports the same world rectangles.
    supports = view._query_supports(set())
    assert len(supports) == 1
    assert len(supports[0].rects) == 2
    assert supports[0].rects[0].cx == pytest.approx(-unit / 2)

    # The placement ghost starts with the same material rectangles.
    window._on_parts_place_requested(block.uid)
    assert view.placing() is True
    assert view._ghost_rects == view._ghost_base_rects
    assert view._ghost_base_rects == tuple(meta['base_rects'])
    window._tabs.setCurrentIndex(0)

    # Restore the default full-square block geometry.
    block.cells_x = 2
    block.cells_y = 2
    block.excluded_cells = []
    window._sync_assemble()
    _reset(window)


def test_assemble_panel_status_routing(window: MainWindow) -> None:
    '''Status messages on the assemble tab land in its own panel label.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    window._set_status("hello assemble")
    assert window._assemble_panel._status_label.text() == "hello assemble"
    window._tabs.setCurrentIndex(0)
    window._set_status("hello planning")
    assert window._status.text() == "hello planning"
    _reset(window)


def test_copy_paste_starts_persistent_ghost(window: MainWindow) -> None:
    '''Paste starts a persistent ghost; one drop appends and dirties.

    The Assemble tab now uses free 3D placement: paste attaches the first
    copied piece's pose to the ghost, drops land via _drop_ghost →
    piece_placed → _on_assemble_piece_placed, and the ghost STAYS until
    Esc / RMB (cancel_placement).
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _place(window, x=5.0, z=7.0, rot_z=2, flipped=True)
    # Select piece 0 and copy it.
    window._aselection = {0}
    window._aprimary = 0
    window._on_copy()
    window._reset_edit_history()
    window._dirty = False
    n0 = len(window._project.assemble.pieces)

    # Paste → ghost active, nothing appended yet.
    window._on_paste()
    view = window._assemble_view
    assert view.placing() is True
    assert view._ghost_actor is not None
    # Ghost carries the copied pose.
    assert view._ghost_rot == 2
    assert view._ghost_flip is True
    assert view._ghost_z == pytest.approx(7.0)
    assert view._ghost_block == 0
    # Nothing appended yet (paste only starts the ghost).
    assert len(window._project.assemble.pieces) == n0
    assert window._dirty is False

    # Drop: set ghost position directly and call _drop_ghost.
    # Piece 0 is at (5,0,7) with half_x=8mm → occupies x ∈ [-3,13].
    # Use x=20 to land in non-overlapping space.
    pieces = window._project.assemble.pieces
    view._ghost_pos = (20.0, 0.0, 7.0)
    view._drop_ghost()
    assert len(pieces) == n0 + 1
    new_piece = pieces[-1]
    # New uid != source uid.
    assert new_piece.uid != pieces[0].uid
    # Position / orientation match the ghost pose.
    assert new_piece.position_x == pytest.approx(20.0)
    assert new_piece.position_y == 0.0
    assert new_piece.position_z == pytest.approx(7.0)
    assert new_piece.rot_z == 2
    assert new_piece.flipped is True
    # Ghost PERSISTS after one drop (explicit requirement).
    assert view.placing() is True
    # The drop was an undoable edit.
    assert window._dirty is True

    # Cancel → ghost gone.
    view.cancel_placement()
    assert view.placing() is False

    # Undo the placement drop.
    window._on_undo()
    pieces = window._project.assemble.pieces
    assert len(pieces) == n0
    _wait_meshes(window)
    window._reset_edit_history()
    _reset(window)


def test_paste_assembled_empty_and_stale(window: MainWindow) -> None:
    '''Empty clipboard / removed source blocks report instead of pasting.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    window._on_paste()
    assert "Clipboard is empty" in window._assemble_panel._status_label.text()
    # Nothing was placed; ghost mode is not active.
    assert window._assemble_view.placing() is False

    window._clip_assembled = [{"block_uid": "gone", "position_x": 0.0}]
    window._on_paste()
    assert (
        "copied blocks no longer exist" in window._assemble_panel._status_label.text()
    )
    assert window._assemble_view.placing() is False
    assert window._project.assemble.pieces == []
    _reset(window)


def test_paste_assembled_updates_linked_quantity(window: MainWindow) -> None:
    '''Pasting copies of a linked block bumps its quantity too.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _place(window)
    window._selection = {0}
    window._primary = 0
    window._on_qty_link_toggled(True)
    assert window._project.blocks[0].quantity == 1

    # Select piece 0, copy, paste → ghost active.
    window._aselection = {0}
    window._aprimary = 0
    window._on_copy()
    window._on_paste()
    view = window._assemble_view
    assert view.placing() is True

    # Drop at non-overlapping position (piece 0 is at origin, default block
    # half-extent is 4mm → occupies x ∈ [-4,4], y ∈ [-4,4]).
    view._ghost_pos = (16.0, 0.0, 0.0)
    view._drop_ghost()

    # Now: piece count +1, linked quantity updated, placement scene follows.
    assert len(window._project.assemble.pieces) == 2
    assert window._project.blocks[0].quantity == 2
    assert len(window._project.print_layout.pieces) == 2

    view.cancel_placement()
    _reset(window)


def test_paste_multi_copies_ghosts_first(window: MainWindow) -> None:
    '''Multi-piece copy: ghost attaches to the FIRST copied piece's block.'''
    _reset(window)
    # Add a second block (mirror test_prune_stale_pieces_on_block_removal).
    window._project.blocks.append(
        BlockInstance(uid="blk2", cells_x=2, cells_y=2, quantity=1)
    )
    # Place pieces of both blocks.
    _place(window, block_index=0)
    _place(window, block_index=1)
    assert len(window._project.assemble.pieces) == 2

    # Select {0, 1} (piece indices), primary = 0.
    window._aselection = {0, 1}
    window._aprimary = 0
    window._on_copy()
    window._on_paste()

    view = window._assemble_view
    # Ghost attaches to the FIRST copied piece's block (index 0).
    assert view._ghost_block == 0
    assert view.placing() is True
    # Status message mentions remaining copies on clipboard.
    status = window._assemble_panel._status_label.text()
    assert "stay on the clipboard" in status

    # Cancel and prune the second block (mirror the prune test).
    view.cancel_placement()
    window._selection = {1}
    window._primary = 1
    window._on_remove_blocks()
    window._sync_assemble()
    _reset(window)


def test_ghost_blocked_drop_refused(window: MainWindow) -> None:
    '''Collision prohibition: intersecting ghost is refused, tinted red.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    # Place piece 0 at origin.
    _place(window)
    bh = window._project.block_settings(0).block_height

    # Start placement from the parts panel → ghost active.
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    view = window._assemble_view
    assert view.placing() is True
    n_before = len(window._project.assemble.pieces)

    # Drop exactly on the existing piece → intersecting → refused.
    view._ghost_pos = (0.0, 0.0, 0.0)
    assert view._ghost_valid_at(0.0, 0.0, 0.0) is False
    view._drop_ghost()
    assert len(window._project.assemble.pieces) == n_before

    # Ghost tinted red (COLLISION_COLOR = "#d02020").
    # _update_ghost wasn't called after setting _ghost_pos, so trigger tint.
    assert view._ghost_actor is not None
    view._tint_ghost(not view._ghost_valid_at(0.0, 0.0, 0.0))
    color_hex = _actor_color_hex(view._ghost_actor)
    assert color_hex[:7] == "#d02020"

    # Move to the legal stack plane (z = block_height).
    view._ghost_pos = (0.0, 0.0, bh)
    assert view._ghost_valid_at(0.0, 0.0, bh) is True
    view._drop_ghost()
    assert len(window._project.assemble.pieces) == n_before + 1
    new_piece = window._project.assemble.pieces[-1]
    assert new_piece.position_z == pytest.approx(bh)

    view.cancel_placement()
    _reset(window)


def test_ghost_wheel_steps_working_plane(window: MainWindow) -> None:
    '''Wheel steps the working plane during placement; clamped at 0.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    view = window._assemble_view
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert view.placing() is True

    step = view.grid_step()
    # Scroll up (positive delta) → height increases.
    ev = _wheel(view, 120)
    assert view._ghost_z == pytest.approx(step)
    assert ev.isAccepted()

    # Scroll down (negative delta) → height decreases back.
    ev = _wheel(view, -120)
    assert view._ghost_z == pytest.approx(0.0)

    # Extra scroll down stays at 0 (clamped).
    ev = _wheel(view, -120)
    assert view._ghost_z == pytest.approx(0.0)
    assert ev.isAccepted()

    view.cancel_placement()
    _reset(window)


def test_drag_collision_commits_last_valid(window: MainWindow) -> None:
    '''Dragging onto a piece tints red; confirm falls back to last valid.'''
    _reset(window)
    _wait_meshes(window)

    # Place piece 0 at origin.
    _place(window)
    unit = window._project.settings.unit_size
    # Place piece 1 aside — 3*unit clears the half-extent of a 2x2 block.
    _place(window, x=3 * unit)
    assert len(window._project.assemble.pieces) == 2

    view = window._assemble_view
    # Select piece 1 and begin drag.
    view.set_selection({1}, 1)
    view._grabbing = True
    view._grab_keyboard = False
    view._grab_orig = {1: (3 * unit, 0.0)}
    view._begin_piece_drag()

    # Drive piece 1 onto piece 0.
    view._apply_move(view._grab_orig, -3 * unit, 0.0)
    assert view._drag_pose_invalid() is True
    # Outline tinted red (COLLISION_COLOR = "#d02020").
    color_hex = _actor_color_hex(view._outline_actors[1])
    assert color_hex[:7] == "#d02020"

    # Confirm → falls back to last valid pose (original position).
    view.confirm_grab()
    piece = window._project.assemble.pieces[1]
    assert piece.position_x == pytest.approx(3 * unit)
    assert piece.position_y == 0.0
    assert piece.rot_z == 0
    # Outline back to selection color (#ff9d2e).
    color_hex = _actor_color_hex(view._outline_actors[1])
    assert color_hex[:7] == "#ff9d2e"
    assert window._dirty is True

    _reset(window)


def test_move_keeps_height_and_rotation_keeps_height(window: MainWindow) -> None:
    '''No auto-stacking: XY moves and rotations preserve the current height.'''
    _reset(window)
    _wait_meshes(window)

    # Place piece at z=7.0 (non-zero height).
    _place(window, z=7.0)
    view = window._assemble_view

    # Fake drag state for piece 0.
    view.set_selection({0}, 0)
    view._grabbing = True
    view._grab_keyboard = False
    view._grab_orig = {0: (0.0, 0.0)}
    view._begin_piece_drag()

    # Move XY → height must stay at 7.0.
    view._apply_move(view._grab_orig, 5.0, 3.0)
    assert view._scene_meta[0]['pos_z'] == pytest.approx(7.0)
    assert view._scene_meta[0]['pos_x'] == pytest.approx(5.0)
    assert view._scene_meta[0]['pos_y'] == pytest.approx(3.0)

    # Space rotate → height must still be 7.0 (old code re-resolved it).
    from PySide6.QtGui import QKeyEvent
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier
    )
    view.keyPressEvent(event)
    assert view._scene_meta[0]['rot_z'] == 1
    assert view._scene_meta[0]['pos_z'] == pytest.approx(7.0)

    # Cancel → fully restored: z=7.0 and the orientation rolls back too
    # (rot_z, halves and rects return to the drag start).
    view.cancel_grab()
    assert view._scene_meta[0]['pos_z'] == pytest.approx(7.0)
    assert view._scene_meta[0]['rot_z'] == 0

    _reset(window)


def test_placement_and_grab_cancel_with_rmb_and_esc(window: MainWindow) -> None:
    '''Placement cancel is RMB or Esc; G-grab cancel also accepts both.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    view = window._assemble_view

    # --- Placement cancel via RMB ---
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert view.placing() is True
    QTest.mousePress(view, Qt.MouseButton.RightButton)
    QApplication.processEvents()
    QTest.mouseRelease(view, Qt.MouseButton.RightButton)
    QApplication.processEvents()
    assert view.placing() is False

    # --- Placement cancel via Esc ---
    window._on_parts_place_requested(uid)
    assert view.placing() is True
    esc_event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(QApplication.focusWidget(), esc_event)
    assert view.placing() is False

    # --- G-grab cancel via RMB ---
    _place(window)
    window._aselection = {0}
    window._aprimary = 0
    view.set_selection({0}, 0)
    view.start_grab()
    assert view._grabbing is True
    QTest.mousePress(view, Qt.MouseButton.RightButton)
    QApplication.processEvents()
    QTest.mouseRelease(view, Qt.MouseButton.RightButton)
    QApplication.processEvents()
    assert view._grabbing is False

    _reset(window)


def test_space_rotates_moved_pieces(window: MainWindow) -> None:
    '''Space mid-drag rotates moved pieces 90 degrees; commit keeps it.'''
    _reset(window)
    _wait_meshes(window)
    _place(window)
    window._sync_assemble()
    window._reset_edit_history()
    window._dirty = False

    view = window._assemble_view
    view._grabbing = True
    view._grab_keyboard = False
    view._grab_orig = {0: (0.0, 0.0)}
    view._begin_piece_drag()

    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier
    )
    view.keyPressEvent(event)
    assert event.isAccepted()
    meta = view._scene_meta[0]
    assert meta["rot_z"] == 1

    view.confirm_grab()
    piece = window._project.assemble.pieces[0]
    assert piece.rot_z == 1
    assert window._dirty is True

    window._on_undo()
    assert window._project.assemble.pieces[0].rot_z == 0
    _wait_meshes(window)
    window._reset_edit_history()
    _reset(window)


def test_space_rotates_ghost_during_placement(window: MainWindow) -> None:
    '''Space during the ghost placement rotates the ghost like R.'''
    _reset(window)
    _wait_meshes(window)
    window._on_parts_place_requested(window._project.blocks[0].uid)
    view = window._assemble_view
    assert view.placing() is True
    assert view._ghost_rot == 0

    for expected in (1, 2, 3, 0):
        event = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier
        )
        view.keyPressEvent(event)
        assert view._ghost_rot == expected
    # R keeps working alongside Space.
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier
    )
    view.keyPressEvent(event)
    assert view._ghost_rot == 1

    view.cancel_placement()
    _reset(window)


def test_placement_focuses_view_for_hotkeys(window: MainWindow) -> None:
    '''Regression: R/Space/F were dead on the first placed ghost.

    Placement starts from a press on a panel card, so the viewport must
    take the keyboard focus itself; previously the hotkeys only worked
    after the first drop click (which focused the view) — i.e. of two
    same-type parts placed in a row the first never rotated.
    '''
    _reset(window)
    _wait_meshes(window)
    window._tabs.setCurrentIndex(1)
    view = window._assemble_view
    window._on_parts_place_requested(window._project.blocks[0].uid)
    assert view.placing() is True
    # The viewport holds the keyboard focus right away.
    assert QApplication.focusWidget() is view

    # A key press through the application focus (what a user's R becomes)
    # rotates the ghost.
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(QApplication.focusWidget(), event)
    assert view._ghost_rot == 1

    view.cancel_placement()
    _reset(window)


def test_panel_click_while_placing_never_drops(window: MainWindow) -> None:
    '''Regression: while placing, a press on the parts panel dropped a copy.

    The viewport used to grab the mouse for the whole placement mode, so
    panel presses were redirected into its LMB drop handler and landed a
    piece at the ghost's stale position. The viewport must not hold the
    grab: panel presses stay with the panel (switching parts, toggles).
    '''
    _reset(window)
    _wait_meshes(window)
    window._tabs.setCurrentIndex(1)
    view = window._assemble_view
    window._on_parts_place_requested(window._project.blocks[0].uid)
    assert view.placing() is True
    assert QWidget.mouseGrabber() is not view

    # Clicking the card again while placing re-selects the part without
    # appending a piece.
    n0 = len(window._project.assemble.pieces)
    card = window._parts_panel._cards[0]
    QTest.mousePress(card, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    QTest.mouseRelease(card, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert len(window._project.assemble.pieces) == n0
    assert view.placing() is True

    view.cancel_placement()
    _reset(window)


def test_del_action_gated_to_planning(window: MainWindow) -> None:
    '''The File>Del shortcut is armed only on the planning tab so Del on
    the assemble tab deletes placed copies, not planning blocks.'''
    _reset(window)
    _place(window)
    window._selection = {0}
    window._primary = 0
    window._sync_block_inspector()

    window._tabs.setCurrentIndex(1)
    assert not window._act_remove_blocks.isEnabled()
    window._tabs.setCurrentIndex(0)
    assert window._act_remove_blocks.isEnabled()
    _reset(window)
