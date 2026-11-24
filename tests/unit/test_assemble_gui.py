'''
File:   test_assemble_gui.py
Brief:  Integration tests for the Assemble tab (MainWindow level).
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import os
import time
from collections.abc import Iterator

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent  # noqa: E402
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
        QPointF(10.0, 10.0),
        QPointF(10.0, 10.0),
        QPoint(0, 0),
        QPoint(0, dy),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
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


def test_paste_multi_drops_whole_group(window: MainWindow) -> None:
    '''Multi-piece copy: one drop appends ALL copied pieces atomically.

    Two pieces of the same block at different positions are copied, pasted
    (ghost starts), then ONE drop → exactly 2 new pieces appended with
    correct relative offsets and orientations preserved. One undo step
    removes both. Status no longer mentions "stay on the clipboard".
    '''
    _reset(window)
    unit = window._project.settings.unit_size  # 4.0 mm default

    # Place two pieces of block 0 at different positions (~40mm apart).
    _place(window, x=0.0, y=0.0)
    _place(window, x=10.0 * unit, y=5.0 * unit)
    assert len(window._project.assemble.pieces) == 2

    # Select both pieces and copy them.
    window._aselection = {0, 1}
    window._aprimary = 0
    window._on_copy()
    window._reset_edit_history()
    window._dirty = False
    n_before = len(window._project.assemble.pieces)

    # Paste → ghost active.
    window._on_paste()
    view = window._assemble_view
    assert view.placing() is True
    # Ghost attaches to the FIRST copied piece's block (index 0).
    assert view._ghost_block == 0
    # Status message no longer mentions clipboard.
    status = window._assemble_panel._status_label.text()
    assert "stay on the clipboard" not in status
    assert f"group of {n_before} piece(s)" in status

    # One drop: set ghost position directly and call _drop_ghost.
    # Piece 0 is at (0,0), piece 1 at (40,20). Ghost starts at origin.
    # Drop at offset (20, 10) → piece 0 lands at (20,10), piece 1 at
    # (60,30) — non-overlapping with originals or each other.
    view._ghost_pos = (20.0, 10.0, 0.0)
    view._drop_ghost()

    assert len(window._project.assemble.pieces) == n_before + 2
    new_pieces = window._project.assemble.pieces[n_before:]
    assert len(new_pieces) == 2

    # Both pieces reference the same block uid.
    for p in new_pieces:
        assert p.block_uid == window._project.blocks[0].uid

    # Positions: anchor at ghost pos, second at offset from anchor.
    # Piece n_before (anchor): x=20, y=10 (ghost pose).
    assert new_pieces[0].position_x == pytest.approx(20.0)
    assert new_pieces[0].position_y == pytest.approx(10.0)
    # Piece n_before+1 (offset): x = 20 + (40-0) = 60, y = 10 + (20-0) = 30
    assert new_pieces[1].position_x == pytest.approx(60.0)
    assert new_pieces[1].position_y == pytest.approx(30.0)

    # Orientations preserved (both were placed with default rot_z=0, flipped=False).
    assert new_pieces[0].rot_z == 0
    assert new_pieces[0].flipped is False
    assert new_pieces[1].rot_z == 0
    assert new_pieces[1].flipped is False

    # Ghost PERSISTS after one drop.
    assert view.placing() is True
    assert window._dirty is True

    # Cancel → ghost gone.
    view.cancel_placement()
    assert view.placing() is False

    # One undo step removes BOTH pieces.
    window._on_undo()
    pieces = window._project.assemble.pieces
    assert len(pieces) == n_before
    _wait_meshes(window)
    window._reset_edit_history()
    _reset(window)


def test_paste_group_refused_on_collision(window: MainWindow) -> None:
    '''Multi-piece paste group collides with existing piece → drop refused.

    Arrange an existing piece so the pasted group would overlap it. One
    drop is refused, piece count unchanged, and the ghost survives (placing
    still True) for a retry.
    '''
    _reset(window)
    unit = window._project.settings.unit_size  # 4.0 mm

    # Place two pieces of block 0 at different positions.
    _place(window, x=0.0, y=0.0)
    _place(window, x=10.0 * unit, y=5.0 * unit)
    assert len(window._project.assemble.pieces) == 2

    # Select both and copy.
    window._aselection = {0, 1}
    window._aprimary = 0
    window._on_copy()
    window._reset_edit_history()
    window._dirty = False
    n_before = len(window._project.assemble.pieces)

    # Paste → ghost active.
    window._on_paste()
    view = window._assemble_view
    assert view.placing() is True

    # Drop at origin (overlaps piece 0). Both pieces in the group overlap.
    view._ghost_pos = (0.0, 0.0, 0.0)
    view._drop_ghost()

    # Refused: no new pieces appended.
    assert len(window._project.assemble.pieces) == n_before
    # Ghost still active for retry.
    assert view.placing() is True
    # Ghost tinted red (collision).
    assert view._ghost_actor is not None
    color_hex = _actor_color_hex(view._ghost_actor)
    assert color_hex[:7] == "#d02020"

    # Move to a legal position and drop again.
    view._ghost_pos = (30.0, 15.0, 0.0)
    view._drop_ghost()
    assert len(window._project.assemble.pieces) == n_before + 2

    view.cancel_placement()
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


def test_ghost_wheel_zooms_without_stepping_height(window: MainWindow) -> None:
    '''Wheel no longer steps the ghost height: it zooms natively; z stays.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    view = window._assemble_view
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert view.placing() is True
    _ = view.grid_step()  # verify grid_step accessible during placement
    ev = _wheel(view, 120)
    assert view._ghost_z == pytest.approx(0.0)
    assert ev.isAccepted()
    ev = _wheel(view, -120)
    assert view._ghost_z == pytest.approx(0.0)
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


def test_space_rotates_group_about_common_center(window: MainWindow) -> None:
    '''Multi-selection Space rotates the group rigidly about the bbox centre.

    Two pieces with DIFFERENT footprints are placed so that mean-of-centers
    pivot differs from AABB-center pivot — this exercises the fix for the
    round-1 bug where the pivot was computed as mean of centres instead of
    the selection bounding-box centre (which is what MoveGizmo draws at).

    Block 0: default 1x1 → piece 0 at pos=(0,0), half_x=2.
    Block 1: 3x1 cells  → piece 1 at pos=(12,0), half_x=6.

    AABB center in X = ((-2)+18)/2 = 8 (not mean-of-centers which would be
    (0+12)/2 = 6).  This makes the test fail if the old mean-pivot code runs.
    '''
    _reset(window)
    _wait_meshes(window)

    unit = window._project.settings.unit_size  # 4.0 mm

    # Ensure block 0 is the default 1x1 (some earlier tests modify cells).
    block0 = window._project.blocks[0]
    block0.cells_x = 1
    block0.cells_y = 1

    # Add a second block with different footprint (3x1 cells) so the two
    # pieces have different half-extents, making AABB-centre pivot differ from
    # mean-of-centers pivot.
    from brickgen.core.project import BlockInstance as BI

    window._project.blocks.append(BI(uid="blk2", cells_x=3, cells_y=1, quantity=1))

    # Trigger regeneration so both blocks have fresh meshes and correct dims.
    window._mark_scene_dirty()
    window._applied_context = None  # clear stale context from initial gen
    window._on_generate()
    _wait_applied(window)

    view = window._assemble_view

    # Place piece 0 at origin using block 0 (1x1, half_x=2).
    _place(window, x=0.0)
    # Place piece 1 at x=12 using block 1 (3x1, half_x=6).
    _place(window, block_index=1, x=3 * unit)

    # Now sync the assemble scene — this rebuilds actors from the project
    # pieces using the freshly regenerated meshes.
    window._sync_assemble()
    window._reset_edit_history()
    window._dirty = False

    assert len(view._scene_meta) == 2
    for i, m in enumerate(view._scene_meta):
        assert abs(m['pos_x']) < 1e-6 or abs(m['pos_x'] - 3 * unit) < 1e-6

    # Select both pieces and start a drag.
    view.set_selection({0, 1}, 0)
    view._grabbing = True
    view._grab_keyboard = False
    view._grab_orig = {
        0: (0.0, 0.0),
        1: (3 * unit, 0.0),
    }
    view._begin_piece_drag()

    # Press Space mid-drag.
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier
    )
    view.keyPressEvent(event)
    assert event.isAccepted()

    # --- Verify _scene_meta during drag ---
    for idx in (0, 1):
        meta = view._scene_meta[idx]
        assert meta['rot_z'] == 1
        # Halves must have been swapped. Piece 0 is square so halves stay equal;
        # piece 1 was rectangular (6×2) and should now be (2×6).

    # Compute expected pivot: AABB centre of pre-rotation bounds.
    # Piece 0 (1x1, pos=0): x in [-2, 2], y in [-2, 2]
    # Piece 1 (3x1, pos=12): x in [6, 18], y in [-2, 2]
    # AABB centre = ((-2)+18)/2, (-2+2)/2 = (8, 0).
    # Piece 0: dx=0-8=-8, dy=0 → new=(8-0, 0+(-8)) = (8, -8)
    # Piece 1: dx=12-8=4,  dy=0 → new=(8-0, 0+4)     = (8, 4)
    expected = {
        0: (8.0, -8.0),
        1: (8.0, 4.0),
    }
    for idx in (0, 1):
        meta = view._scene_meta[idx]
        ex, ey = expected[idx]
        dx_val = abs(meta['pos_x'] - ex)
        dy_val = abs(meta['pos_y'] - ey)
        assert dx_val < 1e-6, f"Piece {idx} pos_x: {meta['pos_x']} != {ex}"
        assert dy_val < 1e-6, f"Piece {idx} pos_y: {meta['pos_y']} != {ey}"

    # Verify group bbox extents swap.
    xs = [view._scene_meta[i]['pos_x'] - view._scene_meta[i]['half_x'] for i in (0, 1)]
    ys = [view._scene_meta[i]['pos_y'] - view._scene_meta[i]['half_y'] for i in (0, 1)]
    xe = [view._scene_meta[i]['pos_x'] + view._scene_meta[i]['half_x'] for i in (0, 1)]
    ye = [view._scene_meta[i]['pos_y'] + view._scene_meta[i]['half_y'] for i in (0, 1)]
    width = max(xe) - min(xs)
    height = max(ye) - min(ys)
    # Pre-rotation: x-range [-2,18] → w=20; y-range [-2,2] → h=4.
    # Post-rot: should swap to w=4, h=20.
    assert abs(width - 4.0) < 1e-6, f"Post-rot width: {width} != 4.0"
    assert abs(height - 20.0) < 1e-6, f"Post-rot height: {height} != 20.0"

    # Commit the drag: pieces_moved must carry the rotated positions.
    view.confirm_grab()
    for idx in (0, 1):
        piece = window._project.assemble.pieces[idx]
        assert piece.rot_z == 1
        ex, ey = expected[idx]
        assert abs(piece.position_x - ex) < 1e-6
        assert abs(piece.position_y - ey) < 1e-6
    assert window._dirty is True

    # Undo restores pre-drag positions and rot_z=0.
    window._on_undo()
    _wait_meshes(window)
    for idx in (0, 1):
        piece = window._project.assemble.pieces[idx]
        assert piece.rot_z == 0
    assert abs(window._project.assemble.pieces[0].position_x - 0.0) < 1e-6
    assert abs(window._project.assemble.pieces[1].position_x - 3 * unit) < 1e-6
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


# ---------------------------------------------------------------- axis-lock


def _hover_move(widget: QWidget, pos: QPoint) -> None:
    '''A plain mouse-move event (no buttons held) at *pos*.'''
    widget._pick_scene_point = lambda pos=None: None  # type: ignore[method-assign]
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, move)


def _assemble_stub_ray(view, target_x: float, target_y: float, target_z: float = 2.5):
    '''Monkeypatch ``view._cursor_ray`` for deterministic ray targeting *target*.

    The camera origin is offset sideways so rays never run parallel to the
    Z axis line (the single-axis closest-point guard would reject them).
    Also stubs _view_plane_normal to (0,0,1) so existing tests keep legacy
    horizontal-plane semantics.
    '''
    origin = np.array([150.0, 120.0, 400.0], dtype=np.float64)
    target = np.array([target_x, target_y, target_z], dtype=np.float64)
    direction = target - origin
    view._cursor_ray = lambda pos=None: (origin, direction)
    view._view_plane_normal = lambda: np.array([0.0, 0.0, 1.0])  # type: ignore[method-assign]


# ------------------------------------------------------------------ Z lock lift


def test_drag_z_lock_lifts_piece(window: MainWindow) -> None:
    '''Z-axis lock during a grab lifts the piece along world-Z.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    _place(window, z=0.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    view._begin_piece_drag()

    # Engage the lock with a near-anchor ray (absorbs into the origin).
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    assert view._axis_lock == "z"
    _hover_move(view, QPoint(200, 150))

    # Retarget the ray high above: constrained delta lifts z only.
    _assemble_stub_ray(view, 10.0, 5.0, 100.0)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    assert meta['pos_z'] > 0.5
    # X/Y unchanged in Z-lock mode.
    assert meta['pos_x'] == pytest.approx(0.0, abs=1e-6)
    assert meta['pos_y'] == pytest.approx(0.0, abs=1e-6)

    view.cancel_grab()
    _reset(window)


# ------------------------------------------------------------- Z clamp floor


def test_drag_z_lock_clamps_at_floor(window: MainWindow) -> None:
    '''Z-lock drag below z=0 clamps the piece at the floor.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    _place(window, z=0.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    view._begin_piece_drag()

    # Engage the lock with a near-anchor ray first.
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    assert view._axis_lock == "z"
    _hover_move(view, QPoint(200, 150))

    # Retarget the ray well below floor (z negative): clamped at z=0.
    _assemble_stub_ray(view, 10.0, 5.0, -50.0)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    # pos_z clamped >= 0.
    assert meta['pos_z'] >= 0.0
    # X/Y unchanged.
    assert meta['pos_x'] == pytest.approx(0.0, abs=1e-6)

    view.cancel_grab()
    _reset(window)


# -------------------------------------------------------- plane lock XY


def test_shift_z_lock_moves_xy_not_z(window: MainWindow) -> None:
    '''Shift+Z (plane mode) excludes Z; only X/Y change.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    _place(window, z=5.0)  # start at non-zero height

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    view._begin_piece_drag()

    # Engage the plane lock with a near-anchor ray first.
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", True)
    assert view._axis_lock == "z"
    assert view._axis_plane is True
    _hover_move(view, QPoint(200, 150))

    # Retarget the ray offset in all axes: X/Y move, Z stays frozen.
    _assemble_stub_ray(view, 10.0, 15.0, 80.0)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    # Z unchanged (locked axis excluded).
    assert meta['pos_z'] == pytest.approx(5.0, abs=1e-6)
    # X/Y moved.
    assert meta['pos_x'] != pytest.approx(0.0, abs=1e-9)
    assert meta['pos_y'] != pytest.approx(0.0, abs=1e-9)

    view.cancel_grab()
    _reset(window)


# ------------------------------------------------------- ghost Z lock snap


def test_ghost_z_lock_constrains_and_snaps_corner(window: MainWindow) -> None:
    '''During placement, Z-lock constrains ghost to Z with plain corner snapping.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    view = window._assemble_view
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert view.placing() is True

    # Engage the lock with a near-anchor ray first (absorbed at engagement).
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    assert view._axis_lock == "z"
    _hover_move(view, QPoint(200, 150))

    # Retarget the ray at z=20: ghost z follows, X/Y stay at the anchor.
    _assemble_stub_ray(view, 10.0, 5.0, 20.0)
    _hover_move(view, QPoint(200, 150))
    assert view._ghost_pos is not None
    gx, gy, gz = view._ghost_pos
    # Ghost z moved up (from 0 toward the ray target).
    assert gz > 0.5
    # X/Y unchanged (locked axis excluded).
    assert gx == pytest.approx(0.0, abs=1e-6)
    assert gy == pytest.approx(0.0, abs=1e-6)

    # Unlock Z: placement should still be active.
    view._set_axis_lock("z", False)
    assert view._axis_lock is None
    assert view.placing() is True
    assert view._ghost_actor is not None

    view.cancel_placement()
    _reset(window)


# ----------------------------------------------------- plane lock YZ


def test_ghost_plane_lock_x_moves_yz(window: MainWindow) -> None:
    '''During placement, Shift+X (plane mode) locks X; ghost moves in Y/Z.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    view = window._assemble_view
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert view.placing() is True

    # Engage the plane lock with a near-anchor ray first.
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("x", True)
    assert view._axis_lock == "x"
    assert view._axis_plane is True
    _hover_move(view, QPoint(200, 150))

    # Retarget the ray offset in all axes: X frozen, Y/Z move.
    _assemble_stub_ray(view, 10.0, 15.0, 30.0)
    _hover_move(view, QPoint(200, 150))
    assert view._ghost_pos is not None
    gx, gy, gz = view._ghost_pos
    # X frozen at anchor.
    assert gx == pytest.approx(0.0, abs=1e-6)
    # Y/Z moved.
    assert gy != pytest.approx(0.0, abs=1e-9) or gz != pytest.approx(0.0, abs=1e-9)

    view.cancel_placement()
    _reset(window)


# ---------------------------------- unnormalized ray plane lock regression


def test_shift_plane_lock_unnormalized_ray_no_flyoff(window: MainWindow) -> None:
    '''Shift+Z plane lock with an unnormalized cursor ray must not fly off.

    Regression: ``_cursor_ray`` returns a direction whose length equals the
    camera-to-focal-point distance (hundreds of mm).  The old code computed
    the intersection parameter *t* against the normalised direction but then
    applied *t* to the unnormalised vector, overshooting by factor |d|.

    This test verifies that a ~30 mm in-plane cursor travel produces roughly
    30 mm of piece displacement, not hundreds.
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    _place(window, z=5.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    view._begin_piece_drag()

    # Stub _cursor_ray with a camera origin far away so direction length ~400.
    cam_origin = np.array([0.0, -60.0, 400.0], dtype=np.float64)

    def stub_ray(target_x: float, target_y: float, target_z: float = 2.5):
        target = np.array([target_x, target_y, target_z], dtype=np.float64)
        direction = target - cam_origin
        view._cursor_ray = lambda pos=None: (cam_origin, direction)  # type: ignore[method-assign]
        view._view_plane_normal = lambda: np.array([0.0, 0.0, 1.0])  # type: ignore[method-assign]

    # Engage plane lock at target A (on the engagement plane z=2.5).
    stub_ray(10.0, 5.0)
    view._set_axis_lock("z", True)
    assert view._axis_lock == "z"
    assert view._axis_plane is True
    _hover_move(view, QPoint(200, 150))

    # Retarget to target B: ~30 mm displaced in X, same Y/Z.
    stub_ray(40.0, 5.0)
    _hover_move(view, QPoint(200, 150))

    meta = view._scene_meta[0]
    expected_dx = 30.0  # target_b.x - target_a.x
    assert abs(meta['pos_x'] - expected_dx) < 5.0, (
        f"Expected pos_x near {expected_dx}, got {meta['pos_x']} — "
        "unnormalized ray caused fly-off"
    )
    # Y should barely move (targets share same Y).
    assert abs(meta['pos_y']) < 5.0
    # Z unchanged (plane lock excludes Z).
    assert meta['pos_z'] == pytest.approx(5.0, abs=1e-6)

    view.cancel_grab()
    _reset(window)


# ------------------------------------------------------- gizmo has Z axis


def test_assemble_gizmo_has_z_axis(window: MainWindow) -> None:
    '''Assemble gizmo exposes X, Y and Z axes (planning is XY only).'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    view = window._assemble_view
    # Enable the gizmo.
    view.set_show_gizmo(True)
    # Select a piece so the gizmo becomes visible.
    _place(window)
    view.set_selection({0}, 0)
    view._update_gizmo()

    assert view._gizmo is not None
    # The gizmo's internal axis list should contain z.
    assert "z" in view._gizmo._axes
    assert "x" in view._gizmo._axes
    assert "y" in view._gizmo._axes

    _reset(window)


# ----------------------------------------------------- planning gizmo XY only


def test_planning_preview_gizmo_xy_only(window: MainWindow) -> None:
    '''Planning tab PreviewWidget gizmo has only X and Y axes (no Z).'''
    window._tabs.setCurrentIndex(0)
    _reset(window)

    view = window._preview
    view.set_show_gizmo(True)
    # Select the block so gizmo becomes visible.
    view.set_selection({0}, 0)
    view._update_gizmo()

    assert view._gizmo is not None
    axes_list = view._gizmo._axes
    assert "x" in axes_list
    assert "y" in axes_list
    assert "z" not in axes_list


# ---------------------------------------------------------------- click/release drag


def test_click_piece_release_ends_drag(window: MainWindow) -> None:
    '''A plain click on a piece must not leave the move-drag state lingering.

    Regression: pressing LMB on a piece starts a grab; releasing without
    movement must call ``_finish_click`` which clears ``_grabbing`` and
    keeps the selection alive (the old code left ``_grabbing`` True).
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    # Place one piece at z=0.
    _place(window, z=0.0)

    view = window._assemble_view
    # Stub the pickers so a click lands on block 0 and no gizmo axis.
    view._pick_scene = lambda pos: ("block", 0)  # type: ignore[method-assign]
    view._gizmo_axis_at = lambda pos: None  # type: ignore[method-assign]

    # Press + release at the same QPoint (plain click).
    pt = QPoint(100, 100)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(pt),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(pt),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view, press)
    QApplication.processEvents()
    QApplication.sendEvent(view, release)
    QApplication.processEvents()

    # After release: grab ended, selection kept.
    assert view._grabbing is False
    assert view._selection == {0}
    # Cursor was released (WA_SetCursor cleared by unsetCursor).
    assert not view.testAttribute(Qt.WidgetAttribute.WA_SetCursor)

    _reset(window)


def test_axis_click_release_keeps_selection(window: MainWindow) -> None:
    '''Clicking a gizmo arrow and releasing without movement keeps selection.

    When the gizmo axis is hit, ``mousePressEvent`` calls ``start_move_drag``
    which sets ``_grabbing = True``. On release, if no real move happened,
    the selection must NOT be cleared (the old code called ``_select_clear``).
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    _place(window, z=0.0)

    view = window._assemble_view
    # Select the piece first so it has a selection before the gizmo click.
    view.set_selection({0}, 0)

    # Stub: arrow pixel beyond any block → axis hit, not block pick.
    view._gizmo_axis_at = lambda pos: "x"  # type: ignore[method-assign]
    view._pick_scene = lambda pos: None  # type: ignore[method-assign]

    pt = QPoint(100, 100)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(pt),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(pt),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(view, press)
    QApplication.processEvents()
    QApplication.sendEvent(view, release)
    QApplication.processEvents()

    assert view._grabbing is False
    assert view._selection == {0}

    _reset(window)


# ---------------------------------------------------------------- Z lock lift


def test_z_lock_engagement_keeps_lifted_height(window: MainWindow) -> None:
    '''A lifted piece must NOT jump when the Z lock engages.

    Regression: ``_set_axis_lock("z", False)`` during a grab used to
    re-resolve the piece height from a fresh ray, dropping a piece that
    was sitting at z=10 back to z=0. The fix sets the constraint origin
    so the first constrained update is a no-move (zero delta).
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    # Place a piece at z=10 (non-zero height).
    _place(window, z=10.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    # Do NOT call _begin_piece_drag manually (that was masking the bug).
    assert view._grab_orig_z == {0: 10.0}

    # Engage Z lock with a near-anchor ray (absorbed into origin).
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    _hover_move(view, QPoint(200, 150))

    # Engagement is a no-move: height stays at ~10.0.
    assert view._scene_meta[0]["pos_z"] == pytest.approx(10.0, abs=1e-6)

    # Lift via a high ray target: piece follows the Z delta.
    _assemble_stub_ray(view, 10.0, 5.0, 60.0)
    _hover_move(view, QPoint(200, 150))
    lifted = view._scene_meta[0]["pos_z"]
    assert lifted > 10.5  # should have risen above the starting z=10

    # Retarget the ray even higher: continue lifting smoothly along Z.
    _assemble_stub_ray(view, 10.0, 5.0, 200.0)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    # The piece should lift significantly above its starting height of 10.0.
    assert meta["pos_z"] > 50

    view.cancel_grab()
    _reset(window)


def test_plane_z_lock_keeps_height(window: MainWindow) -> None:
    '''Shift+Z (plane mode) keeps the height gained earlier in the same drag.

    In plane-lock mode, the constraint origin is set at the ray-plane
    intersection through the anchor, so the first constrained update
    retains the current Z offset (the "lifted" height).
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    # Place a piece at z=10.
    _place(window, z=10.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()

    # Engage Z lock (single-axis) with near-anchor ray.
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    _hover_move(view, QPoint(200, 150))

    # Lift via a high ray target: piece follows the Z delta.
    _assemble_stub_ray(view, 10.0, 5.0, 60.0)
    _hover_move(view, QPoint(200, 150))
    lifted = view._scene_meta[0]["pos_z"]
    assert lifted > 10.5  # should have risen above the starting z=10

    # Switch to plane mode (Shift+Z).
    view._set_axis_lock("z", True)
    assert view._axis_plane is True

    # Retarget a ray offset in XY: height retained, XY moves.
    _assemble_stub_ray(view, 40.0, 30.0)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    assert meta["pos_z"] == pytest.approx(lifted, abs=1e-6)
    # X/Y should have changed from the original (0, 0).
    assert meta["pos_x"] != pytest.approx(0.0, abs=1e-6)
    assert meta["pos_y"] != pytest.approx(0.0, abs=1e-6)

    view.cancel_grab()
    _reset(window)


# --------------------------------------------------------- shared gizmo flag


def test_gizmo_toggle_shared_across_tabs(window: MainWindow) -> None:
    '''The View-menu gizmo action controls one shared flag for all tabs.

    Toggling ``_act_show_gizmo`` must update ``project.view.show_gizmo``
    AND every preview widget's ``_gizmo_enabled``, regardless of which tab
    is currently active.
    '''
    _reset(window)
    # Start with gizmo enabled at the project level.
    window._project.view.show_gizmo = True
    window._apply_view_settings()

    # --- On assemble tab: toggle off ---
    window._tabs.setCurrentIndex(1)
    window._act_show_gizmo.setChecked(False)
    QApplication.processEvents()

    assert window._assemble_view._gizmo_enabled is False
    assert window._project.view.show_gizmo is False
    # Legacy per-project field untouched (the shared flag lives in view).
    assert window._project.assemble.view.show_gizmo is True

    # --- Switch to planning tab: gizmo stays off ---
    window._tabs.setCurrentIndex(0)
    QApplication.processEvents()
    assert window._act_show_gizmo.isChecked() is False
    assert window._preview._gizmo_enabled is False

    # --- Switch back to assemble: still disabled ---
    window._tabs.setCurrentIndex(1)
    QApplication.processEvents()
    assert window._assemble_view._gizmo_enabled is False

    # --- Re-enable via action: all previews follow ---
    window._act_show_gizmo.setChecked(True)
    QApplication.processEvents()
    assert window._assemble_view._gizmo_enabled is True

    window._dirty = False


# ------------------------------------------------------------------ gizmo visibility


# ------------------------------------------------------- Z snap step (block height)


def test_ghost_z_snap_uses_block_height(window: MainWindow) -> None:
    '''Ghost Z snap uses block_height * grid_scale, not grid_step.'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    view = window._assemble_view
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert view.placing() is True

    # Default: grid_step = 4.0 * 0.5 = 2.0, block_height = 3.0
    # So _z_snap_step = 3.0 * 0.5 = 1.5 (not 2.0)
    assert view._z_snap_step() == pytest.approx(1.5)
    assert view.grid_step() == pytest.approx(2.0)

    # Engage Z lock so ghost uses constrained path with _update_ghost_constrained
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    assert view._axis_lock == "z"

    # Move cursor to z=7.0: ghost z should be a multiple of _z_snap_step (1.5)
    _assemble_stub_ray(view, 10.0, 5.0, 7.0)
    _hover_move(view, QPoint(200, 150))
    assert view._ghost_pos is not None
    _, _, gz = view._ghost_pos
    z_step = view._z_snap_step()
    remainder_z = abs(gz / z_step - round(gz / z_step))
    assert remainder_z < 1e-9, f"gz={gz} is not a multiple of _z_snap_step={z_step}"

    # Move cursor high to z=100: ghost z should also be a multiple of _z_snap_step
    _assemble_stub_ray(view, 10.0, 5.0, 100.0)
    _hover_move(view, QPoint(200, 150))
    assert view._ghost_pos is not None
    _, _, gz = view._ghost_pos
    remainder_z = abs(gz / z_step - round(gz / z_step))
    assert remainder_z < 1e-9, f"gz={gz} is not a multiple of _z_snap_step={z_step}"

    # Verify Z does NOT land on multiples of grid_step (2.0) for this movement.
    # With step=1.5 the snap point differs from what step=2.0 would produce.
    assert abs(gz - round(gz / 2.0) * 2.0) > 1e-9

    view.cancel_placement()
    _reset(window)


def test_drag_z_lock_dz_is_multiple_of_z_snap_step(window: MainWindow) -> None:
    '''Z-lock drag during piece grab produces dz multiples of _z_snap_step().'''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    _place(window, z=0.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    view._begin_piece_drag()

    # Engage Z lock with near-anchor ray
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    assert view._axis_lock == "z"
    _hover_move(view, QPoint(200, 150))

    # Retarget ray high: constrained delta lifts z only.
    _assemble_stub_ray(view, 10.0, 5.0, 100.0)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    dz = float(meta['pos_z'])

    # dz must be a multiple of _z_snap_step() (block_height * grid_scale = 1.5)
    z_step = view._z_snap_step()
    assert z_step == pytest.approx(1.5)
    remainder = abs(dz / z_step - round(dz / z_step))
    assert remainder < 1e-9, f"dz={dz} is not a multiple of z_step={z_step}"

    view.cancel_grab()
    _reset(window)


def test_ghost_snap_to_pick_point_on_tall_piece(window: MainWindow) -> None:
    '''Ghost placement uses the actual cursor hit point on the picked piece.

    Place a piece, start ghost placement, stub _pick_scene_point to return a
    hit on the placed piece's top surface, stub the ray as usual, _hover_move
    → _ghost_pos z == that piece's stack_top and x/y near the pick point.
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    # Place a reference piece at z=25 so its stack_top (28.0) is above body bottom
    # but far enough from ghost's working plane (z=0) that stacking wins over free.
    _place(window, z=25.0)
    bh = window._project.block_settings(0).block_height
    # stack_top = position_z + block_height = 25.0 + 3.0 = 28.0
    expected_stack_top = pytest.approx(25.0 + bh)

    view = window._assemble_view
    uid = window._project.blocks[0].uid
    window._on_parts_place_requested(uid)
    assert view.placing() is True

    # Stub the ray for deterministic floor-plane intersection.
    _assemble_stub_ray(view, 10.0, 5.0, 2.5)

    # Verify supports are populated (before any hover calls).
    assert len(view._drag_supports) > 0, (
        f"_drag_supports empty: snap_parts={view._snap_parts}, "
        f"provider={view._supports_provider}"
    )

    # Move the ghost once with no hit to initialize position.
    _hover_move(view, QPoint(100, 100))

    # Now set pick_scene_point stub (after _hover_move so it doesn't get overwritten).
    total_h = window._project.block_settings(0).total_height
    pick_z = 25.0 + total_h
    hit_val: tuple[tuple[str, int], np.ndarray] = (
        ("block", 0),
        np.array([0.0, 0.0, pick_z], dtype=np.float64),
    )
    view._pick_scene_point = lambda pos: hit_val  # type: ignore[method-assign]

    # Call _update_ghost directly so the stub is used (not overwritten by _hover_move).
    view._update_ghost(QPoint(100, 100))

    assert view._ghost_pos is not None
    gx, gy, gz = view._ghost_pos
    # Ghost z should be at the stacked piece's stack_top (position_z + block_height).
    assert gz == expected_stack_top
    # Ghost x/y should be near the pick point (grid-rounded to origin).
    assert gx == pytest.approx(0.0, abs=4.0)
    assert gy == pytest.approx(0.0, abs=4.0)

    view.cancel_placement()
    _reset(window)


# ----------------------------------------------------- view-plane free movement


def test_ghost_free_path_moves_z_with_angled_camera(window: MainWindow) -> None:
    '''Ghost placement with angled camera moves Z via mouse, no axis lock.

    The view-plane normal is set to (0, 1, 1)/sqrt(2), a 45-degree upward angle.
    Two cursor positions produce different intersections with the plane through
    the ghost anchor; the z values must differ and must clamp at 0 when aimed
    below the floor.
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    view = window._assemble_view
    uid = window._project.blocks[0].uid
    _ = window._project.block_settings(0)  # trigger settings load
    window._on_parts_place_requested(uid)
    assert view.placing() is True

    # Stub ray with normal pointing up 45 degrees in YZ.
    origin = np.array([150.0, 120.0, 400.0], dtype=np.float64)
    target1 = np.array([10.0, 5.0, 2.5], dtype=np.float64)
    direction1 = target1 - origin
    view._cursor_ray = lambda pos=None: (origin, direction1)
    angled_normal = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    view._view_plane_normal = lambda: angled_normal  # type: ignore[method-assign]

    # First hover: ray target at z=2.5 → intersection plane through ghost anchor.
    # ghost_free_anchor starts at (0, 0, 0) since ghost_pos is None and ghost_z=0.
    # Plane normal = (0, 1/sqrt(2), 1/sqrt(2)), origin = (150, 120, 400).
    # denom = dot(direction1, normal) = 0 + (-115)/sqrt(2) + (-397.5)/sqrt(2)
    #       = -512.5/sqrt(2) ≈ -362.4
    # point = ghost_free_anchor = [0, 0, 0]
    # dot(point - origin, normal) = dot([-150, -120, -400], [0, 1/sqrt(2), 1/sqrt(2)])
    #   = -120/sqrt(2) + (-400)/sqrt(2) = -520/sqrt(2) ≈ -367.7
    # t = (-520/sqrt(2)) / (-512.5/sqrt(2)) = 520/512.5 ≈ 1.0146
    # raw[2] = 400 + (2.5 - 400) * t = 400 + (-397.5) * 1.0146 ≈ 400 - 403.3 ≈ -3.3
    _hover_move(view, QPoint(100, 100))
    assert view._ghost_free_anchor is not None
    z_after_first = float(view._ghost_free_anchor[2])

    # Second hover: ray target at z=100 → different intersection.
    target2 = np.array([10.0, 5.0, 100.0], dtype=np.float64)
    direction2 = target2 - origin
    view._cursor_ray = lambda pos=None: (origin, direction2)
    # denom = dot(direction2, normal) = 0 + (-115)/sqrt(2) + (-300)/sqrt(2)
    #       = -415/sqrt(2) ≈ -293.4
    # t = (-520/sqrt(2)) / (-415/sqrt(2)) = 520/415 ≈ 1.2530
    # raw[2] = 400 + (100 - 400) * t = 400 + (-300) * 1.2530 ≈ 400 - 375.9 ≈ 24.1
    _hover_move(view, QPoint(200, 150))
    assert view._ghost_free_anchor is not None
    z_after_second = float(view._ghost_free_anchor[2])

    # Z must have changed between the two hover moves (angled camera).
    assert abs(z_after_first - z_after_second) > 0.5, (
        f"Z did not change: {z_after_first} vs {z_after_second}"
    )

    # Third hover: aim well below floor → raw[2] negative → clamp at 0.
    target3 = np.array([10.0, 5.0, -400.0], dtype=np.float64)
    direction3 = target3 - origin
    view._cursor_ray = lambda pos=None: (origin, direction3)
    # denom = dot(direction3, normal) = 0 + (-115)/sqrt(2) + (-800)/sqrt(2)
    #       = -915/sqrt(2) ≈ -647.0
    # t = (-520/sqrt(2)) / (-915/sqrt(2)) = 520/915 ≈ 0.5683
    # raw[2] = 400 + (-800) * 0.5683 = 400 - 454.6 ≈ -54.6 → clamped to 0
    _hover_move(view, QPoint(100, 100))
    assert view._ghost_pos is not None
    _, _, ghost_z = view._ghost_pos
    assert ghost_z == pytest.approx(0.0, abs=1e-6), (
        f"Ghost z should clamp at 0, got {ghost_z}"
    )

    view.cancel_placement()
    _reset(window)


def test_drag_free_path_moves_z_with_angled_camera(window: MainWindow) -> None:
    '''Drag with angled camera moves Z via mouse, no axis lock.

    Place a piece, start grab, hover with angled normal → pos_z changes.
    Aim below floor → z clamps at 0. Then engage Z-lock mid-grab and verify
    the height is retained.
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)

    # Place a reference piece at z=5 so its pos_z != 0 for clarity.
    _place(window, z=5.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    view._begin_piece_drag()

    # Stub ray with angled normal [0, -1, -1]/sqrt(2).
    origin = np.array([150.0, 120.0, 400.0], dtype=np.float64)
    target1 = np.array([10.0, 5.0, 2.5], dtype=np.float64)
    direction1 = target1 - origin
    view._cursor_ray = lambda pos=None: (origin, direction1)
    angled_normal = np.array([0.0, -1.0, -1.0]) / np.sqrt(2.0)
    view._view_plane_normal = lambda: angled_normal  # type: ignore[method-assign]

    # First hover: initializes grab_free_anchor at the piece's current position.
    _hover_move(view, QPoint(100, 100))
    assert view._grab_free_anchor is not None

    # Second hover: ray target at z=100 → different intersection.
    # With normal [0,-1,-1]/sqrt(2) and origin at (150,120,400), raw[2] is positive
    # but delta from anchor is negative → piece drops toward floor.
    target2 = np.array([10.0, 5.0, 100.0], dtype=np.float64)
    direction2 = target2 - origin
    view._cursor_ray = lambda pos=None: (origin, direction2)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    second_z = float(meta['pos_z'])

    # Third hover: aim well below floor → raw[2] decreases further.
    target3 = np.array([10.0, 5.0, -400.0], dtype=np.float64)
    direction3 = target3 - origin
    view._cursor_ray = lambda pos=None: (origin, direction3)
    _hover_move(view, QPoint(100, 100))
    meta = view._scene_meta[0]
    third_z = float(meta['pos_z'])

    # Z should decrease (drop toward floor) as cursor aims lower.
    assert third_z <= second_z, (
        f"Z should not increase: second={second_z}, third={third_z}"
    )

    # Engage Z-lock mid-grab: height should be retained (no jump).
    _assemble_stub_ray(view, 10.0, 5.0)
    view._set_axis_lock("z", False)
    assert view._axis_lock == "z"
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    # Engagement is a no-move: height stays approximately at the current value.
    # (Small snapping differences may occur due to z_step rounding.)
    diff = abs(meta["pos_z"] - third_z)
    assert diff < 2.0, (
        f"Z-lock engagement should retain height: expected ~{third_z}, "
        f"got {meta['pos_z']}"
    )

    # Lift via Z-lock: ray target high → piece lifts.
    _assemble_stub_ray(view, 10.0, 5.0, 200.0)
    _hover_move(view, QPoint(200, 150))
    meta = view._scene_meta[0]
    assert meta["pos_z"] > 50, f"Z-lock should lift piece above 50, got {meta['pos_z']}"

    view.cancel_grab()
    _reset(window)


def test_drag_free_path_accumulates_xy_moves(window: MainWindow) -> None:
    '''Guard the fixed free-path anchor: per-frame cursor increments accumulate.

    Regression test for the bug where ``_grab_free_anchor`` was re-anchored to
    the cursor every mouse-move frame and the per-frame increment was applied as
    a total offset from drag-start, causing XY pieces to jitter in place.

    With stubbed ray targets advancing steadily in +X, the accumulated position
    trace must be strictly non-decreasing and the final ``pos_x`` far past the
    first-increment offset (~20 mm).
    '''
    _reset(window)
    window._tabs.setCurrentIndex(1)
    _wait_meshes(window)
    _place(window, x=0.0, y=0.0, z=0.0)

    view = window._assemble_view
    view.set_selection({0}, 0)
    assert view.start_grab()
    view._begin_piece_drag()

    # Initialize the free anchor (no move yet — first hover locks anchor).
    _assemble_stub_ray(view, 10.0, 5.0)
    _hover_move(view, QPoint(100, 100))
    assert view._grab_free_anchor is not None

    # Five successive hover moves with ray target advancing +20 mm in X each step.
    pos_trace: list[float] = []
    for k in range(1, 6):
        _assemble_stub_ray(view, 30.0 + 20.0 * k, 5.0)
        _hover_move(view, QPoint(100 + 10 * k, 100))
        pos_trace.append(float(view._scene_meta[0]['pos_x']))

    # The cursor travelled +100 mm total (5 steps x +20 px each).
    # Even with grid snap the accumulated position must exceed 40.0.
    assert pos_trace[-1] > 40.0, (
        f"Final pos_x={pos_trace[-1]} should exceed 40.0; "
        "free-path anchor may not be accumulating displacement"
    )

    # Strictly non-decreasing trace — a regression that replaces total offset
    # with per-frame increments would park the piece near ~20 and stay flat.
    for i in range(1, len(pos_trace)):
        assert pos_trace[i] >= pos_trace[i - 1], (
            f"Trace not non-decreasing at step {i}: {pos_trace[i - 1]} → {pos_trace[i]}"
        )

    view.cancel_grab()
    _reset(window)
