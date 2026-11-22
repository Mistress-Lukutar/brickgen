'''
File:   test_app_tabs.py
Brief:  Integration tests for the MainWindow tab integration (S4).
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.18.0
'''

from __future__ import annotations

import os
import time
from collections.abc import Iterator

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from brickgen.core.project import (  # noqa: E402
    PRINT_STAGING_BED,
    BlockInstance,
    PrintBedSettings,
    ViewSettings,
)
from brickgen.gui.app import MainWindow  # noqa: E402
from brickgen.services.placement_service import (  # noqa: E402
    BED_GAP_MM,
    block_footprints,
    local_position,
    piece_dims,
    sync_pieces,
    world_position,
)

# vtk 9.6 numpy_support sets .shape on arrays, deprecated in numpy 2.5; the
# warning fires from the preview pipeline exercised by the app smoke test.
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
    '''One MainWindow for the whole module: a full generation run takes
    tens of seconds and spawns worker processes, so per-test windows would
    exhaust memory (the OpenBLAS allocation failures seen earlier).'''
    _ensure_app()
    win = MainWindow()
    win.resize(1200, 760)
    win.show()  # offscreen; makes isVisible()/isHidden() assertions meaningful
    _wait_applied(win)
    yield win
    # Clear the dirty flag first: closeEvent opens a modal "unsaved changes"
    # prompt otherwise, which would block forever under the offscreen platform.
    win._dirty = False
    win._jobs.close()  # first so the executor is shut down even without closeEvent
    win.close()


def _reset_state(win: MainWindow) -> None:
    '''Restore the pristine one-block project state before a test.

    Blocks are trimmed to the first default block (quantity 1), bed
    settings return to defaults and the piece list is rebuilt from
    scratch (which also resets flips and selection).
    '''
    win._project.blocks[:] = [win._project.blocks[0]]
    win._project.blocks[0].quantity = 1
    win._project.print_layout.beds = PrintBedSettings()
    win._project.print_layout.pieces.clear()
    win._block_meshes_flipped = [None] * len(win._block_meshes)
    win._pselection = set()
    win._pprimary = None
    win._placement_panel.set_beds(win._project.print_layout.beds)
    win._placement_view.set_print_beds(win._project.print_layout.beds)
    win._sync_placement()


# ===========================================================================


def test_tabs_switch_sidebar_and_library(window: MainWindow) -> None:
    '''2 tabs with correct titles; switch to 1 -> sidebar stack index 1,
    library hidden, quantity actions disabled; back to 0 restores.'''
    QApplication.processEvents()

    # Verify tab structure.
    assert window._tabs.count() == 2
    assert window._tabs.tabText(0) == "Planning"
    assert window._tabs.tabText(1) == "Print placement"

    # Switch to tab 1 (placement).
    window._tabs.setCurrentIndex(1)
    QApplication.processEvents()
    assert window._sidebar_stack.currentIndex() == 1
    assert window._library.isHidden()
    assert not window._act_qty_labels.isEnabled()

    # Switch back to tab 0 (planning).
    window._tabs.setCurrentIndex(0)
    QApplication.processEvents()
    assert window._sidebar_stack.currentIndex() == 0
    assert not window._library.isHidden()
    assert window._act_qty_labels.isEnabled()


def test_default_project_syncs_pieces(window: MainWindow) -> None:
    '''The default one-block project produces exactly one piece on bed 0
    and the placement view has zone overlays.'''
    _reset_state(window)

    pieces = window._project.print_layout.pieces
    assert len(pieces) == 1
    assert pieces[0].bed == 0

    # Placement view should have zone actors (at minimum).
    assert len(window._placement_view._zone_actors) >= 2


def test_flip_selected_piece(window: MainWindow) -> None:
    '''Select piece 0 via _on_placement_selection; flip it True.
    piece.flipped is True, scene still has actors, project dirty flag set.'''
    _reset_state(window)
    pieces = window._project.print_layout.pieces
    assert len(pieces) == 1

    # Select piece 0.
    window._on_placement_selection((frozenset({0}), 0))
    assert window._pselection == {0}
    assert window._pprimary == 0

    # Flip it.
    window._on_piece_flip_changed(True)
    assert pieces[0].flipped is True
    assert window._dirty is True


def test_frame_selection_picks_primary(window: MainWindow) -> None:
    '''A rubber-band payload (selection, None) picks the lowest selected
    index as primary, so the inspector stays active on framed selections.

    The frame selection emits no primary (no piece was clicked); without
    the fallback the "Selected piece" group stayed disabled although move
    operations worked fine.
    '''
    _reset_state(window)
    window._project.blocks[0].quantity = 2
    window._sync_placement()
    assert len(window._project.print_layout.pieces) == 2

    window._on_placement_selection((frozenset({0, 1}), None))
    assert window._pselection == {0, 1}
    assert window._pprimary == 0
    assert window._placement_panel._inspector_group.isEnabled()

    # An empty frame selection keeps the inspector disabled.
    window._on_placement_selection((frozenset(), None))
    assert window._pselection == set()
    assert window._pprimary is None
    assert not window._placement_panel._inspector_group.isEnabled()


def test_bed_count_clamp(window: MainWindow) -> None:
    '''_on_placement_beds_changed with count=2; _on_piece_bed_changed(2) ->
    piece.bed==1; then beds count=1 -> piece.bed==-1.'''
    _reset_state(window)

    beds = PrintBedSettings(width=220, depth=220, count=2, gap=4)
    window._project.print_layout.beds = beds
    window._placement_panel.set_beds(beds)

    pieces = window._project.print_layout.pieces
    assert len(pieces) == 1

    # Move the piece to bed index 1. The app handler receives the already
    # panel-mapped 0-based bed index (spin value 2 maps to index 1).
    window._pselection = {0}
    window._pprimary = 0
    window._on_piece_bed_changed(1)
    assert pieces[0].bed == 1

    # Reduce the bed count back to 1: the piece no longer fits any bed.
    beds_small = PrintBedSettings(width=220, depth=220, count=1, gap=4)
    window._on_placement_beds_changed(beds_small)
    assert pieces[0].bed == PRINT_STAGING_BED


def test_move_piece_between_zones(window: MainWindow) -> None:
    '''Beds count 2; a drop past the zone-assignment boundary lands the piece
    in bed 1, CLAMPED so its footprint stays inside bed 1's margins (a drop
    past the edge must not poke into the neighboring bed).'''
    _reset_state(window)

    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    window._project.print_layout.beds = beds
    window._placement_panel.set_beds(beds)

    pieces = window._project.print_layout.pieces
    assert len(pieces) >= 1

    # Move piece 0 to bed 1 zone (5 mm past the boundary, outside bed 1's
    # left margin: the raw local x would be negative).
    target_wx = beds.width + BED_GAP_MM / 2.0 + 5.0
    moves = {0: (target_wx, 10.0)}
    window._on_placement_pieces_moved(moves)

    assert pieces[0].bed == 1
    fp = block_footprints(window._project)
    w, d = piece_dims(pieces[0], fp)
    assert pieces[0].position_x == pytest.approx(beds.gap + w / 2, abs=0.01)
    # y was already inside the margins: kept as dropped.
    assert pieces[0].position_y == pytest.approx(10.0, abs=0.01)

    # A drop inside the valid area keeps the exact local position.
    target_wx2 = beds.width + BED_GAP_MM + 60.0
    window._on_placement_pieces_moved({0: (target_wx2, 30.0)})
    expected_local = local_position(target_wx2, 30.0, 1, beds)
    assert pieces[0].position_x == pytest.approx(expected_local[0], abs=0.01)
    assert pieces[0].position_y == pytest.approx(expected_local[1], abs=0.01)


def test_move_piece_resolves_overlap(window: MainWindow) -> None:
    '''Dropping a piece onto another shifts it to the nearest free spot.'''
    _reset_state(window)

    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    window._project.print_layout.beds = beds
    window._project.blocks[0].quantity = 2
    window._sync_placement()
    pieces = window._project.print_layout.pieces
    assert len(pieces) == 2

    # Drop piece 1 exactly onto piece 0.
    wx, wy = world_position(pieces[0], beds)
    window._on_placement_pieces_moved({1: (wx, wy)})

    # Piece 1 was shifted to the nearest conflict-free spot, piece 0 kept
    # its position; nothing overlaps.
    fp = block_footprints(window._project)
    w0, d0 = piece_dims(pieces[0], fp)
    w1, d1 = piece_dims(pieces[1], fp)
    separated_x = (
        abs(pieces[0].position_x - pieces[1].position_x)
        >= (w0 + w1) / 2 + beds.gap - 1e-6
    )
    separated_y = (
        abs(pieces[0].position_y - pieces[1].position_y)
        >= (d0 + d1) / 2 + beds.gap - 1e-6
    )
    assert separated_x or separated_y
    assert pieces[1].bed == 0


def test_auto_place_handler(window: MainWindow) -> None:
    '''Two blocks (2x2 qty 2 each); _on_placement_auto_place([0])
    -> 4 pieces all on bed 0, none staged.'''
    _reset_state(window)

    # Make the default block a 2x2 with quantity 2 and add a second block.
    window._project.blocks[0].quantity = 2
    window._project.blocks.append(
        BlockInstance(uid="blk2", cells_x=2, cells_y=2, quantity=2)
    )
    sync_pieces(window._project)

    # Auto-place on bed 0.
    window._on_placement_auto_place([0])

    pieces = window._project.print_layout.pieces
    assert len(pieces) == 4  # 2 + 2
    assert all(p.bed == 0 for p in pieces)
    assert all(p.bed != PRINT_STAGING_BED for p in pieces)


def test_panel_beds_signal_updates_project(window: MainWindow) -> None:
    '''_on_placement_beds_changed(width=200) -> project and view pick it up;
    restoring the original width returns to the default.'''
    _reset_state(window)

    original_width = window._project.print_layout.beds.width

    # Emit beds_changed from panel with width=200.
    new_beds = PrintBedSettings(
        width=200.0,
        depth=window._project.print_layout.beds.depth,
        count=window._project.print_layout.beds.count,
        gap=window._project.print_layout.beds.gap,
    )
    window._on_placement_beds_changed(new_beds)

    assert window._project.print_layout.beds.width == 200.0
    assert window._placement_view.beds().width == 200.0

    # Restore original.
    orig_beds = PrintBedSettings(
        width=original_width,
        depth=window._project.print_layout.beds.depth,
        count=window._project.print_layout.beds.count,
        gap=window._project.print_layout.beds.gap,
    )
    window._on_placement_beds_changed(orig_beds)
    assert window._project.print_layout.beds.width == original_width


def test_rotated_piece_builds_variant_mesh(window: MainWindow) -> None:
    '''A rotated90 piece renders from the orientation variant cache.'''
    _reset_state(window)
    pieces = window._project.print_layout.pieces
    assert len(pieces) == 1

    pieces[0].rotated90 = True
    window._after_placement_change()

    # The variant (block 0, not flipped, rotated) was built and cached.
    assert (0, False, True) in window._mesh_variants
    variant = window._mesh_variants[(0, False, True)]
    assert variant.n_points > 0
    # The viewport still shows the piece.
    assert len(window._placement_view._block_actors) == 1


def test_view_settings_stored_in_project(window: MainWindow) -> None:
    '''Grid/View menu toggles mutate ``project.view`` and dirty the project;
    ``_apply_view_settings`` pushes the values into the previews.'''
    _reset_state(window)
    window._project.view = ViewSettings()
    window._apply_view_settings()
    window._dirty = False

    # Planning-tab grid toggle lands in the project view.
    window._tabs.setCurrentIndex(0)
    window._act_show_grid.setChecked(True)
    assert window._project.view.grid_enabled is True
    assert window._preview._grid_enabled is True
    assert window._dirty is True

    # Placement-tab grid toggle uses its own flag (per-bed grids, never the
    # base class's infinite floor grid).
    window._tabs.setCurrentIndex(1)
    window._act_show_grid.setChecked(True)
    assert window._project.view.placement_grid_enabled is True
    assert window._placement_view._bed_grid_enabled is True
    assert window._placement_view._grid_actors == []

    # Scale / snap / labels are shared project-level values.
    window._tabs.setCurrentIndex(0)
    window._act_grid_scale[2.0].trigger()
    window._act_snap.setChecked(True)
    window._act_qty_labels.setChecked(True)
    window._act_qty_positions["top"].trigger()
    view = window._project.view
    assert view.grid_scale == 2.0
    assert view.grid_snap is True
    assert view.quantity_labels_enabled is True
    assert view.quantity_label_position == "top"
    assert window._preview._grid_scale == 2.0
    assert window._preview._grid_snap is True
    assert window._preview._qty_enabled is True
    assert window._preview._qty_position == "top"

    # Restore pristine state for the following tests.
    window._project.view = ViewSettings()
    window._apply_view_settings()
    window._dirty = False


def test_enforce_no_overlap_repairs_packed_state(window: MainWindow) -> None:
    '''The post-pack invariant repair fixes any overlapping piece pair.'''
    _reset_state(window)

    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    window._project.print_layout.beds = beds
    window._project.blocks[0].quantity = 2
    window._sync_placement()
    pieces = window._project.print_layout.pieces
    assert len(pieces) == 2

    # Simulate an overlap reaching the viewport (e.g. from a stale path):
    # piece 1 sits exactly on piece 0.
    pieces[1].bed = 0
    pieces[1].position_x = pieces[0].position_x
    pieces[1].position_y = pieces[0].position_y

    window._enforce_no_overlap()

    fp = block_footprints(window._project)
    w0, d0 = piece_dims(pieces[0], fp)
    w1, d1 = piece_dims(pieces[1], fp)
    separated_x = (
        abs(pieces[0].position_x - pieces[1].position_x)
        >= (w0 + w1) / 2 + beds.gap - 1e-6
    )
    separated_y = (
        abs(pieces[0].position_y - pieces[1].position_y)
        >= (d0 + d1) / 2 + beds.gap - 1e-6
    )
    assert separated_x or separated_y
    assert pieces[1].bed == 0
