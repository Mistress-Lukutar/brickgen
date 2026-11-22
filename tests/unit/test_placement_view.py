'''
File:   test_placement_view.py
Brief:  Offscreen PlacementView tests: zone overlays, scene updates, camera.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.17.0
'''

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import pyvista as pv
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from brickgen.core.project import PrintBedSettings
from brickgen.gui.placement_view import PlacementView
from brickgen.gui.preview import BLOCK_COLOR

# vtk 9.6 numpy_support sets .shape on arrays, deprecated in numpy 2.5; the
# warning fires on every cell_data read and is a VTK-internal issue.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array:DeprecationWarning"
    ),
]

_W, _H = 400, 300


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


def _small_cube_mesh() -> pv.PolyData:
    '''Triangulated 10x10x5 block mesh on z=0..5 with a semantic boundary.'''
    mesh = pv.Cube(x_length=10, y_length=10, z_length=5, center=(0, 0, 2.5))
    mesh = mesh.triangulate()
    parts = np.zeros(mesh.n_cells, dtype=np.int8)
    parts[0] = 1
    mesh.cell_data['part'] = parts
    return mesh


def _scene_meta(pos_x: float, pos_y: float) -> dict:
    '''Scene metadata entry of one test block.'''
    return {
        'pos_x': pos_x,
        'pos_y': pos_y,
        'half_x': 5.0,
        'half_y': 5.0,
        'top_z': 5.0,
        'quantity': 1,
    }


# ------------------------------------------------------------------- fixtures


@pytest.fixture()
def placement_view_default() -> Iterator[PlacementView]:
    '''A PlacementView with default PrintBedSettings (count=1).'''
    _ensure_app()
    widget = PlacementView()
    widget.resize(_W, _H)
    widget.renderer.GetRenderWindow().SetSize(_W, _H)
    # Give it an initial empty scene so the camera is placed once.
    widget.update_scene([], [], [])
    try:
        yield widget
    finally:
        widget.close()


@pytest.fixture()
def placement_view_three_beds() -> Iterator[PlacementView]:
    '''A PlacementView configured with 3 beds.'''
    _ensure_app()
    widget = PlacementView()
    widget.resize(_W, _H)
    widget.renderer.GetRenderWindow().SetSize(_W, _H)
    widget.set_print_beds(PrintBedSettings(count=3))
    widget.update_scene([], [], [])
    try:
        yield widget
    finally:
        widget.close()


# ------------------------------------------------------------------ tests


def test_default_view_has_zone_overlays(
    placement_view_default: PlacementView,
) -> None:
    '''Fresh view has zone overlays for the default 1 bed + staging.

    Each bed/staging zone contributes a fill plate, a frame actor and a
    label PointLabels, so 2 zones = 6 entries in _zone_actors.
    '''
    assert len(placement_view_default._zone_actors) == 6


def test_set_print_beds_rebuilds_zones(
    placement_view_three_beds: PlacementView,
) -> None:
    '''set_print_beds(count=3) rebuilds: 12 entries (4 zones x 3 actors).'''
    view = placement_view_three_beds
    # 3 beds + 1 staging = 4 zones; each zone has fill + frame + label.
    assert len(view._zone_actors) == 12
    assert view.beds().count == 3


def test_set_print_beds_no_duplicates(
    placement_view_default: PlacementView,
) -> None:
    '''Changing bed count replaces old actors; no duplicates remain.'''
    view = placement_view_default
    initial_count = len(view._zone_actors)
    assert initial_count == 6  # default: (1 bed + staging) * 3 (fill+frame+label)

    view.set_print_beds(PrintBedSettings(count=3))
    new_count = len(view._zone_actors)
    assert new_count == 12  # (3 beds + staging) * 3

    # Old actors are gone (different list content).
    all_ids = {id(a) for a in view._zone_actors}
    assert len(all_ids) == len(view._zone_actors)


def test_update_scene_keeps_zone_actors_and_adds_pieces(
    placement_view_default: PlacementView,
) -> None:
    '''update_scene with two pieces keeps zone actors and adds per-piece actors.'''
    view = placement_view_default
    assert len(view._zone_actors) == 6

    mesh1 = _small_cube_mesh()
    mesh2 = _small_cube_mesh()
    view.update_scene(
        [mesh1, mesh2],
        [BLOCK_COLOR, '#bf7f3f'],
        [_scene_meta(0.0, 0.0), _scene_meta(50.0, 0.0)],
    )

    # Zone actors are untouched by update_scene.
    assert len(view._zone_actors) == 6

    # Two block actors created with matching meta.
    assert len(view._block_actors) == 2
    assert view._actor_map.get(view._block_actors[0][0]) == 0
    assert view._actor_map.get(view._block_actors[1][0]) == 1


def test_camera_survives_set_print_beds(
    placement_view_default: PlacementView,
) -> None:
    '''Camera position is preserved across set_print_beds.'''
    view = placement_view_default

    # Set a known camera position.
    cam = view.renderer.GetActiveCamera()
    cam.SetPosition(100.0, -200.0, 50.0)
    cam.SetFocalPoint(50.0, 0.0, 2.5)
    cam.SetViewUp(0, 0, 1)
    view.renderer.ResetCameraClippingRange()

    before = list(cam.GetPosition())
    assert before != [0.0, 0.0, 0.0]

    # Change beds — camera must survive unchanged.
    view.set_print_beds(PrintBedSettings(count=3))

    after = list(cam.GetPosition())
    assert after == pytest.approx(before)


def test_zone_actors_not_pickable(
    placement_view_default: PlacementView,
) -> None:
    '''Every actor in the private zone list has GetPickable() == 0.'''
    view = placement_view_default
    for actor in view._zone_actors:
        assert actor.GetPickable() == 0


def test_picked_ray_at_bed_frame_does_not_select_piece(
    placement_view_default: PlacementView,
) -> None:
    '''A pick ray aimed at a bed frame pixel hits the frame (non-pickable)
    and falls through to the background — no piece is selected.'''
    view = placement_view_default

    # Place a piece at the origin.
    mesh = _small_cube_mesh()
    view.update_scene(
        [mesh],
        [BLOCK_COLOR],
        [_scene_meta(0.0, 0.0)],
    )

    # Compute the bed-0 frame bottom-left corner in Qt coords.
    # Bed 0 origin: (0, 0), corners include (0, 0).
    renderer = view.renderer
    renderer.SetWorldPoint(0.0, 0.0, 0.0, 1.0)
    renderer.WorldToDisplay()
    dx, dy, _ = renderer.GetDisplayPoint()[:3]
    _ = QPoint(round(dx), round(view.height() - dy))

    # Pick at the corner pixel — the frame line is there but non-pickable.
    view._picker.Pick(float(dx), float(view.height() - dy), 0, view.renderer)
    actor = view._picker.GetActor()

    # The picker either returns None (no pickable hit) or an actor not in
    # _actor_map (the frame itself). Either way, no block index is resolved.
    piece_idx = view._actor_map.get(actor)
    assert piece_idx is None


def test_grid_enabled_draws_per_bed_grids(
    placement_view_three_beds: PlacementView,
) -> None:
    '''Grid toggle draws the combined minor+major bed grids, not the floor grid.

    4 zones x 3 actors = 12, plus one minor and one major line mesh across
    all beds = 14.  The base class's infinite floor grid stays empty.
    '''
    view = placement_view_three_beds
    view.set_grid_enabled(True)
    assert len(view._zone_actors) == 14
    # The base infinite origin-centered grid must never be built here.
    assert view._grid_actors == []

    view.set_grid_enabled(False)
    assert len(view._zone_actors) == 12


def test_grid_unit_change_rebuilds_grids(
    placement_view_three_beds: PlacementView,
) -> None:
    '''Changing the grid unit with grids on keeps the actor set consistent.'''
    view = placement_view_three_beds
    view.set_grid_enabled(True)
    assert len(view._zone_actors) == 14

    view.set_grid_unit(8.0)  # step 8 * 0.5 = 4 mm: same 2 line meshes
    assert len(view._zone_actors) == 14
    assert view._grid_actors == []


def test_snap_move_target_is_zone_relative(
    placement_view_three_beds: PlacementView,
) -> None:
    '''Snap rounds to zone-local grid multiples, not world origin ones.

    Bed 1 origin x = 220 + 20 (BED_GAP_MM) = 240; default step = 4 * 0.5
    = 2 mm.  Staging origin x = -240.
    '''
    view = placement_view_three_beds
    # Bed 1: local 13.4 -> 14 (world 254); y 6.6 -> 6.
    x, y = view._snap_move_target(253.4, 6.6)
    assert x == pytest.approx(254.0)
    assert y == pytest.approx(6.0)
    # Staging: local 13.4 -> 14 (world -226).
    x, y = view._snap_move_target(-226.6, 0.0)
    assert x == pytest.approx(-226.0)
    assert y == pytest.approx(0.0)
