'''
File:   test_preview.py
Brief:  Unit tests for the pyvista mesh conversion used by the preview.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.9.2
'''

from __future__ import annotations

import numpy as np
import pyvista as pv
import pytest

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings

# vtk 9.6 numpy_support sets .shape on arrays, deprecated in numpy 2.5; the
# warning fires on every cell_data read and is a VTK-internal issue.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)

from brickgen.gui.preview import (
    DISPLAY_MODES,
    MODE_SOLID,
    MODE_SOLID_FEATURE_EDGES,
    MODE_SOLID_WIREFRAME,
    MODE_WIREFRAME,
    PART_BODY,
    PART_BOTTOM,
    PART_STUD,
    PART_TUBE,
    make_face_classifier,
    semantic_edges,
    solid_to_mesh,
)


@pytest.fixture
def mesh_2x4():
    '''Cleaned preview mesh of a 2x4 block.'''
    solid = build_block(BlockSettings(cells_x=2, cells_y=4)).val()
    return solid_to_mesh(solid)


def test_mesh_is_triangles(mesh_2x4) -> None:
    '''All cells of the converted mesh must be triangles.'''
    assert mesh_2x4.n_cells > 0
    assert all(cell.n_points == 3 for cell in mesh_2x4.cell)


def test_mesh_bounding_box(mesh_2x4) -> None:
    '''Mesh bounds match the block footprint 8x16x5 mm.'''
    bounds = mesh_2x4.bounds
    assert bounds.x_max - bounds.x_min == pytest.approx(8.0, abs=1e-3)
    assert bounds.y_max - bounds.y_min == pytest.approx(16.0, abs=1e-3)
    assert bounds.z_max - bounds.z_min == pytest.approx(5.0, abs=1e-3)


def test_mesh_has_no_duplicate_points(mesh_2x4) -> None:
    '''clean() must merge tessellation-adjacent duplicate vertices.'''
    assert mesh_2x4.clean(tolerance=1e-6).n_points == mesh_2x4.n_points


def test_display_modes_list() -> None:
    '''Four display modes: solid, solid + feature edges, solid + wireframe, wireframe.'''
    assert DISPLAY_MODES == [
        MODE_SOLID,
        MODE_SOLID_FEATURE_EDGES,
        MODE_SOLID_WIREFRAME,
        MODE_WIREFRAME,
    ]


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def classified_4x2_mesh():
    '''Classified preview mesh of a 4x2 block (all four part ids present).'''
    settings = BlockSettings(cells_x=4, cells_y=2)
    solid = build_block(settings).val()
    classifier = make_face_classifier(settings)
    return solid_to_mesh(solid, face_part=classifier), settings


@pytest.fixture
def classified_1x1_mesh():
    '''Classified preview mesh of a 1x1 block (no tubes).'''
    settings = BlockSettings(cells_x=1, cells_y=1)
    solid = build_block(settings).val()
    classifier = make_face_classifier(settings)
    return solid_to_mesh(solid, face_part=classifier), settings


@pytest.fixture
def classified_slope_mesh():
    '''Classified preview mesh of a slope block (no studs, no tubes).'''
    settings = BlockSettings(
        cells_x=3, cells_y=2, pins_enabled=False, slope_rise_units=1
    )
    solid = build_block(settings).val()
    classifier = make_face_classifier(settings)
    return solid_to_mesh(solid, face_part=classifier), settings


# ------------------------------------------------------------------ helpers
def _get_centroids(mesh: pv.PolyData) -> np.ndarray:
    '''Return triangle-centroid array from a PyVista PolyData mesh.

    Reads mesh.faces as (n, 4) via reshape(-1, 4)[:, 1:], then uses
    mesh.points[tris].mean(axis=1).
    '''
    tris = mesh.faces.reshape(-1, 4)[:, 1:]
    return mesh.points[tris].mean(axis=1)


# ------------------------------------------------------------------ tests


def test_parts_present_4x2(classified_4x2_mesh) -> None:
    '''A 4x2 flat block classified by part id carries all four ids.'''
    mesh, settings = classified_4x2_mesh
    assert "part" in mesh.cell_data
    parts = np.asarray(mesh.cell_data["part"])
    counts = np.bincount(parts)
    # body (0), stud (1), tube (2), bottom (3) all have > 0 triangles on a 4x2 block
    for pid in (PART_BODY, PART_STUD, PART_TUBE, PART_BOTTOM):
        assert counts[pid] > 0


def test_stud_triangles_above_top_plane(classified_4x2_mesh) -> None:
    '''Every triangle classified as PART_STUD sits above the top face.'''
    mesh, settings = classified_4x2_mesh
    parts = np.asarray(mesh.cell_data["part"])
    centroids = _get_centroids(mesh)
    stud_mask = parts == PART_STUD
    assert stud_mask.any()
    assert all(centroids[stud_mask, 2] > settings.block_height)


def test_tube_triangles_near_tube_centers(classified_4x2_mesh) -> None:
    '''Every tube-classified triangle centroid is within the tube outer diameter of a tube center.'''
    mesh, settings = classified_4x2_mesh
    parts = np.asarray(mesh.cell_data["part"])
    centroids = _get_centroids(mesh)
    tube_mask = parts == PART_TUBE
    assert tube_mask.any()
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0
    centers = [
        (ix * settings.unit_size - half_x, iy * settings.unit_size - half_y)
        for ix in range(1, settings.cells_x)
        for iy in range(1, settings.cells_y)
    ]
    r_out = settings.tube_outer_diameter
    cx = centroids[tube_mask, 0]
    cy = centroids[tube_mask, 1]
    for x, y in zip(cx, cy):
        dist2 = min((x - tx) ** 2 + (y - ty) ** 2 for tx, ty in centers)
        assert dist2 < r_out**2


def test_bottom_triangles_at_zero(classified_4x2_mesh) -> None:
    '''Every PART_BOTTOM triangle centroid has z < 1e-3 (at the bottom plane).'''
    mesh, _ = classified_4x2_mesh
    parts = np.asarray(mesh.cell_data["part"])
    centroids = _get_centroids(mesh)
    bottom_mask = parts == PART_BOTTOM
    assert bottom_mask.any()
    assert all(centroids[bottom_mask, 2] < 1e-3)


def test_no_classifier_zero_parts() -> None:
    '''solid_to_mesh without face_part carries only part value 0.'''
    settings = BlockSettings(cells_x=2, cells_y=2)
    solid = build_block(settings).val()
    mesh = solid_to_mesh(solid)
    parts = np.asarray(mesh.cell_data["part"])
    assert all(parts == 0)


def test_classifier_1x1_no_tubes(classified_1x1_mesh) -> None:
    '''A 1x1 block has zero PART_TUBE triangles and at least one PART_STUD.'''
    mesh, _ = classified_1x1_mesh
    parts = np.asarray(mesh.cell_data["part"])
    counts = np.bincount(parts)
    assert counts[PART_TUBE] == 0
    assert counts[PART_STUD] > 0


def test_classifier_slope_block(classified_slope_mesh) -> None:
    '''A slope block (pins_enabled=False, slope_rise_units=1) has no studs/tubes but some body.'''
    mesh, _ = classified_slope_mesh
    parts = np.asarray(mesh.cell_data["part"])
    counts = np.bincount(parts)
    assert counts[PART_STUD] == 0
    assert counts[PART_TUBE] == 0
    assert counts[PART_BODY] > 0


def test_semantic_edges_deterministic(classified_4x2_mesh) -> None:
    '''semantic_edges returns identical cell count on repeated calls and zero for unclassified meshes.'''
    mesh, _ = classified_4x2_mesh
    edges1 = semantic_edges(mesh)
    edges2 = semantic_edges(mesh)
    assert edges1.n_cells == edges2.n_cells > 0
    # Unclassified mesh: no "part" cell data -> empty PolyData
    unclassified = solid_to_mesh(build_block(BlockSettings(cells_x=2, cells_y=2)).val())
    edges_empty = semantic_edges(unclassified)
    assert edges_empty.n_cells == 0


def test_feature_edges_mode_position() -> None:
    '''MODE_SOLID_FEATURE_EDGES in DISPLAY_MODES and index == 1.'''
    assert MODE_SOLID_FEATURE_EDGES in DISPLAY_MODES
    assert DISPLAY_MODES.index(MODE_SOLID_FEATURE_EDGES) == 1
