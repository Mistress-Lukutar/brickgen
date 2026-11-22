'''
File:   test_preview.py
Brief:  Unit tests for the pyvista mesh conversion used by the preview.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.19.0
'''

from __future__ import annotations

import math

import numpy as np
import pytest
import pyvista as pv

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings

# vtk 9.6 numpy_support sets .shape on arrays, deprecated in numpy 2.5; the
# warning fires on every cell_data read and is a VTK-internal issue, so the
# filter must be active before preview (and thus VTK) is imported.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)

from brickgen.gui.preview import (  # noqa: E402  # see filterwarnings note above
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
    orbit_offset,
    piece_variant_mesh,
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
    '''Four display modes: solid, +feature edges, +wireframe, wireframe-only.'''
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
        cells_x=3,
        cells_y=2,
        pins_enabled=False,
        sloped_cells=[[ix, iy, "+X", 1] for iy in range(2) for ix in range(3)],
    )
    solid = build_block(settings).val()
    classifier = make_face_classifier(settings)
    return solid_to_mesh(solid, face_part=classifier), settings


@pytest.fixture
def classified_negative_clearance_mesh():
    '''Classified preview mesh of a block with oversized studs.'''
    settings = BlockSettings(cells_x=2, cells_y=2, clearance_xy=-0.5, clearance_z=-0.5)
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
    '''Tube-classified centroids sit within one tube outer diameter of a center.'''
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
    '''Slope block (no pins, rise=1): no studs/tubes, but some body parts.'''
    mesh, _ = classified_slope_mesh
    parts = np.asarray(mesh.cell_data["part"])
    counts = np.bincount(parts)
    assert counts[PART_STUD] == 0
    assert counts[PART_TUBE] == 0
    assert counts[PART_BODY] > 0


def test_parts_present_negative_clearance(classified_negative_clearance_mesh) -> None:
    '''A block with oversized studs still shows all four part ids.'''
    mesh, _ = classified_negative_clearance_mesh
    parts = np.asarray(mesh.cell_data["part"])
    counts = np.bincount(parts)
    for pid in (PART_BODY, PART_STUD, PART_TUBE, PART_BOTTOM):
        assert counts[pid] > 0


def test_stud_triangles_above_top_plane_negative_clearance(
    classified_negative_clearance_mesh,
) -> None:
    '''Oversized stud triangles sit above the nominal top plane.'''
    mesh, settings = classified_negative_clearance_mesh
    parts = np.asarray(mesh.cell_data["part"])
    centroids = _get_centroids(mesh)
    stud_mask = parts == PART_STUD
    assert stud_mask.any()
    assert all(centroids[stud_mask, 2] > settings.block_height + 1e-6)


def test_semantic_edges_deterministic(classified_4x2_mesh) -> None:
    '''Repeated semantic_edges calls agree; unclassified meshes yield no edges.'''
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


# ------------------------------------------------- turntable orbit math


def test_orbit_offset_azimuth_rotates_world_z() -> None:
    '''Azimuth rotates the offset around world Z; the camera height stays.'''
    offset = np.array([0.0, -40.0, 10.0])
    up = np.array([0.0, 0.0, 1.0])
    rotated, rotated_up = orbit_offset(offset, up, azimuth_deg=90.0, elevation_deg=0.0)
    assert rotated[2] == pytest.approx(10.0)
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(offset))
    # (0, -40) rotated CCW by 90 degrees around Z lands on (40, 0).
    assert rotated[0] == pytest.approx(40.0, abs=1e-9)
    assert rotated[1] == pytest.approx(0.0, abs=1e-9)
    assert rotated_up[2] == pytest.approx(1.0)  # world up is Z-invariant


def test_orbit_offset_elevation_raises_camera_without_azimuth() -> None:
    '''Positive elevation raises the camera, preserving orbit radius/azimuth.'''
    offset = np.array([0.0, -40.0, 10.0])
    up = np.array([0.0, 0.0, 1.0])
    radius = np.linalg.norm(offset)
    azimuth_before = math.atan2(offset[1], offset[0])
    raised, _ = orbit_offset(offset, up, azimuth_deg=0.0, elevation_deg=45.0)
    assert raised[2] > offset[2]
    assert np.linalg.norm(raised) == pytest.approx(radius)
    assert math.atan2(raised[1], raised[0]) == pytest.approx(azimuth_before)


def test_orbit_offset_elevation_lowers_camera() -> None:
    '''Negative elevation lowers the camera below the horizon.'''
    offset = np.array([0.0, -40.0, 10.0])
    up = np.array([0.0, 0.0, 1.0])
    lowered, _ = orbit_offset(offset, up, azimuth_deg=0.0, elevation_deg=-45.0)
    assert lowered[2] < offset[2]


def test_orbit_offset_carries_view_up_without_roll() -> None:
    '''The view up rotates with the camera: perpendicular and roll-free.'''
    offset = np.array([0.0, -40.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    moved, moved_up = orbit_offset(offset, up, azimuth_deg=30.0, elevation_deg=40.0)
    view_dir = -moved / np.linalg.norm(moved)
    assert float(np.dot(moved_up, view_dir)) == pytest.approx(0.0, abs=1e-9)
    # No roll: zero component along the world-horizontal camera right.
    right = np.cross(view_dir, (0.0, 0.0, 1.0))
    right = right / np.linalg.norm(right)
    assert float(np.dot(moved_up, right)) == pytest.approx(0.0, abs=1e-9)


def test_orbit_offset_crosses_pole_continuously() -> None:
    '''Crossing the pole mirrors the orbit instead of flipping the view.'''
    offset = np.array([0.0, -40.0, 39.0])
    radius = float(np.linalg.norm(offset))
    # Roll-free up of this oblique view (world Z orthogonalized).
    view_dir = -offset / radius
    world_up = np.array([0.0, 0.0, 1.0])
    up = world_up - view_dir * float(np.dot(world_up, view_dir))
    up = up / np.linalg.norm(up)
    crossed, crossed_up = orbit_offset(offset, up, azimuth_deg=0.0, elevation_deg=60.0)
    polar = math.degrees(math.asin(offset[2] / radius))
    assert np.linalg.norm(crossed) == pytest.approx(radius)
    # Past the pole: still above the focal plane, horizontal side mirrored.
    assert crossed[2] == pytest.approx(radius * math.sin(math.radians(polar + 60.0)))
    assert crossed[1] > 0.0  # was -40: the camera is now on the far side
    assert float(np.dot(crossed_up, -crossed / radius)) == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(crossed_up) == pytest.approx(1.0)


def test_orbit_offset_spins_at_pole() -> None:
    '''At the pole the azimuth axis is the view axis: the view spins.'''
    offset = np.array([0.0, 0.0, 40.0])
    up = np.array([0.0, 1.0, 0.0])
    spun, spun_up = orbit_offset(offset, up, azimuth_deg=30.0, elevation_deg=0.0)
    np.testing.assert_allclose(spun, offset)
    # (0, 1, 0) rotated CCW by 30 degrees around Z lands on (-0.5, 0.866, 0).
    np.testing.assert_allclose(spun_up, [-0.5, math.sqrt(3) / 2, 0.0], atol=1e-9)


def test_orbit_offset_preserves_radius() -> None:
    '''Combined azimuth/elevation keep the orbit radius (pure rotation).'''
    offset = np.array([12.0, -30.0, 5.0])
    up = np.array([0.0, 0.0, 1.0])
    rotated, _ = orbit_offset(offset, up, azimuth_deg=33.0, elevation_deg=17.0)
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(offset))


# ========================================================= piece variant meshes


def test_piece_variant_mesh_rotated_swaps_bounds() -> None:
    '''Rotated variant swaps the footprint; the base mesh stays untouched.'''
    mesh = pv.Cube(x_length=8, y_length=4, z_length=2).triangulate()
    variant = piece_variant_mesh(mesh, rotated90=True, flipped=False, lift_z=0.0)
    xmin, xmax, ymin, ymax, zmin, zmax = variant.bounds
    assert (xmax - xmin, ymax - ymin) == pytest.approx((4.0, 8.0))
    # Centered on origin like the base mesh.
    assert (xmin + xmax) / 2 == pytest.approx(0.0)
    assert (ymin + ymax) / 2 == pytest.approx(0.0)
    # The base mesh was not mutated.
    bxmin, bxmax, bymin, bymax, _, _ = mesh.bounds
    assert (bxmax - bxmin, bymax - bymin) == pytest.approx((8.0, 4.0))


def test_piece_variant_mesh_flipped_lifts_onto_positive_z() -> None:
    '''Flipped variant mirrors z and lifts so the mesh stays on z >= 0.'''
    mesh = pv.Cube(x_length=4, y_length=4, z_length=6, center=(0, 0, 3)).triangulate()
    variant = piece_variant_mesh(mesh, rotated90=False, flipped=True, lift_z=6.0)
    _, _, _, _, zmin, zmax = variant.bounds
    assert zmin == pytest.approx(0.0)
    assert zmax == pytest.approx(6.0)


def test_piece_variant_mesh_untransformed_returns_copy() -> None:
    '''No rotation/flip requested: an equal-bbox copy comes back.'''
    mesh = pv.Cube(x_length=8, y_length=4, z_length=2).triangulate()
    variant = piece_variant_mesh(mesh, rotated90=False, flipped=False, lift_z=0.0)
    assert variant.bounds == mesh.bounds
    assert variant is not mesh
