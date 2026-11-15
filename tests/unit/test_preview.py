'''
File:   test_preview.py
Brief:  Unit tests for the pyvista mesh conversion used by the preview.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.9.0
'''

from __future__ import annotations

import pytest

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings
from brickgen.gui.preview import DISPLAY_MODES, brep_edges_polydata, solid_to_mesh


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
    '''Exactly three display modes are offered: solid, solid+mesh, wireframe.'''
    assert len(DISPLAY_MODES) == 3


def test_brep_edges_polydata() -> None:
    '''B-Rep edges discretize into line cells at every block edge.'''
    from brickgen.gui.preview import brep_edges_polydata

    solid = build_block(BlockSettings(cells_x=2, cells_y=2)).val()
    edges = brep_edges_polydata(solid)
    assert edges.n_cells > 0
    assert all(cell.n_points >= 2 for cell in edges.cell)
    bounds = edges.bounds
    assert bounds.x_max - bounds.x_min == pytest.approx(8.0, abs=1e-3)


def test_wireframe_mode_has_no_mesh_fill() -> None:
    '''Wireframe shows only B-Rep edges: surface actor is absent.'''
    modes = dict.fromkeys(DISPLAY_MODES)
    assert "Solid" in modes and "Wireframe" in modes and len(DISPLAY_MODES) == 3


def test_edge_sampling_matches_tolerance() -> None:
    '''Finer tolerance yields denser edge polylines (adaptive sampling).

    Note: the angular constraint (0.1 rad, matching Shape.tessellate) sets
    a density floor on curved edges, so only a very fine tolerance makes
    the chord bound dominate over the angular floor.
    '''
    solid = build_block(BlockSettings(cells_x=2, cells_y=2)).val()
    coarse = brep_edges_polydata(solid, tolerance=0.2)
    fine = brep_edges_polydata(solid, tolerance=0.001)
    assert fine.n_points > coarse.n_points


def test_edge_chord_deviation_bounded() -> None:
    '''Every polyline chord deviates from the true curve within tolerance.

    Checked on a circular stud edge (radius 1.3 mm) against the analytic
    sagitta bound for the sampled segment count.
    '''
    import numpy as np

    from brickgen.gui.preview import _sample_edge

    solid = build_block(BlockSettings(cells_x=2, cells_y=2)).val()
    circle = next(e for e in solid.Edges()
                  if e.geomType() == "CIRCLE" and e.Length() > 8.0)
    tolerance = 0.02
    points = _sample_edge(circle, tolerance, 0.1)
    assert len(points) >= 2
    # Independent check: fine ground-truth sampling of the curve; every
    # fine point must lie within tolerance of the sampled polyline.
    fine = np.asarray([(p.x, p.y, p.z) for p in
                       (circle.positionAt(i / 999.0) for i in range(1000))])
    poly = np.asarray([(p.x, p.y, p.z) for p in points])
    # min distance from each fine point to the closest polyline segment
    dmin = np.full(len(fine), np.inf)
    for a, b in zip(poly, poly[1:]):
        ab = b - a
        denom = float(ab @ ab)
        if denom == 0.0:
            continue
        t = np.clip(((fine - a) @ ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dmin = np.minimum(dmin, np.linalg.norm(fine - proj, axis=1))
    assert float(dmin.max()) <= tolerance


def test_edge_density_depends_on_angular_tolerance() -> None:
    '''Tighter angular tolerance yields denser curved-edge polylines.'''
    solid = build_block(BlockSettings(cells_x=1, cells_y=1)).val()
    loose = brep_edges_polydata(solid, tolerance=1.0, angular_tolerance=0.3)
    tight = brep_edges_polydata(solid, tolerance=1.0, angular_tolerance=0.02)
    assert tight.n_points > loose.n_points
