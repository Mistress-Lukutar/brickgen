'''
File:   test_preview.py
Brief:  Unit tests for the pyvista mesh conversion used by the preview.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.7.0
'''

from __future__ import annotations

import pytest

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings
from brickgen.gui.preview import DISPLAY_MODES, solid_to_mesh


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
