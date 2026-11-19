'''
File:   test_slope_builder.py
Brief:  Unit tests for per-cell sloped-top wedge cells.
Author: Mistress-Lukutar
Date:   2026-11-19
Version: v0.11.0
'''

from __future__ import annotations

import math

import cadquery as cq
import pytest

from brickgen.core.builder import _slope_cell_solid, build_block
from brickgen.core.models import BlockSettings


def _wedge_volume(settings: BlockSettings, rise_units: int) -> float:
    '''Analytic outer volume of one slope cell wedge, mm^3.

    Trapezoid cross-section (heights ``block_height`` and
    ``block_height + rise``) over the cell footprint.
    '''
    unit = settings.unit_size
    h = settings.block_height
    return unit * unit * (h + rise_units * h / 2.0)


def test_wedge_cell_helper_volume() -> None:
    '''The slope cell helper builds a trapezoid prism of the exact volume.'''
    settings = BlockSettings(cells_x=2, cells_y=2)
    solid = _slope_cell_solid(settings, 0, 1, "+Y", 1).val()
    assert solid.Volume() == pytest.approx(_wedge_volume(settings, 1), rel=1e-6)
    bb = solid.BoundingBox()
    assert bb.zmax == pytest.approx(2 * settings.block_height, abs=1e-4)
    assert bb.ymin == pytest.approx(0.0, abs=1e-4)
    assert bb.ymax == pytest.approx(settings.unit_size, abs=1e-4)


def test_single_slope_cell_volume() -> None:
    '''1x2 with one slope cell: hollow shell + stud + hollow wedge, one solid.

    Sloped cells keep the standard bottom opening: their cavity fuses with
    the neighbor's, leaving a wall strip only toward the grid boundary.
    '''
    settings = BlockSettings(cells_x=1, cells_y=2, sloped_cells=[[0, 1, "+Y", 1]])
    solid = build_block(settings).val()
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    # each cell's cavity: inset t on boundary sides, open toward the neighbor
    cavity = (u - 2 * t) * (u - t) * (h - t)
    stud = math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
    expected = u * u * h + _wedge_volume(settings, 1) - 2 * cavity + stud
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    assert solid.BoundingBox().zmax == pytest.approx(2 * h, abs=1e-4)
    assert len(build_block(settings).solids().vals()) == 1


def test_slope_cell_has_no_stud() -> None:
    '''The wedge cell top carries no stud even with the default on.'''
    settings = BlockSettings(cells_x=1, cells_y=2, sloped_cells=[[0, 1, "+Y", 1]])
    assert settings.cell_top(0, 1) == "slope"
    assert settings.cell_top(0, 0) == "stud"
    assert (0, 1) not in settings.stud_cells_all()


def test_all_cells_sloped_volume() -> None:
    '''A fully sloped stud-less block keeps one fused cavity under the wedges.'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=4,
        pins_enabled=False,
        tubes_enabled=False,
        sloped_cells=[[ix, iy, "+X", 1] for iy in range(4) for ix in range(2)],
    )
    solid = build_block(settings).val()
    lx = settings.outer_length_x
    ly = settings.outer_length_y
    h = settings.block_height
    t = settings.wall_thickness
    expected = lx * ly * (h + h / 2.0) - (lx - 2 * t) * (ly - 2 * t) * (h - t)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    assert solid.BoundingBox().zmax == pytest.approx(2 * h, abs=1e-4)
    assert len(build_block(settings).solids().vals()) == 1


def test_slope_sides_are_mirror_symmetric() -> None:
    '''+X and -X wedges on mirrored cells have equal volume.'''
    plus = BlockSettings(
        cells_x=2, cells_y=1, pins_enabled=False, sloped_cells=[[1, 0, "+X", 2]]
    )
    minus = BlockSettings(
        cells_x=2, cells_y=1, pins_enabled=False, sloped_cells=[[0, 0, "-X", 2]]
    )
    v_plus = build_block(plus).val().Volume()
    v_minus = build_block(minus).val().Volume()
    assert v_plus == pytest.approx(v_minus, rel=1e-6)


def test_slope_and_chamfer_combine() -> None:
    '''A slope top and a bottom chamfer can share one solid cell.

    The chamfer makes the cell solid (no bottom opening), so the combined
    cell is a wedge on top and a wedge at the bottom.
    '''
    settings = BlockSettings(
        cells_x=1,
        cells_y=2,
        sloped_cells=[[0, 0, "+Y", 1]],
        chamfered_cells=[[0, 0, "-Y"]],
    )
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    # bottom tip toward the chamfer side (-Y): the wedge thins to zero there
    assert not solid.isInside(cq.Vector(0.0, -3.9, 0.1))
    # near the raised edge (+Y side of the cell) the wedge is full height
    assert solid.isInside(cq.Vector(0.0, -0.1, 5.9))
    # the chamfered cell is solid: material right above the bottom wedge
    assert solid.isInside(cq.Vector(0.0, -2.0, 2.0))


def test_slope_cell_cavity_fuses_with_neighbor() -> None:
    '''Sloped cells keep a normal cavity that fuses with plain neighbors.'''
    settings = BlockSettings(
        cells_x=1, cells_y=2, pins_enabled=False, sloped_cells=[[0, 1, "+Y", 1]]
    )
    solid = build_block(settings).val()
    # cavity void in the normal cell (0,0), away from the walls
    assert not solid.isInside(cq.Vector(0.0, -2.0, 0.3))
    # no wall strip anymore: the slope cell's cavity is open toward (0,0)
    assert not solid.isInside(cq.Vector(0.0, -0.4, 0.3))
    # the slope cell keeps its boundary wall on the outer (+Y) side
    assert solid.isInside(cq.Vector(0.0, 3.6, 0.3))
    # and its side walls
    assert solid.isInside(cq.Vector(-1.5, 2.0, 0.3))


def test_explicit_stud_on_studless_block() -> None:
    '''``stud_cells`` add a stud even when the default top is stud-less.'''
    settings = BlockSettings(
        cells_x=2, cells_y=1, pins_enabled=False, stud_cells=[[1, 0]]
    )
    solid = build_block(settings).val()
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    shell = 2 * u * u * h - (2 * u - 2 * t) * (u - 2 * t) * (h - t)
    stud = math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
    assert solid.Volume() == pytest.approx(shell + stud, rel=1e-3)
    assert solid.BoundingBox().zmax == pytest.approx(settings.total_height, abs=1e-4)


def test_flat_override_on_default_block() -> None:
    '''``flat_cells`` removes the stud from one cell only.'''
    settings = BlockSettings(cells_x=2, cells_y=1, flat_cells=[[0, 0]])
    solid = build_block(settings).val()
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    shell = 2 * u * u * h - (2 * u - 2 * t) * (u - 2 * t) * (h - t)
    stud = math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
    assert solid.Volume() == pytest.approx(shell + stud, rel=1e-3)
    assert settings.cell_top(0, 0) == "flat"
    assert settings.cell_top(1, 0) == "stud"
