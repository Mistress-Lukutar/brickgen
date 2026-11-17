'''
File:   test_round_cell.py
Brief:  Unit tests for round-base cells (inscribed cylinders with round holes).
Author: Mistress-Lukutar
Date:   2026-11-17
Version: v0.1.0
'''

from __future__ import annotations

import math

import cadquery as cq
import pytest

from brickgen.core.builder import ROUND_CELL_OVERSHOOT, build_block
from brickgen.core.models import BlockSettings


def _material_at(solid: cq.Shape, point: tuple[float, float, float]) -> bool:
    '''Point membership probe for the solid.'''
    return solid.isInside(cq.Vector(*point))


def _round_volumes(settings: BlockSettings) -> tuple[float, float]:
    '''Outer and hole volumes of one round cell, mm^3.

    The outer cylinder carries ROUND_CELL_OVERSHOOT so neighboring cells
    fuse; the hole is the inscribed circle inset by the wall thickness.
    '''
    r_out = settings.unit_size / 2.0 + ROUND_CELL_OVERSHOOT
    r_in = settings.unit_size / 2.0 - settings.wall_thickness
    h = settings.block_height
    t = settings.wall_thickness
    return math.pi * r_out**2 * h, math.pi * r_in**2 * (h - t)


def test_round_cell_volume() -> None:
    '''A 1x1 round block: cylinder shell minus the round hole plus a stud.'''
    settings = BlockSettings(cells_x=1, cells_y=1, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    outer, hole = _round_volumes(settings)
    stud = math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
    assert solid.Volume() == pytest.approx(outer - hole + stud, rel=1e-3)
    bb = solid.BoundingBox()
    assert bb.xlen == pytest.approx(
        settings.unit_size + 2 * ROUND_CELL_OVERSHOOT, abs=1e-3
    )
    assert bb.zmax == pytest.approx(settings.total_height, abs=1e-4)
    assert len(build_block(settings).solids().vals()) == 1


def test_round_cell_hole_is_round() -> None:
    '''The bottom hole follows the inscribed circle, not the square.'''
    settings = BlockSettings(cells_x=1, cells_y=1, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    t = settings.wall_thickness
    # center of the cell at cavity depth: inside the round hole
    assert not _material_at(solid, (0.0, 0.0, 0.3))
    # inside the wall ring (between r-t and the outer circle): material
    assert _material_at(solid, (settings.unit_size / 2.0 - t / 2.0, 0.0, 0.3))
    # footprint corner is outside the inscribed circle: no material
    corner = settings.unit_size / 2.0 - 0.1
    assert not _material_at(solid, (corner, corner, 0.3))
    # top slab above the hole stays solid
    assert _material_at(solid, (0.0, 0.0, settings.block_height - t / 2.0))


def test_adjacent_round_cells_keep_own_holes() -> None:
    '''Round holes never fuse: a wall remains at the shared cell edge.

    Tangent cylinders get a tiny radial overlap (ROUND_CELL_OVERSHOOT) so
    OCC fuses the outer bodies into a single solid.
    '''
    settings = BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0], [1, 0]])
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    # shared edge midpoint: outside both holes, inside the wall material
    assert _material_at(solid, (0.0, 0.0, 0.3))
    # each hole is a void at its own center
    assert not _material_at(solid, (-2.0, 0.0, 0.3))
    assert not _material_at(solid, (2.0, 0.0, 0.3))
    outer, hole = _round_volumes(settings)
    stud = math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
    assert solid.Volume() == pytest.approx(2 * (outer - hole) + 2 * stud, rel=1e-2)


def test_round_cell_next_to_square_cell() -> None:
    '''A square neighbor keeps a wall strip toward the round cell's recess.'''
    settings = BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    # square cell cavity is inset t on the side facing the round cell
    assert _material_at(solid, (0.4, 0.0, 0.3))
    # and the square cavity is open toward the far (boundary) wall
    assert not _material_at(solid, (2.0, 0.0, 0.3))
    # the round cell hole is void at its center
    assert not _material_at(solid, (-2.0, 0.0, 0.3))


def test_round_cells_do_not_generate_ribs() -> None:
    '''Round cells carry no box cavity, so they do not form rib corners.'''
    all_round = BlockSettings(
        cells_x=2, cells_y=2, round_cells=[[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    solid = build_block(all_round).val()
    assert len(build_block(all_round).solids().vals()) == 1
    # each round cell keeps a void at its center (no fused box cavities)
    for cx, cy in ((-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)):
        assert not _material_at(solid, (cx, cy, 0.3))
    # and the shared wall material between adjacent holes survives
    assert _material_at(solid, (0.0, -2.0, 0.3))
    assert _material_at(solid, (-2.0, 0.0, 0.3))
