'''
File:   test_round_cell.py
Brief:  Unit tests for round-base cells (free corners on the inscribed circle).
Author: Mistress-Lukutar
Date:   2026-11-17
Version: v0.2.0
'''

from __future__ import annotations

import math

import cadquery as cq
import pytest

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings


def _material_at(solid: cq.Shape, point: tuple[float, float, float]) -> bool:
    '''Point membership probe for the solid.'''
    return solid.isInside(cq.Vector(*point))


def _circle_area(settings: BlockSettings) -> float:
    '''Area of the inscribed circle of one cell, mm^2.'''
    return math.pi * (settings.unit_size / 2.0) ** 2


def _hole_volume(settings: BlockSettings) -> float:
    '''Volume of one round cell's concentric bottom hole, mm^3.'''
    r_in = settings.unit_size / 2.0 - settings.wall_thickness
    return math.pi * r_in**2 * (settings.block_height - settings.wall_thickness)


def _stud_volume(settings: BlockSettings) -> float:
    '''Volume of one top stud, mm^3.'''
    return (
        math.pi
        * (settings.pin_diameter / 2.0) ** 2
        * settings.pin_height
    )


def test_round_cell_volume() -> None:
    '''An isolated 1x1 round cell is the plain inscribed cylinder.'''
    settings = BlockSettings(cells_x=1, cells_y=1, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    expected = _circle_area(settings) * settings.block_height
    expected -= _hole_volume(settings)
    expected += _stud_volume(settings)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    bb = solid.BoundingBox()
    assert bb.xlen == pytest.approx(settings.unit_size, abs=1e-4)
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


def test_two_round_cells_form_stadium() -> None:
    '''Adjacent round cells keep a straight seam: the classic stadium.

    The junction edge stays square on both cells, the free outer corners
    follow the inscribed circle: silhouette = two half-caps + a rectangle.
    '''
    settings = BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0], [1, 0]])
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    u = settings.unit_size
    h = settings.block_height
    # stadium cross-section: one full circle + the u x u junction rectangle
    expected = (_circle_area(settings) + u * u) * h
    expected -= 2 * _hole_volume(settings)
    expected += 2 * _stud_volume(settings)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    # the shared edge midpoint: wall material between the two holes
    assert _material_at(solid, (0.0, 0.0, 0.3))
    # each hole is a void at its own center
    assert not _material_at(solid, (-2.0, 0.0, 0.3))
    assert not _material_at(solid, (2.0, 0.0, 0.3))
    # straight side beyond the circle radius (the junction strips)
    assert _material_at(solid, (-1.0, 1.9, 0.3))
    assert _material_at(solid, (1.0, -1.9, 0.3))
    # free outer corners are rounded away
    assert not _material_at(solid, (-3.9, -1.9, 0.3))
    assert not _material_at(solid, (3.9, 1.9, 0.3))


def test_round_cell_next_to_square_cell() -> None:
    '''A round cell joins a square neighbor flush (flat junction side).'''
    settings = BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    # round cell silhouette: "D" = circle + strip toward the square neighbor
    d_area = _circle_area(settings) / 2.0 + u * u / 2.0
    # the square cell's cavity opens toward the round cell: inset only on
    # its boundary sides (x1, y0, y1)
    square_cavity = (u - t) * (u - 2 * t) * (h - t)
    expected = d_area * h
    expected -= _hole_volume(settings)
    expected += u * u * h - square_cavity
    expected += 2 * _stud_volume(settings)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    # wall between the square cavity and the round hole (round cell side)
    assert _material_at(solid, (-0.4, 0.0, 0.3))
    # the square cavity reaches the shared plane (open side)
    assert not _material_at(solid, (0.4, 0.0, 0.3))
    # the round cell hole is a void at its center
    assert not _material_at(solid, (-2.0, 0.0, 0.3))
    # the straight junction side extends beyond the circle (the strip)
    assert _material_at(solid, (-1.0, 1.9, 0.3))


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
