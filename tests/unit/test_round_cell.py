'''
File:   test_round_cell.py
Brief:  Unit tests for round-base cells (stadium silhouette, fused cavities).
Author: Mistress-Lukutar
Date:   2026-11-18
Version: v0.2.0
'''

from __future__ import annotations

import math

import cadquery as cq
import pytest

from brickgen.core.builder import _figure_corner_ribs, build_block
from brickgen.core.models import BlockSettings


def _material_at(solid: cq.Shape, point: tuple[float, float, float]) -> bool:
    '''Point membership probe for the solid.'''
    return solid.isInside(cq.Vector(*point))


def _circle_area(settings: BlockSettings) -> float:
    '''Area of the inscribed circle of one cell, mm^2.'''
    return math.pi * (settings.unit_size / 2.0) ** 2


def _stud_volume(settings: BlockSettings) -> float:
    '''Volume of one top stud, mm^3.'''
    return math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height


def test_round_cell_volume() -> None:
    '''An isolated 1x1 round cell is the plain inscribed cylinder.'''
    settings = BlockSettings(cells_x=1, cells_y=1, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    expected = _circle_area(settings) * settings.block_height
    expected -= (
        math.pi
        * (settings.unit_size / 2.0 - settings.wall_thickness) ** 2
        * (settings.block_height - settings.wall_thickness)
    )
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
    The bottom cavities fuse across the shared edge into one offset
    stadium void without a partition wall (1xN blocks carry no tubes).
    '''
    settings = BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0], [1, 0]])
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    # stadium cross-section: one full circle + the u x u junction rectangle
    expected = (_circle_area(settings) + u * u) * h
    # fused cavity: straight channel (u x (u - 2t)) + two half-circle caps
    cavity_area = u * (u - 2.0 * t) + math.pi * (u / 2.0 - t) ** 2
    expected -= cavity_area * (h - t)
    expected += 2 * _stud_volume(settings)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    # the shared edge midpoint: cavities fused, no partition wall
    assert not _material_at(solid, (0.0, 0.0, 0.3))
    assert not _material_at(solid, (0.0, 0.9, 0.3))
    # each hole is a void at its own center
    assert not _material_at(solid, (-2.0, 0.0, 0.3))
    assert not _material_at(solid, (2.0, 0.0, 0.3))
    # straight side beyond the circle radius (the junction strips)
    assert _material_at(solid, (-1.0, 1.9, 0.3))
    assert _material_at(solid, (1.0, -1.9, 0.3))
    # wall between the fused channel and the outer top edge survives
    assert _material_at(solid, (0.0, 1.8, 0.3))
    # free outer corners are rounded away
    assert not _material_at(solid, (-3.9, -1.9, 0.3))
    assert not _material_at(solid, (3.9, 1.9, 0.3))


def test_round_cell_next_to_square_cell() -> None:
    '''A round cell joins a square neighbor flush (flat junction side).

    The square cavity and the round bore fuse across the shared edge
    through the round cell's opening strip: one continuous void.
    '''
    settings = BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    # round cell silhouette: "D" = circle + strip toward the square neighbor
    d_area = _circle_area(settings) / 2.0 + u * u / 2.0
    # fused cavity: square box (inset only on its boundary sides) plus the
    # round cell's strip-to-edge channel and half-circle cap
    cavity_area = (
        (u - t) * (u - 2.0 * t)
        + (u / 2.0) * (u - 2.0 * t)
        + math.pi * (u / 2.0 - t) ** 2 / 2.0
    )
    expected = d_area * h + u * u * h
    expected -= cavity_area * (h - t)
    expected += 2 * _stud_volume(settings)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    # fused: no wall between the square cavity and the round bore
    assert not _material_at(solid, (-0.4, 0.0, 0.3))
    # the square cavity reaches the shared plane (open side)
    assert not _material_at(solid, (0.4, 0.0, 0.3))
    # the round cell hole is a void at its center
    assert not _material_at(solid, (-2.0, 0.0, 0.3))
    # the straight junction side extends beyond the circle (the strip)
    assert _material_at(solid, (-1.0, 1.9, 0.3))


def test_all_round_2x2_fused_cavity_with_tube() -> None:
    '''A 2x2 all-round block: one merged cavity, no walls, central tube.'''
    settings = BlockSettings(
        cells_x=2, cells_y=2, round_cells=[[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    solid = build_block(settings).val()
    assert len(_figure_corner_ribs(settings)) == 0
    assert len(build_block(settings).solids().vals()) == 1
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    outer_area = 4.0 * u * u - (4.0 - math.pi) * (u / 2.0) ** 2
    cavity_area = 4.0 * (u - t) ** 2 - (4.0 - math.pi) * (u / 2.0 - t) ** 2
    r_out = settings.tube_inner_diameter / 2.0 + t
    r_in = settings.tube_inner_diameter / 2.0
    tube_volume = math.pi * (r_out**2 - r_in**2) * (h - t)
    expected = outer_area * h - cavity_area * (h - t)
    expected += tube_volume + 4 * _stud_volume(settings)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    # each round cell keeps a void at its center
    for cx, cy in ((-2.0, -2.0), (2.0, -2.0), (-2.0, 2.0), (2.0, 2.0)):
        assert not _material_at(solid, (cx, cy, 0.3))
    # the former partition walls between the holes are fused away
    assert not _material_at(solid, (-1.2, -1.2, 0.3))
    assert not _material_at(solid, (1.2, 1.2, 0.3))
    assert not _material_at(solid, (0.0, -1.5, 0.3))
    assert not _material_at(solid, (-1.5, 0.0, 0.3))
    # the center vertex carries the reinforcement tube: bore void, wall
    assert not _material_at(solid, (0.0, 0.0, 0.3))
    assert _material_at(solid, (0.0, 1.0, 0.3))


def test_all_round_2x2_without_tubes() -> None:
    '''tubes_enabled=False removes the central tube, cavity stays fused.'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        round_cells=[[0, 0], [1, 0], [0, 1], [1, 1]],
        tubes_enabled=False,
    )
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    u = settings.unit_size
    h = settings.block_height
    t = settings.wall_thickness
    outer_area = 4.0 * u * u - (4.0 - math.pi) * (u / 2.0) ** 2
    cavity_area = 4.0 * (u - t) ** 2 - (4.0 - math.pi) * (u / 2.0 - t) ** 2
    expected = outer_area * h - cavity_area * (h - t)
    expected += 4 * _stud_volume(settings)
    assert solid.Volume() == pytest.approx(expected, rel=1e-3)
    # fused cavity without the tube: the former tube wall spot is void
    assert not _material_at(solid, (0.0, 0.0, 0.3))
    assert not _material_at(solid, (0.0, 1.0, 0.3))


def test_all_round_3x2_tubes() -> None:
    '''A 3x2 all-round block gets two tubes like the flat variant.'''
    settings = BlockSettings(
        cells_x=3,
        cells_y=2,
        round_cells=[[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
    )
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    # tube bores at the mid-lines between stud columns
    assert not _material_at(solid, (-2.0, 0.0, 0.3))
    assert not _material_at(solid, (2.0, 0.0, 0.3))
    # tube walls
    assert _material_at(solid, (-2.0, 1.0, 0.3))
    assert _material_at(solid, (2.0, -1.0, 0.3))
    # fused cavity between and around the tubes
    assert not _material_at(solid, (0.0, -0.5, 0.3))
    assert not _material_at(solid, (0.0, 0.5, 0.3))
    assert not _material_at(solid, (-2.0, 2.0, 0.3))


def test_round_cell_in_2x2_plain_block() -> None:
    '''One round corner cell in a plain 2x2: all four cavities merge.'''
    settings = BlockSettings(cells_x=2, cells_y=2, round_cells=[[0, 0]])
    solid = build_block(settings).val()
    assert len(_figure_corner_ribs(settings)) == 0
    assert len(build_block(settings).solids().vals()) == 1
    # round bore, square cavities and the center vertex are one void
    assert not _material_at(solid, (-2.0, -2.0, 0.3))
    assert not _material_at(solid, (2.0, 2.0, 0.3))
    assert not _material_at(solid, (2.0, -2.0, 0.3))
    assert not _material_at(solid, (-0.6, -2.0, 0.3))
    assert not _material_at(solid, (-2.0, -0.6, 0.3))
    assert not _material_at(solid, (0.0, 0.0, 0.3))
    # the central tube sits on the merged cavity
    assert _material_at(solid, (0.0, 1.0, 0.3))


def test_round_l_shape_keeps_corner_wall() -> None:
    '''A 2x2 all-round L: arms fuse, the concave corner keeps its wall.

    The excluded cell removes the fourth quadrant, so the vertex gets no
    fill box, no rib and no tube (1 missing + 3 round quadrants).
    '''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        round_cells=[[0, 0], [1, 0], [0, 1]],
        excluded_cells=[[1, 1]],
    )
    solid = build_block(settings).val()
    assert len(_figure_corner_ribs(settings)) == 0
    assert len(build_block(settings).solids().vals()) == 1
    # arm holes fuse across the shared edges
    assert not _material_at(solid, (-2.0, -2.0, 0.3))
    assert not _material_at(solid, (2.0, -2.0, 0.3))
    assert not _material_at(solid, (-2.0, 2.0, 0.3))
    assert not _material_at(solid, (-0.5, -2.0, 0.3))
    assert not _material_at(solid, (-2.0, 0.5, 0.3))
    # the concave corner at the excluded cell stays material
    assert _material_at(solid, (0.0, 0.0, 0.3))
