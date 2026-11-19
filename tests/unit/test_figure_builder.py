'''
File:   test_figure_builder.py
Brief:  Unit tests for blocks with excluded cells (figure variants).
Author: Mistress-Lukutar
Date:   2026-11-19
Version: v0.12.0
'''

from __future__ import annotations

import math

import pytest

from brickgen.core.builder import _figure_corner_ribs, build_block
from brickgen.core.models import BlockSettings


def _figure_volume(settings: BlockSettings) -> float:
    '''Analytic volume of a figure block, mm^3.

    The cavity is the included region eroded by wall thickness (per-cell
    cavity boxes inset only on sides facing the region boundary, merged
    across shared edges). Plus quarter-round ribs in concave corners.
    '''
    t = settings.wall_thickness
    unit = settings.unit_size
    height = settings.block_height
    included = set(_included(settings))
    cavity = 0.0
    for ix, iy in included:
        wx = (
            unit
            - t * ((ix - 1, iy) not in included)
            - t * ((ix + 1, iy) not in included)
        )
        wy = (
            unit
            - t * ((ix, iy - 1) not in included)
            - t * ((ix, iy + 1) not in included)
        )
        cavity += wx * wy * (height - t)
    r = min(2.0 * t, unit - 2.0 * t)
    rib_area = r * r - math.pi * r * r / 4.0
    pins = (
        len(included)
        * math.pi
        * (settings.pin_diameter / 2.0) ** 2
        * settings.pin_height
        if settings.pins_enabled
        else 0.0
    )
    outer = len(included) * unit * unit * height
    return (
        outer
        - cavity
        + len(_figure_corner_ribs(settings)) * rib_area * (height - t)
        + pins
    )


def _included(settings: BlockSettings) -> list[tuple[int, int]]:
    '''Included cells mirrored from the builder logic.'''
    excluded = {(ix, iy) for ix, iy in settings.excluded_cells}
    return [
        (ix, iy)
        for iy in range(settings.cells_y)
        for ix in range(settings.cells_x)
        if (ix, iy) not in excluded
    ]


def test_excluded_cells_validation() -> None:
    '''Bounds, duplicates and the all-excluded mask are rejected.'''
    with pytest.raises(Exception, match="outside"):
        BlockSettings(cells_x=2, cells_y=2, excluded_cells=[[2, 0]])
    with pytest.raises(Exception, match="twice"):
        BlockSettings(cells_x=2, cells_y=2, excluded_cells=[[0, 0], [0, 0]])
    with pytest.raises(Exception, match="at least one"):
        BlockSettings(
            cells_x=2, cells_y=2, excluded_cells=[[0, 0], [1, 0], [0, 1], [1, 1]]
        )


def test_excluded_cells_normalized_sorted() -> None:
    '''The mask is stored as a sorted list of integer pairs.'''
    settings = BlockSettings(cells_x=3, cells_y=3, excluded_cells=[[2, 1], [0, 0]])
    assert settings.excluded_cells == [[0, 0], [2, 1]]
    assert settings.included_count == 7
    assert not settings.is_rectangular


def test_l_shape_2x2_volume() -> None:
    '''L from 2x2 (one corner excluded): analytic volume, single solid.'''
    settings = BlockSettings(cells_x=2, cells_y=2, excluded_cells=[[1, 1]])
    solid = build_block(settings).val()
    assert len(_figure_corner_ribs(settings)) == 1
    assert solid.Volume() == pytest.approx(_figure_volume(settings), rel=1e-3)
    assert len(build_block(settings).solids().vals()) == 1


def test_t_shape_3x2_volume() -> None:
    '''T from 3x2 (two corners excluded): two ribs, analytic volume.'''
    settings = BlockSettings(cells_x=3, cells_y=2, excluded_cells=[[0, 1], [2, 1]])
    assert len(_figure_corner_ribs(settings)) == 2
    solid = build_block(settings).val()
    assert solid.Volume() == pytest.approx(_figure_volume(settings), rel=1e-3)
    assert len(build_block(settings).solids().vals()) == 1


def test_figure_bbox_covers_grid() -> None:
    '''Bounding box still spans the full grid when border cells remain.'''
    settings = BlockSettings(cells_x=3, cells_y=2, excluded_cells=[[0, 1], [2, 1]])
    bb = build_block(settings).val().BoundingBox()
    assert bb.xlen == pytest.approx(12.0, abs=1e-4)
    assert bb.ylen == pytest.approx(8.0, abs=1e-4)
    assert bb.zmax == pytest.approx(settings.total_height, abs=1e-4)


def test_figure_no_pins_variant() -> None:
    '''Figure without studs: pins contribute nothing, still one solid.'''
    settings = BlockSettings(
        cells_x=2, cells_y=2, pins_enabled=False, excluded_cells=[[1, 1]]
    )
    solid = build_block(settings).val()
    assert solid.Volume() == pytest.approx(_figure_volume(settings), rel=1e-3)
    assert len(build_block(settings).solids().vals()) == 1


def test_figure_no_ribs_without_concave_corner() -> None:
    '''A 1x2 strip has no concave corners, hence no ribs.'''
    settings = BlockSettings(cells_x=1, cells_y=2, excluded_cells=[[0, 0]])
    assert _figure_corner_ribs(settings) == []
    solid = build_block(settings).val()
    assert solid.Volume() == pytest.approx(_figure_volume(settings), rel=1e-3)


def test_slope_cell_in_figure_variant() -> None:
    '''A slope cell inside an L figure: hollow wedge, rib at the corner.'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        excluded_cells=[[1, 1]],
        sloped_cells=[[1, 0, "+X", 1]],
    )
    # the sloped cell keeps a normal cavity, so the L corner has exactly
    # 3 cavity quadrants -> one quarter-round rib
    assert len(_figure_corner_ribs(settings)) == 1
    solid = build_block(settings).val()
    assert solid.BoundingBox().zmax == pytest.approx(
        2 * settings.block_height, abs=1e-4
    )
    assert len(build_block(settings).solids().vals()) == 1
