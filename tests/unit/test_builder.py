'''
File:   test_builder.py
Brief:  Unit tests for the CadQuery block builder geometry.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import math

import pytest

from brickgen.core.builder import _tube_centers, build_block
from brickgen.core.models import BlockSettings


def _expected_volume(settings: BlockSettings) -> float:
    '''Analytic volume of the parametric block, mm^3.

    Shell = outer box minus the cavity (open bottom, solid top face of wall
    thickness); studs are cylinders on the top face; tubes are annuli in the
    cavity, closed on top by the top face (their part inside the ceiling
    slab adds no volume).
    '''
    outer = settings.outer_length_x * settings.outer_length_y * settings.block_height
    cavity_x = settings.outer_length_x - 2.0 * settings.wall_thickness
    cavity_y = settings.outer_length_y - 2.0 * settings.wall_thickness
    cavity_depth = settings.block_height - settings.wall_thickness
    shell = outer - cavity_x * cavity_y * cavity_depth
    pin = (
        (
            math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
            if settings.pins_enabled
            else 0.0
        )
        * settings.cells_x
        * settings.cells_y
    )
    r_out = settings.tube_outer_diameter / 2.0
    r_in = settings.tube_inner_diameter / 2.0
    tubes = (
        len(_tube_centers(settings)) * math.pi * (r_out**2 - r_in**2) * cavity_depth
        if settings.has_tubes
        else 0.0
    )
    return shell + pin + tubes


def test_1x1_bounding_box(default_settings: BlockSettings) -> None:
    '''1x1 block: 4x4 mm footprint, height = body + stud.'''
    solid = build_block(default_settings).val()
    bb = solid.BoundingBox()
    assert bb.xlen == pytest.approx(4.0, abs=1e-4)
    assert bb.ylen == pytest.approx(4.0, abs=1e-4)
    assert bb.zlen == pytest.approx(default_settings.total_height, abs=1e-4)
    assert bb.zmin == pytest.approx(0.0, abs=1e-4)


def test_1x1_volume_matches_analytic(default_settings: BlockSettings) -> None:
    '''Solid volume must equal shell + studs analytic value.'''
    solid = build_block(default_settings).val()
    assert solid.Volume() == pytest.approx(_expected_volume(default_settings), rel=1e-3)


def test_negative_clearance_keeps_block_size() -> None:
    '''Negative clearance fattens the stud only; the block stays 8x8 nominal.'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        unit_size=4.0,
        wall_thickness=0.8,
        clearance_xy=-0.5,
        clearance_z=-0.5,
    )
    solid = build_block(settings).val()
    bb = solid.BoundingBox()
    # Footprint unchanged (no grid expansion)...
    assert bb.xlen == pytest.approx(8.0, abs=1e-4)
    assert bb.ylen == pytest.approx(8.0, abs=1e-4)
    assert bb.zmin == pytest.approx(0.0, abs=1e-4)
    # ...only the stud grows: 3.0 shell + 2.7 oversized stud.
    assert bb.zlen == pytest.approx(5.7, abs=1e-4)
    assert bb.zlen == pytest.approx(settings.total_height, abs=1e-4)


def test_negative_clearance_volume_matches_analytic() -> None:
    '''The oversized studs add exactly their cylinder volume (block nominal).'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        unit_size=4.0,
        wall_thickness=0.8,
        clearance_xy=-0.5,
        clearance_z=-0.5,
    )
    solid = build_block(settings).val()
    assert solid.Volume() == pytest.approx(_expected_volume(settings), rel=1e-6)


def test_negative_clearance_figure_variant_builds() -> None:
    '''Exclusions, slopes and round cells survive oversized studs.'''
    settings = BlockSettings(
        cells_x=3,
        cells_y=2,
        wall_thickness=0.8,
        clearance_xy=-0.3,
        clearance_z=-0.2,
        excluded_cells=[[2, 1]],
        sloped_cells=[[0, 0, "+X", 1]],
        round_cells=[[2, 0]],
    )
    solid = build_block(settings).val()
    assert len(solid.Solids()) == 1
    assert solid.Volume() > 0.0


def test_2x6_volume_matches_analytic(settings_2x6: BlockSettings) -> None:
    '''2x6 block volume must equal the analytic value for 12 studs and tubes.'''
    solid = build_block(settings_2x6).val()
    bb = solid.BoundingBox()
    assert bb.xlen == pytest.approx(8.0, abs=1e-4)
    assert bb.ylen == pytest.approx(24.0, abs=1e-4)
    assert solid.Volume() == pytest.approx(_expected_volume(settings_2x6), rel=1e-3)


def test_tube_centers_layout() -> None:
    '''Tube centers sit at mid-lines between stud rows/columns.

    2x2 -> one central tube; 3x2 -> two tubes along the length (as on the
    reference photo); 1xN -> none.
    '''
    two_by_two = _tube_centers(BlockSettings(cells_x=2, cells_y=2))
    assert len(two_by_two) == 1
    assert two_by_two[0] == pytest.approx((0.0, 0.0))

    three_by_two = _tube_centers(BlockSettings(cells_x=3, cells_y=2))
    assert len(three_by_two) == 2
    xs = sorted(x for x, _ in three_by_two)
    assert xs == pytest.approx([-2.0, 2.0])  # 4 mm and 8 mm from the edge

    one_by_four = _tube_centers(BlockSettings(cells_x=1, cells_y=4))
    assert one_by_four == []


def test_2x2_volume_matches_analytic() -> None:
    '''2x2 block with a single central tube matches the analytic volume.'''
    settings = BlockSettings(cells_x=2, cells_y=2)
    solid = build_block(settings).val()
    assert solid.Volume() == pytest.approx(_expected_volume(settings), rel=1e-3)


def test_tubes_can_be_disabled() -> None:
    '''Disabling tubes removes their volume contribution.'''
    settings = BlockSettings(cells_x=3, cells_y=2, tubes_enabled=False)
    solid = build_block(settings).val()
    assert solid.Volume() == pytest.approx(_expected_volume(settings), rel=1e-3)


def test_tube_outer_diameter_uses_wall_thickness() -> None:
    '''Tube wall equals the shell wall thickness.'''
    settings = BlockSettings()
    assert settings.tube_outer_diameter == pytest.approx(
        settings.tube_inner_diameter + 2.0 * settings.wall_thickness
    )


def test_block_is_open_from_below(default_settings: BlockSettings) -> None:
    '''The bottom rim is a hollow ring, not a solid plate.'''
    shell_outer = default_settings.unit_size**2
    cavity = (default_settings.unit_size - 2.0 * default_settings.wall_thickness) ** 2
    wall_ring_area = shell_outer - cavity
    slab_expected = wall_ring_area * 0.1
    from cadquery import Workplane

    slab = (
        Workplane("XY")
        .box(4.0, 4.0, 0.1, centered=(True, True, False))
        .intersect(build_block(default_settings))
    )
    assert slab.val().Volume() == pytest.approx(slab_expected, rel=1e-2)


def test_no_pins_variant(default_settings: BlockSettings) -> None:
    '''With studs disabled the height equals body height only.'''
    settings = default_settings.model_copy(update={"pins_enabled": False})
    solid = build_block(settings).val()
    assert solid.BoundingBox().zlen == pytest.approx(settings.block_height, abs=1e-4)
    assert solid.Volume() == pytest.approx(_expected_volume(settings), rel=1e-3)


def test_single_solid(settings_2x6: BlockSettings) -> None:
    '''Studs and tubes must be fused with the shell into one solid body.'''
    solids = build_block(settings_2x6).solids().vals()
    assert len(solids) == 1
