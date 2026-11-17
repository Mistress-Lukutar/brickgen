'''
File:   test_slope_builder.py
Brief:  Unit tests for the sloped-top stud-less block variant.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.4.1
'''

from __future__ import annotations

import pytest
from cadquery import Workplane

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings


def _sloped_volume(settings: BlockSettings) -> float:
    '''Analytic volume of the sloped block (no studs, no tubes), mm^3.

    Outer prism: trapezoid cross-section heights H and H+R; cavity mirrors
    the profile inset by wall thickness and capped t below the top.
    '''
    lx = settings.outer_length_x
    ly = settings.outer_length_y
    t = settings.wall_thickness
    rise = settings.slope_rise
    h = settings.block_height
    outer = lx * ly * (h + rise / 2.0)
    cavity = (lx - 2 * t) * (ly - 2 * t) * (h - t + rise / 2.0)
    return outer - cavity


@pytest.fixture
def slope_2x4() -> BlockSettings:
    '''2x4 slope block, raised edge on +X, rise of one unit height.'''
    return BlockSettings(
        cells_x=2,
        cells_y=4,
        pins_enabled=False,
        tubes_enabled=False,
        slope_rise_units=1,
        slope_side="+X",
    )


def test_slope_active_flag() -> None:
    '''Slope applies only without studs and with a positive rise.'''
    assert BlockSettings(pins_enabled=False, slope_rise_units=1).slope_active
    assert not BlockSettings(pins_enabled=True, slope_rise_units=1).slope_active
    assert not BlockSettings(pins_enabled=False, slope_rise_units=0).slope_active


def test_slope_bounding_box(slope_2x4: BlockSettings) -> None:
    '''Raised edge tops at block_height + slope_rise.'''
    solid = build_block(slope_2x4).val()
    bb = solid.BoundingBox()
    assert bb.xlen == pytest.approx(8.0, abs=1e-4)
    assert bb.ylen == pytest.approx(16.0, abs=1e-4)
    assert bb.zmax == pytest.approx(6.0, abs=1e-4)  # 3 + 1x3
    assert bb.zmin == pytest.approx(0.0, abs=1e-4)


def test_slope_volume_matches_analytic(slope_2x4: BlockSettings) -> None:
    '''Sloped block volume equals the trapezoid-prism analytic value.'''
    solid = build_block(slope_2x4).val()
    assert solid.Volume() == pytest.approx(_sloped_volume(slope_2x4), rel=1e-3)


def test_slope_sides_are_mirror_symmetric() -> None:
    '''+X and -X variants have equal volume (mirrored profiles).'''
    plus = BlockSettings(
        cells_x=3, cells_y=2, pins_enabled=False, slope_rise_units=2, slope_side="+X"
    )
    minus = plus.model_copy(update={"slope_side": "-X"})
    v_plus = build_block(plus).val().Volume()
    v_minus = build_block(minus).val().Volume()
    assert v_plus == pytest.approx(v_minus, rel=1e-6)
    assert v_plus == pytest.approx(_sloped_volume(plus), rel=1e-3)


def test_slope_along_y() -> None:
    '''Y-direction slopes match the analytic volume as well.'''
    settings = BlockSettings(
        cells_x=2, cells_y=3, pins_enabled=False, slope_rise_units=1, slope_side="+Y"
    )
    solid = build_block(settings).val()
    assert solid.BoundingBox().zmax == pytest.approx(6.0, abs=1e-4)
    assert solid.Volume() == pytest.approx(_sloped_volume(settings), rel=1e-3)


def test_slope_variant_is_single_solid(slope_2x4: BlockSettings) -> None:
    '''The sloped shell is one solid body.'''
    assert len(build_block(slope_2x4).solids().vals()) == 1


def test_slope_ignored_when_pins_enabled() -> None:
    '''With studs on top the slope parameters have no effect.'''
    plain = BlockSettings(cells_x=2, cells_y=2)
    sloped = plain.model_copy(update={"slope_rise_units": 3})
    v_plain = build_block(plain).val().Volume()
    v_sloped = build_block(sloped).val().Volume()
    assert v_plain == pytest.approx(v_sloped, rel=1e-6)


def test_slope_cavity_is_centered(slope_2x4: BlockSettings) -> None:
    '''Bottom rim walls must be uniform: cavity centered in the footprint.

    Regression: the cavity used to be shifted by wall_thickness/2 along the
    slope axis, producing rim walls of t/2 and 3t/2.
    '''
    t = slope_2x4.wall_thickness
    slab = (
        Workplane("XY")
        .box(
            slope_2x4.outer_length_x,
            slope_2x4.outer_length_y,
            0.1,
            centered=(True, True, False),
        )
        .intersect(build_block(slope_2x4))
    )
    ring = slab.val().Volume()
    rim_area = slope_2x4.outer_length_x * slope_2x4.outer_length_y - (
        slope_2x4.outer_length_x - 2 * t
    ) * (slope_2x4.outer_length_y - 2 * t)
    assert ring == pytest.approx(rim_area * 0.1, rel=1e-3)


def test_steep_slope_has_no_hole() -> None:
    '''Steep slopes keep a uniform top face thickness (no slit hole).

    Regression: a non-parallel cavity ceiling used to poke through the
    sloped top face when rise exceeded the slope-axis length.
    '''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        pins_enabled=False,
        tubes_enabled=False,
        slope_rise_units=3,
        slope_side="+X",
    )
    solid = build_block(settings).val()
    bb = solid.BoundingBox()
    assert bb.zmax == pytest.approx(12.0, abs=1e-4)
    # A through-hole in the top face would remove material below analytic.
    assert solid.Volume() == pytest.approx(_sloped_volume(settings), rel=1e-3)


def test_top_face_thickness_is_uniform() -> None:
    '''Vertical distance between outer top plane and cavity ceiling stays t.

    Sampled at several positions along the slope axis via thin slab cuts.
    '''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        pins_enabled=False,
        tubes_enabled=False,
        slope_rise_units=2,
        slope_side="+X",
    )
    t = settings.wall_thickness
    lx = settings.outer_length_x
    solid = build_block(settings)
    for pos in (0.25, 0.5, 0.75):
        # Global x of the sample point along the slope axis.
        x = -lx / 2.0 + pos * lx
        # Thin probe column at the block center: inside the cavity the
        # material spans ceiling..outer-top, so bbox zmin is the ceiling.
        probe = (
            Workplane("XY", origin=(x, 0, 0))
            .rect(0.02, 0.02)
            .extrude(settings.block_height + settings.slope_rise + 1.0)
        )
        column = probe.intersect(solid)
        bb = column.val().BoundingBox()
        outer_top = settings.block_height + settings.slope_rise * pos
        # bbox spans the 0.02 mm probe width; allow that in tolerances.
        assert bb.zmin == pytest.approx(outer_top - t, abs=0.02)
        assert bb.zmax == pytest.approx(outer_top, abs=0.02)
