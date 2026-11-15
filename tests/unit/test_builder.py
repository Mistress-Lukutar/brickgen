'''
File:   test_builder.py
Brief:  Unit tests for the CadQuery block builder geometry.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.1.0
'''

from __future__ import annotations

import math

import pytest

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings


def _expected_volume(settings: BlockSettings) -> float:
    '''Analytic volume of the parametric block, mm^3.

    Shell = outer box minus the cavity (open bottom, solid top face of
    wall thickness); studs are simple cylinders on the top face.
    '''
    outer = settings.outer_length_x * settings.outer_length_y * settings.block_height
    cavity_x = settings.outer_length_x - 2.0 * settings.wall_thickness
    cavity_y = settings.outer_length_y - 2.0 * settings.wall_thickness
    cavity_depth = settings.block_height - settings.wall_thickness
    cavity = cavity_x * cavity_y * cavity_depth
    pin = (
        math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
        if settings.pins_enabled
        else 0.0
    )
    return outer - cavity + pin * settings.cells_x * settings.cells_y


def test_1x1_bounding_box(default_settings: BlockSettings) -> None:
    '''1x1 block: 4x4 mm footprint, 8 mm body + 2 mm stud.'''
    solid = build_block(default_settings).val()
    bb = solid.BoundingBox()
    assert bb.xlen == pytest.approx(4.0, abs=1e-4)
    assert bb.ylen == pytest.approx(4.0, abs=1e-4)
    assert bb.zlen == pytest.approx(10.0, abs=1e-4)
    assert bb.zmin == pytest.approx(0.0, abs=1e-4)


def test_1x1_volume_matches_analytic(default_settings: BlockSettings) -> None:
    '''Solid volume must equal shell + studs analytic value.'''
    solid = build_block(default_settings).val()
    assert solid.Volume() == pytest.approx(_expected_volume(default_settings), rel=1e-3)


def test_2x6_volume_matches_analytic(settings_2x6: BlockSettings) -> None:
    '''2x6 block volume must equal the analytic value for 12 studs.'''
    solid = build_block(settings_2x6).val()
    bb = solid.BoundingBox()
    assert bb.xlen == pytest.approx(8.0, abs=1e-4)
    assert bb.ylen == pytest.approx(24.0, abs=1e-4)
    assert solid.Volume() == pytest.approx(_expected_volume(settings_2x6), rel=1e-3)


def test_block_is_open_from_below(default_settings: BlockSettings) -> None:
    '''The bottom rim is a hollow ring, not a solid plate.'''
    # A section at the very bottom must have the wall-area cross-section.
    shell_outer = 4.0 * 4.0
    cavity = 2.6 * 2.6
    wall_ring_area = shell_outer - cavity
    solid = build_block(default_settings).val()
    section = solid.faces()  # smoke: builder succeeded
    assert section is not None
    # Bottom area check via a thin slab volume difference: solid volume minus
    # a 0.1 mm solid slab of the closed variant equals the open variant only
    # if there is no bottom plate; direct check: volume of solid below z=0.1.
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
    assert solid.BoundingBox().zlen == pytest.approx(8.0, abs=1e-4)
    assert solid.Volume() == pytest.approx(_expected_volume(settings), rel=1e-3)


def test_single_solid(settings_2x6: BlockSettings) -> None:
    '''All studs must be fused with the shell into one solid body.'''
    solids = build_block(settings_2x6).solids().vals()
    assert len(solids) == 1
