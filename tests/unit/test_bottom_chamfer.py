'''
File:   test_bottom_chamfer.py
Brief:  Unit tests for bottom-chamfered cells (sloped underside wedges).
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.21.0
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


def test_sloped_cells_validation() -> None:
    '''Chamfer sides must look outward; conflicts are rejected.'''
    # neighbor across +X is an included cell -> invalid
    with pytest.raises(Exception, match="outward"):
        BlockSettings(cells_x=2, cells_y=2, chamfered_cells=[[0, 0, "+X"]])
    # neighbor excluded -> valid
    settings = BlockSettings(
        cells_x=2, cells_y=2, excluded_cells=[[1, 0]], chamfered_cells=[[0, 0, "+X"]]
    )
    assert settings.chamfered_cells == [[0, 0, "+X"]]
    # sloped cell cannot be excluded
    with pytest.raises(Exception, match="excluded and chamfered"):
        BlockSettings(
            cells_x=2,
            cells_y=2,
            excluded_cells=[[1, 1]],
            chamfered_cells=[[1, 1, "+Y"]],
        )
    # unknown side rejected
    with pytest.raises(Exception, match="must be one of"):
        BlockSettings(cells_x=2, cells_y=1, chamfered_cells=[[0, 0, "+Z"]])
    # duplicate triplet rejected
    with pytest.raises(Exception, match="twice"):
        BlockSettings(
            cells_x=2, cells_y=1, chamfered_cells=[[0, 0, "-X"], [0, 0, "-X"]]
        )
    # out of bounds rejected
    with pytest.raises(Exception, match="outside"):
        BlockSettings(cells_x=2, cells_y=1, chamfered_cells=[[2, 0, "+X"]])


def test_two_sides_on_one_cell_allowed() -> None:
    '''A corner cell chamfered on two outward sides is a valid hip wedge.'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        excluded_cells=[[1, 1]],
        chamfered_cells=[[1, 0, "+X"], [1, 0, "+Y"]],
    )
    assert settings.chamfered_cells == [[1, 0, "+X"], [1, 0, "+Y"]]


def test_sloped_l_shape_probes() -> None:
    '''L sloped from two sides: wedge void/material at expected spots.'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        excluded_cells=[[1, 1]],
        chamfered_cells=[[1, 0, "+X"], [0, 1, "+Y"]],
    )
    solid = build_block(settings).val()
    # +X arm: bottom void near the outer face, material near the top
    assert not _material_at(solid, (3.9, -2.0, 0.5))
    assert _material_at(solid, (3.9, -2.0, 2.95))
    # wedge thins toward +X: no material at inner-bottom depth 2.0 (below plane)
    assert not _material_at(solid, (2.0, -2.0, 0.3))
    # full material at the inner face (merges with the rest of the block)
    assert _material_at(solid, (0.2, -2.0, 0.3))
    # +Y arm mirrors the same behavior
    assert not _material_at(solid, (-2.0, 3.9, 0.5))
    assert _material_at(solid, (-2.0, 3.9, 2.95))
    # normal corner cell keeps its cavity
    assert not _material_at(solid, (-2.0, -2.0, 0.3))
    # outer wall of the normal cell stays solid
    assert _material_at(solid, (-3.9, -2.0, 0.3))
    # single fused solid
    assert len(build_block(settings).solids().vals()) == 1


def test_chamfer_reduces_volume() -> None:
    '''Adding chamfers only removes material.'''
    base = BlockSettings(cells_x=2, cells_y=2, excluded_cells=[[1, 1]])
    chamfered = base.model_copy(update={"chamfered_cells": [[1, 0, "+X"]]})
    v_base = build_block(base).val().Volume()
    v_cham = build_block(chamfered).val().Volume()
    assert 0 < v_cham < v_base


def test_wedge_bar_2x1_solid() -> None:
    '''A 1x2 bar chamfered on both ends is one solid wedge pair.'''
    settings = BlockSettings(
        cells_x=2, cells_y=1, chamfered_cells=[[0, 0, "-X"], [1, 0, "+X"]]
    )
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    # tips: no material at the outer bottom corners
    assert not _material_at(solid, (-3.9, -0.5, 0.1))
    assert not _material_at(solid, (3.9, -0.5, 0.1))
    # full material at the shared middle bottom
    assert _material_at(solid, (0.0, -0.5, 0.1))
    # top face is flat: studs present at cell centers
    assert _material_at(solid, (1.0, 0.0, settings.total_height - 0.1))
    # volume: two wedges (2 * U^2*H/2) plus two studs
    u, h = settings.unit_size, settings.block_height
    studs = 2 * math.pi * (settings.pin_diameter / 2.0) ** 2 * settings.pin_height
    assert solid.Volume() == pytest.approx(u * u * h + studs, rel=1e-3)


def test_rectangular_without_chamfers_uses_tubes() -> None:
    '''A plain block is untouched by the chamfer machinery.'''
    settings = BlockSettings(cells_x=2, cells_y=2)
    solid = build_block(settings).val()
    assert len(build_block(settings).solids().vals()) == 1
    # tube bore void at the central midpoint
    assert not _material_at(solid, (0.0, 0.0, 1.0))


def test_internal_wall_between_cavity_and_wedge_void() -> None:
    '''Regression (user report): the bottom opening must not extend into
    neighboring cells. The normal cell's cavity is inset by wall thickness
    on sides facing a chamfered cell, creating an internal wall between
    the cavity and the void under the wedge.
    '''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        excluded_cells=[[1, 1]],
        chamfered_cells=[[1, 0, "+X"], [0, 1, "+Y"]],
    )
    solid = build_block(settings).val()
    # wall strip between the corner cavity and the +X wedge void
    # (cell (0,0) spans x -4..0; cavity now ends at x = -wall = -0.8)
    assert _material_at(solid, (-0.4, -2.0, 0.3))
    # cavity side of the wall stays void
    assert not _material_at(solid, (-1.5, -2.0, 0.3))
    # and the same wall against the +Y wedge void
    assert _material_at(solid, (-2.0, -0.4, 0.3))
    assert not _material_at(solid, (-2.0, -1.5, 0.3))


def test_no_rounded_rib_beside_chamfered_cells() -> None:
    '''A single cavity between chamfered cells must have square corners.'''
    settings = BlockSettings(
        cells_x=2,
        cells_y=2,
        excluded_cells=[[1, 1]],
        chamfered_cells=[[1, 0, "+X"], [0, 1, "+Y"]],
    )
    assert _figure_corner_ribs(settings) == []
    solid = build_block(settings).val()
    t = settings.wall_thickness
    # Both probes lie in the cavity, including the former rib's arc region.
    assert not _material_at(solid, (-t - 0.02, -t - 0.02, 1.0))
    assert not _material_at(solid, (-t - 1.0, -t - 1.0, 1.0))
