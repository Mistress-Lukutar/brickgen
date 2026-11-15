'''
File:   builder.py
Brief:  CadQuery geometry builder for parametric LEGO-like stud blocks.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.1.0
'''

from __future__ import annotations

import cadquery as cq

from brickgen.core.models import BlockSettings

# Small overshoot used when cutting through faces to avoid coincident boolean faces.
CUT_OVERSHOOT = 1.0


def build_block(settings: BlockSettings) -> cq.Workplane:
    '''Build the solid model of the block described by ``settings``.

    Coordinate convention (same as the reference LPe12000 project):
    X/Y - footprint plane, Z - up, block bottom rim sits at Z=0,
    top face at Z=``block_height``, studs extend up to Z=``total_height``.

    Args:
        settings: Validated parametric block description.

    Returns:
        A CadQuery workplane holding a single fused solid.
    '''
    length_x = settings.outer_length_x
    length_y = settings.outer_length_y

    # Outer shell: box from Z=0 up to the top face.
    outer = cq.Workplane("XY").box(
        length_x, length_y, settings.block_height, centered=(True, True, False)
    )

    # Inner cavity: open from below, leaving a top face of wall thickness.
    cavity_x = length_x - 2.0 * settings.wall_thickness
    cavity_y = length_y - 2.0 * settings.wall_thickness
    cavity_z = settings.block_height - settings.wall_thickness + CUT_OVERSHOOT
    outer = outer.cut(
        cq.Workplane("XY")
        .box(cavity_x, cavity_y, cavity_z, centered=(True, True, False))
        .translate((0, 0, -CUT_OVERSHOOT))
    )

    if settings.pins_enabled:
        outer = _fuse_pins(outer, settings)

    return outer


def _fuse_pins(shell: cq.Workplane, settings: BlockSettings) -> cq.Workplane:
    '''Fuse one cylindrical stud per 1x1 cell onto the shell top face.

    Args:
        shell: Hollow shell to attach studs to.
        settings: Validated parametric block description.

    Returns:
        The shell with all studs fused into a single solid.
    '''
    radius = settings.pin_diameter / 2.0
    for ix in range(settings.cells_x):
        for iy in range(settings.cells_y):
            cx = (ix + 0.5) * settings.unit_size - settings.outer_length_x / 2.0
            cy = (iy + 0.5) * settings.unit_size - settings.outer_length_y / 2.0
            stud = (
                cq.Workplane("XY", origin=(cx, cy, settings.block_height))
                .circle(radius)
                .extrude(settings.pin_height)
            )
            shell = shell.union(stud)
    return shell
