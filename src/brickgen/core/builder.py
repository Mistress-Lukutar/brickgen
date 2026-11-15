'''
File:   builder.py
Brief:  CadQuery geometry builder for parametric LEGO-like stud blocks.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.4.1
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

    Stud-less blocks with ``slope_rise_units > 0`` get a sloped top: the
    ``slope_side`` edge rises by ``slope_rise`` and the top face slopes down
    to the opposite edge (tubes and studs do not apply in that variant).

    Args:
        settings: Validated parametric block description.

    Returns:
        A CadQuery workplane holding a single fused solid.
    '''
    if settings.slope_active:
        return _build_sloped_block(settings)
    return _build_flat_block(settings)


def _build_flat_block(settings: BlockSettings) -> cq.Workplane:
    '''Build the standard flat-topped block (studs and/or tubes).

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
    if settings.has_tubes:
        outer = _fuse_tubes(outer, settings)

    return outer


def _build_sloped_block(settings: BlockSettings) -> cq.Workplane:
    '''Build the stud-less block with one raised edge and a sloped top face.

    The outer profile is a trapezoid in the plane of the slope axis: heights
    ``block_height`` on the low side and ``block_height + slope_rise`` on the
    raised side. The cavity mirrors that profile inset by the wall thickness,
    leaving a sloped top face of the wall thickness and an open bottom.

    Args:
        settings: Validated parametric block description.

    Returns:
        A CadQuery workplane holding a single fused solid.
    '''
    t = settings.wall_thickness
    length_x = settings.outer_length_x
    length_y = settings.outer_length_y
    height = settings.block_height
    top_high = height + settings.slope_rise

    along_x = settings.slope_side in ("+X", "-X")
    profile_len = length_x if along_x else length_y
    extrude_len = length_y if along_x else length_x
    high_first = settings.slope_side in ("-X", "-Y")

    # Outer trapezoid: (0,0) -> (L,0) -> (L,top_a) -> (0,top_b).
    top_a = height if high_first else top_high
    top_b = top_high if high_first else height
    outer = _extrude_profile(
        [(0.0, 0.0), (profile_len, 0.0), (profile_len, top_a), (0.0, top_b)],
        extrude_len,
        along_x,
    )

    # Cavity: XY-inset by t on all sides; ceiling strictly parallel to the
    # sloped outer top (sampled from the outer top line at the inset ends),
    # so the top face keeps a uniform vertical thickness t for any rise.
    # Pushed below the rim so the bottom stays open.
    def _top_at(pos: float) -> float:
        '''Outer top-plane height along the slope axis at ``pos``.'''
        fraction = pos / profile_len
        return top_b + (top_a - top_b) * fraction

    cavity = _extrude_profile(
        [
            (t, -CUT_OVERSHOOT),
            (profile_len - t, -CUT_OVERSHOOT),
            (profile_len - t, _top_at(profile_len - t) - t),
            (t, _top_at(t) - t),
        ],
        extrude_len - 2.0 * t,
        along_x,
    )
    outer = outer.cut(cavity)
    return outer


def _extrude_profile(
    points: list[tuple[float, float]],
    extrude_len: float,
    along_x: bool,
) -> cq.Workplane:
    '''Extrude an XZ (or YZ) polygon symmetrically across the other axis.

    The polygon is re-centered on the profile axis using the midpoint of its
    own extent (min..max), so profiles that do not start at zero (e.g. the
    wall-inset cavity) stay centered in the block.

    Args:
        points: Polygon vertices in (axis, height) local coordinates, where
            ``axis`` is X when ``along_x`` else Y.
        extrude_len: Total extrusion length across the perpendicular axis.
        along_x: True to sketch in the XZ plane (extrude along Y).

    Returns:
        The extruded solid, centered in both the profile axis and the
        perpendicular axis.
    '''
    plane = "XZ" if along_x else "YZ"
    solid = (
        cq.Workplane(plane)
        .polyline(points)
        .close()
        .extrude(extrude_len / 2.0, both=True)
    )
    axis_values = [p[0] for p in points]
    center = (min(axis_values) + max(axis_values)) / 2.0
    offset = (-center, 0.0, 0.0) if along_x else (0.0, -center, 0.0)
    return solid.translate(offset)


def _tube_centers(settings: BlockSettings) -> list[tuple[float, float]]:
    '''Compute inner tube centers: mid-lines between stud rows/columns.

    One tube per sliding 2x2 cell group, i.e. at every intersection of the
    mid-lines between adjacent stud columns (X) and rows (Y). A 2x2 block
    gets a single central tube; a 3x2 block gets two tubes along its length.

    Args:
        settings: Validated parametric block description.

    Returns:
        List of (x, y) centers in block coordinates (block centered at origin).
    '''
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0
    return [
        (ix * settings.unit_size - half_x, iy * settings.unit_size - half_y)
        for ix in range(1, settings.cells_x)
        for iy in range(1, settings.cells_y)
    ]


def _fuse_tubes(shell: cq.Workplane, settings: BlockSettings) -> cq.Workplane:
    '''Fuse hollow reinforcement tubes under the inner top face.

    Each tube is an annulus (bore ``tube_inner_diameter``, wall equal to
    ``wall_thickness``) spanning the full cavity depth, closed on top by the
    solid top face. The outer cylinder overshoots into the top face for a
    robust boolean union.

    Args:
        shell: Hollow shell to attach tubes to.
        settings: Validated parametric block description.

    Returns:
        The shell with all tubes fused into a single solid.
    '''
    r_out = settings.tube_outer_diameter / 2.0
    r_in = settings.tube_inner_diameter / 2.0
    for cx, cy in _tube_centers(settings):
        tube = (
            cq.Workplane("XY", origin=(cx, cy, 0))
            .circle(r_out)
            .circle(r_in)
            .extrude(settings.block_height)
        )
        shell = shell.union(tube)
    return shell


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
