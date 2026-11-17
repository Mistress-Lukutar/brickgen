'''
File:   builder.py
Brief:  CadQuery geometry builder for parametric stud blocks.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.9.5
'''

from __future__ import annotations

from typing import cast

import cadquery as cq

from brickgen.core.models import BlockSettings

# Small overshoot used when cutting through faces to avoid coincident boolean faces.
CUT_OVERSHOOT = 1.0


def _shape_of(wp: cq.Workplane) -> cq.Shape:
    '''Typed view of a workplane's first object.

    CadQuery's ``Workplane.val()`` declares a loose union (Vector | Location |
    Shape | Sketch); every call site here holds a Shape, so narrow it once.
    '''
    return cast(cq.Shape, wp.val())


def _as_single_solid(shape: cq.Shape) -> cq.Shape:
    '''Merge same-domain faces and unwrap a compound holding one solid.

    One-shot multi-tool booleans leave adjacent coplanar/co-cylindrical faces
    unmerged (sequential unions used to merge them implicitly), which both
    inflates the face count and crashes the OCC mesher on large blocks
    (24x24 and up). ``Shape.clean`` (UnifySameDomain) restores the compact
    topology; the operation is geometrically exact.
    '''
    unified = shape.clean()
    solids = unified.Solids()
    return solids[0] if len(solids) == 1 else unified


def _fuse_all(base: cq.Shape, tools: list[cq.Shape]) -> cq.Shape:
    '''Fuse all tools into the base in a single boolean operation.

    Args:
        base: The base solid to fuse into.
        tools: List of tool shapes to fuse with the base.

    Returns:
        A new Shape that is the result of fusing all tools onto the base,
        or *base* unchanged when the tools list is empty.
    '''
    if not tools:
        return base
    return base.fuse(*tools)


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
    if not settings.is_rectangular:
        return _build_figure_block(settings)
    return _build_flat_block(settings)


def _included_cells(settings: BlockSettings) -> list[tuple[int, int]]:
    '''List the material (ix, iy) cells of the figure, sorted.

    Material cells include both normal and bottom-chamfered cells; only
    excluded cells are missing.

    Args:
        settings: Validated parametric block description.

    Returns:
        Sorted list of material cell indices.
    '''
    excluded = {(ix, iy) for ix, iy in settings.excluded_cells}
    return [
        (ix, iy)
        for iy in range(settings.cells_y)
        for ix in range(settings.cells_x)
        if (ix, iy) not in excluded
    ]


def _build_figure_block(settings: BlockSettings) -> cq.Workplane:
    '''Build a block with excluded and/or bottom-chamfered cells.

    The figure variant follows the reference parts: the cavity is one piece
    shaped like the included region (perimeter walls step around excluded
    cells, no internal partition walls), and every concave corner of the
    region carries an inward quarter-round rib instead of a tube. Tubes do
    not apply in this variant; studs remain on top of every material cell.

    Chamfered cells are solid wedges: their cavity box is omitted and a
    triangular prism is cut away below the sloped underside plane, which
    rises from the full height at the inner face to the top edge at the
    outward face.

    Args:
        settings: Validated parametric block description.

    Returns:
        A CadQuery workplane holding a single fused solid.
    '''
    t = settings.wall_thickness
    unit = settings.unit_size
    height = settings.block_height
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0

    material = set(_included_cells(settings))
    chamfered = {(ix, iy) for ix, iy, _ in settings.sloped_cells}
    # For cavity-side decisions a chamfered neighbor counts as boundary:
    # the wedge cell has no cavity of its own, so a wall strip is required
    # between the normal cell's cavity and the void under the wedge,
    # otherwise the bottom opening extends under the wedge (too wide).
    cavity_neighbors = material - chamfered

    cells: list[cq.Workplane] = []
    cavities: list[cq.Workplane] = []
    for ix, iy in sorted(material):
        x0 = ix * unit - half_x
        y0 = iy * unit - half_y
        cell = cq.Workplane("XY", origin=(x0 + unit / 2.0, y0 + unit / 2.0, 0)).box(
            unit, unit, height, centered=(True, True, False)
        )
        cells.append(cell)
        if (ix, iy) in chamfered:
            continue  # solid wedge: no cavity opening in chamfered cells
        # Cavity box: inset t only on sides facing the region boundary or a
        # chamfered cell; sides shared with a normal material neighbor stay
        # open so the cavities of adjacent cells fuse into a single
        # contour-shaped cavity.
        inset_x0 = t if (ix - 1, iy) not in cavity_neighbors else 0.0
        inset_x1 = t if (ix + 1, iy) not in cavity_neighbors else 0.0
        inset_y0 = t if (ix, iy - 1) not in cavity_neighbors else 0.0
        inset_y1 = t if (ix, iy + 1) not in cavity_neighbors else 0.0
        cav_w = unit - inset_x0 - inset_x1
        cav_h = unit - inset_y0 - inset_y1
        cell_cavity = (
            cq.Workplane(
                "XY",
                origin=(x0 + inset_x0 + cav_w / 2.0, y0 + inset_y0 + cav_h / 2.0, 0),
            )
            .box(cav_w, cav_h, height - t + CUT_OVERSHOOT, centered=(True, True, False))
            .translate((0, 0, -CUT_OVERSHOOT))
        )
        cavities.append(cell_cavity)

    # Fuse all cells into one outer shape in a single boolean operation.
    if len(cells) == 1:
        outer_shape = _shape_of(cells[0])
    else:
        outer_shape = _fuse_all(_shape_of(cells[0]), [_shape_of(c) for c in cells[1:]])

    # Cut all cavity boxes in one operation.
    if cavities:
        outer_shape = outer_shape.cut(*(_shape_of(c) for c in cavities))

    solid = cq.Workplane("XY").newObject([_as_single_solid(outer_shape)])

    # Fuse corner ribs in a single boolean operation.
    ribs = _figure_corner_ribs(settings)
    if ribs:
        fused = _fuse_all(_shape_of(solid), [_shape_of(rib) for rib in ribs])
        solid = cq.Workplane("XY").newObject([_as_single_solid(fused)])

    if settings.pins_enabled:
        solid = _fuse_pins(solid, settings)

    # Cut all chamfer cutters in one operation.
    cutters = [
        _bottom_chamfer_cutter(settings, ix, iy, side)
        for ix, iy, side in settings.sloped_cells
    ]
    if cutters:
        result = _shape_of(solid).cut(*(_shape_of(c) for c in cutters))
        solid = cq.Workplane("XY").newObject([_as_single_solid(result)])
    return solid


def _bottom_chamfer_cutter(
    settings: BlockSettings, ix: int, iy: int, side: str
) -> cq.Workplane:
    '''Triangular prism cutting the chamfered underside of one cell.

    The prism profile is a right triangle along the slope axis: full cell
    footprint at z=0 with the apex at the top edge of the outward face, so
    the remaining material below the top face rises from zero thickness at
    the outward face to the full height at the inner face.

    Args:
        settings: Validated parametric block description.
        ix: Cell index along X.
        iy: Cell index along Y.
        side: Outward face the wedge thins toward (``+X``/``-X``/``+Y``/``-Y``).

    Returns:
        The cutter solid, spanning the cell footprint exactly.
    '''
    unit = settings.unit_size
    height = settings.block_height
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0
    x0 = ix * unit - half_x
    x1 = x0 + unit
    y0 = iy * unit - half_y
    y1 = y0 + unit

    if side == "+X":
        points, plane, perp_center = (
            [(x0, 0), (x1, 0), (x1, height)],
            "XZ",
            (0.0, (y0 + y1) / 2.0),
        )
    elif side == "-X":
        points, plane, perp_center = (
            [(x0, 0), (x1, 0), (x0, height)],
            "XZ",
            (0.0, (y0 + y1) / 2.0),
        )
    elif side == "+Y":
        points, plane, perp_center = (
            [(y0, 0), (y1, 0), (y1, height)],
            "YZ",
            ((x0 + x1) / 2.0, 0.0),
        )
    else:  # "-Y"
        points, plane, perp_center = (
            [(y0, 0), (y1, 0), (y0, height)],
            "YZ",
            ((x0 + x1) / 2.0, 0.0),
        )

    prism = cq.Workplane(plane).polyline(points).close().extrude(unit / 2.0, both=True)
    return prism.translate((perp_center[0], perp_center[1], 0))


def _figure_corner_ribs(settings: BlockSettings) -> list[cq.Workplane]:
    '''Quarter-round inward ribs at every concave corner of the figure.

    A concave corner is a lattice vertex where exactly 3 of the 4 adjacent
    cells have cavities (the missing cell creates the inner angle). Chamfered
    cells have no cavity and therefore do not contribute to such corners. The
    rib fills that corner with a quarter-round fillet (instead of a tube),
    spans the full cavity height and overshoots into the top face.

    Args:
        settings: Validated parametric block description.

    Returns:
        Rib solids to fuse into the shell.
    '''
    t = settings.wall_thickness
    unit = settings.unit_size
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0
    radius = min(2.0 * t, unit - 2.0 * t)
    included = set(_included_cells(settings))
    chamfered = {(ix, iy) for ix, iy, _ in settings.sloped_cells}
    cavity_cells = included - chamfered

    ribs: list[cq.Workplane] = []
    for i in range(1, settings.cells_x):
        for j in range(1, settings.cells_y):
            quadrants = {
                (-1, -1): (i - 1, j - 1),
                (1, -1): (i, j - 1),
                (-1, 1): (i - 1, j),
                (1, 1): (i, j),
            }
            missing = [
                off for off, cell in quadrants.items() if cell not in cavity_cells
            ]
            if len(missing) != 1:
                continue  # concave corner needs exactly 3 cells with cavities
            mx, my = missing[0]
            vx = i * unit - half_x
            vy = j * unit - half_y
            into_x, into_y = -mx, -my  # into the diagonal cavity cell
            rib_box = cq.Workplane(
                "XY",
                origin=(vx + into_x * radius / 2.0, vy + into_y * radius / 2.0, 0),
            ).box(radius, radius, settings.block_height, centered=(True, True, False))
            arc = (
                cq.Workplane(
                    "XY", origin=(vx + into_x * radius, vy + into_y * radius, 0)
                )
                .circle(radius)
                .extrude(settings.block_height)
            )
            ribs.append(rib_box.cut(arc))
    return ribs


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
    tubes: list[cq.Workplane] = []
    for cx, cy in _tube_centers(settings):
        tube = (
            cq.Workplane("XY", origin=(cx, cy, 0))
            .circle(r_out)
            .circle(r_in)
            .extrude(settings.block_height)
        )
        tubes.append(tube)
    if not tubes:
        return shell
    fused = _fuse_all(_shape_of(shell), [_shape_of(tube) for tube in tubes])
    return cq.Workplane("XY").newObject([_as_single_solid(fused)])


def _fuse_pins(shell: cq.Workplane, settings: BlockSettings) -> cq.Workplane:
    '''Fuse one cylindrical stud per 1x1 cell onto the shell top face.

    Args:
        shell: Hollow shell to attach studs to.
        settings: Validated parametric block description.

    Returns:
        The shell with all studs fused into a single solid.
    '''
    radius = settings.pin_diameter / 2.0
    excluded = {(ix, iy) for ix, iy in settings.excluded_cells}
    studs: list[cq.Workplane] = []
    for ix in range(settings.cells_x):
        for iy in range(settings.cells_y):
            if (ix, iy) in excluded:
                continue  # excluded cells have no material to carry a stud
            cx = (ix + 0.5) * settings.unit_size - settings.outer_length_x / 2.0
            cy = (iy + 0.5) * settings.unit_size - settings.outer_length_y / 2.0
            stud = (
                cq.Workplane("XY", origin=(cx, cy, settings.block_height))
                .circle(radius)
                .extrude(settings.pin_height)
            )
            studs.append(stud)
    if not studs:
        return shell
    fused = _fuse_all(_shape_of(shell), [_shape_of(stud) for stud in studs])
    return cq.Workplane("XY").newObject([_as_single_solid(fused)])
