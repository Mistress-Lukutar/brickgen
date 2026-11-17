'''
File:   builder.py
Brief:  CadQuery geometry builder for parametric stud blocks.
Author: Mistress-Lukutar
Date:   2026-11-17
Version: v0.10.0
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

    Cells with per-cell features (sloped tops, bottom chamfers, round
    bases, exclusions) switch to the figure variant; sloped cells become
    trapezoid wedges whose ``side`` edge rises by ``rise x block_height``
    while keeping the standard bottom opening.

    Args:
        settings: Validated parametric block description.

    Returns:
        A CadQuery workplane holding a single fused solid.
    '''
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
    '''Build a block with excluded, chamfered, sloped and/or round cells.

    The figure variant follows the reference parts: the cavity is one piece
    shaped like the included region (perimeter walls step around excluded
    cells, no internal partition walls), and every concave corner of the
    region carries an inward quarter-round rib instead of a tube. Tubes do
    not apply in this variant; studs sit only on cells whose effective top
    is ``stud`` (see :meth:`BlockSettings.cell_top`).

    Chamfered cells are solid wedges: their cavity box is omitted and a
    triangular prism is cut away below the sloped underside plane, which
    rises from the full height at the inner face to the top edge at the
    outward face. Sloped cells are trapezoid prisms that keep the standard
    cavity: the wedge sits on the shared top slab, so the bottom opening
    looks the same as on plain cells. Round cells replace the square base
    with the inscribed cylinder and carry their own concentric round hole;
    the hole never fuses with neighbors, so square cavities keep a wall
    strip toward round cells.

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
    chamfered = {(ix, iy) for ix, iy, _ in settings.chamfered_cells}
    sloped = settings.slope_cell_set()
    round_cells = settings.round_cell_set()
    slope_by_cell = {
        (int(item[0]), int(item[1])): (str(item[2]), int(item[3]))
        for item in settings.sloped_cells
    }
    # For cavity-side decisions a chamfered (solid wedge) or round neighbor
    # counts as boundary: chamfered cells have no cavity of their own, and
    # a round cell's concentric hole recedes from the shared edge, so a
    # wall strip is required; otherwise the bottom opening would extend
    # under/behind them (too wide). Sloped cells keep a normal cavity and
    # fuse with adjacent ones.
    cavity_neighbors = material - chamfered - round_cells

    cells: list[cq.Workplane] = []
    cavities: list[cq.Workplane] = []
    for ix, iy in sorted(material):
        x0 = ix * unit - half_x
        y0 = iy * unit - half_y
        if (ix, iy) in round_cells:
            cells.append(_round_cell_solid(settings, ix, iy))
        elif (ix, iy) in sloped:
            side, rise_units = slope_by_cell[(ix, iy)]
            cells.append(_slope_cell_solid(settings, ix, iy, side, rise_units))
        else:
            cells.append(
                cq.Workplane(
                    "XY", origin=(x0 + unit / 2.0, y0 + unit / 2.0, 0)
                ).box(unit, unit, height, centered=(True, True, False))
            )
        if (ix, iy) in chamfered:
            continue  # solid wedge: no cavity opening in chamfered cells
        if (ix, iy) in round_cells:
            cavities.append(_round_cell_hole(settings, ix, iy))
            continue
        # Cavity box: inset t only on sides facing the region boundary or a
        # cell without a fusable cavity (chamfered/round); sides shared with
        # a normal or sloped material neighbor stay open so the cavities of
        # adjacent cells fuse into a single contour-shaped cavity.
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

    solid = _fuse_pins(solid, settings)

    # Cut all chamfer cutters in one operation.
    cutters = [
        _bottom_chamfer_cutter(settings, ix, iy, side)
        for ix, iy, side in settings.chamfered_cells
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


def _slope_cell_solid(
    settings: BlockSettings, ix: int, iy: int, side: str, rise_units: int
) -> cq.Workplane:
    '''Trapezoid-prism cell solid with one raised edge and a sloped top.

    The outer profile is a trapezoid in the plane of the slope axis: heights
    ``block_height`` on the low side and ``block_height + rise`` on the
    raised ``side`` edge. The cell is solid (no cavity, no stud).

    Args:
        settings: Validated parametric block description.
        ix: Cell index along X.
        iy: Cell index along Y.
        side: Edge that rises (``+X``/``-X``/``+Y``/``-Y``).
        rise_units: Rise of that edge in unit block heights (N x block_height).

    Returns:
        The slope cell solid, spanning the cell footprint exactly.
    '''
    unit = settings.unit_size
    height = settings.block_height
    top_high = height + rise_units * height
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0
    x0 = ix * unit - half_x
    x1 = x0 + unit
    y0 = iy * unit - half_y
    y1 = y0 + unit

    if side == "+X":
        points, plane, perp_center = (
            [(x0, 0), (x1, 0), (x1, top_high), (x0, height)],
            "XZ",
            (0.0, (y0 + y1) / 2.0),
        )
    elif side == "-X":
        points, plane, perp_center = (
            [(x0, 0), (x1, 0), (x1, height), (x0, top_high)],
            "XZ",
            (0.0, (y0 + y1) / 2.0),
        )
    elif side == "+Y":
        points, plane, perp_center = (
            [(y0, 0), (y1, 0), (y1, top_high), (y0, height)],
            "YZ",
            ((x0 + x1) / 2.0, 0.0),
        )
    else:  # "-Y"
        points, plane, perp_center = (
            [(y0, 0), (y1, 0), (y1, height), (y0, top_high)],
            "YZ",
            ((x0 + x1) / 2.0, 0.0),
        )

    prism = cq.Workplane(plane).polyline(points).close().extrude(unit / 2.0, both=True)
    return prism.translate((perp_center[0], perp_center[1], 0))


def _cell_center(settings: BlockSettings, ix: int, iy: int) -> tuple[float, float]:
    '''XY center of cell (ix, iy) in block coordinates, mm.'''
    return (
        (ix + 0.5) * settings.unit_size - settings.outer_length_x / 2.0,
        (iy + 0.5) * settings.unit_size - settings.outer_length_y / 2.0,
    )


# Radial bulge of the round cell cylinder over the exact inscribed circle.
# Adjacent round cells (or a cylinder and a neighboring box) touch the
# shared cell edge along a single tangent line, which OCC refuses to fuse
# into one solid; the tiny overlap makes the union robust while staying
# visually identical to the inscribed circle.
ROUND_CELL_OVERSHOOT = 0.05


def _round_cell_solid(settings: BlockSettings, ix: int, iy: int) -> cq.Workplane:
    '''Cylinder replacing the square base of a round cell.

    The cylinder is inscribed in the cell footprint (diameter
    ``unit_size``) plus :data:`ROUND_CELL_OVERSHOOT` so neighboring cells
    fuse, and spans the full block height.

    Args:
        settings: Validated parametric block description.
        ix: Cell index along X.
        iy: Cell index along Y.

    Returns:
        The round cell solid, centered in the cell footprint.
    '''
    cx, cy = _cell_center(settings, ix, iy)
    return (
        cq.Workplane("XY", origin=(cx, cy, 0))
        .circle(settings.unit_size / 2.0 + ROUND_CELL_OVERSHOOT)
        .extrude(settings.block_height)
    )


def _round_cell_hole(settings: BlockSettings, ix: int, iy: int) -> cq.Workplane:
    '''Concentric round bottom hole of a round cell.

    The bore diameter is ``unit_size - 2 x wall_thickness`` (the inscribed
    circle inset by the wall thickness) and the ceiling sits at
    ``block_height - wall_thickness``, open from below.

    Args:
        settings: Validated parametric block description.
        ix: Cell index along X.
        iy: Cell index along Y.

    Returns:
        The hole cutter solid, spanning the cell height minus the top slab.
    '''
    cx, cy = _cell_center(settings, ix, iy)
    return (
        cq.Workplane("XY", origin=(cx, cy, 0))
        .circle(settings.unit_size / 2.0 - settings.wall_thickness)
        .extrude(settings.block_height - settings.wall_thickness + CUT_OVERSHOOT)
        .translate((0, 0, -CUT_OVERSHOOT))
    )


def _figure_corner_ribs(settings: BlockSettings) -> list[cq.Workplane]:
    '''Quarter-round inward ribs at every concave corner of the figure.

    A concave corner is a lattice vertex where exactly 3 of the 4 adjacent
    cells have cavities (the missing cell creates the inner angle). Chamfered
    cells are solid (no cavity) and round cells keep a hole that recedes
    from the lattice vertex, so neither contributes to such corners. The
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
    chamfered = {(ix, iy) for ix, iy, _ in settings.chamfered_cells}
    cavity_cells = included - chamfered - settings.round_cell_set()

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

    Studs follow the per-cell top resolution: cells with a sloped or flat
    top get none, explicit ``stud_cells`` get one even on stud-less blocks.

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

    outer = _fuse_pins(outer, settings)
    if settings.has_tubes:
        outer = _fuse_tubes(outer, settings)

    return outer


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
    '''Fuse one cylindrical stud onto every cell whose top is ``stud``.

    The per-cell resolution comes from :meth:`BlockSettings.stud_cells_all`:
    sloped and flat cells get none, explicit ``stud_cells`` get one even on
    stud-less blocks, plain cells follow ``pins_enabled``.

    Args:
        shell: Hollow shell to attach studs to.
        settings: Validated parametric block description.

    Returns:
        The shell with all studs fused into a single solid.
    '''
    radius = settings.pin_diameter / 2.0
    studs: list[cq.Workplane] = []
    for ix, iy in sorted(settings.stud_cells_all()):
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
