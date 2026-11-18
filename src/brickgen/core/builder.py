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
    cells, no internal partition walls). Every concave corner of the region
    carries an inward quarter-round rib; lattice vertices whose four cells
    all keep a cavity carry a reinforcement tube when ``has_tubes``. Studs
    sit only on cells whose effective top is ``stud`` (see
    :meth:`BlockSettings.cell_top`).

    Chamfered cells are solid wedges: their cavity box is omitted and a
    triangular prism is cut away below the sloped underside plane, which
    rises from the full height at the inner face to the top edge at the
    outward face. Sloped cells are trapezoid prisms that keep the standard
    cavity: the wedge sits on the shared top slab, so the bottom opening
    looks the same as on plain cells. Round cells keep the square base
    where material neighbors join and round their free corners with the
    inscribed circle (adjacent round cells form the stadium profile); the
    bottom cavity is the concentric inscribed bore, joined by strips to
    every neighboring cavity so round and square voids merge into one
    contour-shaped cavity without partition walls.

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
    # For cavity-side decisions only a chamfered neighbor counts as
    # boundary: it is a solid wedge with no cavity of its own, so a wall
    # strip is required between the square cavity and its body. Sloped
    # and round cells keep material right up to their (straight) shared
    # edges, and a round hole opens toward cavity neighbors through a
    # fused strip, so square cavities may open toward them like toward
    # plain cells.
    cavity_neighbors = material - chamfered

    cells: list[cq.Workplane] = []
    cavities: list[cq.Workplane] = []
    for ix, iy in sorted(material):
        x0 = ix * unit - half_x
        y0 = iy * unit - half_y
        if (ix, iy) in round_cells:
            cells.append(_round_cell_solid(settings, ix, iy, material))
        elif (ix, iy) in sloped:
            side, rise_units = slope_by_cell[(ix, iy)]
            cells.append(_slope_cell_solid(settings, ix, iy, side, rise_units))
        else:
            cells.append(
                cq.Workplane("XY", origin=(x0 + unit / 2.0, y0 + unit / 2.0, 0)).box(
                    unit, unit, height, centered=(True, True, False)
                )
            )
        if (ix, iy) in chamfered:
            continue  # solid wedge: no cavity opening in chamfered cells
        if (ix, iy) in round_cells:
            cavities.append(_round_cell_hole(settings, ix, iy, cavity_neighbors))
            continue
        # Cavity box: inset t only on sides facing the region boundary or a
        # solid (chamfered) cell; sides shared with a material neighbor that
        # keeps its cavity (plain, sloped or round) stay open so adjacent
        # cavities fuse into a single contour-shaped cavity.
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

    # Fill boxes open the cavity across lattice vertices where four
    # cavity-keeping cells meet (round holes and strips stop short of
    # the vertex and would leave a material pillar inside the void).
    cavities.extend(_round_vertex_cavities(settings, cavity_neighbors))

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

    # Reinforcement tubes at lattice vertices whose four cells all keep a
    # cavity (the flat-variant rule, restricted to the cavity region).
    if settings.has_tubes:
        solid = _fuse_tubes(solid, settings, _tube_centers(settings, cavity_neighbors))

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


def _round_cell_solid(
    settings: BlockSettings, ix: int, iy: int, material: set[tuple[int, int]]
) -> cq.Workplane:
    '''Square cell whose free corners are rounded by the inscribed circle.

    The base shape is the square cell; a corner is replaced by the
    inscribed-circle quarter arc (between the midpoints of the adjacent
    edges) only when BOTH of its edges face no material neighbor. Edges
    joined to any material cell stay straight and full-length, so adjacent
    round cells merge with a straight seam (the classic stadium profile)
    and a round cell next to a square one joins flush.

    Args:
        settings: Validated parametric block description.
        ix: Cell index along X.
        iy: Cell index along Y.
        material: Set of (ix, iy) cells present in the figure.

    Returns:
        The round cell solid, centered in the cell footprint.
    '''
    unit = settings.unit_size
    height = settings.block_height
    half = unit / 2.0
    cx, cy = _cell_center(settings, ix, iy)
    shape = cq.Workplane("XY", origin=(cx, cy, 0)).circle(half).extrude(height)
    connected = {
        "+X": (ix + 1, iy) in material,
        "-X": (ix - 1, iy) in material,
        "+Y": (ix, iy + 1) in material,
        "-Y": (ix, iy - 1) in material,
    }
    # A straight edge contributes a strip from the cell center to the edge;
    # the union of circle and strips yields square corners exactly at the
    # joined edges and inscribed-circle arcs elsewhere.
    strips: list[tuple[float, float, float, float]] = []
    if connected["+X"]:
        strips.append((cx + half / 2.0, cy, half, unit))
    if connected["-X"]:
        strips.append((cx - half / 2.0, cy, half, unit))
    if connected["+Y"]:
        strips.append((cx, cy + half / 2.0, unit, half))
    if connected["-Y"]:
        strips.append((cx, cy - half / 2.0, unit, half))
    if not strips:
        return shape
    fused = _fuse_all(
        _shape_of(shape),
        [
            _shape_of(
                cq.Workplane("XY", origin=(sx, sy, 0)).box(
                    w, d, height, centered=(True, True, False)
                )
            )
            for sx, sy, w, d in strips
        ],
    )
    return cq.Workplane("XY").newObject([fused])


def _round_cell_hole(
    settings: BlockSettings,
    ix: int,
    iy: int,
    cavity_neighbors: set[tuple[int, int]],
) -> cq.Workplane:
    '''Round bottom cavity of a round cell, fused with neighboring cavities.

    The bore is the inscribed circle inset by the wall thickness (radius
    ``unit_size / 2 - wall_thickness``); toward every neighbor that keeps a
    cavity (round bore or square cell cavity) a strip of the same width
    runs from the cell center to the shared edge, so adjacent voids merge
    into one contour-shaped cavity without partition walls. The ceiling
    sits at ``block_height - wall_thickness``, open from below.

    Args:
        settings: Validated parametric block description.
        ix: Cell index along X.
        iy: Cell index along Y.
        cavity_neighbors: Material cells that keep a bottom cavity.

    Returns:
        The hole cutter solid, spanning the cell height minus the top slab.
    '''
    half = settings.unit_size / 2.0
    width = settings.unit_size - 2.0 * settings.wall_thickness
    depth = settings.block_height - settings.wall_thickness + CUT_OVERSHOOT
    cx, cy = _cell_center(settings, ix, iy)
    shape = (
        cq.Workplane("XY", origin=(cx, cy, 0))
        .circle(half - settings.wall_thickness)
        .extrude(depth)
    )
    connected = {
        "+X": (ix + 1, iy) in cavity_neighbors,
        "-X": (ix - 1, iy) in cavity_neighbors,
        "+Y": (ix, iy + 1) in cavity_neighbors,
        "-Y": (ix, iy - 1) in cavity_neighbors,
    }
    # A strip runs from the cell center to the shared edge, inset by the
    # wall thickness on both sides; the union of bore and strips opens the
    # cavity toward every cavity-keeping neighbor.
    strips: list[tuple[float, float, float, float]] = []
    if connected["+X"]:
        strips.append((cx + half / 2.0, cy, half, width))
    if connected["-X"]:
        strips.append((cx - half / 2.0, cy, half, width))
    if connected["+Y"]:
        strips.append((cx, cy + half / 2.0, width, half))
    if connected["-Y"]:
        strips.append((cx, cy - half / 2.0, width, half))
    if not strips:
        return shape.translate((0, 0, -CUT_OVERSHOOT))
    fused = _fuse_all(
        _shape_of(shape),
        [
            _shape_of(
                cq.Workplane("XY", origin=(sx, sy, 0)).box(
                    w, d, depth, centered=(True, True, False)
                )
            )
            for sx, sy, w, d in strips
        ],
    )
    return cq.Workplane("XY").newObject([fused]).translate((0, 0, -CUT_OVERSHOOT))


def _round_vertex_cavities(
    settings: BlockSettings, cavity_neighbors: set[tuple[int, int]]
) -> list[cq.Workplane]:
    '''Fill boxes that open the cavity across fully surrounded lattice vertices.

    Round bores and their strips are per-cell cutters; where four
    cavity-keeping cells meet, the strips run along the cell center lines
    and stop short of the vertex, leaving a small material pillar inside
    the merged cavity. One square fill box (``unit_size - 2 x
    wall_thickness`` per side) per such vertex removes it. Vertices with
    fewer cavity cells keep their walls: they sit on the region boundary,
    next to an excluded cell, or at a chamfered cell (which carries a rib).

    Args:
        settings: Validated parametric block description.
        cavity_neighbors: Material cells that keep a bottom cavity.

    Returns:
        Fill box cutter solids spanning the cavity height.
    '''
    unit = settings.unit_size
    width = unit - 2.0 * settings.wall_thickness
    depth = settings.block_height - settings.wall_thickness + CUT_OVERSHOOT
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0
    round_cells = settings.round_cell_set()
    boxes: list[cq.Workplane] = []
    for ix in range(1, settings.cells_x):
        for iy in range(1, settings.cells_y):
            quadrants = [
                (ix - 1, iy - 1),
                (ix, iy - 1),
                (ix - 1, iy),
                (ix, iy),
            ]
            if not all(cell in cavity_neighbors for cell in quadrants):
                continue
            if not any(cell in round_cells for cell in quadrants):
                continue  # square cavity boxes already tile such vertices
            boxes.append(
                cq.Workplane(
                    "XY",
                    origin=(ix * unit - half_x, iy * unit - half_y, 0),
                )
                .box(width, width, depth, centered=(True, True, False))
                .translate((0, 0, -CUT_OVERSHOOT))
            )
    return boxes


def _figure_corner_ribs(settings: BlockSettings) -> list[cq.Workplane]:
    '''Quarter-round inward ribs at every concave corner of the figure.

    A concave corner is a lattice vertex where exactly 3 of the 4 adjacent
    cells have cavities (the missing cell creates the inner angle).
    Chamfered cells are solid (no cavity) and round cells open their
    cavity through a fill box, so neither contributes to such corners: a
    rib there would stand inside the merged void. The rib fills the corner
    with a quarter-round fillet (instead of a tube), spans the full cavity
    height and overshoots into the top face.

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
            if quadrants[missing[0]] in settings.round_cell_set():
                continue  # round cavity opens through a fill box here
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


def _tube_centers(
    settings: BlockSettings,
    cavity_neighbors: set[tuple[int, int]] | None = None,
) -> list[tuple[float, float]]:
    '''Compute inner tube centers: mid-lines between stud rows/columns.

    One tube per sliding 2x2 cell group, i.e. at every intersection of the
    mid-lines between adjacent stud columns (X) and rows (Y). A 2x2 block
    gets a single central tube; a 3x2 block gets two tubes along its length.
    With ``cavity_neighbors`` the centers are restricted to vertices whose
    four surrounding cells all keep a cavity (figure variant: vertices next
    to excluded or chamfered cells carry a rib instead of a tube).

    Args:
        settings: Validated parametric block description.
        cavity_neighbors: Optional cavity-keeping cell set to filter by.

    Returns:
        List of (x, y) centers in block coordinates (block centered at origin).
    '''
    half_x = settings.outer_length_x / 2.0
    half_y = settings.outer_length_y / 2.0
    return [
        (ix * settings.unit_size - half_x, iy * settings.unit_size - half_y)
        for ix in range(1, settings.cells_x)
        for iy in range(1, settings.cells_y)
        if cavity_neighbors is None
        or all(
            (ix + dx, iy + dy) in cavity_neighbors for dx in (-1, 0) for dy in (-1, 0)
        )
    ]


def _fuse_tubes(
    shell: cq.Workplane,
    settings: BlockSettings,
    centers: list[tuple[float, float]] | None = None,
) -> cq.Workplane:
    '''Fuse hollow reinforcement tubes under the inner top face.

    Each tube is an annulus (bore ``tube_inner_diameter``, wall equal to
    ``wall_thickness``) spanning the full cavity depth, closed on top by the
    solid top face. The outer cylinder overshoots into the top face for a
    robust boolean union.

    Args:
        shell: Hollow shell to attach tubes to.
        settings: Validated parametric block description.
        centers: Explicit tube centers; defaults to all sliding 2x2 centers.

    Returns:
        The shell with all tubes fused into a single solid.
    '''
    r_out = settings.tube_outer_diameter / 2.0
    r_in = settings.tube_inner_diameter / 2.0
    tubes: list[cq.Workplane] = []
    for cx, cy in centers if centers is not None else _tube_centers(settings):
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
