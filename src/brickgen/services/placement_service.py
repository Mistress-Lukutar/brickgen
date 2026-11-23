'''
File:   placement_service.py
Brief:  Print-placement layout engine: zone geometry, sync and auto-place.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import dataclasses

import numpy as np

from brickgen.core.project import (
    PRINT_STAGING_BED,
    PrintBedSettings,
    PrintPiece,
    Project,
)

# World-space layout of the placement scene: identical zones (beds and the
# staging area) are arranged on a near-square grid, separated by this gap.
# The staging zone sits to the LEFT of bed 0 (row 0).
BED_GAP_MM = 20.0

# This module stays pure logic: no cadquery / pyvista / Qt imports allowed.


def bed_grid_shape(count: int) -> tuple[int, int]:
    '''Near-square ``(columns, rows)`` bed arrangement for *count* beds.

    The column count is ``ceil(sqrt(count))`` and the rows fill the rest,
    so 2 beds form a 2x1 grid, 3-4 beds a 2x2 grid, 5-6 beds a 3x2 grid,
    7-9 beds a 3x3 grid and so on. Bed indices fill the grid row by row,
    left to right.

    Args:
        count: Number of beds (>= 1).

    Returns:
        ``(columns, rows)`` with ``columns * rows >= count``.
    '''
    columns = max(1, int(np.ceil(np.sqrt(max(1, count)))))
    rows = max(1, int(np.ceil(count / columns)))
    return (columns, rows)


def bed_origin(index: int, beds: PrintBedSettings) -> tuple[float, float]:
    '''World position (bottom-left corner) of *index* bed.

    The beds sit on a near-square grid (see :func:`bed_grid_shape`) and
    are numbered in READING order: left to right, TOP to BOTTOM — bed 0
    is the top-left cell (the grid's highest-Y row, since the world Y
    axis points up), higher indices continue to the right, then wrap to
    the next row below.

    Args:
        index: Zero-based bed index.
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(x, y)`` world coordinates of the bottom-left corner of that bed.
    '''
    columns, rows = bed_grid_shape(beds.count)
    column = index % columns
    row = index // columns
    return (
        column * (beds.width + BED_GAP_MM),
        (rows - 1 - row) * (beds.depth + BED_GAP_MM),
    )


def staging_origin(beds: PrintBedSettings) -> tuple[float, float]:
    '''World position (bottom-left corner) of the staging zone.

    The staging zone always sits to the LEFT of bed 0 (which occupies the
    top-left cell of the bed grid), in the same top row.

    Args:
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(x, y)`` world coordinates of the staging zone origin.
    '''
    _columns, rows = bed_grid_shape(beds.count)
    return (
        -(beds.width + BED_GAP_MM),
        (rows - 1) * (beds.depth + BED_GAP_MM),
    )


def zone_for_world(x: float, y: float, beds: PrintBedSettings) -> int:
    '''Map a world-space point to a zone index.

    Args:
        x: World X coordinate.
        y: World Y coordinate.
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``PRINT_STAGING_BED`` when the point falls in the staging zone
        (with a half-gap tolerance around its rect), otherwise the bed
        index containing the point; when the point lies in no zone rect
        (in the gap between beds or far outside), the nearest bed by
        center distance.
    '''
    half = BED_GAP_MM / 2.0
    sx, sy = staging_origin(beds)
    if (
        sx - half <= x <= sx + beds.width + half
        and sy - half <= y <= sy + beds.depth + half
    ):
        return PRINT_STAGING_BED
    for index in range(beds.count):
        ox, oy = bed_origin(index, beds)
        if (
            ox - half <= x <= ox + beds.width + half
            and oy - half <= y <= oy + beds.depth + half
        ):
            return index
    # Far outside every zone: fall back to the nearest bed center so the
    # caller always receives a usable zone (matches the old clamping).
    centers = [
        (
            bed_origin(index, beds)[0] + beds.width / 2.0,
            bed_origin(index, beds)[1] + beds.depth / 2.0,
        )
        for index in range(beds.count)
    ]
    return min(
        range(beds.count),
        key=lambda index: (x - centers[index][0]) ** 2 + (y - centers[index][1]) ** 2,
    )


def world_position(piece: PrintPiece, beds: PrintBedSettings) -> tuple[float, float]:
    '''Convert a piece's zone-local position to world coordinates.

    Args:
        piece: The print piece (pydantic model).
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(world_x, world_y)`` in mm.
    '''
    if piece.bed == PRINT_STAGING_BED:
        ox, oy = staging_origin(beds)
    else:
        ox, oy = bed_origin(piece.bed, beds)
    return (ox + piece.position_x, oy + piece.position_y)


def local_position(
    x: float,
    y: float,
    bed: int,
    beds: PrintBedSettings,
) -> tuple[float, float]:
    '''Inverse of :func:`world_position` for a piece in zone *bed*.

    Args:
        x: World X coordinate.
        y: World Y coordinate.
        bed: Zone index (0..count-1 or ``PRINT_STAGING_BED``).
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(local_x, local_y)`` in zone-local mm.
    '''
    if bed == PRINT_STAGING_BED:
        ox, oy = staging_origin(beds)
    else:
        ox, oy = bed_origin(bed, beds)
    return (x - ox, y - oy)


def block_footprints(project: Project) -> dict[str, tuple[float, float]]:
    '''Map every block uid to its ``(width, depth)`` footprint.

    Args:
        project: The active project.

    Returns:
        ``{uid: (outer_length_x, outer_length_y)}`` for each block.
    '''
    result: dict[str, tuple[float, float]] = {}
    for block in project.blocks:
        merged = block.to_settings(project.settings)
        result[block.uid] = (merged.outer_length_x, merged.outer_length_y)
    return result


def block_colors(project: Project) -> dict[str, str]:
    '''Map every block uid to its color hex string.

    Args:
        project: The active project.

    Returns:
        ``{uid: color}`` for each block.
    '''
    return {block.uid: block.color for block in project.blocks}


def piece_dims(
    piece: PrintPiece, footprints: dict[str, tuple[float, float]]
) -> tuple[float, float]:
    '''Orientation-aware ``(width, depth)`` footprint of a piece.

    A piece rotated 90 degrees about Z swaps its footprint axes.

    Args:
        piece: The print piece.
        footprints: ``{block_uid: (outer_length_x, outer_length_y)}`` from
            :func:`block_footprints`.

    Returns:
        ``(width, depth)`` in mm for the piece's current orientation.
    '''
    w, d = footprints[piece.block_uid]
    return (d, w) if piece.rotated90 else (w, d)


def fits_bed(width: float, depth: float, beds: PrintBedSettings) -> bool:
    '''Check whether a rectangle of *width* x *depth* fits inside one bed.

    The piece needs ``2 * gap`` clearance from every bed edge.

    Args:
        width: Piece width in mm.
        depth: Piece depth in mm.
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``True`` when the piece fits with the required margin.
    '''
    return width <= beds.width - 2 * beds.gap and depth <= beds.depth - 2 * beds.gap


def _overlaps(
    ax: float,
    ay: float,
    aw: float,
    ad: float,
    bx: float,
    by: float,
    bw: float,
    bd: float,
    gap: float,
) -> bool:
    '''Axis-aligned AABB overlap test with *gap* clearance.

    Both rectangles are centred at ``(ax, ay)`` and ``(bx, by)`` with the
    given half-widths/depths (full extents ``aw``, ``ad``, ``bw``, ``bd``).
    Returns ``True`` when they collide with at least *gap* clearance between
    surfaces.

    Args:
        ax, ay: Centre of rect A.
        aw, ad: Full width and depth of rect A.
        bx, by: Centre of rect B.
        bw, bd: Full width and depth of rect B.
        gap: Required clearance between surfaces.

    Returns:
        ``True`` when the two rects overlap with less than *gap* clearance.
    '''
    return abs(ax - bx) < (aw + bw) / 2.0 + gap and abs(ay - by) < (ad + bd) / 2.0 + gap


def clamp_into_zone(
    piece: PrintPiece,
    footprints: dict[str, tuple[float, float]],
    beds: PrintBedSettings,
) -> None:
    '''Clamp a piece's zone-local position into its zone's margins.

    Keeps the footprint fully inside ``[gap, size-gap]`` of the piece's
    zone (beds and the staging zone are bed-sized): a drop past a zone
    edge can no longer poke into the neighboring zone.  Pieces too large
    for the zone are left untouched (they are staged elsewhere).

    Args:
        piece: The piece (position mutated in place).
        footprints: ``{block_uid: (w, d)}`` from :func:`block_footprints`.
        beds: The project's ``PrintBedSettings``.
    '''
    if piece.block_uid not in footprints:
        return
    w, d = piece_dims(piece, footprints)
    lo_x = beds.gap + w / 2.0
    hi_x = beds.width - beds.gap - w / 2.0
    lo_y = beds.gap + d / 2.0
    hi_y = beds.depth - beds.gap - d / 2.0
    if lo_x > hi_x or lo_y > hi_y:
        return
    piece.position_x = min(max(piece.position_x, lo_x), hi_x)
    piece.position_y = min(max(piece.position_y, lo_y), hi_y)


def nearest_free_spot(
    placed: list[tuple[float, float, float, float]],
    width: float,
    depth: float,
    beds: PrintBedSettings,
    near_x: float,
    near_y: float,
) -> tuple[float, float] | None:
    '''Conflict-free spot closest to a requested position.

    Candidate centers are built from the requested position, the zone
    margins and the edges of the already-placed rects (the cross product
    also covers diagonal neighbors); every candidate is bounds- and
    clearance-checked, and the one with the smallest Euclidean distance
    to ``(near_x, near_y)`` wins (ties: lower y, lower x).

    Args:
        placed: Existing rects as ``(cx, cy, w, d)`` in zone-local coords.
        width: New piece width in mm.
        depth: New piece depth in mm.
        beds: The project's ``PrintBedSettings``.
        near_x: Requested center X (may be out of bounds; it is clamped).
        near_y: Requested center Y.

    Returns:
        ``(cx, cy)`` of the closest conflict-free spot, or ``None`` when
        no candidate is conflict-free.
    '''
    gap = beds.gap
    lo_x = gap + width / 2.0
    hi_x = beds.width - gap - width / 2.0
    lo_y = gap + depth / 2.0
    hi_y = beds.depth - gap - depth / 2.0
    if lo_x > hi_x or lo_y > hi_y:
        return None
    xs_set = {near_x, lo_x, hi_x}
    ys_set = {near_y, lo_y, hi_y}
    for rx, ry, rw, rd in placed:
        xs_set.add(rx + rw / 2.0 + gap + width / 2.0)  # right of the rect
        xs_set.add(rx - rw / 2.0 - gap - width / 2.0)  # left of the rect
        ys_set.add(ry + rd / 2.0 + gap + depth / 2.0)  # above the rect
        ys_set.add(ry - rd / 2.0 - gap - depth / 2.0)  # below the rect
    xs = sorted(x for x in xs_set if lo_x - 1e-6 <= x <= hi_x + 1e-6)
    ys = sorted(y for y in ys_set if lo_y - 1e-6 <= y <= hi_y + 1e-6)
    best: tuple[float, float, float] | None = None
    pos: tuple[float, float] | None = None
    for cy in ys:
        for cx in xs:
            cx = min(max(cx, lo_x), hi_x)
            cy = min(max(cy, lo_y), hi_y)
            if any(
                _overlaps(cx, cy, width, depth, rx, ry, rw, rd, gap)
                for rx, ry, rw, rd in placed
            ):
                continue
            dist = (cx - near_x) ** 2 + (cy - near_y) ** 2
            key = (dist, cy, cx)
            if best is None or key < best:
                best = key
                pos = (cx, cy)
    return pos


def resolve_piece_placements(
    pieces: list[PrintPiece],
    footprints: dict[str, tuple[float, float]],
    beds: PrintBedSettings,
    moved: list[PrintPiece],
) -> None:
    '''Repair moved pieces that overlap others in their zones.

    Every moved piece keeps its position when it clears all pieces in the
    same zone by ``gap``; otherwise it shifts to the nearest conflict-free
    spot in that zone (:func:`nearest_free_spot`), or goes to staging
    when the zone is full (:func:`_stage_piece`, MaxRects-packed into the
    free staging space).  Manual moves can therefore never produce
    overlapping volumes.

    Args:
        pieces: The full piece list (mutated in place).
        footprints: ``{block_uid: (w, d)}`` from :func:`block_footprints`.
        beds: The project's ``PrintBedSettings``.
        moved: The pieces repositioned by this move (any order).
    '''
    moved_uids = {p.uid for p in moved}
    # Orientation-aware rects of every untouched piece, grouped by zone.
    static: dict[int, list[tuple[float, float, float, float]]] = {}
    for p in pieces:
        if p.uid in moved_uids or p.block_uid not in footprints:
            continue
        w, d = piece_dims(p, footprints)
        static.setdefault(p.bed, []).append((p.position_x, p.position_y, w, d))

    for piece in moved:
        if piece.block_uid not in footprints:
            continue  # unknown block: keep the clamped position
        w, d = piece_dims(piece, footprints)
        rects = static.setdefault(piece.bed, [])
        if not any(
            _overlaps(
                piece.position_x, piece.position_y, w, d, rx, ry, rw, rd, beds.gap
            )
            for rx, ry, rw, rd in rects
        ):
            rects.append((piece.position_x, piece.position_y, w, d))
            continue
        spot = nearest_free_spot(rects, w, d, beds, piece.position_x, piece.position_y)
        if spot is not None:
            piece.position_x, piece.position_y = spot
            rects.append((spot[0], spot[1], w, d))
            continue
        # Zone full: stage the piece (MaxRects-packed into the free
        # staging space around the already-staged pieces).
        fw, fd = footprints[piece.block_uid]
        _stage_piece(piece, fw, fd, static.setdefault(PRINT_STAGING_BED, []), beds)


def resolve_manual_moves(
    pieces: list[PrintPiece],
    footprints: dict[str, tuple[float, float]],
    beds: PrintBedSettings,
    moves: dict[int, tuple[float, float]],
) -> None:
    '''Commit dropped piece positions with collision resolution.

    Each moved piece is assigned to the zone under its dropped world
    center (:func:`zone_for_world`), clamped into the zone margins
    (:func:`clamp_into_zone`) and conflict-repaired via
    :func:`resolve_piece_placements`.

    Args:
        pieces: The full piece list (mutated in place).
        footprints: ``{block_uid: (w, d)}`` from :func:`block_footprints`.
        beds: The project's ``PrintBedSettings``.
        moves: ``{piece index: (world_x, world_y)}`` requested positions.
    '''
    moved: list[PrintPiece] = []
    for index, (wx, wy) in moves.items():
        if not 0 <= index < len(pieces):
            continue
        piece = pieces[index]
        zone = zone_for_world(wx, wy, beds)
        piece.bed = zone
        piece.position_x, piece.position_y = local_position(wx, wy, zone, beds)
        clamp_into_zone(piece, footprints, beds)
        moved.append(piece)
    resolve_piece_placements(pieces, footprints, beds, moved)


def _stage_piece(
    piece: PrintPiece,
    w: float,
    d: float,
    staged_rects: list[tuple[float, float, float, float]],
    beds: PrintBedSettings,
) -> None:
    '''Send a piece to the staging zone, packed into the free staging space.

    The already-staged rects act as obstacles: the piece is MaxRects-
    packed (gap-aware) into the remaining space and never overlaps them.
    When staging is full the piece keeps its coordinates, clamped into
    the zone margins — overlap is allowed there.  *staged_rects* tracks
    already-staged pieces as ``(cx, cy, w, d)`` zone-local rects and grows
    by one entry per staged piece.

    Args:
        piece: The piece to stage (bed, position and ``rotated90`` are
            mutated).
        w: UNROTATED piece footprint width in mm.
        d: UNROTATED piece footprint depth in mm.
        staged_rects: Running list of staged piece rects.
        beds: The project's ``PrintBedSettings``.
    '''
    piece.bed = PRINT_STAGING_BED
    placements, _leftovers = _maxrects_pack(
        [(piece.uid, w, d)], beds, obstacles=staged_rects
    )
    if piece.uid in placements:
        cx, cy, rotated = placements[piece.uid]
        piece.position_x = cx
        piece.position_y = cy
        piece.rotated90 = rotated
    else:
        # Staging full: clamp into the zone margins (overlap allowed).
        cw, cd = (d, w) if piece.rotated90 else (w, d)
        lo_x = beds.gap + cw / 2.0
        hi_x = beds.width - beds.gap - cw / 2.0
        lo_y = beds.gap + cd / 2.0
        hi_y = beds.depth - beds.gap - cd / 2.0
        if lo_x <= hi_x and lo_y <= hi_y:
            piece.position_x = min(max(piece.position_x, lo_x), hi_x)
            piece.position_y = min(max(piece.position_y, lo_y), hi_y)
    rw, rd = (d, w) if piece.rotated90 else (w, d)
    staged_rects.append((piece.position_x, piece.position_y, rw, rd))


def stage_pieces(
    pieces: list[PrintPiece],
    footprints: dict[str, tuple[float, float]],
    beds: PrintBedSettings,
    moved: list[PrintPiece],
) -> None:
    '''Send *moved* pieces to the staging zone, packed into the free space.

    The pieces already staged keep their positions and act as obstacles:
    the moved pieces are MaxRects-packed (gap-aware) into the remaining
    staging space in one batch.  Pieces that fit nowhere keep their
    coordinates, clamped into the staging zone margins — overlap is
    allowed in a packed staging zone.  Pieces with an unknown block uid
    keep their coordinates untouched.

    Args:
        pieces: The full piece list.
        footprints: ``{block_uid: (w, d)}`` from :func:`block_footprints`.
        beds: The project's ``PrintBedSettings``.
        moved: The pieces to stage (bed and position mutated in place).
    '''
    moved_uids = {p.uid for p in moved}
    staged_rects = [
        (p.position_x, p.position_y, *piece_dims(p, footprints))
        for p in pieces
        if p.bed == PRINT_STAGING_BED
        and p.uid not in moved_uids
        and p.block_uid in footprints
    ]
    items: list[tuple[str, float, float]] = []
    for piece in moved:
        piece.bed = PRINT_STAGING_BED
        if piece.block_uid in footprints:
            items.append((piece.uid, *footprints[piece.block_uid]))
    placements, _leftovers = _maxrects_pack(items, beds, obstacles=staged_rects)
    for piece in moved:
        if piece.block_uid not in footprints:
            continue  # unknown block: keep the coords
        if piece.uid in placements:
            cx, cy, rotated = placements[piece.uid]
            piece.position_x = cx
            piece.position_y = cy
            piece.rotated90 = rotated
        else:
            # Staging full: clamp into the zone margins (overlap allowed).
            clamp_into_zone(piece, footprints, beds)


@dataclasses.dataclass
class SyncReport:
    '''Return value of :func:`sync_pieces`.

    Attributes:
        added: Number of pieces created.
        removed: Number of existing pieces dropped.
        kept: Number of existing pieces retained.
        staged: Pieces added that landed in the staging zone (no free spot on bed 0).
    '''

    added: int = 0
    removed: int = 0
    kept: int = 0
    staged: int = 0


def sync_pieces(project: Project) -> SyncReport:
    '''Reconcile ``project.print_layout.pieces`` with ``project.blocks``.

    Mutates the pieces list in place and returns a ``SyncReport``.

    Steps:
        1. Drop pieces whose ``block_uid`` is not among current blocks.
        2. Per block (in block order): keep at most ``quantity`` earliest entries.
        3. Per block (in block order): append missing pieces, MaxRects-packed
           into bed 0's free space first, staging as fallback (existing
           pieces never move; they act as packing obstacles).
        4. Return counts.

    Args:
        project: The active project.

    Returns:
        A ``SyncReport`` with the changes made.
    '''
    beds = project.print_layout.beds
    fp = block_footprints(project)
    blocks_by_uid = {b.uid: b for b in project.blocks}
    valid_uids = set(blocks_by_uid.keys())

    pieces = project.print_layout.pieces

    # Step 1: drop unknown block_uids.
    existing = [p for p in pieces if p.block_uid in valid_uids]
    removed_count = len(pieces) - len(existing)
    pieces.clear()
    pieces.extend(existing)

    report = SyncReport(removed=removed_count, kept=0, added=0, staged=0)

    # Steps 2 & 3: per block in order.
    for block in project.blocks:
        uid = block.uid
        qty = block.quantity
        footprint = fp[uid]
        w, d = footprint

        # Rebuild list keeping at most *qty* earliest entries per uid.
        new_pieces: list[PrintPiece] = []
        kept_for_block = 0
        for p in pieces:
            if p.block_uid == uid:
                if kept_for_block < qty:
                    new_pieces.append(p)
                    kept_for_block += 1
                else:
                    report.removed += 1
            else:
                new_pieces.append(p)

        pieces[:] = new_pieces
        report.kept += kept_for_block

        # Append missing pieces for this block.
        to_add = qty - kept_for_block
        bed0_rects: list[tuple[float, float, float, float]] = []
        staged_rects: list[tuple[float, float, float, float]] = []
        if to_add > 0:
            # Orientation-aware rects of the pieces already in each zone:
            # obstacles for the packing of the new pieces (they never move).
            bed0_rects = [
                (p.position_x, p.position_y, *piece_dims(p, fp))
                for p in pieces
                if p.bed == 0
            ]
            staged_rects = [
                (p.position_x, p.position_y, *piece_dims(p, fp))
                for p in pieces
                if p.bed == PRINT_STAGING_BED
            ]
        for _i in range(to_add):
            new_piece = PrintPiece(block_uid=uid)
            # Try bed 0 first, then staging: each new piece is MaxRects-
            # packed (gap-aware) into the free space around the pieces
            # already in the zone, so existing pieces keep their positions.
            placements, _leftovers = _maxrects_pack(
                [(new_piece.uid, w, d)], beds, obstacles=bed0_rects
            )
            if new_piece.uid in placements:
                cx, cy, rotated = placements[new_piece.uid]
                new_piece.bed = 0
                new_piece.position_x = cx
                new_piece.position_y = cy
                new_piece.rotated90 = rotated
                rw, rd = (d, w) if rotated else (w, d)
                bed0_rects.append((cx, cy, rw, rd))
            else:
                _stage_piece(new_piece, w, d, staged_rects, beds)
                report.staged += 1
            pieces.append(new_piece)
            report.added += 1

    return report


def _pool_key(uid: str, w: float, d: float) -> tuple[float, float, str]:
    '''Deterministic packing order key: area DESC, longer side DESC, uid ASC.

    Args:
        uid: Piece/block identifier (final tie-breaker).
        w: UNROTATED footprint width in mm.
        d: UNROTATED footprint depth in mm.

    Returns:
        The sort key tuple (ascending sort = placement priority).
    '''
    return (-w * d, -max(w, d), uid)


def _carve_free_rects(
    rx: np.ndarray,
    ry: np.ndarray,
    rw: np.ndarray,
    rh: np.ndarray,
    px: float,
    py: float,
    pw: float,
    ph: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Split the free rects intersecting the occupied rect ``(px, py, pw, ph)``.

    Every intersecting free rect is cut into up to four maximal strips
    around the occupied rect; strips contained in a surviving rect or in
    an already-kept strip are pruned (the free rects stay pairwise
    maximal).  Touching rects survive — the strict overlap test keeps
    them.

    Args:
        rx, ry: Free-rect bottom-left corner arrays (zone-local mm).
        rw, rh: Free-rect size arrays.
        px, py: Occupied rect bottom-left corner.
        pw, ph: Occupied rect size.

    Returns:
        The new free-rect state as four arrays ``(rx, ry, rw, rh)``.
    '''
    eps = 1e-9
    inter = (
        (rx < px + pw - eps)
        & (rx + rw > px + eps)
        & (ry < py + ph - eps)
        & (ry + rh > py + eps)
    )
    sx: list[float] = []
    sy: list[float] = []
    sw: list[float] = []
    sh: list[float] = []
    for fx, fy, fw, fh in zip(rx[inter], ry[inter], rw[inter], rh[inter]):
        fx1 = fx + fw
        fy1 = fy + fh
        if px - fx > eps:  # left strip
            sx.append(fx)
            sy.append(fy)
            sw.append(px - fx)
            sh.append(fh)
        if fx1 - (px + pw) > eps:  # right strip
            sx.append(px + pw)
            sy.append(fy)
            sw.append(fx1 - (px + pw))
            sh.append(fh)
        if py - fy > eps:  # bottom strip
            sx.append(fx)
            sy.append(fy)
            sw.append(fw)
            sh.append(py - fy)
        if fy1 - (py + ph) > eps:  # top strip
            sx.append(fx)
            sy.append(py + ph)
            sw.append(fw)
            sh.append(fy1 - (py + ph))
    rx, ry, rw, rh = rx[~inter], ry[~inter], rw[~inter], rh[~inter]

    if sx:
        nx = np.array(sx)
        ny = np.array(sy)
        nw = np.array(sw)
        nh = np.array(sh)
        # Prune redundant strips: any strip contained in an old rect or
        # in an already-kept strip adds nothing.  Old rects are pairwise
        # maximal (invariant), so only new strips can be contained.
        # Area-DESC processing makes exact duplicates keep one copy.
        kx: list[float] = []
        ky: list[float] = []
        kw: list[float] = []
        kh: list[float] = []
        for j in np.argsort(-(nw * nh), kind="stable"):
            ax, ay, aw, ah = nx[j], ny[j], nw[j], nh[j]
            contained = bool(
                (
                    (rx <= ax + eps)
                    & (rx + rw >= ax + aw - eps)
                    & (ry <= ay + eps)
                    & (ry + rh >= ay + ah - eps)
                ).any()
            )
            if not contained and kx:
                kxa = np.array(kx)
                kya = np.array(ky)
                kwa = np.array(kw)
                kha = np.array(kh)
                contained = bool(
                    (
                        (kxa <= ax + eps)
                        & (kxa + kwa >= ax + aw - eps)
                        & (kya <= ay + eps)
                        & (kya + kha >= ay + ah - eps)
                    ).any()
                )
            if not contained:
                kx.append(float(ax))
                ky.append(float(ay))
                kw.append(float(aw))
                kh.append(float(ah))
        if kx:
            rx = np.concatenate([rx, np.array(kx)])
            ry = np.concatenate([ry, np.array(ky)])
            rw = np.concatenate([rw, np.array(kw)])
            rh = np.concatenate([rh, np.array(kh)])
    return rx, ry, rw, rh


def _maxrects_pack(
    items: list[tuple[str, float, float]],
    beds: PrintBedSettings,
    obstacles: list[tuple[float, float, float, float]] | None = None,
) -> tuple[dict[str, tuple[float, float, bool]], list[str]]:
    '''MaxRects packing of one bed zone, gap-aware and centered.

    Free space is tracked as maximal rectangles: every placement splits
    all intersecting free rects into strips, so left-over holes are REUSED
    by later pieces instead of being skipped (the old row-wise shelf pack
    wasted whole rows and left big voids).  Overlaps and out-of-bed
    placements are impossible by construction: a piece is only ever placed
    inside a free rect that was carved around all previously placed
    pieces.

    Per piece the placement candidate (free rect x orientation) is chosen
    so the cluster bbox stays as SQUARE as possible (minimize the larger
    bbox side; a pure area-growth score degenerates into bands/towers for
    identical pieces), then by minimal bbox area growth so holes inside
    the blob fill before the blob widens, then by Best-Short-Side-Fit,
    leftover extent, bottom-left position and the unrotated orientation.

    Spacing model: pieces are packed as rects INFLATED by ``gap`` inside
    the bed shrunk by ``gap/2`` per side — touching inflated rects leave
    exactly ``gap`` between real pieces and between the cluster and the
    bed edges.  After packing, the cluster bounding box is translated so
    it sits centered on the bed (zone-local coordinates stay zone-local).
    With *obstacles* the packer instead packs around fixed rects (they
    are inflated by ``gap`` and carved out of the free space first) and
    the result is NOT centered — placements are absolute zone-local
    coords compatible with the obstacles.

    Performance: the per-piece scan over all (free rect, orientation)
    pairs is numpy-vectorized; hundreds of pieces pack in milliseconds on
    CPU.  A GPU is deliberately not used — the workload is branchy and
    sequential per piece with the whole state fitting in CPU cache, so
    host<->device transfer would dominate, and CUDA would add a hard
    dependency.  Should piece counts ever explode, this function is pure
    and picklable, so it can move into the existing JobQueue process pool
    unchanged.

    Args:
        items: ``(unique_id, width, depth)`` tuples where *unique_id* is an
            opaque identifier (typically ``piece.uid``) used for look-up
            after packing and ``(width, depth)`` is the UNROTATED
            footprint.  Order in the list is the placement priority
            (callers sort by :func:`_pool_key`).
        beds: The project's ``PrintBedSettings``.
        obstacles: Already-placed REAL rects ``(cx, cy, w, d)`` (zone-local
            center coords) that must not be touched.  A list (possibly
            empty) selects incremental mode: items pack around the
            obstacles and the result is NOT centered.  ``None`` (default)
            packs an empty zone and centers the packed cluster.

    Returns:
        ``(placements, leftovers)`` where *placements* maps unique_id to
        ``(cx, cy, rotated90)`` — zone-local CENTER coords plus the chosen
        orientation flag — and *leftovers* lists ids that fit in no
        orientation.
    '''
    gap = beds.gap
    placements: dict[str, tuple[float, float, bool]] = {}
    leftovers: list[str] = []
    if not items:
        return placements, leftovers

    eps = 1e-9
    # Free-rect state as four column arrays (grown via concatenation).
    rx = np.array([gap / 2.0])
    ry = np.array([gap / 2.0])
    rw = np.array([beds.width - gap])
    rh = np.array([beds.depth - gap])
    # Seed the free space with the obstacles: every fixed real rect is
    # inflated by gap (exactly the space a packed item would occupy) and
    # carved out, so new items keep the gap clearance from it.
    for ox, oy, ow, od in obstacles or ():
        rx, ry, rw, rh = _carve_free_rects(
            rx,
            ry,
            rw,
            rh,
            ox - (ow + gap) / 2.0,
            oy - (od + gap) / 2.0,
            ow + gap,
            od + gap,
        )
    # Cluster bounding box (inflated extents) + packed inflated area, for
    # the bbox-growth score. +inf/-inf start: the first item's growth is
    # its own area for every candidate, so the tie-break chain decides.
    bx0, by0, bx1, by1 = np.inf, np.inf, -np.inf, -np.inf
    cur_area = 0.0

    for uid, w, d in items:
        # Inflated candidate sizes; square footprints try one orientation.
        cands: list[tuple[float, float, bool]] = [(w + gap, d + gap, False)]
        if abs(w - d) > eps:
            cands.append((d + gap, w + gap, True))

        best: tuple[tuple[float, ...], int, float, float, bool] | None = None
        for oi, (cw, ch, rotated) in enumerate(cands):
            fit = (rw >= cw - eps) & (rh >= ch - eps)
            if not fit.any():
                continue
            fi = np.nonzero(fit)[0]
            # Score, vectorized over the free rects.  Primary: the cluster
            # bbox's LARGER side after placement — growing the shorter
            # side beats stretching the longer one, so identical pieces
            # build a square-ish blob instead of a band/tower (a pure
            # area-growth score degenerates: filling the long dimension
            # adds less bbox area than opening a new row).  Secondary:
            # bbox area growth — holes inside the blob (growth 0) fill
            # before the blob widens.  Then Best-Short-Side-Fit so tight
            # spots win, then leftover extent, position, orientation.
            nx0 = np.minimum(bx0, rx[fi])
            nx1 = np.maximum(bx1, rx[fi] + cw)
            ny0 = np.minimum(by0, ry[fi])
            ny1 = np.maximum(by1, ry[fi] + ch)
            growth = (nx1 - nx0) * (ny1 - ny0) - cur_area
            max_dim = np.maximum(nx1 - nx0, ny1 - ny0)
            short = np.minimum(rw[fi] - cw, rh[fi] - ch)
            long_left = np.maximum(rw[fi] - cw, rh[fi] - ch)
            # Ties: smaller longer leftover, lower y, left x, unrotated
            # preferred (lexsort: last key is primary).
            order = np.lexsort(
                (
                    np.full(len(fi), oi),
                    rx[fi],
                    ry[fi],
                    long_left,
                    short,
                    growth,
                    max_dim,
                )
            )
            k = int(fi[order[0]])
            key = (
                float(max_dim[order[0]]),
                float(growth[order[0]]),
                float(short[order[0]]),
                float(long_left[order[0]]),
                float(ry[k]),
                float(rx[k]),
                oi,
            )
            if best is None or key < best[0]:
                best = (key, k, cw, ch, rotated)
        if best is None:
            leftovers.append(uid)
            continue

        _, k, cw, ch, rotated = best
        px = float(rx[k])
        py = float(ry[k])
        placements[uid] = (px + cw / 2.0, py + ch / 2.0, rotated)
        px1 = px + cw
        py1 = py + ch
        bx0, by0 = min(bx0, px), min(by0, py)
        bx1, by1 = max(bx1, px1), max(by1, py1)
        cur_area += cw * ch

        # Split every free rect intersecting the placed inflated rect into
        # up to four maximal strips (drops the intersecting rects; touching
        # rects survive — the strict overlap test keeps them).
        rx, ry, rw, rh = _carve_free_rects(rx, ry, rw, rh, px, py, cw, ch)

    if not placements:
        return placements, leftovers

    if obstacles is not None:
        # Packing around fixed rects (incremental placement): placements
        # are absolute (centering would move items onto the obstacles /
        # out of the zone).  An EMPTY obstacles list still selects this
        # mode — only a from-scratch pack (None) is centered.
        return placements, leftovers

    # Center the packed cluster: translate so the occupied bounding box
    # middle coincides with the bed middle.
    dims_by_uid = {uid: (w, d) for uid, w, d in items}
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for uid, (cx, cy, rotated) in placements.items():
        w0, d0 = dims_by_uid[uid]
        hw, hd = (d0, w0) if rotated else (w0, d0)
        min_x = min(min_x, cx - hw / 2.0)
        max_x = max(max_x, cx + hw / 2.0)
        min_y = min(min_y, cy - hd / 2.0)
        max_y = max(max_y, cy + hd / 2.0)
    dx = beds.width / 2.0 - (min_x + max_x) / 2.0
    dy = beds.depth / 2.0 - (min_y + max_y) / 2.0
    placements = {
        uid: (cx + dx, cy + dy, rot) for uid, (cx, cy, rot) in placements.items()
    }
    return placements, leftovers


@dataclasses.dataclass
class AutoPlaceReport:
    '''Return value of :func:`auto_place`.

    Attributes:
        placed: Pieces now on selected beds.
        staged: Pieces sent to staging.
        per_bed: Piece count per selected bed index.
    '''

    placed: int = 0
    staged: int = 0
    per_bed: dict[int, int] = dataclasses.field(default_factory=dict)


def _apply_placements(
    placed: dict[str, tuple[int, float, float, bool]],
    pool: list[tuple[PrintPiece, float, float]],
) -> None:
    '''Write packer placements into their pieces.

    Args:
        placed: ``{piece_uid: (bed, cx, cy, rotated90)}`` — zone-local
            CENTER coords per piece uid.
        pool: The ``(piece, w, d)`` pool the uids refer to.
    '''
    for piece, _w, _d in pool:
        if piece.uid in placed:
            bed_idx, cx, cy, rotated = placed[piece.uid]
            piece.bed = bed_idx
            piece.position_x = cx
            piece.position_y = cy
            piece.rotated90 = rotated


def _pack_sequential(
    valid_pool: list[tuple[PrintPiece, float, float]],
    selected: list[int],
    beds: PrintBedSettings,
) -> None:
    '''Fill each selected bed before moving to the next one.

    Plain packing with no modifiers: pieces of all colors may share a
    bed.  Pieces that fit nowhere keep their staging coordinates (they
    are reset to staging and simply never re-assigned).

    Args:
        valid_pool: The ``(piece, w, d)`` pool to place.
        selected: Sorted selected bed indices.
        beds: The project's ``PrintBedSettings``.
    '''
    # Sort by priority (area DESC, longer side DESC, uid ASC) and reset
    # all pool pieces to staging so they can be repacked.
    valid_pool.sort(key=lambda t: _pool_key(t[0].uid, t[1], t[2]))
    for piece, _w, _d in valid_pool:
        piece.bed = PRINT_STAGING_BED

    # MaxRects-pack each selected bed in order.
    remaining: list[tuple[PrintPiece, float, float]] = list(valid_pool)
    for bed_idx in selected:
        items: list[tuple[str, float, float]] = [(p.uid, w, d) for p, w, d in remaining]
        placements, _leftovers = _maxrects_pack(items, beds)
        placed = {
            uid: (bed_idx, cx, cy, rot) for uid, (cx, cy, rot) in placements.items()
        }
        _apply_placements(placed, remaining)
        # Remaining pieces move to the next bed (leftovers stay staged).
        remaining = [t for t in remaining if t[0].uid not in placements]


def _pack_even(
    valid_pool: list[tuple[PrintPiece, float, float]],
    selected: list[int],
    beds: PrintBedSettings,
    staged_rects: list[tuple[float, float, float, float]],
) -> None:
    '''Balance total piece area across the selected beds, then pack each.

    Greedy LPT (area DESC) assigns every piece to the bed with the
    smallest total assigned area; every bed is MaxRects-packed
    afterwards.  Pieces the packer rejects (the area assignment
    overflowed the real packing) go to staging.

    Args:
        valid_pool: The ``(piece, w, d)`` pool to place.
        selected: Sorted selected bed indices.
        beds: The project's ``PrintBedSettings``.
        staged_rects: Running list of staged piece rects (grown by staging).
    '''
    # Sort LPT by area DESC (ties as in _pool_key).
    valid_pool.sort(key=lambda t: _pool_key(t[0].uid, t[1], t[2]))

    assigned_area: dict[int, float] = {b: 0.0 for b in selected}
    per_bed_items: dict[int, list[tuple[str, float, float]]] = {b: [] for b in selected}
    for piece, w, d in valid_pool:
        best_bed = min(selected, key=lambda b: assigned_area[b])
        per_bed_items[best_bed].append((piece.uid, w, d))
        assigned_area[best_bed] += w * d

    placed: dict[str, tuple[int, float, float, bool]] = {}
    for bed_idx in selected:
        placements, _leftovers = _maxrects_pack(per_bed_items[bed_idx], beds)
        for uid, (cx, cy, rotated) in placements.items():
            placed[uid] = (bed_idx, cx, cy, rotated)
    _apply_placements(placed, valid_pool)
    for piece, w, d in valid_pool:
        if piece.uid not in placed:
            _stage_piece(piece, w, d, staged_rects, beds)


def _color_groups_sorted(
    valid_pool: list[tuple[PrintPiece, float, float]],
    colors: dict[str, str],
) -> list[tuple[str, list[tuple[PrintPiece, float, float]]]]:
    '''Group the pool by block color, largest total area first.

    Args:
        valid_pool: The ``(piece, w, d)`` pool to group.
        colors: ``{block_uid: color}`` from :func:`block_colors`.

    Returns:
        ``(color, items)`` pairs ordered by total area DESC, ties by hex
        ASC.
    '''
    groups: dict[str, list[tuple[PrintPiece, float, float]]] = {}
    for piece, w, d in valid_pool:
        groups.setdefault(colors.get(piece.block_uid, ""), []).append((piece, w, d))
    return sorted(
        groups.items(),
        key=lambda t: (-sum(w * d for _, w, d in t[1]), t[0]),
    )


def _pack_by_color(
    valid_pool: list[tuple[PrintPiece, float, float]],
    colors: dict[str, str],
    selected: list[int],
    beds: PrintBedSettings,
    staged_rects: list[tuple[float, float, float, float]],
) -> None:
    '''Color-exclusive sequential fill: one print job per color.

    Colors are ordered by total area DESC.  Each color consumes unused
    beds in selection order, spilling onto further beds when one bed is
    not enough; pieces left when the beds run out go to staging.  A color
    never shares a bed with another color.

    Args:
        valid_pool: The ``(piece, w, d)`` pool to place.
        colors: ``{block_uid: color}`` from :func:`block_colors`.
        selected: Sorted selected bed indices.
        beds: The project's ``PrintBedSettings``.
        staged_rects: Running list of staged piece rects (grown by staging).
    '''
    free_beds = list(selected)
    for _color, items in _color_groups_sorted(valid_pool, colors):
        items_sorted = sorted(items, key=lambda t: _pool_key(t[0].uid, t[1], t[2]))
        pending = items_sorted
        while pending and free_beds:
            bed_idx = free_beds.pop(0)
            pack_items = [(p.uid, w, d) for p, w, d in pending]
            placements, _leftovers = _maxrects_pack(pack_items, beds)
            if not placements:  # pragma: no cover - pre-filter guarantees a fit
                break
            placed = {
                uid: (bed_idx, cx, cy, rot) for uid, (cx, cy, rot) in placements.items()
            }
            _apply_placements(placed, pending)
            pending = [t for t in pending if t[0].uid not in placements]
        for piece, w, d in pending:
            _stage_piece(piece, w, d, staged_rects, beds)


def _pack_even_by_color(
    valid_pool: list[tuple[PrintPiece, float, float]],
    colors: dict[str, str],
    selected: list[int],
    beds: PrintBedSettings,
    staged_rects: list[tuple[float, float, float, float]],
) -> None:
    '''Color-exclusive AND area-balanced packing.

    Colors are ordered by total area DESC.  Each color first claims a
    PROPORTIONAL share of the free beds (floor of ``bed_count *
    area_share``, at least one bed) and spreads its pieces across that
    allocation as evenly as possible (greedy LPT by area, then MaxRects
    per bed) — a color spreads even when its pieces would fit on fewer
    beds.  When pieces still do not fit, the allocation grows one bed at
    a time, re-distributing ALL of the color's pieces on every growth
    (the best attempt wins when the beds run out); beds that received no
    pieces are released back for the remaining colors.  Pieces left when
    the beds run out go to staging.

    Args:
        valid_pool: The ``(piece, w, d)`` pool to place.
        colors: ``{block_uid: color}`` from :func:`block_colors`.
        selected: Sorted selected bed indices.
        beds: The project's ``PrintBedSettings``.
        staged_rects: Running list of staged piece rects (grown by staging).
    '''
    free_beds = list(selected)
    groups = _color_groups_sorted(valid_pool, colors)
    total_area = sum(w * d for _c, items in groups for _p, w, d in items)
    if total_area <= 0:  # pragma: no cover - groups are non-empty here
        return
    n_beds = len(selected)
    for color, items in groups:
        ideal = n_beds * sum(w * d for _p, w, d in items) / total_area
        k_target = max(1, int(ideal))
        items_sorted = sorted(items, key=lambda t: _pool_key(t[0].uid, t[1], t[2]))
        used_beds: list[int] = []
        # Claim the proportional allocation upfront so the pieces spread
        # across it instead of filling one bed before opening the next.
        for _i in range(min(k_target, len(free_beds))):
            used_beds.append(free_beds.pop(0))
        best_placements: dict[str, tuple[int, float, float, bool]] = {}
        best_count = -1
        pending = items_sorted
        while True:
            if used_beds:
                # LPT the color's pieces across its beds (balance area).
                items_by_uid = [(p.uid, w, d) for p, w, d in items_sorted]
                loads = {b: 0.0 for b in used_beds}
                per_bed_items: dict[int, list[tuple[str, float, float]]] = {
                    b: [] for b in used_beds
                }
                for uid, w, d in items_by_uid:
                    target_bed = min(used_beds, key=lambda b: loads[b])
                    per_bed_items[target_bed].append((uid, w, d))
                    loads[target_bed] += w * d
                # MaxRects-pack every bed of this color.
                final: dict[str, tuple[int, float, float, bool]] = {}
                for bed_idx in used_beds:
                    placements, _leftovers = _maxrects_pack(
                        per_bed_items[bed_idx], beds
                    )
                    for uid, (cx, cy, rotated) in placements.items():
                        final[uid] = (bed_idx, cx, cy, rotated)
                if len(final) > best_count:
                    best_count = len(final)
                    best_placements = final
                pending = [t for t in items_sorted if t[0].uid not in final]
            if not pending or not free_beds:
                break
            # Grow the allocation by one bed and re-distribute.
            used_beds.append(free_beds.pop(0))
        # Release beds that received no pieces back to the free pool.
        placed_beds = {bed for bed, _cx, _cy, _rot in best_placements.values()}
        released = [b for b in used_beds if b not in placed_beds]
        if released:
            free_beds = sorted(free_beds + released)
        _apply_placements(best_placements, items_sorted)
        for piece, w, d in items_sorted:
            if piece.uid not in best_placements:
                _stage_piece(piece, w, d, staged_rects, beds)


def auto_place(
    project: Project,
    bed_indices: list[int],
    *,
    by_color: bool = True,
    even: bool = True,
) -> AutoPlaceReport:
    '''Full repack of the selected beds using MaxRects packing.

    Two independent modifiers shape the packing strategy:

    * ``by_color`` — pieces of different colors never share a bed (one
      print job per color).
    * ``even`` — pieces spread across ALL selected beds (greedy LPT by
      area) instead of filling each bed before moving to the next one.

    Both flags combine: every color then gets exclusive beds — a
    proportional share of the selection (by total piece area) — and
    spreads its pieces across that allocation as evenly as possible,
    claiming more beds only while pieces do not fit.  Without any
    modifier the packer fills the first selected bed and moves to the
    next one when it is full.

    Args:
        project: The active project.
        bed_indices: Selected bed indices (will be normalised).
        by_color: Never mix colors on one bed.
        even: Balance total piece area across the selected beds.

    Returns:
        An ``AutoPlaceReport`` with placement counts.
    '''
    beds = project.print_layout.beds
    fp = block_footprints(project)
    colors = block_colors(project)

    # Normalise selection.
    selected = sorted(set(i for i in bed_indices if 0 <= i < beds.count))
    if not selected:
        return AutoPlaceReport()

    report = AutoPlaceReport(per_bed={b: 0 for b in selected})

    # Pool: pieces on selected beds + staging.
    pool_pieces = [
        p
        for p in project.print_layout.pieces
        if p.bed in selected or p.bed == PRINT_STAGING_BED
    ]

    # Shared pre-filter: unknown blocks and oversized pieces go to staging.
    # Oversized pieces get a free staging spot so they do not overlap.
    valid_pool: list[tuple[PrintPiece, float, float]] = []
    staged_rects: list[tuple[float, float, float, float]] = []
    for piece in pool_pieces:
        uid = piece.block_uid
        if uid not in fp:
            piece.bed = PRINT_STAGING_BED
            continue
        w, d = fp[uid]
        # Oversized in BOTH orientations can never fit.
        if not (fits_bed(w, d, beds) or fits_bed(d, w, beds)):
            _stage_piece(piece, w, d, staged_rects, beds)
            continue
        valid_pool.append((piece, w, d))

    if by_color and even:
        _pack_even_by_color(valid_pool, colors, selected, beds, staged_rects)
    elif by_color:
        _pack_by_color(valid_pool, colors, selected, beds, staged_rects)
    elif even:
        _pack_even(valid_pool, selected, beds, staged_rects)
    else:
        _pack_sequential(valid_pool, selected, beds)

    # Final counts from actual state after all beds processed.
    report.placed = sum(1 for p in pool_pieces if p.bed in selected)
    report.staged = sum(1 for p in pool_pieces if p.bed == PRINT_STAGING_BED)
    for b in selected:
        report.per_bed[b] = sum(1 for p in pool_pieces if p.bed == b)

    return report
