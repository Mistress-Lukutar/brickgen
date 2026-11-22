'''
File:   placement_service.py
Brief:  Print-placement layout engine: zone geometry, sync, auto-place, distribute.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import dataclasses

from brickgen.core.project import (
    PRINT_STAGING_BED,
    PrintBedSettings,
    PrintPiece,
    Project,
)

# World-space layout of the placement scene: identical zones (beds and the
# staging area) sit side by side along X, separated by this gap. The staging
# zone is to the LEFT of bed 0.
BED_GAP_MM = 20.0

# Distribution modes for :func:`distribute`. This module stays pure logic:
# no cadquery / pyvista / Qt imports allowed here.
DISTRIBUTE_BY_COLOR = "by_color"
DISTRIBUTE_EVEN = "even"


def bed_origin(index: int, beds: PrintBedSettings) -> tuple[float, float]:
    '''World position (bottom-left corner) of *index* bed.

    Args:
        index: Zero-based bed index.
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(x, y)`` world coordinates of the bottom-left corner of that bed.
    '''
    return (index * (beds.width + BED_GAP_MM), 0.0)


def staging_origin(beds: PrintBedSettings) -> tuple[float, float]:
    '''World position (bottom-left corner) of the staging zone.

    Args:
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(x, y)`` world coordinates of the staging zone origin.
    '''
    return (-(beds.width + BED_GAP_MM), 0.0)


def zone_for_world_x(x: float, beds: PrintBedSettings) -> int:
    '''Map a world-space *x* coordinate to a zone index.

    Args:
        x: World X coordinate.
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``PRINT_STAGING_BED`` when *x* falls in the staging gap, otherwise
        a bed index clamped to ``[0, beds.count - 1]``.
    '''
    boundary = -(BED_GAP_MM / 2.0)
    if x < boundary:
        return PRINT_STAGING_BED
    idx = int((x + BED_GAP_MM / 2.0) // (beds.width + BED_GAP_MM))
    return max(0, min(idx, beds.count - 1))


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


def find_free_spot(
    placed: list[tuple[float, float, float, float]],
    width: float,
    depth: float,
    beds: PrintBedSettings,
) -> tuple[float, float] | None:
    '''Find the first free spot in a zone for a piece of given dimensions.

    Scans from the bottom-left corner using zone-local coordinates.  The
    candidate set is built from existing rect edges plus the initial edge;
    the first valid position wins.

    Args:
        placed: Existing rects as ``(cx, cy, w, d)`` in zone-local coords.
        width: New piece width in mm.
        depth: New piece depth in mm.
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(cx, cy)`` centre of the first free spot, or ``None`` when no
        position fits within zone margins.
    '''
    gap = beds.gap
    # Build candidate sets for x and y centres.
    xs_set: set[float] = {gap + width / 2.0}
    ys_set: set[float] = {gap + depth / 2.0}
    for rx, ry, rw, rd in placed:
        xs_set.add(rx + rw / 2.0 + gap + width / 2.0)
        ys_set.add(ry + rd / 2.0 + gap + depth / 2.0)
    xs = sorted(xs_set)
    ys = sorted(ys_set)

    for cy in ys:
        for cx in xs:
            # Check zone bounds with gap margin.
            if not (gap + width / 2.0 <= cx <= beds.width - gap - width / 2.0):
                continue
            if not (gap + depth / 2.0 <= cy <= beds.depth - gap - depth / 2.0):
                continue
            # Check overlap against every placed rect.
            ok = True
            for rx, ry, rw, rd in placed:
                if _overlaps(cx, cy, width, depth, rx, ry, rw, rd, gap):
                    ok = False
                    break
            if ok:
                return (cx, cy)
    return None


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
        3. Per block (in block order): append missing pieces, trying bed 0 then staging.
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
        for _i in range(to_add):
            new_piece = PrintPiece(block_uid=uid)
            # Try bed 0 first (convert existing pieces to tuples, using
            # their ORIENTATION-AWARE footprints — pieces may be rotated).
            bed0_tuples = [
                (p.position_x, p.position_y, *piece_dims(p, fp))
                for p in pieces
                if p.bed == 0
            ]
            spot = find_free_spot(bed0_tuples, w, d, beds)
            if spot is not None:
                new_piece.bed = 0
                new_piece.position_x, new_piece.position_y = spot
            else:
                # Try staging.
                staged_tuples = [
                    (p.position_x, p.position_y, *piece_dims(p, fp))
                    for p in pieces
                    if p.bed == PRINT_STAGING_BED
                ]
                spot = find_free_spot(staged_tuples, w, d, beds)
                if spot is not None:
                    new_piece.bed = PRINT_STAGING_BED
                    new_piece.position_x, new_piece.position_y = spot
                    report.staged += 1
                else:
                    # Fallback: overlap allowed in staging.
                    new_piece.bed = PRINT_STAGING_BED
                    new_piece.position_x = beds.gap + w / 2.0
                    new_piece.position_y = beds.gap + d / 2.0
                    report.staged += 1
            pieces.append(new_piece)
            report.added += 1

    return report


def _shelf_pack(
    items: list[tuple[str, float, float]],
    beds: PrintBedSettings,
) -> tuple[dict[str, tuple[float, float, bool]], list[str]]:
    '''Row-wise ("shelf") packing into one bed zone, with 90-degree turns.

    Items are placed left-to-right in rows; when a row overflows the bed
    width a new row starts.  Each item is tried in BOTH orientations
    (unrotated first, then rotated 90 degrees); the first orientation
    fitting the remaining row width AND the bed depth wins, so a piece is
    only turned when it would not fit otherwise.  Items that fit in no
    orientation go to leftovers.

    Args:
        items: ``(unique_id, width, depth)`` tuples where *unique_id* is an
            opaque identifier (typically ``piece.uid``) used for look-up
            after packing and ``(width, depth)`` is the UNROTATED
            footprint.  Order in the list is the placement priority.
        beds: The project's ``PrintBedSettings``.

    Returns:
        ``(placements, leftovers)`` where *placements* maps unique_id to
        ``(cx, cy, rotated90)`` — centre coords in zone-local mm plus the
        chosen orientation flag.
    '''
    gap = beds.gap
    cursor_x = gap
    cursor_y = gap
    row_h = 0.0
    placements: dict[str, tuple[float, float, bool]] = {}
    leftovers: list[str] = []

    def _fit(cw: float, cd: float) -> bool:
        return (
            cursor_x + cw <= beds.width - gap and cursor_y + cd <= beds.depth - gap
        )

    for uid, w, d in items:
        candidates = [(w, d, False), (d, w, True)]
        chosen: tuple[float, float, bool] | None = None
        for candidate in candidates:
            if _fit(candidate[0], candidate[1]):
                chosen = candidate
                break
        if chosen is None:
            # Wrap to the next row and retry both orientations.
            cursor_x = gap
            cursor_y += row_h + gap
            row_h = 0.0
            for candidate in candidates:
                if _fit(candidate[0], candidate[1]):
                    chosen = candidate
                    break
        if chosen is None:
            leftovers.append(uid)
            continue
        cw, cd, rotated = chosen
        placements[uid] = (cursor_x + cw / 2.0, cursor_y + cd / 2.0, rotated)
        cursor_x += cw + gap
        row_h = max(row_h, cd)

    return placements, leftovers


def _pool_sort(
    items: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    '''Sort pool by area DESC, longer side DESC, uid ASC.

    Deterministic ordering for :func:`auto_place` and :func:`distribute`.

    Args:
        items: ``(uid, width, depth)`` tuples.

    Returns:
        Sorted list in priority order.
    '''
    return sorted(items, key=lambda t: (-t[1] * t[2], -max(t[1], t[2]), t[0]))


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


def auto_place(project: Project, bed_indices: list[int]) -> AutoPlaceReport:
    '''Full repack of the selected beds using shelf packing.

    Args:
        project: The active project.
        bed_indices: Selected bed indices (will be normalised).

    Returns:
        An ``AutoPlaceReport`` with placement counts.
    '''
    beds = project.print_layout.beds
    fp = block_footprints(project)

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

    # Pre-filter: oversized and unknown-block pieces go to staging.
    valid_pool: list[tuple[PrintPiece, float, float]] = []
    for piece in pool_pieces:
        uid = piece.block_uid
        if uid not in fp:
            piece.bed = PRINT_STAGING_BED
            continue
        w, d = fp[uid]
        # Oversized in BOTH orientations can never fit — straight to staging.
        if not (fits_bed(w, d, beds) or fits_bed(d, w, beds)):
            piece.bed = PRINT_STAGING_BED
            continue
        valid_pool.append((piece, w, d))

    # Sort by priority (area DESC, longer side DESC, uid ASC).
    valid_pool.sort(key=lambda t: (-t[1] * t[2], -max(t[1], t[2]), t[0].block_uid))

    # Reset all pool pieces to staging so they can be repacked.
    for piece in pool_pieces:
        piece.bed = PRINT_STAGING_BED

    # Shelf-pack each selected bed in order via _shelf_pack.
    remaining_pieces: list[tuple[PrintPiece, float, float]] = list(valid_pool)
    for bed_idx in selected:
        items: list[tuple[str, float, float]] = [
            (p.uid, w, d) for p, w, d in remaining_pieces
        ]
        placements, leftovers_uids = _shelf_pack(items, beds)

        # Apply placements to matching pieces.
        for piece, w, d in remaining_pieces:
            if piece.uid in placements:
                cx, cy, rotated = placements[piece.uid]
                piece.bed = bed_idx
                piece.position_x = cx
                piece.position_y = cy
                piece.rotated90 = rotated
                report.per_bed[bed_idx] += 1
                report.placed += 1

        # Remaining pieces (leftovers stay staged).
        remaining_pieces = [
            (p, w, d) for p, w, d in remaining_pieces if p.uid not in placements
        ]

    # Final counts from actual state after all beds processed.
    report.placed = sum(1 for p in pool_pieces if p.bed in selected)
    report.staged = sum(1 for p in pool_pieces if p.bed == PRINT_STAGING_BED)
    for b in selected:
        report.per_bed[b] = sum(1 for p in pool_pieces if p.bed == b)

    return report


@dataclasses.dataclass
class DistributeReport:
    '''Return value of :func:`distribute`.

    Attributes:
        placed: Pieces now on selected beds.
        staged: Pieces sent to staging.
        per_bed: Piece count per selected bed index.
    '''

    placed: int = 0
    staged: int = 0
    per_bed: dict[int, int] = dataclasses.field(default_factory=dict)


def distribute(project: Project, mode: str, bed_indices: list[int]) -> DistributeReport:
    '''Distribute pieces across selected beds, then shelf-pack each.

    Args:
        project: The active project.
        mode: Distribution mode (``"even"`` or ``"by_color"``).
        bed_indices: Selected bed indices (will be normalised).

    Returns:
        A ``DistributeReport`` with placement counts.

    Raises:
        ValueError: When *mode* is not recognised.
    '''
    beds = project.print_layout.beds
    fp = block_footprints(project)
    colors = block_colors(project)

    # Normalise selection.
    selected = sorted(set(i for i in bed_indices if 0 <= i < beds.count))
    if not selected:
        return DistributeReport()

    report = DistributeReport(per_bed={b: 0 for b in selected})

    # Pool: pieces on selected beds + staging.
    pool_pieces = [
        p
        for p in project.print_layout.pieces
        if p.bed in selected or p.bed == PRINT_STAGING_BED
    ]

    if mode == DISTRIBUTE_EVEN:
        # Pre-filter oversized and unknown.
        valid_pool: list[tuple[PrintPiece, float, float]] = []
        for piece in pool_pieces:
            uid = piece.block_uid
            if uid not in fp:
                piece.bed = PRINT_STAGING_BED
                continue
            w, d = fp[uid]
            # Oversized in BOTH orientations can never fit.
            if not (fits_bed(w, d, beds) or fits_bed(d, w, beds)):
                piece.bed = PRINT_STAGING_BED
                continue
            valid_pool.append((piece, w, d))

        # Sort LPT by area DESC (ties as in _pool_sort).
        valid_pool.sort(key=lambda t: (-t[1] * t[2], -max(t[1], t[2]), t[0].block_uid))

        # Greedy LPT: assign each piece to bed with smallest total area.
        assigned_area: dict[int, float] = {b: 0.0 for b in selected}
        per_bed_pieces: dict[int, list[tuple[PrintPiece, float, float]]] = {
            b: [] for b in selected
        }

        for piece, w, d in valid_pool:
            area = w * d
            best_bed = min(selected, key=lambda b: assigned_area[b])
            per_bed_pieces[best_bed].append((piece, w, d))
            assigned_area[best_bed] += area

    elif mode == DISTRIBUTE_BY_COLOR:
        # Group by color, filtering oversized and unknown.
        color_groups: dict[str, list[tuple[PrintPiece, float, float]]] = {}
        for piece in pool_pieces:
            uid = piece.block_uid
            if uid not in fp:
                piece.bed = PRINT_STAGING_BED
                continue
            w, d = fp[uid]
            # Oversized in BOTH orientations can never fit.
            if not (fits_bed(w, d, beds) or fits_bed(d, w, beds)):
                piece.bed = PRINT_STAGING_BED
                continue
            color = colors.get(uid, "")
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append((piece, w, d))

        # Order groups by total area DESC, ties by hex ASC.
        group_totals = [
            (c, sum(w * d for _, w, d in items), items)
            for c, items in color_groups.items()
        ]
        group_totals.sort(key=lambda t: (-t[1], t[0]))

        # Capacity per bed.
        capacity = (beds.width - 2 * beds.gap) * (beds.depth - 2 * beds.gap)

        # Walk groups, fill beds sequentially.
        color_area: dict[int, float] = {b: 0.0 for b in selected}
        color_per_bed_pieces: dict[int, list[tuple[PrintPiece, float, float]]] = {
            b: [] for b in selected
        }

        for color, _, items in group_totals:
            # Sort items within group by area DESC (_pool_sort order).
            items_sorted = sorted(
                items, key=lambda t: (-t[1] * t[2], -max(t[1], t[2]), t[0].block_uid)
            )
            for piece, w, d in items_sorted:
                area = w * d
                placed_on_bed = False
                for bed_idx in selected:
                    if color_area[bed_idx] + area <= capacity:
                        color_per_bed_pieces[bed_idx].append((piece, w, d))
                        color_area[bed_idx] += area
                        placed_on_bed = True
                        break
                if not placed_on_bed:
                    piece.bed = PRINT_STAGING_BED

    else:
        raise ValueError(f"unknown distribute mode: {mode!r}")

    # Shelf-pack each selected bed with its assigned pieces via _shelf_pack.
    for bed_idx in selected:
        items_list = (
            color_per_bed_pieces[bed_idx]
            if mode == DISTRIBUTE_BY_COLOR
            else per_bed_pieces[bed_idx]
        )
        pack_items: list[tuple[str, float, float]] = [
            (p.uid, w, d) for p, w, d in items_list
        ]
        placements, leftovers_uids = _shelf_pack(pack_items, beds)

        # Apply placements to matching pieces.
        for piece, w, d in items_list:
            if piece.uid in placements:
                cx, cy, rotated = placements[piece.uid]
                piece.bed = bed_idx
                piece.position_x = cx
                piece.position_y = cy
                piece.rotated90 = rotated

    # Final counts from actual state after all beds processed.
    report.placed = sum(1 for p in pool_pieces if p.bed in selected)
    report.staged = sum(1 for p in pool_pieces if p.bed == PRINT_STAGING_BED)
    for b in selected:
        report.per_bed[b] = sum(1 for p in pool_pieces if p.bed == b)

    return report
