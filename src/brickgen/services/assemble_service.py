'''
File:   assemble_service.py
Brief:  Pure Assemble-tab domain logic: part grouping, counts, snap and stacking.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.models import BlockSettings
from brickgen.core.project import AssembledPiece, Project

# Overlap threshold (mm): footprint insets shrink by this much before the
# overlap test so exact edge-touching never counts as a real overlap.
_OVERLAP_EPS = 1e-6

# Sanity caps: a footprint decomposes into at most this many material
# rectangles (absurd exclusion patterns fall back to the bounding box),
# and side-flush candidates pair at most this many support/query rects.
_MAX_FOOTPRINT_RECTS = 64
_MAX_SIDE_PAIRS = 256


class Rect(NamedTuple):
    '''Axis-aligned footprint rectangle: center (cx, cy) + half extents.

    Material rectangles are the cells a block actually occupies; excluded
    cells leave voids the bounding box alone ignores.
    '''

    cx: float
    cy: float
    hx: float
    hy: float


class PartVariant(NamedTuple):
    '''One planned block inside a part group (one color variant).

    Attributes:
        block_index: Zero-based index of the block in ``project.blocks``
            (``index`` would shadow ``tuple.index``).
        uid: The block uid (assembled pieces reference it).
        color: Planned color of the block ('#rrggbb').
        quantity: Planned BOM quantity of the block.
        linked: True when the quantity follows the assemble count.
    '''

    block_index: int
    uid: str
    color: str
    quantity: int
    linked: bool


class PartGroup(NamedTuple):
    '''Planned-part panel entry: one geometry, one or more color variants.

    Blocks with identical geometry (grid size, flags and per-cell layout)
    share a group; the group name comes from the first variant's recipe
    name.

    Attributes:
        key: Canonical geometry signature (stable across renames/colors).
        name: Recipe name of the representative block.
        variants: The planned blocks of this geometry, in scene order.
    '''

    key: str
    name: str
    variants: list[PartVariant]


class Support(NamedTuple):
    '''World footprints of one assembled piece, for snap queries.

    ``hx``/``hy`` describe the material bounding box (exact for blocks
    without excluded cells), while ``rects`` carries the exact occupied
    rectangles — excluded cells leave voids the bounding box ignores, and
    only the rectangles count for stacking and side-flush queries.

    Attributes:
        x, y: Footprint (grid) center, mm.
        z: Bottom height of the piece, mm.
        hx, hy: Half extents of the material bounding box, mm.
        height: Piece height (bottom to top), mm.
        stack_top: Absolute height a piece dropped on top rests at, mm.
            Normal pieces stack at ``z + block_height`` — the studs of
            this piece enter the bottom cavities of the dropped one —
            while flipped pieces rest rim-to-rim at ``z + total_height``.
        rects: World-space material rectangles of the footprint.
    '''

    x: float
    y: float
    z: float
    hx: float
    hy: float
    height: float
    stack_top: float
    rects: tuple[Rect, ...]


# Block fields that are NOT part of the geometry signature.
_INSTANCE_FIELDS: tuple[str, ...] = (
    "uid",
    "position_x",
    "position_y",
    "color",
    "quantity",
    "quantity_linked",
)


def _geometry_payload(block: object) -> dict:
    '''Block fields that define its geometry (everything but scene/BOM data).'''
    return {
        key: value
        for key, value in block.model_dump().items()  # type: ignore[attr-defined]
        if key not in _INSTANCE_FIELDS
    }


def assemble_groups(project: Project) -> list[PartGroup]:
    '''Group the project blocks into planned-part entries by geometry.

    Args:
        project: The project whose blocks are grouped.

    Returns:
        One :class:`PartGroup` per distinct block geometry, in first-seen
        scene order.
    '''
    groups: dict[str, PartGroup] = {}
    order: list[str] = []
    for index, block in enumerate(project.blocks):
        payload = _geometry_payload(block)
        key = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        if key not in groups:
            try:
                name = block.to_settings(project.settings).recipe_name()
            except SettingsValidationError:
                name = f"{block.cells_x}x{block.cells_y}"  # type: ignore[attr-defined]
            groups[key] = PartGroup(key=key, name=name, variants=[])
            order.append(key)
        groups[key].variants.append(
            PartVariant(
                block_index=index,
                uid=block.uid,  # type: ignore[attr-defined]
                color=block.color,  # type: ignore[attr-defined]
                quantity=block.quantity,  # type: ignore[attr-defined]
                linked=block.quantity_linked,  # type: ignore[attr-defined]
            )
        )
    return [groups[key] for key in order]


def placed_counts(pieces: list[AssembledPiece]) -> dict[str, int]:
    '''Number of assembled pieces per referenced block uid.

    Pieces referencing removed blocks are counted too; callers filter by
    the current block uids when they need live counts only.

    Args:
        pieces: The assembled pieces of a project.

    Returns:
        Mapping of block uid -> placed copy count.
    '''
    counts: dict[str, int] = {}
    for piece in pieces:
        counts[piece.block_uid] = counts.get(piece.block_uid, 0) + 1
    return counts


def base_footprint_rects(settings: BlockSettings) -> tuple[Rect, ...]:
    '''Material rectangles of the unrotated, unflipped block footprint.

    The footprint is the full grid minus the excluded cells; row runs of
    consecutive occupied cells merge down across identical rows so usual
    shapes (T, L, steps) yield a handful of rectangles. Piece-centered
    local coordinates: cell ``(ix, iy)`` spans
    ``[ix*unit - outer/2, (ix+1)*unit - outer/2]`` per axis.

    Args:
        settings: Block geometry (``excluded_cells`` define the voids).

    Returns:
        One :class:`Rect` per merged material run; a block without
        excluded cells (or with a pathological exclusion pattern beyond
        ``_MAX_FOOTPRINT_RECTS``) is the single full-grid rectangle.
    '''
    unit = settings.unit_size
    half_x = settings.outer_length_x / 2
    half_y = settings.outer_length_y / 2
    excluded = settings.excluded_cell_set()
    if not excluded:
        return (Rect(0.0, 0.0, half_x, half_y),)
    rects: list[Rect] = []
    for iy in range(settings.cells_y):
        y0 = iy * unit - half_y
        y1 = y0 + unit
        run_start: int | None = None
        for ix in range(settings.cells_x + 1):
            occupied = ix < settings.cells_x and (ix, iy) not in excluded
            if run_start is None:
                if occupied:
                    run_start = ix
                continue
            if occupied:
                continue
            x0 = run_start * unit - half_x
            x1 = ix * unit - half_x
            run_start = None
            cx = (x0 + x1) / 2
            hx = (x1 - x0) / 2
            for index, rect in enumerate(rects):
                if (
                    abs(rect.cx - cx) <= _OVERLAP_EPS
                    and abs(rect.hx - hx) <= _OVERLAP_EPS
                    and abs(rect.cy + rect.hy - y0) <= _OVERLAP_EPS
                ):
                    # Same run as the previous row: merge downward.
                    prev_y0 = rect.cy - rect.hy
                    rects[index] = Rect(
                        rect.cx, (prev_y0 + y1) / 2, rect.hx, (y1 - prev_y0) / 2
                    )
                    break
            else:
                rects.append(Rect(cx, (y0 + y1) / 2, hx, (y1 - y0) / 2))
    if len(rects) > _MAX_FOOTPRINT_RECTS:
        return (Rect(0.0, 0.0, half_x, half_y),)
    return tuple(rects)


def transform_footprint_rects(
    rects: Iterable[Rect], rot_z: int, flipped: bool
) -> tuple[Rect, ...]:
    '''Piece-relative rectangles for one piece orientation.

    Applies ``rot_z`` quarter turns about Z (90 degrees counter-clockwise
    each) and then the flip's Y mirror — the same order as the mesh
    variant transform (rotate about Z, then 180 degrees about X).

    Args:
        rects: Base-orientation rectangles (as :func:`base_footprint_rects`).
        rot_z: Quarter-turn count 0..3.
        flipped: Mirror Y (pins-down flip).

    Returns:
        The transformed rectangle tuple.
    '''
    out: list[Rect] = []
    for rect in rects:
        cx, cy, hx, hy = rect
        for _ in range(int(rot_z) % 4):
            cx, cy, hx, hy = -cy, cx, hy, hx
        if flipped:
            cy = -cy
        out.append(Rect(cx, cy, hx, hy))
    return tuple(out)


def footprint_rects(
    settings: BlockSettings, rot_z: int, flipped: bool
) -> tuple[Rect, ...]:
    '''Piece-relative material rectangles for one block orientation.

    Args:
        settings: Block geometry.
        rot_z: Quarter-turn count 0..3.
        flipped: Mirror Y (pins-down flip).

    Returns:
        :func:`base_footprint_rects` transformed by
        :func:`transform_footprint_rects`.
    '''
    return transform_footprint_rects(base_footprint_rects(settings), rot_z, flipped)


def offset_rects(rects: Iterable[Rect], dx: float, dy: float) -> tuple[Rect, ...]:
    '''Translate piece-relative rectangles into world coordinates.'''
    return tuple(Rect(r.cx + dx, r.cy + dy, r.hx, r.hy) for r in rects)


def _bbox_halves(rects: Sequence[Rect]) -> tuple[float, float]:
    '''Half extents of the bounding box covering the given rectangles.'''
    return (
        (max(r.cx + r.hx for r in rects) - min(r.cx - r.hx for r in rects)) / 2,
        (max(r.cy + r.hy for r in rects) - min(r.cy - r.hy for r in rects)) / 2,
    )


def assemble_supports(
    project: Project, exclude_uids: set[str] | None = None
) -> list[Support]:
    '''World footprints of the assembled pieces (for snapping/stacking).

    Args:
        project: Project holding the assembled pieces and shared geometry.
        exclude_uids: Piece uids to skip (e.g. the pieces being dragged).

    Returns:
        One :class:`Support` per assembled piece — rotation/flip applied
        to the material rectangles, ``stack_top`` per the flip state.
    '''
    skip = exclude_uids or set()
    blocks_by_uid = {block.uid: block for block in project.blocks}
    supports: list[Support] = []
    for piece in project.assemble.pieces:
        if piece.uid in skip:
            continue
        block = blocks_by_uid.get(piece.block_uid)
        if block is None:
            continue
        settings: BlockSettings = block.to_settings(project.settings)
        rects = footprint_rects(settings, piece.rot_z, piece.flipped)
        hx, hy = _bbox_halves(rects)
        # Normal pieces stack at block_height (studs into the bottom
        # cavities of the dropped piece); flipped pieces rest rim-to-rim.
        stack_offset = settings.total_height if piece.flipped else settings.block_height
        supports.append(
            Support(
                x=piece.position_x,
                y=piece.position_y,
                z=piece.position_z,
                hx=hx,
                hy=hy,
                height=settings.total_height,
                stack_top=piece.position_z + stack_offset,
                rects=offset_rects(rects, piece.position_x, piece.position_y),
            )
        )
    return supports


def _rects_overlap(a: Sequence[Rect], b: Sequence[Rect]) -> bool:
    '''True when any rectangle of *a* materially overlaps one of *b*.

    Exact edge touching (flush faces) is not an overlap.
    '''
    return any(
        abs(ra.cx - rb.cx) < ra.hx + rb.hx - _OVERLAP_EPS
        and abs(ra.cy - rb.cy) < ra.hy + rb.hy - _OVERLAP_EPS
        for ra in a
        for rb in b
    )


def assemble_pose_collides(
    supports: list[Support],
    rects: Sequence[Rect],
    z: float,
    height: float,
    stack_offset: float,
) -> bool:
    '''True when the piece pose intersects any support volume.

    The XY test uses the material rectangles only — excluded-cell voids
    stay enterable. The Z test is a plain interval overlap with two
    legal contacts: resting ON a support (pose bottom exactly at its
    :attr:`Support.stack_top`, studs entering its top cavities) and
    sitting UNDER one (support bottom exactly at the pose stud plane
    ``z + stack_offset``). Exact edge touching anywhere is never an
    intersection.

    Args:
        supports: Other pieces (the moving one must be excluded).
        rects: World-space material rectangles of the pose.
        z: Bottom height of the pose, mm.
        height: Total height of the pose, mm.
        stack_offset: Stud plane offset of the pose — ``block_height``
            for a normal piece, ``total_height`` for a flipped one.

    Returns:
        True when the pose volume intersects a support volume.
    '''
    top = z + height
    for support in supports:
        if not _rects_overlap(support.rects, rects):
            continue
        if z >= support.z + support.height - _OVERLAP_EPS:
            continue  # fully above the support
        if support.z >= top - _OVERLAP_EPS:
            continue  # fully below the support
        if abs(z - support.stack_top) <= _OVERLAP_EPS:
            continue  # studs enter the support's top cavities
        if abs(support.z - (z + stack_offset)) <= _OVERLAP_EPS:
            continue  # the support's cavities sit on this pose's studs
        return True
    return False


def _grid_round(value: float, step: float) -> float:
    '''Round *value* to a multiple of *step*.'''
    return round(value / step) * step


def assemble_snap_target(
    cursor_x: float,
    cursor_y: float,
    cursor_z: float,
    rects: Sequence[Rect],
    height: float,
    block_height: float,
    flipped: bool,
    supports: list[Support],
    step: float,
    snap_to_grid: bool,
    snap_to_parts: bool,
) -> tuple[float, float, float]:
    '''Resolve the free-3D drop position of a piece from the cursor point.

    The base target is the cursor point on the caller's working plane
    (``cursor_z`` is the piece bottom height): X/Y are rounded so the
    footprint minimum corner — not the center — lands on a grid-step
    multiple when ``snap_to_grid``, Z is rounded to the step. With
    ``snap_to_parts`` magnet candidates compete by their 3D distance to
    the raw cursor: the free base position itself, stacking ON TOP of
    any piece whose material overlaps the footprint (at its
    ``stack_top`` — any height, so towers and floating pieces stack
    alike), and aligning FLUSH to the sides of a nearby material
    rectangle at that piece's bottom or top plane. Candidates that would
    intersect another piece are discarded; ties prefer stacking.

    The returned position may still intersect a piece when every
    candidate collides (the free base position is then returned for the
    red "blocked" feedback) — callers must gate the actual drop with
    :func:`assemble_pose_collides`.

    Args:
        cursor_x, cursor_y: Unsnapped cursor point on the working
            plane, mm.
        cursor_z: Working plane height (piece bottom), mm.
        rects: Piece-relative material rectangles of the moving piece.
        height: Total height of the moving piece, mm.
        block_height: Stud plane height of the moving piece, mm.
        flipped: Flip state of the moving piece (stacks rim-to-rim).
        supports: Footprints of the other pieces.
        step: Grid step in mm (> 0).
        snap_to_grid: Round the base target to grid-step multiples.
        snap_to_parts: Enable the stacking / side-flush candidates.

    Returns:
        The ``(x, y, z)`` drop position of the footprint center; *z* is
        the height of the piece bottom.
    '''
    step = max(step, 1e-9)
    hx, hy = _bbox_halves(rects)
    base_x = _grid_round(cursor_x - hx, step) + hx if snap_to_grid else cursor_x
    base_y = _grid_round(cursor_y - hy, step) + hy if snap_to_grid else cursor_y
    base_z = _grid_round(cursor_z, step) if snap_to_grid else cursor_z
    stack_offset = height if flipped else block_height

    best: tuple[float, float, float] | None = None
    best_score: tuple[float, int] | None = None

    def consider(x: float, y: float, z: float, stacked: bool) -> None:
        '''Keep the candidate nearest to the cursor (stacking wins ties).'''
        nonlocal best, best_score
        score = (
            (cursor_x - x) ** 2 + (cursor_y - y) ** 2 + (cursor_z - z) ** 2,
            0 if stacked else 1,
        )
        if best_score is None or score < best_score:
            best_score = score
            best = (x, y, z)

    def valid_pose(x: float, y: float, z: float) -> bool:
        '''False when the pose at (x, y, z) would intersect a piece.'''
        return not assemble_pose_collides(
            supports, offset_rects(rects, x, y), z, height, stack_offset
        )

    if snap_to_parts:
        base_rects_w = offset_rects(rects, base_x, base_y)
        # The free working-plane candidate competes too — but only when
        # it does not intersect a piece: a blocked base position must
        # never hide a legal stacking/flush magnet nearby.
        if valid_pose(base_x, base_y, base_z):
            consider(base_x, base_y, base_z, stacked=False)
        for support in supports:
            if _rects_overlap(support.rects, base_rects_w):
                # Stacking on this piece at its contact plane (legal at
                # ANY height: towers, floating pieces, slot bottoms).
                top = support.stack_top
                if valid_pose(base_x, base_y, top):
                    consider(base_x, base_y, top, stacked=True)
            # Flush against every material rectangle of the support and
            # the moved piece, at the support's bottom AND top planes;
            # huge rect sets fall back to the bounding boxes to keep the
            # candidate count bounded.
            support_rects: Sequence[Rect] = support.rects
            query_rects: Sequence[Rect] = rects
            if len(support_rects) * len(query_rects) > _MAX_SIDE_PAIRS:
                support_rects = (Rect(support.x, support.y, support.hx, support.hy),)
                query_rects = (Rect(0.0, 0.0, hx, hy),)
            for support_rect in support_rects:
                for query_rect in query_rects:
                    qx, qy = query_rect.cx, query_rect.cy
                    sx = support_rect.hx + query_rect.hx
                    sy = support_rect.hy + query_rect.hy
                    for plane in (support.z, support.stack_top):
                        for cx, cy in (
                            (support_rect.cx + sx, support_rect.cy),
                            (support_rect.cx - sx, support_rect.cy),
                            (support_rect.cx, support_rect.cy + sy),
                            (support_rect.cx, support_rect.cy - sy),
                        ):
                            x, y = cx - qx, cy - qy
                            if valid_pose(x, y, plane):
                                consider(x, y, plane, stacked=False)

    if best is None:
        return (base_x, base_y, base_z)
    return best  # type: ignore[return-value]


def assemble_summary(project: Project) -> str:
    '''Human-readable assemble status: placed copies vs planned total.

    Args:
        project: The project to summarize.

    Returns:
        ``"Assembly: placed/planned pieces (K part types)"`` or the empty
        scene notice.
    '''
    planned = sum(block.quantity for block in project.blocks)
    placed = len(project.assemble.pieces)
    groups = len(assemble_groups(project))
    if not project.blocks:
        return "Assembly is empty - add blocks on the Planning tab."
    return f"Assembly: {placed}/{planned} pieces placed ({groups} part types)"
