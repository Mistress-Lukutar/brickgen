'''
File:   assemble_service.py
Brief:  Pure Assemble-tab domain logic: part grouping, counts, snap and stacking.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.20.0
'''

from __future__ import annotations

import json
from typing import NamedTuple

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.models import BlockSettings
from brickgen.core.project import AssembledPiece, Project

# Overlap threshold (mm): footprint insets shrink by this much before the
# overlap test so exact edge-touching never counts as a real overlap.
_OVERLAP_EPS = 1e-6


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
    '''Axis-aligned footprint of one assembled piece, for snap queries.

    Attributes:
        x, y: Footprint center, mm.
        z: Bottom height of the piece, mm.
        hx, hy: Half extents of the (rotation-applied) footprint, mm.
        height: Piece height (bottom to top), mm.
    '''

    x: float
    y: float
    z: float
    hx: float
    hy: float
    height: float

    @property
    def top(self) -> float:
        '''Height of the piece top face, mm.'''
        return self.z + self.height


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


def assemble_supports(
    project: Project, exclude_uids: set[str] | None = None
) -> list[Support]:
    '''World footprints of the assembled pieces (for snapping/stacking).

    Args:
        project: Project holding the assembled pieces and shared geometry.
        exclude_uids: Piece uids to skip (e.g. the pieces being dragged).

    Returns:
        One :class:`Support` per assembled piece (rotation applied to the
        footprint halves).
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
        hx = settings.outer_length_x / 2
        hy = settings.outer_length_y / 2
        if piece.rot_z % 2:
            hx, hy = hy, hx
        supports.append(
            Support(
                x=piece.position_x,
                y=piece.position_y,
                z=piece.position_z,
                hx=hx,
                hy=hy,
                height=settings.total_height,
            )
        )
    return supports


def support_top_z(
    supports: list[Support], cx: float, cy: float, hx: float, hy: float
) -> float | None:
    '''Top height of the highest piece overlapping the given footprint.

    Args:
        supports: Candidate pieces (the dragged ones must be excluded).
        cx, cy: Center of the queried footprint, mm.
        hx, hy: Half extents of the queried footprint, mm.

    Returns:
        The top height of the highest overlapping piece, or None when the
        footprint overlaps nothing.
    '''
    best: float | None = None
    for support in supports:
        if (
            abs(cx - support.x) < hx + support.hx - _OVERLAP_EPS
            and abs(cy - support.y) < hy + support.hy - _OVERLAP_EPS
        ):
            top = support.top
            if best is None or top > best:
                best = top
    return best


def _grid_round(value: float, step: float) -> float:
    '''Round *value* to a multiple of *step*.'''
    return round(value / step) * step


def assemble_snap_target(
    cursor_x: float,
    cursor_y: float,
    hx: float,
    hy: float,
    supports: list[Support],
    step: float,
    snap_to_grid: bool,
    snap_to_parts: bool,
) -> tuple[float, float, float]:
    '''Resolve the drop position of a piece from the cursor point.

    The base target is the cursor point (rounded to the grid step when
    ``snap_to_grid``). With ``snap_to_parts`` two extra candidate kinds
    compete by their XY distance to the raw cursor: dropping ON TOP of
    the highest piece under the footprint, and aligning FLUSH to the
    sides of a nearby piece (at that piece's bottom height). Ties prefer
    the stacking candidate.

    Args:
        cursor_x, cursor_y: Unsnapped cursor point on the floor, mm.
        hx, hy: Half extents of the (rotation-applied) footprint, mm.
        supports: Footprints of the other pieces.
        step: Grid step in mm (> 0).
        snap_to_grid: Round the base target to grid-step multiples.
        snap_to_parts: Enable the stacking / side-flush candidates.

    Returns:
        The ``(x, y, z)`` drop position of the footprint center; *z* is
        the height of the piece bottom.
    '''
    step = max(step, 1e-9)
    base_x = _grid_round(cursor_x, step) if snap_to_grid else cursor_x
    base_y = _grid_round(cursor_y, step) if snap_to_grid else cursor_y

    best: tuple[float, float, float] | None = None
    best_score: tuple[float, int] | None = None

    def consider(x: float, y: float, z: float, stacked: bool) -> None:
        '''Keep the candidate nearest to the cursor (stacking wins ties).'''
        nonlocal best, best_score
        score = ((cursor_x - x) ** 2 + (cursor_y - y) ** 2, 0 if stacked else 1)
        if best_score is None or score < best_score:
            best_score = score
            best = (x, y, z)

    if snap_to_parts:
        # The plain floor candidate competes too: a side candidate far away
        # must never beat simply dropping at the cursor point.
        consider(base_x, base_y, 0.0, stacked=False)
        top = support_top_z(supports, base_x, base_y, hx, hy)
        if top is not None:
            consider(base_x, base_y, top, stacked=True)
        for support in supports:
            sx = support.hx + hx
            sy = support.hy + hy
            consider(support.x + sx, support.y, support.z, stacked=False)
            consider(support.x - sx, support.y, support.z, stacked=False)
            consider(support.x, support.y + sy, support.z, stacked=False)
            consider(support.x, support.y - sy, support.z, stacked=False)

    if best is None:
        return (base_x, base_y, 0.0)
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
