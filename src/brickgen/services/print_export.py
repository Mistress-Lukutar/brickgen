'''
File:   print_export.py
Brief:  Assembly and export of print-bed compounds from the placement layout.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.20.0
'''

from __future__ import annotations

from typing import cast

import cadquery as cq

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings
from brickgen.core.project import Project
from brickgen.services.placement_service import bed_origin, piece_dims


def collect_bed_entries(project: Project, bed_index: int) -> tuple[list[dict], int]:
    '''Collect export entries for the pieces sitting fully inside one bed.

    Args:
        project: The active project.
        bed_index: Zero-based bed index to collect from.

    Returns:
        ``(entries, skipped)`` where *entries* is a list of picklable dicts
        ``{'settings': dict, 'position_x': float, 'position_y': float,
        'flipped': bool, 'rotated90': bool}`` in piece list order and
        *skipped* counts the pieces assigned to *bed_index* whose footprint
        is not fully inside the bed rectangle (orientation-aware: a rotated
        piece swaps its axes). Touching the edges is allowed:
        ``0 <= px - w/2 and px + w/2 <= beds.width`` (same for y/depth).
        Staged pieces and other beds are not this function's concern.
    '''
    beds = project.print_layout.beds
    if bed_index < 0 or bed_index >= beds.count:
        return ([], 0)

    # Merged settings are per block; pieces of one block share them.
    settings_cache: dict[str, BlockSettings] = {}
    footprints: dict[str, tuple[float, float]] = {}
    for i, block in enumerate(project.blocks):
        settings = project.block_settings(i)
        settings_cache[block.uid] = settings
        footprints[block.uid] = (settings.outer_length_x, settings.outer_length_y)

    entries: list[dict] = []
    skipped = 0
    for piece in project.print_layout.pieces:
        if piece.bed != bed_index:
            continue
        piece_settings = settings_cache.get(piece.block_uid)
        if piece_settings is None:
            skipped += 1
            continue
        w, d = piece_dims(piece, footprints)
        if not (
            0 <= piece.position_x - w / 2
            and piece.position_x + w / 2 <= beds.width
            and 0 <= piece.position_y - d / 2
            and piece.position_y + d / 2 <= beds.depth
        ):
            skipped += 1
            continue
        entries.append(
            {
                'settings': piece_settings.model_dump(),
                'position_x': piece.position_x,
                'position_y': piece.position_y,
                'flipped': piece.flipped,
                'rotated90': piece.rotated90,
            }
        )
    return (entries, skipped)


def bed_compound(entries: list[dict]) -> cq.Workplane:
    '''Build one Compound of all piece solids of a single bed.

    Each entry carries ``{'settings': dict, 'position_x', 'position_y',
    'flipped', 'rotated90'}``. The block solid is built centered on the
    origin. Transforms (matching the preview's mesh math exactly): a
    rotated piece first turns 90 degrees about Z; a flipped piece then
    rotates 180 degrees about the X axis (pins down) and is lifted by
    ``total_height`` so its bounding box sits on z=0 again. Every solid
    ends up translated to ``(position_x, position_y, 0)``.

    Args:
        entries: Piece entry dicts (see :func:`collect_bed_entries`).

    Returns:
        A Workplane wrapping the compound of all piece solids.

    Raises:
        ValueError: If *entries* is empty.
    '''
    if not entries:
        raise ValueError('no pieces in bed')
    solids: list[cq.Shape] = []
    for entry in entries:
        settings = BlockSettings.model_validate(entry['settings'])
        model = build_block(settings)
        shape = cast(cq.Shape, model.val())
        if entry.get('rotated90', False):
            # 90 deg about Z: the footprint swaps its axes.
            shape = shape.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 90.0)
        if entry.get('flipped', False):
            # 180 deg about X: z -> -z, y -> -y; lift back onto z >= 0.
            shape = shape.rotate((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 180.0)
            shape = shape.translate((0.0, 0.0, settings.total_height))
        shape = shape.translate(
            (float(entry['position_x']), float(entry['position_y']), 0.0)
        )
        solids.append(shape)
    compound = cq.Compound.makeCompound(solids)
    return cq.Workplane(obj=compound)


def collect_all_beds(project: Project) -> tuple[list[dict], int]:
    '''Collect export payloads for every bed that has printable pieces.

    Args:
        project: The active project.

    Returns:
        ``(beds, total_skipped)`` where *beds* holds one dict per bed with
        at least one fully-inside piece: ``{'index': i, 'pieces': [...],
        'skipped': n}``. *total_skipped* sums the per-bed skipped counts.
    '''
    beds_payload: list[dict] = []
    total_skipped = 0
    for index in range(project.print_layout.beds.count):
        entries, skipped = collect_bed_entries(project, index)
        total_skipped += skipped
        if entries:
            beds_payload.append({'index': index, 'pieces': entries, 'skipped': skipped})
    return (beds_payload, total_skipped)


def collect_world_entries(project: Project) -> tuple[list[dict], int]:
    '''Collect export entries for every bed in ONE world-space scene.

    Single-file export: pieces from all beds keep their zone-local
    positions converted to world mm (each bed's origin offset added), so
    the whole layout lands in one compound exactly as the placement
    viewport shows it. Staged pieces are never exported.

    Args:
        project: The active project.

    Returns:
        ``(entries, skipped)`` where *entries* is one flat list of
        picklable piece dicts with world positions, ordered by bed index
        then piece order, and *skipped* counts every piece not fully
        inside its bed.
    '''
    beds = project.print_layout.beds
    entries: list[dict] = []
    total_skipped = 0
    for index in range(beds.count):
        bed_entries, skipped = collect_bed_entries(project, index)
        total_skipped += skipped
        ox, oy = bed_origin(index, beds)
        for entry in bed_entries:
            entries.append(
                {
                    **entry,
                    'position_x': entry['position_x'] + ox,
                    'position_y': entry['position_y'] + oy,
                }
            )
    return (entries, total_skipped)


__all__ = [
    'collect_all_beds',
    'collect_bed_entries',
    'collect_world_entries',
    'bed_compound',
]
