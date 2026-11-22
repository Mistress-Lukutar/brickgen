'''
File:   test_print_export.py
Brief:  Unit tests for print-bed entry collection and compound assembly.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.17.0
'''

from __future__ import annotations

import pytest

from brickgen.core.models import BlockSettings
from brickgen.core.project import (
    PRINT_STAGING_BED,
    BlockInstance,
    PrintBedSettings,
    PrintPiece,
    Project,
)
from brickgen.services.print_export import (
    bed_compound,
    collect_all_beds,
    collect_bed_entries,
)


def _make_project(
    *blocks: BlockInstance, beds: PrintBedSettings | None = None
) -> Project:
    '''Project with shared default geometry and explicit bed settings.'''
    project = Project(blocks=list(blocks))
    if beds is not None:
        project.print_layout.beds = beds
    return project


def _piece(
    block_uid: str, x: float, y: float, bed: int = 0, flipped: bool = False
) -> PrintPiece:
    '''Convenience piece builder.'''
    return PrintPiece(
        block_uid=block_uid,
        bed=bed,
        position_x=x,
        position_y=y,
        flipped=flipped,
    )


# ===========================================================================
# collect_bed_entries
# ===========================================================================


def test_collect_bed_entries_inside_piece() -> None:
    '''A fully inside piece yields one entry with passthrough fields.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=1),
        beds=PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0),
    )
    proj.print_layout.pieces.append(_piece("a", 8.0, 8.0, bed=0, flipped=True))

    entries, skipped = collect_bed_entries(proj, 0)
    assert skipped == 0
    assert len(entries) == 1
    entry = entries[0]
    assert entry["position_x"] == 8.0
    assert entry["position_y"] == 8.0
    assert entry["flipped"] is True
    assert entry["settings"]["cells_x"] == 2


def test_collect_bed_entries_skips_overhanging_piece() -> None:
    '''A piece whose footprint crosses the bed edge is skipped.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=2),
        beds=PrintBedSettings(width=220.0, depth=220.0, count=1, gap=0.0),
    )
    # 8x8 footprint: center 216.0 -> x range [212, 220] touches the edge: OK.
    proj.print_layout.pieces.append(_piece("a", 216.0, 8.0))
    # Center 216.1 -> x max 220.1 > 220: overhang.
    proj.print_layout.pieces.append(_piece("a", 216.1, 8.0))

    entries, skipped = collect_bed_entries(proj, 0)
    assert len(entries) == 1
    assert skipped == 1


def test_collect_bed_entries_out_of_range_bed() -> None:
    '''Out-of-range bed index yields no entries.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=1, cells_y=1),
        beds=PrintBedSettings(count=1),
    )
    proj.print_layout.pieces.append(_piece("a", 4.0, 4.0, bed=0))
    entries, skipped = collect_bed_entries(proj, 5)
    assert entries == []
    assert skipped == 0


def test_collect_bed_entries_ignores_staged_pieces() -> None:
    '''Staged pieces belong to no bed and are not collected.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=1, cells_y=1),
        beds=PrintBedSettings(count=1),
    )
    proj.print_layout.pieces.append(_piece("a", 4.0, 4.0, bed=PRINT_STAGING_BED))
    entries, skipped = collect_bed_entries(proj, 0)
    assert entries == []
    assert skipped == 0


# ===========================================================================
# bed_compound
# ===========================================================================


def _entry(cells_x: int, cells_y: int, x: float, y: float, flipped: bool) -> dict:
    settings = BlockSettings(cells_x=cells_x, cells_y=cells_y)
    return {
        'settings': settings.model_dump(),
        'position_x': x,
        'position_y': y,
        'flipped': flipped,
    }


def test_bed_compound_bbox_and_position() -> None:
    '''One unflipped 2x2 piece at (10, 20): bbox matches footprint/height.'''
    settings = BlockSettings(cells_x=2, cells_y=2)
    half_x = settings.outer_length_x / 2
    half_y = settings.outer_length_y / 2
    height = settings.total_height

    compound = bed_compound([_entry(2, 2, 10.0, 20.0, False)])
    bbox = compound.val().BoundingBox()
    assert bbox.xmin == pytest.approx(10.0 - half_x, abs=1e-3)
    assert bbox.xmax == pytest.approx(10.0 + half_x, abs=1e-3)
    assert bbox.ymin == pytest.approx(20.0 - half_y, abs=1e-3)
    assert bbox.ymax == pytest.approx(20.0 + half_y, abs=1e-3)
    assert bbox.zmin == pytest.approx(0.0, abs=1e-3)
    assert bbox.zmax == pytest.approx(height, abs=1e-3)


def test_bed_compound_flip_is_rigid_invariant() -> None:
    '''A flipped piece has the same bounding box and volume (rigid motion).'''
    straight = bed_compound([_entry(2, 2, 10.0, 20.0, False)])
    flipped = bed_compound([_entry(2, 2, 10.0, 20.0, True)])

    bbox_straight = straight.val().BoundingBox()
    bbox_flipped = flipped.val().BoundingBox()
    for attr in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        assert getattr(bbox_flipped, attr) == pytest.approx(
            getattr(bbox_straight, attr), abs=1e-3
        )
    assert flipped.val().Volume() == pytest.approx(straight.val().Volume(), rel=1e-6)


def test_bed_compound_two_pieces_volume_sums() -> None:
    '''A compound of two pieces has their combined volume.'''
    single = bed_compound([_entry(2, 2, 10.0, 20.0, False)])
    one_by_one = bed_compound([_entry(1, 1, 0.0, 0.0, False)])
    double = bed_compound(
        [_entry(2, 2, 10.0, 20.0, False), _entry(1, 1, 30.0, 20.0, False)]
    )
    expected = single.val().Volume() + one_by_one.val().Volume()
    assert double.val().Volume() == pytest.approx(expected, rel=1e-6)


def test_bed_compound_empty_rejected() -> None:
    '''An empty entry list is rejected.'''
    with pytest.raises(ValueError, match="no pieces in bed"):
        bed_compound([])


# ===========================================================================
# collect_all_beds
# ===========================================================================


def test_collect_all_beds_reports_skipped() -> None:
    '''Beds without printable pieces are omitted; skips are summed.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=3),
        beds=PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0),
    )
    # Bed 0: one inside, one overhanging.
    proj.print_layout.pieces.append(_piece("a", 8.0, 8.0, bed=0))
    proj.print_layout.pieces.append(_piece("a", 216.1, 8.0, bed=0))
    # Bed 1: one inside.
    proj.print_layout.pieces.append(_piece("a", 8.0, 8.0, bed=1))

    beds_payload, total_skipped = collect_all_beds(proj)
    assert [b["index"] for b in beds_payload] == [0, 1]
    assert beds_payload[0]["skipped"] == 1
    assert beds_payload[1]["skipped"] == 0
    assert len(beds_payload[0]["pieces"]) == 1
    assert total_skipped == 1


# ===========================================================================
# rotated90 integration
# ===========================================================================


def test_collect_bed_entries_rotated_bounds() -> None:
    '''Bounds checks use the swapped footprint of a rotated piece.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=4, cells_y=2, quantity=3),
        beds=PrintBedSettings(width=220.0, depth=220.0, count=1, gap=0.0),
    )
    # Rotated footprint is 8x16: center (100, 100) is fine.
    proj.print_layout.pieces.append(_piece("a", 100.0, 100.0, flipped=False))
    proj.print_layout.pieces[0].rotated90 = True
    # Rotated depth 16: center y=213 -> y max 221 > 220 -> overhang.
    overhanging = _piece("a", 100.0, 213.0)
    overhanging.rotated90 = True
    proj.print_layout.pieces.append(overhanging)

    entries, skipped = collect_bed_entries(proj, 0)
    assert len(entries) == 1
    assert entries[0]["rotated90"] is True
    assert skipped == 1


def test_bed_compound_rotated_bbox_swaps() -> None:
    '''A rotated 4x2 piece occupies (8 x 16) on the bed instead of (16 x 8).'''
    settings = BlockSettings(cells_x=4, cells_y=2)  # outer 16 x 8
    height = settings.total_height

    compound = bed_compound(
        [
            {
                'settings': settings.model_dump(),
                'position_x': 10.0,
                'position_y': 20.0,
                'flipped': False,
                'rotated90': True,
            }
        ]
    )
    bbox = compound.val().BoundingBox()
    assert bbox.xmin == pytest.approx(10.0 - 4.0, abs=1e-3)
    assert bbox.xmax == pytest.approx(10.0 + 4.0, abs=1e-3)
    assert bbox.ymin == pytest.approx(20.0 - 8.0, abs=1e-3)
    assert bbox.ymax == pytest.approx(20.0 + 8.0, abs=1e-3)
    assert bbox.zmin == pytest.approx(0.0, abs=1e-3)
    assert bbox.zmax == pytest.approx(height, abs=1e-3)


def test_bed_compound_flipped_and_rotated_volume_invariant() -> None:
    '''Flip + 90-degree rotation is a rigid motion: same volume and the
    same bbox as the unflipped rotated piece (the footprint stays swapped).'''
    settings = BlockSettings(cells_x=4, cells_y=2)
    entry_base = {
        'settings': settings.model_dump(),
        'position_x': 10.0,
        'position_y': 20.0,
        'flipped': False,
        'rotated90': False,
    }
    rotated = bed_compound([{**entry_base, 'rotated90': True}])
    combo = bed_compound([{**entry_base, 'flipped': True, 'rotated90': True}])

    bbox_rotated = rotated.val().BoundingBox()
    bbox_combo = combo.val().BoundingBox()
    for attr in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        assert getattr(bbox_combo, attr) == pytest.approx(
            getattr(bbox_rotated, attr), abs=1e-3
        )
    assert combo.val().Volume() == pytest.approx(rotated.val().Volume(), rel=1e-6)
