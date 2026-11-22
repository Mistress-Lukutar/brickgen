'''
File:   test_placement_service.py
Brief:  Unit tests for print-placement layout engine.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.16.0
'''

from __future__ import annotations

import random

import pytest

from brickgen.core.project import (
    PRINT_STAGING_BED,
    BlockInstance,
    PrintBedSettings,
    PrintPiece,
    Project,
)
from brickgen.services.placement_service import (
    DISTRIBUTE_BY_COLOR,
    DISTRIBUTE_EVEN,
    auto_place,
    bed_origin,
    block_colors,
    block_footprints,
    clamp_into_zone,
    distribute,
    find_free_spot,
    fits_bed,
    local_position,
    nearest_free_spot,
    piece_dims,
    resolve_manual_moves,
    resolve_piece_placements,
    staging_origin,
    sync_pieces,
    world_position,
    zone_for_world_x,
)

# ------------------------------------------------------------------ helpers


def _make_project(
    *blocks: BlockInstance,
    beds: PrintBedSettings | None = None,
) -> Project:
    '''Build a minimal project with explicit block uids and default geometry.

    Args:
        *blocks: ``BlockInstance`` objects (callers must set ``uid=`` and
            ``quantity=``).
        beds: Optional ``PrintBedSettings``; defaults to factory settings.

    Returns:
        A validated ``Project`` ready for placement tests.
    '''
    if not blocks:
        blocks = (BlockInstance(),)
    project = Project(blocks=list(blocks))
    if beds is not None:
        project.print_layout.beds = beds
    return project


def _assert_bed_invariants(proj: Project, beds: PrintBedSettings) -> None:
    '''Every piece on a real bed stays in bounds and ``gap``-clear of others.

    In-bounds means the piece footprint lies inside
    ``[gap, width-gap] x [gap, depth-gap]``; clearance means every pair on
    the same bed is separated by at least ``gap`` on one axis.

    Args:
        proj: The project whose pieces are checked.
        beds: The project's ``PrintBedSettings``.
    '''
    fp = block_footprints(proj)
    by_bed: dict[int, list[PrintPiece]] = {}
    for piece in proj.print_layout.pieces:
        if piece.bed == PRINT_STAGING_BED:
            continue
        w, d = piece_dims(piece, fp)
        assert piece.position_x - w / 2 >= beds.gap - 1e-6
        assert piece.position_x + w / 2 <= beds.width - beds.gap + 1e-6
        assert piece.position_y - d / 2 >= beds.gap - 1e-6
        assert piece.position_y + d / 2 <= beds.depth - beds.gap + 1e-6
        by_bed.setdefault(piece.bed, []).append(piece)
    for bed_pieces in by_bed.values():
        for i, p1 in enumerate(bed_pieces):
            w1, d1 = piece_dims(p1, fp)
            for p2 in bed_pieces[i + 1 :]:
                w2, d2 = piece_dims(p2, fp)
                separated_x = (
                    abs(p1.position_x - p2.position_x)
                    >= (w1 + w2) / 2 + beds.gap - 1e-6
                )
                separated_y = (
                    abs(p1.position_y - p2.position_y)
                    >= (d1 + d2) / 2 + beds.gap - 1e-6
                )
                assert separated_x or separated_y, (
                    f"pieces {p1.uid} and {p2.uid} overlap or violate the gap"
                )


# ================================================================== zone geometry


def test_bed_origin_exact_values() -> None:
    '''bed_origin returns exact world coords for every bed index.

    With the default W=220, GAP=20 (BED_GAP_MM):
      step = 240 mm per zone.
    '''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=3, gap=4.0)

    assert bed_origin(0, beds) == (0.0, 0.0)
    assert bed_origin(1, beds) == (240.0, 0.0)
    assert bed_origin(2, beds) == (480.0, 0.0)


def test_staging_origin_exact_value() -> None:
    '''staging_origin is to the left of bed 0 by one full step.

    W=220 → step = 240 → staging at (-240.0, 0.0).
    '''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=3, gap=4.0)
    assert staging_origin(beds) == (-240.0, 0.0)


def test_zone_for_world_x() -> None:
    '''zone_for_world_x maps world X to the correct zone index.

    Uses W=225 so that bed 0 spans [0, 225) and step = BED_GAP_MM + W = 245 mm.
    The global boundary is -(BED_GAP_MM/2) = -10.0.
    '''
    beds = PrintBedSettings(width=225.0, depth=220.0, count=2, gap=20.0)

    # Staging zone (x < -10).
    assert zone_for_world_x(-241, beds) == PRINT_STAGING_BED

    # Exact boundary: x = -BED_GAP_MM/2 = -10 is NOT strictly less → bed 0.
    assert zone_for_world_x(-10.0, beds) == 0

    # Bed 0 range.
    assert zone_for_world_x(0, beds) == 0

    # Edge between beds: (230+10)//245 = 0 → bed 0; (235+10)//245 = 1 → bed 1.
    assert zone_for_world_x(230, beds) == 0
    assert zone_for_world_x(235, beds) == 1

    # Beyond last bed clamped to count-1.
    assert zone_for_world_x(9999, beds) == 1


def test_zone_for_world_x_staging_boundary() -> None:
    '''With BED_GAP_MM=20 the boundary is always -10; x=-11 → staging.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    assert zone_for_world_x(-11, beds) == PRINT_STAGING_BED


def test_world_position_and_local_position_round_trip() -> None:
    '''world_position and local_position are inverses for bed and staging pieces.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)

    # Bed piece: zone-local (10, 20).
    BlockInstance(uid="b1", cells_x=1, cells_y=1)
    local_wp = world_position(
        type('P', (), {'bed': 0, 'position_x': 10.0, 'position_y': 20.0})(), beds
    )
    assert local_wp == (10.0, 20.0)

    # Staging piece: zone-local (5, 15).
    local_sp = world_position(
        type(
            'P',
            (),
            {
                'bed': PRINT_STAGING_BED,
                'position_x': 5.0,
                'position_y': 15.0,
            },
        )(),
        beds,
    )
    staging_ox, _ = staging_origin(beds)
    assert local_sp == (staging_ox + 5.0, 15.0)

    # Inverse: local_position for bed piece.
    lp_bed = local_position(10.0, 20.0, 0, beds)
    assert lp_bed == (10.0, 20.0)

    # Inverse: local_position for staging piece.
    lp_stg = local_position(staging_ox + 5.0, 15.0, PRINT_STAGING_BED, beds)
    assert lp_stg == (5.0, 15.0)


# ================================================================== block lookups


def test_block_footprints() -> None:
    '''block_footprints returns correct outer dimensions for every block.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=1),
        BlockInstance(uid="b", cells_x=4, cells_y=2, quantity=1),
    )
    fps = block_footprints(proj)
    # unit_size=4.0 → 2x2 = 8x8, 4x2 = 16x8.
    assert fps["a"] == (8.0, 8.0)
    assert fps["b"] == (16.0, 8.0)


def test_block_colors() -> None:
    '''block_colors maps every uid to its hex color.'''
    proj = _make_project(
        BlockInstance(uid="red", cells_x=1, cells_y=1, color="#ff0000", quantity=1),
        BlockInstance(uid="blue", cells_x=1, cells_y=1, color="#0000ff", quantity=1),
    )
    cols = block_colors(proj)
    assert cols["red"] == "#ff0000"
    assert cols["blue"] == "#0000ff"


# ================================================================== fit / collision


def test_fits_bed_exact_fit() -> None:
    '''Exact fit returns True; 0.1 mm over returns False.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, gap=4.0)
    # Effective space: 220 - 8 = 212.
    assert fits_bed(212.0, 212.0, beds) is True
    assert fits_bed(212.1, 212.0, beds) is False


def test_fits_bed_small() -> None:
    '''A small piece always fits.'''
    beds = PrintBedSettings(width=20.0, depth=20.0, gap=4.0)
    assert fits_bed(8.0, 8.0, beds) is True


def test_find_free_spot_empty_bed() -> None:
    '''First free spot on an empty bed is at the corner margin.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, gap=4.0)
    spot = find_free_spot([], 8.0, 8.0, beds)
    assert spot == (8.0, 8.0)


def test_find_free_spot_second_sequential() -> None:
    '''Two sequential finds on a bed with gap=4: second spot x=20.0.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, gap=4.0)
    # First piece at (8, 8).
    placed: list[tuple[float, float, float, float]] = [(8.0, 8.0, 8.0, 8.0)]
    spot = find_free_spot(placed, 8.0, 8.0, beds)
    # xs = {8} ∪ {8+4+4} = {8, 16}; ys = {8, 16}.
    # (8,8) overlaps; (8,16) overlaps; (16,8): abs(8)=8 < 12 ✓ → overlaps.
    # (16,16): abs(8)=8<12 and abs(8)=8<12 → overlaps.
    # (20,8): abs(12)=12 < 12? No → free spot at (20, 8).
    assert spot == (20.0, 8.0)


def test_find_free_spot_no_space() -> None:
    '''When no position fits, returns None.'''
    beds = PrintBedSettings(width=20.0, depth=20.0, gap=4.0)
    # Fill bed with overlapping pieces at every candidate.
    placed = [(8.0, 8.0, 8.0, 8.0)]
    spot = find_free_spot(placed, 8.0, 8.0, beds)
    assert spot is None


# ================================================================== sync_pieces


def test_sync_pieces_new_block_creates_two() -> None:
    '''New project, 1 block (uid "a", 2x2, qty 2) → 2 pieces on bed 0.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=2),
    )
    report = sync_pieces(proj)
    assert report.added == 2
    assert report.removed == 0
    assert len(proj.print_layout.pieces) == 2
    # First piece at corner margin.
    p0 = proj.print_layout.pieces[0]
    assert p0.bed == 0
    assert p0.position_x == pytest.approx(8.0)
    assert p0.position_y == pytest.approx(8.0)


def test_sync_pieces_increase_qty() -> None:
    '''qty 2→5 → added=3, kept=2.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=2),
    )
    sync_pieces(proj)  # creates 2 pieces.
    proj.blocks[0].quantity = 5
    report = sync_pieces(proj)
    assert report.added == 3
    assert report.kept == 2
    assert len(proj.print_layout.pieces) == 5


def test_sync_pieces_decrease_qty() -> None:
    '''qty 5→1 → 1 piece kept, removed=4 net.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=5),
    )
    sync_pieces(proj)  # creates 5 pieces.
    proj.blocks[0].quantity = 1
    report = sync_pieces(proj)
    assert report.removed == 4
    assert len(proj.print_layout.pieces) == 1
    assert proj.print_layout.pieces[0].block_uid == "a"


def test_sync_pieces_block_removed() -> None:
    '''Block removed from project → its pieces dropped.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=3),
    )
    sync_pieces(proj)
    assert len(proj.print_layout.pieces) == 3

    proj.blocks.clear()
    report = sync_pieces(proj)
    assert report.removed == 3
    assert len(proj.print_layout.pieces) == 0


def test_sync_pieces_second_block_appended() -> None:
    '''Adding a second block → its pieces appended.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=1),
    )
    sync_pieces(proj)
    assert len(proj.print_layout.pieces) == 1

    proj.blocks.append(BlockInstance(uid="b", cells_x=2, cells_y=2, quantity=1))
    report = sync_pieces(proj)
    assert report.added == 1
    assert len(proj.print_layout.pieces) == 2
    # Second block's piece appended at end.
    assert proj.print_layout.pieces[1].block_uid == "b"


def test_sync_pieces_staging_fallback() -> None:
    '''Tiny bed with qty 3 → 1 on bed 0, 2 staged.'''
    beds = PrintBedSettings(width=20.0, depth=20.0, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=3),
        beds=beds,
    )
    report = sync_pieces(proj)
    assert report.added == 3
    assert report.staged == 2

    bed0_pieces = [p for p in proj.print_layout.pieces if p.bed == 0]
    staged_pieces = [p for p in proj.print_layout.pieces if p.bed == PRINT_STAGING_BED]
    assert len(bed0_pieces) == 1
    assert len(staged_pieces) == 2


def test_sync_pieces_print_layout_valid() -> None:
    '''PrintLayout validator accepts staging pieces.'''
    beds = PrintBedSettings(width=20.0, depth=20.0, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=3),
        beds=beds,
    )
    sync_pieces(proj)
    # Re-validate — should not raise.
    Project.model_validate(proj.model_dump())


def test_sync_pieces_different_footprints_no_overlap() -> None:
    '''FIX-1 regression: existing pieces use their own footprint in find_free_spot.

    Two blocks with DIFFERENT footprints: a 4x2 (16x8) qty 1 and a 2x2 (8x8) qty 1.
    After sync, the big piece sits at (12, 8). The small piece must find a spot
    beside it using the big piece's real 16x8 footprint — not its own 8x8.

    Candidate xs = {4+4} U {12 + 16/2 + 4 + 8/2} = {8, 12+8+4+4} = {8, 28}.
    At (8,8): overlaps big piece centred at (12,8) with w=16,d=8, gap=4:
      abs(8-12)=4 < (8+16)/2 + 4 = 16 → True. Overlaps. Skip.
    At (8,16): abs(8-12)=4<16 and abs(16-8)=8<(8+8)/2+4=12 → overlaps. Skip.
    At (28,8): abs(28-12)=16 < 16? No → no overlap. Free spot at (28, 8).
    '''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="big", cells_x=4, cells_y=2, quantity=1),  # 16x8
        BlockInstance(uid="small", cells_x=2, cells_y=2, quantity=1),  # 8x8
        beds=beds,
    )
    sync_pieces(proj)

    assert len(proj.print_layout.pieces) == 2
    big_p = next(p for p in proj.print_layout.pieces if p.block_uid == "big")
    small_p = next(p for p in proj.print_layout.pieces if p.block_uid == "small")

    # Big piece at corner (16x8 → center (12, 8)).
    assert big_p.bed == 0
    assert big_p.position_x == pytest.approx(12.0)
    assert big_p.position_y == pytest.approx(8.0)

    # Small piece beside big one at (28, 8).
    assert small_p.bed == 0
    assert small_p.position_x == pytest.approx(28.0)
    assert small_p.position_y == pytest.approx(8.0)


def test_distribute_by_color_unknown_uid_staged() -> None:
    '''FIX-3: piece with unknown block_uid goes to staging in by_color mode.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="known", cells_x=2, cells_y=2, color="#ff0000", quantity=1),
        beds=beds,
    )
    # One piece for the known block, one with an unknown block_uid.
    known_piece = PrintPiece(block_uid="known")
    unknown_piece = PrintPiece(block_uid="unknown_block_999")
    proj.print_layout.pieces.extend([known_piece, unknown_piece])
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0])

    assert report.placed == 1
    assert known_piece.bed == 0
    # The unknown-uid piece must be staged.
    assert unknown_piece.bed == PRINT_STAGING_BED
    assert report.staged == 1


# ================================================================== auto_place


def test_auto_place_positions_and_counts() -> None:
    '''auto_place repacks selected beds: counts, margins and no overlaps.

    2 beds; block 4x2 (16x8) qty 1, block 2x2 (8x8) qty 2 — all three fit
    bed 0 (MaxRects places them centered on it), bed 1 stays empty.
    '''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=4, cells_y=2, quantity=1),
        BlockInstance(uid="B", cells_x=2, cells_y=2, quantity=2),
        beds=beds,
    )
    sync_pieces(proj)
    report = auto_place(proj, [0, 1])

    assert report.placed == 3
    assert report.staged == 0
    assert report.per_bed[0] == 3
    assert report.per_bed.get(1, 0) == 0  # bed 1 has no pieces (all fit on bed 0)

    # Every piece landed on bed 0, inside the margins, gap-clear.
    assert all(p.bed == 0 for p in proj.print_layout.pieces)
    _assert_bed_invariants(proj, beds)

    # The three-piece cluster is centered on the bed.
    fp = block_footprints(proj)
    min_x = min(
        p.position_x - piece_dims(p, fp)[0] / 2 for p in proj.print_layout.pieces
    )
    max_x = max(
        p.position_x + piece_dims(p, fp)[0] / 2 for p in proj.print_layout.pieces
    )
    min_y = min(
        p.position_y - piece_dims(p, fp)[1] / 2 for p in proj.print_layout.pieces
    )
    max_y = max(
        p.position_y + piece_dims(p, fp)[1] / 2 for p in proj.print_layout.pieces
    )
    assert (min_x + max_x) / 2 == pytest.approx(beds.width / 2)
    assert (min_y + max_y) / 2 == pytest.approx(beds.depth / 2)


def test_auto_place_non_selected_bed_untouched() -> None:
    '''Piece on a non-selected bed stays untouched.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, quantity=1),
        beds=beds,
    )
    sync_pieces(proj)

    # Manually place one piece on bed 1.
    proj.print_layout.pieces[0].bed = 1
    proj.print_layout.pieces[0].position_x = 50.0
    proj.print_layout.pieces[0].position_y = 60.0

    report = auto_place(proj, [0])
    # Pool is empty (no pieces on bed 0 or staging), so nothing moves.
    assert report.placed == 0
    p0 = proj.print_layout.pieces[0]
    assert p0.bed == 1
    assert p0.position_x == pytest.approx(50.0)
    assert p0.position_y == pytest.approx(60.0)


def test_auto_place_overflow() -> None:
    '''Overflow: tiny bed with qty 5 → staged pieces have bed=-1.'''
    beds = PrintBedSettings(width=20.0, depth=20.0, count=2, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, quantity=5),
        beds=beds,
    )
    sync_pieces(proj)  # 1 on bed 0, 4 staged.
    report = auto_place(proj, [0, 1])

    assert report.staged > 0
    staged = [p for p in proj.print_layout.pieces if p.bed == PRINT_STAGING_BED]
    assert len(staged) > 0
    # All staged pieces must have bed == -1.
    for p in staged:
        assert p.bed == PRINT_STAGING_BED


def test_auto_place_oversize_staged() -> None:
    '''Oversize piece (16x16 in 20x20 bed with gap 4) goes straight to staging.'''
    beds = PrintBedSettings(width=20.0, depth=20.0, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=4, cells_y=4, quantity=1),  # 16x16
        beds=beds,
    )
    sync_pieces(proj)
    report = auto_place(proj, [0])

    assert report.staged == 1
    assert proj.print_layout.pieces[0].bed == PRINT_STAGING_BED


def test_auto_place_empty_selection() -> None:
    '''Empty bed_indices → zero report, no change.'''
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, quantity=1),
    )
    sync_pieces(proj)
    report = auto_place(proj, [])
    assert report.placed == 0
    assert report.staged == 0


# ================================================================== distribute even


def test_distribute_even_exact_assignment() -> None:
    '''Distribute even on 2 beds with red qty 4 (8x8) and blue qty 1 (16x8).

    LPT by area DESC: [blue(128), red, red, red, red].
    bed0 gets blue + one red; bed1 gets three reds.
    '''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="red", cells_x=2, cells_y=2, color="#ff0000", quantity=4),
        BlockInstance(uid="blue", cells_x=4, cells_y=2, color="#0000ff", quantity=1),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_EVEN, [0, 1])

    assert report.placed == 5
    assert report.staged == 0
    # LPT: blue→bed0(128), red1→bed1(64), red2→bed1(128), red3→bed0(192), red4→bed1(192)
    assert report.per_bed[0] == 2
    assert report.per_bed[1] == 3

    # Count pieces per bed by uid.
    red_on_0 = sum(
        1 for p in proj.print_layout.pieces if p.block_uid == "red" and p.bed == 0
    )
    red_on_1 = sum(
        1 for p in proj.print_layout.pieces if p.block_uid == "red" and p.bed == 1
    )
    blue_on_0 = sum(
        1 for p in proj.print_layout.pieces if p.block_uid == "blue" and p.bed == 0
    )

    assert red_on_0 == 1
    assert red_on_1 == 3
    assert blue_on_0 == 1

    # All positions within bed margins.
    for p in proj.print_layout.pieces:
        assert p.position_x >= beds.gap + 4
        assert p.position_x <= beds.width - beds.gap - 4


def test_distribute_even_both_beds_used() -> None:
    '''Both beds receive pieces (not all on one bed).'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, quantity=6),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_EVEN, [0, 1])
    assert report.per_bed[0] > 0
    assert report.per_bed[1] > 0


# ================================================================== distribute by_color


def test_distribute_by_color_groups_on_separate_beds() -> None:
    '''3 beds, blue group (6 pieces 8x8), red group (2 pieces 8x8).

    One print job per color: blue packs bed 0 (3 fit) and spills onto bed 1
    exclusively (3 more); red takes bed 2.  Nothing staged.
    '''
    beds = PrintBedSettings(width=40.0, depth=20.0, count=3, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="red", cells_x=2, cells_y=2, color="#ff0000", quantity=2),
        BlockInstance(uid="blue", cells_x=2, cells_y=2, color="#0000ff", quantity=6),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0, 1, 2])

    assert report.placed == 8
    assert report.staged == 0
    # Shelf pack on W=40,D=20,GAP=4 fits 3 per bed: blue = beds 0 + 1.
    blue_beds = [p.bed for p in proj.print_layout.pieces if p.block_uid == "blue"]
    assert blue_beds.count(0) == 3
    assert blue_beds.count(1) == 3
    # Red owns bed 2 exclusively.
    red_beds = [p.bed for p in proj.print_layout.pieces if p.block_uid == "red"]
    assert red_beds == [2, 2]


def test_distribute_by_color_never_mixes_beds() -> None:
    '''Two colors never share a bed even when both would fit together.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="red", cells_x=2, cells_y=2, color="#ff0000", quantity=1),
        BlockInstance(uid="blue", cells_x=2, cells_y=2, color="#0000ff", quantity=1),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0, 1])

    assert report.placed == 2
    assert report.staged == 0
    bed_red = next(p.bed for p in proj.print_layout.pieces if p.block_uid == "red")
    bed_blue = next(p.bed for p in proj.print_layout.pieces if p.block_uid == "blue")
    assert bed_red != bed_blue


def test_distribute_by_color_spill_to_second_bed() -> None:
    '''Single color with more pieces than one bed holds → second bed gets spill.'''
    beds = PrintBedSettings(width=40.0, depth=20.0, count=2, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, color="#ff0000", quantity=7),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0, 1])

    # Shelf pack fits 3 per bed: bed0 + bed1 hold 6, the 7th is staged.
    assert report.placed == 6
    assert report.staged == 1


def test_distribute_by_color_exact_fit() -> None:
    '''6 pieces of 8x8 exactly fill one bed (capacity 384).

    With W=40,D=20,GAP=4: shelf pack fits only 3 per bed.
    Use larger beds for full placement.
    '''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, color="#ff0000", quantity=6),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0])

    # All 6 fit on one large bed.
    assert report.placed == 6
    assert report.staged == 0


def test_distribute_by_color_spill_to_staging() -> None:
    '''When remaining beds cannot hold the spill piece, it goes to staging.'''
    beds = PrintBedSettings(width=40.0, depth=20.0, count=1, gap=4.0)
    # 7 pieces with capacity 384 per bed (each 64 area).
    # All assigned to bed0 (area 448 > 384? No, piece-by-piece: 6 fill, 7th overflows).
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, color="#ff0000", quantity=7),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0])

    # 6 assigned to bed0 (capacity 384), 7th has no more beds → staging.
    assert report.staged >= 1


def test_distribute_by_color_single_bed_overflow() -> None:
    '''Single bed can only hold 6 of 7 pieces; 7th goes to staging.'''
    beds = PrintBedSettings(width=40.0, depth=20.0, count=1, gap=4.0)
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, color="#ff0000", quantity=7),
        beds=beds,
    )
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0])

    # 6 pieces assigned to bed0 (area 384), 7th piece overflows.
    assert report.staged >= 1


# ================================================================== unknown mode


def test_distribute_unknown_mode_raises() -> None:
    '''Unknown distribute mode raises ValueError.'''
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, quantity=1),
    )
    sync_pieces(proj)
    with pytest.raises(ValueError, match="unknown distribute mode"):
        distribute(proj, "invalid_mode", [0])


# ======================================================== validation round-trip


def test_sync_preserves_print_layout_validity() -> None:
    '''After sync_pieces the project still validates.'''
    proj = _make_project(
        BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=3),
    )
    sync_pieces(proj)
    Project.model_validate(proj.model_dump())


def test_auto_place_preserves_print_layout_validity() -> None:
    '''After auto_place the project still validates.'''
    proj = _make_project(
        BlockInstance(uid="A", cells_x=2, cells_y=2, quantity=3),
    )
    sync_pieces(proj)
    auto_place(proj, [0])
    Project.model_validate(proj.model_dump())


def test_distribute_preserves_print_layout_validity() -> None:
    '''After distribute the project still validates.'''
    proj = _make_project(
        BlockInstance(uid="red", cells_x=2, cells_y=2, color="#ff0000", quantity=4),
        BlockInstance(uid="blue", cells_x=4, cells_y=2, color="#0000ff", quantity=1),
    )
    sync_pieces(proj)
    distribute(proj, DISTRIBUTE_EVEN, [0, 1])
    Project.model_validate(proj.model_dump())


# ================================================================== 90-degree rotation


def test_piece_dims_swaps_for_rotated() -> None:
    '''piece_dims swaps (w, d) when the piece is rotated 90 degrees.'''
    proj = _make_project(BlockInstance(uid="a", cells_x=4, cells_y=2))
    fp = block_footprints(proj)
    assert fp["a"] == (16.0, 8.0)

    straight = PrintPiece(block_uid="a")
    assert piece_dims(straight, fp) == (16.0, 8.0)
    rotated = PrintPiece(block_uid="a", rotated90=True)
    assert piece_dims(rotated, fp) == (8.0, 16.0)


def test_auto_place_rotates_to_fit() -> None:
    '''A piece that only fits turned 90 degrees is placed rotated.'''
    # A 4x2 block (16x8) does not fit a 12x24 bed unrotated (16 > 12),
    # but the rotated footprint (8x16) fits.
    beds = PrintBedSettings(width=12.0, depth=24.0, count=1, gap=0.0)
    proj = _make_project(
        BlockInstance(uid="a", cells_x=4, cells_y=2, quantity=1), beds=beds
    )
    sync_pieces(proj)

    report = auto_place(proj, [0])

    assert report.placed == 1
    piece = proj.print_layout.pieces[0]
    assert piece.bed == 0
    assert piece.rotated90 is True
    # Single centered piece on the 12x24 bed (gap 0).
    assert piece.position_x == pytest.approx(6.0)
    assert piece.position_y == pytest.approx(12.0)


def test_auto_place_prefers_unrotated() -> None:
    '''A piece that fits unrotated is NOT turned by the packing.'''
    proj = _make_project(BlockInstance(uid="a", cells_x=4, cells_y=2, quantity=1))
    sync_pieces(proj)
    auto_place(proj, [0])
    piece = proj.print_layout.pieces[0]
    assert piece.rotated90 is False
    assert piece.position_x == pytest.approx(110.0)  # centered on the 220 mm bed


def test_auto_place_rotation_nonoverlap() -> None:
    '''Pieces packed with rotations never overlap their actual footprints.'''
    # Two elongated blocks on a tight bed force a rotated placement.
    beds = PrintBedSettings(width=30.0, depth=30.0, count=1, gap=2.0)
    proj = _make_project(
        BlockInstance(uid="a", cells_x=4, cells_y=2, quantity=2), beds=beds
    )
    sync_pieces(proj)
    auto_place(proj, [0])

    pieces = proj.print_layout.pieces
    assert all(p.bed == 0 for p in pieces)
    fp = block_footprints(proj)
    placed = [piece_dims(p, fp) for p in pieces]
    # No pair overlaps (axis-separated AABB with the gap clearance).
    (p1, p2) = pieces
    (w1, d1), (w2, d2) = placed
    assert abs(p1.position_x - p2.position_x) >= (w1 + w2) / 2 or (
        abs(p1.position_y - p2.position_y) >= (d1 + d2) / 2
    )


def test_sync_respects_rotated_existing_pieces() -> None:
    '''Free-spot scanning in sync uses the rotated footprint of existing pieces.'''
    beds = PrintBedSettings(width=40.0, depth=40.0, count=1, gap=0.0)
    proj = _make_project(
        BlockInstance(uid="a", cells_x=4, cells_y=2, quantity=2), beds=beds
    )
    # First piece already placed ROTATED (8 wide, 16 deep) at (4, 8).
    proj.print_layout.pieces.append(
        PrintPiece(block_uid="a", bed=0, position_x=4.0, position_y=8.0, rotated90=True)
    )
    sync_pieces(proj)

    pieces = proj.print_layout.pieces
    assert len(pieces) == 2
    fp = block_footprints(proj)
    w1, d1 = piece_dims(pieces[0], fp)
    w2, d2 = piece_dims(pieces[1], fp)
    # The new piece must not overlap the rotated one (touching is OK).
    assert abs(pieces[1].position_x - pieces[0].position_x) >= (w1 + w2) / 2 or (
        abs(pieces[1].position_y - pieces[0].position_y) >= (d1 + d2) / 2
    )


def test_distribute_keeps_rotation_flag_consistent() -> None:
    '''distribute re-packs positions and the rotated90 flags stay valid.'''
    beds = PrintBedSettings(width=12.0, depth=24.0, count=2, gap=0.0)
    proj = _make_project(
        BlockInstance(uid="a", cells_x=4, cells_y=2, color="#ff0000", quantity=2),
        beds=beds,
    )
    sync_pieces(proj)
    distribute(proj, DISTRIBUTE_EVEN, [0, 1])
    fp = block_footprints(proj)
    for piece in proj.print_layout.pieces:
        w, d = piece_dims(piece, fp)
        # Every piece sits inside its bed in its chosen orientation.
        assert piece.bed in (0, 1)
        assert piece.position_x - w / 2 >= 0
        assert piece.position_x + w / 2 <= beds.width
        assert piece.position_y - d / 2 >= 0
        assert piece.position_y + d / 2 <= beds.depth


# ================================================================== MaxRects packing


def test_auto_place_reuses_holes_beyond_shelf() -> None:
    '''MaxRects fills the strips around a big block that shelf packing wasted.

    Bed 36x36, one 28x28 block, seven 8x8 blocks: a row-wise pack fits only
    4 squares (one row below, none beside — the row is full), while MaxRects
    packs all seven (3 in the top strip + 4 stacked in the right column).
    '''
    beds = PrintBedSettings(width=36.0, depth=36.0, count=1, gap=0.0)
    proj = _make_project(
        BlockInstance(uid="big", cells_x=7, cells_y=7, quantity=1),
        BlockInstance(uid="small", cells_x=2, cells_y=2, quantity=7),
        beds=beds,
    )
    sync_pieces(proj)
    report = auto_place(proj, [0])

    assert report.placed == 8
    assert report.staged == 0
    _assert_bed_invariants(proj, beds)


def test_auto_place_stress_no_overlap_in_bounds() -> None:
    '''Seeded mixed blocks: every placed piece stays in bounds and gap-clear.'''
    rng = random.Random(42)
    blocks = [
        BlockInstance(
            uid=f"b{i}",
            cells_x=rng.randint(1, 8),
            cells_y=rng.randint(1, 8),
            color="#800000",
            quantity=rng.randint(2, 20),
        )
        for i in range(12)
    ]
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(*blocks, beds=beds)
    sync_pieces(proj)
    report = auto_place(proj, [0, 1])

    assert report.placed + report.staged == sum(b.quantity for b in blocks)
    _assert_bed_invariants(proj, beds)


def test_distribute_even_stress_invariants() -> None:
    '''Distribute even: placed pieces never overlap or leave the beds.'''
    rng = random.Random(7)
    blocks = [
        BlockInstance(
            uid=f"e{i}",
            cells_x=rng.randint(1, 6),
            cells_y=rng.randint(1, 6),
            color="#008000",
            quantity=rng.randint(2, 12),
        )
        for i in range(10)
    ]
    beds = PrintBedSettings(width=220.0, depth=220.0, count=3, gap=4.0)
    proj = _make_project(*blocks, beds=beds)
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_EVEN, [0, 1, 2])

    assert report.placed + report.staged == sum(b.quantity for b in blocks)
    _assert_bed_invariants(proj, beds)


def test_distribute_by_color_stress_invariants() -> None:
    '''Distribute by color: placed pieces never overlap or leave the beds.'''
    rng = random.Random(11)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    blocks = [
        BlockInstance(
            uid=f"c{i}",
            cells_x=rng.randint(1, 6),
            cells_y=rng.randint(1, 6),
            color=colors[i % len(colors)],
            quantity=rng.randint(2, 12),
        )
        for i in range(10)
    ]
    beds = PrintBedSettings(width=220.0, depth=220.0, count=4, gap=4.0)
    proj = _make_project(*blocks, beds=beds)
    sync_pieces(proj)
    report = distribute(proj, DISTRIBUTE_BY_COLOR, [0, 1, 2, 3])

    assert report.placed + report.staged == sum(b.quantity for b in blocks)
    _assert_bed_invariants(proj, beds)

    # One print job per color: no bed holds two colors.
    colors_map = block_colors(proj)
    bed_colors: dict[int, set[str]] = {}
    for piece in proj.print_layout.pieces:
        if piece.bed != PRINT_STAGING_BED:
            bed_colors.setdefault(piece.bed, set()).add(colors_map[piece.block_uid])
    assert all(len(cs) == 1 for cs in bed_colors.values())


# ========================================================= manual move resolution


def test_clamp_into_zone_keeps_footprint_inside() -> None:
    '''clamp_into_zone pulls the piece center into the zone margins.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    proj = _make_project(BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=1))
    sync_pieces(proj)
    piece = proj.print_layout.pieces[0]
    fp = block_footprints(proj)

    piece.position_x, piece.position_y = -5.0, 500.0
    clamp_into_zone(piece, fp, beds)
    assert piece.position_x == pytest.approx(8.0)  # gap 4 + 8/2
    assert piece.position_y == pytest.approx(212.0)  # 220 - 4 - 8/2


def test_nearest_free_spot_shifts_to_closest_side() -> None:
    '''A request inside an occupant resolves to the closest free center.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    placed = [(60.0, 60.0, 16.0, 8.0)]
    spot = nearest_free_spot(placed, 8.0, 8.0, beds, 60.0, 60.0)
    # Free centers 12 mm away: (60, 48) below and (60, 72) above; the tie
    # breaks to the lower y.
    assert spot == (60.0, 48.0)


def test_nearest_free_spot_full_zone_returns_none() -> None:
    '''No conflict-free candidate in a packed zone → None.'''
    beds = PrintBedSettings(width=40.0, depth=20.0, count=1, gap=4.0)
    placed = [(20.0, 8.0, 32.0, 8.0), (20.0, 16.0, 32.0, 8.0)]
    assert nearest_free_spot(placed, 8.0, 8.0, beds, 20.0, 10.0) is None


def test_resolve_manual_moves_keeps_valid_drop() -> None:
    '''A drop in free space keeps its exact position and zone.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=2))
    sync_pieces(proj)
    resolve_manual_moves(
        proj.print_layout.pieces, block_footprints(proj), beds, {1: (100.0, 100.0)}
    )
    piece = proj.print_layout.pieces[1]
    assert piece.bed == 0
    assert piece.position_x == pytest.approx(100.0)
    assert piece.position_y == pytest.approx(100.0)


def test_resolve_manual_moves_repairs_overlap() -> None:
    '''A drop onto another piece shifts to the nearest conflict-free spot.'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=1, gap=4.0)
    proj = _make_project(BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=2))
    sync_pieces(proj)
    # Piece 0 sits at (8, 8); drop piece 1 right on top of it.
    resolve_manual_moves(
        proj.print_layout.pieces, block_footprints(proj), beds, {1: (8.0, 8.0)}
    )
    p0, p1 = proj.print_layout.pieces
    assert p1.bed == 0
    assert (p1.position_x, p1.position_y) == pytest.approx((20.0, 8.0))
    _assert_bed_invariants(proj, beds)
    # Piece 0 untouched.
    assert (p0.position_x, p0.position_y) == pytest.approx((8.0, 8.0))


def test_resolve_manual_moves_stages_when_zone_full() -> None:
    '''A drop into a packed zone stages the piece at a free staging spot.'''
    beds = PrintBedSettings(width=20.0, depth=20.0, count=1, gap=4.0)
    proj = _make_project(BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=2))
    sync_pieces(proj)
    # Only one 8x8 piece fits the 20x20 bed: piece 0 on the bed, piece 1
    # staged by sync.  Force piece 1 onto the (full) bed.
    resolve_manual_moves(
        proj.print_layout.pieces, block_footprints(proj), beds, {1: (12.0, 12.0)}
    )
    p1 = proj.print_layout.pieces[1]
    assert p1.bed == PRINT_STAGING_BED
    # Staged at the free staging spot, not on top of bed pieces.
    assert p1.position_x == pytest.approx(8.0)
    assert p1.position_y == pytest.approx(8.0)


def test_resolve_piece_placements_repairs_bed_change_overlap() -> None:
    '''resolve_piece_placements repairs pre-positioned pieces (bed change).'''
    beds = PrintBedSettings(width=220.0, depth=220.0, count=2, gap=4.0)
    proj = _make_project(BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=2))
    sync_pieces(proj)
    p0, p1 = proj.print_layout.pieces
    # Simulate a bed change that lands piece 1 on top of piece 0 on bed 0.
    p1.bed = 0
    p1.position_x, p1.position_y = p0.position_x, p0.position_y
    resolve_piece_placements(
        proj.print_layout.pieces, block_footprints(proj), beds, [p1]
    )
    _assert_bed_invariants(proj, beds)
    assert p1.bed == 0
