'''
File:   test_assemble_service.py
Brief:  Unit tests for the pure Assemble-tab service (groups, counts, snapping).
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.21.0
'''

from __future__ import annotations

import pytest

from brickgen.core.project import (
    AssembledPiece,
    BlockInstance,
    Project,
)
from brickgen.services.assemble_service import (
    Support,
    assemble_groups,
    assemble_snap_target,
    assemble_summary,
    assemble_supports,
    placed_counts,
    support_top_z,
)


def test_assemble_groups_merges_same_geometry() -> None:
    '''Blocks differing only in color/quantity/position share one group.'''
    project = Project(
        blocks=[
            BlockInstance(uid="a", cells_x=2, cells_y=2, color="#ff0000", quantity=3),
            BlockInstance(uid="b", cells_x=2, cells_y=2, color="#00ff00", quantity=5),
            BlockInstance(uid="c", cells_x=3, cells_y=1, color="#ff0000", quantity=1),
        ]
    )
    groups = assemble_groups(project)
    assert len(groups) == 2
    first, second = groups
    assert [variant.uid for variant in first.variants] == ["a", "b"]
    assert [variant.color for variant in first.variants] == ["#ff0000", "#00ff00"]
    assert [variant.uid for variant in second.variants] == ["c"]
    # The name comes from the representative recipe.
    assert first.name == second.name or first.name.startswith("2x2")
    assert "3x1" in second.name


def test_assemble_groups_distinct_layouts() -> None:
    '''Same grid size but a different per-cell layout splits groups.'''
    project = Project(
        blocks=[
            BlockInstance(uid="a", cells_x=2, cells_y=2),
            BlockInstance(uid="b", cells_x=2, cells_y=2, excluded_cells=[[0, 0]]),
        ]
    )
    assert len(assemble_groups(project)) == 2


def test_placed_counts() -> None:
    '''placed_counts tallies copies per referenced block uid.'''
    pieces = [
        AssembledPiece(block_uid="a"),
        AssembledPiece(block_uid="a"),
        AssembledPiece(block_uid="b"),
    ]
    assert placed_counts(pieces) == {"a": 2, "b": 1}
    assert placed_counts([]) == {}


def test_support_top_z_none_and_stack() -> None:
    '''Empty space returns None; overlap returns the highest top; a pure
    edge touch is not an overlap.'''
    supports = [
        Support(x=0.0, y=0.0, z=0.0, hx=10.0, hy=10.0, height=8.0),
        Support(x=30.0, y=0.0, z=8.0, hx=10.0, hy=10.0, height=8.0),
    ]
    assert support_top_z(supports, 100.0, 100.0, 5.0, 5.0) is None
    # Inside the first piece: top 8; the second starts at x=20 (30-10), so
    # a footprint touching x=20 exactly does not count.
    assert support_top_z(supports, 5.0, 0.0, 5.0, 5.0) == pytest.approx(8.0)
    assert support_top_z(supports, 20.0, 0.0, 0.0, 5.0) is None
    # Overlapping both pieces: the higher top wins.
    assert support_top_z(supports, 15.0, 0.0, 6.0, 5.0) == pytest.approx(16.0)


def test_assemble_supports_swap_halves_on_odd_rotation() -> None:
    '''rot_z 1 swaps the footprint halves (90-degree footprint).'''
    project = Project(blocks=[BlockInstance(uid="a", cells_x=4, cells_y=2)])
    project.assemble.pieces = [
        AssembledPiece(block_uid="a", rot_z=0, position_x=1.0, position_y=2.0),
        AssembledPiece(block_uid="a", rot_z=1, position_x=3.0, position_y=4.0),
    ]
    supports = assemble_supports(project)
    unit = project.settings.unit_size
    first, second = supports
    assert first.x == pytest.approx(1.0) and first.y == pytest.approx(2.0)
    assert second.x == pytest.approx(3.0) and second.y == pytest.approx(4.0)
    assert first.hx == pytest.approx(4 * unit / 2)
    assert first.hy == pytest.approx(2 * unit / 2)
    # Rotation swaps the halves.
    assert second.hx == pytest.approx(2 * unit / 2)
    assert second.hy == pytest.approx(4 * unit / 2)


def test_assemble_supports_exclude_and_missing_block() -> None:
    '''Excluded uids and stale block references are skipped.'''
    project = Project(blocks=[BlockInstance(uid="a")])
    project.assemble.pieces = [
        AssembledPiece(uid="p1", block_uid="a"),
        AssembledPiece(uid="p2", block_uid="gone"),
    ]
    assert len(assemble_supports(project)) == 1
    assert assemble_supports(project, {"p1"}) == []


def test_snap_target_grid_only() -> None:
    '''Without part snapping the target is the grid-rounded cursor, z=0.'''
    x, y, z = assemble_snap_target(
        3.0, 7.0, 4.0, 4.0, [], 4.0, snap_to_grid=True, snap_to_parts=False
    )
    assert (x, y, z) == (4.0, 8.0, 0.0)


def test_snap_target_no_snapping() -> None:
    '''All snapping off: the raw cursor point on the floor.'''
    x, y, z = assemble_snap_target(
        3.3, 7.7, 4.0, 4.0, [], 4.0, snap_to_grid=False, snap_to_parts=False
    )
    assert (x, y, z) == (3.3, 7.7, 0.0)


def test_snap_target_stacks_on_overlap() -> None:
    '''A footprint overlapping a piece drops onto its top at the cursor.'''
    supports = [Support(x=0.0, y=0.0, z=0.0, hx=8.0, hy=8.0, height=8.8)]
    x, y, z = assemble_snap_target(
        3.0, 3.0, 4.0, 4.0, supports, 4.0, snap_to_grid=True, snap_to_parts=True
    )
    assert (x, y) == (4.0, 4.0)  # grid-rounded cursor
    assert z == pytest.approx(8.8)  # on top of the support


def test_snap_target_side_flush_wins_when_closer() -> None:
    '''A cursor near a piece side snaps flush to that side at its height.'''
    supports = [Support(x=0.0, y=0.0, z=0.0, hx=8.0, hy=8.0, height=8.8)]
    # Cursor just outside the +X side (side contact needs 8 + 4 = 12).
    x, y, z = assemble_snap_target(
        13.0, 0.5, 4.0, 4.0, supports, 4.0, snap_to_grid=True, snap_to_parts=True
    )
    assert x == pytest.approx(12.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)  # same layer as the neighbor


def test_snap_target_stacking_preferred_on_tie() -> None:
    '''When stacking and side distances tie, stacking wins.'''
    supports = [Support(x=0.0, y=0.0, z=0.0, hx=8.0, hy=8.0, height=8.8)]
    # The grid-rounded cursor (8, 0) overlaps the piece edge strip; the
    # side-flush candidate (12, 0) is the same XY distance away. Equal
    # scores prefer the stacking candidate (higher z, same XY distance).
    x, y, z = assemble_snap_target(
        10.0, 0.0, 4.0, 4.0, supports, 4.0, snap_to_grid=True, snap_to_parts=True
    )
    assert z == pytest.approx(8.8)
    assert (x, y) == (8.0, 0.0)


def test_snap_target_far_from_parts_falls_back_to_grid() -> None:
    '''No nearby pieces: the plain grid-rounded floor position.'''
    supports = [Support(x=0.0, y=0.0, z=0.0, hx=8.0, hy=8.0, height=8.8)]
    x, y, z = assemble_snap_target(
        200.0, 100.0, 4.0, 4.0, supports, 4.0, snap_to_grid=True, snap_to_parts=True
    )
    assert (x, y, z) == (200.0, 100.0, 0.0)


def test_assemble_summary() -> None:
    '''Summary reports placed/planned counts and part types.'''
    project = Project(
        blocks=[
            BlockInstance(uid="a", cells_x=2, cells_y=2, quantity=3),
            BlockInstance(uid="b", cells_x=1, cells_y=1, quantity=2),
        ]
    )
    assert assemble_summary(project) == ("Assembly: 0/5 pieces placed (2 part types)")
    project.assemble.pieces.append(AssembledPiece(block_uid="a"))
    assert assemble_summary(project) == ("Assembly: 1/5 pieces placed (2 part types)")


def test_assemble_summary_empty_scene() -> None:
    '''An empty project reports the planning hint.'''
    project = Project()
    assert "Planning" in assemble_summary(project)
