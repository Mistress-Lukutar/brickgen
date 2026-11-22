'''
File:   test_assemble_service.py
Brief:  Unit tests for the pure Assemble-tab service (groups, counts, snapping).
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

import pytest

from brickgen.core.models import BlockSettings
from brickgen.core.project import (
    AssembledPiece,
    BlockInstance,
    Project,
)
from brickgen.services.assemble_service import (
    Rect,
    Support,
    assemble_groups,
    assemble_snap_target,
    assemble_summary,
    assemble_supports,
    base_footprint_rects,
    placed_counts,
    support_top_z,
    transform_footprint_rects,
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
    '''Empty space returns None; overlap returns the highest stacking
    plane; a pure edge touch is not an overlap.'''
    supports = [
        Support(
            x=0.0,
            y=0.0,
            z=0.0,
            hx=10.0,
            hy=10.0,
            height=8.0,
            stack_top=8.0,
            rects=(Rect(0.0, 0.0, 10.0, 10.0),),
        ),
        Support(
            x=30.0,
            y=0.0,
            z=8.0,
            hx=10.0,
            hy=10.0,
            height=8.0,
            stack_top=16.0,
            rects=(Rect(30.0, 0.0, 10.0, 10.0),),
        ),
    ]
    assert support_top_z(supports, (Rect(100.0, 100.0, 5.0, 5.0),)) is None
    # Inside the first piece: stacking plane 8; the second starts at x=20
    # (30-10), so a footprint touching x=20 exactly does not count.
    assert support_top_z(supports, (Rect(5.0, 0.0, 5.0, 5.0),)) == pytest.approx(8.0)
    assert support_top_z(supports, (Rect(20.0, 0.0, 0.0, 5.0),)) is None
    # Overlapping both pieces: the higher stacking plane wins.
    assert support_top_z(supports, (Rect(15.0, 0.0, 6.0, 5.0),)) == pytest.approx(16.0)


def test_assemble_supports_stack_top_by_flip_state() -> None:
    '''Normal pieces stack at block_height; flipped ones rim-to-rim.'''
    project = Project(blocks=[BlockInstance(uid="a", cells_x=2, cells_y=2)])
    settings = project.blocks[0].to_settings(project.settings)
    project.assemble.pieces = [
        AssembledPiece(uid="p1", block_uid="a"),
        AssembledPiece(uid="p2", block_uid="a", flipped=True, position_z=3.0),
    ]
    normal, flipped = assemble_supports(project)
    assert normal.stack_top == pytest.approx(settings.block_height)
    assert normal.height == pytest.approx(settings.total_height)
    assert flipped.stack_top == pytest.approx(3.0 + settings.total_height)


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


def _t_block_project() -> tuple[Project, BlockSettings]:
    '''Project with one 3x2 block whose cell (2, 0) is excluded (a notch).'''
    project = Project(
        blocks=[BlockInstance(uid="a", cells_x=3, cells_y=2, excluded_cells=[[2, 0]])]
    )
    return project, project.blocks[0].to_settings(project.settings)


def test_base_footprint_rects_full_grid_without_exclusions() -> None:
    '''No excluded cells: one full-grid rectangle.'''
    project = Project(blocks=[BlockInstance(uid="a", cells_x=4, cells_y=2)])
    settings = project.blocks[0].to_settings(project.settings)
    assert base_footprint_rects(settings) == (
        Rect(0.0, 0.0, 2 * settings.unit_size, settings.unit_size),
    )


def test_base_footprint_rects_decompose_exclusions() -> None:
    '''Excluded cells split the footprint into merged material rectangles.'''
    project, settings = _t_block_project()
    unit = settings.unit_size
    rects = base_footprint_rects(settings)
    # Row 0 keeps cells 0-1, row 1 keeps the full width: two rectangles.
    assert len(rects) == 2
    first, second = rects
    assert first.cx == pytest.approx(-unit / 2)
    assert first.hx == pytest.approx(unit)
    assert first.cy == pytest.approx(-unit / 2)
    assert first.hy == pytest.approx(unit / 2)
    assert second.cx == pytest.approx(0.0)
    assert second.hx == pytest.approx(3 * unit / 2)
    assert second.cy == pytest.approx(unit / 2)


def test_transform_footprint_rects_rot_and_flip() -> None:
    '''Quarter turns rotate rectangles; the flip mirrors Y after them.'''
    rects = (Rect(1.0, 2.0, 3.0, 4.0),)
    assert transform_footprint_rects(rects, 1, False) == (Rect(-2.0, 1.0, 4.0, 3.0),)
    assert transform_footprint_rects(rects, 2, False) == (Rect(-1.0, -2.0, 3.0, 4.0),)
    assert transform_footprint_rects(rects, 0, True) == (Rect(1.0, -2.0, 3.0, 4.0),)
    # The flip applies after the rotation (mesh transform order).
    assert transform_footprint_rects(rects, 1, True) == (Rect(-2.0, -1.0, 4.0, 3.0),)


def test_assemble_supports_rects_world_offset_and_rotation() -> None:
    '''Support rects are world rectangles honoring position and rotation.'''
    project, settings = _t_block_project()
    unit = settings.unit_size  # type: ignore[attr-defined]
    project.assemble.pieces = [
        AssembledPiece(
            uid="p", block_uid="a", position_x=10.0, position_y=20.0, rot_z=1
        )
    ]
    support = assemble_supports(project)[0]
    first, second = support.rects
    assert (first.cx, first.cy) == pytest.approx((10 + unit / 2, 20 - unit / 2))
    assert (first.hx, first.hy) == pytest.approx((unit / 2, unit))
    assert (second.cx, second.cy) == pytest.approx((10 - unit / 2, 20.0))
    assert (second.hx, second.hy) == pytest.approx((unit / 2, 3 * unit / 2))
    # The bbox halves follow the rotated material box.
    assert support.hx == pytest.approx(unit)
    assert support.hy == pytest.approx(3 * unit / 2)


def test_support_top_z_drops_into_excluded_void() -> None:
    '''A footprint fully inside an excluded-cell void overlaps nothing.

    Regression: the T-piece notch is not material, so pieces drop into
    the empty space instead of snapping onto the bounding box top.
    '''
    project, settings = _t_block_project()
    unit = settings.unit_size
    project.assemble.pieces = [AssembledPiece(uid="p", block_uid="a")]
    supports = assemble_supports(project)
    # Void cell (2, 0) spans x in [0.5u, 1.5u], y in [-u, 0].
    void = (Rect(unit, -unit / 2, unit / 2, unit / 2),)
    assert support_top_z(supports, void) is None
    # Material overlap still stacks (cell (0, 0) center).
    material = (Rect(-unit, -unit / 2, unit / 2, unit / 2),)
    assert support_top_z(supports, material) == pytest.approx(settings.block_height)


def test_support_top_z_stacks_into_notch_on_base_plate() -> None:
    '''Inside a T-void the piece stacks on the plate under the T-piece.'''
    project = Project(
        blocks=[
            BlockInstance(uid="plate", cells_x=5, cells_y=5),
            BlockInstance(uid="t", cells_x=3, cells_y=2, excluded_cells=[[2, 0]]),
        ]
    )
    unit = project.settings.unit_size
    project.assemble.pieces = [
        AssembledPiece(uid="p1", block_uid="plate"),
        AssembledPiece(uid="p2", block_uid="t", position_z=3.0),
    ]
    supports = assemble_supports(project)
    # The void cell of the T-piece sits above plate material: the plate's
    # stacking plane wins over the T-piece's bounding-box top.
    void = (Rect(unit, -unit / 2, unit / 2, unit / 2),)
    assert support_top_z(supports, void) == pytest.approx(3.0)


def test_snap_target_grid_only() -> None:
    '''Without part snapping the corner-rounded cursor target, z=0.'''
    x, y, z = assemble_snap_target(
        3.0,
        7.0,
        (Rect(0.0, 0.0, 4.0, 4.0),),
        [],
        4.0,
        snap_to_grid=True,
        snap_to_parts=False,
    )
    # Corners at 0/4: the footprint fills one grid cell per axis.
    assert (x, y, z) == (4.0, 8.0, 0.0)


def test_snap_target_no_snapping() -> None:
    '''All snapping off: the raw cursor point on the floor.'''
    x, y, z = assemble_snap_target(
        3.3,
        7.7,
        (Rect(0.0, 0.0, 4.0, 4.0),),
        [],
        4.0,
        snap_to_grid=False,
        snap_to_parts=False,
    )
    assert (x, y, z) == (3.3, 7.7, 0.0)


def _box_support(
    x: float = 0.0, y: float = 0.0, hx: float = 8.0, hy: float = 8.0
) -> Support:
    '''One rectangular support at floor level with stack_top 3.'''
    return Support(
        x=x,
        y=y,
        z=0.0,
        hx=hx,
        hy=hy,
        height=8.8,
        stack_top=3.0,
        rects=(Rect(x, y, hx, hy),),
    )


def test_snap_target_stacks_on_overlap() -> None:
    '''A footprint overlapping a piece drops onto its stacking plane.'''
    supports = [_box_support()]
    x, y, z = assemble_snap_target(
        3.0,
        3.0,
        (Rect(0.0, 0.0, 4.0, 4.0),),
        supports,
        4.0,
        snap_to_grid=True,
        snap_to_parts=True,
    )
    assert (x, y) == (4.0, 4.0)  # corner-rounded cursor
    assert z == pytest.approx(3.0)  # block_height above the support bottom


def test_snap_target_side_flush_wins_when_closer() -> None:
    '''A cursor near a piece side snaps flush to that side at its height.'''
    supports = [_box_support()]
    # Cursor just outside the +X side (side contact needs 8 + 4 = 12).
    x, y, z = assemble_snap_target(
        13.0,
        0.5,
        (Rect(0.0, 0.0, 4.0, 4.0),),
        supports,
        4.0,
        snap_to_grid=True,
        snap_to_parts=True,
    )
    assert x == pytest.approx(12.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)  # same layer as the neighbor


def test_snap_target_stacking_preferred_on_tie() -> None:
    '''When stacking and side distances tie, stacking wins.'''
    # Support spans x in [-10, 6]: the corner-rounded cursor (9, 0) drops
    # at (8, 0) — overlapping the support (stacking candidate) and exactly
    # as far from the +X flush side (10, 0).
    supports = [_box_support(x=-2.0)]
    x, y, z = assemble_snap_target(
        9.0,
        0.0,
        (Rect(0.0, 0.0, 4.0, 4.0),),
        supports,
        4.0,
        snap_to_grid=True,
        snap_to_parts=True,
    )
    assert z == pytest.approx(3.0)
    assert (x, y) == (8.0, 0.0)


def test_snap_target_far_from_parts_falls_back_to_grid() -> None:
    '''No nearby pieces: the plain corner-rounded floor position.'''
    supports = [_box_support()]
    x, y, z = assemble_snap_target(
        200.0,
        100.0,
        (Rect(0.0, 0.0, 4.0, 4.0),),
        supports,
        4.0,
        snap_to_grid=True,
        snap_to_parts=True,
    )
    assert (x, y, z) == (200.0, 100.0, 0.0)


def test_snap_target_drops_into_excluded_void() -> None:
    '''Dropping into an excluded-cell void lands on the floor.

    Regression: the stacking candidate must come from material overlap
    only — the T-piece bounding box top (in the air above the notch)
    never wins.
    '''
    project, settings = _t_block_project()
    unit = settings.unit_size
    project.assemble.pieces = [AssembledPiece(uid="p", block_uid="a")]
    supports = assemble_supports(project)
    # Cursor inside the void cell (2, 0): spans x [0.5u, 1.5u].
    x, y, z = assemble_snap_target(
        1.5 * unit,
        -unit / 2,
        (Rect(0.0, 0.0, unit / 2, unit / 2),),
        supports,
        unit,
        snap_to_grid=True,
        snap_to_parts=True,
    )
    assert (x, y) == pytest.approx((1.5 * unit, -unit / 2))
    assert z == pytest.approx(0.0)


def test_snap_target_side_flush_against_notch_wall() -> None:
    '''A piece flushes against the inner wall of an excluded-cell notch.'''
    project, settings = _t_block_project()
    unit = settings.unit_size
    project.assemble.pieces = [AssembledPiece(uid="p", block_uid="a")]
    supports = assemble_supports(project)
    # Cursor inside the void, near the flush position against the notch's
    # -X wall (the row-0 material edge at x = 0.5u): the one-cell piece
    # center lands at 0.5u + u/2 = u.
    x, y, z = assemble_snap_target(
        unit + unit / 8,
        -unit / 2,
        (Rect(0.0, 0.0, unit / 2, unit / 2),),
        supports,
        unit,
        snap_to_grid=True,
        snap_to_parts=True,
    )
    assert x == pytest.approx(unit)
    assert y == pytest.approx(-unit / 2)
    assert z == pytest.approx(0.0)


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
