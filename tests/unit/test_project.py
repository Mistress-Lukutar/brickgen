'''
File:   test_project.py
Brief:  Unit tests for Project and BlockInstance models.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.18.0
'''

from __future__ import annotations

import pytest

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.models import BlockSettings
from brickgen.core.project import (
    DEFAULT_BLOCK_COLOR,
    GRID_SCALE_PRESETS,
    PRINT_STAGING_BED,
    QUANTITY_LABEL_POSITIONS,
    BlockInstance,
    PrintBedSettings,
    PrintLayout,
    PrintPiece,
    PrintToolsSettings,
    Project,
    ViewSettings,
)


# ---------------------------------------------------------- defaults
def test_default_block_instance_validates() -> None:
    '''A BlockInstance with all defaults must validate cleanly.'''
    inst = BlockInstance()
    assert inst.cells_x == 1
    assert inst.cells_y == 1
    assert inst.position_x == 0.0
    assert inst.position_y == 0.0
    assert inst.color == DEFAULT_BLOCK_COLOR
    assert inst.quantity == 1
    assert inst.excluded_cells == []
    assert inst.flat_cells == []
    assert inst.stud_cells == []
    assert inst.sloped_cells == []
    assert inst.chamfered_cells == []
    assert inst.round_cells == []


def test_invalid_color_rejected() -> None:
    '''A non-hex color must raise SettingsValidationError.'''
    with pytest.raises(SettingsValidationError, match="color"):
        BlockInstance(color="red")


def test_quantity_zero_rejected() -> None:
    '''quantity=0 is below ge=1 and must be rejected.'''
    with pytest.raises(SettingsValidationError):
        BlockInstance(quantity=0)


def test_non_finite_position_rejected() -> None:
    '''NaN and infinity positions must raise SettingsValidationError.'''
    with pytest.raises(SettingsValidationError, match="position"):
        BlockInstance(position_x=float("nan"))
    with pytest.raises(SettingsValidationError, match="position"):
        BlockInstance(position_y=float("inf"))


# ------------------------------------------------------ layout validation
def test_sloped_cell_outside_grid_rejected() -> None:
    '''A sloped cell beyond the grid bounds must fail.'''
    with pytest.raises(SettingsValidationError, match="outside"):
        BlockInstance(cells_x=2, cells_y=1, sloped_cells=[[2, 0, "+X", 1]])


def test_sloped_and_stud_conflict_rejected() -> None:
    '''A cell cannot be both sloped and stud.'''
    with pytest.raises(SettingsValidationError, match="sloped and stud"):
        BlockInstance(
            cells_x=2,
            cells_y=1,
            sloped_cells=[[0, 0, "+X", 1]],
            stud_cells=[[0, 0]],
        )


def test_duplicate_excluded_pair_rejected() -> None:
    '''Duplicate [ix, iy] in excluded_cells must fail.'''
    with pytest.raises(SettingsValidationError, match="twice"):
        BlockInstance(
            cells_x=2,
            cells_y=1,
            excluded_cells=[[0, 0], [0, 0]],
        )


def test_valid_sloped_quad_accepted() -> None:
    '''A correct sloped quadruplet must pass validation.'''
    inst = BlockInstance(
        cells_x=2,
        cells_y=1,
        sloped_cells=[[0, 0, "+X", 2]],
    )
    assert inst.sloped_cells == [[0, 0, "+X", 2]]


# ---------------------------------------------------------- to_settings
def test_to_settings_merges_shared_and_layout() -> None:
    '''to_settings combines shared geometry with instance layout.'''
    shared = BlockSettings(
        block_height=5.0,
        clearance_xy=-0.25,
        cells_x=1,
        cells_y=1,
    )
    inst = BlockInstance(
        cells_x=3,
        cells_y=1,
        excluded_cells=[[0, 0]],
    )
    merged = inst.to_settings(shared)

    assert merged.block_height == 5.0
    assert merged.cells_x == 3
    assert merged.excluded_cells == [[0, 0]]
    # Shared fields that weren't overridden are preserved
    assert merged.clearance_xy == -0.25
    assert merged.unit_size == shared.unit_size
    assert merged.wall_thickness == shared.wall_thickness


def test_to_settings_block_flags_win_over_shared() -> None:
    '''The block's own pins/tubes flags take priority over shared settings.

    Regression: adding a studless block must never flip the studs of the
    already placed blocks (the flags used to be project-wide).
    '''
    shared = BlockSettings(pins_enabled=False, tubes_enabled=False)
    pinned = BlockInstance(cells_x=2, cells_y=2)
    studless = BlockInstance(cells_x=2, cells_y=2, pins_enabled=False)

    merged_pinned = pinned.to_settings(BlockSettings(pins_enabled=True))
    merged_studless = studless.to_settings(shared)

    assert merged_pinned.pins_enabled is True
    assert merged_studless.pins_enabled is False
    assert merged_studless.tubes_enabled is True


# ------------------------------------------------------------- Project
def test_project_with_multiple_blocks() -> None:
    '''Project accepts multiple distinct BlockInstances.'''
    inst1 = BlockInstance(
        cells_x=2,
        cells_y=2,
        color="#ff0000",
        quantity=5,
    )
    inst2 = BlockInstance(
        cells_x=3,
        cells_y=3,
        color="#00ff00",
        quantity=3,
        sloped_cells=[[0, 0, "+X", 1]],
    )
    proj = Project(blocks=[inst1, inst2])

    assert len(proj.blocks) == 2
    assert proj.blocks[0].color == "#ff0000"
    assert proj.blocks[1].sloped_cells == [[0, 0, "+X", 1]]


def test_block_settings_returns_merged() -> None:
    '''block_settings(i) returns per-instance merged BlockSettings.'''
    shared = BlockSettings(block_height=5.0, clearance_xy=-0.25)
    inst = BlockInstance(
        cells_x=3,
        cells_y=1,
        excluded_cells=[[0, 0]],
        color="#aabbcc",
    )
    proj = Project(settings=shared, blocks=[inst])

    bs = proj.block_settings(0)
    assert bs.block_height == 5.0
    assert bs.cells_x == 3
    assert bs.excluded_cells == [[0, 0]]
    assert bs.clearance_xy == -0.25


def test_total_quantity_sums() -> None:
    '''total_quantity sums the quantity of every block instance.'''
    proj = Project(
        blocks=[
            BlockInstance(quantity=10),
            BlockInstance(quantity=20),
            BlockInstance(quantity=30),
        ]
    )
    assert proj.total_quantity() == 60


def test_project_roundtrip() -> None:
    '''Project.model_validate(project.model_dump()) must be equal.'''
    proj = Project(
        settings=BlockSettings(block_height=4.5, clearance_xy=-0.1),
        blocks=[
            BlockInstance(cells_x=2, cells_y=3, color="#ff5500", quantity=7),
            BlockInstance(
                cells_x=1,
                cells_y=1,
                sloped_cells=[[0, 0, "+Y", 1]],
                chamfered_cells=[[0, 0, "-X"]],
            ),
        ],
    )
    roundtripped = Project.model_validate(proj.model_dump())
    assert roundtripped == proj


# ------------------------------------------------------- PrintBedSettings
def test_print_bed_settings_defaults() -> None:
    '''PrintBedSettings must expose the documented defaults.'''
    bs = PrintBedSettings()
    assert bs.width == 220.0
    assert bs.depth == 220.0
    assert bs.count == 1
    assert bs.gap == 1.0


def test_print_bed_settings_rejects_invalid_width() -> None:
    '''Width below 10 or above 2000 must raise.'''
    with pytest.raises(Exception):
        PrintBedSettings(width=5.0)
    with pytest.raises(Exception):
        PrintBedSettings(width=3000.0)


def test_print_bed_settings_rejects_invalid_count() -> None:
    '''Count 0 and count 65 must raise.'''
    with pytest.raises(Exception):
        PrintBedSettings(count=0)
    with pytest.raises(Exception):
        PrintBedSettings(count=65)


def test_print_bed_settings_rejects_invalid_gap() -> None:
    '''Gap -1 and gap 101 must raise.'''
    with pytest.raises(Exception):
        PrintBedSettings(gap=-1.0)
    with pytest.raises(Exception):
        PrintBedSettings(gap=101.0)


# ------------------------------------------------------- BlockInstance.uid
def test_block_instance_uid_is_nonempty() -> None:
    '''A BlockInstance must always have a non-empty uid.'''
    inst = BlockInstance()
    assert len(inst.uid) > 0


def test_block_instance_uids_differ() -> None:
    '''Two freshly created instances must have distinct uids.'''
    inst1 = BlockInstance()
    inst2 = BlockInstance()
    assert inst1.uid != inst2.uid


# ------------------------------------------------------- PrintPiece
def test_print_piece_defaults() -> None:
    '''PrintPiece defaults: bed == staging, uid auto-assigned, no rotation.'''
    piece = PrintPiece(block_uid="abc123")
    assert piece.bed == PRINT_STAGING_BED
    assert len(piece.uid) > 0
    assert piece.flipped is False
    assert piece.rotated90 is False


def test_print_piece_empty_block_uid_rejected() -> None:
    '''An empty block_uid must raise SettingsValidationError.'''
    with pytest.raises(SettingsValidationError):
        PrintPiece(block_uid="")


def test_print_piece_bed_negative_two_rejected() -> None:
    '''bed == -2 is below PRINT_STAGING_BED and must be rejected.'''
    with pytest.raises(SettingsValidationError, match="bed"):
        PrintPiece(block_uid="abc", bed=-2)


def test_print_piece_nan_position_rejected() -> None:
    '''NaN position_x / position_y must raise SettingsValidationError.'''
    with pytest.raises(SettingsValidationError, match="position"):
        PrintPiece(block_uid="abc", position_x=float("nan"))
    with pytest.raises(SettingsValidationError, match="position"):
        PrintPiece(block_uid="abc", position_y=float("inf"))


# ------------------------------------------------------- PrintLayout
def test_print_layout_valid_pieces_pass() -> None:
    '''Valid pieces with staging or valid bed indices must pass.'''
    layout = PrintLayout(
        beds=PrintBedSettings(count=2),
        pieces=[
            PrintPiece(block_uid="a", bed=-1),
            PrintPiece(block_uid="b", bed=0),
            PrintPiece(block_uid="c", bed=1),
        ],
    )
    assert len(layout.pieces) == 3


def test_print_layout_bed_out_of_range_rejected() -> None:
    '''piece with bed 1 when count==1 must raise SettingsValidationError.'''
    with pytest.raises(SettingsValidationError, match="references bed"):
        PrintLayout(
            beds=PrintBedSettings(count=1),
            pieces=[PrintPiece(block_uid="x", bed=1)],
        )


def test_print_layout_bed_0_count_2_passes() -> None:
    '''bed 0 with count 2 must pass.'''
    layout = PrintLayout(
        beds=PrintBedSettings(count=2),
        pieces=[PrintPiece(block_uid="x", bed=0)],
    )
    assert len(layout.pieces) == 1


def test_print_layout_bed_minus_one_always_passes() -> None:
    '''Staging (bed -1) must always pass regardless of count.'''
    layout = PrintLayout(
        beds=PrintBedSettings(count=1),
        pieces=[PrintPiece(block_uid="x", bed=-1)],
    )
    assert len(layout.pieces) == 1


# ------------------------------------------------------- Project.print_layout
def test_project_default_print_layout() -> None:
    '''A project without explicit print_layout gets defaults.'''
    proj = Project()
    assert isinstance(proj.print_layout, PrintLayout)
    assert proj.print_layout.beds.count == 1
    assert proj.print_layout.pieces == []


# ------------------------------------------------------ PrintToolsSettings
def test_print_tools_settings_defaults() -> None:
    '''PrintToolsSettings exposes the factory defaults and PrintLayout
    carries them when no explicit tools state is given.'''
    tools = PrintToolsSettings()
    assert tools.auto_from == 1
    assert tools.auto_to == 1
    assert tools.distribute_by_color is True
    assert tools.distribute_evenly is True
    assert PrintLayout().tools == tools


def test_print_layout_tools_range_clamped_to_beds() -> None:
    '''The auto-place range is clamped into the existing beds (e.g. after
    a bed count shrank), so persisted values stay usable.'''
    layout = PrintLayout(
        beds=PrintBedSettings(count=2),
        tools=PrintToolsSettings(auto_from=1, auto_to=5),
    )
    assert layout.tools.auto_from == 1
    assert layout.tools.auto_to == 2


def test_print_layout_tools_range_normalized() -> None:
    '''from > to (hand-edited JSON) normalizes to from <= to.'''
    layout = PrintLayout(
        beds=PrintBedSettings(count=4),
        tools=PrintToolsSettings(auto_from=3, auto_to=1),
    )
    assert layout.tools.auto_from == 3
    assert layout.tools.auto_to == 3
    layout = PrintLayout(
        beds=PrintBedSettings(count=1),
        tools=PrintToolsSettings(auto_from=3, auto_to=1),
    )
    assert layout.tools.auto_from == 1
    assert layout.tools.auto_to == 1


def test_project_with_valid_pieces_validates() -> None:
    '''A project with pieces validates when bed indices are in range.'''
    proj = Project(
        print_layout=PrintLayout(
            beds=PrintBedSettings(count=2),
            pieces=[
                PrintPiece(block_uid="a", bed=0),
                PrintPiece(block_uid="b", bed=1),
                PrintPiece(block_uid="c", bed=-1),
            ],
        )
    )
    assert len(proj.print_layout.pieces) == 3


# -------------------------------------------------------- Project.view
def test_view_settings_defaults() -> None:
    '''ViewSettings exposes the documented factory defaults.'''
    view = ViewSettings()
    assert view.grid_enabled is False
    assert view.grid_scale == 0.5
    assert view.grid_snap is False
    assert view.placement_grid_enabled is False
    assert view.quantity_labels_enabled is False
    assert view.quantity_label_position == "right"


def test_view_settings_scale_bounds() -> None:
    '''Grid scale must be a finite positive number within the cap.'''
    ViewSettings(grid_scale=1000.0)
    with pytest.raises(Exception):
        ViewSettings(grid_scale=0.0)
    with pytest.raises(Exception):
        ViewSettings(grid_scale=-0.5)
    with pytest.raises(Exception):
        ViewSettings(grid_scale=1000.5)
    # NaN / inf fail the gt/le field constraints.
    with pytest.raises(Exception):
        ViewSettings(grid_scale=float("nan"))
    with pytest.raises(Exception):
        ViewSettings(grid_scale=float("inf"))


def test_view_settings_position_literal_rejected() -> None:
    '''Only the documented anchor sides are accepted.'''
    assert set(QUANTITY_LABEL_POSITIONS) == {"left", "right", "top", "bottom"}
    with pytest.raises(Exception):
        ViewSettings(quantity_label_position="diagonal")


def test_view_settings_unknown_key_rejected() -> None:
    '''Unknown keys are rejected (extra="forbid").'''
    with pytest.raises(Exception):
        ViewSettings(grid_scalee=1.0)  # type: ignore[call-arg]


def test_grid_scale_presets_offered() -> None:
    '''The Grid menu offers the documented scale presets incl. 0.5 / 1 / 10.'''
    for scale in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        assert scale in GRID_SCALE_PRESETS


def test_project_view_default_and_custom() -> None:
    '''A project without explicit view gets defaults; a custom one sticks.'''
    proj = Project()
    assert proj.view == ViewSettings()
    proj2 = Project(view=ViewSettings(grid_snap=True, grid_scale=2.0))
    assert proj2.view.grid_snap is True
    assert proj2.view.grid_scale == 2.0
