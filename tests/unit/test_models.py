'''
File:   test_models.py
Brief:  Unit tests for BlockSettings validation and derived properties.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.17.0
'''

from __future__ import annotations

import math

import pytest

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.models import BlockSettings, validate_settings


def test_defaults_match_specification(default_settings: BlockSettings) -> None:
    '''Factory defaults must match the specified 1x1 block dimensions.'''
    assert default_settings.unit_size == 4.0
    assert default_settings.wall_thickness == 0.8
    assert default_settings.block_height == 3.0
    assert default_settings.clearance_xy == -0.25
    assert default_settings.clearance_z == 0.2
    # Derived values from clearance fields
    assert default_settings.pin_diameter == pytest.approx(2.65, abs=1e-9)
    assert default_settings.pin_height == pytest.approx(2.0, abs=1e-9)
    tube_od = (math.sqrt(2.0) - 1.0) * 4.0 + 2.0 * 0.8
    tube_id = tube_od - 2.0 * 0.8
    assert default_settings.tube_outer_diameter == pytest.approx(tube_od, abs=1e-9)
    assert default_settings.tube_inner_diameter == pytest.approx(tube_id, abs=1e-9)
    assert default_settings.cells_x == 1
    assert default_settings.cells_y == 1


def test_derived_properties(settings_2x6: BlockSettings) -> None:
    '''Footprint and height are derived from cell counts and unit size.'''
    assert settings_2x6.outer_length_x == pytest.approx(8.0)
    assert settings_2x6.outer_length_y == pytest.approx(24.0)
    assert settings_2x6.total_height == pytest.approx(5.0)
    assert settings_2x6.describe() == "2x6"


def test_total_height_without_pins() -> None:
    '''Disabling studs removes pin height from the total.'''
    settings = BlockSettings(pins_enabled=False)
    assert settings.total_height == pytest.approx(settings.block_height)


@pytest.mark.parametrize(
    "field, value",
    [
        ("unit_size", 0.0),
        ("wall_thickness", -1.0),
        ("block_height", 0.0),
    ],
)
def test_rejects_non_positive_dimensions(field: str, value: float) -> None:
    '''Zero and negative values must fail for gt=0 dimension fields.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({field: value})


def test_negative_clearance_xy_enlarges_stud_only() -> None:
    '''Negative clearance_xy fattens the stud; the block size stays nominal.'''
    settings = BlockSettings(
        cells_x=2, cells_y=3, unit_size=4.0, wall_thickness=0.8, clearance_xy=-0.5
    )
    assert settings.outer_length_x == pytest.approx(8.0, abs=1e-9)
    assert settings.outer_length_y == pytest.approx(12.0, abs=1e-9)
    # Socket 2.4: the stud is modeled 0.5 fatter (shrinkage interference).
    assert settings.pin_diameter == pytest.approx(2.9, abs=1e-9)
    tube_od = (math.sqrt(2.0) - 1.0) * 4.0 + 2.0 * 0.8
    assert settings.tube_outer_diameter == pytest.approx(tube_od, abs=1e-9)
    # Just inside the -2*wall bound the stud still fits its cell footprint.
    loose = BlockSettings(unit_size=4.0, wall_thickness=0.8, clearance_xy=-1.5)
    assert loose.pin_diameter == pytest.approx(3.9, abs=1e-9)


def test_negative_clearance_z_enlarges_stud_only() -> None:
    '''Negative clearance_z makes the stud taller; the shell stays nominal.'''
    settings = BlockSettings(clearance_z=-0.5)
    assert settings.cavity_depth == pytest.approx(2.2, abs=1e-9)
    assert settings.pin_height == pytest.approx(2.7, abs=1e-9)
    assert settings.total_height == pytest.approx(5.7, abs=1e-9)
    sloped = BlockSettings(sloped_cells=[[0, 0, "+Y", 2]], clearance_z=-0.5)
    assert sloped.slope_rise == pytest.approx(6.0, abs=1e-9)


@pytest.mark.parametrize(
    "field, value",
    [
        ("clearance_xy", -1.6),
        ("clearance_xy", -3.0),
        ("clearance_z", -2.2),
        ("clearance_z", -3.0),
    ],
)
def test_rejects_over_sized_studs(field: str, value: float) -> None:
    '''clearance_xy <= -2*wall would overlap neighboring studs (defaults:
    -1.6); clearance_z <= -cavity depth fails the sanity bound (-2.2).'''
    with pytest.raises(SettingsValidationError):
        validate_settings({field: value})


def test_zero_clearances_allowed() -> None:
    '''Zero clearance_xy and clearance_z are valid (pin_diameter == 2.0).'''
    settings = BlockSettings(
        clearance_xy=0.0, clearance_z=0.0, unit_size=4.0, wall_thickness=1.0
    )
    assert settings.pin_diameter == pytest.approx(2.0, abs=1e-9)
    assert settings.pin_height == pytest.approx(2.0, abs=1e-9)


def test_rejects_wall_leaving_no_cavity() -> None:
    '''Wall thickness of half the unit size leaves no cavity.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({"wall_thickness": 2.0})


def test_rejects_clearance_leaving_no_stud() -> None:
    '''clearance_xy >= socket width or clearance_z >= cavity depth must fail.'''
    # At defaults: socket = 4.0 - 2*0.8 = 2.4; equality (>=) raises
    with pytest.raises(SettingsValidationError, match="clearance_xy"):
        validate_settings({"clearance_xy": 2.4})
    # cavity_depth = 3.0 - 0.8 = 2.2; equality (>=) raises
    with pytest.raises(SettingsValidationError, match="clearance_z"):
        validate_settings({"clearance_z": 2.2})


def test_rejects_unknown_fields() -> None:
    '''Unknown keys must be rejected to catch preset typos.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({"cell_x": 2})  # typo on purpose


def test_rejects_tube_that_does_not_fit_cavity() -> None:
    '''A tube wider than the cavity must fail validation (via derived sizes).'''
    # unit_size=4.0, wall_thickness=1.8 → tube_outer ≈ 5.26 ≥ cavity width 4.4
    with pytest.raises(SettingsValidationError, match="tube_outer_diameter"):
        validate_settings(
            {
                "cells_x": 2,
                "cells_y": 2,
                "unit_size": 4.0,
                "wall_thickness": 1.8,
            }
        )


def test_has_tubes_requires_2x2() -> None:
    '''Tubes apply only when both dimensions are at least 2 cells.'''
    assert BlockSettings(cells_x=2, cells_y=2).has_tubes
    assert not BlockSettings(cells_x=1, cells_y=4).has_tubes
    assert not BlockSettings(cells_x=2, cells_y=2, tubes_enabled=False).has_tubes


def test_rejects_non_finite_dimension() -> None:
    '''NaN and infinity must fail validation.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({"unit_size": float("nan")})
    with pytest.raises(SettingsValidationError):
        validate_settings({"unit_size": float("inf")})


# ------------------------------------------------- per-cell top features
def test_cell_top_resolution() -> None:
    '''Slope wins over explicit overrides; plain cells follow the default.'''
    settings = BlockSettings(
        cells_x=1,
        cells_y=3,
        pins_enabled=False,
        stud_cells=[[0, 0]],
        sloped_cells=[[0, 1, "+Y", 1]],
    )
    assert settings.cell_top(0, 0) == "stud"  # explicit override
    assert settings.cell_top(0, 1) == "slope"
    assert settings.cell_top(0, 2) == "flat"  # default
    assert settings.has_studs
    assert settings.stud_cells_all() == {(0, 0)}

    default_on = BlockSettings(cells_x=1, cells_y=2, flat_cells=[[0, 1]])
    assert default_on.cell_top(0, 0) == "stud"
    assert default_on.cell_top(0, 1) == "flat"


def test_total_height_accounts_for_slopes_and_studs() -> None:
    '''Total height covers the tallest slope and the stud height.'''
    sloped = BlockSettings(
        cells_x=1, cells_y=2, pins_enabled=False, sloped_cells=[[0, 1, "-Y", 2]]
    )
    assert sloped.total_height == pytest.approx(sloped.block_height * 3)
    assert sloped.max_slope_rise_units == 2
    assert sloped.slope_rise == pytest.approx(sloped.block_height * 2)
    assert not sloped.has_studs

    mixed = BlockSettings(
        cells_x=1, cells_y=2, sloped_cells=[[0, 1, "-Y", 1]], stud_cells=[[0, 0]]
    )
    # pin height 2 vs slope rise 3: the slope wins
    assert mixed.total_height == pytest.approx(mixed.block_height * 2)


def test_slope_cell_validation() -> None:
    '''Rise bounds, duplicates and conflicts with other tops are rejected.'''
    # rise below 1 / above the cap
    with pytest.raises(SettingsValidationError, match="rise"):
        BlockSettings(cells_x=2, cells_y=1, sloped_cells=[[0, 0, "+X", 0]])
    with pytest.raises(SettingsValidationError, match="rise"):
        BlockSettings(cells_x=2, cells_y=1, sloped_cells=[[0, 0, "+X", 17]])
    # one wedge per cell
    with pytest.raises(SettingsValidationError, match="twice"):
        BlockSettings(
            cells_x=2, cells_y=1, sloped_cells=[[0, 0, "+X", 1], [0, 0, "-X", 1]]
        )
    # unknown side
    with pytest.raises(SettingsValidationError, match="must be one of"):
        BlockSettings(cells_x=2, cells_y=1, sloped_cells=[[0, 0, "+Z", 1]])
    # malformed quadruplet (a 3-element entry is treated as a v1 bottom
    # chamfer by the migration, so the malformed case uses a bad rise)
    with pytest.raises(SettingsValidationError, match="quadruplet"):
        BlockSettings(cells_x=2, cells_y=1, sloped_cells=[[0, 0, "+X", "x"]])
    # out of bounds / excluded cell
    with pytest.raises(SettingsValidationError, match="outside"):
        BlockSettings(cells_x=2, cells_y=1, stud_cells=[[5, 0]])
    with pytest.raises(SettingsValidationError, match="excluded"):
        BlockSettings(
            cells_x=2,
            cells_y=1,
            excluded_cells=[[0, 0]],
            sloped_cells=[[0, 0, "+X", 1]],
        )


def test_top_conflicts_rejected() -> None:
    '''slope+stud, slope+flat and flat+stud cannot share a cell.'''
    with pytest.raises(SettingsValidationError, match="sloped and stud"):
        BlockSettings(
            cells_x=2, cells_y=1, sloped_cells=[[0, 0, "+X", 1]], stud_cells=[[0, 0]]
        )
    with pytest.raises(SettingsValidationError, match="sloped and flat"):
        BlockSettings(
            cells_x=2, cells_y=1, sloped_cells=[[0, 0, "+X", 1]], flat_cells=[[0, 0]]
        )
    with pytest.raises(SettingsValidationError, match="flat and stud"):
        BlockSettings(cells_x=2, cells_y=1, flat_cells=[[0, 0]], stud_cells=[[0, 0]])


def test_slope_inside_figure_is_allowed() -> None:
    '''Unlike chamfers, slope directions need not look outward.'''
    settings = BlockSettings(cells_x=3, cells_y=1, sloped_cells=[[1, 0, "+Y", 1]])
    assert settings.sloped_cells == [[1, 0, "+Y", 1]]
    assert not settings.is_rectangular


# ------------------------------------------------------- round base cells
def test_round_cell_validation() -> None:
    '''Bounds, duplicates and wedge conflicts are rejected for round cells.'''
    with pytest.raises(SettingsValidationError, match="outside"):
        BlockSettings(cells_x=2, cells_y=1, round_cells=[[5, 0]])
    with pytest.raises(SettingsValidationError, match="twice"):
        BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0], [0, 0]])
    with pytest.raises(SettingsValidationError, match="round and sloped"):
        BlockSettings(
            cells_x=2, cells_y=1, round_cells=[[0, 0]], sloped_cells=[[0, 0, "+X", 1]]
        )
    with pytest.raises(SettingsValidationError, match="round and chamfered"):
        BlockSettings(
            cells_x=2, cells_y=1, round_cells=[[0, 0]], chamfered_cells=[[0, 0, "-X"]]
        )
    with pytest.raises(SettingsValidationError, match="excluded"):
        BlockSettings(
            cells_x=2, cells_y=1, excluded_cells=[[0, 0]], round_cells=[[0, 0]]
        )


def test_round_cell_combines_with_top_overrides() -> None:
    '''A round base is independent of the top type: stud or flat both fine.'''
    round_stud = BlockSettings(cells_x=2, cells_y=1, round_cells=[[0, 0]])
    assert round_stud.cell_top(0, 0) == "stud"
    assert round_stud.cell_top(1, 0) == "stud"
    assert (0, 0) in round_stud.stud_cells_all()
    assert not round_stud.is_rectangular

    round_flat = BlockSettings(
        cells_x=2, cells_y=1, round_cells=[[0, 0]], flat_cells=[[0, 0]]
    )
    assert round_flat.cell_top(0, 0) == "flat"
    assert round_flat.has_studs  # only cell (1,0) carries a stud


def test_v1_migration() -> None:
    '''v1 keys map onto the v2 fields on load.'''
    # global stud-less slope -> per-cell wedges covering the whole grid
    migrated = validate_settings(
        {
            "cells_x": 2,
            "cells_y": 2,
            "pins_enabled": False,
            "tubes_enabled": False,
            "slope_rise_units": 1,
            "slope_side": "+X",
        }
    )
    assert migrated.sloped_cells == sorted(
        [[ix, iy, "+X", 1] for iy in range(2) for ix in range(2)]
    )
    assert not migrated.has_studs

    # v1 bottom chamfers (sloped_cells triplets) -> chamfered_cells
    migrated_chamfers = validate_settings(
        {"cells_x": 2, "cells_y": 2, "sloped_cells": [[1, 0, "+X"], [0, 1, "+Y"]]}
    )
    assert migrated_chamfers.chamfered_cells == [[0, 1, "+Y"], [1, 0, "+X"]]
    assert migrated_chamfers.sloped_cells == []

    # a global slope with the default studs on was ignored in v1: dropped
    dropped = validate_settings(
        {"cells_x": 2, "cells_y": 2, "pins_enabled": True, "slope_rise_units": 3}
    )
    assert dropped.sloped_cells == []
    assert dropped.has_studs

    # v1 quadruplets (already v2-style) pass through untouched
    kept = validate_settings(
        {"cells_x": 2, "cells_y": 1, "sloped_cells": [[0, 0, "+X", 2]]}
    )
    assert kept.sloped_cells == [[0, 0, "+X", 2]]


# ------------------------------------------------------------ recipe names
def test_recipe_name_distinct_stud_modes() -> None:
    '''Plain blocks with and without studs must never share a name.'''
    assert BlockSettings(cells_x=2, cells_y=2).recipe_name() == "2x2-studs"
    assert (
        BlockSettings(cells_x=2, cells_y=2, pins_enabled=False).recipe_name()
        == "2x2-studless"
    )


def test_recipe_name_encodes_negative_clearance() -> None:
    '''Negative mm deltas encode with a minus sign and stay file-name safe.'''
    settings = BlockSettings(clearance_xy=-0.5, clearance_z=-0.25)
    name = settings.recipe_name()
    assert "cx-0.5" in name
    assert "cz-0.25" in name


def test_recipe_name_encodes_cell_features() -> None:
    '''Per-cell features appear with coordinates in a stable order.'''
    slope = BlockSettings(cells_x=1, cells_y=2, sloped_cells=[[0, 1, "+Y", 1]])
    assert slope.recipe_name() == "1x2-studs_sl0,1,+Y,1"

    combo = BlockSettings(
        cells_x=1,
        cells_y=2,
        sloped_cells=[[0, 0, "+Y", 1]],
        chamfered_cells=[[0, 0, "-Y"]],
    )
    assert combo.recipe_name() == "1x2-studs_sl0,0,+Y,1_ch0,0,-Y"

    excl = BlockSettings(cells_x=2, cells_y=2, excluded_cells=[[1, 1], [0, 0]])
    assert excl.recipe_name() == "2x2-studs_ex0,0;1,1"

    round_cell = BlockSettings(cells_x=1, cells_y=1, round_cells=[[0, 0]])
    assert round_cell.recipe_name() == "1x1-studs_rd0,0"

    overrides = BlockSettings(
        cells_x=2,
        cells_y=1,
        pins_enabled=False,
        stud_cells=[[1, 0]],
        flat_cells=[[0, 0]],
    )
    assert overrides.recipe_name() == "2x1-studless_fl0,0_st1,0"


def test_recipe_name_tubes_and_dimensions() -> None:
    '''Tube state and non-default millimeter dimensions are encoded.'''
    assert (
        BlockSettings(cells_x=2, cells_y=2, tubes_enabled=False).recipe_name()
        == "2x2-studs-notubes"
    )
    # 1-wide blocks never carry tubes: no token even when disabled
    assert (
        BlockSettings(cells_x=1, cells_y=4, tubes_enabled=False).recipe_name()
        == "1x4-studs"
    )
    tall = BlockSettings(cells_x=2, cells_y=2, block_height=6.0)
    assert tall.recipe_name() == "2x2-studs_h6"
    # Non-default mm fields produce tokens in field order; no legacy pd/ph/tb
    exotic = BlockSettings(
        cells_x=2,
        cells_y=2,
        unit_size=5.6,
        block_height=4.5,
        clearance_xy=0.3,
        clearance_z=0.1,
    )
    name = exotic.recipe_name()
    assert "u5.6" in name
    assert "h4.5" in name
    assert "cx0.3" in name
    assert "cz0.1" in name
    assert "pd" not in name
    assert "ph" not in name
    assert "tb" not in name
    # Defaults produce no cx/cz token
    defaults = BlockSettings()
    assert "cx" not in defaults.recipe_name()
    assert "cz" not in defaults.recipe_name()
