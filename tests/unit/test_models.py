'''
File:   test_models.py
Brief:  Unit tests for BlockSettings validation and derived properties.
Author: Mistress-Lukutar
Date:   2026-11-17
Version: v0.10.0
'''

from __future__ import annotations

import pytest

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.models import BlockSettings, validate_settings


def test_defaults_match_specification(default_settings: BlockSettings) -> None:
    '''Factory defaults must match the specified 1x1 block dimensions.'''
    assert default_settings.unit_size == 4.0
    assert default_settings.wall_thickness == 0.8
    assert default_settings.block_height == 3.0
    assert default_settings.pin_diameter == 2.6
    assert default_settings.pin_height == 2.0
    assert default_settings.tube_inner_diameter == 1.2
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
        ("pin_diameter", 0.0),
        ("pin_height", -0.5),
    ],
)
def test_rejects_non_positive_dimensions(field: str, value: float) -> None:
    '''Zero and negative dimensions must fail validation.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({field: value})


def test_rejects_wall_leaving_no_cavity() -> None:
    '''Wall thickness of half the unit size leaves no cavity.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({"wall_thickness": 2.0})


def test_rejects_pin_wider_than_cell() -> None:
    '''A stud wider than the unit cell cannot fit.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({"pin_diameter": 5.0})


def test_rejects_unknown_fields() -> None:
    '''Unknown keys must be rejected to catch preset typos.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({"cell_x": 2})  # typo on purpose


def test_rejects_tube_that_does_not_fit_cavity() -> None:
    '''A tube wider than the cavity must fail validation.'''
    with pytest.raises(SettingsValidationError):
        validate_settings({"cells_x": 2, "cells_y": 2, "tube_inner_diameter": 5.2})


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
    settings = BlockSettings(
        cells_x=3, cells_y=1, sloped_cells=[[1, 0, "+Y", 1]]
    )
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
