'''
File:   test_models.py
Brief:  Unit tests for BlockSettings validation and derived properties.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.4.1
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
