'''
File:   models.py
Brief:  Pydantic settings model describing the parametric block geometry.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.3.0
'''

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from brickgen.core.exceptions import SettingsValidationError

MAX_CELLS = 64  # sanity limit per axis for GUI/CLI input


def validate_settings(values: dict[str, object]) -> BlockSettings:
    '''Validate raw values into BlockSettings, mapping pydantic errors to ours.

    Args:
        values: Raw field values (e.g. from GUI form or CLI options).

    Returns:
        Validated settings instance.

    Raises:
        SettingsValidationError: If any value fails validation.
    '''
    try:
        return BlockSettings.model_validate(values)
    except ValidationError as exc:
        raise SettingsValidationError(str(exc)) from exc


class BlockSettings(BaseModel):
    '''Parametric description of a LEGO-like hollow stud block.

    The block is a hollow rectangular shell without a bottom: outer footprint
    ``cells_x * unit_size`` by ``cells_y * unit_size``, wall thickness
    ``wall_thickness``, open from below, with a solid top face carrying one
    cylindrical stud per 1x1 cell. For blocks of at least 2x2 cells, hollow
    reinforcement tubes are placed at the intersections of the mid-lines
    between stud rows/columns (one tube per sliding 2x2 cell group center).

    Attributes:
        cells_x: Number of cells along X (block length in cells).
        cells_y: Number of cells along Y (block width in cells).
        unit_size: Footprint of a single 1x1 cell in mm.
        wall_thickness: Shell wall thickness in mm (also used for tubes).
        block_height: Height of the shell from bottom rim to top face, mm.
        pin_diameter: Top stud diameter in mm.
        pin_height: Top stud height above the top face in mm.
        pins_enabled: Whether to place studs on the top face.
        tubes_enabled: Whether to place inner reinforcement tubes (2x2+ blocks).
        tube_inner_diameter: Inner bore diameter of the tubes, mm.
    '''

    cells_x: int = Field(default=1, ge=1, le=MAX_CELLS)
    cells_y: int = Field(default=1, ge=1, le=MAX_CELLS)
    unit_size: float = Field(default=4.0, gt=0)
    wall_thickness: float = Field(default=0.8, gt=0)
    block_height: float = Field(default=3.0, gt=0)
    pin_diameter: float = Field(default=2.5, gt=0)
    pin_height: float = Field(default=2.0, gt=0)
    pins_enabled: bool = True
    tubes_enabled: bool = True
    tube_inner_diameter: float = Field(default=1.0, gt=0)

    model_config = {"extra": "forbid"}

    @field_validator("unit_size", "wall_thickness", "block_height",
                     "pin_diameter", "pin_height", "tube_inner_diameter")
    @classmethod
    def _validate_positive_finite(cls, value: float) -> float:
        '''Reject non-finite values (NaN/inf slip past ``gt=0``).'''
        if not value < float("inf") or value != value:
            raise SettingsValidationError("dimension must be a finite positive number")
        return value

    @property
    def tube_outer_diameter(self) -> float:
        '''Outer tube diameter: bore plus two wall-thickness shells, mm.'''
        return self.tube_inner_diameter + 2.0 * self.wall_thickness

    @property
    def has_tubes(self) -> bool:
        '''True when tubes apply: enabled and both cell counts are 2+.'''
        return self.tubes_enabled and self.cells_x >= 2 and self.cells_y >= 2

    @model_validator(mode="after")
    def _validate_geometry(self) -> BlockSettings:
        '''Reject cross-field conflicts (no cavity, oversized stud/tube).'''
        if 2.0 * self.wall_thickness >= self.unit_size:
            raise SettingsValidationError(
                "wall_thickness*2 must be smaller than unit_size to keep a cavity"
            )
        if self.block_height <= self.wall_thickness:
            raise SettingsValidationError(
                "block_height must exceed wall_thickness (top face keeps wall thickness)"
            )
        if self.pins_enabled and self.pin_diameter >= self.unit_size:
            raise SettingsValidationError(
                "pin_diameter must be smaller than unit_size"
            )
        if self.has_tubes:
            cavity_min = min(
                self.outer_length_x, self.outer_length_y
            ) - 2.0 * self.wall_thickness
            if self.tube_outer_diameter >= cavity_min:
                raise SettingsValidationError(
                    "tube_outer_diameter must fit inside the cavity"
                )
        return self

    @property
    def outer_length_x(self) -> float:
        '''Outer footprint size along X, mm.'''
        return self.cells_x * self.unit_size

    @property
    def outer_length_y(self) -> float:
        '''Outer footprint size along Y, mm.'''
        return self.cells_y * self.unit_size

    @property
    def total_height(self) -> float:
        '''Total height including studs, mm.'''
        return self.block_height + (self.pin_height if self.pins_enabled else 0.0)

    def describe(self) -> str:
        '''Human-readable form label, e.g. ``2x6``.'''
        return f"{self.cells_x}x{self.cells_y}"


__all__ = ["BlockSettings", "validate_settings"]
