'''
File:   models.py
Brief:  Pydantic settings model describing the parametric block geometry.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.5.0
'''

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from brickgen.core.exceptions import SettingsValidationError

MAX_CELLS = 64  # sanity limit per axis for GUI/CLI input
MAX_SLOPE_RISE = 16  # sanity limit for the slope rise in unit heights

# Side of the raised edge that forms the slope.
SlopeSide = Literal["+X", "-X", "+Y", "-Y"]
SLOPE_SIDES: tuple[str, ...] = ("+X", "-X", "+Y", "-Y")


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

    Stud-less blocks may instead have a sloped top: one top edge rises by
    ``slope_rise_units * block_height`` and the top face slopes down to the
    opposite edge, forming a wedge.

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
        slope_rise_units: Rise of one top edge in unit block heights
            (N x ``block_height``), forming a sloped top face. Applies only
            to stud-less blocks (``pins_enabled=False``); 0 disables it.
        slope_side: Which top edge is raised: ``+X``/``-X``/``+Y``/``-Y``.
        excluded_cells: 0-based ``[ix, iy]`` cells removed from the figure
            (top view, ix along X, iy along Y). Non-empty masks switch the
            builder to the figure variant: contour walls follow the shape,
            concave corners get inward quarter-round ribs instead of tubes.
    '''

    cells_x: int = Field(default=1, ge=1, le=MAX_CELLS)
    cells_y: int = Field(default=1, ge=1, le=MAX_CELLS)
    unit_size: float = Field(default=4.0, gt=0)
    wall_thickness: float = Field(default=0.8, gt=0)
    block_height: float = Field(default=3.0, gt=0)
    pin_diameter: float = Field(default=2.6, gt=0)
    pin_height: float = Field(default=2.0, gt=0)
    pins_enabled: bool = True
    tubes_enabled: bool = True
    tube_inner_diameter: float = Field(default=1.2, gt=0)
    slope_rise_units: int = Field(default=0, ge=0, le=MAX_SLOPE_RISE)
    slope_side: SlopeSide = "+X"
    excluded_cells: list[list[int]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @field_validator("unit_size", "wall_thickness", "block_height",
                     "pin_diameter", "pin_height", "tube_inner_diameter")
    @classmethod
    def _validate_positive_finite(cls, value: float) -> float:
        '''Reject non-finite values (NaN/inf slip past ``gt=0``).'''
        if not value < float("inf") or value != value:
            raise SettingsValidationError("dimension must be a finite positive number")
        return value

    @field_validator("slope_side")
    @classmethod
    def _validate_slope_side(cls, value: str) -> str:
        '''Restrict the slope side to the supported set.'''
        if value not in SLOPE_SIDES:
            raise SettingsValidationError(
                f"slope_side must be one of {', '.join(SLOPE_SIDES)}"
            )
        return value

    @field_validator("excluded_cells")
    @classmethod
    def _validate_excluded_cells(cls, value: list) -> list[list[int]]:
        '''Normalize the exclusion mask to a sorted list of [ix, iy] pairs.'''
        result: list[list[int]] = []
        for item in value:
            try:
                ix, iy = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise SettingsValidationError(
                    f"excluded cell {item!r} must be an [ix, iy] pair"
                ) from exc
            result.append([ix, iy])
        return sorted(result)

    @property
    def tube_outer_diameter(self) -> float:
        '''Outer tube diameter: bore plus two wall-thickness shells, mm.'''
        return self.tube_inner_diameter + 2.0 * self.wall_thickness

    @property
    def has_tubes(self) -> bool:
        '''True when tubes apply: enabled and both cell counts are 2+.'''
        return self.tubes_enabled and self.cells_x >= 2 and self.cells_y >= 2

    @property
    def is_rectangular(self) -> bool:
        '''True when no cells are excluded from the figure.'''
        return not self.excluded_cells

    @property
    def included_count(self) -> int:
        '''Number of cells present in the figure.'''
        return self.cells_x * self.cells_y - len(self.excluded_cells)

    @property
    def slope_active(self) -> bool:
        '''True when the sloped-top variant applies (no studs, rise > 0).'''
        return not self.pins_enabled and self.slope_rise_units > 0

    @property
    def slope_rise(self) -> float:
        '''Absolute rise of the raised edge above the top face, mm.'''
        return self.slope_rise_units * self.block_height

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
        if self.has_tubes and not self.slope_active:
            cavity_min = min(
                self.outer_length_x, self.outer_length_y
            ) - 2.0 * self.wall_thickness
            if self.tube_outer_diameter >= cavity_min:
                raise SettingsValidationError(
                    "tube_outer_diameter must fit inside the cavity"
                )
        seen: set[tuple[int, int]] = set()
        for ix, iy in self.excluded_cells:
            if not (0 <= ix < self.cells_x and 0 <= iy < self.cells_y):
                raise SettingsValidationError(
                    f"excluded cell [{ix}, {iy}] is outside the {self.cells_x}x"
                    f"{self.cells_y} grid"
                )
            if (ix, iy) in seen:
                raise SettingsValidationError(
                    f"excluded cell [{ix}, {iy}] is listed twice"
                )
            seen.add((ix, iy))
        if len(self.excluded_cells) >= self.cells_x * self.cells_y:
            raise SettingsValidationError("at least one cell must remain in the figure")
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
        '''Total height including studs or the raised slope edge, mm.'''
        if self.pins_enabled:
            return self.block_height + self.pin_height
        return self.block_height + self.slope_rise

    def describe(self) -> str:
        '''Human-readable form label, e.g. ``2x6``.'''
        return f"{self.cells_x}x{self.cells_y}"


__all__ = ["BlockSettings", "validate_settings", "SlopeSide", "SLOPE_SIDES"]
