'''
File:   project.py
Brief:  Project and placed-block models for the multi-block GUI scene.
Author: Mistress-Lukutar
Date:   2026-11-20
Version: v0.13.0
'''

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.models import MAX_CELLS, BlockSettings, validate_settings

# Factory default block color (matches the preview BLOCK_COLOR).
DEFAULT_BLOCK_COLOR = "#3f7fbf"

# BlockSettings fields describing geometry shared by the whole project
# (identical for every block) versus per-cell layout owned by each instance.
SHARED_SETTING_FIELDS: tuple[str, ...] = (
    "unit_size",
    "wall_thickness",
    "block_height",
    "clearance_xy",
    "clearance_z",
    "pins_enabled",
    "tubes_enabled",
)
LAYOUT_FIELDS: tuple[str, ...] = (
    "excluded_cells",
    "flat_cells",
    "stud_cells",
    "sloped_cells",
    "chamfered_cells",
    "round_cells",
)

_QUANTITY_MAX = 100000
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class BlockInstance(BaseModel):
    '''A single placed block within a project scene.

    Each instance carries its own grid size, per-cell layout, scene position,
    color, and quantity (for BOM tracking). Shared geometry is inherited from
    the parent :class:`Project.settings`.

    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    cells_x: int = Field(default=1, ge=1, le=MAX_CELLS)
    cells_y: int = Field(default=1, ge=1, le=MAX_CELLS)
    position_x: float = Field(
        default=0.0,
        description=(
            "Scene offset of the block center along X (mm); blocks are built "
            "centered on the origin."
        ),
    )
    position_y: float = Field(
        default=0.0,
        description=(
            "Scene offset of the block center along Y (mm); blocks are built "
            "centered on the origin."
        ),
    )
    color: str = Field(
        default=DEFAULT_BLOCK_COLOR,
        description="'#rrggbb' hex preview color for visual distinction.",
    )
    quantity: int = Field(
        default=1,
        description=(
            "How many physical copies this placed block represents, for BOM tracking."
        ),
    )
    excluded_cells: list[list[int]] = Field(
        default_factory=list,
        description="See BlockSettings.excluded_cells.",
    )
    flat_cells: list[list[int]] = Field(
        default_factory=list,
        description="See BlockSettings.flat_cells.",
    )
    stud_cells: list[list[int]] = Field(
        default_factory=list,
        description="See BlockSettings.stud_cells.",
    )
    sloped_cells: list[list] = Field(
        default_factory=list,
        description="See BlockSettings.sloped_cells.",
    )
    chamfered_cells: list[list] = Field(
        default_factory=list,
        description="See BlockSettings.chamfered_cells.",
    )
    round_cells: list[list[int]] = Field(
        default_factory=list,
        description="See BlockSettings.round_cells.",
    )

    @field_validator("quantity")
    @classmethod
    def _validate_quantity(cls, value: int) -> int:
        '''Wrap pydantic bounds in SettingsValidationError.'''
        if not (1 <= value <= _QUANTITY_MAX):
            raise SettingsValidationError(
                f"quantity must be between 1 and {_QUANTITY_MAX}"
            )
        return value

    @field_validator("color")
    @classmethod
    def _validate_color(cls, value: str) -> str:
        '''Ensure color is a valid ``#rrggbb`` hex string.'''
        if not _COLOR_RE.match(value):
            raise SettingsValidationError("color must be a '#rrggbb' hex value")
        return value

    @field_validator("position_x", "position_y")
    @classmethod
    def _validate_position(cls, value: float) -> float:
        '''Reject non-finite position values (NaN / inf).'''
        if not value < float("inf") or value != value:
            raise SettingsValidationError("position must be a finite number")
        return value

    @model_validator(mode="after")
    def _validate_layout(self) -> BlockInstance:
        '''Reuse all BlockSettings per-cell rules against this grid size.'''
        # Explicit bounds check for sloped cells (BlockSettings does not
        # call _check_cell_bounds for the sloped set, only flat/stud/round).
        for cell in self.sloped_cells:
            ix, iy = int(cell[0]), int(cell[1])
            if not (0 <= ix < self.cells_x and 0 <= iy < self.cells_y):
                raise SettingsValidationError(
                    f"sloped cell [{ix}, {iy}] is outside the "
                    f"{self.cells_x}x{self.cells_y} grid"
                )
        values = {
            "cells_x": self.cells_x,
            "cells_y": self.cells_y,
        }
        for field_name in LAYOUT_FIELDS:
            values[field_name] = getattr(self, field_name)
        try:
            BlockSettings.model_validate(values)
        except Exception as exc:  # noqa: BLE001
            raise SettingsValidationError(str(exc)) from exc
        return self

    def to_settings(self, shared: BlockSettings) -> BlockSettings:
        '''Merge the shared project geometry with this block's grid and layout.

        Args:
            shared: The project-level ``BlockSettings`` carrying shared
                geometry (unit_size, wall_thickness, etc.).

        Returns:
            A fully-populated ``BlockSettings`` instance combining shared
            geometry and this instance's per-cell layout.
        '''
        values = {field: getattr(shared, field) for field in SHARED_SETTING_FIELDS}
        values["cells_x"] = self.cells_x
        values["cells_y"] = self.cells_y
        values.update({field: getattr(self, field) for field in LAYOUT_FIELDS})
        return validate_settings(values)


class Project(BaseModel):
    '''A multi-block project scene.

    A project holds shared block geometry (``settings``) and a list of
    placed block instances, each with its own grid size, per-cell layout,
    scene position, color, and quantity.

    Convention: ``settings`` carries **shared geometry only**; its per-cell
    lists are unused (kept empty). Per-instance layout lives in the
    :class:`BlockInstance` objects.

    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    settings: BlockSettings = Field(
        default_factory=BlockSettings,
        description=(
            "Shared block geometry for the whole project. Carries shared "
            "geometry only; per-cell lists are unused (kept empty)."
        ),
    )
    blocks: list[BlockInstance] = Field(
        default_factory=list,
        description="List of placed block instances in the scene.",
    )

    def block_settings(self, index: int) -> BlockSettings:
        '''Full BlockSettings of the placed block at *index*.

        Args:
            index: Zero-based index into ``blocks``.

        Returns:
            A merged ``BlockSettings`` combining shared geometry with the
            instance's per-cell layout.
        '''
        return self.blocks[index].to_settings(self.settings)

    def total_quantity(self) -> int:
        '''Sum of every block's quantity (total BOM count).'''
        return sum(b.quantity for b in self.blocks)


__all__ = [
    "Project",
    "BlockInstance",
    "DEFAULT_BLOCK_COLOR",
    "SHARED_SETTING_FIELDS",
    "LAYOUT_FIELDS",
]
