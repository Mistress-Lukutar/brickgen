'''
File:   project.py
Brief:  Project and placed-block models for the multi-block GUI scene.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.models import MAX_CELLS, BlockSettings, validate_settings

# Factory default block color (matches the preview BLOCK_COLOR).
DEFAULT_BLOCK_COLOR = "#3f7fbf"

# BlockSettings fields describing geometry shared by the whole project
# (identical for every block) versus block-level data owned by each
# instance (grid size, stud/tube defaults, per-cell layout).
SHARED_SETTING_FIELDS: tuple[str, ...] = (
    "unit_size",
    "wall_thickness",
    "block_height",
    "clearance_xy",
    "clearance_z",
)
# Block-level BlockSettings flags carried by every instance/template.
BLOCK_FLAG_FIELDS: tuple[str, ...] = (
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

# Anchor side of the per-block BOM quantity label (relative to each block).
QuantityLabelPosition = Literal["left", "right", "top", "bottom"]
QUANTITY_LABEL_POSITIONS: tuple[str, ...] = ("left", "right", "top", "bottom")

# Grid cell scale presets offered by the Grid menu: the actual grid step is
# ``unit_size * grid_scale`` mm (1.0 = one cell per Unit Size).
GRID_SCALE_PRESETS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

_GRID_SCALE_MAX = 1000.0

# Export format names offered by the placement-tab export panel. Plain
# strings only: the cadquery ExportTypes mapping lives in
# ``services.export_service.SUPPORTED_FORMATS`` and must stay keyed by
# exactly these names (core never imports services).
EXPORT_FORMATS: tuple[str, ...] = ("STEP", "STL", "AMF", "3MF", "SVG", "VRML", "BREP")

# Print-placement piece.bed value meaning "not placed on any bed".
PRINT_STAGING_BED = -1
_BED_COUNT_MAX = 64
_BED_SIZE_MIN, _BED_SIZE_MAX = 10.0, 2000.0
_GAP_MIN, _GAP_MAX = 0.0, 100.0


def _validate_finite(value: float, message: str) -> float:
    '''Reject non-finite values (NaN / inf).

    Args:
        value: The numeric value to check.
        message: Error message to raise on failure.

    Returns:
        The original *value* when finite.

    Raises:
        SettingsValidationError: When *value* is NaN or infinite.
    '''
    if not value < float("inf") or value != value:
        raise SettingsValidationError(message)
    return value


class BlockLayout(BaseModel):
    '''Scene-independent definition of one block: grid size, stud/tube
    defaults and the per-cell layout lists.

    This is everything a template stores and everything a placed instance
    contributes on top of the shared project geometry. ``pins_enabled``
    sets the default top of cells without an explicit per-cell assignment;
    ``tubes_enabled`` toggles the inner tubes of 2x2+ blocks.

    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    cells_x: int = Field(default=1, ge=1, le=MAX_CELLS)
    cells_y: int = Field(default=1, ge=1, le=MAX_CELLS)
    pins_enabled: bool = Field(
        default=True,
        description=(
            "Default top for cells without an explicit per-cell assignment: "
            "True puts a stud on every plain cell."
        ),
    )
    tubes_enabled: bool = Field(
        default=True,
        description="Build the inner tubes (centers of 2x2+ blocks) of this block.",
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

    @model_validator(mode="after")
    def _validate_layout(self) -> BlockLayout:
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
        '''Merge the shared project geometry with this block's definition.

        The block's own ``pins_enabled``/``tubes_enabled`` flags win over
        the shared settings, which only contribute the numeric geometry.

        Args:
            shared: The project-level ``BlockSettings`` carrying shared
                geometry (unit_size, wall_thickness, etc.).

        Returns:
            A fully-populated ``BlockSettings`` instance combining shared
            geometry and this block's grid, flags and per-cell layout.
        '''
        values = {field: getattr(shared, field) for field in SHARED_SETTING_FIELDS}
        values["cells_x"] = self.cells_x
        values["cells_y"] = self.cells_y
        values.update({field: getattr(self, field) for field in BLOCK_FLAG_FIELDS})
        values.update({field: getattr(self, field) for field in LAYOUT_FIELDS})
        return validate_settings(values)


class BlockInstance(BlockLayout):
    '''A single placed block within a project scene.

    Each instance carries its own grid size, stud/tube defaults, per-cell
    layout, scene position, color, and quantity (for BOM tracking). Shared
    geometry is inherited from the parent :class:`Project.settings`.
    '''

    uid: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description=(
            "Stable unique id linking print-placement pieces to this block "
            "across edits."
        ),
    )
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
            "How many physical copies this placed block represents, for BOM "
            "tracking. 0 is allowed for blocks whose quantity is linked to "
            "the Assemble tab before their first placed copy."
        ),
    )
    quantity_linked: bool = Field(
        default=False,
        description=(
            "True derives ``quantity`` from the number of this block's "
            "copies placed on the Assemble tab; False keeps the manual "
            "BOM quantity."
        ),
    )

    @field_validator("uid")
    @classmethod
    def _validate_uid(cls, value: str) -> str:
        '''Ensure uid is a non-empty stripped string.'''
        if not value.strip():
            raise SettingsValidationError("uid must be a non-empty string")
        return value

    @field_validator("quantity")
    @classmethod
    def _validate_quantity(cls, value: int) -> int:
        '''Wrap pydantic bounds in SettingsValidationError.'''
        if not (0 <= value <= _QUANTITY_MAX):
            raise SettingsValidationError(
                f"quantity must be between 0 and {_QUANTITY_MAX}"
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
        return _validate_finite(value, "position must be a finite number")


class PrintBedSettings(BaseModel):
    '''Print-bed configuration shared by every bed of the project.

    All beds are identical rectangles; ``gap`` is the spacing used by
    auto-placement both between pieces and from the bed edges.
    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    width: float = Field(
        default=220.0,
        ge=_BED_SIZE_MIN,
        le=_BED_SIZE_MAX,
        description="Bed X extent, mm.",
    )
    depth: float = Field(
        default=220.0,
        ge=_BED_SIZE_MIN,
        le=_BED_SIZE_MAX,
        description="Bed Y extent, mm.",
    )
    count: int = Field(
        default=1,
        ge=1,
        le=_BED_COUNT_MAX,
        description="Number of print beds.",
    )
    gap: float = Field(
        default=1.0,
        ge=_GAP_MIN,
        le=_GAP_MAX,
        description="Spacing between pieces and to bed edges, mm.",
    )


class PrintPiece(BaseModel):
    '''One physical copy of a block on the print-placement tabs.

    A block with quantity N produces N pieces (copy index is implicit in
    the list order). ``bed`` is -1 (staging zone) or a bed index; the
    position is the piece footprint CENTER in bed-local mm (staging-local
    when staged). ``flipped`` rotates the piece 180 degrees about the X
    axis so the pins point down.
    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    uid: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Stable piece id.",
    )
    block_uid: str = Field(
        description=("uid of the BlockInstance this piece is a copy of."),
    )
    bed: int = Field(
        default=PRINT_STAGING_BED,
        description="Bed index, or -1 for the staging zone.",
    )
    position_x: float = Field(
        default=0.0,
        description="Piece footprint center, zone-local X, mm.",
    )
    position_y: float = Field(
        default=0.0,
        description="Piece footprint center, zone-local Y, mm.",
    )
    flipped: bool = Field(
        default=False,
        description="180-deg rotation: pins point down.",
    )
    rotated90: bool = Field(
        default=False,
        description=(
            "Piece rotated 90 degrees about Z by the packing algorithm; "
            "the footprint swaps (depth becomes X)."
        ),
    )

    @field_validator("uid", "block_uid")
    @classmethod
    def _validate_uid_field(cls, value: str) -> str:
        '''Ensure uid / block_uid are non-empty stripped strings.'''
        if not value.strip():
            raise SettingsValidationError("uid and block_uid must be non-empty strings")
        return value

    @field_validator("bed")
    @classmethod
    def _validate_bed(cls, value: int) -> int:
        '''Reject bed values below PRINT_STAGING_BED.'''
        if value < PRINT_STAGING_BED:
            raise SettingsValidationError("bed must be -1 (staging) or a bed index")
        return value

    @field_validator("position_x", "position_y")
    @classmethod
    def _validate_position(cls, value: float) -> float:
        '''Reject non-finite position values (NaN / inf).'''
        return _validate_finite(value, "position must be a finite number")


class PrintToolsSettings(BaseModel):
    '''Print-placement tools state: auto-place bed range and modifiers.

    The bed range is stored 1-based exactly as the tools panel displays it
    and is normalized (clamped into the existing beds, from <= to) by
    :class:`PrintLayout`, so the persisted values are always usable.
    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    auto_from: int = Field(
        default=1,
        ge=1,
        le=_BED_COUNT_MAX,
        description="First bed (1-based) of the auto-place range.",
    )
    auto_to: int = Field(
        default=1,
        ge=1,
        le=_BED_COUNT_MAX,
        description="Last bed (1-based) of the auto-place range.",
    )
    distribute_by_color: bool = Field(
        default=True,
        description=(
            "Auto-place never mixes piece colors on one bed (one print job per color)."
        ),
    )
    distribute_evenly: bool = Field(
        default=True,
        description=(
            "Auto-place spreads pieces across all selected beds instead of "
            "filling one bed before moving to the next."
        ),
    )


class ExportSettings(BaseModel):
    '''Per-project print-layout export configuration (placement tab panel).

    Everything the placement-tab export panel edits: the mesh precision,
    the output file format and whether each bed is written to its own
    file or every bed is combined into one file. The values travel inside
    the project file, so every project restores its own export defaults
    on open. The angular mesh tolerance stays application-level
    (``AppSettings.export_angular_tolerance``).

    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    format: str = Field(
        default="STL",
        description="Export file format (one of ``EXPORT_FORMATS``).",
    )
    tolerance: float = Field(
        default=0.1,
        gt=0,
        le=1.0,
        description="Mesh tessellation tolerance for tessellated formats, mm.",
    )
    per_bed_files: bool = Field(
        default=True,
        description=(
            "True writes one file per bed (<stem>_bedNN.ext); False writes "
            "every bed's pieces into a single file (<stem>.ext) at their "
            "world positions."
        ),
    )

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        '''Restrict the export format to the supported name set.'''
        normalized = value.strip().upper()
        if normalized not in EXPORT_FORMATS:
            raise SettingsValidationError(
                f"format must be one of {', '.join(EXPORT_FORMATS)}"
            )
        return normalized


class PrintLayout(BaseModel):
    '''Print-placement state of a project: bed configuration, pieces and
    the auto-place tools state.'''

    model_config = {"extra": "forbid"}

    beds: PrintBedSettings = Field(
        default_factory=PrintBedSettings,
        description="Print-bed configuration (all beds identical).",
    )
    pieces: list[PrintPiece] = Field(
        default_factory=list,
        description="Placement pieces: one per physical copy of a block.",
    )
    tools: PrintToolsSettings = Field(
        default_factory=PrintToolsSettings,
        description="Placement tools state: auto-place range and modifiers.",
    )

    @model_validator(mode="after")
    def _validate_pieces(self) -> PrintLayout:
        '''Every piece.bed must reference an existing bed or staging; the
        tools bed range is normalized into the existing beds.'''
        for piece in self.pieces:
            if not (piece.bed == PRINT_STAGING_BED or 0 <= piece.bed < self.beds.count):
                raise SettingsValidationError(
                    f"piece {piece.uid} references bed {piece.bed}, "
                    f"but only {self.beds.count} beds exist"
                )
        # A shrunken bed count (or hand-edited JSON) can leave the stored
        # auto-place range outside the existing beds: clamp it instead of
        # failing the whole project.
        auto_from = max(1, min(self.tools.auto_from, self.beds.count))
        auto_to = max(auto_from, min(self.tools.auto_to, self.beds.count))
        if (auto_from, auto_to) != (self.tools.auto_from, self.tools.auto_to):
            self.tools = self.tools.model_copy(
                update={"auto_from": auto_from, "auto_to": auto_to}
            )
        return self


class AssembledPiece(BaseModel):
    '''One placed copy of a block on the Assemble tab.

    ``block_uid`` references the :class:`BlockInstance` the copy is built
    from (the copy inherits the block's geometry and color). The position
    is the piece footprint CENTER in mm; ``position_z`` is the height of
    the piece BOTTOM (0 = on the floor). ``rot_z`` counts 90-degree turns
    about the Z axis (0..3, footprint swaps on odd values); ``flipped``
    turns the piece 180 degrees about the X axis (pins down) and lifts it
    by the block height, matching the print-placement flip semantics.
    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    uid: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Stable assembled-piece id.",
    )
    block_uid: str = Field(
        description="uid of the BlockInstance this assembled copy is built from."
    )
    position_x: float = Field(
        default=0.0,
        description="Piece footprint center X, mm.",
    )
    position_y: float = Field(
        default=0.0,
        description="Piece footprint center Y, mm.",
    )
    position_z: float = Field(
        default=0.0,
        description="Height of the piece bottom above the floor, mm.",
    )
    rot_z: int = Field(
        default=0,
        description="Quarter turns (90 degrees each) about the Z axis, 0..3.",
    )
    flipped: bool = Field(
        default=False,
        description="180-deg rotation about X: pins point down.",
    )

    @field_validator("uid", "block_uid")
    @classmethod
    def _validate_uid_field(cls, value: str) -> str:
        '''Ensure uid / block_uid are non-empty stripped strings.'''
        if not value.strip():
            raise SettingsValidationError("uid and block_uid must be non-empty strings")
        return value

    @field_validator("rot_z")
    @classmethod
    def _validate_rot_z(cls, value: int) -> int:
        '''Restrict the quarter-turn count to the 0..3 range.'''
        if not 0 <= value <= 3:
            raise SettingsValidationError("rot_z must be a quarter-turn count 0..3")
        return value

    @field_validator("position_x", "position_y", "position_z")
    @classmethod
    def _validate_position(cls, value: float) -> float:
        '''Reject non-finite position values (NaN / inf).'''
        return _validate_finite(value, "position must be a finite number")


class AssembleViewSettings(BaseModel):
    '''Per-project Assemble-tab view preferences (snapping, floor grid).

    The floor grid reuses the shared ``ViewSettings.grid_scale`` cell scale
    (one Grid menu scale for every viewport); only the assemble-specific
    toggles live here.
    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    grid_enabled: bool = Field(
        default=True,
        description="Show the XY floor grid in the assemble viewport.",
    )
    snap_to_grid: bool = Field(
        default=True,
        description="Round piece placement to the grid step.",
    )
    snap_to_parts: bool = Field(
        default=True,
        description=(
            "Magnetize pieces to other assembled pieces: drop onto the "
            "topmost support under the footprint and align flush to the "
            "sides of nearby pieces."
        ),
    )
    show_gizmo: bool = Field(
        default=True,
        description=(
            "Show the move gizmo (draggable axis arrows) in the assemble viewport."
        ),
    )


class AssembleState(BaseModel):
    '''Assemble-tab state of a project: placed copies and view settings.'''

    model_config = {"extra": "forbid"}

    pieces: list[AssembledPiece] = Field(
        default_factory=list,
        description="Copies placed on the assemble scene.",
    )
    view: AssembleViewSettings = Field(
        default_factory=AssembleViewSettings,
        description="Assemble viewport snapping and grid preferences.",
    )


class ViewSettings(BaseModel):
    '''Per-project view preferences (View/Grid menu state).

    Everything the user toggles through the View menu (BOM quantity labels
    and their anchor side) and the Grid menu (floor grid, cell scale,
    snap-to-grid) — for both the planning preview and the print-placement
    viewport — lives here and is saved with the project file, so every
    project restores its own view state on open.

    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    grid_enabled: bool = Field(
        default=False,
        description="Show the XY floor grid in the planning preview.",
    )
    grid_scale: float = Field(
        default=0.5,
        gt=0,
        le=_GRID_SCALE_MAX,
        description=(
            "Grid cell scale factor; the grid step is ``unit_size * grid_scale`` "
            "mm (1.0 = one cell per Unit Size)."
        ),
    )
    grid_snap: bool = Field(
        default=False,
        description="Snap block movement to the grid step.",
    )
    placement_grid_enabled: bool = Field(
        default=False,
        description="Show the floor grid on the print-placement viewport.",
    )
    show_gizmo: bool = Field(
        default=True,
        description=(
            "Show the move gizmo (draggable axis arrows) in the planning and "
            "print-placement viewports."
        ),
    )
    quantity_labels_enabled: bool = Field(
        default=False,
        description="Show the BOM quantity label next to every placed block.",
    )
    quantity_label_position: QuantityLabelPosition = Field(
        default="right",
        description="Anchor side of the quantity label, relative to each block.",
    )


class Project(BaseModel):
    '''A multi-block project scene.

    A project holds shared block geometry (``settings``), a list of placed
    block instances, each with its own grid size, per-cell layout, scene
    position, color, and quantity, plus the per-project view settings
    (grids, snapping, BOM labels).

    Convention: ``settings`` carries **shared geometry only**; its per-cell
    lists and stud/tube flags are unused. Per-instance layout and flags
    live in the :class:`BlockInstance` objects.

    Model config:
        extra: "forbid" — rejects unknown keys to catch payload typos.
    '''

    model_config = {"extra": "forbid"}

    settings: BlockSettings = Field(
        default_factory=BlockSettings,
        description=(
            "Shared block geometry for the whole project. Carries shared "
            "geometry only; per-cell lists and stud/tube flags are unused."
        ),
    )
    blocks: list[BlockInstance] = Field(
        default_factory=list,
        description="List of placed block instances in the scene.",
    )
    print_layout: PrintLayout = Field(
        default_factory=PrintLayout,
        description="Print-placement state: bed configuration and pieces.",
    )
    assemble: AssembleState = Field(
        default_factory=AssembleState,
        description="Assemble-tab state: placed copies and view settings.",
    )
    view: ViewSettings = Field(
        default_factory=ViewSettings,
        description=(
            "Per-project view settings: floor grids, cell scale, snap and "
            "BOM quantity labels (View/Grid menu state)."
        ),
    )
    export: ExportSettings = Field(
        default_factory=ExportSettings,
        description=(
            "Per-project export configuration: mesh precision, file format "
            "and per-bed vs single-file output (placement export panel)."
        ),
    )

    def block_settings(self, index: int) -> BlockSettings:
        '''Full BlockSettings of the placed block at *index*.

        Args:
            index: Zero-based index into ``blocks``.

        Returns:
            A merged ``BlockSettings`` combining shared geometry with the
            instance's flags and per-cell layout.
        '''
        return self.blocks[index].to_settings(self.settings)

    def total_quantity(self) -> int:
        '''Sum of every block's quantity (total BOM count).'''
        return sum(b.quantity for b in self.blocks)


__all__ = [
    "Project",
    "BlockLayout",
    "BlockInstance",
    "DEFAULT_BLOCK_COLOR",
    "SHARED_SETTING_FIELDS",
    "BLOCK_FLAG_FIELDS",
    "LAYOUT_FIELDS",
    "PRINT_STAGING_BED",
    "PrintBedSettings",
    "PrintPiece",
    "PrintToolsSettings",
    "PrintLayout",
    "AssembledPiece",
    "AssembleViewSettings",
    "AssembleState",
    "ViewSettings",
    "ExportSettings",
    "EXPORT_FORMATS",
    "QuantityLabelPosition",
    "QUANTITY_LABEL_POSITIONS",
    "GRID_SCALE_PRESETS",
]
