'''
File:   models.py
Brief:  Pydantic settings model describing the parametric block geometry.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from brickgen.core.exceptions import SettingsValidationError

MAX_CELLS = 64  # sanity limit per axis for GUI/CLI input
MAX_SLOPE_RISE = 16  # sanity limit for the slope rise in unit heights

# Outer tube diameter factor: the tube surface keeps clearance_xy/2 to the
# four surrounding stud surfaces along the stud-tube center line (stud
# centers sit at unit_size*sqrt(2)/2 from the tube vertex), so the clearance
# cancels out and the diameter depends only on unit and wall thickness.
_TUBE_PRESS_RATIO: float = math.sqrt(2.0) - 1.0

# Side of the raised edge that forms a slope (top) or a chamfer (bottom).
SlopeSide = Literal["+X", "-X", "+Y", "-Y"]
SLOPE_SIDES: tuple[str, ...] = ("+X", "-X", "+Y", "-Y")

# Millimeter dimensions encoded into recipe_name() as ``token + value``
# when they deviate from the factory default (token, field name).
_RECIPE_MM_FIELDS: tuple[tuple[str, str], ...] = (
    ("u", "unit_size"),
    ("w", "wall_thickness"),
    ("h", "block_height"),
    ("cx", "clearance_xy"),
    ("cz", "clearance_z"),
)


def _cell_entries(prefix: str, cells: list[list]) -> str:
    '''Encode a per-cell group as ``prefix`` plus ``;``-joined entries.

    Each entry is its comma-separated cell record (e.g. ``ix,iy`` or
    ``ix,iy,side,rise``); the validators keep the lists sorted, so the
    encoding is deterministic.

    Args:
        prefix: Short group tag (``ex``/``fl``/``st``/``sl``/``ch``/``rd``).
        cells: The cell records to encode.

    Returns:
        The group token, e.g. ``ex0,1;2,1``.
    '''
    return prefix + ";".join(",".join(str(v) for v in cell) for cell in cells)


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
    '''Parametric description of a hollow stud block.

    The block is a hollow rectangular shell without a bottom: outer footprint
    ``cells_x * unit_size`` by ``cells_y * unit_size``, wall thickness
    ``wall_thickness``, open from below, with a solid top face. For blocks of
    at least 2x2 cells, hollow reinforcement tubes are derived (outer diameter
    depends only on the unit and wall thickness) and placed at the intersections
    of the mid-lines between stud rows/columns; their role is to press the
    surrounding studs toward the cavity walls (they replace the missing wall at
    inner lattice vertices), with a bore that only saves plastic.

    Every 1x1 cell carries a per-cell top feature and an optional bottom
    chamfer:

    - ``stud`` - cylindrical stud on the top face. This is the default top
      for plain cells when ``pins_enabled`` is true; ``stud_cells`` /
      ``flat_cells`` override the default per cell.
    - ``sloped cell`` - top wedge: the ``side`` edge of the cell rises by
      ``rise x block_height`` and the cell top slopes down to the opposite
      edge. The cell keeps the standard bottom opening (only chamfered
      cells are solid), so a slope combines with the usual hollow shell.
    - ``chamfered cell`` - ``[ix, iy, side]`` triplet: the cell bottom is a
      solid wedge thinned toward ``side`` (no cavity opening). ``side`` must
      look outward (neighbor across it excluded or outside the grid). The
      chamfer makes the cell solid, so a sloped+chamfered cell has no
      bottom opening either.
    - ``round cell`` - the square base keeps its straight edges wherever a
      material neighbor joins (adjacent round cells form the classic
      stadium profile) and rounds its free corners with the inscribed
      circle. The bottom hole is concentric and round, inset by the wall
      thickness, and never fuses with neighboring cavities. A round cell
      cannot carry a slope or a chamfer.
    - ``excluded cell`` - removed from the figure entirely.

    Exclusions, chamfers, slopes or round cells switch the builder to the
    figure variant: contour walls follow the shape, concave corners get
    inward quarter-round ribs instead of tubes.

    Attributes:
        cells_x: Number of cells along X (block length in cells).
        cells_y: Number of cells along Y (block width in cells).
        unit_size: Footprint of a single 1x1 cell in mm.
        wall_thickness: Shell wall thickness in mm (also used for tubes).
        block_height: Height of the shell from bottom rim to top face, mm.
        clearance_xy: Fit clearance in the X/Y plane, mm. Positive: the stud
            diameter shrinks by this value relative to the cell socket
            (``unit_size - 2*wall_thickness``) so stacked blocks slide together.
            Negative: the stud is modeled fatter than the socket by
            ``|clearance_xy|`` - an interference fit compensating print
            shrinkage; the block dimensions stay nominal.
        clearance_z: Vertical fit clearance, mm. Positive: the stud height
            shrinks by this value relative to the cavity depth
            (``block_height - wall_thickness``) so studs never bottom out.
            Negative: the stud is modeled taller than the cavity depth by
            ``|clearance_z|`` (shrinkage compensation); the block dimensions
            stay nominal.
        pins_enabled: Default top for plain cells: studs. Cells listed in
            ``flat_cells``/``stud_cells`` override this per cell; sloped and
            excluded cells ignore it.
        tubes_enabled: Whether to place inner reinforcement tubes (2x2+ blocks).
        excluded_cells: 0-based ``[ix, iy]`` cells removed from the figure
            (top view, ix along X, iy along Y).
        flat_cells: ``[ix, iy]`` cells explicitly without a stud (plain top).
        stud_cells: ``[ix, iy]`` cells explicitly with a stud (used when
            ``pins_enabled`` is false).
        sloped_cells: ``[ix, iy, side, rise]`` quadruplets for cells whose
            top is a wedge: the ``side`` edge rises by
            ``rise x block_height``. ``rise`` is 1..MAX_SLOPE_RISE.
        chamfered_cells: ``[ix, iy, side]`` triplets for cells whose bottom
            is chamfered (solid wedge, no cavity opening). A cell may be
            chamfered on several sides; combined freely with any top.
        round_cells: ``[ix, iy]`` cells whose base keeps straight edges only
            toward material neighbors (free corners follow the inscribed
            circle) with a concentric round bottom hole.
    '''

    cells_x: int = Field(default=1, ge=1, le=MAX_CELLS)
    cells_y: int = Field(default=1, ge=1, le=MAX_CELLS)
    unit_size: float = Field(default=4.0, gt=0)
    wall_thickness: float = Field(default=0.8, gt=0)
    block_height: float = Field(default=3.0, gt=0)
    clearance_xy: float = Field(default=-0.25)
    clearance_z: float = Field(default=0.2)
    pins_enabled: bool = True
    tubes_enabled: bool = True
    excluded_cells: list[list[int]] = Field(default_factory=list)
    flat_cells: list[list[int]] = Field(default_factory=list)
    stud_cells: list[list[int]] = Field(default_factory=list)
    sloped_cells: list[list] = Field(default_factory=list)
    chamfered_cells: list[list] = Field(default_factory=list)
    round_cells: list[list[int]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_settings(cls, values: object) -> object:
        '''Map v1 preset keys and drop derived-v3 legacy keys.

        v1 stored bottom chamfers in ``sloped_cells`` as ``[ix, iy, side]``
        triplets (now ``chamfered_cells``) and described the stud-less
        whole-block slope with global ``slope_side``/``slope_rise_units``.
        Triplet entries are moved to ``chamfered_cells``; a positive global
        rise on a stud-less block is expanded into per-cell slopes covering
        the whole grid.  v3 derived ``pin_diameter``, ``pin_height``, and
        ``tube_inner_diameter`` from fit-clearance fields, so old presets that
        still carry those keys are silently stripped (``extra="forbid"`` must
        not reject them).
        '''
        if not isinstance(values, dict):
            return values
        values = dict(values)
        legacy_slope = values.pop("sloped_cells", None)
        if isinstance(legacy_slope, list):
            triplets: list[list] = []
            quadruplets: list[list] = []
            for item in legacy_slope:
                if isinstance(item, (list, tuple)) and len(item) >= 4:
                    quadruplets.append(list(item))
                else:
                    triplets.append(list(item))
            if triplets:
                chamfers = list(values.get("chamfered_cells", []))
                values["chamfered_cells"] = chamfers + triplets
            if quadruplets:
                values["sloped_cells"] = quadruplets
        rise = values.pop("slope_rise_units", None)
        side = values.pop("slope_side", None)
        if (
            rise is not None
            and side is not None
            and not values.get("sloped_cells")
            and not values.get("pins_enabled", True)
        ):
            # v1 whole-block slope: every cell gets the same wedge.
            cells_x = int(values.get("cells_x", 1))
            cells_y = int(values.get("cells_y", 1))
            values["sloped_cells"] = [
                [ix, iy, str(side), int(rise)]
                for iy in range(cells_y)
                for ix in range(cells_x)
            ]
        # v3: pin_diameter, pin_height, tube_inner_diameter are now derived.
        values.pop("pin_diameter", None)
        values.pop("pin_height", None)
        values.pop("tube_inner_diameter", None)
        return values

    @field_validator(
        "unit_size",
        "wall_thickness",
        "block_height",
        "clearance_xy",
        "clearance_z",
    )
    @classmethod
    def _validate_finite(cls, value: float) -> float:
        '''Reject non-finite values (NaN/inf slip past the Field bounds).'''
        if not value < float("inf") or value != value:
            raise SettingsValidationError("dimension must be a finite number")
        return value

    @field_validator("sloped_cells")
    @classmethod
    def _validate_sloped_cells(cls, value: list) -> list[list]:
        '''Normalize the top-slope list to sorted [ix, iy, side, rise] quads.'''
        result: list[list] = []
        seen: set[tuple[int, int]] = set()
        for item in value:
            try:
                ix, iy = int(item[0]), int(item[1])
                side = str(item[2])
                rise = int(item[3])
            except (TypeError, ValueError, IndexError) as exc:
                raise SettingsValidationError(
                    f"sloped cell {item!r} must be an [ix, iy, side, rise] quadruplet"
                ) from exc
            if side not in SLOPE_SIDES:
                raise SettingsValidationError(
                    f"sloped cell side '{side}' must be one of {', '.join(SLOPE_SIDES)}"
                )
            if not 1 <= rise <= MAX_SLOPE_RISE:
                raise SettingsValidationError(
                    f"sloped cell rise must be between 1 and {MAX_SLOPE_RISE}"
                )
            if (ix, iy) in seen:
                raise SettingsValidationError(
                    f"sloped cell [{ix}, {iy}] is listed twice"
                )
            seen.add((ix, iy))
            result.append([ix, iy, side, rise])
        return sorted(result)

    @field_validator("chamfered_cells")
    @classmethod
    def _validate_chamfered_cells(cls, value: list) -> list[list]:
        '''Normalize the bottom-chamfer list to sorted [ix, iy, side] triplets.'''
        result: list[list] = []
        seen: set[tuple[int, int, str]] = set()
        for item in value:
            try:
                triplet = (int(item[0]), int(item[1]), str(item[2]))
            except (TypeError, ValueError, IndexError) as exc:
                raise SettingsValidationError(
                    f"chamfered cell {item!r} must be an [ix, iy, side] triplet"
                ) from exc
            if triplet[2] not in SLOPE_SIDES:
                raise SettingsValidationError(
                    f"chamfered cell side '{triplet[2]}' must be one of "
                    f"{', '.join(SLOPE_SIDES)}"
                )
            if triplet in seen:
                raise SettingsValidationError(
                    f"chamfered cell {triplet[:2]} with side '{triplet[2]}' "
                    "is listed twice"
                )
            seen.add(triplet)
            result.append([triplet[0], triplet[1], triplet[2]])
        return sorted(result)

    @field_validator("excluded_cells")
    @classmethod
    def _validate_excluded_cells(cls, value: list) -> list[list[int]]:
        '''Normalize the exclusion mask to a sorted list of [ix, iy] pairs.'''
        return cls._normalize_cell_pairs(value, "excluded cell")

    @field_validator("flat_cells")
    @classmethod
    def _validate_flat_cells(cls, value: list) -> list[list[int]]:
        '''Normalize the explicit no-stud list to sorted [ix, iy] pairs.'''
        return cls._normalize_cell_pairs(value, "flat cell")

    @field_validator("stud_cells")
    @classmethod
    def _validate_stud_cells(cls, value: list) -> list[list[int]]:
        '''Normalize the explicit stud list to sorted [ix, iy] pairs.'''
        return cls._normalize_cell_pairs(value, "stud cell")

    @field_validator("round_cells")
    @classmethod
    def _validate_round_cells(cls, value: list) -> list[list[int]]:
        '''Normalize the round-base list to sorted [ix, iy] pairs.'''
        return cls._normalize_cell_pairs(value, "round cell")

    @staticmethod
    def _normalize_cell_pairs(value: list, label: str) -> list[list[int]]:
        '''Normalize a raw cell list to sorted [ix, iy] integer pairs.

        Duplicate cells are rejected so preset typos surface early.
        '''
        result: list[list[int]] = []
        seen: set[tuple[int, int]] = set()
        for item in value:
            try:
                ix, iy = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise SettingsValidationError(
                    f"{label} {item!r} must be an [ix, iy] pair"
                ) from exc
            if (ix, iy) in seen:
                raise SettingsValidationError(f"{label} [{ix}, {iy}] is listed twice")
            seen.add((ix, iy))
            result.append([ix, iy])
        return sorted(result)

    @property
    def cavity_depth(self) -> float:
        '''Internal cavity depth from the bottom rim to the ceiling, mm.'''
        return self.block_height - self.wall_thickness

    @property
    def pin_diameter(self) -> float:
        '''Derived stud diameter: cell socket minus the X/Y clearance, mm.

        The socket a stud enters is ``unit_size - 2*wall_thickness`` wide (the
        same hole a round cell gets); positive ``clearance_xy`` leaves that
        much diametral play. A negative clearance models the stud fatter than
        the socket - an interference fit compensating print shrinkage; the
        block dimensions stay nominal.
        '''
        return self.unit_size - 2.0 * self.wall_thickness - self.clearance_xy

    @property
    def pin_height(self) -> float:
        '''Derived stud height: cavity depth minus the Z clearance, mm.

        When a block is stacked onto the studs, its rim rests on the host top
        face while each stud top stays ``clearance_z`` below the cavity
        ceiling. A negative clearance models the stud taller than the cavity
        depth (shrinkage compensation); the block dimensions stay nominal.
        '''
        return self.cavity_depth - self.clearance_z

    @property
    def tube_outer_diameter(self) -> float:
        '''Derived outer tube diameter, mm.

        The tube presses the four surrounding studs toward the cavity walls:
        its surface keeps ``clearance_xy / 2`` to each stud surface along the
        stud-tube center line, so the clearance cancels out of the formula and
        the diameter depends only on the unit and wall thickness. Under a
        negative clearance the gap turns into an interference that the shrunk
        printed stud takes up.
        '''
        return _TUBE_PRESS_RATIO * self.unit_size + 2.0 * self.wall_thickness

    @property
    def tube_inner_diameter(self) -> float:
        '''Derived tube bore diameter: outer minus two wall shells, mm.

        The bore only saves plastic; at very small units it collapses towards
        zero and the builder fuses a solid rod instead.
        '''
        return self.tube_outer_diameter - 2.0 * self.wall_thickness

    @property
    def has_tubes(self) -> bool:
        '''True when tubes apply: enabled and both cell counts are 2+.'''
        return self.tubes_enabled and self.cells_x >= 2 and self.cells_y >= 2

    @property
    def is_rectangular(self) -> bool:
        '''True when the figure is the full rectangle (no per-cell features).'''
        return not (
            self.excluded_cells
            or self.chamfered_cells
            or self.sloped_cells
            or self.round_cells
        )

    @property
    def has_bottom_chamfers(self) -> bool:
        '''True when at least one cell has a chamfered bottom.'''
        return bool(self.chamfered_cells)

    @property
    def has_sloped_tops(self) -> bool:
        '''True when at least one cell has a sloped top wedge.'''
        return bool(self.sloped_cells)

    @property
    def has_round_cells(self) -> bool:
        '''True when at least one cell has a round (cylinder) base.'''
        return bool(self.round_cells)

    @property
    def max_slope_rise_units(self) -> int:
        '''Largest per-cell slope rise in unit heights (0 when none).'''
        return max((int(item[3]) for item in self.sloped_cells), default=0)

    @property
    def included_count(self) -> int:
        '''Number of cells present in the figure.'''
        return self.cells_x * self.cells_y - len(self.excluded_cells)

    def cell_top(self, ix: int, iy: int) -> str:
        '''Effective top feature of a cell: ``"stud"``, ``"flat"`` or ``"slope"``.

        Sloped cells always win; explicit ``flat_cells``/``stud_cells``
        override the ``pins_enabled`` default; excluded cells are not
        material and report ``"flat"``.
        '''
        if (ix, iy) in self.slope_cell_set():
            return "slope"
        if (ix, iy) in self.flat_cell_set():
            return "flat"
        if (ix, iy) in self.stud_cell_set() or self.pins_enabled:
            return "stud"
        return "flat"

    def slope_cell_set(self) -> set[tuple[int, int]]:
        '''Set of (ix, iy) cells with a sloped top wedge.'''
        return {(int(item[0]), int(item[1])) for item in self.sloped_cells}

    def round_cell_set(self) -> set[tuple[int, int]]:
        '''Set of (ix, iy) cells with a round (cylinder) base.'''
        return {(int(item[0]), int(item[1])) for item in self.round_cells}

    def flat_cell_set(self) -> set[tuple[int, int]]:
        '''Set of (ix, iy) cells explicitly without a stud.'''
        return {(int(item[0]), int(item[1])) for item in self.flat_cells}

    def stud_cell_set(self) -> set[tuple[int, int]]:
        '''Set of (ix, iy) cells explicitly with a stud.'''
        return {(int(item[0]), int(item[1])) for item in self.stud_cells}

    def excluded_cell_set(self) -> set[tuple[int, int]]:
        '''Set of (ix, iy) cells removed from the figure.'''
        return {(int(item[0]), int(item[1])) for item in self.excluded_cells}

    def stud_cells_all(self) -> set[tuple[int, int]]:
        '''Set of (ix, iy) cells that carry a stud (default or explicit).'''
        result: set[tuple[int, int]] = set()
        excluded = self.excluded_cell_set()
        sloped = self.slope_cell_set()
        flat = self.flat_cell_set()
        explicit = self.stud_cell_set()
        for iy in range(self.cells_y):
            for ix in range(self.cells_x):
                if (ix, iy) in excluded or (ix, iy) in sloped or (ix, iy) in flat:
                    continue
                if (ix, iy) in explicit or self.pins_enabled:
                    result.add((ix, iy))
        return result

    @property
    def has_studs(self) -> bool:
        '''True when at least one cell carries a stud (default or explicit).'''
        return bool(self.stud_cells_all())

    @property
    def slope_rise(self) -> float:
        '''Absolute rise of the tallest slope edge above the top face, mm.'''
        return self.max_slope_rise_units * self.block_height

    @model_validator(mode="after")
    def _validate_geometry(self) -> BlockSettings:
        '''Reject cross-field conflicts (no cavity, oversized stud/tube).'''
        if 2.0 * self.wall_thickness >= self.unit_size:
            raise SettingsValidationError(
                "wall_thickness*2 must be smaller than unit_size to keep a cavity"
            )
        if self.block_height <= self.wall_thickness:
            raise SettingsValidationError(
                "block_height must exceed wall_thickness "
                "(top face keeps wall thickness)"
            )
        socket = self.unit_size - 2.0 * self.wall_thickness
        if not -2.0 * self.wall_thickness < self.clearance_xy < socket:
            raise SettingsValidationError(
                "clearance_xy must stay within (-2*wall, unit_size - 2*wall) "
                "so the stud stays positive and neighboring studs do not overlap"
            )
        depth = self.block_height - self.wall_thickness
        if not -depth < self.clearance_z < depth:
            raise SettingsValidationError(
                "|clearance_z| must stay below the cavity depth "
                "(block_height - wall_thickness)"
            )
        if self.has_tubes:
            cavity_min = (
                min(self.outer_length_x, self.outer_length_y)
                - 2.0 * self.wall_thickness
            )
            if self.tube_outer_diameter >= cavity_min:
                raise SettingsValidationError(
                    "tube_outer_diameter must fit inside the cavity"
                )
        excluded = self.excluded_cell_set()
        sloped = self.slope_cell_set()
        flat = self.flat_cell_set()
        stud = self.stud_cell_set()
        round_cells = self.round_cell_set()
        chamfered = {(int(item[0]), int(item[1])) for item in self.chamfered_cells}
        for ix, iy in excluded:
            if not (0 <= ix < self.cells_x and 0 <= iy < self.cells_y):
                raise SettingsValidationError(
                    f"excluded cell [{ix}, {iy}] is outside the {self.cells_x}x"
                    f"{self.cells_y} grid"
                )
        self._check_cell_bounds(flat, "flat")
        self._check_cell_bounds(stud, "stud")
        self._check_cell_bounds(round_cells, "round")
        for ix, iy in sloped | flat | stud | round_cells:
            if (ix, iy) in excluded:
                raise SettingsValidationError(
                    f"cell [{ix}, {iy}] cannot be both excluded and featured"
                )
        conflict = flat & stud
        if conflict:
            ix, iy = min(conflict)
            raise SettingsValidationError(
                f"cell [{ix}, {iy}] cannot be both flat and stud"
            )
        conflict = sloped & flat
        if conflict:
            ix, iy = min(conflict)
            raise SettingsValidationError(
                f"cell [{ix}, {iy}] cannot be both sloped and flat"
            )
        conflict = sloped & stud
        if conflict:
            ix, iy = min(conflict)
            raise SettingsValidationError(
                f"cell [{ix}, {iy}] cannot be both sloped and stud: a wedge "
                "top cannot carry a stud"
            )
        conflict = round_cells & sloped
        if conflict:
            ix, iy = min(conflict)
            raise SettingsValidationError(
                f"cell [{ix}, {iy}] cannot be both round and sloped: a wedge "
                "top needs a square base"
            )
        conflict = round_cells & chamfered
        if conflict:
            ix, iy = min(conflict)
            raise SettingsValidationError(
                f"cell [{ix}, {iy}] cannot be both round and chamfered"
            )
        side_offsets = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}
        for ix, iy, side in self.chamfered_cells:
            if not (0 <= ix < self.cells_x and 0 <= iy < self.cells_y):
                raise SettingsValidationError(
                    f"chamfered cell [{ix}, {iy}] is outside the {self.cells_x}x"
                    f"{self.cells_y} grid"
                )
            if (ix, iy) in excluded:
                raise SettingsValidationError(
                    f"cell [{ix}, {iy}] cannot be both excluded and chamfered"
                )
            dx, dy = side_offsets[side]
            nx, ny = ix + dx, iy + dy
            neighbor_out = not (0 <= nx < self.cells_x and 0 <= ny < self.cells_y)
            if not neighbor_out and (nx, ny) not in excluded:
                raise SettingsValidationError(
                    f"chamfered cell [{ix}, {iy}] side '{side}' must look outward "
                    "(neighbor across it must be excluded or outside the grid)"
                )
        if len(self.excluded_cells) >= self.cells_x * self.cells_y:
            raise SettingsValidationError("at least one cell must remain in the figure")
        return self

    def _check_cell_bounds(self, cells: set[tuple[int, int]], label: str) -> None:
        '''Raise when an override cell sits outside the grid.'''
        for ix, iy in cells:
            if not (0 <= ix < self.cells_x and 0 <= iy < self.cells_y):
                raise SettingsValidationError(
                    f"{label} cell [{ix}, {iy}] is outside the {self.cells_x}x"
                    f"{self.cells_y} grid"
                )

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
        '''Total height: block height plus studs or the tallest slope, mm.'''
        return self.block_height + max(
            self.pin_height if self.has_studs else 0.0, self.slope_rise
        )

    def describe(self) -> str:
        '''Human-readable form label, e.g. ``2x6``.'''
        return f"{self.cells_x}x{self.cells_y}"

    def recipe_name(self) -> str:
        '''Compact export-file stem encoding the full geometry recipe.

        The name keeps blocks that would otherwise silently overwrite each
        other apart (``2x2-studs`` vs ``2x2-studless``): dimensions, top
        type, tube state, every per-cell feature with its coordinates and
        sides, and any millimeter dimension deviating from the factory
        defaults. Tokens that carry no geometry are omitted, so common names
        stay short:

        - ``2x2-studs`` / ``2x2-studless`` - plain blocks;
        - ``1x2-studs_sl0,1,+Y,1`` - the top-slope showcase brick;
        - ``2x4-studs_ex0,1;2,1_h4.5`` - exclusions and a taller block.

        All characters are file-name safe on Windows.

        Returns:
            The deterministic file stem for this recipe.
        '''
        head: list[str] = [
            self.describe(),
            "studs" if self.pins_enabled else "studless",
        ]
        if not self.tubes_enabled and self.cells_x >= 2 and self.cells_y >= 2:
            head.append("notubes")
        tail: list[str] = []
        for token, field in _RECIPE_MM_FIELDS:
            default = BlockSettings.model_fields[field].default
            value = getattr(self, field)
            if value != default:
                tail.append(f"{token}{value:g}")
        if self.excluded_cells:
            tail.append(_cell_entries("ex", self.excluded_cells))
        if self.flat_cells:
            tail.append(_cell_entries("fl", self.flat_cells))
        if self.stud_cells:
            tail.append(_cell_entries("st", self.stud_cells))
        if self.sloped_cells:
            tail.append(_cell_entries("sl", self.sloped_cells))
        if self.chamfered_cells:
            tail.append(_cell_entries("ch", self.chamfered_cells))
        if self.round_cells:
            tail.append(_cell_entries("rd", self.round_cells))
        name = "-".join(head)
        if tail:
            name += "_" + "_".join(tail)
        return name


__all__ = ["BlockSettings", "validate_settings", "SlopeSide", "SLOPE_SIDES"]
