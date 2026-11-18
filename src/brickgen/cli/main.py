'''
File:   main.py
Brief:  Typer CLI entry point for headless block generation and export.
Author: Mistress-Lukutar
Date:   2026-11-17
Version: v0.10.0
'''

from __future__ import annotations

import logging
from pathlib import Path

import typer

from brickgen.core.builder import build_block
from brickgen.core.exceptions import BrickGenError
from brickgen.core.models import SLOPE_SIDES, BlockSettings, validate_settings
from brickgen.services.export_service import SUPPORTED_FORMATS, export_model
from brickgen.services.settings_io import load_settings

app = typer.Typer(
    help="BrickGen - parametric stud block generator (headless mode).",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)


def _parse_cell_list(spec: str, arity: int, flag: str) -> list[list]:
    '''Parse a cell list spec like ``"0,1;2,1"`` into fixed-arity entries.

    Args:
        spec: Comma/semicolon-separated ``ix,iy[,side[,rise]]`` entries.
        arity: Expected number of comma-separated fields per entry.
        flag: Flag name used in error messages.

    Returns:
        List of parsed entries (ints, upper-cased side strings; empty when
        the spec is blank).

    Raises:
        typer.BadParameter: If an entry is malformed.
    '''
    shapes = {
        2: "ix,iy",
        3: "ix,iy,side",
        4: "ix,iy,side,rise",
    }
    entries: list[list] = []
    for part in spec.replace(":", ",").split(";"):
        fields = [p.strip() for p in part.split(",") if p.strip()]
        if not fields:
            continue
        if len(fields) != arity:
            raise typer.BadParameter(
                f"Invalid --{flag} item '{part.strip()}', "
                f"expected '{shapes.get(arity, 'ix,iy')}'"
            )
        try:
            values: list[int | str] = [int(field) for field in fields[:2]]
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid --{flag} item '{part.strip()}': ix/iy must be integers"
            ) from exc
        if arity >= 3:
            side = fields[2].upper()
            if side not in SLOPE_SIDES:
                raise typer.BadParameter(
                    f"Invalid --{flag} side '{fields[2]}', "
                    f"expected one of {'/'.join(SLOPE_SIDES)}"
                )
            values.append(side)
        if arity == 4:
            try:
                values.append(int(fields[3]))
            except ValueError as exc:
                raise typer.BadParameter(
                    f"Invalid --{flag} item '{part.strip()}': rise must be an integer"
                ) from exc
        entries.append(values)
    return entries


@app.command()
def generate(
    cells_x: int = typer.Option(
        1, "--cells-x", "-x", min=1, max=64, help="Cells along X (length)."
    ),
    cells_y: int = typer.Option(
        1, "--cells-y", "-y", min=1, max=64, help="Cells along Y (width)."
    ),
    unit_size: float = typer.Option(4.0, "--unit-size", help="1x1 cell footprint, mm."),
    wall: float = typer.Option(0.8, "--wall", help="Wall thickness, mm."),
    height: float = typer.Option(3.0, "--height", help="Block height, mm."),
    pin_diameter: float = typer.Option(
        2.6, "--pin-diameter", help="Stud diameter, mm."
    ),
    pin_height: float = typer.Option(2.0, "--pin-height", help="Stud height, mm."),
    no_pins: bool = typer.Option(
        False,
        "--no-pins",
        help="Omit top studs on plain cells (explicit --stud-cells still apply).",
    ),
    no_tubes: bool = typer.Option(False, "--no-tubes", help="Omit inner tubes."),
    tube_inner_diameter: float = typer.Option(
        1.2, "--tube-bore", help="Inner tube bore diameter, mm."
    ),
    exclude: str = typer.Option(
        "", "--exclude", help="Cells to remove, e.g. '0,1;2,1' (ix,iy from 0)."
    ),
    flat_cells: str = typer.Option(
        "", "--flat-cells", help="Explicit stud-less plain cells 'ix,iy;...'."
    ),
    stud_cells: str = typer.Option(
        "", "--stud-cells", help="Explicit stud cells 'ix,iy;...' (with --no-pins)."
    ),
    slope_cells: str = typer.Option(
        "",
        "--slope-cells",
        help="Top-slope wedge cells 'ix,iy,side,rise;...', side "
        f"{'/'.join(SLOPE_SIDES)}, rise in unit heights.",
    ),
    chamfer_cells: str = typer.Option(
        "",
        "--chamfer-cells",
        help=f"Bottom-chamfered cells 'ix,iy,side;...', side {'/'.join(SLOPE_SIDES)}.",
    ),
    round_cells: str = typer.Option(
        "",
        "--round-cells",
        help="Round-base cells 'ix,iy;...' (inscribed cylinder, round hole).",
    ),
    settings_file: Path | None = typer.Option(
        None, "--settings", "-s", help="Load parameters from a JSON preset first."
    ),
    output: Path = typer.Argument(..., help="Output file (extension picks format)."),
) -> None:
    '''Generate a block and export it to the file given by ``output``.

    The export format is derived from the output extension
    (step/stl/amf/3mf/svg/vrml/brep, case-insensitive).
    '''
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        if settings_file is not None:
            settings = load_settings(settings_file)
        else:
            settings = BlockSettings()
        # Explicit CLI values always win over the preset defaults.
        overrides = {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "unit_size": unit_size,
            "wall_thickness": wall,
            "block_height": height,
            "pin_diameter": pin_diameter,
            "pin_height": pin_height,
            "pins_enabled": not no_pins,
            "tubes_enabled": not no_tubes,
            "tube_inner_diameter": tube_inner_diameter,
            "excluded_cells": _parse_cell_list(exclude, 2, "exclude"),
            "flat_cells": _parse_cell_list(flat_cells, 2, "flat-cells"),
            "stud_cells": _parse_cell_list(stud_cells, 2, "stud-cells"),
            "sloped_cells": _parse_cell_list(slope_cells, 4, "slope-cells"),
            "chamfered_cells": _parse_cell_list(chamfer_cells, 3, "chamfer-cells"),
            "round_cells": _parse_cell_list(round_cells, 2, "round-cells"),
        }
        settings = settings.model_copy(update=overrides)
        settings = validate_settings(settings.model_dump())
        model = build_block(settings)
        written = export_model(model, output, output.suffix.lstrip(".") or "STEP")
    except BrickGenError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"Generated {settings.recipe_name()} -> {written}",
        fg=typer.colors.GREEN,
    )


@app.command()
def formats() -> None:
    '''List supported export formats.'''
    typer.echo(", ".join(sorted(SUPPORTED_FORMATS)))


if __name__ == "__main__":
    app()
