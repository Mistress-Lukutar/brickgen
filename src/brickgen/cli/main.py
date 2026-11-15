'''
File:   main.py
Brief:  Typer CLI entry point for headless block generation and export.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.4.1
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
    help="BrickGen - parametric LEGO-like block generator (headless mode).",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)


@app.command()
def generate(
    cells_x: int = typer.Option(1, "--cells-x", "-x", min=1, max=64,
                                help="Cells along X (length)."),
    cells_y: int = typer.Option(1, "--cells-y", "-y", min=1, max=64,
                                help="Cells along Y (width)."),
    unit_size: float = typer.Option(4.0, "--unit-size", help="1x1 cell footprint, mm."),
    wall: float = typer.Option(0.8, "--wall", help="Wall thickness, mm."),
    height: float = typer.Option(3.0, "--height", help="Block height, mm."),
    pin_diameter: float = typer.Option(2.6, "--pin-diameter", help="Stud diameter, mm."),
    pin_height: float = typer.Option(2.0, "--pin-height", help="Stud height, mm."),
    no_pins: bool = typer.Option(False, "--no-pins", help="Omit top studs."),
    no_tubes: bool = typer.Option(False, "--no-tubes", help="Omit inner tubes."),
    tube_inner_diameter: float = typer.Option(
        1.2, "--tube-bore", help="Inner tube bore diameter, mm."
    ),
    slope_rise: int = typer.Option(
        0, "--slope-rise", min=0, max=16,
        help="Slope rise in unit heights (stud-less blocks only, 0 = off).",
    ),
    slope_side: str = typer.Option(
        "+X", "--slope-side", help=f"Raised slope edge: {'/'.join(SLOPE_SIDES)}."
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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
            "slope_rise_units": slope_rise,
            "slope_side": slope_side,
        }
        settings = settings.model_copy(update=overrides)
        settings = validate_settings(settings.model_dump())
        model = build_block(settings)
        written = export_model(model, output, output.suffix.lstrip(".") or "STEP")
    except BrickGenError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(
        f"Generated {settings.describe()} block -> {written}",
        fg=typer.colors.GREEN,
    )


@app.command()
def formats() -> None:
    '''List supported export formats.'''
    typer.echo(", ".join(sorted(SUPPORTED_FORMATS)))


if __name__ == "__main__":
    app()
