'''
File:   main.py
Brief:  Typer CLI entry point for headless project export of print layouts.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import logging
from pathlib import Path

import typer

from brickgen.core.exceptions import BrickGenError
from brickgen.services.export_service import SUPPORTED_FORMATS, export_model
from brickgen.services.placement_service import sync_pieces
from brickgen.services.print_export import bed_compound, collect_all_beds
from brickgen.services.project_io import load_project

app = typer.Typer(
    help="BrickGen - parametric stud block generator (headless mode).",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)


@app.command()
def export(
    project_file: Path = typer.Argument(..., help="BrickGen project (*.json)."),
    out_dir: Path = typer.Option(
        Path.cwd(), "--out-dir", "-o", help="Output directory for the bed files."
    ),
    file_format: str = typer.Option(
        "STL", "--format", "-f", help="Export format (step/stl/amf/3mf/svg/vrml/brep)."
    ),
    beds: str = typer.Option(
        "",
        "--beds",
        help="Comma-separated 1-based bed numbers to export "
        "(default: every bed with pieces).",
    ),
) -> None:
    '''Export a project's print placement: one file per bed.

    The pieces are reconciled with the blocks first (sync), so freshly
    migrated projects export correctly without the GUI. Each bed becomes
    ``<stem>_bedNN.<ext>``; pieces outside their bed bounds are skipped and
    reported.
    '''
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        fmt = file_format.upper()
        if fmt not in SUPPORTED_FORMATS:
            raise typer.BadParameter(
                f"Unsupported format '{file_format}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        # Requested beds (1-based display numbers -> 0-based indices).
        requested: set[int] | None = None
        if beds.strip():
            requested = set()
            for part in beds.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    number = int(part)
                except ValueError as exc:
                    raise typer.BadParameter(
                        f"Invalid --beds item '{part}': bed numbers must be integers"
                    ) from exc
                if number < 1:
                    raise typer.BadParameter(
                        f"Invalid --beds item '{part}': bed numbers are 1-based"
                    )
                requested.add(number - 1)

        project = load_project(project_file)
        sync_pieces(project)
        if not project.print_layout.pieces:
            typer.secho(
                "Error: project has no print pieces.", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)

        beds_payload, skipped = collect_all_beds(project)
        if not beds_payload:
            typer.secho(
                "Error: no pieces are placed on any bed.", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = project_file.stem
        written: list[Path] = []
        for bed in beds_payload:
            if requested is not None and bed["index"] not in requested:
                continue
            path = out_dir / f"{stem}_bed{bed['index'] + 1:02d}.{fmt.lower()}"
            compound = bed_compound(bed["pieces"])
            written.append(
                export_model(
                    compound,
                    path,
                    fmt,
                    tolerance=0.1,
                    angular_tolerance=0.1,
                )
            )
            typer.secho(
                f"Bed {bed['index'] + 1}: {len(bed['pieces'])} piece(s) -> {path}",
                fg=typer.colors.GREEN,
            )

        if requested is not None:
            missing = sorted(
                n + 1 for n in requested if n not in {b["index"] for b in beds_payload}
            )
            for number in missing:
                typer.secho(
                    f"Bed {number}: no pieces to export (skipped).",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
        if skipped:
            typer.secho(
                f"{skipped} piece(s) outside their bed bounds were skipped.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        typer.secho(f"Exported {len(written)} bed file(s).", fg=typer.colors.GREEN)
    except typer.BadParameter:
        raise
    except BrickGenError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def formats() -> None:
    '''List supported export formats.'''
    typer.echo(", ".join(sorted(SUPPORTED_FORMATS)))


if __name__ == "__main__":
    app()
