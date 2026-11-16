# BrickGen — Parametric Stud-Block Generator

BrickGen generates simple construction blocks (hollow shells with studs on top)
from parametric settings, with an interactive 3D preview GUI and multi-format
CAD export.

**Stack:** Python ≥ 3.11 · CadQuery (OpenCASCADE, exact B-Rep geometry) →
STEP/STL/AMF/3MF/SVG/VRML/BREP · GUI: PySide6 + PyVista/VTK (OpenGL render) ·
CLI: Typer · settings: Pydantic v2 + JSON presets.

## Block Geometry

The block is a hollow box **without a bottom** — an open cavity underneath, a
solid top wall, and one cylindrical stud per 1×1 cell:

| Parameter | Default | Description |
|---|---:|---|
| `cells_x` / `cells_y` | 1 / 1 | Shape size in cells (`2x6` shape → `cells_x=2`, `cells_y=6`) |
| `unit_size` | 4.0 mm | 1×1 cell footprint (4×4 mm square) |
| `wall_thickness` | 0.8 mm | Wall, top-wall and tube thickness |
| `block_height` | 3.0 mm | Shell height from bottom rim to top face |
| `pin_diameter` | 2.6 mm | Stud diameter |
| `pin_height` | 2.0 mm | Stud height above the top face |
| `tube_inner_diameter` | 1.2 mm | Inner tube bore diameter |
| `pins_enabled` | true | Top studs on/off |
| `tubes_enabled` | true | Inner tubes on/off |
| `slope_rise_units` | 0 | Slope rise in unit heights (stud-less blocks only) |
| `slope_side` | `+X` | Raised slope edge: `+X`/`-X`/`+Y`/`-Y` |
| `excluded_cells` | — | 0-based `[ix, iy]` cells removed from the figure |
| `sloped_cells` | — | `[ix, iy, side]` cells with a chamfered (wedge) bottom |

- Footprint: `cells_x × unit_size` by `cells_y × unit_size`; Z=0 is the bottom
  plane; the cavity depth is `block_height − wall_thickness`.
- **Inner tubes** (2×2 and larger): hollow cylinders at the mid-line
  intersections between stud rows/columns — one tube per sliding 2×2 cell
  group (2×2 → 1 center tube; 3×2 → 2 tubes along the length). They span the
  full cavity depth and are closed by the top wall; disabled via
  `tubes_enabled`.
- **Figure blocks** (`excluded_cells`): the shell walls contour the remaining
  shape; concave corners get inward quarter-round ribs (radius = 2 × wall
  thickness) instead of tubes. Studs remain on every cell that still has
  material.
- **Bottom chamfers** (`sloped_cells`): the cell's bottom becomes a solid
  wedge tapering from zero thickness at the outward face to full height —
  the cell has no cavity opening there. Allowed only on faces that look
  outward (the neighbor across the face is excluded or outside the grid);
  a corner cell may be chamfered on two sides. Chamfers switch the builder
  to the figure variant; tubes do not apply.
- **Sloped top** (stud-less blocks only): one top edge rises by
  `slope_rise_units × block_height` and the top face slopes down to the
  opposite edge, forming a wedge. The cavity ceiling keeps a constant
  vertical top-wall thickness at any rise, so it never breaks through the
  slope (regression-tested). Studs and tubes do not apply.

## Quick Start (Windows)

```bat
start.bat
```

Checks the `.venv` and imports; if anything is missing it runs
`scripts\setup.ps1` (Python 3.11+ via the `py` launcher, `.venv`,
`pip install -e ".[dev]"`), then launches the GUI through `pythonw` (no
console window). Subsequent launches are instant.

### Manual Install

```bash
pip install -e ".[dev]"        # with dev tools (pytest, ruff, mypy)
# or runtime-only:
pip install cadquery pydantic numpy pyvista pyvistaqt PySide6 typer
```

## GUI

```bash
python -m brickgen.gui.app     # or, after install: brickgen-gui
```

Features:

- Parameter form; the model rebuilds **automatically** after any change
  (250 ms debounce) — there is no Generate button.
- **Layout (top view)** grid editor: click a cell to cycle
  included → excluded → bottom-chamfered (arrow marks the outward chamfer
  side; extra clicks add a second outward side on corner cells);
  *Include all cells* resets the grid.
- **Slope group** for stud-less blocks: rise in unit heights and the raised
  edge side (`+X/-X/+Y/-Y`).
- **Template library** on the right: cards with rendered previews; click
  applies a template, *Save as template…* stores the current parameters.
  Built-in templates ship with the package
  (`src/brickgen/data/templates/`, 13 shapes); user templates live in
  `Documents/BrickGen/templates/`.
- **Display modes:** Solid · Solid + feature edges (part-boundary outlines:
  stud base circles, tube bores, bottom rim) · Solid + wireframe · Wireframe.
  Wireframe lines are the edges of the tessellated mesh itself, not B-Rep
  geometry; mode switches only toggle actors and never rebuild geometry.
- **Semantic coloring:** the mesh is classified per part (body, studs, tube
  bores, bottom rim) and shaded accordingly; SSAO is enabled when the GPU
  supports it.
- *Show grid* — metric axis grid (mm) around the model.
- *Export model…* — format is chosen by the file extension.
- *Save settings… / Load settings…* — JSON presets.
- *Settings* — preferences dialog: preview tessellation tolerances (linear +
  angular), export mesh tolerances (STL/AMF/3MF), default export format.
  Persisted in `Documents/BrickGen/settings.json`.
- Status bar with the actual bounding-box size and shape label (e.g. `2x6`).

All heavy work (model build, meshing) runs in a **worker process**
(`gui/jobs.py`), so the UI stays responsive even on large blocks (up to
64×64 cells); meshes cross the process boundary as numpy arrays.

## CLI

```bash
python -m brickgen.cli.main generate -x 2 -y 6 block_2x6.step
python -m brickgen.cli.main generate --settings preset.json block_1x4.stl
python -m brickgen.cli.main generate -x 3 -y 2 --exclude "1,1" corner.step
python -m brickgen.cli.main generate -x 2 -y 2 --no-pins --slope-rise 2 --slope-side +X wedge.step
python -m brickgen.cli.main generate -x 2 -y 2 --slope-cells "1,0,+X;0,1,+Y" chamfers.step
python -m brickgen.cli.main formats          # list supported formats
```

Options: `--cells-x/-x`, `--cells-y/-y`, `--unit-size`, `--wall`, `--height`,
`--pin-diameter`, `--pin-height`, `--no-pins`, `--no-tubes`, `--tube-bore`,
`--slope-rise`, `--slope-side`, `--exclude "ix,iy;..."`,
`--slope-cells "ix,iy,side;..."`, `--settings/-s preset.json` (explicit CLI
options always win over preset values). The export format is derived from the
output extension (`step`, `stl`, `amf`, `3mf`, `svg`, `vrml`, `brep` —
case-insensitive).

## JSON Preset

```json
{
  "schema": {"name": "brickgen-settings", "version": 1},
  "cells_x": 2,
  "cells_y": 6,
  "unit_size": 4.0,
  "wall_thickness": 0.7,
  "block_height": 8.0,
  "pin_diameter": 2.6,
  "pin_height": 2.0,
  "pins_enabled": true
}
```

The `schema` block is optional on load and always written on save. Unknown
keys (`extra: forbid`) and out-of-range values are rejected by validation,
e.g. `wall_thickness * 2 < unit_size`, `pin_diameter < unit_size`,
`block_height > wall_thickness`, tube must fit the cavity.

## Project Layout (src-layout)

```
start.bat / scripts/setup.ps1   # Windows bootstrap and GUI launcher
tools/bump-version.py           # version/date bump across the project
src/brickgen/
├── core/                       # pure geometry domain (no Qt)
│   ├── models.py               # BlockSettings (pydantic) + validate_settings()
│   ├── builder.py              # CadQuery block construction
│   └── exceptions.py           # BrickGenError hierarchy
├── services/                   # orchestration (no Qt)
│   ├── export_service.py       # STEP/STL/AMF/3MF/SVG/VRML/BREP export
│   ├── settings_io.py          # JSON preset load/save
│   ├── template_library.py     # built-in + user templates
│   └── app_settings.py         # persistent GUI preferences
├── gui/                        # PySide6 application
│   ├── app.py                  # main window
│   ├── grid_editor.py          # top-view cell grid editor
│   ├── preview.py              # PyVista/VTK preview (modes, coloring, SSAO)
│   ├── jobs.py / jobtasks.py   # process-based background job queue
│   ├── library.py              # template panel
│   └── preferences.py          # preferences dialog
├── cli/main.py                 # Typer CLI
└── data/templates/*.json       # built-in block templates (package data)
```

## Development

All commands go through the project venv (`.venv/Scripts/python.exe` on
Windows); `ruff`/`uv` are not assumed on PATH:

```bash
.venv/Scripts/python.exe -m pytest -q          # unit tests, no network/hardware
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --config "format.quote-style='preserve'" src tests
.venv/Scripts/python.exe -m mypy src
.venv/Scripts/python.exe tools/bump-version.py --bump patch
```

Tests verify analytically: bounding-box dimensions, solid volume (shell +
studs), open-bottom invariant, studs fused into a single body, settings
validation, JSON preset roundtrip, and export file writing.

Conventions: single-quote (`'''`) docstrings with a
`File/Brief/Author/Date/Version` header in every file; Google pydocstyle;
ruff line length 88, rules E/F/W/I/D. Never run plain `ruff format` — the
default rewrites all docstring quotes.

## Assumptions

1. One stud per 1×1 cell, centered; toggleable via `pins_enabled`.
2. The top wall is solid and has the wall thickness; the bottom edge is
   open (no bottom plate).
3. Inner tubes are placed automatically for shapes ≥ 2×2 at mid-line
   intersections, spanning the full cavity depth.
4. B-Rep exports (STEP/BREP) are exact geometry; STL/AMF/3MF tessellate with
   configurable linear/angular tolerances (0.1 mm / 0.1 rad defaults); the
   preview tolerance (0.15 mm default) is set in Preferences.
