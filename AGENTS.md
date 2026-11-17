# BrickGen — Agent Instructions

Parametric stud-block (LEGO-like) generator built on CadQuery/OCC, with a PySide6 + PyVista GUI, Typer CLI, and multi-format export (STEP/STL/AMF/3MF/SVG/VRML/BREP). Windows-first; requires Python ≥3.11. No README/docs — this file is the primary guide.

## Layout

- `src/brickgen/core/` — geometry domain: `models.py` (Pydantic `BlockSettings`, validators, `MAX_CELLS=64` sanity limits), `builder.py` (CadQuery geometry), `exceptions.py`. No Qt imports allowed here.
- `src/brickgen/services/` — export, template library, settings I/O. No Qt.
- `src/brickgen/gui/` — PySide6 + PyVistaQt UI (`app.py`, `preview.py`), background jobs (`jobs.py` process queue, `jobtasks.py` worker-side tasks).
- `src/brickgen/cli/` — Typer headless entry (`brickgen` script); GUI entry is `brickgen-gui` / `python -m brickgen.gui.app`.
- `src/brickgen/data/templates/*.json` — built-in block templates (packaged as package data).
- `tests/unit/` — pytest; `tools/bump-version.py` — version/date bump across files; `scripts/setup.ps1` + `start.bat` — venv bootstrap and GUI launcher.

## Commands

`uv` and `ruff` are NOT on PATH; system Python lacks the deps. Always use the venv (Git Bash paths):

```bash
.venv/Scripts/python.exe -m pytest -q                    # all tests
.venv/Scripts/python.exe -m pytest tests/unit/test_builder.py -q   # focused
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format --config "format.quote-style='preserve'" src tests
.venv/Scripts/python.exe -m mypy src
.venv/Scripts/python.exe tools/bump-version.py patch     # bumps pyproject, __init__.py, file headers
```

## Conventions

- Every file starts with a `'''` header docstring: File / Brief / Author (Mistress-Lukutar) / Date / Version. Follow it on new files.
- The whole repo uses `'''` (single-quote) docstrings. **Never run plain `ruff format`** — the default rewrites all quotes to `"""`; use the `--config "format.quote-style='preserve'"` invocation above.
- Ruff: line-length 88, rules E/F/W/I. mypy: python 3.11, `mypy_path=src`; `py.typed` is shipped.

## Architecture / gotchas

- **OCC topology:** any grouped multi-tool boolean in `builder.py` must keep the `Shape.clean()` (UnifySameDomain) pass via `_as_single_solid` — one-shot `fuse(*tools)` leaves unmerged coplanar faces that SEGFAULT the OCC mesher on ≥24×24 blocks.
- **Threading:** OCC booleans release the GIL, but Python-side loops don't; threads freeze the UI. All heavy GUI work must go through `gui/jobs.py` JobQueue (single-worker spawn `ProcessPoolExecutor`). `cq.Workplane`/OCC shapes cannot pickle: models stay cached in the worker (LRU in `jobtasks.py`); meshes cross the process boundary only as numpy arrays (`preview.mesh_to_arrays` / `mesh_from_arrays`).
- **Preview "edges"** = mesh wireframe derived from the triangle mesh only. Never reintroduce B-Rep edge sampling or wireframe-from-topology.
- `BRepMesh_IncrementalMesh` silently skips already-meshed shapes — mesh benchmarks must run in fresh processes or numbers are fake.
- Settings/templates are Pydantic-validated (`validate_settings`); CLI parses `--exclude` as `ix,iy` pairs joined by `;`.

## History note

README.md and docs/ were purged from all git history (2026-11-16); old history exists only in `brickgen-backup-20261116.bundle`. Don't try to browse docs from git log.
