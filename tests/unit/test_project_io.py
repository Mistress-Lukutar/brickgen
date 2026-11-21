'''
File:   test_project_io.py
Brief:  Unit tests for JSON save/load of multi-block projects.
Author: Mistress-Lukutar
Date:   2026-11-20
Version: v0.13.0
'''

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brickgen.core.exceptions import ProjectIOError
from brickgen.core.models import BlockSettings
from brickgen.core.project import (
    LAYOUT_FIELDS,
    BlockInstance,
    Project,
)
from brickgen.services.project_io import load_project, save_project


# ----------------------------------------------------------- roundtrip
def test_save_load_roundtrip(tmp_path: Path) -> None:
    '''Saved project must load back to an equal model.'''
    proj = Project(
        settings=BlockSettings(
            unit_size=4.0,
            wall_thickness=0.8,
            block_height=3.5,
            clearance_xy=-0.25,
            clearance_z=0.2,
            pins_enabled=True,
            tubes_enabled=True,
        ),
        blocks=[
            BlockInstance(
                cells_x=2,
                cells_y=4,
                position_x=10.0,
                position_y=5.0,
                color="#ff3300",
                quantity=12,
                sloped_cells=[[0, 0, "+X", 1]],
            ),
            BlockInstance(
                cells_x=1,
                cells_y=1,
                position_x=-3.0,
                position_y=7.5,
                color="#00aaff",
                quantity=3,
            ),
        ],
    )

    path = save_project(proj, tmp_path / "project.json")
    loaded = load_project(path)

    # Shared settings match
    assert loaded.settings.block_height == proj.settings.block_height
    assert loaded.settings.unit_size == proj.settings.unit_size
    assert loaded.settings.wall_thickness == proj.settings.wall_thickness
    assert loaded.settings.clearance_xy == proj.settings.clearance_xy
    assert loaded.settings.clearance_z == proj.settings.clearance_z
    assert loaded.settings.pins_enabled == proj.settings.pins_enabled
    assert loaded.settings.tubes_enabled == proj.settings.tubes_enabled

    # Blocks match
    assert len(loaded.blocks) == 2
    for i in range(len(proj.blocks)):
        assert loaded.blocks[i].model_dump() == proj.blocks[i].model_dump()


# ------------------------------------------------------- saved file format
def test_saved_file_has_correct_schema(tmp_path: Path) -> None:
    '''Saved JSON must contain schema brickgen-project version 4.'''
    proj = Project(blocks=[BlockInstance()])
    path = save_project(proj, tmp_path / "project.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"]["name"] == "brickgen-project"
    assert payload["schema"]["version"] == 4


def test_saved_settings_has_no_layout_fields(tmp_path: Path) -> None:
    '''The "settings" key must NOT contain any layout field keys.'''
    proj = Project(
        blocks=[
            BlockInstance(
                cells_x=2,
                cells_y=2,
                excluded_cells=[[0, 0]],
                sloped_cells=[[1, 1, "+X", 1]],
            )
        ],
    )
    path = save_project(proj, tmp_path / "project.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    for field in LAYOUT_FIELDS:
        assert field not in payload["settings"], (
            f"Layout field '{field}' must not appear in settings"
        )


# ------------------------------------------------------- legacy migration
def test_legacy_settings_migration(tmp_path: Path) -> None:
    '''A brickgen-settings preset migrates to a one-block project.'''
    legacy = {
        "schema": {"name": "brickgen-settings", "version": 3},
        "cells_x": 4,
        "cells_y": 2,
        "block_height": 4.5,
        "excluded_cells": [[0, 0], [1, 1]],
        "chamfered_cells": [[3, 0, "+X"]],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    proj = load_project(path)

    assert len(proj.blocks) == 1
    block = proj.blocks[0]
    assert block.cells_x == 4
    assert block.cells_y == 2
    assert block.excluded_cells == [[0, 0], [1, 1]]
    assert block.chamfered_cells == [[3, 0, "+X"]]
    assert block.position_x == 0.0
    assert block.position_y == 0.0
    assert block.quantity == 1
    assert proj.settings.block_height == 4.5


def test_legacy_without_schema_marker_migrates(tmp_path: Path) -> None:
    '''A plain settings file (no schema, no "blocks") also migrates.'''
    legacy = {
        "cells_x": 3,
        "cells_y": 3,
        "block_height": 5.0,
        "unit_size": 4.0,
        "wall_thickness": 0.8,
        "clearance_xy": -0.25,
        "clearance_z": 0.2,
        "pins_enabled": True,
        "tubes_enabled": True,
    }
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    proj = load_project(path)

    assert len(proj.blocks) == 1
    assert proj.blocks[0].cells_x == 3
    assert proj.settings.block_height == 5.0


# ----------------------------------------------------------- error cases
def test_unknown_schema_name_raises(tmp_path: Path) -> None:
    '''A file with an unknown schema marker must raise ProjectIOError.'''
    path = tmp_path / "other.json"
    path.write_text(
        json.dumps(
            {
                "schema": {"name": "some-other-tool", "version": 1},
                "blocks": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectIOError):
        load_project(path)


def test_unsupported_version_raises(tmp_path: Path) -> None:
    '''A project schema with version 99 must raise ProjectIOError.'''
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps(
            {
                "schema": {"name": "brickgen-project", "version": 99},
                "settings": {},
                "blocks": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectIOError, match="Unsupported project schema version"):
        load_project(path)


def test_broken_json_raises(tmp_path: Path) -> None:
    '''Malformed JSON must raise ProjectIOError.'''
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ProjectIOError, match="Invalid JSON"):
        load_project(path)


def test_nonexistent_file_raises(tmp_path: Path) -> None:
    '''Loading a missing file must raise ProjectIOError.'''
    with pytest.raises(ProjectIOError):
        load_project(tmp_path / "missing.json")


def test_invalid_block_color_raises(tmp_path: Path) -> None:
    '''A project with an invalid block color must raise ProjectIOError.'''
    payload = {
        "schema": {"name": "brickgen-project", "version": 4},
        "settings": {},
        "blocks": [
            {
                "cells_x": 2,
                "cells_y": 2,
                "color": "red",
                "quantity": 1,
            }
        ],
    }
    path = tmp_path / "bad_color.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectIOError, match="Invalid project"):
        load_project(path)
