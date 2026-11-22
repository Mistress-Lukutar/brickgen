'''
File:   test_template_library.py
Brief:  Unit tests for the template library service.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brickgen.core.exceptions import TemplateError
from brickgen.core.project import BlockLayout
from brickgen.services import template_library as tl
from brickgen.services.template_library import (
    TEMPLATE_SCHEMA_NAME,
    TEMPLATE_SCHEMA_VERSION,
    list_templates,
    save_template,
    user_templates_dir,
)


@pytest.fixture
def isolated_user_dir(monkeypatch, tmp_path: Path) -> Path:
    '''Redirect the user templates dir to a temp folder.'''
    target = tmp_path / "templates"
    monkeypatch.setattr(tl, "user_templates_dir", lambda: target)
    return target


def test_builtin_templates_are_valid(monkeypatch, tmp_path: Path) -> None:
    '''All shipped templates parse and validate (user dir isolated).'''
    monkeypatch.setattr(tl, "user_templates_dir", lambda: tmp_path / "u")
    templates = list_templates()
    assert len(templates) >= 11
    assert all(isinstance(t.layout, BlockLayout) for t in templates)
    assert all(t.builtin for t in templates)
    names = {t.name for t in templates}
    assert {"Brick 2x4", "T-shape 3x2", "L-shape 2x2"} <= names


def test_load_new_format_file(monkeypatch, tmp_path: Path) -> None:
    '''A new-format block-template v1 file loads with layout fields.'''
    target = tmp_path / "templates"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tl, "user_templates_dir", lambda: target)
    payload = {
        "schema": {"name": TEMPLATE_SCHEMA_NAME, "version": TEMPLATE_SCHEMA_VERSION},
        "template_name": "Test 2x4",
        "cells_x": 2,
        "cells_y": 4,
        "pins_enabled": False,
        "tubes_enabled": True,
    }
    (target / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    templates = list_templates()
    mine = [t for t in templates if t.name == "Test 2x4"]
    assert len(mine) == 1
    tpl = mine[0]
    assert tpl.layout.cells_x == 2
    assert tpl.layout.cells_y == 4
    assert tpl.layout.pins_enabled is False
    assert tpl.layout.tubes_enabled is True


def test_load_legacy_settings_file(monkeypatch, tmp_path: Path) -> None:
    '''A legacy brickgen-settings v3 file extracts only block-level fields.'''
    target = tmp_path / "templates"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tl, "user_templates_dir", lambda: target)
    payload = {
        "schema": {"name": "brickgen-settings", "version": 3},
        "template_name": "Legacy Block",
        "cells_x": 3,
        "cells_y": 2,
        "unit_size": 4.0,
        "wall_thickness": 0.8,
        "block_height": 3.0,
        "clearance_xy": -0.25,
        "clearance_z": 0.2,
        "pins_enabled": False,
        "tubes_enabled": False,
        "sloped_cells": [[0, 0, "+X", 1]],
    }
    (target / "legacy.json").write_text(json.dumps(payload), encoding="utf-8")
    templates = list_templates()
    mine = [t for t in templates if t.name == "Legacy Block"]
    assert len(mine) == 1
    tpl = mine[0]
    assert tpl.layout.cells_x == 3
    assert tpl.layout.cells_y == 2
    assert tpl.layout.pins_enabled is False
    assert tpl.layout.tubes_enabled is False
    assert tpl.layout.sloped_cells == [[0, 0, "+X", 1]]


def test_save_and_reload_roundtrip(isolated_user_dir: Path) -> None:
    '''A saved user template appears in the list with its layout.'''
    layout = BlockLayout(
        cells_x=3,
        cells_y=2,
        pins_enabled=False,
        sloped_cells=[[0, 0, "-X", 1], [1, 0, "-X", 1], [2, 0, "-X", 1]],
    )
    path = save_template("My Slope", layout)
    assert path.parent == isolated_user_dir
    # Verify the file contains the new schema marker.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"]["name"] == TEMPLATE_SCHEMA_NAME
    assert payload["schema"]["version"] == TEMPLATE_SCHEMA_VERSION
    templates = list_templates()
    mine = [t for t in templates if t.name == "My Slope"]
    assert len(mine) == 1
    assert not mine[0].builtin
    assert mine[0].layout.cells_x == 3
    assert mine[0].layout.cells_y == 2
    assert mine[0].layout.pins_enabled is False
    assert mine[0].layout.sloped_cells == [
        [0, 0, "-X", 1],
        [1, 0, "-X", 1],
        [2, 0, "-X", 1],
    ]


def test_save_template_overwrites(isolated_user_dir: Path) -> None:
    '''Saving the same name twice keeps a single file (overwritten).'''
    save_template("Dup", BlockLayout(cells_x=2, cells_y=2))
    save_template("Dup", BlockLayout(cells_x=3, cells_y=3, tubes_enabled=False))
    templates = [t for t in list_templates() if t.name == "Dup"]
    assert len(templates) == 1
    assert templates[0].layout.cells_x == 3
    assert templates[0].layout.cells_y == 3
    assert templates[0].layout.tubes_enabled is False


def test_invalid_user_template_skipped(isolated_user_dir: Path) -> None:
    '''Broken user files are skipped without breaking the listing.'''
    save_template("Good", BlockLayout(cells_x=2, cells_y=2))
    bad = isolated_user_dir / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    names = [t.name for t in list_templates()]
    assert "Good" in names
    assert "bad" not in names


def test_load_template_file_invalid_raises(tmp_path: Path) -> None:
    '''Malformed JSON raises TemplateError on direct load.'''
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(TemplateError):
        tl._load_template_file(path, builtin=False)


def test_load_template_file_invalid_payload_raises(tmp_path: Path) -> None:
    '''A new-format file with invalid layout fields raises TemplateError.'''
    path = tmp_path / "invalid.json"
    payload = {
        "schema": {"name": TEMPLATE_SCHEMA_NAME, "version": TEMPLATE_SCHEMA_VERSION},
        "template_name": "Invalid",
        "cells_x": 2,
        "cells_y": 4,
        "pins_enabled": False,
        "tubes_enabled": True,
        "unknown_field": 99,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TemplateError):
        tl._load_template_file(path, builtin=False)


def test_user_templates_dir_exists() -> None:
    '''The default user templates directory is created on demand.'''
    assert user_templates_dir().is_dir()
