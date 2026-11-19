'''
File:   test_template_library.py
Brief:  Unit tests for the template library service.
Author: Mistress-Lukutar
Date:   2026-11-19
Version: v0.11.0
'''

from __future__ import annotations

from pathlib import Path

import pytest

from brickgen.core.exceptions import TemplateError
from brickgen.core.models import BlockSettings
from brickgen.services import template_library as tl
from brickgen.services.template_library import (
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
    assert all(isinstance(t.settings, BlockSettings) for t in templates)
    assert all(t.builtin for t in templates)
    names = {t.name for t in templates}
    assert {"Brick 2x4", "T-shape 3x2", "L-shape 2x2"} <= names


def test_save_and_reload_roundtrip(isolated_user_dir: Path) -> None:
    '''A saved user template appears in the list with its settings.'''
    settings = BlockSettings(
        cells_x=3,
        cells_y=2,
        pins_enabled=False,
        sloped_cells=[[0, 0, "-X", 1], [1, 0, "-X", 1], [2, 0, "-X", 1]],
    )
    path = save_template("My Slope", settings)
    assert path.parent == isolated_user_dir
    templates = list_templates()
    mine = [t for t in templates if t.name == "My Slope"]
    assert len(mine) == 1
    assert not mine[0].builtin
    assert mine[0].settings == settings


def test_save_template_overwrites(isolated_user_dir: Path) -> None:
    '''Saving the same name twice keeps a single file (overwritten).'''
    save_template("Dup", BlockSettings())
    save_template("Dup", BlockSettings(cells_x=2, cells_y=2))
    templates = [t for t in list_templates() if t.name == "Dup"]
    assert len(templates) == 1
    assert templates[0].settings == BlockSettings(cells_x=2, cells_y=2)


def test_invalid_user_template_skipped(isolated_user_dir: Path) -> None:
    '''Broken user files are skipped without breaking the listing.'''
    save_template("Good", BlockSettings(cells_x=2, cells_y=2))
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


def test_user_templates_dir_exists() -> None:
    '''The default user templates directory is created on demand.'''
    assert user_templates_dir().is_dir()
