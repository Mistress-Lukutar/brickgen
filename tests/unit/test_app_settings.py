'''
File:   test_app_settings.py
Brief:  Unit tests for persistent application preferences.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.19.0
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from brickgen.services import app_settings as asm
from brickgen.services.app_settings import (
    MAX_RECENT_PROJECTS,
    AppSettings,
    app_settings_path,
    load_app_settings,
    save_app_settings,
)


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path: Path) -> Path:
    '''Redirect the settings file to a temp location.'''
    target = tmp_path / "settings.json"
    monkeypatch.setattr(asm, "app_settings_path", lambda: target)
    return target


def test_roundtrip(isolated_settings: Path) -> None:
    '''Saved preferences load back equal.'''
    settings = AppSettings(
        preview_tolerance=0.05, export_tolerance=0.02, default_format="STL"
    )
    save_app_settings(settings)
    assert load_app_settings() == settings


def test_missing_file_yields_defaults(isolated_settings: Path) -> None:
    '''No settings file means factory defaults.'''
    assert load_app_settings() == AppSettings()


def test_broken_file_yields_defaults(isolated_settings: Path) -> None:
    '''Corrupt settings fall back to defaults instead of crashing.'''
    isolated_settings.write_text("{broken", encoding="utf-8")
    assert load_app_settings() == AppSettings()


def test_invalid_values_rejected(isolated_settings: Path) -> None:
    '''Out-of-domain values fail validation on load -> defaults.'''
    save_app_settings(AppSettings())
    payload = json.loads(isolated_settings.read_text(encoding="utf-8"))
    payload["preview_tolerance"] = 5.0
    isolated_settings.write_text(json.dumps(payload), encoding="utf-8")
    assert load_app_settings() == AppSettings()


def test_unknown_format_rejected() -> None:
    '''Default format must be one of the supported export formats.'''
    with pytest.raises(ValueError, match="default_format"):
        AppSettings(default_format="OBJ")


def test_settings_path_uses_documents() -> None:
    '''The settings file lives next to the template library.'''
    path = app_settings_path()
    assert path.name == "settings.json"
    assert path.parent.name == "BrickGen"


def test_angular_tolerance_defaults_and_bounds() -> None:
    '''Angular tolerance persists and must be within (0, 1] rad.'''
    settings = AppSettings(preview_angular_tolerance=0.05, export_angular_tolerance=0.2)
    save_app_settings(settings)
    assert load_app_settings().preview_angular_tolerance == 0.05
    assert load_app_settings().export_angular_tolerance == 0.2
    with pytest.raises(ValueError):
        AppSettings(preview_angular_tolerance=0.0)
    with pytest.raises(ValueError):
        AppSettings(export_angular_tolerance=2.0)


def test_legacy_view_keys_are_ignored(isolated_settings: Path) -> None:
    '''Pre-v0.16 settings files carrying per-project view keys still load.

    View settings (grids, snap, labels) moved into the project file with
    schema v8; old settings.json payloads with the grid/label keys must be
    trimmed instead of invalidating the tolerance preferences.
    '''
    payload = AppSettings().model_dump()
    payload.update(
        {
            "grid_enabled": True,
            "grid_scale": 2.0,
            "grid_snap": True,
            "placement_grid_enabled": True,
            "quantity_labels_enabled": True,
            "quantity_label_position": "top",
        }
    )
    isolated_settings.write_text(json.dumps(payload), encoding="utf-8")
    assert load_app_settings() == AppSettings()


def test_undo_depth_default_and_bounds() -> None:
    '''undo_depth defaults to 50 steps and must stay within 1..500.'''
    settings = AppSettings()
    assert settings.undo_depth == 50
    with pytest.raises(ValueError):
        AppSettings(undo_depth=0)
    with pytest.raises(ValueError):
        AppSettings(undo_depth=501)
    assert AppSettings(undo_depth=1).undo_depth == 1
    assert AppSettings(undo_depth=500).undo_depth == 500


def test_undo_depth_roundtrip(isolated_settings: Path) -> None:
    '''A saved undo depth is restored verbatim by load_app_settings.'''
    save_app_settings(AppSettings(undo_depth=123))
    assert load_app_settings().undo_depth == 123


def test_recent_projects_default_empty_and_roundtrip(isolated_settings: Path) -> None:
    '''recent_projects defaults to an empty list and persists verbatim.'''
    assert AppSettings().recent_projects == []
    paths = [str(Path.home() / "one.json"), str(Path.home() / "two.json")]
    save_app_settings(AppSettings(recent_projects=paths))
    assert load_app_settings().recent_projects == [
        str(Path.home() / "one.json"),
        str(Path.home() / "two.json"),
    ]


def test_recent_projects_normalized_deduped_capped(tmp_path: Path) -> None:
    '''The validator makes paths absolute, drops junk, dedupes and caps.'''
    entries: list[Any] = [
        str(tmp_path / "a.json"),
        str(tmp_path / "a.json"),  # exact duplicate -> dropped
        str(tmp_path / "A.json"),  # case-insensitive duplicate -> dropped
        "",  # empty -> dropped
        "   ",  # whitespace-only -> dropped
        42,  # non-string -> dropped
        None,  # non-string -> dropped
        str(tmp_path / "b.json"),
    ]
    result = AppSettings(recent_projects=entries).recent_projects
    assert result == [str(tmp_path / "a.json"), str(tmp_path / "b.json")]

    many = [str(tmp_path / f"p{i}.json") for i in range(12)]
    capped = AppSettings(recent_projects=many).recent_projects
    assert len(capped) == MAX_RECENT_PROJECTS == 10
    assert [Path(p) for p in capped] == [tmp_path / f"p{i}.json" for i in range(10)]


def test_recent_projects_relative_paths_made_absolute(tmp_path: Path) -> None:
    '''Relative and non-list payloads still yield a sane list.'''
    settings = AppSettings(recent_projects=["a.json"])
    assert len(settings.recent_projects) == 1
    assert Path(settings.recent_projects[0]).is_absolute()

    assert AppSettings(recent_projects="oops").recent_projects == []  # type: ignore[arg-type]


def test_recent_projects_junk_file_entries(
    isolated_settings: Path, tmp_path: Path
) -> None:
    '''A settings file with junk recent entries loads with them cleaned.'''
    save_app_settings(AppSettings())
    payload = json.loads(isolated_settings.read_text(encoding="utf-8"))
    payload["recent_projects"] = [1, None, "", "   ", str(tmp_path / "p.json")]
    isolated_settings.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_app_settings()
    assert [Path(p) for p in loaded.recent_projects] == [tmp_path / "p.json"]
