'''
File:   test_app_settings.py
Brief:  Unit tests for persistent application preferences.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.16.0
'''

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brickgen.services import app_settings as asm
from brickgen.services.app_settings import (
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
