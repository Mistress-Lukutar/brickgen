'''
File:   test_settings_io.py
Brief:  Unit tests for JSON settings persistence.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.21.0
'''

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brickgen.core.exceptions import SettingsIOError
from brickgen.core.models import BlockSettings
from brickgen.services.settings_io import load_settings, save_settings


def test_roundtrip(tmp_path: Path) -> None:
    '''Saved settings must load back to an equal model.'''
    settings = BlockSettings(cells_x=2, cells_y=6, block_height=10.0)
    path = save_settings(settings, tmp_path / "preset.json")
    loaded = load_settings(path)
    assert loaded == settings


def test_saved_file_has_schema_marker(tmp_path: Path) -> None:
    '''Preset JSON must carry the schema marker block.'''
    path = save_settings(BlockSettings(), tmp_path / "preset.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"]["name"] == "brickgen-settings"
    assert payload["schema"]["version"] == 3


def test_load_missing_file(tmp_path: Path) -> None:
    '''Loading a missing file raises SettingsIOError.'''
    with pytest.raises(SettingsIOError):
        load_settings(tmp_path / "missing.json")


def test_load_invalid_json(tmp_path: Path) -> None:
    '''Malformed JSON raises SettingsIOError.'''
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SettingsIOError):
        load_settings(path)


def test_load_foreign_preset(tmp_path: Path) -> None:
    '''JSON with a foreign schema marker is rejected.'''
    path = tmp_path / "other.json"
    path.write_text(
        json.dumps({"schema": {"name": "other-tool", "version": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(SettingsIOError):
        load_settings(path)


def test_load_invalid_values(tmp_path: Path) -> None:
    '''Out-of-domain values fail validation during load.'''
    path = save_settings(BlockSettings(), tmp_path / "good.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["wall_thickness"] = 3.0  # leaves no cavity
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SettingsIOError):
        load_settings(bad)
