'''
File:   settings_io.py
Brief:  JSON import/export of BlockSettings presets.
Author: Mistress-Lukutar
Date:   2026-11-20
Version: v0.13.0
'''

from __future__ import annotations

import json
import logging
from pathlib import Path

from brickgen.core.exceptions import SettingsIOError
from brickgen.core.models import BlockSettings

logger = logging.getLogger(__name__)

SETTINGS_SCHEMA_KEY = "schema"
SETTINGS_SCHEMA_NAME = "brickgen-settings"
# v3: stud/tube sizes derived from clearance_xy/clearance_z; legacy v2
# ``pin_diameter``/``pin_height``/``tube_inner_diameter`` keys are dropped on
# load by BlockSettings._migrate_legacy_settings.
SETTINGS_SCHEMA_VERSION = 3


def save_settings(settings: BlockSettings, path: Path) -> Path:
    '''Serialize block settings to a JSON preset file.

    Args:
        settings: Settings to persist.
        path: Destination ``*.json`` path (parent directories are created).

    Returns:
        The resolved path of the written file.

    Raises:
        SettingsIOError: If the file cannot be written.
    '''
    path = path.expanduser().resolve()
    payload = {
        SETTINGS_SCHEMA_KEY: {
            "name": SETTINGS_SCHEMA_NAME,
            "version": SETTINGS_SCHEMA_VERSION,
        },
        **settings.model_dump(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        raise SettingsIOError(f"Cannot write settings to {path}: {exc}") from exc
    logger.info("Settings saved to %s", path)
    return path


def load_settings(path: Path) -> BlockSettings:
    '''Load block settings from a JSON preset file.

    Args:
        path: Source ``*.json`` path.

    Returns:
        Validated settings instance.

    Raises:
        SettingsIOError: If the file is missing, unreadable, not a BrickGen
            preset, or contains invalid values.
    '''
    path = path.expanduser().resolve()
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except OSError as exc:
        raise SettingsIOError(f"Cannot read settings file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SettingsIOError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SettingsIOError(f"Invalid preset structure in {path}")
    schema = payload.get(SETTINGS_SCHEMA_KEY)
    if isinstance(schema, dict):
        if schema.get("name") != SETTINGS_SCHEMA_NAME:
            raise SettingsIOError(f"{path} is not a {SETTINGS_SCHEMA_NAME} preset")
        payload = {k: v for k, v in payload.items() if k != SETTINGS_SCHEMA_KEY}
    try:
        return BlockSettings.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - pydantic wraps its own hierarchy
        raise SettingsIOError(f"Invalid settings in {path}: {exc}") from exc
