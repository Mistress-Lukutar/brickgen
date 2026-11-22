'''
File:   app_settings.py
Brief:  Persistent application preferences (precisions, default format).
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.19.0
'''

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from brickgen.services.export_service import SUPPORTED_FORMATS

logger = logging.getLogger(__name__)


class AppSettings(BaseModel):
    '''Application-level preferences persisted across sessions.

    View state (grids, snapping, BOM labels) is NOT stored here anymore:
    since v0.16.0 it is per-project data in ``core.project.ViewSettings``
    and travels inside the project file.

    Attributes:
        preview_tolerance: Tessellation tolerance and edge chord-deviation
            bound for the interactive preview, mm. One value drives both,
            so mesh circles and edge polylines always match.
        export_tolerance: Mesh tolerance for tessellated export formats
            (STL/AMF/3MF), mm.
        default_format: Format preselected in the export dialog.
        undo_depth: Maximum number of undo (Ctrl+Z) steps kept in the
            edit history shared by the planning and placement tabs.
    '''

    preview_tolerance: float = Field(default=0.15, gt=0, le=1.0)
    preview_angular_tolerance: float = Field(default=0.1, gt=0, le=1.0)
    export_tolerance: float = Field(default=0.1, gt=0, le=1.0)
    export_angular_tolerance: float = Field(default=0.1, gt=0, le=1.0)
    default_format: str = "STEP"
    undo_depth: int = Field(default=50, ge=1, le=500)

    model_config = {"extra": "forbid"}

    @field_validator("default_format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        '''Restrict the default export format to the supported set.'''
        normalized = value.upper()
        if normalized not in SUPPORTED_FORMATS:
            raise ValueError(
                f"default_format must be one of {', '.join(SUPPORTED_FORMATS)}"
            )
        return normalized


def app_settings_path() -> Path:
    '''Return the settings file path, creating the parent directory.

    Returns:
        ``Documents/BrickGen/settings.json`` (home fallback like templates).
    '''
    documents = Path.home() / "Documents"
    base = documents if documents.is_dir() else Path.home()
    directory = base / "BrickGen"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "settings.json"


def load_app_settings() -> AppSettings:
    '''Load application preferences, falling back to defaults.

    A missing, unreadable or invalid file yields factory defaults (with a
    warning in the log), so the GUI always starts. Keys unknown to the
    current model — e.g. the per-project view settings stored here before
    v0.16.0 — are dropped with a note instead of failing the whole file.

    Returns:
        Loaded or default AppSettings.
    '''
    path = app_settings_path()
    if not path.is_file():
        return AppSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            known = AppSettings.model_fields
            unknown = set(payload) - known.keys()
            if unknown:
                logger.info(
                    "App settings %s ignored legacy keys: %s",
                    path,
                    ", ".join(sorted(unknown)),
                )
                payload = {k: v for k, v in payload.items() if k in known}
        return AppSettings.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to defaults
        logger.warning("App settings %s unreadable, using defaults: %s", path, exc)
        return AppSettings()


def save_app_settings(settings: AppSettings) -> Path:
    '''Persist application preferences to the settings file.

    Args:
        settings: Preferences to store.

    Returns:
        Path of the written file.

    Raises:
        OSError: If the file cannot be written.
    '''
    path = app_settings_path()
    path.write_text(
        json.dumps(settings.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("App settings saved to %s", path)
    return path
