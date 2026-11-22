'''
File:   app_settings.py
Brief:  Persistent application preferences (precisions, default format).
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from brickgen.services.export_service import SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

# Anchor side of the per-block BOM quantity label (relative to each block).
QuantityLabelPosition = Literal["left", "right", "top", "bottom"]
QUANTITY_LABEL_POSITIONS: tuple[str, ...] = ("left", "right", "top", "bottom")

# Grid cell scale presets offered by the Grid menu: the actual grid step is
# ``unit_size * grid_scale`` mm (1.0 = one cell per Unit Size).
GRID_SCALE_PRESETS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


class AppSettings(BaseModel):
    '''Application-level preferences persisted across sessions.

    Attributes:
        preview_tolerance: Tessellation tolerance and edge chord-deviation
            bound for the interactive preview, mm. One value drives both,
            so mesh circles and edge polylines always match.
        export_tolerance: Mesh tolerance for tessellated export formats
            (STL/AMF/3MF), mm.
        default_format: Format preselected in the export dialog.
        quantity_labels_enabled: Show the BOM quantity label next to every
            placed block in the preview (View menu toggle).
        quantity_label_position: Anchor side of the quantity label,
            relative to each block.
        grid_enabled: Show the XY floor grid in the preview (Grid menu).
        grid_scale: Grid cell scale factor; the grid step is
            ``unit_size * grid_scale`` mm (1.0 = one cell per Unit Size).
        grid_snap: Snap block movement to the grid step (Grid menu).
        placement_grid_enabled: Show the floor grid on the print-placement
            viewport.
    '''

    preview_tolerance: float = Field(default=0.15, gt=0, le=1.0)
    preview_angular_tolerance: float = Field(default=0.1, gt=0, le=1.0)
    export_tolerance: float = Field(default=0.1, gt=0, le=1.0)
    export_angular_tolerance: float = Field(default=0.1, gt=0, le=1.0)
    default_format: str = "STEP"
    quantity_labels_enabled: bool = False
    quantity_label_position: QuantityLabelPosition = "right"
    grid_enabled: bool = False
    grid_scale: float = Field(default=0.5, gt=0, le=1000.0)
    grid_snap: bool = False
    placement_grid_enabled: bool = Field(
        default=False,
        description="Show the floor grid on the print-placement viewport.",
    )

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
    warning in the log), so the GUI always starts.

    Returns:
        Loaded or default AppSettings.
    '''
    path = app_settings_path()
    if not path.is_file():
        return AppSettings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
