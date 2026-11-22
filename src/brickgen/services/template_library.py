'''
File:   template_library.py
Brief:  Template library: block layout templates from builtin and user dirs.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from brickgen.core.exceptions import TemplateError
from brickgen.core.models import validate_settings
from brickgen.core.project import LAYOUT_FIELDS, BlockLayout

logger = logging.getLogger(__name__)

SCHEMA_KEY = "schema"
TEMPLATE_NAME_KEY = "template_name"
TEMPLATE_SCHEMA_NAME = "brickgen-block-template"
TEMPLATE_SCHEMA_VERSION = 1
SETTINGS_SCHEMA_NAME = "brickgen-settings"
SETTINGS_SCHEMA_VERSION = 3

# Read-only templates shipped with the application.
BUILTIN_DIR = Path(__file__).resolve().parent.parent / "data" / "templates"


@dataclass(frozen=True)
class Template:
    '''A named block-layout template for the template library.

    Attributes:
        name: Display name.
        layout: Validated block layout (grid size, flags, per-cell lists).
        builtin: True for templates shipped with the application.
    '''

    name: str
    layout: BlockLayout
    builtin: bool


def user_templates_dir() -> Path:
    '''Return the writable user templates directory, creating it if needed.

    Returns:
        ``Documents/BrickGen/templates`` (falls back to ``~/.brickgen``).
    '''
    documents = Path.home() / "Documents"
    base = documents if documents.is_dir() else Path.home()
    directory = base / "BrickGen" / "templates"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _load_template_file(path: Path, builtin: bool) -> Template:
    '''Load and validate a single template JSON file.

    Supports the new block-template format (schema name
    ``brickgen-block-template``) and the legacy settings format
    (``brickgen-settings``). New-format files are parsed as ``BlockLayout``;
    legacy files are parsed as full ``BlockSettings`` and converted to a
    ``BlockLayout`` carrying only the block-level fields.

    Args:
        path: Source file.
        builtin: Whether the file belongs to the builtin library.

    Returns:
        Parsed template.

    Raises:
        TemplateError: If the file is unreadable, malformed or invalid.
    '''
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TemplateError(f"Cannot read template {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TemplateError(f"Invalid JSON in template {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TemplateError(f"Invalid template structure in {path}")
    name = str(payload.pop(TEMPLATE_NAME_KEY, path.stem))
    schema = payload.pop(SCHEMA_KEY, None)
    # New format: try BlockLayout first (extra keys forbidden).
    if (
        isinstance(schema, dict)
        and schema.get("name") == TEMPLATE_SCHEMA_NAME
        and schema.get("version") == TEMPLATE_SCHEMA_VERSION
    ):
        try:
            layout = BlockLayout.model_validate(payload)
        except Exception as exc:
            raise TemplateError(f"Invalid template layout in {path}: {exc}") from exc
        return Template(name=name, layout=layout, builtin=builtin)
    # Legacy format: full BlockSettings → extract block-level fields.
    try:
        settings = validate_settings(payload)
    except Exception as exc:
        raise TemplateError(f"Invalid template settings in {path}: {exc}") from exc
    try:
        layout_dict = {
            "cells_x": settings.cells_x,
            "cells_y": settings.cells_y,
            "pins_enabled": settings.pins_enabled,
            "tubes_enabled": settings.tubes_enabled,
        }
        for field in LAYOUT_FIELDS:
            layout_dict[field] = getattr(settings, field)
        layout = BlockLayout.model_validate(layout_dict)
    except Exception as exc:
        raise TemplateError(f"Cannot extract block layout from {path}: {exc}") from exc
    return Template(name=name, layout=layout, builtin=builtin)


def list_templates() -> list[Template]:
    '''List all block-layout templates: builtin first, then user templates.

    Unreadable files are skipped with a warning.

    Returns:
        Templates from the builtin directory and the user directory.
    '''
    templates: list[Template] = []
    for directory, builtin in ((BUILTIN_DIR, True), (user_templates_dir(), False)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                templates.append(_load_template_file(path, builtin))
            except TemplateError as exc:
                logger.warning("Skipping template: %s", exc)
    logger.info("Template library: %d templates", len(templates))
    return templates


def save_template(name: str, layout: BlockLayout) -> Path:
    '''Save current block layout as a named user template.

    Args:
        name: Display name (also sanitized into the file name).
        layout: Block layout to persist.

    Returns:
        Path of the written template file (existing file is overwritten).

    Raises:
        TemplateError: If the file cannot be written.
    '''
    display_name = name.strip() or "Template"
    safe = re.sub(r"[^\w\-]+", "_", display_name, flags=re.UNICODE).strip("_")
    path = user_templates_dir() / f"{safe or 'template'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        SCHEMA_KEY: {
            "name": TEMPLATE_SCHEMA_NAME,
            "version": TEMPLATE_SCHEMA_VERSION,
        },
        TEMPLATE_NAME_KEY: display_name,
        **layout.model_dump(),
    }
    try:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        raise TemplateError(f"Cannot save template to {path}: {exc}") from exc
    logger.info("Template saved: %s -> %s", display_name, path)
    return path
