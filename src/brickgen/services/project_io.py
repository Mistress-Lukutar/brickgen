'''
File:   project_io.py
Brief:  JSON save/load of multi-block GUI projects with legacy settings migration.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from brickgen.core.exceptions import ProjectIOError
from brickgen.core.models import BlockSettings
from brickgen.core.project import (
    BLOCK_FLAG_FIELDS,
    LAYOUT_FIELDS,
    SHARED_SETTING_FIELDS,
    BlockInstance,
    Project,
)
from brickgen.services.settings_io import SETTINGS_SCHEMA_NAME

logger = logging.getLogger(__name__)

PROJECT_SCHEMA_KEY = "schema"
PROJECT_SCHEMA_NAME = "brickgen-project"
PROJECT_SCHEMA_VERSION = 7
# Versions accepted by load_project: v4/v5 payloads are migrated on the fly
# (v4 kept pins/tubes flags on the shared settings, v5 moved them onto
# each block, v6 added print_layout and block uids, v7 added the per-piece
# rotated90 flag).
_MIGRATABLE_VERSIONS = (4, 5, 6, 7)


def save_project(project: Project, path: Path) -> Path:
    '''Serialize a project to a JSON file.

    Args:
        project: The project to persist.
        path: Destination ``*.json`` path (parent directories are created).

    Returns:
        The resolved path of the written file.

    Raises:
        ProjectIOError: If the file cannot be written.
    '''
    path = path.expanduser().resolve()
    payload = {
        PROJECT_SCHEMA_KEY: {
            "name": PROJECT_SCHEMA_NAME,
            "version": PROJECT_SCHEMA_VERSION,
        },
        "settings": {
            field: getattr(project.settings, field) for field in SHARED_SETTING_FIELDS
        },
        "blocks": [block.model_dump() for block in project.blocks],
        "print_layout": project.print_layout.model_dump(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        raise ProjectIOError(f"Cannot write project to {path}: {exc}") from exc
    logger.info("Project saved to %s", path)
    return path


def load_project(path: Path) -> Project:
    '''Load a project from a JSON file.

    Supports the current project schema (v6), previous v5 (per-block
    pins/tubes flags) and v4 (pins/tubes on shared settings), with
    automatic migration, plus legacy single-block preset files that lack
    a ``blocks`` key or carry the old ``brickgen-settings`` schema marker.

    Args:
        path: Source ``*.json`` path.

    Returns:
        Validated project instance.

    Raises:
        ProjectIOError: If the file is missing, unreadable, not a BrickGen
            project, or contains invalid values.
    '''
    path = path.expanduser().resolve()
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except OSError as exc:
        raise ProjectIOError(f"Cannot read project file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectIOError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectIOError(f"Invalid project structure in {path}")

    schema = payload.get(PROJECT_SCHEMA_KEY)

    # --- Legacy detection ---
    legacy = False
    if isinstance(schema, dict) and schema.get("name") == SETTINGS_SCHEMA_NAME:
        # Legacy single-block settings file.
        legacy = True
    elif schema is None and "blocks" not in payload:
        # Schema-less old preset.
        legacy = True

    if legacy:
        return _legacy_project(payload, path)

    # --- Current project format ---
    if isinstance(schema, dict) and schema.get("name") == PROJECT_SCHEMA_NAME:
        version = schema.get("version")
        if version not in _MIGRATABLE_VERSIONS:
            raise ProjectIOError(
                f"Unsupported project schema version in {path}: {version!r}"
            )
        payload = {k: v for k, v in payload.items() if k != PROJECT_SCHEMA_KEY}
        if version != PROJECT_SCHEMA_VERSION:
            if version < 5:
                payload = _migrate_v4_blocks(payload)
            if version < 6:
                payload = _migrate_v5_to_v6(payload)
            if version < 7:
                payload = _migrate_v6_to_v7(payload)
    else:
        raise ProjectIOError(f"{path} is not a {PROJECT_SCHEMA_NAME} project")

    try:
        project = Project.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - pydantic wraps its own hierarchy
        raise ProjectIOError(f"Invalid project in {path}: {exc}") from exc

    logger.info("Project loaded from %s (%d blocks)", path, len(project.blocks))
    return project


def _legacy_project(payload: dict, path: Path) -> Project:
    '''Convert a legacy single-block preset into a one-block project.

    Args:
        payload: The parsed JSON dict (without the schema key removed yet).
        path: Source path for error messages.

    Returns:
        A ``Project`` wrapping the legacy block as a single instance.
    '''
    cleaned = {k: v for k, v in payload.items() if k != PROJECT_SCHEMA_KEY}
    try:
        settings = BlockSettings.model_validate(cleaned)
    except Exception as exc:  # noqa: BLE001
        raise ProjectIOError(f"Invalid settings in {path}: {exc}") from exc

    shared = BlockSettings.model_validate(
        {field: getattr(settings, field) for field in SHARED_SETTING_FIELDS}
    )
    block = BlockInstance(
        cells_x=settings.cells_x,
        cells_y=settings.cells_y,
        **{field: getattr(settings, field) for field in BLOCK_FLAG_FIELDS},
        **{field: getattr(settings, field) for field in LAYOUT_FIELDS},
    )
    return Project(settings=shared, blocks=[block])


def _migrate_v4_blocks(payload: dict) -> dict:
    '''Backfill the per-block pins/tubes flags of a v4 payload.

    v4 stored ``pins_enabled``/``tubes_enabled`` only on the shared project
    settings; v5 stores them on every block. The shared values are copied
    into each block so an old scene renders exactly as it did.

    Args:
        payload: The parsed project dict (schema key already removed).

    Returns:
        The payload with every block carrying explicit flag fields.
    '''
    settings = payload.get("settings")
    shared = settings if isinstance(settings, dict) else {}
    for block in payload.get("blocks") or []:
        if isinstance(block, dict):
            for field in BLOCK_FLAG_FIELDS:
                block.setdefault(field, shared.get(field, True))
    return payload


def _migrate_v5_to_v6(payload: dict) -> dict:
    '''Backfill uids and print_layout on a v5 payload.

    v5 did not store ``uid`` on blocks or a ``print_layout`` dict; v6 adds
    both so that print-placement pieces can be stably linked to their block
    definitions across edits.

    Args:
        payload: The parsed project dict (schema key already removed).

    Returns:
        The payload with every block carrying a ``uid`` and a populated
        ``print_layout`` dict.
    '''
    for block in payload.get("blocks") or []:
        if isinstance(block, dict):
            block.setdefault("uid", uuid.uuid4().hex)

    layout = payload.get("print_layout")
    if not isinstance(layout, dict):
        payload["print_layout"] = {}
    else:
        pieces = layout.setdefault("pieces", [])
        for piece in pieces:
            if isinstance(piece, dict):
                piece.setdefault("uid", uuid.uuid4().hex)
        layout.setdefault("beds", {})

    return payload


def _migrate_v6_to_v7(payload: dict) -> dict:
    '''Backfill the per-piece rotated90 flag on a v6 payload.

    v7 adds ``rotated90`` (90-degree packing rotation set by the placement
    engine); v6 pieces did not store it, so every piece dict gains the
    ``False`` default.

    Args:
        payload: The parsed project dict (schema key already removed).

    Returns:
        The payload with every piece carrying an explicit ``rotated90``.
    '''
    layout = payload.get("print_layout")
    if isinstance(layout, dict):
        for piece in layout.get("pieces") or []:
            if isinstance(piece, dict):
                piece.setdefault("rotated90", False)
    return payload
