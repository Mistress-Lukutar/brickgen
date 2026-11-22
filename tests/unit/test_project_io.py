'''
File:   test_project_io.py
Brief:  Unit tests for JSON save/load of multi-block projects.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.18.0
'''

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brickgen.core.exceptions import ProjectIOError
from brickgen.core.models import BlockSettings
from brickgen.core.project import (
    LAYOUT_FIELDS,
    BlockInstance,
    PrintBedSettings,
    PrintLayout,
    PrintPiece,
    PrintToolsSettings,
    Project,
    ViewSettings,
)
from brickgen.services.project_io import (
    PROJECT_SCHEMA_NAME,
    PROJECT_SCHEMA_VERSION,
    load_project,
    save_project,
)


# ----------------------------------------------------------- roundtrip
def test_save_load_roundtrip(tmp_path: Path) -> None:
    '''Saved project must load back to an equal model.'''
    proj = Project(
        settings=BlockSettings(
            unit_size=4.0,
            wall_thickness=0.8,
            block_height=3.5,
            clearance_xy=-0.25,
            clearance_z=0.2,
            pins_enabled=True,
            tubes_enabled=True,
        ),
        blocks=[
            BlockInstance(
                cells_x=2,
                cells_y=4,
                position_x=10.0,
                position_y=5.0,
                color="#ff3300",
                quantity=12,
                sloped_cells=[[0, 0, "+X", 1]],
            ),
            BlockInstance(
                cells_x=1,
                cells_y=1,
                position_x=-3.0,
                position_y=7.5,
                color="#00aaff",
                quantity=3,
            ),
        ],
    )

    path = save_project(proj, tmp_path / "project.json")
    loaded = load_project(path)

    # Shared settings match
    assert loaded.settings.block_height == proj.settings.block_height
    assert loaded.settings.unit_size == proj.settings.unit_size
    assert loaded.settings.wall_thickness == proj.settings.wall_thickness
    assert loaded.settings.clearance_xy == proj.settings.clearance_xy
    assert loaded.settings.clearance_z == proj.settings.clearance_z
    assert loaded.settings.pins_enabled == proj.settings.pins_enabled
    assert loaded.settings.tubes_enabled == proj.settings.tubes_enabled

    # Blocks match
    assert len(loaded.blocks) == 2
    for i in range(len(proj.blocks)):
        assert loaded.blocks[i].model_dump() == proj.blocks[i].model_dump()


# ------------------------------------------------------- saved file format
def test_saved_file_has_correct_schema(tmp_path: Path) -> None:
    '''Saved JSON must contain the brickgen-project schema marker and version.'''
    proj = Project(blocks=[BlockInstance()])
    path = save_project(proj, tmp_path / "project.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"]["name"] == PROJECT_SCHEMA_NAME
    assert payload["schema"]["version"] == PROJECT_SCHEMA_VERSION


def test_saved_settings_has_no_layout_fields(tmp_path: Path) -> None:
    '''The "settings" key must NOT contain any layout field keys.'''
    proj = Project(
        blocks=[
            BlockInstance(
                cells_x=2,
                cells_y=2,
                excluded_cells=[[0, 0]],
                sloped_cells=[[1, 1, "+X", 1]],
            )
        ],
    )
    path = save_project(proj, tmp_path / "project.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    for field in LAYOUT_FIELDS:
        assert field not in payload["settings"], (
            f"Layout field '{field}' must not appear in settings"
        )


def test_saved_blocks_carry_flag_fields(tmp_path: Path) -> None:
    '''The per-block pins/tubes flags must be stored on every block.'''
    proj = Project(
        blocks=[
            BlockInstance(cells_x=2, cells_y=2),
            BlockInstance(
                cells_x=2,
                cells_y=2,
                pins_enabled=False,
                tubes_enabled=False,
            ),
        ],
    )
    path = save_project(proj, tmp_path / "project.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["blocks"][0]["pins_enabled"] is True
    assert payload["blocks"][0]["tubes_enabled"] is True
    assert payload["blocks"][1]["pins_enabled"] is False
    assert payload["blocks"][1]["tubes_enabled"] is False
    # The flags are block-level now: they must not leak into shared settings.
    assert "pins_enabled" not in payload["settings"]
    assert "tubes_enabled" not in payload["settings"]


# ------------------------------------------------------------- v4 migration
def test_v4_project_migrates_flags_from_shared_settings(tmp_path: Path) -> None:
    '''A v4 project loads with per-block flags backfilled from the shared
    settings, so old scenes keep their stud/tube look.'''
    payload = {
        "schema": {"name": "brickgen-project", "version": 4},
        "settings": {
            "unit_size": 8.0,
            "wall_thickness": 1.0,
            "block_height": 5.0,
            "clearance_xy": -0.25,
            "clearance_z": 0.0,
            "pins_enabled": False,
            "tubes_enabled": False,
        },
        "blocks": [
            {"cells_x": 2, "cells_y": 2},
            {"cells_x": 1, "cells_y": 1},
        ],
    }
    path = tmp_path / "v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)

    for block in proj.blocks:
        assert block.pins_enabled is False
        assert block.tubes_enabled is False


def test_v4_project_defaults_flags_to_true(tmp_path: Path) -> None:
    '''A v4 payload whose shared settings lack the flag keys migrates with
    the factory default (studs and tubes on).'''
    payload = {
        "schema": {"name": "brickgen-project", "version": 4},
        "settings": {"unit_size": 8.0},
        "blocks": [{"cells_x": 2, "cells_y": 2}],
    }
    path = tmp_path / "v4_min.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)

    assert proj.blocks[0].pins_enabled is True
    assert proj.blocks[0].tubes_enabled is True


def test_v4_project_mixed_flags_roundtrip(tmp_path: Path) -> None:
    '''A migrated v4 project saves as v5 and reloads to the same state.'''
    payload = {
        "schema": {"name": "brickgen-project", "version": 4},
        "settings": {
            "unit_size": 8.0,
            "wall_thickness": 1.0,
            "block_height": 5.0,
            "clearance_xy": -0.25,
            "clearance_z": 0.0,
            "pins_enabled": False,
            "tubes_enabled": True,
        },
        "blocks": [{"cells_x": 2, "cells_y": 4, "position_x": 1.0}],
    }
    path = tmp_path / "v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)
    saved = save_project(proj, tmp_path / "v5.json")
    reloaded = load_project(saved)

    # The v4 shared flag keys are vestigial after migration (flags are
    # block-level now), so compare the fields that matter for rendering.
    assert reloaded.blocks == proj.blocks
    assert reloaded.settings.unit_size == proj.settings.unit_size
    assert reloaded.settings.block_height == proj.settings.block_height
    assert reloaded.blocks[0].pins_enabled is False
    assert reloaded.blocks[0].tubes_enabled is True


# ------------------------------------------------------- legacy migration
def test_legacy_settings_migration(tmp_path: Path) -> None:
    '''A brickgen-settings preset migrates to a one-block project.'''
    legacy = {
        "schema": {"name": "brickgen-settings", "version": 3},
        "cells_x": 4,
        "cells_y": 2,
        "block_height": 4.5,
        "excluded_cells": [[0, 0], [1, 1]],
        "chamfered_cells": [[3, 0, "+X"]],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    proj = load_project(path)

    assert len(proj.blocks) == 1
    block = proj.blocks[0]
    assert block.cells_x == 4
    assert block.cells_y == 2
    assert block.excluded_cells == [[0, 0], [1, 1]]
    assert block.chamfered_cells == [[3, 0, "+X"]]
    assert block.position_x == 0.0
    assert block.position_y == 0.0
    assert block.quantity == 1
    assert proj.settings.block_height == 4.5


def test_legacy_without_schema_marker_migrates(tmp_path: Path) -> None:
    '''A plain settings file (no schema, no "blocks") also migrates.'''
    legacy = {
        "cells_x": 3,
        "cells_y": 3,
        "block_height": 5.0,
        "unit_size": 4.0,
        "wall_thickness": 0.8,
        "clearance_xy": -0.25,
        "clearance_z": 0.2,
        "pins_enabled": True,
        "tubes_enabled": True,
    }
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    proj = load_project(path)

    assert len(proj.blocks) == 1
    assert proj.blocks[0].cells_x == 3
    assert proj.settings.block_height == 5.0


# ----------------------------------------------------------- error cases
def test_unknown_schema_name_raises(tmp_path: Path) -> None:
    '''A file with an unknown schema marker must raise ProjectIOError.'''
    path = tmp_path / "other.json"
    path.write_text(
        json.dumps(
            {
                "schema": {"name": "some-other-tool", "version": 1},
                "blocks": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectIOError):
        load_project(path)


def test_unsupported_version_raises(tmp_path: Path) -> None:
    '''A project schema with version 99 must raise ProjectIOError.'''
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps(
            {
                "schema": {"name": "brickgen-project", "version": 99},
                "settings": {},
                "blocks": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProjectIOError, match="Unsupported project schema version"):
        load_project(path)


def test_broken_json_raises(tmp_path: Path) -> None:
    '''Malformed JSON must raise ProjectIOError.'''
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ProjectIOError, match="Invalid JSON"):
        load_project(path)


def test_nonexistent_file_raises(tmp_path: Path) -> None:
    '''Loading a missing file must raise ProjectIOError.'''
    with pytest.raises(ProjectIOError):
        load_project(tmp_path / "missing.json")


def test_invalid_block_color_raises(tmp_path: Path) -> None:
    '''A project with an invalid block color must raise ProjectIOError.'''
    payload = {
        "schema": {"name": "brickgen-project", "version": 4},
        "settings": {},
        "blocks": [
            {
                "cells_x": 2,
                "cells_y": 2,
                "color": "red",
                "quantity": 1,
            }
        ],
    }
    path = tmp_path / "bad_color.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectIOError, match="Invalid project"):
        load_project(path)


# ------------------------------------------------------- v6 migration from v5
def _build_v5_payload(blocks: list[dict] | None = None) -> dict:
    '''Build a v5-style project payload for testing migration.'''
    return {
        "schema": {"name": PROJECT_SCHEMA_NAME, "version": 5},
        "settings": {
            "unit_size": 6.4,
            "wall_thickness": 0.8,
            "block_height": 3.2,
            "clearance_xy": -0.15,
            "clearance_z": 0.1,
        },
        "blocks": blocks or [],
    }


def test_load_v5_payload_adds_block_uids(tmp_path: Path) -> None:
    '''Loading a v5 payload succeeds; every block gets a non-empty uid.'''
    payload = _build_v5_payload(
        [
            {"cells_x": 2, "cells_y": 2},
            {"cells_x": 1, "cells_y": 1},
        ]
    )
    path = tmp_path / "v5.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)

    assert len(proj.blocks) == 2
    for block in proj.blocks:
        assert len(block.uid) > 0
    assert proj.blocks[0].uid != proj.blocks[1].uid


def test_load_v5_payload_has_print_layout_defaults(tmp_path: Path) -> None:
    '''V5 load yields print_layout with default beds (count 1) and no pieces.'''
    payload = _build_v5_payload([{"cells_x": 2, "cells_y": 2}])
    path = tmp_path / "v5_pl.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)

    assert proj.print_layout.beds.count == 1
    assert proj.print_layout.pieces == []


def test_load_v5_payload_with_print_layout_adds_piece_uids(tmp_path: Path) -> None:
    '''Pieces missing uid in an existing print_layout gain one; bed preserved.'''
    payload = _build_v5_payload(
        [
            {"cells_x": 2, "cells_y": 2},
        ]
    )
    payload["print_layout"] = {
        "beds": {"width": 200.0, "depth": 200.0, "count": 1, "gap": 5.0},
        "pieces": [
            {"block_uid": "abc", "bed": 0, "position_x": 10.0},
            {"block_uid": "def", "bed": -1},
        ],
    }
    path = tmp_path / "v5_pl_pieces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)

    assert len(proj.print_layout.pieces) == 2
    for piece in proj.print_layout.pieces:
        assert len(piece.uid) > 0
    assert proj.print_layout.pieces[0].bed == 0
    assert proj.print_layout.pieces[1].bed == -1


def test_load_v6_payload_backfills_rotated90(tmp_path: Path) -> None:
    '''Loading a v6 payload migrates it to v7: pieces gain rotated90=False.'''
    payload = _build_v5_payload([{"cells_x": 2, "cells_y": 2}])
    # A v6 payload: uids already present (simulate a saved v6 file).
    payload["schema"] = {"name": PROJECT_SCHEMA_NAME, "version": 6}
    payload["blocks"][0]["uid"] = "blk_v6"
    payload["print_layout"] = {
        "beds": {"width": 220.0, "depth": 220.0, "count": 1, "gap": 4.0},
        "pieces": [
            {
                "uid": "pc1",
                "block_uid": "blk_v6",
                "bed": 0,
                "position_x": 8.0,
                "position_y": 8.0,
                "flipped": False,
            },
        ],
    }
    path = tmp_path / "v6_pieces.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)

    assert proj.print_layout.pieces[0].rotated90 is False
    assert proj.print_layout.pieces[0].uid == "pc1"
    # Saving again writes the CURRENT schema version.
    reloaded = load_project(save_project(proj, tmp_path / "resaved.json"))
    assert reloaded.print_layout.pieces[0].rotated90 is False


def test_roundtrip_project_with_print_layout(tmp_path: Path) -> None:
    '''Project with explicit uids and print_layout round-trips exactly.'''
    block_a_uid = "aaaa1111bbbb2222cccc3333"
    block_b_uid = "dddd4444eeee5555ffff6666"

    piece1 = PrintPiece(
        uid="piece_0001",
        block_uid=block_a_uid,
        bed=0,
        position_x=50.0,
        position_y=30.0,
        flipped=False,
        rotated90=True,
    )
    piece2 = PrintPiece(
        uid="piece_0002",
        block_uid=block_b_uid,
        bed=1,
        position_x=-20.0,
        position_y=70.0,
        flipped=True,
        rotated90=True,
    )
    piece3 = PrintPiece(
        uid="piece_0003",
        block_uid=block_a_uid,
        bed=-1,
        position_x=0.0,
        position_y=0.0,
        flipped=False,
    )

    proj = Project(
        settings=BlockSettings(
            unit_size=6.4,
            wall_thickness=0.8,
            block_height=3.2,
            clearance_xy=-0.15,
            clearance_z=0.1,
        ),
        blocks=[
            BlockInstance(uid=block_a_uid, cells_x=2, cells_y=2),
            BlockInstance(uid=block_b_uid, cells_x=3, cells_y=3),
        ],
        print_layout=PrintLayout(
            beds=PrintBedSettings(count=2, width=180.0, depth=250.0, gap=2.5),
            pieces=[piece1, piece2, piece3],
        ),
    )

    path = save_project(proj, tmp_path / "roundtrip.json")
    loaded = load_project(path)

    # Block uids survive
    assert loaded.blocks[0].uid == block_a_uid
    assert loaded.blocks[1].uid == block_b_uid

    # Bed settings survive exactly
    assert loaded.print_layout.beds.count == 2
    assert loaded.print_layout.beds.width == 180.0
    assert loaded.print_layout.beds.depth == 250.0
    assert loaded.print_layout.beds.gap == 2.5

    # Pieces survive exactly
    assert len(loaded.print_layout.pieces) == 3
    assert loaded.print_layout.pieces[0].uid == "piece_0001"
    assert loaded.print_layout.pieces[0].bed == 0
    assert loaded.print_layout.pieces[0].position_x == 50.0
    assert loaded.print_layout.pieces[0].flipped is False
    assert loaded.print_layout.pieces[0].rotated90 is True

    assert loaded.print_layout.pieces[1].uid == "piece_0002"
    assert loaded.print_layout.pieces[1].bed == 1
    assert loaded.print_layout.pieces[1].position_y == 70.0
    assert loaded.print_layout.pieces[1].flipped is True
    assert loaded.print_layout.pieces[1].rotated90 is True

    assert loaded.print_layout.pieces[2].uid == "piece_0003"
    assert loaded.print_layout.pieces[2].bed == -1


def test_v4_chain_migration_gains_uid(tmp_path: Path) -> None:
    '''Loading a v4 payload yields blocks with uid after chained migration.'''
    payload = {
        "schema": {"name": PROJECT_SCHEMA_NAME, "version": 4},
        "settings": {
            "unit_size": 8.0,
            "wall_thickness": 1.0,
            "block_height": 5.0,
            "clearance_xy": -0.25,
            "clearance_z": 0.0,
            "pins_enabled": True,
            "tubes_enabled": True,
        },
        "blocks": [
            {"cells_x": 2, "cells_y": 2},
            {"cells_x": 1, "cells_y": 1},
        ],
    }
    path = tmp_path / "v4_uid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    proj = load_project(path)

    assert len(proj.blocks) == 2
    for block in proj.blocks:
        assert len(block.uid) > 0
    assert proj.blocks[0].uid != proj.blocks[1].uid


def test_save_includes_print_layout_key(tmp_path: Path) -> None:
    '''Saving a project writes "print_layout" key into the JSON file.'''
    proj = Project(blocks=[BlockInstance()])
    path = save_project(proj, tmp_path / "has_pl.json")
    text = path.read_text(encoding="utf-8")
    assert '"print_layout"' in text


# ------------------------------------------------------------ view settings
def test_save_includes_view_key(tmp_path: Path) -> None:
    '''Saving a project writes "view" key into the JSON file.'''
    proj = Project(blocks=[BlockInstance()])
    path = save_project(proj, tmp_path / "has_view.json")
    assert '"view"' in path.read_text(encoding="utf-8")


def test_view_settings_roundtrip(tmp_path: Path) -> None:
    '''Custom per-project view settings survive save/load unchanged.'''
    proj = Project(blocks=[BlockInstance()])
    proj.view.grid_enabled = True
    proj.view.grid_scale = 5.0
    proj.view.grid_snap = True
    proj.view.placement_grid_enabled = True
    proj.view.quantity_labels_enabled = True
    proj.view.quantity_label_position = "top"

    path = save_project(proj, tmp_path / "view.json")
    loaded = load_project(path)
    assert loaded.view == proj.view


def test_invalid_view_value_raises(tmp_path: Path) -> None:
    '''A payload with an out-of-domain view value must raise ProjectIOError.'''
    payload = {
        "schema": {"name": PROJECT_SCHEMA_NAME, "version": PROJECT_SCHEMA_VERSION},
        "blocks": [],
        "view": {"grid_scale": 0.0},
    }
    path = tmp_path / "bad_view.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectIOError, match="Invalid project"):
        load_project(path)


def test_v7_payload_loads_with_default_view(tmp_path: Path) -> None:
    '''A v7 payload without "view" loads with ViewSettings defaults.

    Pre-v8 projects carry no view settings; loading falls back to the
    factory defaults and the next save writes the current schema version.
    '''
    payload = {
        "schema": {"name": PROJECT_SCHEMA_NAME, "version": 7},
        "settings": {
            field: getattr(BlockSettings(), field)
            for field in (
                "unit_size",
                "wall_thickness",
                "block_height",
                "clearance_xy",
                "clearance_z",
            )
        },
        "blocks": [],
        "print_layout": {"beds": {}, "pieces": []},
    }
    path = tmp_path / "v7.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    proj = load_project(path)
    assert proj.view == ViewSettings()

    resaved = save_project(proj, tmp_path / "resaved.json")
    resaved_payload = json.loads(resaved.read_text(encoding="utf-8"))
    assert resaved_payload["schema"]["version"] == PROJECT_SCHEMA_VERSION
    assert resaved_payload["view"]["grid_scale"] == 0.5


# ------------------------------------------------------------- tools state
def test_tools_state_roundtrip(tmp_path: Path) -> None:
    '''Custom placement tools state survives save/load unchanged.'''
    proj = Project(
        blocks=[BlockInstance()],
        print_layout=PrintLayout(
            beds=PrintBedSettings(count=4),
            tools=PrintToolsSettings(
                auto_from=2,
                auto_to=3,
                distribute_by_color=False,
                distribute_evenly=False,
            ),
        ),
    )

    path = save_project(proj, tmp_path / "tools.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["print_layout"]["tools"] == {
        "auto_from": 2,
        "auto_to": 3,
        "distribute_by_color": False,
        "distribute_evenly": False,
    }

    loaded = load_project(path)
    assert loaded.print_layout.tools == proj.print_layout.tools


def test_load_v8_payload_defaults_tools(tmp_path: Path) -> None:
    '''A v8 payload without print_layout.tools loads with defaults.

    v9 added the tools state; pre-v9 projects have no "tools" key, so
    loading falls back to the factory defaults and the next save writes
    the current schema version.
    '''
    proj = Project(blocks=[BlockInstance()])
    payload = {
        "schema": {"name": PROJECT_SCHEMA_NAME, "version": 8},
        "settings": {
            field: getattr(BlockSettings(), field)
            for field in (
                "unit_size",
                "wall_thickness",
                "block_height",
                "clearance_xy",
                "clearance_z",
            )
        },
        "blocks": [b.model_dump() for b in proj.blocks],
        # A v8 print_layout carries no "tools" key at all.
        "print_layout": {"beds": {}, "pieces": []},
        "view": {},
    }
    path = tmp_path / "v8.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_project(path)
    assert loaded.print_layout.tools == PrintToolsSettings()

    resaved = save_project(loaded, tmp_path / "resaved.json")
    resaved_payload = json.loads(resaved.read_text(encoding="utf-8"))
    assert resaved_payload["schema"]["version"] == PROJECT_SCHEMA_VERSION
    assert resaved_payload["print_layout"]["tools"] == {
        "auto_from": 1,
        "auto_to": 1,
        "distribute_by_color": True,
        "distribute_evenly": True,
    }
