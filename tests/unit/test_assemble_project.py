'''
File:   test_assemble_project.py
Brief:  Project-model and IO tests for the Assemble state (schema v11).
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brickgen.core.exceptions import SettingsValidationError
from brickgen.core.project import (
    AssembledPiece,
    AssembleState,
    BlockInstance,
    Project,
)
from brickgen.services.project_io import (
    PROJECT_SCHEMA_VERSION,
    load_project,
    save_project,
)


def test_default_project_has_empty_assemble() -> None:
    '''A fresh project carries an empty assemble state with defaults.'''
    project = Project()
    assert project.assemble == AssembleState()
    assert project.assemble.pieces == []
    assert project.assemble.view.snap_to_grid is True
    assert project.assemble.view.snap_to_parts is True
    assert project.assemble.view.grid_enabled is True


def test_block_defaults_not_linked() -> None:
    '''Blocks default to a manual quantity (no assemble link).'''
    assert BlockInstance().quantity_linked is False


def test_assembled_piece_validation() -> None:
    '''rot_z is a quarter-turn count; uids must be non-empty; positions
    must be finite.'''
    with pytest.raises(SettingsValidationError):
        AssembledPiece(block_uid="a", rot_z=4)
    with pytest.raises(SettingsValidationError):
        AssembledPiece(block_uid="")
    with pytest.raises(SettingsValidationError):
        AssembledPiece(block_uid="a", position_z=float("nan"))
    piece = AssembledPiece(block_uid="a", rot_z=3, flipped=True)
    assert piece.rot_z == 3 and piece.flipped is True


def test_save_load_roundtrip_with_assemble(tmp_path: Path) -> None:
    '''Assemble pieces, view settings and linked quantities survive IO.'''
    project = Project(blocks=[BlockInstance(uid="a", quantity=2)])
    project.blocks[0].quantity_linked = True
    project.assemble.pieces = [
        AssembledPiece(block_uid="a", position_x=10.0, position_z=8.0, rot_z=2),
        AssembledPiece(block_uid="a", flipped=True),
    ]
    path = save_project(project, tmp_path / "proj.json")
    loaded = load_project(path)
    assert len(loaded.assemble.pieces) == 2
    assert loaded.assemble.pieces[0].position_z == pytest.approx(8.0)
    assert loaded.assemble.pieces[0].rot_z == 2
    assert loaded.assemble.pieces[1].flipped is True
    assert loaded.blocks[0].quantity_linked is True


def test_saved_file_has_schema_v11(tmp_path: Path) -> None:
    '''Saved JSON carries the assemble key and the current version.'''
    project = Project(blocks=[BlockInstance(uid="a")])
    project.assemble.pieces.append(AssembledPiece(block_uid="a"))
    path = save_project(project, tmp_path / "proj.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == {
        "name": "brickgen-project",
        "version": PROJECT_SCHEMA_VERSION,
    }
    assert "assemble" in payload
    assert len(payload["assemble"]["pieces"]) == 1


def test_v10_project_loads_with_default_assemble(tmp_path: Path) -> None:
    '''A v10 payload (no assemble key) loads with the model defaults.'''
    payload = {
        "schema": {"name": "brickgen-project", "version": 10},
        "settings": {},
        "blocks": [
            {
                "uid": "blk",
                "cells_x": 2,
                "cells_y": 1,
            }
        ],
        "print_layout": {},
        "view": {},
        "export": {},
    }
    path = tmp_path / "v10.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    project = load_project(path)
    assert project.assemble == AssembleState()
    assert len(project.blocks) == 1
    # Saving again writes the CURRENT schema version.
    saved = save_project(project, tmp_path / "upgraded.json")
    reloaded = load_project(saved)
    assert json.loads(saved.read_text(encoding="utf-8"))["schema"]["version"] == (
        PROJECT_SCHEMA_VERSION
    )
    assert reloaded.blocks[0].uid == "blk"


def test_linked_quantity_zero_roundtrip(tmp_path: Path) -> None:
    '''A linked block with zero quantity (no placed copies) roundtrips.'''
    project = Project(blocks=[BlockInstance(uid="a", quantity=0)])
    project.blocks[0].quantity_linked = True
    path = save_project(project, tmp_path / "proj.json")
    loaded = load_project(path)
    assert loaded.blocks[0].quantity == 0
    assert loaded.blocks[0].quantity_linked is True


def test_invalid_assemble_piece_rejected(tmp_path: Path) -> None:
    '''A corrupt assembled piece fails the whole load (payload typos).'''
    payload = {
        "schema": {"name": "brickgen-project", "version": PROJECT_SCHEMA_VERSION},
        "settings": {},
        "blocks": [{"uid": "blk", "cells_x": 1, "cells_y": 1}],
        "assemble": {"pieces": [{"block_uid": "blk", "rot_z": 9}]},
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="rot_z"):
        load_project(path)
