'''
File:   test_export_service.py
Brief:  Unit tests for model export to 3D formats.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.21.0
'''

from __future__ import annotations

from pathlib import Path

import pytest

from brickgen.core.builder import build_block
from brickgen.core.exceptions import ExportError
from brickgen.core.models import BlockSettings
from brickgen.services.export_service import export_model


@pytest.fixture
def model_1x1():
    '''A generated default 1x1 block model.'''
    return build_block(BlockSettings())


def test_export_step(model_1x1, tmp_path: Path) -> None:
    '''STEP export writes a non-empty file.'''
    out = export_model(model_1x1, tmp_path / "block.step", "STEP")
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_stl(model_1x1, tmp_path: Path) -> None:
    '''STL export writes a non-empty file.'''
    out = export_model(model_1x1, tmp_path / "block.stl", "STL")
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_3mf(model_1x1, tmp_path: Path) -> None:
    '''3MF export writes a non-empty file.'''
    out = export_model(model_1x1, tmp_path / "block.3mf", "3MF")
    assert out.exists()
    assert out.stat().st_size > 0


def test_export_unsupported_format(model_1x1, tmp_path: Path) -> None:
    '''Unknown formats raise ExportError.'''
    with pytest.raises(ExportError):
        export_model(model_1x1, tmp_path / "block.obj", "OBJ")


def test_export_creates_parent_dirs(model_1x1, tmp_path: Path) -> None:
    '''Missing parent directories are created automatically.'''
    out = export_model(model_1x1, tmp_path / "deep" / "dir" / "block.stl", "STL")
    assert out.exists()
