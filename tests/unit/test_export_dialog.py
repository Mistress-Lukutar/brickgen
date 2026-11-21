'''
File:   test_export_dialog.py
Brief:  Unit tests for the export dialog format/extension resolution.
Author: Mistress-Lukutar
Date:   2026-11-20
Version: v0.13.0
'''

from __future__ import annotations

from brickgen.gui.app import resolve_export_target
from brickgen.services.export_service import SUPPORTED_FORMATS


def test_step_is_first_default_format() -> None:
    '''STEP must be the first entry of the filter order.'''
    assert list(SUPPORTED_FORMATS)[0] == "STEP"


def test_filter_wins_over_typed_suffix() -> None:
    '''3MF filter with a .step name yields block.3mf and 3MF format.'''
    path, fmt = resolve_export_target("block_2x2.step", "3MF (*.3mf)")
    assert fmt == "3MF"
    assert path.endswith("block_2x2.3mf")


def test_step_filter_keeps_step() -> None:
    '''STEP filter keeps the typed .step name unchanged.'''
    path, fmt = resolve_export_target("block_2x2.step", "STEP (*.step)")
    assert fmt == "STEP"
    assert path.endswith("block_2x2.step")


def test_stl_filter_replaces_suffix() -> None:
    '''STL filter replaces a .step suffix with .stl.'''
    path, fmt = resolve_export_target("block.step", "STL (*.stl)")
    assert fmt == "STL"
    assert path.endswith("block.stl")


def test_stl_filter_appends_missing_suffix() -> None:
    '''A name without suffix gains the filter extension.'''
    path, fmt = resolve_export_target("block", "STL (*.stl)")
    assert fmt == "STL"
    assert path.endswith("block.stl")


def test_all_files_uses_typed_known_suffix() -> None:
    '''With the All-files filter a known suffix is kept as the format.'''
    path, fmt = resolve_export_target("block.stl", "All files (*.*)")
    assert fmt == "STL"
    assert path.endswith("block.stl")


def test_all_files_unknown_suffix_falls_back_to_step() -> None:
    '''With the All-files filter an unknown suffix falls back to STEP.'''
    path, fmt = resolve_export_target("block.obj", "All files (*.*)")
    assert fmt == "STEP"
    assert path.endswith("block.step")
