'''
File:   test_export_dialog.py
Brief:  Unit tests for the export format menu order.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.18.0
'''

from __future__ import annotations

from brickgen.services.export_service import SUPPORTED_FORMATS


def test_step_is_first_default_format() -> None:
    '''STEP must be the first entry of the filter order.'''
    assert list(SUPPORTED_FORMATS)[0] == "STEP"
