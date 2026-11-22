'''
File:   conftest.py
Brief:  Shared pytest fixtures for BrickGen tests.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.15.0
'''

from __future__ import annotations

import pytest

from brickgen.core.models import BlockSettings


@pytest.fixture
def default_settings() -> BlockSettings:
    '''Factory defaults matching the specification (1x1 block).'''
    return BlockSettings()


@pytest.fixture
def settings_2x6() -> BlockSettings:
    '''A 2x6 form example.'''
    return BlockSettings(cells_x=2, cells_y=6)
