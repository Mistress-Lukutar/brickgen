'''
File:   __init__.py
Brief:  BrickGen package root - parametric LEGO-like block generator.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.1.0
'''

from __future__ import annotations

from brickgen.core.models import BlockSettings
from brickgen.core.builder import build_block

__all__ = ["BlockSettings", "build_block", "__version__"]

__version__ = "v0.1.0"
