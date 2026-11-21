'''
File:   __init__.py
Brief:  BrickGen package root - parametric stud block generator.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.14.0
'''

from __future__ import annotations

from brickgen.core.builder import build_block
from brickgen.core.models import BlockSettings

__all__ = ["BlockSettings", "build_block", "__version__"]

__version__ = "0.14.0"
