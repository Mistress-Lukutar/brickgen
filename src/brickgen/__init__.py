'''
File:   __init__.py
Brief:  BrickGen package root - parametric stud block generator.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.4.0
'''

from __future__ import annotations

from brickgen.core.models import BlockSettings
from brickgen.core.builder import build_block

__all__ = ["BlockSettings", "build_block", "__version__"]

__version__ = "0.8.1"
