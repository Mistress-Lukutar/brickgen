'''
File:   exceptions.py
Brief:  Custom exception hierarchy for BrickGen.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations


class BrickGenError(Exception):
    '''Base exception for all BrickGen errors.'''


class SettingsValidationError(BrickGenError):
    '''Raised when block settings fail validation.'''


class SettingsIOError(BrickGenError):
    '''Raised when settings import/export fails (unreadable/invalid JSON file).'''


class TemplateError(BrickGenError):
    '''Raised when a template file is unreadable or invalid.'''


class ExportError(BrickGenError):
    '''Raised when a 3D model export fails or the format is unsupported.'''


class ProjectIOError(BrickGenError):
    '''Raised when a project file cannot be read, written, or parsed.'''
