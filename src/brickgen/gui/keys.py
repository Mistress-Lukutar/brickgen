'''
File:   keys.py
Brief:  Layout-independent keyboard-key resolution for hotkey handlers.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent


def native_vk_to_key(vk: int) -> Qt.Key | None:
    '''Map a native virtual-key code to a Qt key for letter keys, else None.

    Windows letter VK codes are positional Latin regardless of layout:
    VK_A=0x41..VK_Z=0x5A → return ``Qt.Key(vk)`` (Qt.Key_A..Key_Z share
    these values). Also accept lowercase codes 0x61..0x7A (X11 keysyms of
    Latin layouts) → return ``Qt.Key(vk - 0x20)``. Any other value → None.

    WHY: Qt resolves character keys through the active layout, so
    ``event.key()`` returns locale character codes (e.g. Cyrillic) and
    letter comparisons fail; native virtual keys stay positional.
    '''
    if 0x41 <= vk <= 0x5A:  # A-Z uppercase VK codes
        return Qt.Key(vk)
    if 0x61 <= vk <= 0x7A:  # a-z X11 keysyms of Latin layouts
        return Qt.Key(vk - 0x20)
    return None


def event_key(event: QKeyEvent) -> Qt.Key:
    '''Layout-independent key for a key event.

    Falls back to ``event.key()`` when the native code is unknown/unset
    (0 — synthetic events, non-letter keys like Escape/Delete/Enter, and
    non-Windows platforms).
    '''
    _vk = native_vk_to_key(event.nativeVirtualKey())
    if _vk is not None:
        return _vk
    return Qt.Key(event.key())
