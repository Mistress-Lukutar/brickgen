'''
File:   test_keys.py
Brief:  Layout-independent keyboard-key resolution unit tests.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

from brickgen.gui.keys import event_key, native_vk_to_key


def test_native_vk_to_key_letter_mapping() -> None:
    '''Map Windows positional VK codes to Qt letter keys.'''
    # Uppercase range 0x41..0x5A (A-Z)
    assert native_vk_to_key(0x41) is Qt.Key.Key_A
    assert native_vk_to_key(0x52) is Qt.Key.Key_R
    assert native_vk_to_key(0x5A) is Qt.Key.Key_Z
    # Lowercase X11 keysyms 0x61..0x7A (a-z → subtract 0x20)
    assert native_vk_to_key(0x72) is Qt.Key.Key_R
    # Non-letter codes return None
    assert native_vk_to_key(0x20) is None   # space
    assert native_vk_to_key(0x1B) is None   # escape
    assert native_vk_to_key(0) is None      # unset


def test_event_key_fallback() -> None:
    '''event_key falls back to event.key() when nativeVirtualKey is 0.'''
    # Synthetic events (nativeVirtualKey defaults to 0) fall through to
    # event.key().
    ev_r = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_R,
        Qt.KeyboardModifier.NoModifier,
    )
    assert event_key(ev_r) == Qt.Key.Key_R

    ev_esc = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Escape,
        Qt.KeyboardModifier.NoModifier,
    )
    assert event_key(ev_esc) == Qt.Key.Key_Escape


def test_layout_bug_regression_cyrillic() -> None:
    '''Cyrillic layout press resolves via native VK code, not key().

    On a Russian layout the physical R key produces Cyrillic char 0x043A
    in event.key(), but nativeVirtualKey stays at 0x52. The resolver must
    pick up the positional Latin key from the native path.
    '''
    # Build an event where key() returns Cyrillic but nativeVK is Latin R
    ev = QKeyEvent(
        QEvent.Type.KeyPress,
        0x043A,           # Cyrillic char → Qt.Key(0x043A) == 1082
        Qt.KeyboardModifier.NoModifier,
        0,                # nativeScanCode
        0x52,             # nativeVirtualKey = VK_R
        0,                # nativeModifiers
    )
    assert ev.key() != Qt.Key.Key_R         # key() returns Cyrillic code
    assert ev.nativeVirtualKey() == 0x52     # native path has Latin R
    assert event_key(ev) is Qt.Key.Key_R    # resolver picks up native VK
