'''
File:   test_start_dialog.py
Brief:  Unit tests for the startup welcome dialog (new/open/recent).
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from brickgen.gui.start_dialog import (  # noqa: E402
    _MAX_VISIBLE_ROWS,
    _ROW_HEIGHT,
    StartDialog,
)


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


@pytest.fixture
def dialog():
    '''A bare start dialog with no recent projects.'''
    _ensure_app()
    dlg = StartDialog([])
    yield dlg
    dlg.deleteLater()


def test_new_button_sets_choice(dialog: StartDialog) -> None:
    '''The New Project card reports choice "new" and accepts the dialog.'''
    dialog.show()
    dialog._btn_new.click()
    assert dialog.choice == "new"
    assert dialog.result() == int(dialog.DialogCode.Accepted)


def test_open_button_sets_choice(dialog: StartDialog) -> None:
    '''The Open Project card reports choice "open" and accepts the dialog.'''
    dialog.show()
    dialog._btn_open.click()
    assert dialog.choice == "open"
    assert dialog.result() == int(dialog.DialogCode.Accepted)


def test_empty_recent_shows_placeholder(dialog: StartDialog) -> None:
    '''Without recents the list is hidden and a muted hint is shown.'''
    assert dialog._list.isHidden()
    assert dialog._empty.isVisibleTo(dialog)


def test_recent_click_sets_path(tmp_path: Path) -> None:
    '''Clicking a recent entry reports choice "recent" and the file path.'''
    _ensure_app()
    path = tmp_path / "proj.json"
    path.write_text("{}", encoding="utf-8")
    dlg = StartDialog([str(path)])
    dlg.show()
    item = dlg._list.item(0)
    assert item is not None
    dlg._list.itemClicked.emit(item)
    assert dlg.choice == "recent"
    assert Path(dlg.recent_path) == path
    dlg.deleteLater()


def test_missing_recent_entry_disabled(tmp_path: Path) -> None:
    '''Entries whose file no longer exists are grayed and unclickable.'''
    _ensure_app()
    existing = tmp_path / "proj.json"
    existing.write_text("{}", encoding="utf-8")
    ghost = tmp_path / "ghost.json"
    dlg = StartDialog([str(existing), str(ghost)])
    good, bad = dlg._list.item(0), dlg._list.item(1)
    assert good.flags() & Qt.ItemFlag.ItemIsEnabled
    assert not bad.flags() & Qt.ItemFlag.ItemIsEnabled
    assert bad.toolTip() == "File not found"
    # Clicking a disabled entry is a no-op (no choice recorded).
    dlg._on_recent_clicked(bad)
    assert dlg.choice is None
    dlg.deleteLater()


def test_many_recent_entries_cap_visible_rows(tmp_path: Path) -> None:
    '''Long recents scroll instead of growing the dialog endlessly.'''
    _ensure_app()
    entries = [str(tmp_path / f"p{i}.json") for i in range(12)]
    dlg = StartDialog(entries)
    assert dlg._list.count() == 12
    assert dlg._list.maximumHeight() == _MAX_VISIBLE_ROWS * _ROW_HEIGHT + 8
    dlg.deleteLater()


def test_reject_leaves_choice_none(dialog: StartDialog) -> None:
    '''Escaping the dialog records no choice (caller keeps defaults).'''
    dialog.choice = None
    dialog.reject()
    assert dialog.choice is None


def test_frameless_and_centered_on_screen() -> None:
    '''The dialog is a frameless card centered over the screen.'''
    _ensure_app()
    dlg = StartDialog([])
    dlg.show()
    assert bool(dlg.windowFlags() & Qt.WindowType.FramelessWindowHint)
    assert dlg.geometry().center() == dlg.screen().availableGeometry().center()
    dlg.deleteLater()


def test_centered_over_parent_window() -> None:
    '''With a parent the card centers over the parent window geometry.'''
    _ensure_app()
    parent = QWidget()
    parent.resize(400, 300)
    parent.move(50, 40)
    parent.show()
    dlg = StartDialog([], parent)
    dlg.show()
    assert dlg.geometry().center() == parent.window().frameGeometry().center()
    dlg.deleteLater()
    parent.deleteLater()
