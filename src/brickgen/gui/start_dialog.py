'''
File:   start_dialog.py
Brief:  Startup welcome dialog: new project, open project, recent list.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.20.0
'''

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QShowEvent
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from brickgen import __version__

# Palette mirrors the 3D viewport (preview.VIEWPORT_BG / DEFAULT_BLOCK_COLOR)
# so the modal blends into the scene shown behind it.
_BG = "#2e3338"
_LIST_BG = "#262b30"
_CARD = "#3a4046"
_CARD_HOVER = "#464d55"
_ACCENT = "#3f7fbf"
_ACCENT_HOVER = "#4f8fcb"
_TEXT = "#e8eaed"
_MUTED = "#9aa3ab"
_DIM = "#6e767d"

# Frameless card geometry: the outer dialog reserves margins around the card
# for the drop shadow (frameless windows have no native frame or shadow).
_CARD_WIDTH = 560
_WINDOW_MARGIN = 32

_ROW_HEIGHT = 46
_MAX_VISIBLE_ROWS = 6

_STYLE = f'''
QFrame#startCard {{
    background-color: {_BG};
    border-radius: 12px;
}}
QFrame#startCard QLabel {{
    color: {_TEXT};
}}
QLabel#startTitle {{
    font-size: 24pt;
    font-weight: bold;
}}
QLabel#startSubtitle {{
    color: {_MUTED};
}}
QPushButton#cardNew {{
    background-color: {_ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 16px;
    font-size: 11pt;
    font-weight: bold;
}}
QPushButton#cardNew:hover {{
    background-color: {_ACCENT_HOVER};
}}
QPushButton#cardOpen {{
    background-color: {_CARD};
    color: {_TEXT};
    border: 1px solid {_CARD_HOVER};
    border-radius: 8px;
    padding: 16px;
    font-size: 11pt;
    font-weight: bold;
}}
QPushButton#cardOpen:hover {{
    background-color: {_CARD_HOVER};
}}
QLabel#sectionLabel {{
    color: {_MUTED};
    font-size: 9pt;
    font-weight: bold;
}}
QListWidget#recentList {{
    background-color: {_LIST_BG};
    border: 1px solid {_CARD};
    border-radius: 6px;
    outline: none;
    padding: 2px;
}}
QListWidget#recentList::item {{
    border-radius: 4px;
}}
QListWidget#recentList::item:hover {{
    background-color: {_CARD_HOVER};
}}
QListWidget#recentList::item:selected {{
    background-color: {_ACCENT};
}}
QLabel#mutedLabel {{
    color: {_MUTED};
}}
'''


class StartDialog(QDialog):
    '''Startup modal offering a new project, an open dialog and recent files.

    Rendered as a frameless rounded card floating over the parent window:
    no Windows frame, not draggable, centered over the parent (the
    maximized main window) on every show.

    The window reads the outcome after :meth:`exec`:

    Attributes:
        choice: ``"new"``, ``"open"`` or ``"recent"``; ``None`` when the
            dialog was dismissed (Esc — there is no close button), which
            the caller treats the same as ``"new"`` — the default project
            is already loaded.
        recent_path: Absolute path picked from the recent list (for
            ``choice == "recent"``).
    '''

    def __init__(
        self, recent_projects: list[str], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.choice: str | None = None
        self.recent_path: str | None = None

        self.setObjectName("startDialog")
        self.setWindowTitle("BrickGen")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(_STYLE)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(48)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 150))

        card = QFrame()
        card.setObjectName("startCard")
        card.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            _WINDOW_MARGIN, _WINDOW_MARGIN, _WINDOW_MARGIN, _WINDOW_MARGIN
        )
        outer.addWidget(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(32, 28, 32, 18)
        layout.setSpacing(12)

        title = QLabel("BrickGen")
        title.setObjectName("startTitle")
        subtitle = QLabel("Parametric stud-block generator")
        subtitle.setObjectName("startSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        self._btn_new = QPushButton("New Project")
        self._btn_new.setObjectName("cardNew")
        self._btn_new.setMinimumHeight(64)
        self._btn_new.clicked.connect(self._on_new)
        self._btn_open = QPushButton("Open Project...")
        self._btn_open.setObjectName("cardOpen")
        self._btn_open.setMinimumHeight(64)
        self._btn_open.clicked.connect(self._on_open)
        actions.addWidget(self._btn_new)
        actions.addWidget(self._btn_open)
        layout.addLayout(actions)

        layout.addSpacing(8)
        section = QLabel("RECENT PROJECTS")
        section.setObjectName("sectionLabel")
        layout.addWidget(section)

        self._list = QListWidget()
        self._list.setObjectName("recentList")
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.itemClicked.connect(self._on_recent_clicked)
        self._list.itemActivated.connect(self._on_recent_clicked)
        layout.addWidget(self._list)
        self._empty = QLabel("No recent projects yet")
        self._empty.setObjectName("mutedLabel")
        layout.addWidget(self._empty)

        layout.addStretch(1)
        footer = QHBoxLayout()
        hint = QLabel("Esc — start a new project")
        hint.setObjectName("mutedLabel")
        version = QLabel(f"v{__version__}")
        version.setObjectName("mutedLabel")
        footer.addWidget(hint)
        footer.addStretch(1)
        footer.addWidget(version)
        layout.addLayout(footer)

        self._fill_recent(recent_projects)
        # Pinned after the layout is built: the height follows the content.
        self.setFixedSize(_CARD_WIDTH + 2 * _WINDOW_MARGIN, self.sizeHint().height())

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt naming
        '''Center the card over the parent window before it becomes usable.'''
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            anchor = parent.window().frameGeometry()
        else:
            anchor = self.screen().availableGeometry()
        self.move(anchor.center() - self.rect().center())

    def _fill_recent(self, recent_projects: list[str]) -> None:
        '''Populate the recent list with two-line rows (name + full path).'''
        entries = [str(p) for p in recent_projects if str(p).strip()]
        self._list.setVisible(bool(entries))
        self._empty.setVisible(not entries)
        if not entries:
            return
        self._list.setFixedHeight(
            min(len(entries), _MAX_VISIBLE_ROWS) * _ROW_HEIGHT + 8
        )
        for entry in entries:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setSizeHint(QSize(self._list.width(), _ROW_HEIGHT))
            exists = Path(entry).is_file()
            if not exists:
                item.setFlags(
                    item.flags()
                    & ~Qt.ItemFlag.ItemIsEnabled
                    & ~Qt.ItemFlag.ItemIsSelectable
                )
                item.setToolTip("File not found")
            self._list.addItem(item)
            self._list.setItemWidget(item, self._make_row(entry, exists))

    def _make_row(self, entry: str, exists: bool) -> QWidget:
        '''Build the name/path label pair shown inside one recent row.'''
        row = QWidget()
        lay = QVBoxLayout(row)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(1)
        name = QLabel(Path(entry).name)
        name.setStyleSheet(
            f"color: {_TEXT if exists else _MUTED}; font-weight: bold;"
            + (" font-style: italic;" if not exists else "")
        )
        path = QLabel(entry)
        path.setStyleSheet(f"color: {_MUTED if exists else _DIM}; font-size: 8pt;")
        lay.addWidget(name)
        lay.addWidget(path)
        return row

    def _on_new(self) -> None:
        '''Continue with the default (already loaded) project.'''
        self.choice = "new"
        self.accept()

    def _on_open(self) -> None:
        '''Ask the main window to run the regular open-project flow.'''
        self.choice = "open"
        self.accept()

    def _on_recent_clicked(self, item: QListWidgetItem) -> None:
        '''Open the project behind the clicked recent entry.'''
        if not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        self.choice = "recent"
        self.recent_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
