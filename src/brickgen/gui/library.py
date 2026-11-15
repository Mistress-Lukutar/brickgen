'''
File:   library.py
Brief:  Template library panel: thumbnail previews, select, save, refresh.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.5.0
'''

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from brickgen.core.builder import build_block
from brickgen.core.exceptions import BrickGenError
from brickgen.core.models import BlockSettings
from brickgen.gui.preview import BLOCK_COLOR, solid_to_mesh
from brickgen.services.template_library import Template, list_templates, save_template

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = (192, 144)


def render_thumbnail(settings: BlockSettings) -> QPixmap:
    '''Render an off-screen preview image for the given settings.

    Args:
        settings: Validated block settings to render.

    Returns:
        Thumbnail pixmap (white background, iso camera view).
    '''
    mesh = solid_to_mesh(build_block(settings).val())
    plotter = pv.Plotter(off_screen=True, window_size=THUMBNAIL_SIZE)
    try:
        plotter.set_background("white")
        plotter.add_mesh(mesh, color=BLOCK_COLOR, lighting="three lights")
        plotter.camera_position = "iso"
        img = plotter.screenshot(return_img=True)
    finally:
        plotter.close()
    height, width = img.shape[0], img.shape[1]
    image = QImage(
        img.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(image)


class TemplateLibraryPanel(QWidget):
    '''Right-side template library: click a card to apply its parameters.

    Templates come from the builtin directory and the user directory.
    '''

    # Emitted with the settings of the template the user clicked.
    apply_requested = Signal(object)

    def __init__(
        self,
        current_settings: Callable[[], BlockSettings],
        parent: QWidget | None = None,
    ) -> None:
        '''Create the library panel.

        Args:
            current_settings: Callback returning the settings currently
                entered in the form (used by "Save as template").
            parent: Optional parent widget.
        '''
        super().__init__(parent)
        self._get_current_settings = current_settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Template library")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(*THUMBNAIL_SIZE))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setWordWrap(True)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list, stretch=1)

        save_btn = QPushButton("Save as template...")
        save_btn.clicked.connect(self._on_save_template)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(save_btn)
        layout.addWidget(refresh_btn)

        self.refresh()

    def refresh(self) -> None:
        '''Reload the template list and (re)render all thumbnails.'''
        self._list.clear()
        templates = list_templates()
        for template in templates:
            item = QListWidgetItem(template.name)
            item.setData(Qt.ItemDataRole.UserRole, template)
            item.setToolTip(
                f"{template.settings.describe()} | height "
                f"{template.settings.total_height:.1f} mm"
            )
            self._list.addItem(item)
            try:
                item.setIcon(render_thumbnail(template.settings))
            except BrickGenError as exc:
                logger.warning("Thumbnail render failed for %s: %s", template.name, exc)
        if not templates:
            self._list.addItem(QListWidgetItem("(no templates found)"))

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        '''Emit the settings of the clicked template.'''
        template = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(template, Template):
            self.apply_requested.emit(template.settings)

    def _on_save_template(self) -> None:
        '''Ask for a name and save the current form settings as a template.'''
        name, ok = QInputDialog.getText(
            self, "Save as template", "Template name:",
            text=f"Block {self._get_current_settings().describe()}",
        )
        if not ok or not name.strip():
            return
        try:
            save_template(name, self._get_current_settings())
        except BrickGenError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.refresh()
