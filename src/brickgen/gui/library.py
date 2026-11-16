'''
File:   library.py
Brief:  Template library panel: thumbnail previews, select, save, refresh.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.8.0
'''

from __future__ import annotations

import logging
from collections.abc import Callable

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

from brickgen.core.exceptions import BrickGenError
from brickgen.core.models import BlockSettings
from brickgen.gui.jobs import SKIPPED, JobError, JobQueue
from brickgen.gui.preview import BLOCK_COLOR, mesh_from_arrays
from brickgen.services.template_library import Template, list_templates, save_template

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = (192, 144)


def render_thumbnail(mesh: pv.PolyData) -> QPixmap:
    '''Render an off-screen preview image for a prepared preview mesh.

    UI-thread only: creates a small off-screen VTK plotter. The heavy
    build + tessellation happens beforehand on the background job queue.

    Args:
        mesh: Pre-tessellated PyVista PolyData to render.

    Returns:
        Thumbnail pixmap (white background, iso camera view).
    '''
    plotter = pv.Plotter(off_screen=True, window_size=THUMBNAIL_SIZE)
    try:
        plotter.set_background("white")
        plotter.add_mesh(mesh, color=BLOCK_COLOR, lighting="three lights")
        plotter.camera_position = "iso"
        img = plotter.screenshot(return_img=True)
    finally:
        plotter.close()
    height, width = img.shape[0], img.shape[1]
    image = QImage(img.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


class TemplateLibraryPanel(QWidget):
    '''Right-side template library: click a card to apply its parameters.

    Templates come from the builtin directory and the user directory.
    Thumbnails are built asynchronously on the background job queue so that
    the UI thread stays responsive during the heavy OCC tessellation.
    '''

    # Emitted with the settings of the template the user clicked.
    apply_requested = Signal(object)

    def __init__(
        self,
        current_settings: Callable[[], BlockSettings],
        jobs: JobQueue,
        parent: QWidget | None = None,
    ) -> None:
        '''Create the library panel.

        Args:
            current_settings: Callback returning the settings currently
                entered in the form (used by "Save as template").
            jobs: Background job queue on which thumbnail build work is
                dispatched.
            parent: Optional parent widget.
        '''
        super().__init__(parent)
        self._get_current_settings = current_settings
        self._jobs = jobs
        self._refresh_seq = 0
        self._thumb_meta: dict[int, tuple[str, int]] = {}  # job id -> (name, seq)
        self._jobs.job_done.connect(self._on_job_done)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Template library")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(*THUMBNAIL_SIZE))
        # Fixed grid cells: without a grid size the icon-mode flow layout lets
        # multi-line item labels overlap neighbouring cards (icons painted on
        # top of each other) once thumbnails fill in asynchronously.
        self._list.setGridSize(QSize(THUMBNAIL_SIZE[0] + 12, THUMBNAIL_SIZE[1] + 40))
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
        '''Reload the template list and (re)render all thumbnails.

        The item list renders immediately; each thumbnail fills in as its
        build job completes on the background queue.
        '''
        self._refresh_seq += 1
        refresh_seq = self._refresh_seq
        self._thumb_meta.clear()
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
            payload = {'settings': template.settings.model_dump()}
            job_id = self._jobs.submit('thumbnail', payload)
            self._thumb_meta[job_id] = (template.name, refresh_seq)

        if not templates:
            self._list.addItem(QListWidgetItem("(no templates found)"))

    def _find_item(self, name: str) -> QListWidgetItem | None:
        '''Find a list item by its display text.

        Args:
            name: Display text to match.

        Returns:
            The matching item, or ``None`` if not found.
        '''
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.text() == name:
                return item
        return None

    def _on_job_done(self, job_id: int, result: object) -> None:
        '''Handle a completed thumbnail-build job.

        Args:
            job_id: The job identifier returned by :meth:`JobQueue.submit`.
            result: Return value from the job callable, :data:`SKIPPED`, or
                a :class:`JobError`.
        '''
        meta = self._thumb_meta.pop(job_id, None)
        if meta is None:
            return  # Foreign job (e.g. generation/export — other slots handle those).

        name, refresh_seq = meta
        if refresh_seq != self._refresh_seq:
            return  # A newer refresh replaced the list; discard stale result.

        if isinstance(result, JobError):
            logger.warning("Thumbnail build failed for %s: %s", name, result.exc)
            return

        if result is SKIPPED:
            return  # Not expected (thumbnail jobs are never keyed); defensive.

        item = self._find_item(name)
        if item is None:
            return
        try:
            mesh = mesh_from_arrays(result)
            item.setIcon(render_thumbnail(mesh))
        except Exception:  # noqa: BLE001 - VTK off-screen render can fail on broken GL
            logger.warning("Thumbnail VTK render failed for %s", name)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        '''Emit the settings of the clicked template.'''
        template = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(template, Template):
            self.apply_requested.emit(template.settings)

    def _on_save_template(self) -> None:
        '''Ask for a name and save the current form settings as a template.'''
        name, ok = QInputDialog.getText(
            self,
            "Save as template",
            "Template name:",
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
