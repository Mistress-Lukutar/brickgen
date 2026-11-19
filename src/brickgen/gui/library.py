'''
File:   library.py
Brief:  Template library panel: thumbnail previews, select, save, refresh.
Author: Mistress-Lukutar
Date:   2026-11-19
Version: v0.11.0
'''

from __future__ import annotations

import logging
from collections.abc import Callable

import pyvista as pv
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from brickgen.core.exceptions import BrickGenError
from brickgen.core.models import BlockSettings
from brickgen.gui.jobs import SKIPPED, JobError, JobQueue
from brickgen.gui.preview import BLOCK_COLOR, mesh_from_arrays
from brickgen.services.template_library import list_templates, save_template

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = (192, 144)

# Fixed card cell: thumbnail + caption, matching the old icon-grid cell.
CARD_SIZE = (THUMBNAIL_SIZE[0] + 12, THUMBNAIL_SIZE[1] + 40)


def render_thumbnail(mesh: pv.PolyData) -> QPixmap:
    '''Render an off-screen preview image for a prepared preview mesh.

    UI-thread only: creates a small off-screen VTK plotter. The heavy
    build + tessellation happens beforehand on the background job queue.

    Args:
        mesh: Pre-tessellated PyVista PolyData to render.

    Returns:
        Thumbnail pixmap (white background, iso camera view).
    '''
    plotter = pv.Plotter(off_screen=True, window_size=list(THUMBNAIL_SIZE))
    try:
        plotter.set_background("white")  # type: ignore[arg-type]  # pyvista stub bug
        # lighting="three lights" matches the preview shading (stub says bool).
        plotter.add_mesh(
            mesh,
            color=BLOCK_COLOR,
            lighting="three lights",  # type: ignore[arg-type]
        )
        plotter.camera_position = "iso"
        img = plotter.screenshot(return_img=True)
    finally:
        plotter.close()
    if img is None:  # return_img=True always returns an ndarray; defensive
        raise RuntimeError("Off-screen screenshot returned no image")
    height, width = img.shape[0], img.shape[1]
    image = QImage(img.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


class TemplateLibraryPanel(QWidget):
    '''Right-side template library: click a card to apply its parameters.

    Templates come from the builtin directory and the user directory.
    Thumbnails are built asynchronously on the background job queue so that
    the UI thread stays responsive during the heavy OCC tessellation.

    The cards are laid out in an explicit ``QGridLayout`` (fixed-size
    buttons in as many columns as the current width allows) instead of a
    ``QListWidget`` in icon mode: the list's flow layout computes item
    positions from a viewport width that can be stale around the startup
    splitter/maximize resizes, permanently offsetting the grid.
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
        # Two columns of thumbnail cards minimum: 2 grid cells + scrollbar
        # and margins, so the splitter never squeezes the panel to one column.
        self.setMinimumWidth(2 * CARD_SIZE[0] + 28)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Template library")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cards_host = QWidget()
        self._cards_layout = QGridLayout(self._cards_host)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(4)
        self._scroll.setWidget(self._cards_host)
        layout.addWidget(self._scroll, stretch=1)

        # Card widgets in display order; rebuilt by refresh(), positioned by
        # _relayout_cards(). A dict name -> button maps thumbnails to cards.
        self._cards: dict[str, QToolButton] = {}
        self._card_order: list[QWidget] = []

        save_btn = QPushButton("Save as template...")
        save_btn.clicked.connect(self._on_save_template)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(save_btn)
        layout.addWidget(refresh_btn)

        self.refresh()

    def resizeEvent(self, event: QResizeEvent) -> None:
        '''Re-wrap the card grid for the new width (once the layout settles).

        The deferred call runs after the scroll-area viewport has been
        resized along with the panel, so the column count is computed from
        the settled width.
        '''
        super().resizeEvent(event)
        QTimer.singleShot(0, self._relayout_cards)

    def force_relayout(self) -> None:
        '''Re-wrap the card grid at the panel's current width.'''
        self._relayout_cards()

    def _relayout_cards(self) -> None:
        '''Place the card buttons into the grid, wrapping by viewport width.

        ``QGridLayout`` positions are computed exactly here — no flow
        layout is involved, so a resize cannot leave the cards offset.
        '''
        viewport_w = self._scroll.viewport().width()
        cols = max(1, viewport_w // CARD_SIZE[0])
        for index, card in enumerate(self._card_order):
            self._cards_layout.addWidget(
                card,
                index // cols,
                index % cols,
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            )

    def refresh(self) -> None:
        '''Reload the template list and (re)render all thumbnails.

        The card grid renders immediately; each thumbnail fills in as its
        build job completes on the background queue.
        '''
        self._refresh_seq += 1
        refresh_seq = self._refresh_seq
        self._thumb_meta.clear()
        self._cards.clear()
        self._card_order.clear()
        self._clear_cards_layout()
        templates = list_templates()
        for template in templates:
            button = QToolButton()
            button.setText(template.name)
            button.setIconSize(QSize(*THUMBNAIL_SIZE))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setFixedSize(*CARD_SIZE)
            button.setToolTip(
                f"{template.settings.describe()} | height "
                f"{template.settings.total_height:.1f} mm"
            )
            button.clicked.connect(
                lambda checked=False, tpl=template: self.apply_requested.emit(
                    tpl.settings
                )
            )
            self._cards[template.name] = button
            self._card_order.append(button)
            payload = {'settings': template.settings.model_dump()}
            job_id = self._jobs.submit('thumbnail', payload)
            self._thumb_meta[job_id] = (template.name, refresh_seq)

        if not templates:
            empty = QLabel("(no templates found)")
            self._card_order.append(empty)

        self._relayout_cards()

    def _clear_cards_layout(self) -> None:
        '''Remove and dispose every widget currently in the card grid.'''
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

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

        button = self._cards.get(name)
        if button is None:
            return
        if not isinstance(result, dict):
            logger.warning("Thumbnail job for %s returned unexpected payload", name)
            return
        try:
            mesh = mesh_from_arrays(result)
            button.setIcon(render_thumbnail(mesh))
        except Exception:  # noqa: BLE001 - VTK off-screen render can fail on broken GL
            logger.warning("Thumbnail VTK render failed for %s", name)

    def _on_save_template(self) -> None:
        '''Ask for a name and save the current form settings as a template.

        The recipe name is pre-filled, so templates of different blocks
        stay distinguishable by default.
        '''
        name, ok = QInputDialog.getText(
            self,
            "Save as template",
            "Template name:",
            text=self._get_current_settings().recipe_name(),
        )
        if not ok or not name.strip():
            return
        try:
            save_template(name, self._get_current_settings())
        except BrickGenError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.refresh()
