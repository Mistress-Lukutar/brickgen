'''
File:   parts_panel.py
Brief:  Right-side Assemble-tab panel: planned parts with thumbnails, color
        variants and placed/planned counts.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import json
import logging

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from brickgen.gui.jobs import SKIPPED, JobError, JobQueue
from brickgen.gui.library import THUMBNAIL_SIZE, render_thumbnail
from brickgen.gui.preview import mesh_from_arrays
from brickgen.services.assemble_service import PartGroup

logger = logging.getLogger(__name__)

THUMBNAIL_PAYLOAD_TOLERANCE = 0.15
THUMBNAIL_PAYLOAD_ANGULAR = 0.1

# Fixed card cell: thumbnail + name + color combo + counts (taller than a
# library card, same width so both right panels align in the splitter).
CARD_WIDTH = THUMBNAIL_SIZE[0] + 12
CARD_HEIGHT = THUMBNAIL_SIZE[1] + 96
PART_CARD_SIZE = (CARD_WIDTH, CARD_HEIGHT)


def _color_chip(color: str) -> QPixmap:
    '''Small color-swatch pixmap for a combo item ('#rrggbb').'''
    chip = QPixmap(12, 12)
    chip.fill(QColor(color))
    return chip


class _PartCard(QFrame):
    '''One planned-part card: thumbnail, name, color combo, counts.

    A click anywhere on the card body (except the combo, which swallows
    its own presses) starts placing that part: the window attaches a
    ghost copy of the color-selected block to the cursor. Switching the
    combo re-renders the thumbnail in the selected variant's color.
    '''

    # Emitted with the block uid of the color variant selected on the card.
    place_requested = Signal(str)
    # Emitted when the color combo selection changes (thumbnail recolor).
    variant_changed = Signal()

    def __init__(
        self,
        name: str,
        uids: list[str],
        colors: list[str],
        tooltip: str,
        parent: QWidget | None = None,
    ) -> None:
        '''Build the card widgets.

        Args:
            name: Recipe name of the part geometry.
            uids: Block uids of the color variants (combo order).
            colors: Planned colors of the group (first one preselected).
            tooltip: Geometry description shown on hover.
            parent: Optional parent widget.
        '''
        super().__init__(parent)
        self._uids = list(uids)
        self._colors = list(colors)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedSize(*PART_CARD_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._thumb = QLabel()
        self._thumb.setFixedSize(QSize(*THUMBNAIL_SIZE))
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._thumb)

        self._name = QLabel(name)
        self._name.setStyleSheet('font-size: 11pt;')
        self._name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name.setToolTip(tooltip)
        layout.addWidget(self._name)

        self._combo = QComboBox()
        self._combo.setToolTip("Planned color of the copy to place.")
        for color in colors:
            self._combo.addItem(_color_chip(color), color)
        layout.addWidget(self._combo)
        if len(colors) <= 1:
            self._combo.setEnabled(False)
        # Connected after population: the initial addItem(0) must not fire.
        self._combo.currentIndexChanged.connect(self._on_combo_index_changed)

        self._counts = QLabel("")
        self._counts.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._counts)
        layout.addStretch(1)

    def selected_uid(self) -> str:
        '''Block uid of the currently selected color variant.'''
        index = max(0, min(self._combo.currentIndex(), len(self._uids) - 1))
        return self._uids[index]

    def selected_color(self) -> str:
        '''Planned color of the currently selected variant.'''
        index = max(0, min(self._combo.currentIndex(), len(self._colors) - 1))
        return self._colors[index]

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        '''Show the rendered part thumbnail.'''
        self._thumb.setPixmap(pixmap)

    def set_counts(self, planned: int, placed: int, per_color: list[str]) -> None:
        '''Refresh the counts label and the per-color combo captions.

        Args:
            planned: Total planned quantity of the group.
            placed: Total placed copies of the group.
            per_color: One caption per combo item (empty keeps the plain
                color text).
        '''
        self._counts.setText(f"Planned: {planned} · Placed: {placed}")
        for index, caption in enumerate(per_color):
            if caption and index < self._combo.count():
                self._combo.setItemText(index, caption)

    def _on_combo_index_changed(self, _index: int) -> None:
        '''The color combo changed: the thumbnail must follow the color.'''
        self.variant_changed.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''Any press on the card body starts placing the part.'''
        self.place_requested.emit(self.selected_uid())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''Swallow the release so the click never leaks to other widgets.'''
        event.accept()


class PlannedPartsPanel(QWidget):
    '''Right-side panel of the Assemble tab: every planned part geometry
    with its planned colors and placed/planned counts.

    Cards rebuild only when the part structure changes (geometry, colors,
    block set); count changes update the labels in place. Thumbnails are
    built asynchronously on the background job queue and cached per
    (geometry, color).
    '''

    # Emitted with the block uid whose copy the user wants to place.
    place_requested = Signal(str)

    def __init__(self, jobs: JobQueue, parent: QWidget | None = None) -> None:
        '''Create the panel.

        Args:
            jobs: Background job queue for thumbnail builds.
            parent: Optional parent widget.
        '''
        super().__init__(parent)
        self._jobs = jobs
        self._refresh_seq = 0
        self._thumb_meta: dict[int, tuple[str, str, int]] = {}
        # (geometry key, color) -> rendered pixmap.
        self._thumb_cache: dict[tuple[str, str], QPixmap] = {}
        # (geometry key, color) builds currently running on the queue.
        self._pending_thumbs: set[tuple[str, str]] = set()
        self._jobs.job_done.connect(self._on_job_done)
        # Two columns of cards minimum (mirrors the template library).
        self.setMinimumWidth(2 * CARD_WIDTH + 28)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        header = QLabel("Planned parts")
        header.setStyleSheet("font-weight: bold;")
        header.setToolTip(
            "Parts planned on the Planning tab. Click a card to attach a "
            "copy to the cursor, click the scene to drop it (Esc exits)."
        )
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

        self._cards: list[_PartCard] = []
        self._card_keys: list[str] = []
        self._signature: tuple | None = None

    # ------------------------------------------------------------- public

    def refresh(self, groups: list[PartGroup], counts: dict[str, int]) -> None:
        '''Sync the cards with the planned part groups.

        Args:
            groups: Planned part groups (geometry-grouped blocks).
            counts: Placed copy count per block uid.
        '''
        signature = tuple(
            (
                group.key,
                group.name,
                tuple((variant.uid, variant.color) for variant in group.variants),
            )
            for group in groups
        )
        structure_changed = signature != self._signature
        self._signature = signature

        if structure_changed:
            self._rebuild_cards(groups)
            self._submit_thumbnails(groups)
        self._update_counts(groups, counts)
        if structure_changed:
            self._relayout_cards()

    def force_relayout(self) -> None:
        '''Re-wrap the card grid at the panel's current width.'''
        self._relayout_cards()

    # ------------------------------------------------------------ internals

    def _rebuild_cards(self, groups: list[PartGroup]) -> None:
        '''Recreate every card widget for the new part structure.'''
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards.clear()
        self._card_keys.clear()
        for group in groups:
            colors = [variant.color for variant in group.variants]
            uids = [variant.uid for variant in group.variants]
            planned = sum(variant.quantity for variant in group.variants)
            tooltip = f"{group.name} · {planned} planned"
            card = _PartCard(group.name, uids, colors, tooltip)
            card.place_requested.connect(self.place_requested)
            card.variant_changed.connect(
                lambda card=card: self._on_card_variant_changed(card)
            )
            self._cards.append(card)
            self._card_keys.append(group.key)

    def _update_counts(self, groups: list[PartGroup], counts: dict[str, int]) -> None:
        '''Refresh the placed/planned labels of every card.'''
        for card, group in zip(self._cards, groups):
            planned = sum(variant.quantity for variant in group.variants)
            placed = sum(counts.get(variant.uid, 0) for variant in group.variants)
            # Caption duplicates get a "placed/planned" suffix per variant.
            per_color: list[str] = []
            for variant in group.variants:
                if len(group.variants) > 1:
                    variant_placed = counts.get(variant.uid, 0)
                    per_color.append(
                        f"{variant.color} · {variant_placed}/{variant.quantity}"
                    )
                else:
                    per_color.append("")
            card.set_counts(planned, placed, per_color)

    def _submit_thumbnails(self, groups: list[PartGroup]) -> None:
        '''Queue thumbnail builds for every card (cache-first).

        Each card's thumbnail renders in its SELECTED variant's color, so
        the initial submission uses the combo selection (index 0 right
        after a rebuild).
        '''
        self._refresh_seq += 1
        seq = self._refresh_seq
        self._thumb_meta.clear()
        # Cards were rebuilt: running builds are stale (their results are
        # discarded by the seq guard) and their pending marks must not
        # block resubmission of the same (key, color).
        self._pending_thumbs.clear()
        for card, group in zip(self._cards, groups):
            if not group.variants:
                continue
            color = card.selected_color()
            self._show_or_submit_thumbnail(card, group.key, color, seq)

    def _show_or_submit_thumbnail(
        self, card: _PartCard, key: str, color: str, seq: int
    ) -> None:
        '''Show the cached thumbnail for (key, color) or queue its build.

        Args:
            card: The card asking for the thumbnail.
            key: Geometry key of the card's part group.
            color: Selected variant color to render the part in.
            seq: Current refresh sequence number for the job result guard.
        '''
        cache_key = (key, color)
        cached = self._thumb_cache.get(cache_key)
        if cached is not None:
            card.set_thumbnail(cached)
            return
        if cache_key in self._pending_thumbs:
            return  # a build for this (key, color) is already running
        payload = {
            # The group key IS the canonical geometry JSON (layout
            # fields only): the worker rebuilds the merged settings
            # from it plus the shared geometry defaults.
            'settings': json.loads(key),
            'tolerance': THUMBNAIL_PAYLOAD_TOLERANCE,
            'angular': THUMBNAIL_PAYLOAD_ANGULAR,
        }
        job_id = self._jobs.submit('thumbnail', payload)
        self._thumb_meta[job_id] = (key, color, seq)
        self._pending_thumbs.add(cache_key)

    def _on_card_variant_changed(self, card: _PartCard) -> None:
        '''A card's color combo changed: re-render its thumbnail color.'''
        try:
            index = self._cards.index(card)
        except ValueError:  # pragma: no cover - card always known here
            return
        key = self._card_keys[index]
        self._show_or_submit_thumbnail(
            card, key, card.selected_color(), self._refresh_seq
        )

    def _relayout_cards(self) -> None:
        '''Place the cards into the grid, wrapping by viewport width.'''
        viewport_w = self._scroll.viewport().width()
        cols = max(1, viewport_w // CARD_WIDTH)
        for index, card in enumerate(self._cards):
            self._cards_layout.addWidget(
                card,
                index // cols,
                index % cols,
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            )
        if not self._cards:
            # Empty state: rebuild the single placeholder cell (a resize
            # must not stack duplicates).
            while self._cards_layout.count():
                item = self._cards_layout.takeAt(0)
                if item is None:
                    break
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            empty = QLabel("(no blocks planned)")
            self._cards_layout.addWidget(empty, 0, 0)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Re-wrap the card grid for the new width (once settled).'''
        super().resizeEvent(event)
        QTimer.singleShot(0, self._relayout_cards)

    def _on_job_done(self, job_id: int, result: object) -> None:
        '''Fill in a completed thumbnail build.

        Args:
            job_id: The job identifier returned by :meth:`JobQueue.submit`.
            result: Handler result dict, :data:`SKIPPED`, or a
                :class:`JobError`.
        '''
        meta = self._thumb_meta.pop(job_id, None)
        if meta is None:
            return  # Foreign job (generation/export — other slots handle those).
        key, color, seq = meta
        self._pending_thumbs.discard((key, color))
        if seq != self._refresh_seq:
            return  # A newer refresh replaced the cards; discard stale result.
        if isinstance(result, JobError):
            logger.warning("Thumbnail build failed: %.24s: %s", key, result.exc)
            return
        if result is SKIPPED or not isinstance(result, dict):
            return
        try:
            mesh = mesh_from_arrays(result)
            pixmap = render_thumbnail(mesh, color)
        except Exception:  # noqa: BLE001 - VTK off-screen render can fail on GL
            logger.warning("Thumbnail VTK render failed for a planned part")
            return
        self._thumb_cache[(key, color)] = pixmap
        for card, card_key in zip(self._cards, self._card_keys):
            # Only the card whose selected variant requested this color.
            if card_key == key and card.selected_color() == color:
                card.set_thumbnail(pixmap)
