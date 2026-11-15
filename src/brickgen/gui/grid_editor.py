'''
File:   grid_editor.py
Brief:  Top-view cell grid editor: click cells to include/exclude them.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.5.0
'''

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

_CELL_PX = 26  # rendered cell size
_CELL_GAP = 2  # gap between cells

_FILL_INCLUDED = QColor("#3f7fbf")
_FILL_EXCLUDED = QColor("#e8e8e8")
_EDGE_COLOR = QColor("#444444")


class CellGridEditor(QWidget):
    '''Schematic top view of the cell grid (X right, Y up).

    Clicking a cell toggles it between included (filled) and excluded
    (grayed). Excluded cells are removed from the figure.
    '''

    # Emitted after any cell toggle.
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the editor with a single included cell.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent)
        self._cols = 1
        self._rows = 1
        self._excluded: set[tuple[int, int]] = set()
        self._update_size()

    # ------------------------------------------------------------------ API
    def set_dimensions(self, cols: int, rows: int) -> None:
        '''Resize the grid, dropping exclusions outside the new bounds.

        Args:
            cols: Number of cells along X.
            rows: Number of cells along Y.
        '''
        self._cols = max(1, cols)
        self._rows = max(1, rows)
        self._excluded = {
            (ix, iy) for ix, iy in self._excluded
            if 0 <= ix < self._cols and 0 <= iy < self._rows
        }
        self._update_size()
        self.update()

    def excluded_cells(self) -> list[list[int]]:
        '''Current exclusion mask as sorted [ix, iy] pairs.'''
        return sorted([ix, iy] for ix, iy in self._excluded)

    def set_excluded(self, cells: list[list[int]]) -> None:
        '''Replace the exclusion mask.

        Args:
            cells: List of [ix, iy] pairs (out-of-range pairs are dropped).
        '''
        self._excluded = {
            (int(ix), int(iy)) for ix, iy in cells
            if 0 <= int(ix) < self._cols and 0 <= int(iy) < self._rows
        }
        self.update()

    def include_all(self) -> None:
        '''Clear the exclusion mask.'''
        self._excluded.clear()
        self.update()
        self.changed.emit()

    # ------------------------------------------------------------- painting
    def _update_size(self) -> None:
        '''Adjust the widget size to the current grid dimensions.'''
        width = self._cols * _CELL_PX + (self._cols + 1) * _CELL_GAP
        height = self._rows * _CELL_PX + (self._rows + 1) * _CELL_GAP
        self.setFixedSize(width, height)

    def _cell_rect(self, ix: int, iy: int) -> QRectF:
        '''Pixel rect of cell (ix, iy); row 0 is drawn at the top (+Y up).'''
        row = self._rows - 1 - iy
        x = _CELL_GAP + ix * (_CELL_PX + _CELL_GAP)
        y = _CELL_GAP + row * (_CELL_PX + _CELL_GAP)
        return QRectF(x, y, _CELL_PX, _CELL_PX)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Draw the grid cells.'''
        painter = QPainter(self)
        try:
            pen = QPen(_EDGE_COLOR, 1)
            painter.setPen(pen)
            for iy in range(self._rows):
                for ix in range(self._cols):
                    painter.setBrush(
                        _FILL_EXCLUDED if (ix, iy) in self._excluded
                        else _FILL_INCLUDED
                    )
                    painter.drawRect(self._cell_rect(ix, iy))
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Toggle the cell under the cursor.'''
        pos: QPointF = event.position()
        ix = int((pos.x() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        row = int((pos.y() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        iy = self._rows - 1 - row
        if 0 <= ix < self._cols and 0 <= iy < self._rows:
            if len(self._excluded) < self._cols * self._rows - 1 or (ix, iy) in self._excluded:
                self._excluded.symmetric_difference_update({(ix, iy)})
                self.update()
                self.changed.emit()
        super().mousePressEvent(event)
