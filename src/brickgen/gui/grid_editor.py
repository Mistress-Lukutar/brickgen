'''
File:   grid_editor.py
Brief:  Top-view cell grid editor: normal / excluded / bottom-chamfered.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.6.0
'''

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

_CELL_PX = 26  # rendered cell size
_CELL_GAP = 2  # gap between cells

_FILL_INCLUDED = QColor("#3f7fbf")
_FILL_EXCLUDED = QColor("#e8e8e8")
_FILL_SLOPED = QColor("#e8a33d")
_EDGE_COLOR = QColor("#444444")
_ARROW_COLOR = QColor("#222222")

_SIDE_OFFSETS = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}


class CellGridEditor(QWidget):
    '''Schematic top view of the cell grid (X right, Y up).

    Clicking a cell cycles its state:
    normal (blue) -> excluded (gray) -> bottom-chamfered (orange, an arrow
    shows the outward side the wedge thins toward) -> normal. A chamfer is
    only offered on sides that look outward of the figure, and a cell is
    chamfered in exactly one direction; clicking a chamfered cell again
    rotates the chamfer to its next outward side, and after the last one
    the cell returns to normal.
    '''

    # Emitted after any state change.
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
        self._sloped: dict[tuple[int, int], list[str]] = {}
        self._update_size()

    # ------------------------------------------------------------------ API
    def set_dimensions(self, cols: int, rows: int) -> None:
        '''Resize the grid, dropping states outside the new bounds.

        Args:
            cols: Number of cells along X.
            rows: Number of cells along Y.
        '''
        self._cols = max(1, cols)
        self._rows = max(1, rows)
        keep = lambda c: 0 <= c[0] < self._cols and 0 <= c[1] < self._rows  # noqa: E731
        self._excluded = {c for c in self._excluded if keep(c)}
        self._sloped = {c: sides for c, sides in self._sloped.items() if keep(c)}
        self._update_size()
        self.update()

    def excluded_cells(self) -> list[list[int]]:
        '''Current exclusion mask as sorted [ix, iy] pairs.'''
        return sorted([ix, iy] for ix, iy in self._excluded)

    def sloped_cells(self) -> list[list]:
        '''Current chamfer list as sorted [ix, iy, side] triplets.'''
        triplets: list[list] = []
        for (ix, iy), sides in self._sloped.items():
            for side in sides:
                triplets.append([ix, iy, side])
        return sorted(triplets)

    def set_excluded(self, cells: list[list[int]]) -> None:
        '''Replace the exclusion mask.

        Args:
            cells: List of [ix, iy] pairs (out-of-range pairs are dropped).
        '''
        self._excluded = {
            (int(ix), int(iy))
            for ix, iy in cells
            if 0 <= int(ix) < self._cols and 0 <= int(iy) < self._rows
        }
        self.update()

    def set_sloped(self, cells: list[list]) -> None:
        '''Replace the chamfer list.

        A cell is chamfered in exactly one direction; when several triplets
        name the same cell, only the first one is kept.

        Args:
            cells: List of [ix, iy, side] triplets (invalid are dropped).
        '''
        sloped: dict[tuple[int, int], list[str]] = {}
        for item in cells:
            try:
                cell = (int(item[0]), int(item[1]))
                side = str(item[2])
            except (TypeError, ValueError, IndexError):
                continue
            if cell in self._excluded or side not in _SIDE_OFFSETS:
                continue
            if not (0 <= cell[0] < self._cols and 0 <= cell[1] < self._rows):
                continue
            if cell not in sloped:
                sloped[cell] = [side]
        self._sloped = sloped
        self.update()

    def include_all(self) -> None:
        '''Reset every cell back to normal.'''
        self._excluded.clear()
        self._sloped.clear()
        self.update()
        self.changed.emit()

    # ----------------------------------------------------------- click cycle
    def _available_sides(self, ix: int, iy: int) -> list[str]:
        '''Outward sides of a cell: neighbor excluded or outside the grid.

        Args:
            ix: Cell index along X.
            iy: Cell index along Y.

        Returns:
            List of side names in stable order.
        '''
        order = ["+X", "-X", "+Y", "-Y"]
        result: list[str] = []
        for side in order:
            dx, dy = _SIDE_OFFSETS[side]
            nx, ny = ix + dx, iy + dy
            inside = 0 <= nx < self._cols and 0 <= ny < self._rows
            if not inside or (nx, ny) in self._excluded:
                result.append(side)
        return result

    def _on_click(self, ix: int, iy: int) -> None:
        '''Cycle the cell state: normal -> excluded -> chamfered -> normal.'''
        cell = (ix, iy)
        if cell not in self._excluded and cell not in self._sloped:
            # to excluded, unless it is the last material cell
            if len(self._excluded) < self._cols * self._rows - 1:
                self._excluded.add(cell)
        elif cell in self._excluded:
            # to chamfered with the first available outward side
            available = self._available_sides(ix, iy)
            self._excluded.discard(cell)
            if available:
                self._sloped[cell] = [available[0]]
        else:
            # chamfered: rotate to the next available outward side; after
            # the last one, return to normal
            available = self._available_sides(ix, iy)
            current = self._sloped.get(cell, [])
            idx = (
                available.index(current[0])
                if current and current[0] in available
                else -1
            )
            if idx + 1 < len(available):
                self._sloped[cell] = [available[idx + 1]]
            else:
                del self._sloped[cell]
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
        '''Draw the grid cells with their states.'''
        painter = QPainter(self)
        try:
            painter.setPen(QPen(_EDGE_COLOR, 1))
            for iy in range(self._rows):
                for ix in range(self._cols):
                    cell = (ix, iy)
                    if cell in self._excluded:
                        painter.setBrush(_FILL_EXCLUDED)
                    elif cell in self._sloped:
                        painter.setBrush(_FILL_SLOPED)
                    else:
                        painter.setBrush(_FILL_INCLUDED)
                    rect = self._cell_rect(ix, iy)
                    painter.drawRect(rect)
                    for side in self._sloped.get(cell, []):
                        self._draw_arrow(painter, rect, side)
        finally:
            painter.end()

    def _draw_arrow(self, painter: QPainter, rect: QRectF, side: str) -> None:
        '''Draw an outward arrow marking the chamfer direction.'''
        dx, dy = _SIDE_OFFSETS[side]
        cx, cy = rect.center().x(), rect.center().y()
        half = _CELL_PX / 2.0 - 4.0
        tail = QPointF(cx - dx * half * 0.5, cy - dy * half * 0.5)
        tip = QPointF(cx + dx * half, cy + dy * half)
        painter.setPen(QPen(_ARROW_COLOR, 2))
        painter.setBrush(_ARROW_COLOR)
        painter.drawLine(tail, tip)
        perp = QPointF(-dy, dx)
        size = 3.5
        left = QPointF(
            tip.x() - dx * size + perp.x() * size, tip.y() - dy * size + perp.y() * size
        )
        right = QPointF(
            tip.x() - dx * size - perp.x() * size, tip.y() - dy * size - perp.y() * size
        )
        painter.drawPolygon(QPolygonF([tip, left, right]))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Cycle the state of the cell under the cursor.'''
        pos: QPointF = event.position()
        ix = int((pos.x() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        row = int((pos.y() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        iy = self._rows - 1 - row
        if 0 <= ix < self._cols and 0 <= iy < self._rows:
            self._on_click(ix, iy)
        super().mousePressEvent(event)
