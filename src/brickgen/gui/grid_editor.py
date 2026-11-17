'''
File:   grid_editor.py
Brief:  Top-view cell grid editor: selection and per-cell feature states.
Author: Mistress-Lukutar
Date:   2026-11-17
Version: v0.10.0
'''

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

_CELL_PX = 26  # rendered cell size
_CELL_GAP = 2  # gap between cells

_FILL_INCLUDED = QColor("#3f7fbf")
_FILL_EXCLUDED = QColor("#e8e8e8")
_FILL_CHAMFERED = QColor("#e8a33d")
_EDGE_COLOR = QColor("#444444")
_ARROW_COLOR = QColor("#222222")
_WEDGE_COLOR = QColor("#1c2f45")
_STUD_COLOR = QColor("#f5f7fa")
_STUD_RIM_COLOR = QColor("#20415f")
_SELECTION_COLOR = QColor("#d64545")

_SIDE_OFFSETS = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}


@dataclass
class CellState:
    '''Per-cell feature state as edited in the layout.

    Attributes:
        excluded: Cell removed from the figure. The remaining fields are
            kept (hidden) so un-excluding restores the previous features;
            excluded cells are skipped by the export methods.
        stud: Explicit top stud; None follows the global default. A slope
            replaces the top entirely, so it clears this override.
        slope: ``(side, rise)`` of the top wedge; None when the top is flat
            or studded.
        chamfer: Single outward side of the bottom chamfer; the core model
            supports several sides per cell, the GUI edits exactly one.
    '''

    excluded: bool = False
    stud: bool | None = None
    slope: tuple[str, int] | None = None
    chamfer: str | None = None

    def is_default(self) -> bool:
        '''True when the cell needs no explicit record.'''
        return (
            not self.excluded
            and self.stud is None
            and self.slope is None
            and self.chamfer is None
        )


class CellGridEditor(QWidget):
    '''Schematic top view of the cell grid (X right, Y up).

    Left-clicking a cell selects it (red outline); the features of the
    selected cell - exclusion, top stud, top slope, bottom chamfer - are
    edited through :meth:`set_state` by the inspector panel in the main
    window. Cell colors: blue = included, gray = excluded, orange = bottom
    chamfer. Icons: white dot = stud, dark wedge = slope (full edge marks
    the raised side), black arrow = chamfer direction.
    '''

    # Emitted after any cell state change.
    changed = Signal()
    # Emitted when the selected cell changes (including cleared).
    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the editor with a single included cell.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent)
        self._cols = 1
        self._rows = 1
        self._states: dict[tuple[int, int], CellState] = {}
        self._selected: tuple[int, int] | None = None
        self._default_studs = True
        self._update_size()

    # ------------------------------------------------------------------ API
    def set_dimensions(self, cols: int, rows: int) -> None:
        '''Resize the grid, dropping states and selection outside the bounds.

        Args:
            cols: Number of cells along X.
            rows: Number of cells along Y.
        '''
        self._cols = max(1, cols)
        self._rows = max(1, rows)
        keep = lambda c: 0 <= c[0] < self._cols and 0 <= c[1] < self._rows  # noqa: E731
        self._states = {c: s for c, s in self._states.items() if keep(c)}
        if self._selected is not None and not keep(self._selected):
            self._selected = None
            self.selection_changed.emit()
        self._update_size()
        self.update()

    def set_default_studs(self, enabled: bool) -> None:
        '''Set the top applied to plain cells without an explicit override.

        Args:
            enabled: True when plain cells carry studs.
        '''
        self._default_studs = enabled
        self.update()

    def selected_cell(self) -> tuple[int, int] | None:
        '''Currently selected (ix, iy) cell, or None.'''
        return self._selected

    def clear_selection(self) -> None:
        '''Drop the current selection, if any.'''
        if self._selected is not None:
            self._selected = None
            self.update()
            self.selection_changed.emit()

    def state(self, ix: int, iy: int) -> CellState:
        '''Feature state of cell (ix, iy); default state when untouched.'''
        return self._states.get((ix, iy), CellState())

    def set_state(self, ix: int, iy: int, state: CellState) -> None:
        '''Store the feature state of cell (ix, iy).

        Args:
            ix: Cell index along X.
            iy: Cell index along Y.
            state: New feature state.
        '''
        if not (0 <= ix < self._cols and 0 <= iy < self._rows):
            return
        if state.is_default():
            self._states.pop((ix, iy), None)
        else:
            self._states[(ix, iy)] = state
        self.update()
        self.changed.emit()

    def include_all(self) -> None:
        '''Reset every cell back to plain.'''
        self._states.clear()
        self.update()
        self.changed.emit()

    def effective_stud(self, ix: int, iy: int) -> bool:
        '''True when the cell shows a stud (explicit override or default).

        Sloped cells never carry a stud.

        Args:
            ix: Cell index along X.
            iy: Cell index along Y.
        '''
        state = self.state(ix, iy)
        if state.excluded or state.slope is not None:
            return False
        if state.stud is not None:
            return state.stud
        return self._default_studs

    def available_chamfer_sides(self, ix: int, iy: int) -> list[str]:
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
            neighbor_excluded = self.state(nx, ny).excluded if inside else False
            if not inside or neighbor_excluded:
                result.append(side)
        return result

    # ------------------------------------------------- model list exporting
    def excluded_cells(self) -> list[list[int]]:
        '''Current exclusion mask as sorted [ix, iy] pairs.'''
        return sorted(
            [ix, iy] for (ix, iy), st in self._states.items() if st.excluded
        )

    def flat_cells(self) -> list[list[int]]:
        '''Explicit no-stud cells as sorted [ix, iy] pairs.'''
        return sorted(
            [ix, iy]
            for (ix, iy), st in self._states.items()
            if not st.excluded and st.slope is None and st.stud is False
        )

    def stud_cells(self) -> list[list[int]]:
        '''Explicit stud cells as sorted [ix, iy] pairs.'''
        return sorted(
            [ix, iy]
            for (ix, iy), st in self._states.items()
            if not st.excluded and st.slope is None and st.stud is True
        )

    def sloped_cells(self) -> list[list]:
        '''Top-slope list as sorted [ix, iy, side, rise] quadruplets.'''
        result: list[list] = []
        for (ix, iy), st in self._states.items():
            if st.excluded or st.slope is None:
                continue
            side, rise = st.slope
            result.append([ix, iy, side, rise])
        return sorted(result)

    def chamfered_cells(self) -> list[list]:
        '''Bottom-chamfer list as sorted [ix, iy, side] triplets.'''
        result: list[list] = []
        for (ix, iy), st in self._states.items():
            if st.excluded or st.chamfer is None:
                continue
            result.append([ix, iy, st.chamfer])
        return sorted(result)

    def set_from_lists(
        self,
        excluded: list[list[int]],
        flat: list[list[int]],
        stud: list[list[int]],
        sloped: list[list],
        chamfered: list[list],
    ) -> None:
        '''Replace all cell states from model-style lists.

        A cell that is both chamfered and sloped gets both features; the
        chamfer keeps its first side (the GUI edits a single direction).

        Args:
            excluded: [ix, iy] pairs.
            flat: [ix, iy] pairs of explicit no-stud cells.
            stud: [ix, iy] pairs of explicit stud cells.
            sloped: [ix, iy, side, rise] quadruplets.
            chamfered: [ix, iy, side] triplets.
        '''
        states: dict[tuple[int, int], CellState] = {}

        def _cell_of(item: list) -> tuple[int, int] | None:
            try:
                cell = (int(item[0]), int(item[1]))
            except (TypeError, ValueError, IndexError):
                return None
            if not (0 <= cell[0] < self._cols and 0 <= cell[1] < self._rows):
                return None
            return cell

        for item in excluded:
            cell = _cell_of(item)
            if cell is not None:
                states[cell] = CellState(excluded=True)
        for item in flat:
            cell = _cell_of(item)
            if cell is not None and cell not in states:
                states[cell] = CellState(stud=False)
        for item in stud:
            cell = _cell_of(item)
            if cell is not None and cell not in states:
                states[cell] = CellState(stud=True)
        for item in sloped:
            cell = _cell_of(item)
            if cell is None:
                continue
            try:
                slope = (str(item[2]), int(item[3]))
            except (TypeError, ValueError, IndexError):
                continue
            state = states.setdefault(cell, CellState())
            if not state.excluded:
                state.slope = slope
                state.stud = None
        for item in chamfered:
            cell = _cell_of(item)
            if cell is None:
                continue
            if not isinstance(item[2], str) or item[2] not in _SIDE_OFFSETS:
                continue
            state = states.setdefault(cell, CellState())
            if not state.excluded and state.chamfer is None:
                state.chamfer = item[2]
        self._states = {c: s for c, s in states.items() if not s.is_default()}
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
        '''Draw the grid cells with their feature states.

        Pen and brush are set explicitly for every cell: the icon and
        selection helpers below switch them, and leaked painter state
        would tint the borders of every cell drawn after the changed one.
        '''
        painter = QPainter(self)
        try:
            for iy in range(self._rows):
                for ix in range(self._cols):
                    state = self._states.get((ix, iy), CellState())
                    rect = self._cell_rect(ix, iy)
                    if state.excluded:
                        painter.setBrush(_FILL_EXCLUDED)
                    elif state.chamfer is not None:
                        painter.setBrush(_FILL_CHAMFERED)
                    else:
                        painter.setBrush(_FILL_INCLUDED)
                    painter.setPen(QPen(_EDGE_COLOR, 1))
                    painter.drawRect(rect)
                    if state.excluded:
                        continue
                    if state.slope is not None:
                        self._draw_wedge(painter, rect, state.slope[0])
                    elif self.effective_stud(ix, iy):
                        self._draw_stud(painter, rect)
                    if state.chamfer is not None:
                        self._draw_arrow(painter, rect, state.chamfer)
                    if self._selected == (ix, iy):
                        self._draw_selection(painter, rect)
        finally:
            painter.end()

    def _draw_stud(self, painter: QPainter, rect: QRectF) -> None:
        '''Draw a small circle marking a stud on the cell top.'''
        painter.setPen(QPen(_STUD_RIM_COLOR, 1))
        painter.setBrush(_STUD_COLOR)
        radius = _CELL_PX / 6.0
        center = rect.center()
        painter.drawEllipse(center, radius, radius)

    def _draw_wedge(self, painter: QPainter, rect: QRectF, side: str) -> None:
        '''Draw a filled triangle marking the slope: its full edge sits on
        the raised side, the apex points down the slope.'''
        inset = 4.0
        left, top = rect.left() + inset, rect.top() + inset
        right, bottom = rect.right() - inset, rect.bottom() - inset
        cx, cy = rect.center().x(), rect.center().y()
        if side == "+X":
            points = [QPointF(right, top), QPointF(right, bottom), QPointF(left, cy)]
        elif side == "-X":
            points = [QPointF(left, top), QPointF(left, bottom), QPointF(right, cy)]
        elif side == "+Y":
            points = [QPointF(left, top), QPointF(right, top), QPointF(cx, bottom)]
        else:  # "-Y"
            points = [QPointF(left, bottom), QPointF(right, bottom), QPointF(cx, top)]
        painter.setPen(QPen(_WEDGE_COLOR, 1))
        painter.setBrush(_WEDGE_COLOR)
        painter.drawPolygon(QPolygonF(points))

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

    def _draw_selection(self, painter: QPainter, rect: QRectF) -> None:
        '''Draw the red outline around the selected cell.'''
        painter.setPen(QPen(_SELECTION_COLOR, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(1.0, 1.0, -1.0, -1.0))

    # ------------------------------------------------------------- input
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Select the cell under the cursor.'''
        pos: QPointF = event.position()
        ix = int((pos.x() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        row = int((pos.y() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        iy = self._rows - 1 - row
        if 0 <= ix < self._cols and 0 <= iy < self._rows:
            self._selected = (ix, iy)
            self.update()
            self.selection_changed.emit()
        super().mousePressEvent(event)
