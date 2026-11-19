'''
File:   grid_editor.py
Brief:  Top-view cell grid editor: selection and per-cell feature states.
Author: Mistress-Lukutar
Date:   2026-11-19
Version: v0.11.0
'''

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

_CELL_PX = 26  # rendered cell size
_CELL_GAP = 2  # gap between cells
_BAND_MIN_DRAG = 4.0  # px of movement before a press becomes a band drag

_FILL_INCLUDED = QColor("#3f7fbf")
_FILL_EXCLUDED = QColor("#e8e8e8")
_FILL_CHAMFERED = QColor("#e8a33d")
_EDGE_COLOR = QColor("#444444")
_ARROW_COLOR = QColor("#222222")
_WEDGE_COLOR = QColor("#1c2f45")
_STUD_COLOR = QColor("#f5f7fa")
_STUD_RIM_COLOR = QColor("#20415f")
_SELECTION_COLOR = QColor("#d64545")
_SELECTION_SECONDARY_COLOR = QColor("#e88a8a")
_BAND_FILL = QColor(214, 69, 69, 40)

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
        round: Round base - the square is replaced by the inscribed circle.
            Conflicts with the slope and chamfer wedges, so enabling it
            clears both.
        chamfer: Single outward side of the bottom chamfer; the core model
            supports several sides per cell, the GUI edits exactly one.
    '''

    excluded: bool = False
    stud: bool | None = None
    slope: tuple[str, int] | None = None
    round: bool = False
    chamfer: str | None = None

    def is_default(self) -> bool:
        '''True when the cell needs no explicit record.'''
        return (
            not self.excluded
            and self.stud is None
            and self.slope is None
            and not self.round
            and self.chamfer is None
        )


class CellGridEditor(QWidget):
    '''Schematic top view of the cell grid (X right, Y up).

    Left-clicking a cell selects it; Ctrl+click toggles a cell in the
    selection, dragging with the left button frames a rectangular area
    (plain drag replaces the selection, Ctrl+drag extends it), clicking
    the background clears it and Escape clears it too. Selected cells
    are framed red around their silhouette - bright for the primary
    (last clicked, the inspector target), lighter for the rest. The
    features of the selected cells - exclusion, top stud, top slope,
    round base, bottom chamfer - are edited through :meth:`set_state` by
    the inspector panel in the main window, which applies changes to all
    selected cells. Cell colors: blue = included, gray = excluded,
    orange = bottom chamfer. Round cells render with inscribed-circle
    arcs at their free corners and straight edges toward material
    neighbors (matching the built solid). Icons: white dot = stud, dark
    wedge = slope (full edge marks the raised side), black arrow =
    chamfer direction.
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
        self._selected: set[tuple[int, int]] = set()
        self._primary: tuple[int, int] | None = None
        self._press_cell: tuple[int, int] | None = None
        self._band_origin: QPointF | None = None
        self._band_rect: QRectF | None = None
        self._band_active = False
        self._band_cells: set[tuple[int, int]] = set()
        self._default_studs = True
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        new_selection = {c for c in self._selected if keep(c)}
        selection_dropped = new_selection != self._selected or (
            self._primary is not None and not keep(self._primary)
        )
        self._selected = new_selection
        if self._primary is not None and not keep(self._primary):
            self._primary = None
        if selection_dropped:
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
        '''Primary (last clicked) selected (ix, iy) cell, or None.'''
        return self._primary

    def selected_cells(self) -> list[tuple[int, int]]:
        '''All selected cells in sorted (ix, iy) order.'''
        return sorted(self._selected)

    def clear_selection(self) -> None:
        '''Drop the current selection, if any.'''
        self._apply_selection(set(), None)

    def _apply_selection(
        self, cells: set[tuple[int, int]], primary: tuple[int, int] | None
    ) -> None:
        '''Replace the selection, repaint and notify on actual change.

        Args:
            cells: New selected-cell set.
            primary: New primary cell; must be None or a member of cells.
        '''
        if cells == self._selected and primary == self._primary:
            return
        self._selected = set(cells)
        self._primary = primary
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

    def _material_neighbor(self, ix: int, iy: int) -> bool:
        '''True when the neighbor cell exists and is not excluded.'''
        if not (0 <= ix < self._cols and 0 <= iy < self._rows):
            return False
        return not self._states.get((ix, iy), CellState()).excluded

    def _round_path(self, ix: int, iy: int, rect: QRectF) -> QPainterPath:
        '''Outline of a round cell: inscribed-circle arcs at free corners,
        straight edges toward material neighbors (stadium junctions).

        A corner is rounded only when both of its edges face no material
        neighbor - the same rule the solid builder applies.

        Args:
            ix: Cell index along X.
            iy: Cell index along Y.
            rect: Pixel rect of the cell.

        Returns:
            Closed painter path of the cell silhouette.
        '''
        connected = {
            "+X": self._material_neighbor(ix + 1, iy),
            "-X": self._material_neighbor(ix - 1, iy),
            "+Y": self._material_neighbor(ix, iy + 1),
            "-Y": self._material_neighbor(ix, iy - 1),
        }
        # Corner order walks the outline counter-clockwise: top-right,
        # top-left, bottom-left, bottom-right (Qt arc angles: 0 deg at 3
        # o'clock, positive counter-clockwise).
        spans = [
            (
                0.0,
                not (connected["+X"] or connected["+Y"]),
                QPointF(rect.right(), rect.top()),
            ),
            (
                90.0,
                not (connected["-X"] or connected["+Y"]),
                QPointF(rect.left(), rect.top()),
            ),
            (
                180.0,
                not (connected["-X"] or connected["-Y"]),
                QPointF(rect.left(), rect.bottom()),
            ),
            (
                270.0,
                not (connected["+X"] or connected["-Y"]),
                QPointF(rect.right(), rect.bottom()),
            ),
        ]

        def _at(angle: float) -> QPointF:
            rad = math.radians(angle)
            return QPointF(
                rect.center().x() + rect.width() / 2.0 * math.cos(rad),
                rect.center().y() - rect.height() / 2.0 * math.sin(rad),
            )

        path = QPainterPath()
        path.moveTo(_at(0.0))
        for start, rounded, corner_pt in spans:
            if rounded:
                path.arcTo(rect, start, 90.0)
            else:
                path.lineTo(corner_pt)
                path.lineTo(_at(start + 90.0))
        return path

    # ------------------------------------------------- model list exporting
    def excluded_cells(self) -> list[list[int]]:
        '''Current exclusion mask as sorted [ix, iy] pairs.'''
        return sorted([ix, iy] for (ix, iy), st in self._states.items() if st.excluded)

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

    def round_cells(self) -> list[list[int]]:
        '''Round-base cells as sorted [ix, iy] pairs.'''
        return sorted([ix, iy] for (ix, iy), st in self._states.items() if st.round)

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
        round_cells: list[list[int]] | None = None,
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
            round_cells: [ix, iy] pairs of round-base cells.
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
        for item in round_cells or []:
            cell = _cell_of(item)
            if cell is not None and cell not in states:
                states[cell] = CellState(round=True)
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
                    if state.round and not state.excluded:
                        # round base: free corners follow the inscribed circle
                        fill_rect = rect.adjusted(1.0, 1.0, -1.0, -1.0)
                        painter.drawPath(self._round_path(ix, iy, fill_rect))
                    else:
                        painter.drawRect(rect)
                    if state.excluded:
                        continue
                    if state.slope is not None:
                        self._draw_wedge(painter, rect, state.slope[0])
                    elif self.effective_stud(ix, iy):
                        self._draw_stud(painter, rect)
                    if state.chamfer is not None:
                        self._draw_arrow(painter, rect, state.chamfer)
            for ix, iy in self._band_cells:
                self._paint_frame(painter, ix, iy, _SELECTION_SECONDARY_COLOR)
            ordered = [c for c in sorted(self._selected) if c != self._primary]
            if self._primary is not None:
                ordered.append(self._primary)
            for ix, iy in ordered:
                color = (
                    _SELECTION_COLOR
                    if (ix, iy) == self._primary
                    else _SELECTION_SECONDARY_COLOR
                )
                self._paint_frame(painter, ix, iy, color)
            if self._band_active and self._band_rect is not None:
                painter.setPen(QPen(_SELECTION_COLOR, 1, Qt.PenStyle.DashLine))
                painter.setBrush(_BAND_FILL)
                painter.drawRect(self._band_rect)
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

    def _paint_frame(self, painter: QPainter, ix: int, iy: int, color: QColor) -> None:
        '''Draw a selection frame around cell (ix, iy).

        The 2 px frame sits in the 2 px gap around the cell, so adjacent
        selections touch without covering cell fills. Round cells get a
        frame matching their rounded silhouette.

        Args:
            painter: Active painter.
            ix: Cell index along X.
            iy: Cell index along Y.
            color: Frame color (bright for the primary selection).
        '''
        state = self._states.get((ix, iy), CellState())
        rect = self._cell_rect(ix, iy).adjusted(-1.0, -1.0, 1.0, 1.0)
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if state.round and not state.excluded:
            painter.drawPath(self._round_path(ix, iy, rect))
        else:
            painter.drawRect(rect)

    # ------------------------------------------------------------- input
    def _cell_at(self, pos: QPointF) -> tuple[int, int] | None:
        '''Cell (ix, iy) under a widget position, or None outside cells.

        Args:
            pos: Position in widget coordinates.

        Returns:
            The cell hit test result; gap pixels resolve to the preceding
            cell, margins and beyond resolve to None.
        '''
        ix = int((pos.x() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        row = int((pos.y() - _CELL_GAP) // (_CELL_PX + _CELL_GAP))
        iy = self._rows - 1 - row
        if 0 <= ix < self._cols and 0 <= iy < self._rows:
            return (ix, iy)
        return None

    def _cells_in_rect(self, rect: QRectF) -> set[tuple[int, int]]:
        '''Set of cells intersecting a widget-coordinate rectangle.

        Args:
            rect: Rubber band rectangle.

        Returns:
            All cells whose pixel rect intersects rect.
        '''
        return {
            (ix, iy)
            for iy in range(self._rows)
            for ix in range(self._cols)
            if self._cell_rect(ix, iy).intersects(rect)
        }

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Apply the selection click semantics and arm the rubber band.

        Plain click selects the cell under the cursor (or clears on the
        background). Ctrl+click defers its toggle to the release, which
        commits either the band cells (a drag) or the pressed-cell toggle
        (a click) - so a Ctrl+drag never toggles the starting cell.
        '''
        if event.button() == Qt.MouseButton.LeftButton:
            pos: QPointF = event.position()
            self._band_origin = pos
            self._band_rect = None
            self._band_active = False
            self._band_cells = set()
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            cell = self._cell_at(pos)
            if ctrl:
                self._press_cell = cell
            else:
                self._press_cell = None
                if cell is None:
                    self._apply_selection(set(), None)
                else:
                    self._apply_selection({cell}, cell)
            self.setFocus()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Track the rubber band rectangle and its live cell highlight.'''
        if self._band_origin is not None:
            pos: QPointF = event.position()
            rect = QRectF(self._band_origin, pos).normalized()
            self._band_active = (
                rect.width() > _BAND_MIN_DRAG or rect.height() > _BAND_MIN_DRAG
            )
            self._band_rect = rect
            self._band_cells = self._cells_in_rect(rect) if self._band_active else set()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Commit the rubber band area or the deferred Ctrl toggle.

        Plain drag replaces the selection with the band cells (primary
        falls to the last band cell); Ctrl+drag extends the selection,
        keeping the primary; a plain drag over empty background clears.
        Without a drag, a Ctrl+release toggles the pressed cell (the
        added cell becomes primary). A sub-threshold movement keeps the
        press behavior, so simple clicks stay simple.
        '''
        if (
            self._band_origin is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if self._band_active and self._band_rect is not None:
                band = self._cells_in_rect(self._band_rect)
                if ctrl:
                    cells = self._selected | band
                    primary = (
                        self._primary
                        if self._primary in cells
                        else (max(band) if band else None)
                    )
                else:
                    cells = band
                    primary = max(band) if band else None
                self._apply_selection(cells, primary)
            elif ctrl and self._press_cell is not None:
                cell = self._press_cell
                if cell in self._selected:
                    remaining = self._selected - {cell}
                    self._apply_selection(
                        remaining, max(remaining) if remaining else None
                    )
                else:
                    self._apply_selection(self._selected | {cell}, cell)
            self._press_cell = None
            self._band_origin = None
            self._band_rect = None
            self._band_active = False
            self._band_cells = set()
            self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Clear the selection on Escape.'''
        if event.key() == Qt.Key.Key_Escape:
            self.clear_selection()
            event.accept()
            return
        super().keyPressEvent(event)
