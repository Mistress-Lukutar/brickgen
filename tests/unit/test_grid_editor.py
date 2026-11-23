'''
File:   test_grid_editor.py
Brief:  Unit tests for the layout CellGridEditor selection behavior.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest  # noqa: E402 - platform first
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402 - platform first
from PySide6.QtGui import QMouseEvent  # noqa: E402 - platform first
from PySide6.QtTest import QTest  # noqa: E402 - platform first
from PySide6.QtWidgets import QApplication  # noqa: E402 - platform first

from brickgen.gui.grid_editor import (  # noqa: E402 - platform first
    _CELL_GAP,
    _CELL_PX,
    CellGridEditor,
)

# The drag helper sends a hand-built QMouseEvent; its classic ctor is
# deprecated in PySide6 in favor of the QEventPoint-based one, which is
# not worth constructing for a synthetic move.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Function.*QMouseEvent.*is marked as deprecated:DeprecationWarning"
)


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


@pytest.fixture()
def editor() -> CellGridEditor:
    '''A 4x3 grid editor with cleared selection.'''
    _ensure_app()
    grid = CellGridEditor()
    grid.set_dimensions(4, 3)
    return grid


def _cell_center(grid: CellGridEditor, ix: int, iy: int) -> QPointF:
    '''Widget position of the center of cell (ix, iy).'''
    x = _CELL_GAP + ix * (_CELL_PX + _CELL_GAP) + _CELL_PX / 2.0
    row = grid._rows - 1 - iy  # noqa: SLF001 - tests inspect privates
    y = _CELL_GAP + row * (_CELL_PX + _CELL_GAP) + _CELL_PX / 2.0
    return QPointF(x, y)


def _click(grid: CellGridEditor, ix: int, iy: int, ctrl: bool = False) -> None:
    '''Mouse-click cell (ix, iy), optionally with Ctrl held.'''
    modifier = (
        Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    )
    QTest.mouseClick(
        grid,
        Qt.MouseButton.LeftButton,
        modifier,
        _cell_center(grid, ix, iy).toPoint(),
    )


def _drag(
    grid: CellGridEditor,
    start: QPointF,
    end: QPointF,
    ctrl: bool = False,
) -> None:
    '''Press at start, move the mouse to end, release there.

    The intermediate moves are delivered as raw QMouseEvents because
    QTest cannot simulate a drag with the button held.
    '''
    modifier = (
        Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    )
    QTest.mousePress(grid, Qt.MouseButton.LeftButton, modifier, start.toPoint())
    steps = 5
    for i in range(1, steps + 1):
        t = i / steps
        pt = QPointF(
            start.x() + (end.x() - start.x()) * t,
            start.y() + (end.y() - start.y()) * t,
        )
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            pt,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            modifier,
        )
        QApplication.sendEvent(grid, event)
    QTest.mouseRelease(grid, Qt.MouseButton.LeftButton, modifier, end.toPoint())


def test_click_selects_single_cell(editor: CellGridEditor) -> None:
    '''A plain click selects exactly the clicked cell as primary.'''
    _click(editor, 1, 1)
    assert editor.selected_cells() == [(1, 1)]
    assert editor.selected_cell() == (1, 1)


def test_click_replaces_previous_selection(editor: CellGridEditor) -> None:
    '''A plain click replaces the whole previous selection.'''
    _click(editor, 0, 0)
    _click(editor, 2, 1)
    assert editor.selected_cells() == [(2, 1)]


def test_ctrl_click_adds_cell_as_primary(editor: CellGridEditor) -> None:
    '''Ctrl+click extends the selection; the added cell becomes primary.'''
    _click(editor, 0, 0)
    _click(editor, 2, 1, ctrl=True)
    assert editor.selected_cells() == [(0, 0), (2, 1)]
    assert editor.selected_cell() == (2, 1)


def test_ctrl_click_toggles_selected_out(editor: CellGridEditor) -> None:
    '''Ctrl+click on a selected cell removes it; primary falls back.

    Removing the last cell empties the selection entirely.
    '''
    _click(editor, 1, 1)
    _click(editor, 2, 1, ctrl=True)
    assert editor.selected_cells() == [(1, 1), (2, 1)]
    assert editor.selected_cell() == (2, 1)
    _click(editor, 2, 1, ctrl=True)
    assert editor.selected_cells() == [(1, 1)]
    assert editor.selected_cell() == (1, 1)
    _click(editor, 1, 1, ctrl=True)
    assert editor.selected_cells() == []
    assert editor.selected_cell() is None


def test_background_click_clears_selection(editor: CellGridEditor) -> None:
    '''Clicking the outer margin clears the selection.'''
    _click(editor, 1, 1)
    QTest.mouseClick(
        editor,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(1, 1),
    )
    assert editor.selected_cells() == []
    assert editor.selected_cell() is None


def test_escape_clears_selection(editor: CellGridEditor) -> None:
    '''Escape drops the whole selection.'''
    _click(editor, 1, 1)
    _click(editor, 2, 0, ctrl=True)
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    assert editor.selected_cells() == []
    assert editor.selected_cell() is None


def test_set_dimensions_drops_out_of_bounds(
    editor: CellGridEditor,
) -> None:
    '''Shrinking the grid drops selection cells outside the bounds.'''
    _click(editor, 3, 2)
    changes: list[int] = []
    editor.selection_changed.connect(lambda: changes.append(1))
    editor.set_dimensions(2, 2)
    assert editor.selected_cells() == []
    assert editor.selected_cell() is None
    assert len(changes) == 1


def test_band_drag_replaces_selection(editor: CellGridEditor) -> None:
    '''A plain drag selects every cell under the rectangle.'''
    _click(editor, 3, 0)  # replaced by the band
    _drag(editor, _cell_center(editor, 0, 2), _cell_center(editor, 1, 1))
    assert editor.selected_cells() == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert editor.selected_cell() == (1, 2)


def test_band_drag_with_ctrl_extends_selection(
    editor: CellGridEditor,
) -> None:
    '''Ctrl+drag adds the band cells to the existing selection.'''
    _click(editor, 3, 0)
    _drag(
        editor,
        _cell_center(editor, 0, 2),
        _cell_center(editor, 1, 1),
        ctrl=True,
    )
    assert editor.selected_cells() == [(0, 1), (0, 2), (1, 1), (1, 2), (3, 0)]
    assert editor.selected_cell() == (3, 0)


def test_click_after_band_drag_starts_new_selection(
    editor: CellGridEditor,
) -> None:
    '''After a band drag a plain click goes back to single selection.'''
    _drag(editor, _cell_center(editor, 0, 2), _cell_center(editor, 1, 1))
    _click(editor, 3, 0)
    assert editor.selected_cells() == [(3, 0)]


def test_selection_changed_emitted_on_change(
    editor: CellGridEditor,
) -> None:
    '''selection_changed fires once per actual selection change.'''
    changes: list[int] = []
    editor.selection_changed.connect(lambda: changes.append(1))
    _click(editor, 1, 1)
    _click(editor, 1, 1)  # same cell again: no change, no signal
    _click(editor, 2, 1, ctrl=True)
    editor.clear_selection()
    editor.clear_selection()  # already empty: no signal
    assert len(changes) == 3
