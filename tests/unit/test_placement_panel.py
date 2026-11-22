'''
File:   test_placement_panel.py
Brief:  Offscreen PlacementPanel tests: bed-range signals, tools state,
        tri-state flip checkbox and sizing.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.18.0
'''

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from brickgen.core.project import (
    PRINT_STAGING_BED,
    BlockInstance,
    PrintBedSettings,
    PrintPiece,
    PrintToolsSettings,
)
from brickgen.gui.placement_panel import PlacementPanel


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


@pytest.fixture()
def panel() -> Iterator[PlacementPanel]:
    '''A plain (unshown) PlacementPanel.'''
    _ensure_app()
    widget = PlacementPanel()
    try:
        yield widget
    finally:
        widget.close()


def _make_pieces(flips: list[bool]) -> tuple[list[PrintPiece], list[BlockInstance]]:
    '''Pieces (all copies of one 2x2 block) with the given flip states.'''
    pieces = [
        PrintPiece(block_uid="b1", bed=PRINT_STAGING_BED, flipped=flipped)
        for flipped in flips
    ]
    blocks = [BlockInstance(uid="b1", cells_x=2, cells_y=2)]
    return pieces, blocks


def test_bed_range_emitted_zero_based(panel: PlacementPanel) -> None:
    '''"Beds from 1 to 2" emits 0-based indices [0, 1].

    The spinboxes and viewport labels both display 1-based bed numbers; the
    signal carries 0-based indices for the placement service.
    '''
    panel.set_beds(PrintBedSettings(count=4))
    panel._spin_from.setValue(1)
    panel._spin_to.setValue(2)
    seen: list[tuple[list[int], bool, bool]] = []
    panel.auto_place_requested.connect(
        lambda beds, by_color, even: seen.append((beds, by_color, even))
    )
    panel._on_auto_place()
    assert seen == [([0, 1], True, True)]


def test_auto_place_emits_checkbox_states(panel: PlacementPanel) -> None:
    '''The modifier checkboxes ride along with the auto-place signal.

    Both default to True ("Distribute by color", "Distribute evenly");
    unchecking a box flips the corresponding flag in the emitted request.
    '''
    panel.set_beds(PrintBedSettings(count=4))
    # The from/to coupling resets `from` when it exceeds `to`: raise `to`
    # first (as a user would with the spinners).
    panel._spin_to.setValue(3)
    panel._spin_from.setValue(2)
    seen: list[tuple[list[int], bool, bool]] = []
    panel.auto_place_requested.connect(
        lambda beds, by_color, even: seen.append((beds, by_color, even))
    )
    panel._on_auto_place()
    assert seen == [([1, 2], True, True)]

    panel._cb_by_color.setChecked(False)
    panel._cb_even.setChecked(False)
    panel._on_auto_place()
    assert seen[-1] == ([1, 2], False, False)

    panel._cb_by_color.setChecked(True)
    panel._on_auto_place()
    assert seen[-1] == ([1, 2], True, False)


def test_panel_minimum_width_pinned(panel: PlacementPanel) -> None:
    '''The panel pins its minimum width so controls are never clipped.

    The horizontal scrollbar of the hosting scroll area is off, so without
    an explicit minimum the form would clip on the right.
    '''
    assert panel.minimumWidth() >= 200


# ------------------------------------------------------- flip checkbox
def test_flip_checkbox_tri_state(panel: PlacementPanel) -> None:
    '''The flip checkbox mirrors the whole selection, not just the primary:
    checked when every selected piece is flipped, unchecked when none is,
    partially checked when the selection mixes both.'''
    blocks = [BlockInstance(uid="b1", cells_x=2, cells_y=2)]

    pieces, _ = _make_pieces([False, False])
    panel.sync_inspector(pieces, {}, {}, blocks, {0, 1}, 0)
    assert panel._cb_flip.checkState() == Qt.CheckState.Unchecked

    pieces, _ = _make_pieces([True, True])
    panel.sync_inspector(pieces, {}, {}, blocks, {0, 1}, 0)
    assert panel._cb_flip.checkState() == Qt.CheckState.Checked

    pieces, _ = _make_pieces([True, False])
    panel.sync_inspector(pieces, {}, {}, blocks, {0, 1}, 0)
    assert panel._cb_flip.checkState() == Qt.CheckState.PartiallyChecked

    # A single selected piece simply follows its own state.
    flipped_only, _ = _make_pieces([True])
    panel.sync_inspector(flipped_only, {}, {}, blocks, {0}, 0)
    assert panel._cb_flip.checkState() == Qt.CheckState.Checked


def test_flip_checkbox_click_semantics(panel: PlacementPanel) -> None:
    '''A click applies to ALL selected pieces: checked or partially
    checked -> clear all, unchecked -> flip all; the partial state is
    never the result of a click.'''
    blocks = [BlockInstance(uid="b1", cells_x=2, cells_y=2)]
    pieces, _ = _make_pieces([True, False])
    panel.sync_inspector(pieces, {}, {}, blocks, {0, 1}, 0)
    assert panel._cb_flip.checkState() == Qt.CheckState.PartiallyChecked

    seen: list[bool] = []
    panel.piece_flip_changed.connect(seen.append)

    # Click on a partially-checked box clears the flip for all.
    panel._cb_flip.click()
    assert seen == [False]
    assert panel._cb_flip.checkState() == Qt.CheckState.Unchecked

    # Click on an unchecked box flips all.
    panel._cb_flip.click()
    assert seen == [False, True]
    assert panel._cb_flip.checkState() == Qt.CheckState.Checked

    # Click on a checked box clears all again.
    panel._cb_flip.click()
    assert seen == [False, True, False]
    assert panel._cb_flip.checkState() == Qt.CheckState.Unchecked


def test_flip_selected_button_removed(panel: PlacementPanel) -> None:
    '''The "Flip selected" button is gone: the tri-state checkbox covers it.'''
    assert not hasattr(panel, "_btn_flip_sel")
    assert not hasattr(PlacementPanel, "flip_selection_requested")


# ---------------------------------------------------------- tools state
def test_tools_changed_emitted_on_user_edit(panel: PlacementPanel) -> None:
    '''User edits of the bed range and the distribute checkboxes emit the
    full PrintToolsSettings state.'''
    panel.set_beds(PrintBedSettings(count=4))
    panel.set_tools(PrintToolsSettings(auto_from=1, auto_to=4))
    seen: list[PrintToolsSettings] = []
    panel.tools_changed.connect(seen.append)

    # Raise `to` first (as a user would with the spinners), then `from`.
    panel._spin_to.setValue(4)
    panel._spin_from.setValue(2)
    assert seen[-1] == PrintToolsSettings(auto_from=2, auto_to=4)

    panel._cb_even.setChecked(False)
    assert seen[-1] == PrintToolsSettings(
        auto_from=2, auto_to=4, distribute_evenly=False
    )
    panel._cb_by_color.setChecked(False)
    assert seen[-1] == PrintToolsSettings(
        auto_from=2, auto_to=4, distribute_by_color=False, distribute_evenly=False
    )


def test_set_tools_syncs_without_emitting(panel: PlacementPanel) -> None:
    '''set_tools mirrors the project state into the widgets without any
    tools_changed emission (programmatic sync is not a user edit).'''
    panel.set_beds(PrintBedSettings(count=4))
    seen: list[PrintToolsSettings] = []
    panel.tools_changed.connect(seen.append)

    panel.set_tools(
        PrintToolsSettings(
            auto_from=2,
            auto_to=3,
            distribute_by_color=False,
            distribute_evenly=False,
        )
    )
    assert seen == []
    assert panel._spin_from.value() == 2
    assert panel._spin_to.value() == 3
    assert panel._cb_by_color.isChecked() is False
    assert panel._cb_even.isChecked() is False


def test_set_tools_clamped_by_bed_count(panel: PlacementPanel) -> None:
    '''set_beds shrinks the from/to widgets below the new bed count.'''
    panel.set_beds(PrintBedSettings(count=4))
    panel.set_tools(PrintToolsSettings(auto_from=3, auto_to=4))
    panel.set_beds(PrintBedSettings(count=2))
    assert panel._spin_from.value() == 2
    assert panel._spin_to.value() == 2
