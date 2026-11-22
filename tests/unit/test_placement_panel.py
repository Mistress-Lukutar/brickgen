'''
File:   test_placement_panel.py
Brief:  Offscreen PlacementPanel tests: bed-range signals and sizing.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.17.0
'''

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from brickgen.core.project import PrintBedSettings
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
