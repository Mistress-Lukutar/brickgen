'''
File:   test_placement_panel.py
Brief:  Offscreen PlacementPanel tests: bed-range signals and sizing.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.16.0
'''

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from brickgen.core.project import PrintBedSettings
from brickgen.gui.placement_panel import PlacementPanel
from brickgen.services.placement_service import DISTRIBUTE_BY_COLOR


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
    signals carry 0-based indices for the placement service.
    '''
    panel.set_beds(PrintBedSettings(count=4))
    panel._spin_from.setValue(1)
    panel._spin_to.setValue(2)
    seen: list[list[int]] = []
    panel.auto_place_requested.connect(seen.append)
    panel._on_auto_place()
    assert seen == [[0, 1]]


def test_distribute_range_emitted_zero_based(panel: PlacementPanel) -> None:
    '''Distribute emits (mode, 0-based bed indices) for the selected range.'''
    panel.set_beds(PrintBedSettings(count=4))
    # The from/to coupling resets `from` when it exceeds `to`: raise `to`
    # first (as a user would with the spinners).
    panel._spin_to.setValue(3)
    panel._spin_from.setValue(2)
    seen: list[tuple[str, list[int]]] = []
    panel.distribute_requested.connect(lambda mode, beds: seen.append((mode, beds)))
    panel._on_distribute(DISTRIBUTE_BY_COLOR)
    assert seen == [(DISTRIBUTE_BY_COLOR, [1, 2])]


def test_panel_minimum_width_pinned(panel: PlacementPanel) -> None:
    '''The panel pins its minimum width so controls are never clipped.

    The horizontal scrollbar of the hosting scroll area is off, so without
    an explicit minimum the form would clip on the right.
    '''
    assert panel.minimumWidth() >= 200
