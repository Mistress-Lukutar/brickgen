'''
File:   test_export_panel.py
Brief:  Offscreen ExportPanel tests: settings sync, change signals, wiring.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from brickgen.core.project import ExportSettings
from brickgen.gui.export_panel import ExportPanel


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


@pytest.fixture()
def panel() -> Iterator[ExportPanel]:
    '''A plain (unshown) ExportPanel.'''
    _ensure_app()
    widget = ExportPanel()
    try:
        yield widget
    finally:
        widget.close()


def test_collect_default_settings(panel: ExportPanel) -> None:
    '''A fresh panel collects the factory export settings.'''
    settings = panel.collect_settings()
    assert settings == ExportSettings()


def test_set_settings_syncs_widgets(panel: ExportPanel) -> None:
    '''set_settings mirrors the project's export settings into the widgets.'''
    panel.set_settings(
        ExportSettings(format="3MF", tolerance=0.25, per_bed_files=False)
    )
    assert panel._combo_format.currentText() == "3MF"
    assert panel._spin_tolerance.value() == pytest.approx(0.25)
    assert panel._radio_per_bed.isChecked() is False
    assert panel._radio_single.isChecked() is True
    # Programmatic sync must not emit.
    seen: list[ExportSettings] = []
    panel.export_settings_changed.connect(seen.append)
    panel.set_settings(ExportSettings())
    assert seen == []


def test_widget_edits_emit_settings(panel: ExportPanel) -> None:
    '''User edits emit the combined ExportSettings.'''
    panel.set_settings(ExportSettings())
    seen: list[ExportSettings] = []
    panel.export_settings_changed.connect(seen.append)
    panel._combo_format.setCurrentText("BREP")
    panel._spin_tolerance.setValue(0.4)
    panel._radio_single.setChecked(True)
    assert len(seen) == 3
    assert seen[-1] == ExportSettings(format="BREP", tolerance=0.4, per_bed_files=False)


def test_export_button_emits_request(panel: ExportPanel) -> None:
    '''Clicking Export emits export_requested exactly once.'''
    seen: list[None] = []
    panel.export_requested.connect(lambda: seen.append(None))
    panel._btn_export.click()
    assert len(seen) == 1


def test_set_status_styles(panel: ExportPanel) -> None:
    '''set_status mirrors the shared status text with its styling.'''
    panel.set_status("Exported: x.stl")
    assert panel._status_label.text() == "Exported: x.stl"
    assert "color: #0a6;" in panel._status_label.styleSheet()
    panel.set_status("Export failed.", error=True)
    assert "color: #b00;" in panel._status_label.styleSheet()
