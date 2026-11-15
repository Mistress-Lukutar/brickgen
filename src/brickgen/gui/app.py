'''
File:   app.py
Brief:  BrickGen PySide6 main window: settings panel, VTK preview, export.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.2.0
'''

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from brickgen.core.builder import build_block
from brickgen.core.exceptions import BrickGenError
from brickgen.core.models import BlockSettings, validate_settings
from brickgen.gui.preview import DISPLAY_MODES, MODE_SOLID, PreviewWidget
from brickgen.services.export_service import SUPPORTED_FORMATS, export_model
from brickgen.services.settings_io import load_settings, save_settings

logger = logging.getLogger(__name__)

_SETTINGS_FILETYPES = "BrickGen settings (*.json);;All files (*.*)"


class MainWindow(QMainWindow):
    '''Main application window: parameters on the left, VTK preview on the right.'''

    def __init__(self) -> None:
        '''Build the layout and render the default 1x1 block.'''
        super().__init__()
        self._settings = BlockSettings()
        self._model = None
        self.setWindowTitle("BrickGen - LEGO-like block generator")
        self.resize(1200, 720)
        self.setMinimumSize(900, 560)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        self._preview = PreviewWidget()
        splitter.addWidget(self._preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._sync_form_from_settings()
        self._on_generate()

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self) -> QWidget:
        '''Create the parameters panel with actions and status label.'''
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_size_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_actions_group())
        layout.addStretch(1)
        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        return panel

    def _build_size_group(self) -> QGroupBox:
        '''Create the block-geometry form group.'''
        group = QGroupBox("Block parameters")
        form = QFormLayout(group)

        self._cells_x = QSpinBox()
        self._cells_x.setRange(1, 64)
        self._cells_y = QSpinBox()
        self._cells_y.setRange(1, 64)
        form.addRow("Cells X (length)", self._cells_x)
        form.addRow("Cells Y (width)", self._cells_y)

        self._floats: dict[str, QDoubleSpinBox] = {}
        for label, field, maximum in [
            ("Unit size, mm", "unit_size", 100.0),
            ("Wall thickness, mm", "wall_thickness", 20.0),
            ("Block height, mm", "block_height", 200.0),
            ("Pin diameter, mm", "pin_diameter", 50.0),
            ("Pin height, mm", "pin_height", 100.0),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(0.05, maximum)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            self._floats[field] = spin
            form.addRow(label, spin)

        self._pins = QCheckBox("Studs on top")
        form.addRow(self._pins)
        return group

    def _build_display_group(self) -> QGroupBox:
        '''Create the preview display-mode group.'''
        group = QGroupBox("Preview")
        layout = QFormLayout(group)
        self._mode = QComboBox()
        self._mode.addItems(DISPLAY_MODES)
        self._mode.setCurrentText(MODE_SOLID)
        self._mode.currentTextChanged.connect(self._on_mode_changed)
        self._grid = QCheckBox("Show grid")
        self._grid.toggled.connect(self._on_grid_toggled)
        layout.addRow("Display mode", self._mode)
        layout.addRow(self._grid)
        return group

    def _build_actions_group(self) -> QWidget:
        '''Create the action buttons group.'''
        group = QGroupBox("Actions")
        layout = QVBoxLayout(group)
        generate = QPushButton("Generate")
        generate.clicked.connect(self._on_generate)
        export = QPushButton("Export model...")
        export.clicked.connect(self._on_export)
        save = QPushButton("Save settings...")
        save.clicked.connect(self._on_save_settings)
        load = QPushButton("Load settings...")
        load.clicked.connect(self._on_load_settings)
        layout.addWidget(generate)
        layout.addWidget(export)
        layout.addWidget(save)
        layout.addWidget(load)
        return group

    # ----------------------------------------------------------------- actions
    def _collect_settings(self) -> BlockSettings:
        '''Build BlockSettings from the form widgets.

        Returns:
            Validated settings.

        Raises:
            SettingsValidationError: If values fail geometry validation.
        '''
        values: dict[str, int | float | bool] = {
            "cells_x": self._cells_x.value(),
            "cells_y": self._cells_y.value(),
            "pins_enabled": self._pins.isChecked(),
        }
        values.update({field: spin.value() for field, spin in self._floats.items()})
        return validate_settings(values)

    def _sync_form_from_settings(self) -> None:
        '''Push settings values into the form widgets.'''
        self._cells_x.setValue(self._settings.cells_x)
        self._cells_y.setValue(self._settings.cells_y)
        self._pins.setChecked(self._settings.pins_enabled)
        for field, spin in self._floats.items():
            spin.setValue(getattr(self._settings, field))

    def _on_mode_changed(self, mode: str) -> None:
        '''Apply the selected display mode to the preview.'''
        self._preview.set_display_mode(mode)

    def _on_grid_toggled(self, enabled: bool) -> None:
        '''Show or hide the preview grid.'''
        self._preview.set_grid_enabled(enabled)

    def _on_generate(self) -> None:
        '''Rebuild the model from form values and refresh the preview.'''
        try:
            self._settings = self._collect_settings()
        except BrickGenError as exc:
            self._set_status("Invalid settings.", error=True)
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        try:
            self._model = build_block(self._settings)
        except BrickGenError as exc:
            self._set_status("Generation failed.", error=True)
            QMessageBox.critical(self, "Generation failed", str(exc))
            return
        self._status_style(False)
        self._status.setText(
            f"Form {self._settings.describe()} | footprint "
            f"{self._settings.outer_length_x:.2f} x {self._settings.outer_length_y:.2f} mm"
            f" | height {self._settings.total_height:.2f} mm"
        )
        self._preview.show_model(self._model)
        self._preview.set_display_mode(self._mode.currentText())
        self._preview.set_grid_enabled(self._grid.isChecked())

    def _on_export(self) -> None:
        '''Ask for a destination file and export the current model.'''
        if self._model is None:
            QMessageBox.information(self, "Nothing to export", "Generate a block first.")
            return
        formats = sorted(SUPPORTED_FORMATS)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export model",
            f"block_{self._settings.describe()}.step",
            ";;".join(f"{fmt} (*.{fmt.lower()})" for fmt in formats)
            + ";;All files (*.*)",
        )
        if not path:
            return
        suffix = path.rsplit(".", 1)[-1].upper() if "." in path else "STEP"
        file_format = suffix if suffix in SUPPORTED_FORMATS else "STEP"
        try:
            written = export_model(self._model, Path(path), file_format)
        except BrickGenError as exc:
            self._set_status("Export failed.", error=True)
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._set_status(f"Exported: {written.name}")

    def _on_save_settings(self) -> None:
        '''Persist current form values into a JSON preset.'''
        path, _ = QFileDialog.getSaveFileName(
            self, "Save settings", "brickgen_settings.json", _SETTINGS_FILETYPES
        )
        if not path:
            return
        try:
            self._settings = self._collect_settings()
            written = save_settings(self._settings, Path(path))
        except BrickGenError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._set_status(f"Settings saved: {written.name}")

    def _on_load_settings(self) -> None:
        '''Load a JSON preset into the form and regenerate the model.'''
        path, _ = QFileDialog.getOpenFileName(
            self, "Load settings", "", _SETTINGS_FILETYPES
        )
        if not path:
            return
        try:
            self._settings = load_settings(Path(path))
        except BrickGenError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._sync_form_from_settings()
        self._on_generate()

    # ------------------------------------------------------------------ helpers
    def _status_style(self, error: bool) -> None:
        '''Color the status label green (ok) or red (error).'''
        self._status.setStyleSheet(
            "color: #0a6;" if not error else "color: #b00;"
        )

    def _set_status(self, text: str, error: bool = False) -> None:
        '''Set status text with the matching color.'''
        self._status_style(error)
        self._status.setText(text)


def main() -> None:
    '''Application entry point: configure logging and run the Qt GUI loop.'''
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
