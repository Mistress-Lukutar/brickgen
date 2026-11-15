'''
File:   app.py
Brief:  BrickGen PySide6 main window: parameters, VTK preview, template library.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.7.0
'''

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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
from brickgen.core.models import SLOPE_SIDES, BlockSettings, validate_settings
from brickgen.gui.grid_editor import CellGridEditor
from brickgen.gui.library import TemplateLibraryPanel
from brickgen.gui.preview import DISPLAY_MODES, MODE_SOLID, PreviewWidget
from brickgen.services.export_service import SUPPORTED_FORMATS, export_model
from brickgen.services.settings_io import load_settings, save_settings

logger = logging.getLogger(__name__)

_SETTINGS_FILETYPES = "BrickGen settings (*.json);;All files (*.*)"

# Debounce for automatic regeneration after any parameter change, ms.
AUTO_GENERATE_DELAY_MS = 250


def resolve_export_target(path: str, selected_filter: str) -> tuple[str, str]:
    '''Derive the export format and a matching file path.

    The selected filter defines the format; when the typed file name has a
    different suffix it is replaced so the saved file always matches the
    chosen format (``block.step`` + 3MF filter -> ``block.3mf``). With the
    "All files" filter the typed suffix is used, falling back to STEP.

    Args:
        path: File name as typed by the user.
        selected_filter: Filter string returned by the file dialog.

    Returns:
        Tuple of (corrected path, export format name).
    '''
    formats = list(SUPPORTED_FORMATS)  # STEP first, dict insertion order
    file_format = next(
        (fmt for fmt in formats if selected_filter.startswith(fmt)), None
    )
    if file_format is None:
        suffix = Path(path).suffix.lstrip(".").upper()
        file_format = suffix if suffix in SUPPORTED_FORMATS else "STEP"
    return str(Path(path).with_suffix("." + file_format.lower())), file_format


class MainWindow(QMainWindow):
    '''Parameters on the left, VTK preview in the middle, template library on
    the right. The model regenerates automatically after any change.'''

    def __init__(self) -> None:
        '''Build the layout and render the default 1x1 block.'''
        super().__init__()
        self._settings = BlockSettings()
        self._model = None
        self.setWindowTitle("BrickGen - construction block generator")
        self.resize(1400, 760)
        self.setMinimumSize(1100, 600)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        self._preview = PreviewWidget()
        splitter.addWidget(self._preview)
        self._library = TemplateLibraryPanel(self._collect_settings_or_default)
        self._library.apply_requested.connect(self._on_template_selected)
        splitter.addWidget(self._library)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

        # Auto-generation: debounced rebuild after any parameter change.
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(AUTO_GENERATE_DELAY_MS)
        self._regen_timer.timeout.connect(self._on_generate)

        self._sync_form_from_settings()
        self._connect_auto_generate()
        self._on_generate()

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self) -> QWidget:
        '''Create the parameters panel with actions and status label.'''
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_size_group())
        layout.addWidget(self._build_layout_group())
        layout.addWidget(self._build_slope_group())
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
        self._pins.toggled.connect(self._on_pins_toggled)
        form.addRow(self._pins)

        self._tubes = QCheckBox("Inner tubes (2x2+ centers)")
        form.addRow(self._tubes)
        tube_spin = QDoubleSpinBox()
        tube_spin.setRange(0.1, 20.0)
        tube_spin.setSingleStep(0.1)
        tube_spin.setDecimals(2)
        self._floats["tube_inner_diameter"] = tube_spin
        form.addRow("Tube bore diameter, mm", tube_spin)
        return group

    def _build_layout_group(self) -> QGroupBox:
        '''Create the top-view cell grid editor group.'''
        group = QGroupBox("Layout (top view: click = exclude / chamfer)")
        layout = QVBoxLayout(group)
        self._cell_grid = CellGridEditor()
        self._cell_grid.changed.connect(self._schedule_rebuild)
        include_all = QPushButton("Include all cells")
        include_all.clicked.connect(self._on_include_all)
        layout.addWidget(self._cell_grid, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(include_all)
        return group

    def _build_slope_group(self) -> QGroupBox:
        '''Create the slope group (only applies to stud-less blocks).'''
        self._slope_group = QGroupBox("Slope (stud-less blocks)")
        form = QFormLayout(self._slope_group)
        self._slope_side = QComboBox()
        self._slope_side.addItems(list(SLOPE_SIDES))
        form.addRow("Raised edge", self._slope_side)
        self._slope_rise = QSpinBox()
        self._slope_rise.setRange(0, 16)
        self._slope_rise.setSuffix(" x unit height")
        form.addRow("Edge rise", self._slope_rise)
        note = QLabel(
            "One top edge rises N x block_height, the top face slopes down\n"
            "to the opposite edge (full rectangle). Studs and tubes do not\n"
            "apply; cell exclusions are ignored in this mode."
        )
        note.setWordWrap(True)
        form.addRow(note)
        return self._slope_group

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
        export = QPushButton("Export model...")
        export.clicked.connect(self._on_export)
        save = QPushButton("Save settings...")
        save.clicked.connect(self._on_save_settings)
        load = QPushButton("Load settings...")
        load.clicked.connect(self._on_load_settings)
        layout.addWidget(export)
        layout.addWidget(save)
        layout.addWidget(load)
        return group

    # ------------------------------------------------------- auto generation
    def _connect_auto_generate(self) -> None:
        '''Wire every parameter widget to the debounced regeneration.'''
        self._cells_x.valueChanged.connect(self._on_cells_changed)
        self._cells_y.valueChanged.connect(self._on_cells_changed)
        for spin in self._floats.values():
            spin.valueChanged.connect(self._schedule_rebuild)
        self._tubes.toggled.connect(self._schedule_rebuild)
        self._slope_side.currentTextChanged.connect(self._schedule_rebuild)
        self._slope_rise.valueChanged.connect(self._schedule_rebuild)

    def _schedule_rebuild(self) -> None:
        '''(Re)start the debounced regeneration timer.'''
        self._regen_timer.start()

    def _on_cells_changed(self) -> None:
        '''Resize the grid editor when the cell counts change; regenerate.'''
        self._cell_grid.set_dimensions(self._cells_x.value(), self._cells_y.value())
        self._schedule_rebuild()

    def _on_include_all(self) -> None:
        '''Reset the exclusion mask from the "Include all" button.'''
        self._cell_grid.include_all()
        self._schedule_rebuild()

    def _on_pins_toggled(self, checked: bool) -> None:
        '''Enable the slope group only for stud-less blocks; regenerate.'''
        self._slope_group.setEnabled(not checked)
        self._schedule_rebuild()

    # ----------------------------------------------------------------- actions
    def _collect_settings(self) -> BlockSettings:
        '''Build BlockSettings from the form widgets.

        Returns:
            Validated settings.

        Raises:
            SettingsValidationError: If values fail geometry validation.
        '''
        values: dict[str, int | float | bool | str | list] = {
            "cells_x": self._cells_x.value(),
            "cells_y": self._cells_y.value(),
            "pins_enabled": self._pins.isChecked(),
            "tubes_enabled": self._tubes.isChecked(),
            "slope_side": self._slope_side.currentText(),
            "slope_rise_units": self._slope_rise.value(),
            "excluded_cells": self._cell_grid.excluded_cells(),
            "sloped_cells": self._cell_grid.sloped_cells(),
        }
        values.update({field: spin.value() for field, spin in self._floats.items()})
        return validate_settings(values)

    def _collect_settings_or_default(self) -> BlockSettings:
        '''Best-effort settings snapshot for the template library panel.

        Returns:
            Current form settings, or the last valid settings when the form
            is currently in an invalid state.
        '''
        try:
            return self._collect_settings()
        except BrickGenError:
            return self._settings

    def _sync_form_from_settings(self) -> None:
        '''Push settings values into the form widgets.'''
        self._cells_x.setValue(self._settings.cells_x)
        self._cells_y.setValue(self._settings.cells_y)
        self._pins.setChecked(self._settings.pins_enabled)
        self._tubes.setChecked(self._settings.tubes_enabled)
        self._slope_side.setCurrentText(self._settings.slope_side)
        self._slope_rise.setValue(self._settings.slope_rise_units)
        self._slope_group.setEnabled(not self._settings.pins_enabled)
        self._cell_grid.set_dimensions(self._settings.cells_x, self._settings.cells_y)
        self._cell_grid.set_excluded(self._settings.excluded_cells)
        self._cell_grid.set_sloped(self._settings.sloped_cells)
        for field, spin in self._floats.items():
            spin.setValue(getattr(self._settings, field))

    def _on_mode_changed(self, mode: str) -> None:
        '''Apply the selected display mode to the preview.'''
        self._preview.set_display_mode(mode)

    def _on_grid_toggled(self, enabled: bool) -> None:
        '''Show or hide the preview grid.'''
        self._preview.set_grid_enabled(enabled)

    def _on_template_selected(self, settings: BlockSettings) -> None:
        '''Apply a template picked in the library panel.'''
        self._settings = settings
        self._sync_form_from_settings()
        self._on_generate()
        self._set_status(f"Template applied: {settings.describe()}")

    def _on_generate(self) -> None:
        '''Rebuild the model from form values and refresh the preview.

        Invalid intermediate form states are reported in the status label
        only (no dialogs), so typing stays uninterrupted.
        '''
        try:
            self._settings = self._collect_settings()
        except BrickGenError as exc:
            self._set_status(f"Invalid settings: {exc}", error=True)
            return
        try:
            self._model = build_block(self._settings)
        except BrickGenError as exc:
            self._set_status(f"Generation failed: {exc}", error=True)
            return
        slope_note = (
            f" | slope {self._settings.slope_side} +{self._settings.slope_rise_units}u"
            if self._settings.slope_active
            else ""
        )
        figure_note = (
            f" | figure {self._settings.included_count} cells"
            if not self._settings.is_rectangular
            else ""
        )
        chamfer_note = (
            f" | {len(self._settings.sloped_cells)} chamfered"
            if self._settings.sloped_cells
            else ""
        )
        self._status_style(False)
        self._status.setText(
            f"Form {self._settings.describe()} | footprint "
            f"{self._settings.outer_length_x:.2f} x {self._settings.outer_length_y:.2f} mm"
            f" | height {self._settings.total_height:.2f} mm{slope_note}{figure_note}{chamfer_note}"
        )
        self._preview.show_model(self._model)
        self._preview.set_display_mode(self._mode.currentText())
        self._preview.set_grid_enabled(self._grid.isChecked())

    def _on_export(self) -> None:
        '''Ask for a destination file and export the current model.

        STEP is the default: it is first in the filter list and the initial
        file name. The selected filter defines the format and the file
        extension (see ``resolve_export_target``).
        '''
        if self._model is None:
            QMessageBox.information(self, "Nothing to export", "Generate a block first.")
            return
        formats = list(SUPPORTED_FORMATS)
        default_filter = f"{formats[0]} (*.{formats[0].lower()})"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export model",
            f"block_{self._settings.describe()}.step",
            ";;".join(f"{fmt} (*.{fmt.lower()})" for fmt in formats)
            + ";;All files (*.*)",
            default_filter,
        )
        if not path:
            return
        path, file_format = resolve_export_target(path, selected_filter)
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
