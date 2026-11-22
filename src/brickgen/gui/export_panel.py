'''
File:   export_panel.py
Brief:  Right-sidebar export panel for the print-placement tab: precision,
        format, per-bed vs single-file output and the export button.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.19.0
'''

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from brickgen.core.project import EXPORT_FORMATS, ExportSettings


class ExportPanel(QWidget):
    '''Right-sidebar panel for the print-placement tab: the project's
    export configuration and the big export button.

    Every control edits the per-project :class:`ExportSettings` (persisted
    inside the project JSON); the "Export" button only emits
    ``export_requested`` — the main window owns the actual export flow.
    '''

    # User clicked the Export button.
    export_requested = Signal()
    # ExportSettings built from the current widget states.
    export_settings_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the panel with the export group inside a scroll area.'''
        super().__init__(parent)
        self._syncing = False
        self._build_ui()
        # Start from the factory defaults (the combo's first item is STEP,
        # but the ExportSettings default format is STL).
        self.set_settings(ExportSettings())

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        '''Build the complete widget tree.'''
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)

        container = QWidget()
        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(10)

        # --- Export configuration group ---
        self._export_group = QGroupBox("Export (project)")
        form = QFormLayout(self._export_group)
        form.setVerticalSpacing(8)

        self._spin_tolerance = QDoubleSpinBox()
        self._spin_tolerance.setRange(0.01, 1.0)
        self._spin_tolerance.setDecimals(3)
        self._spin_tolerance.setSingleStep(0.01)
        self._spin_tolerance.setSuffix(" mm")
        self._spin_tolerance.setToolTip(
            "Mesh tessellation precision for tessellated formats "
            "(STL/AMF/3MF); smaller is finer and slower."
        )
        self._spin_tolerance.valueChanged.connect(self._emit_settings)
        form.addRow("Precision (mesh), mm", self._spin_tolerance)

        self._combo_format = QComboBox()
        self._combo_format.addItems(list(EXPORT_FORMATS))
        self._combo_format.currentTextChanged.connect(self._emit_settings)
        form.addRow("File format", self._combo_format)

        output_label = QLabel("Output files:")
        form.addRow(output_label)

        self._radio_per_bed = QRadioButton("One file per bed")
        self._radio_per_bed.setToolTip(
            "Every bed with printable pieces is written to its own file "
            "(<name>_bedNN.<ext>)."
        )
        self._radio_per_bed.setChecked(True)
        self._radio_single = QRadioButton("Single file (all beds)")
        self._radio_single.setToolTip(
            "All beds' pieces are combined into one file at their world "
            "positions (as shown in the viewport)."
        )
        group = QButtonGroup(self)
        group.addButton(self._radio_per_bed)
        group.addButton(self._radio_single)
        self._radio_per_bed.toggled.connect(self._emit_settings)
        form.addRow(self._radio_per_bed)
        form.addRow(self._radio_single)

        inner.addWidget(self._export_group)

        # --- Big export button ---
        self._btn_export = QPushButton("Export...")
        self._btn_export.setMinimumHeight(44)
        self._btn_export.setStyleSheet("font-weight: bold; font-size: 13pt;")
        self._btn_export.setToolTip("Export the print layout (Ctrl+E).")
        self._btn_export.clicked.connect(self.export_requested.emit)
        inner.addWidget(self._btn_export)

        # --- Status label (export feedback mirrors the panel status) ---
        self._status_label = QLabel("Ready.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #0a6;")
        inner.addWidget(self._status_label)
        inner.addStretch(1)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # The horizontal scrollbar is off, so anything narrower than the
        # form's natural width would clip it: pin the panel to at least
        # that width plus a scrollbar (mirrors the other sidebar panels).
        form_width = (
            container.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
        )
        self.setMinimumWidth(form_width)
        scroll.setMinimumWidth(form_width)

    # ------------------------------------------------------------ handlers

    def _emit_settings(self) -> None:
        '''Any control changed: emit the updated ExportSettings
        (suppressed during programmatic sync).'''
        if self._syncing:
            return
        self.export_settings_changed.emit(self.collect_settings())

    def collect_settings(self) -> ExportSettings:
        '''Build the ExportSettings from the current widget states.

        Returns:
            The validated per-project export configuration.
        '''
        return ExportSettings(
            format=self._combo_format.currentText(),
            tolerance=self._spin_tolerance.value(),
            per_bed_files=self._radio_per_bed.isChecked(),
        )

    # ---------------------------------------------------------- public API

    def set_settings(self, settings: ExportSettings) -> None:
        '''Sync the widgets from the project's export settings.'''
        self._syncing = True
        try:
            self._combo_format.setCurrentText(settings.format)
            self._spin_tolerance.setValue(settings.tolerance)
            self._radio_per_bed.setChecked(settings.per_bed_files)
            self._radio_single.setChecked(not settings.per_bed_files)
        finally:
            self._syncing = False

    def set_status(self, text: str, error: bool = False) -> None:
        '''Set the status label with green (ok) or red (error) styling.'''
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            "color: #0a6;" if not error else "color: #b00;"
        )
