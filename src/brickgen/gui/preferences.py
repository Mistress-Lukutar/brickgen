'''
File:   preferences.py
Brief:  Modal preferences dialog: precisions and default export format.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

from pydantic import ValidationError
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from brickgen.services.app_settings import AppSettings
from brickgen.services.export_service import SUPPORTED_FORMATS


class PreferencesDialog(QDialog):
    '''Modal editor for application preferences (AppSettings).'''

    def __init__(self, settings: AppSettings, parent=None) -> None:
        '''Build the form from current preferences.

        Args:
            settings: Current application settings.
            parent: Optional parent widget.
        '''
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._preview_tol = QDoubleSpinBox()
        self._preview_tol.setRange(0.01, 1.0)
        self._preview_tol.setSingleStep(0.01)
        self._preview_tol.setDecimals(3)
        self._preview_tol.setValue(settings.preview_tolerance)
        form.addRow("Preview precision (mesh), mm", self._preview_tol)

        self._export_tol = QDoubleSpinBox()
        self._export_tol.setRange(0.01, 1.0)
        self._export_tol.setSingleStep(0.01)
        self._export_tol.setDecimals(3)
        self._export_tol.setValue(settings.export_tolerance)
        form.addRow("Export mesh tolerance, mm", self._export_tol)

        self._preview_ang = QDoubleSpinBox()
        self._preview_ang.setRange(0.01, 1.0)
        self._preview_ang.setSingleStep(0.01)
        self._preview_ang.setDecimals(3)
        self._preview_ang.setValue(settings.preview_angular_tolerance)
        form.addRow("Preview angular tolerance, rad", self._preview_ang)

        self._export_ang = QDoubleSpinBox()
        self._export_ang.setRange(0.01, 1.0)
        self._export_ang.setSingleStep(0.01)
        self._export_ang.setDecimals(3)
        self._export_ang.setValue(settings.export_angular_tolerance)
        form.addRow("Export angular tolerance, rad", self._export_ang)

        self._format = QComboBox()
        self._format.addItems(list(SUPPORTED_FORMATS))
        self._format.setCurrentText(settings.default_format)
        form.addRow("Default export format", self._format)

        self._undo_depth = QSpinBox()
        self._undo_depth.setRange(1, 500)
        self._undo_depth.setSingleStep(10)
        self._undo_depth.setValue(settings.undo_depth)
        self._undo_depth.setToolTip(
            "How many edit steps the shared Ctrl+Z / Ctrl+Y history keeps."
        )
        form.addRow("Undo depth (steps)", self._undo_depth)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> AppSettings | None:
        '''Collect the form into AppSettings.

        Returns:
            Validated settings, or None (with a warning box) when invalid.
        '''
        values = {
            "preview_tolerance": self._preview_tol.value(),
            "preview_angular_tolerance": self._preview_ang.value(),
            "export_tolerance": self._export_tol.value(),
            "export_angular_tolerance": self._export_ang.value(),
            "default_format": self._format.currentText(),
            "undo_depth": self._undo_depth.value(),
        }
        try:
            return AppSettings.model_validate(values)
        except ValidationError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return None
