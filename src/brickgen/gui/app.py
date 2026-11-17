'''
File:   app.py
Brief:  BrickGen PySide6 main window: parameters, VTK preview, template library.
Author: Mistress-Lukutar
Date:   2026-11-17
Version: v0.10.0
'''

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from brickgen.core.exceptions import BrickGenError
from brickgen.core.models import (
    MAX_SLOPE_RISE,
    SLOPE_SIDES,
    BlockSettings,
    validate_settings,
)
from brickgen.gui.grid_editor import CellGridEditor, CellState
from brickgen.gui.jobs import SKIPPED, JobError, JobQueue
from brickgen.gui.library import TemplateLibraryPanel
from brickgen.gui.preferences import PreferencesDialog
from brickgen.gui.preview import (
    DISPLAY_MODES,
    MODE_SOLID,
    PreviewWidget,
    mesh_from_arrays,
)
from brickgen.services.app_settings import (
    load_app_settings,
    save_app_settings,
)
from brickgen.services.export_service import SUPPORTED_FORMATS
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
    the right. The model regenerates automatically after any change.
    '''

    def __init__(self) -> None:
        '''Build the layout and render the default 1x1 block.'''
        super().__init__()
        self._settings = BlockSettings()
        self._applied_cache_key: str | None = None
        self._app_settings = load_app_settings()
        self.setWindowTitle("BrickGen - construction block generator")
        self.resize(1400, 760)
        self.setMinimumSize(1100, 600)

        # Background job queue for heavy meshing off the UI thread. The model
        # itself lives in the worker process, keyed by generation.
        self._jobs = JobQueue(self)
        self._gen_seq = 0
        self._job_meta: dict[int, tuple[int, str, BlockSettings]] = {}
        self._export_meta: dict[int, str] = {}  # job id -> resolved export path
        self._jobs.job_done.connect(self._on_job_done)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._sidebar = self._build_sidebar()
        splitter.addWidget(self._sidebar)
        splitter.addWidget(self._build_preview_pane())
        self._library = TemplateLibraryPanel(
            self._collect_settings_or_default, self._jobs
        )
        self._library.apply_requested.connect(self._on_template_selected)
        splitter.addWidget(self._library)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self._splitter = splitter
        self._layout_finalized = False
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
        '''Create the parameters panel with actions and status label.

        The content is wrapped in a scroll area so nothing is clipped when
        the window is too short for the whole form.
        '''
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.addWidget(self._build_size_group())
        layout.addWidget(self._build_layout_group())
        layout.addWidget(self._build_cell_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_actions_group())
        layout.addStretch(1)
        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The horizontal scrollbar is off, so anything narrower than the
        # form's natural width would clip it: pin the splitter child (the
        # scroll area itself) to at least that width plus a scrollbar.
        form_width = (
            panel.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
        )
        panel.setMinimumWidth(form_width)
        scroll.setMinimumWidth(form_width)
        return scroll

    def _build_size_group(self) -> QGroupBox:
        '''Create the block-geometry form group.'''
        group = QGroupBox("Block parameters")
        form = QFormLayout(group)
        form.setVerticalSpacing(8)

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

        self._pins = QCheckBox("Studs on plain cells")
        self._pins.setToolTip(
            "Default top for cells without an explicit per-cell assignment.\n"
            "Individual cells are edited in the layout below (click a cell)."
        )
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
        group = QGroupBox("Layout (top view: click selects a cell)")
        layout = QVBoxLayout(group)
        self._cell_grid = CellGridEditor()
        self._cell_grid.changed.connect(self._schedule_rebuild)
        self._cell_grid.selection_changed.connect(self._on_cell_selection_changed)
        include_all = QPushButton("Include all cells")
        include_all.clicked.connect(self._on_include_all)
        layout.addWidget(self._cell_grid, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(include_all)
        return group

    def _build_preview_pane(self) -> QWidget:
        '''Build the preview area with the preferences button bottom-left.

        Returns:
            Container widget holding the preview and the button row.
        '''
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        self._preview = PreviewWidget()
        layout.addWidget(self._preview, stretch=1)
        bottom = QHBoxLayout()
        prefs = QPushButton("Settings")
        prefs.setFixedWidth(110)
        prefs.clicked.connect(self._on_preferences)
        bottom.addWidget(prefs)
        bottom.addStretch(1)
        layout.addLayout(bottom)
        return pane

    def _on_preferences(self) -> None:
        '''Open the preferences modal and apply/save the results.'''
        dialog = PreferencesDialog(self._app_settings, self)
        if not dialog.exec():
            return
        updated = dialog.get_settings()
        if updated is None:
            return
        self._app_settings = updated
        try:
            save_app_settings(updated)
        except OSError as exc:
            QMessageBox.critical(self, "Preferences", f"Cannot save: {exc}")
        self._on_generate()  # re-render the preview with the new precision
        self._set_status("Preferences saved.")

    def _build_cell_group(self) -> QGroupBox:
        '''Create the inspector of the cell selected in the layout.

        A checklist of the actions available for the selected cell; checking
        an item reveals its parameter pairs indented below it. Slope and
        stud are mutually exclusive (a wedge top cannot carry a stud), the
        round base conflicts with the slope and chamfer wedges, exclusion
        hides everything else.
        '''
        self._cell_group = QGroupBox("Cell (click the layout to select)")
        outer = QVBoxLayout(self._cell_group)
        outer.setSpacing(4)

        self._cb_exclude = QCheckBox("Exclude cell")
        self._cb_exclude.setToolTip("Remove the cell from the figure entirely.")
        self._cb_stud = QCheckBox("Stud on top")
        self._cb_stud.setToolTip(
            "Cylindrical stud on the cell top. Unchecking makes the cell\n"
            "plain even when 'Studs on plain cells' is on."
        )
        self._cb_slope = QCheckBox("Slope")
        self._cb_slope.setToolTip(
            "Top wedge: the raised edge rises N x unit heights, the cell\n"
            "keeps the usual bottom opening. Geometrically impossible\n"
            "together with a stud or a round base."
        )
        self._cb_round = QCheckBox("Round base")
        self._cb_round.setToolTip(
            "Replace the square base with the inscribed circle; the bottom\n"
            "hole becomes round as well. Conflicts with slope and chamfer."
        )
        self._cb_chamfer = QCheckBox("Chamfer (bottom)")
        self._cb_chamfer.setToolTip(
            "Solid bottom wedge thinned toward the chosen side; only the\n"
            "outward sides of the cell are offered. Combines with any top."
        )

        # Parameter pairs, shown indented under their checked item.
        self._slope_params = QWidget()
        form = QFormLayout(self._slope_params)
        form.setContentsMargins(24, 0, 0, 0)
        form.setVerticalSpacing(4)
        self._slope_edge = QComboBox()
        self._slope_edge.addItems(list(SLOPE_SIDES))
        self._slope_rise = QSpinBox()
        self._slope_rise.setRange(1, MAX_SLOPE_RISE)
        self._slope_rise.setSuffix(" x unit height")
        form.addRow("Raised edge", self._slope_edge)
        form.addRow("Height", self._slope_rise)

        self._chamfer_params = QWidget()
        chamfer_form = QFormLayout(self._chamfer_params)
        chamfer_form.setContentsMargins(24, 0, 0, 0)
        chamfer_form.setVerticalSpacing(4)
        self._chamfer_edge = QComboBox()
        chamfer_form.addRow("Direction", self._chamfer_edge)

        for widget in (
            self._cb_exclude,
            self._cb_stud,
            self._cb_slope,
            self._slope_params,
            self._cb_round,
            self._cb_chamfer,
            self._chamfer_params,
        ):
            outer.addWidget(widget)

        self._cb_exclude.toggled.connect(self._on_inspector_exclude)
        self._cb_stud.toggled.connect(self._on_inspector_stud)
        self._cb_slope.toggled.connect(self._on_inspector_slope)
        self._cb_round.toggled.connect(self._on_inspector_round)
        self._cb_chamfer.toggled.connect(self._on_inspector_chamfer)
        self._slope_edge.currentTextChanged.connect(self._on_slope_params_changed)
        self._slope_rise.valueChanged.connect(self._on_slope_params_changed)
        self._chamfer_edge.currentTextChanged.connect(self._on_chamfer_param_changed)
        return self._cell_group

    def _build_display_group(self) -> QGroupBox:
        '''Create the preview display-mode group.'''
        group = QGroupBox("Preview")
        layout = QFormLayout(group)
        layout.setVerticalSpacing(8)
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

    def showEvent(self, event: QShowEvent) -> None:
        '''Pin the splitter to deterministic sizes on the first show.

        The automatic initial distribution can leave the library list laid
        out for a stale width (offset card grid, clipped second column);
        after the final window geometry is in, set the panel widths
        explicitly and re-flow the library grid at that width.
        '''
        super().showEvent(event)
        if self._layout_finalized:
            return
        self._layout_finalized = True
        QTimer.singleShot(0, self._finalize_layout)

    def _finalize_layout(self) -> None:
        '''Apply the final splitter sizes and re-layout the library cards.'''
        sidebar_w = self._sidebar.minimumWidth()
        library_w = self._library.minimumWidth()
        preview_w = max(self._splitter.width() - sidebar_w - library_w, 200)
        self._splitter.setSizes([sidebar_w, preview_w, library_w])
        # Deferred: run once more after the whole geometry pass has settled.
        QTimer.singleShot(0, self._library.force_relayout)
        self._library.force_relayout()

    # ------------------------------------------------------- auto generation
    def _connect_auto_generate(self) -> None:
        '''Wire every parameter widget to the debounced regeneration.'''
        self._cells_x.valueChanged.connect(self._on_cells_changed)
        self._cells_y.valueChanged.connect(self._on_cells_changed)
        for spin in self._floats.values():
            spin.valueChanged.connect(self._schedule_rebuild)
        self._tubes.toggled.connect(self._schedule_rebuild)

    def _schedule_rebuild(self) -> None:
        '''(Re)start the debounced regeneration timer.'''
        self._regen_timer.start()

    def _on_cells_changed(self) -> None:
        '''Resize the grid editor when the cell counts change; regenerate.'''
        self._cell_grid.set_dimensions(self._cells_x.value(), self._cells_y.value())
        self._schedule_rebuild()

    def _on_include_all(self) -> None:
        '''Reset every cell feature from the "Include all" button.'''
        self._cell_grid.include_all()
        self._sync_inspector()
        self._schedule_rebuild()

    def _on_pins_toggled(self, checked: bool) -> None:
        '''Apply the new plain-cell default and regenerate.'''
        self._cell_grid.set_default_studs(checked)
        self._sync_inspector()
        self._schedule_rebuild()

    # -------------------------------------------------------- cell inspector
    def _on_cell_selection_changed(self) -> None:
        '''Refresh the inspector after the layout selection changed.'''
        self._sync_inspector()

    def _sync_inspector(self) -> None:
        '''Push the selected cell state into the inspector widgets.

        Checkbox signals are blocked while syncing: the widgets mirror the
        editor state, they do not drive it. Checking a box reveals its
        parameter block indented below; an excluded cell disables the other
        actions (their hidden state is kept and restored on un-excluding).
        '''
        cell = self._cell_grid.selected_cell()
        boxes = (
            self._cb_exclude,
            self._cb_stud,
            self._cb_slope,
            self._cb_round,
            self._cb_chamfer,
        )
        for box in boxes:
            box.blockSignals(True)
        try:
            if cell is None:
                self._cell_group.setTitle("Cell (click the layout to select)")
                for box in boxes:
                    box.setEnabled(False)
                self._slope_params.setVisible(False)
                self._chamfer_params.setVisible(False)
                return
            ix, iy = cell
            state = self._cell_grid.state(ix, iy)
            sides = self._cell_grid.available_chamfer_sides(ix, iy)
            self._cell_group.setTitle(f"Cell [{ix}, {iy}]")
            self._cb_exclude.setEnabled(True)
            self._cb_exclude.setChecked(state.excluded)
            self._cb_stud.setEnabled(not state.excluded)
            self._cb_stud.setChecked(self._cell_grid.effective_stud(ix, iy))
            self._cb_slope.setEnabled(not state.excluded)
            self._cb_slope.setChecked(state.slope is not None)
            self._cb_round.setEnabled(not state.excluded)
            self._cb_round.setChecked(state.round)
            self._cb_chamfer.setEnabled(not state.excluded and bool(sides))
            self._cb_chamfer.setChecked(state.chamfer is not None)
            self._sync_slope_params(state)
            self._sync_chamfer_params(state, sides)
        finally:
            for box in boxes:
                box.blockSignals(False)

    def _sync_slope_params(self, state: CellState) -> None:
        '''Show/refresh the slope parameter pair for the selected cell.'''
        for widget in (self._slope_edge, self._slope_rise):
            widget.blockSignals(True)
        try:
            if state.slope is not None:
                side, rise = state.slope
                self._slope_edge.setCurrentText(side)
                self._slope_rise.setValue(rise)
            self._slope_params.setVisible(state.slope is not None)
        finally:
            for widget in (self._slope_edge, self._slope_rise):
                widget.blockSignals(False)

    def _sync_chamfer_params(self, state: CellState, sides: list[str]) -> None:
        '''Show/refresh the chamfer direction combo for the selected cell.'''
        if state.chamfer is not None and state.chamfer not in sides:
            # Stale direction (the neighbor across it was included): fall
            # back to the first outward side, or drop the chamfer entirely.
            state.chamfer = sides[0] if sides else None
            self._cb_chamfer.setChecked(state.chamfer is not None)
            cell = self._cell_grid.selected_cell()
            if cell is not None:
                self._cell_grid.set_state(*cell, state)
        self._chamfer_edge.blockSignals(True)
        try:
            self._chamfer_edge.clear()
            self._chamfer_edge.addItems(sides)
            if state.chamfer is not None:
                self._chamfer_edge.setCurrentText(state.chamfer)
            self._chamfer_params.setVisible(state.chamfer is not None)
        finally:
            self._chamfer_edge.blockSignals(False)

    def _edit_selected(self, mutate) -> None:
        '''Apply a mutation to the selected cell state, then re-sync.

        Args:
            mutate: Callable receiving the (mutable) CellState.
        '''
        cell = self._cell_grid.selected_cell()
        if cell is None:
            return
        ix, iy = cell
        state = self._cell_grid.state(ix, iy)
        mutate(state)
        self._cell_grid.set_state(ix, iy, state)
        self._sync_inspector()

    def _on_inspector_exclude(self, checked: bool) -> None:
        '''Exclude or restore the selected cell (features are preserved).'''
        self._edit_selected(lambda state: setattr(state, "excluded", checked))

    def _on_inspector_stud(self, checked: bool) -> None:
        '''Toggle the stud top; a stud replaces a slope (conflict).'''

        def mutate(state: CellState) -> None:
            state.stud = checked
            if checked:
                state.slope = None

        self._edit_selected(mutate)

    def _on_inspector_slope(self, checked: bool) -> None:
        '''Toggle the top slope; it replaces the stud and round base.'''

        def mutate(state: CellState) -> None:
            state.slope = (
                (self._slope_edge.currentText(), self._slope_rise.value())
                if checked
                else None
            )
            if checked:
                state.stud = None  # the wedge top replaces the stud
                state.round = False  # a wedge needs a square base

        self._edit_selected(mutate)

    def _on_inspector_round(self, checked: bool) -> None:
        '''Toggle the round base; it clears the wedge features (conflict).'''

        def mutate(state: CellState) -> None:
            state.round = checked
            if checked:
                state.slope = None
                state.chamfer = None

        self._edit_selected(mutate)

    def _on_inspector_chamfer(self, checked: bool) -> None:
        '''Toggle the bottom chamfer of the selected cell.'''

        def mutate(state: CellState) -> None:
            state.chamfer = (
                self._chamfer_edge.currentText() if checked else None
            )
            if checked:
                state.round = False  # a chamfer needs a square base

        self._edit_selected(mutate)

    def _on_slope_params_changed(self) -> None:
        '''Apply edited slope parameters to the selected cell.'''
        if not self._cb_slope.isChecked():
            return
        slope = (self._slope_edge.currentText(), self._slope_rise.value())
        self._edit_selected(lambda state: setattr(state, "slope", slope))

    def _on_chamfer_param_changed(self) -> None:
        '''Apply the edited chamfer direction to the selected cell.'''
        if not self._cb_chamfer.isChecked():
            return
        side = self._chamfer_edge.currentText()
        self._edit_selected(lambda state: setattr(state, "chamfer", side))

    # ----------------------------------------------------------------- actions
    def _collect_settings(self) -> BlockSettings:
        '''Build BlockSettings from the form widgets.

        Returns:
            Validated settings.

        Raises:
            SettingsValidationError: If values fail geometry validation.
        '''
        values: dict[str, object] = {
            "cells_x": self._cells_x.value(),
            "cells_y": self._cells_y.value(),
            "pins_enabled": self._pins.isChecked(),
            "tubes_enabled": self._tubes.isChecked(),
            "excluded_cells": self._cell_grid.excluded_cells(),
            "flat_cells": self._cell_grid.flat_cells(),
            "stud_cells": self._cell_grid.stud_cells(),
            "sloped_cells": self._cell_grid.sloped_cells(),
            "chamfered_cells": self._cell_grid.chamfered_cells(),
            "round_cells": self._cell_grid.round_cells(),
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
        self._cell_grid.set_dimensions(self._settings.cells_x, self._settings.cells_y)
        self._cell_grid.set_default_studs(self._settings.pins_enabled)
        self._cell_grid.clear_selection()
        self._cell_grid.set_from_lists(
            self._settings.excluded_cells,
            self._settings.flat_cells,
            self._settings.stud_cells,
            self._settings.sloped_cells,
            self._settings.chamfered_cells,
            self._settings.round_cells,
        )
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
        '''Schedule a background rebuild of the model from form values.

        Invalid intermediate form states are reported in the status label
        only (no dialogs), so typing stays uninterrupted.  A
        ``build_preview`` task (settings snapshot + tolerances) is sent to
        the worker process; the model is built and cached there under a
        generation cache key, and the mesh comes back as arrays via
        ``_on_job_done``.  Same-key submissions buffer (latest-wins).
        '''
        try:
            self._settings = self._collect_settings()
        except BrickGenError as exc:
            self._set_status(f"Invalid settings: {exc}", error=True)
            return

        self._gen_seq += 1
        seq = self._gen_seq
        settings = self._settings  # snapshot: the payload must not reference live state
        cache_key = f'gen{seq}'

        self._set_status(f"Generating {settings.describe()}…")

        payload = {
            'settings': settings.model_dump(),
            'tolerance': self._app_settings.preview_tolerance,
            'angular': self._app_settings.preview_angular_tolerance,
            'cache_key': cache_key,
        }
        job_id = self._jobs.submit('build_preview', payload, key='generation')
        self._job_meta[job_id] = (seq, cache_key, settings)

    def _on_job_done(self, job_id: int, result: object) -> None:
        '''Handle a completed background job.

        Dispatches generation results (by ``_job_meta`` seq mapping) and export
        results (by ``_export_meta`` path lookup), dropping unknown ids as
        foreign jobs.  Skipped/stale generation results are silently discarded;
        export errors always show a dialog because the user explicitly acted.

        Args:
            job_id: The job identifier returned by :meth:`JobQueue.submit`.
            result: Handler result dict, :data:`SKIPPED`, or a :class:`JobError`.
        '''
        # --- export routing (checked before generation) ---
        if job_id in self._export_meta:
            self._export_meta.pop(job_id)
            if result is SKIPPED:
                logger.warning("Export job %d was skipped (unexpected)", job_id)
                return
            if isinstance(result, JobError):
                self._set_status("Export failed.", error=True)
                QMessageBox.critical(self, "Export failed", str(result.exc))
                return
            if not isinstance(result, dict) or 'written' not in result:
                logger.error("Export job %d returned unexpected payload", job_id)
                self._set_status("Export failed.", error=True)
                return
            written = Path(str(result['written']))
            self._set_status(f"Exported: {written.name}")
            return

        # --- generation routing ---
        meta = self._job_meta.pop(job_id, None)
        if meta is None:
            # Foreign job (e.g. from a later slice) — drop silently.
            if result is not SKIPPED and not isinstance(result, JobError):
                logger.debug("Dropping foreign job id=%d", job_id)
            return
        seq, cache_key, settings = meta

        if result is SKIPPED:
            return  # Superseded generation; silent discard.

        # Stale check: a newer request superseded this between scheduling and
        # delivery; failures of stale generations are not user-relevant either.
        if seq != self._gen_seq:
            return

        if isinstance(result, JobError):
            self._set_status(f"Generation failed: {result.exc}", error=True)
            return

        if not isinstance(result, dict):
            logger.error("Generation job %d returned unexpected payload", job_id)
            return
        mesh = mesh_from_arrays(result)
        self._settings = settings
        self._applied_cache_key = result.get('cache_key', cache_key)

        slope_note = (
            f" | {len(settings.sloped_cells)} sloped"
            if settings.sloped_cells
            else ""
        )
        figure_note = (
            f" | figure {settings.included_count} cells"
            if not settings.is_rectangular
            else ""
        )
        chamfer_note = (
            f" | {len(settings.chamfered_cells)} chamfered"
            if settings.chamfered_cells
            else ""
        )
        round_note = (
            f" | {len(settings.round_cells)} round"
            if settings.round_cells
            else ""
        )
        self._status_style(False)
        self._status.setText(
            f"Form {settings.describe()} | footprint "
            f"{settings.outer_length_x:.2f} x {settings.outer_length_y:.2f} mm"
            f" | height {settings.total_height:.2f} mm"
            f"{slope_note}{figure_note}{chamfer_note}{round_note}"
        )
        self._preview.update_mesh(mesh, settings)
        self._preview.set_display_mode(self._mode.currentText())
        self._preview.set_grid_enabled(self._grid.isChecked())

    def _on_export(self) -> None:
        '''Ask for a destination file and queue the export on the background job
        queue.

        The exported model is the one cached in the worker process under the
        applied generation's key: a later generation re-render never affects
        this export.  Every user-confirmed export runs exactly once.  If the
        worker's LRU has already evicted the key (many newer generations),
        the error surfaces as a normal export-failure dialog.

        STEP is the default: it is first in the filter list and the initial
        file name. The selected filter defines the format and the file
        extension (see ``resolve_export_target``).
        '''
        if self._applied_cache_key is None:
            QMessageBox.information(
                self, "Nothing to export", "Generate a block first."
            )
            return
        formats = list(SUPPORTED_FORMATS)
        default_format = self._app_settings.default_format
        default_filter = f"{default_format} (*.{default_format.lower()})"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export model",
            f"{self._settings.recipe_name()}.{default_format.lower()}",
            ";;".join(f"{fmt} (*.{fmt.lower()})" for fmt in formats)
            + ";;All files (*.*)",
            default_filter,
        )
        if not path:
            return
        path, file_format = resolve_export_target(path, selected_filter)

        payload = {
            'cache_key': self._applied_cache_key,
            'path': path,
            'format': file_format,
            'tolerance': self._app_settings.export_tolerance,
            'angular': self._app_settings.export_angular_tolerance,
        }

        self._set_status(f"Exporting {Path(path).name}…")

        job_id = self._jobs.submit('export', payload)
        self._export_meta[job_id] = path

    def _on_save_settings(self) -> None:
        '''Persist current form values into a JSON preset.

        The default file name is the generated recipe stem, so presets of
        different blocks do not overwrite each other either.
        '''
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save settings",
            f"{self._settings.recipe_name()}.json",
            _SETTINGS_FILETYPES,
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
        self._status.setStyleSheet("color: #0a6;" if not error else "color: #b00;")

    def _set_status(self, text: str, error: bool = False) -> None:
        '''Set status text with the matching color.'''
        self._status_style(error)
        self._status.setText(text)


def main() -> None:
    '''Application entry point: configure logging and run the Qt GUI loop.'''
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
