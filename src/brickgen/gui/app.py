'''
File:   app.py
Brief:  BrickGen PySide6 main window: project scene, parameters, template library.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.14.0
'''

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import cast

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
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
from brickgen.core.project import (
    LAYOUT_FIELDS,
    SHARED_SETTING_FIELDS,
    BlockInstance,
    Project,
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
    GRID_SCALE_PRESETS,
    QUANTITY_LABEL_POSITIONS,
    load_app_settings,
    save_app_settings,
)
from brickgen.services.export_service import SUPPORTED_FORMATS
from brickgen.services.project_io import load_project, save_project

logger = logging.getLogger(__name__)

_PROJECT_FILETYPES = "BrickGen project (*.json);;All files (*.*)"
_SIDE_OFFSETS = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}

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
    '''Project scene in the middle, parameters on the left, template library
    on the right. File/View menus replace the old action buttons; the model
    regenerates automatically after any geometry change.
    '''

    def __init__(self) -> None:
        '''Build the layout and render the default one-block project.'''
        super().__init__()
        self._app_settings = load_app_settings()
        self._project = Project(blocks=[BlockInstance()])
        self._project_path: Path | None = None
        self._dirty = False
        self._selection: set[int] = {0}
        self._primary: int | None = 0
        self._loading_layout = False
        self._syncing_inspector = False
        self._inspector_pos: tuple[float, float] = (0.0, 0.0)
        self.setWindowTitle("Untitled[*] - BrickGen")
        self.resize(1400, 760)
        self.setMinimumSize(1100, 600)

        # Background job queue for heavy meshing off the UI thread. The model
        # itself lives in the worker process, keyed by generation.
        self._jobs = JobQueue(self)
        self._gen_seq = 0
        self._job_meta: dict[int, tuple[int, dict]] = {}
        self._applied_context: dict | None = None
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
        # Menus after the preview pane: the View actions bind to the preview.
        self._build_menus()
        # Floor grid: unit size from the shared settings, the rest persisted.
        self._preview.set_grid_unit(self._project.settings.unit_size)
        self._preview.set_grid_scale(self._app_settings.grid_scale)
        self._preview.set_snap_enabled(self._app_settings.grid_snap)
        self._preview.set_grid_enabled(self._app_settings.grid_enabled)

        # Auto-generation: debounced rebuild after any parameter change.
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(AUTO_GENERATE_DELAY_MS)
        self._regen_timer.timeout.connect(self._on_generate)

        self._sync_all()
        self._connect_auto_generate()
        self._on_generate()

    # ------------------------------------------------------------------ menus
    def _build_menus(self) -> None:
        '''Create the File and View menus (replacing the old button groups).'''
        file_menu = self.menuBar().addMenu("&File")
        act_new = QAction("&New Project", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._on_new_project)
        file_menu.addAction(act_new)
        act_open = QAction("&Open Project...", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open_project)
        file_menu.addAction(act_open)
        act_save = QAction("&Save Project", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_save_project)
        file_menu.addAction(act_save)
        act_save_as = QAction("Save Project &As...", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self._on_save_project_as)
        file_menu.addAction(act_save_as)
        file_menu.addSeparator()
        self._act_add_block = QAction("&Add Block", self)
        self._act_add_block.setShortcut("Ctrl+B")
        self._act_add_block.triggered.connect(self._on_add_block)
        file_menu.addAction(self._act_add_block)
        self._act_remove_blocks = QAction("&Remove Selected Blocks", self)
        self._act_remove_blocks.setShortcut("Del")
        self._act_remove_blocks.setEnabled(False)
        self._act_remove_blocks.triggered.connect(self._on_remove_blocks)
        file_menu.addAction(self._act_remove_blocks)
        file_menu.addSeparator()
        act_export = QAction("&Export Scene...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._on_export)
        file_menu.addAction(act_export)
        file_menu.addSeparator()
        act_prefs = QAction("S&ettings...", self)
        act_prefs.setShortcut(QKeySequence.StandardKey.Preferences)
        act_prefs.triggered.connect(self._on_preferences)
        file_menu.addAction(act_prefs)
        file_menu.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        view_menu = self.menuBar().addMenu("&View")
        mode_group = QActionGroup(self)
        for mode in DISPLAY_MODES:
            act_mode = QAction(mode, self)
            act_mode.setCheckable(True)
            act_mode.setChecked(mode == MODE_SOLID)
            act_mode.triggered.connect(
                lambda checked=False, m=mode: self._preview.set_display_mode(m)
            )
            mode_group.addAction(act_mode)
            view_menu.addAction(act_mode)
        view_menu.addSeparator()
        self._act_qty_labels = QAction("Show &block quantities", self)
        self._act_qty_labels.setCheckable(True)
        self._act_qty_labels.setChecked(self._app_settings.quantity_labels_enabled)
        self._act_qty_labels.toggled.connect(self._on_view_qty_labels)
        view_menu.addAction(self._act_qty_labels)
        position_menu = view_menu.addMenu("Quantity label &position")
        position_group = QActionGroup(self)
        for position in QUANTITY_LABEL_POSITIONS:
            act_pos = QAction(position.capitalize(), self)
            act_pos.setCheckable(True)
            act_pos.setChecked(position == self._app_settings.quantity_label_position)
            act_pos.triggered.connect(
                lambda checked=False, p=position: self._on_view_qty_position(p)
            )
            position_group.addAction(act_pos)
            position_menu.addAction(act_pos)

        grid_menu = self.menuBar().addMenu("&Grid")
        act_show_grid = QAction("Show &grid", self)
        act_show_grid.setCheckable(True)
        act_show_grid.setChecked(self._app_settings.grid_enabled)
        act_show_grid.toggled.connect(self._on_grid_show)
        grid_menu.addAction(act_show_grid)
        grid_menu.addSeparator()
        scale_menu = grid_menu.addMenu("Cell &scale")
        scale_group = QActionGroup(self)
        for scale in GRID_SCALE_PRESETS:
            act_scale = QAction(self._grid_scale_label(scale), self)
            act_scale.setCheckable(True)
            act_scale.setChecked(scale == self._app_settings.grid_scale)
            act_scale.triggered.connect(
                lambda checked=False, s=scale: self._on_grid_scale(s)
            )
            scale_group.addAction(act_scale)
            scale_menu.addAction(act_scale)
        grid_menu.addSeparator()
        act_snap = QAction("&Snap to grid", self)
        act_snap.setCheckable(True)
        act_snap.setChecked(self._app_settings.grid_snap)
        act_snap.setToolTip(
            "Gizmo and G-grab moves round to the grid step (unit size x cell scale)."
        )
        act_snap.toggled.connect(self._on_grid_snap)
        grid_menu.addAction(act_snap)

    # ------------------------------------------------------------------ layout
    def _build_sidebar(self) -> QWidget:
        '''Create the parameters panel with the status label.

        The content is wrapped in a scroll area so nothing is clipped when
        the window is too short for the whole form.
        '''
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.addWidget(self._build_size_group())
        layout.addWidget(self._build_blocks_group())
        layout.addWidget(self._build_block_group())
        layout.addWidget(self._build_layout_group())
        layout.addWidget(self._build_cell_group())
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
        '''Create the shared block-geometry form group.

        The grid size (Cells X/Y) is per block and lives in the selected
        block inspector; everything here applies to the whole project.
        '''
        group = QGroupBox("Block parameters (shared)")
        form = QFormLayout(group)
        form.setVerticalSpacing(8)

        self._floats: dict[str, QDoubleSpinBox] = {}
        for label, field, minimum, maximum in [
            ("Unit size, mm", "unit_size", 0.05, 100.0),
            ("Wall thickness, mm", "wall_thickness", 0.05, 20.0),
            ("Block height, mm", "block_height", 0.05, 200.0),
            ("Clearance X/Y, mm", "clearance_xy", -10.0, 10.0),
            ("Clearance Z, mm", "clearance_z", -10.0, 10.0),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            self._floats[field] = spin
            form.addRow(label, spin)

        self._derived_label = QLabel("")
        self._derived_label.setWordWrap(False)
        self._derived_label.setStyleSheet("color: #888;")
        form.addRow(self._derived_label)

        self._pins = QCheckBox("Studs on plain cells")
        self._pins.setToolTip(
            "Default top for cells without an explicit per-cell assignment.\n"
            "Individual cells are edited in the layout below (click a cell)."
        )
        form.addRow(self._pins)

        self._tubes = QCheckBox("Inner tubes (2x2+ centers)")
        form.addRow(self._tubes)
        return group

    def _build_blocks_group(self) -> QGroupBox:
        '''Create the placed-blocks list with add/duplicate/remove buttons.'''
        group = QGroupBox("Blocks")
        layout = QVBoxLayout(group)
        self._blocks_list = QListWidget()
        self._blocks_list.setMaximumHeight(140)
        self._blocks_list.setToolTip("Blocks placed on the scene; click to select.")
        self._blocks_list.itemClicked.connect(self._on_blocks_list_clicked)
        layout.addWidget(self._blocks_list)
        buttons = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.setToolTip("Add a new default block to the scene (Ctrl+B).")
        add_btn.clicked.connect(self._on_add_block)
        dup_btn = QPushButton("Duplicate")
        dup_btn.setToolTip("Duplicate the selected block next to it.")
        dup_btn.clicked.connect(self._on_duplicate_block)
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove the selected blocks from the scene.")
        remove_btn.clicked.connect(self._on_remove_blocks)
        buttons.addWidget(add_btn)
        buttons.addWidget(dup_btn)
        buttons.addWidget(remove_btn)
        layout.addLayout(buttons)
        return group

    def _build_block_group(self) -> QGroupBox:
        '''Create the inspector of the blocks selected on the scene.

        Grid size (Cells X/Y), preview color and BOM quantity are per block;
        edits apply to every selected block. With several blocks selected
        the position fields move the whole selection by the same offset.
        '''
        self._block_group = QGroupBox("Block (select in the scene)")
        form = QFormLayout(self._block_group)
        form.setVerticalSpacing(6)

        self._block_cells_x = QSpinBox()
        self._block_cells_x.setRange(1, 64)
        self._block_cells_x.setToolTip("Cells along X (length) of this block.")
        self._block_cells_y = QSpinBox()
        self._block_cells_y.setRange(1, 64)
        self._block_cells_y.setToolTip("Cells along Y (width) of this block.")
        form.addRow("Cells X (length)", self._block_cells_x)
        form.addRow("Cells Y (width)", self._block_cells_y)

        self._block_color = QPushButton("#3f7fbf")
        self._block_color.setFixedHeight(24)
        self._block_color.setToolTip(
            "Preview color of the block, for visual distinction."
        )
        form.addRow("Color", self._block_color)

        self._block_quantity = QSpinBox()
        self._block_quantity.setRange(1, 100000)
        self._block_quantity.setToolTip(
            "How many physical copies this block represents (BOM tracking)."
        )
        form.addRow("Quantity", self._block_quantity)

        self._block_pos_x = QDoubleSpinBox()
        self._block_pos_x.setRange(-100000.0, 100000.0)
        self._block_pos_x.setDecimals(2)
        self._block_pos_x.setSingleStep(1.0)
        self._block_pos_x.setSuffix(" mm")
        self._block_pos_x.setToolTip("Scene position of the block center along X.")
        self._block_pos_y = QDoubleSpinBox()
        self._block_pos_y.setRange(-100000.0, 100000.0)
        self._block_pos_y.setDecimals(2)
        self._block_pos_y.setSingleStep(1.0)
        self._block_pos_y.setSuffix(" mm")
        self._block_pos_y.setToolTip("Scene position of the block center along Y.")
        form.addRow("Position X", self._block_pos_x)
        form.addRow("Position Y", self._block_pos_y)
        return self._block_group

    def _build_layout_group(self) -> QGroupBox:
        '''Create the top-view cell grid editor of the selected block(s).'''
        group = QGroupBox("Layout (top view: click selects, Ctrl adds, drag frames)")
        layout = QVBoxLayout(group)
        self._cell_grid = CellGridEditor()
        self._cell_grid.changed.connect(self._on_grid_changed)
        self._cell_grid.selection_changed.connect(self._on_cell_selection_changed)
        include_all = QPushButton("Include all cells")
        include_all.clicked.connect(self._on_include_all)
        layout.addWidget(self._cell_grid, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(include_all)
        self._layout_group = group
        return group

    def _build_cell_group(self) -> QGroupBox:
        '''Create the inspector of the cells selected in the layout.

        A checklist of the actions available for the selected cells; checking
        an item reveals its parameter pairs indented below it and applies
        to every selected cell. The widgets mirror the primary (last
        clicked) cell. Slope and stud are mutually exclusive (a wedge top
        cannot carry a stud), the round base conflicts with the slope and
        chamfer wedges, exclusion hides everything else.
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

    def _build_preview_pane(self) -> QWidget:
        '''Build the scene preview area (settings moved to the File menu).'''
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        self._preview = PreviewWidget()
        layout.addWidget(self._preview, stretch=1)
        return pane

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
        for spin in self._floats.values():
            spin.valueChanged.connect(self._on_shared_param_changed)
            spin.valueChanged.connect(self._update_derived_label)
        self._pins.toggled.connect(self._on_pins_toggled)
        self._tubes.toggled.connect(self._on_shared_param_changed)
        self._preview.selection_changed.connect(self._on_scene_selection)
        self._preview.blocks_moved.connect(self._on_preview_blocks_moved)
        self._block_cells_x.valueChanged.connect(self._on_block_cells_changed)
        self._block_cells_y.valueChanged.connect(self._on_block_cells_changed)
        self._block_quantity.valueChanged.connect(self._on_block_quantity_changed)
        self._block_pos_x.valueChanged.connect(self._on_block_position_changed)
        self._block_pos_y.valueChanged.connect(self._on_block_position_changed)
        self._block_color.clicked.connect(self._on_block_color_clicked)

    def _update_derived_label(self) -> None:
        '''Update the read-only derived-dimensions label from current spin values.

        A temporary ``BlockSettings`` is built from the live form values so that
        the computed properties (``pin_diameter``, ``pin_height``,
        ``tube_outer_diameter``) are always consistent.  Invalid combos mid-edit
        show a ``"–"`` placeholder instead of crashing.
        '''
        try:
            s = BlockSettings(
                unit_size=self._floats["unit_size"].value(),
                wall_thickness=self._floats["wall_thickness"].value(),
                block_height=self._floats["block_height"].value(),
                clearance_xy=self._floats["clearance_xy"].value(),
                clearance_z=self._floats["clearance_z"].value(),
            )
        except Exception:
            self._derived_label.setText("–")
            return
        self._derived_label.setText(
            f"Stud ⌀{s.pin_diameter:.2f} × {s.pin_height:.2f} mm "
            f"· Tube ⌀{s.tube_outer_diameter:.2f} mm"
        )

    def _schedule_rebuild(self) -> None:
        '''(Re)start the debounced regeneration timer.'''
        self._regen_timer.start()

    def _on_shared_param_changed(self) -> None:
        '''A shared geometry control changed: mark dirty and regenerate.'''
        self._mark_dirty()
        self._schedule_rebuild()

    def _on_pins_toggled(self, checked: bool) -> None:
        '''Apply the new plain-cell default and regenerate.'''
        self._cell_grid.set_default_studs(checked)
        self._sync_inspector()
        self._on_shared_param_changed()

    def _on_include_all(self) -> None:
        '''Reset every cell feature from the "Include all" button.'''
        self._cell_grid.include_all()
        self._sync_inspector()

    # ------------------------------------------------------- project actions
    def _confirm_discard(self) -> bool:
        '''Ask about unsaved changes before a destructive action.

        Returns:
            True when the caller may proceed (no changes, or resolved).
        '''
        if not self._dirty:
            return True
        ret = QMessageBox.question(
            self,
            "BrickGen",
            "The project has unsaved changes. Save them?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if ret == QMessageBox.StandardButton.Save:
            return self._write_project()
        return ret == QMessageBox.StandardButton.Discard

    def _on_new_project(self) -> None:
        '''Start a fresh project with one default block.'''
        if not self._confirm_discard():
            return
        self._project = Project(blocks=[BlockInstance()])
        self._project_path = None
        self._applied_context = None
        self._selection = {0}
        self._primary = 0
        self._mark_clean()
        self._sync_all()
        self._on_generate()
        self._set_status("New project created.")

    def _on_open_project(self) -> None:
        '''Load a project file into the window and regenerate the scene.'''
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", self._default_project_dir(), _PROJECT_FILETYPES
        )
        if not path:
            return
        try:
            project = load_project(Path(path))
        except BrickGenError as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._project = project
        self._project_path = Path(path)
        self._applied_context = None
        self._selection = {0} if project.blocks else set()
        self._primary = 0 if project.blocks else None
        self._mark_clean()
        self._sync_all()
        self._on_generate()
        self._set_status(f"Project opened: {Path(path).name}")

    def _on_save_project(self) -> None:
        '''Save the project under its current path (Save As when untitled).'''
        self._write_project()

    def _on_save_project_as(self) -> bool:
        '''Ask for a destination and save the project there.

        Returns:
            True when the file was written successfully.
        '''
        stem = self._project_path.stem if self._project_path else "project"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            str(Path(self._default_project_dir()) / f"{stem}.json"),
            _PROJECT_FILETYPES,
        )
        if not path:
            return False
        return self._write_project(Path(path))

    def _write_project(self, path: Path | None = None) -> bool:
        '''Serialize the project to *path* (or the current project path).

        Returns:
            True when the file was written successfully.
        '''
        target = path if path is not None else self._project_path
        if target is None:
            return self._on_save_project_as()
        try:
            written = save_project(self._project, Path(target))
        except BrickGenError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self._project_path = Path(target)
        self._mark_clean()
        self._set_status(f"Project saved: {written.name}")
        return True

    def _default_project_dir(self) -> str:
        '''Default directory for project file dialogs.'''
        documents = Path.home() / "Documents" / "BrickGen"
        return str(documents) if documents.is_dir() else str(Path.home())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        '''Prompt about unsaved changes, then stop the background queue.'''
        if self._dirty and not self._confirm_discard():
            event.ignore()
            return
        # Non-blocking worker shutdown: without it the interpreter exit can
        # hang joining a live ProcessPoolExecutor worker.
        self._jobs.close()
        event.accept()

    # -------------------------------------------------------- block actions
    def _on_add_block(self) -> None:
        '''Append a default block to the scene and select it.'''
        merged = BlockInstance().to_settings(self._project.settings)
        half_x = merged.outer_length_x / 2
        half_y = merged.outer_length_y / 2
        position = self._auto_place_position(half_x, half_y)
        self._project.blocks.append(
            BlockInstance(position_x=position[0], position_y=position[1])
        )
        self._select_index(len(self._project.blocks) - 1)
        self._mark_dirty()
        self._sync_all()
        self._on_generate()
        self._set_status("Block added.")

    def _on_duplicate_block(self) -> None:
        '''Duplicate the primary block next to it and select the copy.'''
        if self._primary is None:
            return
        source = self._project.blocks[self._primary]
        merged = source.to_settings(self._project.settings)
        gap = self._project.settings.unit_size
        values = {
            **source.model_dump(),
            "position_x": source.position_x + merged.outer_length_x + gap,
        }
        self._project.blocks.append(BlockInstance.model_validate(values))
        self._select_index(len(self._project.blocks) - 1)
        self._mark_dirty()
        self._sync_all()
        self._on_generate()
        self._set_status("Block duplicated.")

    def _on_remove_blocks(self) -> None:
        '''Remove every selected block from the scene.'''
        if not self._selection:
            return
        for index in sorted(self._selection, reverse=True):
            del self._project.blocks[index]
        self._selection = set()
        self._primary = None
        self._preview.set_selection(set(), None)
        self._mark_dirty()
        self._sync_all()
        self._on_generate()
        self._set_status("Block(s) removed.")

    def _select_index(self, index: int) -> None:
        '''Select the single block *index* on the scene (programmatic).'''
        self._selection = {index}
        self._primary = index
        self._preview.set_selection(self._selection, index)
        self._on_scene_selection((frozenset(self._selection), index))

    def _on_blocks_list_clicked(self, item) -> None:
        '''Select the scene block matching the clicked list row.'''
        index = self._blocks_list.row(item)
        if 0 <= index < len(self._project.blocks):
            self._select_index(index)

    def _auto_place_position(self, half_x: float, half_y: float) -> tuple[float, float]:
        '''Pick a free spot to the right of the current scene bounds.'''
        blocks = self._project.blocks
        if not blocks:
            return (0.0, 0.0)
        settings = self._project.settings
        max_x = max(
            b.position_x + b.to_settings(settings).outer_length_x / 2 for b in blocks
        )
        return (max_x + half_x + settings.unit_size / 2, blocks[-1].position_y)

    def _on_scene_selection(self, payload: object) -> None:
        '''Sync the inspector after the scene selection changed.

        Args:
            payload: ``(frozenset of indices, primary index or None)``.
        '''
        selection, primary = cast(tuple[frozenset, int | None], payload)
        count = len(self._project.blocks)
        self._selection = {i for i in selection if 0 <= i < count}
        self._primary = primary if primary in self._selection else None
        self._sync_blocks_list()
        self._sync_block_inspector()
        self._sync_layout_from_primary()
        self._sync_inspector()

    # ---------------------------------------------------- block inspector
    def _sync_blocks_list(self) -> None:
        '''Rebuild the placed-blocks list and highlight the primary row.'''
        self._blocks_list.blockSignals(True)
        try:
            self._blocks_list.clear()
            for index, block in enumerate(self._project.blocks):
                self._blocks_list.addItem(
                    f"{index + 1}. {block.cells_x}x{block.cells_y} ×{block.quantity}"
                )
            if self._primary is not None and self._primary in self._selection:
                self._blocks_list.setCurrentRow(self._primary)
            else:
                self._blocks_list.setCurrentRow(-1)
        finally:
            self._blocks_list.blockSignals(False)

    def _sync_block_inspector(self) -> None:
        '''Push the primary selected block into the inspector widgets.

        Widgets are signal-blocked while syncing: they mirror the model
        state. The position snapshot is kept so a multi-selection edit
        shifts the whole selection by the same delta.
        '''
        self._syncing_inspector = True
        try:
            selection = sorted(self._selection)
            self._act_remove_blocks.setEnabled(bool(selection))
            if not selection:
                self._primary = None
                self._block_group.setEnabled(False)
                self._block_group.setTitle("Block (select in the scene)")
                self._layout_group.setEnabled(False)
                return
            if self._primary is None or self._primary not in self._selection:
                self._primary = selection[-1]
            block = self._project.blocks[self._primary]
            extra = len(selection) - 1
            title = f"Block [{self._primary + 1}]"
            if extra > 0:
                title = f"{title} +{extra} more"
            self._block_group.setEnabled(True)
            self._block_group.setTitle(title)
            self._block_cells_x.setValue(block.cells_x)
            self._block_cells_y.setValue(block.cells_y)
            self._block_color.setText(block.color)
            self._block_color.setStyleSheet(
                f"QPushButton {{ background-color: {block.color}; }}"
            )
            self._block_quantity.setValue(block.quantity)
            self._block_pos_x.setValue(block.position_x)
            self._block_pos_y.setValue(block.position_y)
            self._inspector_pos = (block.position_x, block.position_y)
            self._layout_group.setEnabled(True)
        finally:
            self._syncing_inspector = False

    def _on_block_cells_changed(self) -> None:
        '''Resize the selected blocks' grids and push the edited layout.'''
        if self._syncing_inspector or self._primary is None:
            return
        cells_x = self._block_cells_x.value()
        cells_y = self._block_cells_y.value()
        self._loading_layout = True
        try:
            self._cell_grid.set_dimensions(cells_x, cells_y)
        finally:
            self._loading_layout = False
        self._apply_layout_to_selection(cells=(cells_x, cells_y))
        self._mark_dirty()
        self._sync_blocks_list()
        self._sync_inspector()
        self._schedule_rebuild()

    def _on_block_color_clicked(self) -> None:
        '''Pick a color and apply it to every selected block.'''
        if self._primary is None:
            return
        color = QColorDialog.getColor(
            QColor(self._project.blocks[self._primary].color),
            self,
            "Select block color",
        )
        if not color.isValid():
            return
        hex_color = color.name()
        changed: dict[int, str] = {}
        for index in sorted(self._selection):
            self._project.blocks[index].color = hex_color
            changed[index] = hex_color
        self._preview.set_block_colors(changed)
        self._sync_block_inspector()
        self._mark_dirty()

    def _on_block_quantity_changed(self) -> None:
        '''Apply the edited BOM quantity to every selected block.'''
        if self._syncing_inspector or self._primary is None:
            return
        quantity = self._block_quantity.value()
        changed: dict[int, int] = {}
        for index in sorted(self._selection):
            self._project.blocks[index].quantity = quantity
            changed[index] = quantity
        self._preview.set_block_quantities(changed)
        self._sync_blocks_list()
        self._mark_dirty()

    def _on_block_position_changed(self) -> None:
        '''Apply the edited position to the selection.

        A single selected block moves to the absolute coordinates; with a
        multi-selection the whole group shifts by the same delta so the
        arrangement is preserved.
        '''
        if self._syncing_inspector or self._primary is None:
            return
        new_x = self._block_pos_x.value()
        new_y = self._block_pos_y.value()
        old_x, old_y = self._inspector_pos
        if len(self._selection) == 1:
            targets = {index: (new_x, new_y) for index in sorted(self._selection)}
        else:
            dx, dy = new_x - old_x, new_y - old_y
            targets = {
                index: (
                    self._project.blocks[index].position_x + dx,
                    self._project.blocks[index].position_y + dy,
                )
                for index in sorted(self._selection)
            }
        for index, (x, y) in targets.items():
            self._project.blocks[index].position_x = x
            self._project.blocks[index].position_y = y
            self._preview.set_block_position(index, x, y)
        self._inspector_pos = (new_x, new_y)
        self._mark_dirty()

    def _on_preview_blocks_moved(self, positions: dict) -> None:
        '''Commit the positions of a drag/grab move into the project.

        Args:
            positions: Mapping of block index to the final (x, y) position.
        '''
        moved = False
        for index, (x, y) in positions.items():
            if not 0 <= index < len(self._project.blocks):
                continue
            block = self._project.blocks[index]
            if block.position_x != x or block.position_y != y:
                moved = True
            block.position_x = x
            block.position_y = y
        if not moved:
            return
        self._mark_dirty()
        self._sync_block_inspector()
        self._set_status(self._scene_summary())

    # ------------------------------------------------------- layout editing
    def _sync_layout_from_primary(self) -> None:
        '''Load the primary selected block's layout into the grid editor.'''
        self._loading_layout = True
        try:
            if self._primary is None:
                self._cell_grid.set_dimensions(1, 1)
                self._cell_grid.include_all()
                return
            block = self._project.blocks[self._primary]
            self._cell_grid.set_dimensions(block.cells_x, block.cells_y)
            self._cell_grid.set_default_studs(self._project.settings.pins_enabled)
            self._cell_grid.set_from_lists(
                block.excluded_cells,
                block.flat_cells,
                block.stud_cells,
                block.sloped_cells,
                block.chamfered_cells,
                block.round_cells,
            )
        finally:
            self._loading_layout = False

    def _on_grid_changed(self) -> None:
        '''Push the edited grid layout into every selected block.'''
        if self._loading_layout or not self._selection:
            return
        self._apply_layout_to_selection()
        self._mark_dirty()
        self._schedule_rebuild()

    def _apply_layout_to_selection(self, cells: tuple[int, int] | None = None) -> None:
        '''Write the grid editor's layout into every selected block.

        Chamfer directions are filtered per block: a side is only kept when
        it is outward in THAT block's own layout. Blocks whose merged
        settings fail validation are left untouched.
        '''
        grid = self._cell_grid
        grid_lists = {
            "excluded_cells": grid.excluded_cells(),
            "flat_cells": grid.flat_cells(),
            "stud_cells": grid.stud_cells(),
            "sloped_cells": grid.sloped_cells(),
            "chamfered_cells": grid.chamfered_cells(),
            "round_cells": grid.round_cells(),
        }
        for index in sorted(self._selection):
            block = self._project.blocks[index]
            cells_x = block.cells_x if cells is None else cells[0]
            cells_y = block.cells_y if cells is None else cells[1]
            excluded = {tuple(pair) for pair in grid_lists["excluded_cells"]}
            # Chamfer directions are filtered per block: keep only sides that
            # are outward in THIS block's own layout.
            chamfered = [
                [ix, iy, side]
                for ix, iy, side in grid_lists["chamfered_cells"]
                if side in self._outward_sides(cells_x, cells_y, excluded, ix, iy)
            ]
            values = {
                **block.model_dump(),
                "cells_x": cells_x,
                "cells_y": cells_y,
                **grid_lists,
                "chamfered_cells": chamfered,
            }
            try:
                self._project.blocks[index] = BlockInstance.model_validate(values)
            except Exception:  # noqa: BLE001 - invalid layout: keep the block as is
                logger.debug("Layout rejected for block %d", index, exc_info=True)

    @staticmethod
    def _outward_sides(
        cells_x: int,
        cells_y: int,
        excluded: set[tuple[int, int]],
        ix: int,
        iy: int,
    ) -> list[str]:
        '''Outward sides of a cell: neighbor excluded or outside the grid.'''
        result: list[str] = []
        for side, (dx, dy) in _SIDE_OFFSETS.items():
            nx, ny = ix + dx, iy + dy
            inside = 0 <= nx < cells_x and 0 <= ny < cells_y
            if not inside or (nx, ny) in excluded:
                result.append(side)
        return result

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
        With several cells selected the widgets mirror the primary (last
        clicked) cell and edits apply to the whole selection.
        '''
        cells = self._cell_grid.selected_cells()
        cell = self._cell_grid.selected_cell()
        if cell is None and cells:
            cell = cells[0]
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
            extra = len(cells) - 1
            title = f"Cell [{ix}, {iy}]"
            if extra > 0:
                title = f"{title} +{extra} more"
            self._cell_group.setTitle(title)
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

    def _edit_selected_cells(
        self,
        mutate,
        *,
        skip_excluded: bool = True,
        chamfer_side: str | None = None,
    ) -> None:
        '''Apply a mutation to every selected cell state, then re-sync.

        Feature mutations skip excluded cells: their hidden state is kept
        and restored on un-excluding. A chamfer direction is applied only
        to the cells where that side is actually outward-facing.

        Args:
            mutate: Callable receiving the (mutable) CellState.
            skip_excluded: False when the mutation targets exclusion itself.
            chamfer_side: Side being applied; cells without that outward
                side are skipped. None skips nothing (chamfer removal).
        '''
        grid = self._cell_grid
        cells = grid.selected_cells()
        if not cells:
            return
        for ix, iy in cells:
            state = grid.state(ix, iy)
            if skip_excluded and state.excluded:
                continue
            if (
                chamfer_side is not None
                and chamfer_side not in grid.available_chamfer_sides(ix, iy)
            ):
                continue
            mutate(state)
            grid.set_state(ix, iy, state)
        self._sync_inspector()

    def _on_inspector_exclude(self, checked: bool) -> None:
        '''Exclude or restore the selected cells (features are preserved).'''
        self._edit_selected_cells(
            lambda state: setattr(state, "excluded", checked),
            skip_excluded=False,
        )

    def _on_inspector_stud(self, checked: bool) -> None:
        '''Toggle the stud top; a stud replaces a slope (conflict).'''

        def mutate(state: CellState) -> None:
            state.stud = checked
            if checked:
                state.slope = None

        self._edit_selected_cells(mutate)

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

        self._edit_selected_cells(mutate)

    def _on_inspector_round(self, checked: bool) -> None:
        '''Toggle the round base; it clears the wedge features (conflict).'''

        def mutate(state: CellState) -> None:
            state.round = checked
            if checked:
                state.slope = None
                state.chamfer = None

        self._edit_selected_cells(mutate)

    def _on_inspector_chamfer(self, checked: bool) -> None:
        '''Toggle the bottom chamfer of the selected cells.'''

        def mutate(state: CellState) -> None:
            state.chamfer = self._chamfer_edge.currentText() if checked else None
            if checked:
                state.round = False  # a chamfer needs a square base

        if checked:
            self._edit_selected_cells(
                mutate, chamfer_side=self._chamfer_edge.currentText()
            )
        else:
            self._edit_selected_cells(mutate)

    def _on_slope_params_changed(self) -> None:
        '''Apply edited slope parameters to the selected cells.'''
        if not self._cb_slope.isChecked():
            return
        slope = (self._slope_edge.currentText(), self._slope_rise.value())
        self._edit_selected_cells(lambda state: setattr(state, "slope", slope))

    def _on_chamfer_param_changed(self) -> None:
        '''Apply the edited chamfer direction to the selected cells.'''
        if not self._cb_chamfer.isChecked():
            return
        side = self._chamfer_edge.currentText()
        self._edit_selected_cells(
            lambda state: setattr(state, "chamfer", side), chamfer_side=side
        )

    # ------------------------------------------------------- state sync
    def _collect_shared_settings(self) -> BlockSettings:
        '''Build the shared BlockSettings from the form widgets.

        Returns:
            Validated settings (shared geometry only; per-cell layout lives
            on the placed blocks).

        Raises:
            SettingsValidationError: If values fail geometry validation.
        '''
        values: dict[str, object] = {
            field: spin.value() for field, spin in self._floats.items()
        }
        values["pins_enabled"] = self._pins.isChecked()
        values["tubes_enabled"] = self._tubes.isChecked()
        return validate_settings(values)

    def _collect_settings_or_default(self) -> BlockSettings:
        '''Best-effort settings snapshot for the template library panel.

        Returns:
            Current shared settings, or the last valid project settings
            when the form is currently in an invalid state.
        '''
        try:
            return self._collect_shared_settings()
        except BrickGenError:
            return self._project.settings

    def _sync_all(self) -> None:
        '''Push the whole project state into every form widget.'''
        self._sync_shared_form()
        self._sync_blocks_list()
        self._sync_block_inspector()
        self._sync_layout_from_primary()
        self._sync_inspector()
        self._update_title()

    def _sync_shared_form(self) -> None:
        '''Push shared settings values into the form widgets.'''
        settings = self._project.settings
        for field, spin in self._floats.items():
            spin.setValue(getattr(settings, field))
        self._pins.setChecked(settings.pins_enabled)
        self._tubes.setChecked(settings.tubes_enabled)
        self._update_derived_label()

    def _mark_dirty(self) -> None:
        '''Flag the project as modified (window title marker).'''
        self._dirty = True
        self.setWindowModified(True)

    def _mark_clean(self) -> None:
        '''Clear the modified flag (after save/open/new).'''
        self._dirty = False
        self.setWindowModified(False)

    def _update_title(self) -> None:
        '''Reflect the current project file name in the window title.'''
        name = self._project_path.name if self._project_path else "Untitled"
        self.setWindowTitle(f"{name}[*] - BrickGen")

    # -------------------------------------------------------- view menu
    @staticmethod
    def _grid_scale_label(scale: float) -> str:
        '''Menu label of a grid cell scale preset (e.g. ``50%``).'''
        return f"{round(scale * 100)}%"

    def _on_grid_show(self, checked: bool) -> None:
        '''Toggle the floor grid and persist it.'''
        self._app_settings.grid_enabled = checked
        self._save_app_settings()
        self._preview.set_grid_enabled(checked)

    def _on_grid_scale(self, scale: float) -> None:
        '''Set the grid cell scale factor and persist it.'''
        self._app_settings.grid_scale = scale
        self._save_app_settings()
        self._preview.set_grid_scale(scale)

    def _on_grid_snap(self, checked: bool) -> None:
        '''Toggle snap-to-grid movement and persist it.'''
        self._app_settings.grid_snap = checked
        self._save_app_settings()
        self._preview.set_snap_enabled(checked)

    def _on_view_qty_labels(self, checked: bool) -> None:
        '''Toggle the per-block BOM quantity labels and persist it.'''
        self._app_settings.quantity_labels_enabled = checked
        self._save_app_settings()
        self._preview.set_quantity_labels(
            checked, self._app_settings.quantity_label_position
        )

    def _on_view_qty_position(self, position: str) -> None:
        '''Set the quantity label anchor side and persist it.'''
        self._app_settings.quantity_label_position = position  # type: ignore[assignment]
        self._save_app_settings()
        self._preview.set_quantity_labels(
            self._app_settings.quantity_labels_enabled, position
        )

    def _save_app_settings(self) -> None:
        '''Persist application preferences, reporting failures inline.'''
        try:
            save_app_settings(self._app_settings)
        except OSError as exc:
            QMessageBox.critical(self, "Preferences", f"Cannot save: {exc}")

    def _on_preferences(self) -> None:
        '''Open the preferences modal and apply/save the results.'''
        dialog = PreferencesDialog(self._app_settings, self)
        if not dialog.exec():
            return
        updated = dialog.get_settings()
        if updated is None:
            return
        self._app_settings = updated
        self._save_app_settings()
        self._on_generate()  # re-render the preview with the new precision
        self._set_status("Preferences saved.")

    # ------------------------------------------------------------- templates
    def _on_template_selected(self, settings: BlockSettings) -> None:
        '''Apply a template picked in the library panel.

        The shared geometry is replaced from the template and the template
        itself is added to the scene as a new block with its own layout.
        '''
        self._project.settings = validate_settings(
            {field: getattr(settings, field) for field in SHARED_SETTING_FIELDS}
        )
        block = BlockInstance(
            cells_x=settings.cells_x,
            cells_y=settings.cells_y,
            **{field: getattr(settings, field) for field in LAYOUT_FIELDS},
        )
        self._project.blocks.append(block)
        self._select_index(len(self._project.blocks) - 1)
        self._mark_dirty()
        self._sync_all()
        self._on_generate()
        self._set_status(f"Template applied: {settings.describe()}")

    # ------------------------------------------------------------- generation
    def _on_generate(self) -> None:
        '''Schedule a background rebuild of the whole scene.

        Invalid intermediate form states are reported in the status label
        only (no dialogs), so typing stays uninterrupted.  A
        ``build_scene`` task (per-block settings snapshots + tolerances) is
        sent to the worker process; every block is built and meshed there,
        and the meshes come back as arrays via ``_on_job_done``.  Same-key
        submissions buffer (latest-wins).
        '''
        try:
            shared = self._collect_shared_settings()
        except BrickGenError as exc:
            self._set_status(f"Invalid settings: {exc}", error=True)
            return
        self._project.settings = shared
        # The grid cells and the snap step follow the unit size.
        self._preview.set_grid_unit(shared.unit_size)

        self._gen_seq += 1
        seq = self._gen_seq
        merged = [block.to_settings(shared) for block in self._project.blocks]
        payload = {
            'blocks': [
                {
                    'position_x': block.position_x,
                    'position_y': block.position_y,
                    'settings': settings.model_dump(),
                }
                for block, settings in zip(self._project.blocks, merged)
            ],
            'tolerance': self._app_settings.preview_tolerance,
            'angular': self._app_settings.preview_angular_tolerance,
            'cache_key': f'gen{seq}',
        }
        context = {'payload': payload, 'settings': merged}
        count = len(merged)
        self._set_status(
            f"Generating {count} blocks…" if count else "Scene is empty - add a block."
        )
        job_id = self._jobs.submit('build_scene', payload, key='generation')
        self._job_meta[job_id] = (seq, context)

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
            # Foreign job (e.g. thumbnail) — drop silently.
            if result is not SKIPPED and not isinstance(result, JobError):
                logger.debug("Dropping foreign job id=%d", job_id)
            return
        seq, context = meta

        if result is SKIPPED:
            return  # Superseded generation; silent discard.

        # Stale check: a newer request superseded this between scheduling and
        # delivery; failures of stale generations are not user-relevant either.
        if seq != self._gen_seq:
            return

        if isinstance(result, JobError):
            self._set_status(f"Generation failed: {result.exc}", error=True)
            return

        if not isinstance(result, dict) or 'blocks' not in result:
            logger.error("Generation job %d returned unexpected payload", job_id)
            return
        meshes = [mesh_from_arrays(arrays) for arrays in result['blocks']]
        self._applied_context = context

        # Colors/quantities/positions come from the CURRENT project state:
        # a matching sequence number means the block list is unchanged, and
        # the non-geometry edits (color/quantity/position) pushed their
        # values directly to the preview actors meanwhile.
        meta_list: list[dict] = []
        colors: list[str] = []
        for block, settings in zip(self._project.blocks, context['settings']):
            meta_list.append(
                {
                    'pos_x': block.position_x,
                    'pos_y': block.position_y,
                    'half_x': settings.outer_length_x / 2,
                    'half_y': settings.outer_length_y / 2,
                    'top_z': settings.total_height,
                    'quantity': block.quantity,
                }
            )
            colors.append(block.color)
        self._preview.update_scene(meshes, colors, meta_list)
        self._set_status(self._scene_summary())

    # ------------------------------------------------------------- export
    def _on_export(self) -> None:
        '''Ask for a destination file and queue the scene export.

        The exported model is the one built from the applied generation's
        payload snapshot: a later re-render never affects this export. All
        blocks are combined into one file, each block sitting at its scene
        position as a separate solid.
        '''
        context = self._applied_context
        if context is None or not context['payload']['blocks']:
            QMessageBox.information(
                self, "Nothing to export", "Add a block to the scene first."
            )
            return
        formats = list(SUPPORTED_FORMATS)
        default_format = self._app_settings.default_format
        default_filter = f"{default_format} (*.{default_format.lower()})"
        stem = self._project_path.stem if self._project_path else "brickgen-scene"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export scene",
            f"{stem}.{default_format.lower()}",
            ";;".join(f"{fmt} (*.{fmt.lower()})" for fmt in formats)
            + ";;All files (*.*)",
            default_filter,
        )
        if not path:
            return
        path, file_format = resolve_export_target(path, selected_filter)

        payload = {
            'blocks': context['payload']['blocks'],
            'path': path,
            'format': file_format,
            'tolerance': self._app_settings.export_tolerance,
            'angular': self._app_settings.export_angular_tolerance,
        }

        self._set_status(f"Exporting {Path(path).name}…")

        job_id = self._jobs.submit('export_scene', payload)
        self._export_meta[job_id] = path

    # ------------------------------------------------------------------ helpers
    def _scene_summary(self) -> str:
        '''Human-readable scene status: counts, BOM total, bounds.'''
        blocks = self._project.blocks
        if not blocks:
            return "Scene is empty - add a block."
        settings = self._project.settings
        merged = [block.to_settings(settings) for block in blocks]
        min_x = min(b.position_x - s.outer_length_x / 2 for b, s in zip(blocks, merged))
        max_x = max(b.position_x + s.outer_length_x / 2 for b, s in zip(blocks, merged))
        min_y = min(b.position_y - s.outer_length_y / 2 for b, s in zip(blocks, merged))
        max_y = max(b.position_y + s.outer_length_y / 2 for b, s in zip(blocks, merged))
        total_qty = sum(b.quantity for b in blocks)
        return (
            f"Scene: {len(blocks)} blocks | {total_qty} pcs total | "
            f"bounds {(max_x - min_x):.2f} x {(max_y - min_y):.2f} mm"
        )

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
