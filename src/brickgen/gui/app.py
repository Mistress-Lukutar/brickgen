'''
File:   app.py
Brief:  BrickGen PySide6 main window: project scene, parameters, template library.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.16.0
'''

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pyvista as pv
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
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
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
    BLOCK_FLAG_FIELDS,
    GRID_SCALE_PRESETS,
    LAYOUT_FIELDS,
    PRINT_STAGING_BED,
    QUANTITY_LABEL_POSITIONS,
    BlockInstance,
    BlockLayout,
    PrintBedSettings,
    Project,
)
from brickgen.gui.grid_editor import CellGridEditor, CellState
from brickgen.gui.jobs import SKIPPED, JobError, JobQueue
from brickgen.gui.library import TemplateLibraryPanel
from brickgen.gui.placement_panel import PlacementPanel
from brickgen.gui.placement_view import PlacementView
from brickgen.gui.preferences import PreferencesDialog
from brickgen.gui.preview import (
    DISPLAY_MODES,
    MODE_SOLID,
    PreviewWidget,
    mesh_from_arrays,
)
from brickgen.services.app_settings import load_app_settings, save_app_settings
from brickgen.services.export_service import SUPPORTED_FORMATS
from brickgen.services.placement_service import (
    auto_place,
    block_colors,
    block_footprints,
    distribute,
    find_free_spot,
    local_position,
    piece_dims,
    sync_pieces,
    world_position,
    zone_for_world_x,
)
from brickgen.services.print_export import collect_all_beds
from brickgen.services.project_io import load_project, save_project
from brickgen.services.template_library import Template

logger = logging.getLogger(__name__)

_PROJECT_FILETYPES = "BrickGen project (*.json);;All files (*.*)"
_SIDE_OFFSETS = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}

# Debounce for automatic regeneration after any parameter change, ms.
AUTO_GENERATE_DELAY_MS = 250


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
        self._applied_context: dict[str, Any] | None = None
        self._export_meta: dict[int, str] = {}  # job id -> resolved export path
        # Per-block regeneration flags (the "scene engine"): every edit of a
        # block's geometry marks that block dirty; a whole-scene rebuild is
        # requested only by structural changes (add/remove/open/template) and
        # by shared geometry edits that every block inherits. An edit made
        # while a generation is in flight stays dirty: the flag carries the
        # edit epoch, which is always newer than the epoch the generation
        # committed. `_compute_changed_mask` turns the flags into the per-block
        # mask consumed by ``update_scene``.
        self._edit_epoch = 0
        self._scene_dirty = True
        self._dirty_epoch: dict[int, int] = {}  # block index -> last edit epoch
        self._committed_epoch: dict[int, int] = {}  # block index -> applied epoch
        self._jobs.job_done.connect(self._on_job_done)

        # Placement tab state.
        self._pselection: set[int] = set()
        self._pprimary: int | None = None
        self._block_meshes: list[pv.PolyData | None] = []
        # Orientation mesh variants keyed (block index, flipped, rotated90):
        # every piece shares one of the four per-block variants.
        self._mesh_variants: dict[tuple[int, bool, bool], pv.PolyData] = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Sidebar stack: page 0 = planning sidebar, page 1 = placement panel.
        self._sidebar_stack = QStackedWidget()
        self._sidebar = self._build_sidebar()
        self._sidebar_stack.addWidget(self._sidebar)
        self._placement_panel = PlacementPanel()
        self._sidebar_stack.addWidget(self._placement_panel)

        # Tab widget: page 0 = planning preview, page 1 = placement view.
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_preview_pane(), "Planning")
        self._placement_view = PlacementView()
        self._tabs.addTab(self._placement_view, "Print placement")

        splitter.addWidget(self._sidebar_stack)
        splitter.addWidget(self._tabs)
        self._library = TemplateLibraryPanel(self._selected_block_layout, self._jobs)
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
        # Floor grid: unit size from the shared settings; every other view
        # option comes from the per-project view settings.
        self._preview.set_grid_unit(self._project.settings.unit_size)
        self._apply_view_settings()

        # Auto-generation: debounced rebuild after any parameter change.
        self._regen_timer = QTimer(self)
        self._regen_timer.setSingleShot(True)
        self._regen_timer.setInterval(AUTO_GENERATE_DELAY_MS)
        self._regen_timer.timeout.connect(self._on_generate)

        self._sync_all()
        self._connect_auto_generate()
        self._on_generate()

        # Placement view init.
        self._placement_view.set_print_beds(self._project.print_layout.beds)
        self._placement_panel.set_beds(self._project.print_layout.beds)
        self._tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------ tabs

    def _active_preview(self) -> PreviewWidget:
        '''Return the currently active preview widget.'''
        if self._tabs.currentIndex() == 0:
            return self._preview
        return self._placement_view

    def _on_tab_changed(self, index: int) -> None:
        '''Switch the sidebar page, library visibility and view-menu state.'''
        self._sidebar_stack.setCurrentIndex(index)
        self._library.setVisible(index == 0)

        # Enable/disable quantity actions only on planning tab.
        self._act_qty_labels.setEnabled(index == 0)

        # Refresh "Show grid" checked state from the active preview.
        self._refresh_grid_action_state()

        # When entering tab 1: push shared settings into placement view.
        if index == 1:
            self._placement_view.set_grid_unit(self._project.settings.unit_size)
            self._placement_view.set_grid_scale(self._project.view.grid_scale)
            self._placement_view.set_snap_enabled(self._project.view.grid_snap)
            self._placement_view.set_grid_enabled(
                self._project.view.placement_grid_enabled
            )
            self._placement_view.set_print_beds(self._project.print_layout.beds)
            self._sync_placement()

    def _refresh_grid_action_state(self) -> None:
        '''Update the "Show grid" action checked state from the active preview.

        The check change is signal-blocked: a programmatic sync must not
        re-enter ``_on_grid_show`` (which would treat it as a user edit and
        dirty the project on a mere tab switch).
        '''
        if self._tabs.currentIndex() == 0:
            checked = self._project.view.grid_enabled
        else:
            checked = self._project.view.placement_grid_enabled
        self._act_show_grid.blockSignals(True)
        try:
            self._act_show_grid.setChecked(checked)
        finally:
            self._act_show_grid.blockSignals(False)

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
        act_export = QAction("&Export Print Layout...", self)
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self._on_export_print)
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
                lambda checked=False, m=mode: self._active_preview().set_display_mode(m)
            )
            mode_group.addAction(act_mode)
            view_menu.addAction(act_mode)
        view_menu.addSeparator()
        self._act_qty_labels = QAction("Show &block quantities", self)
        self._act_qty_labels.setCheckable(True)
        self._act_qty_labels.setChecked(self._project.view.quantity_labels_enabled)
        self._act_qty_labels.toggled.connect(self._on_view_qty_labels)
        view_menu.addAction(self._act_qty_labels)
        position_menu = view_menu.addMenu("Quantity label &position")
        position_group = QActionGroup(self)
        self._act_qty_positions: dict[str, QAction] = {}
        for position in QUANTITY_LABEL_POSITIONS:
            act_pos = QAction(position.capitalize(), self)
            act_pos.setCheckable(True)
            act_pos.setChecked(position == self._project.view.quantity_label_position)
            act_pos.triggered.connect(
                lambda checked=False, p=position: self._on_view_qty_position(p)
            )
            position_group.addAction(act_pos)
            position_menu.addAction(act_pos)
            self._act_qty_positions[position] = act_pos

        grid_menu = self.menuBar().addMenu("&Grid")
        self._act_show_grid = QAction("Show &grid", self)
        self._act_show_grid.setCheckable(True)
        self._act_show_grid.setChecked(
            self._project.view.grid_enabled
            if self._tabs.currentIndex() == 0
            else self._project.view.placement_grid_enabled
        )
        self._act_show_grid.toggled.connect(self._on_grid_show)
        grid_menu.addAction(self._act_show_grid)
        grid_menu.addSeparator()
        scale_menu = grid_menu.addMenu("Cell &scale")
        scale_group = QActionGroup(self)
        self._act_grid_scale: dict[float, QAction] = {}
        for scale in GRID_SCALE_PRESETS:
            act_scale = QAction(self._grid_scale_label(scale), self)
            act_scale.setCheckable(True)
            act_scale.setChecked(scale == self._project.view.grid_scale)
            act_scale.triggered.connect(
                lambda checked=False, s=scale: self._on_grid_scale(s)
            )
            scale_group.addAction(act_scale)
            scale_menu.addAction(act_scale)
            self._act_grid_scale[scale] = act_scale
        grid_menu.addSeparator()
        self._act_snap = QAction("&Snap to grid", self)
        self._act_snap.setCheckable(True)
        self._act_snap.setChecked(self._project.view.grid_snap)
        self._act_snap.setToolTip(
            "Gizmo and G-grab moves round to the grid step (unit size x cell scale)."
        )
        self._act_snap.toggled.connect(self._on_grid_snap)
        grid_menu.addAction(self._act_snap)

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

        self._block_tubes = QCheckBox("Inner tubes (2x2+ centers)")
        self._block_tubes.setToolTip(
            "Build the inner tubes of this block (2x2+ blocks only)."
        )
        form.addRow(self._block_tubes)

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
        '''Build the planning preview widget (tab 0 of the tab widget).'''
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
        self._block_tubes.toggled.connect(self._on_block_tubes_toggled)
        self._preview.selection_changed.connect(self._on_scene_selection)
        self._preview.blocks_moved.connect(self._on_preview_blocks_moved)
        self._block_cells_x.valueChanged.connect(self._on_block_cells_changed)
        self._block_cells_y.valueChanged.connect(self._on_block_cells_changed)
        self._block_quantity.valueChanged.connect(self._on_block_quantity_changed)
        self._block_pos_x.valueChanged.connect(self._on_block_position_changed)
        self._block_pos_y.valueChanged.connect(self._on_block_position_changed)
        self._block_color.clicked.connect(self._on_block_color_clicked)

        # Placement signals.
        self._placement_view.selection_changed.connect(self._on_placement_selection)
        self._placement_view.blocks_moved.connect(self._on_placement_pieces_moved)
        self._placement_panel.beds_changed.connect(self._on_placement_beds_changed)
        self._placement_panel.auto_place_requested.connect(
            self._on_placement_auto_place
        )
        self._placement_panel.distribute_requested.connect(
            self._on_placement_distribute
        )
        self._placement_panel.piece_bed_changed.connect(self._on_piece_bed_changed)
        self._placement_panel.piece_position_changed.connect(
            self._on_piece_position_changed
        )
        self._placement_panel.piece_flip_changed.connect(self._on_piece_flip_changed)
        self._placement_panel.flip_selection_requested.connect(self._on_flip_selection)
        self._placement_panel.stage_selection_requested.connect(
            self._on_stage_selection
        )

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
        self._mark_scene_dirty()
        self._mark_dirty()
        self._schedule_rebuild()

    # ----------------------------------------------------- scene engine flags
    def _mark_blocks_dirty(self, indices: Iterable[int]) -> None:
        '''Flag the given blocks as needing a rebuild on the next generation.

        MUST be called by every handler that changes a block's geometry
        (per-cell layout, stud/tube flags, cell counts). Forgetting the flag
        leaves the block stale on screen until something else rebuilds it.

        Args:
            indices: Scene indices of the edited blocks.
        '''
        self._edit_epoch += 1
        for index in indices:
            self._dirty_epoch[index] = self._edit_epoch

    def _mark_scene_dirty(self) -> None:
        '''Request a full-scene rebuild (structural or shared-geometry edit).

        Used whenever the per-block flags are not enough: the block list
        changed, a shared numeric field changed (every block inherits it),
        or the tessellation tolerances changed.
        '''
        self._edit_epoch += 1
        self._scene_dirty = True

    @staticmethod
    def _compute_changed_mask(
        scene_dirty: bool,
        applied_blocks: int,
        block_count: int,
        dirty_epoch: dict[int, int],
        committed_epoch: dict[int, int],
    ) -> list[bool]:
        '''Per-block rebuild mask for the next generation.

        A block is rebuilt when its edit epoch is newer than the epoch of the
        generation last applied for it. A whole-scene rebuild happens when
        the structure changed (or nothing was applied yet); then the per-block
        bookkeeping is ignored and every block is rebuilt.

        Args:
            scene_dirty: True after a structural or shared-geometry edit.
            applied_blocks: Block count of the last applied generation
                (0 when nothing was applied yet).
            block_count: Current block count.
            dirty_epoch: Block index -> epoch of the last geometry edit.
            committed_epoch: Block index -> epoch committed by the last
                generation that rebuilt that block.

        Returns:
            One bool per block: True = rebuild this block's mesh and actors.
        '''
        if scene_dirty or applied_blocks != block_count:
            return [True] * block_count
        return [
            dirty_epoch.get(index, -1) > committed_epoch.get(index, -1)
            for index in range(block_count)
        ]

    def _commit_applied_mask(self, changed: list[bool], epoch: int) -> None:
        '''Record which blocks the applied generation actually rebuilt.

        Blocks with ``True`` in *changed* are committed at *epoch*; an edit
        of the same block made after the submission keeps a newer epoch and
        therefore stays dirty for the next generation. A full-scene rebuild
        resets the per-block bookkeeping.

        Args:
            changed: The rebuild mask of the applied generation.
            epoch: The edit epoch at the moment the generation was submitted.
        '''
        if all(changed) or not changed:
            self._scene_dirty = False
            self._dirty_epoch.clear()
            self._committed_epoch.clear()
            return
        for index, rebuilt in enumerate(changed):
            if rebuilt:
                self._committed_epoch[index] = epoch
        self._scene_dirty = False

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
        self._mark_scene_dirty()
        self._selection = {0}
        self._primary = 0
        # Reset placement state.
        self._pselection = set()
        self._pprimary = None
        self._block_meshes = []
        self._mesh_variants = {}
        self._mark_clean()
        self._apply_view_settings()
        self._sync_all()
        self._placement_view.set_print_beds(self._project.print_layout.beds)
        self._placement_panel.set_beds(self._project.print_layout.beds)
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
        self._mark_scene_dirty()
        self._selection = {0} if project.blocks else set()
        self._primary = 0 if project.blocks else None
        # Reset placement state.
        self._pselection = set()
        self._pprimary = None
        self._block_meshes = []
        self._mesh_variants = {}
        self._mark_clean()
        # Restore the project's own view state (grids, snap, labels).
        self._apply_view_settings()
        self._sync_all()
        self._placement_view.set_print_beds(self._project.print_layout.beds)
        self._placement_panel.set_beds(self._project.print_layout.beds)
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
        self._mark_scene_dirty()
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
        self._mark_scene_dirty()
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
        self._mark_scene_dirty()
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

    def _selected_block_layout(self) -> BlockLayout | None:
        '''Block-level definition of the primary selected block (None if empty).'''
        if self._primary is None:
            return None
        return self._project.blocks[self._primary]

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
            expected = [
                f"{index + 1}. {block.cells_x}x{block.cells_y} ×{block.quantity}"
                for index, block in enumerate(self._project.blocks)
            ]
            if self._blocks_list.count() == len(expected) and all(
                self._blocks_list.item(row).text() == text
                for row, text in enumerate(expected)
            ):
                # Content unchanged: only the selection moved. Keep the
                # items (rebuilding a long list on every scene click stalls
                # the UI) and just move the highlighted row.
                if self._primary is not None and self._primary in self._selection:
                    self._blocks_list.setCurrentRow(self._primary)
                else:
                    self._blocks_list.setCurrentRow(-1)
                return
            self._blocks_list.clear()
            for text in expected:
                self._blocks_list.addItem(text)
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
            self._block_tubes.setChecked(block.tubes_enabled)
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

    def _on_block_tubes_toggled(self, checked: bool) -> None:
        '''Apply the block-level inner-tubes toggle to every selected block.'''
        if self._syncing_inspector or self._primary is None:
            return
        for index in sorted(self._selection):
            self._project.blocks[index].tubes_enabled = bool(checked)
        self._mark_blocks_dirty(sorted(self._selection))
        self._mark_dirty()
        self._schedule_rebuild()

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
            self._cell_grid.set_default_studs(block.pins_enabled)
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
        self._mark_blocks_dirty(sorted(self._selection))

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
            Validated settings (shared geometry only; per-cell layout and
            stud/tube flags live on the placed blocks).

        Raises:
            SettingsValidationError: If values fail geometry validation.
        '''
        values: dict[str, object] = {
            field: spin.value() for field, spin in self._floats.items()
        }
        return validate_settings(values)

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

    def _apply_view_settings(self) -> None:
        '''Push the per-project view settings into both previews and menus.

        Called on startup and whenever the whole project is replaced
        (new/open), so the view state always matches ``self._project``.
        '''
        view = self._project.view
        self._preview.set_grid_scale(view.grid_scale)
        self._preview.set_snap_enabled(view.grid_snap)
        self._preview.set_grid_enabled(view.grid_enabled)
        self._preview.set_quantity_labels(
            view.quantity_labels_enabled, view.quantity_label_position
        )
        self._placement_view.set_grid_scale(view.grid_scale)
        self._placement_view.set_snap_enabled(view.grid_snap)
        self._placement_view.set_grid_enabled(view.placement_grid_enabled)
        self._refresh_view_actions()

    def _refresh_view_actions(self) -> None:
        '''Sync the View/Grid menu checked states from the project view.

        Signal-blocked: the programmatic check changes must not re-enter
        the toggle handlers (which would dirty the project on open/new).
        '''
        view = self._project.view
        actions: list[QAction] = [
            self._act_show_grid,
            self._act_snap,
            self._act_qty_labels,
            *self._act_grid_scale.values(),
            *self._act_qty_positions.values(),
        ]
        for action in actions:
            action.blockSignals(True)
        try:
            self._act_show_grid.setChecked(view.grid_enabled)
            self._act_snap.setChecked(view.grid_snap)
            self._act_qty_labels.setChecked(view.quantity_labels_enabled)
            for scale, action in self._act_grid_scale.items():
                action.setChecked(scale == view.grid_scale)
            for position, action in self._act_qty_positions.items():
                action.setChecked(position == view.quantity_label_position)
        finally:
            for action in actions:
                action.blockSignals(False)

    def _on_grid_show(self, checked: bool) -> None:
        '''Toggle the floor grid on the active preview (stored per project).'''
        if self._tabs.currentIndex() == 0:
            self._project.view.grid_enabled = checked
        else:
            self._project.view.placement_grid_enabled = checked
        self._mark_dirty()
        self._active_preview().set_grid_enabled(checked)

    def _on_grid_scale(self, scale: float) -> None:
        '''Set the grid cell scale factor (stored per project, both previews).'''
        self._project.view.grid_scale = scale
        self._mark_dirty()
        self._preview.set_grid_scale(scale)
        self._placement_view.set_grid_scale(scale)

    def _on_grid_snap(self, checked: bool) -> None:
        '''Toggle snap-to-grid movement (stored per project, both previews).'''
        self._project.view.grid_snap = checked
        self._mark_dirty()
        self._preview.set_snap_enabled(checked)
        self._placement_view.set_snap_enabled(checked)

    def _on_view_qty_labels(self, checked: bool) -> None:
        '''Toggle the per-block BOM quantity labels (stored per project).'''
        self._project.view.quantity_labels_enabled = checked
        self._mark_dirty()
        self._preview.set_quantity_labels(
            checked, self._project.view.quantity_label_position
        )

    def _on_view_qty_position(self, position: str) -> None:
        '''Set the quantity label anchor side (stored per project).'''
        self._project.view.quantity_label_position = position  # type: ignore[assignment]
        self._mark_dirty()
        self._preview.set_quantity_labels(
            self._project.view.quantity_labels_enabled, position
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
        self._mark_scene_dirty()  # the tessellation tolerances changed
        self._on_generate()  # re-render the preview with the new precision
        self._set_status("Preferences saved.")

    # ------------------------------------------------------------- templates
    def _on_template_selected(self, template: Template) -> None:
        '''Apply a template picked in the library panel.

        Builds a block from ``template.layout`` merged against the current
        shared settings, auto-places it, appends and selects it; the shared
        geometry is never replaced by the template.
        '''
        merged = template.layout.to_settings(self._project.settings)
        position = self._auto_place_position(
            merged.outer_length_x / 2, merged.outer_length_y / 2
        )
        block = BlockInstance(
            cells_x=template.layout.cells_x,
            cells_y=template.layout.cells_y,
            **{field: getattr(template.layout, field) for field in BLOCK_FLAG_FIELDS},
            **{field: getattr(template.layout, field) for field in LAYOUT_FIELDS},
            position_x=position[0],
            position_y=position[1],
        )
        self._project.blocks.append(block)
        self._select_index(len(self._project.blocks) - 1)
        self._mark_scene_dirty()
        self._mark_dirty()
        self._sync_all()
        self._on_generate()
        self._set_status(f"Template applied: {template.name}")

    # ------------------------------------------------------------- generation
    def _on_generate(self) -> None:
        '''Schedule a background rebuild of the scene.

        Invalid intermediate form states are reported in the status label
        only (no dialogs), so typing stays uninterrupted.  A
        ``build_scene`` task (per-block settings snapshots + tolerances) is
        sent to the worker process; every block is built and meshed there,
        and the meshes come back as arrays via ``_on_job_done``.  Same-key
        submissions buffer (latest-wins).  The per-block rebuild mask comes
        from the dirty flags (see the scene engine fields in ``__init__``):
        only edited blocks change their preview actors, structural and
        shared-geometry edits rebuild the whole scene.
        '''
        try:
            shared = self._collect_shared_settings()
        except BrickGenError as exc:
            self._set_status(f"Invalid settings: {exc}", error=True)
            return
        self._project.settings = shared
        # The grid cells and the snap step follow the unit size.
        self._preview.set_grid_unit(shared.unit_size)
        self._placement_view.set_grid_unit(shared.unit_size)

        self._gen_seq += 1
        seq = self._gen_seq
        merged = [block.to_settings(shared) for block in self._project.blocks]
        payload: dict[str, Any] = {
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
        # Per-block rebuild mask from the dirty flags: unchanged blocks keep
        # their actors, so editing one block never rebuilds the whole scene.
        changed = self._compute_changed_mask(
            scene_dirty=self._scene_dirty,
            applied_blocks=(
                len(self._applied_context['payload']['blocks'])
                if self._applied_context is not None
                else -1
            ),
            block_count=len(merged),
            dirty_epoch=self._dirty_epoch,
            committed_epoch=self._committed_epoch,
        )
        context = {
            'payload': payload,
            'settings': merged,
            'changed': changed,
            'epoch': self._edit_epoch,
        }
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
        Uses the per-block rebuild mask so unchanged blocks keep their
        existing preview actors, and commits the applied epochs afterwards
        (edits made while the generation was in flight stay dirty).

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
            written = [Path(str(p)) for p in result['written']]
            self._set_status(f"Exported: {', '.join(p.name for p in written)}")
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
        meshes = [
            mesh_from_arrays(arrays) if flag else None
            for arrays, flag in zip(result['blocks'], context['changed'])
        ]
        self._applied_context = context
        self._commit_applied_mask(context['changed'], context['epoch'])

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
        self._preview.update_scene(meshes, colors, meta_list, context['changed'])

        # Store resolved meshes for placement tab; reset orientation cache.
        self._block_meshes = list(meshes)
        self._mesh_variants = {}
        self._sync_placement()

        self._set_status(self._scene_summary())

    # ------------------------------------------------------------- export
    def _on_export_print(self) -> None:
        '''Ask for a destination folder and format, then queue the layout
        export: one compound file per print bed.

        Only pieces fully inside their bed are exported (a piece hanging
        over the edge would not print); skipped pieces are reported in the
        status afterwards.
        '''
        beds, _skipped = collect_all_beds(self._project)
        if not beds:
            QMessageBox.information(
                self, "Nothing to export", "Place pieces on a bed first."
            )
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select export folder", self._default_project_dir()
        )
        if not folder:
            return
        formats = list(SUPPORTED_FORMATS)
        file_format, ok = QInputDialog.getItem(
            self,
            "Export print layout",
            "Format:",
            formats,
            formats.index(self._app_settings.default_format),
            False,
        )
        if not ok:
            return

        stem = self._project_path.stem if self._project_path else "brickgen-layout"
        payload_beds = []
        for bed in beds:
            name = f"{stem}_bed{bed['index'] + 1:02d}.{file_format.lower()}"
            path = Path(folder) / name
            payload_beds.append(
                {'index': bed['index'], 'path': str(path), 'pieces': bed['pieces']}
            )

        payload = {
            'beds': payload_beds,
            'format': file_format,
            'tolerance': self._app_settings.export_tolerance,
            'angular': self._app_settings.export_angular_tolerance,
        }

        self._set_status(f"Exporting {len(payload_beds)} bed file(s)…")

        job_id = self._jobs.submit('export_print', payload)
        self._export_meta[job_id] = folder

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
        if self._tabs.currentIndex() == 0:
            self._status.setText(text)
        else:
            self._placement_panel.set_status(text, error)

    # ------------------------------------------------------- placement sync

    def _sync_placement(self) -> None:
        '''Reconcile pieces with blocks and rebuild the placement scene.'''
        report = sync_pieces(self._project)
        if report.added or report.removed:
            self._mark_dirty()
        self._refresh_placement_scene()
        self._placement_panel.sync_inspector(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            block_colors(self._project),
            self._project.blocks,
            self._pselection,
            self._pprimary,
        )
        if self._tabs.currentIndex() == 1:
            self._set_status(self._placement_summary())

    def _refresh_placement_scene(self) -> None:
        '''Rebuild the placement viewport actors from cached block meshes.'''
        beds = self._project.print_layout.beds
        footprints = block_footprints(self._project)
        heights = {
            b.uid: self._project.block_settings(i).total_height
            for i, b in enumerate(self._project.blocks)
        }
        colors_map = block_colors(self._project)

        # uid -> block index map.
        uid_to_idx: dict[str, int] = {}
        for idx, blk in enumerate(self._project.blocks):
            uid_to_idx[blk.uid] = idx

        meshes_out: list[pv.PolyData | None] = []
        colors_out: list[str] = []
        meta_out: list[dict] = []

        pieces = self._project.print_layout.pieces
        for piece in pieces:
            blk_idx = uid_to_idx.get(piece.block_uid)
            if blk_idx is None or blk_idx >= len(self._block_meshes):
                continue
            base_mesh = self._block_meshes[blk_idx]
            if base_mesh is None:
                continue

            mesh: pv.PolyData = base_mesh
            if piece.flipped or piece.rotated90:
                # Lazy variant cache: one transformed mesh per orientation.
                key = (blk_idx, piece.flipped, piece.rotated90)
                variant = self._mesh_variants.get(key)
                if variant is None:
                    variant = base_mesh.copy()
                    if piece.rotated90:
                        # 90 deg about Z: footprint swaps (matches export).
                        variant.rotate_z(90.0)
                    if piece.flipped:
                        # 180 deg about X: pins down; lift onto z >= 0.
                        variant.rotate_x(180.0)
                        variant.translate((0.0, 0.0, heights.get(piece.block_uid, 0)))
                    self._mesh_variants[key] = variant
                mesh = variant

            wx, wy = world_position(piece, beds)
            half_x, half_y = piece_dims(piece, footprints)
            meta_out.append(
                {
                    'pos_x': wx,
                    'pos_y': wy,
                    'half_x': half_x / 2.0,
                    'half_y': half_y / 2.0,
                    'top_z': heights.get(piece.block_uid, 0),
                    'quantity': 1,
                }
            )
            colors_out.append(colors_map.get(piece.block_uid, "#3f7fbf"))
            meshes_out.append(mesh)

        self._placement_view.update_scene(meshes_out, colors_out, meta_out)

        # Prune selection to current piece count.
        n_pieces = len(pieces)
        self._pselection = {i for i in self._pselection if i < n_pieces}
        if self._pprimary is not None and self._pprimary >= n_pieces:
            self._pprimary = None

    def _placement_summary(self) -> str:
        '''"Placement: N pieces on K beds, M staged, X outside beds".'''
        pieces = self._project.print_layout.pieces
        if not pieces:
            return "Placement: 0 pieces."
        beds = self._project.print_layout.beds
        total = len(pieces)
        staged = sum(1 for p in pieces if p.bed == PRINT_STAGING_BED)
        on_beds = total - staged

        # Count pieces whose footprint extends outside their zone bounds.
        footprints = block_footprints(self._project)
        outside = 0
        for piece in pieces:
            if piece.bed == PRINT_STAGING_BED:
                continue
            if piece.block_uid not in footprints:
                outside += 1
                continue
            w, d = piece_dims(piece, footprints)
            px = piece.position_x
            py = piece.position_y
            # Check if footprint extends beyond [gap, width-gap] x [gap, depth-gap].
            if (
                px - w / 2 < beds.gap
                or px + w / 2 > beds.width - beds.gap
                or py - d / 2 < beds.gap
                or py + d / 2 > beds.depth - beds.gap
            ):
                outside += 1

        return (
            f"Placement: {total} pieces on {on_beds} bed(s), "
            f"{staged} staged, {outside} outside"
        )

    def _after_placement_change(self) -> None:
        '''Refresh scene, inspector, and status after any placement edit.'''
        self._refresh_placement_scene()
        self._placement_panel.sync_inspector(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            block_colors(self._project),
            self._project.blocks,
            self._pselection,
            self._pprimary,
        )
        if self._tabs.currentIndex() == 1:
            self._set_status(self._placement_summary())

    # --------------------------------------------------- placement handlers

    def _on_placement_selection(self, payload: object) -> None:
        '''Sync placement selection state.'''
        selection, primary = cast(tuple[frozenset, int | None], payload)
        n_pieces = len(self._project.print_layout.pieces)
        self._pselection = {i for i in selection if 0 <= i < n_pieces}
        self._pprimary = primary
        self._placement_panel.sync_inspector(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            block_colors(self._project),
            self._project.blocks,
            self._pselection,
            self._pprimary,
        )

    def _on_placement_pieces_moved(self, moves: dict) -> None:
        '''Commit moved piece positions to the project.'''
        beds = self._project.print_layout.beds
        for index, (wx, wy) in moves.items():
            if not 0 <= index < len(self._project.print_layout.pieces):
                continue
            zone = zone_for_world_x(wx, beds)
            lx, ly = local_position(wx, wy, zone, beds)
            piece = self._project.print_layout.pieces[index]
            piece.bed = zone
            piece.position_x = lx
            piece.position_y = ly
        self._mark_dirty()
        self._refresh_placement_scene()
        self._placement_panel.sync_inspector(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            block_colors(self._project),
            self._project.blocks,
            self._pselection,
            self._pprimary,
        )
        if self._tabs.currentIndex() == 1:
            self._set_status(self._placement_summary())

    def _on_placement_beds_changed(self, beds: PrintBedSettings) -> None:
        '''Apply new bed settings to the project.'''
        self._project.print_layout.beds = beds
        # Pieces with bed >= new count -> staging.
        for piece in self._project.print_layout.pieces:
            if piece.bed >= 0 and piece.bed >= beds.count:
                piece.bed = PRINT_STAGING_BED
        self._placement_view.set_print_beds(beds)
        self._placement_panel.set_beds(beds)
        self._mark_dirty()
        self._sync_placement()

    def _on_placement_auto_place(self, indices: list[int]) -> None:
        '''Handle auto-place request.'''
        report = auto_place(self._project, indices)
        self._mark_dirty()
        self._refresh_placement_scene()
        self._placement_panel.sync_inspector(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            block_colors(self._project),
            self._project.blocks,
            self._pselection,
            self._pprimary,
        )
        if self._tabs.currentIndex() == 1:
            self._set_status(
                f"Auto-placed {report.placed} pieces ({report.staged} staged)"
            )

    def _on_placement_distribute(self, mode: str, indices: list[int]) -> None:
        '''Handle distribute request.'''
        report = distribute(self._project, mode, indices)
        self._mark_dirty()
        self._refresh_placement_scene()
        self._placement_panel.sync_inspector(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            block_colors(self._project),
            self._project.blocks,
            self._pselection,
            self._pprimary,
        )
        if self._tabs.currentIndex() == 1:
            self._set_status(
                f"Distributed {report.placed} pieces ({report.staged} staged)"
            )

    def _on_piece_bed_changed(self, bed: int) -> None:
        '''Move selected pieces to a different bed.'''
        if not self._pselection:
            return
        beds = self._project.print_layout.beds
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            piece = self._project.print_layout.pieces[idx]
            # Preserve world position: compute current zone, then local in target.
            wx, wy = world_position(piece, beds)
            lx, ly = local_position(wx, wy, bed, beds)
            piece.bed = bed
            piece.position_x = lx
            piece.position_y = ly
        self._mark_dirty()
        self._after_placement_change()

    def _on_piece_position_changed(self, x: float, y: float) -> None:
        '''Apply delta position to selected pieces.'''
        if not self._pselection or self._pprimary is None:
            return
        primary = self._project.print_layout.pieces[self._pprimary]
        dx = x - primary.position_x
        dy = y - primary.position_y
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            piece = self._project.print_layout.pieces[idx]
            piece.position_x += dx
            piece.position_y += dy
        self._mark_dirty()
        self._after_placement_change()

    def _on_piece_flip_changed(self, flipped: bool) -> None:
        '''Set flip state on all selected pieces.'''
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            self._project.print_layout.pieces[idx].flipped = flipped
        self._mark_dirty()
        self._after_placement_change()

    def _on_flip_selection(self) -> None:
        '''Toggle flip on all selected pieces.'''
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            p = self._project.print_layout.pieces[idx]
            p.flipped = not p.flipped
        self._mark_dirty()
        self._after_placement_change()

    def _on_stage_selection(self) -> None:
        '''Move selected pieces to staging.'''
        beds = self._project.print_layout.beds
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            piece = self._project.print_layout.pieces[idx]
            piece.bed = PRINT_STAGING_BED
            # Try to find a free spot; fall back to current coords.
            fps = block_footprints(self._project)
            w, d = piece_dims(piece, fps) if piece.block_uid in fps else (0.0, 0.0)
            existing: list[tuple[float, float, float, float]] = [
                (p.position_x, p.position_y, *piece_dims(p, fps))
                for p in self._project.print_layout.pieces
                if p.bed == PRINT_STAGING_BED and p.uid != piece.uid
            ]
            spot = find_free_spot(existing, w, d, beds)
            if spot is not None:
                piece.position_x, piece.position_y = spot
        self._mark_dirty()
        self._after_placement_change()


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
