'''
File:   app.py
Brief:  BrickGen PySide6 main window: project scene, parameters, template library.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterable
from copy import deepcopy
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
    AssembledPiece,
    BlockInstance,
    BlockLayout,
    ExportSettings,
    PrintBedSettings,
    PrintPiece,
    PrintToolsSettings,
    Project,
)
from brickgen.gui.assemble_panel import AssemblePanel
from brickgen.gui.assemble_view import AssembleView
from brickgen.gui.export_panel import ExportPanel
from brickgen.gui.grid_editor import CellGridEditor, CellState
from brickgen.gui.jobs import SKIPPED, JobError, JobQueue
from brickgen.gui.library import TemplateLibraryPanel
from brickgen.gui.parts_panel import PlannedPartsPanel
from brickgen.gui.placement_panel import PlacementPanel
from brickgen.gui.placement_view import PlacementView
from brickgen.gui.preferences import PreferencesDialog
from brickgen.gui.preview import (
    DISPLAY_MODES,
    MODE_SOLID,
    PreviewWidget,
    assemble_variant_mesh,
    mesh_from_arrays,
    piece_variant_mesh,
)
from brickgen.gui.start_dialog import StartDialog
from brickgen.services.app_settings import load_app_settings, save_app_settings
from brickgen.services.assemble_service import (
    Support,
    assemble_groups,
    assemble_summary,
    assemble_supports,
    base_footprint_rects,
    placed_counts,
    transform_footprint_rects,
)
from brickgen.services.placement_service import (
    auto_place,
    block_colors,
    block_footprints,
    clamp_into_zone,
    local_position,
    piece_dims,
    resolve_manual_moves,
    resolve_piece_placements,
    stage_pieces,
    sync_pieces,
    world_position,
)
from brickgen.services.print_export import collect_all_beds, collect_world_entries
from brickgen.services.project_io import load_project, save_project
from brickgen.services.template_library import Template

logger = logging.getLogger(__name__)

_PROJECT_FILETYPES = "BrickGen project (*.json);;All files (*.*)"
_SIDE_OFFSETS = {"+X": (1, 0), "-X": (-1, 0), "+Y": (0, 1), "-Y": (0, -1)}

# Debounce for automatic regeneration after any parameter change, ms.
AUTO_GENERATE_DELAY_MS = 250

# Same-label undo coalescing window, ms: continuous edits (spinboxes, grid
# tweaks) arriving closer than this share one undo step.
UNDO_BURST_MS = 800


class MainWindow(QMainWindow):
    '''Project scene in the middle, parameters on the left, template library
    on the right. File/View menus replace the old action buttons; the model
    regenerates automatically after any geometry change.
    '''

    def __init__(self, defer_start: bool = False) -> None:
        '''Build the layout and render the default one-block project.

        Args:
            defer_start: Real-app startup mode. The initial scene
                generation is paused until the startup modal resolves, so
                the empty viewport (grid only) stays behind it; see
                ``maybe_show_start_dialog``. Tests construct with the
                default, which generates immediately.
        '''
        super().__init__()
        self._app_settings = load_app_settings()
        # Boot guard: the initial placement sync adds the default pieces and
        # would flag the untouched project dirty (unsaved-changes prompt on
        # close, [*] in the title) with zero user edits.
        self._booting = True
        self._generation_paused = defer_start
        self._project = Project(blocks=[BlockInstance()])
        self._project_path: Path | None = None
        self._dirty = False
        self._selection: set[int] = {0}
        self._primary: int | None = 0
        self._loading_layout = False
        self._syncing_inspector = False
        self._syncing_ui = False
        self._inspector_pos: tuple[float, float] = (0.0, 0.0)

        # Shared edit history (Edit menu: Ctrl+Z / Ctrl+Y). Snapshot-based:
        # every mutating handler calls ``_begin_edit`` FIRST, which stores a
        # deep copy of the pre-edit project. Continuous sources (spinboxes,
        # grid edits) coalesce same-label edits inside the burst window into
        # one undo step.
        self._undo_stack: list[Project] = []
        self._redo_stack: list[Project] = []
        self._edit_burst: str | None = None
        self._burst_timer = QTimer(self)
        self._burst_timer.setSingleShot(True)
        self._burst_timer.setInterval(UNDO_BURST_MS)
        self._burst_timer.timeout.connect(self._close_edit_burst)

        # Edit-menu clipboards (block dumps for the planning tab, piece
        # dumps for the placement tab, assembled-piece dumps for the
        # assemble tab) and the cross-select toggle.
        self._clip_blocks: list[dict] = []
        self._clip_pieces: list[dict] = []
        self._clip_assembled: list[dict] = []
        self._cross_select = False
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

        # Assemble tab state.
        self._aselection: set[int] = set()
        self._aprimary: int | None = None
        # Assemble orientation mesh variants keyed (block index, rot_z,
        # flipped): every assembled piece shares one of the eight per-block
        # variants.
        self._assemble_variants: dict[tuple[int, int, bool], pv.PolyData] = {}

        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Sidebar stack: page 0 = planning sidebar, page 1 = assemble panel,
        # page 2 = placement panel.
        self._sidebar_stack = QStackedWidget()
        self._sidebar = self._build_sidebar()
        self._sidebar_stack.addWidget(self._sidebar)
        self._assemble_panel = AssemblePanel()
        self._sidebar_stack.addWidget(self._assemble_panel)
        self._placement_panel = PlacementPanel()
        self._sidebar_stack.addWidget(self._placement_panel)

        # Tab widget: page 0 = planning preview, page 1 = assemble view,
        # page 2 = placement view.
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_preview_pane(), "Planning")
        self._assemble_view = AssembleView()
        self._tabs.addTab(self._assemble_view, "Assemble")
        self._placement_view = PlacementView()
        self._tabs.addTab(self._placement_view, "Print placement")

        splitter.addWidget(self._sidebar_stack)
        splitter.addWidget(self._tabs)
        # Right panel stack: page 0 = template library (planning), page 1 =
        # planned parts (assemble), page 2 = export panel (placement).
        self._library = TemplateLibraryPanel(self._selected_block_layout, self._jobs)
        self._library.apply_requested.connect(self._on_template_selected)
        self._parts_panel = PlannedPartsPanel(self._jobs)
        self._parts_panel.place_requested.connect(self._on_parts_place_requested)
        self._export_panel = ExportPanel()
        self._export_panel.export_requested.connect(self._on_export_print)
        self._export_panel.export_settings_changed.connect(
            self._on_export_settings_changed
        )
        self._right_stack = QStackedWidget()
        self._right_stack.addWidget(self._library)
        self._right_stack.addWidget(self._parts_panel)
        self._right_stack.addWidget(self._export_panel)
        splitter.addWidget(self._right_stack)
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
        if not self._generation_paused:
            self._on_generate()

        # Placement view init.
        self._placement_view.set_print_beds(self._project.print_layout.beds)
        self._placement_panel.set_beds(self._project.print_layout.beds)
        self._placement_panel.set_tools(self._project.print_layout.tools)
        # Assemble view init: world footprints of the placed pieces come
        # from the pure service (supports for stacking/side snapping).
        self._assemble_view.set_supports_provider(self._assemble_supports_excluding)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._booting = False
        self._mark_clean()  # fresh start: an untouched project is clean

    # ------------------------------------------------------------------ tabs

    def _active_preview(self) -> PreviewWidget:
        '''Return the currently active preview widget.'''
        if self._tabs.currentIndex() == 0:
            return self._preview
        if self._tabs.currentIndex() == 1:
            return self._assemble_view
        return self._placement_view

    def _on_tab_changed(self, index: int) -> None:
        '''Switch the sidebar page, right panel and view-menu state.'''
        # Leaving the assemble tab drops its modal placement mode (the
        # ghost grabs the mouse — it must not survive the tab switch).
        if index != 1:
            self._assemble_view.cancel_placement()
        self._sidebar_stack.setCurrentIndex(index)
        self._right_stack.setCurrentIndex(index)

        # Enable/disable quantity actions only on planning tab; the Del
        # shortcut must never remove PLANNING blocks from another tab.
        self._act_qty_labels.setEnabled(index == 0)
        self._act_remove_blocks.setEnabled(index == 0 and bool(self._selection))

        # Refresh "Show grid" checked state from the active preview.
        self._refresh_grid_action_state()

        # When entering the planning tab: push shared view settings.
        if index == 0:
            self._preview.set_show_gizmo(self._project.view.show_gizmo)

        # When entering the assemble tab: push the view settings and sync.
        if index == 1:
            self._assemble_view.set_grid_unit(self._project.settings.unit_size)
            self._assemble_view.set_grid_scale(self._project.view.grid_scale)
            self._assemble_view.set_snap_enabled(
                self._project.assemble.view.snap_to_grid
            )
            self._assemble_view.set_snap_parts(
                self._project.assemble.view.snap_to_parts
            )
            self._assemble_view.set_grid_enabled(
                self._project.assemble.view.grid_enabled
            )
            self._assemble_panel.set_snapping(
                self._project.assemble.view.snap_to_grid,
                self._project.assemble.view.snap_to_parts,
            )
            self._assemble_view.set_show_gizmo(self._project.view.show_gizmo)
            self._sync_assemble()

        # When entering the placement tab: push shared settings into the
        # placement view.
        elif index == 2:
            # Cross-select mode: the pieces of every block selected on the
            # planning tab become the placement selection.
            if self._cross_select:
                self._apply_cross_selection()
            self._placement_view.set_grid_unit(self._project.settings.unit_size)
            self._placement_view.set_grid_scale(self._project.view.grid_scale)
            self._placement_view.set_snap_enabled(self._project.view.grid_snap)
            self._placement_view.set_grid_enabled(
                self._project.view.placement_grid_enabled
            )
            self._placement_view.set_print_beds(self._project.print_layout.beds)
            self._placement_view.set_show_gizmo(self._project.view.show_gizmo)
            self._sync_placement()

    def _refresh_grid_action_state(self) -> None:
        '''Update the "Show grid" action checked state from the active preview.

        The check change is signal-blocked: a programmatic sync must not
        re-enter ``_on_grid_show`` (which would treat it as a user edit and
        dirty the project on a mere tab switch).
        '''
        index = self._tabs.currentIndex()
        if index == 0:
            checked = self._project.view.grid_enabled
        elif index == 1:
            checked = self._project.assemble.view.grid_enabled
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
        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()
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

        # --- Edit menu: shared undo/redo + clipboard commands for BOTH tabs,
        # plus the cross-select toggle. The QActions are window-level
        # shortcuts; line edits (spinboxes) keep their own Ctrl+A/C/V/Z/Y
        # text handling via Qt's ShortcutOverride.
        edit_menu = self.menuBar().addMenu("&Edit")
        self._act_undo = QAction("&Undo", self)
        self._act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._act_undo.setToolTip("Undo the last edit on the active tab (Ctrl+Z).")
        self._act_undo.setEnabled(False)
        self._act_undo.triggered.connect(self._on_undo)
        edit_menu.addAction(self._act_undo)
        self._act_redo = QAction("&Redo", self)
        self._act_redo.setShortcuts(
            [QKeySequence("Ctrl+Y"), QKeySequence(QKeySequence.StandardKey.Redo)]
        )
        self._act_redo.setToolTip("Redo the last undone edit (Ctrl+Y).")
        self._act_redo.setEnabled(False)
        self._act_redo.triggered.connect(self._on_redo)
        edit_menu.addAction(self._act_redo)
        edit_menu.addSeparator()
        self._act_copy = QAction("&Copy", self)
        self._act_copy.setShortcut(QKeySequence.StandardKey.Copy)
        self._act_copy.setToolTip(
            "Copy the selected blocks (planning), assembled copies (assemble) "
            "or pieces (placement) (Ctrl+C)."
        )
        self._act_copy.triggered.connect(self._on_copy)
        edit_menu.addAction(self._act_copy)
        self._act_paste = QAction("&Paste", self)
        self._act_paste.setShortcut(QKeySequence.StandardKey.Paste)
        self._act_paste.setToolTip(
            "Paste the copied blocks (planning), assembled copies (assemble) "
            "or pieces (placement) (Ctrl+V)."
        )
        self._act_paste.triggered.connect(self._on_paste)
        edit_menu.addAction(self._act_paste)
        edit_menu.addSeparator()
        self._act_select_all = QAction("Select &All", self)
        self._act_select_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        self._act_select_all.setToolTip(
            "Select every block (planning) or piece (assemble/placement) (Ctrl+A)."
        )
        self._act_select_all.triggered.connect(self._on_select_all)
        edit_menu.addAction(self._act_select_all)
        edit_menu.addSeparator()
        self._act_cross_select = QAction("&Cross-select mode", self)
        self._act_cross_select.setCheckable(True)
        self._act_cross_select.setToolTip(
            "When switching to Print placement with blocks selected on "
            "Planning, select all placement pieces of those blocks."
        )
        self._act_cross_select.toggled.connect(self._on_cross_select_toggled)
        edit_menu.addAction(self._act_cross_select)

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

        self._act_show_gizmo = QAction("Show move &gizmo", self)
        self._act_show_gizmo.setCheckable(True)
        self._act_show_gizmo.setChecked(self._project.view.show_gizmo)
        self._act_show_gizmo.setToolTip(
            "Arrow cross to drag the selection along the world axes "
            "(X/Y/Z keys constrain a move)."
        )
        self._act_show_gizmo.toggled.connect(self._on_view_gizmo)
        view_menu.addAction(self._act_show_gizmo)

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
        self._block_quantity.setRange(0, 100000)
        self._block_quantity.setToolTip(
            "How many physical copies this block represents (BOM tracking); "
            "0 keeps the block planned without any printed copy."
        )
        form.addRow("Quantity", self._block_quantity)

        self._cb_qty_linked = QCheckBox("Linked to Assemble count")
        self._cb_qty_linked.setToolTip(
            "Derive the quantity from the copies of this block currently\n"
            "placed on the Assemble tab (placing/deleting there updates it).\n"
            "Unchecked keeps the manual BOM quantity above."
        )
        self._cb_qty_linked.toggled.connect(self._on_qty_link_toggled)
        form.addRow(self._cb_qty_linked)

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
        # The stack hosts both sidebar pages; the splitter slot must fit the
        # wider of the two or the hidden page's controls get clipped.
        sidebar_w = max(
            self._sidebar.minimumWidth(),
            self._placement_panel.minimumWidth(),
        )
        self._sidebar_stack.setMinimumWidth(sidebar_w)
        # Same for the right-side stack (library / planned parts / export).
        right_w = max(
            self._library.minimumWidth(),
            self._parts_panel.minimumWidth(),
            self._export_panel.minimumWidth(),
        )
        preview_w = max(self._splitter.width() - sidebar_w - right_w, 200)
        self._splitter.setSizes([sidebar_w, preview_w, right_w])
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
        self._placement_panel.tools_changed.connect(self._on_placement_tools_changed)
        self._placement_panel.piece_bed_changed.connect(self._on_piece_bed_changed)
        self._placement_panel.piece_position_changed.connect(
            self._on_piece_position_changed
        )
        self._placement_panel.piece_flip_changed.connect(self._on_piece_flip_changed)
        self._placement_panel.stage_selection_requested.connect(
            self._on_stage_selection
        )

        # Assemble signals.
        self._assemble_view.selection_changed.connect(self._on_assemble_selection)
        self._assemble_view.blocks_moved.connect(self._on_assemble_pieces_moved)
        self._assemble_view.piece_placed.connect(self._on_assemble_piece_placed)
        self._assemble_view.delete_requested.connect(self._on_assemble_delete)
        self._assemble_panel.snap_grid_changed.connect(self._on_assemble_snap_grid)
        self._assemble_panel.snap_parts_changed.connect(self._on_assemble_snap_parts)
        self._assemble_panel.piece_position_changed.connect(
            self._on_assemble_position_changed
        )
        self._assemble_panel.piece_rotation_changed.connect(
            self._on_assemble_rotation_changed
        )
        self._assemble_panel.piece_flip_changed.connect(self._on_assemble_flip_changed)
        self._assemble_panel.delete_requested.connect(self._on_assemble_delete)

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
        '''A shared geometry control changed: mark dirty and regenerate.

        Programmatic syncs (open/new/restore) are guarded by
        ``_syncing_ui``: they must not create undo steps or dirty flags.
        '''
        if self._syncing_ui:
            return
        self._begin_edit("shared-params", coalesce=True)
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
        # Reset assemble state (scene rebuilds on tab enter / generation).
        self._aselection = set()
        self._aprimary = None
        self._assemble_variants = {}
        self._assemble_view.cancel_placement()
        self._mark_clean()
        self._reset_edit_history()
        self._syncing_ui = True
        try:
            self._apply_view_settings()
            self._sync_all()
            self._placement_view.set_print_beds(self._project.print_layout.beds)
            self._placement_panel.set_beds(self._project.print_layout.beds)
            self._placement_panel.set_tools(self._project.print_layout.tools)
        finally:
            self._syncing_ui = False
        self._on_generate()
        self._set_status("New project created.")

    def _on_open_project(self) -> bool:
        '''Load a project file into the window and regenerate the scene.

        Returns:
            True when a project was loaded; False when the action was
            cancelled or declined.
        '''
        if not self._confirm_discard():
            return False
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", self._default_project_dir(), _PROJECT_FILETYPES
        )
        if not path:
            return False
        return self._load_project_path(path)

    def _load_project_path(self, path: str) -> bool:
        '''Load *path* into the window and regenerate the scene.

        Shared by the File > Open dialog, the File > Open Recent submenu and
        the startup dialog's recent list.

        Returns:
            True when the project was loaded successfully.
        '''
        try:
            project = load_project(Path(path))
        except BrickGenError as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return False
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
        # Reset assemble state (scene rebuilds on tab enter / generation).
        self._aselection = set()
        self._aprimary = None
        self._assemble_variants = {}
        self._assemble_view.cancel_placement()
        self._mark_clean()
        self._reset_edit_history()
        # Restore the project's own view state (grids, snap, labels).
        self._syncing_ui = True
        try:
            self._apply_view_settings()
            self._sync_all()
            self._placement_view.set_print_beds(self._project.print_layout.beds)
            self._placement_panel.set_beds(self._project.print_layout.beds)
            self._placement_panel.set_tools(self._project.print_layout.tools)
        finally:
            self._syncing_ui = False
        self._on_generate()
        self._add_recent_project(Path(path))
        self._set_status(f"Project opened: {Path(path).name}")
        return True

    def _add_recent_project(self, path: Path) -> None:
        '''Record *path* as the newest recent project and persist settings.'''
        self._app_settings.recent_projects = [
            str(path),
            *self._app_settings.recent_projects,
        ]
        self._save_app_settings()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        '''Repopulate the File > Open Recent submenu from the settings.'''
        menu = self._recent_menu
        menu.clear()
        entries = self._app_settings.recent_projects
        for entry in entries:
            act = menu.addAction(Path(entry).name)
            act.setData(entry)
            act.setEnabled(Path(entry).is_file())
            act.triggered.connect(
                lambda checked=False, p=entry: self._on_open_recent(p)
            )
        menu.addSeparator()
        act_clear = menu.addAction("Clear Recent List")
        act_clear.setEnabled(bool(entries))
        act_clear.triggered.connect(self._on_clear_recent)

    def _on_open_recent(self, path: str) -> None:
        '''Open a project picked from the File > Open Recent submenu.'''
        if not self._confirm_discard():
            return
        self._load_project_path(path)

    def _on_clear_recent(self) -> None:
        '''Empty the recent project list.'''
        self._app_settings.recent_projects = []
        self._save_app_settings()
        self._rebuild_recent_menu()

    def maybe_show_start_dialog(self) -> None:
        '''Show the startup modal (called from ``main()`` after the window
        shows).

        Scene generation is paused until the modal resolves, so the
        viewport behind it stays empty (grid only). Every exit builds a
        scene: the chosen/recent project, or the default 1x1 project for
        "new", Esc and cancelled open dialogs.
        '''
        dialog = StartDialog(self._app_settings.recent_projects, self)
        dialog.exec()
        self._generation_paused = False
        loaded = False
        if dialog.choice == "open":
            loaded = self._on_open_project()
        elif dialog.choice == "recent" and dialog.recent_path:
            if self._confirm_discard():
                loaded = self._load_project_path(dialog.recent_path)
        if not loaded:
            self._on_generate()
            self._set_status("New project created.")

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
        self._add_recent_project(Path(target))
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

    # ----------------------------------------------------------- undo/redo
    def _begin_edit(self, label: str, coalesce: bool = False) -> None:
        '''Snapshot the project state before a user edit (undo boundary).

        MUST be called at the START of every mutating handler: the snapshot
        captures the pre-edit state. Continuous sources (spinboxes, grid
        edits) pass ``coalesce=True`` — same-label edits inside the burst
        window share one undo step; discrete commands (add/remove/
        auto-place/...) never coalesce.

        Args:
            label: Edit kind used for burst coalescing.
            coalesce: True to merge same-label edits inside the burst window.
        '''
        if self._syncing_ui:
            return
        if coalesce and self._edit_burst == label:
            self._burst_timer.start()
            return
        self._close_edit_burst()
        self._push_undo_snapshot()
        self._edit_burst = label
        self._burst_timer.start()

    def _close_edit_burst(self) -> None:
        '''End the coalescing window: the next edit opens a new undo step.'''
        self._edit_burst = None
        self._burst_timer.stop()

    def _push_undo_snapshot(self) -> None:
        '''Store the current project on the undo stack (trimmed by depth).'''
        self._undo_stack.append(deepcopy(self._project))
        self._trim_undo()
        self._redo_stack.clear()
        self._update_undo_actions()

    def _trim_undo(self) -> None:
        '''Cap the undo stack at the configured depth (app settings).'''
        depth = self._app_settings.undo_depth
        if len(self._undo_stack) > depth:
            del self._undo_stack[: len(self._undo_stack) - depth]

    def _update_undo_actions(self) -> None:
        '''Enable the Edit-menu undo/redo actions from the stack states.'''
        self._act_undo.setEnabled(bool(self._undo_stack))
        self._act_redo.setEnabled(bool(self._redo_stack))

    def _reset_edit_history(self) -> None:
        '''Clear the undo/redo stacks (new/open boundaries).'''
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._close_edit_burst()
        self._update_undo_actions()

    def _on_undo(self) -> None:
        '''Restore the pre-edit snapshot of the last undo step.'''
        if not self._undo_stack:
            return
        self._close_edit_burst()
        snapshot = self._undo_stack.pop()
        self._redo_stack.append(deepcopy(self._project))
        self._restore_project(snapshot)
        self._update_undo_actions()
        self._set_status("Undo.")

    def _on_redo(self) -> None:
        '''Restore the post-edit snapshot of the last undone step.'''
        if not self._redo_stack:
            return
        self._close_edit_burst()
        snapshot = self._redo_stack.pop()
        self._undo_stack.append(deepcopy(self._project))
        self._trim_undo()
        self._restore_project(snapshot)
        self._update_undo_actions()
        self._set_status("Redo.")

    def _restore_project(self, project: Project) -> None:
        '''Replace the whole project and re-sync every widget (undo/redo).

        Selections are reset like on project open; the placement meshes are
        rebuilt by the generation triggered at the end.
        '''
        self._project = project
        self._applied_context = None
        self._mark_scene_dirty()
        count = len(self._project.blocks)
        self._selection = {0} if count else set()
        self._primary = 0 if count else None
        self._pselection = set()
        self._pprimary = None
        self._block_meshes = []
        self._mesh_variants = {}
        self._aselection = set()
        self._aprimary = None
        self._assemble_variants = {}
        self._assemble_view.cancel_placement()
        self._syncing_ui = True
        try:
            self._apply_view_settings()
            self._sync_all()
            self._placement_view.set_print_beds(self._project.print_layout.beds)
            self._placement_panel.set_beds(self._project.print_layout.beds)
            self._placement_panel.set_tools(self._project.print_layout.tools)
        finally:
            self._syncing_ui = False
        self._on_generate()
        self._mark_dirty()

    # ------------------------------------------------------------- clipboard
    def _on_copy(self) -> None:
        '''Copy the active tab's selection into its per-tab clipboard.'''
        index = self._tabs.currentIndex()
        if index == 0:
            if not self._selection:
                self._set_status("Nothing to copy: no blocks selected.")
                return
            self._clip_blocks = [
                deepcopy(self._project.blocks[index].model_dump())
                for index in sorted(self._selection)
            ]
            self._set_status(f"Copied {len(self._clip_blocks)} block(s).")
        elif index == 1:
            if not self._aselection:
                self._set_status("Nothing to copy: no pieces selected.")
                return
            assembled = self._project.assemble.pieces
            self._clip_assembled = [
                deepcopy(assembled[index].model_dump())
                for index in sorted(self._aselection)
                if index < len(assembled)
            ]
            self._set_status(f"Copied {len(self._clip_assembled)} copy(ies).")
        else:
            pieces = self._project.print_layout.pieces
            if not self._pselection:
                self._set_status("Nothing to copy: no pieces selected.")
                return
            self._clip_pieces = [
                deepcopy(pieces[index].model_dump())
                for index in sorted(self._pselection)
                if index < len(pieces)
            ]
            self._set_status(f"Copied {len(self._clip_pieces)} piece(s).")

    def _on_paste(self) -> None:
        '''Paste the active tab's clipboard: blocks on planning, pieces on
        the placement tab.'''
        index = self._tabs.currentIndex()
        if index == 0:
            self._paste_blocks()
        elif index == 1:
            self._paste_assembled()
        else:
            self._paste_pieces()

    def _paste_blocks(self) -> None:
        '''Paste the copied block definitions as new blocks (new uids,
        auto-placed next to the scene bounds).'''
        if not self._clip_blocks:
            self._set_status("Clipboard is empty: copy blocks first (Ctrl+C).")
            return
        self._begin_edit("paste-blocks")
        pasted: list[int] = []
        for values in self._clip_blocks:
            block = BlockInstance.model_validate({**values, "uid": uuid.uuid4().hex})
            merged = block.to_settings(self._project.settings)
            position = self._auto_place_position(
                merged.outer_length_x / 2, merged.outer_length_y / 2
            )
            block = block.model_copy(
                update={"position_x": position[0], "position_y": position[1]}
            )
            self._project.blocks.append(block)
            pasted.append(len(self._project.blocks) - 1)
        self._selection = set(pasted)
        self._primary = pasted[-1]
        self._preview.set_selection(self._selection, self._primary)
        self._on_scene_selection((frozenset(self._selection), self._primary))
        self._mark_scene_dirty()
        self._mark_dirty()
        self._sync_all()
        self._on_generate()
        self._set_status(f"Pasted {len(pasted)} block(s).")

    def _paste_pieces(self) -> None:
        '''Paste the copied pieces as new physical copies.

        Every pasted piece increments its block's BOM quantity by one (the
        placement scene always mirrors "one piece per block x quantity");
        the new piece starts near the copied one and is shifted to the
        nearest conflict-free spot by the no-overlap repair.
        '''
        if not self._clip_pieces:
            self._set_status("Clipboard is empty: copy pieces first (Ctrl+C).")
            return
        pieces = self._project.print_layout.pieces
        blocks_by_uid = {block.uid: block for block in self._project.blocks}
        if not any(
            str(values.get("block_uid", "")) in blocks_by_uid
            for values in self._clip_pieces
        ):
            self._set_status("Nothing to paste: copied blocks no longer exist.")
            return
        self._begin_edit("paste-pieces")
        beds = self._project.print_layout.beds
        pasted = 0
        linked_skipped = 0
        for values in self._clip_pieces:
            block = blocks_by_uid.get(str(values.get("block_uid", "")))
            if block is None:
                continue  # The source block was removed since the copy.
            if block.quantity_linked:
                # The count of a linked block belongs to the Assemble tab:
                # extra placement pieces would be dropped by the next sync.
                linked_skipped += 1
                continue
            block.quantity += 1
            bed = int(values.get("bed", PRINT_STAGING_BED))
            if not (PRINT_STAGING_BED <= bed < beds.count):
                bed = PRINT_STAGING_BED
            piece = PrintPiece.model_validate(
                {
                    **values,
                    "uid": uuid.uuid4().hex,
                    "bed": bed,
                    "position_x": float(values.get("position_x", 0.0)) + 10.0,
                    "position_y": float(values.get("position_y", 0.0)),
                }
            )
            pieces.append(piece)
            pasted += 1
        if not pasted:
            if linked_skipped:
                self._set_status(
                    "Nothing pasted: the copied blocks are linked to the "
                    "Assemble tab (place copies there instead)."
                )
            return
        self._sync_placement()
        self._mark_dirty()
        message = f"Pasted {pasted} piece(s) (+{pasted} BOM quantity)."
        if linked_skipped:
            message += f" {linked_skipped} skipped (linked to the Assemble count)."
        self._set_status(message)

    def _paste_assembled(self) -> None:
        '''Paste the copied assembled copies as a cursor ghost.

        The first copied piece attaches to the viewport ghost with its
        orientation and height (exactly like a parts-panel placement): every
        left click drops one copy, Esc/right click cancels — the ghost stays
        until explicitly cancelled. Pieces land through the normal placement
        handler, so linked block quantities follow on every drop.
        '''
        if not self._clip_assembled:
            self._set_status("Clipboard is empty: copy pieces first (Ctrl+C).")
            return
        blocks_by_uid = {block.uid: block for block in self._project.blocks}
        values = next(
            (
                entry
                for entry in self._clip_assembled
                if str(entry.get("block_uid", "")) in blocks_by_uid
            ),
            None,
        )
        if values is None:
            self._set_status("Nothing to paste: copied blocks no longer exist.")
            return
        block_uid = str(values.get("block_uid", ""))
        index = next(
            i for i, block in enumerate(self._project.blocks) if block.uid == block_uid
        )
        mesh = self._block_meshes[index] if index < len(self._block_meshes) else None
        if mesh is None or mesh.n_points == 0:
            self._set_status(
                "Part mesh is not ready yet - wait for the generation.",
                error=True,
            )
            return
        settings = self._project.block_settings(index)
        self._assemble_view.start_placement(
            mesh,
            index,
            base_footprint_rects(settings),
            settings.total_height,
            self._project.blocks[index].color,
            settings.block_height,
            rot_z=int(values.get("rot_z", 0)) % 4,
            flipped=bool(values.get("flipped", False)),
            z=float(values.get("position_z", 0.0)),
        )
        extra = len(self._clip_assembled) - 1
        message = (
            "Placing pasted copy: click drops it, R/Space rotates, F flips, "
            "wheel changes height, Esc/RMB finishes."
        )
        if extra:
            message += f" {extra} more copied piece(s) stay on the clipboard."
        self._set_status(message)

    # ------------------------------------------------------------ select all
    def _on_select_all(self) -> None:
        '''Select every block (planning) or every piece (assemble/placement).'''
        index = self._tabs.currentIndex()
        if index == 0:
            count = len(self._project.blocks)
            if not count:
                return
            self._selection = set(range(count))
            self._primary = 0
            self._preview.set_selection(self._selection, self._primary)
            self._on_scene_selection((frozenset(self._selection), self._primary))
        elif index == 1:
            count = len(self._project.assemble.pieces)
            if not count:
                return
            self._aselection = set(range(count))
            self._aprimary = 0
            self._assemble_view.set_selection(self._aselection, self._aprimary)
            self._on_assemble_selection((frozenset(self._aselection), self._aprimary))
        else:
            count = len(self._project.print_layout.pieces)
            if not count:
                return
            self._pselection = set(range(count))
            self._pprimary = 0
            self._placement_view.set_selection(self._pselection, self._pprimary)
            self._on_placement_selection((frozenset(self._pselection), self._pprimary))

    # ----------------------------------------------------------- cross-select
    def _on_cross_select_toggled(self, checked: bool) -> None:
        '''Store the cross-select toggle (session-level, not persisted).'''
        self._cross_select = bool(checked)

    def _apply_cross_selection(self) -> None:
        '''Select every placement piece whose block is selected on planning.

        The primary piece is the first piece of the primary block (falls
        back to the lowest selected piece index).
        '''
        block_uids = {
            self._project.blocks[index].uid
            for index in self._selection
            if 0 <= index < len(self._project.blocks)
        }
        pieces = self._project.print_layout.pieces
        self._pselection = {
            index for index, piece in enumerate(pieces) if piece.block_uid in block_uids
        }
        primary_uid = (
            self._project.blocks[self._primary].uid
            if self._primary is not None
            and 0 <= self._primary < len(self._project.blocks)
            else None
        )
        self._pprimary = next(
            (
                index
                for index, piece in enumerate(pieces)
                if piece.block_uid == primary_uid
            ),
            min(self._pselection) if self._pselection else None,
        )
        self._placement_view.set_selection(self._pselection, self._pprimary)

    # -------------------------------------------------------- block actions
    def _on_add_block(self) -> None:
        '''Append a default block to the scene and select it.'''
        self._begin_edit("add-block")
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
        self._begin_edit("duplicate-block")
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
        self._begin_edit("remove-blocks")
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
            # The Del shortcut must never remove PLANNING blocks while a
            # non-planning tab is active (it deletes assemble/placement
            # pieces there through the view's own Del handling).
            self._act_remove_blocks.setEnabled(
                self._tabs.currentIndex() == 0 and bool(selection)
            )
            if not selection:
                self._primary = None
                self._block_group.setEnabled(False)
                self._block_group.setTitle("Block (select in the scene)")
                self._layout_group.setEnabled(False)
                self._cb_qty_linked.setChecked(False)
                self._block_quantity.setEnabled(True)
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
            self._cb_qty_linked.setChecked(block.quantity_linked)
            # Linked blocks derive their quantity from the assemble count:
            # the spin is read-only while the checkbox is set.
            self._block_quantity.setEnabled(not block.quantity_linked)
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
        self._begin_edit("block-cells", coalesce=True)
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
        self._begin_edit("block-color")
        hex_color = color.name()
        changed: dict[int, str] = {}
        for index in sorted(self._selection):
            self._project.blocks[index].color = hex_color
            changed[index] = hex_color
        self._preview.set_block_colors(changed)
        self._sync_block_inspector()
        self._mark_dirty()

    def _on_block_quantity_changed(self) -> None:
        '''Apply the edited BOM quantity to every selected block.

        Linked blocks are skipped: their quantity follows the Assemble
        count and a manual edit would be overwritten immediately.
        '''
        if self._syncing_inspector or self._primary is None:
            return
        quantity = self._block_quantity.value()
        targets = [
            index
            for index in sorted(self._selection)
            if not self._project.blocks[index].quantity_linked
        ]
        if not targets:
            return
        self._begin_edit("block-quantity", coalesce=True)
        changed: dict[int, int] = {}
        for index in targets:
            self._project.blocks[index].quantity = quantity
            changed[index] = quantity
        self._preview.set_block_quantities(changed)
        self._sync_blocks_list()
        self._mark_dirty()

    def _on_qty_link_toggled(self, checked: bool) -> None:
        '''Toggle the assemble-count link for every selected block.

        Checking the box derives the quantity from the currently placed
        copies immediately (so the placement scene stays consistent);
        unchecking keeps the current value as the manual BOM quantity.
        '''
        if self._syncing_inspector or self._primary is None:
            return
        self._begin_edit("quantity-link")
        for index in sorted(self._selection):
            self._project.blocks[index].quantity_linked = bool(checked)
        self._sync_block_inspector()
        self._mark_dirty()
        if checked:
            # Snap the linked quantities to the current assemble counts;
            # a quantity change resyncs the placement pieces too.
            self._apply_linked_quantities(sorted(self._selection))

    def _apply_linked_quantities(self, indices: list[int]) -> None:
        '''Set the quantity of every linked block to its assemble count.

        Args:
            indices: Block indices to process (linked ones only change).

        Returns:
            None. Marks the project dirty and resyncs the planning
            quantity labels / placement pieces when a value changed.
        '''
        counts = placed_counts(self._project.assemble.pieces)
        changed: dict[int, int] = {}
        for index in indices:
            block = self._project.blocks[index]
            if not block.quantity_linked:
                continue
            placed = counts.get(block.uid, 0)
            if block.quantity != placed:
                block.quantity = placed
                changed[index] = placed
        if not changed:
            return
        self._preview.set_block_quantities(changed)
        self._sync_blocks_list()
        self._sync_block_inspector()
        self._mark_dirty()
        self._sync_placement()

    def _on_block_tubes_toggled(self, checked: bool) -> None:
        '''Apply the block-level inner-tubes toggle to every selected block.'''
        if self._syncing_inspector or self._primary is None:
            return
        self._begin_edit("block-tubes")
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
        self._begin_edit("block-position", coalesce=True)
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
        valid_moves = {
            index: (x, y)
            for index, (x, y) in positions.items()
            if 0 <= index < len(self._project.blocks)
        }
        if not any(
            self._project.blocks[index].position_x != x
            or self._project.blocks[index].position_y != y
            for index, (x, y) in valid_moves.items()
        ):
            return
        self._begin_edit("scene-move")
        for index, (x, y) in valid_moves.items():
            block = self._project.blocks[index]
            block.position_x = x
            block.position_y = y
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
        self._begin_edit("layout-edit", coalesce=True)
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
        self._export_panel.set_settings(self._project.export)
        # The planned-parts panel mirrors the block list (structure only;
        # the assemble scene itself rebuilds on tab enter / generation).
        self._prune_assemble_pieces()
        self._sync_assemble_panel()
        self._update_title()

    def _sync_shared_form(self) -> None:
        '''Push shared settings values into the form widgets.'''
        settings = self._project.settings
        for field, spin in self._floats.items():
            spin.setValue(getattr(settings, field))
        self._update_derived_label()

    def _mark_dirty(self) -> None:
        '''Flag the project as modified (window title marker).

        Ignored while ``_booting`` is set: programmatic updates during
        construction must not count as user edits.
        '''
        if self._booting:
            return
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
        self._assemble_view.set_grid_scale(view.grid_scale)
        self._assemble_view.set_snap_enabled(self._project.assemble.view.snap_to_grid)
        self._assemble_view.set_snap_parts(self._project.assemble.view.snap_to_parts)
        self._assemble_view.set_grid_enabled(self._project.assemble.view.grid_enabled)
        self._assemble_panel.set_snapping(
            self._project.assemble.view.snap_to_grid,
            self._project.assemble.view.snap_to_parts,
        )
        self._placement_view.set_grid_scale(view.grid_scale)
        self._placement_view.set_snap_enabled(view.grid_snap)
        self._placement_view.set_grid_enabled(view.placement_grid_enabled)
        self._preview.set_show_gizmo(view.show_gizmo)
        self._placement_view.set_show_gizmo(view.show_gizmo)
        self._assemble_view.set_show_gizmo(view.show_gizmo)
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
            self._act_show_gizmo,
            *self._act_grid_scale.values(),
            *self._act_qty_positions.values(),
        ]
        for action in actions:
            action.blockSignals(True)
        try:
            self._act_show_grid.setChecked(view.grid_enabled)
            self._act_snap.setChecked(view.grid_snap)
            self._act_qty_labels.setChecked(view.quantity_labels_enabled)
            self._act_show_gizmo.setChecked(view.show_gizmo)
            for scale, action in self._act_grid_scale.items():
                action.setChecked(scale == view.grid_scale)
            for position, action in self._act_qty_positions.items():
                action.setChecked(position == view.quantity_label_position)
        finally:
            for action in actions:
                action.blockSignals(False)

    def _on_grid_show(self, checked: bool) -> None:
        '''Toggle the floor grid on the active preview (stored per project).'''
        index = self._tabs.currentIndex()
        if index == 0:
            self._project.view.grid_enabled = checked
        elif index == 1:
            self._project.assemble.view.grid_enabled = checked
        else:
            self._project.view.placement_grid_enabled = checked
        self._mark_dirty()
        self._active_preview().set_grid_enabled(checked)

    def _on_view_gizmo(self, checked: bool) -> None:
        '''Toggle the move gizmo shared flag (stored in project view).'''
        self._project.view.show_gizmo = checked
        self._mark_dirty()
        self._active_preview().set_show_gizmo(checked)

    def _on_grid_scale(self, scale: float) -> None:
        '''Set the grid cell scale factor (stored per project, all previews).'''
        self._project.view.grid_scale = scale
        self._mark_dirty()
        self._preview.set_grid_scale(scale)
        self._assemble_view.set_grid_scale(scale)
        self._placement_view.set_grid_scale(scale)

    def _on_grid_snap(self, checked: bool) -> None:
        '''Toggle snap-to-grid movement (stored per project, per viewport).'''
        index = self._tabs.currentIndex()
        if index == 1:
            # The assemble tab keeps its own snap preference.
            self._project.assemble.view.snap_to_grid = checked
            self._assemble_view.set_snap_enabled(checked)
        else:
            self._project.view.grid_snap = checked
            self._preview.set_snap_enabled(checked)
            self._placement_view.set_snap_enabled(checked)
        self._mark_dirty()

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
        self._trim_undo()
        self._update_undo_actions()
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
        self._begin_edit("template")
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
        if self._generation_paused:
            # Start-screen pause: the viewport stays empty (grid only)
            # until the startup modal resolves.
            return
        try:
            shared = self._collect_shared_settings()
        except BrickGenError as exc:
            self._set_status(f"Invalid settings: {exc}", error=True)
            return
        self._project.settings = shared
        # The grid cells and the snap step follow the unit size.
        self._preview.set_grid_unit(shared.unit_size)
        self._assemble_view.set_grid_unit(shared.unit_size)
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

        # Store resolved meshes for the placement/assemble tabs; reset the
        # orientation caches.
        self._block_meshes = list(meshes)
        self._mesh_variants = {}
        self._assemble_variants = {}
        self._sync_placement()
        self._sync_assemble()

        self._set_status(self._scene_summary())

    # ------------------------------------------------------------- export
    def _on_export_print(self) -> None:
        '''Ask for a destination folder, then queue the layout export using
        the project's export settings (placement export panel): one
        compound file per print bed, or every bed combined into a single
        file at world positions.

        Only pieces fully inside their bed are exported (a piece hanging
        over the edge would not print); skipped pieces are reported in the
        status afterwards.
        '''
        cfg = self._project.export
        bed_payloads: list[dict] = []
        world_entries: list[dict] = []
        if cfg.per_bed_files:
            bed_payloads, _skipped = collect_all_beds(self._project)
            if not bed_payloads:
                QMessageBox.information(
                    self, "Nothing to export", "Place pieces on a bed first."
                )
                return
        else:
            world_entries, _skipped = collect_world_entries(self._project)
            if not world_entries:
                QMessageBox.information(
                    self, "Nothing to export", "Place pieces on a bed first."
                )
                return
        folder = QFileDialog.getExistingDirectory(
            self, "Select export folder", self._default_project_dir()
        )
        if not folder:
            return
        stem = self._project_path.stem if self._project_path else "brickgen-layout"
        ext = cfg.format.lower()
        payload: dict[str, Any]
        if cfg.per_bed_files:
            payload_beds = [
                {
                    'index': bed['index'],
                    'path': str(
                        Path(folder) / f"{stem}_bed{bed['index'] + 1:02d}.{ext}"
                    ),
                    'pieces': bed['pieces'],
                }
                for bed in bed_payloads
            ]
            payload = {
                'beds': payload_beds,
                'format': cfg.format,
                'tolerance': cfg.tolerance,
                'angular': self._app_settings.export_angular_tolerance,
            }
            self._set_status(f"Exporting {len(payload_beds)} bed file(s)…")
        else:
            payload = {
                'beds': [
                    {
                        'index': 0,
                        'path': str(Path(folder) / f"{stem}.{ext}"),
                        'pieces': world_entries,
                    }
                ],
                'format': cfg.format,
                'tolerance': cfg.tolerance,
                'angular': self._app_settings.export_angular_tolerance,
            }
            self._set_status("Exporting single file…")

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
        index = self._tabs.currentIndex()
        if index == 0:
            self._status.setText(text)
        elif index == 1:
            self._assemble_panel.set_status(text, error)
        else:
            self._placement_panel.set_status(text, error)
            self._export_panel.set_status(text, error)

    # ------------------------------------------------------- placement sync

    def _enforce_no_overlap(self) -> None:
        '''Repair any overlapping piece pair before the scene renders.

        Belt-and-braces after packing: auto_place is fuzz-verified
        overlap-free, but this guarantees the no-overlap
        invariant for EVERY path that reaches the viewport — a piece that
        overlaps another in its zone shifts to the nearest free spot (or
        staging when full) exactly like a repaired manual move.
        '''
        pieces = self._project.print_layout.pieces
        before = [(p.bed, p.position_x, p.position_y) for p in pieces]
        resolve_piece_placements(
            pieces,
            block_footprints(self._project),
            self._project.print_layout.beds,
            list(pieces),
        )
        after = [(p.bed, p.position_x, p.position_y) for p in pieces]
        if before != after:
            self._mark_dirty()

    def _sync_placement(self) -> None:
        '''Reconcile pieces with blocks and rebuild the placement scene.'''
        report = sync_pieces(self._project)
        if report.added or report.removed:
            self._mark_dirty()
        self._enforce_no_overlap()
        self._refresh_placement_scene()
        self._placement_panel.sync_inspector(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            block_colors(self._project),
            self._project.blocks,
            self._pselection,
            self._pprimary,
        )
        if self._tabs.currentIndex() == 2:
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
                    variant = piece_variant_mesh(
                        base_mesh,
                        piece.rotated90,
                        piece.flipped,
                        heights.get(piece.block_uid, 0),
                    )
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
        if self._tabs.currentIndex() == 2:
            self._set_status(self._placement_summary())

    # --------------------------------------------------- placement handlers

    def _on_placement_selection(self, payload: object) -> None:
        '''Sync placement selection state.

        A rubber-band (frame) selection carries no primary — no piece was
        clicked; the lowest selected index becomes the primary so the
        inspector stays active on framed selections.
        '''
        selection, primary = cast(tuple[frozenset, int | None], payload)
        n_pieces = len(self._project.print_layout.pieces)
        self._pselection = {i for i in selection if 0 <= i < n_pieces}
        if primary is None and self._pselection:
            primary = min(self._pselection)
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
        '''Commit moved piece positions to the project (collision-safe).

        Drops past a zone edge are clamped into the zone; drops onto other
        pieces shift to the nearest conflict-free spot instead.
        '''
        self._begin_edit("piece-move")
        resolve_manual_moves(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            self._project.print_layout.beds,
            moves,
        )
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
        if self._tabs.currentIndex() == 2:
            self._set_status(self._placement_summary())

    def _on_placement_beds_changed(self, beds: PrintBedSettings) -> None:
        '''Apply new bed settings to the project.

        A gap-only change never touches the pieces: the gap only feeds
        packing and collision checks, so every existing position stays
        valid.  For width/depth/count changes, pieces referencing removed
        beds or sticking out of the resized bed rectangle are moved to
        staging (packed into the free staging space), so no piece ever
        stays out of bounds after a bed resize.
        '''
        self._begin_edit("bed-settings", coalesce=True)
        old = self._project.print_layout.beds
        self._project.print_layout.beds = beds
        if (beds.width, beds.depth, beds.count) == (old.width, old.depth, old.count):
            # Gap-only change: keep every piece exactly where it is.
            self._placement_view.set_print_beds(beds)
            self._placement_panel.set_beds(beds)
            self._mark_dirty()
            return
        footprints = block_footprints(self._project)
        outside: list[PrintPiece] = []
        for piece in self._project.print_layout.pieces:
            if piece.bed == PRINT_STAGING_BED:
                continue
            if piece.bed >= beds.count:
                outside.append(piece)
                continue
            if piece.block_uid not in footprints:
                continue
            w, d = piece_dims(piece, footprints)
            if (
                piece.position_x - w / 2 < beds.gap - 1e-6
                or piece.position_x + w / 2 > beds.width - beds.gap + 1e-6
                or piece.position_y - d / 2 < beds.gap - 1e-6
                or piece.position_y + d / 2 > beds.depth - beds.gap + 1e-6
            ):
                outside.append(piece)
        if outside:
            stage_pieces(self._project.print_layout.pieces, footprints, beds, outside)
        self._placement_view.set_print_beds(beds)
        self._placement_panel.set_beds(beds)
        # Clamp the persisted auto-place range into the new bed count and
        # re-sync the tools widgets from the normalized values.
        tools = self._project.print_layout.tools
        auto_from = max(1, min(tools.auto_from, beds.count))
        auto_to = max(auto_from, min(tools.auto_to, beds.count))
        if (auto_from, auto_to) != (tools.auto_from, tools.auto_to):
            self._project.print_layout.tools = tools.model_copy(
                update={"auto_from": auto_from, "auto_to": auto_to}
            )
        self._placement_panel.set_tools(self._project.print_layout.tools)
        self._mark_dirty()
        self._sync_placement()

    def _on_placement_auto_place(
        self,
        indices: list[int],
        by_color: bool = True,
        even: bool = True,
    ) -> None:
        '''Handle auto-place request (modifiers from the panel checkboxes).'''
        self._begin_edit("auto-place")
        report = auto_place(self._project, indices, by_color=by_color, even=even)
        self._enforce_no_overlap()
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
        if self._tabs.currentIndex() == 2:
            self._set_status(
                f"Auto-placed {report.placed} pieces ({report.staged} staged)"
            )

    def _on_piece_bed_changed(self, bed: int) -> None:
        '''Move selected pieces to a different bed (collision-safe).'''
        if not self._pselection:
            return
        self._begin_edit("piece-bed", coalesce=True)
        beds = self._project.print_layout.beds
        fps = block_footprints(self._project)
        moved: list[PrintPiece] = []
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            piece = self._project.print_layout.pieces[idx]
            # Preserve world position: compute current zone, then local in target.
            wx, wy = world_position(piece, beds)
            piece.bed = bed
            piece.position_x, piece.position_y = local_position(wx, wy, bed, beds)
            clamp_into_zone(piece, fps, beds)
            moved.append(piece)
        resolve_piece_placements(self._project.print_layout.pieces, fps, beds, moved)
        self._mark_dirty()
        self._after_placement_change()

    def _on_piece_position_changed(self, x: float, y: float) -> None:
        '''Apply delta position to selected pieces (collision-safe).'''
        if not self._pselection or self._pprimary is None:
            return
        self._begin_edit("piece-position", coalesce=True)
        beds = self._project.print_layout.beds
        fps = block_footprints(self._project)
        primary = self._project.print_layout.pieces[self._pprimary]
        dx = x - primary.position_x
        dy = y - primary.position_y
        moved: list[PrintPiece] = []
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            piece = self._project.print_layout.pieces[idx]
            piece.position_x += dx
            piece.position_y += dy
            clamp_into_zone(piece, fps, beds)
            moved.append(piece)
        resolve_piece_placements(self._project.print_layout.pieces, fps, beds, moved)
        self._mark_dirty()
        self._after_placement_change()

    def _on_piece_flip_changed(self, flipped: bool) -> None:
        '''Set flip state on all selected pieces.'''
        if not self._pselection:
            return
        self._begin_edit("piece-flip")
        for idx in sorted(self._pselection):
            if idx >= len(self._project.print_layout.pieces):
                continue
            self._project.print_layout.pieces[idx].flipped = flipped
        self._mark_dirty()
        self._after_placement_change()

    def _on_placement_tools_changed(self, tools: PrintToolsSettings) -> None:
        '''Store the placement tools state in the project (persisted JSON).'''
        self._begin_edit("tools", coalesce=True)
        self._project.print_layout.tools = tools
        self._mark_dirty()

    def _on_export_settings_changed(self, settings: ExportSettings) -> None:
        '''Store the placement export settings in the project (persisted JSON).'''
        self._begin_edit("export-settings", coalesce=True)
        self._project.export = settings
        self._mark_dirty()

    def _on_stage_selection(self) -> None:
        '''Move selected pieces to staging (packed into the free space).'''
        if not self._pselection:
            return
        self._begin_edit("stage-pieces")
        pieces = [
            self._project.print_layout.pieces[idx]
            for idx in sorted(self._pselection)
            if idx < len(self._project.print_layout.pieces)
        ]
        stage_pieces(
            self._project.print_layout.pieces,
            block_footprints(self._project),
            self._project.print_layout.beds,
            pieces,
        )
        self._mark_dirty()
        self._after_placement_change()

    # ------------------------------------------------------- assemble sync

    def _assemble_supports_excluding(self, exclude_uids: set[str]) -> list[Support]:
        '''World footprints of the placed pieces for the assemble viewport.

        Args:
            exclude_uids: Piece uids to skip (the pieces being dragged).

        Returns:
            The remaining pieces' footprints (pure service call).
        '''
        return assemble_supports(self._project, exclude_uids)

    def _prune_assemble_pieces(self) -> None:
        '''Drop assembled pieces whose block no longer exists.'''
        uids = {block.uid for block in self._project.blocks}
        pieces = self._project.assemble.pieces
        kept = [piece for piece in pieces if piece.block_uid in uids]
        if len(kept) != len(pieces):
            self._project.assemble.pieces = kept
            self._mark_dirty()

    def _assemble_block_labels(self) -> dict[str, str]:
        '''Block uid -> "CxM" display label for the assemble inspector.'''
        return {
            block.uid: f"{block.cells_x}x{block.cells_y}"
            for block in self._project.blocks
        }

    def _sync_assemble_panel(self) -> None:
        '''Refresh the planned-parts cards from the current block list.'''
        groups = assemble_groups(self._project)
        counts = placed_counts(self._project.assemble.pieces)
        self._parts_panel.refresh(groups, counts)

    def _refresh_assemble_scene(self) -> None:
        '''Rebuild the assemble viewport actors from cached block meshes.'''
        uid_to_idx = {block.uid: i for i, block in enumerate(self._project.blocks)}
        heights = {
            block.uid: self._project.block_settings(i).total_height
            for i, block in enumerate(self._project.blocks)
        }
        block_heights = {
            block.uid: self._project.block_settings(i).block_height
            for i, block in enumerate(self._project.blocks)
        }
        dims = {
            block.uid: (
                self._project.block_settings(i).outer_length_x / 2,
                self._project.block_settings(i).outer_length_y / 2,
            )
            for i, block in enumerate(self._project.blocks)
        }
        # Base-orientation material rectangles per block uid (excluded
        # cells leave voids pieces may drop into).
        rects_map = {
            block.uid: base_footprint_rects(self._project.block_settings(i))
            for i, block in enumerate(self._project.blocks)
        }
        colors_map = {block.uid: block.color for block in self._project.blocks}

        meshes_out: list[pv.PolyData | None] = []
        colors_out: list[str] = []
        meta_out: list[dict] = []
        for piece in self._project.assemble.pieces:
            blk_idx = uid_to_idx.get(piece.block_uid)
            if blk_idx is None or blk_idx >= len(self._block_meshes):
                continue
            base_mesh = self._block_meshes[blk_idx]
            if base_mesh is None:
                continue
            mesh: pv.PolyData = base_mesh
            if piece.rot_z or piece.flipped:
                # Lazy variant cache: one transformed mesh per orientation.
                key = (blk_idx, piece.rot_z, piece.flipped)
                variant = self._assemble_variants.get(key)
                if variant is None:
                    variant = assemble_variant_mesh(
                        base_mesh,
                        piece.rot_z,
                        piece.flipped,
                        heights.get(piece.block_uid, 0),
                    )
                    self._assemble_variants[key] = variant
                mesh = variant
            half_x, half_y = dims.get(piece.block_uid, (0.0, 0.0))
            if piece.rot_z % 2:
                half_x, half_y = half_y, half_x
            base_rects = rects_map.get(piece.block_uid, ())
            meta_out.append(
                {
                    'pos_x': piece.position_x,
                    'pos_y': piece.position_y,
                    'pos_z': piece.position_z,
                    'half_x': half_x,
                    'half_y': half_y,
                    'base_rects': base_rects,
                    'rects': transform_footprint_rects(
                        base_rects, piece.rot_z, piece.flipped
                    ),
                    'top_z': heights.get(piece.block_uid, 0),
                    'block_h': block_heights.get(piece.block_uid, 0),
                    'quantity': 1,
                    'uid': piece.uid,
                    'rot_z': piece.rot_z,
                    'flipped': piece.flipped,
                }
            )
            colors_out.append(colors_map.get(piece.block_uid, "#3f7fbf"))
            meshes_out.append(mesh)

        self._assemble_view.update_scene(meshes_out, colors_out, meta_out)

        # Prune the selection to the current piece count.
        n_pieces = len(self._project.assemble.pieces)
        self._aselection = {i for i in self._aselection if i < n_pieces}
        if self._aprimary is not None and self._aprimary >= n_pieces:
            self._aprimary = None

    def _sync_assemble(self) -> None:
        '''Prune stale pieces and rebuild the assemble scene + panel.'''
        self._prune_assemble_pieces()
        self._refresh_assemble_scene()
        self._sync_assemble_panel()
        self._assemble_panel.sync_inspector(
            self._project.assemble.pieces,
            self._aselection,
            self._aprimary,
            self._assemble_block_labels(),
        )
        if self._tabs.currentIndex() == 1:
            self._set_status(assemble_summary(self._project))

    # ---------------------------------------------------- assemble handlers

    def _on_assemble_selection(self, payload: object) -> None:
        '''Sync the assemble selection state into the window + inspector.

        A rubber-band (frame) selection carries no primary — the lowest
        selected index becomes the primary (mirrors the placement tab).
        '''
        selection, primary = cast(tuple[frozenset, int | None], payload)
        n_pieces = len(self._project.assemble.pieces)
        self._aselection = {i for i in selection if 0 <= i < n_pieces}
        if primary is None and self._aselection:
            primary = min(self._aselection)
        self._aprimary = primary
        self._assemble_panel.sync_inspector(
            self._project.assemble.pieces,
            self._aselection,
            self._aprimary,
            self._assemble_block_labels(),
        )

    def _on_assemble_pieces_moved(self, moves: dict) -> None:
        '''Commit moved assembled pieces into the project.

        Args:
            moves: Mapping of piece index to the final ``(x, y, z, rot_z)``
                state (Space mid-drag may have rotated the pieces).
        '''
        pieces = self._project.assemble.pieces
        valid = {
            index: (x, y, z, rot_z)
            for index, (x, y, z, rot_z) in moves.items()
            if 0 <= index < len(pieces)
        }
        if not any(
            (
                pieces[index].position_x,
                pieces[index].position_y,
                pieces[index].position_z,
                pieces[index].rot_z,
            )
            != (x, y, z, rot_z % 4)
            for index, (x, y, z, rot_z) in valid.items()
        ):
            return
        self._begin_edit("assemble-move")
        for index, (x, y, z, rot_z) in valid.items():
            piece = pieces[index]
            piece.position_x = x
            piece.position_y = y
            piece.position_z = z
            piece.rot_z = int(rot_z) % 4
        self._mark_dirty()
        self._assemble_panel.sync_inspector(
            pieces,
            self._aselection,
            self._aprimary,
            self._assemble_block_labels(),
        )
        if self._tabs.currentIndex() == 1:
            self._set_status(assemble_summary(self._project))

    def _on_assemble_piece_placed(self, info: dict) -> None:
        '''Drop a ghost copy from the parts panel onto the assembly.

        Args:
            info: Payload of :data:`AssembleView.piece_placed` with keys
                ``block`` (block index), ``x``, ``y``, ``z``, ``rot_z``
                and ``flipped``.
        '''
        block_index = int(info.get('block', -1))
        if not 0 <= block_index < len(self._project.blocks):
            return
        self._begin_edit("assemble-place")
        self._project.assemble.pieces.append(
            AssembledPiece(
                block_uid=self._project.blocks[block_index].uid,
                position_x=float(info['x']),
                position_y=float(info['y']),
                position_z=float(info['z']),
                rot_z=int(info.get('rot_z', 0)),
                flipped=bool(info.get('flipped', False)),
            )
        )
        self._mark_dirty()
        self._sync_assemble()
        # A linked block's quantity follows the new count (the placement
        # scene resyncs inside _apply_linked_quantities).
        self._apply_linked_quantities([block_index])
        self._set_status(assemble_summary(self._project))

    def _linked_block_indices(self, uids: set[str]) -> list[int]:
        '''Indices of the linked blocks among those referenced by *uids*.'''
        return [
            index
            for index, block in enumerate(self._project.blocks)
            if block.uid in uids and block.quantity_linked
        ]

    def _on_assemble_delete(self) -> None:
        '''Delete the selected assembled pieces (Del hotkey / panel button).

        Only the placed copies are removed: the planned part (template)
        stays in the panel with a placed count of 0, and linked block
        quantities follow the new counts.
        '''
        if not self._aselection:
            return
        pieces = self._project.assemble.pieces
        removed_uids = {
            pieces[index].block_uid for index in self._aselection if index < len(pieces)
        }
        self._begin_edit("assemble-delete")
        self._project.assemble.pieces = [
            piece for index, piece in enumerate(pieces) if index not in self._aselection
        ]
        self._aselection = set()
        self._aprimary = None
        self._assemble_view.set_selection(set(), None)
        self._mark_dirty()
        self._sync_assemble()
        self._apply_linked_quantities(self._linked_block_indices(removed_uids))
        self._set_status(assemble_summary(self._project))

    def _on_assemble_position_changed(self, x: float, y: float, z: float) -> None:
        '''Apply the inspector position to the selected pieces.

        The primary piece moves to the absolute coordinates; with a
        multi-selection the whole group shifts by the same delta.
        '''
        if not self._aselection or self._aprimary is None:
            return
        pieces = self._project.assemble.pieces
        if self._aprimary >= len(pieces):
            return
        self._begin_edit("assemble-position", coalesce=True)
        primary = pieces[self._aprimary]
        dx = x - primary.position_x
        dy = y - primary.position_y
        dz = z - primary.position_z
        for index in sorted(self._aselection):
            if index >= len(pieces):
                continue
            piece = pieces[index]
            piece.position_x += dx
            piece.position_y += dy
            piece.position_z += dz
        self._mark_dirty()
        self._refresh_assemble_scene()
        self._assemble_panel.sync_inspector(
            pieces,
            self._aselection,
            self._aprimary,
            self._assemble_block_labels(),
        )

    def _on_assemble_rotation_changed(self, rot_z: int) -> None:
        '''Set the 90-degree Z rotation on all selected pieces.'''
        if not self._aselection:
            return
        self._begin_edit("assemble-rotation", coalesce=True)
        pieces = self._project.assemble.pieces
        for index in sorted(self._aselection):
            if index >= len(pieces):
                continue
            pieces[index].rot_z = int(rot_z) % 4
        self._mark_dirty()
        self._refresh_assemble_scene()
        self._assemble_panel.sync_inspector(
            pieces,
            self._aselection,
            self._aprimary,
            self._assemble_block_labels(),
        )

    def _on_assemble_flip_changed(self, flipped: bool) -> None:
        '''Set the 180-degree flip on all selected pieces.'''
        if not self._aselection:
            return
        self._begin_edit("assemble-flip")
        pieces = self._project.assemble.pieces
        for index in sorted(self._aselection):
            if index >= len(pieces):
                continue
            pieces[index].flipped = bool(flipped)
        self._mark_dirty()
        self._refresh_assemble_scene()
        self._assemble_panel.sync_inspector(
            pieces,
            self._aselection,
            self._aprimary,
            self._assemble_block_labels(),
        )

    def _on_assemble_snap_grid(self, checked: bool) -> None:
        '''Store the assemble snap-to-grid preference (per project).'''
        self._project.assemble.view.snap_to_grid = bool(checked)
        self._assemble_view.set_snap_enabled(bool(checked))
        self._mark_dirty()

    def _on_assemble_snap_parts(self, checked: bool) -> None:
        '''Store the assemble snap-to-parts preference (per project).'''
        self._project.assemble.view.snap_to_parts = bool(checked)
        self._assemble_view.set_snap_parts(bool(checked))
        self._mark_dirty()

    def _on_parts_place_requested(self, block_uid: str) -> None:
        '''Attach a ghost copy of the requested planned part to the cursor.

        Args:
            block_uid: uid of the block (color variant) to place.
        '''
        index = next(
            (
                i
                for i, block in enumerate(self._project.blocks)
                if block.uid == block_uid
            ),
            None,
        )
        if index is None:
            return
        mesh = self._block_meshes[index] if index < len(self._block_meshes) else None
        if mesh is None or mesh.n_points == 0:
            self._set_status(
                "Part mesh is not ready yet - wait for the generation.",
                error=True,
            )
            return
        settings = self._project.block_settings(index)
        self._assemble_view.start_placement(
            mesh,
            index,
            base_footprint_rects(settings),
            settings.total_height,
            self._project.blocks[index].color,
            settings.block_height,
        )
        self._set_status(
            "Placing: click drops a copy, R/Space rotates, F flips, "
            "wheel changes height, Esc/RMB finishes."
        )


def main() -> None:
    '''Application entry point: configure logging and run the Qt GUI loop.'''
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    window = MainWindow(defer_start=True)
    window.showMaximized()
    # Startup modal over the empty grid scene; singleShot so the main loop
    # is running when the dialog opens. Scene generation starts once the
    # modal resolves (see maybe_show_start_dialog).
    QTimer.singleShot(0, window.maybe_show_start_dialog)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
