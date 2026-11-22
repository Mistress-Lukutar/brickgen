'''
File:   placement_panel.py
Brief:  Left-sidebar panel for the print-placement tab: bed configuration,
        placement tools and the selected-piece inspector.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.19.0
'''

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from brickgen.core.project import (
    PRINT_STAGING_BED,
    PrintBedSettings,
    PrintPiece,
    PrintToolsSettings,
)


class _FlipCheckBox(QCheckBox):
    '''"Flip 180\u00b0" checkbox with tri-state display and click-to-all
    semantics.

    The state mirrors the whole selection (checked when every selected
    piece is flipped, unchecked when none is, partially checked when the
    selection mixes both), but a user click never lands on the partial
    state: a click on a checked or partially-checked box clears the flip
    on all selected pieces, a click on an unchecked box flips them all.
    '''

    def nextCheckState(self) -> None:  # noqa: N802 - Qt naming
        '''Click/keyboard toggle: unchecked -> checked, otherwise unchecked.'''
        if self.checkState() == Qt.CheckState.Unchecked:
            self.setCheckState(Qt.CheckState.Checked)
        else:
            self.setCheckState(Qt.CheckState.Unchecked)


class PlacementPanel(QWidget):
    '''Left-sidebar panel for the print-placement tab: bed configuration,
    placement tools and the selected-piece inspector.'''

    beds_changed = Signal(object)  # PrintBedSettings
    # list[int] 0-based bed indices, "Distribute by color", "Distribute evenly"
    auto_place_requested = Signal(object, bool, bool)
    tools_changed = Signal(object)  # PrintToolsSettings
    piece_bed_changed = Signal(int)  # target bed index or PRINT_STAGING_BED
    piece_position_changed = Signal(float, float)  # delta for primary's zone-local pos
    piece_flip_changed = Signal(bool)  # flip state for ALL selected pieces
    stage_selection_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the panel with all groups inside a scroll area.'''
        super().__init__(parent)
        self._syncing = False
        self._build_ui()

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

        # --- Print beds group ---
        self._beds_group = QGroupBox("Print beds (project)")
        beds_form = QFormLayout(self._beds_group)
        beds_form.setVerticalSpacing(8)

        self._spin_bed_width = QDoubleSpinBox()
        self._spin_bed_width.setRange(10.0, 2000.0)
        self._spin_bed_width.setDecimals(1)
        self._spin_bed_width.setSingleStep(1.0)
        self._spin_bed_width.setSuffix(" mm")
        self._spin_bed_width.valueChanged.connect(self._on_beds_changed)

        self._spin_bed_depth = QDoubleSpinBox()
        self._spin_bed_depth.setRange(10.0, 2000.0)
        self._spin_bed_depth.setDecimals(1)
        self._spin_bed_depth.setSingleStep(1.0)
        self._spin_bed_depth.setSuffix(" mm")
        self._spin_bed_depth.valueChanged.connect(self._on_beds_changed)

        self._spin_bed_count = QSpinBox()
        self._spin_bed_count.setRange(1, 64)
        self._spin_bed_count.setSingleStep(1)
        self._spin_bed_count.valueChanged.connect(self._on_beds_changed)

        self._spin_piece_gap = QDoubleSpinBox()
        self._spin_piece_gap.setRange(0.0, 100.0)
        self._spin_piece_gap.setDecimals(1)
        self._spin_piece_gap.setSingleStep(0.5)
        self._spin_piece_gap.setSuffix(" mm")
        self._spin_piece_gap.valueChanged.connect(self._on_beds_changed)

        beds_form.addRow("Bed width, mm", self._spin_bed_width)
        beds_form.addRow("Bed depth, mm", self._spin_bed_depth)
        beds_form.addRow("Bed count", self._spin_bed_count)
        beds_form.addRow("Piece gap, mm", self._spin_piece_gap)

        inner.addWidget(self._beds_group)

        # --- Tools group ---
        self._tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout(self._tools_group)
        tools_layout.setSpacing(8)

        from_to_row = QHBoxLayout()
        self._spin_from = QSpinBox()
        self._spin_from.setRange(1, 64)
        self._spin_from.setSingleStep(1)
        self._spin_from.valueChanged.connect(self._on_from_changed)
        from_to_row.addWidget(QLabel("Beds from:"))
        from_to_row.addWidget(self._spin_from)

        self._spin_to = QSpinBox()
        self._spin_to.setRange(1, 64)
        self._spin_to.setSingleStep(1)
        self._spin_to.valueChanged.connect(self._on_to_changed)
        from_to_row.addWidget(QLabel("Beds to:"))
        from_to_row.addWidget(self._spin_to)

        tools_layout.addLayout(from_to_row)

        # Auto-place modifiers (not separate operations): the checkboxes
        # shape how the single Auto-place button packs the bed range.
        self._cb_by_color = QCheckBox("Distribute by color")
        self._cb_by_color.setChecked(True)
        self._cb_by_color.setToolTip(
            "Auto-place never mixes piece colors on one bed (one print job per color)."
        )
        self._cb_by_color.toggled.connect(self._on_tool_flag_changed)
        tools_layout.addWidget(self._cb_by_color)

        self._cb_even = QCheckBox("Distribute evenly")
        self._cb_even.setChecked(True)
        self._cb_even.setToolTip(
            "Auto-place spreads pieces across all selected beds instead of "
            "filling one bed before moving to the next."
        )
        self._cb_even.toggled.connect(self._on_tool_flag_changed)
        tools_layout.addWidget(self._cb_even)

        # Full-width stacked buttons: a horizontal row of three would force
        # a panel width far beyond the sidebar slot and clip on the right.
        self._btn_auto_place = QPushButton("Auto-place (selected beds)")
        self._btn_auto_place.clicked.connect(self._on_auto_place)
        tools_layout.addWidget(self._btn_auto_place)

        inner.addWidget(self._tools_group)

        # --- Selected piece inspector group ---
        self._inspector_group = QGroupBox("Selected piece")
        self._inspector_group.setEnabled(False)
        insp_form = QFormLayout(self._inspector_group)
        insp_form.setVerticalSpacing(8)

        self._lbl_inspector_summary = QLabel("No piece selected")
        self._lbl_inspector_summary.setWordWrap(True)
        insp_form.addRow(self._lbl_inspector_summary)

        self._spin_piece_bed = QSpinBox()
        self._spin_piece_bed.setRange(0, 1)
        self._spin_piece_bed.setSpecialValueText("Staging")
        self._spin_piece_bed.valueChanged.connect(self._on_piece_bed_changed)
        insp_form.addRow("Bed", self._spin_piece_bed)

        self._spin_piece_x = QDoubleSpinBox()
        self._spin_piece_x.setRange(-100000.0, 100000.0)
        self._spin_piece_x.setDecimals(2)
        self._spin_piece_x.setSingleStep(1.0)
        self._spin_piece_x.setSuffix(" mm")
        self._spin_piece_x.valueChanged.connect(self._on_piece_position_changed)
        insp_form.addRow("X, mm", self._spin_piece_x)

        self._spin_piece_y = QDoubleSpinBox()
        self._spin_piece_y.setRange(-100000.0, 100000.0)
        self._spin_piece_y.setDecimals(2)
        self._spin_piece_y.setSingleStep(1.0)
        self._spin_piece_y.setSuffix(" mm")
        self._spin_piece_y.valueChanged.connect(self._on_piece_position_changed)
        insp_form.addRow("Y, mm", self._spin_piece_y)

        self._cb_flip = _FlipCheckBox("Flip 180\u00b0 (pins down)")
        self._cb_flip.checkStateChanged.connect(self._on_piece_flip_state_changed)
        insp_form.addRow(self._cb_flip)

        # Full-width stacked buttons (same reason as the tool buttons).
        self._btn_stage_sel = QPushButton("Move to staging")
        self._btn_stage_sel.clicked.connect(self._on_stage_selection)
        insp_form.addRow(self._btn_stage_sel)

        inner.addWidget(self._inspector_group)

        # --- Status label ---
        self._status_label = QLabel("Ready.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #0a6;")
        inner.addWidget(self._status_label)
        inner.addStretch(1)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # The horizontal scrollbar is off, so anything narrower than the
        # form's natural width would clip it: pin the panel to at least
        # that width plus a scrollbar (mirrors the planning sidebar).
        form_width = (
            container.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
        )
        self.setMinimumWidth(form_width)
        scroll.setMinimumWidth(form_width)

    # --- Bed spinbox handlers ---

    def _on_beds_changed(self) -> None:
        '''Any bed control changed: emit the updated PrintBedSettings.'''
        if self._syncing:
            return
        beds = PrintBedSettings(
            width=self._spin_bed_width.value(),
            depth=self._spin_bed_depth.value(),
            count=self._spin_bed_count.value(),
            gap=self._spin_piece_gap.value(),
        )
        self.beds_changed.emit(beds)

    def _on_from_changed(self) -> None:
        '''User edit of "Beds from": clamp the range and emit the tools
        state (programmatic sync never clamps or emits).'''
        if self._syncing:
            return
        to_val = self._spin_to.value()
        from_val = self._spin_from.value()
        if from_val > to_val:
            self._spin_from.setValue(to_val)
        elif from_val < 1:
            self._spin_from.setValue(1)
        self._emit_tools()

    def _on_to_changed(self) -> None:
        '''User edit of "Beds to": clamp the range and emit the tools
        state (programmatic sync never clamps or emits).'''
        if self._syncing:
            return
        to_val = self._spin_to.value()
        from_val = self._spin_from.value()
        if to_val < from_val:
            self._spin_to.setValue(from_val)
        elif to_val > self._spin_bed_count.value():
            self._spin_to.setValue(self._spin_bed_count.value())
        self._emit_tools()

    def _on_tool_flag_changed(self, _checked: bool) -> None:
        '''A distribute modifier checkbox changed: emit the tools state.'''
        self._emit_tools()

    def _emit_tools(self) -> None:
        '''Emit the current tools state (suppressed during programmatic sync).'''
        if self._syncing:
            return
        self.tools_changed.emit(
            PrintToolsSettings(
                auto_from=self._spin_from.value(),
                auto_to=self._spin_to.value(),
                distribute_by_color=self._cb_by_color.isChecked(),
                distribute_evenly=self._cb_even.isChecked(),
            )
        )

    def _on_auto_place(self) -> None:
        '''Emit auto-place request for the selected bed range.

        The spinboxes display 1-based bed numbers (as the viewport labels
        do); the signal carries 0-based bed indices plus the states of the
        two modifier checkboxes ("Distribute by color", "Distribute evenly").
        '''
        beds = list(range(self._spin_from.value() - 1, self._spin_to.value()))
        self.auto_place_requested.emit(
            beds, self._cb_by_color.isChecked(), self._cb_even.isChecked()
        )

    # --- Piece inspector handlers ---

    def _on_piece_bed_changed(self, value: int) -> None:
        '''Bed spinbox changed: emit the target bed index.'''
        if self._syncing:
            return
        # 0 -> staging, v>=1 -> v-1 (0-based bed index)
        if value == 0:
            self.piece_bed_changed.emit(PRINT_STAGING_BED)
        else:
            self.piece_bed_changed.emit(value - 1)

    def _on_piece_position_changed(self, _value: float) -> None:
        '''X or Y spinbox changed: emit the current values.'''
        if self._syncing:
            return
        x = self._spin_piece_x.value()
        y = self._spin_piece_y.value()
        self.piece_position_changed.emit(x, y)

    def _on_piece_flip_state_changed(self, state: Qt.CheckState) -> None:
        '''Flip checkbox state changed by the user: apply to ALL selected.

        ``nextCheckState`` guarantees the user-driven state is fully
        checked or unchecked, never partial.
        '''
        if self._syncing:
            return
        self.piece_flip_changed.emit(state == Qt.CheckState.Checked)

    def _on_stage_selection(self) -> None:
        '''Move to staging button clicked.'''
        self.stage_selection_requested.emit()

    # --- Public sync methods ---

    def set_beds(self, beds: PrintBedSettings) -> None:
        '''Sync the bed spinboxes (gap, size, count) and the from/to ranges
        from the project.

        The piece-gap spin MUST be synced here too: without it a loaded
        project's gap never reaches the widget, and the next bed edit
        rebuilds ``PrintBedSettings`` from the stale widget value —
        silently resetting the project's persisted gap.
        '''
        self._syncing = True
        try:
            self._spin_bed_width.setValue(beds.width)
            self._spin_bed_depth.setValue(beds.depth)
            self._spin_piece_gap.setValue(beds.gap)
            self._spin_bed_count.setRange(1, 64)
            self._spin_bed_count.setValue(beds.count)
            # Clamp the bed count value if it exceeded the new max.
            if self._spin_bed_count.value() > beds.count:
                self._spin_bed_count.setValue(beds.count)

            # Update from/to spinbox ranges and clamp values.
            self._spin_from.setRange(1, beds.count)
            self._spin_to.setRange(1, beds.count)
            if self._spin_from.value() > beds.count:
                self._spin_from.setValue(beds.count)
            if self._spin_to.value() > beds.count:
                self._spin_to.setValue(beds.count)

            # Update piece bed spin max (count+1 for staging option).
            self._spin_piece_bed.setRange(0, beds.count)

        finally:
            self._syncing = False

    def set_tools(self, tools: PrintToolsSettings) -> None:
        '''Sync the tools widgets (bed range, distribute modifiers) from
        the project.'''
        self._syncing = True
        try:
            self._spin_from.setValue(tools.auto_from)
            self._spin_to.setValue(tools.auto_to)
            self._cb_by_color.setChecked(tools.distribute_by_color)
            self._cb_even.setChecked(tools.distribute_evenly)
        finally:
            self._syncing = False

    def sync_inspector(
        self,
        pieces: list[PrintPiece],
        footprints: dict[str, tuple[float, float]],
        colors: dict[str, str],
        blocks: list,  # BlockInstance list
        selection: set[int],
        primary: int | None,
    ) -> None:
        '''Refresh the inspector from the given selection.

        Empty selection disables the group and shows "No piece selected".
        Otherwise show primary piece: summary label
        "Piece {i+1} of {n} \u00b7 block {cx}x{cy} \u00b7 Bed {k}"
        (k = "staging" or 1-based),
        bed spin (0..count), x/y spins from the primary piece.position_x/y,
        flip checkbox as the tri-state of the whole selection (checked when
        every selected piece is flipped, unchecked when none is, partially
        checked when mixed).
        '''
        self._syncing = True
        try:
            if not selection or primary is None:
                self._inspector_group.setEnabled(False)
                self._lbl_inspector_summary.setText("No piece selected")
                return

            self._inspector_group.setEnabled(True)
            sel_sorted = sorted(selection)
            idx = sel_sorted.index(primary)
            n = len(sel_sorted)

            # Get the primary piece.
            piece = pieces[primary] if primary < len(pieces) else None
            if piece is None:
                self._inspector_group.setEnabled(False)
                self._lbl_inspector_summary.setText("No piece selected")
                return

            # The piece's block is found by uid — piece indices are unrelated
            # to block indices.
            block_instance = next((b for b in blocks if b.uid == piece.block_uid), None)
            cx, cy = (
                (block_instance.cells_x, block_instance.cells_y)
                if block_instance is not None
                else (1, 1)
            )

            bed_num = piece.bed
            if bed_num == PRINT_STAGING_BED:
                bed_str = "staging"
            else:
                bed_str = str(bed_num + 1)  # 1-based for display

            summary = (
                f"Piece {idx + 1} of {n} \u00b7 block {cx}x{cy} \u00b7 Bed {bed_str}"
            )
            self._lbl_inspector_summary.setText(summary)

            # Bed spin: 0 = staging, v = v+1 for display.
            if piece.bed == PRINT_STAGING_BED:
                self._spin_piece_bed.setValue(0)
            else:
                self._spin_piece_bed.setValue(piece.bed + 1)

            # X/Y spins.
            self._spin_piece_x.setValue(piece.position_x)
            self._spin_piece_y.setValue(piece.position_y)

            # Flip checkbox: tri-state over the whole selection, not just
            # the primary piece.
            flips = [pieces[i].flipped for i in sel_sorted if i < len(pieces)]
            if flips and all(flips):
                self._cb_flip.setCheckState(Qt.CheckState.Checked)
            elif flips and not any(flips):
                self._cb_flip.setCheckState(Qt.CheckState.Unchecked)
            else:
                self._cb_flip.setCheckState(Qt.CheckState.PartiallyChecked)

        finally:
            self._syncing = False

    def set_status(self, text: str, error: bool = False) -> None:
        '''Set the status label with green (ok) or red (error) styling.'''
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            "color: #0a6;" if not error else "color: #b00;"
        )
