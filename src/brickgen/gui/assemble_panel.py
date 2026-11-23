'''
File:   assemble_panel.py
Brief:  Left-sidebar panel for the Assemble tab: snapping settings and the
        selected-piece inspector.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from brickgen.core.project import AssembledPiece

# Display strings of the rotation combo; index == quarter-turn count.
ROTATION_LABELS = ("0°", "90°", "180°", "270°")


class AssemblePanel(QWidget):
    '''Left-sidebar panel for the Assemble tab.

    Holds the snapping preferences (grid + magnetic part snapping) and the
    inspector of the selected assembled pieces (position X/Y/Z, 90-degree
    Z rotation, flip, delete). There are deliberately no block-geometry
    controls here: parts are planned on the Planning tab and only
    assembled here.
    '''

    snap_grid_changed = Signal(bool)
    snap_parts_changed = Signal(bool)
    # Absolute XYZ of the primary piece; the window shifts the whole
    # selection by the same delta.
    piece_position_changed = Signal(float, float, float)
    piece_rotation_changed = Signal(int)  # quarter-turn count 0..3
    piece_flip_changed = Signal(bool)  # flip state for ALL selected pieces
    delete_requested = Signal()

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

        # --- Snapping group ---
        self._snap_group = QGroupBox("Snapping")
        snap_layout = QVBoxLayout(self._snap_group)
        snap_layout.setSpacing(8)

        self._cb_snap_grid = QCheckBox("Snap to grid")
        self._cb_snap_grid.setChecked(True)
        self._cb_snap_grid.setToolTip(
            "Round piece drops and drags to the grid step (Unit Size x Cell Scale)."
        )
        self._cb_snap_grid.toggled.connect(self._on_snap_grid)
        snap_layout.addWidget(self._cb_snap_grid)

        self._cb_snap_parts = QCheckBox("Snap to parts")
        self._cb_snap_parts.setChecked(True)
        self._cb_snap_parts.setToolTip(
            "Magnetize pieces to the assembly: drops land on the topmost\n"
            "support under the footprint and align flush to the sides of\n"
            "nearby pieces. Without it pieces keep their height when dragged."
        )
        self._cb_snap_parts.toggled.connect(self._on_snap_parts)
        snap_layout.addWidget(self._cb_snap_parts)

        inner.addWidget(self._snap_group)

        # --- Selected piece inspector group ---
        self._inspector_group = QGroupBox("Selected piece")
        self._inspector_group.setEnabled(False)
        insp_form = QFormLayout(self._inspector_group)
        insp_form.setVerticalSpacing(8)

        self._lbl_inspector_summary = QLabel("No piece selected")
        self._lbl_inspector_summary.setWordWrap(True)
        insp_form.addRow(self._lbl_inspector_summary)

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

        self._spin_piece_z = QDoubleSpinBox()
        self._spin_piece_z.setRange(-100000.0, 100000.0)
        self._spin_piece_z.setDecimals(2)
        self._spin_piece_z.setSingleStep(1.0)
        self._spin_piece_z.setSuffix(" mm")
        self._spin_piece_z.setToolTip("Height of the piece bottom above the floor.")
        self._spin_piece_z.valueChanged.connect(self._on_piece_position_changed)
        insp_form.addRow("Z, mm", self._spin_piece_z)

        self._combo_rotation = QComboBox()
        self._combo_rotation.addItems(ROTATION_LABELS)
        self._combo_rotation.setToolTip("Rotation about the vertical axis (90° steps).")
        self._combo_rotation.currentIndexChanged.connect(
            self._on_piece_rotation_changed
        )
        insp_form.addRow("Rotation", self._combo_rotation)

        self._cb_flip = QCheckBox("Flip 180° (pins down)")
        self._cb_flip.checkStateChanged.connect(self._on_piece_flip_state_changed)
        insp_form.addRow(self._cb_flip)

        self._btn_delete = QPushButton("Delete piece")
        self._btn_delete.setToolTip("Remove the selected copies from the assembly.")
        self._btn_delete.clicked.connect(self._on_delete)
        insp_form.addRow(self._btn_delete)

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
        # that width plus a scrollbar (mirrors the other sidebars).
        form_width = (
            container.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
        )
        self.setMinimumWidth(form_width)
        scroll.setMinimumWidth(form_width)

    # ------------------------------------------------------------ handlers

    def _on_snap_grid(self, checked: bool) -> None:
        '''Snap-to-grid checkbox changed by the user.'''
        if self._syncing:
            return
        self.snap_grid_changed.emit(bool(checked))

    def _on_snap_parts(self, checked: bool) -> None:
        '''Snap-to-parts checkbox changed by the user.'''
        if self._syncing:
            return
        self.snap_parts_changed.emit(bool(checked))

    def _on_piece_position_changed(self, _value: float) -> None:
        '''X, Y or Z spinbox changed: emit the current absolute position.'''
        if self._syncing:
            return
        self.piece_position_changed.emit(
            self._spin_piece_x.value(),
            self._spin_piece_y.value(),
            self._spin_piece_z.value(),
        )

    def _on_piece_rotation_changed(self, index: int) -> None:
        '''Rotation combo changed: emit the quarter-turn count.'''
        if self._syncing:
            return
        self.piece_rotation_changed.emit(int(index))

    def _on_piece_flip_state_changed(self, state: Qt.CheckState) -> None:
        '''Flip checkbox changed by the user: apply to ALL selected.'''
        if self._syncing:
            return
        self.piece_flip_changed.emit(state == Qt.CheckState.Checked)

    def _on_delete(self) -> None:
        '''Delete-piece button clicked.'''
        self.delete_requested.emit()

    # ---------------------------------------------------------- public sync

    def set_snapping(self, snap_grid: bool, snap_parts: bool) -> None:
        '''Sync the snapping checkboxes from the project (programmatic).'''
        self._syncing = True
        try:
            self._cb_snap_grid.setChecked(bool(snap_grid))
            self._cb_snap_parts.setChecked(bool(snap_parts))
        finally:
            self._syncing = False

    def sync_inspector(
        self,
        pieces: list[AssembledPiece],
        selection: set[int],
        primary: int | None,
        block_labels: dict[str, str],
    ) -> None:
        '''Refresh the inspector from the given selection.

        Empty selection disables the group and shows "No piece selected";
        otherwise the widgets mirror the primary piece and the flip
        checkbox shows the tri-state of the whole selection.

        Args:
            pieces: The assembled pieces of the project.
            selection: Selected piece indices.
            primary: Primary (last clicked) piece index or None.
            block_labels: Mapping of block uid -> display label (grid size).
        '''
        self._syncing = True
        try:
            if not selection or primary is None or primary >= len(pieces):
                self._inspector_group.setEnabled(False)
                self._lbl_inspector_summary.setText("No piece selected")
                return
            self._inspector_group.setEnabled(True)
            piece = pieces[primary]
            n = len(selection)
            label = block_labels.get(piece.block_uid, "block")
            extra = f" +{n - 1} more" if n > 1 else ""
            self._lbl_inspector_summary.setText(
                f"Piece [{primary + 1}] of {n} · {label}{extra}"
            )
            self._spin_piece_x.setValue(piece.position_x)
            self._spin_piece_y.setValue(piece.position_y)
            self._spin_piece_z.setValue(piece.position_z)
            self._combo_rotation.setCurrentIndex(piece.rot_z)
            flips = [pieces[i].flipped for i in sorted(selection) if i < len(pieces)]
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
