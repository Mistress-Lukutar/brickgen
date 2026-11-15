'''
File:   app.py
Brief:  Main BrickGen tkinter window: settings panel, 3D preview, export.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.1.0
'''

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cadquery as cq

from brickgen.core.builder import build_block
from brickgen.core.exceptions import BrickGenError
from brickgen.core.models import BlockSettings, validate_settings
from brickgen.gui.preview import PreviewPanel
from brickgen.services.export_service import SUPPORTED_FORMATS, export_model
from brickgen.services.settings_io import load_settings, save_settings

logger = logging.getLogger(__name__)

_SETTINGS_FILETYPES = [
    ("BrickGen settings", "*.json"),
    ("All files", "*.*"),
]

# (label, settings field, unit hint)
_FLOAT_FIELDS: list[tuple[str, str, str]] = [
    ("Unit size", "unit_size", "mm"),
    ("Wall thickness", "wall_thickness", "mm"),
    ("Block height", "block_height", "mm"),
    ("Pin diameter", "pin_diameter", "mm"),
    ("Pin height", "pin_height", "mm"),
]
_INT_FIELDS: list[tuple[str, str, str]] = [
    ("Cells X (length)", "cells_x", "cells"),
    ("Cells Y (width)", "cells_y", "cells"),
]


class BrickGenApp:
    '''Main application window assembling settings, preview and export.'''

    def __init__(self, root: tk.Tk) -> None:
        '''Build the window layout and render the default 1x1 block.

        Args:
            root: Tk root window to populate.
        '''
        self._root = root
        self._settings = BlockSettings()
        self._model: cq.Workplane | None = None

        root.title("BrickGen - LEGO-like block generator")
        root.geometry("1100x680")
        root.minsize(900, 560)

        self._build_layout()
        self._on_generate()
        logger.info("BrickGen window ready")

    # ------------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        '''Create the two-pane layout: settings sidebar + preview area.'''
        paned = ttk.PanedWindow(self._root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        sidebar = ttk.Frame(paned, padding=10)
        paned.add(sidebar, weight=0)
        preview_frame = ttk.Frame(paned)
        paned.add(preview_frame, weight=1)

        self._preview = PreviewPanel(preview_frame)
        self._build_sidebar(sidebar)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        '''Create the settings form, action buttons and status bar.'''
        parent.columnconfigure(0, weight=1)

        header = ttk.Label(
            parent, text="Block parameters", font=("", 11, "bold")
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self._vars: dict[str, tk.Variable] = {}
        row = 1
        for label, field, unit in _INT_FIELDS:
            self._add_spinbox(parent, row, label, field, unit, 1, 64, 1)
            row += 1
        for label, field, unit in _FLOAT_FIELDS:
            self._add_spinbox(parent, row, label, field, unit, 0.05, 500.0, 0.1)
            row += 1

        pins_var = tk.BooleanVar(value=self._settings.pins_enabled)
        pins_check = ttk.Checkbutton(
            parent, text="Studs on top", variable=pins_var,
            command=self._on_pins_toggled,
        )
        pins_check.grid(row=row, column=0, sticky="w", pady=6)
        self._vars["pins_enabled"] = pins_var
        row += 1

        self._form_label = ttk.Label(parent, text="", foreground="#555")
        self._form_label.grid(row=row, column=0, sticky="w")
        row += 1

        buttons = ttk.Frame(parent)
        buttons.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Generate", command=self._on_generate).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(buttons, text="Export model...", command=self._on_export).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(buttons, text="Save settings...", command=self._on_save_settings).grid(
            row=2, column=0, sticky="ew", pady=2, padx=(0, 2)
        )
        ttk.Button(buttons, text="Load settings...", command=self._on_load_settings).grid(
            row=2, column=1, sticky="ew", pady=2, padx=(2, 0)
        )
        row += 1

        self._status = tk.StringVar(value="Ready.")
        ttk.Label(parent, textvariable=self._status, foreground="#0a6").grid(
            row=row, column=0, sticky="ew", pady=(8, 0)
        )

    def _add_spinbox(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        field: str,
        unit: str,
        min_value: float,
        max_value: float,
        step: float,
    ) -> None:
        '''Add one labelled spinbox row bound to a settings field.

        Args:
            parent: Frame to place the row in.
            row: Grid row index.
            label: Human-readable field label.
            field: BlockSettings field name.
            unit: Unit hint shown after the field.
            min_value: Minimal spinbox value.
            max_value: Maximal spinbox value.
            step: Spinbox increment.
        '''
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=str(getattr(self._settings, field)))
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="e")
        spin = ttk.Spinbox(
            frame, textvariable=var, from_=min_value, to=max_value,
            increment=step, width=7,
        )
        spin.pack(side="left")
        ttk.Label(frame, text=unit, foreground="#888").pack(side="left", padx=(4, 0))
        self._vars[field] = var

    # ----------------------------------------------------------------- actions
    def _collect_settings(self) -> BlockSettings:
        '''Build BlockSettings from the form fields.

        Returns:
            Validated settings.

        Raises:
            ValueError: If a numeric field contains a non-numeric value.
            SettingsValidationError: If values fail geometry validation.
        '''
        values: dict[str, int | float | bool] = {}
        for _, field, _ in _INT_FIELDS:
            values[field] = int(float(self._vars[field].get()))
        for _, field, _ in _FLOAT_FIELDS:
            values[field] = float(self._vars[field].get())
        values["pins_enabled"] = bool(self._vars["pins_enabled"].get())
        return validate_settings(values)

    def _on_pins_toggled(self) -> None:
        '''React to the studs checkbox without rebuilding the model.'''
        self._settings.pins_enabled = bool(self._vars["pins_enabled"].get())
        self._on_generate()

    def _on_generate(self) -> None:
        '''Rebuild the model from form values and refresh the preview.'''
        try:
            self._settings = self._collect_settings()
        except (ValueError, BrickGenError) as exc:
            self._status.set("Invalid settings.")
            messagebox.showwarning("Invalid settings", str(exc), parent=self._root)
            return
        try:
            self._model = build_block(self._settings)
        except BrickGenError as exc:
            self._status.set("Generation failed.")
            messagebox.showerror("Generation failed", str(exc), parent=self._root)
            return
        self._form_label.config(
            text=f"Form: {self._settings.describe()}   "
                 f"Footprint: {self._settings.outer_length_x:.2f} x "
                 f"{self._settings.outer_length_y:.2f} mm   "
                 f"Height: {self._settings.total_height:.2f} mm"
        )
        self._preview.show_model(self._model)
        self._status.set(
            f"Generated {self._settings.describe()} block "
            f"({self._settings.outer_length_x:.1f} x "
            f"{self._settings.outer_length_y:.1f} x "
            f"{self._settings.total_height:.1f} mm)."
        )

    def _on_export(self) -> None:
        '''Ask for a destination file and export the current model.'''
        if self._model is None:
            messagebox.showinfo(
                "Nothing to export", "Generate a block first.", parent=self._root
            )
            return
        formats = sorted(SUPPORTED_FORMATS)
        default_name = f"block_{self._settings.describe()}.step"
        path = filedialog.asksaveasfilename(
            parent=self._root,
            title="Export model",
            defaultextension=".step",
            initialfile=default_name,
            filetypes=[(fmt, f"*.{fmt.lower()}") for fmt in formats]
            + [("All files", "*.*")],
        )
        if not path:
            return
        suffix = Path(path).suffix.lstrip(".").upper()
        file_format = suffix if suffix in SUPPORTED_FORMATS else "STEP"
        try:
            written = export_model(self._model, Path(path), file_format)
        except BrickGenError as exc:
            self._status.set("Export failed.")
            messagebox.showerror("Export failed", str(exc), parent=self._root)
            return
        self._status.set(f"Exported: {written.name}")

    def _on_save_settings(self) -> None:
        '''Persist current form values into a JSON preset.'''
        path = filedialog.asksaveasfilename(
            parent=self._root,
            title="Save settings",
            defaultextension=".json",
            initialfile="brickgen_settings.json",
            filetypes=_SETTINGS_FILETYPES,
        )
        if not path:
            return
        try:
            self._settings = self._collect_settings()
            written = save_settings(self._settings, Path(path))
        except (ValueError, BrickGenError) as exc:
            messagebox.showerror("Save failed", str(exc), parent=self._root)
            return
        self._status.set(f"Settings saved: {written.name}")

    def _on_load_settings(self) -> None:
        '''Load a JSON preset into the form and regenerate the model.'''
        path = filedialog.askopenfilename(
            parent=self._root, title="Load settings", filetypes=_SETTINGS_FILETYPES
        )
        if not path:
            return
        try:
            self._settings = load_settings(Path(path))
        except BrickGenError as exc:
            messagebox.showerror("Load failed", str(exc), parent=self._root)
            return
        self._sync_form_from_settings()
        self._on_generate()

    def _sync_form_from_settings(self) -> None:
        '''Push current settings values back into the form widgets.'''
        for _, field, _ in _INT_FIELDS + _FLOAT_FIELDS:
            self._vars[field].set(str(getattr(self._settings, field)))
        self._vars["pins_enabled"].set(self._settings.pins_enabled)


def main() -> None:
    '''Application entry point: configure logging and run the GUI loop.'''
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    root = tk.Tk()
    BrickGenApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
