'''
File:   placement_view.py
Brief:  Print-placement viewport: piece actors plus static bed/staging zone overlays.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.16.0
'''

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QWidget

from brickgen.core.project import PrintBedSettings
from brickgen.gui.preview import GRID_Z_OFFSET, PreviewWidget
from brickgen.services.placement_service import bed_origin, staging_origin

# Visual palette for print-placement zone overlays.
BED_OUTLINE_COLOR = "#e8c84a"
STAGING_OUTLINE_COLOR = "#888888"
BED_LABEL_COLOR = "#f0dca0"
STAGING_LABEL_COLOR = "#aaaaaa"


class PlacementView(PreviewWidget):
    '''Placement viewport: piece actors plus static bed/staging zone overlays.

    Pieces are ordinary scene blocks for the base PreviewWidget (per-piece
    meshes are passed through :meth:`update_scene`); this subclass only
    adds the static furniture: one rectangular frame per print bed with a
    label, and the staging zone ("Unplaced") frame left of bed 0.
    '''

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the placement viewport with default bed configuration.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent=parent)
        self._beds = PrintBedSettings()
        self._zone_actors: list[pv.Actor] = []
        self._draw_zones()

    # ------------------------------------------------------------------ API

    def beds(self) -> PrintBedSettings:
        '''Current bed configuration.'''
        return self._beds

    def set_print_beds(self, beds: PrintBedSettings) -> None:
        '''Store the bed configuration and rebuild the zone overlays.

        No camera reset: the camera is only placed by the first scene build.

        Args:
            beds: New print-bed settings.
        '''
        self._beds = beds
        self._draw_zones()

    # ---------------------------------------------------------- overlay drawing

    def _draw_zones(self) -> None:
        '''Remove old zone overlays and rebuild frames + labels for current beds.'''
        # Remove every previously-created zone actor.
        for actor in self._zone_actors:
            try:
                self.remove_actor(actor)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._zone_actors.clear()

        beds = self._beds

        # --- Draw one frame per bed ---
        for i in range(beds.count):
            ox, oy = bed_origin(i, beds)
            corners = np.array(
                [
                    [ox, oy, GRID_Z_OFFSET],
                    [ox + beds.width, oy, GRID_Z_OFFSET],
                    [ox + beds.width, oy + beds.depth, GRID_Z_OFFSET],
                    [ox, oy + beds.depth, GRID_Z_OFFSET],
                    [ox, oy, GRID_Z_OFFSET],  # close the loop
                ]
            )
            frame = pv.lines_from_points(corners)
            actor = self.add_mesh(
                frame,
                color=BED_OUTLINE_COLOR,
                line_width=2.0,
                lighting=False,
                pickable=False,
                reset_camera=False,
            )
            actor.SetPickable(False)
            self._zone_actors.append(actor)

            # Label: centre x, slightly above bottom edge y, at GRID_Z_OFFSET.
            label_x = ox + beds.width / 2.0
            label_y = oy + 2.0
            label_z = GRID_Z_OFFSET
            label_actor = self.add_point_labels(
                np.array([[label_x, label_y, label_z]]),
                [f"Bed {i + 1} \u00b7 {beds.width:g}\u00d7{beds.depth:g}"],
                always_visible=True,
                show_points=False,
                shape=None,
                font_size=9,
                text_color=BED_LABEL_COLOR,
                name=f"bed_label_{i}",
                render=False,
            )
            if label_actor is not None:
                self._zone_actors.append(label_actor)

        # --- Draw the staging zone frame (always present) ---
        sx, sy = staging_origin(beds)
        corners = np.array(
            [
                [sx, sy, GRID_Z_OFFSET],
                [sx + beds.width, sy, GRID_Z_OFFSET],
                [sx + beds.width, sy + beds.depth, GRID_Z_OFFSET],
                [sx, sy + beds.depth, GRID_Z_OFFSET],
                [sx, sy, GRID_Z_OFFSET],  # close the loop
            ]
        )
        frame = pv.lines_from_points(corners)
        actor = self.add_mesh(
            frame,
            color=STAGING_OUTLINE_COLOR,
            line_width=2.0,
            lighting=False,
            pickable=False,
            reset_camera=False,
        )
        actor.SetPickable(False)
        self._zone_actors.append(actor)

        # Staging label.
        label_x = sx + beds.width / 2.0
        label_y = sy + 2.0
        label_z = GRID_Z_OFFSET
        label_actor = self.add_point_labels(
            np.array([[label_x, label_y, label_z]]),
            ["Unplaced"],
            always_visible=True,
            show_points=False,
            shape=None,
            font_size=9,
            text_color=STAGING_LABEL_COLOR,
            name="staging_label",
            render=False,
        )
        if label_actor is not None:
            self._zone_actors.append(label_actor)

        self.render()
