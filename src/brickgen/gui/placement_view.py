'''
File:   placement_view.py
Brief:  Print-placement viewport: piece actors plus per-bed grids and zone overlays.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.20.0
'''

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QWidget

from brickgen.core.project import PRINT_STAGING_BED, PrintBedSettings
from brickgen.gui.preview import (
    GRID_MAJOR_COLOR,
    GRID_MAJOR_EVERY,
    GRID_MINOR_COLOR,
    GRID_Z_OFFSET,
    PreviewWidget,
)
from brickgen.services.placement_service import (
    bed_origin,
    staging_origin,
    zone_for_world,
)

# Visual palette for print-placement zone overlays.
BED_OUTLINE_COLOR = "#e8c84a"
STAGING_OUTLINE_COLOR = "#888888"
BED_LABEL_COLOR = "#f0dca0"
STAGING_LABEL_COLOR = "#aaaaaa"

# Layer heights: pieces sit at z >= 0; the zone furniture stacks just below
# the piece plane, each layer slightly above the previous one so coplanar
# lines never z-fight (do NOT use a coincident-topology offset here).
BED_FILL_Z = GRID_Z_OFFSET  # translucent bed plate, bottom-most layer
BED_GRID_Z = GRID_Z_OFFSET + 0.01  # per-bed cell grid lines
BED_FRAME_Z = GRID_Z_OFFSET + 0.02  # bed outline on top of plate + grid
BED_FILL_OPACITY = 0.10
STAGING_FILL_OPACITY = 0.06


class PlacementView(PreviewWidget):
    '''Placement viewport: piece actors plus per-bed grids and zone overlays.

    Pieces are ordinary scene blocks for the base PreviewWidget (per-piece
    meshes are passed through :meth:`update_scene`); this subclass only
    adds the static furniture: one translucent plate with a cell grid per
    print bed plus a frame and label, and the staging zone ("Unplaced")
    frame left of bed 0.  The base class's infinite origin-centered floor
    grid is never used here — the Grid menu toggle instead shows the
    per-bed cell grids.
    '''

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the placement viewport with default bed configuration.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent=parent)
        self._beds = PrintBedSettings()
        self._bed_grid_enabled = False
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

    def set_grid_enabled(self, enabled: bool) -> None:
        '''Show or hide the per-bed cell grids (Grid menu toggle).

        Overrides the base floor-grid toggle: the placement viewport never
        shows the infinite origin-centered grid, only grids inside beds.

        Args:
            enabled: True to display the per-bed grids.
        '''
        enabled = bool(enabled)
        if enabled == self._bed_grid_enabled:
            return
        self._bed_grid_enabled = enabled
        self._draw_zones()

    def set_grid_unit(self, unit_size: float) -> None:
        '''Set the shared block unit size (grid step = unit size * scale).

        Args:
            unit_size: Unit size in mm.
        '''
        if unit_size > 0 and unit_size != self._grid_unit:
            self._grid_unit = float(unit_size)
            self._draw_zones()

    def set_grid_scale(self, scale: float) -> None:
        '''Set the grid cell scale factor (Grid menu).

        Args:
            scale: Cell scale factor (> 0).
        '''
        if scale > 0 and scale != self._grid_scale:
            self._grid_scale = float(scale)
            self._draw_zones()

    # ---------------------------------------------------------------- snap

    def _snap_move_target(self, x: float, y: float) -> tuple[float, float]:
        '''Snap a move target to the grid of the zone under the target.

        Pieces live in zone-local coordinates; snapping to world-origin
        multiples would fight the per-bed grids drawn inside each zone.

        Args:
            x: Unsnapped world X of the move target.
            y: Unsnapped world Y of the move target.

        Returns:
            The ``(x, y)`` target snapped to zone-local grid multiples.
        '''
        step = max(self.grid_step(), 1e-9)
        zone = zone_for_world(x, y, self._beds)
        if zone == PRINT_STAGING_BED:
            ox, oy = staging_origin(self._beds)
        else:
            ox, oy = bed_origin(zone, self._beds)
        return (
            round((x - ox) / step) * step + ox,
            round((y - oy) / step) * step + oy,
        )

    # ---------------------------------------------------------- overlay drawing

    def _draw_zones(self) -> None:
        '''Remove old zone overlays and rebuild plates, grids and labels.'''
        # Remove every previously-created zone actor.
        for actor in self._zone_actors:
            try:
                self.remove_actor(actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._zone_actors.clear()

        beds = self._beds

        # --- One plate + frame + label per bed ---
        for i in range(beds.count):
            ox, oy = bed_origin(i, beds)
            self._zone_actors.append(
                self._add_zone_fill(
                    ox,
                    oy,
                    beds.width,
                    beds.depth,
                    color=BED_OUTLINE_COLOR,
                    opacity=BED_FILL_OPACITY,
                )
            )
            frame = self._add_zone_frame(
                ox, oy, beds.width, beds.depth, BED_FRAME_Z, BED_OUTLINE_COLOR
            )
            self._zone_actors.append(frame)

            # Label: centre x, slightly above bottom edge y.
            label_x = ox + beds.width / 2.0
            label_y = oy + 2.0
            label_actor = self.add_point_labels(
                np.array([[label_x, label_y, GRID_Z_OFFSET]]),
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

        # --- Per-bed cell grids (Grid menu toggle) ---
        if self._bed_grid_enabled:
            minor, major = self._bed_grid_segments()
            for mesh, color in ((minor, GRID_MINOR_COLOR), (major, GRID_MAJOR_COLOR)):
                if len(mesh):
                    self._zone_actors.append(
                        self._add_grid_lines(mesh, color=color, line_width=1.0)
                    )

        # --- Staging zone: plate + frame + label (always present) ---
        sx, sy = staging_origin(beds)
        self._zone_actors.append(
            self._add_zone_fill(
                sx,
                sy,
                beds.width,
                beds.depth,
                color=STAGING_OUTLINE_COLOR,
                opacity=STAGING_FILL_OPACITY,
            )
        )
        frame = self._add_zone_frame(
            sx, sy, beds.width, beds.depth, BED_FRAME_Z, STAGING_OUTLINE_COLOR
        )
        self._zone_actors.append(frame)

        label_x = sx + beds.width / 2.0
        label_y = sy + 2.0
        label_actor = self.add_point_labels(
            np.array([[label_x, label_y, GRID_Z_OFFSET]]),
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

    def _add_zone_fill(
        self,
        ox: float,
        oy: float,
        width: float,
        depth: float,
        color: str,
        opacity: float,
    ) -> pv.Actor:
        '''Add one translucent rectangular zone plate at ``BED_FILL_Z``.

        Args:
            ox, oy: Bottom-left corner of the zone in world mm.
            width: Zone width in mm.
            depth: Zone depth in mm.
            color: Plate color.
            opacity: Plate opacity (low: furniture must stay subtle).

        Returns:
            The plate actor (non-pickable).
        '''
        z = np.full(4, BED_FILL_Z)
        points = np.array(
            [
                [ox, oy, z[0]],
                [ox + width, oy, z[1]],
                [ox + width, oy + depth, z[2]],
                [ox, oy + depth, z[3]],
            ]
        )
        quad = pv.PolyData(points, faces=np.array([[4, 0, 1, 2, 3]]))
        actor = self.add_mesh(
            quad,
            color=color,
            opacity=opacity,
            lighting=False,
            pickable=False,
            reset_camera=False,
            render=False,
        )
        actor.SetPickable(False)
        return actor

    def _add_zone_frame(
        self, ox: float, oy: float, width: float, depth: float, z: float, color: str
    ) -> pv.Actor:
        '''Add one rectangular zone outline at height *z*.

        Args:
            ox, oy: Bottom-left corner of the zone in world mm.
            width: Zone width in mm.
            depth: Zone depth in mm.
            z: Line height (slightly above the plate/grid layers).
            color: Outline color.

        Returns:
            The frame actor (non-pickable).
        '''
        corners = np.array(
            [
                [ox, oy, z],
                [ox + width, oy, z],
                [ox + width, oy + depth, z],
                [ox, oy + depth, z],
                [ox, oy, z],  # close the loop
            ]
        )
        frame = pv.lines_from_points(corners)
        actor = self.add_mesh(
            frame,
            color=color,
            line_width=2.0,
            lighting=False,
            pickable=False,
            reset_camera=False,
            render=False,
        )
        actor.SetPickable(False)
        return actor

    def _bed_grid_segments(self) -> tuple[np.ndarray, np.ndarray]:
        '''Interior cell-grid line segments inside every bed.

        Lines at the zone borders are skipped: the bed frame already draws
        them (at a higher z).  Every ``GRID_MAJOR_EVERY``-th cell line is a
        major line.

        Returns:
            ``(minor, major)`` segment arrays shaped ``(N, 2, 3)``.
        '''
        step = max(self.grid_step(), 1e-9)
        beds = self._beds
        minor: list[list[list[float]]] = []
        major: list[list[list[float]]] = []

        def add_line(
            start: tuple[float, float], end: tuple[float, float], k: int
        ) -> None:
            seg = [
                [start[0], start[1], BED_GRID_Z],
                [end[0], end[1], BED_GRID_Z],
            ]
            (major if k % GRID_MAJOR_EVERY == 0 else minor).append(seg)

        for i in range(beds.count):
            ox, oy = bed_origin(i, beds)
            # Cell count per axis; an exact last cell sits on the frame and
            # is skipped like the k=0 line.
            eps = 1e-6
            nx = int(np.floor(beds.width / step + eps))
            ny = int(np.floor(beds.depth / step + eps))
            xs = (
                range(1, nx) if abs(nx * step - beds.width) <= eps else range(1, nx + 1)
            )
            ys = (
                range(1, ny) if abs(ny * step - beds.depth) <= eps else range(1, ny + 1)
            )
            for k in xs:
                x = ox + k * step
                add_line((x, oy), (x, oy + beds.depth), k)
            for k in ys:
                y = oy + k * step
                add_line((ox, y), (ox + beds.width, y), k)

        return (
            np.array(minor, dtype=float).reshape(-1, 2, 3),
            np.array(major, dtype=float).reshape(-1, 2, 3),
        )
