'''
File:   preview.py
Brief:  PyVista/VTK-based 3D preview widget with display modes.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.2.0
'''

from __future__ import annotations

import logging

import cadquery as cq
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

logger = logging.getLogger(__name__)

# Tessellation tolerance for the interactive preview, mm.
PREVIEW_TESSELLATION_TOLERANCE = 0.15

BLOCK_COLOR = "#3f7fbf"
EDGE_COLOR = "#173a57"

# Display modes offered by the GUI.
MODE_SOLID = "Solid"
MODE_SOLID_MESH = "Solid + mesh"
MODE_WIREFRAME = "Wireframe"
DISPLAY_MODES: list[str] = [MODE_SOLID, MODE_SOLID_MESH, MODE_WIREFRAME]


def solid_to_mesh(solid: cq.Shape, tolerance: float = PREVIEW_TESSELLATION_TOLERANCE) -> pv.PolyData:
    '''Convert a CadQuery shape into a cleaned PyVista triangle mesh.

    Args:
        solid: CadQuery shape to tessellate.
        tolerance: Tessellation tolerance in mm.

    Returns:
        Triangle PolyData with merged duplicate vertices.
    '''
    vertices, triangles = solid.tessellate(tolerance)
    points = np.asarray([(v.x, v.y, v.z) for v in vertices])
    faces = np.hstack(
        [np.full((len(triangles), 1), 3), np.asarray(triangles, dtype=np.int64)]
    ).ravel()
    return pv.PolyData(points, faces).clean(tolerance=1e-6)


class PreviewWidget(QtInteractor):
    '''Interactive 3D preview: shaded solid / wireframe, optional mm grid.

    Wraps a pyvista QtInteractor; the mesh is tessellated once per model and
    display-mode/grid changes only touch actor properties, not geometry.
    '''

    def __init__(self, parent: object | None = None) -> None:
        '''Create the renderer with a white background.

        Args:
            parent: Optional parent Qt widget.
        '''
        super().__init__(parent=parent)
        self.set_background("white")
        self._mesh: pv.PolyData | None = None
        self._mesh_actor = None
        self._grid_actor = None
        self._mode = MODE_SOLID
        self._grid_enabled = False
        self.enable_anti_aliasing("msaa")

    def show_model(
        self,
        model: cq.Workplane,
        tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
    ) -> None:
        '''Tessellate the model solid and (re)create the mesh actor.

        Args:
            model: Workplane whose first solid is displayed.
            tolerance: Tessellation tolerance in mm.
        '''
        solid = model.val()
        if solid is None:
            return
        self._mesh = solid_to_mesh(solid, tolerance)
        if self._mesh_actor is not None:
            self.remove_actor(self._mesh_actor)
        self._mesh_actor = self._add_mesh_actor()
        self._update_grid()
        self.reset_camera()
        logger.debug(
            "Preview mesh: %d points / %d cells", self._mesh.n_points, self._mesh.n_cells
        )

    def set_display_mode(self, mode: str) -> None:
        '''Switch the rendering style without retessellating.

        Args:
            mode: One of ``DISPLAY_MODES``.
        '''
        if mode not in DISPLAY_MODES or mode == self._mode:
            return
        self._mode = mode
        if self._mesh_actor is not None:
            self.remove_actor(self._mesh_actor)
            self._mesh_actor = self._add_mesh_actor()
            self.render()

    def set_grid_enabled(self, enabled: bool) -> None:
        '''Show or hide the metric grid around the model.

        Args:
            enabled: True to display the grid.
        '''
        self._grid_enabled = enabled
        self._update_grid()

    # ------------------------------------------------------------------ internals
    def _add_mesh_actor(self):
        '''Add the current mesh with the styling of the active mode.

        Returns:
            The created actor, or None when no mesh is loaded.
        '''
        if self._mesh is None:
            return None
        if self._mode == MODE_WIREFRAME:
            return self.add_mesh(
                self._mesh, color=BLOCK_COLOR, style="wireframe", line_width=1.5
            )
        return self.add_mesh(
            self._mesh,
            color=BLOCK_COLOR,
            show_edges=(self._mode == MODE_SOLID_MESH),
            edge_color=EDGE_COLOR,
            line_width=0.5,
            smooth_shading=False,
            specular=0.3,
            lighting="three lights",
        )

    def _update_grid(self) -> None:
        '''Recreate or remove the metric grid actor.'''
        if self._grid_actor is not None:
            self.remove_actor(self._grid_actor)
            self._grid_actor = None
        if self._grid_enabled and self._mesh is not None:
            self._grid_actor = self.show_grid(font_size=8)
        self.render()
