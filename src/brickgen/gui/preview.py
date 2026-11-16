'''
File:   preview.py
Brief:  PyVista/VTK-based 3D preview widget with display modes.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.9.2
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
LINE_COLOR = "#111111"

# Display modes offered by the GUI. Wireframe lines are the edges of the
# tessellated mesh itself, not separate B-Rep geometry.
MODE_SOLID = "Solid"
MODE_SOLID_WIREFRAME = "Solid + wireframe"
MODE_WIREFRAME = "Wireframe"
DISPLAY_MODES: list[str] = [MODE_SOLID, MODE_SOLID_WIREFRAME, MODE_WIREFRAME]

# Angular tessellation tolerance for the mesh, rad (Shape.tessellate default).
ANGULAR_TOLERANCE = 0.1


def solid_to_mesh(
    solid: cq.Shape,
    tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
    angular_tolerance: float = ANGULAR_TOLERANCE,
) -> pv.PolyData:
    '''Convert a CadQuery shape into a cleaned PyVista triangle mesh.

    Args:
        solid: CadQuery shape to tessellate.
        tolerance: Linear tessellation tolerance in mm.
        angular_tolerance: Angular tessellation tolerance in rad.

    Returns:
        Triangle PolyData with merged duplicate vertices.
    '''
    vertices, triangles = solid.tessellate(tolerance, angular_tolerance)
    points = np.asarray([(v.x, v.y, v.z) for v in vertices])
    faces = np.hstack(
        [np.full((len(triangles), 1), 3), np.asarray(triangles, dtype=np.int64)]
    ).ravel()
    return pv.PolyData(points, faces).clean(tolerance=1e-6)


class PreviewWidget(QtInteractor):
    '''Interactive 3D preview: shaded mesh with optional wireframe modes.

    Wraps a pyvista QtInteractor. The surface is the OCC tessellation
    (required for shaded rendering in any OpenGL viewer); wireframe lines
    are the edges of that same mesh. The mesh is tessellated once per model
    and display-mode/grid changes only touch actors, not geometry.
    '''

    def __init__(self, parent: object | None = None) -> None:
        '''Create the renderer with a white background.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent=parent)
        self.set_background("white")
        self._mesh: pv.PolyData | None = None
        self._mesh_actor = None
        self._wire_actor = None
        self._grid_actor = None
        self._mode = MODE_SOLID
        self._grid_enabled = False
        self.enable_anti_aliasing("msaa")

    def show_model(
        self,
        model: cq.Workplane,
        tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
        angular_tolerance: float = ANGULAR_TOLERANCE,
    ) -> None:
        '''Tessellate the model solid and rebuild the display actors.

        Args:
            model: Workplane whose first solid is displayed.
            tolerance: Linear tessellation tolerance in mm.
            angular_tolerance: Angular tessellation tolerance in rad.
        '''
        solid = model.val()
        if solid is None:
            return
        self._mesh = solid_to_mesh(solid, tolerance, angular_tolerance)
        self._rebuild_actors()
        self._update_grid()
        self.reset_camera()
        logger.debug(
            "Preview mesh: %d points / %d cells",
            self._mesh.n_points,
            self._mesh.n_cells,
        )

    def set_display_mode(self, mode: str) -> None:
        '''Switch the rendering style without retessellating.

        Args:
            mode: One of ``DISPLAY_MODES``.
        '''
        if mode not in DISPLAY_MODES or mode == self._mode:
            return
        self._mode = mode
        self._rebuild_actors()
        self.render()

    def set_grid_enabled(self, enabled: bool) -> None:
        '''Show or hide the metric grid around the model.

        Args:
            enabled: True to display the grid.
        '''
        self._grid_enabled = enabled
        self._update_grid()

    # ------------------------------------------------------------------ internals
    def _rebuild_actors(self) -> None:
        '''Remove and re-add the mesh/wireframe actors for the active mode.'''
        for attr in ("_mesh_actor", "_wire_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.remove_actor(actor)
                setattr(self, attr, None)
        if self._mesh is None:
            return
        if self._mode != MODE_WIREFRAME:
            self._mesh_actor = self.add_mesh(
                self._mesh,
                color=BLOCK_COLOR,
                smooth_shading=False,
                specular=0.3,
                lighting="three lights",
            )
        if self._mode == MODE_SOLID:
            return
        if self._mode == MODE_SOLID_WIREFRAME:
            # Tubes give the lines real volume so they win the depth test
            # against the surface (plain polylines z-fight with the mesh).
            self._wire_actor = self.add_mesh(
                self._mesh,
                color=LINE_COLOR,
                style="wireframe",
                line_width=2.5,
                lighting=False,
                render_lines_as_tubes=True,
            )
        else:  # MODE_WIREFRAME
            self._wire_actor = self.add_mesh(
                self._mesh,
                color=LINE_COLOR,
                style="wireframe",
                line_width=1.5,
                lighting=False,
            )

    def _update_grid(self) -> None:
        '''Recreate or remove the metric grid actor.'''
        if self._grid_actor is not None:
            self.remove_actor(self._grid_actor)
            self._grid_actor = None
        if self._grid_enabled and self._mesh is not None:
            self._grid_actor = self.show_grid(font_size=8)
        self.render()
