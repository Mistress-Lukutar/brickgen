'''
File:   preview.py
Brief:  PyVista/VTK-based 3D preview widget with display modes.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.8.1
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
EDGE_COLOR = "#111111"

# Display modes offered by the GUI. Edges are topological B-Rep edges
# (CAD-style), not triangulation edges.
MODE_SOLID = "Solid"
MODE_EDGES = "Solid + edges"
MODE_WIREFRAME = "Wireframe"
DISPLAY_MODES: list[str] = [MODE_SOLID, MODE_EDGES, MODE_WIREFRAME]

# B-Rep edges are discretized adaptively with the SAME two constraints as
# the OCC tessellation (Shape.tessellate): linear chord deviation <= tolerance
# and turn angle between segments <= angular tolerance (0.1 rad default).
# Without the angular bound, coarse tolerances produce polygonal edges over
# a visually smooth meshed surface (the angular constraint keeps the mesh
# dense on cylinders while the chord-only edges collapse to triangles).
ANGULAR_TOLERANCE = 0.1  # rad, matches Shape.tessellate default
MAX_EDGE_SEGMENTS = 256


def _sample_edge(edge: cq.Edge, tolerance: float) -> list[cq.Vector]:
    '''Sample an edge as a polyline matching the mesh constraints.

    The segment count doubles until every chord deviates from the true
    curve by no more than ``tolerance`` AND the turn angle between adjacent
    chords stays within ``ANGULAR_TOLERANCE`` (or the segment cap is hit).

    Args:
        edge: CadQuery edge to sample.
        tolerance: Maximum chord deviation in mm.

    Returns:
        Sampled points of the polyline (including both endpoints).
    '''
    segments = 1
    while True:
        fractions = [i / segments for i in range(segments + 1)]
        points = [edge.positionAt(f) for f in fractions]
        max_dev = 0.0
        for pa, pb, fa, fb in zip(points, points[1:], fractions, fractions[1:]):
            mid = edge.positionAt((fa + fb) / 2.0)
            chord = cq.Vector((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0,
                              (pa.z + pb.z) / 2.0)
            max_dev = max(max_dev, (mid - chord).Length)
        max_turn = 0.0
        for a, b, c in zip(points, points[1:], points[2:]):
            v1 = cq.Vector(b.x - a.x, b.y - a.y, b.z - a.z)
            v2 = cq.Vector(c.x - b.x, c.y - b.y, c.z - b.z)
            if v1.Length > 0 and v2.Length > 0:
                max_turn = max(max_turn, v1.getAngle(v2))
        if (max_dev <= tolerance and max_turn <= ANGULAR_TOLERANCE) \
                or segments >= MAX_EDGE_SEGMENTS:
            return points
        segments *= 2


def brep_edges_polydata(solid: cq.Shape, tolerance: float = 0.15) -> pv.PolyData:
    '''Discretize the topological edges of a shape into line polylines.

    Sampling respects the same linear (chord deviation <= ``tolerance``)
    and angular (<= 0.1 rad) constraints as the OCC surface tessellation,
    so curved edges visually match the mesh at any precision. This yields
    CAD-style feature edges independent of the mesh; discretizing curves
    is far cheaper than meshing.

    Args:
        solid: CadQuery shape whose edges are sampled.
        tolerance: Maximum chord deviation in mm (matches mesh tolerance).

    Returns:
        PolyData containing one line cell per edge.
    '''
    points: list[tuple[float, float, float]] = []
    lines: list[int] = []
    for edge in solid.Edges():
        sampled = _sample_edge(edge, tolerance)
        indices = []
        for p in sampled:
            indices.append(len(points))
            points.append((p.x, p.y, p.z))
        lines.extend([len(indices), *indices])
    if not lines:
        return pv.PolyData()
    return pv.PolyData(
        np.asarray(points), lines=np.asarray(lines, dtype=np.int64)
    )


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
    '''Interactive 3D preview: shaded B-Rep solid with CAD-style edges.

    Wraps a pyvista QtInteractor. The surface is the OCC tessellation
    (required for shaded rendering in any OpenGL viewer); the "edges" are
    topological B-Rep curves discretized independently of the mesh, exactly
    like feature edges in CAD viewers. The mesh is tessellated once per
    model and display-mode/grid changes only touch actors, not geometry.
    '''

    def __init__(self, parent: object | None = None) -> None:
        '''Create the renderer with a white background.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent=parent)
        self.set_background("white")
        self._mesh: pv.PolyData | None = None
        self._edges: pv.PolyData | None = None
        self._mesh_actor = None
        self._edges_actor = None
        self._grid_actor = None
        self._mode = MODE_SOLID
        self._grid_enabled = False
        self.enable_anti_aliasing("msaa")

    def show_model(
        self,
        model: cq.Workplane,
        tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
    ) -> None:
        '''Tessellate the model solid and rebuild both display actors.

        Args:
            model: Workplane whose first solid is displayed.
            tolerance: Tessellation tolerance in mm.
        '''
        solid = model.val()
        if solid is None:
            return
        self._mesh = solid_to_mesh(solid, tolerance)
        self._edges = brep_edges_polydata(solid, tolerance)
        self._rebuild_actors()
        self._update_grid()
        self.reset_camera()
        logger.debug(
            "Preview mesh: %d points / %d cells; %d B-Rep edges",
            self._mesh.n_points, self._mesh.n_cells, self._edges.n_cells,
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
        '''Remove and re-add the surface/edges actors for the active mode.'''
        for attr in ("_mesh_actor", "_edges_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.remove_actor(actor)
                setattr(self, attr, None)
        if self._mesh is not None and self._mode != MODE_WIREFRAME:
            self._mesh_actor = self.add_mesh(
                self._mesh,
                color=BLOCK_COLOR,
                smooth_shading=False,
                specular=0.3,
                lighting="three lights",
            )
        if self._edges is not None and self._mode != MODE_SOLID:
            # Tubes give the lines real volume so they win the depth test
            # against the surface (plain polylines z-fight with the mesh).
            self._edges_actor = self.add_mesh(
                self._edges,
                color=EDGE_COLOR,
                line_width=2.5,
                lighting=False,
                render_lines_as_tubes=True,
            )

    def _update_grid(self) -> None:
        '''Recreate or remove the metric grid actor.'''
        if self._grid_actor is not None:
            self.remove_actor(self._grid_actor)
            self._grid_actor = None
        if self._grid_enabled and self._mesh is not None:
            self._grid_actor = self.show_grid(font_size=8)
        self.render()
