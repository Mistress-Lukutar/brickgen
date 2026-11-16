'''
File:   preview.py
Brief:  PyVista/VTK-based 3D preview widget with display modes.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.10.0
'''

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import cadquery as cq
import numpy as np
import pyvista as pv
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from pyvistaqt import QtInteractor

from brickgen.core.models import BlockSettings

logger = logging.getLogger(__name__)

# Tessellation tolerance for the interactive preview, mm.
PREVIEW_TESSELLATION_TOLERANCE = 0.15

BLOCK_COLOR = "#3f7fbf"
LINE_COLOR = "#111111"

# Display modes offered by the GUI. Wireframe lines are the edges of the
# tessellated mesh itself, not separate B-Rep geometry.
# Feature-edge mode renders part-boundary outlines (stud base circles, tube bore
# rims, bottom rim) from cell_data["part"] differences.
MODE_SOLID = "Solid"
MODE_SOLID_FEATURE_EDGES = "Solid + feature edges"
MODE_SOLID_WIREFRAME = "Solid + wireframe"
MODE_WIREFRAME = "Wireframe"
DISPLAY_MODES: list[str] = [
    MODE_SOLID,
    MODE_SOLID_FEATURE_EDGES,
    MODE_SOLID_WIREFRAME,
    MODE_WIREFRAME,
]

# Angular tessellation tolerance for the mesh, rad (Shape.tessellate default).
ANGULAR_TOLERANCE = 0.1

# Semantic part ids stored in mesh cell_data["part"] by the classifier.
PART_BODY = 0
PART_STUD = 1
PART_TUBE = 2
PART_BOTTOM = 3
# Colors per part id for scalars-based surface coloring (body, stud, tube, bottom rim).
PART_COLORS: list[str] = ["#3f7fbf", "#8ec1f0", "#2b5c8a", "#3a78b5"]
# Color for interior (back) faces so the cavity reads darker than the outside.
INTERIOR_COLOR = "#1f3d5c"


def solid_to_mesh(
    solid: cq.Shape,
    tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
    angular_tolerance: float = ANGULAR_TOLERANCE,
    face_part: Callable[[cq.Face], int] | None = None,
) -> pv.PolyData:
    '''Convert a CadQuery shape into a cleaned PyVista triangle mesh.

    Pre-meshes the B-Rep in parallel via OCC then collects per-face
    tessellation (a fast lookup when pre-meshed).  Optionally annotates
    each triangle with a semantic part id supplied by ``face_part``.

    Args:
        solid: CadQuery shape to tessellate.
        tolerance: Linear tessellation tolerance in mm.
        angular_tolerance: Angular tessellation tolerance in rad.
        face_part: Optional callable that maps a cq.Face to an int part id.
            When given, the returned mesh carries ``cell_data["part"]``.

    Returns:
        Triangle PolyData with merged duplicate vertices.
    '''
    # Parallel pre-mesh: fills OCC triangulation so per-face tessellate is O(1).
    BRepMesh_IncrementalMesh(solid.wrapped, tolerance, True, angular_tolerance, True)

    all_vertices: list[tuple[float, float, float]] = []
    all_triangles: list[tuple[int, int, int]] = []
    part_values: list[int] = []
    vertex_offset = 0

    for face in solid.Faces():
        fv, ft = face.tessellate(tolerance, angular_tolerance)
        n_tris = len(ft)
        all_vertices.extend((v.x, v.y, v.z) for v in fv)
        all_triangles.extend(
            (a + vertex_offset, b + vertex_offset, c + vertex_offset) for a, b, c in ft
        )
        part = face_part(face) if face_part is not None else 0
        part_values.extend([part] * n_tris)
        vertex_offset += len(fv)

    points = np.asarray(all_vertices, dtype=np.float64)
    triangles = np.asarray(all_triangles, dtype=np.int64)
    faces = np.hstack([np.full((len(all_triangles), 1), 3), triangles]).ravel()

    mesh = pv.PolyData(points, faces)
    if part_values:
        mesh.cell_data["part"] = np.asarray(part_values, dtype=np.int8)

    return mesh.clean(tolerance=1e-6)


def make_face_classifier(settings: BlockSettings) -> Callable[[cq.Face], int]:
    '''Build a face -> part-id classifier from the block settings.

    Classification order matters: studs first (faces above the top plane),
    then cylindrical tube walls near tube centers, then bottom-rim faces at
    Z=0, everything else is body.
    '''
    half_x = settings.outer_length_x / 2
    half_y = settings.outer_length_y / 2
    centers = [
        (ix * settings.unit_size - half_x, iy * settings.unit_size - half_y)
        for ix in range(1, settings.cells_x)
        for iy in range(1, settings.cells_y)
    ]
    r2 = (settings.tube_outer_diameter / 2) ** 2

    def classifier(face: cq.Face) -> int:
        c = face.Center()
        if settings.pins_enabled and c.z > settings.block_height + 1e-6:
            return PART_STUD
        nearest2 = min(
            ((c.x - tx) ** 2 + (c.y - ty) ** 2 for tx, ty in centers),
            default=math.inf,
        )
        if (
            settings.has_tubes
            and settings.is_rectangular
            and c.z < settings.block_height - 1e-6
            and "cylinder" in face.geomType().lower()
            and nearest2 < r2
        ):
            return PART_TUBE
        if c.z < 1e-6:
            return PART_BOTTOM
        return PART_BODY

    return classifier


def semantic_edges(mesh: pv.PolyData) -> pv.PolyData:
    '''Line mesh of mesh edges where adjacent triangles have different part ids.

    Angle-based feature extraction is useless here: small cylinder facets
    (studs, tube bores) meet at 55-90 degree dihedral angles, the same as real
    geometric edges. Part boundaries give exact outlines instead: stud base
    circles, tube bore rims and the bottom rim.
    '''
    if "part" not in mesh.cell_data:
        return pv.PolyData()

    tris = mesh.faces.reshape(-1, 4)[:, 1:]
    part = np.asarray(mesh.cell_data["part"])
    n = mesh.n_points
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    lo, hi = e.min(axis=1), e.max(axis=1)
    key = lo.astype(np.int64) * n + hi
    p = np.concatenate([part, part, part])
    order = np.argsort(key, kind="stable")
    key_s, p_s = key[order], p[order]
    dup = key_s[1:] == key_s[:-1]
    boundary = np.flatnonzero(dup & (p_s[1:] != p_s[:-1]))
    seg = np.stack([key_s[boundary] // n, key_s[boundary] % n], axis=1)
    uniq, inv = np.unique(mesh.points[seg.reshape(-1)], axis=0, return_inverse=True)
    lines = inv.reshape(-1, 2)
    out = pv.PolyData(uniq)
    out.lines = np.hstack([np.full((len(lines), 1), 2), lines]).ravel()
    return out


def add_surface_actor(plotter, mesh: pv.PolyData) -> object:
    '''Add the shaded surface actor with semantic part coloring to a plotter.

    Works with any pyvista plotter (QtInteractor or off-screen Plotter).
    Falls back to the plain block color when the mesh carries no part ids.

    Args:
        plotter: A PyVista Plotter or QtInteractor instance.
        mesh: Triangle PolyData, optionally carrying ``cell_data["part"]``.

    Returns:
        The added actor (for caller convenience).
    '''
    if "part" in mesh.cell_data:
        return plotter.add_mesh(
            mesh,
            scalars="part",
            cmap=PART_COLORS,
            clim=(-0.5, float(len(PART_COLORS)) - 0.5),
            show_scalar_bar=False,
            smooth_shading=False,
            specular=0.3,
            lighting="three lights",
            backface_params={"color": INTERIOR_COLOR, "specular": 0.1},
        )
    return plotter.add_mesh(
        mesh,
        color=BLOCK_COLOR,
        smooth_shading=False,
        specular=0.3,
        lighting="three lights",
        backface_params={"color": INTERIOR_COLOR, "specular": 0.1},
    )


class PreviewWidget(QtInteractor):
    '''Interactive 3D preview: shaded mesh with optional wireframe modes.

    Wraps a pyvista QtInteractor. The surface is the OCC tessellation
    (required for shaded rendering in any OpenGL viewer); wireframe lines
    are the edges of that same mesh. The mesh is tessellated once per model
    and display-mode/grid changes only touch actors, not geometry.
    Feature-edge mode renders part-boundary outlines: stud base circles,
    tube bore rims, bottom rim.
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
        self._edge_mesh: pv.PolyData | None = None
        self._edge_actor = None
        self._wire_actor = None
        self._grid_actor = None
        self._mode = MODE_SOLID
        self._grid_enabled = False
        self.enable_anti_aliasing("msaa")

    def show_model(
        self,
        model: cq.Workplane,
        settings: BlockSettings,
        tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
        angular_tolerance: float = ANGULAR_TOLERANCE,
    ) -> None:
        '''Tessellate the model solid and rebuild the display actors.

        Args:
            model: Workplane whose first solid is displayed.
            settings: BlockSettings used to build a face-part classifier.
            tolerance: Linear tessellation tolerance in mm.
            angular_tolerance: Angular tessellation tolerance in rad.
        '''
        solid = model.val()
        if solid is None:
            return
        face_part = make_face_classifier(settings)
        self._mesh = solid_to_mesh(
            solid, tolerance, angular_tolerance, face_part=face_part
        )
        self._edge_mesh = None
        try:
            self.enable_ssao(radius=max(2.0, settings.unit_size), bias=0.02)
        except Exception:  # noqa: BLE001 - SSAO is optional eye-candy; GL may lack it
            logger.warning("SSAO unavailable in this environment", exc_info=True)
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
        '''Remove and re-add the mesh/edge/wireframe actors for the active mode.'''
        for attr in ("_mesh_actor", "_edge_actor", "_wire_actor"):
            actor = getattr(self, attr)
            if actor is not None:
                self.remove_actor(actor)
                setattr(self, attr, None)
        if self._mesh is None:
            return
        if self._mode in (MODE_SOLID, MODE_SOLID_FEATURE_EDGES):
            self._mesh_actor = add_surface_actor(self, self._mesh)
            if self._mode == MODE_SOLID:
                return
            # FEATURE_EDGES: build edge lines lazily
            if self._edge_mesh is None:
                self._edge_mesh = semantic_edges(self._mesh)
            self._edge_actor = self.add_mesh(
                self._edge_mesh,
                color=LINE_COLOR,
                line_width=2.0,
                lighting=False,
                render_lines_as_tubes=True,
            )
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
