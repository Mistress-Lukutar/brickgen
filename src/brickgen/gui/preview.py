'''
File:   preview.py
Brief:  PyVista/VTK-based 3D preview widget with display modes.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.12.0
'''

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import cast

import cadquery as cq
import numpy as np
import pyvista as pv
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from PySide6.QtWidgets import QWidget
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
# Consumed by feature-edge extraction (semantic_edges), not by coloring.
PART_BODY = 0
PART_STUD = 1
PART_TUBE = 2
PART_BOTTOM = 3
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


def add_surface_actor(plotter, mesh: pv.PolyData) -> pv.Actor:
    '''Add the shaded surface actor with a uniform block color to a plotter.

    Works with any pyvista plotter (QtInteractor or off-screen Plotter).
    SSAO provides the depth cues, so the surface is tinted with the single
    block color; ``cell_data["part"]`` is only consumed by feature-edge
    extraction, not by coloring.

    Args:
        plotter: A PyVista Plotter or QtInteractor instance.
        mesh: Triangle PolyData, optionally carrying ``cell_data["part"]``.

    Returns:
        The added actor (for caller convenience).
    '''
    return plotter.add_mesh(
        mesh,
        color=BLOCK_COLOR,
        smooth_shading=False,
        specular=0.3,
        lighting="three lights",
        backface_params={"color": INTERIOR_COLOR, "specular": 0.1},
    )


def mesh_to_arrays(mesh: pv.PolyData) -> dict:
    '''Pack a preview mesh into picklable numpy arrays for IPC.

    Returns:
        Dict with 'points' (N,3 float64), 'faces' (int64 face buffer
        [3, i, j, k, ...] as produced by pv.PolyData), and 'part'
        (M, int8 per-triangle ids, empty when absent).
    '''
    points: np.ndarray = np.asarray(mesh.points, dtype=np.float64)
    faces: np.ndarray = np.asarray(mesh.faces, dtype=np.int64).ravel()
    if 'part' in mesh.cell_data:
        part: np.ndarray = np.asarray(mesh.cell_data['part'], dtype=np.int8)
    else:
        part = np.empty(0, dtype=np.int8)
    return {'points': points, 'faces': faces, 'part': part}


def mesh_from_arrays(data: dict) -> pv.PolyData:
    '''Rebuild a pv.PolyData from :func:`mesh_to_arrays` output (UI thread).

    Args:
        data: Dict with keys 'points', 'faces', and 'part' as produced by
            :func:`mesh_to_arrays`.

    Returns:
        A reconstructed ``pv.PolyData`` mesh.
    '''
    points = data['points']
    faces = data['faces']
    mesh = pv.PolyData(points, faces)
    part = data.get('part')
    if part is not None and part.size > 0:
        mesh.cell_data['part'] = part
    return mesh


def build_preview_mesh(
    model: cq.Workplane,
    settings: BlockSettings,
    tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
    angular_tolerance: float = ANGULAR_TOLERANCE,
) -> pv.PolyData:
    '''Build the classified preview mesh for a model (safe off the UI thread).

    Heavy OCC tessellation lives here so callers can run it on a worker
    thread; widget-facing ``update_mesh`` only touches VTK actors.

    The *model* must contain at least one solid (i.e. ``model.val()`` must
    return a non-None :class:`cq.Shape`).  Callers that guard empty models
    (e.g. the main-window generation code) are responsible for this
    invariant.

    Args:
        model: CadQuery Workplane whose first solid is tessellated.
        settings: BlockSettings used to build a face-part classifier.
        tolerance: Linear tessellation tolerance in mm.
        angular_tolerance: Angular tessellation tolerance in rad.

    Returns:
        Triangle PolyData with per-face part classification.
    '''
    return solid_to_mesh(
        cast(cq.Shape, model.val()),  # cadquery val() is loosely typed
        tolerance,
        angular_tolerance,
        make_face_classifier(settings),
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

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the renderer with a white background.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent=parent)
        self.set_background("white")  # type: ignore[arg-type]  # pyvista stub bug
        self._mesh: pv.PolyData | None = None
        self._mesh_actor: pv.Actor | None = None
        self._edge_mesh: pv.PolyData | None = None
        self._edge_actor: pv.Actor | None = None
        self._wire_actor: pv.Actor | None = None
        self._grid_actor: pv.CubeAxesActor | None = None
        self._mode = MODE_SOLID
        self._grid_enabled = False
        self.enable_anti_aliasing("msaa")  # type: ignore[call-arg]  # pyvista stub bug

    def update_mesh(self, mesh: pv.PolyData, settings: BlockSettings) -> None:
        '''Update the preview with a pre-tessellated mesh.

        Stores the mesh, resets edge data, enables SSAO, rebuilds actors,
        and refreshes the grid.  All VTK-side work only — no OCC
        tessellation is performed here.  The user's camera is kept across
        updates: only the first display frames the scene.  VTK re-tightens
        the clipping range on every render, so a grown block stays visible
        from the kept viewpoint.

        Args:
            mesh: Pre-tessellated PolyData (e.g. from :func:`build_preview_mesh`).
            settings: BlockSettings used for SSAO radius calculation.
        '''
        camera_position = None if self._mesh is None else self.camera_position
        self._mesh = mesh
        self._edge_mesh = None
        try:
            self.enable_ssao(  # type: ignore[call-arg]  # pyvista stub bug
                radius=max(2.0, settings.unit_size), bias=0.02
            )
        except Exception:  # noqa: BLE001 - SSAO is optional eye-candy; GL may lack it
            logger.warning("SSAO unavailable in this environment", exc_info=True)
        self._rebuild_actors()
        self._update_grid()
        if camera_position is not None:
            self.camera_position = camera_position
            self.camera_set = True
        else:
            # reset_camera: pyvista stub bug (decorated _Wrapped signature).
            self.reset_camera()  # type: ignore[call-arg]
        logger.debug(
            "Preview mesh: %d points / %d cells",
            self._mesh.n_points,
            self._mesh.n_cells,
        )

    def show_model(
        self,
        model: cq.Workplane,
        settings: BlockSettings,
        tolerance: float = PREVIEW_TESSELLATION_TOLERANCE,
        angular_tolerance: float = ANGULAR_TOLERANCE,
    ) -> None:
        '''Synchronous convenience path: tessellate and display.

        Builds the mesh via :func:`build_preview_mesh` then forwards to
        :meth:`update_mesh`.  Blocks the caller; the main window uses the
        queued async path instead.

        Args:
            model: Workplane whose first solid is displayed.
            settings: BlockSettings used to build a face-part classifier.
            tolerance: Linear tessellation tolerance in mm.
            angular_tolerance: Angular tessellation tolerance in rad.
        '''
        self.update_mesh(
            build_preview_mesh(model, settings, tolerance, angular_tolerance), settings
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
            # show_grid: pyvista stub bug (decorated _Wrapped signature).
            self._grid_actor = self.show_grid(font_size=8)  # type: ignore[call-arg]
        self.render()
