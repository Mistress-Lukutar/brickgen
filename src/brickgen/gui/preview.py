'''
File:   preview.py
Brief:  PyVista/VTK-based 3D scene preview: multi-block meshes, selection, drag.
Author: Mistress-Lukutar
Date:   2026-11-20
Version: v0.13.0
'''

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any, cast

import cadquery as cq
import numpy as np
import pyvista as pv
import vtk  # type: ignore[import-untyped]
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget
from pyvistaqt import QtInteractor

from brickgen.core.models import BlockSettings

logger = logging.getLogger(__name__)

# Tessellation tolerance for the interactive preview, mm.
PREVIEW_TESSELLATION_TOLERANCE = 0.15

BLOCK_COLOR = "#3f7fbf"
LINE_COLOR = "#111111"
# Thin black outline around selected blocks.
SELECTION_COLOR = "#000000"

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

# Quantity label anchors, relative to each block (top view: +Y is up).
LABEL_POSITIONS: tuple[str, ...] = ("left", "right", "top", "bottom")
_LABEL_MARGIN_RATIO = 0.3  # of the larger half-size, plus 1 mm floor


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

    Classification order matters: studs first (faces above the top plane
    sitting within a stud radius of a stud cell center; slope wedges also
    rise above the top plane but stay away from stud centers), then
    cylindrical tube walls near tube centers, then bottom-rim faces at Z=0,
    everything else is body.
    '''
    half_x = settings.outer_length_x / 2
    half_y = settings.outer_length_y / 2
    unit = settings.unit_size
    stud_centers = [
        ((ix + 0.5) * unit - half_x, (iy + 0.5) * unit - half_y)
        for ix, iy in sorted(settings.stud_cells_all())
    ]
    stud_r2 = (settings.pin_diameter / 2 + 1e-6) ** 2
    centers = [
        (ix * unit - half_x, iy * unit - half_y)
        for ix in range(1, settings.cells_x)
        for iy in range(1, settings.cells_y)
    ]
    r2 = (settings.tube_outer_diameter / 2) ** 2

    def classifier(face: cq.Face) -> int:
        c = face.Center()
        if c.z > settings.block_height + 1e-6:
            nearest_stud2 = min(
                ((c.x - sx) ** 2 + (c.y - sy) ** 2 for sx, sy in stud_centers),
                default=math.inf,
            )
            if nearest_stud2 <= stud_r2:
                return PART_STUD
            return PART_BODY  # slope wedge faces above the top plane
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


def add_surface_actor(plotter, mesh: pv.PolyData, color: str = BLOCK_COLOR) -> pv.Actor:
    '''Add the shaded surface actor for one block to a plotter.

    Works with any pyvista plotter (QtInteractor or off-screen Plotter).
    SSAO provides the depth cues, so the surface is tinted with the block
    color (per block on a scene, so placed blocks are visually distinct);
    ``cell_data["part"]`` is only consumed by feature-edge extraction, not
    by coloring.

    Args:
        plotter: A PyVista Plotter or QtInteractor instance.
        mesh: Triangle PolyData, optionally carrying ``cell_data["part"]``.
        color: Surface color of the block.

    Returns:
        The added actor (for caller convenience).
    '''
    return plotter.add_mesh(
        mesh,
        color=color,
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


def _ray_plane_hit(renderer, x: float, y: float, z_plane: float) -> np.ndarray | None:
    '''Intersect the display ray at (x, y) with the horizontal plane z=z_plane.

    Args:
        renderer: Active vtkRenderer (used for display -> world unprojection).
        x: Display coordinate.
        y: Display coordinate.
        z_plane: Height of the horizontal plane, mm.

    Returns:
        World point of the intersection, or None when the ray misses the
        plane (looks away from it or behind the camera).
    '''
    renderer.SetDisplayPoint(float(x), float(y), 0.0)
    renderer.DisplayToWorld()
    near = np.asarray(renderer.GetWorldPoint()[:3], dtype=np.float64)
    renderer.SetDisplayPoint(float(x), float(y), 1.0)
    renderer.DisplayToWorld()
    far = np.asarray(renderer.GetWorldPoint()[:3], dtype=np.float64)
    direction = far - near
    if abs(direction[2]) < 1e-9:
        return None
    t = (z_plane - near[2]) / direction[2]
    if t <= 0.0:
        return None
    return near + t * direction


def _settings_meta(settings: BlockSettings) -> dict:
    '''Scene metadata dict for a single-block (legacy) preview update.'''
    return {
        'pos_x': 0.0,
        'pos_y': 0.0,
        'half_x': settings.outer_length_x / 2,
        'half_y': settings.outer_length_y / 2,
        'top_z': settings.total_height,
        'quantity': 1,
    }


class _SceneInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    '''Trackball camera style that routes block picking/dragging to the widget.

    VTK events are observed on the style itself (the pattern pyvista's own
    styles use). Because a style that observes an event never gets the C++
    ``On*`` dispatch for it (``ProcessEvents`` prefers observers when
    ``HandleObservers`` is on), the widget drives the left button itself and
    calls the base ``OnLeftButtonDown``/``OnMouseMove``/``OnLeftButtonUp``
    explicitly, in balanced pairs. The base ``OnLeftButtonDown`` grab is
    what makes the follow-up move/release events reach the style on this
    platform, and the paired ``OnLeftButtonUp`` always runs in a ``finally``
    so a failing callback can never leave the grab or the ROTATE state
    armed. The C++ state machine stays untouched for the middle/right
    buttons and the wheel.
    '''

    def __init__(self, widget: PreviewWidget) -> None:  # noqa: F821
        super().__init__()
        self._widget = widget
        self.AddObserver('LeftButtonPressEvent', self._on_press)
        self.AddObserver('MouseMoveEvent', self._on_move)
        self.AddObserver('LeftButtonReleaseEvent', self._on_release)

    def _on_press(self, *_args) -> None:
        self._widget._style_press(self)

    def _on_move(self, *_args) -> None:
        self._widget._style_move(self)

    def _on_release(self, *_args) -> None:
        self._widget._style_release(self)


class PreviewWidget(QtInteractor):
    '''Interactive 3D scene preview: one shaded actor per placed block.

    Wraps a pyvista QtInteractor. Each block of the project scene renders as
    its own actor tinted with the block color; selection adds a thin black
    outline around the block. Clicking a block selects it (Ctrl+click toggles it in
    the selection), dragging selected blocks moves them along the XY plane
    (Z is fixed), and rotating the camera stays available on empty space.
    The mesh is tessellated once per block; display-mode/grid/label changes
    only touch actors, not geometry. Quantity labels render the per-block
    BOM count next to each block (View menu).
    '''

    # Emitted when scene selection changed: (frozenset of indices, primary
    # index of the last-clicked block or None).
    selection_changed = Signal(object)
    # Emitted when a drag committed: total (dx, dy) of the dragged selection.
    blocks_dragged = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the renderer with a white background.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent=parent)
        self.set_background("white")  # type: ignore[arg-type]  # pyvista stub bug
        self._scene_meshes: list[pv.PolyData] = []
        self._scene_colors: list[str] = []
        self._scene_meta: list[dict] = []
        self._block_actors: list[pv.Actor] = []
        self._extra_actors: list[pv.Actor] = []
        self._outline_actors: dict[int, pv.Actor] = {}
        self._qty_actors: dict[int, Any] = {}
        self._actor_map: dict[pv.Actor, int] = {}
        self._selection: set[int] = set()
        self._primary: int | None = None
        self._qty_enabled = False
        self._qty_position = "right"
        self._mesh: pv.PolyData | None = None
        self._grid_actor: pv.CubeAxesActor | None = None
        self._mode = MODE_SOLID
        self._grid_enabled = False
        # Drag/pick transient state. The left button is driven by our
        # callbacks in balanced On* pairs (see _SceneInteractorStyle); the
        # always-running release cleanup keeps VTK's state machine clean.
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.01)
        self._press_ctrl = False
        self._press_block: int | None = None
        self._left_state_armed = False
        self._rotated = False
        self._drag_block: int | None = None
        self._drag_plane_z = 0.0
        self._drag_origin: tuple[float, float] | None = None
        self._drag_start: dict[int, tuple[float, float]] = {}
        self._drag_delta: tuple[float, float] | None = None
        self.enable_anti_aliasing("msaa")  # type: ignore[call-arg]  # pyvista stub bug
        self._install_scene_style()

    def _install_scene_style(self) -> None:
        '''Replace the default trackball style with the scene-aware one.'''
        style = _SceneInteractorStyle(self)
        try:
            self.iren.style = style  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - fall back to the raw VTK setter
            logger.warning("pyvista style property unavailable, using raw setter")
            self.iren.interactor.SetInteractorStyle(style)  # type: ignore[union-attr]

    # ------------------------------------------------------------- scene API
    def update_scene(
        self, meshes: list[pv.PolyData], colors: list[str], meta: list[dict]
    ) -> None:
        '''Replace the whole scene with per-block meshes.

        Args:
            meshes: One pre-tessellated PolyData per block (each built
                centered on the origin; placement comes from the meta
                ``pos_x``/``pos_y`` via actor positions).
            colors: One hex color per block, aligned with *meshes*.
            meta: One dict per block with keys ``pos_x``, ``pos_y``,
                ``half_x``, ``half_y``, ``top_z`` and ``quantity``.

        Raises:
            ValueError: If the lists are not aligned.
        '''
        if not (len(meshes) == len(colors) == len(meta)):
            raise ValueError("meshes/colors/meta must be aligned lists")
        had_scene = bool(self._block_actors)
        camera_position = None if not had_scene else self.camera_position
        self._scene_meshes = list(meshes)
        self._scene_colors = list(colors)
        self._scene_meta = [dict(item) for item in meta]
        self._mesh = meshes[0] if meshes else None  # legacy single-mesh attr
        self._selection = {i for i in self._selection if i < len(meshes)}
        if self._primary is not None and self._primary >= len(meshes):
            self._primary = None
        try:
            radius = max(2.0, *(m['half_x'] + m['half_y'] for m in self._scene_meta))
            self.enable_ssao(radius=radius, bias=0.02)  # type: ignore[call-arg]
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
            "Preview scene: %d blocks / %d points total",
            len(self._scene_meshes),
            sum(m.n_points for m in self._scene_meshes),
        )

    def update_mesh(self, mesh: pv.PolyData, settings: BlockSettings) -> None:
        '''Legacy single-block entry point: display one mesh at the origin.

        Args:
            mesh: Pre-tessellated PolyData (e.g. from
                :func:`build_preview_mesh`).
            settings: BlockSettings used for SSAO radius and label anchors.
        '''
        self.update_scene([mesh], [BLOCK_COLOR], [_settings_meta(settings)])

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
        '''Show or hide the metric grid around the scene.

        Args:
            enabled: True to display the grid.
        '''
        self._grid_enabled = enabled
        self._update_grid()

    def set_selection(self, selection: set[int], primary: int | None = None) -> None:
        '''Replace the scene selection programmatically (no signal).

        Args:
            selection: Set of block indices to frame.
            primary: Last-clicked block index; None clears the primary.
        '''
        self._selection = {i for i in selection if i < len(self._scene_meshes)}
        self._primary = (
            primary if primary is not None and primary in self._selection else None
        )
        self._update_selection_outlines()
        self.render()

    def selected_blocks(self) -> set[int]:
        '''Set of currently selected block indices.'''
        return set(self._selection)

    def set_block_position(self, index: int, x: float, y: float) -> None:
        '''Place block *index* at the scene position (x, y) (UI thread).

        Only touches actors and labels - no geometry is rebuilt.

        Args:
            index: Block index in the current scene.
            x: New position of the block center along X, mm.
            y: New position of the block center along Y, mm.
        '''
        if not 0 <= index < len(self._scene_meta):
            return
        self._scene_meta[index]['pos_x'] = float(x)
        self._scene_meta[index]['pos_y'] = float(y)
        self._position_actor(index, float(x), float(y))
        self._refresh_quantity_labels()
        self.render()

    def set_block_colors(self, colors: dict[int, str]) -> None:
        '''Recolor the given blocks without rebuilding geometry.

        Args:
            colors: Mapping of block index to hex color.
        '''
        for index, color in colors.items():
            if 0 <= index < len(self._block_actors):
                self._block_actors[index].prop.color = color
        self.render()

    def set_block_quantities(self, quantities: dict[int, int]) -> None:
        '''Update per-block BOM quantities without rebuilding geometry.

        Args:
            quantities: Mapping of block index to the new quantity.
        '''
        for index, quantity in quantities.items():
            if 0 <= index < len(self._scene_meta):
                self._scene_meta[index]['quantity'] = int(quantity)
        self._refresh_quantity_labels()
        self.render()

    def set_quantity_labels(self, enabled: bool, position: str) -> None:
        '''Show per-block BOM quantity labels and set their anchor side.

        Args:
            enabled: True to render the labels.
            position: One of ``LABEL_POSITIONS`` (relative to each block).
        '''
        self._qty_enabled = enabled
        if position in LABEL_POSITIONS:
            self._qty_position = position
        self._refresh_quantity_labels()
        self.render()

    # ------------------------------------------------------------- internals
    def _position_actor(self, index: int, x: float, y: float) -> None:
        '''Move the surface actor and its outline of block *index*.'''
        if 0 <= index < len(self._block_actors):
            self._block_actors[index].SetPosition(float(x), float(y), 0.0)
        outline = self._outline_actors.get(index)
        if outline is not None:
            outline.SetPosition(float(x), float(y), 0.0)

    def _clear_scene_actors(self) -> None:
        '''Remove every surface/edge/wireframe/outline/label actor.'''
        for actor in (
            *self._block_actors,
            *self._extra_actors,
            *self._outline_actors.values(),
            *self._qty_actors.values(),
        ):
            try:
                self.remove_actor(actor)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._block_actors.clear()
        self._extra_actors.clear()
        self._outline_actors.clear()
        self._qty_actors.clear()
        self._actor_map.clear()

    def _rebuild_actors(self) -> None:
        '''Remove and re-add every per-block actor for the active mode.'''
        self._clear_scene_actors()
        if not self._scene_meshes:
            return
        for index, mesh in enumerate(self._scene_meshes):
            surface = add_surface_actor(self, mesh, color=self._scene_colors[index])
            surface.SetPickable(True)
            position = self._scene_meta[index]
            surface.SetPosition(position['pos_x'], position['pos_y'], 0.0)
            self._block_actors.append(surface)
            self._actor_map[surface] = index
            self._add_mode_actors(index, mesh)
        self._update_selection_outlines()
        self._refresh_quantity_labels()

    def _add_mode_actors(self, index: int, mesh: pv.PolyData) -> None:
        '''Add the display-mode decoration actors of one block.'''
        if self._mode == MODE_SOLID_FEATURE_EDGES:
            edges = semantic_edges(mesh)
            if edges.n_points == 0:
                return
            actor = self.add_mesh(
                edges,
                color=LINE_COLOR,
                line_width=2.0,
                lighting=False,
                render_lines_as_tubes=True,
                pickable=False,
            )
            self._extra_actors.append(actor)
        elif self._mode == MODE_SOLID_WIREFRAME:
            # Tubes give the lines real volume so they win the depth test
            # against the surface (plain polylines z-fight with the mesh).
            actor = self.add_mesh(
                mesh,
                color=LINE_COLOR,
                style="wireframe",
                line_width=2.5,
                lighting=False,
                render_lines_as_tubes=True,
                pickable=False,
            )
            self._extra_actors.append(actor)
        elif self._mode == MODE_WIREFRAME:
            actor = self.add_mesh(
                mesh,
                color=LINE_COLOR,
                style="wireframe",
                line_width=1.5,
                lighting=False,
                pickable=False,
            )
            self._extra_actors.append(actor)

    def _update_selection_outlines(self) -> None:
        '''Rebuild the thin black outline actors around selected blocks.'''
        for actor in self._outline_actors.values():
            try:
                self.remove_actor(actor)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._outline_actors.clear()
        for index in sorted(self._selection):
            if not 0 <= index < len(self._scene_meshes):
                continue
            actor = self.add_mesh(
                self._scene_meshes[index].outline(),
                color=SELECTION_COLOR,
                line_width=1.5,
                lighting=False,
                pickable=False,
            )
            position = self._scene_meta[index]
            actor.SetPosition(position['pos_x'], position['pos_y'], 0.0)
            self._outline_actors[index] = actor

    def _label_point(self, index: int) -> np.ndarray:
        '''World anchor point of the quantity label of block *index*.'''
        meta = self._scene_meta[index]
        margin = max(meta['half_x'], meta['half_y']) * _LABEL_MARGIN_RATIO + 1.0
        x, y = meta['pos_x'], meta['pos_y']
        if self._qty_position == 'left':
            return np.array([x - meta['half_x'] - margin, y, meta['top_z'] + 0.5])
        if self._qty_position == 'right':
            return np.array([x + meta['half_x'] + margin, y, meta['top_z'] + 0.5])
        if self._qty_position == 'top':
            return np.array([x, y + meta['half_y'] + margin, meta['top_z'] + 0.5])
        return np.array([x, y - meta['half_y'] - margin, meta['top_z'] + 0.5])

    def _refresh_quantity_labels(self) -> None:
        '''Rebuild the per-block BOM quantity label actors.'''
        for actor in self._qty_actors.values():
            try:
                self.remove_actor(actor)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._qty_actors.clear()
        if not self._qty_enabled or not self._scene_meshes:
            return
        for index, meta in enumerate(self._scene_meta):
            actor = self.add_point_labels(
                np.array([self._label_point(index)]),
                [f"×{meta['quantity']}"],
                always_visible=True,
                show_points=False,
                shape=None,
                font_size=13,
                text_color='black',
                name=f'quantity_{index}',
            )
            self._qty_actors[index] = actor

    def _update_grid(self) -> None:
        '''Recreate or remove the metric grid actor.'''
        if self._grid_actor is not None:
            self.remove_actor(self._grid_actor)
            self._grid_actor = None
        if self._grid_enabled and self._mesh is not None:
            # show_grid: pyvista stub bug (decorated _Wrapped signature).
            self._grid_actor = self.show_grid(font_size=8)  # type: ignore[call-arg]
        self.render()

    # ------------------------------------------------- scene input (style cb)
    def _pick_block(self, style: _SceneInteractorStyle) -> tuple[int | None, list]:
        '''Block index and world point under the current cursor, or (None, []).

        Args:
            style: The interactor style the event came from.
        '''
        x, y = style.GetInteractor().GetEventPosition()
        if not self._picker.Pick(x, y, 0, self.renderer):
            return None, []
        index = self._actor_map.get(self._picker.GetActor())
        if index is None:
            return None, []
        return index, list(self._picker.GetPickPosition())

    def _emit_selection(self) -> None:
        '''Notify about the current selection (set + primary index).'''
        self.selection_changed.emit((frozenset(self._selection), self._primary))

    def _toggle_selection(self, index: int) -> None:
        '''Add/remove *index* from the selection and notify.'''
        if index in self._selection:
            remaining = self._selection - {index}
            self._selection = remaining
            self._primary = max(remaining) if remaining else None
        else:
            self._selection.add(index)
            self._primary = index
        self._update_selection_outlines()
        self._emit_selection()

    def _select_only(self, index: int) -> None:
        '''Replace the selection with the single block *index* and notify.'''
        self._selection = {index}
        self._primary = index
        self._update_selection_outlines()
        self._emit_selection()

    def _arm_drag(self, index: int, pick_position: list) -> None:
        '''Arm an XY drag of the whole selection starting at *pick_position*.

        Args:
            index: Block the press landed on (drag anchor).
            pick_position: World point under the cursor (defines the drag
                plane height so the block follows the cursor naturally).
        '''
        self._drag_block = index
        self._drag_plane_z = float(pick_position[2])
        self._drag_origin = (float(pick_position[0]), float(pick_position[1]))
        self._drag_start = {
            i: (m['pos_x'], m['pos_y'])
            for i, m in enumerate(self._scene_meta)
            if i in self._selection
        }
        self._drag_delta = None

    def _cancel_drag_visual(self) -> None:
        '''Restore actor positions after a non-committed drag.'''
        for index, (x, y) in self._drag_start.items():
            self._position_actor(index, x, y)
        self.render()

    def _style_press(self, style: _SceneInteractorStyle) -> None:
        '''Handle a left-button press: pick a block or arm a camera orbit.

        Every press path ends with ``style.OnLeftButtonDown()``: its grab is
        what makes the follow-up move/release events reach the style on this
        platform, and the paired ``OnLeftButtonUp`` runs unconditionally on
        release (in a ``finally``), so the VTK state machine stays balanced.

        Plain press on a block selects it immediately (unless already in a
        multi-selection) and arms an XY drag of the whole selection.
        Ctrl+press on an unselected block defers its toggle to the release
        (a Ctrl+drag stays a no-op, mirroring the layout editor). A press on
        empty space arms the native camera rotate.
        '''
        iren = style.GetInteractor()
        self._press_ctrl = bool(iren.GetControlKey())
        self._left_state_armed = False
        self._rotated = False
        self._drag_delta = None
        index, pick_position = self._pick_block(style)
        self._press_block = index
        if index is None:
            self._drag_block = None
            style.OnLeftButtonDown()  # grab + arm the native rotate state
            self._left_state_armed = True
            return
        if self._press_ctrl and index not in self._selection:
            # Deferred toggle only: grab the events (via the base press) but
            # suppress the move handling below, so Ctrl+drag is a no-op.
            style.OnLeftButtonDown()
            self._left_state_armed = True
            return
        if not self._press_ctrl and index not in self._selection:
            self._select_only(index)
        self._arm_drag(index, pick_position)
        style.OnLeftButtonDown()
        self._left_state_armed = True

    def _style_move(self, style: _SceneInteractorStyle) -> None:
        '''Drag the selection along the XY plane or rotate the camera.

        While a block drag is armed the cursor ray is intersected with the
        horizontal plane of the grab point and every selected block follows
        the cursor; the model commit happens on release. Otherwise the move
        goes to the native trackball (camera rotate/spin/pan), and a
        deferred Ctrl+click stays a no-op until its release.
        '''
        if self._drag_block is not None and self._drag_origin is not None:
            x, y = style.GetInteractor().GetEventPosition()
            hit = _ray_plane_hit(self.renderer, x, y, self._drag_plane_z)
            if hit is None:
                return
            dx = float(hit[0] - self._drag_origin[0])
            dy = float(hit[1] - self._drag_origin[1])
            if (
                self._drag_delta is not None
                and abs(dx - self._drag_delta[0]) < 1e-9
                and abs(dy - self._drag_delta[1]) < 1e-9
            ):
                return
            self._drag_delta = (dx, dy)
            for index, (sx, sy) in self._drag_start.items():
                self._position_actor(index, sx + dx, sy + dy)
            self.render()
            return
        if self._press_ctrl and self._press_block is not None:
            return  # deferred Ctrl+click: no spin while waiting for release
        style.OnMouseMove()
        if self._left_state_armed:
            self._rotated = True

    def _style_release(self, style: _SceneInteractorStyle) -> None:
        '''Commit a finished drag or apply the click selection semantics.

        A drag over a block emits :attr:`blocks_dragged` with the total
        delta; the window applies it to the project and pushes exact
        positions back. Without movement: plain click selects the pressed
        block, Ctrl+click toggles it, and a click on empty space (without
        camera rotation) clears the selection. The paired
        ``OnLeftButtonUp`` always runs - a failure above must never leave
        the grab or the ROTATE state armed.
        '''
        try:
            rotated = self._rotated
            moved = (
                self._drag_delta is not None
                and abs(self._drag_delta[0]) + abs(self._drag_delta[1]) > 1e-9
            )
            if self._drag_block is not None:
                if moved and self._drag_delta is not None:
                    dx, dy = self._drag_delta
                    # Keep the visual until the window pushes exact positions.
                    for index, (sx, sy) in self._drag_start.items():
                        self._position_actor(index, sx + dx, sy + dy)
                    self.blocks_dragged.emit(dx, dy)
                else:
                    self._cancel_drag_visual()
                    if self._press_ctrl:
                        self._toggle_selection(self._drag_block)
                    else:
                        self._select_only(self._drag_block)
            elif self._press_block is not None and self._press_ctrl:
                if not rotated:
                    self._toggle_selection(self._press_block)
            elif self._press_block is None and not rotated:
                # Plain click on empty space (no rotation): clear the selection.
                self._select_clear()
        finally:
            if self._left_state_armed:
                style.OnLeftButtonUp()  # EndRotate + ReleaseFocus
            self._drag_block = None
            self._drag_delta = None
            self._drag_origin = None
            self._drag_start = {}
            self._press_block = None
            self._left_state_armed = False
            self._rotated = False

    def _select_clear(self) -> None:
        '''Clear the selection and notify.'''
        if not self._selection:
            return
        self._selection = set()
        self._primary = None
        self._update_selection_outlines()
        self._emit_selection()
        self.render()
