'''
File:   preview.py
Brief:  PyVista/VTK-based 3D scene preview: multi-block meshes, selection, navigation.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import Any, cast

import cadquery as cq
import numpy as np
import pyvista as pv
import vtk  # type: ignore[import-untyped]
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QCursor,
    QKeyEvent,
    QMouseEvent,
    QSurfaceFormat,
    QWheelEvent,
)
from PySide6.QtWidgets import QRubberBand, QWidget
from pyvistaqt import QtInteractor

from brickgen.core.models import BlockSettings
from brickgen.gui.keys import event_key
from brickgen.gui.move_gizmo import GIZMO_LENGTH_STEPS, MoveGizmo

logger = logging.getLogger(__name__)


def _use_compatibility_gl_profile() -> None:
    '''Switch the pyvistaqt GL context to a compatibility profile.

    pyvistaqt pins a desktop-GL 3.2 *core* process-wide QSurfaceFormat
    (``rwi._default_format``) and forces ``MaximumHardwareLineWidth`` to 1
    when its widget initializes. In a core context ``glLineWidth`` above
    1.0 is illegal, so VTK renders every wider line through its wide-line
    geometry shader — whose quad expansion draws fat, camera-angle
    dependent bands. A compatibility profile keeps the native
    constant-width rasterization (this project is Windows-first: desktop
    GL always offers compatibility contexts). :meth:`PreviewWidget.
    initializeGL` re-applies the hardware line-width limit afterwards.
    '''
    try:
        from pyvistaqt import rwi as _rwi

        original = _rwi._default_format  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - pyvistaqt layout changed: keep defaults
        return

    def compatibility_default_format() -> QSurfaceFormat:
        fmt = original()
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        return fmt

    _rwi._default_format = compatibility_default_format  # type: ignore[attr-defined]


_use_compatibility_gl_profile()

# Tessellation tolerance for the interactive preview, mm.
PREVIEW_TESSELLATION_TOLERANCE = 0.15

BLOCK_COLOR = "#3f7fbf"
LINE_COLOR = "#111111"
# Thin bright outline around selected blocks (readable on the dark viewport).
SELECTION_COLOR = "#ff9d2e"
# Viewport / thumbnail background (dark, 3D-editor style).
VIEWPORT_BG = "#2e3338"
# Floor grid: minor lines, every-10th major lines, X/Y axes and axis labels.
GRID_MINOR_COLOR = "#454b53"
GRID_MAJOR_COLOR = "#565e68"
GRID_AXIS_X_COLOR = "#d86a6a"
GRID_AXIS_Y_COLOR = "#6abf83"
GRID_LABEL_X_COLOR = "#cf8d8d"
GRID_LABEL_Y_COLOR = "#8dcfa4"
# Axis-lock guide-line colors (Z axis, blue-ish to match the muted palette).
GRID_AXIS_Z_COLOR = "#6a86d8"
# Guide line half-length in grid steps (lines run BOTH directions from center).
MOVE_GUIDE_LENGTH_STEPS = 8
# Mapping of axis names to their color strings.
AXIS_COLORS: dict[str, str] = {
    "x": GRID_AXIS_X_COLOR,
    "y": GRID_AXIS_Y_COLOR,
    "z": GRID_AXIS_Z_COLOR,
}
# Unit vectors for each axis (float64).
AXIS_VECTORS: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}

# Turntable navigation: the model follows the cursor - horizontal drags
# turn it around the world Z axis, vertical drags tip it up/down. Degrees
# covered by a full-window drag, matched between the axes for a uniform feel.
ORBIT_DEGREES_PER_AXIS = 200.0
# LMB press-to-move distance (manhattan, px) before the rubber band appears
# (and the drag distance a block move must exceed to count as a move, not a
# click).
RUBBER_BAND_THRESHOLD_PX = 4
# World up vector of the scene (blocks sit on the XY plane).
WORLD_UP = np.array([0.0, 0.0, 1.0])
# Floor grid: static plane at z=0, centered on the world origin. The line
# plane sits a hair below z=0 so it never z-fights with block bottoms.
GRID_Z_OFFSET = -0.05
# The colored axes sit slightly ABOVE the line plane: coplanar lines
# z-fight into dashed bands when the plane is seen at a grazing angle.
GRID_AXIS_Z_OFFSET = -0.02
# World-anchored mm labels float this high above the axes (screen overlays:
# z only matters for the 3D projection coherence).
GRID_LABEL_Z = 0.2
# Static half-extent of the floor grid, in cell steps.
GRID_EXTENT_CELLS = 1000
# mm coordinate labels span only ±WINDOW cells around the origin. Labels
# render at a constant screen size, so letting them reach the full extent
# crowds the far ones into wide bands at grazing view angles.
GRID_LABEL_WINDOW_CELLS = 50
# Zoom-adaptive axis labels: adjacent labels keep at least this on-screen
# distance. The label step is the smallest power-of-two multiple of the
# grid step whose spacing clears the threshold at the current camera.
GRID_LABEL_MIN_SPACING_PX = 40.0
GRID_LABEL_MAX_MULTIPLIER = 256
GRID_MAJOR_EVERY = 10
# Widest line the app ever renders (grid, axes, wireframe, feature edges
# stay at or below this). It gates VTK's native ``glLineWidth`` path: see
# ``PreviewWidget.initializeGL``.
MAX_HARDWARE_LINE_WIDTH = 8.0
# Snap: block move drags and G-grabs round to the grid step (Grid menu).
# Floor-plane moves require the cursor ray to dip at least this sine below
# the horizon (~5.7 deg); flatter views would explode pixel deltas.
_FLOOR_RAY_MIN_SIN = 0.1

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
# Fixed gap between the quantity label anchor and the block face it
# adjoins, mm - the same for every block size.
_LABEL_MARGIN_MM = 1.5


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
        reset_camera=False,
        render=False,  # batched: callers render once after the actor batch
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


def assemble_variant_mesh(
    base_mesh: pv.PolyData,
    rot_z: int,
    flipped: bool,
    lift_z: float,
) -> pv.PolyData:
    '''Build the assemble-tab mesh variant of one piece orientation.

    The piece turns ``rot_z`` quarter turns (90 degrees each) about Z,
    then flipped pieces turn 180 degrees about X (pins down) and lift so
    the geometry stays on ``z >= 0`` — the same transform order as the
    print-placement variants and the placement export.

    Note: the pyvista transform methods must have their RESULT assigned —
    newer pyvista releases return a transformed copy instead of mutating
    in place, so calling them without assignment silently renders the
    untransformed mesh.

    Args:
        base_mesh: The untransformed block mesh (centered at origin).
        rot_z: Quarter-turn count about Z (0..3).
        flipped: Rotate 180 degrees about X (pins down).
        lift_z: Z translation applied after the flip (``total_height``).

    Returns:
        The transformed mesh copy (the input mesh is left untouched).
    '''
    variant = base_mesh.copy()
    angle = (int(rot_z) % 4) * 90.0
    if angle:
        variant = variant.rotate_z(angle)
    if flipped:
        # 180 deg about X: pins down; lift onto z >= 0.
        variant = variant.rotate_x(180.0)
        variant = variant.translate((0.0, 0.0, lift_z))
    return variant


def piece_variant_mesh(
    base_mesh: pv.PolyData,
    rotated90: bool,
    flipped: bool,
    lift_z: float,
) -> pv.PolyData:
    '''Build the print-placement mesh variant of one piece orientation.

    Rotated pieces turn 90 degrees about Z (footprint swaps, matching the
    export); flipped pieces turn 180 degrees about X (pins down) and are
    lifted so the geometry stays on ``z >= 0``.

    Args:
        base_mesh: The untransformed block mesh (centered at origin).
        rotated90: Rotate 90 degrees about Z.
        flipped: Rotate 180 degrees about X (pins down).
        lift_z: Z translation applied after the flip (``total_height``).

    Returns:
        The transformed mesh copy (the input mesh is left untouched).
    '''
    return assemble_variant_mesh(base_mesh, 1 if rotated90 else 0, flipped, lift_z)


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


def orbit_offset(
    offset: np.ndarray, up: np.ndarray, azimuth_deg: float, elevation_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    '''Turntable-rotate the camera offset and view up around the scene.

    The azimuth component rotates around the world Z axis through the focal
    point and the elevation component around the camera's own right axis.
    Both rotations are rigid - the view up is carried along with the
    position - so the camera never rolls relative to the world (a roll-free
    up stays roll-free under both rotations) and crossing the pole stays
    continuous instead of flipping: at the pole the horizontal drag simply
    spins the view. Unlike the native VTK trackball, which azimuths around
    the tilted camera up and orthogonalizes the drifted view up in place,
    the orbit radius and the roll-free state are preserved exactly.

    Args:
        offset: Vector from the focal point to the camera position.
        up: Camera view up vector (kept perpendicular to the view direction).
        azimuth_deg: Horizontal rotation around world Z, positive
            counterclockwise viewed from above (+Z), degrees.
        elevation_deg: Vertical rotation (positive raises the camera),
            degrees.

    Returns:
        Tuple of the rotated (offset, up) vectors.
    '''
    offset = np.asarray(offset, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    radius = float(np.linalg.norm(offset))
    if radius < 1e-9:
        return offset.copy(), up.copy()
    up = up / max(float(np.linalg.norm(up)), 1e-12)

    def rotated(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (
            vector * cos_a
            + np.cross(axis, vector) * sin_a
            + axis * float(np.dot(axis, vector)) * (1.0 - cos_a)
        )

    # Azimuth around the world Z; the up rotates along, which keeps the
    # roll-free state (both vectors turn rigidly about the world axis).
    azimuth = math.radians(azimuth_deg)
    offset = rotated(offset, WORLD_UP, azimuth)
    up = rotated(up, WORLD_UP, azimuth)

    # Elevation around the camera's own right axis (view direction x up):
    # well defined even at the pole, unlike the world-up cross product.
    view_dir = -offset / radius
    right = np.cross(view_dir, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:  # degenerate up: fall back to a horizontal axis
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm
    elevation = math.radians(-elevation_deg)  # positive angle raises
    offset = rotated(offset, right, elevation)
    up = rotated(up, right, elevation)
    return offset, up


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


class PreviewWidget(QtInteractor):
    '''Interactive 3D scene preview: one shaded actor per placed block.

    Wraps a pyvista QtInteractor. Each block of the project scene renders as
    its own actor tinted with the block color; selection adds a thin bright
    outline around the block. The mouse scheme is the common 3D-editor one:
    the left button pressed on a block picks it up (a drag moves the whole
    selection on the floor plane, a release without a real move falls back
    to the click semantics), a left drag on the background draws a rubber
    band selecting every block fully inside it, Ctrl+click toggles a block;
    the middle button orbits the camera turntable style (the model follows
    the cursor) and Shift+middle pans; the right button and the wheel keep
    the native VTK zoom. LMB/MMB events are consumed on the Qt level, so the
    VTK interactor never arms its native rotate/pan/spin states. Blocks are
    also moved by the G (grab) hotkey: the selection follows the cursor on
    the floor plane until a click/Enter confirms or Esc/right-click cancels.
    When snap is enabled every move rounds to the grid step. The floor grid
    is a static XY plane at the world origin with colored X/Y axes and mm
    coordinate labels; its cell scale is ``unit_size * scale``. The mesh is
    tessellated once
    per block; display-mode/grid/label changes only touch actors, not
    geometry. Quantity labels render the per-block BOM count next to each
    block (View menu).
    '''

    # Emitted when scene selection changed: (frozenset of indices, primary
    # index of the last-clicked block or None).
    selection_changed = Signal(object)
    # Emitted after a move drag or a grab move committed: final scene
    # positions {block index: (x, y)} of every moved block.
    blocks_moved = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        '''Create the renderer with a dark background.

        Args:
            parent: Optional parent widget.
        '''
        super().__init__(parent=parent)
        self.set_background(VIEWPORT_BG)  # type: ignore[arg-type]  # pyvista stub bug
        self._scene_meshes: list[pv.PolyData] = []
        self._scene_colors: list[str] = []
        self._scene_meta: list[dict] = []
        # One list of actors (surface + mode decorations) per block, aligned
        # with the scene: everything of a block follows its position.
        self._block_actors: list[list[pv.Actor]] = []
        self._outline_actors: dict[int, pv.Actor] = {}
        self._qty_actors: dict[int, Any] = {}
        # Anchor-point PolyData of each quantity label pipeline (keyed like
        # _qty_actors): drag updates only reassign these points instead of
        # rebuilding the actors (see _move_quantity_labels).
        self._qty_points: dict[int, pv.PolyData] = {}
        self._actor_map: dict[pv.Actor, int] = {}
        self._selection: set[int] = set()
        self._primary: int | None = None
        self._qty_enabled = False
        self._qty_position = "right"
        self._mesh: pv.PolyData | None = None
        self._mode = MODE_SOLID
        # Floor grid state: enabled flag, unit size of the shared block
        # geometry (set by the window), cell scale factor.
        self._grid_enabled = False
        self._grid_unit = 4.0
        self._grid_scale = 0.5
        self._grid_snap = False
        self._grid_actors: list[pv.Actor] = []
        # Zoom-adaptive axis label actors, rebuilt whenever the per-axis
        # spacing multipliers flip (X, Y) during camera navigation.
        self._grid_label_actors: list[pv.Actor] = []
        self._label_multipliers: tuple[int, int] = (1, 1)
        # SSAO state for incremental enable/disable.
        self._ssao_radius: float | None = None
        # Selection/navigation transient state. The left button picks up a
        # block (move drag) or draws the rubber band on the background, the
        # middle button orbits, Shift+middle pans; the right button and the
        # wheel stay native VTK zoom. G starts the keyboard grab move.
        self._picker = vtk.vtkCellPicker()
        # CellPicker tolerance is a FRACTION of the viewport diagonal, not
        # world units: 0.01 thickens the pick ray into a cylinder ~1% of
        # the visible scene diagonal wide (tens of px), so clicks outside
        # a block's silhouette still selected it. Keep it sub-pixel: the
        # ray itself must pass through the mesh triangles.
        self._picker.SetTolerance(1e-05)
        self._left_down = False
        self._press_ctrl = False
        self._press_pos = QPoint()
        self._rubbing = False
        self._rubber: QRubberBand | None = None
        self._orbiting = False
        self._panning = False
        self._last_mouse = QPoint()
        # Active move drag (LMB pressed on a block, or the G grab): original
        # positions of the moved blocks, the world anchor of the press on
        # the floor plane, and whether a real move happened. *keyboard*
        # marks the G grab (confirm/cancel via keys/clicks); the mouse drag
        # commits on release or falls back to the click semantics.
        self._grabbing = False
        self._grab_keyboard = False
        self._grab_moved = False
        self._drag_from_axis = False
        self._grab_press_pos = QPoint()
        self._grab_orig: dict[int, tuple[float, float]] = {}
        self._grab_anchor: np.ndarray | None = None
        # Axis-lock state (X/Y/Z lock movement to a single axis or plane).
        # Move gizmo (View menu): draggable arrow cross at the selection center.
        self._gizmo_enabled = False
        self._gizmo: MoveGizmo | None = None
        self._gizmo_step: float | None = None
        # Axis-lock state (X/Y/Z lock movement to a single axis or plane).
        self._axis_lock: str | None = None
        self._axis_plane = False
        self._constraint_anchor: np.ndarray | None = None
        self._constraint_origin: np.ndarray | None = None
        self._constraint_zero: np.ndarray | None = None
        self._last_grab_pos: QPoint | None = None
        self._guide_actors: list[pv.Actor] = []
        self._guides_key: frozenset[str] | None = None
        # Keyboard grabs (G) need the viewport to take and keep focus.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.enable_anti_aliasing("msaa")  # type: ignore[call-arg]  # pyvista stub bug
        # Camera moves change the on-screen grid spacing: watch VTK-level
        # interactions (e.g. native right-button zoom) and refresh the
        # axis labels when their spacing multiplier flips.
        self.iren.add_observer("InteractionEvent", self._on_camera_interaction)

    # ------------------------------------------------------------- scene API
    def update_scene(
        self,
        meshes: list[pv.PolyData | None],
        colors: list[str],
        meta: list[dict],
        changed: list[bool] | None = None,
    ) -> None:
        '''Replace the whole scene with per-block meshes.

        Args:
            meshes: One pre-tessellated PolyData (or ``None``) per block.
                A ``None`` entry means "keep the mesh currently stored at
                this index" (ignored when there is no existing scene).
            colors: One hex color per block, aligned with *meshes*.
            meta: One dict per block with keys ``pos_x``, ``pos_y``,
                ``half_x``, ``half_y``, ``top_z`` and ``quantity``.
            changed: Optional bool mask; ``True`` at index *i* means the
                actor for block *i* must be rebuilt.  ``None`` (or length
                / count mismatch) triggers a full rebuild of every block.

        Raises:
            ValueError: If the lists are not aligned.
        '''
        if not (len(meshes) == len(colors) == len(meta)):
            raise ValueError("meshes/colors/meta must be aligned lists")
        had_scene = bool(self._block_actors)
        camera_position = None if not had_scene else self.camera_position

        # Resolve None entries: keep existing mesh when no old scene exists.
        resolved_meshes: list[pv.PolyData] = []
        for i, m in enumerate(meshes):
            if m is not None:
                resolved_meshes.append(m)
            elif i < len(self._scene_meshes):
                resolved_meshes.append(self._scene_meshes[i])
            else:
                resolved_meshes.append(pv.PolyData())  # empty sentinel
        self._scene_meshes = resolved_meshes
        self._scene_colors = list(colors)
        self._scene_meta = [dict(item) for item in meta]
        self._mesh = next(
            (m for m in self._scene_meshes if m is not None and m.n_points > 0), None
        )
        self._selection = {i for i in self._selection if i < len(meshes)}
        if self._primary is not None and self._primary >= len(meshes):
            self._primary = None

        # SSAO: only enable when the radius changes (an empty scene keeps
        # the last configuration instead of warning on every generation).
        try:
            if self._scene_meta:
                radius = max(
                    2.0, *(m['half_x'] + m['half_y'] for m in self._scene_meta)
                )
                if self._ssao_radius != radius:
                    self._ssao_radius = radius
                    self.enable_ssao(radius=radius, bias=0.02)  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001 - SSAO is optional eye-candy; GL may lack it
            logger.warning("SSAO unavailable in this environment", exc_info=True)

        self._rebuild_actors(changed)
        if camera_position is not None:
            self.camera_position = camera_position
            self.camera_set = True
        else:
            # reset_camera: pyvista stub bug (decorated _Wrapped signature).
            self.reset_camera()  # type: ignore[call-arg]
            # Mark the camera as deliberately placed: pyvista's add_mesh
            # resets the camera on later actor additions whenever camera_set
            # is False, which would yank the view on the first selection.
            self.camera_set = True
        self._refresh_grid_labels()
        self._update_gizmo()
        # Single render for the whole batch: the actor add/remove calls run
        # with render=False (one pyvista render per actor would stall the UI
        # on large scenes), so the batch needs this explicit draw.
        self.render()
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
        '''Show or hide the floor grid.

        Args:
            enabled: True to display the grid.
        '''
        self._grid_enabled = enabled
        self._update_grid()

    def set_grid_unit(self, unit_size: float) -> None:
        '''Set the shared block unit size the grid cells are derived from.

        Args:
            unit_size: Unit size in mm (grid step = unit size * scale).
        '''
        if unit_size > 0 and unit_size != self._grid_unit:
            self._grid_unit = float(unit_size)
            self._update_grid()

    def set_grid_scale(self, scale: float) -> None:
        '''Set the grid cell scale factor (Grid menu).

        Args:
            scale: Cells are drawn every ``unit_size * scale`` mm.
        '''
        if scale > 0 and scale != self._grid_scale:
            self._grid_scale = float(scale)
            self._update_grid()
            self._update_gizmo()

    def set_snap_enabled(self, enabled: bool) -> None:
        '''Enable or disable snap-to-grid for block moves (Grid menu).

        Args:
            enabled: True rounds moves to the grid step multiples.
        '''
        self._grid_snap = bool(enabled)

    def grid_step(self) -> float:
        '''Current grid step in mm (unit size * cell scale).'''
        return self._grid_unit * self._grid_scale

    # ------------------------------------------------------- axis-lock helpers
    def active_axes(self) -> frozenset[str]:
        '''Axes available for movement on this widget.

        The base planning/placement widgets are floor-locked (XY only); the
        assemble tab overrides to allow Z movement too.

        Returns:
            Set of allowed axis names ('x', 'y', optionally 'z').
        '''
        return frozenset({"x", "y"})

    def _allowed_axes(self) -> frozenset[str] | None:
        '''Axes that are actually constrained by the current lock state.

        Returns ``None`` when unconstrained (lock is off); otherwise returns
        the intersection of ``{lock}`` (or ``active_axes() - {lock}`` in
        plane mode) with ``active_axes()``. An empty result falls back to
        ``None``.

        Returns:
            The allowed axes, or ``None`` when no lock is active.
        '''
        if self._axis_lock is None:
            return None
        if self._axis_plane:
            candidate = self.active_axes() - {self._axis_lock}
        else:
            candidate = frozenset({self._axis_lock})
        allowed = candidate & self.active_axes()
        return allowed if allowed else None

    def set_show_gizmo(self, enabled: bool) -> None:
        '''Toggle the move gizmo visibility.

        Args:
            enabled: True to show the draggable arrow cross.
        '''
        self._gizmo_enabled = bool(enabled)
        self._update_gizmo()
        self.render()

    def _gizmo_axis_at(self, pos: "QPoint") -> str | None:
        '''Which gizmo axis lies under *pos*, or ``None``.

        Only returns an axis when the gizmo is enabled and visible.

        Args:
            pos: Cursor position in Qt widget coordinates.

        Returns:
            Axis name ('x', 'y', or 'z'), or None.
        '''
        if not self._gizmo_enabled or self._gizmo is None or not self._gizmo.visible():
            return None
        return self._gizmo.pick_axis(self, pos)

    def _update_gizmo(self) -> None:
        '''(Re)build or hide the move gizmo based on current state.

        No render inside — callers render once. The gizmo is visible when
        enabled, there is a selection, and no grab/placement is active.
        Grid-scale changes trigger a rebuild of the arm geometry.
        '''
        visible = (
            self._gizmo_enabled
            and bool(self._selection)
            and not self._grabbing
            and not getattr(self, "_placing", False)
        )
        if not visible:
            if self._gizmo is not None:
                self._gizmo.set_visible(False)
            return

        step = max(self.grid_step(), 1e-9)
        length = GIZMO_LENGTH_STEPS * step
        if self._gizmo is None:
            self._gizmo = MoveGizmo(
                self.active_axes(), AXIS_COLORS, AXIS_VECTORS
            )
            self._gizmo.build(length)
            self._gizmo_step = self.grid_step()
            self._gizmo.add_actors(self)
        elif (
            self._gizmo_step is None
            or abs(self.grid_step() - self._gizmo_step) > 1e-9
        ):
            # Grid scale changed: rebuild the arm geometry through the view.
            self._gizmo.remove_actors(self)
            self._gizmo.build(length)
            self._gizmo_step = self.grid_step()
            self._gizmo.add_actors(self)
        self._gizmo.place(self._selection_center())
        self._gizmo.set_visible(True)

    def _set_axis_lock(self, axis: str, plane: bool) -> bool:
        '''Toggle the move axis constraint; re-anchors the active move.

        Pressing X/Y/Z (or Shift+X/Y/Z for plane mode) constrains movement
        to that axis or to the plane perpendicular to it. Repeated presses of
        the same key combo toggle the lock off.

        Args:
            axis: Axis name ('x', 'y', or 'z').
            plane: True = exclude the locked axis (Shift modifier).

        Returns:
            ``True`` when the key was consumed, ``False`` otherwise.
        '''
        if axis not in self.active_axes():
            return False
        if self._axis_lock == axis and self._axis_plane == plane:
            self._axis_lock = None
            self._axis_plane = False
        else:
            self._axis_lock = axis
            self._axis_plane = plane
        self._reengage_constraint()
        return True

    def _reengage_constraint(self) -> None:
        '''Reset constraint fields for lazy re-engagement on the next move.

        Clears the anchor/origin/zero state so the constrained path will
        re-engage from the current cursor position with delta=0 at engagement.
        When currently grabbing, replays the last grab position to keep the
        visual in sync.
        '''
        self._constraint_anchor = None
        self._constraint_origin = None
        self._constraint_zero = None
        if self._grabbing and self._last_grab_pos is not None:
            self._update_grab(QPoint(self._last_grab_pos))

    def _selection_center(self) -> np.ndarray:
        '''World center of the selection bounding box.

        Computes the midpoint of the axis-aligned bounding box spanning all
        selected pieces. For each selected block, the bounds are derived from
        ``pos_x ± half_x``, ``pos_y ± half_y``, and ``pos_z`` to
        ``pos_z + top_z`` (using ``meta.get(...)`` defaults of 0.0).

        Returns:
            ``(x, y, z)`` world-coordinate center as float64 array.
            Empty selection returns zeros(3).
        '''
        if not self._selection:
            return np.zeros(3, dtype=np.float64)
        valid = {i for i in self._selection if 0 <= i < len(self._scene_meta)}
        if not valid:
            return np.zeros(3, dtype=np.float64)
        mins = np.full(3, np.inf, dtype=np.float64)
        maxs = np.full(3, -np.inf, dtype=np.float64)
        for idx in valid:
            meta = self._scene_meta[idx]
            hx = float(meta.get('half_x', 0.0))
            hy = float(meta.get('half_y', 0.0))
            pz = float(meta.get('pos_z', 0.0))
            tz = float(meta.get('top_z', 0.0))
            mins[0] = min(mins[0], meta['pos_x'] - hx)
            maxs[0] = max(maxs[0], meta['pos_x'] + hx)
            mins[1] = min(mins[1], meta['pos_y'] - hy)
            maxs[1] = max(maxs[1], meta['pos_y'] + hy)
            mins[2] = min(mins[2], pz)
            maxs[2] = max(maxs[2], pz + tz)
        return (mins + maxs) / 2.0

    def _update_guides(self, center: np.ndarray) -> None:
        '''Draw thin colored guide lines along the allowed axes from *center*.

        Guide lines are 2-point polylines extending ``MOVE_GUIDE_LENGTH_STEPS``
        grid steps in both directions along each allowed axis. They are rebuilt
        when the allowed-set changes; otherwise only repositioned.

        Args:
            center: World-coordinate anchor point for the guide lines.
        '''
        allowed = self._allowed_axes()
        if not allowed:
            # Unconstrained move: every active axis is movable, so guide all
            # of them (call sites only run during an active move).
            allowed = self.active_axes()

        step = max(self.grid_step(), 1e-9)
        L = MOVE_GUIDE_LENGTH_STEPS * step

        if self._guides_key != allowed:
            # Rebuild actors — clear old ones (batched, no render).
            for actor in self._guide_actors:
                try:
                    self.remove_actor(actor, reset_camera=False, render=False)
                except Exception:  # noqa: BLE001 - may be detached
                    pass
            self._guide_actors.clear()

            for axis in sorted(allowed):
                u = AXIS_VECTORS[axis]
                start = -u * L
                end = u * L
                segments = np.array([[start, end]], dtype=np.float64)
                points = segments.reshape(-1, 3)
                index = np.arange(len(segments) * 2, dtype=np.int64).reshape(-1, 2)
                lines = np.hstack([np.full((len(index), 1), 2), index]).ravel()
                mesh = pv.PolyData(points, lines=lines)
                actor = self.add_mesh(
                    mesh,
                    color=AXIS_COLORS[axis],
                    line_width=2.0,
                    lighting=False,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                # Position at origin for now; SetPosition applied in every call.
                self._guide_actors.append(actor)

            self._guides_key = allowed

        # Reposition all guide actors to the current center.
        for actor in self._guide_actors:
            actor.SetPosition(*center)

    def _clear_guides(self) -> None:
        '''Remove all guide-line actors and reset guide state.'''
        for actor in self._guide_actors:
            try:
                self.remove_actor(actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - may be detached
                pass
        self._guide_actors.clear()
        self._guides_key = None

    def _cursor_ray(self, pos: QPoint) -> tuple[np.ndarray, np.ndarray] | None:
        '''Reconstruct the cursor ray from camera through the unprojected cursor.

        The ray origin is the camera position; direction is the vector from
        origin to the focal-depth-unprojected cursor point (NOT normalized).

        Args:
            pos: Cursor position in Qt widget coordinates.

        Returns:
            ``(origin, direction)`` where both are ``np.ndarray`` of shape (3,)
            and direction is not normalized; ``None`` when no renderer/camera.
        '''
        renderer = self.renderer
        camera = renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        renderer.SetWorldPoint(focal[0], focal[1], focal[2], 1.0)
        renderer.WorldToDisplay()
        depth = float(renderer.GetDisplayPoint()[2])
        display_x, display_y = self._to_display(pos)
        renderer.SetDisplayPoint(display_x, display_y, depth)
        renderer.DisplayToWorld()
        world = renderer.GetWorldPoint()
        weight = float(world[3])
        point = np.asarray(world[:3], dtype=np.float64)
        if abs(weight) > 1e-12:
            point = point / weight
        origin = np.asarray(camera.GetPosition(), dtype=np.float64)
        direction = point - origin
        return (origin, direction)

    def _constrained_delta(self, pos: QPoint) -> np.ndarray | None:
        '''Return the total world delta (3-vector) for the current constraint.

        Lazily engages when ``_constraint_anchor is None`` by computing the
        anchor from the selection center and the current applied offset of
        the primary grabbed piece. Then computes the raw cursor-to-anchor
        delta using either single-axis or plane-mode intersection math.

        Args:
            pos: Cursor position in Qt widget coordinates.

        Returns:
            The total world delta (``_constraint_zero + raw_for_cursor``) as a
            3-vector, or ``None`` when the cursor cannot be mapped to the
            constrained space.
        '''
        if self._constraint_anchor is None:
            center = self._selection_center()
            self._constraint_anchor = center.copy()
            # Compute _constraint_zero from primary piece's current offset.
            primary = (
                self._primary if self._primary in self._grab_orig
                else min(self._grab_orig)
            )
            orig = self._grab_orig[primary]
            orig_z = float(getattr(self, '_grab_orig_z', {}).get(primary, 0.0))
            meta = self._scene_meta[primary]
            current_offset = np.array([
                meta['pos_x'] - orig[0],
                meta['pos_y'] - orig[1],
                float(meta.get('pos_z', 0.0)) - orig_z,
            ], dtype=np.float64)
            self._constraint_zero = current_offset

            # Compute _constraint_origin based on lock mode.
            if self._axis_lock is not None and not self._axis_plane:
                # Single-axis: absorb the engagement cursor's projection onto
                # the axis line into the origin, so the first constrained
                # frame is a no-move (the lock never jumps the selection).
                self._constraint_origin = center.copy()
                raw0 = self._constrained_raw_delta(pos)
                if raw0 is not None:
                    idx = {'x': 0, 'y': 1, 'z': 2}[str(self._axis_lock)]
                    self._constraint_origin[idx] += float(raw0[idx])
            else:
                # Plane mode: origin = cursor ray intersection with the plane
                # through the center (fallback: the center itself when the
                # ray is too parallel to the plane).
                lock = self._axis_lock
                assert lock is not None  # constrained path only when locked
                self._constraint_origin = center.copy()
                ray = self._cursor_ray(pos)
                if ray is not None:
                    o, d = ray
                    n = AXIS_VECTORS[lock]  # unit normal
                    d_hat = d / max(float(np.linalg.norm(d)), 1e-12)
                    denom = float(np.dot(d_hat, n))
                    if abs(denom) >= _FLOOR_RAY_MIN_SIN:
                        t = float(np.dot(self._constraint_anchor - o, n) / denom)
                        if t > 0.0:
                            self._constraint_origin = o + d * t

        raw = self._constrained_raw_delta(pos)
        if raw is None:
            return None
        return self._constraint_zero + raw

    def _constrained_raw_delta(self, pos: QPoint) -> np.ndarray | None:
        '''Pure cursor math for constrained delta (no state mutation).

        Computes the raw displacement from ``_constraint_origin`` to the
        closest point on the cursor ray relative to the anchor.

        Args:
            pos: Cursor position in Qt widget coordinates.

        Returns:
            Raw 3-vector delta, or ``None`` when the ray is too parallel.
        '''
        lock = self._axis_lock
        assert lock is not None  # _constrained_raw_delta called only when locked
        u = AXIS_VECTORS[lock]
        ray = self._cursor_ray(pos)
        if ray is None:
            return None
        o, d = ray

        anchor = self._constraint_anchor
        assert anchor is not None  # lazily engaged before this call
        origin = self._constraint_origin
        assert origin is not None  # set during lazy engagement

        if not self._axis_plane:
            # Single-axis mode: closest point of cursor ray to the line (anchor, u).
            w = anchor - o
            uu = float(np.dot(u, u))  # =1.0 for unit vectors but keep generic
            du = float(np.dot(d, u))
            ddot = float(np.dot(d, d))
            denom = ddot * uu - du * du
            if denom < 1e-12:
                return None
            # Guard against ray nearly parallel to the line.
            norm_d = float(np.linalg.norm(d))
            if abs(du) / (norm_d * max(float(np.linalg.norm(u)), 1e-12)) > (1 - 1e-3):
                return None
            t_ray = ((float(np.dot(w, d)) * uu) - (float(np.dot(w, u)) * du)) / denom
            if t_ray <= 0.0:
                return None
            # Line parameter s (mm from the anchor along u): b*e - c*d0 over
            # the denominator with a = u*u, b = u*d, c = d*d, d0 = u*w,
            # e = d*w — same numerator order as t_ray, NOT mirrored.
            s = ((float(np.dot(w, d)) * du) - (float(np.dot(w, u)) * ddot)) / denom

            # The line runs anchor + s*u, so *s* is a parameter in mm from
            # the anchor; the zero reference is the origin's own offset from
            # the anchor along the axis (captured at engagement). raw = 0
            # when the cursor sits exactly where the lock engaged.
            lock_idx = {'x': 0, 'y': 1, 'z': 2}[lock]
            origin_offset = float(origin[lock_idx]) - float(anchor[lock_idx])
            coord = s - origin_offset

            raw = np.zeros(3, dtype=np.float64)
            raw[lock_idx] = coord
            return raw
        else:
            # Plane mode: intersect cursor ray with plane through anchor.
            n = AXIS_VECTORS[lock]  # unit normal
            d_hat = d / max(float(np.linalg.norm(d)), 1e-12)
            denom = float(np.dot(d_hat, n))
            if abs(denom) < _FLOOR_RAY_MIN_SIN:
                return None
            t = float(np.dot(anchor - o, n) / denom)
            if t <= 0.0:
                return None
            point = o + d * t
            raw = point - origin
            return raw

    def _clamp_move_dz(self, dz: float) -> float:
        '''Clamp the Z component of a constrained drag delta.

        Base implementation returns ``dz`` unchanged; subclasses may override
        to enforce additional bounds (e.g. AssembleView clamps to the lowest
        grabbed piece's original height).

        Args:
            dz: The Z delta from constrained math.

        Returns:
            Clamped Z delta.
        '''
        return dz

    def _retained_move_dz(self) -> float:
        '''Z offset kept when the active constraint excludes the Z axis.

        The base widgets are floor-locked, so nothing to retain.
        AssembleView overrides this to hold the drag's accumulated Z offset
        (e.g. Shift+Z plane moves keep the current height instead of
        dropping the piece back to its drag-start height).

        Returns:
            The Z offset (mm) to keep applying while Z is constrained out.
        '''
        return 0.0

    def _snap_move_target(
        self, x: float, y: float, hx: float = 0.0, hy: float = 0.0
    ) -> tuple[float, float]:
        '''Snap a move target so the footprint minimum corner hits the grid.

        Scene positions are footprint centers (the meshes are centered at
        the origin); rounding the center itself would center unit blocks
        on the grid lines. Subtracting the half extents before rounding
        and adding them back puts the bottom-left corner on a grid-step
        multiple, so a unit block fills exactly one grid cell.

        Hook for subclasses: PlacementView snaps relative to the zone under
        the target instead, so pieces align with the per-bed grids.

        Args:
            x: Unsnapped world X of the move target (footprint center).
            y: Unsnapped world Y of the move target (footprint center).
            hx: Half footprint extent along X, mm.
            hy: Half footprint extent along Y, mm.

        Returns:
            The ``(x, y)`` center with its minimum corner snapped to
            grid-step multiples.
        '''
        step = max(self.grid_step(), 1e-9)
        return (
            round((x - hx) / step) * step + hx,
            round((y - hy) / step) * step + hy,
        )

    # ------------------------------------------------------------- floor grid
    def _update_grid(self) -> None:
        '''Rebuild the static floor-grid actors around the world origin.

        The grid is centred at (0, 0) with a fixed extent; it is rebuilt only
        by ``set_grid_unit`` / ``set_grid_scale`` / ``set_grid_enabled``.
        '''
        for actor in (*self._grid_actors, *self._grid_label_actors):
            try:
                self.remove_actor(actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._grid_actors.clear()
        self._grid_label_actors.clear()
        if self._grid_enabled:
            self._add_grid_actors()
            self._refresh_grid_labels()
        self.render()

    def _add_grid_actors(self) -> None:
        '''Build the line/axis/label actors for the static centered grid.'''
        step = max(self.grid_step(), 1e-9)
        center_x = center_y = 0.0
        half = GRID_EXTENT_CELLS * step
        cells = GRID_EXTENT_CELLS
        ks = np.arange(-cells, cells + 1, dtype=np.int64)

        def segments(coords: np.ndarray, vertical: bool) -> np.ndarray:
            '''(N, 2, 3) segment array of full-length grid lines.'''
            fixed = np.full_like(coords, half)
            z = np.full_like(coords, GRID_Z_OFFSET)
            if vertical:
                first = np.stack([coords, -fixed, z], axis=1)
                second = np.stack([coords, fixed, z], axis=1)
            else:
                first = np.stack([-fixed, coords, z], axis=1)
                second = np.stack([fixed, coords, z], axis=1)
            return np.stack([first, second], axis=1)

        # Major lines sit on world-aligned multiples of 10 cells.
        vertical_major = ks % GRID_MAJOR_EVERY == 0
        horizontal_major = ks % GRID_MAJOR_EVERY == 0
        verticals = segments(center_x + ks * step, vertical=True)
        horizontals = segments(center_y + ks * step, vertical=False)
        minor = np.concatenate(
            [verticals[~vertical_major], horizontals[~horizontal_major]]
        )
        major = np.concatenate(
            [verticals[vertical_major], horizontals[horizontal_major]]
        )
        for mesh, color, width in (
            (minor, GRID_MINOR_COLOR, 1.0),
            (major, GRID_MAJOR_COLOR, 1.5),
        ):
            if len(mesh):
                self._grid_actors.append(
                    self._add_grid_lines(mesh, color=color, line_width=width)
                )
        # Colored world axes: X along y=0, Y along x=0, spanning the window.
        # They sit slightly above the line plane (GRID_AXIS_Z_OFFSET):
        # coplanar lines z-fight into dashed bands at grazing view angles.
        axes = np.array(
            [
                [
                    [center_x - half, 0.0, GRID_AXIS_Z_OFFSET],
                    [center_x + half, 0.0, GRID_AXIS_Z_OFFSET],
                ],
                [
                    [0.0, center_y - half, GRID_AXIS_Z_OFFSET],
                    [0.0, center_y + half, GRID_AXIS_Z_OFFSET],
                ],
            ]
        )
        axis_x = self._add_grid_lines(axes[:1], color=GRID_AXIS_X_COLOR, line_width=2.5)
        axis_y = self._add_grid_lines(axes[1:], color=GRID_AXIS_Y_COLOR, line_width=2.5)
        self._grid_actors.extend([axis_x, axis_y])

    def _add_grid_lines(
        self, segments: np.ndarray, color: str, line_width: float
    ) -> pv.Actor:
        '''Add one grid line mesh built from an (N, 2, 3) segment array.'''
        points = segments.reshape(-1, 3)
        index = np.arange(len(segments) * 2, dtype=np.int64).reshape(-1, 2)
        lines = np.hstack([np.full((len(index), 1), 2), index]).ravel()
        return self.add_mesh(
            pv.PolyData(points, lines=lines),
            color=color,
            line_width=line_width,
            lighting=False,
            pickable=False,
            reset_camera=False,
        )

    def _refresh_grid_labels(self) -> None:
        '''Rebuild the axis labels when the zoom-adaptive spacing changed.

        Evaluates the per-axis spacing multipliers at the current camera and
        rebuilds the label actors only when they differ from the rendered
        ones (camera interaction handlers call this on every move; the
        evaluation is two point projections, the rebuild is rare).

        The grid must be on and its line actors present; otherwise this is
        a no-op.
        '''
        if not self._grid_enabled or not self._grid_actors:
            return
        step = max(self.grid_step(), 1e-9)
        multipliers = (
            self._grid_label_multiplier(step, 0),
            self._grid_label_multiplier(step, 1),
        )
        if multipliers == self._label_multipliers and self._grid_label_actors:
            return
        self._label_multipliers = multipliers
        for actor in self._grid_label_actors:
            try:
                self.remove_actor(actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._grid_label_actors.clear()
        self._add_grid_labels(step, multipliers)
        self.render()

    def _on_camera_interaction(self, *_args: Any) -> None:
        '''VTK InteractionEvent: re-evaluate the zoom-adaptive axis labels.'''
        self._refresh_grid_labels()

    def _grid_label_multiplier(self, step: float, axis: int) -> int:
        '''Smallest power-of-two multiple of *step* keeping labels readable.

        The on-screen length of one grid step is measured along *axis* at
        the camera focal point (it shrinks toward 0 when the axis recedes
        from the camera); the returned multiplier is the first power of two
        whose spacing clears :data:`GRID_LABEL_MIN_SPACING_PX`.

        Args:
            step: Grid step in mm.
            axis: World axis index (0 = X, 1 = Y).

        Returns:
            Label spacing multiplier (1, 2, 4, ... capped at
            :data:`GRID_LABEL_MAX_MULTIPLIER`).
        '''
        pixels = math.inf  # no renderer/camera yet: densest labels
        try:
            renderer = self.renderer
            camera = renderer.GetActiveCamera()
            focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
            far = focal.copy()
            far[axis] += step
            renderer.SetWorldPoint(focal[0], focal[1], focal[2], 1.0)
            renderer.WorldToDisplay()
            x0, y0, _ = renderer.GetDisplayPoint()[:3]
            renderer.SetWorldPoint(far[0], far[1], far[2], 1.0)
            renderer.WorldToDisplay()
            x1, y1, _ = renderer.GetDisplayPoint()[:3]
            projected = math.hypot(x1 - x0, y1 - y0)
            if math.isfinite(projected) and projected >= 0.0:
                pixels = projected
        except Exception:  # noqa: BLE001 - keep the densest fallback
            pass
        multiplier = 1
        while (
            multiplier * pixels < GRID_LABEL_MIN_SPACING_PX
            and multiplier < GRID_LABEL_MAX_MULTIPLIER
        ):
            multiplier *= 2
        return multiplier

    def _add_grid_labels(self, step: float, multipliers: tuple[int, int]) -> None:
        '''Zoom-adaptive mm coordinate labels directly on the X and Y axes.

        One label per ``multiplier * step`` mm per axis, centered on the
        axis lines (the multiplier keeps adjacent labels readable at the
        current camera). The origin (0) is skipped: both axes would draw it
        at the same spot. The labels span only ±:data:`GRID_LABEL_WINDOW_CELLS`
        cells around the origin: they render at a constant screen size, so
        labels reaching the full extent crowd into wide bands when the plane
        is seen at a grazing angle (and the count stays bounded at any zoom).

        Args:
            step: Grid step in mm.
            multipliers: Per-axis label spacing multipliers (X, Y), each a
                power of two.
        '''
        label_half = min(GRID_EXTENT_CELLS, GRID_LABEL_WINDOW_CELLS) * step
        for axis, multiplier in enumerate(multipliers):
            label_step = step * multiplier
            first = int(math.ceil(-label_half / label_step))
            last = int(math.floor(label_half / label_step))
            values = np.arange(first, last + 1, dtype=np.int64) * label_step
            values = values[values != 0.0]
            if len(values) > 400:  # defensive cap: pathological zoom levels
                values = values[:: int(math.ceil(len(values) / 400))]
            if not len(values):
                continue
            zeros = np.zeros(len(values))
            if axis == 0:
                points = np.stack([values, zeros, zeros + GRID_LABEL_Z], axis=1)
                color = GRID_LABEL_X_COLOR
            else:
                points = np.stack([zeros, values, zeros + GRID_LABEL_Z], axis=1)
                color = GRID_LABEL_Y_COLOR
            actor = self.add_point_labels(
                points,
                [f"{value:g}" for value in values],
                always_visible=True,
                show_points=False,
                shape=None,
                font_size=9,
                text_color=color,
                name=f"grid_labels_{axis}",
                render=False,
            )
            if actor is not None:
                self._grid_label_actors.append(actor)

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
        # Renders itself once when the outline set changed.
        self._update_selection_outlines()

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
        outline = self._outline_actors.get(index)
        if outline is not None:
            outline.SetPosition(
                float(x), float(y), float(self._scene_meta[index].get('pos_z', 0.0))
            )
        self._move_quantity_labels([index])
        self.render()

    def set_block_colors(self, colors: dict[int, str]) -> None:
        '''Recolor the given blocks without rebuilding geometry.

        The first actor of a block carries its color (the shaded surface, or
        the wireframe in the wireframe-only mode); decorations stay black.

        Args:
            colors: Mapping of block index to hex color.
        '''
        for index, color in colors.items():
            if 0 <= index < len(self._block_actors) and self._block_actors[index]:
                self._block_actors[index][0].prop.color = color
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
        '''Move every actor of block *index* to the scene position.

        The scene meta may carry ``pos_z`` (the Assemble tab places pieces
        at height); planning blocks default to the floor plane.
        '''
        if 0 <= index < len(self._block_actors):
            z = float(self._scene_meta[index].get('pos_z', 0.0))
            for actor in self._block_actors[index]:
                actor.SetPosition(float(x), float(y), z)

    def _clear_scene_actors(self) -> None:
        '''Remove every surface/edge/wireframe/outline/label actor.

        Batched with ``render=False``: a scene rebuild removes up to several
        actors per block, and one render per removal would multiply into a
        visible stall. The caller renders once after the batch.
        '''
        for index in list(self._outline_actors):
            self._drop_outline(index)
        for actor in (
            *(actor for actors in self._block_actors for actor in actors),
            *self._qty_actors.values(),
        ):
            try:
                self.remove_actor(actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - actor may already be detached
                pass
        self._block_actors.clear()
        self._qty_actors.clear()
        self._qty_points.clear()
        self._actor_map.clear()

    def _build_block_actors(self, index: int) -> list[pv.Actor]:
        '''Build the actors for a single block and register them in ``_actor_map``.

        Args:
            index: Block index in the current scene.

        Returns:
            List of actors created for this block (surface, feature edges,
            wireframe decorations).
        '''
        mesh = self._scene_meshes[index]
        position = self._scene_meta[index]
        x, y = float(position['pos_x']), float(position['pos_y'])
        # Stacked assemble pieces render above the floor (meta pos_z).
        z = float(position.get('pos_z', 0.0))
        actors: list[pv.Actor] = []
        if mesh.n_points == 0:
            # Empty placeholder mesh (e.g. a None entry with no previous
            # scene): pyvista refuses to plot empty meshes, so add nothing.
            return actors
        if self._mode == MODE_WIREFRAME:
            # No solid underneath: the wireframe itself is the pick
            # target (the cell picker intersects the mesh triangles, not
            # the rendered lines), so blocks stay selectable.
            wire = self._add_wireframe_actor(mesh, line_width=1.5, tubes=False)
            wire.SetPickable(True)
            actors.append(wire)
            self._actor_map[wire] = index
        else:
            surface = add_surface_actor(self, mesh, color=self._scene_colors[index])
            surface.SetPickable(True)
            actors.append(surface)
            self._actor_map[surface] = index
            if self._mode == MODE_SOLID_FEATURE_EDGES:
                edges = self._add_feature_edge_actor(mesh)
                if edges is not None:
                    actors.append(edges)
            elif self._mode == MODE_SOLID_WIREFRAME:
                actors.append(
                    self._add_wireframe_actor(mesh, line_width=2.5, tubes=True)
                )
        for actor in actors:
            actor.SetPosition(x, y, z)
        return actors

    def _rebuild_actors(self, changed: list[bool] | None = None) -> None:
        '''Remove and re-add per-block actors for the active mode.

        When *changed* is ``None`` (or length/count mismatch) every block
        is rebuilt.  Otherwise only indices where ``changed[i]`` is ``True``
        are rebuilt; unchanged blocks keep their existing actors.

        Args:
            changed: Optional bool mask for incremental rebuild.
        '''
        n_blocks = len(self._scene_meshes)
        if not n_blocks:
            # The scene went empty: clear the actors of the removed blocks,
            # otherwise they linger as ghosts until the next non-empty
            # full rebuild (only outlines/labels would be refreshed below).
            self._clear_scene_actors()
            return

        # Determine whether we do a full or incremental rebuild.
        if (
            changed is None
            or len(changed) != n_blocks
            or len(self._block_actors) != n_blocks
        ):
            self._clear_scene_actors()
            for index in range(n_blocks):
                self._block_actors.append(self._build_block_actors(index))
        else:
            for index, do_build in enumerate(changed):
                if do_build:
                    # Remove existing actors for this block (batched with
                    # render=False - the caller renders once afterwards;
                    # pyvista's remove_actor would render per actor). The
                    # outline is dropped too: the rebuilt mesh may have new
                    # bounds, so the delta update must recreate it.
                    self._drop_outline(index)
                    old_actors = self._block_actors[index]
                    for actor in old_actors:
                        try:
                            self.remove_actor(actor, reset_camera=False, render=False)
                        except Exception:  # noqa: BLE001 - may be detached
                            pass
                        self._actor_map.pop(actor, None)
                    self._block_actors[index] = self._build_block_actors(index)

        self._update_selection_outlines()
        self._refresh_quantity_labels()

    def _add_feature_edge_actor(self, mesh: pv.PolyData) -> pv.Actor | None:
        '''Semantic part-boundary outline actor of one block (None if bare).

        Feature edges are derived from ``cell_data["part"]`` differences; a
        mesh without classification (or without boundaries) adds nothing.
        '''
        edges = semantic_edges(mesh)
        if edges.n_points == 0:
            return None
        return self.add_mesh(
            edges,
            color=LINE_COLOR,
            line_width=2.0,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
            reset_camera=False,
            render=False,  # batched: _rebuild_actors renders once afterwards
        )

    def _add_wireframe_actor(
        self, mesh: pv.PolyData, line_width: float, tubes: bool
    ) -> pv.Actor:
        '''Wireframe rendering of the block mesh.

        With *tubes* the lines get real volume so they win the depth test
        against the surface (plain polylines z-fight with the mesh); the
        bare wireframe mode renders plain lines without a solid underneath.
        Decoration geometry always sits on the block itself, so the tube
        geometry shader only ever sees short in-front-of-camera segments -
        the fat camera-angle dependent bands it draws for huge floor-grid
        lines (whose endpoints fall behind the camera) cannot appear here.
        '''
        return self.add_mesh(
            mesh,
            color=LINE_COLOR,
            style="wireframe",
            line_width=line_width,
            lighting=False,
            render_lines_as_tubes=tubes,
            pickable=False,
            reset_camera=False,
            render=False,  # batched: _rebuild_actors renders once afterwards
        )

    def _update_selection_outlines(self) -> None:
        '''Sync the thin bright outline actors with the current selection.

        Outlines are updated incrementally: only the delta between the
        rendered actors and the selection is added/removed. Everything is
        batched with ``render=False`` and followed by ONE render - pyvista's
        ``add_mesh``/``remove_actor`` render by default, which would cost
        one full viewport render per selected block per change (seconds of
        lag when selecting or deselecting many blocks).
        '''
        count = len(self._scene_meshes)
        wanted = {index for index in self._selection if 0 <= index < count}
        stale = sorted(index for index in self._outline_actors if index not in wanted)
        fresh = sorted(index for index in wanted if index not in self._outline_actors)
        if not stale and not fresh:
            return
        for index in stale:
            self._drop_outline(index)
        for index in fresh:
            actor = self.add_mesh(
                self._scene_meshes[index].outline(),
                color=SELECTION_COLOR,
                line_width=1.5,
                lighting=False,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            position = self._scene_meta[index]
            actor.SetPosition(
                position['pos_x'],
                position['pos_y'],
                float(position.get('pos_z', 0.0)),
            )
            self._outline_actors[index] = actor
        self._update_gizmo()
        self.render()

    def _drop_outline(self, index: int) -> None:
        '''Remove one selection outline actor (batched, no render).'''
        actor = self._outline_actors.pop(index, None)
        if actor is None:
            return
        try:
            self.remove_actor(actor, reset_camera=False, render=False)
        except Exception:  # noqa: BLE001 - actor may already be detached
            pass

    # ------------------------------------------------------- move drag & grab
    def start_move_drag(self, pos: QPoint) -> bool:
        '''Start dragging the selection with the mouse (LMB pressed on a block).

        The selection follows the cursor on the floor plane; the release
        commits the move, or falls back to the click semantics when nothing
        really moved (a sub-threshold jiggle is undone first).

        Args:
            pos: Press position in Qt widget coordinates.

        Returns:
            True when the drag was started.
        '''
        if self._grabbing or not self._selection:
            return False
        self._grabbing = True
        self._grab_keyboard = False
        self._grab_moved = False
        self._drag_from_axis = False
        self._grab_press_pos = QPoint(pos)
        self._grab_orig = {
            index: (
                float(self._scene_meta[index]['pos_x']),
                float(self._scene_meta[index]['pos_y']),
            )
            for index in self._selection
            if 0 <= index < len(self._scene_meta)
        }
        self._grab_anchor = None
        # Reset axis-lock state for a fresh move.
        self._axis_lock = None
        self._axis_plane = False
        self._constraint_anchor = None
        self._constraint_origin = None
        self._constraint_zero = None
        self._last_grab_pos = None
        self._update_grab(pos)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        return True

    def start_grab(self, pos: QPoint | None = None) -> bool:
        '''Start the grab move of the whole selection (G hotkey).

        The selection follows the cursor on the floor plane until a left
        click or Enter confirms it, Esc or a right click cancels.

        Args:
            pos: Anchor position in Qt widget coordinates; the global
                cursor position when None.

        Returns:
            True when grab mode was entered.
        '''
        if self._grabbing or not self._selection:
            return False
        self._grabbing = True
        self._grab_keyboard = True
        self._grab_moved = False
        self._drag_from_axis = False
        self._grab_press_pos = QPoint()
        self._grab_orig = {
            index: (
                float(self._scene_meta[index]['pos_x']),
                float(self._scene_meta[index]['pos_y']),
            )
            for index in self._selection
            if 0 <= index < len(self._scene_meta)
        }
        self._grab_anchor = None
        # Reset axis-lock state for a fresh move.
        self._axis_lock = None
        self._axis_plane = False
        self._constraint_anchor = None
        self._constraint_origin = None
        self._constraint_zero = None
        self._last_grab_pos = None
        self._update_grab(self.mapFromGlobal(QCursor.pos()) if pos is None else pos)
        if QWidget.mouseGrabber() is not self:
            self.grabMouse()  # keep receiving moves outside the viewport
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        return True

    def _update_grab(self, pos: QPoint) -> None:
        '''Move the selection with the cursor; snap to the grid when enabled.

        When an axis lock is active (X/Y/Z key), delegates to the constrained
        path; otherwise uses the original free-plane floor intersection.
        '''
        if not self._grabbing or not self._grab_orig:
            return
        self._last_grab_pos = QPoint(pos)
        if self._axis_lock is not None:
            self._update_grab_constrained(pos)
            return
        world = self._cursor_on_floor(pos)
        if world is None:
            return
        if self._grab_anchor is None:
            self._grab_anchor = world[:2].copy()
            return
        primary = (
            self._primary if self._primary in self._grab_orig else min(self._grab_orig)
        )
        original_x, original_y = self._grab_orig[primary]
        target_x = original_x + float(world[0] - self._grab_anchor[0])
        target_y = original_y + float(world[1] - self._grab_anchor[1])
        if self._grid_snap:
            meta = self._scene_meta[primary]
            target_x, target_y = self._snap_move_target(
                target_x,
                target_y,
                float(meta.get('half_x', 0.0)),
                float(meta.get('half_y', 0.0)),
            )
        dx, dy = target_x - original_x, target_y - original_y
        if dx != 0.0 or dy != 0.0:
            self._grab_moved = True
        self._apply_move(self._grab_orig, dx, dy)
        self._update_guides(self._selection_center())

    def _update_grab_constrained(self, pos: QPoint) -> None:
        '''Constrained-path move: delta computed via axis-lock intersection.

        Uses ``_constrained_delta`` to compute the total world displacement,
        then applies grid snapping and calls ``_apply_move`` with the result.

        Args:
            pos: Cursor position in Qt widget coordinates.
        '''
        delta = self._constrained_delta(pos)
        if delta is None:
            return
        primary = (
            self._primary if self._primary in self._grab_orig else min(self._grab_orig)
        )
        original_x, original_y = self._grab_orig[primary]
        allowed = self._allowed_axes()
        assert allowed is not None  # constrained path only when lock is set
        target_x = original_x + float(delta[0]) if "x" in allowed else original_x
        target_y = original_y + float(delta[1]) if "y" in allowed else original_y
        dz = float(delta[2]) if "z" in allowed else self._retained_move_dz()

        dz = self._clamp_move_dz(dz)

        if self._grid_snap:
            step = max(self.grid_step(), 1e-9)
            meta = self._scene_meta[primary]
            hx = float(meta.get('half_x', 0.0))
            hy = float(meta.get('half_y', 0.0))
            if "x" in allowed:
                target_x = round((target_x - hx) / step) * step + hx
            if "y" in allowed:
                target_y = round((target_y - hy) / step) * step + hy
            if "z" in allowed:
                dz = round(dz / step) * step

        # Check if we actually moved compared to current meta position.
        dx = target_x - original_x
        dy = target_y - original_y
        moved_z = abs(dz - float(self._scene_meta[primary].get('pos_z', 0.0))) > 1e-9
        if dx != 0.0 or dy != 0.0 or moved_z:
            self._grab_moved = True

        self._update_guides(self._selection_center())
        self._apply_move(self._grab_orig, dx, dy, dz=dz)

    def confirm_grab(self) -> None:
        '''Finish the grab move and commit the new positions.'''
        if not self._grabbing:
            return
        self._end_grab()
        positions = {
            index: (
                float(self._scene_meta[index]['pos_x']),
                float(self._scene_meta[index]['pos_y']),
            )
            for index in self._grab_orig
        }
        self._grab_orig.clear()
        self._grab_anchor = None
        self.blocks_moved.emit(positions)

    def cancel_grab(self) -> None:
        '''Finish the grab move and restore the original positions.'''
        if not self._grabbing:
            return
        self._end_grab()
        self._apply_move(self._grab_orig, 0.0, 0.0, restore=True)
        self._grab_orig.clear()
        self._grab_anchor = None

    def _finish_move_drag(self, event: QMouseEvent) -> None:
        '''Release of an LMB move drag: commit the move or apply the click.

        A drag beyond the rubber-band threshold counts as a move and emits
        the committed positions; anything shorter is undone and routed
        through the regular click semantics.
        '''
        dragged = (
            event.position().toPoint() - self._grab_press_pos
        ).manhattanLength() > RUBBER_BAND_THRESHOLD_PX
        moved = self._grab_moved
        from_axis = self._drag_from_axis
        self._end_grab()
        if moved and dragged:
            positions = {
                index: (
                    float(self._scene_meta[index]['pos_x']),
                    float(self._scene_meta[index]['pos_y']),
                )
                for index in self._grab_orig
            }
            self._grab_orig.clear()
            self._grab_anchor = None
            self.blocks_moved.emit(positions)
            return
        if moved:
            self._apply_move(self._grab_orig, 0.0, 0.0, restore=True)
        self._grab_orig.clear()
        self._grab_anchor = None
        if not from_axis:
            self._finish_click(event)

    def _end_grab(self) -> None:
        '''Leave the move drag: drop the mouse grab and the drag cursor.

        Also clears guide lines and resets all axis-lock state.
        '''
        self._grabbing = False
        self._grab_keyboard = False
        self._grab_moved = False
        self._drag_from_axis = False
        self._clear_guides()
        self._axis_lock = None
        self._axis_plane = False
        self._constraint_anchor = None
        self._constraint_origin = None
        self._constraint_zero = None
        self._last_grab_pos = None
        self._update_gizmo()
        if QWidget.mouseGrabber() is self:
            self.releaseMouse()
        self.unsetCursor()

    # ------------------------------------------------------------- move helpers
    def _apply_move(
        self,
        originals: dict[int, tuple[float, float]],
        dx: float,
        dy: float,
        restore: bool = False,
        dz: float = 0.0,
    ) -> None:
        '''Shift (or restore) the moved blocks and their decorations.

        Args:
            originals: Block index -> (x, y) the move started from.
            dx: Delta applied to every X coordinate.
            dy: Delta applied to every Y coordinate.
            restore: True writes the original coordinates back verbatim
                (the deltas are ignored).
            dz: Z delta applied on the non-restore path. Planning blocks
                start at z=0 and never lift — this is a no-op on the planning
                tab; does NOT read-modify meta z.
        '''
        for index, (original_x, original_y) in originals.items():
            if not 0 <= index < len(self._scene_meta):
                continue
            z = float(self._scene_meta[index].get('pos_z', 0.0))
            if restore:
                x, y = original_x, original_y
            else:
                x, y = original_x + dx, original_y + dy
                self._scene_meta[index]['pos_z'] = dz
            self._scene_meta[index]['pos_x'] = x
            self._scene_meta[index]['pos_y'] = y
            self._position_actor(index, x, y)
            outline = self._outline_actors.get(index)
            if outline is not None:
                outline.SetPosition(x, y, z)
        self._move_quantity_labels(originals)
        self.render()

    def _cursor_on_floor(self, pos: QPoint) -> np.ndarray | None:
        '''World point of the cursor ray on the z=0 floor plane.'''
        return self._cursor_on_plane(pos, 0.0)

    def _cursor_on_plane(self, pos: QPoint, plane_z: float) -> np.ndarray | None:
        '''World point of the cursor ray on the horizontal plane *plane_z*.

        The ray is reconstructed from the camera through the unprojected
        cursor (at the focal-plane depth, reusing the pan projection path)
        and intersected with the plane.

        Args:
            pos: Cursor position in Qt widget coordinates.
            plane_z: Height of the horizontal intersection plane, mm.

        Returns:
            World (x, y, z) point, or None when the ray is too parallel to
            the plane (side views would translate pixel deltas into huge
            distances) or points away from it.
        '''
        ray = self._cursor_ray(pos)
        if ray is None:
            return None
        origin, direction = ray
        length = float(np.linalg.norm(direction))
        if length < 1e-9 or abs(direction[2]) < _FLOOR_RAY_MIN_SIN * length:
            return None
        t = (plane_z - origin[2]) / direction[2]
        if t <= 0.0:
            return None
        return origin + direction * t

    def _label_point(self, index: int) -> np.ndarray:
        '''World anchor point of the quantity label of block *index*.

        The label sits on the floor beside the block footprint (like the
        grid axis labels) instead of hanging above the block top, at a
        fixed gap from the face it adjoins regardless of the block size.
        '''
        meta = self._scene_meta[index]
        margin = _LABEL_MARGIN_MM
        x, y = meta['pos_x'], meta['pos_y']
        if self._qty_position == 'left':
            return np.array([x - meta['half_x'] - margin, y, GRID_LABEL_Z])
        if self._qty_position == 'right':
            return np.array([x + meta['half_x'] + margin, y, GRID_LABEL_Z])
        if self._qty_position == 'top':
            return np.array([x, y + meta['half_y'] + margin, GRID_LABEL_Z])
        return np.array([x, y - meta['half_y'] - margin, GRID_LABEL_Z])

    def _refresh_quantity_labels(self) -> None:
        '''Rebuild the per-block BOM quantity label actors.

        Each label is its own tiny pipeline (single-point PolyData -> label
        hierarchy -> placement mapper -> 2D overlay actor). The anchor
        PolyData is kept in ``_qty_points`` so a drag only reassigns the
        moved points (:meth:`_move_quantity_labels`) instead of rebuilding
        actors every frame. Everything is added with ``render=False``: the
        caller renders once after the batch (an ``add_point_labels``/default
        ``add_actor`` render here would cost one full viewport render per
        label per call).
        '''
        for index in list(self._qty_actors):
            self.remove_actor(
                f'quantity_{index}',  # type: ignore[arg-type]  # pyvista stub bug
                reset_camera=False,
                render=False,
            )
        self._qty_actors.clear()
        self._qty_points.clear()
        if not self._qty_enabled or not self._scene_meshes:
            return
        # Shared text property (mirrors the previous add_point_labels args:
        # size 13, bold, no shape/background, always on top of the scene).
        text_property = pv.TextProperty(font_size=13, bold=True, color='#d8dde2')
        for index, meta in enumerate(self._scene_meta):
            pd = pv.PolyData(self._label_point(index).reshape(1, 3))
            labels = vtk.vtkStringArray()
            labels.SetName('labels')
            labels.InsertNextValue(f"×{meta['quantity']}")
            pd.GetPointData().AddArray(labels)
            hierarchy = vtk.vtkPointSetToLabelHierarchy()
            hierarchy.SetInputData(pd)
            hierarchy.SetLabelArrayName('labels')
            hierarchy.SetTextProperty(text_property)
            mapper = vtk.vtkLabelPlacementMapper()
            mapper.SetInputConnection(hierarchy.GetOutputPort())
            mapper.SetShapeToNone()
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            self.add_actor(
                actor,
                name=f'quantity_{index}',
                reset_camera=False,
                pickable=False,
                render=False,
            )
            self._qty_points[index] = pd
            self._qty_actors[index] = actor

    def _move_quantity_labels(self, indices: Iterable[int]) -> None:
        '''Reposition the quantity labels of *indices* in place.

        Runs on every mouse-move frame of a drag/grab move: only the single
        anchor point of each affected pipeline is reassigned, the hierarchy
        re-executes on the next render, and no actors are rebuilt. The one
        ``render()`` after the batch comes from the caller.

        Args:
            indices: Scene indices of the moved blocks.
        '''
        for index in indices:
            pd = self._qty_points.get(index)
            if pd is not None:
                pd.points = self._label_point(index).reshape(1, 3)

    # ------------------------------------------------ scene input (Qt events)
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''LMB picks up a block or arms the frame, MMB orbits, Shift+MMB pans.

        During a keyboard grab (G) the left button confirms and the right
        button cancels; during a mouse move drag extra buttons are ignored.
        Otherwise a left press on a block starts the move drag (unless Ctrl
        is held: it keeps the plain toggle-click semantics) and a left press
        on the background arms the rubber band. Left/middle presses are
        consumed here so the VTK interactor never arms its native
        rotate/pan/spin states; the right button falls through to the
        native VTK zoom.
        '''
        pos = event.position().toPoint()
        button = event.button()
        if self._grabbing and self._grab_keyboard:
            if button == Qt.MouseButton.LeftButton:
                self.confirm_grab()
            elif button == Qt.MouseButton.RightButton:
                self.cancel_grab()
            event.accept()
            return
        if self._grabbing:
            event.accept()  # a move drag is already running on the LMB
            return
        if button == Qt.MouseButton.LeftButton:
            # Grab an axis arrow to start an axis-constrained move of the
            # current selection through the same machinery as X/Y/Z keys.
            axis = self._gizmo_axis_at(pos)
            if axis is not None:
                self.setFocus()
                self.start_move_drag(pos)
                if self._grabbing:
                    self._drag_from_axis = True
                    self._axis_lock = axis
                    self._axis_plane = False
                    self._constraint_anchor = None
                    self._constraint_origin = None
                    self._constraint_zero = None
                    self._update_grab(pos)
                event.accept()
                return
            self.setFocus()
            hit = self._pick_scene(pos)
            if (
                hit is not None
                and hit[0] == "block"
                and not event.modifiers() & Qt.KeyboardModifier.ControlModifier
            ):
                index = hit[1]
                if index not in self._selection:
                    self._select_only(index)  # pick up the pressed block
                elif self._primary != index:
                    self._primary = index
                    self._emit_selection()
                self.start_move_drag(pos)
                event.accept()
                return
            self._left_down = True
            self._press_ctrl = bool(
                event.modifiers() & Qt.KeyboardModifier.ControlModifier
            )
            self._press_pos = pos
            self._rubbing = False
            event.accept()
            return
        if button == Qt.MouseButton.MiddleButton:
            self._last_mouse = pos
            self._panning = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._orbiting = not self._panning
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''Route moves to the grab/drag, the rubber band, the camera orbit
        or the pan.'''
        pos = event.position().toPoint()
        if self._grabbing:
            self._update_grab(pos)
            event.accept()
            return
        if self._left_down:
            if (
                not self._rubbing
                and (pos - self._press_pos).manhattanLength() > RUBBER_BAND_THRESHOLD_PX
            ):
                self._rubbing = True
            if self._rubbing:
                rubber = self._ensure_rubber_band()
                rubber.setGeometry(QRect(self._press_pos, pos).normalized())
                rubber.show()
            event.accept()
            return
        if self._orbiting:
            self._orbit_camera(
                pos.x() - self._last_mouse.x(), pos.y() - self._last_mouse.y()
            )
            self._last_mouse = pos
            event.accept()
            return
        if self._panning:
            self._pan_camera(self._last_mouse, pos)
            self._last_mouse = pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt naming
        '''Finish the move drag, the rubber band, or apply the click.'''
        button = event.button()
        if self._grabbing and not self._grab_keyboard:
            if button == Qt.MouseButton.LeftButton:
                self._finish_move_drag(event)
            event.accept()
            return
        if self._grabbing:
            event.accept()  # press already confirmed/canceled the grab
            return
        if button == Qt.MouseButton.LeftButton and self._left_down:
            self._left_down = False
            if self._rubbing:
                self._rubbing = False
                rubber = self._ensure_rubber_band()
                rect = rubber.geometry()
                rubber.hide()
                self._select_from_rubber(rect, add=self._press_ctrl)
            else:
                self._finish_click(event)
            event.accept()
            return
        if button == Qt.MouseButton.MiddleButton:
            self._orbiting = False
            self._panning = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt naming
        '''G starts the grab move; Enter confirms and Esc cancels an active
        move. X/Y/Z lock movement to a single axis (or Shift+X/Y/Z for plane
        constraint) during a move drag. Every other key falls through to the
        VTK interactor.'''
        key = event_key(event)
        if self._grabbing:
            lock_map: dict[int, str] = {
                Qt.Key.Key_X: "x",
                Qt.Key.Key_Y: "y",
                Qt.Key.Key_Z: "z",
            }
            if key in lock_map and self._set_axis_lock(
                lock_map[key],
                bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
            ):
                event.accept()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.confirm_grab()
                event.accept()
                return
            if key == Qt.Key.Key_Escape:
                self.cancel_grab()
                event.accept()
                return
        elif key == Qt.Key.Key_G:
            self.start_grab()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt naming
        '''Zoom natively (except during a move).'''
        if self._grabbing:
            event.accept()
            return
        super().wheelEvent(event)
        self._refresh_grid_labels()

    def initializeGL(self) -> None:  # noqa: N802 - Qt naming
        '''GL context setup: restore the native wide-line rendering path.

        pyvistaqt forces ``MaximumHardwareLineWidth`` to 1 inside its own
        ``initializeGL`` (it pins a core-profile context where line widths
        above 1 px are illegal). VTK then renders every wider line through
        its wide-line geometry shader, whose quad expansion draws fat,
        camera-angle dependent bands. With the compatibility profile from
        :func:`_use_compatibility_gl_profile` the native ``glLineWidth``
        path works and keeps every line width constant, so the hardware
        limit is re-applied after pyvistaqt's reset here.
        '''
        super().initializeGL()
        try:
            render_window = self.render_window
            if render_window is not None:
                render_window.SetForceMaximumHardwareLineWidth(MAX_HARDWARE_LINE_WIDTH)
        except Exception:  # noqa: BLE001 - window may already be torn down
            pass

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        '''Route a rapid second click through the press handler.

        Qt replaces the second press of a fast same-spot click with a
        MouseButtonDblClick; treating it as a press keeps every physical
        click's selection semantics instead of dropping it. Middle/right
        double clicks are swallowed (VTK never sees them).
        '''
        if event.button() == Qt.MouseButton.LeftButton:
            self.mousePressEvent(event)
            return
        event.accept()

    def _ensure_rubber_band(self) -> QRubberBand:
        '''The lazily created Qt rubber-band overlay of the frame selection.'''
        if self._rubber is None:
            self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        return self._rubber

    def _to_display(self, pos: QPoint) -> tuple[float, float]:
        '''Qt widget coordinates -> VTK display coordinates (y flipped).'''
        return float(pos.x()), float(self.size().height() - pos.y())

    # ------------------------------------------------------ camera navigation
    def _orbit_camera(self, dx: float, dy: float) -> None:
        '''Turntable-orbit the camera by a mouse delta (Qt pixels).

        Both axes grab the model - its front face follows the cursor:
        horizontal movement turns it around the world Z axis, vertical
        movement tips it up/down. The view up is carried through the
        rotations, so the orbit never rolls the camera and crossing the
        pole stays continuous.
        '''
        camera = self.renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        offset = np.asarray(camera.GetPosition(), dtype=np.float64) - focal
        up = np.asarray(camera.GetViewUp(), dtype=np.float64)
        width, height = self.size().width(), self.size().height()
        # Grab signs: dragging right turns the model right (the camera
        # orbits counterclockwise-negative), dragging down tips it down.
        azimuth = -ORBIT_DEGREES_PER_AXIS * dx / max(width, 1)
        elevation = ORBIT_DEGREES_PER_AXIS * dy / max(height, 1)
        offset, up = orbit_offset(offset, up, azimuth, elevation)
        camera.SetPosition(*(focal + offset))
        camera.SetFocalPoint(*focal)
        camera.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()
        self.render()
        self._refresh_grid_labels()

    def _pan_camera(self, last: QPoint, current: QPoint) -> None:
        '''Pan the camera so the world point under the cursor follows it.

        The display delta is unprojected at the focal-plane depth and
        applied as a translation of both the position and the focal point
        (the view direction is preserved).
        '''
        renderer = self.renderer
        camera = renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        renderer.SetWorldPoint(focal[0], focal[1], focal[2], 1.0)
        renderer.WorldToDisplay()
        depth = float(renderer.GetDisplayPoint()[2])
        x0, y0 = self._to_display(last)
        x1, y1 = self._to_display(current)
        renderer.SetDisplayPoint(x0, y0, depth)
        renderer.DisplayToWorld()
        world0 = np.asarray(renderer.GetWorldPoint()[:3], dtype=np.float64)
        renderer.SetDisplayPoint(x1, y1, depth)
        renderer.DisplayToWorld()
        world1 = np.asarray(renderer.GetWorldPoint()[:3], dtype=np.float64)
        delta = world0 - world1  # grab: the scene follows the cursor
        position = np.asarray(camera.GetPosition(), dtype=np.float64)
        camera.SetPosition(*(position + delta))
        camera.SetFocalPoint(*(focal + delta))
        renderer.ResetCameraClippingRange()
        self.render()
        self._refresh_grid_labels()

    # ------------------------------------------------------ scene selection
    def _pick_scene(self, pos: QPoint) -> tuple[str, int] | None:
        '''Depth-sorted block pick under the Qt cursor position.

        Args:
            pos: Cursor position in Qt widget coordinates.

        Returns:
            ``("block", index)`` when a block surface is hit, or None for
            empty space.
        '''
        x, y = self._to_display(pos)
        if not self._picker.Pick(x, y, 0, self.renderer):
            return None
        index = self._actor_map.get(self._picker.GetActor())
        return ("block", index) if index is not None else None

    def _finish_click(self, event: QMouseEvent) -> None:
        '''Apply the click semantics of an LMB press without a drag.

        A plain click selects the pressed block (or clears the selection on
        empty space); Ctrl+click toggles the pressed block in the selection.
        '''
        ctrl = self._press_ctrl or bool(
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        )
        hit = self._pick_scene(event.position().toPoint())
        if hit is None:
            if not ctrl:
                self._select_clear()
            return
        if ctrl:
            self._toggle_selection(hit[1])
        else:
            self._select_only(hit[1])

    def _select_from_rubber(self, rect: QRect, add: bool) -> None:
        '''Select every block fully inside the rubber-band rect (Qt coords).

        With *add* (Ctrl held) the framed blocks extend the current
        selection; otherwise the selection is replaced.

        Args:
            rect: Rubber-band rectangle in Qt widget coordinates.
            add: True to extend the selection instead of replacing it.
        '''
        selection = self._blocks_in_rect(rect)
        if add:
            selection |= self._selection
        self._selection = selection
        self._primary = None  # no clicked block: let the window pick one
        self._update_selection_outlines()
        self._emit_selection()

    def _blocks_in_rect(self, rect: QRect) -> set[int]:
        '''Indices of blocks whose screen bounding box lies fully in *rect*.'''
        inside: set[int] = set()
        for index, mesh in enumerate(self._scene_meshes):
            block_rect = self._block_screen_rect(index, mesh)
            if block_rect is not None and rect.contains(block_rect):
                inside.add(index)
        return inside

    def _block_screen_rect(self, index: int, mesh: pv.PolyData) -> QRect | None:
        '''Qt screen bounding rect of block *index*, or None off-screen.

        The eight world-space corners of the mesh bounds (plus the block
        position) are projected to display coordinates. A corner behind the
        camera projects to garbage, which just yields an oversized rect and
        a missed selection - the scene stays outside the camera in normal
        navigation.
        '''
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        meta = self._scene_meta[index]
        position_x, position_y = float(meta['pos_x']), float(meta['pos_y'])
        position_z = float(meta.get('pos_z', 0.0))
        renderer = self.renderer
        height = self.size().height()
        xs: list[float] = []
        ys: list[float] = []
        for x in (xmin, xmax):
            for y in (ymin, ymax):
                for z in (zmin, zmax):
                    renderer.SetWorldPoint(
                        x + position_x, y + position_y, z + position_z, 1.0
                    )
                    renderer.WorldToDisplay()
                    display_x, display_y, _ = renderer.GetDisplayPoint()[:3]
                    xs.append(display_x)
                    ys.append(height - display_y)
        return QRect(
            QPoint(round(min(xs)), round(min(ys))),
            QPoint(round(max(xs)), round(max(ys))),
        )

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

    def _select_clear(self) -> None:
        '''Clear the selection and notify.'''
        if not self._selection:
            return
        self._selection = set()
        self._primary = None
        self._update_selection_outlines()
        self._emit_selection()
