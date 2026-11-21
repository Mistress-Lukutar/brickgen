'''
File:   test_preview_widget.py
Brief:  Offscreen PreviewWidget tests: selection scheme, camera navigation, modes.
Author: Mistress-Lukutar
Date:   2026-11-21
Version: v0.13.0
'''

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import pyvista as pv
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from brickgen.gui.preview import (
    BLOCK_COLOR,
    MODE_SOLID,
    MODE_SOLID_FEATURE_EDGES,
    MODE_SOLID_WIREFRAME,
    MODE_WIREFRAME,
    PreviewWidget,
)

# vtk 9.6 numpy_support sets .shape on arrays, deprecated in numpy 2.5; the
# warning fires on every cell_data read and is a VTK-internal issue, so the
# filter must be active before preview (and thus VTK) is imported. QTest
# cannot simulate a drag with the button held, so the helper below sends a
# hand-built QMouseEvent whose classic ctor is deprecated in PySide6.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Setting the shape on a NumPy array:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Function.*QMouseEvent.*is marked as deprecated:DeprecationWarning"
    ),
]

_W, _H = 400, 300
_CENTER = QPoint(_W // 2, _H // 2)

# VTK property representation ids (vtkProperty::RepresentationType).
_VTK_WIREFRAME = 1
_VTK_SURFACE = 2


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


def _cube_mesh() -> pv.PolyData:
    '''Triangulated 10x10x5 block mesh on z=0..5 with a semantic boundary.'''
    mesh = pv.Cube(x_length=10, y_length=10, z_length=5, center=(0, 0, 2.5))
    mesh = mesh.triangulate()  # pv.Cube yields quads; the filters return new meshes
    parts = np.zeros(mesh.n_cells, dtype=np.int8)
    parts[0] = 1  # one tagged triangle -> a part-boundary line for the edges
    mesh.cell_data['part'] = parts
    return mesh


def _scene_meta(pos_x: float, pos_y: float) -> dict:
    '''Scene metadata entry of one test block.'''
    return {
        'pos_x': pos_x,
        'pos_y': pos_y,
        'half_x': 5.0,
        'half_y': 5.0,
        'top_z': 5.0,
        'quantity': 1,
    }


@contextmanager
def _preview_widget(
    meshes: list[pv.PolyData],
    meta: list[dict],
    camera: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> Iterator[PreviewWidget]:
    '''A PreviewWidget with the given scene and a deterministic camera.

    The camera looks along +Y from *camera[0]* at *camera[1]* with world-up
    view up, so the focal point projects exactly to the widget center. The
    render window size is pinned because the offscreen platform never
    realizes a real OpenGL context (matrices are computed on demand).
    '''
    _ensure_app()
    widget = PreviewWidget()
    widget.resize(_W, _H)
    widget.renderer.GetRenderWindow().SetSize(_W, _H)
    widget.update_scene(meshes, [BLOCK_COLOR, '#bf7f3f'][: len(meshes)], meta)
    position, focal = camera
    cam = widget.renderer.GetActiveCamera()
    cam.SetPosition(*position)
    cam.SetFocalPoint(*focal)
    cam.SetViewUp(0, 0, 1)
    cam.SetViewAngle(30.0)
    widget.renderer.ResetCameraClippingRange()
    try:
        yield widget
    finally:
        widget.close()


@pytest.fixture()
def preview() -> Iterator[PreviewWidget]:
    '''One block placed at (30, 20) under the widget center.'''
    with _preview_widget(
        [_cube_mesh()],
        [_scene_meta(30.0, 20.0)],
        camera=((30.0, -100.0, 20.0), (30.0, 20.0, 2.5)),
    ) as widget:
        yield widget


@pytest.fixture()
def preview_two() -> Iterator[PreviewWidget]:
    '''Two blocks at (0, 0) and (100, 0), both fully inside the view.'''
    with _preview_widget(
        [_cube_mesh(), _cube_mesh()],
        [_scene_meta(0.0, 0.0), _scene_meta(100.0, 0.0)],
        camera=((50.0, -200.0, 2.5), (50.0, 0.0, 2.5)),
    ) as widget:
        yield widget


def _world_to_qt(widget: PreviewWidget, x: float, y: float, z: float) -> QPoint:
    '''Project a world point to Qt widget coordinates (y flipped).'''
    renderer = widget.renderer
    renderer.SetWorldPoint(float(x), float(y), float(z), 1.0)
    renderer.WorldToDisplay()
    display_x, display_y, _ = renderer.GetDisplayPoint()[:3]
    return QPoint(round(display_x), round(widget.height() - display_y))


def _mouse_drag(
    widget: PreviewWidget,
    start: QPoint,
    end: QPoint,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    modifier: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    '''Press-move-release sequence (QTest cannot simulate held drags).'''
    QTest.mousePress(widget, button, modifier, start)
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(end),
        Qt.MouseButton.NoButton,
        button,
        modifier,
    )
    QApplication.sendEvent(widget, move)
    QTest.mouseRelease(widget, button, modifier, end)


def _camera_offset(widget: PreviewWidget) -> np.ndarray:
    '''Focal-to-camera offset vector of the widget camera.'''
    camera = widget.renderer.GetActiveCamera()
    return np.asarray(camera.GetPosition(), dtype=float) - np.asarray(
        camera.GetFocalPoint(), dtype=float
    )


# ------------------------------------------------------------------ modes


@pytest.mark.parametrize(
    'mode',
    [
        MODE_SOLID,
        MODE_SOLID_FEATURE_EDGES,
        MODE_SOLID_WIREFRAME,
        MODE_WIREFRAME,
    ],
)
def test_mode_actors_follow_block_position(preview: PreviewWidget, mode: str) -> None:
    '''Every actor of a block sits at the block position, not the origin.'''
    preview.set_display_mode(mode)
    actors = preview._block_actors[0]
    assert actors
    for actor in actors:
        x, y, z = actor.GetPosition()
        assert (x, y, z) == pytest.approx((30.0, 20.0, 0.0))


def test_solid_mode_single_surface(preview: PreviewWidget) -> None:
    '''MODE_SOLID renders exactly one pickable shaded surface.'''
    preview.set_display_mode(MODE_SOLID)
    actors = preview._block_actors[0]
    assert len(actors) == 1
    assert actors[0].GetProperty().GetRepresentation() == _VTK_SURFACE
    assert actors[0].GetPickable()


def test_feature_edges_mode_pairs_surface_with_lines(
    preview: PreviewWidget,
) -> None:
    '''MODE_SOLID_FEATURE_EDGES: pickable surface + non-pickable outlines.'''
    preview.set_display_mode(MODE_SOLID_FEATURE_EDGES)
    actors = preview._block_actors[0]
    assert len(actors) == 2
    assert actors[0].GetProperty().GetRepresentation() == _VTK_SURFACE
    assert actors[0].GetPickable()
    # The outline actor is a lines-only PolyData: rendered as 2 px lines,
    # excluded from picking so clicks go to the solid underneath.
    assert not actors[1].GetPickable()
    assert actors[1].GetProperty().GetLineWidth() == pytest.approx(2.0)


def test_solid_wireframe_mode_pairs_surface_with_lines(
    preview: PreviewWidget,
) -> None:
    '''MODE_SOLID_WIREFRAME: pickable surface + mesh wireframe overlay.'''
    preview.set_display_mode(MODE_SOLID_WIREFRAME)
    actors = preview._block_actors[0]
    assert len(actors) == 2
    assert actors[0].GetProperty().GetRepresentation() == _VTK_SURFACE
    assert actors[1].GetProperty().GetRepresentation() == _VTK_WIREFRAME


def test_wireframe_mode_has_no_solid(preview: PreviewWidget) -> None:
    '''MODE_WIREFRAME renders a single wireframe that is itself pickable.'''
    preview.set_display_mode(MODE_WIREFRAME)
    actors = preview._block_actors[0]
    assert len(actors) == 1
    actor = actors[0]
    assert actor.GetProperty().GetRepresentation() == _VTK_WIREFRAME
    assert actor.GetPickable()
    assert preview._actor_map.get(actor) == 0


def test_set_block_position_moves_decorations(preview: PreviewWidget) -> None:
    '''Inspector moves relocate every actor of the block, decorations too.'''
    for mode in (MODE_SOLID_FEATURE_EDGES, MODE_SOLID_WIREFRAME, MODE_WIREFRAME):
        preview.set_display_mode(mode)
        preview.set_block_position(0, 50.0, 60.0)
        for actor in preview._block_actors[0]:
            x, y, z = actor.GetPosition()
            assert (x, y, z) == pytest.approx((50.0, 60.0, 0.0)), mode


# ------------------------------------------------------------- selection


def test_click_selects_block(preview: PreviewWidget) -> None:
    '''A plain click on a block selects it and emits the signal.'''
    received: list[tuple[frozenset, int | None]] = []
    preview.selection_changed.connect(lambda payload: received.append(payload))
    QTest.mouseClick(
        preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _CENTER
    )
    assert preview.selected_blocks() == {0}
    assert received and received[-1] == (frozenset({0}), 0)


def test_click_empty_space_clears_selection(preview: PreviewWidget) -> None:
    '''A plain click on empty space clears the selection.'''
    preview.set_selection({0}, 0)
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(15, 15),
    )
    assert preview.selected_blocks() == set()


def test_ctrl_click_toggles_block(preview: PreviewWidget) -> None:
    '''Ctrl+click adds the block to the selection, again removes it.'''
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        _CENTER,
    )
    assert preview.selected_blocks() == {0}
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        _CENTER,
    )
    assert preview.selected_blocks() == set()


def test_rubber_band_selects_blocks_inside(preview_two: PreviewWidget) -> None:
    '''A drag frame selects every block fully inside it.'''
    _mouse_drag(preview_two, QPoint(5, 5), QPoint(_W - 5, _H - 5))
    assert preview_two.selected_blocks() == {0, 1}


def test_rubber_band_requires_full_containment(preview: PreviewWidget) -> None:
    '''A frame overlapping the block without covering it selects nothing.'''
    preview.set_selection({0}, 0)
    # The block projects to roughly x 176..224, y 136..165: cutting the
    # bottom of the bounding box must not select it.
    _mouse_drag(preview, QPoint(150, 100), QPoint(250, 145))
    assert preview.selected_blocks() == set()


def test_ctrl_rubber_band_extends_selection(preview_two: PreviewWidget) -> None:
    '''Ctrl+drag frames extend the current selection.'''
    preview_two.set_selection({0}, 0)
    center = _world_to_qt(preview_two, 100.0, 0.0, 2.5)
    _mouse_drag(
        preview_two,
        QPoint(center.x() - 40, center.y() - 40),
        QPoint(center.x() + 40, center.y() + 40),
        modifier=Qt.KeyboardModifier.ControlModifier,
    )
    assert preview_two.selected_blocks() == {0, 1}


def test_lmb_drag_never_rotates_camera(preview: PreviewWidget) -> None:
    '''A left-button drag is a frame: the camera stays exactly where it was.'''
    camera = preview.renderer.GetActiveCamera()
    position_before = list(camera.GetPosition())
    _mouse_drag(preview, QPoint(200, 150), QPoint(300, 250))
    assert list(camera.GetPosition()) == position_before


# -------------------------------------------------------------- navigation


def test_mmb_horizontal_drag_turns_model_right(preview: PreviewWidget) -> None:
    '''MMB drag right turns the model right: height and radius are kept.'''
    camera = preview.renderer.GetActiveCamera()
    offset_before = _camera_offset(preview)
    _mouse_drag(
        preview,
        QPoint(200, 150),
        QPoint(400, 150),
        button=Qt.MouseButton.MiddleButton,
    )
    offset_after = _camera_offset(preview)
    assert offset_after[2] == pytest.approx(offset_before[2])
    assert float(np.linalg.norm(offset_after)) == pytest.approx(
        float(np.linalg.norm(offset_before))
    )
    # Grab: the model turns right with the cursor, so the camera azimuth
    # decreases - 200 px of a 400 px wide drag at ORBIT_DEGREES_PER_AXIS.
    azimuth_before = math.atan2(offset_before[1], offset_before[0])
    azimuth_after = math.atan2(offset_after[1], offset_after[0])
    delta = math.degrees(
        (azimuth_after - azimuth_before + math.pi) % (2.0 * math.pi) - math.pi
    )
    assert delta == pytest.approx(-100.0)
    # No roll: the view up keeps zero component along the camera right
    # axis (it stays in the vertical plane spanned by the view direction
    # and the world Z).
    view_dir = -offset_after / float(np.linalg.norm(offset_after))
    right = np.cross(view_dir, (0.0, 0.0, 1.0))
    right = right / float(np.linalg.norm(right))
    up = np.asarray(camera.GetViewUp(), dtype=float)
    assert float(np.dot(up, right)) == pytest.approx(0.0, abs=1e-9)


def test_mmb_vertical_drag_grabs_model(preview: PreviewWidget) -> None:
    '''Vertical drags grab the model: up tips it up, the camera descends.'''
    camera = preview.renderer.GetActiveCamera()
    z_before = camera.GetPosition()[2]
    _mouse_drag(
        preview,
        QPoint(200, 200),
        QPoint(200, 80),
        button=Qt.MouseButton.MiddleButton,
    )
    z_lowered = camera.GetPosition()[2]
    assert z_lowered < z_before
    # And dragging down raises the camera again (model front follows down).
    _mouse_drag(
        preview,
        QPoint(200, 80),
        QPoint(200, 140),
        button=Qt.MouseButton.MiddleButton,
    )
    z_raised = camera.GetPosition()[2]
    assert z_raised > z_lowered


def test_mmb_drag_does_not_change_selection(preview: PreviewWidget) -> None:
    '''MMB navigation leaves the scene selection untouched.'''
    preview.set_selection({0}, 0)
    _mouse_drag(
        preview,
        QPoint(200, 150),
        QPoint(300, 150),
        button=Qt.MouseButton.MiddleButton,
    )
    assert preview.selected_blocks() == {0}


def test_shift_mmb_pans_camera(preview: PreviewWidget) -> None:
    '''Shift+MMB translates the camera: the scene follows the cursor.'''
    camera = preview.renderer.GetActiveCamera()
    position_before = np.asarray(camera.GetPosition(), dtype=float)
    focal_before = np.asarray(camera.GetFocalPoint(), dtype=float)
    _mouse_drag(
        preview,
        QPoint(200, 150),
        QPoint(300, 150),
        button=Qt.MouseButton.MiddleButton,
        modifier=Qt.KeyboardModifier.ShiftModifier,
    )
    position_after = np.asarray(camera.GetPosition(), dtype=float)
    focal_after = np.asarray(camera.GetFocalPoint(), dtype=float)
    # Pure translation: the view direction is preserved.
    np.testing.assert_allclose(
        (position_after - position_before) - (focal_after - focal_before),
        0.0,
        atol=1e-9,
    )
    # Dragging right moves the camera left along the world X (view +Y).
    assert focal_after[0] < focal_before[0]
