'''
File:   test_preview_widget.py
Brief:  Offscreen PreviewWidget tests: selection scheme, camera navigation, modes.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.16.0
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


# ------------------------------------------------------------- move drag (LMB)


def test_lmb_drag_on_block_moves_it(preview: PreviewWidget) -> None:
    '''A left drag on a block moves the selection and commits on release.'''
    received: list[dict] = []
    preview.blocks_moved.connect(lambda payload: received.append(payload))
    preview.set_selection({0}, 0)
    _mouse_drag(preview, _CENTER, QPoint(_CENTER.x() + 40, _CENTER.y()))
    meta = preview._scene_meta[0]
    assert meta['pos_x'] > 30.0
    assert meta['pos_y'] == pytest.approx(20.0)
    assert received and set(received[-1]) == {0}
    assert received[-1][0] == pytest.approx((meta['pos_x'], meta['pos_y']))
    for actor in preview._block_actors[0]:
        x, y, _ = actor.GetPosition()
        assert x == pytest.approx(meta['pos_x'])
        assert y == pytest.approx(20.0)
    assert not preview._grabbing


def test_lmb_press_on_block_selects_it(preview: PreviewWidget) -> None:
    '''Pressing on an unselected block selects it before the drag starts.'''
    QTest.mousePress(
        preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _CENTER
    )
    assert preview.selected_blocks() == {0}
    QTest.mouseRelease(
        preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _CENTER
    )


def test_lmb_click_on_block_without_move_selects(preview: PreviewWidget) -> None:
    '''A click on a block without a real move selects, never moves.'''
    received: list[dict] = []
    preview.blocks_moved.connect(lambda payload: received.append(payload))
    QTest.mouseClick(
        preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _CENTER
    )
    assert received == []
    assert preview.selected_blocks() == {0}
    assert preview._scene_meta[0]['pos_x'] == pytest.approx(30.0)


def test_lmb_subthreshold_jiggle_commits_nothing(preview: PreviewWidget) -> None:
    '''A sub-threshold drag is undone and falls back to the click.'''
    received: list[dict] = []
    preview.blocks_moved.connect(lambda payload: received.append(payload))
    preview.set_selection({0}, 0)
    _mouse_drag(preview, _CENTER, QPoint(_CENTER.x() + 2, _CENTER.y()))
    assert received == []
    assert preview._scene_meta[0]['pos_x'] == pytest.approx(30.0)
    assert preview.selected_blocks() == {0}


def test_ctrl_click_on_block_does_not_move(preview: PreviewWidget) -> None:
    '''Ctrl+click keeps the toggle semantics and never starts a move.'''
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        _CENTER,
    )
    assert preview.selected_blocks() == {0}
    assert preview._scene_meta[0]['pos_x'] == pytest.approx(30.0)
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        _CENTER,
    )
    assert preview.selected_blocks() == set()
    assert preview._scene_meta[0]['pos_x'] == pytest.approx(30.0)


def test_lmb_drag_on_block_snap_rounds_to_step(preview: PreviewWidget) -> None:
    '''With snap on, the dragged block lands on grid step multiples.'''
    preview.set_selection({0}, 0)
    preview.set_grid_unit(4.0)
    preview.set_grid_scale(0.5)  # 2 mm step
    preview.set_snap_enabled(True)
    _mouse_drag(preview, _CENTER, QPoint(_CENTER.x() + 40, _CENTER.y()))
    x = preview._scene_meta[0]['pos_x']
    assert x == pytest.approx(round(x / 2.0) * 2.0, abs=1e-9)
    assert x != pytest.approx(30.0)


def test_lmb_drag_moves_whole_selection(preview_two: PreviewWidget) -> None:
    '''Dragging a selected block shifts the whole selection by one delta.'''
    # The fixture camera is level with the floor: floor-plane moves need a
    # view raised above the horizon (see _FLOOR_RAY_MIN_SIN).
    camera = preview_two.renderer.GetActiveCamera()
    camera.SetPosition(50.0, -150.0, 60.0)
    camera.SetFocalPoint(50.0, 0.0, 2.5)
    camera.SetViewUp(0, 0, 1)
    preview_two.renderer.ResetCameraClippingRange()
    preview_two.set_selection({0, 1}, 0)
    press = _world_to_qt(preview_two, 0.0, 0.0, 5.0)
    _mouse_drag(preview_two, press, QPoint(press.x() + 30, press.y() - 10))
    meta = preview_two._scene_meta
    delta_x = meta[0]['pos_x'] - 0.0
    delta_y = meta[0]['pos_y'] - 0.0
    assert delta_x != 0.0
    assert meta[1]['pos_x'] == pytest.approx(100.0 + delta_x)
    assert meta[1]['pos_y'] == pytest.approx(0.0 + delta_y)
    assert preview_two.selected_blocks() == {0, 1}


def test_set_block_position_moves_outline(preview: PreviewWidget) -> None:
    '''Inspector moves relocate the selection outline with the block.'''
    preview.set_selection({0}, 0)
    preview.set_block_position(0, 60.0, 40.0)
    assert preview._outline_actors[0].GetPosition()[:2] == pytest.approx((60.0, 40.0))


# ------------------------------------------------------------- grab (G key)


def _hover_move(widget: PreviewWidget, pos: QPoint) -> None:
    '''A plain mouse-move event (no buttons held) at *pos*.'''
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, move)


def test_g_key_starts_grab(preview: PreviewWidget) -> None:
    '''Pressing G enters grab mode, Esc leaves it.'''
    preview.set_selection({0}, 0)
    QTest.keyClick(preview, Qt.Key.Key_G)
    assert preview._grabbing
    QTest.keyClick(preview, Qt.Key.Key_Escape)
    assert not preview._grabbing


def test_g_key_without_selection_is_noop(preview: PreviewWidget) -> None:
    '''G with an empty selection does not enter grab mode.'''
    QTest.keyClick(preview, Qt.Key.Key_G)
    assert not preview._grabbing


def test_grab_moves_selection_and_confirms_with_click(preview: PreviewWidget) -> None:
    '''A grab follows the mouse; an LMB click commits the new positions.'''
    received: list[dict] = []
    preview.blocks_moved.connect(lambda payload: received.append(payload))
    preview.set_selection({0}, 0)
    assert preview.start_grab(_CENTER)
    _hover_move(preview, QPoint(_CENTER.x() + 60, _CENTER.y() + 30))
    assert preview._scene_meta[0]['pos_x'] != pytest.approx(30.0)
    assert preview._scene_meta[0]['pos_y'] != pytest.approx(20.0)
    QTest.mouseClick(
        preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _CENTER
    )
    assert not preview._grabbing
    assert received and set(received[-1]) == {0}
    assert received[-1][0] == (
        pytest.approx(preview._scene_meta[0]['pos_x']),
        pytest.approx(preview._scene_meta[0]['pos_y']),
    )


def test_grab_cancel_restores_original_positions(preview: PreviewWidget) -> None:
    '''Esc during a grab restores the scene exactly as it was.'''
    preview.set_selection({0}, 0)
    assert preview.start_grab(_CENTER)
    _hover_move(preview, QPoint(_CENTER.x() + 60, _CENTER.y() + 30))
    assert preview._scene_meta[0]['pos_x'] != pytest.approx(30.0)
    QTest.keyClick(preview, Qt.Key.Key_Escape)
    assert not preview._grabbing
    assert preview._scene_meta[0]['pos_x'] == pytest.approx(30.0)
    assert preview._scene_meta[0]['pos_y'] == pytest.approx(20.0)
    for actor in preview._block_actors[0]:
        assert actor.GetPosition()[:2] == pytest.approx((30.0, 20.0))


def test_grab_moves_whole_selection_by_same_delta(preview_two: PreviewWidget) -> None:
    '''A grab of a multi-selection shifts every block by the same delta.'''
    preview_two.set_selection({0, 1}, 1)
    assert preview_two.start_grab(_CENTER)
    _hover_move(preview_two, QPoint(_CENTER.x() + 50, _CENTER.y()))
    meta = preview_two._scene_meta
    delta_x = meta[1]['pos_x'] - 100.0
    delta_y = meta[1]['pos_y'] - 0.0
    assert meta[0]['pos_x'] == pytest.approx(0.0 + delta_x)
    assert meta[0]['pos_y'] == pytest.approx(0.0 + delta_y)
    QTest.keyClick(preview_two, Qt.Key.Key_Escape)


def test_grab_snap_positions_on_grid(preview: PreviewWidget) -> None:
    '''With snap on, both grabbed coordinates land on grid step multiples.'''
    preview.set_selection({0}, 0)
    preview.set_grid_unit(4.0)
    preview.set_grid_scale(0.5)  # 2 mm step
    preview.set_snap_enabled(True)
    assert preview.start_grab(_CENTER)
    _hover_move(preview, QPoint(_CENTER.x() + 37, _CENTER.y() - 13))
    preview.confirm_grab()
    x = preview._scene_meta[0]['pos_x']
    y = preview._scene_meta[0]['pos_y']
    assert x == pytest.approx(round(x / 2.0) * 2.0, abs=1e-9)
    assert y == pytest.approx(round(y / 2.0) * 2.0, abs=1e-9)


def test_grab_blocked_navigation(preview: PreviewWidget) -> None:
    '''MMB orbit and the wheel are ignored while a grab is active.'''
    preview.set_selection({0}, 0)
    assert preview.start_grab(_CENTER)
    camera = preview.renderer.GetActiveCamera()
    position_before = list(camera.GetPosition())
    _mouse_drag(
        preview,
        QPoint(200, 150),
        QPoint(300, 150),
        button=Qt.MouseButton.MiddleButton,
    )
    assert list(camera.GetPosition()) == position_before
    QTest.keyClick(preview, Qt.Key.Key_Escape)


# --------------------------------------------------------------- floor grid


def test_grid_actors_created_and_cleared(preview: PreviewWidget) -> None:
    '''Enabling the grid builds the line/axis/label actors, disabling clears.'''
    preview.set_grid_unit(4.0)
    preview.set_grid_scale(0.5)
    assert preview.grid_step() == pytest.approx(2.0)
    preview.set_grid_enabled(True)
    # Minor lines, major lines, X axis, Y axis, X labels, Y labels.
    assert len(preview._grid_actors) >= 6
    preview.set_grid_enabled(False)
    assert preview._grid_actors == []


def test_grid_scale_extends_step(preview: PreviewWidget) -> None:
    '''The grid step is the unit size times the cell scale factor.'''
    preview.set_grid_unit(4.0)
    preview.set_grid_scale(1.0)
    assert preview.grid_step() == pytest.approx(4.0)
    preview.set_grid_scale(10.0)
    assert preview.grid_step() == pytest.approx(40.0)
    preview.set_grid_scale(0.5)
    assert preview.grid_step() == pytest.approx(2.0)


def test_grid_present_with_empty_scene(preview: PreviewWidget) -> None:
    '''Grid is visible even when the scene has no blocks.'''
    preview.set_grid_unit(4.0)
    preview.set_grid_scale(0.5)
    preview.set_grid_enabled(True)
    # Clear the scene and call update_scene with empty lists.
    preview.update_scene([], [], [])
    assert len(preview._grid_actors) >= 6


def test_grid_survives_scene_update(preview: PreviewWidget) -> None:
    '''Grid actors persist across an update_scene call.'''
    preview.set_grid_unit(4.0)
    preview.set_grid_scale(0.5)
    preview.set_grid_enabled(True)
    grid_actors_before = list(preview._grid_actors)
    # Re-trigger the scene with the same data.
    meshes = [preview._scene_meshes[0]] if preview._scene_meshes else [_cube_mesh()]
    preview.update_scene(meshes, [BLOCK_COLOR], [_scene_meta(30.0, 20.0)])
    assert len(preview._grid_actors) >= 6
    # The grid actor objects are the same instances (not rebuilt).
    for old_actor in grid_actors_before:
        assert old_actor in preview._grid_actors


def test_grid_rebuilt_by_scale_and_unit(preview: PreviewWidget) -> None:
    '''set_grid_scale / set_grid_unit rebuild the grid actors.'''
    preview.set_grid_unit(4.0)
    preview.set_grid_scale(0.5)
    preview.set_grid_enabled(True)
    first_actors = list(preview._grid_actors)
    preview.set_grid_scale(1.0)
    assert len(preview._grid_actors) != len(first_actors) or (
        len(preview._grid_actors) > 0 and preview._grid_actors[0] is not first_actors[0]
    )
    second_actors = list(preview._grid_actors)
    preview.set_grid_unit(8.0)
    assert len(preview._grid_actors) != len(second_actors) or (
        len(preview._grid_actors) > 0
        and preview._grid_actors[0] is not second_actors[0]
    )


def test_incremental_update_preserves_unchanged_actors(
    preview_two: PreviewWidget,
) -> None:
    '''Incremental update: changed=True rebuilds, changed=False keeps actors.'''
    preview_two.set_display_mode(MODE_SOLID)
    actors_idx0_old = list(preview_two._block_actors[0])
    actors_idx1_old = list(preview_two._block_actors[1])

    # Build a new mesh for block 0 only; None for block 1 (keep old).
    new_mesh = _cube_mesh()
    preview_two.update_scene(
        [new_mesh, None],
        [BLOCK_COLOR, '#bf7f3f'],
        [_scene_meta(0.0, 0.0), _scene_meta(100.0, 0.0)],
        changed=[True, False],
    )
    # Block 0: new actors.
    assert preview_two._block_actors[0] != actors_idx0_old
    # Block 1: same actor objects.
    for old_actor in actors_idx1_old:
        assert old_actor in preview_two._block_actors[1]
    # Actor map: every pickable actor maps to a valid index.
    for actor, idx in preview_two._actor_map.items():
        assert idx < len(preview_two._scene_meshes)


def test_changed_none_triggers_full_rebuild(preview: PreviewWidget) -> None:
    '''changed=None causes a full rebuild of all actors.'''
    preview.set_display_mode(MODE_SOLID)
    actors_before = list(preview._block_actors[0])
    preview.update_scene(
        [_cube_mesh()],
        [BLOCK_COLOR],
        [_scene_meta(30.0, 20.0)],
        changed=None,
    )
    # All actors are new objects after a full rebuild.
    assert preview._block_actors[0] != actors_before


def test_empty_scene_update_removes_block_actors(
    preview_two: PreviewWidget,
) -> None:
    '''Regression: emptying the scene must remove the block actors from the
    renderer — the early-return-before-clear left them as visible ghosts.'''
    preview_two.set_display_mode(MODE_SOLID)
    block_actors = [a for actors in preview_two._block_actors for a in actors]
    assert block_actors

    preview_two.update_scene([], [], [], [])

    assert preview_two._block_actors == []
    assert preview_two._actor_map == {}
    # None of the previously rendered block actors remains in the renderer.
    renderer_actors = set(map(id, preview_two.actors.values()))
    for actor in block_actors:
        assert id(actor) not in renderer_actors
