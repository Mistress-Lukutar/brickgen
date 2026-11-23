'''
File:   test_parts_panel.py
Brief:  Offscreen PlannedPartsPanel tests: per-variant thumbnail rendering.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import pyvista as pv
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from brickgen.gui import parts_panel
from brickgen.gui.parts_panel import PlannedPartsPanel
from brickgen.gui.preview import mesh_to_arrays
from brickgen.services.assemble_service import PartGroup, PartVariant


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


class _FakeJobs(QObject):
    '''Duck-typed JobQueue: records submits, delivers results manually.'''

    job_done = Signal(int, object)

    def __init__(self) -> None:
        super().__init__()
        self._next_id = 0
        self.submits: list[tuple[str, dict]] = []

    def submit(self, task: str, payload: dict, *, key: str | None = None) -> int:
        '''Record the submit and return a fresh job id.'''
        self._next_id += 1
        self.submits.append((task, payload))
        return self._next_id


@pytest.fixture()
def jobs() -> Iterator[_FakeJobs]:
    '''A fake job queue (no worker processes).'''
    _ensure_app()
    queue = _FakeJobs()
    try:
        yield queue
    finally:
        queue.deleteLater()


@pytest.fixture()
def rendered(monkeypatch: pytest.MonkeyPatch) -> dict[str, QPixmap]:
    '''Patch render_thumbnail: one pixmap per color, filled with the color.'''
    pixmaps: dict[str, QPixmap] = {}

    def fake_render(mesh: pv.PolyData, color: str = '') -> QPixmap:
        pixmap = QPixmap(4, 4)
        pixmap.fill(QColor(color if color else '#ffffff'))
        pixmaps[color] = pixmap
        return pixmap

    monkeypatch.setattr(parts_panel, 'render_thumbnail', fake_render)
    return pixmaps


def _thumb_color(card: parts_panel._PartCard) -> str | None:
    '''Fill color of the card's thumbnail (None while empty).'''
    pixmap = card._thumb.pixmap()
    if pixmap is None or pixmap.isNull():
        return None
    return pixmap.toImage().pixelColor(0, 0).name()


def _group(colors: list[str]) -> PartGroup:
    '''One part group with a color variant per given color.'''
    key = json.dumps({'cells_x': 2, 'cells_y': 2})
    return PartGroup(
        key=key,
        name='2x2-studs',
        variants=[
            PartVariant(
                block_index=index,
                uid=f'u{index}',
                color=color,
                quantity=1,
                linked=False,
            )
            for index, color in enumerate(colors)
        ],
    )


def _mesh_arrays() -> dict:
    '''Mesh arrays payload standing in for a worker thumbnail build.'''
    cube = pv.Cube(x_length=10, y_length=10, z_length=5, center=(0, 0, 2.5))
    return mesh_to_arrays(cube.triangulate())


def test_initial_thumbnail_uses_first_variant(
    jobs: _FakeJobs, rendered: dict[str, QPixmap]
) -> None:
    '''The initial submission renders the first variant's color.'''
    panel = PlannedPartsPanel(jobs)
    panel.refresh([_group(['#ff0000', '#00ff00'])], {})
    card = panel._cards[0]
    assert len(panel._cards) == 1
    assert jobs.submits and jobs.submits[0][0] == 'thumbnail'
    assert card._thumb.pixmap().isNull()  # nothing rendered yet

    panel._on_job_done(1, _mesh_arrays())
    assert _thumb_color(card) == '#ff0000'
    panel.deleteLater()


def test_combo_change_recolors_thumbnail(
    jobs: _FakeJobs, rendered: dict[str, QPixmap]
) -> None:
    '''Selecting another color rebuilds the thumbnail in that color.'''
    panel = PlannedPartsPanel(jobs)
    panel.refresh([_group(['#ff0000', '#00ff00'])], {})
    panel._on_job_done(1, _mesh_arrays())
    card = panel._cards[0]
    assert _thumb_color(card) == '#ff0000'

    card._combo.setCurrentIndex(1)
    assert len(jobs.submits) == 2  # a build for the second color is queued
    panel._on_job_done(2, _mesh_arrays())
    assert _thumb_color(card) == '#00ff00'

    # Switching back hits the cache: no third submit, instant recolor.
    card._combo.setCurrentIndex(0)
    assert len(jobs.submits) == 2
    assert _thumb_color(card) == '#ff0000'
    panel.deleteLater()


def test_stale_color_result_keeps_selection(
    jobs: _FakeJobs, rendered: dict[str, QPixmap]
) -> None:
    '''A late result for a no-longer-selected color never overwrites.'''
    panel = PlannedPartsPanel(jobs)
    panel.refresh([_group(['#ff0000', '#00ff00'])], {})
    card = panel._cards[0]
    card._combo.setCurrentIndex(1)  # switch before variant-0 lands

    panel._on_job_done(1, _mesh_arrays())  # the stale variant-0 build
    assert card._thumb.pixmap().isNull()  # card keeps waiting for its color
    assert '#ff0000' in rendered  # ...but the pixmap is rendered and cached

    panel._on_job_done(2, _mesh_arrays())  # the selected variant's build
    assert _thumb_color(card) == '#00ff00'
    panel.deleteLater()


def test_duplicate_variant_build_is_not_requeued(
    jobs: _FakeJobs, rendered: dict[str, QPixmap]
) -> None:
    '''Rapid combo flips do not queue a second build for the same color.'''
    panel = PlannedPartsPanel(jobs)
    panel.refresh([_group(['#ff0000', '#00ff00'])], {})
    card = panel._cards[0]
    card._combo.setCurrentIndex(1)
    assert len(jobs.submits) == 2
    card._combo.setCurrentIndex(1)  # same selection again: no new build
    assert len(jobs.submits) == 2
    panel.deleteLater()


def test_foreign_job_result_ignored(
    jobs: _FakeJobs, rendered: dict[str, QPixmap]
) -> None:
    '''Results with unknown job ids (generation/export) change nothing.'''
    panel = PlannedPartsPanel(jobs)
    panel.refresh([_group(['#ff0000'])], {})
    panel._on_job_done(999, _mesh_arrays())
    assert rendered == {}
    assert _thumb_color(panel._cards[0]) is None
    panel.deleteLater()
