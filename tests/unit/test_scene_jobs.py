'''
File:   test_scene_jobs.py
Brief:  Worker-handler tests for multi-block scene build and export.
Author: Mistress-Lukutar
Date:   2026-11-20
Version: v0.13.0
'''

import os
import tempfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from brickgen.core.models import BlockSettings  # noqa: E402
from brickgen.gui.jobs import JobError, JobQueue  # noqa: E402
from brickgen.gui.preview import mesh_from_arrays  # noqa: E402

# --------------------------------------------------------------------------- helpers


def _ensure_app() -> QApplication:
    '''Return the singleton QApplication, creating it if necessary.'''
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_until(predicate, timeout: float = 120.0) -> None:
    '''Block until *predicate* returns True or *timeout* seconds elapse.'''
    app = _ensure_app()
    import time  # noqa: E402 - inside function to avoid top-level import

    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert predicate(), 'condition not met within timeout'


def _collect_results(q: JobQueue):
    '''Return a list that captures ``(job_id, result)`` from *q.job_done*.'''
    results: list[tuple[int, object]] = []
    q.job_done.connect(lambda job_id, result: results.append((job_id, result)))
    return results


# --------------------------------------------------------------------------- fixtures

BLOCK_A: dict = {
    'position_x': 0.0,
    'position_y': 0.0,
    'settings': BlockSettings(cells_x=2, cells_y=1).model_dump(),
}
BLOCK_B: dict = {
    'position_x': 12.0,
    'position_y': 0.0,
    'settings': BlockSettings(cells_x=1, cells_y=1, pins_enabled=False).model_dump(),
}


# ===========================================================================
# Test 1: test_build_scene_handler
# ===========================================================================


def test_build_scene_handler():
    '''Build a two-block scene and validate mesh arrays + reconstructed meshes.'''
    q = JobQueue()
    results = _collect_results(q)

    try:
        q.submit(
            'build_scene',
            {
                'blocks': [BLOCK_A, BLOCK_B],
                'tolerance': 0.15,
                'angular': 0.1,
            },
        )

        _wait_until(lambda: len(results) == 1, timeout=120.0)

        assert len(results) == 1
        _, result = results[0]
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert 'blocks' in result
        blocks_list = result['blocks']
        assert isinstance(blocks_list, list) and len(blocks_list) == 2

        for blk_arrays in blocks_list:
            assert isinstance(blk_arrays, dict)
            assert 'points' in blk_arrays
            assert 'faces' in blk_arrays
            assert 'part' in blk_arrays
            pts = blk_arrays['points']
            fcs = blk_arrays['faces']
            prt = blk_arrays['part']
            assert isinstance(pts, np.ndarray) and pts.dtype == np.float64
            assert pts.shape[1] == 3
            assert len(pts) > 0
            assert isinstance(fcs, np.ndarray) and fcs.dtype == np.int64
            assert len(fcs) >= 3
            assert isinstance(prt, np.ndarray) and prt.dtype == np.int8

        # Reconstruct meshes and verify non-empty cells.
        for blk_arrays in blocks_list:
            mesh = mesh_from_arrays(blk_arrays)
            assert mesh.n_cells > 0

        # Block A (2x1) has a bounding box ~8 units wide on X; block B (1x1) ~4.
        pts_a = blocks_list[0]['points']
        pts_b = blocks_list[1]['points']
        width_a = np.ptp(pts_a[:, 0])
        width_b = np.ptp(pts_b[:, 0])
        assert abs(width_a - 8.0) < 1e-6, f'Block A width {width_a} != ~8.0'
        assert abs(width_b - 4.0) < 1e-6, f'Block B width {width_b} != ~4.0'

    finally:
        q.close()


# ===========================================================================
# Test 2: test_export_scene_handler
# ===========================================================================


def test_export_scene_handler():
    '''Export a two-block scene to STEP and verify the output file.'''
    q = JobQueue()
    results = _collect_results(q)

    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        q.submit(
            'export_scene',
            {
                'blocks': [BLOCK_A, BLOCK_B],
                'path': tmp_path,
                'format': 'STEP',
                'tolerance': 0.1,
                'angular': 0.1,
            },
        )

        _wait_until(lambda: len(results) == 1, timeout=120.0)

        assert len(results) == 1
        _, result = results[0]
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert 'written' in result
        written_path = result['written']
        from pathlib import Path as _Path  # noqa: E402

        assert _Path(written_path).exists(), f'{written_path} does not exist'
        assert _Path(written_path).stat().st_size > 0, 'File is empty'

    finally:
        q.close()


# ===========================================================================
# Test 3: test_export_scene_empty_rejected
# ===========================================================================


def test_export_scene_empty_rejected():
    '''An empty blocks list raises a JobError mentioning "no blocks".'''
    q = JobQueue()
    results = _collect_results(q)

    try:
        q.submit(
            'export_scene',
            {
                'blocks': [],
                'path': tempfile.mktemp(suffix='.step'),
                'format': 'STEP',
                'tolerance': 0.1,
                'angular': 0.1,
            },
        )

        _wait_until(lambda: len(results) == 1, timeout=120.0)

        assert len(results) == 1
        _, result = results[0]
        assert isinstance(result, JobError), f'Expected JobError, got {type(result)}'
        assert 'no blocks' in str(result.exc) or 'no blocks' in result.message

    finally:
        q.close()
