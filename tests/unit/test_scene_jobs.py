'''
File:   test_scene_jobs.py
Brief:  Worker-handler tests for multi-block scene build and export.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.18.0
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
# Test 2: test_export_print_handler
# ===========================================================================


def test_export_print_handler():
    '''Export one print bed (two pieces) to STEP and verify the output file.'''
    q = JobQueue()
    results = _collect_results(q)

    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp:
        tmp_path = tmp.name

    bed_payload = {
        'index': 0,
        'path': tmp_path,
        'pieces': [
            {
                'settings': BLOCK_A['settings'],
                'position_x': 0.0,
                'position_y': 0.0,
                'flipped': False,
            },
            {
                'settings': BLOCK_B['settings'],
                'position_x': 20.0,
                'position_y': 0.0,
                'flipped': False,
            },
        ],
    }

    try:
        q.submit(
            'export_print',
            {
                'beds': [bed_payload],
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
        written_list = result['written']
        assert isinstance(written_list, list) and len(written_list) == 1
        written_path = written_list[0]
        from pathlib import Path as _Path  # noqa: E402

        assert _Path(written_path).exists(), f'{written_path} does not exist'
        assert _Path(written_path).stat().st_size > 0, 'File is empty'

    finally:
        q.close()


# ===========================================================================
# Test 3: test_export_print_empty_rejected
# ===========================================================================


def test_export_print_empty_rejected():
    '''An empty beds list raises a JobError mentioning "no beds".'''
    q = JobQueue()
    results = _collect_results(q)

    try:
        q.submit(
            'export_print',
            {
                'beds': [],
                'format': 'STEP',
                'tolerance': 0.1,
                'angular': 0.1,
            },
        )

        _wait_until(lambda: len(results) == 1, timeout=120.0)

        assert len(results) == 1
        _, result = results[0]
        assert isinstance(result, JobError), f'Expected JobError, got {type(result)}'
        assert 'no beds' in str(result.exc) or 'no beds' in result.message

    finally:
        q.close()


# ===========================================================================
# Test 4: test_build_scene_cache_hit_rebuild
# ===========================================================================


def test_build_scene_cache_hit_rebuild():
    '''Same payload submitted twice: second call reports cached == [True, ...].'''
    from brickgen.gui import jobtasks

    jobtasks._ARRAY_CACHE.clear()

    q = JobQueue()
    results = _collect_results(q)

    try:
        payload = {
            'blocks': [BLOCK_A, BLOCK_B],
            'tolerance': 0.15,
            'angular': 0.1,
        }

        # First build -- all cached=False.
        q.submit('build_scene', payload)
        _wait_until(lambda: len(results) == 1, timeout=120.0)
        _, first_result = results[0]
        assert isinstance(first_result, dict)
        assert 'cached' in first_result
        assert first_result['cached'] == [False, False]
        first_points = [blk['points'].copy() for blk in first_result['blocks']]

        # Second build -- all cached=True.
        q.submit('build_scene', payload)
        _wait_until(lambda: len(results) == 2, timeout=120.0)
        _, second_result = results[1]
        assert isinstance(second_result, dict)
        assert second_result['cached'] == [True, True]

        # Points arrays must be identical (cache hit returns exact copy).
        for i in range(2):
            np.testing.assert_array_equal(
                first_points[i], second_result['blocks'][i]['points']
            )

    finally:
        q.close()


# ===========================================================================
# Test 5: test_build_scene_partial_cache_hit
# ===========================================================================


def test_build_scene_partial_cache_hit():
    '''Two blocks: changing one block settings misses only that index.'''
    from brickgen.gui import jobtasks

    jobtasks._ARRAY_CACHE.clear()

    q = JobQueue()
    results = _collect_results(q)

    try:
        # First build with both blocks.
        payload = {
            'blocks': [BLOCK_A, BLOCK_B],
            'tolerance': 0.15,
            'angular': 0.1,
        }
        q.submit('build_scene', payload)
        _wait_until(lambda: len(results) == 1, timeout=120.0)
        _, first_result = results[0]
        assert isinstance(first_result, dict)
        assert first_result['cached'] == [False, False]

        # Second build: change BLOCK_B settings (cells_x=2 instead of 1).
        modified_b = {
            'position_x': 12.0,
            'position_y': 0.0,
            'settings': BlockSettings(
                cells_x=2, cells_y=1, pins_enabled=False
            ).model_dump(),
        }
        payload2 = {
            'blocks': [BLOCK_A, modified_b],
            'tolerance': 0.15,
            'angular': 0.1,
        }
        q.submit('build_scene', payload2)
        _wait_until(lambda: len(results) == 2, timeout=120.0)
        _, second_result = results[1]
        assert isinstance(second_result, dict)
        # BLOCK_A cached=True (unchanged), BLOCK_B cached=False (changed).
        assert second_result['cached'] == [True, False]

    finally:
        q.close()


# ===========================================================================
# Test 6: test_build_scene_tolerance_miss
# ===========================================================================


def test_build_scene_tolerance_miss():
    '''Changing tolerance produces no cache hits.'''
    from brickgen.gui import jobtasks

    jobtasks._ARRAY_CACHE.clear()

    q = JobQueue()
    results = _collect_results(q)

    try:
        payload = {
            'blocks': [BLOCK_A, BLOCK_B],
            'tolerance': 0.15,
            'angular': 0.1,
        }

        # First build.
        q.submit('build_scene', payload)
        _wait_until(lambda: len(results) == 1, timeout=120.0)
        _, first_result = results[0]
        assert isinstance(first_result, dict)
        assert first_result['cached'] == [False, False]

        # Second build with different tolerance.
        payload2 = {
            'blocks': [BLOCK_A, BLOCK_B],
            'tolerance': 0.10,
            'angular': 0.1,
        }
        q.submit('build_scene', payload2)
        _wait_until(lambda: len(results) == 2, timeout=120.0)
        _, second_result = results[1]
        assert isinstance(second_result, dict)
        # Both blocks miss because tolerance changed (key includes tolerance).
        assert second_result['cached'] == [False, False]

    finally:
        q.close()
