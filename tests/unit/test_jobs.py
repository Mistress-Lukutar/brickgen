"""
File:   test_jobs.py
Brief:  Unit tests for JobQueue (process-based background job queue).
Author: Mistress-Lukutar
Date:   2026-11-20
"""

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402 - platform must be set first


def _ensure_app() -> QApplication:
    """Return the singleton QApplication, creating it if necessary."""
    app = QApplication.instance()
    if app is None:  # pragma: no cover
        app = QApplication([])
    return app


def _wait_until(predicate, timeout: float = 10.0) -> None:
    """Block until *predicate* returns True or *timeout* seconds elapse.

    Drives ``QApplication.processEvents()`` so that queued signals are delivered.
    """
    app = _ensure_app()
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate(), 'condition not met within timeout'


import pytest  # noqa: E402 - platform must be set first

from brickgen.gui.jobs import SKIPPED, JobError, JobQueue  # noqa: E402

# vtk 9.6 numpy_support sets .shape on arrays, deprecated in numpy 2.5; the
# warning fires from the preview pipeline exercised by the app smoke test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy Array:DeprecationWarning"
)


# --------------------------------------------------------------------------- helpers


def _collect_results(q: JobQueue):
    """Return a list that captures ``(job_id, result)`` from *q.job_done*."""
    results: list[tuple[int, object]] = []
    q.job_done.connect(lambda job_id, result: results.append((job_id, result)))
    return results


# ===========================================================================
# Test 1: test_ping_roundtrip
# ===========================================================================


def test_ping_roundtrip():
    """A 'ping' task returns {'pong': True}; first id == 1."""
    q = JobQueue()
    results = _collect_results(q)

    job_id = q.submit('ping', {})

    _wait_until(lambda: len(results) == 1, timeout=5.0)

    assert job_id == 1, 'first job on a fresh queue must have id 1'
    rid, result = results[0]
    assert rid == job_id
    assert result == {'pong': True}

    q.close()


# ===========================================================================
# Test 2: test_echo_fifo_order
# ===========================================================================


def test_echo_fifo_order():
    """Three 'echo' payloads submitted in order deliver in order; ids 1,2,3."""
    q = JobQueue()
    results = _collect_results(q)

    q.submit('echo', {'n': 1})
    q.submit('echo', {'n': 2})
    q.submit('echo', {'n': 3})

    _wait_until(lambda: len(results) == 3, timeout=5.0)

    assert len(results) == 3
    assert results[0][0] == 1 and results[0][1] == {'n': 1}
    assert results[1][0] == 2 and results[1][1] == {'n': 2}
    assert results[2][0] == 3 and results[2][1] == {'n': 3}

    q.close()


# ===========================================================================
# Test 3: test_unknown_task_is_joberror
# ===========================================================================


def test_unknown_task_is_joberror():
    """Submit 'no_such_task' -> JobError, message mentions the task name."""
    q = JobQueue()
    results = _collect_results(q)

    job_id = q.submit('no_such_task', {'x': 42})

    _wait_until(lambda: len(results) == 1, timeout=5.0)

    rid, result = results[0]
    assert rid == job_id
    assert isinstance(result, JobError)
    assert 'no_such_task' in result.message

    q.close()


# ===========================================================================
# Test 4: test_boom_error_envelope
# ===========================================================================


def test_boom_error_envelope():
    """'boom' -> JobError whose message contains 'ValueError' and 'boom handler'."""
    q = JobQueue()
    results = _collect_results(q)

    job_id = q.submit('boom', {})

    _wait_until(lambda: len(results) == 1, timeout=5.0)

    rid, result = results[0]
    assert rid == job_id
    assert isinstance(result, JobError)
    assert 'ValueError' in result.message
    assert 'boom handler' in result.message

    q.close()


# ===========================================================================
# Test 5: test_keyed_buffer_latest_wins
# ===========================================================================


def test_keyed_buffer_latest_wins():
    """Keyed latest-wins buffering with SKIPPED emission.

    Submit ('sleep_echo', {'seconds': 1.0, 'value': 'first'}, key='gen'); wait
    until busy; then submit ('echo', {'n': 2}, key='gen') (buffered) and then
    ('echo', {'n': 3}, key='gen') (replaces -> id of job 2 is SKIPPED).

    Collect until 3 deliveries: exactly one SKIPPED, delivered echo results
    are 'first' and 3 (never 2), total deliveries == 3.
    """
    q = JobQueue()
    results = _collect_results(q)

    # Submit first job (sleep_echo with key='gen')
    q.submit('sleep_echo', {'seconds': 1.0, 'value': 'first'}, key='gen')

    # Wait until busy_count == 1 so the child has truly started processing.
    _wait_until(lambda: q.busy_count() == 1, timeout=5.0)
    assert q.busy_count() == 1

    # Submit buffered job (echo n=2, same key) — replaces nothing, just buffers.
    q.submit('echo', {'n': 2}, key='gen')

    # Submit another buffered job (echo n=3, same key) — replaces echo n=2 -> SKIPPED.
    q.submit('echo', {'n': 3}, key='gen')

    # Collect until we have 3 deliveries (1 result + 1 SKIPPED + 1 more result).
    _wait_until(lambda: len(results) == 3, timeout=10.0)

    statuses = [r for _, r in results]
    values = [r for r in statuses if r is not SKIPPED and isinstance(r, dict)]

    # Exactly one SKIPPED
    assert statuses.count(SKIPPED) == 1

    # The echo results should be 'first' (from sleep_echo) and {'n': 3} (from
    # the winning buffered job). Never 2.
    value_list = [v.get('value') if isinstance(v, dict) else v for v in values]
    assert 'first' in value_list, f"Expected 'first' in results: {values}"
    assert {'n': 3} in values or (
        isinstance(values[-1], dict) and values[-1].get('n') == 3
    ), f"Expected echo n=3 in results: {values}"

    # Total deliveries == 3
    assert len(results) == 3

    q.close()


# ===========================================================================
# Test 6: test_keyless_jobs_all_run
# ===========================================================================


def test_keyless_jobs_all_run():
    """Three key=None echo jobs all execute (no buffering)."""
    q = JobQueue()
    results = _collect_results(q)

    q.submit('echo', {'a': 1})
    q.submit('echo', {'b': 2})
    q.submit('echo', {'c': 3})

    _wait_until(lambda: len(results) == 3, timeout=5.0)

    assert len(results) == 3
    result_by_id = {r[0]: r[1] for r in results}
    assert {'a': 1} in result_by_id.values()
    assert {'b': 2} in result_by_id.values()
    assert {'c': 3} in result_by_id.values()

    q.close()


# ===========================================================================
# Test 7: test_busy_count_lifecycle
# ===========================================================================


def test_busy_count_lifecycle():
    """sleep_echo(0.4) -> busy_count()==1 while running (poll until 1, wait until 0)."""
    q = JobQueue()
    results = _collect_results(q)

    q.submit('sleep_echo', {'seconds': 0.4, 'value': 'done'})

    # Poll until busy_count reaches 1 (job dispatched to child).
    _wait_until(lambda: q.busy_count() == 1, timeout=5.0)
    assert q.busy_count() == 1

    # Wait until the job completes.
    _wait_until(lambda: q.busy_count() == 0, timeout=5.0)
    _wait_until(lambda: len(results) == 1, timeout=5.0)

    assert len(results) == 1
    assert results[0][1] == {'value': 'done'}

    q.close()


# ===========================================================================
# Test 8: test_close_shuts_child_down
# ===========================================================================


def test_close_shuts_child_down():
    """After all jobs drain, close() twice (idempotent) returns promptly (< timeout)."""
    q = JobQueue()
    results = _collect_results(q)

    # Drain the queue first.
    q.submit('echo', {'x': 1})
    _wait_until(lambda: len(results) == 1, timeout=5.0)

    start = time.monotonic()
    q.close(timeout=5.0)
    elapsed1 = time.monotonic() - start
    assert elapsed1 < 5.0, f'first close took {elapsed1:.1f}s (expected < 5s)'

    start = time.monotonic()
    q.close(timeout=5.0)
    elapsed2 = time.monotonic() - start
    assert elapsed2 < 5.0, f'second close took {elapsed2:.1f}s (expected < 5s)'

    # Second close should be essentially instant (no-op).
    assert elapsed2 < 0.5, 'idempotent close should be near-instant'


# ===========================================================================
# Test 9: test_build_preview_handler_arrays
# ===========================================================================


def test_build_preview_handler_arrays():
    """Build preview and export roundtrip: validate mesh arrays and model cache."""
    from brickgen.core.models import BlockSettings
    from brickgen.gui.preview import mesh_from_arrays

    q = JobQueue()
    results = _collect_results(q)

    default_settings = BlockSettings()
    settings_dump = default_settings.model_dump()

    # Submit build_preview job.
    q.submit(
        'build_preview',
        {
            'settings': settings_dump,
            'tolerance': 0.15,
            'angular': 0.1,
            'cache_key': 'g1',
        },
        key='preview',
    )

    _wait_until(lambda: len(results) == 1, timeout=60.0)

    assert len(results) == 1
    rid, result = results[0]
    assert isinstance(result, dict), f'Expected dict result, got {type(result)}'
    assert 'points' in result
    assert 'faces' in result
    assert 'part' in result
    assert 'g1' in result.get('cache_key', '')

    points = result['points']
    faces = result['faces']
    part = result['part']

    # Validate array shapes.
    import numpy as np

    assert isinstance(points, np.ndarray) and points.dtype == np.float64
    assert points.shape[1] == 3
    assert len(points) > 0, 'Mesh should have non-empty points'

    assert isinstance(faces, np.ndarray) and faces.dtype == np.int64
    assert len(faces) >= 3  # at least one triangle header

    assert isinstance(part, np.ndarray) and part.dtype == np.int8

    # Reconstruct mesh and validate.
    mesh = mesh_from_arrays(result)
    assert mesh.n_cells > 0
    assert 'part' in mesh.cell_data

    # Now export using the cached model.
    with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as tmp:
        tmp_path = tmp.name

    q.submit(
        'export',
        {
            'cache_key': 'g1',
            'path': tmp_path,
            'format': 'STEP',
            'tolerance': 0.1,
            'angular': 0.1,
        },
    )

    _wait_until(lambda: len(results) == 2, timeout=120.0)

    assert len(results) == 2
    _, export_result = results[1]
    assert isinstance(export_result, dict)
    assert 'written' in export_result
    written_path = export_result['written']
    assert Path(written_path).exists()
    assert Path(written_path).stat().st_size > 0

    q.close()


# ===========================================================================
# Test 10: test_app_generation_and_export_smoke
# ===========================================================================


def test_app_generation_and_export_smoke():
    """MainWindow generates the scene via the process queue and exports it.

    Requires offscreen Qt platform. Timeout is generous because the worker
    process pays the OCC/pyvista import cost on the very first job.
    """
    _ensure_app()

    from brickgen.gui.app import MainWindow

    w = MainWindow()

    # Initial auto-generation completes on the queue (applied context set).
    # NOTE: startup fires _on_generate twice (explicit call + debounced timer
    # from form sync), so the applied key is 'gen2' — latest-wins in action.
    _wait_until(lambda: w._applied_context is not None, timeout=180.0)
    assert len(w._preview._block_actors) >= 1
    assert w._preview._mesh is not None and w._preview._mesh.n_cells > 0

    # Adding a block schedules the next generation (latest-wins key).
    w._on_add_block()
    _wait_until(lambda: len(w._preview._block_actors) == 2, timeout=180.0)

    # Export the applied scene exactly like _on_export does after its dialog.
    out = os.path.join(tempfile.gettempdir(), 'brickgen_app_export_smoke.step')
    context = w._applied_context
    payload = {
        'blocks': context['payload']['blocks'],
        'path': out,
        'format': 'STEP',
        'tolerance': 0.1,
        'angular': 0.1,
    }
    job_id = w._jobs.submit('export_scene', payload)
    w._export_meta[job_id] = out

    _wait_until(lambda: job_id not in w._export_meta, timeout=120.0)
    assert Path(out).exists() and Path(out).stat().st_size > 0

    w._mark_clean()
    w.close()
    w._jobs.close()


# ===========================================================================
# Compatibility: JobError.exc shim
# ===========================================================================


def test_joberror_exc_shim():
    """JobError.exc returns a RuntimeError wrapping the message (compatibility shim)."""
    err = JobError('test error')
    assert err.message == 'test error'
    exc = err.exc
    assert isinstance(exc, RuntimeError)
    assert str(exc) == 'test error'
    # Existing code path: str(result.exc) should work.
    assert str(err.exc) == 'test error'


# ===========================================================================
# Compatibility: SKIPPED import
# ===========================================================================


def test_skipped_is_object():
    """SKIPPED is a module-level sentinel object."""
    from brickgen.gui.jobs import SKIPPED as imported_skipped

    assert imported_skipped is SKIPPED
