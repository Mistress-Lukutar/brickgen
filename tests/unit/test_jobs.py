"""
File:   test_jobs.py
Brief:  Unit tests for JobQueue (serialized background job queue) and GUI threading.
Author: Mistress-Lukutar
Date:   2026-11-16
"""

import os
import threading
import time

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
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)


# --------------------------------------------------------------------------- helpers


def _collect_results(q: JobQueue):
    """Return a list that captures ``(job_id, result)`` from *q.job_done*."""
    results: list[tuple[int, object]] = []
    q.job_done.connect(lambda job_id, result: results.append((job_id, result)))
    return results


# ===========================================================================
# 1. test_job_result_delivered
# ===========================================================================


def test_job_result_delivered():
    """A submitted callable's return value arrives via the job_done signal."""
    q = JobQueue()
    results = _collect_results(q)

    def my_fn():
        return 42

    job_id = q.submit(my_fn)

    _wait_until(lambda: len(results) == 1, timeout=5.0)

    assert job_id == 1, 'first job on a fresh queue must have id 1'
    rid, result = results[0]
    assert rid == job_id
    assert result == 42


# ===========================================================================
# 2. test_fifo_order_and_ids
# ===========================================================================


def test_fifo_order_and_ids():
    """Three jobs execute in submission order (FIFO) with monotonically
    increasing ids on a fresh queue."""
    q = JobQueue()
    results: list[str] = []
    order = _collect_results(q)

    def job_a():
        time.sleep(0.05)
        results.append('a')

    def job_b():
        time.sleep(0.02)
        results.append('b')

    def job_c():
        time.sleep(0.01)
        results.append('c')

    id_a = q.submit(job_a)
    id_b = q.submit(job_b)
    id_c = q.submit(job_c)

    _wait_until(lambda: len(order) == 3, timeout=5.0)

    assert id_a == 1 and id_b == 2 and id_c == 3
    # The worker thread serialises execution; all three append in submission order.
    assert results == ['a', 'b', 'c']


# ===========================================================================
# 3. test_skip_if_prevents_execution
# ===========================================================================


def test_skip_if_prevents_execution():
    """A job whose skip_if returns True is skipped: fn not called, SKIPPED result."""
    q = JobQueue()
    results: list[str] = []
    order = _collect_results(q)

    q.submit(lambda: results.append('ran'), skip_if=lambda: True)

    _wait_until(lambda: len(order) == 1, timeout=5.0)

    assert order[0][1] is SKIPPED
    assert results == []


# ===========================================================================
# 4. test_job_error_wraps_exception
# ===========================================================================


def test_job_error_wraps_exception():
    """An exception raised inside a job is wrapped in JobError on the GUI thread."""
    q = JobQueue()
    order = _collect_results(q)

    def boom():
        raise ValueError('boom')

    q.submit(boom)

    _wait_until(lambda: len(order) == 1, timeout=5.0)

    _, result = order[0]
    assert isinstance(result, JobError)
    assert result.exc.args == ('boom',)


# ===========================================================================
# 5. test_queue_survives_exception_and_processes_next
# ===========================================================================


def test_queue_survives_exception_and_processes_next():
    """The worker thread survives a job exception and processes the next queued job."""
    q = JobQueue()
    results: list[int] = []
    _collect_results(q)  # connect result-collection slot (side effect)

    def fail():
        raise RuntimeError('oops')

    q.submit(fail)
    q.submit(lambda: results.append(99))

    _wait_until(lambda: len(results) == 1, timeout=5.0)

    assert results == [99]


# ===========================================================================
# 6. test_busy_count_lifecycle
# ===========================================================================


def test_busy_count_lifecycle():
    """busy_count() tracks queued + running jobs correctly (increments on submit,
    decrements after completion)."""
    q = JobQueue()
    results = _collect_results(q)
    block_event = threading.Event()

    def blocking_job():
        block_event.wait()

    q.submit(blocking_job)

    # Wait until the job is queued (busy_count > 0 means it has been submitted
    # and not yet finished).
    _wait_until(lambda: q.busy_count() == 1, timeout=5.0)
    assert q.busy_count() == 1

    # Release the blocking job.
    block_event.set()

    # Wait until the job completes.
    _wait_until(lambda: q.busy_count() == 0, timeout=5.0)
    _wait_until(lambda: len(results) == 1, timeout=5.0)


# ===========================================================================
# 7. test_latest_wins_skips_stale_generation
# ===========================================================================


def test_latest_wins_skips_stale_generation():
    """QUEUED stale jobs are skipped via skip_if; only the latest generation
    executes.

    The worker thread evaluates skip_if at execution start.  Because both jobs
    are submitted before the worker picks up job 1, the sequence counter is
    already 2 when job 1's skip_if runs -- so job 1 is skipped and job 2
    (which matches seq==2) executes normally.
    """
    q = JobQueue()
    results: list[tuple[int, object]] = _collect_results(q)

    gen_seq_ref = [0]

    def make_build(seq):
        def build():
            time.sleep(0.05)
            return seq

        return build

    # Submit job 1 (seq=1) -- skip_if checks if seq matches current gen_seq.
    q.submit(make_build(1), skip_if=lambda: 1 != gen_seq_ref[0])

    # Bump the sequence counter before the worker processes job 1's skip_if.
    time.sleep(0.05)
    gen_seq_ref[0] = 2

    # Submit job 2 (seq=2).
    q.submit(make_build(2), skip_if=lambda: 2 != gen_seq_ref[0])

    _wait_until(lambda: len(results) == 2, timeout=10.0)

    # Exactly one skipped and one result of 2.
    statuses = [r for _, r in results]
    assert statuses.count(SKIPPED) == 1
    assert statuses.count(2) == 1


# ===========================================================================
# 8. test_in_flight_job_completes_not_interruptible
# ===========================================================================


def test_in_flight_job_completes_not_interruptible():
    """An in-flight job is NOT interrupted by a later skip_if; it runs to
    completion. Only QUEUED stale jobs skip."""
    q = JobQueue()
    results: list[tuple[int, object]] = _collect_results(q)

    # Start at 1 so the worker's skip_if (1 != gen_seq) returns False immediately
    # when it picks up job 1.  Then fn_started confirms the worker is past skip_if
    # and executing fn() -- i.e. job 1 is truly in-flight.
    gen_seq_ref = [1]
    block_event = threading.Event()
    fn_started = threading.Event()  # set by fn() after skip_if is evaluated

    def make_blocking(seq):
        """Factory that returns a callable which blocks on block_event."""

        def build():
            fn_started.set()  # confirms skip_if ran and returned False
            time.sleep(0.05)
            block_event.wait()
            return seq

        return build

    q.submit(make_blocking(1), skip_if=lambda: 1 != gen_seq_ref[0])

    # Wait until the worker has evaluated skip_if for job 1 and started fn().
    _wait_until(lambda: fn_started.is_set(), timeout=5.0)

    # Bump sequence while job 1 is in-flight.
    gen_seq_ref[0] = 2

    q.submit(make_blocking(2), skip_if=lambda: 2 != gen_seq_ref[0])

    # Release both jobs.
    block_event.set()

    _wait_until(lambda: len(results) == 2, timeout=10.0)

    results_vals = [r for _, r in results]
    # Both jobs complete because job 1 was already running when seq bumped --
    # it cannot be interrupted; only queued jobs check skip_if at start.
    assert results_vals.count(1) == 1
    assert results_vals.count(2) == 1


# ===========================================================================
# 9. test_app_async_smoke
# ===========================================================================


def test_app_async_smoke():
    """Smoke test: instantiate MainWindow, wait for initial auto-generation to
    complete on the background queue, then change a parameter and verify it
    propagates.

    Requires offscreen Qt platform (set via QT_QPA_PLATFORM env var).
    This test is slower (~seconds) because the library panel builds thumbnails
    through the same job queue before _model is set.
    """
    _ensure_app()  # QApplication required for queued signal delivery

    from brickgen.gui.app import MainWindow

    w = MainWindow()
    w.setWindowTitle('test_app_async_smoke')

    # Wait for initial generation to complete (sets self._model in _on_job_done).
    _wait_until(lambda: w._model is not None, timeout=120.0)
    assert w._model is not None

    # Change a parameter and wait for the settings to propagate.
    w._cells_x.setValue(3)
    _wait_until(lambda: w._settings.cells_x == 3, timeout=5.0)
    assert w._settings.cells_x == 3
