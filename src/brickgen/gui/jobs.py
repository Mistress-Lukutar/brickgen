'''
File:   jobs.py
Brief:  Serialized background job queue for CPU-heavy GUI work.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.1.0
'''

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from queue import Queue

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


# Module-level sentinel returned when a job's ``skip_if`` predicate fires.
SKIPPED: object = object()


class JobError:
    '''Wraps an exception raised inside a background job.

    Attributes:
        exc: The original exception captured on the worker thread.
    '''

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class JobQueue(QObject):
    '''Serialized background queue for CPU-heavy jobs.

    One daemon worker thread executes submitted callables in FIFO order and
    reports results back on the thread that owns this QObject via the
    ``job_done`` signal (PySide6 auto-queues cross-thread signal emissions).
    Keeping all heavy work on this single thread guarantees the OCC kernel is
    never driven from two threads concurrently.

    Jobs may carry ``skip_if`` -- a zero-argument callable evaluated on the
    worker thread immediately before execution; when it returns True the job
    is skipped (its result is ``SKIPPED`` and ``fn`` is not called). This is
    the latest-wins mechanism: the caller closes over a generation counter
    and skips anything superseded while sitting in the queue.

    Attributes:
        job_done: Signal emitted after every job completes. Payload is
            ``(job_id, result)`` where *result* is the callable's return value,
            ``SKIPPED``, or a :class:`JobError`.
    '''

    job_done = Signal(int, object)  # job id, result (value | SKIPPED | JobError)

    def __init__(self, parent: QObject | None = None) -> None:
        '''Initialise the queue and start the worker thread.

        The daemon worker thread is created immediately on construction so that
        calls to :meth:`submit` never have to synchronise with a lazy-start
        window.  The thread blocks on :class:`queue.Queue` until the first job
        arrives.

        Args:
            parent: Optional parent QObject for ownership / lifetime management.
        '''
        super().__init__(parent)
        self._q: Queue[tuple[int, Callable[[], object], Callable[[], bool] | None]] = (
            Queue()
        )
        self._next_id: int = 1
        self._lock = threading.Lock()
        self._running_count: int = 0  # jobs currently executing + queued
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def submit(
        self,
        fn: Callable[[], object],
        skip_if: Callable[[], bool] | None = None,
    ) -> int:
        '''Submit a callable for asynchronous execution.

        The callable is placed at the tail of the FIFO queue and will be
        executed by the worker thread in submission order.  Each call returns
        a unique integer id that can be used to correlate results with requests.

        Args:
            fn: Zero-argument callable that performs the heavy work.
                Its return value is delivered as the job result (or wrapped in
                :class:`JobError` if it raises).
            skip_if: Optional zero-argument predicate evaluated on the worker
                thread before *fn*.  When ``skip_if()`` returns ``True``, *fn*
                is not called and the job result is :data:`SKIPPED`.

        Returns:
            A monotonically increasing integer id (starting at 1) assigned in
            submission order.
        '''
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
            self._running_count += 1
        logger.debug('JobQueue submit: id=%d', job_id)
        self._q.put((job_id, fn, skip_if))
        return job_id

    def busy_count(self) -> int:
        '''Return the number of jobs not yet finished (queued + running).

        Returns:
            Integer count of pending and in-flight jobs.
        '''
        with self._lock:
            return self._running_count

    # -- internal -----------------------------------------------------------

    def _run_loop(self) -> None:
        '''Worker thread entry point.

        Blocks on the internal :class:`queue.Queue`, pulling one job item at a
        time.  For each item it evaluates ``skip_if`` (if provided), calls
        ``fn``, and emits ``job_done(id, result)`` via the Qt signal which is
        automatically queued to the GUI thread.

        The loop survives any exception raised by a job body -- the exception
        is wrapped in :class:`JobError`, emitted, and then the worker continues
        to the next queued item.
        '''
        while True:
            job_id, fn, skip_if = self._q.get()  # blocks until available

            if skip_if is not None and skip_if():
                result = SKIPPED
            else:
                try:
                    result = fn()
                except BaseException as exc:
                    result = JobError(exc)

            # Decrease the busy count before the signal so slots querying
            # busy_count() from job_done never see the finished job.
            with self._lock:
                self._running_count -= 1

            self.job_done.emit(job_id, result)

            logger.debug('JobQueue done: id=%d', job_id)
