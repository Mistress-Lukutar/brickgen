'''
File:   jobs.py
Brief:  Process-based serialized background job queue for CPU-heavy GUI work.
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.21.0
'''

from __future__ import annotations

import logging
import multiprocessing
import threading
import traceback
from concurrent.futures import Future, ProcessPoolExecutor

from PySide6.QtCore import QObject, Signal

from brickgen.gui import jobtasks

logger = logging.getLogger(__name__)

# Sentinel emitted for a buffered job that was replaced before dispatch.
SKIPPED: object = object()


class JobError:
    '''Wraps a child-process failure: attribute ``message: str``.

    A compatibility shim exposes ``exc`` (property returning ``RuntimeError``)
    so existing call sites using ``str(result.exc)`` continue to work.

    Attributes:
        message: Full traceback text from the child process.
    '''

    def __init__(self, message: str) -> None:
        '''Store the child-process traceback text.

        Args:
            message: Full traceback text from the child process.
        '''
        self.message = message

    @property
    def exc(self) -> RuntimeError:
        '''Compatibility shim: returns a RuntimeError wrapping *message*.'''
        return RuntimeError(self.message)


def _child_init() -> None:
    '''Executor initializer: import the task handlers in the worker process.'''
    import brickgen.gui.jobtasks  # noqa: F401 - pre-warm the handler registry


class JobQueue(QObject):
    '''Serialized background queue running tasks in a worker process.

    Jobs are ``(task, payload)`` pairs executed by handlers registered in
    :mod:`brickgen.gui.jobtasks` inside a single worker process
    (:class:`concurrent.futures.ProcessPoolExecutor`, spawn context). The
    worker process has its own interpreter, so Python-phase work (mesh
    collection loops and friends) can never hold the GUI thread's GIL —
    the reason plain threads are not enough for OCC + numpy work.

    Results arrive on the GUI thread via ``job_done(id, result)`` where
    *result* is the handler's return dict, a :class:`JobError`, or
    :data:`SKIPPED` for a buffered job replaced before dispatch.

    Key-scoped latest-wins: ``submit(..., key='generation')`` buffers while a
    job with the same key is in flight; a newer submission replaces the
    buffer (the replaced id is emitted with SKIPPED immediately). Buffered
    jobs are dispatched when the in-flight one finishes. ``key=None`` jobs
    are always sent (FIFO, never replaced).

    Built models stay cached inside the worker (:mod:`brickgen.gui.jobtasks`
    LRU keyed by ``cache_key``); export tasks reference them by key, which
    also sidesteps that ``cq.Workplane`` cannot be pickled.
    '''

    job_done = Signal(int, object)  # job id, result (dict | SKIPPED | JobError)

    def __init__(self, parent: QObject | None = None) -> None:
        '''Create the executor and the bookkeeping state.

        Args:
            parent: Optional parent QObject for ownership / lifetime management.
        '''
        super().__init__(parent)
        self._next_id: int = 1
        self._lock = threading.Lock()
        self._inflight: dict[str, int] = {}  # key -> dispatched job id
        self._pending: dict[str, tuple[int, str, dict]] = {}  # key -> buffered job
        self._busy: int = 0
        self._closed: bool = False
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_child_init,
        )
        logger.debug('JobQueue started')

    def submit(self, task: str, payload: dict, *, key: str | None = None) -> int:
        '''Queue a named task with a payload dict for background execution.

        Args:
            task: Handler name registered in :mod:`brickgen.gui.jobtasks`
                (e.g. ``'ping'``, ``'build_preview'``, ``'export'``).
            payload: Picklable dict passed verbatim to the handler.
            key: Optional latest-wins group. When a job with this key is
                in flight, the new job buffers; a buffered job of the same
                key is replaced (its id is emitted with :data:`SKIPPED`).

        Returns:
            A monotonically increasing integer id (starting at 1), or -1
            when the queue is already closed.
        '''
        with self._lock:
            if self._closed:
                logger.debug('JobQueue closed; dropping task %r', task)
                return -1
            job_id = self._next_id
            self._next_id += 1
            if key is not None and key in self._inflight:
                old = self._pending.get(key)
                if old is not None:
                    logger.debug(
                        'JobQueue submit: id=%d replaced (key=%s)', old[0], key
                    )
                    self.job_done.emit(old[0], SKIPPED)
                self._pending[key] = (job_id, task, payload)
                return job_id
            if key is not None:
                self._inflight[key] = job_id
            self._busy += 1
        self._dispatch(job_id, task, payload, key)
        logger.debug('JobQueue submit: id=%d (key=%s)', job_id, key)
        return job_id

    def busy_count(self) -> int:
        '''Return the number of dispatched jobs not yet finished.

        Buffered-not-yet-dispatched jobs do not count (a replaced buffer was
        already reported as SKIPPED at replace time).

        Returns:
            Integer count of in-flight jobs.
        '''
        with self._lock:
            return self._busy

    def close(self, timeout: float = 5.0) -> None:
        '''Shut the worker process down. Idempotent; second call is a no-op.

        Buffered jobs are emitted as SKIPPED; queued futures are cancelled.
        The executor shutdown does not wait for a running job, so the call
        returns promptly.

        Args:
            timeout: Unused compatibility parameter (kept for the previous
                signature); shutdown is non-blocking.
        '''
        with self._lock:
            if self._closed:
                logger.debug('JobQueue close: already closed')
                return
            self._closed = True
            pending_ids = [entry[0] for entry in self._pending.values()]
            self._pending.clear()
        for job_id in pending_ids:
            self.job_done.emit(job_id, SKIPPED)
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.debug('JobQueue closed')

    # ------------------------------------------------------------------ internal

    def _dispatch(self, job_id: int, task: str, payload: dict, key: str | None) -> None:
        '''Submit a job to the executor and arm its completion callback.'''
        try:
            future = self._executor.submit(jobtasks.run_task, task, payload)
        except RuntimeError as exc:  # executor shutdown mid-flight
            logger.warning('JobQueue dispatch failed: %s', exc)
            with self._lock:
                self._busy -= 1
            self.job_done.emit(job_id, JobError(f'Executor unavailable: {exc}'))
            return

        def _on_job_done(fut: Future) -> None:
            self._on_done(job_id, key, fut)

        future.add_done_callback(_on_job_done)

    def _on_done(self, job_id: int, key: str | None, future: Future) -> None:
        '''Completion callback (executor management thread).

        Wraps failures into :class:`JobError`, flushes a same-key buffered
        job (dispatching it before the signal so bookkeeping stays ordered),
        and emits the result.

        Args:
            job_id: The finished job's id.
            key: Its latest-wins key (or None).
            future: The completed future.
        '''
        try:
            result: object = future.result()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            result = JobError(''.join(traceback.format_exception(exc)).strip())
        to_dispatch: tuple[int, str, dict] | None = None
        with self._lock:
            self._busy -= 1
            if key is not None:
                self._inflight.pop(key, None)
                pending = self._pending.pop(key, None)
                if pending is not None:
                    self._busy += 1
                    self._inflight[key] = pending[0]
                    to_dispatch = pending
        if to_dispatch is not None:
            logger.debug('JobQueue flush: id=%d (key=%s)', to_dispatch[0], key)
            self._dispatch(to_dispatch[0], to_dispatch[1], to_dispatch[2], key)
        self.job_done.emit(job_id, result)
        logger.debug('JobQueue done: id=%d', job_id)
