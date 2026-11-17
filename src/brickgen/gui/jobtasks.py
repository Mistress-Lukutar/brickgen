'''
File:   jobtasks.py
Brief:  Child-process task handlers for the process-based JobQueue.
Author: Mistress-Lukutar
Date:   2026-11-16
Version: v0.1.0

Each handler is invoked inside a child process; heavy modules (cadquery, pyvista,
OCC) are imported lazily inside the handler so the first job pays the import cost.
The module maintains an LRU cache of built models keyed by caller-supplied keys.
'''

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cadquery as cq

from brickgen.gui.preview import mesh_to_arrays

_HANDLERS: dict[str, Callable[[dict], dict]] = {}


def handler(name: str) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    '''Register a task handler by name.

    Args:
        name: The task name string used in :meth:`~brickgen.gui.jobs.JobQueue.submit`.

    Returns:
        Decorator that registers the function and returns it unchanged.
    '''

    def wrap(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _HANDLERS[name] = fn
        return fn

    return wrap


def run_task(task: str, payload: dict) -> dict:
    '''Dispatch to a registered handler by task name.

    Args:
        task: The task name (must match a registered handler).
        payload: Dict passed to the handler function.

    Returns:
        Handler's return value (a picklable dict).

    Raises:
        ValueError: If no handler is registered for *task*.
    '''
    fn = _HANDLERS.get(task)
    if fn is None:
        raise ValueError(f"Unknown task: {task!r}")
    return fn(payload)


# LRU cache of models built in this process, keyed by caller-supplied cache_key.
# Maximum 3 entries; used to avoid rebuilding the same model for export.
_MODEL_CACHE: OrderedDict[str, cq.Workplane] = OrderedDict()
_MAX_CACHE_SIZE = 3


@handler('ping')
def _handle_ping(payload: dict) -> dict:  # noqa: ARG001
    '''Simple connectivity check.

    Returns:
        ``{'pong': True}``
    '''
    return {'pong': True}


@handler('echo')
def _handle_echo(payload: dict) -> dict:
    '''Return the payload unchanged.

    Args:
        payload: Arbitrary dict to echo back.

    Returns:
        The same *payload* dict.
    '''
    return payload


@handler('boom')
def _handle_boom(payload: dict) -> dict:  # noqa: ARG001
    '''Raise a ValueError for testing error envelopes.

    Raises:
        ValueError: Always, with message 'boom handler'.
    '''
    raise ValueError('boom handler')


@handler('sleep_echo')
def _handle_sleep_echo(payload: dict) -> dict:
    '''Sleep for the specified seconds then return the value.

    Args:
        payload: Dict with keys ``'seconds'`` (float) and ``'value'`` (any).

    Returns:
        ``{'value': <value>}`` where *value* is from the payload.

    Raises:
        KeyError: If 'seconds' or 'value' are missing from payload.
    '''
    seconds = payload['seconds']
    value = payload['value']
    time.sleep(seconds)
    return {'value': value}


@handler('build_preview')
def _handle_build_preview(payload: dict) -> dict:
    '''Build a meshed preview and cache the underlying model.

    The model is cached under *cache_key* for later export reuse within this
    child process.

    Args:
        payload: Dict with keys::

            'settings': BlockSettings-compatible dict (serialised via model_dump).
            'tolerance': float tessellation tolerance in mm.
            'angular': float angular tolerance in rad.
            'cache_key': str unique key for LRU cache storage.

    Returns:
        Dict with keys ``'points'``, ``'faces'``, ``'part'`` (from
        :func:`mesh_to_arrays`) merged with ``{'cache_key': cache_key}``.

    Raises:
        KeyError: If required payload keys are missing.
    '''
    from brickgen.core.builder import build_block
    from brickgen.core.models import BlockSettings
    from brickgen.gui.preview import (
        make_face_classifier,
        solid_to_mesh,
    )

    settings = BlockSettings.model_validate(payload['settings'])
    model = build_block(settings)

    tol = payload['tolerance']
    ang = payload['angular']
    cache_key = payload['cache_key']

    mesh = solid_to_mesh(
        cast(cq.Shape, model.val()), tol, ang, make_face_classifier(settings)
    )
    arrays = mesh_to_arrays(mesh)

    # Cache model under cache_key (LRU eviction).
    if cache_key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(cache_key)
    else:
        if len(_MODEL_CACHE) >= _MAX_CACHE_SIZE:
            _MODEL_CACHE.popitem(last=False)
        _MODEL_CACHE[cache_key] = model

    arrays['cache_key'] = cache_key
    return arrays


@handler('thumbnail')
def _handle_thumbnail(payload: dict) -> dict:
    '''Build a meshed thumbnail without caching the model.

    Args:
        payload: Dict with keys::

            'settings': BlockSettings-compatible dict (serialised via model_dump).

    Returns:
        Dict with keys ``'points'``, ``'faces'``, ``'part'`` from
        :func:`mesh_to_arrays`.

    Raises:
        KeyError: If 'settings' is missing from payload.
    '''
    from brickgen.core.builder import build_block
    from brickgen.core.models import BlockSettings
    from brickgen.gui.preview import (
        make_face_classifier,
        solid_to_mesh,
    )

    settings = BlockSettings.model_validate(payload['settings'])
    model = build_block(settings)

    mesh = solid_to_mesh(
        cast(cq.Shape, model.val()),
        payload.get('tolerance', 0.15),
        payload.get('angular', 0.1),
        make_face_classifier(settings),
    )
    return mesh_to_arrays(mesh)


@handler('export')
def _handle_export(payload: dict) -> dict:
    '''Export a cached model to a file.

    Args:
        payload: Dict with keys::

            'cache_key': str key matching a previously built model.
            'path': str output file path.
            'format': str export format (e.g. 'STEP').
            'tolerance': float tessellation tolerance in mm.
            'angular': float angular tolerance in rad.

    Returns:
        ``{'written': <str>}`` with the resolved output path.

    Raises:
        ValueError: If *cache_key* is not found in the model cache.
    '''
    from brickgen.services.export_service import export_model as _export_model

    cache_key = payload['cache_key']
    if cache_key not in _MODEL_CACHE:
        raise ValueError(f"model {cache_key!r} not cached")
    model = _MODEL_CACHE[cache_key]

    written = _export_model(
        model,
        Path(payload['path']),
        payload['format'],
        tolerance=payload.get('tolerance', 0.1),
        angular_tolerance=payload.get('angular', 0.1),
    )
    return {'written': str(written)}
