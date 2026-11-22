'''
File:   jobtasks.py
Brief:  Child-process task handlers for the process-based JobQueue.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.16.0

Each handler is invoked inside a child process; heavy modules (cadquery, pyvista,
OCC) are imported lazily inside the handler so the first job pays the import cost.
The module maintains an LRU cache of built models keyed by caller-supplied keys.
'''

from __future__ import annotations

import hashlib
import json
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


def _scene_fingerprint(settings_dump: dict, tolerance: float, angular: float) -> str:
    '''Stable per-block cache key: canonical settings JSON + tessellation tolerances.'''
    canonical = json.dumps(settings_dump, sort_keys=True, separators=(',', ':'))
    raw = f'{canonical}|{tolerance!r}|{angular!r}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


# LRU of finished per-block mesh arrays (picklable dicts), keyed by
# _scene_fingerprint. A hit lets build_scene skip the OCC build and the
# BRepMesh tessellation for blocks whose settings and tolerances did not
# change (the key includes both tolerances, so a precision change misses).
_ARRAY_CACHE: OrderedDict[str, dict] = OrderedDict()
_ARRAY_CACHE_MAX = 64


def _array_cache_get(key: str) -> dict | None:
    '''Return the cached arrays dict for *key*, or None on miss.'''
    if key in _ARRAY_CACHE:
        _ARRAY_CACHE.move_to_end(key)
        return _ARRAY_CACHE[key]
    return None


def _array_cache_put(key: str, arrays: dict) -> None:
    '''Store *arrays* under *key*, evicting oldest entries beyond _ARRAY_CACHE_MAX.'''
    if key in _ARRAY_CACHE:
        _ARRAY_CACHE.move_to_end(key)
    else:
        while len(_ARRAY_CACHE) >= _ARRAY_CACHE_MAX:
            _ARRAY_CACHE.popitem(last=False)
        _ARRAY_CACHE[key] = arrays


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


@handler('build_scene')
def _handle_build_scene(payload: dict) -> dict:
    '''Build meshed previews for every block of the scene.

    Each block is built from its own BlockSettings and returned as picklable
    arrays; points are NOT translated here -- the UI places meshes by setting
    actor positions from the block payload.

    Args:
        payload: Dict with keys::

            'blocks': list of block dicts, each::

                'settings': BlockSettings-compatible dict (model_dump).
                'position_x': float scene offset of the block center, mm.
                'position_y': float scene offset of the block center, mm.

            'tolerance': float tessellation tolerance in mm (optional, 0.15).
            'angular': float angular tolerance in rad (optional, 0.1).

    Returns:
        Dict with keys::

            'blocks': list of mesh_to_arrays results aligned with payload block order;
                each carries ``points``/``faces``/``part``.
            'cached': list[bool] parallel to *blocks*; True when the per-block
                arrays were served from the array cache (no OCC rebuild).
    '''
    from brickgen.core.builder import build_block
    from brickgen.core.models import BlockSettings
    from brickgen.gui.preview import (
        make_face_classifier,
        mesh_to_arrays,
        solid_to_mesh,
    )

    tolerance = payload.get('tolerance', 0.15)
    angular = payload.get('angular', 0.1)
    blocks_data: list[dict] = []
    cached_flags: list[bool] = []
    for entry in payload['blocks']:
        key = _scene_fingerprint(entry['settings'], tolerance, angular)
        cached = _array_cache_get(key)
        if cached is not None:
            blocks_data.append(dict(cached))
            cached_flags.append(True)
        else:
            settings = BlockSettings.model_validate(entry['settings'])
            model = build_block(settings)
            shape = cast(cq.Shape, model.val())
            classifier = make_face_classifier(settings)
            mesh = solid_to_mesh(shape, tolerance, angular, classifier)
            arrays = mesh_to_arrays(mesh)
            _array_cache_put(key, arrays)
            blocks_data.append(arrays)
            cached_flags.append(False)
    return {'blocks': blocks_data, 'cached': cached_flags}


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


@handler('export_print')
def _handle_export_print(payload: dict) -> dict:
    '''Export the print placement: one compound file per bed.

    Args:
        payload: Dict with keys::

            'beds': list of dicts ``{'index': int, 'path': str,
                    'pieces': [entry dicts]}`` where each entry matches the
                    shape produced by ``print_export.collect_bed_entries``.
            'format': str export format (e.g. 'STL').
            'tolerance': float tessellation tolerance in mm (optional, 0.1).
            'angular': float angular tolerance in rad (optional, 0.1).

    Returns:
        ``{'written': [str, ...]}`` with the resolved output paths, one
        per bed.

    Raises:
        ValueError: If 'beds' is empty.
    '''
    from brickgen.services.export_service import export_model as _export_model
    from brickgen.services.print_export import bed_compound as _bed_compound

    tolerance = payload.get('tolerance', 0.1)
    angular = payload.get('angular', 0.1)
    beds_list = payload['beds']

    if not beds_list:
        raise ValueError('no beds with pieces')

    written: list[str] = []
    for bed in beds_list:
        compound = _bed_compound(bed['pieces'])
        path = _export_model(
            compound,
            Path(bed['path']),
            payload['format'],
            tolerance=tolerance,
            angular_tolerance=angular,
        )
        written.append(str(path))
    return {'written': written}
