'''
File:   test_scene_mask.py
Brief:  Unit tests for the per-block rebuild mask of the GUI scene engine.
Author: Mistress-Lukutar
Date:   2026-11-22
Version: v0.21.0
'''

from __future__ import annotations

from brickgen.gui.app import MainWindow

_COMPUTE = MainWindow.__dict__["_compute_changed_mask"]
_COMMIT = MainWindow.__dict__["_commit_applied_mask"]


def _compute(
    scene_dirty: bool,
    applied_blocks: int,
    block_count: int,
    dirty_epoch: dict[int, int],
    committed_epoch: dict[int, int],
) -> list[bool]:
    '''Call the static mask computation without instantiating the window.'''
    return _COMPUTE(
        scene_dirty, applied_blocks, block_count, dirty_epoch, committed_epoch
    )


def _commit(
    dirty_epoch: dict[int, int],
    committed_epoch: dict[int, int],
    changed: list[bool],
    epoch: int,
) -> bool:
    '''Run the commit on a bare window shell; returns the scene_dirty flag.'''
    window = MainWindow.__new__(MainWindow)
    window._scene_dirty = True
    window._dirty_epoch = dirty_epoch
    window._committed_epoch = committed_epoch
    _COMMIT(window, changed, epoch)
    return window._scene_dirty


# ------------------------------------------------------------- whole scene
def test_first_generation_rebuilds_everything() -> None:
    '''Nothing applied yet: every block must be rebuilt.'''
    assert _compute(True, -1, 3, {}, {}) == [True, True, True]


def test_structural_change_rebuilds_everything() -> None:
    '''A block was added/removed: the whole scene is rebuilt.'''
    assert _compute(True, 2, 3, {}, {}) == [True, True, True]


def test_scene_dirty_beats_per_block_flags() -> None:
    '''scene_dirty=True forces a full rebuild even with clean block flags.'''
    assert _compute(True, 2, 2, {0: 5}, {0: 5}) == [True, True]


def test_matching_count_with_clean_flags_rebuilds_nothing() -> None:
    '''No edits since the last apply: nothing is rebuilt.'''
    assert _compute(False, 2, 2, {}, {0: 1, 1: 1}) == [False, False]


# ------------------------------------------------------------ per-block edit
def test_single_block_edit_rebuilds_only_that_block() -> None:
    '''Editing block 1 must flag block 1 only (the reported bug scenario).'''
    mask = _compute(False, 2, 2, {1: 3}, {0: 1, 1: 1})
    assert mask == [False, True]


def test_stale_edit_does_not_rebuild_again() -> None:
    '''A block whose edit was already committed is not rebuilt again.'''
    mask = _compute(False, 2, 2, {0: 1}, {0: 1, 1: 1})
    assert mask == [False, False]


def test_edit_while_generation_in_flight_stays_dirty() -> None:
    '''An edit made after the submission has a newer epoch than the epoch
    the in-flight generation will commit, so the block stays dirty.'''
    dirty = {0: 2}  # edited at epoch 2
    committed = {}
    # generation submitted at epoch 2 rebuilds block 0 and commits it:
    scene_dirty = _commit(dirty, committed, [True, False], epoch=2)
    assert scene_dirty is False
    assert committed == {0: 2}
    # the same block is edited AGAIN while the next generation flies
    # (epoch 3 > commit epoch 2): it must stay dirty.
    dirty[0] = 3
    mask = _compute(False, 2, 2, dirty, committed)
    assert mask == [True, False]


def test_unrelated_block_keeps_flag_after_partial_commit() -> None:
    '''Committing block 0 must not clear block 1's pending edit.'''
    dirty = {0: 2, 1: 2}
    committed: dict[int, int] = {}
    _commit(dirty, committed, [True, False], epoch=2)
    mask = _compute(False, 2, 2, dirty, committed)
    assert mask == [False, True]


def test_stale_generation_discard_keeps_dirty() -> None:
    '''A stale result is never committed, so the edit survives for the
    next generation (commit is only called for applied results).'''
    dirty = {0: 5}
    committed: dict[int, int] = {0: 2}
    mask = _compute(False, 2, 2, dirty, committed)
    assert mask == [True, False]


# ------------------------------------------------------------ full-scene commit
def test_full_rebuild_commit_resets_bookkeeping() -> None:
    '''A whole-scene rebuild clears the per-block flags.'''
    dirty = {0: 7, 1: 3}
    committed = {0: 1, 1: 1}
    scene_dirty = _commit(dirty, committed, [True, True], epoch=7)
    assert scene_dirty is False
    assert dirty == {}
    assert committed == {}


def test_empty_scene_commit_is_safe() -> None:
    '''Committing an empty mask must not raise.'''
    assert _commit({}, {}, [], epoch=1) is False
