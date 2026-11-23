'''
File:   move_gizmo.py
Brief:  Draggable axis-arrow move gizmo (XY on planning/placement, XYZ on assemble).
Author: Mistress-Lukutar
Date:   2026-11-23
Version: v0.22.0
'''

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
import vtk  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from brickgen.gui.preview import PreviewWidget

# Gizmo arm length in grid steps, and proportions of the arrow head.
GIZMO_LENGTH_STEPS = 6
GIZMO_HEAD_FRACTION = 0.35
GIZMO_RADIUS_FRACTION = 0.10


class MoveGizmo:
    '''Arrow-cross gizmo for axis-constrained drag movement.

    Builds thin arrow meshes (shaft + cone head) for each active axis,
    adds them as pickable actors to the preview widget, and provides a
    picker-based method to detect which axis the user clicked on.

    The gizmo is built around the world origin; actor positions place it
    at the selection center. One ``vtkCellPicker`` with generous tolerance
    handles the grab area for the thin arrows.

    Attributes:
        _axes: Ordered list of active axis names ('x', 'y', optionally 'z').
        _colors: Copy of the axis-color mapping passed to the constructor.
        _vectors: Copy of the axis-unit-vector mapping passed to the
            constructor.
        _actors: Actor handle per axis name.
        _actor_axis: Reverse map from actor handle back to axis name.
        _visible: Current visibility state (persisted across rebuilds).
        _built_length: Last arm length used for geometry; triggers rebuild
            only when the grid step changes enough to matter.
    '''

    def __init__(
        self,
        active_axes: Iterable[str],
        axis_colors: Mapping[str, str],
        axis_vectors: Mapping[str, np.ndarray],
    ) -> None:
        '''Create a gizmo for the given axes.

        Args:
            active_axes: Axis names to draw arrows for (e.g. ``['x', 'y']``).
            axis_colors: Mapping of axis name to hex color string.
            axis_vectors: Mapping of axis name to unit vector array.
        '''
        self._axes = [a for a in ("x", "y", "z") if a in set(active_axes)]
        self._colors = dict(axis_colors)
        self._vectors = {k: v.copy() for k, v in axis_vectors.items()}
        self._actors: dict[str, pv.Actor] = {}
        self._actor_axis: dict[pv.Actor, str] = {}
        self._visible = False
        self._built_length = 0.0
        self._meshes: dict[str, pv.PolyData] = {}
        self._picker = vtk.vtkCellPicker()
        # Tolerance is a FRACTION of the viewport diagonal (see PreviewWidget
        # _picker note): thin arrows need a generous grab area, ~1% of the
        # scene diagonal.
        self._picker.SetTolerance(0.01)

    def build(self, length: float) -> None:
        '''(Re)build the per-axis arrow meshes at the given arm length.

        Each axis gets a cylinder shaft and a cone head merged into one
        PolyData. The geometry is built around the world origin so that
        actor positioning places the gizmo at the selection center.

        This method does NOT touch actors; the view owns adding/removing
        (so rebuild works through the view's batching).

        Args:
            length: Total arm length in mm (``GIZMO_LENGTH_STEPS * grid_step``).
        '''
        if abs(length - self._built_length) < 1e-9:
            return

        for axis in self._axes:
            u = self._vectors[axis]
            half = length / 2.0
            # Shaft: cylinder centered at half-length along the axis direction.
            shaft = pv.Cylinder(
                center=u * half,
                direction=u,
                height=length,
                radius=length * GIZMO_RADIUS_FRACTION,
            )
            # Head: cone extending past the shaft tip.
            head_len = length * GIZMO_HEAD_FRACTION
            head = pv.Cone(
                center=u * (length + head_len / 2.0),
                direction=u,
                height=head_len,
                radius=length * GIZMO_RADIUS_FRACTION * 3.0,
            )
            merged = pv.merge([shaft, head])
            self._meshes[axis] = merged

        self._built_length = length

    def meshes(self) -> dict[str, pv.PolyData]:
        '''Return the current arrow PolyData per axis.

        Returns:
            Dict mapping axis name to its ``pv.PolyData`` mesh.
        '''
        return self._meshes

    def add_actors(self, view: "PreviewWidget") -> None:
        '''Add gizmo actors to *view* (invisible to the scene picker).

        Batched with ``reset_camera=False, render=False``: the caller
        renders once after all actors are added.

        Args:
            view: The preview widget to attach actors to.
        '''
        for axis in self._axes:
            mesh = self._meshes.get(axis)
            if mesh is None:
                continue
            actor = view.add_mesh(
                mesh,
                color=self._colors[axis],
                lighting=False,
                pickable=False,  # scene picks must never land on gizmo arrows
                reset_camera=False,
                render=False,
            )
            self._actors[axis] = actor
            self._actor_axis[actor] = axis
            actor.SetVisibility(self._visible)

    def remove_actors(self, view: "PreviewWidget") -> None:
        '''Remove all gizmo actors from *view* (batched).

        Args:
            view: The preview widget to remove actors from.
        '''
        for actor in list(self._actors.values()):
            try:
                view.remove_actor(actor, reset_camera=False, render=False)
            except Exception:  # noqa: BLE001 - may already be detached
                pass
        self._actors.clear()
        self._actor_axis.clear()

    def place(self, center: np.ndarray) -> None:
        '''Move every gizmo actor to *center*.

        The arrows are built around the world origin; the actor position
        IS the gizmo center.

        Args:
            center: ``(x, y, z)`` world coordinates for the gizmo center
                (``np.ndarray`` of shape ``(3,)``).
        '''
        for actor in self._actors.values():
            actor.SetPosition(*center)

    def set_visible(self, visible: bool) -> None:
        '''Set the visibility of all gizmo actors.

        Args:
            visible: True to show the gizmo arrows.
        '''
        self._visible = visible
        for actor in self._actors.values():
            actor.SetVisibility(visible)

    def visible(self) -> bool:
        '''Current visibility state of the gizmo.

        Returns:
            True if the gizmo is currently visible.
        '''
        return self._visible

    def pick_axis(
        self, view: "PreviewWidget", pos: Any
    ) -> str | None:
        '''Pick which axis arrow lies under *pos* in display coordinates.

        Converts Qt widget coords to VTK display coords and uses the
        dedicated cell picker to find a hit. Returns the axis name of the
        hit actor, or ``None`` when nothing was picked.

        Args:
            view: The preview widget (for display-coord conversion).
            pos: Cursor position in Qt widget coordinates.

        Returns:
            Axis name ('x', 'y', or 'z') of the picked arrow, or None.
        '''
        x, y = view._to_display(pos)  # noqa: SLF001 - internal method
        for actor in self._actors.values():
            actor.SetPickable(True)
        try:
            if not self._picker.Pick(x, y, 0, view.renderer):
                return None
            actor = self._picker.GetActor()
            return self._actor_axis.get(actor)
        finally:
            for actor in self._actors.values():
                actor.SetPickable(False)
