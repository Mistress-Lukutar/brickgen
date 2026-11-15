'''
File:   preview.py
Brief:  Matplotlib-based 3D preview panel embedded into the tkinter GUI.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.1.0
'''

from __future__ import annotations

import logging

import cadquery as cq
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)

# Tessellation tolerance for the interactive preview, mm.
PREVIEW_TESSELLATION_TOLERANCE = 0.2

_PREVIEW_COLOR = "#3f7fbf"
_PREVIEW_EDGE_COLOR = "#1d3d5c"


class PreviewPanel:
    '''Tk-embedded 3D mesh preview rendered with matplotlib.'''

    def __init__(self, parent: object) -> None:
        '''Create the preview panel inside ``parent`` (a tk widget).

        Args:
            parent: Parent tkinter widget or grid slot container.
        '''
        self._figure = Figure(figsize=(5, 4), dpi=100)
        self._axes = self._figure.add_subplot(111, projection="3d")
        self._canvas = FigureCanvasTkAgg(self._figure, master=parent)
        self._toolbar = NavigationToolbar2Tk(self._canvas, parent, pack_toolbar=False)
        self._toolbar.config(height=28)
        self._toolbar.pack(side="bottom", fill="x")
        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self._show_empty()

    def show_model(self, model: cq.Workplane, tolerance: float = PREVIEW_TESSELLATION_TOLERANCE) -> None:
        '''Render a CadQuery solid as a shaded mesh.

        Args:
            model: Workplane whose first solid is displayed.
            tolerance: Tessellation tolerance in mm.
        '''
        solid = model.val()
        if solid is None:
            self._show_empty()
            return
        vertices, triangles = solid.tessellate(tolerance)
        verts = np.asarray([(v.x, v.y, v.z) for v in vertices])
        faces = verts[np.asarray(triangles)]
        self._axes.clear()
        collection = Poly3DCollection(
            faces,
            facecolors=_PREVIEW_COLOR,
            edgecolors=_PREVIEW_EDGE_COLOR,
            linewidths=0.15,
            shade=True,
        )
        self._axes.add_collection3d(collection)
        self._fit_axes(verts)
        self._axes.set_xlabel("X, mm")
        self._axes.set_ylabel("Y, mm")
        self._axes.set_zlabel("Z, mm")
        self._axes.view_init(elev=28, azim=-55)
        self._canvas.draw_idle()
        logger.debug("Preview updated: %d verts / %d faces", len(verts), len(faces))

    def _fit_axes(self, verts: np.ndarray) -> None:
        '''Fit axes limits to the mesh bounding box with equal aspect ratio.'''
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        center = (mins + maxs) / 2.0
        half = float(((maxs - mins) / 2.0).max()) * 1.1 or 1.0
        self._axes.set_xlim(center[0] - half, center[0] + half)
        self._axes.set_ylim(center[1] - half, center[1] + half)
        self._axes.set_zlim(center[2] - half, center[2] + half)
        self._axes.set_box_aspect((1, 1, 1))

    def _show_empty(self) -> None:
        '''Display a blank scene before the first model is generated.'''
        self._axes.clear()
        self._axes.set_xlim(-4, 4)
        self._axes.set_ylim(-4, 4)
        self._axes.set_zlim(0, 8)
        self._axes.set_xlabel("X, mm")
        self._axes.set_ylabel("Y, mm")
        self._axes.set_zlabel("Z, mm")
        self._canvas.draw_idle()
