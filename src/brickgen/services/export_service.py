'''
File:   export_service.py
Brief:  Export of generated solids to STEP/STL/AMF/3MF/SVG/VRML/BREP files.
Author: Mistress-Lukutar
Date:   2026-11-15
Version: v0.1.0
'''

from __future__ import annotations

import logging
from pathlib import Path

import cadquery as cq
from cadquery.occ_impl.exporters import ExportTypes, export

from brickgen.core.exceptions import ExportError

logger = logging.getLogger(__name__)

# Publicly offered formats, in GUI/CLI menu order.
SUPPORTED_FORMATS: dict[str, str] = {
    "STEP": ExportTypes.STEP,
    "STL": ExportTypes.STL,
    "AMF": ExportTypes.AMF,
    "3MF": ExportTypes.THREEMF,
    "SVG": ExportTypes.SVG,
    "VRML": ExportTypes.VRML,
    "BREP": ExportTypes.BREP,
}

# Default tessellation tolerance for STL-family export, mm.
DEFAULT_TESSELLATION_TOLERANCE = 0.1


def export_model(
    model: cq.Workplane,
    output_path: Path,
    file_format: str,
    tolerance: float = DEFAULT_TESSELLATION_TOLERANCE,
) -> Path:
    '''Export a CadQuery model to the requested 3D file format.

    Args:
        model: Solid model to export.
        output_path: Destination file path (parent directories are created).
        file_format: One of ``SUPPORTED_FORMATS`` keys (STEP, STL, AMF, 3MF, ...).
        tolerance: Mesh tolerance in mm for tessellated formats (STL/AMF/3MF).

    Returns:
        The resolved path of the written file.

    Raises:
        ExportError: If the format is unknown or the exporter fails.
    '''
    export_type = SUPPORTED_FORMATS.get(file_format.upper())
    if export_type is None:
        raise ExportError(
            f"Unsupported format '{file_format}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    output_path = output_path.expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export(
            model,
            str(output_path),
            exportType=export_type,
            tolerance=tolerance,
        )
    except ExportError:
        raise
    except Exception as exc:  # noqa: BLE001 - OCC raises heterogeneous errors
        raise ExportError(f"Failed to export {file_format}: {exc}") from exc
    logger.info("Exported %s to %s", file_format, output_path)
    return output_path
