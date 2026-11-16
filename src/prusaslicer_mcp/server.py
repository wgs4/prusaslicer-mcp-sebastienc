"""MCP server exposing the PrusaSlicer 2.9.x command line to Claude."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from . import cli, orientation, repair
from .config import Config, ConfigError, load_config
from .gcode import GcodeStats, filament_by_feature, layer_gap_warning, parse_gcode_stats
from .preview_gcode import render_layer, render_profile, scan_gcode
from .preview_model import SHEET_ORDER, render_model, to_png
from .project import ProjectInfo, read_project
from .slicer_output import parse_warnings

if TYPE_CHECKING:
    import trimesh

mcp = MCPServer(
    "prusaslicer",
    instructions=(
        "Headless PrusaSlicer 2.9.x. Discover exact profile names with list_printers and list_profiles, "
        "check a model with inspect_model, then slice_model. Profile names must match exactly."
    ),
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

# The transforms are the same for slice_model and preview_model, so each is described once (ADR 0012, 0015).
_RotateX = Annotated[
    float | None,
    Field(
        description="Degrees to rotate about the X axis. PrusaSlicer applies rotations in the fixed order X, then Y, "
        "then Z, whatever order you think of them. 90 or -90 stands a model that lies along Y upright, e.g. a Y-up "
        "scan (check with preview_model first). Exactly +-90 about Y combined with a Z rotation can come out "
        "differently from that order: preview_model shows what PrusaSlicer really does."
    ),
]
_RotateY = Annotated[float | None, Field(description="Degrees to rotate about the Y axis (applied after rotate_x).")]
_RotateZ = Annotated[float | None, Field(description="Degrees to rotate about the vertical Z axis (applied last).")]
_Scale = Annotated[
    float | None,
    Field(gt=0, description="Scale factor: 0.5 is half size, 2 is double. Not a percentage. Not with scale_to_fit."),
]
_ScaleToFit = Annotated[
    list[float] | None,
    Field(
        min_length=3,
        max_length=3,
        description="[x, y, z] in mm: scale the model uniformly to fit inside this box, e.g. the bed size. Not with scale.",
    ),
]

_MODEL_SUFFIXES = {".stl", ".3mf", ".obj", ".amf", ".step", ".stp"}
_STEP_SUFFIXES = {".step", ".stp"}
_GCODE_SUFFIXES = {".gcode", ".gco"}
_OVERRIDE_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
# Flags the server sets itself, plus `save`, which would write into the user's PrusaSlicer data directory.
# An override with one of these names would silently undo a guarantee (e.g. binary_gcode=1 beats our text G-code).
_RESERVED_OVERRIDES = frozenset(
    {
        "export_gcode",
        "binary_gcode",
        "printer_profile",
        "print_profile",
        "material_profile",
        "output",
        "datadir",
        "save",
        "center",
    }
)
_SHEET_CORNERS = ("top left", "top right", "bottom left", "bottom right")
_PROFILE_NOT_FOUND = re.compile(r"profile '.*' wasn't found", re.IGNORECASE)
_PROFILE_HINT = " (hint: profile names must match exactly; get them from list_printers and list_profiles)"

_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        try:
            _config = load_config()
        except ConfigError as e:
            raise ToolError(str(e)) from e
    return _config


async def _run(cfg: Config, args: list[str], timeout: float | None = None) -> cli.CliResult:
    try:
        return await cli.run(cfg, args, timeout=timeout)
    except cli.CliTimeout as e:
        raise ToolError(str(e)) from e


def _with_profile_hint(detail: str) -> str:
    return detail + _PROFILE_HINT if _PROFILE_NOT_FOUND.search(detail) else detail


# Refusals PrusaSlicer words tersely, with what to try. Each was met slicing real models and the remedy was verified.
_STATIC_HINTS = (
    (
        re.compile(r"no extrusions in the first layer", re.IGNORECASE),
        " (hint: the model's lowest layer has nothing to print, usually because its underside is open, as in many "
        "scans, or it rests on a point. suggest_orientation ranks the ways to stand it, and preview_model, which "
        "takes rotate_x, rotate_y and close_holes too, shows one without slicing, to put a flat, closed face down. "
        "If inspect_model reports a hole that can be closed, repair the mesh with close_holes=true)",
    ),
    (
        re.compile(r"wipe tower is currently not supported for multimaterial sequential", re.IGNORECASE),
        " (hint: this is a conflict in the project's own settings. Change one with overrides, for example "
        '{"wipe_tower": "0"} or {"complete_objects": "0"})',
    ),
)


async def _outside_volume_hint(model: Path, printer_profile: str | None, centred: bool) -> str:
    """Say why a model is outside the print volume when we can tell: too big for the bed, or just off it."""
    centre_hint = " (hint: retry with center=true to place the model in the middle of the bed)"
    if printer_profile is not None:
        try:
            x, y, z = (float(v) for v in (await _load_model_mesh(model)).extents)
            width, depth, height = await _bed_size(printer_profile)
        except ToolError:
            return "" if centred else centre_hint  # sizes unknown: the position is the likeliest cause
        too_big = z > height or not ((x <= width and y <= depth) or (x <= depth and y <= width))
        if too_big:
            fits_diagonally = z <= height and (x + y) / math.sqrt(2) <= min(width, depth)
            limit_z = height if math.isfinite(height) else z  # no height limit known: leave the height alone
            bed = f"{width:g} x {depth:g}" + (f" x {height:g}" if math.isfinite(height) else "")
            fit = f"scale_to_fit=[{width:g}, {depth:g}, {limit_z:g}]"
            rotate = " or rotate_z=45 to lay it across the bed's diagonal" if fits_diagonally else ""
            return (
                f" (hint: the model is {x:.1f} x {y:.1f} x {z:.1f} mm but the bed is {bed} mm. "
                f"Use {fit} to shrink it{rotate})"
            )
    return "" if centred else centre_hint


async def _explain_failure(detail: str, model: Path, printer_profile: str | None, centred: bool) -> str:
    """PrusaSlicer's own message plus advice for the refusals we know how to resolve."""
    detail = _with_profile_hint(detail)
    for pattern, hint in _STATIC_HINTS:
        if pattern.search(detail):
            detail += hint
    if "outside of the print volume" in detail:
        detail += await _outside_volume_hint(model, printer_profile, centred)
    return detail


def _chosen_profiles(
    printer: str | None, print_: str | None, material: str | None, project: ProjectInfo | None
) -> tuple[str, str, str] | None:
    """The three profile names to slice with, or None to use the project's own saved settings (ADR 0007).

    All three or none: a partial set would mix profile values with the project's. None is only for a project.
    """
    if printer is not None and print_ is not None and material is not None:
        return printer, print_, material
    if printer is None and print_ is None and material is None:
        if project is None:
            raise ToolError(
                "printer_profile, print_profile and material_profile are required unless the model is a PrusaSlicer "
                "project (a .3mf saved from PrusaSlicer), which is sliced with its own settings"
            )
        return None
    raise ToolError("Give all three profiles (printer_profile, print_profile, material_profile) or none")


def _load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a model file as one mesh, or raise a ToolError a client can read."""
    import trimesh

    try:
        mesh = trimesh.load(path, force="mesh")
    except (
        Exception
    ) as e:  # trimesh's parsers raise assorted types on bad input (BadZipFile, ValueError, ImportError, ...)
        raise ToolError(f"Could not read {path.name} as a {path.suffix.lower()} model: {e}") from e
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ToolError("Could not load a triangle mesh from this file.")
    return mesh


async def _export_mesh(source: Path, flags: list[str]) -> trimesh.Trimesh:
    """Have PrusaSlicer write `source` as an STL, with transform `flags` applied, and load the result.

    This is how a STEP file becomes a mesh (trimesh cannot read it), and how a preview shows exactly what
    PrusaSlicer's own transforms do (ADR 0015, 0016). The STL lives in a temporary folder.
    """
    cfg = get_config()
    with tempfile.TemporaryDirectory(prefix="prusaslicer-mcp-") as folder:
        target = Path(folder) / f"{source.stem}.stl"
        result = await _run(cfg, ["--export-stl", *flags, "--output", str(target), str(source)])
        if result.returncode != 0 or not target.is_file():
            detail = (result.stderr.strip() or result.stdout.strip() or "no message")[-1000:]
            what = "a valid STEP file" if source.suffix.lower() in _STEP_SUFFIXES else "a model PrusaSlicer can read"
            raise ToolError(f"PrusaSlicer could not convert {source.name} to a mesh (is it {what}?): {detail}")
        return await asyncio.to_thread(_load_mesh, target)


async def _load_model_mesh(path: Path) -> trimesh.Trimesh:
    """The model as one mesh, converting STEP through PrusaSlicer. Loading runs off the event loop."""
    if path.suffix.lower() in _STEP_SUFFIXES:
        return await _export_mesh(path, [])
    return await asyncio.to_thread(_load_mesh, path)


def _closing_note(closing: repair.Closing, doing: Literal["slicing", "drawing", "scoring"]) -> str:
    """What close_holes did, so a result never hides that the mesh used was not the file."""
    used = {"slicing": "sliced", "drawing": "drawn", "scoring": "scored"}[doing]
    if closing.closed:
        flat = (
            f" (an uneven rim was extended by up to {closing.extended_mm:.1f} mm so the base is flat)"
            if closing.extended_mm >= 0.05
            else ""
        )
        rest = f"; {closing.left_open} other open hole(s) were left as they are" if closing.left_open else ""
        return f"close_holes capped {closing.closed} flat hole(s) in a copy of the model before {doing}{flat}{rest}."
    if closing.left_open:
        return (
            f"close_holes could not close any of the {closing.left_open} open hole(s): they are tiny, bent or touch "
            f"themselves. The model was {used} as it is."
        )
    return f"close_holes found no open holes, so the model was {used} as it is."


def _transform_args(
    rotate_x: float | None,
    rotate_y: float | None,
    rotate_z: float | None,
    scale: float | None,
    scale_to_fit: list[float] | None,
    split: bool,
    copies: int | None,
) -> list[str]:
    """PrusaSlicer's transform flags for the requested changes, validated (ADR 0012).

    PrusaSlicer applies them in its own fixed order whatever order they are passed: rotate about X, then Y, then
    Z, then scale. `--scale` takes a factor, and a bare `50` would mean fifty times, so only factors are accepted.
    """
    args: list[str] = []
    for name, flag, degrees in (
        ("rotate_x", "--rotate-x", rotate_x),
        ("rotate_y", "--rotate-y", rotate_y),
        ("rotate_z", "--rotate", rotate_z),
    ):
        if degrees is None:
            continue
        if not math.isfinite(degrees):
            raise ToolError(f"{name} must be a finite number of degrees")
        args += [flag, f"{degrees:g}"]
    if scale is not None and scale_to_fit is not None:
        raise ToolError("Use either scale or scale_to_fit, not both")
    if scale is not None:
        if not math.isfinite(scale) or scale <= 0:
            raise ToolError("scale must be a factor greater than 0 (0.5 is half size, 2 is double)")
        args += ["--scale", f"{scale:g}"]
    if scale_to_fit is not None:
        if len(scale_to_fit) != 3:
            raise ToolError("scale_to_fit needs exactly 3 sizes in millimetres: [x, y, z]")
        if not all(math.isfinite(size) and size > 0 for size in scale_to_fit):
            raise ToolError("scale_to_fit sizes must all be positive")
        args += ["--scale-to-fit", ",".join(f"{size:g}" for size in scale_to_fit)]
    if split:
        args.append("--split")
    if copies is not None:
        if not 1 <= copies <= 100:
            raise ToolError("copies must be between 1 and 100")
        args += ["--duplicate", str(copies)]
    return args


def _model_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ToolError(f"Model file not found: {p}")
    if p.suffix.lower() not in _MODEL_SUFFIXES:
        raise ToolError(f"Unsupported model type '{p.suffix}'. Supported: {sorted(_MODEL_SUFFIXES)}")
    return p


def _gcode_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ToolError(f"G-code file not found: {p}")
    if p.suffix.lower() not in _GCODE_SUFFIXES:
        raise ToolError(f"Unsupported G-code type '{p.suffix}'. Supported: {sorted(_GCODE_SUFFIXES)}")
    with p.open("rb") as f:
        if f.read(4) == b"GCDE":
            raise ToolError("Binary G-code (.bgcode) cannot be drawn; slice_model always writes text G-code.")
    return p


async def _query_json(args: list[str]) -> dict[str, Any]:
    """Run a --query-* action and return the JSON it writes with -o."""
    cfg = get_config()
    with tempfile.TemporaryDirectory(prefix="prusaslicer-mcp-") as tmp:
        out = Path(tmp) / "result.json"
        result = await _run(cfg, [*args, "-o", str(out)], timeout=60)
        # PrusaSlicer 2.9.x exits with status 1 even when a --query-* action succeeds,
        # so success is judged by the output file, not the exit code.
        if not out.is_file() or out.stat().st_size == 0:
            detail = _with_profile_hint(result.stderr.strip() or result.stdout.strip())
            raise ToolError(f"PrusaSlicer query failed (exit {result.returncode}): {detail}")
        try:
            return json.loads(
                out.read_text(encoding="utf-8")
            )  # PrusaSlicer writes UTF-8; the default is a code page on Windows
        except json.JSONDecodeError as e:
            raise ToolError(f"PrusaSlicer produced invalid JSON: {e}") from e


# --------------------------------------------------------------------------- printers / profiles


class PrinterVariant(BaseModel):
    nozzle_diameter: float | str
    printer_profile: str
    extruders: int | None = None
    bed_width: float | None = None
    bed_depth: float | None = None
    max_print_height: float | None = None


class PrinterModel(BaseModel):
    id: str
    name: str
    technology: str
    variants: list[PrinterVariant]


def _display_name(name: str) -> str:
    # Vendor bundles store model names wx-escaped ("&&" renders as "&" in the GUI). Unescape so the
    # name reads naturally and matches the real profile names, which use a single "&".
    return name.replace("&&", "&")


def _as_list(value: Any) -> list[Any]:
    """PrusaSlicer uses a list, or omits the key, or writes an empty string when there are none."""
    return value if isinstance(value, list) else []


def _holders(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Nozzle variants, or the model itself when 2.9 has no `variants` array."""
    variants = model.get("variants")
    if isinstance(variants, list) and variants:
        return [item for item in variants if isinstance(item, dict)]
    return [model]


def _profile_name(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and isinstance(item.get("name"), str):
        return item["name"]
    return None


def _unique_names(values: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in _as_list(values):
        name = _profile_name(item)
        if name is None or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _printer_entries(model: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """System and user printer profiles, each exact name once. The first occurrence wins."""
    found: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for holder in _holders(model):
        for key in ("printer_profiles", "user_printer_profiles"):
            for profile in _as_list(holder.get(key)):
                if not isinstance(profile, dict):
                    continue
                name = profile.get("name")
                if not isinstance(name, str) or name in seen:
                    continue
                seen.add(name)
                found.append((holder, profile))
    return found


_BED_ORIGIN = re.compile(r"^\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]$")


def _bed_origin(bed: dict[str, Any]) -> tuple[float, float]:
    """Machine-origin offset of the bed's minimum corner, in mm.

    PrusaSlicer 2.9 writes this as a string, for example "[110, 110]". A missing
    or unreadable value means the bed starts at zero.
    """
    raw = bed.get("origin")
    if isinstance(raw, str):
        match = _BED_ORIGIN.match(raw.strip())
        if match:
            return float(match.group(1)), float(match.group(2))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return 0.0, 0.0
    return 0.0, 0.0


async def _printer_bed(printer_profile: str) -> dict[str, Any]:
    """Bed record of a known FFF printer profile, system or user."""
    data = await _query_json(["--query-printer-models", "--printer-technology", "FFF"])
    for model in _as_list(data.get("printer_models")):
        if not isinstance(model, dict):
            continue
        for _holder, profile in _printer_entries(model):
            bed = profile.get("bed") or {}
            if (
                profile.get("name") == printer_profile
                and isinstance(bed, dict)
                and bed.get("width")
                and bed.get("height")
            ):
                return bed
    raise ToolError(
        f"printer profile {printer_profile!r} was not found with a bed size; use the exact name from list_printers"
    )


async def _bed_size(printer_profile: str) -> tuple[float, float, float]:
    """Width, depth and maximum print height of a known FFF printer profile's bed, in mm."""
    bed = await _printer_bed(printer_profile)
    return bed["width"], bed["height"], bed.get("max_print_height") or float("inf")


async def _bed_center(printer_profile: str) -> tuple[float, float]:
    """Middle of the bed for a known FFF printer profile, in the printer's own coordinates."""
    try:
        bed = await _printer_bed(printer_profile)
    except ToolError as e:
        raise ToolError(f"Cannot centre on the bed: {e}") from e
    width, depth = float(bed["width"]), float(bed["height"])
    origin_x, origin_y = _bed_origin(bed)
    return width / 2 - origin_x, depth / 2 - origin_y


@mcp.tool(annotations=_READ_ONLY)
async def list_printers(
    technology: Annotated[str, Field(description="'FFF' (filament) or 'SLA' (resin)")] = "FFF",
) -> list[PrinterModel]:
    """List the printer models installed in PrusaSlicer, with the exact printer profile
    names (use these as `printer_profile` elsewhere) and bed sizes."""
    if technology not in ("FFF", "SLA"):
        raise ToolError("technology must be 'FFF' or 'SLA'")
    data = await _query_json(["--query-printer-models", "--printer-technology", technology])
    models: list[PrinterModel] = []
    for m in _as_list(data.get("printer_models")):
        if not isinstance(m, dict):
            continue
        variants = []
        for holder, p in _printer_entries(m):
            bed = p.get("bed") or {}
            variants.append(
                PrinterVariant(
                    nozzle_diameter=holder.get("name"),
                    printer_profile=p["name"],
                    extruders=p.get("extruders_cnt"),
                    bed_width=bed.get("width"),
                    bed_depth=bed.get("height"),
                    max_print_height=bed.get("max_print_height"),
                )
            )
        models.append(
            PrinterModel(
                id=m["id"], name=_display_name(m["name"]), technology=m.get("technology", technology), variants=variants
            )
        )
    return models


class PrintProfile(BaseModel):
    name: str
    filament_profiles: list[str]


class ProfileList(BaseModel):
    printer_profile: str
    print_profiles: list[PrintProfile]


@mcp.tool(annotations=_READ_ONLY)
async def list_profiles(
    printer_profile: Annotated[str, Field(description="Exact printer profile name, from list_printers")],
) -> ProfileList:
    """List the print (quality) profiles and compatible filament profiles for a printer profile."""
    data = await _query_json(["--query-print-filament-profiles", "--printer-profile", printer_profile])
    # System profiles first, then user presets. The same print-profile name is listed once;
    # filaments from a later copy are added if that name did not already have them.
    filaments_for: dict[str, list[str]] = {}
    order: list[str] = []
    for key in ("print_profiles", "user_print_profiles"):
        for profile in _as_list(data.get(key)):
            if not isinstance(profile, dict) or not isinstance(profile.get("name"), str):
                continue
            name = profile["name"]
            seen = set(filaments_for.get(name, []))
            extra: list[str] = []
            for filament_key in ("filament_profiles", "user_filament_profiles"):
                for filament in _unique_names(profile.get(filament_key)):
                    if filament not in seen:
                        seen.add(filament)
                        extra.append(filament)
            if name in filaments_for:
                filaments_for[name].extend(extra)
            else:
                filaments_for[name] = extra
                order.append(name)
    return ProfileList(
        printer_profile=data.get("printer_profile", printer_profile),
        print_profiles=[PrintProfile(name=name, filament_profiles=filaments_for[name]) for name in order],
    )


# --------------------------------------------------------------------------- model inspection


class ModelInfo(BaseModel):
    path: str
    triangles: int
    size_mm: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    volume_mm3: float | None
    surface_area_mm2: float
    watertight: bool
    open_holes: int
    closable_holes: int
    connected_parts: int
    fits_bed: bool | None = None
    project: ProjectInfo | None = None
    warnings: list[str]


@mcp.tool(annotations=_READ_ONLY)
async def inspect_model(
    model_path: Annotated[str, Field(description="Absolute path to an STL/3MF/OBJ file")],
    bed_width: Annotated[float | None, Field(description="Bed X size in mm, to check fit")] = None,
    bed_depth: Annotated[float | None, Field(description="Bed Y size in mm, to check fit")] = None,
    max_height: Annotated[float | None, Field(description="Max print height in mm, to check fit")] = None,
) -> ModelInfo:
    """Check a model before slicing: size, volume, watertightness, number of separate parts,
    and (optionally) whether it fits a given build volume."""
    path = _model_path(model_path)
    mesh = await _load_model_mesh(path)

    size = tuple(float(x) for x in mesh.extents)
    warnings: list[str] = []
    holes = [] if mesh.is_watertight else repair.find_holes(mesh)
    closable = sum(hole.closable for hole in holes)
    if not mesh.is_watertight:
        warnings.append("Mesh is not watertight (holes or non-manifold edges); PrusaSlicer may need to repair it.")
    if closable:
        stay = (
            f" The other {len(holes) - closable} (small, bent or part of an open shell) would stay open."
            if closable < len(holes)
            else ""
        )
        warnings.append(
            f"{closable} of the {len(holes)} open hole(s) are flat gaps, such as a missing base: slice_model can "
            f"cap them with close_holes=true.{stay}"
        )
    elif not mesh.is_winding_consistent:
        warnings.append("Face winding is inconsistent.")
    parts = len(mesh.split(only_watertight=False))
    if parts > 1:
        warnings.append(f"Model has {parts} disconnected parts.")

    fits = None
    if bed_width is not None and bed_depth is not None and max_height is not None:
        fits = size[0] <= bed_width and size[1] <= bed_depth and size[2] <= max_height
        if not fits:
            warnings.append(
                f"Model ({size[0]:.1f}x{size[1]:.1f}x{size[2]:.1f} mm) exceeds the build volume in at least one axis (rotation not considered)."
            )

    # PrusaSlicer places STL/OBJ itself but keeps a 3MF's stored coordinates, so a 3MF that fits by size
    # can still sit off the bed (e.g. modelled around the origin) and fail to slice.
    project = read_project(path)
    if path.suffix.lower() == ".3mf" and bed_width is not None and bed_depth is not None:
        lo, hi = mesh.bounds
        if lo[0] < 0 or lo[1] < 0 or hi[0] > bed_width or hi[1] > bed_depth:
            where = (
                f"Stored position is off the bed (x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f} mm; "
                f"bed is 0..{bed_width:g} by 0..{bed_depth:g}). PrusaSlicer keeps a 3MF's position. "
            )
            remedy = (
                "slice_model centres a plain 3MF on the bed automatically (center=false keeps this position)."
                if project is None
                else "This is a PrusaSlicer project, so slice_model keeps its layout; pass center=true to centre it."
            )
            warnings.append(where + remedy)

    return ModelInfo(
        path=str(path),
        triangles=len(mesh.faces),
        size_mm=size,  # type: ignore[arg-type]
        bounds_min=tuple(float(x) for x in mesh.bounds[0]),  # type: ignore[arg-type]
        bounds_max=tuple(float(x) for x in mesh.bounds[1]),  # type: ignore[arg-type]
        volume_mm3=float(mesh.volume) if mesh.is_watertight else None,
        surface_area_mm2=float(mesh.area),
        watertight=bool(mesh.is_watertight),
        open_holes=len(holes),
        closable_holes=closable,
        connected_parts=parts,
        fits_bed=fits,
        project=project,
        warnings=warnings,
    )


PreviewView = Literal["iso", "front", "side", "top", "sheet"]


def _applied_transforms(
    rotate_x: float | None,
    rotate_y: float | None,
    rotate_z: float | None,
    scale: float | None,
    scale_to_fit: list[float] | None,
) -> list[str]:
    """The transforms that were asked for, as the caller wrote them, for a caption."""
    named = (("rotate_x", rotate_x), ("rotate_y", rotate_y), ("rotate_z", rotate_z), ("scale", scale))
    applied = [f"{name}={value:g}" for name, value in named if value is not None]
    if scale_to_fit is not None:
        applied.append("scale_to_fit=[" + ", ".join(f"{size:g}" for size in scale_to_fit) + "]")
    return applied


@mcp.tool(annotations=_READ_ONLY)
async def preview_model(
    model_path: Annotated[str, Field(description="Absolute path to an STL/3MF/OBJ/STEP file")],
    view: Annotated[
        PreviewView,
        Field(
            description="iso (default), front, side or top; sheet puts all four in one picture. Z is up, and the model "
            "is drawn centred, as stored unless you give transforms."
        ),
    ] = "iso",
    size: Annotated[int, Field(description="Picture width and height in pixels", ge=128, le=1024)] = 640,
    rotate_x: _RotateX = None,
    rotate_y: _RotateY = None,
    rotate_z: _RotateZ = None,
    scale: _Scale = None,
    scale_to_fit: _ScaleToFit = None,
    close_holes: Annotated[
        bool,
        Field(
            description="Draw the model with its flat holes capped, as slice_model(close_holes=true) would slice it: "
            "see what the base of a scan looks like once closed. Not for PrusaSlicer projects."
        ),
    ] = False,
) -> list[Image | str]:
    """Draw a picture of a model, to check its shape and which way up it is before slicing.

    With rotate_*, scale or scale_to_fit it draws the model as PrusaSlicer will transform it, by asking PrusaSlicer
    itself, so an orientation can be checked before spending a slice on it. Returns a PNG and a caption with its
    size. PrusaSlicer prints with z up too, so a model that looks to be lying on its side will be printed that way
    (scans are often Y-up)."""
    path = _model_path(model_path)
    flags = _transform_args(rotate_x, rotate_y, rotate_z, scale, scale_to_fit, False, None)
    if close_holes and read_project(path) is not None:
        raise ToolError(
            "close_holes cannot be used with a PrusaSlicer project: a repaired copy would not keep its saved settings."
        )

    notes: list[str] = []
    mesh: trimesh.Trimesh | None = None
    with tempfile.TemporaryDirectory(prefix="prusaslicer-mcp-") as folder:
        source = path
        if close_holes:
            closing = await asyncio.to_thread(repair.close_holes, await _load_model_mesh(path))
            notes.append(_closing_note(closing, "drawing"))
            mesh = closing.mesh
            if closing.closed and flags:
                source = Path(folder) / f"{path.stem}.stl"  # named like the model, as in slice_model
                await asyncio.to_thread(closing.mesh.export, source)
        if flags:
            mesh = await _export_mesh(source, flags)
        elif mesh is None:
            mesh = await _load_model_mesh(path)

    try:
        picture = render_model(mesh, view, size, bed=True)
    except ValueError as e:
        raise ToolError(str(e)) from e
    x, y, z = (float(v) for v in mesh.extents)
    where = (
        "sheet: " + ", ".join(f"{name} ({corner})" for name, corner in zip(SHEET_ORDER, _SHEET_CORNERS, strict=True))
        if view == "sheet"
        else f"{view} view"
    )
    applied = _applied_transforms(rotate_x, rotate_y, rotate_z, scale, scale_to_fit)
    how = f"after {', '.join(applied)}, as PrusaSlicer will apply them" if applied else "as stored"
    caption = " ".join(
        [
            f"{path.name}, {where}. Size {x:.1f} x {y:.1f} x {z:.1f} mm (x, y, z). Z is up and the model is drawn "
            f"{how}, centred. The pale grid is the bed, level with the model's lowest point. PrusaSlicer also prints "
            "with z up, so a model that looks to be lying on its side would be printed that way.",
            *notes,
        ]
    )
    return [Image(data=to_png(picture), format="png"), caption]


class Measurement(BaseModel):
    """What slicing an orientation with supports on really gave. `refused` is PrusaSlicer's reason, and then the
    numbers are empty."""

    support_g: float | None = None
    filament_g: float | None = None
    estimated_time: str | None = None
    estimated_time_seconds: int | None = None
    refused: str | None = None


class OrientationOption(BaseModel):
    rank: int
    geometry_rank: int
    rotate_x: float
    rotate_y: float
    height_mm: float
    base_area_mm2: float
    base_share: float
    overhang_area_mm2: float
    overhang_share: float
    stable: bool
    as_stored: bool
    measured: Measurement | None = None


class OrientationAdvice(BaseModel):
    surface_area_mm2: float
    options: list[OrientationOption]
    warnings: list[str]


_MAX_MEASURED_OPTIONS = 6


async def _measure_support(
    model: Path,
    option: orientation.Orientation,
    profiles: tuple[str, str, str],
    close_holes: bool,
) -> Measurement:
    """Slice the model in one orientation with automatic supports and read what the slice used (ADR 0019)."""
    with tempfile.TemporaryDirectory(prefix="prusaslicer-mcp-") as folder:
        gcode = Path(folder) / "measure.gcode"
        try:
            result = await slice_model(
                str(model),
                *profiles,
                overrides={"support_material": "1", "support_material_auto": "1"},
                output_path=str(gcode),
                rotate_x=option.rotate_x or None,
                rotate_y=option.rotate_y or None,
                close_holes=close_holes,
            )
        except ToolError as e:
            return Measurement(refused=str(e).splitlines()[0][:300])
        by_feature = await asyncio.to_thread(filament_by_feature, gcode)
    total = sum(by_feature.values())
    support = sum(amount for name, amount in by_feature.items() if name.startswith("Support material"))
    grams = result.stats.filament_used_g
    return Measurement(
        support_g=round(grams * support / total, 2) if grams is not None and total > 0 else 0.0 if total == 0 else None,
        filament_g=grams,
        estimated_time=result.stats.estimated_time,
        estimated_time_seconds=result.stats.estimated_time_seconds,
    )


def _rank_by_measurement(options: list[OrientationOption]) -> list[OrientationOption]:
    """Refused last, then stable before unstable, then the least support and the shortest print.

    The sort is stable and the options arrive in geometric order, so equal measurements keep that order.
    """

    def key(option: OrientationOption) -> tuple[bool, bool, float, int]:
        measured = option.measured or Measurement()
        return (
            measured.refused is not None,
            not option.stable,
            measured.support_g if measured.support_g is not None else 0.0,
            measured.estimated_time_seconds or 0,
        )

    ordered = sorted(options, key=key)
    return [option.model_copy(update={"rank": position}) for position, option in enumerate(ordered, start=1)]


@mcp.tool(annotations=_READ_ONLY)
async def suggest_orientation(
    model_path: Annotated[
        str, Field(description="Absolute path to an STL/3MF/OBJ/STEP file (not a PrusaSlicer project)")
    ],
    count: Annotated[int, Field(description="How many of the best orientations to return", ge=1, le=12)] = 5,
    close_holes: Annotated[
        bool,
        Field(
            description="Cap flat holes first, as slice_model(close_holes=true) would, so a scan's missing base can "
            "be the face it stands on. Without it an open base never counts as a base."
        ),
    ] = False,
    printer_profile: Annotated[
        str | None,
        Field(
            description="Exact printer profile name, from list_printers. Give all three profiles to also slice each "
            "option once with automatic supports and report the real support filament, filament and time (slow: one "
            "slice per option, at most 6 options). Omit all three for the geometric scores alone."
        ),
    ] = None,
    print_profile: Annotated[
        str | None, Field(description="Exact print profile name, from list_profiles. All three profiles or none.")
    ] = None,
    material_profile: Annotated[
        str | None, Field(description="Exact filament profile name, from list_profiles. All three profiles or none.")
    ] = None,
) -> OrientationAdvice:
    """Suggest which way up to print a model, ranked, with what each orientation costs.

    The candidates are the model's large flat faces, its six axis directions and the orientation as stored (always
    listed, with its rank). Each is scored by the surface it puts on the bed (`base_area_mm2`, and `stable` unless that
    is under 5% of the model's silhouette from above), the surface leaning past 45 degrees that would need support
    (`overhang_area_mm2`) and the height. Pass the `rotate_x` and `rotate_y` of an option to preview_model to look
    at it and to slice_model to print it. Overhang area is a rough guide to how much support is needed, not a
    measure: orientations within about 10% of each other are effectively tied. The scores are geometry only: they know
    nothing about strength, how it looks or the material, so use them to choose, not to decide. With profiles each
    option is also sliced with supports on, and `measured` holds the real support filament, total filament and
    estimated time (or why PrusaSlicer refused it); the options are then ranked stable first, then by measured
    support, then by time, and `geometry_rank` keeps the geometric order."""
    if (printer_profile, print_profile, material_profile).count(None) not in (0, 3):
        raise ToolError("Give all three profiles (printer_profile, print_profile, material_profile) or none")
    profiles = (
        (printer_profile, print_profile, material_profile)
        if printer_profile is not None and print_profile is not None and material_profile is not None
        else None
    )
    if profiles is not None and count > _MAX_MEASURED_OPTIONS:
        raise ToolError(
            f"count is at most {_MAX_MEASURED_OPTIONS} when measuring supports, since each option is a slice"
        )
    path = _model_path(model_path)
    if read_project(path) is not None:
        raise ToolError(
            "suggest_orientation is for plain models: a PrusaSlicer project keeps its own placement, and rotations "
            "would not combine with it as these scores assume. Use preview_model on the project to try a rotation."
        )
    mesh = await _load_model_mesh(path)
    warnings: list[str] = []
    if close_holes:
        closing = await asyncio.to_thread(repair.close_holes, mesh)
        mesh = closing.mesh
        warnings.append(_closing_note(closing, "scoring"))
    elif not mesh.is_watertight:
        holes = repair.find_holes(mesh)
        closable = sum(hole.closable for hole in holes)
        if closable:
            warnings.append(
                f"The model has {closable} open hole(s) that close_holes can cap, such as a missing base. They are "
                "not counted as a base here: pass close_holes=true to score the model with them capped."
            )
    options = await asyncio.to_thread(orientation.suggest, mesh, count)
    if not any(option.stable for option in options):
        warnings.append(
            "No orientation rests on a real base (each stands on a point or an edge), so expect to need a brim or "
            "supports, or to cut the model flat."
        )
    results = [
        OrientationOption(
            rank=option.rank,
            geometry_rank=option.rank,
            rotate_x=round(option.rotate_x, 2),
            rotate_y=round(option.rotate_y, 2),
            height_mm=round(option.height_mm, 1),
            base_area_mm2=round(option.base_area_mm2, 1),
            base_share=round(option.base_share, 3),
            overhang_area_mm2=round(option.overhang_area_mm2, 1),
            overhang_share=round(option.overhang_share, 3),
            stable=option.stable,
            as_stored=option.as_stored,
        )
        for option in options
    ]
    if profiles is not None:
        for result, option in zip(results, options, strict=True):
            result.measured = await _measure_support(path, option, profiles, close_holes)
        results = _rank_by_measurement(results)
    return OrientationAdvice(surface_area_mm2=round(float(mesh.area), 1), options=results, warnings=warnings)


GcodeView = Literal["layer", "profile"]


@mcp.tool(annotations=_READ_ONLY)
def preview_gcode(
    gcode_path: Annotated[
        str, Field(description="Absolute path to a text G-code file, such as the gcode_path slice_model returns")
    ],
    view: Annotated[
        GcodeView,
        Field(
            description="layer (default): one layer's toolpaths coloured by feature, with the layer below as a faint "
            "ghost. profile: front and side silhouettes of everything printed, where a stretch of height with nothing "
            "printed shows as a blank band."
        ),
    ] = "layer",
    layer: Annotated[
        int | None,
        Field(description="Layer number for view=layer, from 1 (the first layer); default the middle layer", ge=1),
    ] = None,
    size: Annotated[int, Field(description="Picture width and height in pixels", ge=128, le=1024)] = 640,
) -> list[Image | str]:
    """Draw a picture of sliced G-code, to check what will really be printed before trusting the statistics.

    Use view=profile after slice_model: it shows at a glance if parts of the print float or whole stretches of the
    height are empty. Use view=layer to inspect one layer's perimeters, infill and supports."""
    path = _gcode_path(gcode_path)
    if view == "profile" and layer is not None:
        raise ToolError("layer is only used with view=layer")
    try:
        scan = scan_gcode(path)
        total = len(scan.layer_heights)
        number = layer if layer is not None else (total + 1) // 2
        picture = render_layer(path, scan, number, size) if view == "layer" else render_profile(path, scan, size)
    except ValueError as e:
        raise ToolError(str(e)) from e
    if view == "layer":
        caption = (
            f"{path.name}, layer {number} of {total} (z {scan.layer_heights[number - 1]:g} mm). The legend names this "
            "layer's features; the faint grey is the layer below. Every layer is drawn on the same scale."
        )
    else:
        caption = (
            f"{path.name}: front (x) and side (y) silhouettes of everything printed, {total} layers, "
            f"{max(scan.layer_heights):.1f} mm tall, on a 10 mm grid. A blank band across both panels means nothing "
            "was printed at that height."
        )
        gap_warning = layer_gap_warning(path)
        if gap_warning:
            caption += " " + gap_warning
    if scan.arcs_skipped:
        noun = "move" if scan.arcs_skipped == 1 else "moves"
        caption += f" {scan.arcs_skipped} arc {noun} (G2/G3) {'is' if scan.arcs_skipped == 1 else 'are'} not drawn."
    return [Image(data=to_png(picture), format="png"), caption]


# --------------------------------------------------------------------------- slicing


class SliceResult(BaseModel):
    gcode_path: str
    stats: GcodeStats
    settings_from: Literal["profiles", "project"]
    centered: bool
    holes_closed: int = 0
    warnings: list[str]


@mcp.tool()
async def slice_model(
    model_path: Annotated[str, Field(description="Absolute path to an STL/3MF/OBJ file")],
    printer_profile: Annotated[
        str | None,
        Field(
            description="Exact printer profile name, from list_printers. Give all three profiles, or none for a PrusaSlicer "
            "project (a .3mf saved from PrusaSlicer, see inspect_model's `project`), which is then sliced with its own saved "
            "settings (supports, brim, infill...). Giving profiles makes PrusaSlicer ignore those saved settings."
        ),
    ] = None,
    print_profile: Annotated[
        str | None, Field(description="Exact print profile name, from list_profiles. All three profiles or none.")
    ] = None,
    material_profile: Annotated[
        str | None, Field(description="Exact filament profile name, from list_profiles. All three profiles or none.")
    ] = None,
    overrides: Annotated[
        dict[str, str] | None,
        Field(
            description="Optional PrusaSlicer settings, e.g. {'fill_density': '25%', 'perimeters': '3'}. Keys are config option names (see PrusaSlicer --help-fff); options the server sets itself (profiles, output, binary_gcode, datadir, save) are rejected."
        ),
    ] = None,
    output_path: Annotated[
        str | None,
        Field(
            description="Where to write the G-code (default: <output dir>/<model>.gcode). Its folder is created if missing, but only that one level: the folder above it must already exist."
        ),
    ] = None,
    center: Annotated[
        bool | None,
        Field(
            description="Move the model to the middle of the bed. Omit it to centre plain 3MF files, which are usually "
            "modelled around the origin and otherwise fail with 'All objects are outside of the print volume', and to "
            "leave PrusaSlicer projects (which hold their own layout) and STL/OBJ (which PrusaSlicer places itself) "
            "alone. true centres any model; false keeps a plain 3MF where it is stored. The result says whether it was "
            "centred."
        ),
    ] = None,
    rotate_x: _RotateX = None,
    rotate_y: _RotateY = None,
    rotate_z: _RotateZ = None,
    scale: _Scale = None,
    scale_to_fit: _ScaleToFit = None,
    split: Annotated[
        bool,
        Field(
            description="Separate unconnected solids into their own objects, which PrusaSlicer then arranges "
            "compactly on the bed."
        ),
    ] = False,
    copies: Annotated[
        int | None,
        Field(
            ge=1,
            le=100,
            description="Total number of objects to print, arranged automatically on the bed (3 prints three of this model).",
        ),
    ] = None,
    close_holes: Annotated[
        bool,
        Field(
            description="Cap flat holes in the mesh before slicing, such as the missing base of a scan, which "
            "PrusaSlicer otherwise refuses with 'no extrusions in the first layer'. inspect_model says whether a model "
            "has such holes. Only simple, nearly flat holes are closed; bent holes, tiny gaps and open shells are left "
            "alone. PrusaSlicer slices a repaired copy in a temporary folder, so the model file is never changed. Not "
            "for PrusaSlicer projects, which would lose their saved settings."
        ),
    ] = False,
) -> SliceResult:
    """Slice a model headlessly and return the G-code path plus print time, filament use and layer count.
    Uses the given installed profiles, or, for a PrusaSlicer project 3MF with no profiles given, the settings
    saved in the project. G-code is always written as text (not .bgcode)."""
    cfg = get_config()
    model = _model_path(model_path)
    project = read_project(model)
    profiles = _chosen_profiles(printer_profile, print_profile, material_profile, project)

    if output_path:
        out = Path(output_path).expanduser().resolve()
        # A caller-supplied path may create only its last folder: reaching further is almost always a typo, and
        # silently building a tree for it hides the mistake. The configured output directory is the user's own
        # setting, so that one is created in full.
        if out.parent.exists() and not out.parent.is_dir():
            raise ToolError(f"Cannot write {out}: {out.parent} is not a folder")
        if not out.parent.is_dir() and not out.parent.parent.is_dir():
            raise ToolError(
                f"Cannot write {out}: neither its folder nor that folder's parent exists. "
                "Use an existing folder, or one level below it."
            )
        create_all_parents = False
    else:
        out = cfg.output_dir / f"{model.stem}.gcode"
        create_all_parents = True

    closing: repair.Closing | None = None
    if close_holes:
        if project is not None:
            raise ToolError(
                "close_holes cannot be used with a PrusaSlicer project: slicing a repaired copy would drop its saved "
                "settings. Repair the mesh in PrusaSlicer, or slice the model as a plain STL with profiles."
            )
        # A large scan takes seconds to load and cap, so keep that off the event loop.
        closing = await asyncio.to_thread(repair.close_holes, await _load_model_mesh(model))

    args = ["--export-gcode", "--binary-gcode=0"]
    if profiles is not None:
        printer, print_, material = profiles
        args += ["--printer-profile", printer, "--print-profile", print_, "--material-profile", material]
    for key, value in (overrides or {}).items():
        if not _OVERRIDE_KEY.match(key):
            raise ToolError(f"Invalid override key: {key!r}")
        if key in _RESERVED_OVERRIDES:
            raise ToolError(f"Override {key!r} is controlled by the server and cannot be set")
        if "\n" in value or "\r" in value:
            raise ToolError(f"Invalid override value for {key!r}")
        args.append(f"--{key.replace('_', '-')}={value}")
    args += _transform_args(rotate_x, rotate_y, rotate_z, scale, scale_to_fit, split, copies)
    bed_printer = profiles[0] if profiles is not None else (project.printer_profile if project else None)
    # Omitted means automatic: centre a plain 3MF (ADR 0013), leave projects, STL and OBJ alone.
    automatic = center is None
    centre = (model.suffix.lower() == ".3mf" and project is None) if automatic else center
    centered = False
    if centre:
        if bed_printer is None:
            raise ToolError("Cannot centre on the bed: the project does not name its printer, so pass printer_profile")
        try:
            x, y = await _bed_center(bed_printer)
            args += ["--center", f"{x:g},{y:g}"]
            centered = True
        except ToolError:
            if not automatic:
                raise  # centring was asked for, so say it cannot be done; automatic centring is best effort
    # Everything is validated by now, so a refused request leaves nothing behind on disk.
    try:
        out.parent.mkdir(parents=create_all_parents, exist_ok=True)
        scratch = tempfile.TemporaryDirectory(prefix=".prusaslicer-mcp-", dir=out.parent)
    except OSError as e:
        raise ToolError(f"Cannot create the output folder {out.parent}: {e}") from e

    # Slice into a private scratch folder and move the result into place only on success. PrusaSlicer exits 0 for
    # some refusals ("outside of the print volume"), so a file already sitting at the output path would otherwise
    # be reported as this slice's result, and a failed slice must not destroy the previous file either.
    with scratch as scratch_dir:
        scratch_out = Path(scratch_dir) / out.name
        slicing = model
        if closing is not None and closing.closed:
            slicing = Path(scratch_dir) / f"{model.stem}.stl"  # named like the model, so PrusaSlicer's messages are
            try:
                await asyncio.to_thread(closing.mesh.export, slicing)
            except OSError as e:
                raise ToolError(f"Cannot write the repaired copy of the model: {e}") from e
        result = await _run(cfg, [*args, "--output", str(scratch_out), str(slicing)])
        if result.returncode != 0 or not scratch_out.is_file():
            detail = (result.stderr.strip() or result.stdout.strip())[-2000:]
            bed_printer = profiles[0] if profiles is not None else (project.printer_profile if project else None)
            detail = await _explain_failure(detail, model, bed_printer, centered)
            raise ToolError(f"Slicing failed (exit {result.returncode}): {detail}")
        try:
            os.replace(scratch_out, out)
        except OSError as e:
            raise ToolError(f"Cannot write {out}: {e}") from e

    warnings = ([_closing_note(closing, "slicing")] if closing is not None else []) + parse_warnings(result.stdout)
    try:
        stats = parse_gcode_stats(out)
    except ValueError as e:
        raise ToolError(str(e)) from e
    # PrusaSlicer does not always say when layers are missing, and the statistics look normal either way.
    gap_warning = layer_gap_warning(out)
    if gap_warning:
        warnings.append(gap_warning)
    return SliceResult(
        gcode_path=str(out),
        stats=stats,
        settings_from="profiles" if profiles is not None else "project",
        centered=centered,
        holes_closed=closing.closed if closing is not None else 0,
        warnings=warnings,
    )


def main() -> None:
    mcp.run()
