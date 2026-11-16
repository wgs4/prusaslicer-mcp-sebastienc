import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh

from prusaslicer_mcp import cli, server
from prusaslicer_mcp.config import Config, PrusaSlicerNotFound, find_executable


@pytest.fixture
def set_home(monkeypatch):
    """Make `~` (expanduser, Path.home) mean a folder of the test's choosing, on POSIX and on Windows."""

    def apply(folder) -> None:
        monkeypatch.setenv("HOME", str(folder))  # POSIX
        monkeypatch.setenv("USERPROFILE", str(folder))  # Windows reads this one and ignores HOME

    return apply


def pytest_collection_modifyitems(items):
    """Mark every test that needs a real PrusaSlicer as `integration`, so CI can run the two kinds separately."""
    for item in items:
        if "prusaslicer" in item.fixturenames:
            item.add_marker(pytest.mark.integration)


def fff_model_lines(vendor_text: str) -> list[str]:
    """One `model:<id> = <variants>` line per FFF printer model.

    Variants are joined with `;` and spaces around that separator are removed. SLA models are left out.
    PrusaSlicer 2.9 exits with "Configuration wasn't found" when a data directory has no such lines.
    """
    lines: list[str] = []
    model_id: str | None = None
    technology: str | None = None
    variants: str | None = None

    def flush() -> None:
        nonlocal model_id, technology, variants
        if model_id and technology == "FFF" and variants:
            parts = [part.strip() for part in variants.split(";") if part.strip()]
            if parts:
                lines.append(f"model:{model_id} = {';'.join(parts)}")
        model_id = None
        technology = None
        variants = None

    for raw in vendor_text.splitlines():
        if raw.startswith("[printer_model:") and raw.endswith("]"):
            flush()
            model_id = raw[len("[printer_model:") : -1]
            continue
        if raw.startswith("["):
            flush()
            continue
        if model_id is None or "=" not in raw:
            continue
        key, value = (piece.strip() for piece in raw.split("=", 1))
        if key == "technology":
            technology = value
        elif key == "variants":
            variants = value
    flush()
    return lines


def _vendor_ini(executable: Path) -> Path | None:
    """The PrusaResearch vendor bundle shipped beside the binary, including a macOS .app bundle.

    A `prusa-slicer` on PATH is often a wrapper, so the standard application location is checked too.
    """
    resolved = executable.resolve()
    candidates = [
        resolved.parent.parent / "Resources" / "profiles" / "PrusaResearch.ini",
        resolved.parent / "resources" / "profiles" / "PrusaResearch.ini",
        Path("/Applications/PrusaSlicer.app/Contents/Resources/profiles/PrusaResearch.ini"),
        Path.home() / "Applications" / "PrusaSlicer.app" / "Contents" / "Resources" / "profiles" / "PrusaResearch.ini",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _build_synthetic_datadir(root: Path, executable: Path) -> Path | None:
    vendor = _vendor_ini(executable)
    if vendor is None:
        return None
    (root / "vendor").mkdir()
    shutil.copyfile(vendor, root / "vendor" / "PrusaResearch.ini")
    models = fff_model_lines(vendor.read_text(encoding="utf-8", errors="replace"))
    ini = root / "PrusaSlicer.ini"
    ini.write_text("[vendor:PrusaResearch]\n" + "\n".join(models) + "\n", encoding="utf-8")
    ini.chmod(0o600)
    return root


@pytest.fixture(scope="session", autouse=True)
def synthetic_datadir(tmp_path_factory):
    """Use a temporary data directory for the whole session when PrusaSlicer is installed.

    The directory holds the vendor bundle plus a synthetic PrusaSlicer.ini that enables every FFF model.
    One project test slices a file that names an MK3S MMU printer; the profile fixture names an MK4S.
    Enabling every FFF model covers both. The user's own data directory is never read or written.
    """
    try:
        executable = find_executable()
    except PrusaSlicerNotFound:
        yield None
        return
    root = _build_synthetic_datadir(tmp_path_factory.mktemp("prusaslicer-datadir"), executable)
    if root is None:
        yield None
        return
    previous = os.environ.get("PRUSASLICER_DATADIR")
    os.environ["PRUSASLICER_DATADIR"] = str(root)
    yield root
    if previous is None:
        os.environ.pop("PRUSASLICER_DATADIR", None)
    else:
        os.environ["PRUSASLICER_DATADIR"] = previous


@pytest.fixture(scope="session")
def prusaslicer(synthetic_datadir):
    try:
        executable = find_executable()
    except PrusaSlicerNotFound:
        pytest.skip("PrusaSlicer not installed")
    if synthetic_datadir is None:
        pytest.skip("PrusaSlicer vendor profiles were not found")
    return executable


@pytest.fixture
def cube_stl(tmp_path):
    """A 20 mm ASCII STL cube."""
    v = [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0), (0, 0, 20), (20, 0, 20), (20, 20, 20), (0, 20, 20)]
    f = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    lines = ["solid c"]
    for a, b, c in f:
        lines.append("facet normal 0 0 0\nouter loop")
        lines += [f"vertex {v[i][0]} {v[i][1]} {v[i][2]}" for i in (a, b, c)]
        lines.append("endloop\nendfacet")
    lines.append("endsolid c")
    p = tmp_path / "cube.stl"
    p.write_text("\n".join(lines))
    return p


PROJECT_CONFIG = """; generated by PrusaSlicer 2.9.7+macos

; bottom_solid_layers = 4
; brim_width = 5
; fill_density = 20%
; fill_pattern = gyroid
; filament_settings_id = "Prusament PLA"
; layer_height = 0.2
; perimeters = 2
; print_settings_id = 0.20mm QUALITY @MK3
; printer_settings_id = Original Prusa i3 MK3S & MK3S+ MMU2S Single
; support_material = 1
; support_material_auto = 1
; top_solid_layers = 6
; start_gcode = M104 S0 ; heater = off
"""


@pytest.fixture
def make_3mf(tmp_path):
    """Factory: a 20 mm cube saved as a 3MF. Pass `config` to make it a PrusaSlicer project that embeds it."""
    import zipfile

    import trimesh

    def make(name: str = "part.3mf", config: str | None = None, at=(90.0, 90.0, 10.0)):
        mesh = trimesh.creation.box(extents=(20, 20, 20))
        mesh.apply_translation(at)
        path = tmp_path / name
        mesh.export(path)
        if config is not None:
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("Metadata/Slic3r_PE.config", config)
        return path

    return make


@pytest.fixture
def fake_export(tmp_path, monkeypatch):
    """A fake PrusaSlicer that answers `--export-stl` by writing `.mesh` to the --output path.

    `.state` chooses what it does and says; `.calls` records every command line and `.inputs` the model file as it
    was given (a scratch copy is gone by the time a test looks).
    """
    monkeypatch.setattr(
        server,
        "_config",
        Config(executable=tmp_path / "PrusaSlicer", datadir=None, output_dir=tmp_path / "out", timeout=5),
    )
    calls: list[list[str]] = []
    inputs: list[bytes | None] = []
    state = SimpleNamespace(
        mesh=trimesh.creation.box(extents=(20, 30, 40)), writes_output=True, returncode=0, stdout="", stderr=""
    )

    async def fake_run(config, args, timeout=None):
        calls.append(args)
        given = Path(args[-1])
        inputs.append(given.read_bytes() if given.is_file() else None)
        if state.writes_output:
            state.mesh.export(Path(args[args.index("--output") + 1]))
        return cli.CliResult(state.returncode, state.stdout, state.stderr)

    monkeypatch.setattr(cli, "run", fake_run)
    return SimpleNamespace(calls=calls, inputs=inputs, state=state, tmp=tmp_path)
