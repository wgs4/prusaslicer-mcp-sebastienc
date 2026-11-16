"""slice_model's `center` option.

Plain 3MF files keep their stored coordinates, so a model modelled around the origin lands half
off the bed and PrusaSlicer refuses to slice it. `center=True` passes --center <bed middle>.
The unit tests fake the CLI. The last test slices with a real PrusaSlicer when one is installed.
"""

import json
import shutil
from pathlib import Path

import pytest
import trimesh
from mcp.server.mcpserver.exceptions import ToolError

from prusaslicer_mcp import cli, server
from prusaslicer_mcp.config import Config
from prusaslicer_mcp.preview_gcode import iter_extrusions

PRINTERS = {
    "printer_models": [
        {
            "id": "MINIIS",
            "name": "Original Prusa MINI && MINI+ Input Shaper",
            "variants": [
                {
                    "name": "0.4",
                    "printer_profiles": [
                        {
                            "name": "Mini",
                            "extruders_cnt": 1,
                            "bed": {"width": 180, "height": 180, "max_print_height": 180},
                        }
                    ],
                },
                {
                    "name": "0.4",
                    "printer_profiles": [
                        {
                            "name": "MK3S",
                            "extruders_cnt": 1,
                            "bed": {"width": 250, "height": 210, "max_print_height": 210},
                        }
                    ],
                    "user_printer_profiles": [
                        {
                            "name": "MK3S",
                            "extruders_cnt": 1,
                            "bed": {"width": 10, "height": 10, "max_print_height": 10},
                        },
                        {
                            "name": "Shop",
                            "extruders_cnt": 1,
                            "bed": {"width": 220, "height": 220, "max_print_height": 250},
                        },
                        {
                            "name": "Offset Shop",
                            "extruders_cnt": 1,
                            "bed": {
                                "width": 220,
                                "height": 220,
                                "max_print_height": 250,
                                "origin": "[110, 110]",
                            },
                        },
                    ],
                },
            ],
        },
        {
            "id": "FLAT",
            "name": "Flat",
            "user_printer_profiles": [
                {
                    "name": "Flat User",
                    "extruders_cnt": 1,
                    "bed": {"width": 100, "height": 80, "max_print_height": 40},
                }
            ],
        },
    ]
}


@pytest.fixture
def slicer(tmp_path, monkeypatch):
    """Fake CLI: answers the printer query, records slice calls, writes a stub G-code file."""
    monkeypatch.setattr(
        server,
        "_config",
        Config(executable=tmp_path / "PrusaSlicer", datadir=None, output_dir=tmp_path / "out", timeout=5),
    )
    calls: list[list[str]] = []
    state = {"stderr": ""}

    async def fake_run(config, args, timeout=None):
        if "--query-printer-models" in args:
            Path(args[args.index("-o") + 1]).write_text(json.dumps(PRINTERS))
            return cli.CliResult(1, "", "")
        calls.append(args)
        if state["stderr"]:
            return cli.CliResult(0, "", state["stderr"])
        Path(args[args.index("--output") + 1]).write_text("; layer_count = 1\n")
        return cli.CliResult(0, "", "")

    monkeypatch.setattr(cli, "run", fake_run)
    return type("Slicer", (), {"calls": calls, "state": state})


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "part.3mf"
    p.write_bytes(b"x")
    return p


def slice_args(model, printer="Mini", **extra):
    return {
        "model_path": str(model),
        "printer_profile": printer,
        "print_profile": "q",
        "material_profile": "m",
        **extra,
    }


def centre_passed_to_prusaslicer(slicer) -> str:
    args = slicer.calls[0]
    return args[args.index("--center") + 1]


async def test_center_uses_the_middle_of_a_180_mm_bed(slicer, model):
    # Given a printer profile whose bed is 180 x 180 mm and a model file
    arguments = slice_args(model, printer="Mini", center=True)

    # When the model is sliced with center on
    await server.mcp.call_tool("slice_model", arguments)

    # Then PrusaSlicer is told to centre on 90,90
    assert centre_passed_to_prusaslicer(slicer) == "90,90"


async def test_a_user_printer_preset_is_listed_once_and_centring_uses_its_bed(slicer, model):
    # Given a user preset "Shop" (220 mm bed) and a user copy of "MK3S" that must not replace the vendor bed
    names = [variant.printer_profile for printer in await server.list_printers() for variant in printer.variants]

    # When the printers are listed and a model is centred on the user preset
    await server.mcp.call_tool("slice_model", slice_args(model, printer="Shop", center=True))

    # Then the duplicated name appears once, and centring uses the user preset's own bed
    assert names.count("MK3S") == 1
    assert "Shop" in names
    assert "Flat User" in names
    assert centre_passed_to_prusaslicer(slicer) == "110,110"


def test_bed_origin_parses_the_string_prusaslicer_writes():
    # Given the origin text PrusaSlicer 2.9 puts on a bed, a numeric pair, and a bed with no origin
    # When each one is read
    # Then the offset is in millimetres and a missing origin is the corner
    assert server._bed_origin({"origin": "[110, 110]"}) == (110.0, 110.0)
    assert server._bed_origin({"origin": "[0, 0]"}) == (0.0, 0.0)
    assert server._bed_origin({"origin": [10, 20]}) == (10.0, 20.0)
    assert server._bed_origin({}) == (0.0, 0.0)


async def test_centring_subtracts_a_user_bed_origin_that_is_not_the_corner(slicer, model):
    # Given a user preset whose bed runs from -110 to 110 (PrusaSlicer reports origin "[110, 110]")
    arguments = slice_args(model, printer="Offset Shop", center=True)

    # When the model is centred on that preset
    await server.mcp.call_tool("slice_model", arguments)

    # Then the centre is the machine origin, not half the width
    assert centre_passed_to_prusaslicer(slicer) == "0,0"


async def test_centring_works_for_a_user_preset_on_a_model_without_variants(slicer, model):
    # Given a user printer preset stored on a model that has no nozzle variants
    arguments = slice_args(model, printer="Flat User", center=True)

    # When the model is centred on that preset
    await server.mcp.call_tool("slice_model", arguments)

    # Then the bed middle comes from that preset (100 x 80 mm)
    assert centre_passed_to_prusaslicer(slicer) == "50,40"


async def test_center_follows_the_bed_size_of_the_printer_profile(slicer, model):
    # Given a printer profile whose bed is 250 x 210 mm
    arguments = slice_args(model, printer="MK3S", center=True)

    # When the model is sliced with center on
    await server.mcp.call_tool("slice_model", arguments)

    # Then PrusaSlicer is told to centre on 125,105
    assert centre_passed_to_prusaslicer(slicer) == "125,105"


async def test_a_plain_3mf_is_not_moved_when_center_is_false(slicer, model):
    # Given a plain 3MF and an explicit request not to move it (the default would now centre it, ADR 0013)
    arguments = slice_args(model, center=False)

    # When the model is sliced
    await server.mcp.call_tool("slice_model", arguments)

    # Then no --center flag reaches PrusaSlicer
    assert "--center" not in slicer.calls[0]


async def test_center_with_an_unknown_printer_profile_is_refused_before_slicing(slicer, model):
    # Given a printer profile that is not among the installed ones
    arguments = slice_args(model, printer="Nope", center=True)

    # When / Then slicing is refused naming the profile, and PrusaSlicer is never asked to slice
    with pytest.raises(ToolError, match="Nope"):
        await server.mcp.call_tool("slice_model", arguments)
    assert slicer.calls == []


async def test_outside_print_volume_failure_suggests_center(slicer, model):
    # Given PrusaSlicer reports that the model is off the bed
    slicer.state["stderr"] = "All objects are outside of the print volume."

    # When / Then slicing without centring (switched off, since a plain 3MF is centred by default) fails with the remedy
    with pytest.raises(ToolError, match="center=true"):
        await server.mcp.call_tool("slice_model", slice_args(model, center=False))


def _vendor_ini(executable: Path) -> Path | None:
    """The PrusaResearch bundle next to the binary, including a macOS .app and a PATH wrapper."""
    resolved = executable.resolve()
    candidates = [
        resolved.parent.parent / "Resources" / "profiles" / "PrusaResearch.ini",
        resolved.parent / "resources" / "profiles" / "PrusaResearch.ini",
        Path("/Applications/PrusaSlicer.app/Contents/Resources/profiles/PrusaResearch.ini"),
        Path.home() / "Applications" / "PrusaSlicer.app" / "Contents" / "Resources" / "profiles" / "PrusaResearch.ini",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


async def test_a_user_preset_with_an_offset_bed_origin_slices_on_that_bed(tmp_path, prusaslicer, monkeypatch):
    # Given a user printer whose bed is centred on the machine origin, and a 20 mm cube stored elsewhere on it
    vendor = _vendor_ini(Path(prusaslicer))
    if vendor is None:
        pytest.skip("PrusaSlicer vendor profiles were not found")
    datadir = tmp_path / "datadir"
    (datadir / "vendor").mkdir(parents=True)
    shutil.copyfile(vendor, datadir / "vendor" / "PrusaResearch.ini")
    (datadir / "PrusaSlicer.ini").write_text("[vendor:PrusaResearch]\nmodel:MK4S = 0.4\n", encoding="utf-8")
    (datadir / "printer").mkdir()
    (datadir / "printer" / "Offset Bed.ini").write_text(
        "inherits = Original Prusa MK4S 0.4 nozzle\n"
        "printer_model = MK4S\n"
        "printer_variant = 0.4\n"
        "printer_settings_id = Offset Bed\n"
        "bed_shape = -110x-110,110x-110,110x110,-110x110\n"
        "max_print_height = 250\n",
        encoding="utf-8",
    )
    (datadir / "print").mkdir()
    (datadir / "print" / "Offset Print.ini").write_text(
        "inherits = 0.20mm SPEED @MK4S 0.4\n"
        "print_settings_id = Offset Print\n"
        "compatible_printers_condition =\n"
        "compatible_printers =\n",
        encoding="utf-8",
    )
    (datadir / "filament").mkdir()
    (datadir / "filament" / "Offset Filament.ini").write_text(
        "inherits = Prusament PLA @MK4S\n"
        "filament_settings_id = Offset Filament\n"
        "compatible_printers_condition =\n"
        "compatible_printers =\n"
        "compatible_prints_condition =\n"
        "compatible_prints =\n",
        encoding="utf-8",
    )
    mesh = trimesh.creation.box(extents=(20, 20, 20))
    mesh.apply_translation((50, 50, 10))
    model = tmp_path / "shifted.3mf"
    mesh.export(model)
    monkeypatch.setattr(
        server,
        "_config",
        Config(executable=Path(prusaslicer), datadir=datadir, output_dir=tmp_path / "out", timeout=180),
    )

    # When the preset's centre is read and the cube is sliced onto it
    assert await server._bed_center("Offset Bed") == (0, 0)
    result = await server.slice_model(str(model), "Offset Bed", "Offset Print", "Offset Filament", center=True)

    # Then the plastic is around the machine origin, not the corner half the width would pick
    assert result.centered is True
    assert result.stats.layer_count
    moves = [move for move in iter_extrusions(Path(result.gcode_path)) if move.feature != "Skirt/Brim"]
    assert moves
    xs = [value for move in moves for value in (move.x0, move.x1)]
    ys = [value for move in moves for value in (move.y0, move.y1)]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0, abs=1.5)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(0, abs=1.5)
    assert max(xs) - min(xs) == pytest.approx(20, abs=2)
    assert max(ys) - min(ys) == pytest.approx(20, abs=2)
