"""Unit tests for slice_model's command line, output handling and failures (CLI faked, no PrusaSlicer).

Only `cli.run` is faked, so argument building, validation, output paths, warning extraction and
failure reporting are the real code under test.
"""

import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from conftest import PROJECT_CONFIG
from mcp.server.mcpserver.exceptions import ToolError

from prusaslicer_mcp import cli, server
from prusaslicer_mcp.config import Config

FOOTER = """;LAYER_CHANGE
; filament used [mm] = 100.5
; total filament used [g] = 0.31
; estimated printing time (normal mode) = 12m 5s
"""

PROFILES = ("Mini", "0.20mm SPEED", "PLA")
NO_PROFILES = (None, None, None)
PROFILE_FLAGS = ("--printer-profile", "--print-profile", "--material-profile")
MK3S = "Original Prusa i3 MK3S & MK3S+ MMU2S Single"
PRINTERS = {
    "printer_models": [
        {
            "id": "MK3S",
            "name": "MK3S",
            "variants": [
                {
                    "name": 0.4,
                    "printer_profiles": [{"name": MK3S, "bed": {"width": 250, "height": 210, "max_print_height": 210}}],
                }
            ],
        }
    ]
}


@pytest.fixture
def fake_prusaslicer(tmp_path, monkeypatch):
    """A fake slicer. `.state` chooses what it writes and says; `.calls` records every command line."""
    monkeypatch.setattr(
        server,
        "_config",
        Config(executable=tmp_path / "PrusaSlicer", datadir=None, output_dir=tmp_path / "out", timeout=5),
    )
    calls: list[list[str]] = []
    inputs: list[bytes | None] = []  # the model file as PrusaSlicer was given it (scratch copies are gone afterwards)
    state = SimpleNamespace(gcode=FOOTER.encode(), writes_output=True, returncode=0, stdout="", stderr="")

    async def fake_run(config, args, timeout=None):
        if "--query-printer-models" in args:
            Path(args[args.index("-o") + 1]).write_text(json.dumps(PRINTERS))
            return cli.CliResult(1, "", "")
        calls.append(args)
        given = Path(args[-1])
        inputs.append(given.read_bytes() if given.is_file() else None)
        if state.writes_output:
            Path(args[args.index("--output") + 1]).write_bytes(state.gcode)
        return cli.CliResult(state.returncode, state.stdout, state.stderr)

    monkeypatch.setattr(cli, "run", fake_run)
    return SimpleNamespace(calls=calls, inputs=inputs, state=state, out_dir=tmp_path / "out", tmp=tmp_path)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "part.stl"
    path.write_text("solid x\nendsolid x\n")
    return path


async def slice_it(model, profiles=PROFILES, **extra):
    return await server.slice_model(str(model), *profiles, **extra)


# --------------------------------------------------------------------------- the command line


async def test_command_line_asks_for_text_gcode_with_the_given_profiles(fake_prusaslicer, model):
    # Given a model and three profile names

    # When it is sliced
    await slice_it(model)

    # Then PrusaSlicer is asked for text G-code with exactly those profiles, and the model comes last
    args = fake_prusaslicer.calls[0]
    assert args[:2] == ["--export-gcode", "--binary-gcode=0"]
    assert args[2:8] == ["--printer-profile", "Mini", "--print-profile", "0.20mm SPEED", "--material-profile", "PLA"]
    assert args[-3] == "--output"
    assert args[-1] == str(model.resolve())


async def test_overrides_become_hyphenated_equals_flags(fake_prusaslicer, model):
    # Given two setting overrides written with underscores
    overrides = {"fill_density": "25%", "top_solid_layers": "5"}

    # When the model is sliced
    await slice_it(model, overrides=overrides)

    # Then each is passed as one --name=value argument between the profiles and the output
    args = fake_prusaslicer.calls[0]
    assert args[8:10] == ["--fill-density=25%", "--top-solid-layers=5"]
    assert args[10] == "--output"


async def test_override_values_are_passed_verbatim_as_one_argument(fake_prusaslicer, model):
    # Given a value full of characters a shell would interpret
    value = "a b; echo $(whoami) *"

    # When the model is sliced
    await slice_it(model, overrides={"start_gcode": value})

    # Then it reaches PrusaSlicer unchanged, as a single argument
    assert f"--start-gcode={value}" in fake_prusaslicer.calls[0]


async def test_no_overrides_add_no_extra_arguments(fake_prusaslicer, model):
    # Given no overrides and no centring

    # When the model is sliced
    await slice_it(model)

    # Then the command line is just the fixed part plus output and model
    assert len(fake_prusaslicer.calls[0]) == 8 + 3


@pytest.mark.parametrize("value", ["a\nb", "a\rb"], ids=["newline", "carriage-return"])
async def test_override_value_with_a_line_break_is_refused(fake_prusaslicer, model, value):
    # Given an override value containing a line break
    overrides = {"start_gcode": value}

    # When / Then slicing is refused and PrusaSlicer is never started
    with pytest.raises(ToolError, match="Invalid override value for 'start_gcode'"):
        await slice_it(model, overrides=overrides)
    assert fake_prusaslicer.calls == []


@pytest.mark.parametrize("key", ["Bad", "1x", "a b", "a-b", "", "--evil", "x=y"], ids=repr)
async def test_malformed_override_key_is_refused(fake_prusaslicer, model, key):
    # Given an override key that is not a lower-case option name
    overrides = {key: "1"}

    # When / Then slicing is refused and PrusaSlicer is never started
    with pytest.raises(ToolError, match="Invalid override key"):
        await slice_it(model, overrides=overrides)
    assert fake_prusaslicer.calls == []


@pytest.mark.parametrize(
    "key",
    [
        "binary_gcode",
        "export_gcode",
        "output",
        "datadir",
        "save",
        "post_process",
        "printer_profile",
        "print_profile",
        "material_profile",
    ],
)
async def test_override_cannot_replace_a_setting_the_server_controls(fake_prusaslicer, model, key):
    # Given an override for a flag the server sets itself, or one that writes into the user's PrusaSlicer data
    overrides = {key: "1"}

    # When / Then slicing is refused and PrusaSlicer is never started
    with pytest.raises(ToolError, match="controlled by the server"):
        await slice_it(model, overrides=overrides)
    assert fake_prusaslicer.calls == []


# --------------------------------------------------------------------------- where the G-code goes


async def test_gcode_defaults_to_the_output_directory_named_after_the_model(fake_prusaslicer, model):
    # Given no output path and an output directory that does not exist yet

    # When the model is sliced
    result = await slice_it(model)

    # Then the G-code is <output dir>/<model name>.gcode and the directory was created
    assert result.gcode_path == str(fake_prusaslicer.out_dir / "part.gcode")
    assert (fake_prusaslicer.out_dir / "part.gcode").is_file()


async def test_explicit_output_path_is_used_and_its_last_folder_is_created(fake_prusaslicer, model):
    # Given an output path whose folder does not exist yet, inside a folder that does
    target = fake_prusaslicer.tmp / "results" / "result.gcode"

    # When the model is sliced
    result = await slice_it(model, output_path=str(target))

    # Then the G-code is written exactly there, and that one folder was created for it
    assert result.gcode_path == str(target)
    assert target.is_file()


async def test_tilde_in_output_path_is_expanded(fake_prusaslicer, model, set_home):
    # Given an output path written relative to a home directory that exists
    (fake_prusaslicer.tmp / "home").mkdir()
    set_home(fake_prusaslicer.tmp / "home")

    # When the model is sliced
    result = await slice_it(model, output_path="~/gcode/result.gcode")

    # Then it lands under the home directory
    assert result.gcode_path == str(fake_prusaslicer.tmp / "home" / "gcode" / "result.gcode")


@pytest.mark.parametrize(
    "extra",
    [
        {"overrides": {"Bad": "1"}},
        {"overrides": {"binary_gcode": "1"}},
        {"overrides": {"a": "x\ny"}},
        {"profiles": NO_PROFILES},
        {"profiles": ("Mini", None, None)},
    ],
    ids=["bad-key", "reserved-key", "bad-value", "no-profiles-for-an-stl", "partial-profiles"],
)
async def test_a_refused_request_creates_no_folders(fake_prusaslicer, model, extra):
    # Given a request that will be refused, with an output path in folders that do not exist
    target = fake_prusaslicer.tmp / "should-not-exist" / "result.gcode"

    # When / Then it is refused
    with pytest.raises(ToolError):
        await slice_it(model, output_path=str(target), **extra)

    # And nothing was created on disk for it
    assert not target.parent.exists()


async def test_centring_on_an_unknown_printer_creates_no_folders(fake_prusaslicer, model, monkeypatch):
    # Given centring is requested but the printer's bed cannot be found
    async def no_such_bed(printer_profile):
        raise ToolError("Cannot centre on the bed")

    monkeypatch.setattr(server, "_bed_center", no_such_bed)
    target = fake_prusaslicer.tmp / "should-not-exist" / "result.gcode"

    # When / Then it is refused
    with pytest.raises(ToolError, match="Cannot centre"):
        await slice_it(model, output_path=str(target), center=True)

    # And nothing was created on disk for it
    assert not target.parent.exists()


# --------------------------------------------------------------------------- results and failures


async def test_statistics_come_from_the_gcode_footer(fake_prusaslicer, model):
    # Given PrusaSlicer writes G-code with a footer

    # When the model is sliced
    result = await slice_it(model)

    # Then the statistics are read from it
    assert result.stats.estimated_time == "12m 5s"
    assert result.stats.filament_used_g == 0.31
    assert result.stats.layer_count == 1


async def test_each_single_line_slicer_warning_is_returned_trimmed(fake_prusaslicer, model):
    # Given PrusaSlicer prints progress, a warning and a differently-cased warning
    fake_prusaslicer.state.stdout = "10 => Processing\n  Warning: thin walls  \n20 => Done\nWARNING: low bridge\n"

    # When the model is sliced
    result = await slice_it(model)

    # Then only the warning lines are returned, trimmed
    assert result.warnings == ["Warning: thin walls", "WARNING: low bridge"]


async def test_nonzero_exit_fails_even_if_an_old_output_file_exists(fake_prusaslicer, model):
    # Given a G-code file left over from an earlier run, and PrusaSlicer now exits with 2
    fake_prusaslicer.out_dir.mkdir()
    (fake_prusaslicer.out_dir / "part.gcode").write_bytes(FOOTER.encode())
    fake_prusaslicer.state.writes_output = False
    fake_prusaslicer.state.returncode = 2
    fake_prusaslicer.state.stderr = "boom"

    # When / Then the stale file is not mistaken for a result
    with pytest.raises(ToolError, match=r"Slicing failed \(exit 2\): boom"):
        await slice_it(model)


async def test_clean_exit_without_an_output_file_is_a_failure(fake_prusaslicer, model):
    # Given PrusaSlicer exits 0 but writes nothing
    fake_prusaslicer.state.writes_output = False
    fake_prusaslicer.state.stdout = "nothing to do"

    # When / Then it is reported as failed, using stdout as the explanation
    with pytest.raises(ToolError, match=r"Slicing failed \(exit 0\): nothing to do"):
        await slice_it(model)


async def test_only_the_tail_of_a_very_long_error_is_kept(fake_prusaslicer, model):
    # Given PrusaSlicer fails with 5000 characters of output ending in the real reason
    fake_prusaslicer.state.returncode = 1
    fake_prusaslicer.state.writes_output = False
    fake_prusaslicer.state.stderr = "HEAD" + "x" * 5000 + "the real reason"

    # When / Then the message ends with the reason and drops the start
    with pytest.raises(ToolError) as error:
        await slice_it(model)
    assert str(error.value).endswith("the real reason")
    assert "HEAD" not in str(error.value)


async def test_binary_gcode_output_is_reported_as_a_readable_failure(fake_prusaslicer, model):
    # Given PrusaSlicer writes binary G-code anyway
    fake_prusaslicer.state.gcode = b"GCDE" + b"\0" * 32

    # When / Then the statistics cannot be read, and the tool says why
    with pytest.raises(ToolError, match="Binary G-code"):
        await slice_it(model)


# --------------------------------------------------------------------------- PrusaSlicer projects (ADR 0007)


async def test_a_project_is_sliced_with_its_own_settings_when_no_profiles_are_given(fake_prusaslicer, make_3mf):
    # Given a 3MF saved from PrusaSlicer, with its own settings embedded
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced without profiles
    result = await slice_it(project, profiles=NO_PROFILES)

    # Then PrusaSlicer is given no profile flags, so the project's settings apply, and text G-code is still forced
    args = fake_prusaslicer.calls[0]
    assert not any(flag in args for flag in PROFILE_FLAGS)
    assert args[:2] == ["--export-gcode", "--binary-gcode=0"]
    assert result.settings_from == "project"


async def test_profiles_given_for_a_project_replace_its_settings(fake_prusaslicer, make_3mf):
    # Given a project and explicit profiles
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced with them
    result = await slice_it(project)

    # Then the profile flags are passed, which makes PrusaSlicer use the profiles and not the project's settings
    args = fake_prusaslicer.calls[0]
    assert all(flag in args for flag in PROFILE_FLAGS)
    assert result.settings_from == "profiles"


async def test_a_model_that_is_not_a_project_reports_that_profiles_were_used(fake_prusaslicer, model):
    # Given an STL and profiles

    # When it is sliced
    result = await slice_it(model)

    # Then the result says profiles were used
    assert result.settings_from == "profiles"


async def test_overrides_still_apply_on_top_of_a_projects_own_settings(fake_prusaslicer, make_3mf):
    # Given a project sliced without profiles, with one override
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced
    await slice_it(project, profiles=NO_PROFILES, overrides={"fill_density": "10%"})

    # Then the override is passed, and no profile flags are
    args = fake_prusaslicer.calls[0]
    assert "--fill-density=10%" in args
    assert not any(flag in args for flag in PROFILE_FLAGS)


async def test_centring_a_project_uses_the_bed_of_the_printer_stored_in_it(fake_prusaslicer, make_3mf):
    # Given a project that names a 250 x 210 mm printer, and no printer_profile argument
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced without profiles but centred
    await slice_it(project, profiles=NO_PROFILES, center=True)

    # Then PrusaSlicer is told to centre on that bed's middle
    args = fake_prusaslicer.calls[0]
    assert args[args.index("--center") + 1] == "125,105"


async def test_centring_a_project_that_does_not_name_its_printer_asks_for_one(fake_prusaslicer, make_3mf):
    # Given a project whose configuration has no printer profile name
    project = make_3mf(config="; fill_density = 20%\n")

    # When / Then centring it without a printer_profile is refused with advice
    with pytest.raises(ToolError, match="does not name its printer"):
        await slice_it(project, profiles=NO_PROFILES, center=True)
    assert fake_prusaslicer.calls == []


@pytest.mark.parametrize("kind", ["stl", "plain-3mf"])
async def test_a_model_that_is_not_a_project_needs_profiles(fake_prusaslicer, model, make_3mf, kind):
    # Given a model without embedded PrusaSlicer settings
    target = model if kind == "stl" else make_3mf(config=None)

    # When / Then slicing it without profiles is refused, saying what is needed
    with pytest.raises(ToolError, match="required unless the model is a PrusaSlicer project"):
        await slice_it(target, profiles=NO_PROFILES)
    assert fake_prusaslicer.calls == []


@pytest.mark.parametrize(
    "profiles",
    [("Mini", None, None), (None, "0.20mm SPEED", None), (None, None, "PLA"), ("Mini", "0.20mm SPEED", None)],
    ids=["printer-only", "print-only", "material-only", "no-material"],
)
async def test_some_but_not_all_profiles_is_refused_even_for_a_project(fake_prusaslicer, make_3mf, profiles):
    # Given a project and an incomplete set of profiles
    project = make_3mf(config=PROJECT_CONFIG)

    # When / Then slicing is refused, because a partial set would mix profile values with the project's
    with pytest.raises(ToolError, match="all three"):
        await slice_it(project, profiles=profiles)
    assert fake_prusaslicer.calls == []


async def test_empty_profile_names_count_as_given_profiles_and_are_left_for_prusaslicer_to_reject(
    fake_prusaslicer, make_3mf
):
    # Given a project and profile names that are present but empty
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced with them
    result = await slice_it(project, profiles=("", "", ""))

    # Then they are passed on as profiles (not silently replaced by the project's settings)
    assert result.settings_from == "profiles"
    assert fake_prusaslicer.calls[0][fake_prusaslicer.calls[0].index("--printer-profile") + 1] == ""


# --------------------------------------------------------------------------- how far an output path may reach


async def test_an_output_path_two_folders_deep_is_refused_because_it_is_probably_a_typo(fake_prusaslicer, model):
    # Given an output path where neither its folder nor that folder's parent exists
    target = fake_prusaslicer.tmp / "typo" / "also-typo" / "result.gcode"

    # When / Then it is refused with a message that names the path, and PrusaSlicer is never started
    with pytest.raises(ToolError, match=r"Cannot write .*result\.gcode.*neither"):
        await slice_it(model, output_path=str(target))
    assert fake_prusaslicer.calls == []


async def test_a_refused_deep_output_path_creates_no_folders(fake_prusaslicer, model):
    # Given the same too-deep output path
    target = fake_prusaslicer.tmp / "typo" / "also-typo" / "result.gcode"

    # When it is refused
    with pytest.raises(ToolError):
        await slice_it(model, output_path=str(target))

    # Then nothing was created on disk
    assert not (fake_prusaslicer.tmp / "typo").exists()


async def test_an_output_folder_that_is_a_file_is_refused(fake_prusaslicer, model):
    # Given a path whose "folder" is actually a regular file
    blocker = fake_prusaslicer.tmp / "not-a-folder"
    blocker.write_text("x")

    # When / Then slicing is refused readably, and PrusaSlicer is never started
    with pytest.raises(ToolError, match="not a folder"):
        await slice_it(model, output_path=str(blocker / "result.gcode"))
    assert fake_prusaslicer.calls == []


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores folder permissions")
@pytest.mark.skipif(sys.platform == "win32", reason="chmod does not make a Windows folder read-only")
async def test_a_folder_that_cannot_be_created_is_a_readable_failure(fake_prusaslicer, model):
    # Given a read-only folder, so creating a folder inside it is not allowed
    locked = fake_prusaslicer.tmp / "locked"
    locked.mkdir()
    locked.chmod(0o500)

    # When / Then slicing fails with a message naming the folder, not an opaque crash
    try:
        with pytest.raises(ToolError, match="Cannot create the output folder"):
            await slice_it(model, output_path=str(locked / "new" / "result.gcode"))
    finally:
        locked.chmod(0o700)
    assert fake_prusaslicer.calls == []


async def test_the_configured_output_directory_is_created_however_deep_it_is(fake_prusaslicer, model, monkeypatch):
    # Given an output directory set in the configuration that is several levels below an existing folder
    deep = fake_prusaslicer.tmp / "a" / "b" / "c"
    monkeypatch.setattr(
        server,
        "_config",
        server.get_config().__class__(
            executable=server.get_config().executable, datadir=None, output_dir=deep, timeout=5
        ),
    )

    # When the model is sliced without an output path
    result = await slice_it(model)

    # Then the whole configured directory is created, because that path is the user's own setting
    assert result.gcode_path == str(deep / "part.gcode")


async def test_a_multi_line_slicer_warning_is_returned_whole(fake_prusaslicer, model):
    # Given PrusaSlicer reports gaps in the model as a warning followed by detail and advice
    fake_prusaslicer.state.stdout = (
        "90 => Exporting G-code to /tmp/x.gcode\n"
        "print warning: Empty layer between 0.4 and 9.\n"
        "Empty layer between 9.2 and 11.4.\n"
        "\n"
        "Make sure the object is printable. Try to repair the model.\n"
        "Slicing result exported to /tmp/x.gcode\n"
    )

    # When the model is sliced
    result = await slice_it(model)

    # Then the caller gets one warning with every gap and the advice
    assert len(result.warnings) == 1
    assert "between 9.2 and 11.4" in result.warnings[0]
    assert "Try to repair the model." in result.warnings[0]


# --------------------------------------------------------------------------- layers that were never printed


def gcode_with_layers(*heights: float) -> bytes:
    layers = "".join(f";LAYER_CHANGE\n;Z:{h}\n;HEIGHT:0.2\nG1 X1 Y1\n" for h in heights)
    return (layers + FOOTER).encode()


async def test_a_slice_with_an_empty_stretch_is_flagged_even_if_prusaslicer_said_nothing(fake_prusaslicer, model):
    # Given G-code whose layers jump from 0.4 mm to 9 mm, and a PrusaSlicer that printed no warning
    fake_prusaslicer.state.gcode = gcode_with_layers(0.2, 0.4, 9.0, 9.2, 9.4)

    # When the model is sliced
    result = await slice_it(model)

    # Then the caller is warned that part of the height is empty
    assert len(result.warnings) == 1
    assert "Nothing is printed across" in result.warnings[0]
    assert "between z 0.4 and 9 mm" in result.warnings[0]


async def test_the_empty_stretch_warning_comes_after_prusaslicers_own(fake_prusaslicer, model):
    # Given both PrusaSlicer's warning and a gap in the layers
    fake_prusaslicer.state.gcode = gcode_with_layers(0.2, 0.4, 9.0, 9.2, 9.4)
    fake_prusaslicer.state.stdout = "print warning: Empty layer between 0.4 and 9.\n"

    # When the model is sliced
    result = await slice_it(model)

    # Then PrusaSlicer's warning comes first, followed by ours
    assert result.warnings[0].startswith("print warning: Empty layer")
    assert result.warnings[1].startswith("Nothing is printed across")


async def test_a_slice_without_gaps_adds_no_empty_stretch_warning(fake_prusaslicer, model):
    # Given G-code with evenly spaced layers
    fake_prusaslicer.state.gcode = gcode_with_layers(0.2, 0.4, 0.6, 0.8, 1.0)

    # When the model is sliced
    result = await slice_it(model)

    # Then there are no warnings
    assert result.warnings == []


# --------------------------------------------------------------------------- transforms (ADR 0012)


async def sliced_with(fake_prusaslicer, model, **extra) -> list[str]:
    await slice_it(model, **extra)
    return fake_prusaslicer.calls[0]  # slice_it accepts a `profiles=` override too


async def test_no_transform_flags_are_passed_unless_asked_for(fake_prusaslicer, model):
    # Given no transform parameters

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model)

    # Then none of PrusaSlicer's transform flags appear
    transform_flags = ("--rotate", "--rotate-x", "--rotate-y", "--scale", "--scale-to-fit", "--split", "--duplicate")
    assert not any(flag in args for flag in transform_flags)


@pytest.mark.parametrize(
    ("parameter", "flag"),
    [("rotate_x", "--rotate-x"), ("rotate_y", "--rotate-y"), ("rotate_z", "--rotate")],
    ids=["x", "y", "z-is-plain-rotate"],
)
@pytest.mark.parametrize(("degrees", "text"), [(90, "90"), (-90, "-90"), (12.5, "12.5"), (0, "0")])
async def test_each_rotation_becomes_its_prusaslicer_flag_in_degrees(
    fake_prusaslicer, model, parameter, flag, degrees, text
):
    # Given a rotation about one axis
    extra = {parameter: degrees}

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model, **extra)

    # Then PrusaSlicer is given that flag with the angle
    assert args[args.index(flag) + 1] == text


async def test_all_three_rotations_can_be_combined(fake_prusaslicer, model):
    # Given rotations about all three axes

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model, rotate_x=90, rotate_y=45, rotate_z=10)

    # Then all three flags are passed
    assert args[args.index("--rotate-x") + 1] == "90"
    assert args[args.index("--rotate-y") + 1] == "45"
    assert args[args.index("--rotate") + 1] == "10"


@pytest.mark.parametrize("parameter", ["rotate_x", "rotate_y", "rotate_z"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "infinity"])
async def test_a_rotation_that_is_not_a_finite_number_is_refused(fake_prusaslicer, model, parameter, value):
    # Given an angle that is not a number

    # When / Then slicing is refused and PrusaSlicer is never started
    with pytest.raises(ToolError, match="finite"):
        await slice_it(model, **{parameter: value})
    assert fake_prusaslicer.calls == []


@pytest.mark.parametrize(("factor", "text"), [(2, "2"), (0.5, "0.5"), (1.25, "1.25")])
async def test_scale_is_passed_as_a_factor(fake_prusaslicer, model, factor, text):
    # Given a scale factor, where 0.5 means half size

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model, scale=factor)

    # Then PrusaSlicer is given the plain factor, never a percentage, which it would read differently
    assert args[args.index("--scale") + 1] == text


@pytest.mark.parametrize("factor", [0, -1, -0.5])
async def test_a_scale_that_is_not_positive_is_refused(fake_prusaslicer, model, factor):
    # Given a zero or negative scale

    # When / Then slicing is refused and PrusaSlicer is never started
    with pytest.raises(ToolError, match="greater than 0"):
        await slice_it(model, scale=factor)
    assert fake_prusaslicer.calls == []


async def test_scale_to_fit_passes_the_box_in_millimetres(fake_prusaslicer, model):
    # Given a box to fit the model into

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model, scale_to_fit=[180, 180, 150.5])

    # Then PrusaSlicer is given the three sizes
    assert args[args.index("--scale-to-fit") + 1] == "180,180,150.5"


@pytest.mark.parametrize("box", [[10, 10], [10, 10, 10, 10], []], ids=["two", "four", "none"])
async def test_scale_to_fit_needs_exactly_three_sizes(fake_prusaslicer, model, box):
    # Given a box with the wrong number of sizes

    # When / Then slicing is refused and PrusaSlicer is never started
    with pytest.raises(ToolError, match="3"):
        await slice_it(model, scale_to_fit=box)
    assert fake_prusaslicer.calls == []


@pytest.mark.parametrize("box", [[0, 10, 10], [10, -5, 10], [10, 10, 0]], ids=["x", "y", "z"])
async def test_scale_to_fit_sizes_must_be_positive(fake_prusaslicer, model, box):
    # Given a box with a zero or negative size

    # When / Then slicing is refused naming the problem
    with pytest.raises(ToolError, match="positive"):
        await slice_it(model, scale_to_fit=box)
    assert fake_prusaslicer.calls == []


async def test_scale_and_scale_to_fit_cannot_be_combined(fake_prusaslicer, model):
    # Given both ways of resizing at once

    # When / Then slicing is refused, saying to choose one
    with pytest.raises(ToolError, match="scale or scale_to_fit"):
        await slice_it(model, scale=2, scale_to_fit=[100, 100, 100])
    assert fake_prusaslicer.calls == []


async def test_split_asks_prusaslicer_to_separate_unconnected_parts(fake_prusaslicer, model):
    # Given split is on

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model, split=True)

    # Then the flag is passed
    assert "--split" in args


@pytest.mark.parametrize(("count", "text"), [(1, "1"), (3, "3"), (100, "100")])
async def test_copies_become_the_number_of_objects_on_the_plate(fake_prusaslicer, model, count, text):
    # Given a number of copies, which PrusaSlicer treats as the total number of objects

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model, copies=count)

    # Then PrusaSlicer is given that number
    assert args[args.index("--duplicate") + 1] == text


@pytest.mark.parametrize("count", [0, -2, 101])
async def test_a_number_of_copies_outside_one_to_a_hundred_is_refused(fake_prusaslicer, model, count):
    # Given a number of copies that makes no sense

    # When / Then slicing is refused and PrusaSlicer is never started
    with pytest.raises(ToolError, match=r"1|100"):
        await slice_it(model, copies=count)
    assert fake_prusaslicer.calls == []


async def test_transforms_apply_to_a_project_sliced_with_its_own_settings(fake_prusaslicer, make_3mf):
    # Given a PrusaSlicer project and no profiles
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced rotated and scaled
    await slice_it(project, profiles=NO_PROFILES, rotate_x=90, scale=0.5)

    # Then the transforms are passed and still no profile flags are
    args = fake_prusaslicer.calls[0]
    assert "--rotate-x" in args
    assert "--scale" in args
    assert not any(flag in args for flag in PROFILE_FLAGS)


async def test_transforms_work_together_with_centring(fake_prusaslicer, model):
    # Given a rotation and centring on the bed of a printer whose bed size is known

    # When the model is sliced
    args = await sliced_with(fake_prusaslicer, model, profiles=(MK3S, "q", "m"), rotate_x=90, center=True)

    # Then both are passed
    assert "--rotate-x" in args
    assert "--center" in args


@pytest.mark.parametrize(
    "extra",
    [{"scale": 0}, {"scale_to_fit": [1, 2]}, {"copies": 0}, {"scale": 2, "scale_to_fit": [1, 1, 1]}],
    ids=["scale", "fit-box", "copies", "scale-and-fit"],
)
async def test_a_refused_transform_creates_no_folders(fake_prusaslicer, model, extra):
    # Given a transform request that will be refused, with an output path in folders that do not exist
    target = fake_prusaslicer.tmp / "should-not-exist" / "result.gcode"

    # When / Then it is refused
    with pytest.raises(ToolError):
        await slice_it(model, output_path=str(target), **extra)

    # And nothing was created on disk for it
    assert not target.parent.exists()


async def test_the_model_stays_last_on_the_command_line_whatever_transforms_are_asked_for(fake_prusaslicer, model):
    # Given every transform at once

    # When the model is sliced
    args = await sliced_with(
        fake_prusaslicer, model, rotate_x=90, rotate_y=1, rotate_z=2, scale=0.5, split=True, copies=2, center=False
    )

    # Then the command still ends with --output <file> <model>, with every transform flag before it
    assert args[-3] == "--output"
    assert args[-1] == str(model.resolve())
    for flag in ("--rotate-x", "--rotate-y", "--rotate", "--scale", "--split", "--duplicate"):
        assert args.index(flag) < len(args) - 3


# --------------------------------------------------------------------------- an old file must never pass for a new slice


def leftovers(folder) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


async def test_a_refused_slice_is_not_mistaken_for_success_because_an_old_file_is_there(fake_prusaslicer, model):
    # Given a G-code file left at the output path by an earlier slice, and PrusaSlicer now refusing with exit code 0
    # (which it does for "outside of the print volume" and similar) without writing anything
    fake_prusaslicer.out_dir.mkdir()
    old = fake_prusaslicer.out_dir / "part.gcode"
    old.write_bytes(FOOTER.encode())
    fake_prusaslicer.state.writes_output = False
    fake_prusaslicer.state.returncode = 0
    fake_prusaslicer.state.stdout = "All objects are outside of the print volume."

    # When / Then the slice is reported as failed, with PrusaSlicer's reason
    with pytest.raises(ToolError, match="outside of the print volume"):
        await slice_it(model)

    # And the earlier file is left exactly as it was
    assert old.read_bytes() == FOOTER.encode()


async def test_a_successful_slice_replaces_an_older_file_at_the_output_path(fake_prusaslicer, model):
    # Given an old G-code file at the output path, and a new slice with different statistics
    fake_prusaslicer.out_dir.mkdir()
    old = fake_prusaslicer.out_dir / "part.gcode"
    old.write_text("; old file\n; estimated printing time (normal mode) = 9h 9m 9s\n")
    fake_prusaslicer.state.gcode = FOOTER.encode()

    # When the model is sliced
    result = await slice_it(model)

    # Then the file now holds the new slice and the result reports the new statistics
    assert old.read_bytes() == FOOTER.encode()
    assert result.stats.estimated_time == "12m 5s"


async def test_no_working_files_are_left_in_the_output_folder_after_a_slice(fake_prusaslicer, model):
    # Given a fresh output folder

    # When a slice succeeds
    await slice_it(model)

    # Then the folder holds the G-code and nothing else
    assert leftovers(fake_prusaslicer.out_dir) == ["part.gcode"]


async def test_no_working_files_are_left_in_the_output_folder_after_a_failure(fake_prusaslicer, model):
    # Given a slice that fails
    fake_prusaslicer.state.returncode = 1
    fake_prusaslicer.state.writes_output = False
    fake_prusaslicer.state.stderr = "boom"

    # When it is attempted
    with pytest.raises(ToolError):
        await slice_it(model)

    # Then the output folder holds nothing
    assert leftovers(fake_prusaslicer.out_dir) == []


# --------------------------------------------------------------------------- saying how to fix a refused slice


@pytest.fixture
def stl_of(tmp_path):
    """Factory: a box STL of the given size, so the model's real size is known to the server."""
    import trimesh

    def make(x: float, y: float, z: float):
        path = tmp_path / "sized.stl"
        trimesh.creation.box(extents=(x, y, z)).export(path)
        return path

    return make


def refuse(fake_prusaslicer, message: str, returncode: int = 0) -> None:
    """Make the fake PrusaSlicer refuse with this message (exit 0 for the 'outside the print volume' kind)."""
    fake_prusaslicer.state.writes_output = False
    fake_prusaslicer.state.returncode = returncode
    fake_prusaslicer.state.stdout = message


OUTSIDE = "All objects are outside of the print volume."


async def refusal(model, **extra) -> str:
    with pytest.raises(ToolError) as error:
        await slice_it(model, profiles=(MK3S, "q", "m"), **extra)
    return str(error.value)


async def test_a_model_larger_than_the_bed_is_told_so_with_both_sizes(fake_prusaslicer, stl_of):
    # Given a model 300 mm long on a 250 x 210 x 210 mm bed, and PrusaSlicer refusing it
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced
    message = await refusal(stl_of(300, 20, 20))

    # Then the message gives both sizes and offers scale_to_fit with the bed's size, not a pointless recentre
    assert "300.0 x 20.0 x 20.0 mm" in message
    assert "250 x 210 x 210 mm" in message
    assert "scale_to_fit=[250, 210, 210]" in message
    assert "center=true" not in message


async def test_a_model_that_fits_across_the_diagonal_is_told_to_rotate(fake_prusaslicer, stl_of):
    # Given a model 251 mm long, a hair too long for the 250 mm side but short enough to lie across the diagonal
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced
    message = await refusal(stl_of(251, 20, 20))

    # Then rotate_z is suggested as well as scale_to_fit
    assert "rotate_z=45" in message
    assert "scale_to_fit" in message


async def test_a_model_too_long_even_for_the_diagonal_is_not_told_to_rotate(fake_prusaslicer, stl_of):
    # Given a model 400 mm long, which does not fit even across the bed's diagonal
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced
    message = await refusal(stl_of(400, 20, 20))

    # Then only scaling is suggested
    assert "scale_to_fit" in message
    assert "rotate_z" not in message


async def test_a_model_too_tall_for_the_printer_is_not_told_to_rotate_flat(fake_prusaslicer, stl_of):
    # Given a model 300 mm tall on a printer that prints 210 mm high
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced
    message = await refusal(stl_of(20, 20, 300))

    # Then it says so and offers scale_to_fit, but not rotate_z, which cannot help
    assert "20.0 x 20.0 x 300.0 mm" in message
    assert "scale_to_fit" in message
    assert "rotate_z" not in message


async def test_a_model_that_fits_the_bed_is_told_to_centre_it(fake_prusaslicer, stl_of):
    # Given a small model that PrusaSlicer still refuses as outside the volume (it sits off the bed)
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced without centring
    message = await refusal(stl_of(20, 20, 20), center=False)

    # Then the advice is the original one: centre it
    assert "center=true" in message
    assert "scale_to_fit" not in message


async def test_no_centre_advice_when_it_is_already_centred_and_fits(fake_prusaslicer, stl_of):
    # Given a small model that is refused even though centring was asked for
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced with centring on
    message = await refusal(stl_of(20, 20, 20), center=True)

    # Then no hint is invented: the message is PrusaSlicer's own
    assert message.endswith(OUTSIDE)


async def test_when_the_bed_is_unknown_the_centre_advice_is_still_given(fake_prusaslicer, model):
    # Given an unreadable model and a printer profile the server cannot look up, so no sizes are known
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced
    with pytest.raises(ToolError) as error:
        await slice_it(model)  # the stub model and the "Mini" profile

    # Then the original centre advice is the fallback, and PrusaSlicer's message is intact
    assert "center=true" in str(error.value)
    assert OUTSIDE in str(error.value)


async def test_an_empty_first_layer_points_to_the_preview_and_to_rotating(fake_prusaslicer, model):
    # Given PrusaSlicer refusing a model whose lowest layer has nothing to print
    refuse(
        fake_prusaslicer,
        "There is an object with no extrusions in the first layer.\nObject name: scan.stl",
        returncode=1,
    )

    # When it is sliced
    with pytest.raises(ToolError) as error:
        await slice_it(model)

    # Then the message explains the likely causes and the tools to use
    message = str(error.value)
    assert "no extrusions in the first layer" in message
    assert "preview_model" in message
    assert "rotate_x" in message
    assert "repair" in message
    assert "close_holes" in message
    assert "without slicing" in message  # preview_model takes the same rotations, so one can be tried for free
    assert "suggest_orientation" in message


async def test_the_wipe_tower_conflict_names_the_override_that_fixes_it(fake_prusaslicer, make_3mf):
    # Given a project whose settings combine a wipe tower with sequential printing, which PrusaSlicer refuses
    refuse(fake_prusaslicer, "The Wipe Tower is currently not supported for multimaterial sequential prints.")
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced as saved
    with pytest.raises(ToolError) as error:
        await slice_it(project, profiles=NO_PROFILES)

    # Then the message says it is the project's own settings and names the overrides that resolve it
    message = str(error.value)
    assert "overrides" in message
    assert '"wipe_tower": "0"' in message


async def test_an_unrelated_refusal_gets_no_invented_advice(fake_prusaslicer, model):
    # Given PrusaSlicer refusing for a reason we have no advice for
    refuse(fake_prusaslicer, "Something entirely different went wrong.", returncode=1)

    # When it is sliced
    with pytest.raises(ToolError) as error:
        await slice_it(model)

    # Then the message is PrusaSlicer's own, with none of our hints
    message = str(error.value)
    for hint_word in ("center=true", "scale_to_fit", "preview_model", "overrides"):
        assert hint_word not in message


async def test_a_printer_with_no_height_limit_is_described_without_one(fake_prusaslicer, stl_of, monkeypatch):
    # Given a printer profile whose bed has a width and depth but no maximum print height
    refuse(fake_prusaslicer, OUTSIDE)

    async def no_height(printer_profile):
        return 250.0, 210.0, float("inf")

    monkeypatch.setattr(server, "_bed_size", no_height)

    # When an oversized model is sliced on it
    message = await refusal(stl_of(300, 20, 20))

    # Then the bed is described by width and depth only, and the height is left as the model's own
    assert "250 x 210 mm" in message
    assert "inf" not in message
    assert "scale_to_fit=[250, 210, 20]" in message


async def test_a_model_that_fits_when_turned_a_quarter_is_not_called_too_big(fake_prusaslicer, stl_of):
    # Given a model 205 x 240 mm on a 250 x 210 mm bed: too deep as it lies, but it fits turned 90 degrees
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is refused as outside the volume
    message = await refusal(stl_of(205, 240, 20), center=False)

    # Then it is not blamed on size, because it can fit: the advice is to centre it
    assert "center=true" in message
    assert "scale_to_fit" not in message


async def test_a_project_that_does_not_name_its_printer_still_gets_the_centre_advice(fake_prusaslicer, make_3mf):
    # Given a project whose configuration has no printer profile name, refused as outside the volume
    refuse(fake_prusaslicer, OUTSIDE)
    project = make_3mf(config="; fill_density = 20%\n")

    # When it is sliced as saved
    with pytest.raises(ToolError) as error:
        await slice_it(project, profiles=NO_PROFILES)

    # Then there is no bed to compare with, so the original centre advice is given
    assert "center=true" in str(error.value)


async def test_no_centre_advice_when_sizes_are_unknown_and_it_was_already_centred(fake_prusaslicer, model):
    # Given an unreadable model (so its size is unknown) on a known printer, refused even though centring was asked for
    refuse(fake_prusaslicer, OUTSIDE)

    # When it is sliced with centring on
    message = await refusal(model, center=True)

    # Then telling the caller to centre it would be wrong: the message is PrusaSlicer's own
    assert message.endswith(OUTSIDE)


# --------------------------------------------------------------------------- centring plain 3MF files automatically (ADR 0013)

MK3S_PROFILES = (MK3S, "q", "m")


async def test_a_plain_3mf_is_centred_automatically_on_its_printers_bed(fake_prusaslicer, make_3mf):
    # Given a plain 3MF (no PrusaSlicer settings embedded) and no center option, on a 250 x 210 mm bed
    plain = make_3mf(config=None)

    # When it is sliced
    result = await slice_it(plain, profiles=MK3S_PROFILES)

    # Then it is centred on the bed, and the result says so
    args = fake_prusaslicer.calls[0]
    assert args[args.index("--center") + 1] == "125,105"
    assert result.centered is True


async def test_a_project_is_not_centred_unless_asked(fake_prusaslicer, make_3mf):
    # Given a PrusaSlicer project, whose layout was chosen on purpose, and no center option
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced as saved
    result = await slice_it(project, profiles=NO_PROFILES)

    # Then its layout is left alone
    assert "--center" not in fake_prusaslicer.calls[0]
    assert result.centered is False


async def test_a_project_is_centred_when_asked(fake_prusaslicer, make_3mf):
    # Given a project and an explicit request to centre it
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced
    result = await slice_it(project, profiles=NO_PROFILES, center=True)

    # Then it is centred on the bed of the printer stored in the project
    assert "--center" in fake_prusaslicer.calls[0]
    assert result.centered is True


async def test_an_stl_is_not_centred_unless_asked(fake_prusaslicer, model):
    # Given an STL, which PrusaSlicer places itself, and no center option
    # When it is sliced
    result = await slice_it(model, profiles=MK3S_PROFILES)

    # Then no --center is passed
    assert "--center" not in fake_prusaslicer.calls[0]
    assert result.centered is False


async def test_an_stl_is_centred_when_asked(fake_prusaslicer, model):
    # Given an STL and an explicit request to centre it
    # When it is sliced
    result = await slice_it(model, profiles=MK3S_PROFILES, center=True)

    # Then it is centred
    assert "--center" in fake_prusaslicer.calls[0]
    assert result.centered is True


async def test_center_false_keeps_a_plain_3mfs_stored_position(fake_prusaslicer, make_3mf):
    # Given a plain 3MF and an explicit request not to move it
    plain = make_3mf(config=None)

    # When it is sliced
    result = await slice_it(plain, profiles=MK3S_PROFILES, center=False)

    # Then it is not centred
    assert "--center" not in fake_prusaslicer.calls[0]
    assert result.centered is False


async def test_automatic_centring_is_skipped_quietly_when_the_bed_is_unknown(fake_prusaslicer, make_3mf):
    # Given a plain 3MF on a printer profile whose bed cannot be looked up (the fake knows only the MK3S)
    plain = make_3mf(config=None)

    # When it is sliced with the default
    result = await slice_it(plain, profiles=("Some custom printer", "q", "m"))

    # Then it still slices, as before, just without centring
    assert "--center" not in fake_prusaslicer.calls[0]
    assert result.centered is False


async def test_a_refused_plain_3mf_that_was_not_centred_is_told_to_centre_it(fake_prusaslicer, make_3mf):
    # Given a plain 3MF that fits the bed, sliced with centring switched off, and PrusaSlicer refusing it
    refuse(fake_prusaslicer, OUTSIDE)
    plain = make_3mf(config=None, at=(0.0, 0.0, 10.0))

    # When it is sliced
    message = await refusal(plain, center=False)

    # Then the advice is to centre it
    assert "center=true" in message


async def test_a_refused_3mf_that_was_centred_automatically_is_not_told_to_centre_it(fake_prusaslicer, make_3mf):
    # Given a plain 3MF that fits the bed, centred automatically, and PrusaSlicer refusing it anyway
    refuse(fake_prusaslicer, OUTSIDE)
    plain = make_3mf(config=None)

    # When it is sliced with the default
    message = await refusal(plain)

    # Then telling the caller to centre it would be wrong, since that already happened
    assert "center=true" not in message
    assert message.endswith(OUTSIDE)


# --------------------------------------------------------------------------- closing holes before slicing


@pytest.fixture
def open_scan(tmp_path):
    """A 20 x 30 mm cylinder with no bottom, like a scan whose underside was never captured."""
    full = trimesh.creation.cylinder(radius=10, height=30, sections=48)
    path = tmp_path / "scan.stl"
    trimesh.Trimesh(full.vertices, full.faces[full.face_normals[:, 2] > -0.99]).export(path)
    return path


def given_to_prusaslicer(fake):
    mesh = trimesh.load(io.BytesIO(fake.inputs[0]), file_type="stl")
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


async def test_close_holes_slices_a_capped_copy_named_like_the_model(fake_prusaslicer, open_scan):
    # Given a scan with no base
    # When it is sliced with close_holes
    result = await slice_it(open_scan, close_holes=True)

    # Then PrusaSlicer is given a different, watertight file that keeps the model's name, and the result counts the cap
    given = Path(fake_prusaslicer.calls[0][-1])
    assert given != open_scan.resolve()
    assert given.name == "scan.stl"
    assert given_to_prusaslicer(fake_prusaslicer).is_watertight
    assert result.holes_closed == 1
    assert any("capped 1" in warning for warning in result.warnings)


async def test_the_warning_says_how_far_an_uneven_rim_was_carried_to_make_the_base_flat(fake_prusaslicer, tmp_path):
    # Given a scan with no base whose rim has every other point raised 1 mm
    full = trimesh.creation.cylinder(radius=10, height=30, sections=48)
    vertices = full.vertices.copy()
    rim = np.flatnonzero(vertices[:, 2] < -14.99)
    vertices[rim[::2], 2] += 1.0
    scan = tmp_path / "ragged.stl"
    trimesh.Trimesh(vertices, full.faces[full.face_normals[:, 2] > -0.99], process=False).export(scan)

    # When it is sliced with close_holes
    result = await slice_it(scan, close_holes=True)

    # Then the warning says the base was extended by up to 1.0 mm
    assert any("extended by up to 1.0 mm" in warning for warning in result.warnings)


async def test_a_level_rim_is_capped_without_mentioning_an_extension(fake_prusaslicer, open_scan):
    # Given a scan with no base whose rim is level

    # When it is sliced with close_holes
    result = await slice_it(open_scan, close_holes=True)

    # Then the warning does not talk about extending the base
    assert not any("extended" in warning for warning in result.warnings)


async def test_close_holes_leaves_the_original_file_and_no_working_files(fake_prusaslicer, open_scan):
    # Given a scan with no base and its bytes
    before = open_scan.read_bytes()

    # When it is sliced with close_holes
    await slice_it(open_scan, close_holes=True)

    # Then the original is unchanged and only the G-code is left in the output folder
    assert open_scan.read_bytes() == before
    assert leftovers(fake_prusaslicer.out_dir) == ["scan.gcode"]


async def test_holes_are_not_touched_unless_asked(fake_prusaslicer, open_scan):
    # Given a scan with no base

    # When it is sliced without close_holes
    result = await slice_it(open_scan)

    # Then PrusaSlicer gets the original file and nothing is reported as closed
    assert fake_prusaslicer.calls[0][-1] == str(open_scan.resolve())
    assert result.holes_closed == 0


async def test_close_holes_on_a_closed_model_slices_it_as_it_is_and_says_so(fake_prusaslicer, tmp_path):
    # Given a closed cube
    cube = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(20, 20, 20)).export(cube)

    # When it is sliced with close_holes
    result = await slice_it(cube, close_holes=True)

    # Then the original is sliced and a warning says there was nothing to close
    assert fake_prusaslicer.calls[0][-1] == str(cube.resolve())
    assert result.holes_closed == 0
    assert any("no open holes" in warning for warning in result.warnings)


async def test_a_hole_that_cannot_be_capped_is_reported_and_the_model_sliced_as_it_is(fake_prusaslicer, tmp_path):
    # Given a cube with one triangle missing: a gap too small to cap
    box = trimesh.creation.box(extents=(20, 20, 20))
    gap = tmp_path / "gap.stl"
    trimesh.Trimesh(box.vertices, box.faces[:-1], process=False).export(gap)

    # When it is sliced with close_holes
    result = await slice_it(gap, close_holes=True)

    # Then the original is sliced, and the warning says the hole could not be closed
    assert fake_prusaslicer.calls[0][-1] == str(gap.resolve())
    assert result.holes_closed == 0
    assert any("could not close" in warning for warning in result.warnings)


async def test_close_holes_is_refused_for_a_project_and_nothing_is_created(fake_prusaslicer, make_3mf):
    # Given a PrusaSlicer project, which carries settings a repaired copy would lose
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced with close_holes
    with pytest.raises(ToolError) as error:
        await slice_it(project, NO_PROFILES, close_holes=True)

    # Then it is refused with the reason, and PrusaSlicer is not run and no folder is made
    assert "project" in str(error.value)
    assert fake_prusaslicer.calls == []
    assert not fake_prusaslicer.out_dir.exists()


async def test_close_holes_on_an_unreadable_model_fails_readably_and_creates_nothing(fake_prusaslicer, tmp_path):
    # Given a file with an STL name that is not a mesh
    broken = tmp_path / "broken.stl"
    broken.write_text("not a mesh")

    # When it is sliced with close_holes
    with pytest.raises(ToolError) as error:
        await slice_it(broken, close_holes=True)

    # Then it is called unreadable, PrusaSlicer is not run and no folder is made
    assert "Could not load a triangle mesh" in str(error.value)
    assert fake_prusaslicer.calls == []
    assert not fake_prusaslicer.out_dir.exists()


async def test_a_capped_plain_3mf_is_still_centred_on_the_bed(fake_prusaslicer, tmp_path):
    # Given a plain 3MF whose model has no base, stored away from the bed's centre
    full = trimesh.creation.cylinder(radius=10, height=30, sections=48)
    scan = trimesh.Trimesh(full.vertices, full.faces[full.face_normals[:, 2] > -0.99])
    scan.apply_translation([5, 5, 15])
    model = tmp_path / "scan.3mf"
    scan.export(model)

    # When it is sliced with close_holes and no center option, on a 250 x 210 mm bed
    result = await slice_it(model, profiles=MK3S_PROFILES, close_holes=True)

    # Then the capped copy is centred as a plain 3MF would be
    args = fake_prusaslicer.calls[0]
    assert args[args.index("--center") + 1] == "125,105"
    assert result.centered is True
    assert result.holes_closed == 1
