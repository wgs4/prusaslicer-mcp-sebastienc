"""Integration tests against a real PrusaSlicer install (skipped when absent)."""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from conftest import PROJECT_CONFIG
from mcp.server.mcpserver import Image
from mcp.server.mcpserver.exceptions import ToolError
from step_builders import step_cube

from prusaslicer_mcp import server
from prusaslicer_mcp.preview_gcode import iter_extrusions, scan_gcode


@pytest.fixture(autouse=True)
def _reset_config(prusaslicer, tmp_path, monkeypatch):
    monkeypatch.setenv("PRUSASLICER_MCP_OUTPUT_DIR", str(tmp_path / "out"))
    server._config = None
    yield
    server._config = None


# A stock MK4S triple from the PrusaResearch bundle. Fixed names keep the suite off whichever
# printer a machine lists first, and off presets stored in a user's data directory.
_KNOWN_PRINTER = "Original Prusa MK4S 0.4 nozzle"
_KNOWN_PRINT = "0.20mm SPEED @MK4S 0.4"
_KNOWN_FILAMENT = "Prusament PLA @MK4S"


@pytest.fixture
async def installed():
    """MK4S 0.4, the 0.20 mm SPEED print profile, and Prusament PLA. Fails if this install lacks them."""
    printers = await server.list_printers()
    names = [variant.printer_profile for model in printers for variant in model.variants]
    assert _KNOWN_PRINTER in names, f"{_KNOWN_PRINTER!r} is not installed"
    profiles = await server.list_profiles(_KNOWN_PRINTER)
    print_profile = next((item for item in profiles.print_profiles if item.name == _KNOWN_PRINT), None)
    assert print_profile is not None, f"{_KNOWN_PRINT!r} is not installed for {_KNOWN_PRINTER!r}"
    assert _KNOWN_FILAMENT in print_profile.filament_profiles, (
        f"{_KNOWN_FILAMENT!r} is not compatible with {_KNOWN_PRINT!r}"
    )
    return SimpleNamespace(printer=_KNOWN_PRINTER, print_profile=_KNOWN_PRINT, filament=_KNOWN_FILAMENT)


@pytest.fixture
def origin_cube_3mf(tmp_path):
    """A 20 mm cube saved in a plain 3MF around the origin, so half of it is off the bed."""
    path = tmp_path / "origin_cube.3mf"
    trimesh.creation.box(extents=(20, 20, 20)).export(path)
    return path


async def test_list_printers_returns_models_with_variants():
    # Given a PrusaSlicer install with printer profiles (the prusaslicer fixture)

    # When the installed printers are listed
    printers = await server.list_printers()

    # Then there are models and each one has at least one variant
    assert printers
    assert all(p.variants for p in printers)


async def test_list_printers_shows_model_names_without_doubled_ampersands():
    # Given vendor bundles that store model names GUI-escaped ("&&")

    # When the installed printers are listed
    printers = await server.list_printers()

    # Then no displayed model name still contains "&&"
    assert all("&&" not in p.name for p in printers)


async def test_list_profiles_returns_print_profiles_for_a_listed_printer(installed):
    # Given a printer profile name taken from list_printers

    # When its profiles are listed
    profiles = await server.list_profiles(installed.printer)

    # Then there are print profiles to choose from
    assert profiles.print_profiles


async def test_inspect_model_reports_size_volume_and_fit_for_a_cube_on_a_large_bed(cube_stl):
    # Given a 20 mm cube and a 180 mm build volume

    # When the model is inspected against that volume
    info = await server.inspect_model(str(cube_stl), bed_width=180, bed_depth=180, max_height=180)

    # Then it is watertight, fits, and has the cube's size and volume
    assert info.watertight
    assert info.fits_bed
    assert info.size_mm == pytest.approx((20, 20, 20))
    assert info.volume_mm3 == pytest.approx(8000)


async def test_inspect_model_reports_a_cube_as_not_fitting_a_tiny_bed(cube_stl):
    # Given a 20 mm cube and a 10 mm build volume

    # When the model is inspected against that volume
    info = await server.inspect_model(str(cube_stl), bed_width=10, bed_depth=10, max_height=10)

    # Then it is reported as not fitting
    assert info.fits_bed is False


async def test_slice_rejects_an_invalid_override_key(cube_stl):
    # Given an override key that is not a valid PrusaSlicer option name

    # When / Then slicing is refused with a readable message
    with pytest.raises(ToolError, match="Invalid override key"):
        await server.slice_model(str(cube_stl), "x", "y", "z", overrides={"bad key; rm": "1"})


async def test_slice_returns_time_filament_and_layers_for_a_cube_with_an_override(cube_stl, installed):
    # Given a cube, real installed profiles, and a fill density override

    # When the cube is sliced
    result = await server.slice_model(
        str(cube_stl), installed.printer, installed.print_profile, installed.filament, overrides={"fill_density": "10%"}
    )

    # Then the statistics are filled in
    assert result.stats.estimated_time_seconds
    assert result.stats.filament_used_g
    assert result.stats.layer_count


async def test_a_plain_3mf_around_the_origin_is_centred_automatically(origin_cube_3mf, installed):
    # Given a plain 3MF that keeps its stored coordinates and so sits half off the bed

    # When it is sliced with no center option
    result = await server.slice_model(
        str(origin_cube_3mf), installed.printer, installed.print_profile, installed.filament
    )

    # Then it is centred for the caller and slices, and the result says so (ADR 0013)
    assert result.centered is True
    assert result.stats.layer_count


async def test_slice_refuses_a_plain_3mf_around_the_origin_when_centring_is_off(origin_cube_3mf, installed):
    # Given a plain 3MF that keeps its stored coordinates and so sits half off the bed

    # When / Then slicing it with centring switched off fails with the remedy in the message
    with pytest.raises(ToolError, match="center=true"):
        await server.slice_model(
            str(origin_cube_3mf), installed.printer, installed.print_profile, installed.filament, center=False
        )


async def test_slice_places_a_plain_3mf_around_the_origin_when_centered(origin_cube_3mf, installed):
    # Given the same plain 3MF around the origin

    # When it is sliced with center on
    result = await server.slice_model(
        str(origin_cube_3mf), installed.printer, installed.print_profile, installed.filament, center=True
    )

    # Then it slices
    assert result.stats.layer_count


async def test_a_scan_with_no_base_slices_to_its_full_height_once_the_hole_is_closed(tmp_path, installed):
    # Given a 30 mm tall cylinder with no bottom, the shape of a scan whose underside was never captured
    full = trimesh.creation.cylinder(radius=10, height=30, sections=48)
    scan = tmp_path / "scan.stl"
    trimesh.Trimesh(full.vertices, full.faces[full.face_normals[:, 2] > -0.99]).export(scan)
    before = scan.read_bytes()

    # When it is sliced with close_holes
    result = await server.slice_model(
        str(scan), installed.printer, installed.print_profile, installed.filament, close_holes=True
    )

    # Then PrusaSlicer accepts the capped copy, prints all 30 mm, and the result says a hole was closed
    assert result.holes_closed == 1
    assert max(scan_gcode(Path(result.gcode_path)).layer_heights) == pytest.approx(30, abs=0.3)
    assert scan.read_bytes() == before  # the model file itself is not rewritten


@pytest.fixture
def step_cube_file(tmp_path):
    """A 20 mm cube as a STEP file, which trimesh cannot read."""
    path = tmp_path / "cube.step"
    path.write_text(step_cube(20))
    return path


async def test_inspect_model_measures_a_step_file_through_prusaslicer(step_cube_file):
    # Given a STEP cube, a format trimesh cannot read

    # When it is inspected
    info = await server.inspect_model(str(step_cube_file))

    # Then PrusaSlicer's conversion gives its real size and volume
    assert info.size_mm == pytest.approx((20, 20, 20), abs=0.01)
    assert info.volume_mm3 == pytest.approx(8000, rel=1e-3)
    assert info.watertight is True


async def test_preview_model_draws_a_step_file(step_cube_file):
    # Given a STEP cube

    # When it is previewed
    content = await server.preview_model(str(step_cube_file), "iso", 256)

    # Then there is a picture, and the caption gives the converted size
    assert any(isinstance(item, Image) and item.data for item in content)
    assert any(isinstance(item, str) and "20.0 x 20.0 x 20.0" in item for item in content)


async def test_a_step_file_slices(step_cube_file, installed):
    # Given a STEP cube and installed profiles

    # When it is sliced
    result = await server.slice_model(
        str(step_cube_file), installed.printer, installed.print_profile, installed.filament
    )

    # Then it prints
    assert result.stats.layer_count


async def test_a_broken_step_file_fails_with_a_message_that_says_what_to_check(tmp_path):
    # Given a file with a STEP name and no STEP content
    broken = tmp_path / "broken.step"
    broken.write_text("this is not a STEP file")

    # When it is inspected
    with pytest.raises(ToolError) as error:
        await server.inspect_model(str(broken))

    # Then the message names the file and the likely cause
    assert "could not convert broken.step" in str(error.value)
    assert "valid STEP file" in str(error.value)


def filament_per_layer(gcode: Path, layers: int) -> list[float]:
    """Filament (mm) extruded in each of the first layers, arcs included. Assumes relative extrusion (M83)."""
    totals = [0.0] * layers
    layer = -1
    for line in gcode.read_text(errors="replace").splitlines():
        if line.startswith(";LAYER_CHANGE"):
            layer += 1
            if layer >= layers:
                break
        elif layer >= 0 and line.startswith(("G1", "G2", "G3")) and " E" in line:
            amount = float(line.split(" E", 1)[1].split()[0])
            totals[layer] += max(amount, 0.0)
    return totals


async def test_an_uneven_scan_base_is_closed_flat_so_the_first_layer_has_real_contact(tmp_path, installed):
    # Given a cylinder with no bottom whose rim has every other point raised 1 mm, like a scan's ragged edge.
    # (A cap joined straight to such a rim is bumpy: only the low points touch the bed, and the print starts from
    # spikes. This is what happened to the real scan this was found on.)
    full = trimesh.creation.cylinder(radius=10, height=30, sections=48)
    vertices = full.vertices.copy()
    rim = np.flatnonzero(vertices[:, 2] < -14.99)
    vertices[rim[::2], 2] += 1.0
    scan = tmp_path / "ragged.stl"
    trimesh.Trimesh(vertices, full.faces[full.face_normals[:, 2] > -0.99], process=False).export(scan)

    # When it is sliced with close_holes
    result = await server.slice_model(
        str(scan), installed.printer, installed.print_profile, installed.filament, close_holes=True
    )

    # Then the first layer prints as much as the second: a whole base, not a few points
    first, second = filament_per_layer(Path(result.gcode_path), 2)
    assert first > 0.7 * second


def turned_cone_stl(tmp_path, x: float, y: float, z: float):
    """A flat cone (40 mm across, 10 mm tall, a 48-sided base) exported at an odd angle about fixed X, Y, Z."""
    mesh = trimesh.creation.cone(radius=20, height=10, sections=48)
    for axis, degrees in (([1, 0, 0], x), ([0, 1, 0], y), ([0, 0, 1], z)):
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), axis))
    path = tmp_path / "cone.stl"
    mesh.export(path)
    return path


def rotation_flags(option):
    return server._transform_args(option.rotate_x, option.rotate_y, None, None, None, False, None)


@pytest.mark.parametrize("angles", [(0, 0, 0), (180, 0, 0), (37, -20, 110), (16, 0, 0), (90, 0, 0), (0, -90, 0)])
async def test_the_suggested_rotation_stands_the_model_on_its_base_in_prusaslicer(tmp_path, angles):
    # Given a cone exported at an odd angle, and the best orientation suggested for it
    model = turned_cone_stl(tmp_path, *angles)
    best = (await server.suggest_orientation(str(model))).options[0]

    # When PrusaSlicer applies that rotation
    mesh = await server._export_mesh(model, rotation_flags(best))

    # Then the whole 48-point base is level at the bottom and the cone is 10 mm tall: our matrix is PrusaSlicer's
    assert (mesh.vertices[:, 2] <= mesh.vertices[:, 2].min() + 0.02).sum() >= 48
    assert mesh.extents[2] == pytest.approx(10, abs=0.05)


async def test_a_tilted_base_is_refused_as_stored_and_prints_a_whole_first_layer_when_levelled(tmp_path, installed):
    # Given a cone standing on its base but tilted 16 degrees, like a scan whose base is not square to its axis
    model = turned_cone_stl(tmp_path, 16, 0, 0)
    best = (await server.suggest_orientation(str(model))).options[0]

    # When it is sliced as stored, and then with the suggested rotation
    with pytest.raises(ToolError, match="no extrusions in the first layer"):
        await slice_bar(model, installed)
    levelled = await slice_bar(model, installed, rotate_x=best.rotate_x, rotate_y=best.rotate_y)

    # Then as stored it rests on an edge and is refused, and levelled the first layer carries as much as the second
    first, second = filament_per_layer(Path(levelled.gcode_path), 2)
    assert first > 0.7 * second


async def test_measuring_a_cone_on_its_apex_shows_what_the_support_really_costs(tmp_path, installed):
    # Given a flat cone exported apex down, so as stored it rests on a point
    mesh = trimesh.creation.cone(radius=20, height=10, sections=48)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    model = tmp_path / "cone.stl"
    mesh.export(model)

    # When orientations are suggested with real profiles, so each is sliced with supports on
    advice = await server.suggest_orientation(
        str(model),
        count=3,
        printer_profile=installed.printer,
        print_profile=installed.print_profile,
        material_profile=installed.filament,
    )

    # Then the best option, base down, printed with no support at all, and the as-stored one cost more or was refused
    best = advice.options[0]
    assert best.measured is not None
    assert best.measured.refused is None
    assert best.measured.support_g == pytest.approx(0.0, abs=0.05)
    assert (best.measured.filament_g or 0) > 0
    assert best.measured.estimated_time_seconds
    stored = next(option for option in advice.options if option.as_stored)
    assert stored.measured is not None
    assert stored.measured.refused or (stored.measured.support_g or 0) > 1.0


def recorded_setting(result, key: str) -> str:
    """The value PrusaSlicer wrote for `key` in the config dump at the end of the G-code."""
    for line in reversed(Path(result.gcode_path).read_text().splitlines()):
        if line.startswith(f"; {key} = "):
            return line.split(" = ", 1)[1]
    raise AssertionError(f"{key} not found in the G-code")


async def test_slice_applies_a_projects_saved_settings_when_no_profiles_are_given(make_3mf):
    # Given a project 3MF that saves a 5 mm brim and supports
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced without profiles
    result = await server.slice_model(str(project))

    # Then PrusaSlicer recorded the project's settings in the G-code
    assert result.settings_from == "project"
    assert recorded_setting(result, "brim_width") == "5"
    assert recorded_setting(result, "support_material") == "1"


async def test_slice_with_profiles_ignores_a_projects_saved_settings(make_3mf, installed):
    # Given the same project, and an installed print profile that sets no brim
    project = make_3mf(config=PROJECT_CONFIG)

    # When it is sliced with profiles
    result = await server.slice_model(str(project), installed.printer, installed.print_profile, installed.filament)

    # Then the profile's values win over the project's saved brim
    assert result.settings_from == "profiles"
    assert recorded_setting(result, "brim_width") != "5"


async def test_slice_flags_a_model_that_has_an_empty_stretch_in_its_height(tmp_path, installed):
    # Given an STL of two cubes with 10 mm of air between them, so the upper one floats
    lower = trimesh.creation.box(extents=(20, 20, 10))
    lower.apply_translation((0, 0, 5))
    upper = trimesh.creation.box(extents=(20, 20, 10))
    upper.apply_translation((0, 0, 25))
    model = tmp_path / "floating.stl"
    trimesh.util.concatenate([lower, upper]).export(model)

    # When it is sliced with real profiles
    result = await server.slice_model(str(model), installed.printer, installed.print_profile, installed.filament)

    # Then the caller is told layers are missing, and where, from the real G-code's layer markers
    gap_warnings = [w for w in result.warnings if w.startswith("Nothing is printed across")]
    assert len(gap_warnings) == 1
    # The air is between z 10 and z 20. A substring such as "between z 10" also matches 10.2 and 100,
    # and misses a marker that landed on 9.95. One layer (0.2 mm) of slack covers that rounding.
    found = re.search(r"between z ([0-9.]+) and ([0-9.]+) mm", gap_warnings[0])
    assert found, gap_warnings[0]
    assert float(found.group(1)) == pytest.approx(10, abs=0.25)
    assert float(found.group(2)) == pytest.approx(20, abs=0.25)


async def test_slice_of_a_plain_cube_reports_no_empty_stretch(cube_stl, installed):
    # Given a solid cube

    # When it is sliced with real profiles
    result = await server.slice_model(str(cube_stl), installed.printer, installed.print_profile, installed.filament)

    # Then no layers are reported missing
    assert not any(w.startswith("Nothing is printed across") for w in result.warnings)


# --------------------------------------------------------------------------- transforms on real PrusaSlicer output


@pytest.fixture
def bar_stl(tmp_path):
    """A 10 x 20 x 30 mm box, so rotating or scaling it changes the number of layers in a predictable way."""
    path = tmp_path / "bar.stl"
    trimesh.creation.box(extents=(10, 20, 30)).export(path)
    return path


def ratio(numerator: float | None, denominator: float | None) -> float:
    """numerator / denominator, failing clearly if a statistic was not reported."""
    assert numerator is not None
    assert denominator is not None
    return numerator / denominator


async def slice_bar(model, installed, **extra):
    return await server.slice_model(str(model), installed.printer, installed.print_profile, installed.filament, **extra)


async def test_rotating_about_x_stands_the_box_on_its_side(bar_stl, installed):
    # Given a box 30 mm tall and 20 mm deep, sliced as it is and rotated 90 degrees about X
    plain = await slice_bar(bar_stl, installed)

    # When it is rotated 90 degrees about X
    rotated = await slice_bar(bar_stl, installed, rotate_x=90)

    # Then it is now 20 mm tall, so it has two thirds of the layers
    assert ratio(rotated.stats.layer_count, plain.stats.layer_count) == pytest.approx(20 / 30, abs=0.03)


def caption_of(content) -> str:
    return next(item for item in content if isinstance(item, str))


async def test_a_preview_with_a_rotation_shows_the_height_the_slice_prints(bar_stl, installed):
    # Given a 10 x 20 x 30 mm box rotated 90 degrees about X

    # When it is previewed and sliced with that rotation
    caption = caption_of(await server.preview_model(str(bar_stl), rotate_x=90.0))
    sliced = await slice_bar(bar_stl, installed, rotate_x=90)

    # Then the preview's size is the print's size: 10 x 30 wide, 20 tall
    assert "Size 10.0 x 30.0 x 20.0" in caption
    assert max(scan_gcode(Path(sliced.gcode_path)).layer_heights) == pytest.approx(20, abs=0.25)


@pytest.mark.parametrize(
    ("arguments", "size"),
    [
        ({"scale": 2.0}, "20.0 x 40.0 x 60.0"),
        ({"scale_to_fit": [100.0, 100.0, 15.0]}, "5.0 x 10.0 x 15.0"),
        ({"rotate_z": 90.0}, "20.0 x 10.0 x 30.0"),
    ],
    ids=["scale", "scale to fit", "rotate z"],
)
async def test_a_preview_applies_the_transforms_prusaslicer_applies(bar_stl, installed, arguments, size):
    # Given a 10 x 20 x 30 mm box

    # When it is previewed with a transform
    caption = caption_of(await server.preview_model(str(bar_stl), **arguments))

    # Then the size is what PrusaSlicer makes of it
    assert f"Size {size}" in caption


def tripod_stl(tmp_path):
    """Three arms of different lengths meeting at a corner, so every orientation looks different."""
    arms = [trimesh.creation.box(extents=e) for e in ((40, 8, 8), (8, 25, 8), (8, 8, 15))]
    for arm, shift in zip(arms, ([16, 0, 0], [0, 8.5, 0], [0, 0, 3.5]), strict=True):
        arm.apply_translation(shift)
    faces = np.vstack([arm.faces + sum(len(a.vertices) for a in arms[:i]) for i, arm in enumerate(arms)])
    path = tmp_path / "tripod.stl"
    trimesh.Trimesh(np.vstack([arm.vertices for arm in arms]), faces).export(path)
    return path


@pytest.mark.parametrize(
    "rotation",
    [{"rotate_x": 90.0}, {"rotate_y": 90.0, "rotate_z": -90.0}],
    ids=["plain rotation", "y 90 with z -90"],
)
async def test_a_previews_base_is_the_slices_first_layer_even_where_prusaslicer_is_odd(tmp_path, installed, rotation):
    # Given an asymmetric model. Rotating +-90 about Y and then about Z turns it a different way up from the
    # fixed X, Y, Z order that the other cases follow (measured on PrusaSlicer 2.9.7), which is why the preview asks
    # PrusaSlicer instead of working the rotation out itself.
    model = tripod_stl(tmp_path)

    # When it is previewed as PrusaSlicer transforms it, and sliced the same way
    flags = server._transform_args(
        rotation.get("rotate_x"), rotation.get("rotate_y"), rotation.get("rotate_z"), None, None, False, None
    )
    mesh = await server._export_mesh(model, flags)
    sliced = await slice_bar(model, installed, **rotation)

    # Then the footprint on the bed in the preview is the footprint of the first printed layer
    on_bed = mesh.vertices[:, 2] <= mesh.bounds[0][2] + 0.05
    previewed = np.ptp(mesh.vertices[on_bed][:, :2], axis=0)
    first_layer = [m for m in iter_extrusions(Path(sliced.gcode_path)) if m.layer == 0 and m.feature != "Skirt/Brim"]
    printed = (
        max(max(m.x0, m.x1) for m in first_layer) - min(min(m.x0, m.x1) for m in first_layer),
        max(max(m.y0, m.y1) for m in first_layer) - min(min(m.y0, m.y1) for m in first_layer),
    )
    # X/Y in the G-code are extrusion centerlines, so the span is one extrusion width shorter than the
    # mesh. Elephant-foot compensation (spelled elefant_foot_compensation) insets each side as well.
    width = float(recorded_setting(sliced, "first_layer_extrusion_width"))
    foot = float(recorded_setting(sliced, "elefant_foot_compensation"))
    inset = width + 2 * foot
    expected = (float(previewed[0]) - inset, float(previewed[1]) - inset)
    assert printed == pytest.approx(expected, abs=0.15)


async def test_scale_doubles_the_height(bar_stl, installed):
    # Given a box sliced at its own size
    plain = await slice_bar(bar_stl, installed)

    # When it is scaled by a factor of 2
    scaled = await slice_bar(bar_stl, installed, scale=2)

    # Then it is twice as tall, so it has twice the layers
    assert ratio(scaled.stats.layer_count, plain.stats.layer_count) == pytest.approx(2.0, abs=0.05)


async def test_scale_to_fit_shrinks_the_box_into_the_given_volume(bar_stl, installed):
    # Given a box 30 mm tall

    # When it is scaled to fit a 15 mm cube
    plain = await slice_bar(bar_stl, installed)
    fitted = await slice_bar(bar_stl, installed, scale_to_fit=[15, 15, 15])

    # Then the tallest side is halved, so it has half the layers
    assert ratio(fitted.stats.layer_count, plain.stats.layer_count) == pytest.approx(0.5, abs=0.03)


async def test_copies_print_that_many_objects(bar_stl, installed):
    # Given a box sliced once
    one = await slice_bar(bar_stl, installed)

    # When three copies are requested
    three = await slice_bar(bar_stl, installed, copies=3)

    # Then about three times the filament is used, and the layers are unchanged
    assert ratio(three.stats.filament_used_g, one.stats.filament_used_g) == pytest.approx(3.0, rel=0.1)
    assert three.stats.layer_count == one.stats.layer_count


async def test_split_brings_separate_solids_together_on_the_bed(tmp_path, installed):
    # Given one STL holding two solids 60 mm apart
    left = trimesh.creation.box(extents=(20, 20, 10))
    right = trimesh.creation.box(extents=(10, 10, 30))
    right.apply_translation((60, 0, 0))
    both = trimesh.util.concatenate([left, right])
    model = tmp_path / "two.stl"
    both.export(model)

    # When it is sliced as one object and then split (each width is measured before the next slice, because both
    # write the same default file)
    def printed_width(result) -> float:
        extent = scan_gcode(Path(result.gcode_path)).x_range
        return extent[1] - extent[0]

    plain_width = printed_width(await slice_bar(model, installed))
    split_width = printed_width(await slice_bar(model, installed, split=True))

    # Then PrusaSlicer re-arranges the separated solids, so the printed width shrinks
    assert split_width < plain_width * 0.7


async def test_a_projects_saved_settings_survive_a_rotation(make_3mf):
    # Given a project that saves a 5 mm brim, sliced as saved
    project = make_3mf(config=PROJECT_CONFIG)
    plain = await server.slice_model(str(project))

    # When it is sliced rotated about X with no profiles
    rotated = await server.slice_model(str(project), rotate_x=45)

    # Then it still uses the project's settings, and the model really was rotated
    assert rotated.settings_from == "project"
    assert recorded_setting(rotated, "brim_width") == "5"
    assert rotated.stats.layer_count != plain.stats.layer_count


async def test_a_plain_3mf_can_be_rotated_and_centred_together(origin_cube_3mf, installed):
    # Given a plain 3MF modelled around the origin, which only slices once it is centred on the bed

    # When it is rotated and centred
    result = await server.slice_model(
        str(origin_cube_3mf), installed.printer, installed.print_profile, installed.filament, rotate_x=90, center=True
    )

    # Then it slices
    assert result.stats.layer_count


async def test_a_refused_slice_does_not_reuse_the_file_from_an_earlier_slice(origin_cube_3mf, installed, tmp_path):
    # Given a good slice already written to a path (centred, so PrusaSlicer accepts it)
    target = tmp_path / "same.gcode"
    args = (str(origin_cube_3mf), installed.printer, installed.print_profile, installed.filament)
    await server.slice_model(*args, center=True, output_path=str(target))
    earlier = target.read_bytes()

    # When the same model is sliced to the same path with centring off, which PrusaSlicer refuses with exit code 0
    with pytest.raises(ToolError, match="outside of the print volume"):
        await server.slice_model(*args, center=False, output_path=str(target))

    # Then it is reported as a failure and the earlier file is left exactly as it was
    assert target.read_bytes() == earlier


async def test_a_refused_oversized_model_says_how_big_it_is_and_what_to_do(tmp_path, installed):
    # Given a model a metre long, far larger than any bed
    model = tmp_path / "huge.stl"
    trimesh.creation.box(extents=(1000, 20, 20)).export(model)

    # When it is sliced with real profiles
    with pytest.raises(ToolError) as error:
        await server.slice_model(str(model), installed.printer, installed.print_profile, installed.filament)

    # Then PrusaSlicer's refusal comes with the model's size and a scale_to_fit remedy, not a pointless recentre
    message = str(error.value)
    assert "outside of the print volume" in message
    assert "1000.0 x 20.0 x 20.0 mm" in message
    assert "scale_to_fit=[" in message
    assert "center=true" not in message
