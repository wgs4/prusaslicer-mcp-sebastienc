"""The session data directory is a synthetic fixture, never a user's PrusaSlicer folder."""

import os
from pathlib import Path

import pytest
from conftest import fff_model_lines


def test_fff_model_lines_keep_fff_models_and_strip_variant_spaces():
    # Given a vendor bundle with one FFF model, one SLA model, and a model that has no variants
    text = """
[printer_model:MK4S]
name = Original Prusa MK4S
variants = 0.4; 0.6
technology = FFF

[printer_model:SL1]
variants = 0.05
technology = SLA

[printer_model:BARE]
technology = FFF
"""

    # When the synthetic PrusaSlicer.ini lines are built
    lines = fff_model_lines(text)

    # Then only the FFF model with variants is enabled, and the variant list has no spaces
    assert lines == ["model:MK4S = 0.4;0.6"]


def test_synthetic_datadir_enables_fff_models_outside_the_user_directory(synthetic_datadir):
    # Given PrusaSlicer and its vendor bundle are installed (otherwise there is nothing to point at)
    if synthetic_datadir is None:
        pytest.skip("PrusaSlicer vendor profiles were not found")

    # When the session data directory is inspected
    configured = Path(os.environ["PRUSASLICER_DATADIR"]).resolve()
    ini = (synthetic_datadir / "PrusaSlicer.ini").read_text(encoding="utf-8")
    model_lines = [line for line in ini.splitlines() if line.startswith("model:")]

    # Then it is the fixture, it enables the FFF models, and it holds no printer keys
    assert configured == synthetic_datadir.resolve()
    assert Path.home() not in configured.parents
    assert ini.splitlines()[0] == "[vendor:PrusaResearch]"
    assert any(line.startswith("model:MK4S = ") and "0.4" in line.split("=", 1)[1] for line in model_lines)
    assert any(line.startswith("model:MK3SMMU2S = ") for line in model_lines)
    assert (synthetic_datadir / "vendor" / "PrusaResearch.ini").is_file()
    assert not (synthetic_datadir / "physical_printer").exists()
    assert oct((synthetic_datadir / "PrusaSlicer.ini").stat().st_mode & 0o777) == "0o600"
