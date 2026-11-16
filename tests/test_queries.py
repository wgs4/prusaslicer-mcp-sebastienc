"""Unit tests for the --query-* plumbing and the discovery tools built on it (CLI faked, no PrusaSlicer).

Only `cli.run` is faked. The temp-file handling, failure detection and JSON-to-model shaping in
server.py are the real code under test.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from prusaslicer_mcp import cli, server
from prusaslicer_mcp.config import Config


@pytest.fixture
def prusaslicer_query(tmp_path, monkeypatch):
    """A fake CLI for --query-* actions: set `.state` to choose what PrusaSlicer writes and says."""
    monkeypatch.setattr(
        server,
        "_config",
        Config(executable=tmp_path / "PrusaSlicer", datadir=None, output_dir=tmp_path / "out", timeout=5),
    )
    calls: list[SimpleNamespace] = []
    state = SimpleNamespace(payload=None, raw=None, returncode=1, stdout="", stderr="")

    async def fake_run(config, args, timeout=None):
        out = Path(args[args.index("-o") + 1])
        calls.append(SimpleNamespace(args=args, timeout=timeout, out=out))
        if state.raw is not None:
            out.write_text(state.raw)
        elif state.payload is not None:
            out.write_text(json.dumps(state.payload))
        return cli.CliResult(state.returncode, state.stdout, state.stderr)

    monkeypatch.setattr(cli, "run", fake_run)
    return SimpleNamespace(calls=calls, state=state)


PRINTER_PAYLOAD = {
    "printer_models": [
        {
            "id": "MAKER",
            "name": "Maker && Co One",
            "variants": [
                {
                    "name": 0.4,
                    "printer_profiles": [
                        {
                            "name": "Maker One 0.4",
                            "extruders_cnt": 2,
                            "bed": {"width": 200, "height": 150, "max_print_height": 100},
                        }
                    ],
                }
            ],
        }
    ]
}


# --------------------------------------------------------------------------- failure detection


async def test_query_succeeds_when_prusaslicer_exits_with_status_1(prusaslicer_query):
    # Given PrusaSlicer wrote its answer but exited with 1, which it does even on success in 2.9.x
    prusaslicer_query.state.payload = PRINTER_PAYLOAD
    prusaslicer_query.state.returncode = 1

    # When the printers are listed
    printers = await server.list_printers()

    # Then the exit status is ignored and the answer is used
    assert [p.id for p in printers] == ["MAKER"]


async def test_query_without_an_output_file_fails_with_the_slicers_stderr(prusaslicer_query):
    # Given PrusaSlicer writes no output file and explains on stderr
    prusaslicer_query.state.stderr = "  something went wrong  \n"

    # When / Then the tool fails with the exit status and the trimmed explanation
    with pytest.raises(ToolError, match=r"PrusaSlicer query failed \(exit 1\): something went wrong$"):
        await server.list_printers()


async def test_query_failure_falls_back_to_stdout_when_stderr_is_empty(prusaslicer_query):
    # Given PrusaSlicer writes no output file and only prints to stdout
    prusaslicer_query.state.stdout = "explained on stdout"

    # When / Then the stdout text is used as the explanation
    with pytest.raises(ToolError, match="explained on stdout"):
        await server.list_printers()


async def test_query_with_an_empty_output_file_is_a_failure(prusaslicer_query):
    # Given PrusaSlicer creates the output file but leaves it empty
    prusaslicer_query.state.raw = ""
    prusaslicer_query.state.stderr = "no data"

    # When / Then it is treated like a missing file
    with pytest.raises(ToolError, match="PrusaSlicer query failed"):
        await server.list_printers()


async def test_query_with_malformed_json_is_a_readable_failure(prusaslicer_query):
    # Given PrusaSlicer writes something that is not JSON
    prusaslicer_query.state.raw = "{not json"

    # When / Then the tool fails saying so, instead of leaking a JSON decoder crash
    with pytest.raises(ToolError, match="PrusaSlicer produced invalid JSON"):
        await server.list_printers()


async def test_unknown_printer_profile_in_a_query_points_at_the_listing_tools(prusaslicer_query):
    # Given PrusaSlicer does not know the printer profile it is asked about
    prusaslicer_query.state.stderr = "Printer profile 'Nope' wasn't found among installed printers."

    # When / Then the failure carries a hint about where exact names come from
    with pytest.raises(ToolError, match="list_printers and list_profiles"):
        await server.list_profiles("Nope")


# --------------------------------------------------------------------------- how queries are run


async def test_query_is_run_with_a_60_second_timeout_and_cleans_up_after_itself(prusaslicer_query):
    # Given a successful query
    prusaslicer_query.state.payload = PRINTER_PAYLOAD

    # When the printers are listed
    await server.list_printers()

    # Then the query had its own 60 s limit and its temporary output was removed
    call = prusaslicer_query.calls[0]
    assert call.timeout == 60
    assert not call.out.parent.exists()


@pytest.mark.parametrize("technology", ["FFF", "SLA"])
async def test_list_printers_asks_for_the_requested_technology(prusaslicer_query, technology):
    # Given an empty but valid answer
    prusaslicer_query.state.payload = {"printer_models": []}

    # When printers of one technology are listed
    await server.list_printers(technology)

    # Then PrusaSlicer is asked for exactly that technology
    args = prusaslicer_query.calls[0].args
    assert args[:3] == ["--query-printer-models", "--printer-technology", technology]


async def test_list_profiles_asks_for_the_given_printer_profile_verbatim(prusaslicer_query):
    # Given a printer profile name with spaces and ampersands
    prusaslicer_query.state.payload = {"print_profiles": []}
    name = "Original Prusa MINI & MINI+ Input Shaper"

    # When its profiles are listed
    await server.list_profiles(name)

    # Then the name is passed unchanged as one argument
    args = prusaslicer_query.calls[0].args
    assert args[:3] == ["--query-print-filament-profiles", "--printer-profile", name]


# --------------------------------------------------------------------------- shaping the answers


async def test_list_printers_maps_the_query_fields(prusaslicer_query):
    # Given PrusaSlicer's JSON for one model with one variant
    prusaslicer_query.state.payload = PRINTER_PAYLOAD

    # When the printers are listed
    (model,) = await server.list_printers()

    # Then ids, the unescaped name, nozzle, extruders and bed dimensions are mapped to our fields
    (variant,) = model.variants
    assert (model.id, model.name, model.technology) == ("MAKER", "Maker & Co One", "FFF")
    assert variant.model_dump() == {
        "nozzle_diameter": 0.4,
        "printer_profile": "Maker One 0.4",
        "extruders": 2,
        "bed_width": 200,
        "bed_depth": 150,
        "max_print_height": 100,
    }


async def test_list_printers_leaves_bed_fields_empty_when_the_profile_has_no_bed(prusaslicer_query):
    # Given a printer profile without bed information
    prusaslicer_query.state.payload = {
        "printer_models": [
            {"id": "X", "name": "X", "variants": [{"name": 0.4, "printer_profiles": [{"name": "X 0.4"}]}]}
        ]
    }

    # When the printers are listed
    (model,) = await server.list_printers()

    # Then the missing values are None rather than an error
    variant = model.variants[0]
    assert (variant.bed_width, variant.bed_depth, variant.max_print_height, variant.extruders) == (
        None,
        None,
        None,
        None,
    )


async def test_list_printers_reports_the_requested_technology_when_the_model_omits_it(prusaslicer_query):
    # Given a model entry with no technology field, queried for SLA
    prusaslicer_query.state.payload = {"printer_models": [{"id": "R", "name": "Resin", "variants": []}]}

    # When SLA printers are listed
    (model,) = await server.list_printers("SLA")

    # Then the technology that was asked for is reported
    assert model.technology == "SLA"


async def test_list_printers_with_nothing_installed_returns_an_empty_list(prusaslicer_query):
    # Given a valid answer that has no printer models key at all
    prusaslicer_query.state.payload = {"unrelated": 1}

    # When the printers are listed
    printers = await server.list_printers()

    # Then the result is empty instead of an error
    assert printers == []


async def test_list_profiles_maps_print_profiles_and_their_filaments(prusaslicer_query):
    # Given PrusaSlicer's JSON with one print profile that has filaments and one that has none listed
    prusaslicer_query.state.payload = {
        "printer_profile": "Maker One 0.4",
        "print_profiles": [
            {"name": "0.20mm SPEED", "filament_profiles": ["PLA", "PETG"]},
            {"name": "0.10mm DETAIL"},
        ],
    }

    # When the profiles are listed
    profiles = await server.list_profiles("Maker One 0.4")

    # Then both are mapped, the second with an empty filament list
    assert profiles.printer_profile == "Maker One 0.4"
    assert [(p.name, p.filament_profiles) for p in profiles.print_profiles] == [
        ("0.20mm SPEED", ["PLA", "PETG"]),
        ("0.10mm DETAIL", []),
    ]


async def test_list_printers_includes_a_user_preset_once_including_models_without_variants(prusaslicer_query):
    # Given a vendor profile that is repeated as a user preset, a distinct user preset, and a model
    # that has no nozzle variants (PrusaSlicer 2.9 puts its profiles on the model itself)
    prusaslicer_query.state.payload = {
        "printer_models": [
            {
                "id": "MAKER",
                "name": "Maker",
                "variants": [
                    {
                        "name": 0.4,
                        "printer_profiles": [
                            {
                                "name": "Maker One 0.4",
                                "extruders_cnt": 1,
                                "bed": {"width": 200, "height": 150, "max_print_height": 100},
                            }
                        ],
                        "user_printer_profiles": [
                            {"name": "Maker One 0.4", "bed": {"width": 1, "height": 1, "max_print_height": 1}},
                            {
                                "name": "Shop",
                                "extruders_cnt": 1,
                                "bed": {"width": 220, "height": 220, "max_print_height": 250},
                            },
                        ],
                    }
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

    # When the printers are listed
    printers = {p.id: p for p in await server.list_printers()}

    # Then the duplicated name is listed once, with the vendor bed, and the variant-less user preset is included
    assert [v.printer_profile for v in printers["MAKER"].variants] == ["Maker One 0.4", "Shop"]
    assert printers["MAKER"].variants[0].bed_width == 200
    flat = printers["FLAT"].variants[0]
    assert flat.printer_profile == "Flat User"
    assert (flat.bed_width, flat.bed_depth, flat.max_print_height) == (100, 80, 40)


async def test_list_profiles_merges_user_presets_and_keeps_each_name_once(prusaslicer_query):
    # Given a print profile that also exists as a user preset, plus a user-only print profile
    prusaslicer_query.state.payload = {
        "printer_profile": "Maker One 0.4",
        "print_profiles": [
            {"name": "0.20mm SPEED", "filament_profiles": ["PLA"], "user_filament_profiles": ["PLA", "My PLA"]},
        ],
        "user_print_profiles": [
            {"name": "0.20mm SPEED", "filament_profiles": ["PETG"]},
            {"name": "Shop 0.20mm", "user_filament_profiles": ["Shop PLA"]},
        ],
    }

    # When the profiles are listed
    profiles = await server.list_profiles("Maker One 0.4")

    # Then each print profile name appears once, and filaments from both sources are kept
    assert [(p.name, p.filament_profiles) for p in profiles.print_profiles] == [
        ("0.20mm SPEED", ["PLA", "My PLA", "PETG"]),
        ("Shop 0.20mm", ["Shop PLA"]),
    ]


async def test_list_profiles_echoes_the_requested_printer_when_the_answer_omits_it(prusaslicer_query):
    # Given an answer without the printer_profile field
    prusaslicer_query.state.payload = {"print_profiles": []}

    # When the profiles are listed
    profiles = await server.list_profiles("Asked For")

    # Then the name that was asked for is echoed back
    assert profiles.printer_profile == "Asked For"


def test_query_output_is_read_as_utf_8_whatever_the_machine_default_is(tmp_path):
    # Given PrusaSlicer's JSON with a non-ASCII profile name, written as UTF-8 as PrusaSlicer does, and a Python
    # whose default text encoding is ASCII (what a Windows code page is to UTF-8: not the same)
    payload = tmp_path / "payload.json"
    payload.write_bytes(json.dumps({"name": "Prusament PLA \u00ae Galaxy \u00e9"}, ensure_ascii=False).encode())
    program = textwrap.dedent(
        f"""
        import asyncio, json, sys
        from pathlib import Path
        from types import SimpleNamespace
        from prusaslicer_mcp import cli, server

        server._config = SimpleNamespace()

        async def fake_run(config, args, timeout=None):
            Path(args[args.index("-o") + 1]).write_bytes(Path({str(payload)!r}).read_bytes())
            return cli.CliResult(1, "", "")

        cli.run = fake_run
        result = asyncio.run(server._query_json(["--query-printer-models"]))
        sys.stdout.buffer.write(json.dumps(result, ensure_ascii=True).encode())
        """
    )
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "PYTHONIOENCODING": "ascii"}

    # When the query result is read in that Python
    done = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, check=False)

    # Then the name arrives intact
    assert done.returncode == 0, done.stderr.decode()
    assert json.loads(done.stdout) == {"name": "Prusament PLA \u00ae Galaxy \u00e9"}
