"""Subprocess tests exercise the actual documented single-image entry point."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "pixelperfect.py"
if not SCRIPT.exists():
    SCRIPT = SOURCE.parent / "pixelperfect.py"


def invoke(*arguments, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


def make_input(path):
    rng = np.random.default_rng(18)
    low = Image.fromarray(rng.integers(20, 235, (6, 8, 3), dtype=np.uint8))
    low.resize((64, 48), Image.Resampling.NEAREST).save(path)
    return low


def test_default_cli_creates_only_requested_png(tmp_path):
    source = tmp_path / "input.png"
    truth = make_input(source)
    output = tmp_path / "result.png"
    completed = invoke("-i", source, "-o", output, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert {p.name for p in tmp_path.iterdir()} == {"input.png", "result.png"}
    with Image.open(output) as result:
        assert result.format == "PNG"
        assert result.size == truth.size


def test_cli_target_size_and_scale(tmp_path):
    source = tmp_path / "input.png"
    make_input(source)
    output = tmp_path / "result.png"
    completed = invoke("-i", source, "-o", output, "--target-size", "8x6", "--scale", "4", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    with Image.open(output) as result:
        assert result.size == (32, 24)
        data = np.asarray(result)
        np.testing.assert_array_equal(data, np.repeat(np.repeat(data[::4, ::4], 4, axis=0), 4, axis=1))


def test_cli_explicit_debug_and_verbose(tmp_path):
    source = tmp_path / "input.png"
    make_input(source)
    output = tmp_path / "result.png"
    debug = tmp_path / "debug"
    completed = invoke("-i", source, "-o", output, "--target-size", "8x6", "--debug-dir", debug,
                       "--verbose", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert list(debug.glob("*.json")), list(debug.iterdir())
    assert len(list(debug.glob("*.png"))) >= 2
    assert completed.stderr.strip()


def test_cli_rejects_mutually_exclusive_constraints(tmp_path):
    source = tmp_path / "input.png"
    make_input(source)
    completed = invoke("-i", source, "-o", tmp_path / "out.png", "--pixel-size", "8",
                       "--target-size", "8x6", cwd=tmp_path)
    assert completed.returncode != 0
    assert not (tmp_path / "out.png").exists()


def test_cli_corrupt_input_has_short_error(tmp_path):
    source = tmp_path / "corrupt.jpg"
    source.write_bytes(b"not jpeg")
    completed = invoke("-i", source, "-o", tmp_path / "out.png", cwd=tmp_path)
    assert completed.returncode != 0
    assert "Traceback" not in completed.stderr
    assert completed.stderr.strip()
    assert not (tmp_path / "out.png").exists()


def test_cli_constant_input_warns_and_still_writes(tmp_path):
    source = tmp_path / "constant.png"
    Image.new("RGB", (31, 29), "navy").save(source)
    output = tmp_path / "result.png"
    completed = invoke("-i", source, "-o", output, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.strip()
    with Image.open(output) as result:
        assert result.size == (31, 29)


def test_cli_noninteger_pixel_size_is_accepted(tmp_path):
    source = tmp_path / "input.png"
    make_input(source)
    completed = invoke("-i", source, "-o", tmp_path / "result.png", "--pixel-size", "8.0x8.0", cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
