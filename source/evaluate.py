"""Reproducible generated-data evaluation; no downloads or external image data.

Run from the project directory: python source/evaluate.py
All artifacts are written under the explicitly selected input/output directories.
The automatic quality probes are reported honestly, even when sizes differ.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import PIL
from PIL import Image, ImageFilter
import scipy

from pixelperfect import Config, pixelize, save_result


PROJECT = Path(__file__).resolve().parents[1]

# Preserved measurements from the first actual evaluator run in this workspace.
# This historical record is deliberately immutable: rerunning the script measures
# the current implementation below, rather than remeasuring the old algorithm.
PRE_OPTIMIZATION_BENCHMARK = {
    "provenance": "Measured in this workspace on 2026-09-16 by the first execution of source/evaluate.py, before shared sampling-gradient reuse, safe lower-bound candidate pruning, and vectorized single-mode sampling.",
    "input_size": [1024, 1024], "output_size": [64, 64],
    "wall_seconds": 20.845036999999138,
    "stage_seconds": {
        "read_preprocess": 0.020941900000252645,
        "features": 0.5095590000000811,
        "candidate_search": 0.1003342000003613,
        "local_warp": 0.17297310000139987,
        "sampling": 17.378461500000412,
        "palette": 0.0000073999999585794285,
        "scoring": 2.6595107999983156,
        "total": 20.842087299999548,
    },
    "rgb_mae": 0.0, "alpha_mae": 0.0, "exact_visible_pixels": 1.0,
    "note": "Historical single warm-process run on the reported workstation; separate runs are subject to machine-load variability. This is not a controlled statistical speedup experiment.",
}


def make_truth(width=24, height=24, seed=63, rgba=False):
    """Distinct cells plus known four-neighbor line, dot, and enclosed hole."""
    rng = np.random.default_rng(seed)
    palette = np.array([[28, 33, 58], [228, 70, 84], [242, 203, 93],
                        [42, 156, 148], [112, 78, 171], [225, 226, 232]], np.uint8)
    data = palette[rng.integers(0, len(palette), size=(height, width))]
    # Broad identical runs suppress many otherwise visible grid boundaries.
    if min(width, height) >= 12:
        data[4:10, 4:11] = palette[0]
        data[5:10, 5] = palette[1]
        data[9, 5:10] = palette[1]
        data[6, 9] = [255, 255, 244]
        data[12:17, 12:17] = palette[3]
        data[14, 14] = palette[0]
    if rgba:
        alpha = np.full((height, width, 1), 255, np.uint8)
        alpha[:2] = 0
        alpha[-2:] = 0
        alpha[:, :2] = 0
        alpha[:, -2:] = 0
        if width > 14 and height > 14:
            alpha[14, 14] = 0
        data = np.concatenate((data, alpha), axis=-1)
        data[data[..., 3] == 0, :3] = rng.integers(0, 256, (np.sum(alpha == 0), 3), dtype=np.uint8)
    return Image.fromarray(data)


def nn(image, factor):
    return image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)


def jpeg(image, quality=72):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def generated_cases():
    truth = make_truth()
    clean = nn(truth, 8)
    repeated = np.asarray(truth).copy()
    repeated[:, 6:16] = repeated[:, 6:7]
    texture = np.asarray(clean).copy()
    texture[::3, ::3] = np.clip(texture[::3, ::3].astype(int) + 24, 0, 255)
    widths = np.tile([8, 9, 8, 7], 6)
    heights = np.tile([8, 8, 9, 7], 6)
    drift = Image.fromarray(np.repeat(np.repeat(np.asarray(truth), heights, 0), widths, 1))
    transparent_truth = make_truth(rgba=True)
    return [
        ("clean_integer", clean, truth, "sufficient"),
        ("noninteger_nearest", truth.resize((180, 180), Image.Resampling.NEAREST), truth, "quality"),
        ("noninteger_bilinear", truth.resize((180, 180), Image.Resampling.BILINEAR), truth, "quality"),
        ("shifted_crop", clean.crop((3, 2, 190, 191)), truth, "quality"),
        ("mild_blur", clean.filter(ImageFilter.GaussianBlur(0.65)), truth, "quality"),
        ("jpeg_noise", jpeg(clean), truth, "quality"),
        ("same_color_runs", nn(Image.fromarray(repeated), 8), Image.fromarray(repeated), "quality"),
        ("texture_interference", Image.fromarray(texture), truth, "quality"),
        ("bounded_drift", drift, truth, "quality"),
        ("transparent", nn(transparent_truth, 8), transparent_truth, "sufficient"),
    ]


def comparison(output, truth):
    if output.size != truth.size:
        return {"rgb_mae": None, "alpha_mae": None, "exact_visible_pixels": None}
    actual = np.asarray(output.convert("RGBA"), dtype=np.float32) / 255
    expected = np.asarray(truth.convert("RGBA"), dtype=np.float32) / 255
    alpha = expected[..., 3]
    visible = alpha > 0
    rgb_error = np.abs(actual[..., :3] - expected[..., :3])
    rgb_mae = float((rgb_error * alpha[..., None]).sum() / max(3 * alpha.sum(), 1) * 255)
    alpha_mae = float(np.abs(actual[..., 3] - alpha).mean() * 255)
    exact = np.all(np.abs(actual[..., :3] - expected[..., :3]) < 0.5 / 255, axis=-1)
    exact_fraction = float(exact[visible].mean()) if visible.any() else 1.0
    return {"rgb_mae": rgb_mae, "alpha_mae": alpha_mae, "exact_visible_pixels": exact_fraction}


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def evaluate(input_dir: Path, output_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "platform": platform.platform(), "python": sys.version,
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": os.cpu_count(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pillow": PIL.__version__,
        "thread_env": {key: os.environ.get(key) for key in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    automatic = []
    for name, source, truth, category in generated_cases():
        source.save(input_dir / f"{name}.png")
        truth.save(input_dir / f"{name}_truth.png")
        print(f"Evaluating automatic: {name}", flush=True)
        start = perf_counter()
        result = pixelize(source, Config())
        elapsed = perf_counter() - start
        save_result(result, output_dir / f"{name}.png")
        metrics = comparison(result.image, truth)
        record = {
            "name": name, "category": category, "source_size": source.size,
            "truth_size": truth.size, "output_size": result.image.size,
            "size_match": result.image.size == truth.size,
            "confidence": result.confidence, "wall_seconds": elapsed,
            "stage_seconds": result.timings, "grid": result.grid,
            "diagnostics": result.diagnostics, **metrics,
        }
        record["sufficient_case_pass"] = (
            record["size_match"] and metrics["rgb_mae"] is not None
            and metrics["rgb_mae"] <= 2 and metrics["alpha_mae"] <= 1
        ) if category == "sufficient" else None
        automatic.append(record)

    truth = make_truth(16, 16, seed=35)
    clean = nn(truth, 9)
    impulses = np.asarray(clean).copy()
    impulses[4::9, 4::9] = [255, 0, 255]
    sampling_cases = [
        ("clean", clean),
        ("center_impulses", Image.fromarray(impulses)),
        ("blur_and_jpeg", jpeg(clean.filter(ImageFilter.GaussianBlur(0.6)), quality=70)),
    ]
    sampling = []
    for name, source in sampling_cases:
        source.save(input_dir / f"sampling_{name}.png")
        for method in ("center", "median", "robust"):
            print(f"Evaluating sampling: {name}/{method}", flush=True)
            start = perf_counter()
            result = pixelize(source, Config(target_size=truth.size, sampling=method, local_warp="off"))
            elapsed = perf_counter() - start
            save_result(result, output_dir / f"sampling_{name}_{method}.png")
            sampling.append({"case": name, "method": method, "wall_seconds": elapsed,
                             "output_size": result.image.size, **comparison(result.image, truth)})
    truth.save(input_dir / "sampling_truth.png")

    # One measured 1024x1024 automatic run, after the smaller cases warm imports.
    benchmark_truth = make_truth(64, 64, seed=1729)
    benchmark_source = nn(benchmark_truth, 16)
    benchmark_source.save(input_dir / "benchmark_1024.png")
    benchmark_truth.save(input_dir / "benchmark_truth.png")
    print("Benchmark: one 1024x1024 automatic run", flush=True)
    start = perf_counter()
    result = pixelize(benchmark_source, Config())
    elapsed = perf_counter() - start
    save_result(result, output_dir / "benchmark_1024.png")
    benchmark = {
        "input_size": benchmark_source.size, "truth_size": benchmark_truth.size,
        "output_size": result.image.size, "wall_seconds": elapsed,
        "stage_seconds": result.timings, "confidence": result.confidence,
        "note": "One warm-process run; includes pixelize only, excludes disk PNG read/write. No real-time claim.",
        **comparison(result.image, benchmark_truth),
    }

    report = {"environment": environment, "automatic": automatic, "sampling": sampling,
              "benchmark": benchmark, "pre_optimization_benchmark": PRE_OPTIMIZATION_BENCHMARK,
              "methodology": "Generated truth; sufficient evidence is asserted only for clean integer and clean RGBA inputs. Other automatic cases are quality probes. RGB MAE is alpha weighted and reported only at matching output dimensions. Connectivity: four neighbors for source-supported line/hole checks."}
    report = serializable(report)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "EVALUATION.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"Report: {output_dir / 'EVALUATION.md'}")
    print(f"1024x1024 wall time: {elapsed:.3f} s; output {result.image.width}x{result.image.height}")
    return report


def fmt(value, decimals=3):
    return "n/a (different dimensions)" if value is None else f"{value:.{decimals}f}"


def dimensions(value):
    return "x".join(map(str, value))


def markdown_report(report):
    env, benchmark = report["environment"], report["benchmark"]
    lines = [
        "# Generated-data evaluation", "",
        "This report contains measured results. No external images or downloaded datasets are used.", "",
        "## Environment", "",
        f"- OS: {env['platform']}", f"- CPU: {env['processor']} ({env['logical_cpu_count']} logical CPUs)",
        f"- Python: {env['python'].splitlines()[0]}",
        f"- NumPy {env['numpy']}; SciPy {env['scipy']}; Pillow {env['pillow']}",
        f"- Thread environment: `{env['thread_env']}`", "",
        "## Automatic grid recovery", "",
        "Clean integer enlargement and clean RGBA provide sufficient grid evidence. Other cases are quality probes: blur, crop, repeated colors, and resampling can lose information or admit multiple scales. A matching dimension alone does not prove correct recovery.", "",
        "RGB MAE is measured in 0–255 units, weighted by truth alpha. Hidden transparent RGB never contributes. When dimensions differ, the error is deliberately not computed by resizing the result.", "",
        "| Case | Category | Input | Truth | Output | RGB MAE | Alpha MAE | Confidence* | Time (s) |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in report["automatic"]:
        lines.append(f"| {r['name']} | {r['category']} | {dimensions(r['source_size'])} | {dimensions(r['truth_size'])} | {dimensions(r['output_size'])} | {fmt(r['rgb_mae'])} | {fmt(r['alpha_mae'])} | {r['confidence']:.3f} | {r['wall_seconds']:.3f} |")
    lines += ["", "*Confidence is an uncalibrated heuristic score, not a probability.", "",
              "## Sampling comparison", "",
              "All three samplers use the same explicit 16x16 target grid, with local warp disabled and no palette reduction. This isolates sampling quality from automatic scale selection.", "",
              "| Input | Sampler | RGB MAE | Exact visible pixels | Time (s) |",
              "|---|---|---:|---:|---:|"]
    for r in report["sampling"]:
        exact = "n/a" if r["exact_visible_pixels"] is None else f"{100 * r['exact_visible_pixels']:.2f}%"
        lines.append(f"| {r['case']} | {r['method']} | {fmt(r['rgb_mae'])} | {exact} | {r['wall_seconds']:.3f} |")
    impulses = {r["method"]: r for r in report["sampling"] if r["case"] == "center_impulses"}
    outcome = "passed" if impulses["robust"]["rgb_mae"] < impulses["center"]["rgb_mae"] * 0.25 else "FAILED"
    lines += ["", f"Center-impulse robustness check (robust MAE < 25% of center MAE): **{outcome}**.", "",
              "## Performance", "",
              f"One 1024x1024 automatic run took **{benchmark['wall_seconds']:.3f} seconds** and produced **{dimensions(benchmark['output_size'])}** (truth: {dimensions(benchmark['truth_size'])}).",
              benchmark["note"], "", "| Stage | Seconds |", "|---|---:|"]
    lines += [f"| {key} | {value:.6f} |" for key, value in benchmark["stage_seconds"].items()]
    baseline = report["pre_optimization_benchmark"]
    lines += ["", "### Historical measurement before optimization", "",
              baseline["provenance"], "",
              "The implementation now reuses one sampling-gradient map, prunes candidates only when a safe lower bound proves that they cannot win, and vectorizes the common single-color-mode sampling path. Skipped candidates retain their bounds and parameters in diagnostics; an unevaluated final score is null, not fabricated.", "",
              "| Separate run | Wall seconds | Sampling seconds | Output | RGB MAE |",
              "|---|---:|---:|---|---:|",
              f"| Before optimization (historical) | {baseline['wall_seconds']:.3f} | {baseline['stage_seconds']['sampling']:.3f} | {dimensions(baseline['output_size'])} | {baseline['rgb_mae']:.3f} |",
              f"| Current implementation | {benchmark['wall_seconds']:.3f} | {benchmark['stage_seconds']['sampling']:.3f} | {dimensions(benchmark['output_size'])} | {fmt(benchmark['rgb_mae'])} |", "",
              baseline["note"]]
    sufficient_failures = [r for r in report["automatic"] if r["sufficient_case_pass"] is False]
    wrong_size = [r for r in report["automatic"] if not r["size_match"]]
    lines += ["", "## Verified behavior and unresolved cases", "",
              f"- Sufficient-evidence automatic failures: {len(sufficient_failures)}.",
              "- Exact output dimensions, alpha invariance, tiny/constant inputs, deterministic repetition, CLI side effects, and source-supported one-cell lines/dots/holes are covered separately by pytest.",
              "- Structure checks use four-neighbor connectivity; a diagonal contact is not counted as a line connection.",
              "- These generated examples are a development benchmark, not a held-out natural-image benchmark or evidence of unique recovery from arbitrary AI art."]
    if wrong_size:
        lines.append("- Automatic dimension mismatches (retained in this report): " + ", ".join(r["name"] for r in wrong_size) + ".")
    else:
        lines.append("- All listed automatic cases matched the generating grid dimensions in this run; this does not resolve scale ambiguity in other images.")
    if sufficient_failures:
        lines.append("- Required clean cases that failed: " + ", ".join(r["name"] for r in sufficient_failures) + ".")
    lines += ["- Bilinear noninteger scaling and JPEG can change true cell colors irreversibly; a nonzero reconstruction/color error is expected.",
              "- Repeating texture can dominate periodic evidence. A successful synthetic texture probe does not imply general resistance.",
              "- Full candidate parameters and component scores are preserved in evaluation.json for inspection.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "input" / "synthetic")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output" / "evaluation")
    args = parser.parse_args()
    report = evaluate(args.input_dir.resolve(), args.output_dir.resolve())
    return 1 if any(r["sufficient_case_pass"] is False for r in report["automatic"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
