"""Bounded single-image restoration pipeline; no CLI or filesystem side effects."""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from PIL import Image
from .config import Config
from .image_io import load_image, to_pil
from .features import extract_features
from .grid import generate_candidates, refine_candidate
from .sampling import recover_cells, improve_structure, sampling_gradient
from .palette import quantize_cells
from .scoring import prepare_evidence, score_candidate


@dataclass
class PixelizeResult:
    image: Image.Image
    grid: dict
    confidence: float
    timings: dict
    diagnostics: dict
    cell_confidence: np.ndarray = field(repr=False)


def _grid_dict(g):
    return {"sx": float(g.sx), "sy": float(g.sy), "phase_x": float(g.phase_x),
            "phase_y": float(g.phase_y), "x_lines": np.asarray(g.x_lines).tolist(),
            "y_lines": np.asarray(g.y_lines).tolist(), "warped": bool(g.warped),
            "output_size": [len(g.x_lines) - 1, len(g.y_lines) - 1],
            "coverage": "full input; half-open source pixel boxes", "metadata": g.metadata}


def pixelize(image, config=None):
    """Return a native-resolution image and diagnostics, writing no files.

    Config.scale controls CLI export only; the API image always has native size.
    """
    config = Config() if config is None else config
    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")
    start = perf_counter()
    timings = {}
    data = load_image(image)
    rgba = data.rgba
    h, w = rgba.shape[:2]
    timings["read_preprocess"] = perf_counter() - start
    if config.target_size is not None:
        tw, th = config.target_size
        if tw > w or th > h:
            raise ValueError("target_size cannot exceed the input dimensions; use scale to enlarge")
        if config.square and abs(w / tw - h / th) > 1e-8:
            raise ValueError("square mode requires target_size with exactly equal source x/y spacing")
        # Explicit policy: exact aspect ratio, with at most one input-pixel
        # rounding tolerance. The image is never silently cropped or stretched.
        aspect_error = abs(w / tw - h / th)
        if aspect_error > max(1 / tw, 1 / th) + 1e-8:
            raise ValueError("target_size aspect ratio conflicts with input (more than one source pixel); choose a proportional size")
    t = perf_counter()
    features = extract_features(rgba)
    evidence = prepare_evidence(rgba)
    gradient = sampling_gradient(rgba) if config.sampling == "robust" else None
    timings["features"] = perf_counter() - t
    t = perf_counter()
    candidates = generate_candidates(features, (h, w), config)
    timings["candidate_search"] = perf_counter() - t
    manual = config.pixel_size is not None or config.target_size is not None
    warnings = []
    if not candidates:
        warnings.append("No reliable grid evidence; preserved original dimensions (low confidence).")
        fallback_rgba = rgba
        if config.colors is not None:
            # A no-grid fallback still honors explicitly requested quantization.
            from .sampling import CellResult
            cells = CellResult(rgba=rgba.copy(), confidence=np.ones((h, w), np.float32))
            fallback_rgba = quantize_cells(cells, config.colors).rgba
        timings["total"] = perf_counter() - start
        return PixelizeResult(to_pil(fallback_rgba, data.has_alpha),
                              {"sx": 1.0, "sy": 1.0, "phase_x": 0.0, "phase_y": 0.0,
                               "x_lines": list(range(w + 1)), "y_lines": list(range(h + 1)),
                               "output_size": [w, h], "warped": False, "fallback": True},
                              0.0, timings, {"warnings": warnings, "candidates": [],
                                            "confidence_kind": "uncalibrated heuristic score", "fallback": True},
                              np.zeros((h, w), np.float32))
    results = []
    pruned = []
    for name in ("local_warp", "sampling", "palette", "scoring"):
        timings[name] = 0.0
    for candidate in candidates[:config.max_candidates]:
        # Every term in J is nonnegative, and R includes this unavoidable cell
        # count term. This bound can reject a finer candidate without sampling
        # it. Retain its parameters and bound in diagnostics, never invent a D.
        lower_bound = (config.complexity_weight * (len(candidate.x_lines) - 1)
                       * (len(candidate.y_lines) - 1) / (h * w))
        incumbent = min(results, key=lambda r: r[0]["total"]) if results else None
        if incumbent is not None and lower_bound > incumbent[0]["total"] + 1e-12:
            pruned.append({"grid": _grid_dict(candidate), "scores": {
                "status": "pruned_lower_bound", "total": None, "lower_bound": float(lower_bound),
                "reason": "cell-count complexity alone exceeds incumbent total cost"},
                "support": float(candidate.support), "strip_agreement": float(candidate.strip_agreement)})
            continue
        candidate_timings = {}
        t = perf_counter()
        grid = refine_candidate(candidate, features, config) if config.local_warp == "auto" else candidate
        candidate_timings["local_warp"] = perf_counter() - t
        timings["local_warp"] += candidate_timings["local_warp"]
        t = perf_counter()
        cells = recover_cells(rgba, grid.x_lines, grid.y_lines,
                              method=config.sampling, max_samples=config.max_samples, gradient=gradient)
        if config.sampling == "robust":
            cells = improve_structure(cells, rgba, grid.x_lines, grid.y_lines)
        candidate_timings["sampling"] = perf_counter() - t
        timings["sampling"] += candidate_timings["sampling"]
        t = perf_counter()
        if config.colors is not None:
            cells = quantize_cells(cells, config.colors)
        candidate_timings["palette"] = perf_counter() - t
        timings["palette"] += candidate_timings["palette"]
        t = perf_counter()
        score = score_candidate(evidence, cells, grid, config)
        candidate_timings["scoring"] = perf_counter() - t
        timings["scoring"] += candidate_timings["scoring"]
        score["timings_seconds"] = candidate_timings
        results.append((score, grid, cells))
    results.sort(key=lambda item: (item[0]["total"], item[2].rgba.shape[0] * item[2].rgba.shape[1], item[1].sx))
    best_score, best_grid, best_cells = results[0]
    rival_bounds = [r[0]["total"] for r in results[1:]] + [p["scores"]["lower_bound"] for p in pruned]
    gap = max(0., min(rival_bounds) - best_score["total"]) if rival_bounds else 0.0
    margin = float(np.clip(gap / 0.015, 0, 1))
    fit = float(np.exp(-best_score["reconstruction"] / 0.035))
    effective_min = max(1.25, config.min_pixel_size)
    bounds_hit = (not manual and (min(best_grid.sx, best_grid.sy) <= effective_min * 1.02
                                  or best_grid.sx >= min(config.max_pixel_size, w / 2) * 0.98
                                  or best_grid.sy >= min(config.max_pixel_size, h / 2) * 0.98))
    confidence = float(np.clip(0.35 * best_grid.support + 0.25 * best_grid.strip_agreement
                               + 0.2 * fit + 0.15 * margin + 0.05, 0, 1))
    # A great fit of an unsupported hypothesis is not evidence for its grid.
    confidence *= min(1.0, max(0.0, best_grid.support) / 0.3)
    if bounds_hit:
        confidence *= 0.7
        warnings.append("Best grid touches the search range boundary; consider changing the scale range.")
    conservative = False
    if confidence < config.confidence_threshold:
        warnings.append("Low heuristic grid confidence; the original grid may be ambiguous.")
        if not manual:
            # Only choose a less aggressive candidate if its score is close;
            # do not silently replace a well-supported coarse solution.
            close = [r for r in results if r[0]["total"] <= best_score["total"] + 0.006]
            chosen = max(close, key=lambda r: r[2].rgba.shape[0] * r[2].rgba.shape[1])
            conservative = chosen[1] is not best_grid
            best_score, best_grid, best_cells = chosen
            if conservative:
                fit = float(np.exp(-best_score["reconstruction"] / 0.035))
                # Keep a conservative score for a non-winning candidate; no
                # positive winner margin is attributed to this fallback.
                margin = 0.0
                confidence = min(confidence, float((.35 * best_grid.support + .25 * best_grid.strip_agreement
                                                    + .2 * fit + .05) * min(1., best_grid.support / .3)))
    diagnostics = {
        "confidence_kind": "uncalibrated heuristic score", "warnings": warnings,
        "fallback": False, "manual_constraint": manual, "conservative_selection": conservative,
        "range_boundary": bounds_hit, "score_gap": gap,
        "score_gap_is_lower_bound": bool(pruned),
        "confidence_components": {"support": best_grid.support, "strip_agreement": best_grid.strip_agreement,
                                  "fit": fit, "margin": margin},
        "selected_score": best_score, "structure_checks": best_cells.structure,
        "candidates": [{"grid": _grid_dict(g), "scores": s,
                        "support": float(g.support), "strip_agreement": float(g.strip_agreement)}
                       for s, g, c in results] + pruned,
    }
    output = to_pil(best_cells.rgba, data.has_alpha)
    if config.target_size is not None and output.size != config.target_size:
        raise RuntimeError("internal grid size invariant violated")
    timings["total"] = perf_counter() - start
    return PixelizeResult(output, _grid_dict(best_grid), confidence, timings, diagnostics, best_cells.confidence)
