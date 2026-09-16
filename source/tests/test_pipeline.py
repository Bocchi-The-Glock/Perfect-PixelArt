"""End-to-end requirements using generated inputs with known evidence strength."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from pixelperfect import Config, pixelize, save_result


def truth_image(width=12, height=10, seed=23, rgba=False):
    rng = np.random.default_rng(seed)
    palette = np.array([[25, 35, 60], [235, 82, 73], [245, 210, 92],
                        [55, 165, 155], [118, 82, 183], [230, 230, 235]], np.uint8)
    data = palette[rng.integers(0, len(palette), size=(height, width))]
    if rgba:
        alpha = np.full((height, width, 1), 255, np.uint8)
        alpha[:2] = 0
        alpha[:, :2] = 0
        data = np.concatenate([data, alpha], axis=2)
    return Image.fromarray(data)


def enlarged(image, factor=8):
    return image.resize((image.width * factor, image.height * factor), Image.Resampling.NEAREST)


def test_clean_automatic_grid_dimensions_and_colors():
    truth = truth_image()
    result = pixelize(enlarged(truth), Config(local_warp="off"))
    assert result.image.size == truth.size, result.diagnostics
    assert np.abs(np.asarray(result.image, dtype=float) - np.asarray(truth, dtype=float)).max() <= 2
    assert 0 <= result.confidence <= 1
    assert result.grid
    assert result.timings and all(value >= 0 for value in result.timings.values())


def test_target_size_and_nearest_neighbor_export(tmp_path):
    truth = truth_image(8, 6)
    result = pixelize(enlarged(truth, 7), Config(target_size=truth.size))
    assert result.image.size == (8, 6)
    destination = tmp_path / "enlarged.png"
    save_result(result, destination, scale=4)
    with Image.open(destination) as saved:
        assert saved.size == (32, 24)
        expected = np.asarray(result.image.resize(saved.size, Image.Resampling.NEAREST))
        np.testing.assert_array_equal(np.asarray(saved), expected)


def test_seed_free_repeatability():
    truth = truth_image(9, 7)
    image = enlarged(truth, 9).filter(ImageFilter.GaussianBlur(0.45))
    config = Config(target_size=truth.size, colors=5)
    first = pixelize(image, config)
    second = pixelize(image, config)
    np.testing.assert_array_equal(np.asarray(first.image), np.asarray(second.image))
    assert first.grid == second.grid
    assert first.confidence == second.confidence


def test_fully_transparent_hidden_rgb_does_not_change_result():
    truth = truth_image(10, 8, rgba=True)
    clean = np.asarray(enlarged(truth), dtype=np.uint8).copy()
    changed = clean.copy()
    mask = changed[..., 3] == 0
    changed[mask, :3] = np.random.default_rng(91).integers(0, 256, (mask.sum(), 3), dtype=np.uint8)
    first = pixelize(clean, Config(target_size=truth.size))
    second = pixelize(changed, Config(target_size=truth.size))
    a, b = np.asarray(first.image), np.asarray(second.image)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[..., 3] == 0, np.asarray(truth)[..., 3] == 0)
    assert first.confidence == second.confidence


def test_single_pixel_line_highlight_and_hole_are_not_removed():
    # All structures are one *true grid cell* wide, so their support is unambiguous.
    data = np.full((11, 11, 4), [35, 45, 80, 255], np.uint8)
    data[1:10, 2] = [235, 50, 65, 255]
    data[7, 2:8] = [235, 50, 65, 255]
    data[3, 8] = [255, 250, 225, 255]
    data[5, 5] = [0, 0, 0, 0]
    result = pixelize(enlarged(Image.fromarray(data)), Config(target_size=(11, 11)))
    out = np.asarray(result.image)
    assert np.max(np.abs(out[1:10, 2, :3].astype(int) - data[1:10, 2, :3])) <= 2
    assert np.max(np.abs(out[3, 8, :3].astype(int) - data[3, 8, :3])) <= 2
    assert out[5, 5, 3] == 0


def test_robust_sampling_resists_center_impulses_better_than_center():
    truth = truth_image(8, 6)
    image = np.asarray(enlarged(truth, 9)).copy()
    for row in range(truth.height):
        for column in range(truth.width):
            image[row * 9 + 4, column * 9 + 4] = [255, 0, 255]
    target = np.asarray(truth).astype(float)
    errors = {}
    for sampling in ("center", "median", "robust"):
        result = pixelize(image, Config(target_size=truth.size, sampling=sampling, local_warp="off"))
        errors[sampling] = np.mean(np.abs(np.asarray(result.image).astype(float) - target))
    assert errors["robust"] < errors["center"] * 0.25, errors
    assert errors["robust"] < 5, errors


def test_constant_image_falls_back_without_inventing_grid():
    image = Image.new("RGB", (37, 29), (65, 95, 120))
    result = pixelize(image, Config())
    assert result.image.size == image.size
    assert result.confidence < 0.5
    assert result.diagnostics
    np.testing.assert_array_equal(np.asarray(result.image), np.asarray(image))


@pytest.mark.parametrize("size", [(1, 1), (1, 5), (2, 2)])
def test_tiny_inputs_have_defined_nonempty_outputs(size):
    result = pixelize(Image.new("RGBA", size, (15, 45, 80, 0)), Config())
    assert result.image.width >= 1 and result.image.height >= 1
    assert np.asarray(result.image)[..., 3].max() == 0
    assert result.confidence < 0.5


@pytest.mark.parametrize("kind", ["noninteger", "blur", "jpeg", "crop", "same_color_runs", "texture", "drift"])
def test_degraded_or_ambiguous_inputs_report_diagnostics_and_remain_valid(kind):
    """These cases are quality probes, not assertions of unique automatic recovery."""
    truth = truth_image(10, 8)
    source = enlarged(truth)
    if kind == "noninteger":
        source = truth.resize((75, 60), Image.Resampling.BILINEAR)
    elif kind == "blur":
        source = source.filter(ImageFilter.GaussianBlur(0.7))
    elif kind == "jpeg":
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        source = Image.open(buffer).convert("RGB")
    elif kind == "crop":
        source = source.crop((3, 2, source.width - 2, source.height - 1))
    elif kind == "same_color_runs":
        data = np.asarray(truth).copy()
        data[:, 3:7] = data[:, 3:4]
        source = enlarged(Image.fromarray(data))
    elif kind == "texture":
        data = np.asarray(source).copy()
        data[::3, ::3] = np.clip(data[::3, ::3].astype(int) + 25, 0, 255)
        source = Image.fromarray(data)
    elif kind == "drift":
        data = np.asarray(truth)
        widths = np.array([8, 9, 8, 7, 8, 9, 8, 7, 8, 8])
        heights = np.array([8, 9, 8, 7, 8, 9, 8, 7])
        source = Image.fromarray(np.repeat(np.repeat(data, heights, axis=0), widths, axis=1))
    result = pixelize(source, Config())
    assert 0 <= result.confidence <= 1
    assert result.image.width > 0 and result.image.height > 0
    assert result.image.width <= source.width and result.image.height <= source.height
    assert result.diagnostics and result.timings


def test_numpy_float_and_pillow_inputs_agree():
    image = enlarged(truth_image(6, 5))
    config = Config(target_size=(6, 5))
    as_pillow = pixelize(image, config)
    as_float = pixelize(np.asarray(image, dtype=np.float32) / 255, config)
    np.testing.assert_array_equal(np.asarray(as_pillow.image), np.asarray(as_float.image))


def test_corrupt_input_is_rejected(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not an image")
    with pytest.raises((ValueError, OSError)):
        pixelize(path, Config())


@pytest.mark.parametrize("values", [
    {"pixel_size": 0}, {"target_size": (0, 4)}, {"scale": 0},
    {"colors": 0}, {"sampling": "unknown"}, {"local_warp": "unknown"},
    {"pixel_size": 8, "target_size": (8, 8)},
    {"min_pixel_size": 16, "max_pixel_size": 8},
])
def test_invalid_configuration_is_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        pixelize(Image.new("RGB", (64, 64)), Config(**values))


def test_palette_budget_counts_visible_colors_only():
    truth = truth_image(9, 7, rgba=True)
    result = pixelize(enlarged(truth), Config(target_size=truth.size, colors=3))
    data = np.asarray(result.image)
    assert data.shape[2] == 4
    colors = np.unique(data[data[..., 3] > 0, :3], axis=0)
    assert len(colors) <= 3
    np.testing.assert_array_equal(data[..., 3] == 0, np.asarray(truth)[..., 3] == 0)


def test_semitransparent_alpha_and_straight_color_survive_clean_recovery():
    data = np.asarray(truth_image(6, 4, rgba=True)).copy()
    data[2:, 2:, 3] = np.array([[64, 128, 192, 255], [255, 192, 128, 64]], np.uint8)
    result = pixelize(enlarged(Image.fromarray(data)), Config(target_size=(6, 4)))
    out = np.asarray(result.image)
    visible = data[..., 3] > 0
    assert np.abs(out[..., 3].astype(int) - data[..., 3]).max() <= 1
    assert np.abs(out[visible, :3].astype(int) - data[visible, :3]).max() <= 2


def test_conflicting_target_aspect_ratio_is_explicitly_rejected():
    with pytest.raises(ValueError, match="aspect ratio"):
        pixelize(Image.new("RGB", (80, 60)), Config(target_size=(8, 8)))
