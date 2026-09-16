import numpy as np
import pytest
from PIL import Image
from pixelperfect.image_io import load_image
from pixelperfect.scoring import render_grid, prepare_evidence, _robust_error, expression_complexity
from pixelperfect import Config


def test_fractional_area_reconstruction_and_alpha():
    cells = np.array([[[1, 0, 0, 1], [0, 1, 0, 0]]], np.float32)
    rendered = render_grid(cells, [0, 1.5, 3], [0, 1], (1, 3))
    np.testing.assert_allclose(rendered[0], [[1, 0, 0, 1], [.5, 0, 0, .5], [0, 0, 0, 0]])


def test_hidden_rgb_not_scored():
    a = np.zeros((3, 4, 4), np.float32)
    b = a.copy()
    b[..., :3] = 1
    assert _robust_error(prepare_evidence(a)["premultiplied"], prepare_evidence(b)["premultiplied"]) == 0


def test_spatial_complexity_not_only_histogram():
    a = np.ones((8, 8, 4), np.float32)
    a[:4, :, :3] = 0
    b = np.ones_like(a)
    b[np.indices((8, 8)).sum(0) % 2 == 0, :3] = 0
    ra, _ = expression_complexity(a, 1024)
    rb, _ = expression_complexity(b, 1024)
    assert rb > ra


def test_exif_orientation():
    im = Image.new("RGB", (20, 10))
    im.getexif()[274] = 6
    assert load_image(im).rgba.shape == (20, 10, 4)


@pytest.mark.parametrize("kwargs", [{"pixel_size":0}, {"pixel_size":float("nan")}, {"scale":0},
                                  {"target_size":(0,4)}, {"colors":257}, {"max_pixel_size":1},
                                  {"sampling":"bad"}, {"local_warp":"bad"},
                                  {"pixel_size":4,"target_size":(8,8)}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_bad_float_image():
    with pytest.raises(ValueError):
        load_image(np.ones((2, 2, 3), np.float32) * np.nan)


def test_lower_bound_pruning_records_real_unknowns():
    from pixelperfect import pixelize
    truth = np.random.default_rng(131).integers(0, 256, (10, 12, 3), dtype=np.uint8)
    source = np.repeat(np.repeat(truth, 8, axis=0), 8, axis=1)
    result = pixelize(source, Config(local_warp="off"))
    np.testing.assert_array_equal(np.asarray(result.image), truth)
    skipped = [c for c in result.diagnostics["candidates"] if c["scores"].get("status") == "pruned_lower_bound"]
    assert skipped
    for candidate in skipped:
        assert candidate["scores"]["total"] is None
        assert candidate["scores"]["lower_bound"] > result.diagnostics["selected_score"]["total"]
    assert result.diagnostics["score_gap_is_lower_bound"]


def test_constant_fallback_still_honors_palette_request():
    from pixelperfect import pixelize
    result = pixelize(Image.new("RGBA", (7, 5), (50, 90, 30, 0)), Config(colors=2))
    assert result.image.size == (7, 5)
    assert np.asarray(result.image).sum() == 0


def test_square_target_does_not_silently_use_rectangular_spacing():
    from pixelperfect import pixelize
    with pytest.raises(ValueError, match="square"):
        pixelize(Image.new("RGB", (80, 81)), Config(target_size=(10, 10), square=True))
