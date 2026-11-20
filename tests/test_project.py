"""Run from the project folder:   python -m pytest -q
These tests protect the behaviour the team has agreed on. Run them before every merge.
"""
import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modules import REGISTRY                      # noqa: E402
from pipeline import load_routing, process_video  # noqa: E402

ROUTING = os.path.join(ROOT, "routing.yaml")


def textured(seed=0, size=(360, 480)):
    """A sharp, well-lit, textured test image."""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, (size[0], size[1], 3)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    for _ in range(25):
        x, y = int(rng.integers(0, size[1] - 60)), int(rng.integers(0, size[0] - 60))
        cv2.rectangle(img, (x, y), (x + 40, y + 40), tuple(float(v) for v in rng.integers(0, 255, 3)), -1)
    return img.clip(0, 255).astype(np.uint8)


def avg_score(module, images):
    m = REGISTRY[module]
    return float(np.mean([m.diagnose(i).score for i in images]))


IMAGES = [textured(s) for s in range(4)]


def test_lighting_score_rises_as_image_gets_darker():
    levels = [1.0, 0.6, 0.4, 0.25]
    scores = [avg_score("lighting", [(i * f).astype(np.uint8) for i in IMAGES]) for f in levels]
    assert scores == sorted(scores) and scores[-1] - scores[0] > 0.4, scores


def test_noise_score_rises_with_noise_level():
    rng = np.random.default_rng(1)
    scores = [avg_score("noise", [np.clip(i + rng.normal(0, s, i.shape), 0, 255).astype(np.uint8) for i in IMAGES])
              for s in (0, 5, 10, 20)]
    assert scores == sorted(scores) and scores[-1] - scores[0] > 0.3, scores


def test_blur_score_rises_with_blur_level():
    scores = [avg_score("sharpness", [cv2.GaussianBlur(i, (0, 0), s) if s else i for i in IMAGES])
              for s in (0, 1, 2, 4)]
    assert scores == sorted(scores) and scores[-1] - scores[0] > 0.3, scores


@pytest.mark.parametrize("name", ["lighting", "sharpness", "noise"])
def test_correct_keeps_size_and_type(name):
    m = REGISTRY[name]
    m.reset()
    frame = (IMAGES[0] * 0.4).astype(np.uint8)
    m.observe(frame)
    out = m.correct(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_lighting_fix_brightens_a_dark_frame():
    m = REGISTRY["lighting"]
    m.reset()
    dark = (IMAGES[0] * 0.3).astype(np.uint8)
    m.observe(dark)
    assert m.correct(dark).mean() > dark.mean() + 10


def test_temporal_denoising_reduces_noise():
    m = REGISTRY["noise"]
    m.reset()
    rng = np.random.default_rng(3)
    frames = [np.clip(IMAGES[0] + rng.normal(0, 10, IMAGES[0].shape), 0, 255).astype(np.uint8) for _ in range(6)]
    for f in frames:
        m.observe(f)
    before = m.diagnose(frames[-1]).details["noise_sigma"]
    after = m.diagnose(m.correct(frames[-1])).details["noise_sigma"]
    assert after < before * 0.7, (before, after)


def test_routing_table_is_complete_and_consistent():
    rows = load_routing(ROUTING)
    assert len(rows) == 16
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        if r["status"] == "implemented":
            assert r["module"] in REGISTRY and 0 < r["threshold"] <= 1 and isinstance(r["order"], int)
    orders = {r["id"]: r["order"] for r in rows if r["status"] == "implemented"}
    assert orders["grain"] < orders["poor_indoor_light"]      # denoise BEFORE brightening


def test_pipeline_end_to_end(tmp_path):
    src = str(tmp_path / "clip.mp4")
    w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 15, (160, 120))
    base = cv2.resize(IMAGES[0], (160, 120))
    for i in range(45):
        w.write((base * 0.3).astype(np.uint8))
    w.release()
    rep = process_video(src, str(tmp_path / "out"), sample_every=5, routing_path=ROUTING)
    assert len(rep["samples"]) == 9
    assert os.path.exists(rep["corrected_video"])
    assert {"summary", "stability", "speed"} <= set(rep)
    for s in rep["samples"]:
        assert all(0.0 <= v <= 1.0 for v in s["scores"].values())
        assert all(0.0 <= v <= 1.0 for v in s["scores_after"].values())
    light = [x for x in rep["summary"] if x["id"] == "poor_indoor_light"][0]
    assert light["mean_after"] < light["mean_before"]          # the fix must actually help here


def test_zerodce_colour_transfer_keeps_the_original_hue():
    from modules.lighting_zerodce import keep_colour
    rng = np.random.default_rng(5)
    original = rng.integers(40, 80, (50, 60, 3)).astype(np.uint8)
    tinted = np.clip(original.astype(np.float32) * np.array([1.4, 1.6, 2.4]), 0, 255).astype(np.uint8)   # pretend network output with a colour cast
    out = keep_colour(original, tinted)
    assert out.mean() > original.mean() * 1.3                              # it brightened
    o_ratio = original[..., 0].astype(float) / original[..., 1]
    n_ratio = out[..., 0].astype(float) / np.maximum(out[..., 1], 1)
    assert np.allclose(o_ratio, n_ratio, rtol=0.08)                        # blue:green ratio unchanged = no colour cast


# --- Zero-DCE++ STABLE version (needs torch + weights; skipped otherwise) ---------------
def _zerodce_ready():
    try:
        import torch  # noqa: F401
        from modules.lighting_zerodce import WEIGHTS
        return os.path.exists(WEIGHTS)
    except ImportError:
        return False


needs_zerodce = pytest.mark.skipif(not _zerodce_ready(), reason="torch or Zero-DCE++ weights missing")


def test_zerodce_curve_is_monotonic_and_keeps_black_and_white():
    from modules.lighting_zerodce import curve_from_pair
    rng = np.random.default_rng(1)
    y_in = rng.integers(0, 120, 5000).astype(np.uint8)
    y_out = np.clip(y_in.astype(np.float32) * 2.2 + rng.normal(0, 6, 5000), 0, 255)   # noisy brightening
    c = curve_from_pair(y_in, y_out)
    assert c[0] == 0 and abs(c[255] - 255) < 1e-3
    assert np.all(np.diff(c) >= 0)                       # brighter in -> never darker out
    assert 1.8 < c[60] / 60 < 2.6                         # it measured the ~2.2x lift


def test_zerodce_curve_never_pushes_a_frame_above_target():
    from modules.lighting_zerodce import limit_curve, TARGET_MEAN
    gray = np.full((50, 50), 100, np.uint8)               # an already well-lit frame
    strong = np.clip(np.arange(256, dtype=np.float32) * 2.0, 0, 255)
    limited = limit_curve(strong, gray)
    assert abs(limited[100] - TARGET_MEAN) < 1.0


@needs_zerodce
def test_zerodce_stable_does_not_pulse_when_part_of_the_scene_changes():
    """A big dark region grows and shrinks (like stress_pulse). The brightening curve must
    stay put, so the part of the picture that never changes must keep its brightness."""
    m = REGISTRY["lighting_zerodce_stable"]
    m.reset()
    base = (cv2.resize(IMAGES[1], (240, 180)) * 0.3).astype(np.uint8)
    steady = []
    for i in range(60):
        f = base.copy()
        w = int(110 + 100 * np.sin(i / 4.0))
        f[:, :w] = (f[:, :w] * 0.4).astype(np.uint8)       # changing region on the left
        m.observe(f)
        steady.append(float(m.correct(f)[:, 225:].mean()))  # right edge never changes
    steady = np.array(steady[10:])
    assert steady.mean() > base[:, 225:].mean() * 1.3     # it brightened
    assert steady.std() < 1.5                             # and did not pulse


@needs_zerodce
def test_zerodce_stable_correct_is_repeatable_and_keeps_colour():
    m = REGISTRY["lighting_zerodce_stable"]
    m.reset()
    dark = (IMAGES[2] * 0.3).astype(np.uint8)
    m.observe(dark)
    a, b = m.correct(dark), m.correct(dark)
    assert np.array_equal(a, b)                           # no hidden state changes in correct()
    assert a.shape == dark.shape and a.dtype == np.uint8
    mask = dark.min(axis=2) > 10
    o = dark[..., 0][mask].astype(float) / dark[..., 1][mask]
    n = a[..., 0][mask].astype(float) / np.maximum(a[..., 1][mask], 1)
    assert np.median(np.abs(o - n) / o) < 0.08            # blue:green ratio unchanged
