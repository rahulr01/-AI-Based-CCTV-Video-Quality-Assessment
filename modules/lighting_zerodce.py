"""Lighting fix with the pretrained Zero-DCE++ network.

Setup (once, on your own machine):
    pip install torch
    python get_zerodce.py           # downloads the authors' code + weights
To use it in the pipeline: in routing.yaml change  module: lighting
to  module: lighting_zerodce.   To just compare: python compare_fixes.py ...

Written from the authors' repo (model.py and lowlight_test.py): input is an RGB
tensor in 0..1 shaped (1, 3, H, W) with H and W divisible by scale_factor (12);
the network returns (enhanced_image, curve_maps).
diagnose() is inherited from the classical module; only the FIX is neural.
Check the repo's licence before using it in a company product.
"""
import os

import cv2
import numpy as np

from .lighting import LightingModule

SCALE = 12
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights", "zerodce", "Epoch99.pth")


def to_input(frame, scale=SCALE):
    """BGR uint8 frame -> float32 (1, 3, H', W') in 0..1, RGB, size divisible by scale."""
    h, w = frame.shape[:2]
    h2, w2 = max(scale, h // scale * scale), max(scale, w // scale * scale)
    small = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_AREA) if (h2, w2) != (h, w) else frame
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None], (h, w)


def from_output(arr, orig_hw):
    """(1, 3, H', W') float 0..1 RGB -> BGR uint8 at the ORIGINAL frame size."""
    out = (np.clip(arr[0].transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
    h, w = orig_hw
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class LightingZeroDCE(LightingModule):
    name = "lighting"

    def __init__(self):
        super().__init__()
        self.net = None

    def _load(self):
        import torch                                           # only needed when this fix is used
        from nets.zerodce.zerodce_net import enhance_net_nopool
        if not os.path.exists(WEIGHTS):
            raise FileNotFoundError("Zero-DCE++ weights missing. Run: python get_zerodce.py")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        net = enhance_net_nopool(SCALE).to(self.device)
        net.load_state_dict(torch.load(WEIGHTS, map_location=self.device))
        self.net = net.eval()                                  # loaded ONCE, reused for every frame

    def correct(self, frame):
        if self.net is None:
            self._load()
        x, hw = to_input(frame)
        t = self.torch.from_numpy(x).to(self.device)
        with self.torch.no_grad():
            enhanced, _ = self.net(t)
        return from_output(enhanced.cpu().numpy(), hw)


def keep_colour(original, enhanced, eps=4.0, max_gain=10.0, scale=4):
    """Brighten the ORIGINAL frame by the same factor Zero-DCE++ applied to brightness.
    Hue and saturation stay exactly as in the original, so the network's colour cast
    (the purple tint on grey walls) cannot appear.

    The brightness factor is a smooth map, so it is worked out on a small copy
    (1/scale size) and enlarged: about 4x faster than doing it per pixel, and it also
    stops near-black pixels from being multiplied into noise. eps and max_gain are
    safety limits."""
    h, w = original.shape[:2]
    small = (max(1, w // scale), max(1, h // scale))
    yo = cv2.cvtColor(cv2.resize(original, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ye = cv2.cvtColor(cv2.resize(enhanced, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gain = np.clip((ye + eps) / (yo + eps), 0.0, max_gain)
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(original.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


class LightingZeroDCEKeepColour(LightingZeroDCE):
    """Zero-DCE++ brightness with the original colours."""
    name = "lighting"

    def correct(self, frame):
        return keep_colour(frame, super().correct(frame))


# ---------------------------------------------------------------------------
# STABLE version (use this one for video).
#
# Why the two classes above flicker: the network picks a NEW curve for every frame
# from what is in that frame, so a person walking past, a shadow, or sensor noise
# changes how much the WHOLE picture is brightened (stress_pulse ratio 2.12).
#
# What this version does instead:
#   1. observe() runs the network on a SMALL copy of the frame (fast at any resolution)
#      and turns its answer into ONE brightness curve: input brightness -> output
#      brightness (a 256-entry table), measured from the network's own output.
#   2. That curve is smoothed over time (like the classical module's slow average):
#      it moves SLOWLY towards stronger brightening and FASTER towards weaker
#      brightening, so a sudden 'lights on' is not over-brightened for long.
#   3. correct() only APPLIES the smoothed curve, as a brightness gain on the original
#      pixels, so the original colours are kept (no purple tint) and calling correct()
#      twice on the same frame gives the same answer.
# Trade-off: one curve for the whole picture, so the network's per-region differences
# are not used. The curve SHAPE (lift the dark, protect the bright) is the network's.
# ---------------------------------------------------------------------------
NET_LONG_SIDE = 384      # network input size (long side); the curve hardly depends on resolution
NET_EVERY = 2            # run the network every N frames (curve is smoothed anyway)
BINS = 32                # brightness bins used to measure the curve
MIN_PIXELS = 20          # a bin needs at least this many pixels to be trusted
ALPHA_UP = 0.03          # per network run: move towards MORE brightening slowly
ALPHA_DOWN = 0.5         # per network run: move towards LESS brightening fast (lights switched on)
DEADBAND = 1.5           # curve changes smaller than this (grey levels) are ignored
MAX_GAIN = 8.0           # never multiply a pixel by more than this (noise in near-black)
TARGET_MEAN = 120.0      # never brighten a frame's average above this (the network alone takes
                         # an already well-lit frame to ~155; the classical fix aims for 110)


def curve_from_pair(y_in, y_out, bins=BINS, min_pixels=MIN_PIXELS):
    """Measure the network's brightness curve from an input/output grey pair.
    Returns a 256-entry float table, monotonic, with 0 -> 0 and 255 -> 255
    (Zero-DCE++ curves always keep black and white fixed)."""
    y_in = y_in.ravel().astype(np.float32)
    y_out = y_out.ravel().astype(np.float32)
    idx = np.minimum((y_in * bins / 256.0).astype(np.int32), bins - 1)
    count = np.bincount(idx, minlength=bins)
    total = np.bincount(idx, weights=y_out, minlength=bins)
    total_in = np.bincount(idx, weights=y_in, minlength=bins)
    ok = count >= min_pixels
    xs = np.concatenate([[0.0], total_in[ok] / np.maximum(count[ok], 1), [255.0]])
    ys = np.concatenate([[0.0], total[ok] / np.maximum(count[ok], 1), [255.0]])
    order = np.argsort(xs)
    xs, ys = xs[order], np.maximum.accumulate(ys[order])     # brighter in -> never darker out
    return np.interp(np.arange(256, dtype=np.float32), xs, ys).astype(np.float32)


def limit_curve(curve, gray, target=TARGET_MEAN):
    """Scale back the curve's lift if it would make this frame brighter than target on average."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float32)
    n = max(hist.sum(), 1.0)
    levels = np.arange(256, dtype=np.float32)
    mean_in, mean_out = float(hist @ levels) / n, float(hist @ curve) / n
    if mean_out <= target or mean_out <= mean_in:
        return curve
    k = float(np.clip((target - mean_in) / (mean_out - mean_in), 0.0, 1.0))
    return levels + k * (curve - levels)


def apply_curve(frame, curve, max_gain=MAX_GAIN):
    """Brighten with a 256-entry curve, as a gain on the original pixels (colours kept)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    levels = np.arange(256, dtype=np.float32)
    gain_lut = np.clip(curve / np.maximum(levels, 1.0), 0.0, max_gain)
    gain = gain_lut[gray]
    return cv2.multiply(frame, cv2.merge([gain, gain, gain]), dtype=cv2.CV_8U)   # rounds and clips to 0..255


class LightingZeroDCEStable(LightingZeroDCE):
    """Zero-DCE++ curve, smoothed over time, original colours kept."""
    name = "lighting"

    def reset(self):
        super().reset()
        self._curve = None
        self._count = 0

    def _small(self, frame):
        h, w = frame.shape[:2]
        s = min(1.0, NET_LONG_SIDE / max(h, w))
        return cv2.resize(frame, (max(SCALE, int(w * s)), max(SCALE, int(h * s))), interpolation=cv2.INTER_AREA)

    def frame_curve(self, frame):
        """The network's curve for ONE frame (no smoothing)."""
        small = self._small(frame)
        enhanced = LightingZeroDCE.correct(self, small)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return limit_curve(curve_from_pair(gray, cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)), gray)

    def observe(self, frame):
        super().observe(frame)                        # keeps the classical brightness average too
        self._count += 1
        if self._curve is not None and (self._count - 1) % NET_EVERY:
            return
        cur = self.frame_curve(frame)
        if self._curve is None:
            self._curve = cur
            return
        diff = cur - self._curve
        diff[np.abs(diff) < DEADBAND] = 0.0           # ignore tiny wobbles (noise)
        self._curve += np.where(diff > 0, ALPHA_UP, ALPHA_DOWN) * diff

    def correct(self, frame):
        curve = self._curve if self._curve is not None else self.frame_curve(frame)
        return apply_curve(frame, curve)
