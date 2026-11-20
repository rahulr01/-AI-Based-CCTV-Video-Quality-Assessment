"""Low / poor light.

Stability: the gamma fix uses a SLOW running average of brightness (observe),
so one person walking past does not make the whole picture pulse.

TODO for the owner:
  - diagnose with tile-by-tile brightness (a heatmap), not just the mean
  - detect infrared mode (3 colour channels almost identical) and skip colour steps
  - compare with Zero-DCE++ (modules/lighting_zerodce.py) using compare_fixes.py
"""
import cv2
import numpy as np
from .base import ProblemModule, Diagnosis

# CLAHE is OFF by default: in our stress test (python make_stress_clips.py) it was the main
# cause of brightness pulsing when large parts of the picture change. Turn it on and try to
# make it stable (for example smooth its effect over time) before using it.
USE_CLAHE = False


class LightingModule(ProblemModule):
    name = "lighting"

    def __init__(self):
        self.reset()

    def reset(self):
        self._mean = None

    def observe(self, frame):
        small = cv2.resize(frame, (160, 90))
        m = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        self._mean = m if self._mean is None else 0.95 * self._mean + 0.05 * m

    def diagnose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        dark_share = float((gray < 40).mean())
        bright_share = float((gray > 240).mean())
        score = float(np.clip((90.0 - mean) / 90.0, 0.0, 1.0))   # darker = worse
        return Diagnosis(self.name, score, {
            "mean_brightness": round(mean, 1),
            "dark_pixel_share": round(dark_share, 3),
            "bright_pixel_share": round(bright_share, 3),
        })

    def correct(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = float(gray.mean())
        # slow average for stability, but NEVER brighter than this frame needs
        # (otherwise a sudden 'lights on' gets over-brightened while the average catches up)
        mean = max(max(self._mean, now) if self._mean is not None else now, 1.0)
        gamma = float(np.clip(np.log(110 / 255) / np.log(mean / 255), 0.4, 1.0))
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(frame, lut)
        if not USE_CLAHE:
            return out
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
