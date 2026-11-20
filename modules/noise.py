"""Noise / grain.

Fix = motion-adaptive TEMPORAL averaging: on a fixed camera the background is
still while the noise changes, so averaging the last few frames removes noise
without blurring. Where something moved, the current frame is kept (no ghosts).
The first frames of a video (no history yet) use single-frame denoising.

Rule: noise is fixed BEFORE brightening (the routing 'order' column enforces this).

TODO for the owner:
  - estimate noise from the smoothest tiles so texture does not inflate it
  - tune MOVE_THRESHOLD and the buffer length on real clips
  - single-frame denoising of the moved pixels
"""
from collections import deque

import cv2
import numpy as np
from .base import ProblemModule, Diagnosis

BUFFER = 5
MOVE_THRESHOLD = 15      # gray-level difference that counts as "this pixel moved"


class NoiseModule(ProblemModule):
    name = "noise"

    def __init__(self):
        self.reset()

    def reset(self):
        self._buf = deque(maxlen=BUFFER)

    def observe(self, frame):
        self._buf.append(frame.copy())

    def diagnose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape
        # Dark frames hide noise that the lighting fix will multiply. Judge the noise as it
        # would look after brightening (gain 1 = no change, at most 4x).
        gain = float(np.clip(110.0 / max(gray.mean(), 1.0), 1.0, 4.0))
        gray = gray * gain
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
        resp = cv2.filter2D(gray, -1, kernel)[1:-1, 1:-1]
        # Immerkaer (1996) fast noise-variance estimate
        sigma = float(np.sqrt(np.pi / 2.0) * np.abs(resp).sum() / (6.0 * (w - 2) * (h - 2)))
        score = float(np.clip((sigma - 2.0) / 18.0, 0.0, 1.0))
        return Diagnosis(self.name, score, {"noise_sigma": round(sigma, 2), "brightening_gain": round(gain, 2)})

    def correct(self, frame):
        if len(self._buf) < 3:
            return cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
        mean = np.stack(self._buf).astype(np.float32).mean(axis=0)
        f32 = frame.astype(np.float32)
        diff = np.abs(cv2.cvtColor(f32, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY))
        moved = cv2.blur(diff, (7, 7)) > MOVE_THRESHOLD          # blur: noise averages out, motion stays
        moved = cv2.dilate(moved.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        return np.where(moved[..., None], f32, mean).clip(0, 255).astype(np.uint8)
