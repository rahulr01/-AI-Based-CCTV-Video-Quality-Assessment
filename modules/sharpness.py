"""Blur / sharpness.

TODO for the owner:
  - Laplacian variance depends on the scene; tune the 300 below (the 'sharp' reference) on real clips
  - add camera-shake detection (feature tracking) as a second score
  - compare unsharp mask with a deblurring model on a few frames
"""
import cv2
import numpy as np
from .base import ProblemModule, Diagnosis


class SharpnessModule(ProblemModule):
    name = "sharpness"

    def diagnose(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, int(h * 640 / w)))      # resolution-independent
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # log scale so mild blur and severe blur get DIFFERENT scores (a straight line saturated)
        score = float(np.clip(1.0 - np.log(lap_var + 1.0) / np.log(300.0), 0.0, 1.0))
        return Diagnosis(self.name, score, {"laplacian_variance": round(lap_var, 1)})

    def correct(self, frame):
        amount = 1.5                                              # try 1.0 to 2.5
        blurred = cv2.GaussianBlur(frame, (0, 0), 2)
        return cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)   # unsharp mask
