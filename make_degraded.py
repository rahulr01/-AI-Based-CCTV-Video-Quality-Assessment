"""Make fake BAD versions of GOOD frames at KNOWN levels, then check the diagnosis.

Usage:   python make_degraded.py good_frames degraded_out
For every degradation (dark, noise, blur) at 3 levels it saves the images and prints
the average score. THE SCORE MUST RISE WITH THE LEVEL. If it doesn't, the
diagnosis is wrong (or the test frames are bad: use sharp, well-lit, textured frames).
"""
import glob
import os
import sys

import cv2
import numpy as np

from modules import REGISTRY

good_dir, out_dir = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(os.path.join(good_dir, "*.jpg")) + glob.glob(os.path.join(good_dir, "*.png")))
if not paths:
    sys.exit(f"No .jpg/.png images in {good_dir}")
rng = np.random.default_rng(0)


def dark(img, f):
    return (img.astype(np.float32) * f).clip(0, 255).astype(np.uint8)


def noisy(img, s):
    return np.clip(img.astype(np.float32) + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def blurry(img, s):
    return cv2.GaussianBlur(img, (0, 0), s)


JOBS = {"lighting": ("dark", dark, [0.6, 0.4, 0.25]),
        "noise": ("noise", noisy, [5, 10, 20]),
        "sharpness": ("blur", blurry, [1, 2, 4])}

print(f"{'degradation':12s}{'level':>7s}{'avg score':>11s}   module")
for module, (tag, fn, levels) in JOBS.items():
    mod = REGISTRY[module]
    imgs = [cv2.imread(p) for p in paths]
    print(f"{'original':12s}{'-':>7s}{np.mean([mod.diagnose(i).score for i in imgs]):11.2f}   {module}")
    for lv in levels:
        folder = os.path.join(out_dir, f"{tag}_{lv}")
        os.makedirs(folder, exist_ok=True)
        scores = []
        for p, im in zip(paths, imgs):
            bad = fn(im, lv)
            cv2.imwrite(os.path.join(folder, os.path.splitext(os.path.basename(p))[0] + ".png"), bad)
            scores.append(mod.diagnose(bad).score)
        print(f"{tag:12s}{lv:7}{np.mean(scores):11.2f}   {module}")
