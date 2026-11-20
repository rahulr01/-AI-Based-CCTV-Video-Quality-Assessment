"""Pick a threshold for ONE module from labelled folders, then check it on other clips.

Usage:   python calibrate.py lighting good_A bad_A                 # choose a threshold
         python calibrate.py lighting good_A bad_A good_B bad_B    # ...and test it on held-out clips
A = frames you tune on.  B = frames from DIFFERENT clips, never used for tuning.
Reports: precision (flagged frames that really were bad), recall (bad frames caught),
false alarm rate (good frames wrongly flagged).
"""
import glob
import os
import sys

import cv2
import numpy as np

from modules import REGISTRY
from pipeline import load_routing


def scores(mod, folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        sys.exit(f"No .jpg/.png images in {folder}")
    return np.array([mod.diagnose(cv2.imread(p)).score for p in paths])


def metrics(good, bad, t):
    tp, fn = (bad >= t).sum(), (bad < t).sum()
    fp, tn = (good >= t).sum(), (good < t).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_alarm = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, false_alarm, f1


def show(label, m):
    print(f"  {label:34s} precision {m[0]:.2f}   recall {m[1]:.2f}   false alarms {m[2]:.2f}")


name = sys.argv[1]
mod = REGISTRY[name]
gA, bA = scores(mod, sys.argv[2]), scores(mod, sys.argv[3])
if min(len(gA), len(bA)) < 20:
    print(f"WARNING: only {len(gA)} good / {len(bA)} bad frames. Use 30+ of each from several clips,"
          " or the threshold will not be reliable.\n")

best_t = max(np.arange(0.05, 0.96, 0.01), key=lambda t: (metrics(gA, bA, t)[3], t))
rows = [r for r in load_routing() if r.get("module") == name]
current = rows[0]["threshold"] if rows else None

print(f"Module: {name}   (good avg {gA.mean():.2f}, bad avg {bA.mean():.2f})")
print("On the frames you tuned on (A):")
show(f"suggested threshold {best_t:.2f}", metrics(gA, bA, best_t))
if current is not None:
    show(f"current routing.yaml {current:.2f}", metrics(gA, bA, current))
if len(sys.argv) >= 6:
    gB, bB = scores(mod, sys.argv[4]), scores(mod, sys.argv[5])
    print("On held-out clips (B)  <- the numbers you can trust:")
    show(f"suggested threshold {best_t:.2f}", metrics(gB, bB, best_t))
    if current is not None:
        show(f"current routing.yaml {current:.2f}", metrics(gB, bB, current))
print(f"\nIf you are happy, set  threshold: {best_t:.2f}  for module '{name}' in routing.yaml")
