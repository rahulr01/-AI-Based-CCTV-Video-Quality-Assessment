"""Compare candidate FIXES for one problem on a folder of bad frames.

Usage:   python compare_fixes.py lighting bad_frames compare_out
Saves  original | candidate 1 | candidate 2 ...  side by side, and prints for each
candidate: average score before/after, the TYPICAL (median) milliseconds per frame,
the device it ran on, and the one-off start-up time (model loading) shown separately.

IMPORTANT: for lighting the score only measures brightness, so two very different
fixes can both score 0.00. Judge the QUALITY by looking at the saved images.
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

from modules import REGISTRY

CANDIDATES = {
    "lighting": {"classical (gamma)": "lighting", "Zero-DCE++": "lighting_zerodce",
                 "Zero-DCE++ (colour kept)": "lighting_zerodce_color"},
    "sharpness": {"unsharp mask": "sharpness"},
    "noise": {"temporal/NLM": "noise"},
}
# Add more candidates by registering a module in modules/__init__.py and listing it above.

problem, folder, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
if not paths:
    sys.exit(f"No .jpg/.png images in {folder}")
first = cv2.imread(paths[0])

# Warm-up: the first call of a neural model loads it (seconds). Time that separately.
ready, startup, broken = {}, {}, {}
for name, key in CANDIDATES[problem].items():
    mod = REGISTRY[key]
    try:
        mod.reset()
        mod.observe(first)
        t0 = time.time()
        mod.correct(first)
        startup[name] = time.time() - t0
        ready[name] = mod
    except Exception as e:                       # e.g. torch / weights not installed yet
        broken[name] = f"{type(e).__name__}: {e}"

results = {name: {"before": [], "after": [], "ms": []} for name in ready}
for p in paths:
    frame = cv2.imread(p)
    panels = [frame.copy()]
    cv2.putText(panels[0], "original", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for name, mod in ready.items():
        mod.reset()
        mod.observe(frame)
        before = mod.diagnose(frame).score
        t0 = time.time()
        out = mod.correct(frame)
        ms = 1000 * (time.time() - t0)
        results[name]["before"].append(before)
        results[name]["after"].append(mod.diagnose(out).score)
        results[name]["ms"].append(ms)
        shown = out.copy()
        cv2.putText(shown, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(shown)
    h = 360
    row = [cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h)) for x in panels]
    cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), np.hstack(row))

h0, w0 = first.shape[:2]
print(f"Frame size: {w0}x{h0}   frames compared: {len(paths)}\n")
print(f"{'candidate':26s}{'score before':>13s}{'score after':>12s}{'ms/frame':>10s}{'device':>8s}{'start-up s':>12s}")
for name in CANDIDATES[problem]:
    if name in broken:
        print(f"{name:26s}  SKIPPED  ({broken[name]})")
        continue
    r = results[name]
    device = getattr(ready[name], "device", "cpu")
    print(f"{name:26s}{np.mean(r['before']):13.2f}{np.mean(r['after']):12.2f}"
          f"{np.median(r['ms']):10.1f}{device:>8s}{startup[name]:12.1f}")
print("\nSide-by-side images saved in", out_dir)
print("Remember: judge QUALITY by looking at the images, not at the score.")
