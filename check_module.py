"""Check ONE module on a folder of good frames and a folder of bad frames.

Usage:   python check_module.py lighting good_frames bad_frames
Prints the score for every image, so you can SEE whether the numbers make sense.
A good module: the GOOD average is low and the BAD average is high.
"""
import glob
import os
import sys

import cv2

from modules import REGISTRY

name, good_dir, bad_dir = sys.argv[1:4]
module = REGISTRY[name]

for label, folder in (("GOOD", good_dir), ("BAD", bad_dir)):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
    scores = []
    for path in paths:
        score = module.diagnose(cv2.imread(path)).score
        scores.append(score)
        print(f"{label}  {os.path.basename(path):30s} score = {score:.2f}")
    if scores:
        print(f"--> {label} average = {sum(scores) / len(scores):.2f}\n")
