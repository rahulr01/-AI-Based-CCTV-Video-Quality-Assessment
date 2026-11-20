"""Make two test clips from ONE photo. Use a sharp, well-lit frame from your own footage.

Usage:   python make_stress_clips.py photo.png
  stress_dark.mp4  - dark + noisy, with a moving block  (should be brightened and denoised)
  stress_pulse.mp4 - same, but a big dark region keeps changing size (tests stability)
Then:    python pipeline.py stress_pulse.mp4 out_pulse    and read "stability" in the output.
Target: stability ratio 1.2 or lower on BOTH clips.
"""
import sys

import cv2
import numpy as np

bg = cv2.resize(cv2.imread(sys.argv[1]), (480, 480))
rng = np.random.default_rng(2)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
dark = cv2.VideoWriter("stress_dark.mp4", fourcc, 15, (480, 480))
pulse = cv2.VideoWriter("stress_pulse.mp4", fourcc, 15, (480, 480))
for i in range(15 * 8):
    f = bg.copy()
    x = (i * 5) % 400
    cv2.rectangle(f, (x, 300), (x + 60, 440), (30, 30, 200), -1)                     # a moving "person"
    f = np.clip(f.astype(np.float32) * 0.35 + rng.normal(0, 10, f.shape), 0, 255).astype(np.uint8)
    dark.write(f)
    g = f.copy()
    w = max(int(240 + 200 * np.sin(i / 5.0)), 1)
    g[:, :w] = (g[:, :w] * 0.4).astype(np.uint8)                                     # dark region pulses in size
    pulse.write(g)
dark.release()
pulse.release()
print("wrote stress_dark.mp4 and stress_pulse.mp4")
