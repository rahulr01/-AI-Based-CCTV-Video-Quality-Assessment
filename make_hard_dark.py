"""Make a HARD dark test clip from a normal clip: much darker AND with sensor-like noise.

Usage:   python make_hard_dark.py person_test.mp4 person_dark_hard.mp4
Optional: python make_hard_dark.py in.mp4 out.mp4 0.15 4      (brightness factor, noise level)
Real low-light CCTV is dark AND noisy, so this is a fairer test than just darkening.
Needs ffmpeg on the PATH (you already use it). The clip is saved at high quality so the
noise survives compression.
"""
import subprocess
import sys

import cv2
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
bright = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0

cap = cv2.VideoCapture(src)
if not cap.isOpened():
    sys.exit(f"Could not open {src}")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
       "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", dst]
try:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
except FileNotFoundError:
    sys.exit("ffmpeg was not found. Install it with:  winget install ffmpeg   (then reopen PowerShell)")

rng = np.random.default_rng(0)
n = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    dark = f.astype(np.float32) * bright + rng.normal(0, sigma, f.shape)     # dark first, then sensor noise
    proc.stdin.write(np.clip(dark, 0, 255).astype(np.uint8).tobytes())
    n += 1
proc.stdin.close()
proc.wait()
print(f"Wrote {dst}: {n} frames, brightness x{bright}, noise level {sigma}")
