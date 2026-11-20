"""Makes a fake CCTV-like clip: first half dark + noisy, second half blurry.
Only for testing the pipeline. Replace with real sample footage later."""
import cv2
import numpy as np

w, h, fps, seconds = 320, 240, 15, 6
out = cv2.VideoWriter("test_clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
rng = np.random.default_rng(0)
for i in range(fps * seconds):
    img = np.full((h, w, 3), 150, np.uint8)
    cv2.rectangle(img, (0, 160), (w, h), (110, 120, 100), -1)
    x = 20 + (i * 4) % (w - 60)
    cv2.rectangle(img, (x, 90), (x + 30, 190), (60, 60, 200), -1)     # a "person"
    cv2.circle(img, (x + 15, 80), 12, (170, 190, 220), -1)
    if i < fps * seconds // 2:
        img = (img * 0.30).astype(np.uint8)                            # dark
        img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)  # noise
    else:
        img = cv2.GaussianBlur(img, (0, 0), 4)                         # blur
    out.write(img)
out.release()
print("wrote test_clip.mp4")
