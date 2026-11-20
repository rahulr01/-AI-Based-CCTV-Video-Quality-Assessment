"""How close is the CORRECTED video to the ORIGINAL normal-brightness video?

Use it when you darkened a good clip on purpose:
    python compare_to_reference.py dce_test.mp4 out_dce_dark\\corrected.mp4
reference = the normal clip (the answer we want to get back)
corrected = the pipeline's output for the darkened version of it

It prints, averaged over the clip:
  brightness      mean brightness (0-255). Corrected should be CLOSE to the reference.
  blown share     % of pixels at 250 or above (pure white). If the corrected number is
                  much higher than the reference, the fix is over-exposing.
  colour shift    average difference in colour (LAB a/b). Lower is better. Above ~6 is visible.
  PSNR            similarity to the reference in dB. Higher is better (20+ is decent for this).
"""
import sys

import cv2
import numpy as np

ref, cor = cv2.VideoCapture(sys.argv[1]), cv2.VideoCapture(sys.argv[2])
if not (ref.isOpened() and cor.isOpened()):
    sys.exit("Could not open one of the videos. Check both file names.")

rows = []
i = 0
while True:
    ok1, a = ref.read()
    ok2, b = cor.read()
    if not (ok1 and ok2):
        break
    if i % 5 == 0:
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        la, lb = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
        colour = float(np.abs(la[..., 1:] - lb[..., 1:]).mean())
        rows.append((i, ga.mean(), gb.mean(), (ga >= 250).mean() * 100, (gb >= 250).mean() * 100,
                     colour, cv2.PSNR(a, b)))
    i += 1

if not rows:
    sys.exit("No frames compared. Do both videos have frames?")
r = np.array(rows)
print(f"Frames compared: {len(rows)} (every 5th of {i})\n")
print(f"{'':22s}{'reference':>11s}{'corrected':>11s}")
print(f"{'brightness (0-255)':22s}{r[:, 1].mean():11.1f}{r[:, 2].mean():11.1f}")
print(f"{'blown share (%)':22s}{r[:, 3].mean():11.1f}{r[:, 4].mean():11.1f}")
print(f"\ncolour shift (lower is better): {r[:, 5].mean():.1f}")
print(f"PSNR (higher is better):        {r[:, 6].mean():.1f} dB")
worst = r[np.argmax(r[:, 4] - r[:, 3])]
print(f"\nMost over-exposed moment: frame {int(worst[0])} "
      f"(blown share {worst[3]:.1f}% in reference, {worst[4]:.1f}% in corrected)")
