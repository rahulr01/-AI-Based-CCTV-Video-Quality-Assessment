# Results log: lighting fix experiments

Date: (fill in)   Author: Rahul R   Project: CCTV Stream Diagnosis & Enhancement

All numbers below were measured by running the project's own scripts. Timings vary by
roughly +/- 15 ms between runs. Everything uses clips recorded by the author (one person,
one room), darkened by script. No company footage has been tested yet.

## Experiment 1: dark clip (all pixels x0.3), 10 s, 640 wide, no people

Reference = the original normal-brightness clip (`dce_test.mp4`). Measured with
`compare_to_reference.py` and `pipeline.py`.

| Fix | Lighting score before to after | Brightness (reference 100.6) | Colour shift | PSNR | Stability ratio | ms/frame |
|---|---|---|---|---|---|---|
| Zero-DCE++ plain | 0.675 to 0.016 | 95.7 | 4.6 | 18.5 dB | 0.09 | 63.5 |
| Zero-DCE++ colour kept (first version) | 0.675 to 0.044 | 85.6 | 3.4 | 19.1 dB | 0.10 | 104.5 and 123.6 (two runs) |
| Zero-DCE++ colour kept (faster colour step) | not recorded | 85.7 | 3.4 | 18.9 dB | not recorded | 83.5 |

- Blown-out (pure white) pixels: 3.8% in the reference, 0.0% in every corrected video.
  The fix did not over-expose this clip.
- The colour-kept version keeps colours closer to the original; plain is brighter and faster.

## Experiment 2: person detection, easy clip (all pixels x0.3)

`detect_compare.py`, public model yolov8n, every 5th frame. Fix = Zero-DCE++ colour kept.

| Video | Frames with a person | Avg persons | Avg confidence | Caught vs normal | Extra |
|---|---|---|---|---|---|
| normal | 66% | 0.68 | 0.85 | n/a | n/a |
| dark | 64% | 0.68 | 0.80 | 98% | 0% |
| fixed | 65% | 0.71 | 0.79 | 99% | 0% |

Finding: this darkening barely affected the detector, so the fix showed no measurable
benefit and no harm. The test was too easy.

## Experiment 3: person detection, hard clip (x0.15 brightness plus noise level 4.0)

900 frames, 180 compared per video. Made with `make_hard_dark.py`.

| Video | Frames with a person | Avg persons | Avg confidence | Caught vs normal | Extra | ms/frame |
|---|---|---|---|---|---|---|
| normal | 66% | 0.68 | 0.85 | n/a | n/a | n/a |
| dark and noisy | 54% | 0.56 | 0.70 | 82% | 0% | n/a |
| fixed, Zero-DCE++ colour kept | 63% | 0.67 | 0.76 | 97% | 0% | 103.6 |
| fixed, classical (gamma) | 63% | 0.69 | 0.77 | 97% | 0% | 50.7 |

Pipeline scores (before to after): 

| Fix | Noise | Lighting | Stability ratio |
|---|---|---|---|
| Zero-DCE++ colour kept | 0.465 to 0.435 | 0.84 to 0.335 | 0.16 |
| classical | 0.465 to 0.188 | 0.84 to 0.175 | 0.11 |

## Conclusions (and their limits)

1. The pipeline recovered person detection from 82% to 97% of the reference detections on
   a dark, noisy test clip, with no extra false detections.
2. Zero-DCE++ did not beat the classical gamma fix on this clip: same detection result,
   about 2x slower, brighter-than-needed noise left behind, and it under-brightened.
3. **Limits:** one clip, one person, one room. The darkening was a uniform multiplication,
   which a global brightness fix is well matched to. Zero-DCE++ is designed for unevenly lit
   scenes, which have NOT been tested. The detector is the small public YOLO model, not the
   company's. The noise score scales dark pictures up, so the noise comparison is partly
   unfair to Zero-DCE++. The normal clip is an answer key, not ground truth.
4. Decision for v1: classical fix as the main method; Zero-DCE++ kept as a documented,
   switchable comparison (`routing_dce.yaml`, `routing_dce_color.yaml`).

## Known problems found along the way

- The blur check is unreliable on dark or noisy footage (it flagged a clean dark clip 100%
  and a noisy clip 0%). Needs to ignore noise and darkness.
- The first timing of Zero-DCE++ (1,043 ms) was inflated by model loading in the script;
  fixed in `compare_fixes.py`.
- The stability ratio is not meaningful for very short clips (the 1 s fade-in dominates).

## Next tests

- A naturally unevenly lit clip (a lamp or window, nothing darkened by script).
- Real company footage, if allowed, to calibrate thresholds on held-out clips.
- Repeat the detection test with the company's own trained model.
