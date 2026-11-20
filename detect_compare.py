"""Does the fix help a person detector?  Compares THREE videos of the same scene.

Usage:
    python detect_compare.py person_test.mp4 person_dark.mp4 out_person_fixed\\corrected.mp4
       (normal clip)        (dark clip)        (the pipeline's fixed version of the dark clip)

For every 5th frame it counts the people YOLO finds, then prints:
  frames with a person   % of sampled frames where at least one person was found
  avg persons            average number of people per frame
  avg confidence         how sure YOLO is (higher is better)
  caught vs normal       of the frames where the NORMAL clip found a person, how many this clip found too
  extra detections       % of frames where the normal clip found nobody but this clip did (false alarms)

The NORMAL clip is used as the answer key. It is not perfect proof, so also look at the
video. This uses the small public YOLO model (yolov8n), not the company's own model.
"""
import sys

import cv2
import numpy as np

EVERY = 5
CONF = 0.25


def sampled_frames(path, every=EVERY):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open {path}")
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


def analyse(paths, detector, every=EVERY):
    """detector(frame) -> list of confidences, one per person found."""
    per_video = []
    for p in paths:
        per_video.append([detector(f) for f in sampled_frames(p, every)])
    n = min(len(v) for v in per_video)
    per_video = [v[:n] for v in per_video]
    ref_found = np.array([len(c) > 0 for c in per_video[0]])
    rows = []
    for confs in per_video:
        found = np.array([len(c) > 0 for c in confs])
        all_conf = [x for c in confs for x in c]
        rows.append({
            "with_person": 100.0 * found.mean(),
            "avg_persons": float(np.mean([len(c) for c in confs])),
            "avg_conf": float(np.mean(all_conf)) if all_conf else 0.0,
            "caught": 100.0 * (found & ref_found).sum() / max(ref_found.sum(), 1),
            "extra": 100.0 * (found & ~ref_found).sum() / max((~ref_found).sum(), 1),
        })
    return rows, n


def yolo_detector():
    from ultralytics import YOLO                       # imported here so the logic can be tested without it
    model = YOLO("yolov8n.pt")                         # downloads once (about 6 MB)

    def detect(frame):
        res = model(frame, classes=[0], conf=CONF, verbose=False)[0]     # class 0 = person
        return [float(x) for x in res.boxes.conf]
    return detect


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python detect_compare.py normal.mp4 dark.mp4 fixed.mp4")
    rows, n = analyse(sys.argv[1:4], yolo_detector())
    print(f"Frames compared per video: {n} (every {EVERY}th frame)\n")
    print(f"{'':10s}{'with person':>13s}{'avg persons':>13s}{'avg conf':>10s}{'caught vs normal':>18s}{'extra':>8s}")
    for name, r in zip(("normal", "dark", "fixed"), rows):
        caught = "-" if name == "normal" else f"{r['caught']:.0f}%"
        extra = "-" if name == "normal" else f"{r['extra']:.0f}%"
        print(f"{name:10s}{r['with_person']:12.0f}%{r['avg_persons']:13.2f}{r['avg_conf']:10.2f}{caught:>18s}{extra:>8s}")
    print("\nIf 'fixed' beats 'dark' on 'with person' and 'caught', the fix helps detection.")
