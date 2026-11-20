"""Frame -> Diagnose -> Route -> Correct -> Compare.

Run from the terminal:   python pipeline.py input.mp4 out_folder [routing_file.yaml]
The dashboard (app.py) calls process_video() the same way.

Stability rules built in (all explained in GUIDE.md):
  - diagnose only every N frames, scores are smoothed (EMA)
  - fixes FADE in and out over RAMP_SECONDS instead of switching in one frame
  - hysteresis: a fix turns ON after MIN_SAMPLES_ON samples above the threshold and
    turns OFF only when the score falls below threshold * OFF_RATIO
  - modules keep slow running state through observe() (no brightness pulsing)
  - skip rule: a fix that makes its own score worse is not applied
  - regressions: after the whole chain, any score that got clearly worse is reported
"""
import json
import os
import sys
import time

import cv2
import yaml

from modules import REGISTRY

SKIP_MARGIN = 0.05          # a fix that worsens its own score by more than this is skipped
REGRESSION_MARGIN = 0.15    # after ALL fixes, a score worse than before by this is reported
EMA_ALPHA = 0.5             # score smoothing
OFF_RATIO = 0.7             # hysteresis: switch OFF below threshold * this
MIN_SAMPLES_ON = 2          # hysteresis: samples in a row above threshold before switching ON
RAMP_SECONDS = 1.0          # a fix fades in / out over this long (no sudden brightness step)


def load_routing(path="routing.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)["rows"]


def _brightness(frame):
    """Mean brightness of a 4 x 4 grid of tiles (16 numbers)."""
    g = cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2GRAY).astype("float32")
    return g.reshape(4, 40, 4, 40).mean(axis=(1, 3)).ravel()


def _mean_jump(tiles, trim=0.10, log=False):
    """TYPICAL frame-to-frame brightness change of the picture as a whole.
    Per frame we take the MEDIAN change over the 16 tiles, so one person walking past
    (a few tiles) is ignored, but pulsing of the whole picture (all tiles) shows up.
    The biggest 10% of steps are dropped: scene cuts and the planned fade-in are meant
    to be big."""
    if len(tiles) < 3:
        return 0.0
    import numpy as np
    t = np.log1p(np.array(tiles)) if log else np.array(tiles)
    steps = np.sort(np.median(np.abs(np.diff(t, axis=0)), axis=1))
    keep = steps[: max(1, int(len(steps) * (1 - trim)))]
    return float(keep.mean())


def process_video(src, out_dir, sample_every=15, routing_path="routing.yaml"):
    rows = load_routing(routing_path)
    active_rows = sorted([r for r in rows if r["status"] == "implemented"],
                         key=lambda r: r["order"])
    for r in active_rows:
        REGISTRY[r["module"]].reset()
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    out_path = os.path.join(out_dir, "corrected.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    smoothed, state, streak = {}, {r["id"]: False for r in active_rows}, {r["id"]: 0 for r in active_rows}
    samples = []
    target = {r["id"]: 0.0 for r in active_rows}     # 1.0 = fix wanted, 0.0 = not wanted
    weight = {r["id"]: 0.0 for r in active_rows}     # how strongly the fix is blended in right now
    bright_in, bright_out = [], []
    idx, t_start = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for r in active_rows:                       # cheap per-frame memory update
            REGISTRY[r["module"]].observe(frame)

        if idx % sample_every == 0:
            # --- DIAGNOSE (on the original frame) ---
            raw, scores, details = {}, {}, {}
            for r in active_rows:
                rid = r["id"]
                d = REGISTRY[r["module"]].diagnose(frame)
                raw[rid] = d.score
                prev = smoothed.get(rid, d.score)
                smoothed[rid] = EMA_ALPHA * d.score + (1 - EMA_ALPHA) * prev
                s = smoothed[rid]
                if state[rid]:                                        # hysteresis
                    if s < r["threshold"] * OFF_RATIO:
                        state[rid], streak[rid] = False, 0
                else:
                    streak[rid] = streak[rid] + 1 if s >= r["threshold"] else 0
                    if streak[rid] >= MIN_SAMPLES_ON:
                        state[rid] = True
                scores[rid] = round(s, 3)
                details[rid] = d.details

            # --- ROUTE: which fixes are switched on ---
            to_apply = [r for r in active_rows if state[r["id"]]]

            # --- CORRECT the sampled frame, with the skip rule ---
            test, kept, skipped = frame, [], []
            for r in list(to_apply):
                mod = REGISTRY[r["module"]]
                before = mod.diagnose(test).score
                candidate = mod.correct(test)
                after_own = mod.diagnose(candidate).score
                if after_own > before + SKIP_MARGIN:
                    skipped.append(r["id"])
                else:
                    test = candidate
                    kept.append(r["id"])
            for r in active_rows:
                target[r["id"]] = 1.0 if r["id"] in kept else 0.0

            # --- COMPARE: score the corrected frame on EVERY problem ---
            after = {r["id"]: round(REGISTRY[r["module"]].diagnose(test).score, 3) for r in active_rows}
            regressions = [rid for rid in after if after[rid] > raw[rid] + REGRESSION_MARGIN]

            cv2.imwrite(os.path.join(out_dir, "frames", f"orig_{idx:06d}.jpg"), frame)
            cv2.imwrite(os.path.join(out_dir, "frames", f"corr_{idx:06d}.jpg"), test)
            samples.append({
                "frame": idx, "time_s": round(idx / fps, 2),
                "scores": scores, "scores_after": after, "details": details,
                "applied": kept, "skipped": skipped, "regressions": regressions,
            })

        # --- CORRECT every frame with the currently active fixes ---
        step = 1.0 / max(fps * RAMP_SECONDS, 1.0)
        out = frame
        for r in active_rows:
            rid = r["id"]
            weight[rid] += max(-step, min(step, target[rid] - weight[rid]))
            if weight[rid] > 0.0:
                fixed = REGISTRY[r["module"]].correct(out)
                out = fixed if weight[rid] >= 1.0 else cv2.addWeighted(out, 1 - weight[rid], fixed, weight[rid], 0)
        writer.write(out)
        bright_in.append(_brightness(frame))
        bright_out.append(_brightness(out))
        idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t_start

    summary = []
    for r in rows:
        if r["status"] == "implemented" and samples:
            n = len(samples)
            flagged = sum(1 for s in samples if s["scores"][r["id"]] >= r["threshold"])
            summary.append({"id": r["id"], "problem": r["problem"],
                            "flagged_share": round(flagged / n, 2),
                            "mean_before": round(sum(s["scores"][r["id"]] for s in samples) / n, 3),
                            "mean_after": round(sum(s["scores_after"][r["id"]] for s in samples) / n, 3),
                            "advice": r["camera_fix"]})

    jump_in, jump_out = _mean_jump(bright_in), _mean_jump(bright_out)
    # RELATIVE version (log brightness): a fix that brightens by a FIXED curve scores about
    # 1 or lower even when it brightens a lot; only a fix whose strength moves around
    # (pulsing, flicker) scores above 1. The plain ratio also grows just from brightening.
    rel_in, rel_out = _mean_jump(bright_in, log=True), _mean_jump(bright_out, log=True)
    report = {
        "video": os.path.basename(src), "fps": fps, "size": [w, h],
        "frames": idx, "sample_every": sample_every,
        "stability": {"brightness_jump_original": round(jump_in, 3),
                      "brightness_jump_corrected": round(jump_out, 3),
                      "ratio": round(jump_out / max(jump_in, 0.5), 2),
                      "ratio_relative": round(rel_out / max(rel_in, 0.005), 2)},
        "speed": {"ms_per_frame": round(1000 * elapsed / max(idx, 1), 1)},
        "samples": samples, "summary": summary, "corrected_video": out_path,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    routing = sys.argv[3] if len(sys.argv) > 3 else "routing.yaml"     # optional: another routing file
    rep = process_video(sys.argv[1], sys.argv[2], routing_path=routing)
    print(json.dumps({"summary": rep["summary"], "stability": rep["stability"], "speed": rep["speed"]}, indent=2))
