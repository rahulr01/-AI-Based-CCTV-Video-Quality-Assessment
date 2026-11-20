"""Run the pipeline on EVERY video in a folder and write one summary table.

Usage:   python run_batch.py my_videos batch_out
Output:  batch_out/batch_summary.csv  (one row per video per problem)
Open the CSV in Excel to compare clips: which clips have which problems,
and did the fixes lower the scores (mean_after < mean_before)?
"""
import csv
import glob
import os
import sys

from pipeline import process_video

src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
videos = [p for ext in ("mp4", "avi", "mov", "mkv") for p in glob.glob(os.path.join(src_dir, f"*.{ext}"))]

with open(os.path.join(out_dir, "batch_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["video", "problem", "flagged_share", "mean_before", "mean_after", "stability_ratio", "ms_per_frame"])
    for v in sorted(videos):
        name = os.path.splitext(os.path.basename(v))[0]
        print("Processing", name)
        rep = process_video(v, os.path.join(out_dir, name))
        for s in rep["summary"]:
            w.writerow([name, s["problem"], s["flagged_share"], s["mean_before"], s["mean_after"],
                        rep["stability"]["ratio"], rep["speed"]["ms_per_frame"]])
print("Wrote", os.path.join(out_dir, "batch_summary.csv"))
