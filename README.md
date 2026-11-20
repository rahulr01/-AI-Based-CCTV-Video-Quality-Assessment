# CCTV Stream Diagnosis & Enhancement: v1 skeleton

**Start with GUIDE.md**: the full plan from start to production (workflow, calendar, testing, tuning, stability, RTSP, packaging).

A working end-to-end starting point. Everything runs today; each module is
simple on purpose and is meant to be improved by its owner.

## Run it
    pip install -r requirements.txt
    python make_test_video.py            # makes test_clip.mp4 (fake dark + blurry clip)
    python pipeline.py test_clip.mp4 out # command line run, writes out/report.json
    streamlit run app.py                 # two-panel dashboard

## How it works
Frame -> Diagnose -> Route -> Correct -> Compare -> Dashboard

- `modules/`      one file per problem. Each has `diagnose(frame)` and `correct(frame)`.
- `routing.yaml`  the routing table: one row per researched problem (16 rows).
                  `status: implemented` rows run; `planned` rows show advice only.
- `pipeline.py`   samples every N frames, scores each problem, switches fixes on or
                  off, applies them in `order`, and skips a fix that makes its own
                  score worse (the "skip rule").
- `app.py`        Panel A (scores, findings, advice) and Panel B (original vs corrected).

## Add a new problem in 3 steps
1. Copy `modules/noise.py`, rename it, write `diagnose` and `correct`.
2. Register it in `modules/__init__.py`.
3. Change that problem's row in `routing.yaml` to `status: implemented` and set
   `module`, `threshold` and `order`.

## Known limits of this skeleton (these are the first things to improve)
- Thresholds (0.30 / 0.35 / 0.50) are guesses. Tune them on REAL footage.
- Scores are smoothed (EMA), so the switch-on lags a little after a scene change.
- Laplacian variance reads very low on flat or dark scenes, so "blur" can fire
  there too. A good fix is to skip the blur check when lighting is flagged.
- Noise denoising (fastNlMeans) is slow; measure time per frame.
- No person-detection check yet (does the fix help YOLO?). That is the next module.
- Recorded video only. RTSP is a later stage.

## Learning the code (read in this order, 10 minutes each)
1. `modules/base.py`        the two functions every module has
2. `modules/noise.py`       the simplest real module
3. `routing.yaml`           which module runs, at what threshold, in what order
4. `pipeline.py`           the loop that ties everything together
5. `app.py`                reads report.json and draws the two panels
Tools: `python check_module.py <module> good_dir bad_dir` shows scores per image.
Neural models: copy `modules/_template_neural.py` and follow the steps in it.
