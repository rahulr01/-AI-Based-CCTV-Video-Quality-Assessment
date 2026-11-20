# CCTV Stream Diagnosis & Enhancement: Plan from Start to Production

Team: Ayisha Rima, Rahima, Rahul R. (If only two of you are active, Section 3 shows how to merge roles.)
Company schedule: development 1 Nov to 15 Dec, testing 16 to 31 Dec, documentation 1 to 10 Jan, final demo 11 to 20 Jan.

---

## 0. Read this first: the honest starting point

**What exists today (in this project folder):**

- A working pipeline: Frame, Diagnose, Route, Correct, Compare, Dashboard.
- Three problem modules (noise, lighting, blur), all classical methods, plus a Zero-DCE++ lighting module that is written but **not yet run by anyone** (it needs `torch` on your machine).
- Stability rules (smoothing, hysteresis, fade-in/out, skip rule, slow brightness average) and a stability measurement.
- Tools: `make_degraded.py`, `calibrate.py`, `compare_fixes.py`, `check_module.py`, `run_batch.py`, `make_stress_clips.py`, and 10 automated tests (`python -m pytest -q`).
- A two-panel dashboard (`streamlit run app.py`).

**What does NOT exist yet:**

- Real footage and thresholds tuned on it (all thresholds today are guesses validated only on fake data).
- The YOLO before/after check (does the fix help person detection?).
- User-selectable fixes in the dashboard, RTSP/live input, frame-drop and FPS checks.
- Any pretrained neural model running in the pipeline.
- Packaging (Docker), logging, a config file, a deployment.

**What "production level" can honestly mean for you by 20 January.**
Three interns in about ten weeks can deliver a **production-ready v1**: calibrated on real clips, tested, measured, documented, packaged and demonstrated on a recorded video and a simulated live stream. They cannot deliver a system running 24/7 on every client camera, because that needs deployment, monitoring and support that the company has to staff. Say this plainly in your final presentation. It makes the work look credible, not smaller.

**Acceptance targets (agree on these numbers with your lead in week 1; they are my suggestions):**

| Area | Target | How it is measured |
|---|---|---|
| Diagnosis | At least 85% of bad frames caught, at most 10% of good frames flagged | `calibrate.py` on **held-out** clips |
| Fix quality | Own problem's score goes down on most flagged frames; none of the 3 scores gets worse by 0.15 or more | `run_batch.py` CSV (`mean_before`, `mean_after`), "regressions" in the dashboard |
| Stability | Brightness stability ratio 1.2 or lower on `stress_dark` and `stress_pulse` | `pipeline.py` output, "stability" |
| Downstream | Person detection (recall / mAP50) is not lower on corrected frames than on originals | YOLO check (Section 7) |
| Speed | Measured ms per frame at real camera resolution, and the number of streams one machine can handle, written down | `speed` in the report |
| Quality | All tests pass; every module has a documented primary method and fallback | `python -m pytest -q` |

---

## 1. How the system fits together

```
 video file / RTSP stream
          |
   read frames (OpenCV)
          |
   every frame:  module.observe(frame)        <- cheap running memory (brightness, frame buffer)
   every N frames:
          |-- DIAGNOSE each module  -> score 0..1  -> smooth (EMA) -> compare to threshold
          |-- ROUTE (routing.yaml)  -> hysteresis decides which fixes are ON, in 'order'
          |-- skip rule: a fix that makes its own score worse is NOT applied
   every frame:
          |-- CORRECT in order (noise -> lighting -> blur), each fix fades in/out over 1 s
          |
   write corrected video + report.json (scores before/after, stability, speed, advice)
          |
   dashboard: Panel A (scores, findings, camera advice) | Panel B (original vs corrected)
```

Three ideas to remember:

1. **A frame is a NumPy array of numbers.** *Diagnose* measures them (brightness is `frame.mean()`), *correct* changes them.
2. **`routing.yaml` is the control panel.** One row per researched problem: which module, which threshold, what order, and what camera advice to show when software cannot fix it.
3. **Every module has the same interface** (`reset`, `observe`, `diagnose`, `correct`), so three people can work in parallel without breaking each other.

---

## 2. Tech stack

| Purpose | Tool | Why |
|---|---|---|
| Language | Python 3.10 or newer | Everything below is Python |
| Image / video | OpenCV, NumPy | Fast, standard, CPU-friendly |
| Neural models | PyTorch (`pip install torch`) | The pretrained models are PyTorch |
| Person detection check | `ultralytics` (YOLO) | You already use it; gives the "did the fix help" result |
| Quality metrics | scikit-image (PSNR, SSIM), `pyiqa` (no-reference scores) | Your evaluation plan asks for both; no-reference scores are supplementary |
| Dashboard | Streamlit (v1) | Python-only, quick; keep it unless there is spare time |
| Config | `routing.yaml` (+ a `config.yaml` you will add, Section 12) | Settings out of the code |
| Tests | pytest | Already in the project |
| Version control | Git + GitHub, `.gitignore` already set | Weights, videos and keys must never be committed |
| Task tracking | The company's Jira (or Trello) | The rubric asks for it |
| Fake live stream | MediaMTX (RTSP server) + ffmpeg | Test live input without client access |
| Packaging | Docker (Section 12) | One command to run it anywhere |
| Heavy experiments | Google Colab or the company GPU server | Rahul's 4GB card is tight for big models |

---

## 3. Who does what

| Person | Owns | Done when |
|---|---|---|
| **Ayisha** | **Lighting module**: calibration, backlight score, Zero-DCE++ comparison, fixing the open stability issue (Section 9) | Lighting passes the acceptance table and has a written primary method and fallback |
| **Rahima** | **Blur module** (later camera shake) and the **evaluation**: YOLO before/after check, PSNR/SSIM, the final results table | The results table exists for all three problems |
| **Rahul** | **Noise module**, the **pipeline and dashboard**, user-selectable fixes, RTSP simulation, packaging and the shared repo/board | Dashboard shows all agreed panels; RTSP demo runs; Docker image runs |

**If only two of you are active:** the person with the pipeline also takes noise and packaging; the other takes lighting and blur. Drop camera shake and the neural deblur first (Section 14).

**Team rules:** one shared repo; no one merges code that fails `python -m pytest -q`; each person must be able to explain every module, not just their own; a 15-minute call on Monday (what I will finish this week) and a 15-minute demo on Friday (show something that runs).

---

## 4. The loop you repeat for every problem

This is the core workflow. Do it for lighting, noise and blur, in parallel.

**Step 1. Build the labelled set.** Collect real clips (Section 10). From each clip, save frames into folders `good/` and `bad/` for your problem. You need **30 or more of each, from several different clips**. Keep clips separate: tune on some clips (A), test on others (B).

**Step 2. Check the score is sane on fake damage.**
`python make_degraded.py good_frames degraded_out`
The score must rise with the level (dark, noise, blur at 3 levels each). If it does not, the diagnosis is wrong. Use sharp, well-lit, textured frames.

**Step 3. Calibrate the threshold.**
`python calibrate.py lighting goodA badA goodB badB`
Read the **held-out (B)** line, not the A line. Put the suggested threshold in `routing.yaml`. Write down the false alarms and misses (these go in your methods document).

**Step 4. Make the baseline fix work.** Run `python check_module.py <module> good bad` and the pipeline on a clip. Look at the corrected frames with your own eyes: edges, colours, halos, invented detail.

**Step 5. Compare candidate fixes.**
`python compare_fixes.py lighting bad_frames compare_out`
It saves original | candidate 1 | candidate 2 side by side and prints score before/after and ms per frame. Pick one **primary** and one **fallback**.

**Step 6. Make it stable** (Section 9).

**Step 7. Check it helps detection** (Section 7).

**Step 8. Freeze and integrate.** Merge into the shared repo, run the tests, run `run_batch.py` on all clips.

**Step 9. Write it down.** Fill in your module's research template (the one in your Workflow document) with the *measured* numbers. This becomes the methods document.

---

## 5. Calendar (week by week)

If you are already past week 1, compress weeks 1 and 2.

| Week | Dates | Everyone | Rahul | Ayisha | Rahima | Gate (must be true to move on) |
|---|---|---|---|---|---|---|
| 1 | by 8 Nov | Get the sample clips; agree on the 3 problems and the targets with your lead; shared repo and board | Set up repo, board, run tests | Run all tools on the fake data; read `lighting.py` | Same for `sharpness.py` | Everyone has run `pytest` and `pipeline.py` |
| 2 | 9 to 15 Nov | Label good/bad frames (Step 1), run Step 2 | Noise: noise-estimate fixes, time per frame | Lighting: tile-based score | Blur: skip when too dark/noisy | Fake-damage scores rise with level for all 3 |
| 3 | 16 to 22 Nov | Calibrate (Step 3) on real clips | Calibrate noise | Calibrate lighting | Calibrate blur | Held-out precision/recall recorded for all 3 |
| 4 | 23 to 30 Nov | Integrate (Step 8); **Phase 1 milestone: initial working version** | Merge, dashboard check | Run full pipeline on all clips, log failures | Same | A non-technical person can upload a clip and see what changed |
| 5 | 1 to 7 Dec | YOLO check (Section 7), `torch` installed | User-selectable fixes (checkboxes) | Zero-DCE++ comparison (Section 6) | YOLO before/after for original vs corrected | First results table |
| 6 | 8 to 15 Dec | Stability work (Section 9) | Simulated RTSP (Section 11) | Fix stress_pulse | PSNR/SSIM + no-reference scores | **Phase 2: functional prototype**, stability ratio 1.2 or lower |
| 7 | 16 to 22 Dec | Full testing: combined damage, speed at real resolution | Docker, config, logging | Failure analysis (lighting) | Failure analysis (blur) | Test report with false alarms/misses |
| 8 | 23 to 31 Dec | Fix the worst failures; freeze features | Packaging | Final lighting numbers | Final results table | **Feature freeze 31 Dec** |
| 9 | 1 to 10 Jan | Methods document, README, screenshots, results | Architecture + RTSP doc | Lighting chapter | Evaluation chapter | Document readable by someone who was not there |
| 10 | 11 to 20 Jan | Demo rehearsal x2, presentation | Live demo owner | Explains lighting + stability | Explains evaluation | Everyone can explain the whole system |

**Rule:** at the end of every week there is something that runs. If it does not run, it is not done.

---

## 6. How to integrate a neural model (Zero-DCE++ worked example)

The whole point of the module interface: **a neural model is just a different `correct()`**. The pipeline does not change.

1. `pip install torch` (CPU is fine for a first test; use Colab for heavy models).
2. `python get_zerodce.py` downloads the authors' code and weights into `nets/` and `weights/` (both git-ignored). **Check the repo's licence allows company use** before relying on it.
3. Compare it with the classical fix on the same bad frames: `python compare_fixes.py lighting bad_frames compare_out`. The row "Zero-DCE++" is skipped with an error until steps 1 and 2 are done.
4. Look at the saved images for over-brightening, colour shifts, noise amplification. Note ms per frame.
5. If it wins, switch it on: in `routing.yaml` change `module: lighting` to `module: lighting_zerodce`. Nothing else changes.
6. Re-run Steps 3, 6 and 7 for the new fix (calibrate, stability, detection).

**How `lighting_zerodce.py` works (so you can write the next one):** it follows the authors' own test script: RGB values in 0..1, shaped (1, 3, H, W), H and W divisible by 12, and the network returns (enhanced image, curve maps). `to_input()` and `from_output()` do that conversion and are covered by a size round-trip check; the network itself has **not been run yet**, so your first real run is also the first real test.

**Rules for any model (these make it production-safe):**

- Load the model **once**, never per frame (the module loads it on first use).
- A BGR frame in, a BGR frame of the **same size** out.
- Keep `diagnose()` classical unless a model clearly wins; only the fix needs to be neural.
- Weights live in `weights/`, never in git; share them through Drive.
- Time it at real camera resolution. If a fix is too slow, run it on a smaller copy, run it every N frames, or use the fallback.
- **To build (production):** wrap every neural `correct()` in try/except and fall back to the classical fix if it fails or takes longer than the time budget. A camera feed must never stop because a model crashed.

**Candidates from your research, in the order to try:** lighting: Zero-DCE++ then SCI, Retinexformer only if a measurable gap remains. Noise: temporal denoising first (already built), single-frame NLM as fallback. Blur: unsharp mask is only a baseline; a deblurring model (check whether it was trained on motion blur or defocus) is a later experiment, and for severe blur the honest answer is "flag it, recommend re-focusing".

---

## 7. How to test

| Level | What | Command | Run when |
|---|---|---|---|
| 1. Unit | Each module's score rises with damage; fixes keep frame size/type; routing table is complete; pipeline runs end to end | `python -m pytest -q` | Before every merge |
| 2. Fake damage | Scores rise with known levels | `python make_degraded.py good out` | After changing a `diagnose()` |
| 3. Calibration | Precision / recall / false alarms on held-out clips | `python calibrate.py ...` | After new clips or a changed score |
| 4. Fix comparison | Candidates side by side, score and speed | `python compare_fixes.py ...` | When trying a new fix |
| 5. Stability | Brightness stability on stress clips | `python make_stress_clips.py photo.png`, then `python pipeline.py stress_pulse.mp4 out` | After changing a fix or parameter |
| 6. Batch | Every clip, one CSV | `python run_batch.py clips batch_out` | Before every demo |
| 7. Downstream | Does detection improve? | below | Weekly from week 5 |
| 8. Speed | ms per frame at real resolution | `speed` in the report | After any fix change |

**The YOLO before/after check (the headline result).** Take your labelled frames (your Roboflow annotations are the ground truth). Make a corrected copy of the same images by running them through the pipeline; keep the label files unchanged. Run `yolo detect val` on both copies with the **same model and the same labels** and compare recall and mAP50 for the person class. Use the company's own trained model if your developer can give it to you; a COCO-pretrained model does not know your uniform or phone classes, so evaluate only the person class with it. Report improvements **and** cases where detection got worse. Honest negative results are exactly what the evaluation plan asks for.

**Regression tests:** when a bug is found, add a test that would have caught it. When you finish calibration, store 3 to 5 reference clips and their expected scores in a test with a tolerance, so later changes cannot silently break detection.

---

## 8. Parameters: what each one does and how to tune it

| Parameter | Where | Default | What it does | Raise it when | Lower it when |
|---|---|---|---|---|---|
| `sample_every` | `process_video`, dashboard slider | 15 frames | How often problems are diagnosed | CPU is too slow | Fixes react too slowly |
| `EMA_ALPHA` | `pipeline.py` | 0.5 | Score smoothing (higher = faster reaction) | Fix reacts too late | Scores flicker |
| Thresholds | `routing.yaml` | noise 0.30, lighting 0.35, blur 0.50 | Score at which a problem counts | Too many false alarms | Misses real problems. **Replace with `calibrate.py` results** |
| `OFF_RATIO` | `pipeline.py` | 0.7 | Fix turns OFF only below threshold x this | Fix keeps toggling | Fix stays on too long |
| `MIN_SAMPLES_ON` | `pipeline.py` | 2 | Samples above threshold before a fix turns ON | Fix flickers on briefly | Fix starts too late |
| `RAMP_SECONDS` | `pipeline.py` | 1.0 | Fade-in/out time of a fix | You see a brightness step | Fix is too slow to appear |
| `SKIP_MARGIN` | `pipeline.py` | 0.05 | A fix that worsens its own score by more is skipped | Good fixes get skipped | Bad fixes get through |
| `REGRESSION_MARGIN` | `pipeline.py` | 0.15 | After all fixes, a score worse by this is reported | Too many warnings | Real side effects are missed |
| Lighting score | `lighting.py` | `(90 - mean) / 90` | Darker = worse | n/a | Replace with tile-based score |
| Lighting target / limits | `lighting.py` | target mean 110, gamma 0.4 to 1.0 | How strongly dark frames are lifted | Output still dark | Output washed out |
| Running brightness | `lighting.py` | 0.95 / 0.05 per frame | How slowly the gain follows the scene | Gain lags real changes | Brightness pulses |
| `USE_CLAHE` | `lighting.py` | False | Local contrast boost | Only after it is made stable | Leave off if frames pulse |
| Noise score | `noise.py` | `(sigma - 2) / 18`, judged as if brightened (target 110, gain up to 4x) | Noise strength | n/a | n/a |
| `BUFFER`, `MOVE_THRESHOLD` | `noise.py` | 5 frames, 15 | Temporal averaging length; what counts as "moved" | Noise remains | Ghosting behind people |
| Blur score | `sharpness.py` | log scale, sharp reference 300 | Laplacian variance, low = blurry | n/a | n/a |
| Unsharp mask | `sharpness.py` | amount 1.5, sigma 2 | Sharpening strength | Still soft | Halos or amplified noise |

**Tuning method:** change one parameter, run Steps 3, 5 and 6, keep the change only if the numbers improve and nothing else gets worse. Write every change and the evidence in a `CHANGELOG.md`.

---

## 9. Making it stable

**Already built:** diagnose only every N frames; score smoothing; hysteresis (separate on/off levels and a minimum duration); fixes fade in/out; a slow running brightness average and a rule never to brighten more than the current frame needs; temporal denoising instead of per-frame denoising; a skip rule; regression reporting; a measured stability ratio.

**What the measured stability ratio means:** the typical frame-to-frame brightness change of the output divided by the original. Per frame it uses the *median over a 4x4 grid of tiles*, so one person walking past is ignored while whole-picture pulsing shows up. The biggest 10% of steps are dropped so a scene cut or the planned fade-in does not count. It is a coarse check: it does **not** prove the corrected video looks natural. Always also watch the video.

**Measured today (with `make_stress_clips.py` on one test photo):**

- `stress_dark` (dark, noisy, moving object): ratio **0.39**, passes.
- `stress_pulse` (a large dark region changes size): ratio **1.78**, **fails** the 1.2 target.
- In an ablation, turning off CLAHE cut the pulse ratio from 2.55 to 1.82. It is off by default now. The rest comes from the gain following big content changes.

**Open task (Ayisha, week 6):** get `stress_pulse` to 1.2 or lower. Ideas to try, one at a time: estimate brightness from a **background model** (temporal median) instead of the whole frame, so a moving or changing object does not move the gain; smooth the gain per tile; re-enable CLAHE only with a temporal limit; Zero-DCE++ with output smoothing across frames.

**Other stability rules:** never treat a single dark frame as a camera fault (report "dark for X% of the clip"); skip the blur check when the frame is too dark or too noisy; detect infrared mode (three colour channels almost identical) and skip colour-based steps; test combined damage (dark + noisy + blurry) and compare processing orders.

---

## 10. Working with real videos

- **Get 20 or more clips, 10 to 30 seconds each**, from your developer, including bad ones (dark, blurry, noisy, backlit). Different cameras and times of day. Note the camera, resolution and FPS for each.
- **Privacy:** CCTV footage shows customers and staff. Only use footage the company has approved for the project, keep it inside the team's storage, and do not upload identifiable footage to outside tools (including any AI assistant) unless your lead says it is allowed. For anything you share outside, use your own recorded footage or free sample clips.
- **Storage layout:** `data/clips/` (originals, never edited), `data/frames/<problem>/good|bad` (labelled frames), `data/results/` (outputs). Keep the originals untouched; the system must never silently replace evidence with generated pixels.
- **Compression matters:** a fake test clip written as MP4 lost most of its added noise, so noise detection is weaker on compressed video. Always calibrate on footage as it really arrives.
- **Applying fixes to a video:** `python pipeline.py clip.mp4 out` (one clip) or `python run_batch.py clips batch_out` (all clips, one CSV). Output keeps the original size and frame rate. Audio is not copied (CCTV usually has none).
- **Browser playback:** OpenCV's `mp4v` file may not play in a browser. Convert with `ffmpeg -i corrected.mp4 -vcodec libx264 -pix_fmt yuv420p corrected_h264.mp4`.
- **Speed at real resolution:** the measured ~11 ms per frame was at 480x480. A 1080p frame has about 9 times as many pixels, so measure on real clips. For a 25 FPS stream the whole pipeline has about 40 ms per frame.

---

## 11. RTSP / live streams

**Not built yet.** Build it in week 6, after the recorded pipeline is stable. Live or client footage is a dependency: start only when access is given.

1. **Simulate a stream first.** Run MediaMTX (an open-source RTSP server), then loop a recorded clip into it with ffmpeg (for example `ffmpeg -re -stream_loop -1 -i clip.mp4 -c copy -f rtsp rtsp://localhost:8554/test`). Check the MediaMTX README for the current exact commands.
2. **Read it with OpenCV:** `cv2.VideoCapture("rtsp://localhost:8554/test")`.
3. **Process every N frames**, exactly like the recorded pipeline (same modules).
4. **Handle the real-world problems:** reconnect when the stream drops; keep only the latest frame so processing never falls behind (latency); count dropped frames and measured FPS (these are two of the issues your developer named); log every event with a timestamp.
5. **Store results** (scores, flags, timestamps) in a small SQLite table. The dashboard refreshes every few seconds and shows the current status and the last few issues. This is "near-real-time", which is the honest and achievable version of "live".
6. **Real camera later:** only when your lead gives access, and read-only.

---

## 12. Dashboard and packaging: from prototype to deployable

**Dashboard (Rahul), in this order:**

1. User-selectable fixes: a checkbox per fix, re-run on the chosen video (required by your deliverables).
2. Show **camera advice** for flagged problems that software cannot fix (the 13 planned rows should at least be listed with their advice).
3. Live-stream view (Section 11) with current status and recent issues.
4. Download a CSV/JSON report per clip.
5. A clear "unreliable for AI analytics" label when a problem is too severe to correct.

**Packaging and operations:**

- **`config.yaml`:** move constants (`EMA_ALPHA`, `OFF_RATIO`, ramp, margins, paths) out of `pipeline.py`, with overrides from environment variables.
- **Logging:** use Python's `logging` module; one log file per run; never `print` in library code.
- **Errors:** a bad frame, a missing weight file or a crashed model must be logged and skipped, never stop a stream.
- **Docker:** a `Dockerfile` that installs `requirements.txt` and runs `streamlit run app.py`; pin package versions in `requirements.txt`. (Not written yet; test it on a second machine.)
- **Secrets:** no passwords, SSH keys or server addresses in the repo, in screenshots, or in chats with any tool. Use environment variables.
- **Handover file:** `docs/HANDOVER.md` with how to run, how to retrain/replace a model, the parameter table, known limits, and who to ask.

---

## 13. Production-readiness checklist

| Item | Status today | Plan |
|---|---|---|
| Automated tests pass | Done (10 tests) | Add real-clip regression tests (week 7) |
| Thresholds calibrated on real, held-out footage | Not done | Weeks 2 to 3 |
| Detection impact measured (YOLO) | Not done | Weeks 5 to 6 |
| Stability target met | Partly (1 of 2 stress clips) | Week 6 |
| Speed measured at real resolution | Not done | Week 7 |
| Neural fix integrated with fallback | Module written, not run | Weeks 5 to 7 |
| RTSP (simulated) with reconnect, drops, FPS | Not done | Week 6 |
| Config file, logging, error handling | Not done | Week 7 |
| Docker | Not done | Week 8 |
| Documentation, README, methods document | README done | Week 9 |
| Licence check of every model and weight file | Not done | Week 5 |
| Known limits written down | Started (this guide) | Keep updating |

---

## 14. Risks and what to cut (in this order)

1. Camera shake, backlight and any other extra problem.
2. Neural deblur (keep the classical baseline and the "flag and re-focus" advice).
3. Neural fixes entirely (keep the classical fixes and the comparison you managed).
4. Docker (keep a clear run guide).
5. Real client streams (keep the simulated stream).

**Never cut:** calibration on held-out clips, the YOLO before/after check, the stability measurement, the dashboard, the methods document.

| Risk | Plan |
|---|---|
| Real footage arrives late | Start with free/own clips; keep tools generic |
| A neural model does not run on a teammate's laptop | Use Colab or CPU; CPU is a valid baseline |
| Fixes make things worse | The skip rule and regression report; log them as findings |
| One person is blocked | Friday demo and board show it early; pair up |
| Scope creep | Anything not in the acceptance table needs the team's agreement |

---

## 15. Using Claude (or any AI assistant) well

You are planning a subscription. Check the current plan limits and features on Anthropic's support site before you pay, because they change. These habits work with any plan:

- **Give it context, not just a question.** Paste the error text and the relevant file, say what you expected, and what you tried. Start a task with "Here is the README and `modules/noise.py`".
- **Ask for small, testable steps**, and ask for a test with each change. Run `python -m pytest -q` after every change it suggests.
- **Ask it to explain before you accept code.** If you cannot explain a function, you cannot defend it in the demo, and the lead may ask you to change it live.
- **Use it as a reviewer:** "What could go wrong with this threshold? What edge cases are missing?" is often more valuable than "write this".
- **Keep a decisions log** (what changed, why, what evidence). It becomes your report.
- **Keep secrets and client footage out of it** (Section 10 and 12).
- **Check claims on your own data.** Whatever it says about a method, run `compare_fixes.py` and `calibrate.py` before you believe it.

Useful prompts:

1. "Here is `check_module.py` output and `routing.yaml`. My lighting held-out precision is 0.7. What are three likely causes and a test for each?"
2. "Review this function for stability problems when the scene changes suddenly. Suggest a test that would fail on the current version."
3. "Explain this code line by line as if I must present it to my CEO tomorrow."
4. "Write a pytest test for this bug: [paste the failing frame description and the expected behaviour]."

---

## 16. Cheat sheet

```
python -m pytest -q                                        all tests
python pipeline.py clip.mp4 out                            one clip -> out/report.json, out/corrected.mp4
python run_batch.py clips batch_out                        all clips -> batch_summary.csv
python make_degraded.py good_frames degraded_out           fake damage + score check
python check_module.py lighting good bad                   scores per image
python calibrate.py lighting goodA badA goodB badB         choose and verify a threshold
python compare_fixes.py lighting bad_frames out            compare fixes side by side
python make_stress_clips.py photo.png                      stress clips for stability
python get_zerodce.py                                      download Zero-DCE++ (needs torch)
python -m streamlit run app.py                             dashboard
```

| File | Purpose |
|---|---|
| `routing.yaml` | One row per problem: module, threshold, order, advice |
| `pipeline.py` | The loop, stability rules, report |
| `modules/*.py` | One file per problem (the same 4 functions) |
| `app.py` | Two-panel dashboard |
| `tests/test_project.py` | Automated tests |
| `calibrate.py`, `compare_fixes.py`, `make_degraded.py`, `make_stress_clips.py`, `run_batch.py`, `check_module.py` | Testing and tuning tools |
| `get_zerodce.py`, `modules/lighting_zerodce.py` | Neural lighting fix |
