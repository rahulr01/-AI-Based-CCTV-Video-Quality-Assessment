"""Two-panel dashboard (v1 skeleton).  Run:  streamlit run app.py
Panel A = diagnostics (scores over time, what was found, camera advice)
Panel B = original vs corrected frames side by side
"""
import os
import tempfile

import pandas as pd
import streamlit as st

from pipeline import process_video

st.set_page_config(page_title="CCTV Diagnosis & Enhancement", layout="wide")
st.title("CCTV Stream Diagnosis & Enhancement")
st.caption("v1 skeleton: recorded video only. RTSP comes later.")

up = st.file_uploader("Upload a recorded video", type=["mp4", "avi", "mov", "mkv"])
every = st.sidebar.slider("Diagnose every N frames", 5, 60, 15)

if up and st.button("Analyse"):
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, up.name)
    with open(src, "wb") as f:
        f.write(up.getbuffer())
    with st.spinner("Diagnosing and correcting..."):
        st.session_state["report"] = process_video(src, os.path.join(tmp, "out"), every)
        st.session_state["out_dir"] = os.path.join(tmp, "out")

rep = st.session_state.get("report")
if rep:
    left, right = st.columns(2)

    with left:
        st.subheader("Panel A: Diagnostics")
        view = st.radio("Scores shown", ["Before fix", "After fix", "Both"], horizontal=True)
        rows = []
        for s in rep["samples"]:
            row = {"time_s": s["time_s"]}
            if view in ("Before fix", "Both"):
                row.update({k: v for k, v in s["scores"].items()})
            if view in ("After fix", "Both"):
                row.update({k + " (after)": v for k, v in s["scores_after"].items()})
            rows.append(row)
        st.line_chart(pd.DataFrame(rows).set_index("time_s"))
        m1, m2, m3 = st.columns(3)
        m1.metric("Brightness jump (corrected / original)", rep["stability"]["ratio"],
                  help="Aim for 1.2 or lower. Much higher means the fix makes the picture pulse.")
        m2.metric("ms per frame", rep["speed"]["ms_per_frame"])
        m3.metric("Samples where a fix made another problem worse",
                  sum(1 for s in rep["samples"] if s["regressions"]))
        st.write("**Share of sampled frames where each problem was found**")
        st.dataframe(pd.DataFrame(rep["summary"]), use_container_width=True)
        st.caption("Advice is shown for problems that software cannot fully fix.")

    with right:
        st.subheader("Panel B: Original vs corrected")
        i = st.slider("Sample", 0, len(rep["samples"]) - 1, 0)
        s = rep["samples"][i]
        fdir = os.path.join(st.session_state["out_dir"], "frames")
        c1, c2 = st.columns(2)
        c1.image(os.path.join(fdir, f"orig_{s['frame']:06d}.jpg"), caption="Original")
        c2.image(os.path.join(fdir, f"corr_{s['frame']:06d}.jpg"), caption="Corrected")
        st.write(f"Fixes applied: {s['applied'] or 'none'}  |  skipped (made it worse): {s['skipped'] or 'none'}")
        with open(rep["corrected_video"], "rb") as f:
            st.download_button("Download corrected video", f, file_name="corrected.mp4")
