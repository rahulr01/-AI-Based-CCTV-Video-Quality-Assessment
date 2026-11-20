"""Download the Zero-DCE++ code and weights from the authors' GitHub into this project.
Needs git and internet. Check the repo's licence before company use."""
import os
import shutil
import subprocess
import tempfile

URL = "https://github.com/Li-Chongyi/Zero-DCE_extension.git"
tmp = tempfile.mkdtemp()
try:
    subprocess.check_call(["git", "clone", "--depth", "1", URL, os.path.join(tmp, "z")])
except Exception:
    raise SystemExit("git clone failed. Install git, or download the repo ZIP from GitHub by hand and "
                     "copy Zero-DCE++/model.py to nets/zerodce/zerodce_net.py and "
                     "snapshots_Zero_DCE++/Epoch99.pth to weights/zerodce/Epoch99.pth")
src = os.path.join(tmp, "z", "Zero-DCE++")
os.makedirs("nets/zerodce", exist_ok=True)
os.makedirs("weights/zerodce", exist_ok=True)
shutil.copy(os.path.join(src, "model.py"), "nets/zerodce/zerodce_net.py")
shutil.copy(os.path.join(src, "snapshots_Zero_DCE++", "Epoch99.pth"), "weights/zerodce/Epoch99.pth")
open("nets/__init__.py", "a").close()
open("nets/zerodce/__init__.py", "a").close()
print("Done. Zero-DCE++ is ready (needs: pip install torch).")
