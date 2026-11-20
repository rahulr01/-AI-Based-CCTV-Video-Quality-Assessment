"""TEMPLATE: how to plug a pretrained neural model into a module.
This file is NOT registered, so it does not run until you finish it.

Steps:
 1. Find the model's GitHub repo (for example Zero-DCE++). Put its weights file
    (.pth) in the weights/ folder. Do NOT commit big weight files to git; share
    them through Drive.
 2. Open the repo's demo / test script. It shows how to load the weights and run
    ONE image. Copy that logic into _load() and correct() below.
 3. Copy the repo's network class (model.py) into a models/ folder and import it.
 4. Register the class in modules/__init__.py and point the routing.yaml row to it.

Three rules:
 - Load the model ONCE (lazy load below), never inside the per-frame code.
 - correct() takes a BGR frame and returns a BGR frame of the same size.
 - diagnose() can stay classical. Only the FIX needs a neural model in v1.
"""
import cv2
import numpy as np

from .base import ProblemModule, Diagnosis


class NeuralLightingTemplate(ProblemModule):
    name = "lighting"

    def __init__(self):
        self.net = None
        self.device = "cpu"

    def _load(self):
        import torch                                    # imported here so the project runs without torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # from models.zero_dce import EnhanceNet        # <- the class copied from the repo
        # self.net = EnhanceNet().to(self.device)
        # self.net.load_state_dict(torch.load("weights/model.pth", map_location=self.device))
        # self.net.eval()
        raise NotImplementedError("Fill in the 4 commented lines from the model's repo")

    def diagnose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        return Diagnosis(self.name, float(np.clip((90 - mean) / 90, 0, 1)), {"mean_brightness": round(mean, 1)})

    def correct(self, frame):
        if self.net is None:
            self._load()
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y = self.net(x)        # some models return several outputs: take the enhanced image
        out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype("uint8")
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
