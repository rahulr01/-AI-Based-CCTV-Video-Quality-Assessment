"""The ONE interface every problem module must follow.

Each person owns one or more modules. As long as a module has these methods,
the pipeline and the dashboard work with it automatically.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Diagnosis:
    problem: str                 # module name, e.g. "lighting"
    score: float                 # 0.0 = no problem, 1.0 = severe
    details: dict = field(default_factory=dict)   # raw numbers for the dashboard


class ProblemModule:
    name = "base"

    def reset(self):
        """Called once at the start of every video. Clear any memory here."""

    def observe(self, frame: np.ndarray):
        """Called for EVERY frame, before diagnose/correct. Keep running state here
        (for example a slowly-moving brightness average). Must be cheap."""

    def diagnose(self, frame: np.ndarray) -> Diagnosis:
        """Look at a BGR frame and return how bad the problem is (0 to 1)."""
        raise NotImplementedError

    def correct(self, frame: np.ndarray) -> np.ndarray:
        """Return an improved BGR frame (same size as the input)."""
        raise NotImplementedError
