from .lighting import LightingModule
from .lighting_zerodce import LightingZeroDCE, LightingZeroDCEKeepColour, LightingZeroDCEStable
from .sharpness import SharpnessModule
from .noise import NoiseModule

# To add a module: create the file, import it here, add it to this dict,
# then point a row in routing.yaml to its name. Nothing else changes.
REGISTRY = {
    "lighting": LightingModule(),
    "lighting_zerodce": LightingZeroDCE(),     # loads torch only when first used
    "lighting_zerodce_color": LightingZeroDCEKeepColour(),   # same, original colours kept
    "lighting_zerodce_stable": LightingZeroDCEStable(),      # smoothed over time: use this for video
    "sharpness": SharpnessModule(),
    "noise": NoiseModule(),
}
