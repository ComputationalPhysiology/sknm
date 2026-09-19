"""Membrane models and the seam they plug into.

Two models are available: `base_model_IM`, a human induced pluripotent stem cell derived
cardiomyocyte model generated from `base_model_IM.ode` by
`tools/generate_membrane_models.py`; and `fitzhugh_nagumo`, a hand-written two-state model.

Any object satisfying `MembraneModel` can be used in their place. To use your own `.ode`,
generate it with `tools/generate_membrane_models.py` and wrap it with `from_gotranx`.
"""

from sknm.membrane import base_model_IM
from sknm.membrane.fitzhugh_nagumo import FitzHughNagumo, fitzhugh_nagumo
from sknm.membrane.protocol import GotranxModel, MembraneModel, from_gotranx

__all__ = [
    "FitzHughNagumo",
    "GotranxModel",
    "MembraneModel",
    "base_model_IM",
    "fitzhugh_nagumo",
    "from_gotranx",
]
