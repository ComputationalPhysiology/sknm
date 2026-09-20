"""Membrane models and the seam they plug into.

Three models are available: `base_model_IM`, a human induced pluripotent stem cell derived
cardiomyocyte model; `PBM`, the phantom bursting model of the pancreatic beta cell; and
`fitzhugh_nagumo`, a hand-written two-state model. The first two are generated from
`base_model_IM.ode` and `PBM.ode` by `tools/generate_membrane_models.py`.

The two generated models differ in one way that matters to a caller. `base_model_IM` names its
membrane potential ``V_m`` and integrates ``dV/dt = -I_tot``, so it has no capacitance to
declare. `PBM` names it ``v`` and integrates ``dv/dt = -I / Cm``, so it does::

    from sknm.membrane import PBM, from_gotranx
    from sknm.units import fF

    model = from_gotranx(PBM, v_name="v", capacitance=5300 * fF)

`sknm.presets.beta_membrane_model` builds exactly that, and is what to reach for rather than
retyping the capacitance.

Any object satisfying `MembraneModel` can be used in their place. To use your own `.ode`,
generate it with `tools/generate_membrane_models.py` and wrap it with `from_gotranx`.
"""

from sknm.membrane import PBM, base_model_IM
from sknm.membrane.fitzhugh_nagumo import FitzHughNagumo, fitzhugh_nagumo
from sknm.membrane.protocol import GotranxModel, MembraneModel, from_gotranx

__all__ = [
    "PBM",
    "FitzHughNagumo",
    "GotranxModel",
    "MembraneModel",
    "base_model_IM",
    "fitzhugh_nagumo",
    "from_gotranx",
]
