"""Figure 11: the same beta cell wave under the harshest conditions the paper tries.

Figure 10 is the case where SKNM's assumption holds exactly. This is the case where it holds
least well: the gap junction conductances are spread as widely as the paper spreads them
(``gamma = 1``) so that no single ratio can relate the extracellular and intracellular
conductance of every connection, and the extracellular space is squeezed to 2%.

For hiPSC-CMs that combination is where KNM and SKNM visibly part company -- Figure 4's 2%
panel has them 6% apart at this gamma. For beta cells they stay together, because the gap
junction resistance is so large that the conductance ratio is enormous whatever the
extracellular volume is. The figure prints how far apart the two rows actually are, alongside
Figure 10's, so the comparison is a number rather than an impression.

The spread uses the reference implementation's own draws, committed under `data/`. Unlike the
cardiac sheet, a seeded draw would not do: this sheet has 420 connections and an 8-cell
conduction path, so the choice of draws moves the wave by 20%.

Cheap enough that ``SKNM_EXAMPLES_FULL`` changes nothing: there is no sweep to reduce, only
two runs.

    python examples/fig11_beta_variable_coupling.py
"""

from __future__ import annotations

import common
import fig10_beta_travelling_wave as fig10

#: The paper's parameters for this figure: the widest gap junction spread, and 2% of the
#: volume outside the cells.
GAMMA = 1.0
DELTA_E = 0.02


def main() -> None:
    options = common.options()
    setup = common.BetaSetup(gamma=GAMMA, delta_e=DELTA_E)
    cache = common.ResultCache("fig11", enabled=not options.no_cache)

    recorded = {}
    for variant in fig10.VARIANTS:
        label = setup.label(variant=variant, quantity="snapshots")
        recorded[variant] = cache.compute(label, lambda v=variant: fig10.snapshots(setup, v))

    fig10.draw(setup, recorded, "fig11_beta_variable_coupling", options)
    fig10.report(setup, recorded)


if __name__ == "__main__":
    main()
