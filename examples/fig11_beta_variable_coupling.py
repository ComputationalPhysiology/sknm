# %% [markdown]
# # Figure 11 — the beta cell wave under the harshest conditions the paper tries
#
# Figure 10 is the case where SKNM's assumption holds exactly. This is the case where it holds
# least well: the gap junction conductances are spread as widely as the paper spreads them
# ($\gamma = 1$) so that no single ratio can relate the extracellular and intracellular
# conductance of every connection, and the extracellular space is squeezed to 2%.
#
# For hiPSC-CMs that combination is where KNM and SKNM visibly part company — Figure 4's 2%
# panel has them 6% apart at this $\gamma$. For beta cells they stay together, because the gap
# junction resistance is so large that the conductance ratio is enormous whatever the
# extracellular volume is.
#
# The spread uses the reference implementation's own draws, committed under `data/`. Unlike
# the cardiac sheet, a seeded draw would not do: this sheet has 420 connections and an 8-cell
# conduction path, so the choice of draws moves the wave by 20%.
#
# This page is also the script `examples/fig11_beta_variable_coupling.py`, and runs either
# way. It is cheap enough that `SKNM_EXAMPLES_FULL` changes nothing: there is no sweep to
# reduce, only two runs.

# %%
import matplotlib.pyplot as plt

import common
from sknm import Variant

#: The paper's parameters for this figure: the widest gap junction spread, and 2% of the
#: volume outside the cells.
GAMMA = 1.0
DELTA_E = 0.02

#: The same four moments and the same two models as Figure 10, so the two pages can be read
#: against each other panel for panel.
SNAPSHOT_TIMES = (100.0, 300.0, 500.0, 700.0)
SNAPSHOT_INTERVAL = 100.0

VARIANTS = (Variant.KNM, Variant.SKNM)

options = common.options()
setup = common.BetaSetup(gamma=GAMMA, delta_e=DELTA_E)

# %% [markdown]
# ## The two runs

# %%
cache = common.ResultCache("fig11", enabled=not options.no_cache)

recorded = {}
for variant in VARIANTS:
    label = setup.label(variant=variant, quantity="snapshots")
    recorded[variant] = cache.compute(
        label,
        lambda v=variant: common.snapshots(setup, v, SNAPSHOT_TIMES, SNAPSHOT_INTERVAL),
    )

# %% [markdown]
# ## The figure

# %%
figure = common.snapshot_figure(
    setup,
    recorded,
    VARIANTS,
    [f"t = {time / 1000:g} s" for time in SNAPSHOT_TIMES],
    width=10.0,
    height=4.9,
)
common.write_figure(figure, options.output_dir, "fig11_beta_variable_coupling")
plt.show()

# %% [markdown]
# ## How far apart the rows are, and why
#
# Printed in the same shape as Figure 10's, so the comparison between the easiest case and the
# hardest one is a number rather than an impression.

# %%
common.snapshot_report(recorded, [f"{time / 1000:g}" for time in SNAPSHOT_TIMES], VARIANTS, "t (s)")

network = setup.network()
print(
    f"\nconductance ratio lambda = {network.lam:.0f}, so SKNM's lambda / (1 + lambda) is "
    f"{network.lam / (1 + network.lam):.6f}"
)
