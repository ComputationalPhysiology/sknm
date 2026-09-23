# %% [markdown]
# # Figure 10 — a travelling wave across 15x15 pancreatic beta cells
#
# The beta counterpart of Figure 2, and the same claim: with every connection carrying the
# same conductances, a single ratio relates the extracellular and intracellular conductance of
# all of them, so SKNM's assumption holds exactly and the two rows should be
# indistinguishable.
#
# It holds far more comfortably here than it does for cardiac cells. The gap junction
# resistance between beta cells is a thousand times the cardiac one, so it swamps both
# conductivities and the conductance ratio comes out around 65,000 rather than 40 — which is
# why the wave is 150 times slower, why the snapshots are in seconds rather than
# milliseconds, and why even SKNM(u_e=0) is indistinguishable from KNM here.
#
# This page is also the script `examples/fig10_beta_travelling_wave.py`, and runs either way.
# It is cheap enough that `SKNM_EXAMPLES_FULL` changes nothing: there is no sweep to reduce,
# only two runs.

# %%
import matplotlib.pyplot as plt

import common
from sknm import Variant

#: The paper's four snapshot times in ms, and the interval that lands a sample on each.
SNAPSHOT_TIMES = (100.0, 300.0, 500.0, 700.0)
SNAPSHOT_INTERVAL = 100.0

VARIANTS = (Variant.KNM, Variant.SKNM)

options = common.options()
setup = common.BetaSetup()

# %% [markdown]
# ## The two runs
#
# The beta wave takes twenty times as long as the cardiac one to cross a sheet a seventh the
# size, so the run is 1000 ms rather than 50.

# %%
cache = common.ResultCache("fig10", enabled=not options.no_cache)

recorded = {}
for variant in VARIANTS:
    label = setup.label(variant=variant, quantity="snapshots")
    recorded[variant] = cache.compute(
        label,
        lambda v=variant: common.snapshots(setup, v, SNAPSHOT_TIMES, SNAPSHOT_INTERVAL),
    )

# %% [markdown]
# ## The figure
#
# Sized for four columns of square panels rather than the default three: an image axis keeps
# its aspect, so leftover figure height turns into a gap between the rows rather than into
# larger panels.

# %%
figure = common.snapshot_figure(
    setup,
    recorded,
    VARIANTS,
    [f"t = {time / 1000:g} s" for time in SNAPSHOT_TIMES],
    width=10.0,
    height=4.9,
)
common.write_figure(figure, options.output_dir, "fig10_beta_travelling_wave")
plt.show()

# %% [markdown]
# ## How far apart the rows are, and why
#
# The conductance ratio is what explains the agreement. SKNM replaces the extracellular
# potential with a correction of $\lambda / (1 + \lambda)$; at this $\lambda$ that factor is
# 1 to five figures, so there is nothing left for the two models to disagree about.

# %%
common.snapshot_report(recorded, [f"{time / 1000:g}" for time in SNAPSHOT_TIMES], VARIANTS, "t (s)")

network = setup.network()
print(
    f"\nconductance ratio lambda = {network.lam:.0f}, so SKNM's lambda / (1 + lambda) is "
    f"{network.lam / (1 + network.lam):.6f}"
)
