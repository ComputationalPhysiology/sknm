# %% [markdown]
# # Figure 2 — a travelling wave across 40x40 hiPSC-CMs
#
# Snapshots of the membrane potential at three points in time, one row per model. This is the
# paper's default set-up, where every connection has the same conductances, so a single ratio
# relates the extracellular and intracellular conductance of all of them and the assumption
# SKNM is derived from holds exactly.
#
# The two rows should be indistinguishable. The point of the page is the number at the bottom,
# which says how indistinguishable they actually are.
#
# This page is also the script `examples/fig02_travelling_wave.py`, and runs either way. It is
# cheap enough that `SKNM_EXAMPLES_FULL` changes nothing: there is no sweep to reduce, only
# two runs.

# %%
import matplotlib.pyplot as plt

import common
from sknm import Variant

#: The paper's three snapshot times in ms, and the interval that lands a sample on each.
SNAPSHOT_TIMES = (25.0, 30.0, 35.0)
SNAPSHOT_INTERVAL = 5.0

VARIANTS = (Variant.KNM, Variant.SKNM)

options = common.options()
setup = common.Setup()

# %% [markdown]
# ## The two runs
#
# One run per model over the paper's own 40x40 sheet, each recording the whole sheet at the
# three snapshot times. Results are cached under `examples/results/`, so redrawing costs no
# simulation.

# %%
cache = common.ResultCache("fig02", enabled=not options.no_cache)

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
# Both rows are drawn to one colour scale. On scales of their own, two sheets differing by a
# rounding error would look like two different results.

# %%
figure = common.snapshot_figure(
    setup, recorded, VARIANTS, [f"t = {time:g} ms" for time in SNAPSHOT_TIMES]
)
common.write_figure(figure, options.output_dir, "fig02_travelling_wave")
plt.show()

# %% [markdown]
# ## How far apart the rows are
#
# The last column is the whole claim: the largest difference between the two models at any
# cell, at each moment drawn above.

# %%
common.snapshot_report(recorded, [f"{time:g}" for time in SNAPSHOT_TIMES], VARIANTS, "t (ms)")
