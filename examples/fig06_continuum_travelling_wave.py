# %% [markdown]
# # Figure 6 — a travelling wave, solved as bidomain and as monodomain
#
# The continuum counterpart of Figure 2, and the same claim made about a different pair of
# models: snapshots of the membrane potential at three points in time, one row per model,
# which should be indistinguishable.
#
# Here they are more than indistinguishable. The paper's default sheet has uniform gap
# junction conductances and square cells, so the intracellular conductivity is both spatially
# uniform and isotropic, one ratio relates it to the extracellular conductivity everywhere,
# and the bidomain system reduces to the monodomain one exactly rather than approximately.
# The table at the bottom prints how far apart the two rows are, and the answer is the
# precision of the linear solves.
#
# This page needs `dolfinx` and `fenicsx-beat`, which `sknm` does not depend on. It is also
# the script `examples/fig06_continuum_travelling_wave.py`, and runs either way.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bidomain
import common
import plotting

bidomain.require()

#: The paper's three snapshot times, shared with Figure 2.
SNAPSHOT_TIMES = (25.0, 30.0, 35.0)

MODELS = ("bidomain", "monodomain")
MODEL_LABEL = {"bidomain": "BD", "monodomain": "MD"}

options = common.options()
setup = bidomain.BidomainSetup()

# %% [markdown]
# ## The two runs
#
# Each run is held open to the last snapshot time. Left to itself it would stop as soon as the
# wave reached the far probe, which is earlier than the last panel.

# %%
cache = common.ResultCache("fig06", enabled=not options.no_cache)

recorded = {}
for model in MODELS:
    label = setup.label(model=model, quantity="snapshots")
    recorded[model] = cache.compute(
        label, lambda m=model: bidomain.snapshots(setup, m, SNAPSHOT_TIMES)
    )

# %% [markdown]
# ## The figure
#
# Both rows are drawn to one colour scale, so a difference the size of a solver tolerance
# cannot be magnified into a visible one.

# %%
potentials = np.stack(list(recorded.values()))
low, high = float(potentials.min()), float(potentials.max())

plotting.use_house_style()
figure, grid = plotting.panel_grid(len(MODELS), len(SNAPSHOT_TIMES))
for row, model in enumerate(MODELS):
    for column, time in enumerate(SNAPSHOT_TIMES):
        axis = grid[row][column]
        image = plotting.show_sheet(axis, recorded[model][column], low=low, high=high)
        if row == 0:
            axis.set_title(f"t = {time:g} ms")
        if column == 0:
            axis.set_ylabel(MODEL_LABEL[model], fontsize=11, color=plotting.INK)
plotting.colour_scale(figure, image, grid, "membrane potential (mV)")

common.write_figure(figure, options.output_dir, "fig06_continuum_travelling_wave")
plt.show()

# %% [markdown]
# ## How far apart the rows are
#
# The last column is the collapse of the bidomain model onto the monodomain one, measured
# rather than asserted.

# %%
difference = np.abs(recorded["bidomain"] - recorded["monodomain"])
common.print_table(
    ["t (ms)", "BD min", "BD max", "MD min", "MD max", "max |BD - MD|"],
    [
        [
            f"{time:g}",
            float(recorded["bidomain"][index].min()),
            float(recorded["bidomain"][index].max()),
            float(recorded["monodomain"][index].min()),
            float(recorded["monodomain"][index].max()),
            f"{difference[index].max():.2e}",
        ]
        for index, time in enumerate(SNAPSHOT_TIMES)
    ],
)
