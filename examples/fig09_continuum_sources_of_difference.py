# %% [markdown]
# # Figure 9 — why the two continuum models differ
#
# The counterpart of Figure 5, and the same argument about a different pair of models. The
# error the monodomain model makes is the product of two things: how far the conductivities
# are from being related by a single ratio, and how much the extracellular potential varies.
#
# - **Left.** $F(\lambda)$, the misfit the ratio is chosen to minimize, as the gap junction
#   conductances are spread. It rises from zero, so the first factor grows with the spread.
#   This needs no mesh at all — the intracellular conductivity is constant within a cell and
#   the extracellular one is constant everywhere, so the integral is an exact finite sum.
# - **Right.** How far apart the extracellular potential gets across the sheet during a run,
#   as the extracellular volume fraction grows. It falls, so the second factor shrinks. This
#   needs the bidomain model, the only one of the two that solves for an extracellular
#   potential.
#
# Figure 5's left panel is the same picture of a different quantity. That one is equation
# (29), a sum over the network's connections of conductances, in (mS/cm)²; this one is
# equation (31), an area integral of conductivities, in mS². The two differ by four orders of
# magnitude and neither can be computed from the other.
#
# This page needs `dolfinx` and `fenicsx-beat`, which `sknm` does not depend on. It is also
# the script `examples/fig09_continuum_sources_of_difference.py`, and runs either way. Set
# `SKNM_EXAMPLES_FULL=1` to sweep the paper's full sample rather than every other point.

# %% [markdown]
# :::{note}
# This page shows the code and the figure it draws, but the site does not run it. The four
# continuum examples need `dolfinx` and sweep for hours, which is more than a pull request
# can carry; the seven network and beta pages are executed as you read them. The figure
# below is the one this code drew, committed under `docs/figures/` and refreshed by a
# scheduled job that reruns these four. Run the script yourself for the printed tables.
# :::

# %%
import matplotlib.pyplot as plt
import numpy as np

import bidomain
import common
import plotting

bidomain.require()

#: The paper draws the left panel at the default extracellular volume and the right panel at a
#: spread halfway to the widest, as Figure 5 does.
MISFIT_DELTA_E = 0.2
RANGE_GAMMA = 0.5

options = common.options()
fractions = common.sample(common.VOLUME_FRACTIONS, options)
print(fractions)

# %% [markdown]
# ## The left panel: the misfit
#
# Equation (31) over the sheet's conductivities. No mesh and no solve: the fields are
# piecewise constant, so the integral is a finite sum.

# %%
misfits = [
    bidomain.BidomainSetup(delta_e=MISFIT_DELTA_E, gamma=gamma).conductivity_misfit()
    for gamma in common.MISFIT_VARIATIONS
]

# %% [markdown]
# ## The right panel: the extracellular spread
#
# Read every step rather than every millisecond, because the spread peaks as the wave front
# passes, which takes about as long as the upstroke does. The run is held open for the whole
# action potential rather than stopping when the wave reaches the far probe, so that a sheet
# whose spread peaks late is not cut off.


# %%
def extracellular_range(setup):
    """The widest the extracellular potential gets across the sheet at any one moment."""
    widest = 0.0

    def watch(t, pde):
        nonlocal widest
        potential = pde.u_e.x.array
        widest = max(widest, float(potential.max() - potential.min()))

    bidomain.run(setup, "bidomain", callback=watch, minimum_time=setup.t_end)
    return widest


cache = common.ResultCache(
    f"fig09-{'full' if options.full else 'fast'}", enabled=not options.no_cache
)
ranges = []
for delta_e in fractions:
    setup = bidomain.BidomainSetup(delta_e=delta_e, gamma=RANGE_GAMMA)
    label = setup.label(quantity="extracellular_range", model="bidomain")
    ranges.append(float(cache.compute(label, lambda s=setup: extracellular_range(s))))

# %% [markdown]
# ## The figure

# %%
plotting.use_house_style()
figure, axes = plotting.small_multiples(2, width=8.0, height=3.3)

axes[0].plot(
    common.MISFIT_VARIATIONS,
    misfits,
    color=plotting.SERIES_COLOUR["bidomain"],
    linestyle="-",
)
axes[0].set_title(r"$F(\lambda)$")
axes[0].set_xlabel(r"$\gamma$")
axes[0].set_ylabel(r"mS$^2$")
axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

axes[1].plot(
    fractions,
    ranges,
    color=plotting.SERIES_COLOUR["bidomain"],
    linestyle="-",
    marker="o",
    markeredgecolor=plotting.SURFACE,
    markeredgewidth=1.0,
)
axes[1].set_title(r"max($u_e$) - min($u_e$)")
axes[1].set_xlabel(r"$\delta_e$")
axes[1].set_ylabel("mV")
axes[1].set_ylim(bottom=0.0)

common.write_figure(figure, options.output_dir, "fig09_continuum_sources_of_difference")
plt.show()

# %% [markdown]
# ```{figure} ../figures/fig09_continuum_sources_of_difference.png
# :alt: The conductivity misfit rising and the extracellular spread falling
#
# One factor grows with the gap junction spread and the other shrinks as the
# extracellular volume grows, which together say where the two models should differ.
# ```


# %% [markdown]
# ## The numbers behind it

# %%
common.print_table(
    ["gamma", "F(lambda) (mS^2)"],
    [
        [f"{gamma:g}", misfit]
        for gamma, misfit in zip(common.MISFIT_VARIATIONS, misfits, strict=True)
        if gamma in np.round(np.arange(0.0, 1.01, 0.2), 3)
    ],
)
print()
common.print_table(
    ["delta_e", "max(u_e) - min(u_e) (mV)"],
    [[f"{fraction:g}", value] for fraction, value in zip(fractions, ranges, strict=True)],
)
