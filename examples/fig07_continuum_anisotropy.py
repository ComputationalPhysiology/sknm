# %% [markdown]
# # Figure 7: continuum conduction velocity against cell shape
#
# The continuum counterpart of Figure 3. Elongating the cells while holding their volume near
# 4 pL makes the intracellular conductivity anisotropic, and the monodomain model has only one
# conductivity to be anisotropic with: it replaces the pair by a single ratio chosen to fit
# both directions at once, which no single number can do when the two differ. The paper's
# finding is that the monodomain model tracks the bidomain one anyway, and that the two part
# company where the extracellular space is small.
#
# The five anisotropy factors are the ones the reference implementation ships meshes for. The
# mesh is rebuilt for each of them, since the cell changes shape: the element size follows the
# cell so that no element straddles a cell boundary.
#
# This page needs `dolfinx` and `fenicsx-beat`, which `sknm` does not depend on. It is also
# the script `examples/fig07_continuum_anisotropy.py`, and runs either way. Set
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

import bidomain
import common
import plotting

bidomain.require()

MODELS = ("bidomain", "monodomain")

options = common.options()
factors = common.sample(common.ANISOTROPY_FACTORS, options)
print(factors)

# %% [markdown]
# ## The sweep
#
# One conduction velocity per extracellular volume fraction, anisotropy factor and model.
# Each is a fresh mesh and a fresh solve, which is why this page is the expensive one.

# %%
cache = common.ResultCache(
    f"fig07-{'full' if options.full else 'fast'}", enabled=not options.no_cache
)

velocities: dict[tuple[float, float, str], float] = {}
for delta_e in common.EXTRACELLULAR_FRACTIONS:
    for alpha in factors:
        setup = bidomain.BidomainSetup(alpha=alpha, delta_e=delta_e)
        for model in MODELS:
            label = setup.label(model=model, quantity="conduction_velocity")
            value = cache.compute(
                label,
                lambda s=setup, m=model: bidomain.run(s, m).conduction_velocity,
            )
            velocities[delta_e, alpha, model] = float(value)

# %% [markdown]
# ## The figure
#
# Drawn in the same colours and line styles as Figure 3, so the network and continuum halves
# of the paper can be laid side by side.

# %%
plotting.use_house_style()
figure, axes = plotting.small_multiples(len(common.EXTRACELLULAR_FRACTIONS))
for axis, delta_e in zip(axes, common.EXTRACELLULAR_FRACTIONS, strict=True):
    for model in MODELS:
        plotting.plot_series(
            axis,
            factors,
            [velocities[delta_e, alpha, model] for alpha in factors],
            model,
        )
    axis.set_title(common.panel_title(delta_e))
    axis.set_xlabel(r"$\alpha$")
    axis.set_xticks([1, 2, 3, 4])
axes[0].set_ylabel("CV (cm/s)")

smallest = common.EXTRACELLULAR_FRACTIONS[-1]
plotting.label_endpoints(
    axes[-1],
    factors[-1],
    {model: velocities[smallest, factors[-1], model] for model in MODELS},
)
plotting.variant_legend(figure, MODELS)

common.write_figure(figure, options.output_dir, "fig07_continuum_anisotropy")
plt.show()

# %% [markdown]
# ```{figure} ../figures/fig07_continuum_anisotropy.png
# :alt: Conduction velocity against anisotropy factor, at four extracellular volume fractions
#
# The monodomain model tracks the bidomain one until the extracellular space is small.
# ```


# %% [markdown]
# ## The numbers behind it

# %%
common.print_table(
    [
        "alpha",
        *(
            f"{common.panel_title(d)} {plotting.SERIES_LABEL[m]}"
            for d in common.EXTRACELLULAR_FRACTIONS
            for m in MODELS
        ),
    ],
    [
        [
            f"{alpha:g}",
            *(
                velocities[delta_e, alpha, model]
                for delta_e in common.EXTRACELLULAR_FRACTIONS
                for model in MODELS
            ),
        ]
        for alpha in factors
    ],
)
