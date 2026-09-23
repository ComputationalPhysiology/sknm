# %% [markdown]
# # Figure 8 — continuum conduction velocity against gap junction variation
#
# The counterpart of Figure 4, and the sharpest test of the monodomain assumption. Spreading
# the gap junction conductances makes the intracellular conductivity vary from cell to cell
# while the extracellular conductivity stays constant everywhere, so no single ratio can
# relate the two and the assumption cannot hold at all. The paper's finding is that the two
# models agree closely anyway, except where the extracellular space is small, and that they
# part company further as the spread widens.
#
# This is the first of these figures where the conductivity is genuinely heterogeneous. At
# $\gamma = 0$ the reference's own variation cancels exactly — a draw $a$ scales the
# resistance by $1/(a + (1 - a))$ — so Figures 6 and 7 are drawn on a uniform field and this
# one is not. It is what the element size is chosen for: every element lies inside one cell,
# so the field is represented exactly rather than averaged across a boundary.
#
# This page needs `dolfinx` and `fenicsx-beat`, which `sknm` does not depend on. It is also
# the script `examples/fig08_continuum_gap_junction_variation.py`, and runs either way. Set
# `SKNM_EXAMPLES_FULL=1` to sweep the paper's full sample rather than every other point.

# %%
import matplotlib.pyplot as plt

import bidomain
import common
import plotting

bidomain.require()

MODELS = ("bidomain", "monodomain")

options = common.options()
variations = common.sample(common.GAP_JUNCTION_VARIATIONS, options)
print(variations)

# %% [markdown]
# ## The sweep

# %%
cache = common.ResultCache(
    f"fig08-{'full' if options.full else 'fast'}", enabled=not options.no_cache
)

velocities: dict[tuple[float, float, str], float] = {}
for delta_e in common.EXTRACELLULAR_FRACTIONS:
    for gamma in variations:
        setup = bidomain.BidomainSetup(delta_e=delta_e, gamma=gamma)
        for model in MODELS:
            label = setup.label(model=model, quantity="conduction_velocity")
            value = cache.compute(
                label,
                lambda s=setup, m=model: bidomain.run(s, m).conduction_velocity,
            )
            velocities[delta_e, gamma, model] = float(value)

# %% [markdown]
# ## The figure

# %%
plotting.use_house_style()
figure, axes = plotting.small_multiples(len(common.EXTRACELLULAR_FRACTIONS))
for axis, delta_e in zip(axes, common.EXTRACELLULAR_FRACTIONS, strict=True):
    for model in MODELS:
        plotting.plot_series(
            axis,
            variations,
            [velocities[delta_e, gamma, model] for gamma in variations],
            model,
        )
    axis.set_title(common.panel_title(delta_e))
    axis.set_xlabel(r"$\gamma$")
    axis.set_xticks([0, 0.5, 1])
axes[0].set_ylabel("CV (cm/s)")

smallest, last = common.EXTRACELLULAR_FRACTIONS[-1], variations[-1]
plotting.label_endpoints(
    axes[-1], last, {model: velocities[smallest, last, model] for model in MODELS}
)
plotting.variant_legend(figure, MODELS)

common.write_figure(figure, options.output_dir, "fig08_continuum_gap_junction_variation")
plt.show()

# %% [markdown]
# ## The numbers behind it

# %%
common.print_table(
    [
        "gamma",
        *(
            f"{common.panel_title(d)} {plotting.SERIES_LABEL[m]}"
            for d in common.EXTRACELLULAR_FRACTIONS
            for m in MODELS
        ),
    ],
    [
        [
            f"{gamma:g}",
            *(
                velocities[delta_e, gamma, model]
                for delta_e in common.EXTRACELLULAR_FRACTIONS
                for model in MODELS
            ),
        ]
        for gamma in variations
    ],
)
