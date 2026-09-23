# %% [markdown]
# # Figure 3 — conduction velocity against cell shape
#
# Elongating the cells while holding their volume near 4 pL breaks the assumption SKNM rests
# on, because an x-direction connection and a y-direction connection no longer share a
# conductance ratio. The paper's finding is that SKNM tracks KNM anyway, and that the two
# part company only where the extracellular space is small.
#
# This page is also the script `examples/fig03_anisotropy.py`, and runs either way. Set
# `SKNM_EXAMPLES_FULL=1` to sweep the paper's full sample rather than every other point of
# it; the reduction changes how many markers the curves carry and nothing else.

# %%
import matplotlib.pyplot as plt

import common
import plotting
from sknm import Variant

VARIANTS = (Variant.KNM, Variant.SKNM)

# %% [markdown]
# ## The sweep
#
# The five anisotropy factors are the ones the reference implementation ships meshes for.
# The cell size at each of them follows from `sknm.presets.hipsc_cell_size` rather than a
# lookup table, so the sweep is not limited to those five.

# %%
options = common.options()
factors = common.sample(common.ANISOTROPY_FACTORS, options)
print(factors)

# %% [markdown]
# ## Measuring it
#
# One conduction velocity for every combination of extracellular volume fraction,
# anisotropy factor and model: each is the paper's own 40x40 sheet, run at its 0.02 ms step
# until the wave crosses the second measurement column. Results are cached under
# `examples/results/`, so drawing the figure a second time costs no simulation.

# %%
cache = common.ResultCache(
    f"fig03-{'full' if options.full else 'fast'}", enabled=not options.no_cache
)

velocities: dict[tuple[float, float, Variant], float] = {}
for delta_e in common.EXTRACELLULAR_FRACTIONS:
    for alpha in factors:
        setup = common.Setup(alpha=alpha, delta_e=delta_e)
        for variant in VARIANTS:
            label = setup.label(variant=variant, quantity="conduction_velocity")
            value = cache.compute(
                label,
                lambda s=setup, v=variant: common.measure_conduction_velocity(s, v),
            )
            velocities[delta_e, alpha, variant] = float(value)

# %% [markdown]
# ## The figure
#
# One panel per extracellular volume fraction, largest first. The two models are drawn in
# the paper's own line styles — KNM solid, SKNM dotted — so the panels can be read against
# it directly.

# %%
plotting.use_house_style()
figure, axes = plotting.small_multiples(len(common.EXTRACELLULAR_FRACTIONS))
for axis, delta_e in zip(axes, common.EXTRACELLULAR_FRACTIONS, strict=True):
    for variant in VARIANTS:
        plotting.plot_series(
            axis,
            factors,
            [velocities[delta_e, alpha, variant] for alpha in factors],
            variant,
        )
    axis.set_title(common.panel_title(delta_e))
    axis.set_xlabel(r"$\alpha$")
    axis.set_xticks([1, 2, 3, 4])
axes[0].set_ylabel("CV (cm/s)")

smallest = common.EXTRACELLULAR_FRACTIONS[-1]
plotting.label_endpoints(
    axes[-1],
    factors[-1],
    {variant: velocities[smallest, factors[-1], variant] for variant in VARIANTS},
)
plotting.variant_legend(figure, VARIANTS)

common.write_figure(figure, options.output_dir, "fig03_anisotropy")
plt.show()

# %% [markdown]
# ## The numbers behind it
#
# The two models stay within a fraction of a percent of each other across the whole sweep at
# 50% extracellular volume, and separate as that volume shrinks. The last column pair is
# where the paper's claim is tested hardest.

# %%
common.print_table(
    [
        "alpha",
        *(
            f"{common.panel_title(d)} {v.name}"
            for d in common.EXTRACELLULAR_FRACTIONS
            for v in VARIANTS
        ),
    ],
    [
        [
            f"{alpha:g}",
            *(
                velocities[delta_e, alpha, variant]
                for delta_e in common.EXTRACELLULAR_FRACTIONS
                for variant in VARIANTS
            ),
        ]
        for alpha in factors
    ],
)
