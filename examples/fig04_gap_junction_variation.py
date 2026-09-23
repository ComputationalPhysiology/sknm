"""Figures 4 and S2: conduction velocity against gap junction variation.

Spreading the gap junction conductances is the sharpest test of SKNM's assumption. The
extracellular conductance of every connection stays the same while the intracellular
conductance varies, so no single ratio can relate them and the assumption cannot hold at all.
The paper's finding is that the two models still agree closely, except where the extracellular
space is small, and that they part company further as the spread widens.

One script for both figures because they are one sweep: **S2 is Figure 4 with SKNM(u_e=0)
added**, and computing it separately would mean running the other two models twice.

    python examples/fig04_gap_junction_variation.py

Set ``SKNM_EXAMPLES_FULL=1`` to sweep the paper's full sample rather than every other
point of it.
"""

from __future__ import annotations

import common
import plotting
from sknm import Variant

#: Figure 4 compares the two full models; S2 adds the one that drops the extracellular
#: potential rather than eliminating it.
FIGURE_4_VARIANTS = (Variant.KNM, Variant.SKNM)
FIGURE_S2_VARIANTS = (Variant.KNM, Variant.SKNM, Variant.SKNM_UE0)


def main() -> None:
    options = common.options()
    variations = common.sample(common.GAP_JUNCTION_VARIATIONS, options)
    cache = common.ResultCache(
        f"fig04-{'full' if options.full else 'fast'}", enabled=not options.no_cache
    )

    velocities: dict[tuple[float, float, Variant], float] = {}
    for delta_e in common.EXTRACELLULAR_FRACTIONS:
        for gamma in variations:
            setup = common.Setup(delta_e=delta_e, gamma=gamma)
            for variant in FIGURE_S2_VARIANTS:
                label = setup.label(variant=variant, quantity="conduction_velocity")
                value = cache.compute(
                    label,
                    lambda s=setup, v=variant: common.measure_conduction_velocity(s, v),
                )
                velocities[delta_e, gamma, variant] = float(value)

    plotting.use_house_style()
    for stem, variants in (
        ("fig04_gap_junction_variation", FIGURE_4_VARIANTS),
        ("figS2_gap_junction_variation", FIGURE_S2_VARIANTS),
    ):
        figure = _draw(variations, velocities, variants)
        common.write_figure(figure, options.output_dir, stem)

    common.print_table(
        [
            "gamma",
            *(
                f"{common.panel_title(d)} {v.name}"
                for d in common.EXTRACELLULAR_FRACTIONS
                for v in FIGURE_S2_VARIANTS
            ),
        ],
        [
            [
                f"{gamma:g}",
                *(
                    velocities[delta_e, gamma, variant]
                    for delta_e in common.EXTRACELLULAR_FRACTIONS
                    for variant in FIGURE_S2_VARIANTS
                ),
            ]
            for gamma in variations
        ],
    )


def _draw(variations, velocities, variants):
    """One small-multiples figure over the four extracellular volume fractions."""
    figure, axes = plotting.small_multiples(len(common.EXTRACELLULAR_FRACTIONS))
    for axis, delta_e in zip(axes, common.EXTRACELLULAR_FRACTIONS, strict=True):
        for variant in variants:
            plotting.plot_series(
                axis,
                variations,
                [velocities[delta_e, gamma, variant] for gamma in variations],
                variant,
            )
        axis.set_title(common.panel_title(delta_e))
        axis.set_xlabel(r"$\gamma$")
        axis.set_xticks([0, 0.5, 1])
    axes[0].set_ylabel("CV (cm/s)")

    smallest, last = common.EXTRACELLULAR_FRACTIONS[-1], variations[-1]
    plotting.label_endpoints(
        axes[-1], last, {variant: velocities[smallest, last, variant] for variant in variants}
    )
    plotting.variant_legend(figure, variants)
    return figure


if __name__ == "__main__":
    main()
