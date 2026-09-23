"""Figures 12 and S1: beta cell conduction velocity against gap junction variation.

The beta counterpart of Figures 4 and S2, and the contrast with them is the point. There, the
three models separate as the extracellular space shrinks, and SKNM(u_e=0) is 12% out at 2%.
Here all four panels and all three models coincide.

That is not a coincidence, and the figure prints the reason rather than leaving it to the eye:
the conductance ratio lambda is around 65,000 at every extracellular volume the paper tries,
because a gap junction resistance a thousand times the cardiac one swamps both conductivities.
SKNM's `lambda / (1 + lambda)` is then 1 to five figures, which is exactly the assumption
SKNM(u_e=0) makes outright -- so the three models are solving the same algebraic system. The
Supplementary predicts this in words ("we would also arrive at the same model if we assumed
Ge >> Gi"); the table below is the number.

One script for both figures because they are one sweep: **S1 is Figure 12 with SKNM(u_e=0)
added**, and computing it separately would mean running the other two models twice.

The sweep uses the reference implementation's own draws, committed under `data/`, because a
seeded draw moves this sheet's velocity by 20% at ``gamma = 1``.

    python examples/fig12_beta_gap_junction_variation.py

Set ``SKNM_EXAMPLES_FULL=1`` to sweep the paper's full sample rather than every other
point of it.
"""

from __future__ import annotations

import common
import plotting
from sknm import Variant

#: Figure 12 compares the two full models; S1 adds the one that drops the extracellular
#: potential rather than eliminating it.
FIGURE_12_VARIANTS = (Variant.KNM, Variant.SKNM)
FIGURE_S1_VARIANTS = (Variant.KNM, Variant.SKNM, Variant.SKNM_UE0)


def main() -> None:
    options = common.options()
    variations = common.sample(common.GAP_JUNCTION_VARIATIONS, options)
    cache = common.ResultCache(
        f"fig12-{'full' if options.full else 'fast'}", enabled=not options.no_cache
    )

    velocities: dict[tuple[float, float, Variant], float] = {}
    for delta_e in common.EXTRACELLULAR_FRACTIONS:
        for gamma in variations:
            setup = common.BetaSetup(delta_e=delta_e, gamma=gamma)
            for variant in FIGURE_S1_VARIANTS:
                label = setup.label(variant=variant, quantity="conduction_velocity")
                value = cache.compute(
                    label,
                    lambda s=setup, v=variant: common.measure_conduction_velocity(s, v),
                )
                velocities[delta_e, gamma, variant] = float(value)

    plotting.use_house_style()
    for stem, variants in (
        ("fig12_beta_gap_junction_variation", FIGURE_12_VARIANTS),
        ("figS1_beta_gap_junction_variation", FIGURE_S1_VARIANTS),
    ):
        figure = _draw(variations, velocities, variants)
        common.write_figure(figure, options.output_dir, stem)

    common.print_table(
        [
            "gamma",
            *(
                f"{common.panel_title(d)} {v.name}"
                for d in common.EXTRACELLULAR_FRACTIONS
                for v in FIGURE_S1_VARIANTS
            ),
        ],
        [
            [
                f"{gamma:g}",
                *(
                    velocities[delta_e, gamma, variant]
                    for delta_e in common.EXTRACELLULAR_FRACTIONS
                    for variant in FIGURE_S1_VARIANTS
                ),
            ]
            for gamma in variations
        ],
    )
    _report_why_the_panels_coincide(variations, velocities)


def _report_why_the_panels_coincide(variations, velocities):
    """Print lambda and the largest spread between models, per panel.

    Four panels a reader cannot tell apart are a result, but a picture of them cannot show
    whether they were computed or copied. These two columns are what make it evidence.
    """
    rows = []
    for delta_e in common.EXTRACELLULAR_FRACTIONS:
        lam = common.BetaSetup(delta_e=delta_e).network().lam
        spread = max(
            max(velocities[delta_e, gamma, v] for v in FIGURE_S1_VARIANTS)
            - min(velocities[delta_e, gamma, v] for v in FIGURE_S1_VARIANTS)
            for gamma in variations
        )
        reference = max(velocities[delta_e, gamma, Variant.KNM] for gamma in variations)
        rows.append(
            [
                common.panel_title(delta_e),
                f"{lam:.0f}",
                f"{lam / (1 + lam):.6f}",
                f"{100 * spread / reference:.4f}%",
            ]
        )
    print()
    common.print_table(["panel", "lambda", "lambda/(1+lambda)", "widest model spread"], rows)
    across_panels = max(
        abs(velocities[a, gamma, Variant.KNM] - velocities[b, gamma, Variant.KNM])
        for gamma in variations
        for a in common.EXTRACELLULAR_FRACTIONS
        for b in common.EXTRACELLULAR_FRACTIONS
    )
    largest = max(
        velocities[d, g, Variant.KNM] for d in common.EXTRACELLULAR_FRACTIONS for g in variations
    )
    print(
        f"\nwidest disagreement between panels, at equal gamma: "
        f"{100 * across_panels / largest:.4f}% of the largest velocity plotted"
    )


def _draw(variations, velocities, variants):
    """One small-multiples figure over the four extracellular volume fractions."""
    # One y scale across all four panels, which the published figure also uses. It is what
    # makes "the panels coincide" visible: on scales of their own, four identical curves are
    # drawn four different sizes and read as four different results.
    figure, axes = plotting.small_multiples(len(common.EXTRACELLULAR_FRACTIONS), share_y=True)
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

    # No direct endpoint labels here, unlike Figures 4 and S2. The models agree on every panel,
    # so a label placed beside one curve would sit on all of them and say nothing; the legend
    # and the line styles carry it instead.
    plotting.variant_legend(figure, variants)
    return figure


if __name__ == "__main__":
    main()
