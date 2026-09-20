"""Figure 3: conduction velocity against cell length-to-width ratio, at four extracellular volumes.

Elongating the cells while holding their volume near 4 pL breaks the assumption SKNM rests on,
because an x-direction connection and a y-direction connection no longer share a conductance
ratio. The paper's finding is that SKNM tracks KNM anyway, and that the two part company only
where the extracellular space is small.

The five anisotropy factors are the ones the reference implementation ships meshes for; the
cell size at each of them follows from `sknm.presets.hipsc_cell_size`, so the sweep is not
limited to them.

    python examples/fig03_anisotropy.py [--full]
"""

from __future__ import annotations

import common
import plotting
from sknm import Variant

VARIANTS = (Variant.KNM, Variant.SKNM)


def main() -> None:
    args = common.parse_args(__doc__.splitlines()[0])
    factors = common.sample(common.ANISOTROPY_FACTORS, args)
    cache = common.ResultCache(
        f"fig03-{'full' if args.full else 'fast'}", enabled=not args.no_cache
    )

    velocities: dict[tuple[float, Variant], float] = {}
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

    common.write_figure(figure, args.output_dir, "fig03_anisotropy")

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


if __name__ == "__main__":
    main()
