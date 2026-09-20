"""Figure 5: why KNM and SKNM differ, as two factors that can be looked at one at a time.

The error SKNM makes on a connection is the product of two things: how far that connection is
from the single conductance ratio SKNM assumes, and how much the extracellular potential
varies across it. Figure 5 shows one of each.

- **Left.** F(lambda), the misfit the ratio is chosen to minimize, as the gap junction
  conductances are spread. It rises from zero, so the first factor grows with the spread.
  This needs no simulation at all -- it is a property of a network's conductances.
- **Right.** How far apart the extracellular potential gets across the sheet during a run,
  as the extracellular volume fraction grows. It falls, so the second factor shrinks. This
  needs KNM, the only model that solves for an extracellular potential.

Together they say the two models should differ most at a wide spread and a small
extracellular space, which is where Figure 4 finds the difference.

    python examples/fig05_sources_of_difference.py [--full]
"""

from __future__ import annotations

import numpy as np

import common
import plotting
from sknm import Variant
from sknm.units import ms

#: The paper draws the left panel at the default extracellular volume and the right panel at
#: a spread halfway to the widest.
MISFIT_DELTA_E = 0.2
RANGE_GAMMA = 0.5


def extracellular_range(setup: common.Setup) -> float:
    """The widest the extracellular potential gets across the sheet at any one moment.

    Recorded every step rather than every millisecond: the spread peaks as the wave front
    passes, which takes about as long as the upstroke does.

    Parameters
    ----------
    setup : common.Setup
        The sheet to run, as KNM.

    Returns
    -------
    float
        ``max(u_e) - min(u_e)`` over the cells, at the moment it is largest, in mV.
    """
    simulation = setup.simulation(Variant.KNM)
    result = simulation.run(common.T_END * ms, record=("u_e",), record_every=setup.dt * ms)
    # The first sample is taken before any step, and the extracellular potential is produced
    # by a step, so that column is nan for every cell.
    potential = result.u_e[:, 1:]
    return float((potential.max(axis=0) - potential.min(axis=0)).max())


def main() -> None:
    args = common.parse_args(__doc__.splitlines()[0])
    fractions = common.sample(common.VOLUME_FRACTIONS, args)
    cache = common.ResultCache(
        f"fig05-{'full' if args.full else 'fast'}", enabled=not args.no_cache
    )

    misfits = [
        common.Setup(delta_e=MISFIT_DELTA_E, gamma=gamma).network().conductance_misfit()
        for gamma in common.MISFIT_VARIATIONS
    ]
    ranges = []
    for delta_e in fractions:
        setup = common.Setup(delta_e=delta_e, gamma=RANGE_GAMMA)
        label = setup.label(quantity="extracellular_range", variant=Variant.KNM)
        ranges.append(float(cache.compute(label, lambda s=setup: extracellular_range(s))))

    plotting.use_house_style()
    figure, axes = plotting.small_multiples(2, width=8.0, height=3.3)

    axes[0].plot(
        common.MISFIT_VARIATIONS,
        misfits,
        color=plotting.SERIES_COLOUR[Variant.KNM],
        linestyle="-",
    )
    axes[0].set_title(r"$F(\lambda)$")
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel(r"(mS/cm)$^2$")
    axes[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    axes[1].plot(
        fractions,
        ranges,
        color=plotting.SERIES_COLOUR[Variant.KNM],
        linestyle="-",
        marker="o",
        markeredgecolor=plotting.SURFACE,
        markeredgewidth=1.0,
    )
    axes[1].set_title(r"max($u_e$) - min($u_e$)")
    axes[1].set_xlabel(r"$\delta_e$")
    axes[1].set_ylabel("mV")
    axes[1].set_ylim(bottom=0.0)

    common.write_figure(figure, args.output_dir, "fig05_sources_of_difference")

    common.print_table(
        ["gamma", "F(lambda) (mS/cm)^2"],
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


if __name__ == "__main__":
    main()
