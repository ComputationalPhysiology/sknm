"""Figure 9: why the two continuum models differ, as two factors looked at one at a time.

The counterpart of Figure 5, and the same argument about a different pair of models. The error
the monodomain model makes is the product of two things: how far the conductivities are from
being related by a single ratio, and how much the extracellular potential varies.

- **Left.** F(lambda), the misfit the ratio is chosen to minimize, as the gap junction
  conductances are spread. It rises from zero, so the first factor grows with the spread. This
  needs no mesh at all -- the intracellular conductivity is constant within a cell and the
  extracellular one is constant everywhere, so the integral is an exact finite sum.
- **Right.** How far apart the extracellular potential gets across the sheet during a run, as
  the extracellular volume fraction grows. It falls, so the second factor shrinks. This needs
  the bidomain model, the only one of the two that solves for an extracellular potential.

Figure 5's left panel is the same picture of a different quantity. That one is equation (29),
a sum over the network's connections of conductances, in (mS/cm)^2; this one is equation (31),
an area integral of conductivities, in mS^2. The two differ by four orders of magnitude and
neither can be computed from the other.

Needs dolfinx and fenicsx-beat for the right panel; see `bidomain.py`.

    python examples/fig09_continuum_sources_of_difference.py [--full]
"""

from __future__ import annotations

import numpy as np

import bidomain
import common
import plotting

#: The paper draws the left panel at the default extracellular volume and the right panel at a
#: spread halfway to the widest, as Figure 5 does.
MISFIT_DELTA_E = 0.2
RANGE_GAMMA = 0.5


def extracellular_range(setup: bidomain.BidomainSetup) -> float:
    """The widest the extracellular potential gets across the sheet at any one moment.

    Read every step rather than every millisecond: the spread peaks as the wave front passes,
    which takes about as long as the upstroke does. The run is held open for the whole action
    potential rather than stopping when the wave reaches the far probe, so that a sheet whose
    spread peaks late is not cut off.

    Parameters
    ----------
    setup : bidomain.BidomainSetup
        The sheet to run, as a bidomain.

    Returns
    -------
    float
        ``max(u_e) - min(u_e)`` over the sheet, at the moment it is largest, in mV.
    """
    widest = 0.0

    def watch(t: float, pde: object) -> None:
        nonlocal widest
        potential = pde.u_e.x.array
        widest = max(widest, float(potential.max() - potential.min()))

    bidomain.run(setup, "bidomain", callback=watch, minimum_time=setup.t_end)
    return widest


def main() -> None:
    args = common.parse_args(__doc__.splitlines()[0])
    if not bidomain.available():
        print(bidomain.REQUIREMENT)
        return

    fractions = common.sample(common.VOLUME_FRACTIONS, args)
    cache = common.ResultCache(
        f"fig09-{'full' if args.full else 'fast'}", enabled=not args.no_cache
    )

    misfits = [
        bidomain.BidomainSetup(delta_e=MISFIT_DELTA_E, gamma=gamma).conductivity_misfit()
        for gamma in common.MISFIT_VARIATIONS
    ]
    ranges = []
    for delta_e in fractions:
        setup = bidomain.BidomainSetup(delta_e=delta_e, gamma=RANGE_GAMMA)
        label = setup.label(quantity="extracellular_range", model="bidomain")
        ranges.append(float(cache.compute(label, lambda s=setup: extracellular_range(s))))

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

    common.write_figure(figure, args.output_dir, "fig09_continuum_sources_of_difference")

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


if __name__ == "__main__":
    main()
