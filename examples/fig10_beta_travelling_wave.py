"""Figure 10: a travelling wave across 15x15 pancreatic beta cells, as KNM and as SKNM.

The beta counterpart of Figure 2, and the same claim: with every connection carrying the same
conductances, a single ratio relates the extracellular and intracellular conductance of all of
them, so SKNM's assumption holds exactly and the two rows should be indistinguishable.

It holds far more comfortably here than it does for cardiac cells. The gap junction resistance
between beta cells is a thousand times the cardiac one, so it swamps both conductivities and
the conductance ratio comes out around 65,000 rather than 40 -- which is why the wave is 150
times slower, why the snapshots are in seconds rather than milliseconds, and why even
SKNM(u_e=0) is indistinguishable from KNM here. The figure prints the ratio and the measured
difference.

Cheap enough that ``SKNM_EXAMPLES_FULL`` changes nothing: there is no sweep to reduce, only
two runs.

    python examples/fig10_beta_travelling_wave.py
"""

from __future__ import annotations

import numpy as np

import common
import plotting
from sknm import Variant
from sknm.units import ms

#: The paper's four snapshot times in ms, and the interval that lands a sample on each of them.
SNAPSHOT_TIMES = (100.0, 300.0, 500.0, 700.0)
SNAPSHOT_INTERVAL = 100.0

VARIANTS = (Variant.KNM, Variant.SKNM)


def snapshots(setup: common.BetaSetup, variant: Variant) -> np.ndarray:
    """Record the membrane potential at each of the paper's four snapshot times.

    Parameters
    ----------
    setup : common.BetaSetup
        The sheet to run.
    variant : Variant
        Which model to solve.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(SNAPSHOT_TIMES), n_cells)``, in mV.
    """
    simulation = setup.simulation(variant)
    result = simulation.run(
        max(SNAPSHOT_TIMES) * ms, record=("v",), record_every=SNAPSHOT_INTERVAL * ms
    )
    wanted = [int(np.argmin(np.abs(result.t - time))) for time in SNAPSHOT_TIMES]
    return np.ascontiguousarray(result.v[:, wanted].T)


def draw(setup, recorded, stem, options):
    """Write one snapshot figure, one row per model, sharing a single colour scale."""
    potentials = np.stack(list(recorded.values()))
    low, high = float(potentials.min()), float(potentials.max())

    plotting.use_house_style()
    # Sized for four columns of square panels rather than `panel_grid`'s default three, which
    # would leave the two rows floating far apart: an image axis keeps its aspect, so the
    # leftover figure height turns into a gap rather than into larger panels.
    figure, grid = plotting.panel_grid(len(VARIANTS), len(SNAPSHOT_TIMES), width=10.0, height=4.9)
    image = None
    for row, variant in enumerate(VARIANTS):
        for column, time in enumerate(SNAPSHOT_TIMES):
            axis = grid[row][column]
            sheet = recorded[variant][column].reshape(setup.ny, setup.nx)
            image = plotting.show_sheet(axis, sheet, low=low, high=high)
            if row == 0:
                axis.set_title(f"t = {time / 1000:g} s")
            if column == 0:
                axis.set_ylabel(plotting.SERIES_LABEL[variant], fontsize=11, color=plotting.INK)
    plotting.colour_scale(figure, image, grid, "membrane potential (mV)")

    return common.write_figure(figure, options.output_dir, stem)


def report(setup, recorded):
    """Print the numbers the figure plots, and the conductance ratio that explains them."""
    difference = np.abs(recorded[Variant.KNM] - recorded[Variant.SKNM])
    common.print_table(
        ["t (s)", "KNM min", "KNM max", "SKNM min", "SKNM max", "max |KNM - SKNM|"],
        [
            [
                f"{time / 1000:g}",
                float(recorded[Variant.KNM][index].min()),
                float(recorded[Variant.KNM][index].max()),
                float(recorded[Variant.SKNM][index].min()),
                float(recorded[Variant.SKNM][index].max()),
                f"{difference[index].max():.2e}",
            ]
            for index, time in enumerate(SNAPSHOT_TIMES)
        ],
    )
    network = setup.network()
    print(
        f"\nconductance ratio lambda = {network.lam:.0f}, so SKNM's lambda / (1 + lambda) is "
        f"{network.lam / (1 + network.lam):.6f}"
    )


def main() -> None:
    options = common.options()
    setup = common.BetaSetup()
    cache = common.ResultCache("fig10", enabled=not options.no_cache)

    recorded = {}
    for variant in VARIANTS:
        label = setup.label(variant=variant, quantity="snapshots")
        recorded[variant] = cache.compute(label, lambda v=variant: snapshots(setup, v))

    draw(setup, recorded, "fig10_beta_travelling_wave", options)
    report(setup, recorded)


if __name__ == "__main__":
    main()
