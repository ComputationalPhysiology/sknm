"""Figure 2: a travelling wave across 40x40 hiPSC-CMs, solved as KNM and as SKNM.

Snapshots of the membrane potential at three points in time, one row per model. This is the
paper's default set-up, where every connection has the same conductances, so a single ratio
relates the extracellular and intracellular conductance of all of them and the assumption
SKNM is derived from holds exactly. The two rows should be indistinguishable, and the figure
prints how far apart they actually are.

Cheap enough that ``SKNM_EXAMPLES_FULL`` changes nothing: there is no sweep to reduce, only
two runs.

    python examples/fig02_travelling_wave.py
"""

from __future__ import annotations

import numpy as np

import common
import plotting
from sknm import Variant
from sknm.units import ms

#: The paper's three snapshot times, and the interval that lands a sample on each of them.
SNAPSHOT_TIMES = (25.0, 30.0, 35.0)
SNAPSHOT_INTERVAL = 5.0

VARIANTS = (Variant.KNM, Variant.SKNM)


def snapshots(setup: common.Setup, variant: Variant) -> np.ndarray:
    """Record the membrane potential at each of the paper's three snapshot times.

    Parameters
    ----------
    setup : common.Setup
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


def main() -> None:
    options = common.options()
    setup = common.Setup()
    cache = common.ResultCache("fig02", enabled=not options.no_cache)

    recorded = {}
    for variant in VARIANTS:
        label = setup.label(variant=variant, quantity="snapshots")
        recorded[variant] = cache.compute(label, lambda v=variant: snapshots(setup, v))

    potentials = np.stack(list(recorded.values()))
    low, high = float(potentials.min()), float(potentials.max())

    plotting.use_house_style()
    figure, grid = plotting.panel_grid(len(VARIANTS), len(SNAPSHOT_TIMES))
    for row, variant in enumerate(VARIANTS):
        for column, time in enumerate(SNAPSHOT_TIMES):
            axis = grid[row][column]
            sheet = recorded[variant][column].reshape(setup.ny, setup.nx)
            image = plotting.show_sheet(axis, sheet, low=low, high=high)
            if row == 0:
                axis.set_title(f"t = {time:g} ms")
            if column == 0:
                axis.set_ylabel(plotting.SERIES_LABEL[variant], fontsize=11, color=plotting.INK)
    plotting.colour_scale(figure, image, grid, "membrane potential (mV)")

    common.write_figure(figure, options.output_dir, "fig02_travelling_wave")

    difference = np.abs(recorded[Variant.KNM] - recorded[Variant.SKNM])
    common.print_table(
        ["t (ms)", "KNM min", "KNM max", "SKNM min", "SKNM max", "max |KNM - SKNM|"],
        [
            [
                f"{time:g}",
                float(recorded[Variant.KNM][index].min()),
                float(recorded[Variant.KNM][index].max()),
                float(recorded[Variant.SKNM][index].min()),
                float(recorded[Variant.SKNM][index].max()),
                f"{difference[index].max():.2e}",
            ]
            for index, time in enumerate(SNAPSHOT_TIMES)
        ],
    )


if __name__ == "__main__":
    main()
