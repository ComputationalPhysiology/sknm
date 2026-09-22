"""Figure 6: a travelling wave across 40x40 hiPSC-CMs, solved as bidomain and monodomain.

The continuum counterpart of Figure 2, and the same claim made about a different pair of
models: snapshots of the membrane potential at three points in time, one row per model, which
should be indistinguishable.

Here they are more than indistinguishable. The paper's default sheet has uniform gap junction
conductances and square cells, so the intracellular conductivity is both spatially uniform and
isotropic, one ratio relates it to the extracellular conductivity everywhere, and the bidomain
system reduces to the monodomain one exactly rather than approximately. The figure prints how
far apart the two rows are, and the answer is the precision of the linear solves.

Needs dolfinx and fenicsx-beat; see `bidomain.py`.

    python examples/fig06_continuum_travelling_wave.py
"""

from __future__ import annotations

import numpy as np

import bidomain
import common
import plotting

#: The paper's three snapshot times, shared with Figure 2.
SNAPSHOT_TIMES = (25.0, 30.0, 35.0)

MODELS = ("bidomain", "monodomain")
MODEL_LABEL = {"bidomain": "BD", "monodomain": "MD"}


def snapshots(setup: bidomain.BidomainSetup, model: str) -> np.ndarray:
    """Record the membrane potential across the sheet at each of the snapshot times.

    Parameters
    ----------
    setup : bidomain.BidomainSetup
        The sheet to run.
    model : {"bidomain", "monodomain"}
        Which model to solve.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(SNAPSHOT_TIMES), ny + 1, nx + 1)``, in mV, laid out as the grid.
    """
    remaining = list(SNAPSHOT_TIMES)
    frames = []

    def capture(t: float, pde: object) -> None:
        if remaining and t >= remaining[0] - 0.5 * setup.dt:
            remaining.pop(0)
            frames.append(bidomain.to_grid(setup, pde.V, pde.v.x.array.copy()))

    bidomain.run(setup, model, callback=capture, minimum_time=max(SNAPSHOT_TIMES))
    if remaining:
        raise ValueError(f"the run ended before {remaining} ms")
    return np.stack(frames)


def main() -> None:
    args = common.parse_args(__doc__.splitlines()[0])
    if not bidomain.available():
        print(bidomain.REQUIREMENT)
        return

    setup = bidomain.BidomainSetup()
    cache = common.ResultCache("fig06", enabled=not args.no_cache)

    recorded = {}
    for model in MODELS:
        label = setup.label(model=model, quantity="snapshots")
        recorded[model] = cache.compute(label, lambda m=model: snapshots(setup, m))

    potentials = np.stack(list(recorded.values()))
    low, high = float(potentials.min()), float(potentials.max())

    plotting.use_house_style()
    figure, grid = plotting.panel_grid(len(MODELS), len(SNAPSHOT_TIMES))
    for row, model in enumerate(MODELS):
        for column, time in enumerate(SNAPSHOT_TIMES):
            axis = grid[row][column]
            image = plotting.show_sheet(axis, recorded[model][column], low=low, high=high)
            if row == 0:
                axis.set_title(f"t = {time:g} ms")
            if column == 0:
                axis.set_ylabel(MODEL_LABEL[model], fontsize=11, color=plotting.INK)
    plotting.colour_scale(figure, image, grid, "membrane potential (mV)")

    common.write_figure(figure, args.output_dir, "fig06_continuum_travelling_wave")

    difference = np.abs(recorded["bidomain"] - recorded["monodomain"])
    common.print_table(
        ["t (ms)", "BD min", "BD max", "MD min", "MD max", "max |BD - MD|"],
        [
            [
                f"{time:g}",
                float(recorded["bidomain"][index].min()),
                float(recorded["bidomain"][index].max()),
                float(recorded["monodomain"][index].min()),
                float(recorded["monodomain"][index].max()),
                f"{difference[index].max():.2e}",
            ]
            for index, time in enumerate(SNAPSHOT_TIMES)
        ],
    )


if __name__ == "__main__":
    main()
