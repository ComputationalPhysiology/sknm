"""Shared machinery for the scripts that reproduce the paper's figures.

The scripts are `fig02` through `fig05`; this module holds what they have in common: the
paper's sweep samples, the setup that turns one point of a sweep into a running simulation,
and a cache so that replotting does not mean resolving.

**Fast by default.** Every script runs a reduced sample of its sweep unless given ``--full``,
and the reduction is *fewer points, never cheaper points*. Each point that is plotted is
computed on the paper's own 40x40 sheet at its own 0.02 ms time step, so a fast figure and a
full figure differ only in how many markers the curves carry. The two reductions that would
have been faster both corrupt the result and are not offered:

- **A smaller sheet.** Conduction velocity is not a local measurement. Halving the sheet's
  height raises it by 13% and narrowing the sheet to the 35 cells the measurement columns
  need raises it by 30%, because the wave stops spreading sideways and the far column moves
  into the boundary. Both dwarf the KNM/SKNM differences these figures exist to show.
- **A coarser time step.** A cell activates on the step its potential crosses the threshold,
  so the velocity is quantized to about one part in the number of steps the wave takes to
  cross. At 0.1 ms that is +/-2%, which is larger than the KNM/SKNM difference at 50%
  extracellular volume -- the figures would show the two models disagreeing where the paper
  shows them agreeing.

What is left is the number of points, which changes no point's value.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sknm
from sknm import Simulation, Variant, analysis, presets
from sknm.membrane import base_model_IM, from_gotranx
from sknm.units import ms, mV

#: Where the scripts live, and the two directories they write.
EXAMPLES_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXAMPLES_DIR / "figures"
CACHE_DIR = EXAMPLES_DIR / "results"

#: The four extracellular volume fractions every panel row of Figures 3 and 4 is drawn at.
EXTRACELLULAR_FRACTIONS = (0.5, 0.2, 0.1, 0.02)
#: The anisotropy factors of Figure 3: the five the reference implementation ships meshes for.
ANISOTROPY_FACTORS = tuple(sorted(presets.CELL_DIMENSIONS))
#: The gap junction variations of Figure 4, from uniform coupling to the widest spread.
GAP_JUNCTION_VARIATIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
#: The extracellular volume fractions of Figure 5's right panel.
VOLUME_FRACTIONS = (0.02, 0.1, 0.2, 0.3, 0.4, 0.5)
#: The gap junction variations of Figure 5's left panel. Dense, because it costs no simulation.
MISFIT_VARIATIONS = tuple(np.round(np.linspace(0.0, 1.0, 21), 3))

#: The paper's sheet, time step, run length and threshold.
NX = 40
NY = 40
DT = 0.02
T_END = 50.0
THRESHOLD = -20.0

#: Seed for the gap junction draws. The paper reuses one set of draws across every value of
#: gamma and every variant, which is what makes its curves comparable point for point; one
#: seed, fixed here, does the same. The reference's own draws are in `random_picks/` and can be
#: passed to `presets.vary_conductances` instead, in the order that directory writes them.
DRAW_SEED = 0

#: Bumped when the layout of a cache file changes, so that older ones are discarded.
CACHE_FORMAT = 1


def parse_args(description: str) -> argparse.Namespace:
    """Read the command line every figure script accepts.

    Parameters
    ----------
    description : str
        What the script reproduces, shown by ``--help``.

    Returns
    -------
    argparse.Namespace
        With `full`, `no_cache` and `output_dir`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--full",
        action="store_true",
        help="sweep the paper's full sample rather than every other point of it",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="resolve every point, ignoring and overwriting any cached results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help=f"where to write the figure, by default {FIGURE_DIR}",
    )
    return parser.parse_args()


def sample(points: Sequence[float], args: argparse.Namespace) -> tuple[float, ...]:
    """Choose how much of a sweep to run, from ``--full``.

    Parameters
    ----------
    points : sequence of float
        The full sample, in order.
    args : argparse.Namespace
        From `parse_args`.

    Returns
    -------
    tuple of float
        `points` itself under ``--full``, otherwise every other one of them with both ends
        kept. A reduced sample is always a subset, so a fast figure plots a subset of the
        markers a full figure does, at the same values.
    """
    if args.full:
        return tuple(points)
    fewer = tuple(points[::2])
    if len(points) and points[-1] not in fewer:
        fewer += (points[-1],)
    return fewer


def panel_title(delta_e: float) -> str:
    """The paper's own panel title for an extracellular volume fraction, such as ``"2% E"``."""
    return f"{delta_e:.0%} E"


@dataclasses.dataclass(frozen=True)
class Setup:
    """One point of a sweep: everything that decides what a simulation computes.

    Every field is part of the cache key, so adding one here invalidates cached results
    rather than silently serving numbers computed without it.

    Parameters
    ----------
    alpha : float, optional
        Anisotropy factor, by default 1.0.
    delta_e : float, optional
        Extracellular volume fraction, by default the paper's 0.2.
    gamma : float, optional
        Gap junction variation, by default 0.0.
    nx, ny : int, optional
        Shape of the sheet, by default the paper's 40 by 40.
    dt : float, optional
        Time step in ms, by default the paper's 0.02.
    seed : int, optional
        Seed for the gap junction draws, by default `DRAW_SEED`.
    """

    alpha: float = 1.0
    delta_e: float = presets.DELTA_E
    gamma: float = 0.0
    nx: int = NX
    ny: int = NY
    dt: float = DT
    seed: int = DRAW_SEED

    def network(self) -> sknm.CellNetwork:
        """Build the sheet, with its gap junction conductances spread by `gamma`.

        Returns
        -------
        sknm.CellNetwork
            The paper's hiPSC-CM sheet.
        """
        network = presets.hipsc_sheet(self.nx, self.ny, alpha=self.alpha, delta_e=self.delta_e)
        draws = np.random.default_rng(self.seed).random(network.n_connections)
        # Applied even at gamma = 0, where it returns the conductances unchanged, so that the
        # reference point of a sweep goes through the same code as the rest of it.
        return network.with_conductances(presets.vary_conductances(network, self.gamma, draws))

    def simulation(self, variant: Variant | str) -> Simulation:
        """Build the simulation, stimulated along the left edge and ready to run.

        Parameters
        ----------
        variant : Variant or str
            Which of the three models to solve.

        Returns
        -------
        sknm.Simulation
            At time zero.
        """
        simulation = Simulation(
            self.network(),
            from_gotranx(base_model_IM),
            variant=variant,
            dt=self.dt * ms,
        )
        simulation.set_parameter(
            "stim_amplitude", presets.hipsc_stimulus_amplitude(self.nx, self.ny)
        )
        return simulation

    def conduction_path(self) -> analysis.ConductionPath:
        """The two cells a conduction velocity is measured between, at this anisotropy factor.

        Returns
        -------
        sknm.analysis.ConductionPath
            Matching this setup's cell size, which is what makes the distance right.
        """
        return presets.hipsc_conduction_path(self.nx, self.ny, alpha=self.alpha)

    def label(self, **extra: Any) -> str:
        """A cache key naming every parameter that decides the result.

        Parameters
        ----------
        **extra
            Anything beyond the setup that the result depends on, such as the variant or which
            quantity is being measured.

        Returns
        -------
        str
            Every field and every extra, sorted by name. Floats are written so that they read
            back exactly, so two labels match only when the numbers do.
        """
        fields = dataclasses.asdict(self) | dict(extra)
        return " ".join(f"{name}={_format(fields[name])}" for name in sorted(fields))


def measure_conduction_velocity(setup: Setup, variant: Variant | str) -> float:
    """Run one simulation and read the conduction velocity off it.

    The run stops as soon as the wave reaches the far end of the conduction path, which on
    the paper's setup is at about 37 ms of the 50 ms a full action potential takes.

    Parameters
    ----------
    setup : Setup
        The point of the sweep to compute.
    variant : Variant or str
        Which of the three models to solve.

    Returns
    -------
    float
        Conduction velocity in cm/s.

    Raises
    ------
    ValueError
        If the wave never reached the far cell, so there is no velocity to report.
    """
    simulation = setup.simulation(variant)
    path = setup.conduction_path()
    recorder = analysis.ActivationRecorder(
        simulation, threshold=THRESHOLD * mV, stop_when_activated=path.end
    )
    simulation.run(T_END * ms, record=(), callback=recorder)
    return float(analysis.conduction_velocity(recorder, path).m_as("cm / s"))


class ResultCache:
    """Measured results held between runs, so that restyling a figure costs no simulation.

    One file per script and sample, holding a label for every point and the value measured
    there. It is written after each point, so an interrupted sweep resumes where it stopped.

    The key is the *parameters* of a run, which is everything except the code that ran them:
    a cache cannot tell that the library has changed underneath it. Delete `CACHE_DIR`, or
    pass ``--no-cache``, after changing anything `sknm` computes.

    Parameters
    ----------
    name : str
        Stem of the cache file, distinguishing both the script and its sample.
    enabled : bool, optional
        Whether to read and write at all, by default True. `False` recomputes everything and
        stores nothing.
    directory : pathlib.Path, optional
        Where the file lives, by default `CACHE_DIR`.
    """

    def __init__(self, name: str, *, enabled: bool = True, directory: Path = CACHE_DIR) -> None:
        self.path = directory / f"{name}.npz"
        self.enabled = enabled
        self._spec = json.dumps(
            {
                "format": CACHE_FORMAT,
                "sknm": sknm.__version__,
                "t_end": T_END,
                "threshold": THRESHOLD,
            },
            sort_keys=True,
        )
        self._values: dict[str, npt.NDArray[np.float64]] = {}
        if enabled and self.path.exists():
            self._load()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path.name!r}, holding={len(self._values)})"

    def _load(self) -> None:
        """Read the file, discarding it whole if it was written for other invariants."""
        with np.load(self.path) as stored:
            if str(stored["spec"].item()) != self._spec:
                return
            self._values = {
                str(label): value
                for label, value in zip(stored["labels"], stored["values"], strict=True)
            }

    def _save(self) -> None:
        """Write every value held, replacing the file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            spec=np.array(self._spec),
            labels=np.array(list(self._values), dtype=np.str_),
            values=np.array(list(self._values.values()), dtype=np.float64),
        )

    def compute(self, label: str, produce: Callable[[], npt.ArrayLike]) -> npt.NDArray[np.float64]:
        """Return the value stored under a label, measuring it first if it is not there.

        Parameters
        ----------
        label : str
            From `Setup.label`, naming every parameter the value depends on.
        produce : callable
            Called with no arguments to measure the value, when it is not cached.

        Returns
        -------
        numpy.ndarray
            The value, as a float array. Scalars come back zero-dimensional.

        Raises
        ------
        ValueError
            If the value has a different shape from those already held, which a single file
            cannot store.
        """
        if label in self._values:
            return self._values[label]
        value = np.asarray(produce(), dtype=np.float64)
        for held_label, held in self._values.items():
            if held.shape != value.shape:
                raise ValueError(
                    f"{self.path.name} holds values of shape {held.shape} under "
                    f"{held_label!r}, so it cannot also hold one of shape {value.shape} "
                    f"under {label!r}; give this sweep a cache of its own"
                )
        self._values[label] = value
        if self.enabled:
            self._save()
        return value


def write_figure(figure: Any, output_dir: Path, stem: str) -> Path:
    """Save a figure and say where it went.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to write.
    output_dir : pathlib.Path
        Directory to write into, created if it does not exist.
    stem : str
        File name without its suffix.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.png"
    figure.savefig(path, dpi=200)
    print(f"wrote {path}")
    return path


def print_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Print the numbers a figure plots, so they can be read and compared without an image.

    Parameters
    ----------
    header : sequence of str
        Column names.
    rows : sequence of sequence
        One row per line. Numbers are shown to four decimal places.
    """
    widths = [len(name) for name in header]
    formatted = [[_cell(value) for value in row] for row in rows]
    for row in formatted:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    lines = [header, ["-" * width for width in widths], *formatted]
    for line in lines:
        print("  ".join(cell.rjust(width) for cell, width in zip(line, widths, strict=True)))


def _cell(value: Any) -> str:
    """One table cell, with numbers to a fixed number of places so columns line up."""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _format(value: Any) -> str:
    """One label field, written so that reading it back gives the same number."""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Variant):
        return value.value
    return str(value)
