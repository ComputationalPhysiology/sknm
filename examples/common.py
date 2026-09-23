"""Shared machinery for the scripts that reproduce the paper's figures.

The scripts are `fig02` through `fig05`; this module holds what they have in common: the
paper's sweep samples, the setup that turns one point of a sweep into a running simulation,
and a cache so that replotting does not mean resolving.

**Fast by default.** Every script runs a reduced sample of its sweep unless
``SKNM_EXAMPLES_FULL`` is set, and the reduction is *fewer points, never cheaper points*.
Each point that is plotted is computed on the paper's own 40x40 sheet at its own 0.02 ms time
step, so a fast figure and a full figure differ only in how many markers the curves carry. The
two reductions that would have been faster both corrupt the result and are not offered:

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

import dataclasses
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import sknm
from sknm import Simulation, Variant, analysis, presets
from sknm.membrane import base_model_IM, from_gotranx
from sknm.units import ms, mV

#: Where the scripts live, the two directories they write, and the one they read.
EXAMPLES_DIR = Path(__file__).resolve().parent
FIGURE_DIR = EXAMPLES_DIR / "figures"
CACHE_DIR = EXAMPLES_DIR / "results"
DATA_DIR = EXAMPLES_DIR / "data"

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

#: The paper's cardiac sheet, time step, run length and threshold.
NX = 40
NY = 40
DT = 0.02
T_END = 50.0
THRESHOLD = presets.HIPSC_THRESHOLD.m_as(mV)

#: The paper's beta cell sheet and the twenty-fold longer run its slower wave needs.
BETA_NX = 15
BETA_NY = 15
BETA_T_END = 1000.0
BETA_THRESHOLD = presets.BETA_THRESHOLD.m_as(mV)

#: Seed for the gap junction draws. The paper reuses one set of draws across every value of
#: gamma and every variant, which is what makes its curves comparable point for point; one
#: seed, fixed here, does the same. The reference's own draws are in `random_picks/` and can be
#: passed to `presets.vary_conductances` instead, in the order that directory writes them.
DRAW_SEED = 0

#: Bumped when the layout of a cache file changes, so that older ones are discarded.
CACHE_FORMAT = 2


#: The environment variables a run is configured with, one per field of `Options`.
#:
#: A command line cannot reach these scripts. Each is also run as a notebook, where the
#: process belongs to the Jupyter kernel and ``sys.argv`` describes the kernel's own
#: invocation rather than the script's; the environment is what both ways of running share.
OPTION_VARIABLES = (
    "SKNM_EXAMPLES_FULL",
    "SKNM_EXAMPLES_NO_CACHE",
    "SKNM_EXAMPLES_OUTPUT_DIR",
)

#: What a flag variable may be set to. An empty value counts as false, because exporting a
#: variable to the empty string is how a shell spells "unset".
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


@dataclasses.dataclass(frozen=True)
class Options:
    """How much of a sweep to run, whether to reuse cached results, and where to write.

    Attributes
    ----------
    full : bool
        Sweep the paper's full sample rather than every other point of it.
    no_cache : bool
        Resolve every point, ignoring and overwriting any cached result.
    output_dir : pathlib.Path
        Where the figure is written.
    """

    full: bool = False
    no_cache: bool = False
    output_dir: Path = FIGURE_DIR


def _flag(name: str) -> bool:
    """Read one environment variable as a boolean.

    Parameters
    ----------
    name : str
        The variable to read. An unset variable is false.

    Returns
    -------
    bool
        What the variable is set to.

    Raises
    ------
    ValueError
        If the variable is set to something that is neither true nor false. Reading an
        unrecognized value as false would run a different sweep than the one asked for and
        say nothing about it.
    """
    value = os.environ.get(name, "")
    lowered = value.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} is set to {value!r}, which is neither true nor false; use one of "
        f"{', '.join(sorted(TRUE_VALUES))} or {', '.join(sorted(FALSE_VALUES - {''}))}"
    )


def options() -> Options:
    """Read how this run was asked to behave from the environment.

    Returns
    -------
    Options
        Set from `OPTION_VARIABLES`, defaulting to a fast, cached run that writes into
        `FIGURE_DIR`.

    Raises
    ------
    ValueError
        If a flag variable is set to something that is neither true nor false.
    """
    directory = os.environ.get("SKNM_EXAMPLES_OUTPUT_DIR", "").strip()
    return Options(
        full=_flag("SKNM_EXAMPLES_FULL"),
        no_cache=_flag("SKNM_EXAMPLES_NO_CACHE"),
        output_dir=Path(directory) if directory else FIGURE_DIR,
    )


def sample(points: Sequence[float], options: Options) -> tuple[float, ...]:
    """Choose how much of a sweep to run, from `Options.full`.

    Parameters
    ----------
    points : sequence of float
        The full sample, in order.
    options : Options
        From `options`.

    Returns
    -------
    tuple of float
        `points` itself when `options` is full, otherwise every other one of them with both
        ends kept. A reduced sample is always a subset, so a fast figure plots a subset of
        the markers a full figure does, at the same values.
    """
    if options.full:
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
    t_end : float, optional
        How long to run for, in ms, by default 50.
    threshold : float, optional
        Membrane potential at which a cell counts as activated, in mV, by default -20.
    seed : int, optional
        Seed for the gap junction draws, by default `DRAW_SEED`.
    """

    alpha: float = 1.0
    delta_e: float = presets.DELTA_E
    gamma: float = 0.0
    nx: int = NX
    ny: int = NY
    dt: float = DT
    t_end: float = T_END
    threshold: float = THRESHOLD
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
        return cache_label(self, **extra)


def beta_draws() -> npt.NDArray[np.float64]:
    """The paper's own gap junction draws for its 15x15 sheet, one per connection.

    Committed under `data/`, where the equivalent cardiac draws are not, because the beta
    figures need them and the cardiac ones do not. On the 40x40 sheet the draws move a
    conduction velocity by about 2% and any seed reproduces the published figure; on this one
    there are 420 connections and the conduction path is 8 cells long, so at ``gamma = 1``
    they move it by 20% and every seed tried fell below the axis of the published Figure S1.

    No number the test suite asserts depends on them: every validation target is at
    ``gamma = 0``, where `sknm.presets.vary_conductances` returns the conductances unchanged
    whatever the draws are.

    Returns
    -------
    numpy.ndarray
        Shape ``(420,)``, in `sknm.sheet`'s connection order.
    """
    return np.loadtxt(DATA_DIR / "gj_scale_15x15.txt")


@dataclasses.dataclass(frozen=True)
class BetaSetup:
    """One point of a beta cell sweep, the sibling of `Setup`.

    A separate dataclass rather than a cell type on `Setup`, because a setup **is** the cache
    key: a shared class would carry an anisotropy factor into every beta label, where a beta
    cell has none. What the two share is everything around them -- `ResultCache`, `sample`,
    `measure_conduction_velocity` and the plotting -- which is what makes two small classes
    cheaper than one general one.

    Parameters
    ----------
    delta_e : float, optional
        Extracellular volume fraction, by default the paper's 0.5.
    gamma : float, optional
        Gap junction variation, by default 0.0.
    nx, ny : int, optional
        Shape of the sheet, by default the paper's 15 by 15.
    dt : float, optional
        Time step in ms, by default the paper's 0.02.
    t_end : float, optional
        How long to run for, in ms, by default 1000 -- twenty times the cardiac run, because a
        beta wave is 150 times slower.
    threshold : float, optional
        Membrane potential at which a cell counts as activated, in mV, by default -50. A beta
        action potential peaks at about -19.5 mV, so the cardiac -20 would activate nothing.
    seed : int or None, optional
        Where the gap junction draws come from, by default `None`, meaning the reference
        implementation's own draws for its 15 by 15 sheet -- which is what the paper's figures
        need and what `beta_draws` returns. An integer seeds an RNG instead, which is the only
        option on a sheet of another shape, since the committed draws describe one sheet.
    """

    delta_e: float = presets.BETA_DELTA_E
    gamma: float = 0.0
    nx: int = BETA_NX
    ny: int = BETA_NY
    dt: float = DT
    t_end: float = BETA_T_END
    threshold: float = BETA_THRESHOLD
    seed: int | None = None

    def network(self) -> sknm.CellNetwork:
        """Build the sheet, with its gap junction conductances spread by `gamma`.

        Returns
        -------
        sknm.CellNetwork
            The paper's beta cell sheet.
        """
        network = presets.beta_sheet(self.nx, self.ny, delta_e=self.delta_e)
        if self.seed is None:
            draws = beta_draws()
            if draws.shape != (network.n_connections,):
                raise ValueError(
                    f"the committed draws are for a 15 by 15 sheet with {draws.size} "
                    f"connections, but this one is {self.nx} by {self.ny} with "
                    f"{network.n_connections}. Give the setup a `seed` to draw its own."
                )
        else:
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
            presets.beta_membrane_model(),
            variant=variant,
            dt=self.dt * ms,
        )
        simulation.set_parameter("gkatpbar", presets.beta_stimulus_conductance(self.nx, self.ny))
        return simulation

    def conduction_path(self) -> analysis.ConductionPath:
        """The two cells a conduction velocity is measured between.

        Returns
        -------
        sknm.analysis.ConductionPath
            Columns 4 and 12 of the measurement row, eight cells apart.
        """
        return presets.beta_conduction_path(self.nx, self.ny)

    def label(self, **extra: Any) -> str:
        """A cache key naming every parameter that decides the result.

        Parameters
        ----------
        **extra
            Anything beyond the setup that the result depends on, such as the variant.

        Returns
        -------
        str
            Every field and every extra, sorted by name.
        """
        return cache_label(self, **extra)


def measure_conduction_velocity(setup: Setup | BetaSetup, variant: Variant | str) -> float:
    """Run one simulation and read the conduction velocity off it.

    The run stops as soon as the wave reaches the far end of the conduction path, which on the
    paper's cardiac setup is at about 37 ms of the 50 ms a full action potential takes.

    The threshold and the run length come from the setup rather than from a constant here,
    because they are exactly the two numbers that differ between the two cell types: a beta
    cell activates at -50 mV and takes twenty times as long, and the cardiac threshold applied
    to it would leave the measurement cell unactivated.

    Parameters
    ----------
    setup : Setup or BetaSetup
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
        simulation, threshold=setup.threshold * mV, stop_when_activated=path.end
    )
    simulation.run(setup.t_end * ms, record=(), callback=recorder)
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
        # The run length and the threshold used to live here. They moved into the label, where
        # `Setup` and `BetaSetup` carry them as fields, because the two setups do not share
        # them: pinned here, one script's cache would discard the other's on every load.
        self._spec = json.dumps({"format": CACHE_FORMAT, "sknm": sknm.__version__}, sort_keys=True)
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


def cache_label(setup: Any, **extra: Any) -> str:
    """The cache key of any setup: every field it declares, plus anything else asked for.

    Built from `dataclasses.asdict`, so a field added to a setup enters its key rather than
    having to be remembered separately.
    """
    fields = dataclasses.asdict(setup) | dict(extra)
    return " ".join(f"{name}={_format(fields[name])}" for name in sorted(fields))


def _format(value: Any) -> str:
    """One label field, written so that reading it back gives the same number."""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Variant):
        return value.value
    return str(value)
