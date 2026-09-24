"""The look of the figures: the paper's layout, drawn in a palette chosen for legibility.

The paper's figures encode the model in two channels at once, a colour and a line style. That
encoding is reproduced here, because it keeps the figures readable when the two curves lie on
top of each other and what keeps them readable in greyscale and to a colour-blind reader.
The plotting package's default appearance is not reproduced: the series colours are a set whose
pairwise separation has been checked under simulated colour vision deficiency, the grid is a
recessive hairline rather than a dashed rule, and the axis furniture is muted so the data is the
darkest thing on the page.

Every figure is drawn on a light surface. These are files written to disk for reading and for
pasting beside the published originals, not a page that follows a reader's theme.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from sknm import Variant

#: Chart surface and ink. The data is the darkest thing drawn; the furniture recedes.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

#: One colour per model, in an order whose pairs stay separable under colour vision
#: deficiency, and one line style per model, so the identity never rests on colour alone.
#: The line styles are the paper's own: KNM solid, SKNM dotted.
#:
#: The two continuum models are keyed by name and share the network models' two colours and
#: styles. Figures 6 to 9 are the counterparts of Figures 2 to 5 one for one: the bidomain model
#: is where KNM's assumption is not yet made, and the monodomain model is where it is. Drawing
#: the counterpart pair in the counterpart colours lets a reader lay the two halves of the paper
#: side by side.
SERIES_COLOUR = {
    Variant.KNM: "#2a78d6",
    Variant.SKNM: "#eb6834",
    Variant.SKNM_UE0: "#1baf7a",
    "bidomain": "#2a78d6",
    "monodomain": "#eb6834",
}
SERIES_STYLE = {
    Variant.KNM: "-",
    Variant.SKNM: ":",
    Variant.SKNM_UE0: "--",
    "bidomain": "-",
    "monodomain": ":",
}
SERIES_MARKER = {
    Variant.KNM: "o",
    Variant.SKNM: "s",
    Variant.SKNM_UE0: "^",
    "bidomain": "o",
    "monodomain": "s",
}
SERIES_LABEL = {
    Variant.KNM: "KNM",
    Variant.SKNM: "SKNM",
    Variant.SKNM_UE0: r"SKNM($u_e$=0)",
    "bidomain": "BD",
    "monodomain": "MD",
}

#: A single hue, light to dark, for the membrane potential snapshots. Magnitude gets a
#: sequential ramp and a scale legend; never a rainbow, whose bands invent boundaries the
#: data does not have.
POTENTIAL_COLOURS = LinearSegmentedColormap.from_list(
    "sknm_potential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

_RC = {
    "figure.facecolor": SURFACE,
    "figure.dpi": 110,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": SECONDARY_INK,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    # A solid hairline. Dashing a gridline adds noise and reads as a series in its own right.
    "grid.color": GRIDLINE,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
    "text.color": INK,
    "font.size": 10,
}


def use_house_style() -> None:
    """Apply the style to every figure made afterwards."""
    mpl.rcParams.update(_RC)


def small_multiples(
    n_panels: int, *, width: float = 11.0, height: float = 3.1, share_y: bool = False
) -> tuple[Any, Any]:
    """A row of panels sharing one x axis meaning, as the paper's Figures 3 and 4 are drawn.

    Parameters
    ----------
    n_panels : int
        How many panels.
    width, height : float, optional
        Figure size in inches, by default 11.0 by 3.1.
    share_y : bool, optional
        Whether every panel is drawn on one y scale, by default `False`.

        Off by default because the paper's Figures 3, 4 and S2 give their 2% panel a range of
        its own, where the models separate far enough to need it. Turn it on where the point of
        the figure is that the panels agree: panels on scales of their own cannot be compared by
        eye at all, so four identical curves would be drawn four different sizes and read as
        four different results.

    Returns
    -------
    tuple
        The figure and a flat array of its axes.
    """
    figure, axes = plt.subplots(
        1, n_panels, figsize=(width, height), layout="constrained", sharey=share_y
    )
    return figure, axes.ravel()


def panel_grid(
    n_rows: int, n_columns: int, *, width: float = 9.0, height: float = 5.6
) -> tuple[Any, Any]:
    """A grid of panels, as the paper's Figure 2 lays its snapshots out.

    Parameters
    ----------
    n_rows, n_columns : int
        Shape of the grid.
    width, height : float, optional
        Figure size in inches, by default 9.0 by 5.6.

    Returns
    -------
    tuple
        The figure and a ``(n_rows, n_columns)`` array of its axes.
    """
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(width, height), squeeze=False)
    return figure, axes


def show_sheet(axis: Any, values: Any, *, low: float, high: float) -> Any:
    """Draw one sheet of cells as an image, one pixel per cell.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Where to draw.
    values : numpy.ndarray
        Shape ``(ny, nx)``, the quantity at each cell.
    low, high : float
        Ends of the colour scale, shared across every panel of a figure so that the panels
        can be compared with each other.

    Returns
    -------
    matplotlib.image.AxesImage
        For attaching a colour bar to.
    """
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return axis.imshow(
        values,
        cmap=POTENTIAL_COLOURS,
        vmin=low,
        vmax=high,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )


def colour_scale(figure: Any, image: Any, axes: Any, label: str) -> None:
    """Attach the one scale legend a sequential encoding needs.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    image : matplotlib.image.AxesImage
        Any of the images the scale covers; they share it.
    axes : array of matplotlib.axes.Axes
        The panels to steal room from.
    label : str
        What the scale measures, with its unit.
    """
    bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label(label, color=SECONDARY_INK)
    bar.outline.set_visible(False)


def plot_series(
    axis: Any,
    x: Sequence[float],
    y: Sequence[float],
    variant: Variant | str,
    **kwargs: Any,
) -> None:
    """Draw one model's curve, in its colour, its line style and its marker.

    The markers carry a ring in the surface colour, so that where two models agree, which on
    most of these panels is everywhere, the marker on top does not swallow the one beneath it.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Where to draw.
    x, y : sequence of float
        The points, in order.
    variant : Variant or str
        Which model this is, choosing every part of the appearance. One of the three network
        variants, or ``"bidomain"`` or ``"monodomain"``.
    **kwargs
        Passed to `matplotlib.axes.Axes.plot`.
    """
    axis.plot(
        x,
        y,
        color=SERIES_COLOUR[variant],
        linestyle=SERIES_STYLE[variant],
        marker=SERIES_MARKER[variant],
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label=SERIES_LABEL[variant],
        **kwargs,
    )


def label_endpoints(axis: Any, x: float, values: Mapping[Variant | str, float]) -> None:
    """Name each model beside the far end of its curve.

    Direct labels selectively, in the one panel where the curves separate far enough to
    label them apart; on a panel where the models agree, a label would sit on all of them at
    once and say nothing. The panel is given room above and below so that the outermost
    labels stay inside it however tightly the curves are packed.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Where to draw.
    x : float
        Where the curves end, in data coordinates.
    values : mapping to float
        Each model's value at `x`, keyed as `plot_series` takes them.
    """
    ordered = sorted(values, key=lambda variant: -values[variant])
    low, high = axis.get_ylim()
    span = high - low
    axis.set_ylim(low - 0.18 * span, high + 0.14 * span)
    offsets = np.linspace(13.0, -17.0, len(ordered))
    for variant, offset in zip(ordered, offsets, strict=True):
        axis.annotate(
            SERIES_LABEL[variant],
            (x, values[variant]),
            textcoords="offset points",
            xytext=(-5, offset),
            ha="right",
            color=SECONDARY_INK,
            fontsize=9,
        )


def variant_legend(figure: Any, variants: Sequence[Variant | str]) -> None:
    """Put one legend across the bottom of a figure, naming every model it draws.

    Placed outside the panels, so it cannot land on an axis label whatever the panels grow
    into. A legend is present whenever more than one model is drawn, even where the curves
    are also named on the plot.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    variants : sequence
        The models drawn, in the order to list them, keyed as `plot_series` takes them.
    """
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=SERIES_COLOUR[variant],
            linestyle=SERIES_STYLE[variant],
            marker=SERIES_MARKER[variant],
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=SERIES_LABEL[variant],
        )
        for variant in variants
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        labelcolor=SECONDARY_INK,
    )
