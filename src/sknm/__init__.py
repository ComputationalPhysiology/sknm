"""The Simplified Kirchhoff Network Model.

A Python implementation of the Simplified Kirchhoff Network Model (SKNM) of Jaeger & Tveito,
*Sci Rep* 13:16434 (2023), together with KNM and SKNM(u_e=0) so that the paper's model
comparisons can be reproduced. See ``CONTEXT.md`` for the domain vocabulary.
"""

from sknm import analysis, presets, units
from sknm.assembly import Operator, Variant, assemble
from sknm.linalg import BiCGSTABSolver, CGSolver, ConvergenceError, DirectSolver, Solver
from sknm.membrane import MembraneModel, from_gotranx
from sknm.network import CellNetwork, Connection, chain, from_edges, sheet
from sknm.simulation import Result, Simulation

__version__ = "0.1.0.dev0"

__all__ = [
    "BiCGSTABSolver",
    "CGSolver",
    "CellNetwork",
    "Connection",
    "ConvergenceError",
    "DirectSolver",
    "MembraneModel",
    "Operator",
    "Result",
    "Simulation",
    "Solver",
    "Variant",
    "__version__",
    "analysis",
    "assemble",
    "chain",
    "from_edges",
    "from_gotranx",
    "presets",
    "sheet",
    "units",
]
