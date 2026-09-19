"""The Simplified Kirchhoff Network Model.

A Python implementation of the Simplified Kirchhoff Network Model (SKNM) of Jaeger & Tveito,
*Sci Rep* 13:16434 (2023), together with KNM and SKNM(u_e=0) so that the paper's model
comparisons can be reproduced. See ``CONTEXT.md`` for the domain vocabulary.
"""

from sknm.membrane import MembraneModel, from_gotranx

__version__ = "0.1.0.dev0"

__all__ = ["MembraneModel", "__version__", "from_gotranx"]
