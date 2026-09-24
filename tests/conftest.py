"""Test configuration.

`examples/` and `tools/` are directories of scripts rather than packages, so neither is
importable by default. The bidomain and monodomain machinery lives in the first, as example
code rather than part of the library, because it requires `dolfinx`, which `sknm` deliberately
does not depend on. The reference page generator lives in the second. Both still need testing,
so this puts them on the path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
TOOLS_DIR = ROOT / "tools"

for directory in (EXAMPLES_DIR, TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
