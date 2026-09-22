"""Test configuration.

`examples/` is a directory of scripts rather than a package, so it is not importable by
default. The bidomain and monodomain machinery lives there -- it is example code, not part of
the library, and it requires `dolfinx`, which `sknm` deliberately does not depend on -- yet it
still needs testing, so this puts it on the path.
"""

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
