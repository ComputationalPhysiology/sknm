#!/usr/bin/env python3
"""Regenerate the committed membrane model code from the `.ode` sources.

The generated Python is committed so that `sknm`'s runtime dependency set stays numpy +
scipy and gotranx is needed only to regenerate. Run this after editing an `.ode`:

    python3 tools/generate_membrane_models.py

Requires the `codegen` extra:  pip install -e ".[codegen]"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "sknm" / "membrane"

# (.ode source, generated module name, what singularity guarding found in this model).
# The paragraph is written out per model rather than interpolated from a fragment, so that
# each generated file carries a note wrapped to this repo's line length.
SOURCES = [
    (
        "base_model_IM.ode",
        "base_model_IM.py",
        "Singularity guarding is on (gotranx's default) and adds one guard here,\n"
        "`ibarca_j` for |V_m| < 0.019 mV, at a cost of roughly 10% per step.",
    ),
    (
        "PBM.ode",
        "PBM.py",
        "Singularity guarding is on (gotranx's default) but adds no guard here: every\n"
        "denominator in this model's right hand side is a parameter, never a state.",
    ),
]

HEADER = '''"""Generated from `{source}`. Do not edit by hand.

Regenerate with `python3 tools/generate_membrane_models.py`.

{guards}

The scheme is `generalized_rush_larsen`; the backend is numpy, so this module imports numpy
and nothing else, which is what keeps `sknm`'s runtime dependencies to numpy + scipy.
"""

'''


def main() -> int:
    try:
        import gotranx
        import gotranx.cli.gotran2py
    except ImportError:
        print(
            'gotranx is not installed. Install the codegen extra:\n    pip install -e ".[codegen]"',
            file=sys.stderr,
        )
        return 1

    for source, module, guards in SOURCES:
        ode_path = ROOT / source
        code = gotranx.cli.gotran2py.get_code(
            gotranx.load_ode(ode_path),
            scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
            # gotranx defaults to black; this repo formats with ruff. Emitting ruff-formatted
            # code keeps regeneration idempotent. Otherwise the formatter rewrites the file and
            # every regeneration shows a few hundred lines of spurious diff.
            format=gotranx.cli.gotran2py.Format.ruff,
        )
        out = DEST / module
        out.write_text(HEADER.format(source=source, guards=guards) + code)
        print(f"wrote {out.relative_to(ROOT)}  ({len(code.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
