"""Elementwise comparison of the committed membrane code against the reference C++.

The suite otherwise asserts scalar targets, which a regression could leave untouched while
still moving a whole trace. This closes that gap by checking the derivatives themselves,
against the reference implementation rather than against a stored fixture.

`tools/validate_base_model_IM.py` compares 500 randomised state/parameter/time cases x 25
derivatives against `references/SKNM_code/base_model_IM.h` compiled with g++, and exits
non-zero if any relative difference exceeds 1e-12.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "tools" / "validate_base_model_IM.py"
REFERENCE_HEADER = ROOT / "references" / "SKNM_code" / "base_model_IM.h"

pytestmark = pytest.mark.reference


@pytest.mark.skipif(
    not REFERENCE_HEADER.exists(),
    reason="references/SKNM_code is gitignored and absent from a fresh clone",
)
@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler to build the reference")
def test_committed_membrane_code_matches_the_c_reference():
    result = subprocess.run(
        [sys.executable, str(VALIDATE)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
