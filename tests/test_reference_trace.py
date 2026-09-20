"""Elementwise comparison of the committed membrane code against the reference C++.

The suite otherwise asserts scalar targets, which a regression could leave untouched while
still moving a whole trace. This closes that gap by checking the derivatives themselves,
against the reference implementation rather than against a stored fixture.

Each validator compares 500 randomised state/parameter/time cases against the corresponding
header in `references/SKNM_code` compiled with g++, and exits non-zero if the agreement is
worse than machine precision.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "references" / "SKNM_code"

pytestmark = pytest.mark.reference

#: (validator, reference header) for every committed generated model.
VALIDATORS = [
    ("validate_base_model_IM.py", "base_model_IM.h"),
    ("validate_PBM.py", "PBM.h"),
]


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler to build the reference")
@pytest.mark.parametrize(("validator", "header"), VALIDATORS)
def test_committed_membrane_code_matches_the_c_reference(validator, header):
    if not (REFERENCE_DIR / header).exists():
        pytest.skip("references/SKNM_code is gitignored and absent from a fresh clone")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / validator)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
