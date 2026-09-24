"""Compare the committed beta cell membrane code against the original gotran C++, elementwise.

This checks the file that ships. It imports `sknm.membrane.PBM` rather than regenerating in
memory, so a stale or hand-edited committed file is caught. Regenerate with
`python3 tools/generate_membrane_models.py`.

The model is short enough to read through, but reading it is not enough: a transcription that
looks right and is wrong by one sign produces a plausible burst at the wrong frequency.

Run directly for the full report, or via `pytest -m reference` for a pass/fail check.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from sknm.membrane import PBM

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path("/tmp/sknm-tools-build")
BUILD.mkdir(parents=True, exist_ok=True)

DRIVER = BUILD / "pbm_rhs_driver"
if not DRIVER.exists():
    subprocess.run(
        [
            "g++",
            "-O2",
            "-I",
            str(ROOT / "references" / "SKNM_code"),
            str(ROOT / "tools" / "pbm_rhs_driver.cpp"),
            "-o",
            str(DRIVER),
        ],
        check=True,
    )

HDR = ROOT / "references" / "SKNM_code" / "PBM.h"
src = HDR.read_text()

NSTATE = 5
NPARAM = 25


# --- name order as the C uses it -------------------------------------------
def names(fn, n):
    blk = src.split("int " + fn + "(")[1].split("};")[0]
    return re.findall(r'"(\w+)"', blk)[:n]


c_states = names("state_index", NSTATE)
c_params = names("parameter_index", NPARAM)


def init_vals(fn):
    body = src.split("void " + fn + "(")[1].split("\n}")[0]
    d = {}
    for m in re.finditer(r"\[(\d+)\]\s*=\s*([^;]+);\s*//\s*(\w+);", body):
        d[m.group(3)] = float(m.group(2))
    return d


s0 = init_vals("init_state_values")
p0 = init_vals("init_parameters_values")

# --- index maps between C order and gotranx order ---------------------------
py_s = [PBM.state_index(n) for n in c_states]  # py_s[c_idx] = py_idx
py_p = [PBM.parameter_index(n) for n in c_params]

# --- build randomised test cases -------------------------------------------
# The right hand side has no conditionals, so there are no branches to cover; what matters is
# spanning the ranges the states actually visit. v is swept across the whole action potential
# and beyond it, and c across the range that drives the Ca-activated K+ current through the
# steep part of its Hill term.
rng = np.random.default_rng(20230101)
NCASE = 500
base_p = np.array([p0[n] for n in c_params])
RANGES = {
    "v": (-90.0, 10.0),
    "n": (0.0, 1.0),
    "c": (0.01, 1.0),
    "a": (0.0, 1.0),
    "cer": (50.0, 200.0),
}

cases = []
for k in range(NCASE):
    s = np.array([rng.uniform(*RANGES[n]) for n in c_states])
    # Sweep v deterministically as well, so every case is a distinct membrane potential rather
    # than 500 draws that might cluster.
    s[c_states.index("v")] = -90.0 + 100.0 * k / (NCASE - 1)
    p = base_p.copy()
    # Half the cases carry the stimulus: the beta stimulus is a halved gkatpbar, so both
    # values have to be exercised.
    if k % 2:
        p[c_params.index("gkatpbar")] = 0.5 * p0["gkatpbar"]
    # The model is autonomous, with t appearing nowhere in the right hand side, but the
    # signature takes it, so pass a spread of values rather than only zero.
    cases.append((s, p, float(rng.uniform(0.0, 1000.0))))

# --- run the C --------------------------------------------------------------
inp = ["%d" % NCASE]
for s, p, t in cases:
    inp.append(" ".join("%.17g" % x for x in s))
    inp.append(" ".join("%.17g" % x for x in p))
    inp.append("%.17g" % t)
out = (
    subprocess.run(
        [str(DRIVER)],
        input="\n".join(inp),
        capture_output=True,
        text=True,
        check=True,
    )
    .stdout.strip()
    .split("\n")
)
c_res = np.array([[float(x) for x in line.split()] for line in out])  # (N,5) C order

# --- run the Python ---------------------------------------------------------
py_res = np.empty_like(c_res)
for k, (s, p, t) in enumerate(cases):
    S = np.empty(NSTATE)
    S[py_s] = s
    P = np.empty(NPARAM)
    P[py_p] = p
    v = PBM.rhs(t, S, P)
    py_res[k] = v[py_s]  # back to C order

# --- compare ----------------------------------------------------------------
# Scaled by each derivative's own range rather than by its value at the case, as the hiPSC
# validator can afford to do. `dcer/dt` is the difference of two ER fluxes that
# very nearly cancel, so individual values pass through zero: at one case it is -2.5e-08 from
# terms of order 1e-2, where a difference of 3e-19, two units in the last place of a double,
# reads as a relative error of 1e-11. Dividing by the state's range measures the agreement that
# is actually there, and still catches a real transcription error, which moves a derivative by
# a sizeable fraction of its own range rather than by an ulp.
difference = np.abs(c_res - py_res)
scale = np.abs(c_res).max(axis=0)
scaled = difference / scale
relative = np.abs(c_res - py_res) / np.maximum(np.abs(c_res), 1e-300)

print("cases: %d x %d derivatives = %d comparisons" % (NCASE, NSTATE, c_res.size))
print(
    "non-finite in C: %d   in Python: %d"
    % ((~np.isfinite(c_res)).sum(), (~np.isfinite(py_res)).sum())
)
print("max difference scaled by each derivative's range: %.3e" % np.nanmax(scaled))
worst = np.unravel_index(np.nanargmax(scaled), scaled.shape)
print(
    "worst at case %d, state %s: C=%.17g  py=%.17g"
    % (worst[0], c_states[worst[1]], c_res[worst], py_res[worst])
)
print()
print("per-state:      max |C - py|     range of |C|      scaled     pointwise relative")
for i, n in enumerate(c_states):
    print(
        "  d%-4s/dt      %.3e       %.3e     %.3e     %.3e"
        % (n, difference[:, i].max(), scale[i], scaled[:, i].max(), np.nanmax(relative[:, i]))
    )

ok = np.nanmax(scaled) < 1e-13
print(
    "\n%s"
    % ("PASS - transcription is numerically identical to the C" if ok else "FAIL - see above")
)
sys.exit(0 if ok else 1)
