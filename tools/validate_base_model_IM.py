"""Compare the transcribed .ode against the original gotran C++, elementwise."""
import re, subprocess, sys
import numpy as np

sys.path.insert(0, "/tmp/claude-0/-home-shared/b50163d5-5b8e-4455-8724-1cc01a2e4380/scratchpad")
import bmIM

HDR = "/home/shared/references/SKNM_code/base_model_IM.h"
src = open(HDR).read()

# --- name order as the C uses it -------------------------------------------
def names(fn, n):
    body = src.split("char names[][")[1] if False else None
    blk = src.split("int " + fn + "(")[1].split("};")[0]
    return re.findall(r'"(\w+)"', blk)[:n]

c_states = names("state_index", 25)
c_params = names("parameter_index", 83)

def init_vals(fn):
    body = src.split("void " + fn + "(")[1].split("\n}")[0]
    d = {}
    for m in re.finditer(r"\[(\d+)\]\s*=\s*([^;]+);\s*//\s*(\w+);", body):
        d[m.group(3)] = float(m.group(2))
    return d

s0 = init_vals("init_state_values")
p0 = init_vals("init_parameters_values")

# --- index maps between C order and gotranx order ---------------------------
py_s = [bmIM.state_index(n) for n in c_states]      # py_s[c_idx] = py_idx
py_p = [bmIM.parameter_index(n) for n in c_params]

# --- build randomised test cases -------------------------------------------
rng = np.random.default_rng(20230101)
NCASE = 500
base_s = np.array([s0[n] for n in c_states])
base_p = np.array([p0[n] for n in c_params])
vm_c = c_states.index("V_m")

cases = []
for k in range(NCASE):
    s = base_s * rng.uniform(0.8, 1.25, 25)
    # sweep V_m across both branches of the aj/bj and i_Stim conditionals
    s[vm_c] = rng.uniform(-90.0, 40.0)
    # exercise the stimulus window: period 5000, start/duration from defaults
    t = rng.choice([0.0, p0["stim_start"] + 0.5 * p0["stim_duration"],
                    p0["stim_start"] - 1.0, 3.0 * p0["stim_period"] + 1.0,
                    rng.uniform(0, 2 * p0["stim_period"])])
    cases.append((s, base_p.copy(), float(t)))

# --- run the C --------------------------------------------------------------
inp = ["%d" % NCASE]
for s, p, t in cases:
    inp.append(" ".join("%.17g" % x for x in s))
    inp.append(" ".join("%.17g" % x for x in p))
    inp.append("%.17g" % t)
out = subprocess.run(
    ["/tmp/claude-0/-home-shared/b50163d5-5b8e-4455-8724-1cc01a2e4380/scratchpad/driver"],
    input="\n".join(inp), capture_output=True, text=True, check=True,
).stdout.strip().split("\n")
c_res = np.array([[float(x) for x in line.split()] for line in out])  # (N,25) C order

# --- run the Python ---------------------------------------------------------
py_res = np.empty_like(c_res)
for k, (s, p, t) in enumerate(cases):
    S = np.empty(25); S[py_s] = s
    P = np.empty(83); P[py_p] = p
    v = bmIM.rhs(t, S, P)
    py_res[k] = v[py_s]   # back to C order

# --- compare ----------------------------------------------------------------
den = np.maximum(np.abs(c_res), np.abs(py_res))
rel = np.where(den > 0, np.abs(c_res - py_res) / np.where(den > 0, den, 1), 0.0)

print("cases: %d x 25 derivatives = %d comparisons" % (NCASE, c_res.size))
print("non-finite in C: %d   in Python: %d"
      % ((~np.isfinite(c_res)).sum(), (~np.isfinite(py_res)).sum()))
print("max relative difference: %.3e" % np.nanmax(rel))
worst = np.unravel_index(np.nanargmax(rel), rel.shape)
print("worst at case %d, state %s: C=%.17g  py=%.17g"
      % (worst[0], c_states[worst[1]], c_res[worst], py_res[worst]))
print()
print("per-state max relative difference:")
for i, n in enumerate(c_states):
    print("  d%-6s/dt  %.3e" % (n, np.nanmax(rel[:, i])))

ok = np.nanmax(rel) < 1e-12
print("\n%s" % ("PASS - transcription is numerically identical to the C"
                if ok else "FAIL - see above"))
sys.exit(0 if ok else 1)
