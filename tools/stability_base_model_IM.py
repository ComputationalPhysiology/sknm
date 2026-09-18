"""Rush-Larsen stability and accuracy of base_model_IM: gotranx Python vs the original C++.

Protocol (ticket 02): single cell, stim_amplitude = 20, 300 ms, V_m sampled every 1 ms.
Reference is the original C++ with forward explicit Euler at dt = 1e-5 ms. Reports, per scheme
and dt, the max |V_m - reference| over the 300 samples, or the time at which it went non-finite.

Usage:  python3 tools/stability_base_model_IM.py [--build-dir DIR]
The C reference is cached in the build dir; delete it to recompute (~1 min).
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C_SRC = ROOT / "references" / "SKNM_code"
ODE = ROOT / "base_model_IM.ode"
T_END, STIM = 300.0, 20.0
GRL_DTS = (0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.00025)


def c_trajectory(build: Path, dt: float, euler: bool) -> np.ndarray:
    exe = build / "conv_driver"
    if not exe.exists():
        subprocess.run(
            [
                "g++",
                "-O2",
                "-I",
                str(C_SRC),
                str(ROOT / "tools" / "conv_driver.cpp"),
                "-o",
                str(exe),
            ],
            check=True,
        )
    cache = build / f"c_{'euler' if euler else 'rl'}_{dt:g}.txt"
    if not cache.exists():
        out = subprocess.run(
            [str(exe), repr(dt), "1" if euler else "0"], capture_output=True, text=True, check=True
        )
        cache.write_text(out.stdout)
    return np.loadtxt(cache)


def python_module(remove_singularities: bool) -> dict:
    import gotranx
    import gotranx.cli.gotran2py
    from gotranx.schemes import Scheme

    ode = gotranx.load_ode(ODE, remove_singularities=remove_singularities)
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[Scheme.generalized_rush_larsen])
    namespace: dict = {}
    exec(compile(code, "<base_model_IM>", "exec"), namespace)
    return namespace


def python_trajectory(mod: dict, dt: float) -> tuple[np.ndarray, float | None, float]:
    """V_m every 1 ms; the time it went non-finite (or None); wall seconds per step."""
    p = mod["init_parameter_values"](stim_amplitude=STIM)
    s = mod["init_state_values"]()
    vi = mod["state_index"]("V_m")
    step = mod["generalized_rush_larsen"]
    n_steps, every = round(T_END / dt), round(1.0 / dt)
    samples, t = [], 0.0
    start = time.perf_counter()
    with np.errstate(all="ignore"):
        for n in range(n_steps):
            s = step(s, t, dt, p)
            t = (n + 1) * dt
            if not np.all(np.isfinite(s)):
                return np.array(samples), t, (time.perf_counter() - start) / (n + 1)
            if (n + 1) % every == 0:
                samples.append(s[vi])
    return np.array(samples), None, (time.perf_counter() - start) / n_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/sknm-tools-build"))
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)

    import gotranx

    ref = c_trajectory(args.build_dir, 1e-5, euler=True)
    c_rl = c_trajectory(args.build_dir, 0.02, euler=False)
    print(f"gotranx {gotranx.__file__}")
    print(f"reference: C forward_explicit_euler, dt = 1e-5 ms, {len(ref)} samples\n")
    print(f"{'scheme':48s} {'dt (ms)':>8s}  result")
    print(f"{'C forward_rush_larsen':48s} {0.02:>8g}  max err {np.max(np.abs(c_rl - ref)):.4g} mV")

    for guards in (True, False):
        mod = python_module(remove_singularities=guards)
        label = f"gotranx generalized_rush_larsen (guards {'on' if guards else 'off'})"
        for dt in GRL_DTS:
            v, blew_up, per_step = python_trajectory(mod, dt)
            if blew_up is not None:
                result = f"BLOWS UP at t = {blew_up:.4g} ms"
            else:
                result = f"max err {np.max(np.abs(v - ref)):.4g} mV"
            print(f"{label:48s} {dt:>8g}  {result:32s} ({per_step * 1e6:.0f} us/step)")


if __name__ == "__main__":
    main()
