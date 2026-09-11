# 05 — Packaging and tooling

Parent: [map](../map.md)
Type: grilling
Status: open
Blocked by: —

## Question

Settle the repository's mechanical decisions. Small, but they are load-bearing for every later
session and cheap to get wrong.

Fixed by the user already: **src layout**, **modern `pyproject.toml`**, package importable as
`import sknm`, a `tests/` folder, an `examples/` folder. Runtime dependencies are **numpy +
scipy only**; matplotlib is an examples extra.

Open:

1. **Build backend**: hatchling, setuptools, flit, or pdm? (Recommendation: hatchling.)
2. **Python versions supported.** Local env is 3.12.3 on aarch64. Floor at 3.10? 3.11?
3. **Version management** — static in `pyproject.toml`, or dynamic from VCS tags?
4. **Lint/format/type**: ruff (lint + format)? mypy or pyright, and at what strictness? Are type
   hints mandatory in the numerical core, where they tend to be noise (`npt.NDArray`)?
5. **Dependency/environment tooling**: `uv` is *not* currently installed. Adopt it, or plain
   pip + venv?
6. **CI**: GitHub Actions? Note the repo currently has **no commits and no remote** — is there a
   remote to push to, and is CI in scope at all?
7. **Optional-dependency extras**: `[examples]` (matplotlib), `[dev]` (pytest, ruff, mypy),
   `[codegen]` (gotranx, needed only to regenerate membrane models)?
8. **Pre-commit hooks** — yes or no?
9. **Repo hygiene, needs doing regardless**: `references/` currently holds ~36 MB of duplicate
   zips (`8340201.zip`, `SKNM_code.zip`), a `__MACOSX/` junk directory, and ~15 MB of
   visualization-only `.mesh`/`.msh` files. Decide what is committed vs gitignored **before the
   first commit**, since the repo has no history yet and this is the moment it is free to fix.
10. **Licence** for `sknm` itself, and the CC-BY-4.0 attribution wording for the port
    (see the map's Notes).

## Answer

_(unresolved)_
