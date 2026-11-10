# What should CI build and run, and how does it fit on a GitHub runner?

Scope, opened 2026-11-10. Nothing here is built. It expires when CI runs
green; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md` (priority 8). The build half is scoped in
`docs/BUILD_SHARED_OBJECTS_SCOPE.md`.

## The state

`.github/workflows/cmake-multi-platform.yml` is `disabled_manually`; its last
run was 2026-04-26 and runs failed for lack of disk. Since #181 it builds
every target (`cmake --build … -j$(nproc)`) in two jobs
(`PLANCK_CC_MAXORDER` 3 and 4) and runs `ctest`. Callgraph Profiling
(`callgraph.yml`) is disabled for the same reason. The #181 fixes (build
every target, `BASIS_PATH` per test) have never run on CI. `ctest` runs the
~94 unit tests only; the regression runner (`tests/run_regressions.py`) is
not in CI at all.

## Steps

### 1. Measure the disk, before changing anything

On a local Linux container (or one manual workflow run with
`df -h` and `du -sh build` steps added): disk used after configure, after
the external projects (libxc, libmsym, libecpint, simple-dftd3, Eigen), and
after the full build, per `PLANCK_CC_MAXORDER`. GitHub's ubuntu runner has
~14 GB free. This says how much the shared-objects change has to save and
whether it is enough on its own.

### 2. Cheap disk relief that needs no build change

- Free runner space first (remove the preinstalled Android/.NET/Haskell
  toolchains: a standard ~20 GB step).
- Build the external projects once and cache them (`actions/cache` keyed on
  their pinned commits).
- Drop debug info (`-g0`) in CI.

If (1) and (2) fit, re-enable CI now; do not wait for step 3.

### 3. Shared objects

`docs/BUILD_SHARED_OBJECTS_SCOPE.md`: one `planck_core` static library
removes ~90 % of duplicate compiles and their objects. This is the durable
fix and also cuts build time.

### 4. What CI runs

- Job A (`MAXORDER=3`): build, `ctest`, then
  `run_regressions.py --suite core` (132 cases; measure its wall time on the
  runner — budget ~30 min).
- Job B (`MAXORDER=4`): build and `ctest` only (the generated kernels compile
  and link; the rank-4 cases are too slow for CI).
- Smoke and extended suites stay local. Record the four known RCCSDT routing
  failures as expected so they do not mask new ones, or keep them out of
  `core`.

The Boys core is out-of-tree and licensed; the `PLANCK_LICENSE` secret is
set, but PRs from forks cannot read it. Decide whether fork PRs skip the
build or CI runs on `push` only.

### 5. First run is also a first Linux build

simple-dftd3 with gfortran and libecpint under g++ on Linux have never been
built by CI. Expect to fix something there; it is not a sign the plan is
wrong.

## Not in this scope

macOS or Windows runners, sanitizers, and performance tracking.
