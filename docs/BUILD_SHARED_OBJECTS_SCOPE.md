# How should the build compile each Planck source once?

Scope, opened 2026-11-07. Nothing here is built. It expires when the work
lands; rewrite it then as the answer. Status lives in
`vault/Status/Open Work.md`.

## The finding

`CMakeLists.txt` has 82 executables and no library of Planck's own sources.
Every executable lists the `src/` files it needs, and CMake compiles them
again for each one. Measured on the full rebuild of 2026-11-07 (after a
`src/base/types.h` change), from the make log in `build-full`:

| | compile jobs |
|---|---|
| all jobs (70 targets that rebuilt) | 930 |
| `src/` jobs | 860, for **82 distinct files** |
| duplicate `src/` jobs | **778 (90 %)** |
| `tests/` jobs (one per test) | 70 |

The most duplicated files are the expensive ones: `basis/spherical.cpp` 53×,
`integrals/{os,hgp,rys,screening,shellpair}.cpp` 51× each, `basis/basis.cpp`
and `gbs_parser.cpp` 50×, `gaussian.cpp` 48×, `integrals/ecp.cpp` 38×. With
the `-j4` cap (heavy generated CC TUs), one header change costs about a
thousand compiles.

The hand-written source lists also break: three test targets failed to link
on 2026-11-07 because a source file moved or gained a dependency
(`planck-multipole-integrals`, `planck-population-analysis`,
`planck-working-state-rebuild`). CI never built them, so nobody noticed.

## What varies between targets (why one library is not a drop-in)

An object can be shared only by targets that would compile it identically.
Four things differ today:

1. **`USE_OPENMP`.** Defined per target, by 37 of the 82 executables. It
   guards code in `integrals/{os,hgp,rys,ecp}.cpp`, `fused_fock.h`,
   `basis/spherical.cpp`, `symmetry/skeleton_eri.h`, `casscf/orbital.cpp`,
   `dft/base/wrapper.h`, `dft/ks_matrix.cpp` and others. Of the 47
   executables that compile the integral engine, **16 build it without
   OpenMP** and so test serial object code.
2. **`USING_Libxc`.** `planck-dft`, `planck-mpi` and `planck-udh-validation`
   define it; `hartree-fock` does not. It changes `io/io.cpp` (libxc
   functional-name lookup) and `io/external_libraries.h`.
3. **`USE_MPI`.** Only `planck-mpi`. It changes the inline `rank()`,
   `size()` and `allreduce_inplace()` in `base/mpi_env.h`, which the
   integral engines and the DFT driver call. So every object that includes
   `mpi_env.h` differs between `planck-mpi` and the rest. Mixing the two
   would be an ODR violation.
4. **`PLANCK_CC_*` options.** `hartree-fock` and `planck-dft` get
   `PLANCK_CC_MAXORDER`, `PLANCK_CC_ARBITRARY_LOWER_RANKS` and
   `PLANCK_CC_DRESS_OPERATORS`; `hartree-fock` also gets `PLANCK_CC_UCC`.
   **`planck-mpi` omits `PLANCK_CC_DRESS_OPERATORS` and `PLANCK_CC_UCC`**, so
   with those options ON it compiles a different CC path from the serial
   binaries. That looks unintended; it should be checked rather than
   carried into the new structure.

Things that do **not** block sharing: `EIGEN_DONT_PARALLELIZE` is global;
the `-O1` pin on `generated_kernel_registry.cpp` and the version definitions
on the three `main.cpp` files are source-file properties, which apply in a
library too; `-UNDEBUG` applies only to `tests/*.cpp`, which stay per test.

## Recommended design

**One static library, `planck_core`,** holding every `src/` file except the
per-binary ones below, compiled once with `USE_OPENMP` (when OpenMP is
found), the `PLANCK_CC_*` options as `PUBLIC` definitions, and `PUBLIC`
usage requirements: include directories, `OpenMP::OpenMP_CXX`, libmsym,
libecpint. Every executable links it.

- **Static, not an OBJECT library.** An OBJECT library links every object
  into every executable, so each test would need every dependency resolved.
  A static archive lets the linker pull only the objects a test references.
  That is also what removes the hand-written per-test source lists, and the
  missing-source link failures with them.
- **One library, not one per directory.** The modules call each other in
  both directions (SCF ↔ post-HF response, integrals ↔ symmetry). Separate
  static libraries would need cycle handling in the link line for no gain.
- **Per-binary sources stay in each executable:** the three entry files
  (`driver.cpp`, `dft/main.cpp`, `mpi/main.cpp`, the only includers of
  `external_libraries.h`) and `io/io.cpp`, the one shared source that
  `USING_Libxc` changes. That is four files compiled up to three times, not
  eighty.
- **libxc** becomes a usage requirement of a small `planck_dft` static
  library (`DFT_LIB_SRC`), linked by `planck-dft`, `planck-mpi` and the DFT
  tests, so `hartree-fock` still does not link libxc.

**`planck-mpi`** cannot share objects that include `mpi_env.h`. Two options:

- **(a)** keep its own full source list, compiled with `USE_MPI` (one extra
  copy of the sources, only when `BUILD_MPI=ON`); or
- **(b)** move the bodies of `rank()`, `size()` and `allreduce_inplace()`
  out of the header into `src/base/mpi_env.cpp` (serial stub) and
  `src/mpi/mpi_env_mpi.cpp` (MPI), linked per binary. The core objects then
  no longer depend on `USE_MPI`, and `planck-mpi` links `planck_core` like
  everything else. The calls are per tensor or per Fock build, never per
  integral, so losing inlining costs nothing measurable.

Recommendation: **(a) first, (b) as its own step.** (b) changes code that
the MPI bitwise gates protect, so it should land separately, with those
gates as its check.

## Consequences to check

- **16 test executables become threaded.** They compile the engines today
  without OpenMP, so they would now run with whatever `OMP_NUM_THREADS`
  says. Planck's reductions are designed to be thread-count invariant, but
  a golden value pinned by a serial run could differ in the last bits.
  Check: `ctest` at `OMP_NUM_THREADS=1` and `4`, and diff the outputs.
- **The CC option mismatch in `planck-mpi`** disappears once the options are
  `PUBLIC` on the library. Confirm first that the omission was not
  deliberate.
- **Link order.** `planck_core` must come before libmsym, libecpint and
  libxc. `PUBLIC` usage requirements handle that.
- **CI time** should drop, since CI now builds every target.

## Steps

- **S0. Baseline.** Time a full rebuild after `touch src/base/types.h`
  (`-j4`, Release) and record the CI build time, so the gain is measured,
  not assumed.
- **S1. `planck_core` for the HF stack.** Build the library and link
  `hartree-fock` and every test that uses only HF-stack sources to it.
  Delete their per-test source lists. Gate: identical `hartree-fock`
  regression output (every printed digit), and `ctest` green at
  `OMP_NUM_THREADS=1` and `4`.
- **S2. `planck_dft`.** Move `planck-dft` and the DFT tests onto
  `planck_core` + `planck_dft`. Same gates for the DFT cases.
- **S3. `planck-mpi`, option (a).** Keep its own list; make its CC options
  match the other binaries (if S1 found the omission unintended). Gate: the
  MPI smoke and scale cases.
- **S4. Option (b), optional.** Move the `mpi_env.h` bodies into per-binary
  `.cpp` files and link `planck-mpi` against `planck_core`. Gate: the MPI
  bitwise cases at 2, 3 and 4 ranks.
- **S5. Measure.** Repeat S0. Expected from the job counts: about 930 jobs
  down to about 250 (82 shared files, a few per-binary variants, 70 test
  mains, and planck-mpi's copy until S4). The wall-time gain depends on
  per-file cost, which S0 measures.

## Not in scope

Unity builds, precompiled headers, ccache, splitting the generated CC
kernels (their own scope, `docs/CCGEN_KERNEL_WIRING_MULTISECTOR_SCOPE.md`),
and shared libraries (`.so`/`.dylib`): static keeps the binaries
self-contained, as they are today.
