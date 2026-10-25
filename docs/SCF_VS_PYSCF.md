# Where does a Planck RHF lose to PySCF, and what closed it?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This file answers one question about direct-SCF
RHF on water chains: where is the time lost relative to PySCF, and which
levers paid? It follows `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, which had
already made the Fock build fast and closed.

## Short answer

Not where it looked. Iteration counts are identical once the convergence
tolerances match. Two real costs were found and removed: a 246 KB zero-fill
per primitive in the nuclear-attraction routine, which dominated setup, and
the lack of an incremental Fock build. After both, Planck beats PySCF on
large Pople-basis runs: 12 H2O / 6-31G\* takes 3.1 s against 7.2 s.

What remains is a per-primitive-quartet kernel gap: 4–16× in a full Fock
build, largest on small, dense cc-pVXZ systems and shrinking with size. It
is **not** general contraction. PySCF given the same segmented
representation Planck uses is just as fast. So the shared-exponent family
work (B2.2) cannot close it.

## Measured: end to end, after both fixes

M3, 4 threads, `tests/benchmarks/pyscf_bench.py`. Water chain, RHF,
Cartesian, no symmetry, direct SCF, hcore guess, Planck `engine auto`
(`build-full`), PySCF 2.13.0 (NumPy on Accelerate). Both converge to
`conv_tol 1e-9` and `conv_tol_grad 1e-7`, the latter matching Planck's 1e-7
DIIS-error and density gates.

| basis | system | nb | iterations Planck / PySCF | wall Planck | wall PySCF |
|---|---|---|---|---|---|
| 6-31G\* | 4 H2O | 76 | 14 / 14 | 0.46 s | 0.11 s |
| 6-31G\* | 8 H2O | 152 | 15 / 15 | 1.38 s | 1.10 s |
| 6-31G\* | 12 H2O | 228 | 15 / 15 | **3.11 s** | 7.22 s |
| cc-pVDZ | 4 H2O | 100 | 15 / 15 | 1.72 s | 0.24 s |
| cc-pVDZ | 6 H2O | 150 | 15 / 15 | 3.96 s | 1.00 s |
| cc-pVDZ | 8 H2O | 200 | 16 / 15 | 7.56 s | 4.02 s |
| cc-pVTZ | 1 H2O | 65 | 14 / 13 | 0.54 s | 0.05 s |
| cc-pVTZ | 2 H2O | 130 | 15 / 14 | 3.98 s | 0.58 s |
| cc-pVTZ | 3 H2O | 195 | 15 / 15 | 12.63 s | 6.86 s |

Energies agree to about 9.6e-9 Eh per water, growing linearly with size.
That is the basis-coefficient precision floor
(`docs/INTEGRAL_PRECISION_FLOOR_SCOPE.md`), not a defect. The harness's
absolute 1e-7 pass line therefore "fails" at 12 waters.

Before the two fixes, 12 H2O / 6-31G\* took 5.6–6.2 s, and 8 H2O / cc-pVDZ
10.7–11.7 s.

## What was found

### 1. Iteration counts are a tolerance artifact, not a DIIS gap

With PySCF's default `conv_tol_grad` (√conv_tol = 3e-5), PySCF stops 2–4
iterations earlier than Planck. Planck's 1e-7 DIIS-error and density gates
are much tighter. Matched at 1e-7, the counts are equal. Screening
thresholds barely matter either: PySCF at `direct_scf_tol` 1e-10 (Planck's
`tol_eri`) instead of its default 1e-13 is under 10 % faster. Process start
is 0.07 s warm and 0.8 s cold, so the 15–50× wall ratios on single waters are
exec time plus cold caches against in-process PySCF.

### 2. Setup was a zero-fill (S1, `f0910687`)

A `sample` of the first 3 s of 12 H2O / 6-31G\* put 77 % of samples in
`_compute_nuclear_attraction`, and 90 % of that in `__bzero` called from
`_os_nuclear_primitive`. The routine declared
`double V[13][13][13][14] = {}`, sized for MAX_L. That is 246 KB cleared per
primitive pair × charge × AO component, even for s–s. Every read in the VRR,
the m = 0 extraction and `_nuclear_hrr` is of an entry the recurrence wrote
first, so the initializers on `V`, `V0` and the derivative's `dV0` were
dropped. The arithmetic is unchanged and the result bitwise-identical.

| case (4 threads) | setup before | setup after |
|---|---|---|
| 12 H2O / 6-31G\* | 2.40–2.46 s | 0.05 s |
| 8 H2O / cc-pVDZ | 1.78–2.35 s | 0.05 s |
| 3 H2O / cc-pVTZ | 0.61–0.83 s | 0.04 s |

The same routine serves PCM (every cavity point, every iteration) and ESP.
Their outputs, and those of SCF, CHELPG, the spherical gradient and RKS-PCM
inputs, were identical before and after.

### 3. The incremental build pays only with a tight density test (S3, `2772788a`)

`G_n = G_{n−1} + G[P_n − P_{n−1}]` in the direct RHF, UHF and ROHF loops
(`direct_fock_rhf` / `direct_fock_uhf`, `src/scf/scf.cpp`, Cartesian and
spherical). G is linear in the density, so this is exact however P was
produced: SOSCF, damping and level shifts need nothing special. The fused
loop skips a block when
`Q_ab Q_cd · max(4|D_ab|, 4|D_cd|, |D_ac|, |D_ad|, |D_bc|, |D_bd|) < tol_eri / 10`
(`screened`, `src/integrals/fused_fock.h`, both traversals), with |D| the
shell-block maximum over every density passed in. In replication mode the
test is off, because a representative block there is scattered onto images
whose density differs. The Schwarz table stays unscreened.

A counting probe came first. It measured the fraction of Schwarz-surviving
work (components × primitives) that also survives the density test. With
the full P that fraction is 0.89–0.96 every iteration, worth only
1.04–1.11×. With ΔP it falls below half by iterations 9–12 and ends at
0.18–0.37, worth 1.25–1.6× over a run.

The tolerance decided the design. At `tol_eri` (1e-10) the set of skipped
blocks changes with ΔP every iteration. The screening error is then
inconsistent between iterations, so DIIS cannot absorb it: the energy
random-walks at 1e-8, and the energy gate fails (12 H2O / 6-31G\*: 15 → 24
iterations, |ΔE| 1.5e-8 against the full build). A full rebuild every 5
iterations made it worse (35 iterations), because the error comes from
inconsistency, not accumulation. Fock-build time, best of 3, against the
pre-S3 binary:

| case | `tol_eri` | `tol_eri / 10` (used) | `tol_eri / 100` |
|---|---|---|---|
| 12 H2O 6-31G\* | 1.19× (24 iters) | 1.24× | 1.15× |
| 8 H2O cc-pVDZ | 1.46× | 1.33× | 1.21× |
| 4 H2O cc-pVDZ | 1.34× | 1.23× | 1.14× |
| 2 H2O cc-pVTZ | 1.28× | 1.19× | 1.12× |

At `/10`, |ΔE| ≤ 6e-10 against the full build and the iteration counts are
unchanged. UHF matches to 5e-10 in the same iterations. ROHF matches to
3e-10 but took 71 iterations against 67. The gain is below PySCF's own: its
in-SCF iteration is 1.5–2× cheaper than its full build, because it screens
at 1e-13 and so has more to drop.

### 4. The remaining gap is the kernel, not the contraction scheme (S2)

Full Fock builds compared like for like: Planck's iteration-1 build (still a
full build after S3) against PySCF's `get_veff` from scratch, timed **warm**
(after one untimed call), with `direct_scf_tol` 1e-10, on the hcore guess:

| case | nb | Planck 1t / 4t | PySCF 1t / 4t | ratio 1t / 4t |
|---|---|---|---|---|
| 4 H2O 6-31G\* | 76 | 0.067 / 0.022 s | 0.011 / 0.0031 s | 6.2× / 7.1× |
| 12 H2O 6-31G\* | 228 | 0.736 / 0.248 s | 0.949 / 0.324 s | 0.78× / 0.77× |
| 4 H2O cc-pVDZ | 100 | 0.389 / 0.136 s | 0.033 / 0.0082 s | 12.0× / 16.6× |
| 8 H2O cc-pVDZ | 200 | 2.00 / 0.567 s | 0.531 / 0.155 s | 3.8× / 3.7× |
| 1 H2O cc-pVTZ | 65 | 0.094 / 0.029 s | 0.0066 / 0.0018 s | 14× / 16× |
| 2 H2O cc-pVTZ | 130 | 1.00 / 0.284 s | 0.091 / 0.0249 s | 11× / 11.4× |
| 3 H2O cc-pVTZ | 195 | 3.15 / 0.860 s | 0.564 / 0.135 s | 5.6× / 6.4× |

- **Where it lies:** in per-quartet cost on dense blocks. It is largest when
  almost every quartet survives screening, and Planck's block screening wins
  it back with size: 6-31G\* crosses over by 12 waters, and cc-pVXZ falls
  from 11–16× to 4–6×.
- **Profile:** the one-thread profile matches the ERI record. On 4 H2O
  cc-pVDZ: closed forms 41 % + Boys 23 %, digestion 15 %, HGP VRR 13 %. On
  2 H2O cc-pVTZ: HGP VRR + finish + prepare 58 %, digestion 21 %, closed
  forms 13 %. Every lever the ERI work found there is already pulled.
- **Not general contraction:** cc bases share primitive exponents across the
  contracted functions of one L, and Planck evaluates each contracted
  shell's primitive quartets separately. The test: split PySCF's own
  general contractions into segmented shells that repeat the primitives, as
  Planck's GBS does (same functions, identical hcore energy). PySCF's
  `get_veff` is unchanged: 0.94–1.02× on 4 H2O cc-pVDZ and 2 H2O cc-pVTZ, at
  1 and 4 threads. libcint pays nothing for the segmented representation, so
  sharing exponents (B2.2) cannot be what separates the two codes. Its
  measured ceiling was ~1.29× of a cc-pVDZ build in any case.

## What invariants matter

1. **Compare at matched convergence criteria.** PySCF's default
   `conv_tol_grad` is √`conv_tol`. Leaving it there makes PySCF look 20–30 %
   faster by stopping early.
2. **Time PySCF warm.** The first `get_veff` on a molecule builds its
   screening object. Timing it cold made PySCF look 3–4× slower than it is,
   and an earlier version of the S2 table in this file understated the
   kernel gap because of it.
3. **Never zero a worst-case-sized buffer in a per-primitive routine.** Clear
   the extent used, or show every read is preceded by a write.
4. **Screening error must be consistent from iteration to iteration.** An
   error that is the same every iteration only shifts the converged answer.
   One that changes each iteration is noise the convergence test sees. That
   is why the density test runs ten times tighter than `tol_eri`.
5. **The Schwarz table is never density-screened**
   (`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, invariant 8).

## Not done

- **The per-quartet kernel gap.** Closing it needs libcint-class kernels
  (`docs/ERI_PERFORMANCE_SCOPE.md`), not a traversal change. B2.2 is
  declined on the segmented-representation result above.
- **DFT:** the KS Coulomb/exchange direct builds (`_compute_2e_j_direct` /
  `_compute_2e_k_direct`) do not yet build incrementally.
- **Gradients:** `docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md`.
- **RI-JK:** D3 in `docs/ERI_FOCK_BLAS_SCOPE.md`.

## Harness fixes made while measuring

`tests/benchmarks/pyscf_bench.py` parsed the SCF table with an 8-field regex
that also matched MO-energy rows. That inflated the iteration count about 3×
(45 for a 15-iteration run) and returned a per-iteration time larger than the
wall clock. It now reads `PLANCK_PHASE_TIMING` lines. It also gained
`--pyscf-conv-tol-grad` and prints iteration counts and wall times for both
codes. It still times PySCF's whole `kernel()`, which includes the cold
first build, so its per-iteration column for PySCF reads slightly high on
small systems.
