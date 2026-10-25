# Where does a Planck RHF lose to PySCF, and what closes it?

Scope for in-flight work (it expires when the work lands; rewrite it then as
the answer). Measured after the shell-quartet Fock-build work
(`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`) closed.

## Short answer

Not where expected. With matched convergence tolerances both codes take the
same number of iterations, and on Pople bases Planck's direct-SCF iteration
is already level with PySCF and ahead at the largest size. Three things
remain:

1. **Setup** (fixed by S1), outside the SCF loop, was dominated by the nuclear-attraction
   matrix: `_os_nuclear_primitive` zero-fills a fixed
   `double V[13][13][13][14]` (246 KB) on every primitive pair, nucleus and AO
   component. That is 2.6 s of the 6.2 s wall on 12 H2O / 6-31G\*: more than
   PySCF's whole run, once removed (~3.6 s vs 6.9 s).
2. **Per-iteration cost on generally contracted bases**: 4.5–7× PySCF on
   cc-pVDZ / cc-pVTZ up to nb ≈ 150, narrowing to 2× by nb ≈ 200. The Pople
   bases do not show it.
3. **No incremental Fock build and no density screening** (fixed by S3,
   1.19–1.33× on the Fock build). Every iteration rebuilt G from the full
   density, and the block skip was `qmax[bra] · qmax[ket] < tol_eri` alone.

## Measured (M3, 4 threads, 2026-10-25)

`tests/benchmarks/pyscf_bench.py`: water chain, RHF, Cartesian, no symmetry,
direct SCF, hcore guess, `engine auto` (`build-full`), PySCF 2.13.0 (NumPy on
Accelerate), `conv_tol 1e-9`, `conv_tol_grad 1e-7` (matches Planck's 1e-7
DIIS-error and density gates). Planck per-iteration time from the
`PLANCK_PHASE_TIMING` probe.

| basis | system | nb | iters Planck / PySCF | s/iter Planck | s/iter PySCF | wall Planck | wall PySCF |
|---|---|---|---|---|---|---|---|
| 6-31G\* | 4 H2O | 76 | 14 / 14 | 0.023 | 0.017 | 0.63 | 0.23 |
| 6-31G\* | 8 H2O | 152 | 15 / 15 | 0.102 | 0.085 | 2.12 | 1.27 |
| 6-31G\* | 12 H2O | 228 | 15 / 15 | 0.232 | 0.417 | 5.26 | 6.26 |
| cc-pVDZ | 4 H2O | 100 | 15 / 15 | 0.111 | 0.015 | 2.22 | 0.22 |
| cc-pVDZ | 6 H2O | 150 | 15 / 15 | 0.303 | 0.066 | 6.95 | 0.99 |
| cc-pVDZ | 8 H2O | 200 | 16 / 15 | 0.522 | 0.249 | 11.72 | 3.73 |
| cc-pVTZ | 1 H2O | 65 | 14 / 13 | 0.024 | 0.004 | 0.50 | 0.05 |
| cc-pVTZ | 2 H2O | 130 | 15 / 14 | 0.258 | 0.038 | 4.34 | 0.53 |
| cc-pVTZ | 3 H2O | 195 | 15 / 15 | 0.854 | 0.366 | 13.69 | 5.49 |

- **Iteration counts:** with PySCF's default `conv_tol_grad` (√conv_tol =
  3e-5) it stops 2–4 iterations earlier. That is a tolerance difference, not a
  DIIS one.
- **Screening threshold:** PySCF at `direct_scf_tol` 1e-10 (Planck's
  `tol_eri`) instead of its default 1e-13 is under 10 % faster (12 H2O / 6-31G\*
  0.485 → 0.460 s/iter, cc-pVDZ 8 H2O 0.261 → 0.250).
- **Process start:** 0.07 s warm, 0.8 s cold, while the program itself
  reports 6 ms on STO-3G water. The 15–50× wall ratios on single waters are
  exec time plus cold caches against in-process PySCF; not a target.
- **Energies:** agree to 9.8e-9 Eh per water, growing linearly with size.
  That is the basis-coefficient precision floor
  (`docs/INTEGRAL_PRECISION_FLOOR_SCOPE.md`), not a defect. The harness's
  absolute 1e-7 pass line therefore "fails" at 12 waters.
- **Setup profile** (12 H2O / 6-31G\*, `sample`, first 3 s):
  `_compute_nuclear_attraction` 77 % of samples, 90 % of that inside
  `__bzero`/`memset` called from `_os_nuclear_primitive`. Wall 6.21 s, SCF loop
  3.63 s (15 iterations).

## Where the logic lives

- `src/integrals/os.cpp`
  - `_os_nuclear_primitive` (~l. 575): the `V[VRR_DIM]³[MMAX] = {}` clear; one
    scalar `Lookup::boys(m, T)` per order.
  - `compute_external_charge_attraction_impl` (~l. 1117) and
    `_compute_electronic_potential` (~l. 1193) call it per AO pair, per point
    charge / nucleus, per primitive pair. PCM calls the second every SCF
    iteration over every cavity point, and ESP uses it too.
  - `V0[...] = {}` (~l. 669) and the nuclear-derivative `dV0[...] = {}`
    (~l. 1572, 17 KB): the same pattern in the gradient's 1e term.
- `src/integrals/fused_fock.h` — the block skip (~l. 550) and the per-build
  `qmax` table; `src/integrals/screening.cpp` `schwarz_table_hgp`, rebuilt on
  every Fock call (`rys_schwarz_matrix`, `rys.cpp` ~l. 1767).
- `src/scf/scf.cpp` `run_rhf` / `run_uhf` / `run_rohf` — full-P Fock build per
  iteration (~l. 607), phase probe (~l. 972).

## Plan, in order

### S1 — Stop zero-filling the nuclear VRR table — LANDED 2026-10-25

Dropped the initializers on `V` and `V0` in `_os_nuclear_primitive` and on
`dV0` in the nuclear-position derivative. Every read in those functions (VRR,
m = 0 extraction, `_nuclear_hrr` over `[0, l*_max]`) is of an entry written
first; the dC path's `V`/`dV` scratch already cleared only its extent. Same
arithmetic, so bitwise.

| case (4 threads) | setup before | setup after | wall before → after | PySCF wall |
|---|---|---|---|---|
| 12 H2O / 6-31G\* | 2.40–2.46 s | 0.05 s | 5.6 → 3.3 s | 6.3 s |
| 8 H2O / cc-pVDZ | 1.78–2.35 s | 0.05 s | 10.7 → 8.6–9.5 s | 3.7 s |
| 3 H2O / cc-pVTZ | 0.61–0.83 s | 0.04 s | 13.4–13.9 → 13.0–13.4 s | 5.5 s |

Gate: outputs identical to the pre-change binary (timing lines stripped) on
the 4 H2O / 6-31G\* single point, RHF PCM+ESP, PCM reaction-field ESP, ESP
points, CHELPG, the spherical RHF gradient, and RKS PBE / B3LYP PCM; core
regression suite 80/80. Not done, deliberately: batching the Boys orders (not
bitwise, and Boys was not visible in the setup profile), and a per-shell 1e
VRR. Setup is now ≲ 1 % of wall, so neither is worth it.

### S2 — Generally contracted bases — MEASURED 2026-10-25, deferred behind S3

Like-for-like full builds (PySCF `get_veff` from scratch on the hcore guess,
3 repeats; Planck median `fock_s`, which is always a full build):

| case | Planck 1t / 4t | PySCF 1t / 4t | Planck / PySCF |
|---|---|---|---|
| 4 H2O cc-pVDZ (nb 100) | 0.386 / 0.108 s | 0.101 / 0.031 s | 3.8× / 3.5× |
| 8 H2O cc-pVDZ (nb 200) | 1.85 / 0.55 s | 1.26 / 0.44 s | 1.5× / 1.25× |
| 2 H2O cc-pVTZ (nb 130) | 0.90 / 0.27 s | 0.19 / 0.061 s | 4.8× / 4.4× |
| 12 H2O 6-31G\* (nb 228) | 0.71 / 0.21 s | 1.51 / 0.61 s | 0.47× / 0.34× |

- **What the gap is:** per-quartet kernel cost on dense blocks. It shrinks
  with size as Planck's block screening wins it back. The one-thread profile
  is the ERI record's: on 4 H2O cc-pVDZ, closed forms 41 % + Boys 23 %,
  digestion 15 %, HGP VRR 13 %; on 2 H2O cc-pVTZ, HGP VRR + finish + prepare
  58 %, digestion 21 %, closed forms 13 %.
- **Why B2.2 is not the cure:** `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md` already
  measured shared-exponent sharing. A basis-chosen traversal pays at most
  ~1.29× of a cc-pVDZ build, and lanes took the HGP half (a scalar HGP family
  pass was 0.48–1.03×).
- **Incremental build is the larger lever:** PySCF's in-SCF iteration is 1.5–2×
  cheaper than its own full `get_veff` (4 H2O cc-pVDZ 0.015 vs 0.031 s; 8 H2O
  cc-pVDZ 0.249 vs 0.436 s; 2 H2O cc-pVTZ 0.038 vs 0.061 s; 12 H2O 6-31G\*
  0.417 vs 0.611 s), on every basis.

Decision: do S3 next. Revisit B2.2 only if cc-pVXZ is still the headline gap
after S3, and then as the basis-chosen traversal.

Original S2 note, kept for the record:

The cc-pVXZ gap is per-iteration Fock build. cc bases share primitive
exponents across contracted functions of one L, and Planck evaluates each
contracted shell's primitive quartets separately. The parked work already
measures the cure: the shared-exponent family pass is 1.58× on cc-pVDZ's
closed forms (B2.2), and D2a's families are ~1.7× on the cc kernels
(`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`). Re-profile one cc-pVDZ and one
cc-pVTZ iteration against PySCF first. Then build B2.2 as the basis-dependent
traversal it was parked as: on for general contractions, off for Pople, whose
family order costs 3.5–7.4 %.

Gate: as for the Fock-build work: block sweep against per-component HGP,
thread-count invariance, every SCF regression, and this benchmark.

### S3 — Incremental ΔP build with density-weighted screening — LANDED 2026-10-25

`G_n = G_{n-1} + G[P_n − P_{n−1}]` in the direct RHF, UHF and ROHF loops
(`direct_fock_rhf` / `direct_fock_uhf` in `src/scf/scf.cpp`, Cartesian and
spherical). The fused loop skips a block when
`Q_ab Q_cd · max(4|D_ab|, 4|D_cd|, |D_ac|, |D_ad|, |D_bc|, |D_bd|) < tol_eri / 10`,
with |D| the shell-block maximum over every density passed in (`screened`
in `src/integrals/fused_fock.h`, both traversals). It is off in replication
mode, where a representative block is scattered onto images whose density
differs. The Schwarz table itself stays unscreened (ERI scope invariant 8).
There is no periodic rebuild, as in PySCF. Linearity makes the update exact
however P was produced, so SOSCF, damping and level shifts need no special
handling.

**Probe first** (temporary, deleted): the fraction of Schwarz-surviving work,
weighted by components × primitives, that also survives the density test.
With full P it is 0.89–0.96 per iteration, worth 1.04–1.11×. With ΔP it
falls below half by iterations 9–12 and ends at 0.18–0.37, worth 1.25–1.6×
over a run.

**The tolerance decided the design.** At `tol_eri` (1e-10) the skipped set
changes with ΔP every iteration, the energy random-walks at 1e-8, and the
energy gate fails (12 H2O / 6-31G\*: 15 → 24 iterations, |ΔE| 1.5e-8). A full
rebuild every 5 iterations made it worse (35 iterations): the error is
inconsistency, not accumulation. Fock-build time, best of 3 at 4 threads,
against the pre-S3 binary:

| case | `tol_eri` | `tol_eri/10` (used) | `tol_eri/100` |
|---|---|---|---|
| 12 H2O 6-31G\* | 1.19× (24 iters) | 1.24× | 1.15× |
| 8 H2O cc-pVDZ | 1.46× | 1.33× | 1.21× |
| 4 H2O cc-pVDZ | 1.34× | 1.23× | 1.14× |
| 2 H2O cc-pVTZ | 1.28× | 1.19× | 1.12× |

At `/10`, |ΔE| ≤ 6e-10 against the full build, with identical iteration
counts. UHF (4 H2O⁺ / cc-pVDZ) matches to 5e-10 in the same iterations;
ROHF matches to 3e-10 but took 71 iterations against 67. Gate: core 80/80
and extended 135/139 (the four known rccsdt routing failures). The gain is
below PySCF's 1.5–2×, because PySCF screens at 1e-13 and has more to drop.

## Not in this scope

- Iteration count and DIIS: matched under equal tolerances.
- Process start-up and small-system wall clock.
- Analytic gradients: `docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md`.
- RI-JK (D3 in `docs/ERI_FOCK_BLAS_SCOPE.md`).

## Harness fixes made while measuring

`tests/benchmarks/pyscf_bench.py` parsed the SCF table with an 8-field regex
that also matched MO-energy rows. That inflated the iteration count about 3×
(45 for a 15-iteration run) and returned a per-iteration time larger than the
wall clock. It now reads `PLANCK_PHASE_TIMING` lines. It also gained
`--pyscf-conv-tol-grad` and prints iteration counts and wall times for both
codes.
