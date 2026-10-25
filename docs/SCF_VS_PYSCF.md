# Where does a Planck RHF lose to PySCF, and what closed it?

Canonical status lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`. This file answers one question about direct-SCF
RHF on water chains: where is the time lost relative to PySCF, and which
levers paid? It follows `docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, which had
already made the Fock build fast and closed.

## Short answer

Not where it looked, twice over.

- **Iteration counts** are identical once the convergence tolerances match.
- **Two real costs were found and removed:** a 246 KB zero-fill per
  primitive in the nuclear-attraction routine, which dominated setup (S1),
  and the lack of an incremental Fock build (S3).
- **The "kernel gap" was mostly a benchmark artifact.** PySCF keeps the ERI
  tensor in memory whenever it fits (`RHF.get_jk`, regardless of
  `direct_scf`), so every PySCF number this comparison first used was
  conventional SCF, not direct. Direct against direct, a full Fock build is
  level or faster in Planck on 6-31G\* (0.27–0.95×), and 1.5–3.7× slower on
  cc-pVXZ, not 4–16×. What remains is analysed in
  `docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md`: Planck's high-L kernels are slower
  than libcint's, and its low-L kernels are faster.
- **The basis file was a cause, and is fixed.** PySCF's cc-pVXZ files drop
  outer primitives that also appear as uncontracted shells (the optimized
  general contraction); the GBS files Planck reads keep them, 28 against 24
  primitives per water in cc-pVDZ. Planck now drops them at read time:
  the Fock build is 2.1–2.3× faster on cc-pVDZ and 1.5–1.8× on cc-pVTZ,
  energies unchanged (`docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md`, K1).
- **General contraction is a small lever for Planck.** It was first declined
  on an in-core test (§4). Remeasured direct, it is worth 1.6× to libcint but
  about 1.1× to Planck, whose per-primitive work is already cheaper.
- **What remains is high-L kernel cost on cc-pVTZ**, rescoped in
  `docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md` (K2).

## Measured: end to end, after both fixes (PySCF in-core)

**These PySCF numbers are conventional SCF**, measured before the in-core
storage was found. They compare Planck's direct SCF with PySCF's stored ERI
tensor, which is cheap per iteration at these sizes. A direct-vs-direct
end-to-end rerun with the fixed harness is pending; the full-build
comparison in §4 is already direct against direct.

M3, 4 threads, `tests/benchmarks/pyscf_bench.py` (before its in-core fix).
Water chain, RHF, Cartesian, no symmetry, hcore guess, Planck direct SCF
with `engine auto` (`build-full`), PySCF 2.13.0 (NumPy on Accelerate). Both
converge to `conv_tol 1e-9` and `conv_tol_grad 1e-7`, the latter matching
Planck's 1e-7 DIIS-error and density gates.

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

Energies agreed to about 9.6e-9 Eh per water, growing linearly with size,
attributed at the time to the basis-coefficient precision floor
(`docs/INTEGRAL_PRECISION_FLOOR_SCOPE.md`). These runs predate Boys core
v1.2.1 (§5), which fixed integral errors of up to 7e-6 on high-L quartets,
so part of that difference may have been the Boys defect. Re-measure before
quoting the floor.

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

At `/10`, with the bench's `tol_density 1e-7`, |ΔE| ≤ 6e-10 against the
full build and the iteration counts are unchanged. UHF matches to 5e-10 in
the same iterations. ROHF matches to 3e-10 but took 71 iterations against 67.

**The screen is switched off for the SCF tail.** At the default
`tol_density` (1e-10) the `/10` screen still random-walks: the DIIS error
stalls at 1–3e-10, which is the convergence gate's own floor, so a run
converges only when the noise lets it. 2 H2O / cc-pVTZ took 88 iterations
against 18, and planck-mpi `-n 2` landed 6.7e-9 from serial on 4 H2O /
cc-pVDZ (`tol_energy 1e-10`), failing `water4_rhf_ccpvdz_auto_direct_mpi_smoke`.
Once the DIIS error falls below 1e-8 (`IncrementalFock::kOffBelow`,
`src/scf/scf.cpp`), the loop drops the screen for the rest of the SCF and
does one full rebuild, which discards the accumulated drift. The threshold
is fixed rather than `tol_density`: the default `tol_density` sits below
the stall, so a switch keyed to it would never fire. A run converged above
1e-8 keeps the screen throughout. Fock-build time at four threads, best of
3, against the same build with the screen never on:

| case | bench tol: screen always on | bench tol: switched | default tol: always on | default tol: switched |
|---|---|---|---|---|
| 12 H2O 6-31G\* | 1.28× | 1.24× | 1.51× (27 iters, 21) | 1.15× |
| 8 H2O cc-pVDZ | 1.21× | 1.20× | 1.20× (53 iters, 21) | 1.15× |
| 4 H2O cc-pVDZ | 1.21× | 1.17× | 1.60× | 1.28× |
| 2 H2O cc-pVTZ | 1.39× | 1.26× | 0.53× (88 iters, 18) | 1.05× |

With the switch, every energy equals the unscreened build's to all printed
digits and the iteration counts match it to within one. Serial and MPI
agree bitwise. The gain is below PySCF's own: its
in-SCF iteration is 1.5–2× cheaper than its full build, because it screens
at 1e-13 and so has more to drop.

### 4. Direct against direct, the gap is small and basis-dependent (S2)

**The trap.** PySCF's `RHF.get_jk` stores the 8-fold ERI tensor whenever
it fits under `max_memory` (4000 MB by default), whatever `direct_scf` says.
A warm `get_veff` is then a dense contraction, not a direct build. On
4 H2O / cc-pVDZ at one thread: in-core 0.030 s, forced direct 0.205 s. An
earlier version of this section compared Planck's direct build with PySCF's
in-core one and reported a 4–16× "kernel gap".

Full Fock builds compared like for like: Planck's iteration-1 build (still a
full build after S3) against PySCF's `get_veff` **forced direct**
(`_is_mem_enough` returns False, `_eri` stays `None`), warm, with
`direct_scf_tol` 1e-10, on the hcore guess:

| case | nb | Planck 1t / 4t | PySCF-direct 1t / 4t | ratio 1t / 4t |
|---|---|---|---|---|
| 4 H2O 6-31G\* | 76 | 0.067 / 0.022 s | 0.088 / 0.081 s | 0.76× / 0.27× |
| 12 H2O 6-31G\* | 228 | 0.736 / 0.248 s | 0.902 / 0.260 s | 0.82× / 0.95× |
| 4 H2O cc-pVDZ | 100 | 0.389 / 0.136 s | 0.172 / 0.093 s | 2.3× / 1.5× |
| 8 H2O cc-pVDZ | 200 | 2.00 / 0.567 s | 0.783 / 0.249 s | 2.6× / 2.3× |
| 1 H2O cc-pVTZ | 65 | 0.094 / 0.029 s | 0.025 / 0.025 s | 3.7× / 1.1× |
| 2 H2O cc-pVTZ | 130 | 1.00 / 0.284 s | 0.280 / 0.101 s | 3.6× / 2.8× |
| 3 H2O cc-pVTZ | 195 | 3.15 / 0.860 s | 0.852 / 0.243 s | 3.7× / 3.5× |

- **Where it lies:** `docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md` (K0) timed both
  codes' kernels on the same quartets. Planck is faster than libcint at total
  L ≤ 3 and slower above, up to 4–5× at L ≥ 8. On 4 H2O cc-pVDZ, Planck's
  kernels are faster overall (0.53×), so the remaining build gap there is
  work: PySCF evaluates fewer quartets.
- **Profile:** the one-thread profile matches the ERI record. On 4 H2O
  cc-pVDZ: closed forms 41 % + Boys 23 %, digestion 15 %, HGP VRR 13 %. On
  2 H2O cc-pVTZ: HGP VRR + finish + prepare 58 %, digestion 21 %, closed
  forms 13 %.
- **General contraction: the test that declined it was not direct.** cc
  bases share primitive exponents across the contracted functions of one L,
  and Planck evaluates each contracted shell's primitive quartets
  separately. The test split PySCF's own general contractions into segmented
  shells and saw `get_veff` unchanged (0.94–1.02×), and B2.2 was declined on
  it. That result almost certainly came from PySCF's in-core path, where
  `get_veff` contracts a stored tensor and cannot see the contraction
  scheme. Forced direct, PySCF is 1.7–2.1× slower on Planck's GBS files than
  on its own, though that comparison mixes general contraction with the
  redundant-primitive difference, now removed on Planck's side
  (`docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md`, K1). After K1, splitting PySCF's
  general contractions costs it 1.6–1.7× on cc-pVDZ, but Planck's family
  probe gains only 1.00× over a whole cc-pVDZ sweep (1.28× on the closed
  forms, a loss on the HGP half), because Planck's per-primitive work is
  already cheaper than libcint's. B2.2 stays a small lever (G in the scope).

### 5. A Boys-function precision defect, found by the kernel comparison

K0 checks each Planck integral against libcint before timing it. After
rescaling for the two normalization conventions, the difference was up to
6.6e-8 (6-31G\*), 4.0e-8 (cc-pVDZ) and 7.2e-6 (cc-pVTZ). It was largest in
pure d and f quartets and grew with L. Planck's Rys engine, which does not use
the Boys function, agreed with libcint to 2e-16, while OS, HGP and the block
kernel agreed with each other and not with libcint. That isolated it to the
Boys function: the out-of-tree core returned high-order values with too few
significant digits over part of its argument range.

Fixed in Boys core **v1.2.1**: max relative error 1.45e-13 against mpmath
over orders 0–59, checked on macOS and Linux builds before release. After the
fix the K0 differences are 3.4e-14 / 2.6e-14 / 7.2e-13, and OS, HGP, the block
kernel and Rys agree to 1e-16. v1.2.1 needs a v1.2.1 license. It is not
bitwise against v1.2.0, so checks pinned to exact integral values move.

## What invariants matter

1. **Compare at matched convergence criteria.** PySCF's default
   `conv_tol_grad` is √`conv_tol`. Leaving it there makes PySCF look 20–30 %
   faster by stopping early.
2. **Make PySCF run direct, and check that it did.** `RHF.get_jk` stores
   the ERI tensor whenever it fits, so `direct_scf = True` alone gives
   conventional SCF. Override `_is_mem_enough` and assert `mf._eri is None`;
   lowering `max_memory` is not enough, since a small molecule's tensor fits
   under any budget.
3. **Time PySCF warm.** The first `get_veff` on a molecule builds its
   screening object (and, in-core, the ERI tensor).
4. **Check values before timing them.** The kernel comparison found the
   Boys defect only because it compared every integral first.
5. **Never zero a worst-case-sized buffer in a per-primitive routine.** Clear
   the extent used, or show every read is preceded by a write.
6. **Screening error must be consistent from iteration to iteration.** An
   error that is the same every iteration only shifts the converged answer.
   One that changes each iteration is noise the convergence test sees. That
   is why the density test runs ten times tighter than `tol_eri`.
7. **The Schwarz table is never density-screened**
   (`docs/ERI_SHELL_QUARTET_FOCK_BUILD.md`, invariant 8).

## Not done

- **The direct-vs-direct end-to-end table.** Rerun
  `tests/benchmarks/pyscf_bench.py` (now forced direct) to replace the
  in-core table above, and re-check the energy agreement with Boys v1.2.1.
- **The remaining cc-pVXZ gap:** after the redundant primitives were
  dropped, a full direct build is 0.79–1.21× PySCF's on cc-pVDZ and
  1.0–2.75× on cc-pVTZ (the 1.5–3.7× in §4 predates it). On cc-pVTZ it is
  high-L kernel cost (K2); on cc-pVDZ it is general contraction, which is
  worth 1.6× to libcint but about 1.1× to Planck. Rescoped in
  `docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md`.
- **DFT:** the KS Coulomb/exchange direct builds (`_compute_2e_j_direct` /
  `_compute_2e_k_direct`) do not yet build incrementally.
- **Gradients:** `docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md`.
- **RI-JK:** D3 in `docs/ERI_FOCK_BLAS_SCOPE.md`.

## Harness fixes made while measuring

`tests/benchmarks/pyscf_bench.py` parsed the SCF table with an 8-field regex
that also matched MO-energy rows. That inflated the iteration count about 3×
(45 for a 15-iteration run) and returned a per-iteration time larger than the
wall clock. It now reads `PLANCK_PHASE_TIMING` lines. It also gained
`--pyscf-conv-tol-grad` (default 1e-7, matching Planck) and prints
iteration counts and wall times for both codes. Later it was fixed to run
PySCF genuinely direct (`_is_mem_enough` overridden, `_eri is None`
asserted); its docstring had claimed both codes ran direct all along. It
also gained `--spherical`. It still times PySCF's whole `kernel()`,
including the first build, so PySCF's per-iteration column reads slightly
high on small systems.
