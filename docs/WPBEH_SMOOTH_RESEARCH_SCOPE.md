# Can wPBEh (HSE) be made smooth without changing what it computes?

Research scope, opened 2026-11-06, rewritten 2026-11-08 after R0, the
first part of R1, and the first R2 result. Nothing has landed in Planck. The engineering response
inside Planck (gate around the defect) is the separate
`docs/WPBEH_S_SCALING_SCOPE.md`; this scope asks the scientific question
behind it.

## Where this stands

The question has partly answered itself. The s = 1 discontinuity in libxc's
HSE06, and a smooth replacement for it, were published in 2009 by the people
who wrote the scaling (Henderson, Izmaylov, Scalmani, Scuseria, JCP 131,
044108). They called the smooth form "the recommended version of HSE" and
shipped it in Gaussian 09 as `HSEH1PBE`. libxc, and so PySCF, Planck and
Psi4, still implement the version they superseded.

What is not published, as far as found, and is what remains here:

- what the seams do to finite-difference gradients, frequencies and kernels
  (R2; the energy-FD gradient result is below, the rest is open);
- the seam inventory beyond s = 1 (R1, measured below);
- how far libxc's HSE06 sits from the recommended version, and from the
  other codes' "HSE06" (measured below);
- getting the recommended version into libxc (R5).

## Recommendation (2026-11-08; the first R2 result agrees)

1. **Planck: change nothing now.** Keep libxc HSE06 as `hse06`. It matches
   PySCF and the rest of the libxc ecosystem to ≤ 1.2e-10, and that
   agreement is what Planck's HSE06 gates rest on. Keep avoiding energy FD
   as a validation tool for HSE06; analytic gradients are unaffected.
2. **The smooth functional to adopt is the 2009 eq. 9 scaling, not a new
   construction.** It is published, it is what Gaussian's recommended HSE
   uses, and libxc already contains it (`s_scaling_3`, unused). Its cost
   relative to libxc HSE06 is 0.006–0.03 kcal/mol in atomization energies.
   That is smaller than the spread between codes that all call their
   functional HSE06 (up to 0.15 kcal/mol for CP2K's convention, 0.6–6 for
   Quantum ESPRESSO's). Keeping libxc's own values to 1e-3 kcal/mol is
   therefore a weak goal.
3. **Demote R3(b), the narrow blend, to a fallback.** It works (≤ 2e-3
   kcal/mol from libxc HSE06), but it is a new, unpublished functional whose
   only virtue is matching libxc's superseded form. Use it only if libxc
   declines eq. 9, and then only with a window h ≳ 1e-2: at h = 1e-3 its
   FD error reaches 4.4e-6 Ha/Bohr (R2).
4. **Upstream (R5): ask libxc for a separately named HSE using eq. 9.** Do
   not change libxc's HSE06 IDs, since published numbers depend on them. A
   complete 2009 implementation also needs the numerical integral for the
   one non-analytic hole term, which removes the ν = 14 seam. Without it, a
   seam remains at negligible energy in the density tails.

On H₂O₂, eq. 9 already reaches the FD noise floor without the integral
(R2). The integral becomes required only if a wider set shows the ν = 14
seam above that floor.

## The seams

libxc (`maple/gga_exc/gga_x_wpbeh.mpl`, `maple/scaling.mpl`), libxc 7.0.0
as shipped with PySCF 2.13.0. Planck pins `4bd0e1e3` (5.0.0-799, 2023),
with the same branch constants. Relative jumps across each branch point,
one-sided at ±1e-10, ρ = 1, ω = 0.11 unless stated. **Every seam is
discontinuous in value; none is even C⁰.**

| seam | F | v_ρ | v_σ | f_ρρ | f_σσ |
|---|---|---|---|---|---|
| `EGscut` s = 0.08 | 6.8e-9 | 3.1e-9 | 4.6e-6 | 2.5e-7 | 7.0e-4 |
| `s_scaling_2` s = 1 | 1.2e-4 | 4.9e-5 | 3.5e-4 | 1.9e-3 | 5.4e-4 |
| `s_scaling_2` s = 15 | 2.4e-5 | 3.8e-4 | v_σ → 0 above | 2.2e-2 | f_σσ → 0 above |
| `wcutoff` ν = 14 (s = 0.5–5) | 0.5–2.3e-4 | 1e-5–4e-4 | 0.8–11 % | 1–5 % | — |

- **s = 1.** `s_scaling_2` is s below 1 and s − ln(1 + e^(s − smax)) above
  (smax = 8.572844). The second branch starts at 1 − 5.15e-4, so s̄ jumps.
  The jump grows weakly as ρ falls (1.7e-4 at ρ = 1e-3). This is eq. 8 of
  the 2009 paper. libxc restored it deliberately (commit `173000399`,
  2020: "even if that leads to a discontinuous function") and labels it
  "apparently used by Gaussian", which was true before 2009.
- **s = 15.** The middle branch reaches smax − 1.6e-3, not smax, before
  s̄ is pinned to smax. Above it v_σ = 0.
- **ν = 14.** Switches between two erfc approximations (exponent 1.4559 →
  2). For ω = 0.11 that is ρ ≈ 1.6e-8. The 2009 numerical integral removes
  it.
- **`EGscut`.** Small-s Taylor branch with inexact constants. Already found
  and fixed upstream: libxc issue #627 (from PySCF issue #3312, where a
  regenerated libxc gave 6.8e-4 Ha/Bohr HSE06 FD failures), MR !806 in
  libxc **7.1.0** (exact constants, cutoff moved to s = 0.13). PySCF moved
  to libxc 7.1.2 (PR #3336). Planck's pin lacks it; a libxc bump inherits
  the fix and moves HSE06 values at s < 0.13 slightly.
- `expfcutoff = 700` (exponent guard) and `m_max(1e-15, …)` were not
  measured; neither is reached at chemical densities.

Two checks make these numbers trustworthy:

- **Independent reference.** It integrates the Ernzerhof–Perdew hole with
  the HSE fits H(s), F(s) against the exact erfc in mpmath (30 digits),
  with G(s) from the hole normalization. It shares only the hole model with
  libxc: not the closed form, the erfc fit, the Taylor branch or the large-ν
  branch.
  - At ν = 0 it agrees with libxc to ≤ 7e-16 (s ≥ 0.3).
  - At ν > 0 they differ by up to 1.2e-6 relative (4e-7 at HSE06's
    ν ≈ 0.036). That is HSE's erfc fit, part of the functional's
    definition.
  - F(0, 0) = 1 − 4.8e-9: the published hole parameters normalize the LDA
    limit only to 8 digits.
- **Attribution.** The reference with `s_scaling_2` applied reproduces the
  s = 1 and s = 15 jumps (−1.18e-4, +2.42e-5), and is continuous to 4e-13
  at s = 0.08. Those two are entirely the scaling.

## Where molecules sit

HSE06/6-31G*, PySCF grid level 3, eight closed-shell molecules (H₂O, H₂O₂,
NH₃, HF, N₂, CO, C₂H₄, C₆H₆). For each seam: the points within 0.1 % of it,
and an upper bound on the energy step if all of them crossed
(|ΔF_rel · w ρ e_x|).

| seam | points in band | energy bound / Eh |
|---|---|---|
| s = 0.08 | 0 | 0 |
| s = 1 | 9–144 | 1.8e-7 – 3.0e-6 |
| s = 15 | 0–8 | ≤ 6e-12 |
| ν = 14 | 0–8 | ≤ 2e-18 |

21–28 % of all grid points have s > 15 (the tails), carrying 1e-5 – 3e-5 Eh
of exchange, all with v_σ = 0. 13–15 % have ν > 14, carrying ≤ 3e-12 Eh.

For energies and energy FD, s = 1 is the only seam that matters. The others
are real discontinuities at negligible energy, but their large jumps in v_σ,
f_σσ and f_ρρ could still matter for kernels and response in diffuse tails
(R2).

## Consequences measured so far

- **Energy FD gradients** (Planck, 2026-11-06). C1 H₂O₂/6-31G*, HSE06,
  150 × 590 grid, δ = 5e-4 Bohr: 7 of 354,000 grid points cross s = 1 and
  carry 2.796e-8 Eh. That is the whole FD error: 2.8e-5 Ha/Bohr off the
  analytic gradient, 4.4e-8 after removing those points. The miss varies
  erratically with δ: 1.3e-9 at 1e-4, 2.8e-5 at 5e-4, 1.2e-5 at 2e-3.
  PySCF reproduces it to 1e-12 (`tests/pyscf/hse06_s_scaling_jump.py`).
- **FD TDDFT kernel** (Planck, fixed). A central difference of V_xc
  (step 1e-5) shifted two spherical water/6-31G* HSE06 Casida roots by
  1.1e-4 and 4.4e-4 eV. Planck now uses the analytic f_xc. Any code with an
  FD kernel is exposed.
- **Analytic quantities are unaffected** at the reference point: Planck and
  PySCF HSE06 energies and analytic gradients agree to ≤ 1.2e-10.

## What "HSE06" means in each code

From source, current master, 2026-11-08 (read, not run):

| code | full-range exchange | scaling | branches |
|---|---|---|---|
| libxc (PySCF, Planck, Psi4) | hole at ω = 0 | eq. 8 (`s_scaling_2`), jumps at 1 and 15 | `EGscut`, `wcutoff` |
| CP2K (`xc_xwpbe.F`; regtests `XWPBE SCALE_X0 1.0`) | hole at ω = 0 | Heyd's original (`s_scaling_1`): s below 8.3, smax − 18.796/s² above; continuous, kinked at 8.3 | `gcutoff = 0.08`, `wcutoff = 14` |
| Quantum ESPRESSO (`pbexsr`) | real PBE (`pbex`) | Heyd's original | `EGscut`, `wcutoff` |
| NWChem (`xc_xwpbe.F` → `hse08fx.F`) | — | none: the HJS hole (Henderson–Janesko–Scuseria 2008) | none; smooth |
| Gaussian 09 `HSEH1PBE` (closed) | per the 2009 paper | eq. 9 | numerical integral, no `wcutoff` |

- QE differentiates with respect to the scaled s above 8.3 without the
  chain-rule factor, so its potential there is not the derivative of its
  energy.
- Not checked, and nothing is asserted about them: later Gaussian releases,
  VASP, Q-Chem, ORCA, FHI-aims.
- The 2009 paper does not say whether s̄ applies to the full-range piece as
  well; libxc and CP2K apply it to both.

**Energetic spread.** Atomization energy differences from libxc HSE06,
kcal/mol, evaluated non-self-consistently at the converged libxc HSE06
density (6-31G*, grid level 4, atoms UKS). Each variant is evaluated through
libxc by feeding it the gradient whose `s_scaling_2` image is the variant's
s̄. That is exact for s̄ < smax − 1.6e-3 (only the s > ~15 tails are
clamped), and was checked against the mpmath reference to 4e-7.

| mol | CP2K | QE | 2009 eq. 9 | blend h = 1e-2 | blend h = 1e-3 |
|---|---|---|---|---|---|
| H₂O | −0.0075 | 1.044 | 0.0081 | 0.0004 | −0.0001 |
| H₂O₂ | −0.0182 | 1.771 | 0.0126 | 0.0002 | −0.0002 |
| NH₃ | −0.0057 | 1.204 | 0.0114 | 0.0007 | 0.0000 |
| HF | −0.0036 | 0.615 | 0.0056 | 0.0002 | −0.0001 |
| N₂ | 0.0016 | 1.033 | 0.0202 | −0.0004 | −0.0000 |
| CO | −0.0148 | 1.302 | 0.0064 | −0.0008 | −0.0009 |
| C₂H₄ | −0.0448 | 2.184 | −0.0086 | 0.0009 | −0.0000 |
| C₆H₆ | −0.1510 | 6.113 | −0.0324 | 0.0016 | −0.0000 |

- **QE is about 1 kcal/mol away (6 for benzene).** libxc's hole at ω = 0
  differs from real PBE exchange by up to 4e-3 in F_x over s = 0.1–6,
  because the hole model reproduces PBE only approximately. This is a
  definitional spread, far larger than any seam.
- **Eq. 9 moves atomization energies by 0.006–0.03 kcal/mol**, the paper's
  "small but non-negligible".
- **The blend** (C² smootherstep between the two `s_scaling_2` branches
  over [1 − h, 1 + h], identical to libxc outside) moves them by ≤ 1.6e-3 at
  h = 1e-2 and ≤ 9e-4 at h = 1e-3. Its curvature in s̄(s) is about
  5e-4/h²: 5 at 1e-2, 500 at 1e-3.
- Caveats: one basis, eight molecules, non-self-consistent. The eq. 9
  column omits the exact erfc integral (≤ 1.2e-6 relative in F) and applies
  eq. 9 to both pieces.

## Remaining phases

**R2. Consequences (first result 2026-11-08).** `tests/pyscf/wpbeh_r2_fd.py`
runs each variant as a PySCF custom functional whose potentials are the
exact chain rule through s̄. Validation:

- in native mode it reproduces PySCF's built-in HSE06 to 1e-13 Eh and
  1e-14 Ha/Bohr;
- every variant's v_ρ and v_σ match pointwise FD to FD precision.

C1 H₂O₂/6-31G*, 150 × 590 grid (PySCF default pruning), libxc 7.0.0,
SCF `conv_tol` 1e-12. Maximum |FD − analytic| over all 12 Cartesian
components, fourth-order central differences, Ha/Bohr:

| variant | δ = 1e-4 | 2.5e-4 | 5e-4 | 1e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|---|---|
| libxc HSE06 | 1.33e-4 | 5.31e-5 | 2.62e-5 | 2.06e-5 | 5.45e-6 | 8.61e-6 |
| 2009 eq. 9 | 4.32e-9 | 1.92e-9 | 7.72e-10 | 6.59e-10 | 6.34e-10 | 4.99e-10 |
| blend h = 1e-2 | 3.40e-9 | 1.50e-9 | 9.30e-10 | 9.46e-10 | 1.36e-9 | 6.67e-9 |
| blend h = 1e-3 | 2.96e-9 | 1.19e-8 | 4.63e-8 | 3.15e-7 | 1.50e-6 | 4.39e-6 |

- **libxc HSE06 cannot be finite-differenced.** The error is 5e-6 to
  1.3e-4 and largest at the smallest δ: a step in E, divided by δ.
- **Eq. 9 sits at the noise floor at every δ.** That floor is set by SCF
  convergence over δ, about 4e-9 at δ = 1e-4. Neither the ν = 14 seam nor
  the emulation's s > 15 clamp is visible at this level.
- **The blend works at h = 1e-2 and fails at h = 1e-3.** At 1e-3 the
  error grows with δ to 4.4e-6, because the window is narrower than the
  step and the blend's curvature (~500) dominates the truncation error. At
  1e-2 it only begins to rise at δ = 4e-3.
- This confirms recommendation 2 (eq. 9) and keeps the blend as a fallback
  only at h ≳ 1e-2.

After that:

- the census set;
- semi-numerical frequencies against the analytic Hessian;
- FD TDDFT kernels;
- geometry optimization near convergence;
- SCF convergence at tight thresholds.

Planck joins only if it gets a variant hook. That choice belongs to the
Planck scope, not here.

**R1, remainder.**

- A larger set (G2-1 or W4-11) with a level-5 grid and a few
  transition-metal complexes.
- Redo the spread table there self-consistently.
- Barrier heights and ionization potentials as well as atomization
  energies.

**R5 prototype: eq. 9 inside libxc 7.1.2, generated with SymPy
(2026-11-08).** libxc 7.1 generates its kernels from SymPy definitions
under `python/` (`scripts/sympy2c/compile.py`), so no Maple is needed.

- **Pipeline control.** Regenerating stock `gga_x_wpbeh` through the
  default `sp.diff` path reproduces the committed C byte for byte
  (184 s). The `--ad` engine emits different and 12× larger code (5.8 MB),
  so it is no control.
- **Two new functionals** (IDs are placeholders, free in 7.1.2):
  - `GGA_X_WPBEH09` (790): `python/gga_exc/gga_x_wpbeh09.py` includes
    `gga_x_wpbeh` verbatim and replaces only the scaling with eq. 9.
  - `HYB_GGA_XC_HSE06_09` (791): HSE06's mix (1 × ω = 0, −β × ω, PBE
    correlation; β = 0.25, ω = 0.11) on the new exchange.
- **Overflow-safe eq. 9.** libxc's unused `s_scaling_3` overflows
  `exp(s − smax)` in the density tails and loses every digit to
  cancellation before that. For s > smax, eq. 9 is written as
  smax − log1p(e^(smax − s)) + e^(−s) log(1 + e^(s − smax)), the same
  analytic function, so the branch point is C^∞.
- **Generated code.** 465 KB, 78 lines different from stock wPBEh after
  renaming, and order 2 like stock wPBEh.
- **Source patch:** `tests/pyscf/libxc-7.1.2-hse06_09.patch` (9 files, 177
  added lines; applies cleanly to the 7.1.2 tag). The generated C is not in
  it: regenerate it with
  `scripts/sympy2c/compile.py gga_x_wpbeh09 --maxorder 2` (about 3 min).
  The build and validation steps are in the docstring of
  `tests/pyscf/wpbeh09_libxc_validate.py`.

Validation (pylibxc on the patched build, and PySCF 2.15 driving it as a
custom functional):

| check | result |
|---|---|
| ω = 0 against the mpmath hole reference at s̄ = eq. 9 (s = 0.05–30) | ≤ 3.9e-10; the residual is 7.1's exactly derived E, Fc1, Fc2 against the published values the reference uses |
| ω = 0.11 against stock wPBEh at the preimage gradient, same library | 2.0e-14 |
| jumps across s = 0.08, 0.13, 1, smax, 15 (±1e-10) | ≤ 3e-9 in value and derivatives, i.e. the local slope; stock: 1.2e-4 at s = 1 |
| v_ρ, v_σ against FD | median 5e-9; worst cases are FD round-off where v_σ ≈ 1e-10 (s ≈ 24) |
| tails, s = 20 to 1e100 | finite, constant from s = 100 |
| AD build against `sp.diff` build, 20k points | values ≤ 8e-11, first derivatives ≤ 6e-10, second ≤ 3e-5, confined to the inherited s < 0.13 series branch |
| H₂O₂/6-31G* HSE06-09 against the R2 emulation | ΔE −3.2e-9 Eh, ΔG ≤ 6.1e-10 Ha/Bohr, most likely the emulation's s > 15 clamp (not confirmed) |
| H₂O₂ energy FD against the analytic gradient, δ = 1e-4 / 5e-4 / 2e-3 | 8.0e-9 / 6.6e-10 / 6.2e-10 Ha/Bohr, the SCF noise floor |

The hybrid reports (ω, α, β) = (0.11, 0, 0.25), as HSE06 does. Not done:
the spin-polarized path, a libxc regression-reference file, and the
numerical integral that would remove the ν = 14 branch.

**R4. Validation of the adopted variant.**

- **Energetics:** report differences from libxc HSE06; there is no pass
  threshold for eq. 9.
- **Smoothness:** FD within truncation error at all δ.
- **Cross-code:** Planck and PySCF agree to 1e-10 on the same patched libxc.
- **Gaussian:** if a Gaussian `HSEH1PBE` number is obtainable, compare
  against it, since it is the published reference implementation.

**R5. Implementation path.**

- Preferred: an upstream libxc functional with new IDs, eq. 9 plus the
  2009 numerical integral, leaving HSE06 untouched. The prototype above is
  the first half and is in a state to propose as a merge request.
- Report the libxc-versus-2009 gap and the stale "apparently used by
  Gaussian" comment to the maintainers either way.
- Planck then exposes the new functional by name.
- Fallback: a fetch-time libxc patch in Planck, as was done for libecpint.

**R6. Write-up.** Now a short note rather than a new-functional paper. It
covers:

- the seam inventory;
- the measured consequences;
- the observation that libxc-based codes run the superseded HSE;
- the inter-code spread of "HSE06";
- availability.

## Kill and shrink criteria

- Fired (R0): the s = 1 discontinuity and its smooth fix are published.
  The scope keeps only what adds to that.
- If R2 shows the effects are confined to energy-FD validation, the note
  shrinks to inventory, spread and availability. Planck's engineering scope
  is then enough for Planck.
- If libxc adopts eq. 9 upstream, R3(b) is dropped entirely.

## Not in scope

- Changing the default HSE06 in Planck or libxc.
- Periodic (plane-wave) HSE.
- Range-separated families other than HSE and the HJS control.
- Meta-GGA scalings.

## Scripts

Run both with `tests/pyscf/.venv/bin/python` (PySCF 2.13.0, mpmath).

- `tests/pyscf/wpbeh_seams.py`: the mpmath reference and the seam
  inventory; `--census` gives the molecule table and `--variants` the
  spread table.
- `tests/pyscf/wpbeh_r2_fd.py`: R2; `--check` validates the custom
  functional.
- `tests/pyscf/libxc-7.1.2-hse06_09.patch` and
  `tests/pyscf/wpbeh09_libxc_validate.py`: the R5 prototype and its
  validation (`points` or `molecule`). These need a patched libxc 7.1.2
  build and PySCF ≥ 2.14, not the gate venv; see the script's docstring.

## Sources

- Henderson, Izmaylov, Scalmani, Scuseria, JCP 131, 044108 (2009), Sec. II
  (eqs. 8–10) and the appendix.
- Henderson, Janesko, Scuseria, JCP 128, 194105 (2008), appendix (the
  computational reduced gradient σ(s), libxc's `s_scaling_4`).
- libxc issue #627, MRs !804 and !806, the 7.1.0 and 7.1.1 release notes,
  and commit `173000399`.
- PySCF issue #3312 and PR #3336.
- Source files: CP2K `src/xc/xc_xwpbe.F` and
  `tests/QS/regtest-hfx-wfn-fitting/CH4-HSE06.inp`; QE
  `XClib/qe_funct_exch_gga.f90` and `XClib/qe_drivers_gga.f90`; NWChem
  `src/nwdft/xc/xc_xwpbe.F` and `src/nwdft/xc/hse08fx.F`.
