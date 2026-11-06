# Can wPBEh (HSE) be made smooth without changing what it computes?

Research scope, opened 2026-11-06. Nothing has landed. Intended as
self-contained work that can be published. The engineering response inside
Planck (gate around the defect) is the separate
`docs/WPBEH_S_SCALING_SCOPE.md`; this scope asks the scientific question
behind it.

## The question

The Heyd–Scuseria–Ernzerhof (HSE) screened hybrids use the short-range
Ernzerhof–Perdew exchange hole model (wPBEh). As implemented in libxc, used
by PySCF, Planck and many other codes, their energy density is not a smooth
function of the density. Three things follow:

1. **Characterize:** where is the libxc wPBEh energy density discontinuous,
   or discontinuous in its first and second derivatives? How large is each
   seam, and how often do real molecules put grid weight on one?
2. **Consequences:** which quantities does each seam corrupt (finite-difference
   gradients, semi-numerical frequencies, finite-difference XC kernels,
   geometry optimization, SCF convergence), and by how much?
3. **Repair:** is there a construction that is C² (or better) everywhere and
   agrees with standard HSE06 to well below chemical relevance, so that
   published HSE06 numbers stay valid?

## What is already known (measured in Planck, 2026-11-06)

- **The s = 1 seam.** libxc rescales the reduced gradient with Henderson's
  first scaling, `s_scaling_2` (`maple/scaling.mpl`): s for s < 1, and
  s − ln(1 + exp(s − smax)) for s ≥ 1, with smax = 8.572844. At s = 1 the
  second branch is 1 − 5.15e-4, so the scaled s jumps. The energy density
  jumps by ~1e-4 relative (+3.3e-6 at ρ = 0.1, +7.2e-5 at ρ = 1 a.u.), and
  ∂e/∂σ jumps too. PBE shows no such feature.
- **It is deliberate upstream.** libxc commit `173000399` (2020-01-27):
  "Changed again s_scaling_2 to follow the old convention (even if that leads
  to a discontinuous function)"; the source comment says this scaling is
  "apparently used by Gaussian". Henderson's smooth second scaling,
  `s_scaling_3` = s − (1 − e^(−s)) ln(1 + e^(s − smax)), is in libxc but no
  functional uses it.
- **It corrupts energy finite differences.** C1 H₂O₂ / 6-31G*, HSE06, 150 × 590
  grid, δ = 5e-4 Bohr: 7 of 354,000 grid points cross s = 1 and carry
  2.796e-8 Eh, the whole FD error (FD 2.8e-5 Ha/Bohr off the analytic gradient;
  4.4e-8 after removing them). The miss varies erratically with δ (1.3e-9 at
  1e-4, 2.8e-5 at 5e-4, 1.2e-5 at 2e-3). PySCF reproduces it to 1e-12
  (`tests/pyscf/hse06_s_scaling_jump.py`).
- **It corrupted a finite-difference TDDFT kernel.** Planck built the TDDFT
  XC kernel as a central difference of V_xc (step 1e-5). In spherical
  water/6-31G* HSE06, points within one step of s = 1 shifted two Casida roots
  by 1.1e-4 and 4.4e-4 eV (with 1e-6 steps, agreement with PySCF's analytic
  kernel was 1e-7 eV). Cartesian water happened to have no such point. Fixed in
  Planck by switching to the analytic f_xc; any code with an FD kernel is
  exposed.
- **Analytic quantities are unaffected at the reference point.** Planck's
  HSE06 energies and analytic gradients match PySCF to ≤ 1.2e-10.

## Other seams to check

The same libxc file (`maple/gga_exc/gga_x_wpbeh.mpl`) has more branches, each
a candidate seam:

| branch | where | form |
|---|---|---|
| `EGscut = 0.08` | small s | Taylor polynomial below, closed form above |
| `wcutoff = 14` | ν = ω/k_F large (low density, density tails) | `term1_largew` vs the series in ω |
| `expfcutoff = 700` | exponent guard | exp/erfc handling (`erfcx` since `b2babe13b`) |
| `s > 15` | very large s | scaled s pinned to smax |
| `m_max(1e-15, …)` | s → 0 | floor |

Which of these are C⁰, C¹ or C² is not known.

The HJS hole (`gga_x_hjs.mpl`, Henderson–Janesko–Scuseria) has no scaling
and no piecewise branches, which makes it the natural control.

## Phases

**R0. Provenance and literature (desk work).**
- Henderson et al., JCP 128, 194105 (2008), and its appendix (the scaling
  functions); the original HSE papers (2003, 2006 erratum) and Ernzerhof–Perdew
  (1998): which scaling, if any, each prescribes.
- What other codes do. Gaussian is "apparently" the first scaling (unverified);
  also check VASP, Q-Chem, ORCA, CP2K, FHI-aims and NWChem. Use published
  energies or source where available, and do not assert behaviour that was
  not checked.
- Search libxc issues and merge requests, and the literature, for prior
  reports of the discontinuity. If it is already published, this scope
  shrinks to R2–R4.

**R1. Characterize every seam, at high precision.**
- Re-implement F_x^wPBEh(s, ν) in mpmath from the papers, independent of
  libxc, and check it against libxc away from the seams. This independent
  reference is what makes the rest credible.
- For each branch: jump in F, ∂F/∂s, ∂²F/∂s² (and the ν derivatives), as a
  function of (ρ, s).
- Map where molecules actually sit: across a test set (G2-1 or W4-11
  geometries; a few transition-metal complexes), the grid weight and density
  near each seam at converged HSE06 densities.

**R2. Consequences, measured.** For each seam that matters in R1:
- energy-FD vs analytic gradients over the test set, as a function of δ;
- semi-numerical frequencies (FD of analytic gradients; the gradient is
  discontinuous too) vs analytic Hessians;
- geometry optimization near convergence (stalls or oscillation);
- finite-difference TDDFT kernels (the Planck case above, generalized);
- SCF convergence at tight thresholds.
Planck and PySCF provide two independent drivers on the same libxc, which
separates functional effects from code effects.

**R3. Smooth constructions.** Candidates, each with derivatives through
third order (TDDFT kernels and analytic Hessians need f_xc and k_xc):
- (a) Henderson's second scaling, `s_scaling_3` (C^∞, changes values away
  from s = 1 too);
- (b) a C² (or C^∞) blend of the first scaling over a narrow window around
  s = 1 that is exactly `s_scaling_2` outside the window, so HSE06 values are
  untouched almost everywhere;
- (c) smooth replacements for any other seam R1 flags (e.g. a matched
  expansion at `EGscut`);
- (d) as a control, HJS-PBE (already smooth).
Derive symbolically (Maple, as libxc does, or SymPy) and generate code.

**R4. Validation.**
- Energetics vs standard HSE06: atomization energies, barrier heights,
  ionization potentials on a standard set. The success criterion for (b) is
  differences far below 0.01 kcal/mol; (a) is reported, not required to pass.
- Smoothness: FD vs analytic gradients agreeing to the FD truncation error at
  all δ; semi-numerical vs analytic frequencies; TDDFT FD vs analytic kernels.
- Cross-code: the same smooth functional in Planck and in PySCF (patched libxc)
  agreeing to 1e-10.

**R5. Implementation path.** Preferred: an upstream libxc functional (new IDs
such as HSE06-smooth, or an external parameter selecting the scaling), so
every libxc client gets it. Planck then exposes it as a named functional.
Fallback: a fetch-time patch in Planck (as was done for libecpint).

**R6. Write-up.** Target a short communication (JCP or JCTC). Contents: the
seam inventory (R1), measured consequences (R2), the smooth construction and
its negligible energetic cost (R3–R4), and availability (R5).

## Kill and shrink criteria

- If R0 finds the discontinuity and its consequences already published, keep
  only the parts that add something (likely R3–R5).
- If R1 shows the s = 1 seam is the only one and R2 shows its effects are
  confined to FD validation, the result is a short note. Planck's engineering
  scope (gate around it) is then enough for Planck itself.
- If no construction in R3 stays below 0.01 kcal/mol from HSE06 while being
  C², report that trade-off instead of a replacement.

## Not in scope

Changing default HSE06 in Planck or libxc; periodic (plane-wave) HSE; other
range-separated families beyond the HJS control; meta-GGA scalings.
