# ESP-derived atomic charges (CHELPG and RESP)

**Status:** scope. Landed: E0 (the ESP kernel), E1 (cross-code validation
against PySCF, which pulled the `%begin_esp` input plumbing forward from E5),
E2 (the vdW radii, which turned out to need a comment fix and one data
correction rather than a new field — see §3), and E3 (the CHELPG grid and the
constrained charge fit), E4 — E4a, the Connolly-shell grid, which also fixes
the rotational variance E3 measured, and E4b, the RESP restraint — and E5, the
input wiring. **E0–E5 are complete for the HF path.**

Not done, and deliberately: RESP stage-2 refitting (§E4b), and ESP charges in
`planck-dft`, which has no population analysis at all and would need either a
duplicated reporter or a shared one extracted (§E5).

Mulliken and Löwdin charges partition the density by *basis function ownership*,
which makes them basis-set dependent and physically arbitrary — a Löwdin charge
can move by several tenths of an electron between STO-3G and cc-pVTZ on the same
geometry. CHELPG and RESP instead fit charges to reproduce the molecule's own
electrostatic potential on a grid of points outside the vdW surface. That is the
observable a force field actually needs, which is why both are the standard route
into MM parameterization.

---

## 1. What this reduces to

Both methods are the same three steps:

1. generate grid points around the molecule,
2. evaluate the molecular ESP at each point,
3. solve a constrained least-squares problem for atomic charges reproducing it.

**Step 2 already existed in this tree before any of this work**, in two pieces
that had never been put together: `_compute_external_charge_attraction`
(`src/integrals/os.cpp`) builds the AO potential matrix for charges at
*arbitrary* points, and `pcm.cpp:292` contracts such a matrix against the density
to get the potential at one point. C-PCM has been doing the whole job for cavity
tesserae all along.

So the genuinely new physics here is small: a grid generator and a least-squares
solve. Most of the remaining work is *plumbing and validation*, not integrals.

---

## 2. E0 — the ESP kernel (LANDED, gated)

**The naive route does not scale, and the reason is structural.**
`_compute_external_charge_attraction` sums its entire charge list into **one** AO
matrix. That is correct for PCM's per-tessera setup, which reuses each matrix
every SCF iteration — but an ESP fit needs φ at each point **separately**.
Getting that from the existing entry means either one call per point (each
re-walking every shell pair and allocating an `nbasis²` matrix that is contracted
once and discarded) or storing one matrix per point. A CHELPG grid is ~10k
points; neither is affordable.

`ObaraSaika::_compute_electronic_potential` (`src/integrals/os.cpp`) fuses the
density contraction into the shell-pair sweep: **O(npoints × npairs) time in
O(npoints) memory, with no `nbasis²` temporary at all.**
`SCF::electrostatic_potential` (`src/populations/esp.{h,cpp}`) adds the nuclear
term and owns the sign.

**Two things that were wrong in the first plan, both found by reading the source:**

- `_os_nuclear_primitive` is file-local `static` in `os.cpp`, so the sweep had to
  live there and be exposed through `os.h`. The original plan put it in
  `src/populations/esp.cpp`, where it could not have compiled.
- `build_shellpairs` emits the **upper triangle only** (`ib >= ia`), so an
  off-diagonal pair stands for two elements of the Frobenius product `P : V` and
  the diagonal for one. This weighting is the only logic in the fusion with no
  counterpart in the matrix builder, and therefore the only place it can silently
  go wrong.

**Parallelized over points, not pairs.** Over pairs would mean accumulating into
a shared per-point vector — the cross-thread reduction that has caused
determinism jitter in the DFT grid layer. Over points, each thread owns disjoint
output entries and there is no reduction at all: the `build_coulomb_from_eri`
shape, which is bitwise thread-count-invariant.

Gated by `planck-esp-points`, whose oracle is the *existing* per-point matrix
builder — same kernel, slow route:

```
fused vs per-point matrix oracle: max |diff| = 2.220e-16
r =  60.0 Bohr: rel = 7.26e-04
r = 120.0 Bohr: rel = 3.66e-04
r = 240.0 Bohr: rel = 1.84e-04
```

The oracle check runs on a **random** symmetric density, deliberately: a
converged density is near-diagonal on STO-3G, which would barely exercise the
off-diagonal weighting that is the actual failure mode. Mutation-verified —
forcing the weight to 1.0 moves the oracle disagreement to **2.002e-01**.

The far-field residual halves as `r` doubles, which is the dipole tail, not
error.

---

## 3. The vdW radii were already there, mislabelled (RESOLVED)

**This section previously claimed Planck had only covalent radii and that
CHELPG would need a new `vdw_radius` field. That was wrong**, and it was wrong
because it trusted a comment over the data.

`ElementData::radius` was documented as `// Covalent radius (Angstrom)` but
holds **van der Waals** radii. H 1.20, C 1.70, N 1.55, O 1.52 and Na 2.27 match
PySCF's Bondi table exactly; the covalent values would be ~0.31 / 0.73 / 0.71 /
0.66 / 1.66. The comment is now corrected in both `elements.h` and
`elements.cpp`. **No new field is needed** — adding one would have duplicated
data already present.

**A retraction:** this section used to accuse `pcm.cpp:88` of claiming Bondi
radii while scaling covalent ones. That accusation was false — the PCM cavity
scales vdW radii, exactly as its comment says. The defect was the `elements.h`
comment, which had propagated the wrong belief.

**One genuine data error, fixed:** F was 1.350, which matches no published set.
Bondi 1.47, Mantina 1.47, Alvarez 1.46, Batsanov 1.45 and Hu 1.48 agree within
0.03 Å. Corrected to 1.470. Blast radius was nil — the radius has exactly one
consumer (`pcm.cpp:103`), the only fluorine-bearing registered case is gas
phase, and both PCM cases are water — so no committed result moved.

### Why Alvarez 2013 is deliberately not imported

Alvarez (*Dalton Trans.* **42**, 8617) is the newest and broadest set: 93
elements, and the first real lanthanide/actinide values derived from non-bonded
distances. It is still the wrong table to import here, for a reason that is
easy to miss.

It measures a **different quantity**. Alvarez radii mark the point of maximum
slope on the *leading edge of the van der Waals peak*, which his §5.4 states
explicitly is **not** a closest-approach cutoff — and he names Pd, Pt, Hg and U
among the elements where Bondi's values are cutoff-like. Both consumers here
want the cutoff sense: a PCM cavity wants an enclosing surface, and a CHELPG
exclusion shell wants points outside the density.

Measured against Planck's table, importing it would move **75 of 93 entries by
≥0.10 Å, essentially always upward** (Planck lower in 85 of 93, mean −0.38 Å),
inflating every cavity by ~0.4 Å and shifting every committed PCM energy.
Alvarez also has no datum for Pm, Po, At, Rn, Fr or Ra, and flags 11 entries as
rough — Bk's `[3.40]` rests on **3 atom pairs**.

Where Alvarez *is* useful: it confirms Planck is right and Bondi is stale on 5
of the 7 elements where the two disagree (F, Pd, Pt, Hg, U). Alvarez notes
Bondi's Cu, Ag and Hg radii point "even to the chemical bond territory".

**PySCF's table is not a drop-in reference either.** Its apparent disagreements
with Planck are mostly its own `unknown = 1.999999` filler, which sits at
Sc–Co, Y–Rh, the lanthanides and the actinides — 49 elements where Bondi
published nothing. Planck carries real values there.

---

## 4. Steps

### E1 — validate the ESP against PySCF (LANDED)

E0's gate proves the fused sweep agrees with *Planck's own* matrix builder. It
does **not** prove either is right; both share the same integral kernel, so a
defect in that kernel would be invisible to it. This is the same trap recorded
for the spherical transform, where every high-L gate was cross-*engine* and so
agreed while all being wrong.

PySCF's `mol.intor('int1e_grids', grids=pts)` is the same one-electron integral
and is an **independent implementation**. Measured on water/STO-3G `cart=True`,
converged RHF, against `water_rhf_esp_points_sto3g.hfinp`:

| point (Bohr) | Planck | PySCF | abs diff |
|---|---|---|---|
| `( 1.3, -0.7,  2.1)` | `0.145838926896` | `0.145838922309` | `4.6e-09` |
| `(-2.4,  1.9, -0.6)` | `-0.011474069305` | `-0.011474069301` | `4.0e-12` |
| `( 0.4,  0.3,  3.7)` | `0.040830829013` | `0.040830828077` | `9.4e-10` |
| `(-5.0, -4.0,  1.2)` | `0.003869442460` | `0.003869442320` | `1.4e-10` |
| `(60.0,  0.0,  0.0)` | `0.000005075137` | `0.000005075137` | `0.0` |

**The ESP kernel is cross-code validated.** The residual ~5e-9 is not an ESP
error: Planck and PySCF carry different roundings of the Angstrom→Bohr
conversion (Planck keeps more digits), so at identical *input* coordinates the
two codes place nuclei a few parts in 1e8 apart. That shifts the nuclear
repulsion by 6.9e-7 Eh and the SCF energy by ~1e-8, which propagates into the
potential. Feeding PySCF Planck's own Bohr coordinates makes the nuclear
repulsions agree to all ten printed digits and leaves the table above.

**So ~1e-8 is the floor for any Planck-vs-PySCF comparison on an Angstrom
input, and a tighter tolerance on this gate would be asserting the rounding,
not the physics.** Do not chase it, and do not "fix" it by tightening SCF
convergence — verified inert: `tol_energy`/`tol_density` at `1e-13` reproduce
the same energy and the same potentials bitwise.

*Trap, and it cost real time:* the first reference table here was computed on
`O 0 0 0.117176`, a geometry **no committed input uses**. The values were
correct and useless. Generate a reference from the geometry in the actual input
file, not from a plausible-looking one.

*Trap:* the density must be the converged RHF one, not the identity-scaled
density E0's gate uses. A fabricated density is only comparable across codes if
the AO ordering matches, and `cart=True` ordering is not guaranteed to. The
converged density is a physical object and sidesteps the question.

**Wiring, which the plan got wrong.** This was scoped as a C++ test binary.
That is the wrong shape: `scf.cpp` includes `post_hf/casscf/aug-hessian.h`,
`casscf/orbital.h` and both response modules, which in turn pull
`post_hf/integrals.h` and `post_hf/ri/ri_eri.h` — CASSCF and RI machinery
linked in to validate two contractions. E1 instead runs through the
**production binary** via a `%begin_esp` section (`OptionsESP` in `types.h`,
`_parse_esp` in `io.cpp`, `log_esp_report` in `hf_driver.cpp`), which needs no
new target and gates the path a user actually takes. This pulls a slice of E5
forward, deliberately.

### E2 — vdW radii (LANDED; no new field was needed)

See §3. The radii were already present and correct, just documented as
covalent. The work was a comment fix in `elements.{h,cpp}` plus one data
correction (F 1.350 → 1.470). Alvarez 2013 was evaluated against the primary
source and deliberately not imported.

### E3 — CHELPG (LANDED)

`chelpg_grid` and `fit_esp_charges` in `src/populations/esp.{h,cpp}`, driven by
`grid chelpg` in the `%begin_esp` section. Breneman & Wiberg (1990) defaults:
0.3 Å lattice spacing, 2.8 Å head-space measured **from the vdW surface** (so
the shell thickness is uniform rather than varying by element), points inside
any atom's scaled vdW radius discarded. The constraint is a Lagrange
multiplier, not a penalty, making the system symmetric-**indefinite** — so
`ldlt()` is required, not merely preferred; `llt()` would fail.

The grid is not a second code path: it fills the same point vector the explicit
`point` lines fill, so `electrostatic_potential` is called once either way.
`grid` and `point` are mutually exclusive, rejected at parse time.

Water/STO-3G, 8485 points: O −0.713342, H +0.356707 / +0.356634, sum exactly
zero, RRMS 5.62e-02. Gated by `water_rhf_chelpg_sto3g`.

**PySCF ships no CHELPG or ESP-fitting module**, so unlike E1 there is no
cross-code oracle. The substitute is an *analytically known* answer: fit a
field generated by point charges placed on the nuclei, which the model
represents exactly, and require them back. Measured 1.1e-14.

#### Rotational variance is real, and is not gated

An invariance check on an **unfittable** field fails on correct code. Measured
drift between two orientations as the lattice is refined, against charges of
order 0.9:

| spacing | drift | spacing | drift |
|---|---|---|---|
| 0.50 Å | 2.2e-01 | 0.20 Å | 6.1e-02 |
| 0.40 Å | 3.6e-01 | 0.15 Å | 7.8e-02 |
| 0.30 Å | 5.8e-02 | 0.10 Å | 2.8e-02 |

It is 3–7% throughout and **does not converge with spacing**, because refining
a cubic lattice does not make it rotationally symmetric — it resamples a field
the atom-centred model cannot represent. This is CHELPG's known rotational
variance, not a defect here. The gate therefore asserts invariance only where
it is genuinely required (an exactly-representable field, where it holds to
2e-14), and records the measurement rather than inventing a tolerance to hide
it. **Do not "fix" this by loosening the bound until an unfittable field
passes.**

It is fixed properly in **E4a**, by sampling on rotationally symmetric Connolly
shells instead of a cubic lattice — which RESP wants anyway. `use_symm true`
also removes it for symmetric molecules, but is not the remedy; see E4a for the
measurement and for why not.

### E4 — RESP, and the Connolly-shell grid that dissolves E3's rotational variance

Two pieces, deliberately in one step because the second is the fix for a defect
E3 measured and could not gate.

#### E4a — Connolly shells (`connolly_grid`) — LANDED

**The first implementation was wrong, and the way it was wrong is the point.**
Calling `fibonacci_sphere` directly and placing those directions around each
atom measured a rotational drift of **3.3e-02** — barely better than the cubic
lattice it replaced. The scope's premise ("a sphere is rotationally symmetric")
is true of the *surface* and false of a *fixed discrete sampling* of it: the
point pattern stays pinned to the lab axes, so rotating the molecule slides
points across each atom's surface exactly as the lattice does.

Measured against sampling density, which is what proved it was not simply a
resolution problem:

| points/shell | 50 | 200 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|
| drift | 1.0e-1 | 3.3e-2 | 8.1e-3 | 8.7e-3 | 1.6e-2 |

Non-monotone, with a floor. The decisive experiment was rotating the direction
set *with* the molecule: **3.3e-02 → 7.1e-15**, independent of density.

The fix is therefore to build the direction set in a **molecule-derived frame**.
`connolly_grid` receives no rotation, so the frame comes from the geometry
itself: Gram-Schmidt from the centroid, seeded by the farthest atom, with ties
broken by atom index — all rotation-invariant choices, so the frame rotates with
the molecule by construction. This also sidesteps the degenerate-eigenvector
discontinuity that ruled out principal axes.

Measured: **6.9e-15** against CHELPG's 5.8e-02 on the identical fixture.
Water/STO-3G over 1016 points gives O −0.709219, H +0.354587 / +0.354632.
Gated by `water_rhf_connolly_sto3g` and by check 7 of `planck-esp-points`, which
runs **both** grids and requires the CHELPG arm to keep failing, so a Connolly
pass cannot become vacuous. Mutation-verified: removing the frame rotation
returns the drift to 3.3e-02 and the check goes red.

`fibonacci_sphere` was promoted from an anonymous namespace in `pcm.cpp` to
`src/base/sphere.h`, and PCM now consumes it — one generator, not two that can
drift apart.

Merz-Kollman-style nested spherical shells: for each atom, points on spheres of
radius `scale_k × r_vdW` for a few scale factors (conventionally 1.4, 1.6, 1.8,
2.0), discarding any point that falls inside another atom's scaled sphere. This
is what RESP conventionally samples on, so it is needed for E4 regardless.

**It also removes the rotational variance E3 recorded**, and removes it
structurally rather than mitigating it: a sphere is rotationally symmetric, so
rotating the molecule rotates the sample set with it instead of resampling a
lattice through a fixed cubic grid. E3's 3–7 % drift has nowhere to come from.

Reuse `fibonacci_sphere` for the point placement. It currently sits in an
anonymous namespace in `src/solvation/pcm.cpp`, so **promote it to a shared
header rather than copying it** — one generator with a parameter, not two
implementations that can drift apart.

**What `use_symm` revealed, and why it is not the fix.** `detectSymmetry` calls
`msymAlignAxes` and writes a standardized frame into `_standard`
(`symmetry.cpp:170`), which is the frame `chelpg_grid` builds in. Measured on a
rigidly rotated C2v water:

| | point group | grid pts | O charge | RRMS |
|---|---|---|---|---|
| canonical, symm off | — | 8485 | −0.71334229 | 5.6198e-02 |
| rotated, **symm on** | C2v | **8485** | **−0.71334223** | 5.6198e-02 |
| rotated, symm off | — | 9232 | −0.71350913 | 5.4665e-02 |

Symmetry-on recovers the canonical answer to 6e-8 on a bit-identical grid.
That is real, but it is **not** a fix for three reasons: it only helps molecules
that *have* symmetry (C1 falls through to `set_standard_from_bohr(bohr_coords)`,
the untouched input frame — and C1 is most of what anyone fits charges for); it
makes charges depend silently on an SCF/geometry keyword; and libmsym reorders
atoms, so the two H charges swap position in the output. Do not present
`use_symm true` as the remedy.

Two options were considered and rejected. A principal-axis frame is cheap and
works for C1, but degenerate or near-degenerate moments of inertia make the
eigenvectors arbitrary, so axes swap discontinuously along a geometry scan —
trading a 3–7 % variance for a discontinuity. Orientation averaging (fit over
several rotated lattices) is trivially correct but N× the cost and still only
mitigates.

#### E4b — the restraint (`fit_resp_charges`) — LANDED, not yet wired

The hyperbolic restraint `a·Σ(√(q²+b²) − b)`, Bayly et al. stage-1 defaults
`a = 0.0005`, `b = 0.1` a.u. Its derivative contributes a **diagonal**
`a/√(q_k²+b²)` to the normal matrix, which depends on `q`, so the same
`(natoms+1)` LDLᵀ system is simply re-solved until the charges stop moving —
no new solver, ~7 iterations on water. Hydrogens are exempt by default
(`exempt_hydrogen`), since the restraint exists to tame buried heavy atoms and
hydrogens always sit where the data is good.

Equivalence groups force chosen atoms to share a charge. They are **hard rows
in the same Lagrange block** as the total charge, not penalties, so grouped
atoms agree to 2.2e-16 rather than approximately.

**Stage 2 (methyl/methylene refitting) is deliberately not implemented** — it
exists for AMBER compatibility specifically, and `equivalence_groups` already
covers the general case of forcing atoms to share a charge.

**Still open: there is no input keyword for RESP.** `fit_resp_charges` is
implemented and gated but has no driver call site, so `grid connolly` currently
runs the *unrestrained* fit. Wiring it is E5's job, and until then RESP is
reachable only from C++.

Gated by check 8 of `planck-esp-points`, which asserts three things:

- **reduction** — at `strength = 0` it reproduces `fit_esp_charges` to
  `0.0e+00`, pinning that the restraint is the only difference between the two
  paths;
- **the trade** — the restraint must both shrink the charges and *worsen* the
  RRMS, measured 4.7e-03 and 1.6e-06;
- **equivalence** — grouped atoms agree exactly, against a 1.34 gap when
  unconstrained (the non-vacuity check: if the free fit already tied them, the
  constraint would be untested).

**The effect-size thresholds are deliberate.** Bare inequalities (`|q|` smaller,
RRMS larger) *do* catch an inert restraint, so they were not vacuous — but the
real effect is 0.5 % in `|q|` and an RRMS rise below the fifth significant
figure, so a bare inequality sits one fixture change away from passing on
rounding noise. The check therefore requires a minimum effect and asserts the
**iteration count > 2** as an independent signal: an inert restraint leaves the
normal matrix unchanged between passes and exits at exactly 2. All three fire on
a mutation that deletes the restraint diagonal.

#### Gating

The check E3 could not have: **fitted charges must be invariant under rigid
motion with `use_symm false`**, on a field the model cannot represent exactly —
the fixture where E3's lattice demonstrably fails. Verify it goes red against a
`chelpg_grid` fit before trusting it, so the gate is known to be measuring the
grid and not the fit. E3's two fixture-too-easy failures (see the test's own
comments) are the trap to avoid repeating here.

Keep `chelpg_grid`: CHELPG charges are a published, cited quantity and people
reproducing them need the cubic lattice, variance and all. The default for
RESP should be the Connolly grid.

### E5 — wiring (LANDED)

The `%begin_esp` block landed early, pulled forward by E1. What E5 added is the
RESP half: `resp`, `resp_strength`, `resp_tightness`, `resp_exempt_hydrogen`
and `equivalent` (1-based atom indices, stored 0-based, matching the `bsse`
`fragment` convention). Range-checked in `parse_input` where `natoms` is known.

The driver dispatches between `fit_esp_charges` and `fit_resp_charges`, both of
which yield the same `ESPChargeFit`, so the report below the dispatch is shared
and there is still one path. RESP additionally prints its iteration count, and
warns if the restraint did not converge — otherwise an unconverged fit is
indistinguishable from a converged one in the charges alone.

Measured on water/STO-3G, one Connolly grid of 1012 points:

| | O | H | H | RRMS | iters |
|---|---|---|---|---|---|
| unrestrained | −0.708804 | 0.354386 | 0.354418 | 3.576e-02 | — |
| RESP, no `equivalent` | −0.700909 | 0.350440 | 0.350469 | 3.746e-02 | 7 |
| RESP + `equivalent 2 3` | −0.700909 | 0.350454 | 0.350454 | 3.746e-02 | 7 |
| RESP at Bayly 0.0005 | −0.708410 | 0.354205 | 0.354205 | 3.577e-02 | 5 |

Gated by `water_rhf_resp_sto3g`. **The fixture uses `resp_strength 0.01`, not
the Bayly stage-1 default**, because at 0.0005 the restraint moves these charges
by ~4e-4 — below what the 8-decimal print distinguishes from the unrestrained
fit — so a case pinning printed charges would assert nothing about the
restraint. The production default is unchanged.

#### A stale-expectation failure worth recording

`water_rhf_connolly_sto3g` was committed asserting 1016 grid points and
O −0.709218. Committed code produces **1012** and **−0.708804**. The numbers
were captured from a `hartree-fock` binary built *before* E4a's molecular-frame
fix and registered without rebuilding — the fix changes which directions are
generated and so which points survive burial, which is exactly a 1016 → 1012
shift.

**Its mutation test passed and did not catch this**, which is the transferable
part: a mutation test proves an assertion is *sensitive to change*, not that its
*value is current*. Perturbing a stale expectation still turns the case red.
Capture regression values from a binary you have just rebuilt, and prefer
verifying the build is current over assuming it.

**DFT is a separate decision.** `planck-dft` has **no population analysis at
all** — it includes `populations/multipole.h` and nothing else. Wiring ESP there
means either duplicating the reporter or extracting a shared one. Recommend
HF-only first.

---

## 5. Gating

`metric_close` on the fit RRMS plus `contains` on the charge lines. Emit as
`^\s*ESP Fit RRMS\s+([-+0-9Ee\.]+)` to match the existing `METRIC_PATTERNS`
shape (`tests/run_regressions.py:20`).

**Non-vacuity is the thing to get right here.** A charge-fitting gate passes
trivially if it asserts only the charges: the total-charge constraint *forces*
them to sum correctly regardless of whether the fit reproduces anything. Assert
the **RRMS**, and verify it goes red when the grid is perturbed.

---

## 6. Traps

- **Do not reuse PCM's precompute-and-store pattern.** It is right for PCM (each
  matrix is reused every SCF iteration) and wrong here (each point is visited
  once). This is what E0 exists to avoid.
- **Do not add a `sym_ops` overload to the fused sweep.** Symmetry folding
  reconstructs AO-pair orbits, which is a property of the *matrix*; a contraction
  already reduced against a full unfolded density has nothing to fold.
- **`ElementData::radius` is vdW, and not every vdW set means the same thing.**
  See §3. The field was mislabelled `// Covalent radius` for years, so check the
  data before trusting a comment about it. And when comparing against a
  published set, check what that set *measures*: Alvarez radii mark a
  distribution slope, not a closest-approach cutoff, so they are not
  interchangeable with the cutoff-sense radii a grid exclusion shell wants.
- **Buried atoms fit badly and that is not a bug.** An atom with no grid points
  in its neighbourhood (a carbon in a bulky group) has almost no leverage on the
  ESP, so its fitted charge is poorly determined. This is the known motivation
  for RESP's restraint, not a defect to chase.
- **Time E1 before committing to a default grid density.** The fused sweep is
  `O(npoints × npairs)`; ~10k points on a real basis is the expensive part of the
  whole feature, and the CHELPG default spacing was chosen in 1990 for molecules
  much smaller than what people will run this on.
- **An exactly-representable fixture cannot gate anything about the grid.**
  This bit twice in E3, in two guises. Asserting the charge constraint on a
  field of nuclear-centred point charges passed even with the Lagrange row
  deleted, because the unconstrained solution already sums correctly there.
  Asserting rotational invariance on the same fixture passed against two
  deliberately broken grids, because an exact fit is insensitive to which
  points sample it. **A property can only be gated on a fixture where it
  binds** — for a constraint or a grid, that means a field the atom-centred
  model cannot represent. Mutation-test every such check before trusting it;
  both of these were caught only that way.
- **Do not invent a tolerance to make a failing check pass.** When the
  rewritten invariance check failed on correct code, the fix was to measure the
  effect (3–7 %, non-convergent in spacing), discover it was CHELPG's real
  behaviour, and narrow the claim — not to widen the bound until it went green.
