# ESP-derived atomic charges (CHELPG and RESP)

**Status:** scope. Landed: E0 (the ESP kernel), E1 (cross-code validation
against PySCF, which pulled the `%begin_esp` input plumbing forward from E5),
and E2 (the vdW radii, which turned out to need a comment fix and one data
correction rather than a new field — see §3). E3-E5 — grid generation and the
CHELPG/RESP fits — are not built.

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

### E3 — CHELPG

Cubic grid, 0.3 Å spacing, 2.8 Å head-space beyond the vdW surface; points inside
any atom's vdW radius discarded. The fit is least-squares with one Lagrange
multiplier for total charge — a symmetric `(natoms+1)` system, so `ldlt()`, as
already used at `geomopt.cpp:662`. Roughly 40 lines.

### E4 — RESP

Same ESP, same solver, plus the hyperbolic restraint `a·Σ(√(q²+b²) − b)` (with
`a = 0.0005`, `b = 0.1` a.u., the Bayly et al. values) solved iteratively to
self-consistency, ~25 iterations. Optional equivalent-atom constraints.

**Skip stage 2** (methyl/methylene refitting). It exists for AMBER
compatibility specifically; add it when someone needs AMBER-compatible charges,
not before.

### E5 — wiring

A `%begin_esp` block following `_parse_bsse` (`src/io/io.cpp:1799`). Call site is
`hf_driver.cpp:1284`, beside `log_population_report`, where `shellpairs` is
already in scope.

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
