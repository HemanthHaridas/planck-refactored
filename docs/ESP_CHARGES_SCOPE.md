# ESP-derived atomic charges (CHELPG and RESP)

**Status:** scope. E0 (the ESP kernel) is landed and gated; E1-E5 are not built.

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

## 3. The real gap: no vdW radii

`src/lookup/elements.h` carries exactly one radius:

```cpp
double radius;  // Covalent radius (Angstrom)
```

CHELPG needs **van der Waals** radii for its exclusion shell, and RESP needs them
for its Connolly shells. Covalent radii are roughly half the size and would place
grid points **inside** the electron density, where the fit is meaningless.

**Incidental finding, pre-existing and out of scope here:** `pcm.cpp:88` claims
its cavity uses "the standard Bondi-radius PCM choice" while actually scaling
`element->radius`, i.e. covalent radii. Either the comment or the cavity is
wrong. Worth a separate look; do **not** fold it into this work, because changing
it moves every committed PCM energy.

Reference values are available for cross-checking at
`pyscf.data.radii.VDW` (104 elements, **in Bohr** — note the unit differs from
Planck's Angstrom convention).

---

## 4. Steps

### E1 — validate the ESP against PySCF

E0's gate proves the fused sweep agrees with *Planck's own* matrix builder. It
does **not** prove either is right; both share the same integral kernel, so a
defect in that kernel would be invisible to it. This is the same trap recorded
for the spherical transform, where every high-L gate was cross-*engine* and so
agreed while all being wrong.

PySCF's `mol.intor('int1e_grids', grids=pts)` is the same one-electron integral
and is an **independent implementation**. Reference values, water/STO-3G
`cart=True`, converged RHF (`E = -74.9629507133`, `tr(PS) = 10.0000000000`):

| point (Bohr) | φ (a.u.) |
|---|---|
| `( 1.3, -0.7,  2.1)` | `-0.085603882576` |
| `(-2.4,  1.9, -0.6)` | `+0.007140177405` |
| `( 0.4,  0.3,  3.7)` | `-0.048251686268` |
| `(-5.0, -4.0,  1.2)` | `-0.003635334371` |
| `(60.0,  0.0,  0.0)` | `-0.000004193544` |

Expect agreement to ~1e-10. A committed script under `tests/pyscf/`, registered
in `cases.json` with `"kind": "esp"`.

*Trap:* the density must be the converged RHF one, not the identity-scaled
density E0's gate uses. A fabricated density is only comparable across codes if
the AO ordering matches, and `cart=True` ordering is not guaranteed to. The
converged density is a physical object and sidesteps the question.

### E2 — vdW radii

Add a `vdw_radius` field to `ElementData`. Bondi (1964) covers H–Rn; ~30 lines of
data. Gate: covered by E4's fit, which cannot be right with wrong radii.

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
- **Covalent ≠ vdW.** See §3. Grid points placed with covalent radii sit inside
  the density and the fit is meaningless rather than merely inaccurate.
- **Buried atoms fit badly and that is not a bug.** An atom with no grid points
  in its neighbourhood (a carbon in a bulky group) has almost no leverage on the
  ESP, so its fitted charge is poorly determined. This is the known motivation
  for RESP's restraint, not a defect to chase.
- **Time E1 before committing to a default grid density.** The fused sweep is
  `O(npoints × npairs)`; ~10k points on a real basis is the expensive part of the
  whole feature, and the CHELPG default spacing was chosen in 1990 for molecules
  much smaller than what people will run this on.
