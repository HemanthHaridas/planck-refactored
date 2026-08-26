# Why did a correct generated CC kernel produce a ~4x wrong energy?

Because the build was configured with `PLANCK_CC_SPIN_ADAPT=OFF` — an emit that
`CMakeLists.txt` itself documented as defective, kept as the **default** for
byte-compatibility with a historical result.

Nothing was wrong with the kernel. The generated rank-3 CCSDT residual matches
the hand-written path to all ten digits on three independent systems once the
flag is ON. This answers how a documented-defective default survived, why it
read convincingly as a kernel defect for an entire investigation, and what
changed so it cannot recur.

The sibling question — *a correct kernel driven by a solver expecting a different
amplitude representation* — is answered in
`docs/CCGEN_RANK3_KERNEL_AND_SOLVER.md`. Both are the same shape of failure:
**a correct kernel, wrong context, converging to a self-consistent wrong answer.**

## The measurements

With `-DPLANCK_CC_SPIN_ADAPT=ON -DPLANCK_CC_ARBITRARY_LOWER_RANKS=ON`:

| case | `SPIN_ADAPT=OFF` | `ON` | hand-written reference |
|---|---|---|---|
| **Be generated rank-3** | −0.0139349127 | **−0.0517702884** | −0.0517702884 — all ten digits |
| **Be rank-4** (warm-started) | 100-iter cap, failed | **−0.0517746458**, 6 steps | −0.0517746458 |
| **LiH generated rank-3** | −0.0093755944, failed | **−0.0204594700**, 62 steps | −0.0204594700 — all ten digits |
| **CH4 generated rank-3** | −0.0565650696, failed | **−0.0791116827**, 35 steps | −0.0791116825 (2e-10) |

CH4 total `-39.8058445098` against the PySCF reference `-39.8058445240` —
**1.4e-08**. Rank 4 warm-starting from the rank-3 solve converges in **6** steps
against 12 cold, i.e. the warm-start doing exactly what it is for.

## What the flag does

`CMakeLists.txt` (the comment predates this investigation and was correct):

> Emit genuine spatial (restricted) RCC kernels via the R1.0 spin-adaptation
> instead of raw spin-orbital algebra bound to spatial storage. **Without this
> the generated energy carries the `0.25*t2*oovv` defect and the correlation
> energy is ~4x wrong (the Be-CCSDTQ defect).**

Measured factor on Be CCSDTQ: **3.63** (`-0.0142553636` vs `-0.0517746458`).

It was OFF *"for byte-compatibility with the historical (defective) emit until
Gap B lands"*. **Gap B had landed** — the rank-4 multi-sector `t4` solve is
wired and gated end to end by `be_rccsdtq_sto3g`
(`docs/CCGEN_CCSDTQ_MULTISECTOR.md`) — so the condition was satisfied and the
comment was stale. The default is **ON as of 2026-08-26**.

## Why it read as a kernel defect for so long

Four things independently pointed the wrong way. Each is worth recognizing again.

**1. The failure is self-consistent, not a crash.** On Be the generated rank-3
solve converged in 12 iterations to `rms(res) = 1.9e-12` at an energy 3.78e-02
from correct. A tight residual at a wrong energy means the kernel is driving a
*different* equation to self-consistency — which is exactly what a spin-orbital
residual bound to spatial storage is. This looks like a kernel bug, and reads as
one at every level of detail short of the build configuration.

**2. No regression case pinned the flag.** `be_rccsdtq_sto3g` carried
`requires_build_option: null`, passed in a default build and **failed** in the
investigation build, with nothing declaring the dependency. So the suite could
be green and the binary under test could still be the defective emit.

**3. The gates that existed asserted the wrong path.** `ch4_rccsdt_sto3g`
asserts `kernels=hand-optimized` — the hand-written path — so it was green
throughout while never executing the generated kernel it was added to protect.
A gate on the generated path (`ch4_rccsdt_generated_sto3g`) was added mid-
investigation and pinned the *observed broken* behaviour, which was the right
instinct but pinned a build artifact as if it were a kernel property.

**4. Two builds differed in three flags, not one.** The investigation build was
`ARBITRARY_LOWER_RANKS=ON`, and that flag was assumed to be the only difference.
It also carried `SPIN_ADAPT=OFF` and `INCLUDE_INTERMEDIATES=OFF`. **Diff the
whole cache, not the flag you changed.**

## The measurement that cracked it

Compare the **CCSDTQ bundle's shared manifolds against the CCSDT bundle's**.

Rank 4 was a verified-correct generated path, and its singles/doubles/triples
kernels solve the *same* equations as CCSDT's. If the CCSDT bundle were
specially broken, they would differ. Dumping both at identical amplitudes and
running `cmp`:

```
rank 1: BYTE-IDENTICAL
rank 2: BYTE-IDENTICAL
rank 3: BYTE-IDENTICAL
```

That excluded "the rank-3 kernel is wrong" in one step, and moved the search to
what the two *builds* differed in. The general lesson: **when one component
looks broken and a sibling that shares its code looks fine, compare them
directly before investigating either.**

The earlier ladder had instead compared C++ against PySCF and against ccgen-in-
Python, found real disagreement, and localised it to "the C++ layer" — correct
but not actionable, because the C++ layer was compiling different algebra than
assumed.

## Two false trails, both plausible

**The warm-start seed.** With `ARBITRARY_LOWER_RANKS=ON`, rank 4 warm-starts
from rank 3, so a broken rank-3 solve poisoning rank 4 explained the rank-4
failure exactly. It was wrong: with `cc_warm_start .false.` — no seed at all,
confirmed by the absent `Warm-started rank 4 from converged rank 3` log line —
rank 4 still failed in that build and still succeeded in the default one.
**Note the `warm_start=on/off` log line reports the FLAG, not whether seeding
happened**; in a default build `generated_floor` is 4, so `rank - 1 >=
generated_floor` is false and no seed occurs regardless of the flag.

**MO phase freedom.** A genuine confound, found and fixed on the way. A
converged SCF fixes each MO only up to a sign, chosen independently by PySCF and
Planck, so `<ij|ab>` picks up `p_i p_j p_a p_b` and nothing is comparable
elementwise. Caught by probing at *zero* amplitudes, where the doubles residual
is the bare driver and no CC algebra runs: every element ratio was exactly ±1,
following an exact rule over all 24 non-zero elements —

```
sign = +1  iff  (i == j) == (a == b)
```

— and one choice, `(1,-1,1,1,1,-1)`, reconciled the two to **1.234e-08**.
`solve_phases` / `apply_phases` in `python/ccgen/tests/dump_cc_fixture.py` handle
it. **The Frobenius norm is phase-invariant**, so it is the right quantity for
any cross-implementation residual comparison.

## Findings that outlived the investigation

**There is no rank-2 generated RCC kernel.** `make_generated_rcc_kernels`
(`generated_kernel_registry.cpp:118`) floors at 3 (4 without
`ARBITRARY_LOWER_RANKS`), and rank 3 dispatches to
`make_generated_ccsdt_kernels()`. So any rank-1/rank-2 residual observed through
the generated path is the **singles/doubles of the CCSDT bundle**, and any
"rank 2 works" datapoint is the *hand-written* CCSD. This is deliberate
(`io.cpp:675`: a generated rank-2 RCC path would have no consumer), not an
asymmetry to fix.

**Be/STO-3G cannot validate a rank-3 kernel.** Both t1 and t3 sit at machine
zero (|t1| 2.3e-15, |t3| 7.4e-19), and the triples residual is 7.4e-18 *even at
converged t1/t2 with t3 forced to zero* — the manifold is never driven. Every
rank-3 signal lives in mixed-spin blocks. **LiH/STO-3G at 1.6 Å** is the
replacement: closed-shell, `no=4 nv=8`, t3 live at 8.2e-04.

**The spatial representative block is not all-alpha.** `spin.py:577` places a
single beta on the **last bra and last ket slot** — rank 2 `abab`, rank 3
`aabaab`. Extracting `aaaaaa` yields an empty t3 and a silently vacuous test.

**Amplitude layout is transposed between the two sides.** ccgen is
`(vir...,occ...)`; C++ `rank_dims` (`amplitudes.cpp:54`) is `(occ...,virt...)`.

## What changed

| | |
|---|---|
| `CMakeLists.txt` | `PLANCK_CC_SPIN_ADAPT` **default ON**; comment records why and how to reproduce the historical emit |
| `run_regressions.py` | `requires_build_option` accepts a **list** — these cases depend on more than one flag |
| `be_rccsdtq_sto3g` | requires `PLANCK_CC_SPIN_ADAPT` |
| `ch4_rccsdt_generated_sto3g` | requires both flags; **inverted** to assert `rccsdt_total_energy == -39.8058445240 ± 1e-06` and exit 0 |
| `src/post_hf/cc/rccgen.cpp` | `PLANCK_CC_FIXTURE_DIR` probe: seed amplitudes, evaluate residuals **once**, report per-rank max\|R\| with index, stop. Inert when unset |
| `python/ccgen/tests/dump_cc_fixture.py` | fixture dumper; refuses to write a non-fixed-point fixture; handles layout, spatial block, and MO phases |
| `python/ccgen/tests/test_iterate_amps_fixed_point.py` | the fixture is a fixed point; Be's t1/t3 are inert |
| `python/ccgen/tests/test_spatial_residual_vs_pyscf.py` | pins the **reference** side independently of C++ |

## If a generated kernel looks wrong again

1. **Check the build cache first** — `grep '^PLANCK_CC' <build>/CMakeCache.txt`,
   and diff it against a known-good tree. Do not assume the flag you set is the
   only one that differs.
2. **Compare against a sibling bundle** that shares the kernels before
   investigating the kernel.
3. **Check the case actually runs the path you think** — `kernels=hand-optimized`
   in a log means the hand-written path ran.
4. Only then reach for the probe. It, the LiH fixture, and the two Python gates
   are in place.

**Existing build trees keep their cached `OFF`.** The new default applies to
fresh configures; reconfigure with `-DPLANCK_CC_SPIN_ADAPT=ON` or delete the
cache.

**Open, minor:** `be_rccsdtq_sto3g` asserts `-14.4036551081` while both builds
produce `-14.4036550465` — a 6.2e-08 gap that passes the 1e-07 tolerance. It is
pre-existing and unrelated to the flag (both configurations give the same
value), but the gate is one tightening away from failing.

---

Status (what is landed, what is open) lives in `vault/Status/Completion.md` and
`vault/Status/Open Work.md`, which are canonical.
