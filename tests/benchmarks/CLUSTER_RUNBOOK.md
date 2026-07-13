# Running `scale_bench.py` on a cluster

What to run, in what order, and what to bring back.

The fixture answers four questions. **Q2 (memory) is already answered** — it was
measured on a laptop, see below. The cluster is needed for **Q1** (strong scaling
past one node's cores), **Q3** (where replicated diagonalization starts to bite),
and **Q4** (the real ceiling).

---

## 0. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON
cmake --build build -j
```

`-DBUILD_MPI=ON` is **required** for the MPI arm — without it there is no
`build/planck-mpi` and Q1/Q3 silently report `NO DATA` (the fixture skips rather
than fails). Confirm all three binaries exist:

```bash
ls build/hartree-fock build/planck-dft build/planck-mpi
```

### Launcher

The fixture calls `mpirun -n <k>` by default. If your site uses something else:

```bash
export MPIRUN="srun"              # SLURM
export MPIRUN_EXTRA="--oversubscribe"   # Open MPI only, if ranks > cores
```

`--oversubscribe` is **not** passed by default — it is an Open-MPI-only flag and
Intel MPI / MPICH / srun reject it as an unknown option, killing every rank.

---

## 1. Correctness gate first (cheap — do not skip)

```bash
tests/benchmarks/scale_bench.py --verify-only --build-dir build
```

Expect every rank count to print `dE = 0.00e+00` and the gate to `PASS`.

**Why this and not the timings first:** MPI bugs are *logic* bugs, not scale
bugs. A wrong reduction gives plausible-but-wrong numbers, not a crash. If this
fails, every timing below is measuring a code that computes the wrong answer,
and the whole allocation is wasted. This costs a couple of minutes on the
smallest system.

The fixture enforces this itself: it **refuses to report a single timing** if the
gate fails.

---

## 2. The real run

```bash
tests/benchmarks/scale_bench.py \
    --build-dir build \
    --sizes 8,12,16,24,32 \
    --ranks 1,2,4,8,16,32 \
    --basis 6-31g \
    --methods hf,dft \
    --threads 1 \
    --out scale.json
```

Notes on the knobs:

- **`--threads 1`.** Keep it there for the scaling sweep. Mixing OpenMP and MPI
  scaling confounds both — you cannot tell a rank-scaling problem from a
  thread-scaling one. Run a separate `--threads N --ranks 1` sweep afterwards if
  you want the OpenMP curve.
- **`--sizes`** are water-chain lengths in *molecules* (3 atoms each). With
  6-31g cartesian, `nb = 13 x nwater`, so `8,12,16,24,32` gives
  `nb = 104,156,208,312,416`. The HPC scope asked for a 30–50 atom fixture;
  `--sizes 16` is 48 atoms.
- **`--ranks`** should reach at least 16 to say anything about strong scaling.
  Two ranks proves distribution works; it does not prove it *scales*.
- **DFT will die** somewhere in that ladder — see Q2 below. That is a result,
  not a failure.

A SLURM wrapper, if useful:

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=32 --cpus-per-task=1 --time=04:00:00
module load openmpi            # or your site's MPI
export OMP_NUM_THREADS=1
tests/benchmarks/scale_bench.py --build-dir build \
    --sizes 8,12,16,24,32 --ranks 1,2,4,8,16,32 \
    --basis 6-31g --methods hf,dft --out scale.json
```

---

## 3. Memory (Q2) — already answered, but re-run to find the ceiling

```bash
tests/benchmarks/scale_bench.py --memory-only --basis 6-31g --out memory.json
```

This is 1 rank, `max_cycles=1` (peak RSS is set by the first iteration — no point
converging an SCF to measure a memory high-water mark), and a size ladder chosen
so the `nb^4` tensor hurts.

**Measured on a laptop already:**

| nb | `nb^4` tensor | HF (fused) | DFT | ratio |
|---|---|---|---|---|
| 52 | 58 MB | **9 MB** | 169 MB | 19x |
| 104 | 936 MB | **12 MB** | 1198 MB | **98x** |

HF stays flat at ~10 MB while `nb` doubles — the fused build never allocates the
tensor. DFT tracks `nb^4` almost exactly and is 98x heavier on the *same* system.

**What the cluster adds:** the ceiling. At `nb = 208` the tensor is 15 GB; at
`nb = 416` it is 240 GB. Somewhere in there DFT dies and HF keeps going. **That
crossover is the single most useful number** for scoping the DFT work — it is the
size of system the fused J/K build unblocks.

---

## What to report

Bring back **`scale.json`** and **`memory.json`**. Everything below is derivable
from them, but the four headline numbers are what matter:

### Q1 — does the MPI Fock build strong-scale?

Report: **parallel efficiency at the highest rank count**, for HF, at the largest
`nb` that ran.

- **> 70%** — Tier 1 holds, and it is done. Nothing further needed.
- **< 70%** — the bra-stripe is load-imbalanced. The triangular quartet loop
  hands rank 0 the long rows. The fix is known: flatten the triangle and stripe
  the *linear* index, exactly as `_compute_2e` already does. Report the
  efficiency-vs-rank curve so the severity is visible.

Ignore DFT's Q1 number — see below.

### Q2 — the DFT memory ceiling

Report: **the `nb` (and atom count) at which DFT dies while HF survives**, plus
the HF/DFT RSS ratio at the largest size where both ran.

That crossover sizes the DFT fused-J/K work (`docs/DFT_FUSED_JK_SCOPE.md`) and is
the number to quote when justifying it.

### Q3 — where does replicated diagonalization bite?

Report: the **non-scaling fraction for HF** at the largest `nb`, at max ranks.

Diagonalization is `O(nb^3)` and is **not** distributed — every rank does all of
it. As the Fock build distributes, diag's *share* grows. The HPC scope deferred
ScaLAPACK "until nb forces it" without ever finding out where that is; this is
where.

- **< 30%** — replicated diag holds. ScaLAPACK stays deferred.
- **> 30%** — distributed diag is the next tier, not a someday item.

**Read HF's number only.** The fixture prints DFT's too, but for DFT the
non-scaling residue is dominated by the *un-distributed J/K build*, not by diag —
Amdahl gives one number for "everything that didn't scale" and cannot tell them
apart. Reading DFT's residue as a diag verdict would send you off to build
ScaLAPACK to fix a problem that is actually the missing fused J/K. The script
says so in its own read-out. DFT's Q3 number only becomes meaningful after Step 3
of `DFT_FUSED_JK_SCOPE.md` lands.

### Q4 — the regression fixture to commit

Report: **the largest system that runs in under ~60 s/iteration serially.**

That becomes the first scale-proving case in `tests/regression_cases.json`. The
HPC scope asked for a 30–50 atom fixture and never got one; everything in the
suite today is <= 6 atoms (ethylene), which is why "it scales" has been a claim
rather than a measurement.

---

## Expected surprises

- **DFT shows ~50% efficiency at 2 ranks and a 1.00x speedup.** Not a bug, and
  not a new finding — DFT's J/K build is not distributed *at all* (it still
  contracts a dense `nb^4` tensor), so there is nothing to distribute. This is
  the predicted symptom of the gap in `DFT_FUSED_JK_SCOPE.md`, already reproduced
  on a laptop. The fixture labels it `EXPECTED` rather than `WEAK`.
- **Small systems report `too small to judge`.** Below ~50 ms per iteration the
  efficiency number is measuring process startup, not scaling. Deliberate.
- **Every run exits non-zero under `--memory-only`.** `max_cycles=1` means the
  SCF is *intentionally* not converged. Peak RSS is still valid — it is set on
  the first iteration. The fixture does not treat this as a failure (an earlier
  version did, and reported a false ceiling on every row).
