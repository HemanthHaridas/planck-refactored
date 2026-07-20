# Scale-Proving Results — Notchpeak (CHPC Utah)

Raw data: `docs/scale_notchpeak.json` (45 runs). Produced by
`tests/benchmarks/scale_bench.py`. Host `notch389`, Linux, 6-31g, engine `os`,
1 thread/rank, water chains, `hf` and `dft` (B3LYP, grid normal).

This is the fixture the HPC scope asked for and never had: before this, "it
scales" and "the memory-direct build saves memory" were claims, not
measurements. These are the measurements.

## Read `per_iter_s` from wall clock, not the old column

The `per_iter_s` in this JSON was scraped from Planck's own SCF `Time(s)`
column. Under `mpirun` that table is printed by ONE rank about ITS OWN timer, so
it does not track rank count: it is bit-identical (`1.371943 s`) at 1, 2, 4, 8
and 16 ranks in the 16-water HF sweep, while the true per-iteration time went
15.75 → 1.62 s. **Reading that column would tell you MPI does not scale, which
is false.**

Derive the real number as `wall_s / n_iters` (`wall_s` is measured outside the
launcher and cannot be fooled). `scale_bench.py` was fixed to do this; the
scraped value is retained as `per_iter_rank_s`.

## Q1 — Does the MPI direct-SCF Fock build strong-scale?

**HF: yes. DFT: no.**

| system | nb | HF best | DFT best |
|---|---|---|---|
| 8 water | 104 | **8.46×** @ 16 ranks | 1.19× @ 16 |
| 16 water | 208 | **9.70×** @ 16 ranks | 1.19× @ 8 |
| 24 water | 312 | **9.71×** @ 16 ranks | 1.04× @ 4 |
| 32 water | 416 | **10.04×** @ 16 ranks | 1.00× (serial only) |

HF holds ~9.7–10× on 16 ranks (~61% efficiency) and *improves* with system size —
the distributed memory-direct Fock build works.

DFT is pinned at ~1.2× no matter the size or rank count. That is the expected
signature of the un-fused J/K: DFT still materializes the `nb^4` ERI tensor
serially, so only the small HF-like remainder scales. **This is the strongest
existing argument for `DFT_FUSED_JK_SCOPE`.**

## Q2 — Is the memory-direct claim real, and is DFT paying for its absence?

Peak RSS per rank, 1 rank:

| system | nb | HF | DFT | ratio |
|---|---|---|---|---|
| 8 water | 104 | 15 MB | 1,288 MB | 86× |
| 12 water | 156 | 23 MB | 5,380 MB | 231× |
| 16 water | 208 | 35 MB | 15,790 MB | 446× |
| 24 water | 312 | 68 MB | 76,294 MB | 1,116× |
| 32 water | 416 | 115 MB | 235,491 MB | **2,053×** |

HF grows roughly `nb^2` (15 → 115 MB across a 4× nb increase). DFT grows like
`nb^4` and reaches **230 GB** at nb=416. The memory-direct claim is real for HF
and DFT is paying the full price for not having it.

## Q4 — Ceiling, and what binds

Two failures in the sweep, both DFT, both resource-bound:

- `24 water, 8 ranks` → **rc=137 (OOM kill)**, immediately after logging
  `Building ERI tensor for KS Coulomb term (75806.8 MB)`. A 76 GB allocation per
  the tensor build; ranks multiply the footprint, so the 8-rank case died where
  1–4 ranks survived.
- `32 water, 1 rank` → rc=1, `RKS did not converge in 100 iterations` at 235 GB
  RSS. Time/convergence-bound rather than an outright kill.

HF completed every case in the ladder, up to 32 water / nb=416.

**Ceiling: DFT is memory-bound from ~24 water (nb≈312) on this hardware; HF has
no ceiling in the tested range.**

## What this scopes

1. **`DFT_FUSED_JK_SCOPE` is the highest-value HPC work available.** It is the
   sole cause of both the DFT scaling wall (1.2×) and the DFT memory wall
   (2053× HF). Fixing it converts DFT from memory-bound at 24 water to
   HF-like scaling.
2. **HF MPI is done** for this range — 9.7–10× on 16 ranks with efficiency
   rising in system size. No load-imbalance or comm problem to chase.
3. **The regression fixture** the scope asked for: 16 water / nb=208 runs in
   ~1.6 s/iter at 16 ranks and ~16 s/iter serially — large enough to be a real
   scale gate, small enough to commit.

## Caveat

Single node generation (`notch389`), single basis (6-31g), single engine (`os`).
Q3 (where replicated serial diagonalization starts to bind) is not answered here
— the runs do not break out diagonalization time. Notchpeak is heterogeneous;
see also the libxc `-march=native` portability fix, which is a separate issue
from anything measured above.
