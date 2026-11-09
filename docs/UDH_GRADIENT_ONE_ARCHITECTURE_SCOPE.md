# UKS double-hybrid gradient: one derivative architecture

Scope for ongoing work. When it lands, rewrite this file as the answer to its
question, or fold the answer into `docs/DH_IMPLEMENTATION.md` and delete it.

**Question.** The production UKS double-hybrid gradient fails. How does it reach
a working state with one derivative architecture shared with RKS, and no dense
path beside the streamed one?

## The defect

Every UKS double-hybrid gradient stops with:

```
UDH driver: invalid correction coordinate layout
```

`d72742df` (2026-10-12) changed `build_dh_eq33_derivative_integrals`
(`src/dft/driver.cpp`) to return only the two streamed callbacks. The UKS path
was not moved with it:

1. `build_udh_gradient_correction` passes that bundle to
   `build_udh_non_xc_gradient`, which reads only the dense fields
   `hamiltonian_ao`, `overlap_ao` and `eri_ao`.
2. Those fields are now empty, so the coordinate count is `nc = 0` and every
   contraction is a zero-length vector.
3. `src/dft/udh_driver.cpp:126` then rejects `non_xc.total.size() != 3 * natoms`.

The last recorded validation is `039bd2df` (2026-10-09). The suite missed the
break because no regression case runs this path, and the
`planck-udh-molecular-runner` ctest entry runs only the audit script's Python
unit tests. The molecular audit itself is run by hand.

The fixture that exposes it is
`tests/inputs/regression/dft/h2o2_cation_uks_b2plyp_v4_gradient_sto3g.hfinp`.

## Decision

**Move UKS onto the streamed contraction, and delete the dense route for both
spins.** Restoring the dense builder is ruled out: it would keep two derivative
architectures, which is what caused this break.

RKS and UKS differ only in where the density coefficients come from, so one
engine can take both:

| Term | RKS | UKS |
|---|---|---|
| Coulomb, Eq. 34 / 46 | `D(μν) P(κτ)` | `D_tot(μν) P_tot(κτ)` |
| Exchange | one term, factor `-½ a_x`, total densities | two terms, factor `-a_x`, spin densities |
| Pair, Eq. 35 / 47 | one channel, `(1+δ_ij) t̃`, one MO frame | three channels `aa`, `ab`, `bb`, weights `c`, `2c`, `c`, two frames |

For a closed shell `D_σ = D/2` and `P_σ = P/2`, so the two UKS exchange terms sum
to `-½ a_x D P`, the RKS form. Both pair layouts are already `(i,a,j,b)` →
`C_i C_a C_j C_b`.

## Where the two paths differ today

| Stage | RKS | UKS | Step |
|---|---|---|---|
| AO derivative contraction | streamed callbacks | dense 3N arrays, **broken** | 1–3 |
| AO pair density | bounded AO tiles | six `n⁴` tensors | 1–3 |
| Dense reference adapters | kept in production types | n/a | 4 |
| MO integrals | five Eq. 40 sectors | three full `n⁴` transforms | 5 |
| Z-vector | matrix-free GMRES | dense joint Jacobian, pivoted QR | 6 |
| Amplitudes | source-backed pair tiles | resident `t2_aa/ab/bb` | 7 |

Steps 1–3 fix the error. Steps 4–7 remove the remaining second architecture.

## Steps

### 1. Spin-general pair generator

Generalize `for_each_dh_eq47_nonseparable_raw_ao_block` to a list of channels:

```cpp
struct DHPairChannel {
    int n_occ_s, n_virt_s, n_occ_t, n_virt_t;
    const Eigen::MatrixXd *c_s, *c_t;
    std::function<double(int i, int j, int a, int b)> amplitude; // scale included
};
```

- The generator sums all channels into each emitted AO tile. A UKS gradient is
  then one derivative traversal, not three.
- RKS is one channel. The `(1+δ_ij)` factor moves into its `amplitude`.
- The existing call becomes this generator. No second generator remains.

Gate: RKS output is bitwise unchanged (see Validation).

### 2. One two-particle source and one contraction

Replace `DHEq46_47TwoParticleDensity` as the callback argument with:

```cpp
struct DHEq33TwoParticleSource {
    Eigen::MatrixXd coulomb_d, coulomb_p;
    struct Exchange { Eigen::MatrixXd d, p; double factor; };
    std::vector<Exchange> exchange;
    std::vector<DHPairChannel> pair;
};
```

`contract_dh_eq33_two_electron_direct` takes only this. Delete its
`has_materialized_eq47` branch, `DHGammaStorage`, and the
`nonseparable_raw_ao` / `nonseparable_symmetric_ao` fields.

### 3. Route UKS through it (fixes the error)

- `build_dh_eq33_non_xc_gradient` takes AO `D`, `W` and the source, so both spins
  call one function. RKS forms its AO matrices before the call.
- `UDHTwoParticleDensity` keeps `relaxed_ao` and `ground_ao` and loses all six
  `UDHERICoefficients` tensors. `build_udh_two_particle_density` builds a source.
- Delete `build_udh_non_xc_gradient` and `UDHERICoefficients`.
- `planck-udh-validation` gets its per-channel ledger by calling the same
  contraction with one channel at a time. Production does not carry channel
  fields.
- Confirm the `W` sign. RKS passes `W` to `one_electron_gradient`, and UDH
  documents `+W:S'`. `tests/udh_two_particle_density.cpp` has an analytic-derivative
  fixture that settles it.

### 4. Remove the dense adapters

- `DHEq33NonXCDerivatives` becomes exactly the two callbacks. Delete the dense
  fields and the dense branch in `build_dh_eq33_non_xc_gradient`.
- Delete `build_dh_eq47_nonseparable_raw_ao`.
- Tests keep their independent dense oracle by supplying it as a callback in the
  test file (`dft_coulomb_response`, `dh_pt2_amplitude_density`,
  `dh_o5_backtransform`, `udh_stationary_transfer`,
  `udh_two_particle_density`). Update `tests/test_dh_o2_workspace.py`, which
  asserts on the removed source.
- Rewrite `docs/DH_IMPLEMENTATION.md` §7 and §9 rule 7, which currently permit
  retained dense adapters.

### 5. MO integrals by sector

Replace `UDHFullMOIntegrals` with the Eq. 40 sectors per spin block, built by
`build_dh_eq40_mo_eri_blocks_direct` with separate spin frames. Establish first
which sectors `build_udh_orbital_rhs` actually reads.

### 6. One Z-vector solver

Generalize `solve_dh_zvector` to a packed vector and an action, so UKS supplies a
joint α/β action. Delete the dense QR in `solve_udh_zvector`. RKS also has a
`DenseReference` backend; move that oracle into the tests in the same step.

### 7. Amplitude storage

Back UKS amplitudes with the same pair-tile storage as RKS. Whether `UMP2Result`
can supply tiles is not yet known, and needs checking before this step is sized.

## Validation

The energy path works today, so the oracle is finite differences of the current
code's energy.

| Gate | Requirement |
|---|---|
| RKS unchanged, steps 1–2 and 4 | Printed gradient bitwise identical on water and C1 H2O2 B2PLYP/STO-3G; `planck-dft-coulomb-response`, `dh_o5_backtransform` and `dh_pt2_amplitude_density` pass |
| RKS FD | Analytic vs energy FD ≤ 5e-8 Ha/Bohr (recorded 1.70e-8, 1.34e-8) |
| UKS FD, step 3 | Fourth-order FD on C1 H2O2⁺, HO2 and the asymmetric water cation via `tests/udh_molecular_audit.py`, ≤ 5e-8 Ha/Bohr (recorded worst 2.68e-9) |
| UKS translation | Gradient sum ≤ 1e-11 Ha/Bohr (recorded 5.9e-13) |
| Closed-shell reduction | `scf_type uhf` on a closed-shell molecule matches the RKS gradient |
| Mutations | Dropping `ab`, setting its weight to `c`, or swapping the exchange factor each fails a gate |
| Steps 5–7 | UKS gradient unchanged within 1e-9 Ha/Bohr at each step |

Register `h2o2_cation_uks_b2plyp_v4_gradient_sto3g` after step 3, tagged
`extended`, with `gradient_close` against the FD-verified gradient. It fails
before the fix, which is the point of it.

Comparing against a build of `039bd2df` was ruled out. Later changes (the
Å→Bohr constant, Boys v1.2.1, dropped redundant primitives) moved values, so it
is not a bitwise reference.

## Out of scope

- UKS double-hybrid geometry optimization and frequencies. Open once this lands.
- The per-AO-element `_compute_eri_deriv_elem` traversal and its hardcoded
  Obara-Saika call, which both spins already share.
- Spherical, solvent, range-separated and RI double-hybrid gradients.
