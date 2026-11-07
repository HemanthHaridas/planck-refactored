# How does Planck add Grimme's D3 dispersion correction?

Answer, landed 2026-11-07 (opened as a scope on 2026-10-26). Status lives in
`vault/Status/Completion.md` and `vault/Status/Open Work.md`.

## Answer

Planck links **simple-dftd3 v1.7.0** (LGPL-3.0-or-later), fetched by
`ExternalProject` at build time and linked statically. No D3 code or data is
in the tree. `planck-dft` takes

```
%begin_dft
    dispersion            d3bj      # none | d3bj | d3zero | d3bj-atm | d3zero-atm
    dispersion_functional b3lyp     # optional; derived from the functional
%end_dft
```

D3 depends only on the nuclear geometry, so it enters where nuclear repulsion
does:

- **Energy:** `run_ks_scf_scaffold` evaluates it once per call, before the SCF
  loop, and adds it to the RKS and UKS totals next to `_nuclear_repulsion`.
  Every DFT energy goes through the scaffold (single points, FD displacements,
  geomopt steps, double-hybrid KS halves). It is printed as
  `Dispersion Energy :` after `DFT Energy :`, which already includes it.
- **Gradient:** `compute_analytic_ks_gradient` adds it next to the HF-like and
  XC terms. Every KS gradient goes through that function: RKS, UKS, and the
  KS half of the RKS and UKS double-hybrid gradients. Geometry optimization
  and frequencies (FD of gradients) need no further code.

Both call `DFT::Dispersion::compute` (`src/dft/dispersion.{h,cpp}`), which
reads `_molecule._standard` and so works in the same frame as the rest of the
gradient. Nothing is cached, so nothing can go stale across a displacement;
it costs microseconds per geometry.

### Parameter names

`parameter_set_name` derives the simple-dftd3 name from the **resolved libxc
ids**, so built-in aliases and raw libxc names behave the same:

| libxc functional(s) | name |
|---|---|
| combined `*_xc_<name>` | `<name>` without underscores: `b3lyp`, `pbeh`, `hse06`, `b2plyp`, `camb3lyp` |
| `*_x_<a>` + `*_c_<a>` | `<a>` (`pbe`, `pw91`) |
| `gga_x_b88` + `gga_c_lyp` / `gga_c_p86` | `blyp` / `bp86` |
| anything else | rejected, naming both functionals |

simple-dftd3 drops `-` but not `_` when it matches a name, hence the
underscore stripping. A name it does not know fails the run with its own
message. Nothing falls back to zero. `dispersion_functional` overrides the
derivation and is passed through unchanged.

Two traps checked on the way:

- libxc's `hyb_gga_xc_b3lyp` is the VWN-RPA (Gaussian) B3LYP, while
  simple-dftd3's `b3lyp` names the VWN5 one (`b3lypg` is VWN-RPA). The D3
  parameters are identical for both in every damping (`param.f90`), so
  `b3lyp` is correct.
- PBE0 is `hyb_gga_xc_pbeh` in libxc, so Planck asks for `pbeh`. v1.7.0 maps
  it to `pbe0`; v1.0.0 does not know the alias.

Ghost atoms are left out of the D3 structure (zero gradient rows), which is
the counterpoise-consistent choice. DFT counterpoise itself is still open.

### Build

`-DPLANCK_WITH_DFTD3` defaults ON. CMake's `check_language(Fortran)` looks
for a Fortran compiler. Without one the option is forced OFF with a warning,
and `dispersion` inputs then fail with a message naming the option. CI
installs gfortran and passes `-DCMAKE_Fortran_COMPILER=gfortran`, so a missing
compiler fails there instead of silently dropping D3. The imported `dftd3`
target is linked globally, like libecpint, because `src/dft/dispersion.cpp` is
in `DFT_LIB_SRC`, which several test targets compile. Its
`IMPORTED_LINK_INTERFACE_LANGUAGES Fortran` pulls in libgfortran. simple-dftd3
enables C to install its header, so CMake passes `CMAKE_C_COMPILER` through
when set (the system `cc` here does not link). Its own CMake fetches mctc-lib
v0.7.0 and toml-f v0.5.1 at pinned tags, and `s-dftd3-dependency-method=fetch`
keeps a system copy out. The library build takes about 14 s.

### Licence

simple-dftd3 is LGPL-3.0-or-later. The C6 reference table and the functional
parameters live inside it, with no separate data licence. Planck ships source
only and fetches the library at build time, so no LGPL file enters the tree
and building for one's own use conveys nothing. Anyone who redistributes a
built binary must meet LGPL §4 for the statically linked library (README,
Requirements). Not legal advice. A native C++ port was rejected because the
C6 table and the code would be LGPL files in a PolyForm tree. The licence of
Grimme's original `dftd3` program and data could not be confirmed, so it is
no alternative source.

## Validation

References come from `tests/pyscf/d3_references.py`. It uses PySCF 2.13.0
`mf.disp` and pyscf-dispersion's wrapper, but over a separately built shared
**simple-dftd3 v1.0.0** (`DFTD3_LIBRARY`): a different release and binary from
Planck's. Settings are `cart=True`, Planck's GBS and Bohr, and the unpruned
150 × 590 grid.

| case | what it pins | agreement |
|---|---|---|
| `water_rks_b3lyp_d3bj_sto3g` | D3, RKS total | D3 every printed digit (−5.7366166775e-4); total 3e-11 |
| `water_cation_uks_b3lyp_d3bj_sto3g` | UKS scaffold total | 7e-12 |
| `water_dimer_rks_pbe_d3bj_gradient_sto3g` | gradient, every component | ≤ 8.2e-10 Ha/Bohr; the D3 part reaches 1.2e-4 |
| `…_gradient_symm_sto3g` | same, under Cs reorientation | equals the C1 run to 2e-12 |
| `benzene_rks_pbe0_d3zero_sto3g` | zero damping, `pbeh` name | D3 every digit (−3.11244711812e-3); total 1e-11 |
| `h2o2_rks_b2plyp_d3bj_atm_sto3g` | ATM, double-hybrid name, C1 | D3 every digit |
| `water_dimer_rks_pbe_d3bj_geomopt_sto3g` | geomopt | E_opt 1.0e-8 above geomeTRIC gau_tight (Planck stops at max\|g\| 4.6e-5) |
| `water_rks_lda_d3bj_rejected_sto3g` | no parameters → error | message names `lda_x` + `lda_c_vwn` |

Mutation check: dropping the D3 gradient term fails both dimer gradient cases
(1.16e-4). Dropping the D3 term from the RKS energy line fails the RKS water
case by exactly the D3 energy. The UKS line has its own gate.

Not separately gated: an FD check of the D3 gradient alone. That would test
simple-dftd3, which its own suite covers, and the PySCF component comparison
already pins Planck's summation and frame. Frequencies are not gated either:
they are central differences of the gated gradient.

### Reference-tool findings

- **pyscf-dispersion 1.0.0 does not load on macOS arm64.** It calls
  `np.ctypeslib.load_library` without the `loader_path` numpy 2 requires,
  numpy 2 does not try `.dylib`, and the wheel bundles an x86_64 library. The
  reference script loads `$DFTD3_LIBRARY` instead and stubs the unused D4
  library.
- **The wrapper ignores simple-dftd3 errors.** An unknown name returns an
  energy of exactly 0 (v1.0.0 with `pbeh`). The script asserts a nonzero D3
  energy.

## Not done

- HF-D3. HF-D3 parameters exist (`hf`), and the hook would be the same two
  lines in the HF SCF and gradient. Nobody has asked for it.
- D4 (charge-dependent C6; needs atomic charges), and VV10-type nonlocal
  correlation, which belongs with meta-GGA (`docs/META_GGA_SCOPE.md`).
- Modified damping variants (`d3mbj`, `d3mzero`, `d3op`). simple-dftd3 has
  them, and one more branch in `compute` would add each.
- `results_json` reports `electronic_energy` as total minus nuclear repulsion,
  so it includes D3. There is no separate JSON field yet.
