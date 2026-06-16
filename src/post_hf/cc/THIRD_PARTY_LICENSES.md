# Third-Party Licenses — `src/post_hf/cc/`

The CCSD and CCSDT implementations in this directory were developed
independently for Planck. However, the PySCF quantum chemistry package
(version 2.12.1) is used as the **reference and regression oracle** in
`tests/pyscf/` to validate the correctness of these implementations.

## PySCF

- **Project**: PySCF — Python-based Simulations of Chemistry Framework
- **Version used for reference**: 2.12.1
- **Homepage**: <http://www.pyscf.org>
- **Repository**: <https://github.com/pyscf/pyscf>
- **License**: Apache License, Version 2.0

The specific PySCF modules consulted as algorithmic references are:

| Module | Description |
|--------|-------------|
| `pyscf/cc/rccsdt.py` | RHF-CCSDT (T1-dressed formalism) |
| `pyscf/cc/rccsdt_highm.py` | RHF-CCSDT with full T3 storage |
| `pyscf/cc/uccsdt.py` | UHF-CCSDT (symmetry-unique T3 storage) |
| `pyscf/cc/uccsdt_highm.py` | UHF-CCSDT with full T3 storage |
| `pyscf/cc/rccsdtq_highm.py` | RHF-CCSDTQ with full T4 storage |

All four CCSDT modules carry the following copyright notice:

```
Copyright 2014-2025 The PySCF Developers. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors: Yu Jin <yjin@flatironinstitute.org>
         Huanchen Zhai <hczhai.ok@gmail.com>
```

The full text of the Apache License 2.0 is reproduced in
`src/post_hf/cc/LICENSE-Apache-2.0.txt`.

## Note on Planck's CCSDT Implementation

No source code from PySCF was copied into Planck. The CCSDT equations
implemented in `ccsdt.cpp` / `ccsdt.h` follow the standard T1-dressed
formalism described in:

- J. Chem. Phys. **142**, 064108 (2015); DOI: 10.1063/1.4907278
- Chem. Phys. Lett. **228**, 233 (1994); DOI: 10.1016/0009-2614(94)00898-1
- Shavitt & Bartlett, *Many-Body Methods in Chemistry and Physics*,
  Cambridge University Press (2009); DOI: 10.1017/CBO9780511596834

PySCF regression values (obtained by running `tests/pyscf/`) are used
solely as numerical benchmarks to verify correctness of the Planck
implementation.
