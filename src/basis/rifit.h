#ifndef HF_BASIS_RIFIT_H
#define HF_BASIS_RIFIT_H

// RI / density-fitting auxiliary basis loader.
//
// Auxiliary basis sets share the Gaussian94 (.gbs) file format with orbital
// basis sets, but they serve a different role: they are the "ket" side of
// 3-center integrals (μν|P) used to factor the 4-center AO ERI tensor as
// (μν|λσ) ≈ Σ_PQ (μν|P) V_{PQ}^{-1} (Q|λσ). Consequently the aux basis
// lives in its own storage, separate from the orbital `HartreeFock::Basis`,
// and is never mixed into the orbital shell list or AO function index space.
//
// Conventionally, aux basis sets stay Cartesian (PySCF default). The mixed
// Cartesian-aux × spherical-orbital case is handled inside the 3-center
// integral routine, not here.

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"

namespace HartreeFock
{
    // A loaded auxiliary basis set. The Shell objects use the same
    // normalization conventions (contracted norm pre-folded into _coefficients,
    // primitive norms held separately) as the orbital basis so the existing
    // integral machinery can consume them unchanged.
    struct AuxBasis
    {
        std::vector<Shell> shells;       // aux shells, in input order
        std::vector<std::size_t> offsets; // offsets[K] = first aux function of shell K
        std::size_t nfunctions = 0;       // total aux function count (Σ (2L+1) or Cartesian count)
        bool cartesian = true;            // aux functions are Cartesian by default
    };

    namespace BasisFunctions
    {
        // Read a Gaussian94-format auxiliary basis file and instantiate aux
        // shells on every atom of `molecule`. Returns the populated AuxBasis
        // or a human-readable error.
        //
        // The aux basis is always built in the Cartesian basis function space:
        // a shell of angular momentum L contributes (L+1)(L+2)/2 functions to
        // the offset table. Spherical-orbital users still load Cartesian aux
        // shells — the contraction with spherical AOs is done downstream by
        // the 3-center integral routine.
        std::expected<AuxBasis, std::string> read_ri_basis(
            const std::string &file_name,
            const Molecule &molecule);
    } // namespace BasisFunctions
} // namespace HartreeFock

#endif // HF_BASIS_RIFIT_H
