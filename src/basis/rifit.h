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
// Aux shells are loaded Cartesian. The FITTED aux functions follow the orbital
// basis, as in PySCF (its auxmol copies mol.cart): with a spherical orbital
// basis, AuxBasis::cart_to_sph maps the Cartesian aux to pure harmonics
// (docs/RI_SPHERICAL_AUX.md). The orbital-index transform of the
// 3-center tensor is handled inside the 3-center integral routine.

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
        std::size_t nfunctions = 0;       // Cartesian aux function count (integral loops, offsets)
        bool cartesian = true;            // shells are always loaded Cartesian
        // Block-diagonal [n_sph_aux × nfunctions] map to the fitted (spherical)
        // aux functions, set when the orbital basis is spherical; empty means
        // the Cartesian functions are fitted directly. Applied once to the
        // metric and the 3-center tensor (ri_eri.cpp).
        Eigen::MatrixXd cart_to_sph;
        std::size_t nfit() const
        {
            return cart_to_sph.size() ? static_cast<std::size_t>(cart_to_sph.rows()) : nfunctions;
        }
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
