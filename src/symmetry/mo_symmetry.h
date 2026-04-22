#ifndef HF_MO_SYMMETRY_H
#define HF_MO_SYMMETRY_H

#include "base/types.h"
#include <expected>

namespace HartreeFock
{
    namespace Symmetry
    {
        // Assign irreducible representation labels to all converged MOs.
        // Fills calculator._info._scf.alpha.mo_symmetry (and beta for UHF).
        // No-op when symmetry is off or the point group is C1 or linear.
        std::expected<void, std::string> assign_mo_symmetry(HartreeFock::Calculator &calculator);

        // Symmetry-adapted orbital (SAO) basis data.
        // Columns of `transform` are SAOs expressed in the AO basis.
        // The SAO basis is orthonormal (U^T S U = I) and groups basis functions
        // by irreducible representation, making the Fock matrix block-diagonal.
        struct SAOBasis
        {
            Eigen::MatrixXd transform;            // U [nb×nb]: col i = SAO i in AO basis
            std::vector<int> sao_irrep_index;     // irrep index per SAO column
            std::vector<std::string> irrep_names; // Mulliken name per irrep index
            std::vector<int> block_sizes;         // n_SAOs per irrep block
            std::vector<int> block_offsets;       // start col offset per block in U
            bool valid = false;
        };

        struct AbelianIrrepProductTable
        {
            std::vector<std::string> irrep_names;
            std::vector<std::vector<int>> product;
            bool valid = false;
        };

        // Build the SAO basis for symmetry-blocked Fock diagonalization.
        // Returns valid=false for linear molecules (C∞v/D∞h), C1, or symmetry off.
        // Must be called after 1e integrals are computed and basis is built.
        std::expected<SAOBasis, std::string> build_sao_basis(HartreeFock::Calculator &calculator);

        // Build an explicit Abelian irrep multiplication table for the current
        // point group. Returns valid=false when symmetry is unavailable or the
        // full point group does not have only 1D irreps.
        std::expected<AbelianIrrepProductTable, std::string> build_abelian_irrep_product_table(
            HartreeFock::Calculator &calculator);
    } // namespace Symmetry
} // namespace HartreeFock

#endif // !HF_MO_SYMMETRY_H
