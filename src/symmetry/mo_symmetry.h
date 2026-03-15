#ifndef HF_MO_SYMMETRY_H
#define HF_MO_SYMMETRY_H

#include "base/types.h"

namespace HartreeFock
{
    namespace Symmetry
    {
        // Assign irreducible representation labels to all converged MOs.
        // Fills calculator._info._scf.alpha.mo_symmetry (and beta for UHF).
        // No-op when symmetry is off or the point group is C1 or linear.
        void assign_mo_symmetry(HartreeFock::Calculator& calculator);

        // Symmetry-adapted orbital (SAO) basis data.
        // Columns of `transform` are SAOs expressed in the AO basis.
        // The SAO basis is orthonormal (U^T S U = I) and groups basis functions
        // by irreducible representation, making the Fock matrix block-diagonal.
        struct SAOBasis {
            Eigen::MatrixXd          transform;        // U [nb×nb]: col i = SAO i in AO basis
            std::vector<int>         sao_irrep_index;  // irrep index per SAO column
            std::vector<std::string> irrep_names;      // Mulliken name per irrep index
            std::vector<int>         block_sizes;      // n_SAOs per irrep block
            std::vector<int>         block_offsets;    // start col offset per block in U
            bool valid = false;
        };

        // Build the SAO basis for symmetry-blocked Fock diagonalization.
        // Returns valid=false for linear molecules (C∞v/D∞h), C1, or symmetry off.
        // Must be called after 1e integrals are computed and basis is built.
        SAOBasis build_sao_basis(HartreeFock::Calculator& calculator);
    }
}

#endif // !HF_MO_SYMMETRY_H
