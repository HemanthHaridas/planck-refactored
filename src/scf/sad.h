#ifndef HF_SAD_H
#define HF_SAD_H

#include <Eigen/Core>
#include <string>
#include <unordered_map>

#include "base/types.h"

namespace HartreeFock
{
    namespace SCF
    {
        // Returns one atomic Basis per unique element symbol present in molecule.
        // Each basis is constructed for a single atom at the origin, using the
        // same contraction/normalization conventions as read_gbs_basis in gaussian.cpp.
        std::unordered_map<std::string, HartreeFock::Basis>
        read_gbs_basis_atomic(const std::string &file_name,
                              const HartreeFock::Molecule &molecule,
                              const HartreeFock::BasisType &basis_type);

        // Compute the SAD initial density for RHF.
        //
        // For each unique element in calc._molecule:
        //   1. Runs atomic UHF (HCore guess, suppressed output).
        //   2. Forms spin-summed density P_atom = P_alpha + P_beta.
        //   3. Applies shell-wise spherical averaging to remove directional bias.
        //      Shell populations are preserved in the full atomic AO metric, so
        //      zeroing cross-shell couplings does not leak charge.
        //
        // Assembles a block-diagonal molecular density, then projects it to the
        // nearest proper RHF density via:
        //   X = S^{-1/2} (thresholded)  →  diagonalize X^T P X
        //   →  take top n_occ eigenvectors  →  P = 2 C_occ C_occ^T
        //
        // Requires:
        //   - calc._overlap already computed (molecular S matrix)
        //   - calc._basis._basis_path + "/" + _basis_name pointing to the GBS file
        //   - n_electrons = sum(Z) - charge must be even
        Eigen::MatrixXd compute_sad_guess_rhf(const HartreeFock::Calculator &calc);

    } // namespace SCF
} // namespace HartreeFock

#endif // !HF_SAD_H
