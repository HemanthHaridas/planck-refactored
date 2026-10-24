#ifndef HF_SAD_H
#define HF_SAD_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <unordered_map>
#include <utility>

#include "base/types.h"

namespace HartreeFock
{
    namespace SCF
    {
        // Returns one atomic Basis per unique element symbol present in molecule.
        // Each basis is constructed for a single atom at the origin, using the
        // same contraction/normalization conventions as read_gbs_basis in gaussian.cpp.
        std::expected<std::unordered_map<std::string, HartreeFock::Basis>, std::string>
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
        // Assembles a block-diagonal molecular density and scales it to
        // n_electrons: P = (n / Tr(P_raw S)) P_raw. Deliberately not idempotent:
        // choosing the top n natural orbitals is ill-posed when a degenerate set
        // straddles the cutoff; the first SCF diagonalization occupies by orbital
        // energy instead (see sad.cpp, scale_raw_sad).
        //
        // Requires:
        //   - calc._overlap already computed (molecular S matrix)
        //   - calc._basis._basis_path + "/" + _basis_name pointing to the GBS file
        //   - n_electrons = sum(Z) - charge must be even
        std::expected<Eigen::MatrixXd, std::string> compute_sad_guess_rhf(
            const HartreeFock::Calculator &calc);

        // Compute spin-resolved SAD initial densities for open-shell SCF.
        //
        // Spin-neutral: both channels are the spin-summed raw RHF-SAD density,
        // scaled to n_alpha and n_beta electrons. The spins polarize at the first
        // SCF iteration through their different occupations.
        //
        // The result can be used directly as the initial {P_alpha, P_beta}
        // guess for both UHF and ROHF.
        std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
        compute_sad_guess_open_shell(
            const HartreeFock::Calculator &calc,
            int n_alpha,
            int n_beta);

    } // namespace SCF
} // namespace HartreeFock

#endif // !HF_SAD_H
