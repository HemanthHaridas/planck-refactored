#ifndef HF_OS_SYMM_H
#define HF_OS_SYMM_H

#include <Eigen/Core>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"
#include "symmetry/group_operations.h"

// ─── Obara-Saika direct Fock with FULL point-group symmetry reduction ──────────
//
// Phase 2 of the full-symmetry ERI reduction (docs/FULL_SYMMETRY_ERI_DESIGN.md).
// Separate `_symm` implementation kept ALONGSIDE the production os.cpp
// _compute_2e_fock so the proven D2h-only path is untouched and the two can be
// A/B benchmarked and cross-validated (symmetry-on == symmetry-off energy).
//
// Method (Dupuis-King skeleton):
//   1. Build a SKELETON ERI tensor over only the petite-list-unique shell quartets
//      under the full group G (each orbit representative once, with the usual
//      8-fold permutational (ij|kl) symmetry); non-representative quartets are
//      skipped entirely — that is the integral-compute saving.
//   2. Contract the skeleton to a skeleton Fock F_skel with the standard
//      J − ½K (RHF) / spin-resolved (UHF) formula.
//   3. Restore the true Fock by projecting onto the totally-symmetric component:
//        F = (1/|G|) Σ_R O_Rᵀ F_skel O_R     (symmetry/fock_symmetrization.h)
//
// The O_R come from symmetry/group_operations.h (Phase 0). If `ops` is invalid
// (symmetry off / C1 / linear) the routines fall back to computing every quartet
// (no reduction) so the result still equals the full Fock.
//
// NOTE: like the production direct path, this still materializes an nb⁴ tensor —
// it saves integral *compute*, not memory. Memory-direct is a later phase.

namespace HartreeFock
{
    namespace ObaraSaika
    {
        // RHF direct Fock G = J − ½K via the skeleton + symmetrization scheme.
        // `basis` supplies the shell ordering that ops.shell_perm is indexed by.
        std::expected<Eigen::MatrixXd, std::string> _compute_2e_fock_symm(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            const Eigen::MatrixXd &density,
            std::size_t nbasis,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);

        // UHF direct Fock {G_alpha, G_beta} via the skeleton + symmetrization scheme.
        std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
        _compute_2e_fock_uhf_symm(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            const Eigen::MatrixXd &Pa,
            const Eigen::MatrixXd &Pb,
            std::size_t nbasis,
            const HartreeFock::Symmetry::GroupOperations &ops,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10);
    } // namespace ObaraSaika
} // namespace HartreeFock

#endif // !HF_OS_SYMM_H
