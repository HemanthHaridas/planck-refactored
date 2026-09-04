#ifndef HF_OS_H
#define HF_OS_H

#include <Eigen/Core>
#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include "base/types.h"
#include "fock_accumulate.h"
#include "shellpair.h"

namespace HartreeFock
{
    namespace ObaraSaika
    {
        double _os_1d(const double gamma, const double distPA, const double distPB, const int lA, const int lB);
        std::tuple<double, double> _compute_3d_overlap_kinetic(const HartreeFock::ShellPair &shell_pair);
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_1e(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const std::size_t nbasis,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);
        Eigen::MatrixXd _compute_nuclear_attraction(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const std::size_t nbasis,
            const HartreeFock::Molecule &molecule,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);
        // Build the AO matrix V_munu = -sum_c q_c <mu | 1/|r - r_c| | nu>
        // for an arbitrary list of point charges. Same Obara-Saika kernel
        // as _compute_nuclear_attraction; nuclear attraction is in fact a
        // thin wrapper that builds the charge list from the molecule and
        // calls into this routine. Used by the C-PCM module to assemble
        // one matrix per cavity tessera (see src/solvation/pcm.cpp).
        Eigen::MatrixXd _compute_external_charge_attraction(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const std::size_t nbasis,
            const std::vector<HartreeFock::ExternalCharge> &charges,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);
        // Build the full AO ERI tensor. Applies Schwarz screening:
        // quartets with Q(i,j)·Q(k,l) < tol_eri are skipped.
        std::vector<double> _compute_2e(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                        std::size_t nbasis,
                                        HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                                        double omega = 0.0,
                                        double tol_eri = 1e-10,
                                        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        // Contracted shell-quartet ERI for explicit angular-momentum components.
        double _contracted_eri_elem(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Shell-quartet block kernel (H-10 step 2a). Fills `block` with every
        // Cartesian-component ERI of the quartet (A B | C D) in [a][b][c][d]
        // row-major order (d fastest). `block` must hold at least
        // gA.n_components * gB.n_components * gC.n_components * gD.n_components
        // doubles. Bitwise-identical to per-component _contracted_eri_elem; not
        // yet wired into the production entry points (see os.cpp).
        void _contracted_eri_block(
            const HartreeFock::Basis &basis,
            const ShellGroup &gA, const ShellGroup &gB,
            const ShellGroup &gC, const ShellGroup &gD,
            HartreeFock::ERIKernel kernel,
            double omega,
            double *block);

        Eigen::MatrixXd _compute_fock_rhf(const std::vector<double> &_eri,
                                          const Eigen::MatrixXd &density,
                                          const std::size_t nbasis);

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_fock_uhf(const std::vector<double> &_eri,
                          const Eigen::MatrixXd &Pa, const Eigen::MatrixXd &Pb,
                          std::size_t nbasis);

        // Build the two-electron Fock contribution G = J - 0.5*K (direct SCF).
        // Applies Schwarz screening before each _contracted_eri call.
        Eigen::MatrixXd _compute_2e_fock(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                         const Eigen::MatrixXd &density,
                                         std::size_t nbasis,
                                         HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                                         double omega = 0.0,
                                         double tol_eri = 1e-10,
                                         const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        // UHF direct-SCF variant: returns {G_alpha, G_beta}.
        // Applies Schwarz screening; builds the ERI tensor once per call.
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                             const Eigen::MatrixXd &Pa,
                             const Eigen::MatrixXd &Pb,
                             std::size_t nbasis,
                             HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                             double omega = 0.0,
                             double tol_eri = 1e-10,
                             const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        // ── Memory-direct Fock builders ─────────────────────────────────────
        //
        // Same result as _compute_2e_fock / _compute_2e_fock_uhf above, but each
        // canonical quartet is contracted straight into G (nb^2) rather than
        // scattered into an nb^4 tensor that is then contracted in a second
        // sweep. The nb^4 array is never allocated — which is the whole point:
        // the two-phase builders above allocate it on EVERY SCF iteration
        // (0.8 GB at nb=100, 500 GB at nb=500), so "direct" mode currently costs
        // more memory than conventional, not less.
        //
        // Equal to the two-phase builders to summation-order noise (~1e-14), not
        // bitwise: the fused orbit accumulates in a different order than the nb^4
        // sweep. Gated by planck-fock-accumulate and planck-fused-fock.
        //
        // Integral symmetry (sym_ops) is handled natively: the ERI is computed
        // once per symmetry-orbit representative and replicated across the orbit
        // with the accumulated AO sign. See the dedup argument in
        // src/integrals/quartet_orbit.h.
        Eigen::MatrixXd _compute_2e_fock_direct(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &density,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr,
            HartreeFock::Integrals::FusedTerm term =
                HartreeFock::Integrals::FusedTerm::Combined);

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_2e_fock_uhf_direct(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &Pa,
            const Eigen::MatrixXd &Pb,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr,
            HartreeFock::Integrals::FusedTerm term =
                HartreeFock::Integrals::FusedTerm::Combined);

        // ── Gradient derivative integrals ──────────────────────────────────────────

        // Returns {dS/dAx, dS/dAy, dS/dAz, dT/dAx, dT/dAy, dT/dAz}
        // GTO-centre derivative of one contracted (μ,ν) shell pair (AM shift rule).
        std::array<double, 6> _compute_1e_deriv_A(const HartreeFock::ShellPair &sp);

        // Returns {dV/dAx, dV/dAy, dV/dAz}
        // Nuclear-attraction GTO-centre derivative (sums over all nuclei in mol).
        std::array<double, 3> _compute_nuclear_deriv_A_elem(
            const HartreeFock::ShellPair &sp,
            const HartreeFock::Molecule &mol);

        // Returns contracted dV_μν/dC_{direction} for one nucleus at C with charge Z.
        // direction: 0=x, 1=y, 2=z
        double _compute_nuclear_deriv_C_elem(
            const HartreeFock::ShellPair &sp,
            const Eigen::Vector3d &C, double Z, int direction);

        // Returns flat array of ERI derivatives for one (μν|λσ) contracted quartet.
        // Layout: [cen*3 + dir], cen∈{0=A,1=B,2=C,3=D}, dir∈{0,1,2}
        std::array<double, 12> _compute_eri_deriv_elem(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            const HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Test hook: return the weighted AM-raising term used by the ERI
        // derivative assembly for one specified centre (0=A, 1=B, 2=C, 3=D).
        double _contracted_eri_elem_weighted_test(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            int weight_center,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Compute the cross-overlap matrix S_cross(μ, ν) = <χ_μ^large | χ_ν^small>
        // between two basis sets centered on the same molecule.
        // Result has dimensions nbasis_large × nbasis_small.
        Eigen::MatrixXd _compute_cross_overlap(const HartreeFock::Basis &large_basis,
                                               const HartreeFock::Basis &small_basis);
    } // namespace ObaraSaika
} // namespace HartreeFock

#endif // !HF_OS_H
