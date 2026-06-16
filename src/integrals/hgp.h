#ifndef HF_HGP_H
#define HF_HGP_H

#include <Eigen/Core>
#include <array>
#include <utility>
#include <vector>

#include "base/types.h"
#include "shellpair.h"

namespace HartreeFock
{
    namespace HeadGordonPople
    {
        double _contracted_eri_elem(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Phase-1 HGP integration surface: keep the public API aligned with the
        // existing ERI engines so the correctness-preserving plumbing lands
        // first, then the internal contracted-quartet kernel can be replaced
        // without touching callers.
        std::vector<double> _compute_2e(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        Eigen::MatrixXd _compute_2e_fock(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &density,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_2e_fock_uhf(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &Pa,
            const Eigen::MatrixXd &Pb,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        // Returns flat array of ERI derivatives for one (mu nu | lambda sigma)
        // contracted quartet.
        // Layout: [cen*3 + dir], cen in {0=A,1=B,2=C,3=D}, dir in {0=x,1=y,2=z}
        std::array<double, 12> _compute_eri_deriv_elem(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Test hook: alias of `_contracted_eri_elem` retained for the gate
        // tests written when the production entry routed screened kernels
        // through OS. Both now compute natively; this exists only so existing
        // tests continue to link.
        double _contracted_eri_elem_native_test(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Test hook: return the weighted AM-raising term used by the HGP ERI
        // derivative assembly for one specified centre (0=A, 1=B, 2=C, 3=D).
        double _contracted_eri_elem_weighted_native_test(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            int weight_center,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);

        // Test hook: alias of `_compute_eri_deriv_elem` retained for the gate
        // tests written when the production entry routed screened kernels
        // through OS. Both now compute natively; this exists only so existing
        // tests continue to link.
        std::array<double, 12> _compute_eri_deriv_elem_native_test(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0);
    } // namespace HeadGordonPople
} // namespace HartreeFock

#endif // HF_HGP_H
