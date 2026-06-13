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

        // Shell-quartet block kernel (H-10 phase A, step A1). Fills `block`
        // with every Cartesian-component ERI of the quartet (A B | C D) in
        // [a][b][c][d] row-major order (d fastest). `block` must hold at least
        // gA.n_components * gB.n_components * gC.n_components * gD.n_components
        // doubles. Bitwise-identical to per-component _contracted_eri_elem; it
        // still calls the per-component kernel once per component (the once-per-
        // shell-quartet VRR/HRR readout is step A4). Not yet wired into the
        // production entry points (see hgp.cpp).
        void _contracted_eri_block(
            const HartreeFock::Basis &basis,
            const ShellGroup &gA, const ShellGroup &gB,
            const ShellGroup &gC, const ShellGroup &gD,
            HartreeFock::ERIKernel kernel,
            double omega,
            double *block);

        // Hoisted shell-quartet block kernel (H-10 step A4-1′). Same output
        // contract as _contracted_eri_block ([a][b][c][d] row-major, d fastest,
        // same size), but amortizes the expensive work: the per-primitive VRR +
        // (a0|c0) contraction runs ONCE per shell quartet at the max AM box and
        // each Cartesian component is then read out via a cheap per-component
        // HRR. Bitwise-identical to _contracted_eri_block (and thus to the per-
        // component _contracted_eri_elem); gated by planck-os-block-kernel.
        // Not yet wired into the production entry points (steps A4-2 / A4-3).
        void _contracted_eri_block_hoisted(
            const HartreeFock::Basis &basis,
            const ShellGroup &gA, const ShellGroup &gB,
            const ShellGroup &gC, const ShellGroup &gD,
            HartreeFock::ERIKernel kernel,
            double omega,
            double *block);

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

        // Test hook (H-10 step A4-pre): contract the (a0|c0) block over all
        // primitive pairs at the AM box (lABx..lCDz) and return the accumulated
        // value at logical coordinate (ax,ay,az,cx,cy,cz). Used to gate that a
        // single max-AM contraction's per-component sub-block is bitwise-equal
        // to a per-component-AM contraction — the §4 invariant the hoisted A4
        // path depends on. ShortRange returns Coulomb − LongRange, matching the
        // production split.
        double _contract_a0c0_at_native_test(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lABx, int lABy, int lABz,
            int lCDx, int lCDy, int lCDz,
            int ax, int ay, int az,
            int cx, int cy, int cz,
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
