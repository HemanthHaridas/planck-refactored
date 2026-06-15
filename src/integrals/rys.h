#ifndef HF_RYS_H
#define HF_RYS_H

#include <Eigen/Core>
#include <utility>
#include <vector>

#include "base/types.h"
#include "shellpair.h"

namespace HartreeFock
{
    namespace RysQuad
    {

        // ── Primitive-level ───────────────────────────────────────────────────────
        //
        // Compute a single primitive (uncontracted) ERI (ab|cd) using Rys quadrature.
        // The caller is responsible for multiplying by ppAB.coeff_product * ppCD.coeff_product.
        double _rys_eri_primitive(
            const HartreeFock::PrimitivePair &ppAB,
            const HartreeFock::PrimitivePair &ppCD,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            double ABx, double ABy, double ABz,
            double CDx, double CDy, double CDz,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0) noexcept;

        // ── Contracted shell quartet ───────────────────────────────────────────────
        //
        // Sum over all primitive pairs in spAB × spCD.
        double _rys_contracted_eri(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0) noexcept;

        // ── Public API — mirrors ObaraSaika:: signatures ───────────────────────────

        // Build 2e Fock contribution G = J - 0.5*K (direct SCF, RHF).
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

        // Build 2e Fock contribution (direct SCF, UHF). Returns {G_alpha, G_beta}.
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_2e_fock_uhf(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &Pa,
            const Eigen::MatrixXd &Pb,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        // ── Auto-dispatch variant ──────────────────────────────────────────────────
        //
        // Per-quartet three-way OS / HGP / Rys selection, keyed on the
        // (L_AB, L_CD) bucket via the data-derived kAutoEngine table (see
        // _auto_engine in rys.cpp and docs/auto_dispatch_fit.json). Rys is
        // selected only at the extreme high-L tail (7,8)/(8,8).

        std::vector<double> _compute_2e_auto(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        Eigen::MatrixXd _compute_2e_fock_auto(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &density,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        _compute_2e_fock_uhf_auto(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const Eigen::MatrixXd &Pa,
            const Eigen::MatrixXd &Pb,
            std::size_t nbasis,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0,
            double tol_eri = 1e-10,
            const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr);

        // ── Test hook (Phase B / B-1): box-size invariance of the 6D sum ───────────
        //
        // Fills `out` with the full per-primitive-pair 6D Rys `sum` buffer at a
        // caller-given box (lABx..lCDz) and explicit root count `n_roots`, BEFORE
        // HRR. The buffer is row-major in RysScratch's stride convention with each
        // dim = box+1. The root count is explicit (it is the quartet quadrature
        // degree, not the summed per-axis box); B-1 builds the max box with n_max
        // and the component box with n_comp and requires them equal at every
        // component coordinate. ShortRange = Coulomb − LongRange. Used only by
        // tests/rys_box_invariance.cpp; no production path calls it.
        void _build_sum_native_test(
            const HartreeFock::PrimitivePair &ppAB,
            const HartreeFock::PrimitivePair &ppCD,
            int lABx, int lABy, int lABz,
            int lCDx, int lCDy, int lCDz,
            int n_roots,
            HartreeFock::ERIKernel kernel,
            double omega,
            std::vector<double> &out) noexcept;

    } // namespace RysQuad
} // namespace HartreeFock

#endif // HF_RYS_H
