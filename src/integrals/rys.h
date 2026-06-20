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

        // ── Test hook (Phase B / B-2b): contract the 6D sum over primitive pairs ───
        //
        // Accumulates, over every primitive pair of the two shell pairs, the
        // per-pair 6D Rys `sum` (built at the given box / n_roots) weighted by
        // `coeff_product`, into a single contracted block returned in `acc` (row-
        // major in RysScratch's stride convention, each dim = box+1). This is the
        // Rys analog of HGP's hgp_contract_a0c0: contract-then-HRR rather than the
        // production per-pair HRR-then-sum.
        //
        // ShortRange = Coulomb − LongRange is composed per pair (matching
        // _rys_contracted_eri's per-pair value), so the contracted block already
        // carries the kernel. Views still carry their folded _component_norm
        // (norm-free max-box readout is B-2c), so callers use a single fixed
        // component box / n_comp; at a single primitive pair the contracted block
        // HRR'd reproduces _rys_contracted_eri bitwise (no cross-pair reorder).
        // Used only by tests/rys_box_invariance.cpp; no production path calls it.
        void _contract_sum_native_test(
            const HartreeFock::ShellPair &spAB,
            const HartreeFock::ShellPair &spCD,
            int lABx, int lABy, int lABz,
            int lCDx, int lCDy, int lCDz,
            int n_roots,
            HartreeFock::ERIKernel kernel,
            double omega,
            std::vector<double> &acc) noexcept;

        // ── Test hook (Phase B / B-2b): HRR a contracted 6D block to a scalar ──────
        //
        // Loads `block` (row-major, dims = box+1, as produced by
        // _contract_sum_native_test at box lAB*+? = the component's own box) into a
        // RysScratch and runs AB-HRR then CD-HRR for the component (lA*..lD*),
        // returning the scalar ERI. Lets the B-2b test compare
        // contract-then-HRR against _rys_contracted_eri. Used only by
        // tests/rys_box_invariance.cpp.
        double _hrr_block_native_test(
            const std::vector<double> &block,
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz,
            int lDx, int lDy, int lDz,
            double ABx, double ABy, double ABz,
            double CDx, double CDy, double CDz) noexcept;

        // ── Test hooks (Phase B / B-2c): norm-free max-box contract + readout ──────
        //
        // The B-2c flow, split so the snapshot is built ONCE per shell quartet
        // (as the production hoist B-3 will), then each component reads out of it.
        //
        // _contract_maxbox_snapshot_native_test: contract the 6D sum from norm-free
        //   component-0 views (_component_norm forced to 1 internally) at the max
        //   box / n_max roots, returning the snapshot block and the AB/CD shell
        //   separations needed by HRR.
        // _maxbox_readout_native_test: gather one component's sub-box from the
        //   snapshot, HRR it, and apply normA·normB·normC·normD (invariant 2 — the
        //   shared norm-free contraction carries no per-component norm).
        //
        // Together they reproduce _rys_contracted_eri (which folds norm per pair
        // and builds per-component) to ≤1e-13, exercising both the n_max-over-
        // component reorder (B-1) and the norm-after-HRR reorder.
        // Used only by tests/rys_box_invariance.cpp.
        void _contract_maxbox_snapshot_native_test(
            const HartreeFock::ContractedView &cvA0,
            const HartreeFock::ContractedView &cvB0,
            const HartreeFock::ContractedView &cvC0,
            const HartreeFock::ContractedView &cvD0,
            int maxAB, int maxCD,
            HartreeFock::ERIKernel kernel,
            double omega,
            std::vector<double> &snapshot,
            double R_AB[3], double R_CD[3]) noexcept;

        double _maxbox_readout_native_test(
            const std::vector<double> &snapshot,
            int maxAB, int maxCD,
            const HartreeFock::ContractedView &cvA,
            const HartreeFock::ContractedView &cvB,
            const HartreeFock::ContractedView &cvC,
            const HartreeFock::ContractedView &cvD,
            const double R_AB[3], const double R_CD[3]) noexcept;

    } // namespace RysQuad
} // namespace HartreeFock

#endif // HF_RYS_H
