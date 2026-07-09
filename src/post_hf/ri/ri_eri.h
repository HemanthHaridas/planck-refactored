#ifndef HF_POST_HF_RI_ERI_H
#define HF_POST_HF_RI_ERI_H

// 2-center and 3-center ERI routines for the RI / density-fitting subsystem.
//
// Conventions:
//   * Auxiliary shells are loaded as Cartesian (matches AuxBasis::cartesian).
//   * The Coulomb metric V_{PQ} = (P|Q) is symmetric positive-semidefinite by
//     construction; small linearly-dependent tails are dropped at the Cholesky
//     step (Step 3), not here.
//   * Normalization follows the same contract as the orbital basis:
//     contracted norm pre-folded into Shell._coefficients, primitive norms
//     held separately on Shell._normalizations. See the Norm Factors gotcha.

#include <Eigen/Core>
#include <array>
#include <expected>
#include <string>

#include "base/types.h"
#include "basis/rifit.h"

namespace HartreeFock::Correlation::RI
{
    struct MetricFactorization
    {
        enum class Method
        {
            Cholesky,
            Eigen
        };

        Method method = Method::Cholesky;
        // Cholesky: lower-triangular L with V = L L^T.
        // Eigen fallback: X = diag(w_kept^-1/2) U_kept^T, so X V X^T = I.
        Eigen::MatrixXd transform;
        Eigen::VectorXd eigenvalues_kept;
        Eigen::VectorXi kept_indices;
    };

    // V_{PQ} = (P|Q) Coulomb metric matrix on the auxiliary basis.
    // Returns an (nfunctions × nfunctions) symmetric positive-semidefinite
    // matrix indexed by aux function (not shell). The Cartesian function
    // ordering within a shell matches the orbital basis convention.
    std::expected<Eigen::MatrixXd, std::string> compute_2c_eri(const AuxBasis &aux);

    // Factor the 2-center metric in the same spirit as PySCF's df.incore:
    // attempt a Cholesky factorization first, then fall back to an eigenvalue
    // decomposition that drops linearly dependent modes below `lindep`.
    std::expected<MetricFactorization, std::string> factorize_2c_metric(
        const Eigen::MatrixXd &metric,
        double lindep);

    // Load, build, and factor the RI metric on Calculator when MP2 RI is
    // enabled. This only prepares the cache; later MP2 code will consume it.
    std::expected<void, std::string> ensure_ri_metric_ready(
        HartreeFock::Calculator &calculator);

    // Packed 3-center AO integrals in chemists' notation: rows index the
    // unique AO pair (μ ≥ ν) via μ(μ+1)/2 + ν, columns index auxiliary
    // functions P. In spherical mode the packed pair space follows the
    // spherical AO ordering after a Cartesian→spherical transform on both AO
    // legs; the auxiliary basis remains Cartesian.
    std::expected<Eigen::MatrixXd, std::string> compute_3c_eri(
        const HartreeFock::Calculator &calculator);

    // Analytic nuclear derivative of one contracted 3-center Cartesian element
    // (μ ν | Q) w.r.t. the three centers it sits on. Layout: [center][axis],
    // center 0 = μ (orbital A), 1 = ν (orbital B), 2 = aux (Q); axis 0/1/2 = x/y/z,
    // so index = center*3 + axis. Contracted over the μν primitive pairs and the
    // aux shell primitives at the given Cartesian momenta. Coulomb kernel only
    // (RI fitting is Coulomb-metric).
    //
    // Uses the Gaussian translational identity, same as the 4-center
    // ObaraSaika::_compute_eri_deriv_elem:
    //   d/dX_q = 2 ζ_X · I(l_X + ê_q)  −  l_Xq · I(l_X − ê_q).
    //
    // RG1a.1: only the μ-center (A) block is populated; ν and aux blocks are
    // filled by RG1a.2. The unpopulated blocks are left zero.
    std::array<double, 9> compute_3c_deriv_elem(
        const HartreeFock::ShellPair &spAB,
        int lAx, int lAy, int lAz,
        int lBx, int lBy, int lBz,
        const HartreeFock::Shell &shellC,
        int lCx, int lCy, int lCz);

    std::expected<void, std::string> ensure_ri_3c_ready(
        HartreeFock::Calculator &calculator);

    // Fitted AO-pair factors B_{(μν),Q} from the cached 3-center tensor and the
    // 2-center metric factorization: applies V^{-1/2} (Cholesky solve or the
    // eigen-pruned transform) to _ri_j3c. Rows index the packed AO pair (μ≥ν),
    // columns index the fitting-metric space. Requires ensure_ri_3c_ready and a
    // populated _ri_metric_factor on the Calculator.
    Eigen::MatrixXd build_ri_pair_factors(const HartreeFock::Calculator &calculator);

    // Contract the packed-pair fitted factors into an MO block B_{(pq),Q},
    // rows indexed p*ncol_q + q over the columns of C_row / C_col. The AO-pair
    // packing (μ≥ν with the off-diagonal doubling) is unfolded here. Reused by
    // MP2 (o/v block) and, from Step 3 on, by the conventional post-HF paths.
    Eigen::MatrixXd build_ri_mo_block(
        const Eigen::MatrixXd &pair_factors,
        const Eigen::MatrixXd &C_row,
        const Eigen::MatrixXd &C_col);

    // RI Coulomb matrix J_{μν} = Σ_Q B_{(μν),Q} c_Q, with the fitted charge
    // c_Q = Σ_{λσ} B_{(λσ),Q} D_{λσ}. Both use the packed (μ≥ν) pair factors, so
    // off-diagonal pairs carry an explicit factor 2 in the c_Q accumulation and
    // the J scatter fills both (μ,ν) and (ν,μ). D must be symmetric. Requires a
    // ready RI cache (build_ri_pair_factors); the AO dimension nb is inferred
    // from the packed pair count.
    Eigen::MatrixXd build_ri_j(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &D);

    // Unpack the fitted pair factors B_{(μν),Q} (packed μ≥ν) into the full
    // symmetric per-aux matrices B[Q](μ,ν) = B[Q](ν,μ). Returned as naux
    // matrices of size nb×nb — the nb²·naux working set the RI exchange build
    // needs (still far below nb⁴). This is fitted (metric already applied), not
    // the raw 3-center tensor.
    std::vector<Eigen::MatrixXd> build_ri_3index_unpacked(
        const HartreeFock::Calculator &calculator);

    // RI exchange K_{μν} = Σ_Q Σ_{λσ} B_{μλ,Q} B_{νσ,Q} D_{λσ}, via the two-step
    // H[Q] = B[Q] D ; K = Σ_Q H[Q] B[Q]ᵀ, using the unpacked per-aux matrices.
    // D symmetric. Optionally pass a prebuilt unpacked tensor to avoid rebuilding
    // it when J and K share one (e.g. build_ri_fock_rhf).
    Eigen::MatrixXd build_ri_k(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &D,
        const std::vector<Eigen::MatrixXd> *unpacked = nullptr);

    // Closed-shell RI Fock contribution G = J - 1/2 K, matching
    // ObaraSaika::_compute_fock_rhf(eri, D) to density-fitting accuracy.
    Eigen::MatrixXd build_ri_fock_rhf(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &D);
} // namespace HartreeFock::Correlation::RI

#endif // HF_POST_HF_RI_ERI_H
