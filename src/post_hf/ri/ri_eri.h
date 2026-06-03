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

    std::expected<void, std::string> ensure_ri_3c_ready(
        HartreeFock::Calculator &calculator);
} // namespace HartreeFock::Correlation::RI

#endif // HF_POST_HF_RI_ERI_H
