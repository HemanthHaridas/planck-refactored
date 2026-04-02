#ifndef HF_POSTHF_CASSCF_INTERNAL_H
#define HF_POSTHF_CASSCF_INTERNAL_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace HartreeFock::Correlation::CASSCFInternal
{

using CIString = std::uint64_t;

// CI strings are encoded as bit patterns, so the bit-width of the host type
// directly limits the number of active orbitals we can represent.
inline constexpr int kCIStringBits = std::numeric_limits<CIString>::digits;
inline constexpr int kMaxSeparateSpinOrbitals = kCIStringBits - 1;
inline constexpr int kMaxPackedSpatialOrbitals = (kCIStringBits - 1) / 2;

inline CIString single_bit_mask(int bit)
{
    if (bit < 0 || bit >= kCIStringBits) return 0;
    return CIString(1) << bit;
}

inline CIString low_bit_mask(int nbits)
{
    if (nbits <= 0) return 0;
    if (nbits >= kCIStringBits) return std::numeric_limits<CIString>::max();
    return (CIString(1) << nbits) - 1;
}

// RAS parameters are shared between the public driver and the determinant
// builders, so this struct carries both the partitioning and the screening caps.
struct RASParams
{
    int nras1 = 0, nras2 = 0, nras3 = 0;
    int max_holes = 100, max_elec = 100;
    bool active = false;
};

// RASSCF constraints are enforced on the combined alpha+beta determinant.
// max_holes counts the total number of electrons missing from a doubly
// occupied RAS1 block; max_elec counts the total number of electrons in RAS3.
inline int ras1_holes(CIString alpha, CIString beta, const RASParams& ras)
{
    const CIString ras1_mask = low_bit_mask(ras.nras1);
    const int occ_alpha = std::popcount(alpha & ras1_mask);
    const int occ_beta = std::popcount(beta & ras1_mask);
    return 2 * ras.nras1 - (occ_alpha + occ_beta);
}

inline int ras3_electrons(CIString alpha, CIString beta, const RASParams& ras)
{
    const int ras3_offset = ras.nras1 + ras.nras2;
    const CIString ras3_mask = low_bit_mask(ras.nras3) << ras3_offset;
    return std::popcount(alpha & ras3_mask) + std::popcount(beta & ras3_mask);
}

inline bool admissible_ras_pair(CIString alpha, CIString beta, const RASParams& ras)
{
    if (!ras.active) return true;
    return ras1_holes(alpha, beta, ras) <= ras.max_holes
        && ras3_electrons(alpha, beta, ras) <= ras.max_elec;
}

// Minimal symmetry metadata needed by the CI determinant selector and root
// tracking. The product table is assumed to be Abelian and indexed by the
// order of `names`.
struct SymmetryContext
{
    std::vector<std::string> names;
    std::vector<std::vector<int>> product;
    bool abelian_1d_only = false;
    int totally_symmetric_irrep = 0;
};

// The active-integral cache stores the one expensive mixed-basis transform used
// repeatedly by Q-matrix contractions across a macroiteration.
struct ActiveIntegralCache
{
    std::vector<double> puvw;
    int nbasis = 0;
    int nact = 0;
    bool valid = false;
};

// Contract the cached mixed-basis two-electron tensor with the active-space
// 2-RDM. The layout is row-major and the tensor sizes must agree exactly.
inline Eigen::MatrixXd contract_q_matrix(
    const std::vector<double>& puvw,
    const std::vector<double>& Gamma,
    int nbasis,
    int nact)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nbasis, nact);
    if (nbasis <= 0 || nact <= 0) return Q;

    const std::size_t expected_puvw = static_cast<std::size_t>(nbasis) * nact * nact * nact;
    const std::size_t expected_gamma = static_cast<std::size_t>(nact) * nact * nact * nact;
    if (puvw.size() != expected_puvw || Gamma.size() != expected_gamma) return Q;

    for (int p = 0; p < nbasis; ++p)
    for (int t = 0; t < nact; ++t)
    {
        double q_pt = 0.0;
        for (int u = 0; u < nact; ++u)
        for (int v = 0; v < nact; ++v)
        for (int w = 0; w < nact; ++w)
            q_pt += Gamma[((t * nact + u) * nact + v) * nact + w]
                 * puvw[((p * nact + u) * nact + v) * nact + w];
        Q(p, t) = q_pt;
    }

    return Q;
}

struct NaturalOrbitalData
{
    Eigen::VectorXd occupations;
    Eigen::MatrixXd rotation;
};

// The natural-orbital rotation diagonalizes the active-space 1-RDM; the
// eigenvalues are returned in descending occupation order to match reporting.
inline NaturalOrbitalData diagonalize_natural_orbitals(const Eigen::MatrixXd& gamma)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(gamma);
    NaturalOrbitalData result;
    result.occupations = eig.eigenvalues().reverse();
    result.rotation = eig.eigenvectors().rowwise().reverse().eval();
    return result;
}

inline int determinant_symmetry(
    CIString alpha,
    CIString beta,
    const std::vector<int>& irr_act,
    const SymmetryContext& sym_ctx)
{
    // Start from the totally symmetric irrep and fold in each occupied active
    // orbital's label for both spin sectors.
    int sym = sym_ctx.totally_symmetric_irrep;
    for (int t = 0; t < static_cast<int>(irr_act.size()); ++t)
    {
        if (irr_act[t] < 0) continue;
        if (alpha & single_bit_mask(t)) sym = sym_ctx.product[sym][irr_act[t]];
        if (beta  & single_bit_mask(t)) sym = sym_ctx.product[sym][irr_act[t]];
    }
    return sym;
}

inline Eigen::MatrixXd compute_root_overlap(
    const Eigen::MatrixXd& c_old,
    const Eigen::MatrixXd& c_new)
{
    if (c_old.rows() == 0 || c_new.rows() == 0 || c_old.rows() != c_new.rows())
        return Eigen::MatrixXd();
    return (c_old.adjoint() * c_new).cwiseAbs();
}

namespace detail
{

inline std::vector<int> hungarian_max_assignment(const Eigen::MatrixXd& weights)
{
    const int nrows = static_cast<int>(weights.rows());
    const int ncols = static_cast<int>(weights.cols());
    std::vector<int> assignment(nrows, -1);

    if (nrows <= 0)
        return assignment;
    if (ncols <= 0)
        return assignment;

    const int n = std::max(nrows, ncols);
    Eigen::MatrixXd square = Eigen::MatrixXd::Zero(n, n);
    square.topLeftCorner(nrows, ncols) = weights.cwiseAbs();

    const double max_weight = square.topLeftCorner(nrows, ncols).maxCoeff();
    Eigen::MatrixXd cost = Eigen::MatrixXd::Constant(n, n, max_weight);
    cost.topLeftCorner(nrows, ncols).array() -= square.topLeftCorner(nrows, ncols).array();

    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0), minv(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);
    std::vector<char> used(n + 1, 0);

    for (int i = 1; i <= n; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), std::numeric_limits<double>::infinity());
        std::fill(used.begin(), used.end(), 0);

        do
        {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;

            for (int j = 1; j <= n; ++j)
            {
                if (used[j]) continue;
                const double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (int j = 0; j <= n; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        }
        while (p[j0] != 0);

        do
        {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
        while (j0 != 0);
    }

    for (int j = 1; j <= n; ++j)
    {
        const int i = p[j];
        if (i <= 0 || i > nrows || j > ncols)
            continue;
        assignment[i - 1] = j - 1;
    }

    return assignment;
}

} // namespace detail

inline std::vector<int> match_roots_by_max_overlap(const Eigen::MatrixXd& overlaps)
{
    // Use a maximum-weight assignment so each previous root is paired with the
    // most similar current root without reusing a new state twice.
    return detail::hungarian_max_assignment(overlaps);
}

// Project a response vector orthogonally to the reference CI root so the
// first-order correction stays in the tangent space of the normalized state.
inline Eigen::VectorXd project_orthogonal(
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& c0)
{
    if (v.size() != c0.size() || c0.size() == 0)
        return v;
    return v - c0.dot(v) * c0;
}

inline Eigen::VectorXd apply_response_diag_preconditioner(
    const Eigen::VectorXd& rhs,
    const Eigen::VectorXd& H_diag,
    double E0,
    double precond_floor,
    double& max_denominator_regularization)
{
    // The diagonal preconditioner is clipped away from zero to avoid exploding
    // corrections when an orbital or CI denominator becomes nearly singular.
    Eigen::VectorXd step = Eigen::VectorXd::Zero(rhs.size());
    for (int i = 0; i < rhs.size(); ++i)
    {
        double denom = H_diag(i) - E0;
        const double abs_denom = std::abs(denom);
        if (abs_denom < precond_floor)
        {
            max_denominator_regularization = std::max(
                max_denominator_regularization,
                precond_floor - abs_denom);
            denom = (denom >= 0.0) ? precond_floor : -precond_floor;
        }
        step(i) = rhs(i) / denom;
    }
    return step;
}

struct CIResponseResult
{
    Eigen::VectorXd c1;
    double residual_norm = std::numeric_limits<double>::infinity();
    int iterations = 0;
    double max_denominator_regularization = 0.0;
    bool converged = false;
};

// Residual for the linearized CI response equation after projecting both the
// right-hand side and the iterate back into the orthogonal complement of c0.
inline Eigen::VectorXd compute_ci_response_residual(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& c1,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& sigma)
{
    Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
    Eigen::VectorXd residual = rhs - (H * c1 - E0 * c1);
    return project_orthogonal(residual, c0);
}

inline CIResponseResult ci_response_diag_precond_single_step(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& H_diag,
    const Eigen::VectorXd& sigma,
    double precond_floor = 1e-4)
{
    // Single-step fallback: one preconditioned update from the projected
    // residual, mainly used when the iterative solver cannot converge.
    CIResponseResult result;
    result.c1 = Eigen::VectorXd::Zero(c0.size());

    const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
    result.c1 = apply_response_diag_preconditioner(
        rhs, H_diag, E0, precond_floor, result.max_denominator_regularization);
    result.c1 = project_orthogonal(result.c1, c0);
    result.residual_norm = compute_ci_response_residual(H, result.c1, c0, E0, sigma).norm();
    result.iterations = 1;
    result.converged = false;
    return result;
}

inline CIResponseResult solve_ci_response_iterative(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& H_diag,
    const Eigen::VectorXd& sigma,
    double tol = 1e-8,
    int max_iter = 32,
    double precond_floor = 1e-4)
{
    // Restartable Davidson-style linear solver for the first-order CI response.
    // The solver keeps the best finite iterate even if convergence stalls.
    CIResponseResult result;
    result.c1 = Eigen::VectorXd::Zero(c0.size());

    if (H.rows() != H.cols() || H.rows() != c0.size() || H.rows() != H_diag.size()
        || H.rows() != sigma.size())
        return result;

    const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
    result.residual_norm = rhs.norm();
    if (!std::isfinite(result.residual_norm))
        return result;
    if (result.residual_norm < tol)
    {
        result.converged = true;
        return result;
    }

    Eigen::VectorXd guess = apply_response_diag_preconditioner(
        rhs, H_diag, E0, precond_floor, result.max_denominator_regularization);
    guess = project_orthogonal(guess, c0);
    const double guess_norm = guess.norm();
    if (!(guess_norm > 1e-14))
        return result;

    Eigen::MatrixXd V(guess.size(), 1);
    V.col(0) = guess / guess_norm;

    for (int iter = 1; iter <= max_iter; ++iter)
    {
        const Eigen::MatrixXd AV = H * V - E0 * V;
        const Eigen::MatrixXd M = V.transpose() * AV;
        const Eigen::VectorXd b = V.transpose() * rhs;
        const Eigen::VectorXd y = M.colPivHouseholderQr().solve(b);
        if (!y.allFinite())
            break;

        result.c1 = project_orthogonal(V * y, c0);
        Eigen::VectorXd residual = rhs - (H * result.c1 - E0 * result.c1);
        residual = project_orthogonal(residual, c0);
        result.residual_norm = residual.norm();
        result.iterations = iter;
        if (!std::isfinite(result.residual_norm))
            break;
        if (result.residual_norm < tol)
        {
            result.converged = true;
            return result;
        }

        Eigen::VectorXd correction = apply_response_diag_preconditioner(
            residual, H_diag, E0, precond_floor, result.max_denominator_regularization);
        correction = project_orthogonal(correction, c0);
        for (int k = 0; k < V.cols(); ++k)
            correction -= V.col(k).dot(correction) * V.col(k);
        for (int k = 0; k < V.cols(); ++k)
            correction -= V.col(k).dot(correction) * V.col(k);

        const double corr_norm = correction.norm();
        if (!(corr_norm > 1e-14))
            break;

        const int m = static_cast<int>(V.cols());
        V.conservativeResize(Eigen::NoChange, m + 1);
        V.col(m) = correction / corr_norm;
    }

    return result;
}

} // namespace HartreeFock::Correlation::CASSCFInternal

#endif // HF_POSTHF_CASSCF_INTERNAL_H
