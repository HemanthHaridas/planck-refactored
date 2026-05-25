#include "fock_symmetrization.h"

std::expected<Eigen::MatrixXd, std::string>
HartreeFock::Symmetry::symmetrize_matrix(const Eigen::MatrixXd &matrix,
                                         const GroupOperations &ops)
{
    if (!ops.valid || ops.operations.empty())
        return std::unexpected("symmetrize_matrix: invalid or empty group operations");

    const Eigen::Index n = matrix.rows();
    if (matrix.cols() != n)
        return std::unexpected("symmetrize_matrix: matrix must be square");

    for (const auto &op : ops.operations)
    {
        if (op.matrix.rows() != n || op.matrix.cols() != n)
            return std::unexpected(
                "symmetrize_matrix: operation '" + op.label +
                "' has shape incompatible with the matrix");
    }

    // F = (1/|G|) Σ_R O_Rᵀ M O_R  — the projection onto the totally-symmetric
    // component. O_R is orthogonal, so for a group-invariant M each term equals M
    // and the average returns M (fixed point); for a general M it averages away the
    // non-symmetric part.
    //
    // Item A (docs/FULL_SYMMETRY_PERF_SCOPE.md): when O_R is monomial (one ±1 per
    // column, classified at build time) the term reduces to a permute-with-signs,
    //   (O_Rᵀ M O_R)(μ,ν) = s_μ s_ν · M(map_μ, map_ν),
    // an O(nb²) accumulate instead of the dense O(nb³) matmul. Non-monomial ops
    // (C3/C4/S4/σ_d, and anything on Cartesian d⁺) take the matmul, unchanged. The
    // result is bitwise the same up to floating-point reassociation; gated by the
    // build-time check that the monomial form equals `matrix` (see classify_monomial).
    Eigen::MatrixXd accum = Eigen::MatrixXd::Zero(n, n);
    for (const auto &op : ops.operations)
    {
        if (op.is_monomial &&
            static_cast<Eigen::Index>(op.mono_map.size()) == n &&
            static_cast<Eigen::Index>(op.mono_sign.size()) == n)
        {
            const auto &map = op.mono_map;
            const auto &sgn = op.mono_sign;
            for (Eigen::Index mu = 0; mu < n; ++mu)
            {
                const Eigen::Index rmu = map[static_cast<std::size_t>(mu)];
                const double smu = static_cast<double>(sgn[static_cast<std::size_t>(mu)]);
                for (Eigen::Index nu = 0; nu < n; ++nu)
                    accum(mu, nu) += smu *
                                     static_cast<double>(sgn[static_cast<std::size_t>(nu)]) *
                                     matrix(rmu, map[static_cast<std::size_t>(nu)]);
            }
        }
        else
        {
            accum.noalias() += op.matrix.transpose() * matrix * op.matrix;
        }
    }

    accum /= static_cast<double>(ops.operations.size());
    return accum;
}
