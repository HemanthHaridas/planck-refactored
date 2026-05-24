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
    Eigen::MatrixXd accum = Eigen::MatrixXd::Zero(n, n);
    for (const auto &op : ops.operations)
        accum.noalias() += op.matrix.transpose() * matrix * op.matrix;

    accum /= static_cast<double>(ops.operations.size());
    return accum;
}
