#include "post_hf/cc/diis.h"

namespace HartreeFock::Correlation::CC
{
    AmplitudeDIIS::AmplitudeDIIS(int max_vecs)
        : max_vecs_(max_vecs > 0 ? static_cast<std::size_t>(max_vecs) : 1u)
    {
    }

    void AmplitudeDIIS::push(const Eigen::VectorXd &amplitudes,
                             const Eigen::VectorXd &error_vector)
    {
        amplitude_history_.push_back(amplitudes);
        error_history_.push_back(error_vector);

        if (amplitude_history_.size() > max_vecs_)
        {
            amplitude_history_.pop_front();
            error_history_.pop_front();
        }
    }

    bool AmplitudeDIIS::ready() const noexcept
    {
        return amplitude_history_.size() >= 2;
    }

    std::size_t AmplitudeDIIS::size() const noexcept
    {
        return amplitude_history_.size();
    }

    std::expected<Eigen::VectorXd, std::string> AmplitudeDIIS::extrapolate() const
    {
        if (!ready())
            return std::unexpected("AmplitudeDIIS::extrapolate: need at least two stored vectors.");

        // The augmented DIIS system is the standard Pulay least-squares problem.
        // Keeping it explicit makes the algebra easy to compare with the SCF DIIS
        // implementation elsewhere in the codebase.
        const Eigen::Index m = static_cast<Eigen::Index>(amplitude_history_.size());
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m + 1, m + 1);
        for (Eigen::Index i = 0; i < m; ++i)
        {
            for (Eigen::Index j = i; j < m; ++j)
            {
                const double bij = error_history_[static_cast<std::size_t>(i)].dot(
                    error_history_[static_cast<std::size_t>(j)]);
                B(i, j) = bij;
                B(j, i) = bij;
            }
            B(i, m) = -1.0;
            B(m, i) = -1.0;
        }

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1);
        rhs(m) = -1.0;

        const Eigen::VectorXd coeffs = B.colPivHouseholderQr().solve(rhs);
        if ((B * coeffs - rhs).norm() > 1e-8)
            return std::unexpected("AmplitudeDIIS::extrapolate: DIIS linear solve did not converge.");

        Eigen::VectorXd out = Eigen::VectorXd::Zero(amplitude_history_.front().size());
        for (Eigen::Index i = 0; i < m; ++i)
            out += coeffs(i) * amplitude_history_[static_cast<std::size_t>(i)];

        return out;
    }

    void AmplitudeDIIS::clear() noexcept
    {
        amplitude_history_.clear();
        error_history_.clear();
    }
} // namespace HartreeFock::Correlation::CC
