#ifndef HF_POSTHF_CC_DIIS_H
#define HF_POSTHF_CC_DIIS_H

#include <Eigen/Core>
#include <Eigen/QR>

#include <deque>
#include <expected>
#include <string>

namespace HartreeFock::Correlation::CC
{
    // Minimal DIIS helper for flattened amplitude vectors. The implementation is
    // intentionally small and mirrors the SCF DIIS code so students can compare
    // the two acceleration patterns directly.
    class AmplitudeDIIS
    {
    public:
        explicit AmplitudeDIIS(int max_vecs = 8);

        void push(const Eigen::VectorXd &amplitudes,
                  const Eigen::VectorXd &error_vector);

        [[nodiscard]] bool ready() const noexcept;
        [[nodiscard]] std::size_t size() const noexcept;

        std::expected<Eigen::VectorXd, std::string> extrapolate() const;

        void clear() noexcept;

    private:
        std::deque<Eigen::VectorXd> amplitude_history_;
        std::deque<Eigen::VectorXd> error_history_;
        std::size_t max_vecs_ = 8;
    };
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_DIIS_H
