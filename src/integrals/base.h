#ifndef HF_INTEGRALS_H
#define HF_INTEGRALS_H

#include <utility>
#include <stdexcept>

#include "base/types.h"
#include "os.h"
#include "tho.h"

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_1e(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis, const HartreeFock::IntegralMethod &engine)
{
    switch (engine)
    {
        case HartreeFock::IntegralMethod::ObaraSaika:
            return HartreeFock::ObaraSaika::_compute_1e(shell_pairs, nbasis);

        case HartreeFock::IntegralMethod::Huzinaga:
            return HartreeFock::Huzinaga::_compute_1e(shell_pairs, nbasis);

        default:
            throw std::runtime_error("Unsupported integral engine");
    }
}

inline Eigen::MatrixXd _compute_nuclear_attraction(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis, const HartreeFock::Molecule &molecule, const HartreeFock::IntegralMethod &engine)
{
    switch (engine)
    {
        case HartreeFock::IntegralMethod::ObaraSaika:
            return HartreeFock::ObaraSaika::_compute_nuclear_attraction(shell_pairs, nbasis, molecule);

        case HartreeFock::IntegralMethod::Huzinaga:
            return HartreeFock::Huzinaga::_compute_nuclear_attraction(shell_pairs, nbasis, molecule);

        default:
            throw std::runtime_error("Unsupported integral engine");
    }
}

inline Eigen::MatrixXd _compute_2e_fock(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                         const Eigen::MatrixXd& density,
                                         const std::size_t nbasis,
                                         const HartreeFock::IntegralMethod& engine)
{
    switch (engine)
    {
        case HartreeFock::IntegralMethod::ObaraSaika:
            return HartreeFock::ObaraSaika::_compute_2e_fock(shell_pairs, density, nbasis);

        default:
            throw std::runtime_error("Unsupported integral engine for 2e Fock");
    }
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
_compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                     const Eigen::MatrixXd& Pa,
                     const Eigen::MatrixXd& Pb,
                     const std::size_t nbasis,
                     const HartreeFock::IntegralMethod& engine)
{
    switch (engine)
    {
        case HartreeFock::IntegralMethod::ObaraSaika:
            return HartreeFock::ObaraSaika::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis);

        case HartreeFock::IntegralMethod::Huzinaga:
            return HartreeFock::Huzinaga::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis);

        default:
            throw std::runtime_error("Unsupported integral engine for UHF 2e Fock");
    }
}

#endif // !HF_INTEGRALS_H
