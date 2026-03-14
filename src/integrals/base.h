#ifndef HF_INTEGRALS_H
#define HF_INTEGRALS_H

#include <utility>

#include "base/types.h"
#include "os.h"

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_1e(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis, const HartreeFock::IntegralMethod &engine)
{
    return HartreeFock::ObaraSaika::_compute_1e(shell_pairs, nbasis);
}

inline Eigen::MatrixXd _compute_nuclear_attraction(const std::vector<HartreeFock::ShellPair> &shell_pairs, const std::size_t nbasis, const HartreeFock::Molecule &molecule, const HartreeFock::IntegralMethod &engine)
{
    return HartreeFock::ObaraSaika::_compute_nuclear_attraction(shell_pairs, nbasis, molecule);
}

inline Eigen::MatrixXd _compute_2e_fock(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                         const Eigen::MatrixXd& density,
                                         const std::size_t nbasis,
                                         const HartreeFock::IntegralMethod& engine,
                                         double tol_eri = 1e-10)
{
    return HartreeFock::ObaraSaika::_compute_2e_fock(shell_pairs, density, nbasis, tol_eri);
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
_compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                     const Eigen::MatrixXd& Pa,
                     const Eigen::MatrixXd& Pb,
                     const std::size_t nbasis,
                     const HartreeFock::IntegralMethod& engine,
                     double tol_eri = 1e-10)
{
    return HartreeFock::ObaraSaika::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, tol_eri);
}

#endif // !HF_INTEGRALS_H
