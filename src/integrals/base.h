#ifndef HF_INTEGRALS_H
#define HF_INTEGRALS_H

#include <utility>

#include "base/types.h"
#include "os.h"
#include "rys.h"

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_1e(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::IntegralMethod &engine,
    const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr)
{
    return HartreeFock::ObaraSaika::_compute_1e(shell_pairs, nbasis, sym_ops);
}

inline Eigen::MatrixXd _compute_nuclear_attraction(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::Molecule &molecule,
    const HartreeFock::IntegralMethod &engine,
    const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr)
{
    return HartreeFock::ObaraSaika::_compute_nuclear_attraction(shell_pairs, nbasis, molecule, sym_ops);
}

inline Eigen::MatrixXd _compute_2e_fock(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                                         const Eigen::MatrixXd& density,
                                         const std::size_t nbasis,
                                         const HartreeFock::IntegralMethod& engine,
                                         double tol_eri = 1e-10,
                                         const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr)
{
    switch (engine) {
        case HartreeFock::IntegralMethod::RysQuadrature:
            return HartreeFock::RysQuad::_compute_2e_fock(shell_pairs, density, nbasis, tol_eri, sym_ops);
        case HartreeFock::IntegralMethod::Auto:
            return HartreeFock::RysQuad::_compute_2e_fock_auto(shell_pairs, density, nbasis, tol_eri, sym_ops);
        default:
            return HartreeFock::ObaraSaika::_compute_2e_fock(shell_pairs, density, nbasis, tol_eri, sym_ops);
    }
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
_compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair>& shell_pairs,
                     const Eigen::MatrixXd& Pa,
                     const Eigen::MatrixXd& Pb,
                     const std::size_t nbasis,
                     const HartreeFock::IntegralMethod& engine,
                     double tol_eri = 1e-10,
                     const std::vector<HartreeFock::SignedAOSymOp>* sym_ops = nullptr)
{
    switch (engine) {
        case HartreeFock::IntegralMethod::RysQuadrature:
            return HartreeFock::RysQuad::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, tol_eri, sym_ops);
        case HartreeFock::IntegralMethod::Auto:
            return HartreeFock::RysQuad::_compute_2e_fock_uhf_auto(shell_pairs, Pa, Pb, nbasis, tol_eri, sym_ops);
        default:
            return HartreeFock::ObaraSaika::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, tol_eri, sym_ops);
    }
}

#endif // !HF_INTEGRALS_H
