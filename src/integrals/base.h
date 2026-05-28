#ifndef HF_INTEGRALS_H
#define HF_INTEGRALS_H

#include <utility>

#include "base/types.h"
#include "hgp.h"
#include "os.h"
#include "rys.h"

// Thin dispatch layer shared by SCF, DFT, gradients, and post-HF code. Callers
// ask for "the integral" they need and choose an engine policy here, while the
// concrete OS/Rys implementations stay behind one stable interface.

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> _compute_1e(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::IntegralMethod &engine,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    return HartreeFock::ObaraSaika::_compute_1e(shell_pairs, nbasis, sym_ops);
}

inline Eigen::MatrixXd _compute_nuclear_attraction(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::Molecule &molecule,
    const HartreeFock::IntegralMethod &engine,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    return HartreeFock::ObaraSaika::_compute_nuclear_attraction(shell_pairs, nbasis, molecule, sym_ops);
}

inline Eigen::MatrixXd _compute_external_charge_attraction(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const std::vector<HartreeFock::ExternalCharge> &charges,
    const HartreeFock::IntegralMethod &engine,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    return HartreeFock::ObaraSaika::_compute_external_charge_attraction(shell_pairs, nbasis, charges, sym_ops);
}

inline std::vector<double> _compute_2e(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::IntegralMethod &engine,
    HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
    double omega = 0.0,
    double tol_eri = 1e-10,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    // Rys handles the higher-angular-momentum and range-separated paths more
    // robustly, while OS remains the default explicit recurrence engine. Auto
    // keeps that choice centralized instead of scattered across callers.
    switch (engine)
    {
    case HartreeFock::IntegralMethod::RysQuadrature:
        return HartreeFock::RysQuad::_compute_2e(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    case HartreeFock::IntegralMethod::HeadGordonPople:
        return HartreeFock::HeadGordonPople::_compute_2e(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    case HartreeFock::IntegralMethod::Auto:
        return HartreeFock::RysQuad::_compute_2e_auto(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    default:
        return HartreeFock::ObaraSaika::_compute_2e(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    }
}

inline Eigen::MatrixXd _compute_2e_fock(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                        const Eigen::MatrixXd &density,
                                        const std::size_t nbasis,
                                        const HartreeFock::IntegralMethod &engine,
                                        HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                                        double omega = 0.0,
                                        double tol_eri = 1e-10,
                                        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    // The Fock-build wrappers follow the same dispatch policy as the raw ERI
    // tensor path so higher layers can swap engines without changing algebra.
    switch (engine)
    {
    case HartreeFock::IntegralMethod::RysQuadrature:
        return HartreeFock::RysQuad::_compute_2e_fock(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops);
    case HartreeFock::IntegralMethod::HeadGordonPople:
        return HartreeFock::HeadGordonPople::_compute_2e_fock(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops);
    case HartreeFock::IntegralMethod::Auto:
        return HartreeFock::RysQuad::_compute_2e_fock_auto(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops);
    default:
        return HartreeFock::ObaraSaika::_compute_2e_fock(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops);
    }
}

inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
_compute_2e_fock_uhf(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                     const Eigen::MatrixXd &Pa,
                     const Eigen::MatrixXd &Pb,
                     const std::size_t nbasis,
                     const HartreeFock::IntegralMethod &engine,
                     HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                     double omega = 0.0,
                     double tol_eri = 1e-10,
                     const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    // UHF reuses the same engine selection, but returns separate alpha/beta
    // effective Fock contributions from one spin-coupled density pair.
    switch (engine)
    {
    case HartreeFock::IntegralMethod::RysQuadrature:
        return HartreeFock::RysQuad::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops);
    case HartreeFock::IntegralMethod::HeadGordonPople:
        return HartreeFock::HeadGordonPople::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops);
    case HartreeFock::IntegralMethod::Auto:
        return HartreeFock::RysQuad::_compute_2e_fock_uhf_auto(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops);
    default:
        return HartreeFock::ObaraSaika::_compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops);
    }
}

#endif // !HF_INTEGRALS_H
