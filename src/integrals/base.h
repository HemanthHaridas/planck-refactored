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
                                        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr,
                                        HartreeFock::Integrals::FusedTerm term =
                                            HartreeFock::Integrals::FusedTerm::Combined)
{
    // The Fock-build wrappers follow the same dispatch policy as the raw ERI
    // tensor path so higher layers can swap engines without changing algebra.
    switch (engine)
    {
    case HartreeFock::IntegralMethod::RysQuadrature:
        return HartreeFock::RysQuad::_compute_2e_fock_direct(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops, term);
    case HartreeFock::IntegralMethod::HeadGordonPople:
        return HartreeFock::HeadGordonPople::_compute_2e_fock_direct(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops, term);
    case HartreeFock::IntegralMethod::Auto:
        return HartreeFock::RysQuad::_compute_2e_fock_auto_direct(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops, term);
    default:
        // OS (the default engine) uses the MEMORY-DIRECT builder: it contracts
        // each quartet straight into G and never allocates the nb^4 tensor. The
        // two-phase ObaraSaika::_compute_2e_fock allocated the full tensor on
        // every SCF iteration (0.8 GB at nb=100, 500 GB at nb=500), so "direct"
        // mode cost more memory than conventional, not less.
        //
        // Every case above is memory-direct too: Rys / HGP / Auto gained fused
        // builders in 2b23971, and 802168b made the fused entry handle integral
        // symmetry (sym_ops) natively rather than delegating back to two-phase.
        // So all four engines share one loop here — the switch selects only the
        // per-quartet ERI kernel.
        return HartreeFock::ObaraSaika::_compute_2e_fock_direct(shell_pairs, density, nbasis, kernel, omega, tol_eri, sym_ops, term);
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
                     const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr,
                     HartreeFock::Integrals::FusedTerm term =
                         HartreeFock::Integrals::FusedTerm::Combined)
{
    // UHF reuses the same engine selection, but returns separate alpha/beta
    // effective Fock contributions from one spin-coupled density pair.
    switch (engine)
    {
    case HartreeFock::IntegralMethod::RysQuadrature:
        return HartreeFock::RysQuad::_compute_2e_fock_uhf_direct(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops, term);
    case HartreeFock::IntegralMethod::HeadGordonPople:
        return HartreeFock::HeadGordonPople::_compute_2e_fock_uhf_direct(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops, term);
    case HartreeFock::IntegralMethod::Auto:
        return HartreeFock::RysQuad::_compute_2e_fock_uhf_auto_direct(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops, term);
    default:
        // Memory-direct: see the RHF dispatcher above.
        return HartreeFock::ObaraSaika::_compute_2e_fock_uhf_direct(shell_pairs, Pa, Pb, nbasis, kernel, omega, tol_eri, sym_ops, term);
    }
}

// ── Single-term memory-direct builds (the DFT entries) ──────────────────────
//
// HF wants G = J - 0.5K and calls the dispatchers above. DFT cannot: it needs J
// alone, K alone scaled by exact_exchange_coefficient, and — for range-separated
// functionals — two K's at different omega added to one J.
//
// These are the same dispatchers with the term selected, so DFT inherits the
// whole fused loop: block-level Schwarz, the fixed-order OpenMP reduction, the
// MPI bra-stripe, and native sym_ops handling. No nb^4 tensor is allocated.
//
// Both return RAW J / RAW K — no coefficient, no sign. The caller applies its
// own (DFT: -0.5 for RKS, -1 for UKS, times the exchange coefficient). See the
// prefactor contract in fock_accumulate.h.

inline Eigen::MatrixXd
_compute_2e_j_direct(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                     const Eigen::MatrixXd &density,
                     const std::size_t nbasis,
                     const HartreeFock::IntegralMethod &engine,
                     HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                     double omega = 0.0,
                     double tol_eri = 1e-10,
                     const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    return _compute_2e_fock(shell_pairs, density, nbasis, engine, kernel, omega,
                            tol_eri, sym_ops,
                            HartreeFock::Integrals::FusedTerm::CoulombOnly);
}

inline Eigen::MatrixXd
_compute_2e_k_direct(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                     const Eigen::MatrixXd &density,
                     const std::size_t nbasis,
                     const HartreeFock::IntegralMethod &engine,
                     HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                     double omega = 0.0,
                     double tol_eri = 1e-10,
                     const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    return _compute_2e_fock(shell_pairs, density, nbasis, engine, kernel, omega,
                            tol_eri, sym_ops,
                            HartreeFock::Integrals::FusedTerm::ExchangeOnly);
}

// Spin-resolved exchange: K_alpha and K_beta from ONE quartet sweep, which is
// what a UKS hybrid needs (the RKS path can just call _compute_2e_k_direct
// twice, but UKS would then pay the sweep twice for no reason).
inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
_compute_2e_k_uhf_direct(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                         const Eigen::MatrixXd &Pa,
                         const Eigen::MatrixXd &Pb,
                         const std::size_t nbasis,
                         const HartreeFock::IntegralMethod &engine,
                         HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
                         double omega = 0.0,
                         double tol_eri = 1e-10,
                         const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr)
{
    return _compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, engine, kernel,
                                omega, tol_eri, sym_ops,
                                HartreeFock::Integrals::FusedTerm::ExchangeOnly);
}

#endif // !HF_INTEGRALS_H
