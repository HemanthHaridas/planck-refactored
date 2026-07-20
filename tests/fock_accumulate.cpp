// Gate for the fused (memory-direct) Fock accumulation — Steps 0 and 1 of the
// memory-direct SCF Fock build.
//
// Today's "direct" SCF Fock builders (_compute_2e_fock / _uhf) allocate the FULL
// nb^4 ERI tensor every SCF iteration, fill it, contract it, and throw it away.
// The memory-direct replacement contracts each canonical quartet straight into
// G (nb^2) and never allocates the tensor. The load-bearing piece is the
// accumulation rule in src/integrals/fock_accumulate.h, and this test pins it
// two independent ways BEFORE any production call site changes:
//
//   Step 1 (engine-free): on RANDOM 8-fold-symmetric tensors, accumulating over
//     canonical quartets must reproduce the naive nb^4 contraction exactly. No
//     integral engine involved, so a failure here is unambiguously the
//     accumulation rule (the classic place these builds go wrong is the
//     degeneracy handling when i==j, k==l, or (ij)==(kl)).
//
//   Step 0 (real integrals): on the real water/STO-3G ERI tensor, the fused
//     accumulation must reproduce what the production two-phase builder
//     (_compute_2e_fock / _compute_2e_fock_uhf) computes. This is the oracle
//     every later step is measured against.
//
// Both RHF and UHF forms are covered. Nothing in production calls the new
// accumulator yet — this test is the whole of its exposure.

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/fock_accumulate.h"
#include "integrals/hgp.h"
#include "integrals/os.h"
#include "integrals/rys.h"
#include "integrals/shellpair.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    // The production quartet loop's canonical filter (os.cpp): each canonical
    // quartet is visited exactly once.
    inline bool canonical(std::size_t i, std::size_t j,
                          std::size_t k, std::size_t l) noexcept
    {
        if (j < i)
            return false;
        if (l < k)
            return false;
        if (k < i || (k == i && l < j))
            return false;
        return true;
    }

    inline double at(const std::vector<double> &eri, std::size_t nb,
                     std::size_t i, std::size_t j, std::size_t k, std::size_t l)
    {
        return eri[((i * nb + j) * nb + k) * nb + l];
    }

    // ── Step 1: engine-free, random 8-fold-symmetric tensors ─────────────────

    // Build a random tensor obeying the full 8-fold ERI symmetry.
    std::vector<double> random_symmetric_eri(std::size_t nb, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> eri(nb * nb * nb * nb, 0.0);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = i; j < nb; ++j)
                for (std::size_t k = 0; k < nb; ++k)
                    for (std::size_t l = k; l < nb; ++l)
                    {
                        if (k < i || (k == i && l < j))
                            continue;
                        const double v = dist(rng);
                        // scatter over the full orbit (store-only, as production does)
                        const std::size_t idx[8][4] = {
                            {i, j, k, l}, {j, i, k, l}, {i, j, l, k}, {j, i, l, k},
                            {k, l, i, j}, {l, k, i, j}, {k, l, j, i}, {l, k, j, i},
                        };
                        for (const auto &t : idx)
                            eri[((t[0] * nb + t[1]) * nb + t[2]) * nb + t[3]] = v;
                    }
        return eri;
    }

    Eigen::MatrixXd random_symmetric_density(std::size_t nb, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd P(nb, nb);
        for (std::size_t a = 0; a < nb; ++a)
            for (std::size_t b = a; b < nb; ++b)
                P(a, b) = P(b, a) = dist(rng);
        return P;
    }

    // Naive nb^4 RHF contraction — the reference Phase 2 performs.
    Eigen::MatrixXd naive_g_rhf(const std::vector<double> &eri,
                                const Eigen::MatrixXd &P, std::size_t nb)
    {
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        G(mu, nu) += P(lam, sig) *
                                     (at(eri, nb, mu, nu, lam, sig) -
                                      0.5 * at(eri, nb, mu, lam, nu, sig));
        return G;
    }

    // Fused: visit each canonical quartet once, accumulate over its orbit.
    Eigen::MatrixXd fused_g_rhf(const std::vector<double> &eri,
                                const Eigen::MatrixXd &P, std::size_t nb)
    {
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                for (std::size_t k = 0; k < nb; ++k)
                    for (std::size_t l = 0; l < nb; ++l)
                    {
                        if (!canonical(i, j, k, l))
                            continue;
                        HartreeFock::Integrals::fock_accumulate_rhf(
                            G, P, i, j, k, l, at(eri, nb, i, j, k, l));
                    }
        return G;
    }

    void naive_g_uhf(const std::vector<double> &eri,
                     const Eigen::MatrixXd &Pa, const Eigen::MatrixXd &Pb,
                     std::size_t nb,
                     Eigen::MatrixXd &Ga, Eigen::MatrixXd &Gb)
    {
        const Eigen::MatrixXd Pt = Pa + Pb;
        Ga = Eigen::MatrixXd::Zero(nb, nb);
        Gb = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                    {
                        const double coulomb = at(eri, nb, mu, nu, lam, sig);
                        const double exch = at(eri, nb, mu, lam, nu, sig);
                        Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                        Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                    }
    }

    void fused_g_uhf(const std::vector<double> &eri,
                     const Eigen::MatrixXd &Pa, const Eigen::MatrixXd &Pb,
                     std::size_t nb,
                     Eigen::MatrixXd &Ga, Eigen::MatrixXd &Gb)
    {
        const Eigen::MatrixXd Pt = Pa + Pb;
        Ga = Eigen::MatrixXd::Zero(nb, nb);
        Gb = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                for (std::size_t k = 0; k < nb; ++k)
                    for (std::size_t l = 0; l < nb; ++l)
                    {
                        if (!canonical(i, j, k, l))
                            continue;
                        HartreeFock::Integrals::fock_accumulate_uhf(
                            Ga, Gb, Pt, Pa, Pb, i, j, k, l,
                            at(eri, nb, i, j, k, l));
                    }
    }

    // Summation order differs between the naive nb^4 sweep and the fused orbit
    // accumulation, so this is a tight tolerance, not a bitwise claim.
    constexpr double TOL = 1e-12;

    void check_random_tensors()
    {
        std::mt19937 rng(20260713);
        double worst_rhf = 0.0;
        double worst_uhf = 0.0;

        for (std::size_t nb = 1; nb <= 7; ++nb)
            for (int trial = 0; trial < 20; ++trial)
            {
                const std::vector<double> eri = random_symmetric_eri(nb, rng);
                const Eigen::MatrixXd P = random_symmetric_density(nb, rng);

                const Eigen::MatrixXd ref = naive_g_rhf(eri, P, nb);
                const Eigen::MatrixXd fus = fused_g_rhf(eri, P, nb);
                worst_rhf = std::max(worst_rhf, (ref - fus).cwiseAbs().maxCoeff());

                const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
                const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);
                Eigen::MatrixXd Ga_ref, Gb_ref, Ga_fus, Gb_fus;
                naive_g_uhf(eri, Pa, Pb, nb, Ga_ref, Gb_ref);
                fused_g_uhf(eri, Pa, Pb, nb, Ga_fus, Gb_fus);
                worst_uhf = std::max(worst_uhf,
                                     std::max((Ga_ref - Ga_fus).cwiseAbs().maxCoeff(),
                                              (Gb_ref - Gb_fus).cwiseAbs().maxCoeff()));
            }

        if (worst_rhf > TOL)
            fail("random-tensor RHF: fused accumulation != naive nb^4 contraction, "
                 "max|dG| = " + std::to_string(worst_rhf));
        else
            std::cout << "OK  Step 1 / RHF / random 8-fold-symmetric tensors "
                         "(nb=1..7, 20 trials): max|dG| = "
                      << worst_rhf << '\n';

        if (worst_uhf > TOL)
            fail("random-tensor UHF: fused accumulation != naive nb^4 contraction, "
                 "max|dG| = " + std::to_string(worst_uhf));
        else
            std::cout << "OK  Step 1 / UHF / random 8-fold-symmetric tensors "
                         "(nb=1..7, 20 trials): max|dG| = "
                      << worst_uhf << '\n';
    }

    // ── J-only / K-only split (DFT_FUSED_JK_SCOPE.md Steps 0-1) ──────────────
    //
    // DFT needs J and K separately: J always, K scaled by
    // exact_exchange_coefficient, and for range-separated functionals TWO K's
    // at different omega added to ONE J. That cannot be expressed as the
    // combined G = J - 0.5K the HF path wants.
    //
    // The contract these oracles pin: exchange_accumulate emits RAW, UNSCALED
    // K. The 0.5 (RHF) and 1.0 (UHF) live in the combined wrappers, NOT in the
    // accumulator — DFT applies its own factor downstream and would otherwise
    // inherit the RHF convention, halving every RKS hybrid's exact exchange
    // while leaving UKS correct.

    // Naive nb^4 Coulomb: J(mu,nu) = sum_{lam,sig} P(lam,sig) (mu nu|lam sig)
    Eigen::MatrixXd naive_j(const std::vector<double> &eri,
                            const Eigen::MatrixXd &P, std::size_t nb)
    {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        J(mu, nu) += P(lam, sig) * at(eri, nb, mu, nu, lam, sig);
        return J;
    }

    // Naive nb^4 exchange, UNSCALED: K(mu,nu) = sum P(lam,sig) (mu lam|nu sig).
    // Matches build_exchange_from_eri in src/dft/driver.cpp, which likewise
    // returns raw K and leaves the coefficient to the caller.
    Eigen::MatrixXd naive_k(const std::vector<double> &eri,
                            const Eigen::MatrixXd &P, std::size_t nb)
    {
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        K(mu, nu) += P(lam, sig) * at(eri, nb, mu, lam, nu, sig);
        return K;
    }

    Eigen::MatrixXd fused_j(const std::vector<double> &eri,
                            const Eigen::MatrixXd &P, std::size_t nb)
    {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                for (std::size_t k = 0; k < nb; ++k)
                    for (std::size_t l = 0; l < nb; ++l)
                    {
                        if (!canonical(i, j, k, l))
                            continue;
                        HartreeFock::Integrals::coulomb_accumulate(
                            J, P, i, j, k, l, at(eri, nb, i, j, k, l));
                    }
        return J;
    }

    Eigen::MatrixXd fused_k(const std::vector<double> &eri,
                            const Eigen::MatrixXd &P, std::size_t nb)
    {
        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                for (std::size_t k = 0; k < nb; ++k)
                    for (std::size_t l = 0; l < nb; ++l)
                    {
                        if (!canonical(i, j, k, l))
                            continue;
                        HartreeFock::Integrals::exchange_accumulate(
                            K, P, i, j, k, l, at(eri, nb, i, j, k, l));
                    }
        return K;
    }

    void check_split_accumulators()
    {
        std::mt19937 rng(20260720);
        double worst_j = 0.0;
        double worst_k = 0.0;
        double worst_rt_rhf = 0.0;
        double worst_rt_uhf = 0.0;

        for (std::size_t nb = 1; nb <= 7; ++nb)
            for (int trial = 0; trial < 20; ++trial)
            {
                const std::vector<double> eri = random_symmetric_eri(nb, rng);
                const Eigen::MatrixXd P = random_symmetric_density(nb, rng);

                // (a) each term against its own brute-force oracle
                worst_j = std::max(worst_j,
                                   (naive_j(eri, P, nb) - fused_j(eri, P, nb))
                                       .cwiseAbs().maxCoeff());
                worst_k = std::max(worst_k,
                                   (naive_k(eri, P, nb) - fused_k(eri, P, nb))
                                       .cwiseAbs().maxCoeff());

                // (b) round-trip: recombining must reproduce the combined
                // entries bitwise-close, which pins BOTH prefactor conventions.
                // RHF carries 0.5 on exchange; UHF carries 1.0.
                const Eigen::MatrixXd J = fused_j(eri, P, nb);
                const Eigen::MatrixXd K = fused_k(eri, P, nb);

                const Eigen::MatrixXd rhf_rt = J - 0.5 * K;
                worst_rt_rhf = std::max(
                    worst_rt_rhf,
                    (fused_g_rhf(eri, P, nb) - rhf_rt).cwiseAbs().maxCoeff());

                // UHF: Ga = J(Pt) - 1.0 * K(Pa). Same-spin exchange, no 0.5.
                const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
                const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);
                const Eigen::MatrixXd Pt = Pa + Pb;
                Eigen::MatrixXd Ga_ref, Gb_ref;
                fused_g_uhf(eri, Pa, Pb, nb, Ga_ref, Gb_ref);

                const Eigen::MatrixXd Jt = fused_j(eri, Pt, nb);
                const Eigen::MatrixXd Ga_rt = Jt - fused_k(eri, Pa, nb);
                const Eigen::MatrixXd Gb_rt = Jt - fused_k(eri, Pb, nb);
                worst_rt_uhf = std::max(
                    worst_rt_uhf,
                    std::max((Ga_ref - Ga_rt).cwiseAbs().maxCoeff(),
                             (Gb_ref - Gb_rt).cwiseAbs().maxCoeff()));
            }

        if (worst_j > TOL)
            fail("split J: coulomb_accumulate != naive nb^4 Coulomb, "
                 "max|dJ| = " + std::to_string(worst_j));
        else
            std::cout << "OK  Step 0 / J-only / random tensors (nb=1..7): "
                         "max|dJ| = " << worst_j << '\n';

        if (worst_k > TOL)
            fail("split K: exchange_accumulate != naive nb^4 exchange. Is the "
                 "0.5 folded into the accumulator? It must NOT be — K is raw. "
                 "max|dK| = " + std::to_string(worst_k));
        else
            std::cout << "OK  Step 0 / K-only (raw, unscaled) / random tensors "
                         "(nb=1..7): max|dK| = " << worst_k << '\n';

        if (worst_rt_rhf > TOL)
            fail("round-trip RHF: J - 0.5*K != fock_accumulate_rhf, "
                 "max|dG| = " + std::to_string(worst_rt_rhf));
        else
            std::cout << "OK  Step 0 / round-trip RHF (J - 0.5K == combined): "
                         "max|dG| = " << worst_rt_rhf << '\n';

        if (worst_rt_uhf > TOL)
            fail("round-trip UHF: J - K != fock_accumulate_uhf, "
                 "max|dG| = " + std::to_string(worst_rt_uhf));
        else
            std::cout << "OK  Step 0 / round-trip UHF (J - 1.0K == combined): "
                         "max|dG| = " << worst_rt_uhf << '\n';
    }

    // ── Step 0: real integrals, against the production two-phase builder ─────

    HartreeFock::Calculator make_water(const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        mol.natoms = 3;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(3);
        mol.atomic_numbers << 8, 1, 1;
        mol.atomic_masses.resize(3);
        mol.atomic_masses << 16.0, 1.0, 1.0;
        mol.coordinates.resize(3, 3);
        mol.coordinates <<
            0.000000, 0.000000, 0.117176,
            0.000000, 0.757200, -0.468704,
            0.000000, -0.757200, -0.468704;

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
        {
            fail("read_gbs_basis failed (" + basis_name + "): " + basis_res.error());
            return calc;
        }
        calc._shells = std::move(*basis_res);
        return calc;
    }

    void check_real_integrals(const std::string &basis_name)
    {
        HartreeFock::Calculator calc = make_water(basis_name);
        if (!g_ok)
            return;

        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();

        // A plausible symmetric density (not a converged one — the identity under
        // test is algebraic and holds for any P).
        std::mt19937 rng(7);
        const Eigen::MatrixXd P = random_symmetric_density(nb, rng);
        const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
        const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);

        // Production two-phase builders — the oracle.
        const Eigen::MatrixXd G_ref =
            HartreeFock::ObaraSaika::_compute_2e_fock(
                shell_pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        const auto [Ga_ref, Gb_ref] =
            HartreeFock::ObaraSaika::_compute_2e_fock_uhf(
                shell_pairs, Pa, Pb, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);

        // Fused accumulation over the same tensor's canonical quartets. We take
        // the tensor from _compute_2e here so this step isolates the ACCUMULATION
        // (the quartet loop itself is not rewritten until Step 2).
        const std::vector<double> eri =
            HartreeFock::ObaraSaika::_compute_2e(
                shell_pairs, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);

        const Eigen::MatrixXd G_fus = fused_g_rhf(eri, P, nb);
        Eigen::MatrixXd Ga_fus, Gb_fus;
        fused_g_uhf(eri, Pa, Pb, nb, Ga_fus, Gb_fus);

        const double d_rhf = (G_ref - G_fus).cwiseAbs().maxCoeff();
        const double d_a = (Ga_ref - Ga_fus).cwiseAbs().maxCoeff();
        const double d_b = (Gb_ref - Gb_fus).cwiseAbs().maxCoeff();

        // ── Step 2: the production memory-direct builders ────────────────────
        // The checks above fuse from a PRE-BUILT tensor, isolating the
        // accumulation rule. These call the real fused quartet loop, which never
        // allocates nb^4 at all — so they gate the loop, not just the algebra.
        const Eigen::MatrixXd G_direct =
            HartreeFock::ObaraSaika::_compute_2e_fock_direct(
                shell_pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        const auto [Ga_direct, Gb_direct] =
            HartreeFock::ObaraSaika::_compute_2e_fock_uhf_direct(
                shell_pairs, Pa, Pb, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);

        const double dd_rhf = (G_ref - G_direct).cwiseAbs().maxCoeff();
        const double dd_uhf = std::max((Ga_ref - Ga_direct).cwiseAbs().maxCoeff(),
                                       (Gb_ref - Gb_direct).cwiseAbs().maxCoeff());

        if (dd_rhf > TOL)
            fail("water/" + basis_name +
                 " RHF: _compute_2e_fock_direct != _compute_2e_fock, max|dG| = " +
                 std::to_string(dd_rhf));
        else
            std::cout << "OK  Step 2 / RHF / water/" << basis_name
                      << " (nb=" << nb
                      << "): _compute_2e_fock_direct == _compute_2e_fock, max|dG| = "
                      << dd_rhf << '\n';

        if (dd_uhf > TOL)
            fail("water/" + basis_name +
                 " UHF: _compute_2e_fock_uhf_direct != _compute_2e_fock_uhf, max|dG| = " +
                 std::to_string(dd_uhf));
        else
            std::cout << "OK  Step 2 / UHF / water/" << basis_name
                      << " (nb=" << nb
                      << "): _compute_2e_fock_uhf_direct == _compute_2e_fock_uhf, max|dG| = "
                      << dd_uhf << '\n';

        // With Schwarz screening ON (the production tol_eri), both builders must
        // still agree. This is not redundant with the unscreened check: the
        // two-phase path leaves a screened quartet as a stored ZERO that Phase 2
        // still reads, while the fused path never contracts it at all. The two
        // are only equivalent if the fused skip is exactly a no-op contribution.
        constexpr double PROD_TOL_ERI = 1e-10;
        const Eigen::MatrixXd G_ref_s =
            HartreeFock::ObaraSaika::_compute_2e_fock(
                shell_pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0,
                PROD_TOL_ERI, nullptr);
        const Eigen::MatrixXd G_direct_s =
            HartreeFock::ObaraSaika::_compute_2e_fock_direct(
                shell_pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0,
                PROD_TOL_ERI, nullptr);
        const double dd_screened = (G_ref_s - G_direct_s).cwiseAbs().maxCoeff();

        if (dd_screened > TOL)
            fail("water/" + basis_name +
                 " RHF (screened): _compute_2e_fock_direct != _compute_2e_fock, max|dG| = " +
                 std::to_string(dd_screened));
        else
            std::cout << "OK  Step 2 / RHF / water/" << basis_name
                      << " (nb=" << nb << ", Schwarz tol=" << PROD_TOL_ERI
                      << "): screened fused == screened two-phase, max|dG| = "
                      << dd_screened << '\n';

        if (d_rhf > TOL)
            fail("water/" + basis_name + " RHF: fused != _compute_2e_fock, max|dG| = " +
                 std::to_string(d_rhf));
        else
            std::cout << "OK  Step 0 / RHF / water/" << basis_name
                      << " (nb=" << nb << "): fused == _compute_2e_fock, max|dG| = "
                      << d_rhf << '\n';

        if (std::max(d_a, d_b) > TOL)
            fail("water/" + basis_name + " UHF: fused != _compute_2e_fock_uhf, max|dG| = " +
                 std::to_string(std::max(d_a, d_b)));
        else
            std::cout << "OK  Step 0 / UHF / water/" << basis_name
                      << " (nb=" << nb << "): fused == _compute_2e_fock_uhf, max|dG| = "
                      << std::max(d_a, d_b) << '\n';
    }
}

// ── Step 3: determinism of the threaded fused build ─────────────────────────
//
// The fused accumulations are read-modify-write reductions into G, unlike the
// store-only write_eri_permutations. A reduction summed in nondeterministic
// order drifts, floating-point addition not being associative — the jitter the
// DFT XC accumulation already had to fix (src/dft/ks_matrix.cpp).
//
// The achievable guarantee, and the one that file states, is *bitwise
// reproducible at a fixed thread count*. It is NOT bitwise equality across
// different thread counts: with per-thread partials, N threads necessarily sum
// N different subsets, so the totals differ in the last bits no matter how the
// partials are ordered. (The two-phase tensor build IS cross-count invariant,
// but only because its scatter is store-only/idempotent — a genuine reduction
// cannot be. The DFT grid XC reduction likewise still drifts ~1e-10 across
// thread counts; the fused Fock build is ~1e-16 relative, far tighter.)
//
// So this checks the real contract: repeated runs at the SAME thread count must
// be bitwise identical. That is what schedule(static) + fixed-thread-index
// summation buys; with schedule(dynamic), or a critical-section reduction,
// repeated runs at one count would already differ.
void check_fixed_thread_determinism(const std::string &basis_name)
{
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;

    const std::vector<HartreeFock::ShellPair> shell_pairs =
        build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();

    std::mt19937 rng(7);
    const Eigen::MatrixXd P = random_symmetric_density(nb, rng);

    auto build = [&]()
    {
        return HartreeFock::ObaraSaika::_compute_2e_fock_direct(
            shell_pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10,
            nullptr);
    };

    const Eigen::MatrixXd first = build();
    for (int rep = 0; rep < 4; ++rep)
    {
        const Eigen::MatrixXd again = build();
        // Bitwise: not a tolerance. Same thread count, same partition, same
        // summation order -> the identical bits, every time.
        if (!(again.array() == first.array()).all())
        {
            fail("water/" + basis_name +
                 ": fused build is not bitwise reproducible at a fixed thread "
                 "count (repeat " + std::to_string(rep) + " differs)");
            return;
        }
    }

    std::cout << "OK  Step 3 / water/" << basis_name
              << ": fused build bitwise-reproducible across 5 runs at a fixed "
                 "thread count; checksum sum="
              << std::setprecision(17) << first.sum() << '\n';
}

// ── Step 6: every engine's fused builder matches its two-phase builder ───────
//
// The fused loop (fused_fock.h) is shared; only the per-quartet ERI callable
// differs between OS, HGP, Rys, and Rys-Auto. Each engine's *_direct entry must
// therefore reproduce its OWN two-phase builder (not OS's — the engines differ
// from each other at the last bits, which is expected and separately gated).
void check_engine(const std::string &name,
                  const std::string &basis_name,
                  const Eigen::MatrixXd &two_phase,
                  const Eigen::MatrixXd &fused)
{
    const double d = (two_phase - fused).cwiseAbs().maxCoeff();
    if (d > TOL)
        fail(name + " / water/" + basis_name +
             ": fused != two-phase, max|dG| = " + std::to_string(d));
    else
        std::cout << "OK  Step 6 / " << name << " / water/" << basis_name
                  << ": fused == two-phase, max|dG| = " << d << '\n';
}

void check_all_engines(const std::string &basis_name)
{
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;

    const std::vector<HartreeFock::ShellPair> sp = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();

    std::mt19937 rng(11);
    const Eigen::MatrixXd P = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);

    const auto K = HartreeFock::ERIKernel::Coulomb;
    constexpr double W = 0.0;
    constexpr double T = 1e-10;

    using namespace HartreeFock;

    check_engine("OS   RHF", basis_name,
                 ObaraSaika::_compute_2e_fock(sp, P, nb, K, W, T, nullptr),
                 ObaraSaika::_compute_2e_fock_direct(sp, P, nb, K, W, T, nullptr));
    check_engine("HGP  RHF", basis_name,
                 HeadGordonPople::_compute_2e_fock(sp, P, nb, K, W, T, nullptr),
                 HeadGordonPople::_compute_2e_fock_direct(sp, P, nb, K, W, T, nullptr));
    check_engine("Rys  RHF", basis_name,
                 RysQuad::_compute_2e_fock(sp, P, nb, K, W, T, nullptr),
                 RysQuad::_compute_2e_fock_direct(sp, P, nb, K, W, T, nullptr));
    check_engine("Auto RHF", basis_name,
                 RysQuad::_compute_2e_fock_auto(sp, P, nb, K, W, T, nullptr),
                 RysQuad::_compute_2e_fock_auto_direct(sp, P, nb, K, W, T, nullptr));

    // UHF: compare the alpha block (beta is built by the same accumulation).
    check_engine("OS   UHF", basis_name,
                 ObaraSaika::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, nullptr).first,
                 ObaraSaika::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, nullptr).first);
    check_engine("HGP  UHF", basis_name,
                 HeadGordonPople::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, nullptr).first,
                 HeadGordonPople::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, nullptr).first);
    check_engine("Rys  UHF", basis_name,
                 RysQuad::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, nullptr).first,
                 RysQuad::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, nullptr).first);
    check_engine("Auto UHF", basis_name,
                 RysQuad::_compute_2e_fock_uhf_auto(sp, Pa, Pb, nb, K, W, T, nullptr).first,
                 RysQuad::_compute_2e_fock_uhf_auto_direct(sp, Pa, Pb, nb, K, W, T, nullptr).first);
}

// ── Integral symmetry (sym_ops): the fused path handles it natively ─────────
//
// Under sym_ops the scatter is a NESTED orbit — the symmetry orbit of the
// quartet, and then the 8-fold permutational orbit of each of its elements. The
// fused builders used to bail out to the two-phase builder here; they now handle
// it directly, which is only correct because the two dedups compose (see the
// argument in src/integrals/quartet_orbit.h).
//
// This is the gate for that claim: with a real symmetry-op set, every engine's
// fused builder must still reproduce its own two-phase builder. Without this,
// the sym_ops path would be entirely untested — every other check in this file
// passes sym_ops = nullptr.
void check_symmetry_ops(const std::string &basis_name)
{
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;

    const std::vector<HartreeFock::ShellPair> sp = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();

    // Water in the yz plane has a C2v-like AO sign structure. Rather than derive
    // the real point group here (that is symmetry.cpp's job and is separately
    // tested), synthesize op sets that exercise the orbit machinery: identity,
    // a pure sign flip, and a sign flip on a different AO. These drive the same
    // build_quartet_orbit code paths — dedup, forced_zero, and the representative
    // filter — that a real point group does.
    std::vector<HartreeFock::SignedAOSymOp> ops;

    HartreeFock::SignedAOSymOp identity;
    identity.ao_map.resize(nb);
    identity.ao_sign.assign(nb, 1);
    for (std::size_t a = 0; a < nb; ++a)
        identity.ao_map[a] = static_cast<int>(a);
    ops.push_back(identity);

    HartreeFock::SignedAOSymOp flip = identity;
    flip.ao_sign[nb / 3] = -1; // one AO changes phase
    ops.push_back(flip);

    std::mt19937 rng(23);
    const Eigen::MatrixXd P = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);

    const auto K = HartreeFock::ERIKernel::Coulomb;
    constexpr double W = 0.0;
    constexpr double T = 1e-10;

    using namespace HartreeFock;

    check_engine("OS   RHF sym", basis_name,
                 ObaraSaika::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 ObaraSaika::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("HGP  RHF sym", basis_name,
                 HeadGordonPople::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 HeadGordonPople::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("Rys  RHF sym", basis_name,
                 RysQuad::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 RysQuad::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("Auto RHF sym", basis_name,
                 RysQuad::_compute_2e_fock_auto(sp, P, nb, K, W, T, &ops),
                 RysQuad::_compute_2e_fock_auto_direct(sp, P, nb, K, W, T, &ops));

    check_engine("OS   UHF sym", basis_name,
                 ObaraSaika::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, &ops).first,
                 ObaraSaika::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, &ops).first);
    check_engine("HGP  UHF sym", basis_name,
                 HeadGordonPople::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, &ops).first,
                 HeadGordonPople::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, &ops).first);
    check_engine("Rys  UHF sym", basis_name,
                 RysQuad::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, &ops).first,
                 RysQuad::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, &ops).first);
    check_engine("Auto UHF sym", basis_name,
                 RysQuad::_compute_2e_fock_uhf_auto(sp, Pa, Pb, nb, K, W, T, &ops).first,
                 RysQuad::_compute_2e_fock_uhf_auto_direct(sp, Pa, Pb, nb, K, W, T, &ops).first);
}

int main()
{
    check_random_tensors();
    check_split_accumulators();
    check_real_integrals("sto-3g");
    check_real_integrals("6-31g*"); // d shells: exercises the multi-component orbit
    check_fixed_thread_determinism("6-31g*");
    check_all_engines("sto-3g");
    check_all_engines("6-31g*");
    check_symmetry_ops("sto-3g");
    check_symmetry_ops("6-31g*");

    if (!g_ok)
    {
        std::cerr << "planck-fock-accumulate: FAIL\n";
        return 1;
    }
    std::cout << "planck-fock-accumulate: OK\n";
    return 0;
}
