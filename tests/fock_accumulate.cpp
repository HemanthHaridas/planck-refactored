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
#include "integrals/os.h"
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

int main()
{
    check_random_tensors();
    check_real_integrals("sto-3g");
    check_real_integrals("6-31g*"); // d shells: exercises the multi-component orbit
    check_fixed_thread_determinism("6-31g*");

    if (!g_ok)
    {
        std::cerr << "planck-fock-accumulate: FAIL\n";
        return 1;
    }
    std::cout << "planck-fock-accumulate: OK\n";
    return 0;
}
