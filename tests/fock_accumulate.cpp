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

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <atomic>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/fock_accumulate.h"
#include "integrals/fused_fock.h"
#include "integrals/hgp_block_eri.h"
#include "integrals/screening.h"
#include "integrals/hgp.h"
#include "integrals/os.h"
#include "integrals/quartet_orbit.h"
#include "integrals/rys.h"
#include "integrals/shellpair.h"
#include "require_boys_library.h"
#include "symmetry/integral_symmetry.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    // distinct_eri_orbit used to find repeats by a pairwise first-occurrence
    // search. That search is kept here verbatim as the oracle: the closed-form
    // version must return the same tuples in the same ORDER (order fixes the
    // accumulation sequence into G, so it is what keeps the Fock build
    // bitwise), on every tuple over 0..5, canonical or not.
    void check_distinct_orbit_matches_search()
    {
        std::size_t cases = 0;
        for (std::size_t i = 0; i < 6; ++i)
            for (std::size_t j = 0; j < 6; ++j)
                for (std::size_t k = 0; k < 6; ++k)
                    for (std::size_t l = 0; l < 6; ++l)
                    {
                        const std::array<std::array<std::size_t, 4>, 8> all = {{
                            {i, j, k, l}, {j, i, k, l}, {i, j, l, k}, {j, i, l, k},
                            {k, l, i, j}, {l, k, i, j}, {k, l, j, i}, {l, k, j, i},
                        }};
                        std::array<std::array<std::size_t, 4>, 8> ref{};
                        std::size_t n_ref = 0;
                        for (const auto &t : all)
                        {
                            bool seen = false;
                            for (std::size_t m = 0; m < n_ref; ++m)
                                seen = seen || ref[m] == t;
                            if (!seen)
                                ref[n_ref++] = t;
                        }
                        std::array<std::array<std::size_t, 4>, 8> got{};
                        const std::size_t n_got =
                            HartreeFock::Integrals::distinct_eri_orbit(i, j, k, l, got);
                        bool same = n_got == n_ref;
                        for (std::size_t m = 0; same && m < n_ref; ++m)
                            same = got[m] == ref[m];
                        if (!same)
                        {
                            fail("distinct_eri_orbit differs from the first-occurrence search at (" +
                                 std::to_string(i) + "," + std::to_string(j) + "," +
                                 std::to_string(k) + "," + std::to_string(l) + ")");
                            return;
                        }
                        ++cases;
                    }
        std::cout << "[INFO] distinct_eri_orbit == first-occurrence search on " << cases
                  << " tuples (order included)\n";
    }

    // is_orbit_representative must agree with build_quartet_orbit's own answer
    // (orbit.front() == the quartet) on every canonical quartet whose orbit is
    // not forced to zero (forced ones are skipped either way). Sign-only ops
    // would make every quartet its own representative, so these ops permute
    // AOs as well: two involutions on nb = 6 with mixed signs, plus their
    // product and the identity.
    void check_orbit_representative_precheck()
    {
        constexpr std::size_t nb = 6;
        const auto make_op = [](std::array<int, nb> map, std::array<int8_t, nb> sign)
        {
            HartreeFock::SignedAOSymOp op;
            op.ao_map.assign(map.begin(), map.end());
            op.ao_sign.assign(sign.begin(), sign.end());
            return op;
        };
        const std::vector<HartreeFock::SignedAOSymOp> ops = {
            make_op({0, 1, 2, 3, 4, 5}, {1, 1, 1, 1, 1, 1}),
            make_op({3, 4, 2, 0, 1, 5}, {1, -1, 1, 1, -1, 1}),
            make_op({0, 1, 5, 3, 4, 2}, {1, 1, -1, 1, 1, -1}),
            make_op({3, 4, 5, 0, 1, 2}, {1, -1, -1, 1, -1, -1}),
        };
        std::size_t compared = 0, reps = 0;
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = i; j < nb; ++j)
                for (std::size_t k = i; k < nb; ++k)
                    for (std::size_t l = k; l < nb; ++l)
                    {
                        if (k == i && l < j)
                            continue;
                        const auto [orbit, forced] =
                            HartreeFock::Integrals::build_quartet_orbit(i, j, k, l, ops);
                        if (forced)
                            continue;
                        const bool ref = orbit.front().i == i && orbit.front().j == j &&
                                         orbit.front().k == k && orbit.front().l == l;
                        const bool got =
                            HartreeFock::Integrals::is_orbit_representative(i, j, k, l, ops);
                        if (got != ref)
                        {
                            fail("is_orbit_representative disagrees with build_quartet_orbit at (" +
                                 std::to_string(i) + "," + std::to_string(j) + "," +
                                 std::to_string(k) + "," + std::to_string(l) + ")");
                            return;
                        }
                        ++compared;
                        reps += ref;
                    }
        std::cout << "[INFO] is_orbit_representative == build_quartet_orbit on " << compared
                  << " canonical quartets (" << reps << " representatives)\n";
        if (reps == 0 || reps == compared)
            fail("is_orbit_representative fixture is vacuous: needs both representatives and non-representatives");
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

    // ── J-only / K-only split (PR #151; docs/HPC_MPI.md) ──────────────
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
        std::cout << "OK  " << name << " / water/" << basis_name
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

// ── Step 2: the single-term builders, on real integrals ─────────────────────
//
// check_split_accumulators proves the accumulation rule on synthetic tensors.
// This proves the whole plumbed path — engine dispatch, Schwarz screening,
// canonical filter, threaded reduction — by the identity that has to hold if
// the term parameter is wired correctly through all four engines:
//
//     J - 0.5*K == combined G      (RHF)
//     J - K_sigma == G_sigma       (UHF, per spin channel)
//
// It goes through the base.h dispatchers, i.e. exactly the entries DFT will
// call, so a mis-wired engine case shows up here rather than in the thread-count check.
void check_single_term_builders(const std::string &basis_name)
{
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;

    const std::vector<HartreeFock::ShellPair> sp = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();

    std::mt19937 rng(2027);
    const Eigen::MatrixXd P = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pt = Pa + Pb;

    const auto K = HartreeFock::ERIKernel::Coulomb;
    constexpr double W = 0.0;
    constexpr double T = 1e-10;

    const std::vector<std::pair<std::string, HartreeFock::IntegralMethod>> engines = {
        {"OS  ", HartreeFock::IntegralMethod::ObaraSaika},
        {"HGP ", HartreeFock::IntegralMethod::HeadGordonPople},
        {"Rys ", HartreeFock::IntegralMethod::RysQuadrature},
        {"Auto", HartreeFock::IntegralMethod::Auto},
    };

    for (const auto &[name, engine] : engines)
    {
        // RHF: J - 0.5K must reproduce the combined build.
        const Eigen::MatrixXd G = _compute_2e_fock(sp, P, nb, engine, K, W, T, nullptr);
        const Eigen::MatrixXd J = _compute_2e_j_direct(sp, P, nb, engine, K, W, T, nullptr);
        const Eigen::MatrixXd Kx = _compute_2e_k_direct(sp, P, nb, engine, K, W, T, nullptr);
        check_engine(name + std::string(" RHF J-0.5K"), basis_name, G, J - 0.5 * Kx);

        // UHF: per spin, J(Pt) - K_sigma must reproduce that channel.
        const auto [Ga, Gb] = _compute_2e_fock_uhf(sp, Pa, Pb, nb, engine, K, W, T, nullptr);
        const Eigen::MatrixXd Jt = _compute_2e_j_direct(sp, Pt, nb, engine, K, W, T, nullptr);
        const auto [Ka, Kb] = _compute_2e_k_uhf_direct(sp, Pa, Pb, nb, engine, K, W, T, nullptr);
        check_engine(name + std::string(" UHF a"), basis_name, Ga, Jt - Ka);
        check_engine(name + std::string(" UHF b"), basis_name, Gb, Jt - Kb);
    }
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

// The sign-flip ops above map every shell group onto itself, so the fused
// loop's block-level orbit rejection (an op sending one of a block's groups
// below the bra's first group) can never fire under them. Water's real C2v
// ops swap the two H atoms, so they move shell groups and exercise it.
void check_real_symmetry_ops(const std::string &basis_name)
{
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;
    calc._molecule._symmetry = true;
    calc._molecule._point_group = "C2v";
    HartreeFock::Symmetry::update_integral_symmetry(calc);
    const std::vector<HartreeFock::SignedAOSymOp> &ops = calc._integral_symmetry_ops;

    bool moves_ao = false;
    for (const auto &op : ops)
        for (std::size_t a = 0; a < op.ao_map.size(); ++a)
            moves_ao = moves_ao || op.ao_map[a] != static_cast<int>(a);
    if (ops.size() < 2 || !moves_ao)
    {
        fail("water/" + basis_name + ": expected C2v ops that swap the H atoms, got " +
             std::to_string(ops.size()) + " ops" + (moves_ao ? "" : ", none permuting AOs"));
        return;
    }

    const std::vector<HartreeFock::ShellPair> sp = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();
    // A density with the molecule's symmetry is not required: fused and
    // two-phase builders apply the same orbit replication, so they must agree
    // for any density.
    std::mt19937 rng(29);
    const Eigen::MatrixXd P = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);

    const auto K = HartreeFock::ERIKernel::Coulomb;
    constexpr double W = 0.0;
    constexpr double T = 1e-10;

    using namespace HartreeFock;

    check_engine("OS   RHF C2v", basis_name,
                 ObaraSaika::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 ObaraSaika::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("HGP  RHF C2v", basis_name,
                 HeadGordonPople::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 HeadGordonPople::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("Rys  RHF C2v", basis_name,
                 RysQuad::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 RysQuad::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("Auto RHF C2v", basis_name,
                 RysQuad::_compute_2e_fock_auto(sp, P, nb, K, W, T, &ops),
                 RysQuad::_compute_2e_fock_auto_direct(sp, P, nb, K, W, T, &ops));
    check_engine("HGP  UHF C2v", basis_name,
                 HeadGordonPople::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, &ops).first,
                 HeadGordonPople::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, &ops).first);
    check_engine("Auto UHF C2v", basis_name,
                 RysQuad::_compute_2e_fock_uhf_auto(sp, Pa, Pb, nb, K, W, T, &ops).first,
                 RysQuad::_compute_2e_fock_uhf_auto_direct(sp, Pa, Pb, nb, K, W, T, &ops).first);
}

// ── S1: skeleton accumulation + final symmetrization ────────────────────────
//
// When every density has the molecule's symmetry the fused loop accumulates
// each orbit representative once, weighted by its orbit size, and symmetrizes
// G at the end instead of replicating. Every check above uses densities with
// no point-group symmetry, so it exercises only the replication fallback (a
// skeleton build would differ from the two-phase builder by O(1) there, so
// their passing also proves the fallback fired). This one group-averages the
// densities, so the skeleton path runs, and requires it to match the
// two-phase replication builder. The average is built from dense O_R matrices,
// independently of the production gather in symmetrize_over_ops.
Eigen::MatrixXd group_average(const Eigen::MatrixXd &P,
                              const std::vector<HartreeFock::SignedAOSymOp> &ops)
{
    const Eigen::Index nb = P.rows();
    Eigen::MatrixXd avg = Eigen::MatrixXd::Zero(nb, nb);
    for (const auto &op : ops)
    {
        Eigen::MatrixXd O = Eigen::MatrixXd::Zero(nb, nb);
        for (Eigen::Index a = 0; a < nb; ++a)
            O(op.ao_map[a], a) = op.ao_sign[a];
        avg += O * P * O.transpose();
    }
    return avg / static_cast<double>(ops.size());
}

void check_skeleton(const std::string &label, const std::string &basis_name,
                    const HartreeFock::Calculator &calc,
                    const std::vector<HartreeFock::SignedAOSymOp> &ops, unsigned seed,
                    bool physical)
{
    const std::vector<HartreeFock::ShellPair> sp = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();
    std::mt19937 rng(seed);
    const Eigen::MatrixXd P = group_average(random_symmetric_density(nb, rng), ops);
    const Eigen::MatrixXd Pa = group_average(random_symmetric_density(nb, rng), ops);
    const Eigen::MatrixXd Pb = group_average(random_symmetric_density(nb, rng), ops);
    const Eigen::MatrixXd Pt = Pa + Pb;
    if (!HartreeFock::Integrals::density_has_symmetry(P, ops, 1e-12))
        fail(label + ": group-averaged density failed the symmetry check");

    const auto K = HartreeFock::ERIKernel::Coulomb;
    constexpr double W = 0.0;
    constexpr double T = 1e-10;
    using namespace HartreeFock;

    check_engine("OS   RHF skel " + label, basis_name,
                 ObaraSaika::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 ObaraSaika::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("HGP  RHF skel " + label, basis_name,
                 HeadGordonPople::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 HeadGordonPople::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("Rys  RHF skel " + label, basis_name,
                 RysQuad::_compute_2e_fock(sp, P, nb, K, W, T, &ops),
                 RysQuad::_compute_2e_fock_direct(sp, P, nb, K, W, T, &ops));
    check_engine("Auto RHF skel " + label, basis_name,
                 RysQuad::_compute_2e_fock_auto(sp, P, nb, K, W, T, &ops),
                 RysQuad::_compute_2e_fock_auto_direct(sp, P, nb, K, W, T, &ops));

    const auto uhf_ref = HeadGordonPople::_compute_2e_fock_uhf(sp, Pa, Pb, nb, K, W, T, &ops);
    const auto uhf_hgp = HeadGordonPople::_compute_2e_fock_uhf_direct(sp, Pa, Pb, nb, K, W, T, &ops);
    check_engine("HGP  UHF a skel " + label, basis_name, uhf_ref.first, uhf_hgp.first);
    check_engine("HGP  UHF b skel " + label, basis_name, uhf_ref.second, uhf_hgp.second);
    const auto uhf_auto_ref = RysQuad::_compute_2e_fock_uhf_auto(sp, Pa, Pb, nb, K, W, T, &ops);
    const auto uhf_auto = RysQuad::_compute_2e_fock_uhf_auto_direct(sp, Pa, Pb, nb, K, W, T, &ops);
    check_engine("Auto UHF a skel " + label, basis_name, uhf_auto_ref.first, uhf_auto.first);
    check_engine("Auto UHF b skel " + label, basis_name, uhf_auto_ref.second, uhf_auto.second);

    // Single-term modes (the DFT J/K path): skeleton vs the no-symmetry build,
    // which is a valid reference only when the ops are true ERI symmetries.
    if (!physical)
        return;
    const auto engine = HartreeFock::IntegralMethod::Auto;
    check_engine("Auto J skel " + label, basis_name,
                 _compute_2e_j_direct(sp, P, nb, engine, K, W, T, nullptr),
                 _compute_2e_j_direct(sp, P, nb, engine, K, W, T, &ops));
    check_engine("Auto K skel " + label, basis_name,
                 _compute_2e_k_direct(sp, P, nb, engine, K, W, T, nullptr),
                 _compute_2e_k_direct(sp, P, nb, engine, K, W, T, &ops));
    const auto k_ref = _compute_2e_k_uhf_direct(sp, Pa, Pb, nb, engine, K, W, T, nullptr);
    const auto k_skel = _compute_2e_k_uhf_direct(sp, Pa, Pb, nb, engine, K, W, T, &ops);
    check_engine("Auto Ka skel " + label, basis_name, k_ref.first, k_skel.first);
    check_engine("Auto Kb skel " + label, basis_name, k_ref.second, k_skel.second);
    check_engine("Auto Jt skel " + label, basis_name,
                 _compute_2e_j_direct(sp, Pt, nb, engine, K, W, T, nullptr),
                 _compute_2e_j_direct(sp, Pt, nb, engine, K, W, T, &ops));
}

void check_skeleton_symmetry(const std::string &basis_name)
{
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;
    const std::size_t nb = calc._shells._basis_functions.size();

    // Synthetic order-2 group: identity plus one AO sign flip (orbits of 1 and
    // 2, forbidden quartets, and no AO movement).
    std::vector<HartreeFock::SignedAOSymOp> flip_ops(2);
    flip_ops[0].ao_map.resize(nb);
    flip_ops[0].ao_sign.assign(nb, 1);
    for (std::size_t a = 0; a < nb; ++a)
        flip_ops[0].ao_map[a] = static_cast<int>(a);
    flip_ops[1] = flip_ops[0];
    flip_ops[1].ao_sign[nb / 3] = -1;
    // Not a true ERI symmetry in general (AO nb/3 is an s function in 6-31g*),
    // so the reference is replication, which applies the same ops.
    check_skeleton("flip", basis_name, calc, flip_ops, 31, false);

    // Water's real C2v ops, which swap the H atoms.
    calc._molecule._symmetry = true;
    calc._molecule._point_group = "C2v";
    HartreeFock::Symmetry::update_integral_symmetry(calc);
    check_skeleton("C2v", basis_name, calc, calc._integral_symmetry_ops, 37, true);
}

// ── K1: dense digestion per block against the per-component loop ─────────────
//
// docs/ERI_SHELL_QUARTET_FOCK_BUILD.md, K1. Same production kernels both ways,
// through fused_fock_build's dense_digest flag; every FusedTerm mode, RHF and
// UHF, at tol 1e-10 and at 1e-6 (where the component Schwarz mask matters).
// A counting kernel proves the dense path ran: it prepares every unordered
// block pair above the block Schwarz bound exactly once. A non-symmetric
// density must take the per-component path, so it must come out bitwise equal.
struct CountingBlockKernel
{
    HartreeFock::Integrals::ChainBlockKernel<HartreeFock::Integrals::ClosedFormBlockKernel,
                                             HartreeFock::Integrals::HgpBlockKernel>
        inner{HartreeFock::Integrals::ClosedFormBlockKernel{3}, HartreeFock::Integrals::HgpBlockKernel{4, 12}};
    std::atomic<std::size_t> *count = nullptr;
    // Blocks whose bra pair sorts after the ket pair (by first AO). The dense
    // traversal visits only p <= q, so it never prepares one; the
    // per-component loop does. Proves which path ran when counts tie.
    std::atomic<std::size_t> *reversed = nullptr;
    bool covers(int l, HartreeFock::ERIKernel k) const { return inner.covers(l, k); }
    bool prepare(const HartreeFock::Integrals::BlockEriRequest &r)
    {
        count->fetch_add(1, std::memory_order_relaxed);
        const auto first = [&](int s) { return r.views[s][0]->_index; };
        if (reversed && std::pair(first(0), first(1)) > std::pair(first(2), first(3)))
            reversed->fetch_add(1, std::memory_order_relaxed);
        return inner.prepare(r);
    }
    double value(std::size_t a, std::size_t b, std::size_t c, std::size_t d) const { return inner.value(a, b, c, d); }
};

// max|A - B| over the outputs a build filled, NaN-safe: a size mismatch or any
// non-finite entry is +inf, so `!(worst <= tol)` cannot pass on NaN the way
// `std::max(worst, NaN)` (which returns worst) did.
double max_output_diff(const Eigen::MatrixXd (&a)[3], const Eigen::MatrixXd (&b)[3])
{
    double worst = 0.0;
    for (int m = 0; m < 3; ++m)
    {
        if (!a[m].size() && !b[m].size())
            continue;
        if (a[m].size() != b[m].size() || !a[m].allFinite() || !b[m].allFinite())
            return std::numeric_limits<double>::infinity();
        worst = std::max(worst, (a[m] - b[m]).cwiseAbs().maxCoeff());
    }
    return worst;
}

double hgp_elem(const HartreeFock::ShellPair &ab, const HartreeFock::ShellPair &cd,
                int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
                int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
                HartreeFock::ERIKernel k, double w) noexcept
{
    return HartreeFock::HeadGordonPople::_contracted_eri_elem(
        ab, cd, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, k, w);
}

// ── Determinism of the threaded fused build, across thread counts ───────────
//
// The fused accumulations are read-modify-write reductions into G, so a sum in
// nondeterministic order drifts (the DFT XC jitter, src/dft/ks_matrix.cpp).
// fused_fock_build keys its partials by bin (bra mod n_bins, n_bins a function
// of nb only) and sums bins in fixed order, so G must be BITWISE identical at
// any thread count, even though bins are scheduled dynamically. Checked on the
// per-component path (OS direct), the dense path with RHF partials, and the
// dense path with UHF partials, each at 1/2/4/8 threads and repeated.
void check_thread_count_invariance(const std::string &basis_name)
{
#ifdef USE_OPENMP
    using namespace HartreeFock::Integrals;
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;
    const std::vector<HartreeFock::ShellPair> sp = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();
    const std::vector<double> flat = HartreeFock::Screening::schwarz_table_hgp(sp, nb, nullptr);
    Eigen::MatrixXd Q(nb, nb);
    for (std::size_t i = 0; i < nb; ++i)
        for (std::size_t j = 0; j < nb; ++j)
            Q(i, j) = flat[i * nb + j];
    std::mt19937 rng(7);
    const Eigen::MatrixXd P = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pt = Pa + Pb;
    const ChainBlockKernel<ClosedFormBlockKernel, HgpBlockKernel> production{ClosedFormBlockKernel{3},
                                                                            HgpBlockKernel{4, 12}};

    // out[0..2] = G, Ga, Gb for one build of the given kind.
    const auto build = [&](int kind, Eigen::MatrixXd (&out)[3])
    {
        if (kind == 0)
        {
            out[0] = HartreeFock::ObaraSaika::_compute_2e_fock_direct(sp, P, nb, HartreeFock::ERIKernel::Coulomb,
                                                                       0.0, 1e-10, nullptr);
            return;
        }
        FusedFockDensities dens;
        if (kind == 1)
            dens.P = &P;
        else
            dens = {nullptr, &Pt, &Pa, &Pb};
        fused_fock_build(sp, nb, Q, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, kind == 2, dens, out[0], out[1],
                         out[2], hgp_elem, nullptr, FusedTerm::Combined, production, /*dense_digest=*/true);
    };
    const char *names[3] = {"per-component (OS direct)", "dense RHF", "dense UHF"};
    const int saved = omp_get_max_threads();
    for (int kind = 0; kind < 3; ++kind)
    {
        const std::string tag = "water/" + basis_name + " " + names[kind];
        Eigen::MatrixXd ref[3];
        omp_set_num_threads(1);
        build(kind, ref);
        bool same = true;
        std::string where;
        for (const int t : {2, 4, 8, 4})
        {
            omp_set_num_threads(t);
            Eigen::MatrixXd got[3];
            build(kind, got);
            for (int m = 0; m < 3; ++m)
                if (got[m].size() != ref[m].size() || !(got[m].array() == ref[m].array()).all())
                    same = false, where = std::to_string(t) + " threads, output " + std::to_string(m);
        }
        omp_set_num_threads(saved);
        // Bitwise: not a tolerance.
        if (!same)
            fail(tag + ": G differs from the 1-thread build at " + where);
        else
            std::cout << "OK  " << tag << ": G bitwise identical at 1/2/4/8 threads (and a repeat at 4)\n";
    }
#else
    std::cout << "SKIP thread-count invariance on water/" << basis_name << ": built without OpenMP\n";
#endif
}

void check_dense_digest(const std::string &basis_name)
{
    using namespace HartreeFock::Integrals;
    HartreeFock::Calculator calc = make_water(basis_name);
    if (!g_ok)
        return;
    const std::vector<HartreeFock::ShellPair> sp = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells._basis_functions.size();
    const std::vector<double> flat = HartreeFock::Screening::schwarz_table_hgp(sp, nb, nullptr);
    Eigen::MatrixXd Q(nb, nb);
    for (std::size_t i = 0; i < nb; ++i)
        for (std::size_t j = 0; j < nb; ++j)
            Q(i, j) = flat[i * nb + j];

    std::mt19937 rng(23);
    const Eigen::MatrixXd P = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pa = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pb = random_symmetric_density(nb, rng);
    const Eigen::MatrixXd Pt = Pa + Pb;

    // Expected dense prepare count: unordered pairs of shell pairs above the
    // block Schwarz bound (shell groups = runs of AOs sharing one Shell).
    std::vector<const HartreeFock::ContractedView *> views;
    const auto groups = fused_shell_groups(sp, nb, views);
    std::vector<double> qmax;
    for (std::size_t a = 0; a < groups.size(); ++a)
        for (std::size_t b = a; b < groups.size(); ++b)
            qmax.push_back(Q.block(groups[a].first_ao, groups[b].first_ao,
                                   groups[a].n_components, groups[b].n_components).maxCoeff());

    const FusedTerm terms[] = {FusedTerm::Combined, FusedTerm::CoulombOnly,
                               FusedTerm::ExchangeOnly, FusedTerm::RawCoulombExchange};
    const char *names[] = {"Combined", "CoulombOnly", "ExchangeOnly", "RawCoulombExchange"};
    for (const double tol : {1e-10, 1e-6})
    {
        std::size_t expect = 0;
        for (std::size_t p = 0; p < qmax.size(); ++p)
            for (std::size_t q = p; q < qmax.size(); ++q)
                expect += qmax[p] * qmax[q] >= tol;
        for (int t = 0; t < 4; ++t)
            for (const bool uhf : {false, true})
            {
                FusedFockDensities dens;
                dens.P = &P;
                if (uhf)
                    dens = {nullptr, &Pt, &Pa, &Pb};
                // [0] per-component, [1] dense (K1).
                Eigen::MatrixXd out[2][3];
                std::atomic<std::size_t> count[2]{0, 0};
                for (int d = 0; d < 2; ++d)
                    fused_fock_build(sp, nb, Q, HartreeFock::ERIKernel::Coulomb, 0.0, tol, uhf, dens,
                                     out[d][0], out[d][1], out[d][2], hgp_elem, nullptr, terms[t],
                                     CountingBlockKernel{.count = &count[d]}, /*dense_digest=*/d == 1);
                const double worst = max_output_diff(out[0], out[1]);
                const std::string tag = "K1 water/" + basis_name + " " + names[t] + (uhf ? " UHF" : " RHF") +
                                        " tol " + (tol < 1e-8 ? "1e-10" : "1e-6");
                // Summation order differs, so the bound is relative to the size
                // of G (cc-pVTZ UHF Coulomb sits at ~1e-13 absolute).
                double gmax = 1.0;
                for (int m = 0; m < 3; ++m)
                    if (out[0][m].size())
                        gmax = std::max(gmax, out[0][m].cwiseAbs().maxCoeff());
                if (!(worst <= 1e-14 * gmax))
                    { std::ostringstream os; os << std::setprecision(3) << worst; fail(tag + ": dense != per-component, max|dG| = " + os.str()); }
                // UHF RawCoulombExchange writes nothing either way, and the dense
                // loop skips it outright.
                else if (count[1] != (uhf && terms[t] == FusedTerm::RawCoulombExchange ? 0 : expect))
                    fail(tag + ": dense prepared " + std::to_string(count[1].load()) + " blocks, expected " +
                         std::to_string(expect) + " (dense path not taken, or a block pair visited twice?)");
                else
                    std::cout << "OK  " << tag << ": max|dG| = " << worst << ", " << expect << " blocks\n";
            }
    }

    // K1.4: skeleton mode (water's real C2v ops, group-averaged densities)
    // goes dense too; replication mode does not.
    {
        // A fresh molecule, not a copy of `calc`: update_integral_symmetry
        // matches basis functions to shells by pointer, and a copy's basis
        // functions still point at the original's shells (only the identity
        // would survive).
        HartreeFock::Calculator cs = make_water(basis_name);
        cs._molecule._symmetry = true;
        cs._molecule._point_group = "C2v";
        HartreeFock::Symmetry::update_integral_symmetry(cs);
        const auto &ops = cs._integral_symmetry_ops;
        if (ops.size() != 4)
            fail("K1.4 water/" + basis_name + ": expected 4 C2v ops, got " + std::to_string(ops.size()));
        std::mt19937 rs(29);
        const Eigen::MatrixXd Ps = group_average(random_symmetric_density(nb, rs), ops);
        const Eigen::MatrixXd Psa = group_average(random_symmetric_density(nb, rs), ops);
        const Eigen::MatrixXd Psb = group_average(random_symmetric_density(nb, rs), ops);
        const Eigen::MatrixXd Pst = Psa + Psb;
        for (int t = 0; t < 4; ++t)
            for (const bool uhf : {false, true})
            {
                FusedFockDensities dens;
                dens.P = &Ps;
                if (uhf)
                    dens = {nullptr, &Pst, &Psa, &Psb};
                Eigen::MatrixXd out[2][3];
                std::atomic<std::size_t> count[2]{0, 0}, reversed[2]{0, 0};
                for (int d = 0; d < 2; ++d)
                    fused_fock_build(sp, nb, Q, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, uhf, dens,
                                     out[d][0], out[d][1], out[d][2], hgp_elem, &ops, terms[t],
                                     CountingBlockKernel{.count = &count[d], .reversed = &reversed[d]},
                                     /*dense_digest=*/d == 1);
                const double worst = max_output_diff(out[0], out[1]);
                const std::string tag = "K1.4 water/" + basis_name + " C2v skeleton " + names[t] +
                                        (uhf ? " UHF" : " RHF");
                const bool nothing = uhf && terms[t] == FusedTerm::RawCoulombExchange;
                // Skeleton weights scale values by up to |G|, and the rounding with them.
                if (!(worst <= 1e-13 * static_cast<double>(ops.size())))
                    { std::ostringstream os; os << std::setprecision(3) << worst; fail(tag + ": dense != per-component, max|dG| = " + os.str()); }
                // Both prepare lazily, only blocks holding a representative, so
                // the counts can tie. Which path ran is decided by the reversed
                // blocks: dense visits p <= q only, the per-component loop does not.
                else if (!nothing && !(count[1] > 0 && count[1] <= count[0] && reversed[1] == 0 &&
                                       reversed[0] > 0))
                    fail(tag + ": dense prepared " + std::to_string(count[1].load()) + " blocks (" +
                         std::to_string(reversed[1].load()) + " reversed) against " +
                         std::to_string(count[0].load()) + " (" + std::to_string(reversed[0].load()) +
                         " reversed): dense skeleton path not taken?");
                else
                    std::cout << "OK  " << tag << ": max|dG| = " << worst << ", blocks " << count[1]
                              << " vs " << count[0] << '\n';
            }
    }

    // A non-symmetric density must fall back to the per-component path.
    Eigen::MatrixXd Pn = P;
    Pn(0, nb - 1) += 1e-3;
    FusedFockDensities dn;
    dn.P = &Pn;
    Eigen::MatrixXd G[2], u1, u2;
    std::atomic<std::size_t> c0{0}, c1{0};
    fused_fock_build(sp, nb, Q, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, false, dn, G[0], u1, u2,
                     hgp_elem, nullptr, FusedTerm::ExchangeOnly, CountingBlockKernel{.count = &c0}, false);
    fused_fock_build(sp, nb, Q, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, false, dn, G[1], u1, u2,
                     hgp_elem, nullptr, FusedTerm::ExchangeOnly, CountingBlockKernel{.count = &c1}, true);
    if (!(G[0].array() == G[1].array()).all() || c0 != c1)
        fail("K1 water/" + basis_name + ": a non-symmetric density did not fall back bitwise");
    else
        std::cout << "OK  K1 water/" << basis_name << ": non-symmetric density falls back, bitwise\n";
}

int main()
{
    // Real-integral fixtures dispatch through the signed, runtime-loaded Boys
    // implementation.  Fail cleanly if it cannot be made available rather
    // than dereferencing the uninitialized dispatch pointer in boys_vec().
    if (!require_boys_library())
        return 1;

    check_distinct_orbit_matches_search();
    check_orbit_representative_precheck();
    check_random_tensors();
    check_split_accumulators();
    check_real_integrals("sto-3g");
    check_real_integrals("6-31g*"); // d shells: exercises the multi-component orbit
    check_all_engines("sto-3g");
    check_all_engines("6-31g*");
    check_single_term_builders("sto-3g");
    check_single_term_builders("6-31g*");
    check_symmetry_ops("sto-3g");
    check_symmetry_ops("6-31g*");
    check_real_symmetry_ops("sto-3g");
    check_real_symmetry_ops("6-31g*");
    check_skeleton_symmetry("sto-3g");
    check_skeleton_symmetry("6-31g*");
    for (const char *b : {"6-31g", "6-31g*", "cc-pVDZ", "cc-pVTZ"})
        check_dense_digest(b);
    check_thread_count_invariance("6-31g*");

    if (!g_ok)
    {
        std::cerr << "planck-fock-accumulate: FAIL\n";
        return 1;
    }
    std::cout << "planck-fock-accumulate: OK\n";
    return 0;
}
