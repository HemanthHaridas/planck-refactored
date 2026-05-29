// Efficiency benchmark for the full-symmetry direct Fock (_symm) vs the production
// direct Fock — docs/FULL_SYMMETRY_ERI_DESIGN.md §7 item 4 / §8.4 item 5 follow-up.
//
// This is the "benchmark _symm vs production" measurement the design defers until
// after parallelism + Schwarz screening land (both now in build_skeleton_eri).
//
// Methodology
// -----------
// We time the *2e Fock build only* — not a full SCF — so the comparison isolates
// the ERI-reduction cost from convergence-path / iteration-count differences. For
// each molecule and each engine (OS, Rys, HGP) we time three Fock builds on the SAME
// symmetry-adapted density P:
//
//   1. nosym   — production _compute_2e_fock with sym_ops = nullptr
//                (no reduction at all; the raw integral cost).
//   2. d2h     — production _compute_2e_fock with the D2h sign-flip ops
//                (today's default SCF path when symmetry is on).
//   3. fullsym — _compute_2e_fock_symm with the FULL point group
//                (the new path: petite list + skeleton symmetrization,
//                 now OpenMP-parallel + Schwarz-screened).
//
// Each variant is warmed up once then timed over N repetitions; we report the
// median wall time and the speedup of d2h / fullsym relative to nosym. The _symm
// result is also re-checked against production so we are not timing a wrong answer.
//
// P must be symmetry-adapted for _symm to equal the true Fock — the same contract
// the SCF integration honors. The density is CONTRAVARIANT (O_R P O_Rᵀ == P), so we
// build it by group-averaging O_R P_raw O_Rᵀ (see the in-body comment for why the
// covariant O_Rᵀ P O_R law would be wrong for Cartesian d-shells).

#include "integrals/os.h"
#include "integrals/rys.h"
#include "integrals/hgp.h"
#include "symmetry/hgp_symm.h"
#include "symmetry/os_symm.h"
#include "symmetry/rys_symm.h"
#include "symmetry/integral_symmetry.h"
#include "integrals/shellpair.h"
#include "symmetry/group_operations.h"
#include "symmetry/symmetry.h"
#include "basis/basis.h"
#include "base/basis.h"
#include "base/types.h"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

    // Median wall time (ms) of `fn` over `reps` timed runs after one warm-up.
    double time_ms(int reps, const std::function<void()> &fn)
    {
        fn(); // warm-up (cache, first-touch allocation, libmsym, …)
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(reps));
        for (int r = 0; r < reps; ++r)
        {
            const auto t0 = Clock::now();
            fn();
            const auto t1 = Clock::now();
            samples.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
    }

    std::expected<HartreeFock::Calculator, std::string> make_calculator(
        const std::vector<int> &Z,
        const std::vector<std::array<double, 3>> &xyz,
        int charge, int multiplicity, const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        const std::size_t n = Z.size();
        mol.natoms = n;
        mol.charge = charge;
        mol.multiplicity = multiplicity;
        mol.atomic_numbers.resize(n);
        mol.atomic_masses.resize(n);
        mol.coordinates.resize(n, 3);
        for (std::size_t i = 0; i < n; ++i)
        {
            mol.atomic_numbers[static_cast<Eigen::Index>(i)] = Z[i];
            mol.atomic_masses[static_cast<Eigen::Index>(i)] = 1.0;
            for (int c = 0; c < 3; ++c)
                mol.coordinates(static_cast<Eigen::Index>(i), c) = xyz[i][static_cast<std::size_t>(c)];
        }
        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        if (auto r = HartreeFock::Symmetry::detectSymmetry(mol, HartreeFock::Units::Angstrom); !r)
            return std::unexpected("detectSymmetry: " + r.error());
        const std::filesystem::path gbs = std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis: " + basis_res.error());
        calc._shells = std::move(*basis_res);
        if (auto r = calc.initialize(); !r)
            return std::unexpected("initialize: " + r.error());
        return calc;
    }

    Eigen::MatrixXd random_symmetric(Eigen::Index n, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd A(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                A(i, j) = dist(rng);
        return 0.5 * (A + A.transpose());
    }

    bool g_ok = true;

    void bench(const std::string &name,
               const std::vector<int> &Z,
               const std::vector<std::array<double, 3>> &xyz,
               int charge, int mult,
               const std::string &basis_name, int reps)
    {
        auto calc_res = make_calculator(Z, xyz, charge, mult, basis_name);
        if (!calc_res)
        {
            std::cerr << "[SKIP] " << name << ": setup: " << calc_res.error() << '\n';
            g_ok = false;
            return;
        }
        HartreeFock::Calculator calc = std::move(*calc_res);

        const std::size_t nb = calc._shells.nbasis();
        const auto pairs = build_shellpairs(calc._shells);

        auto ops_res = HartreeFock::Symmetry::build_group_operations(calc);
        if (!ops_res || !ops_res->valid)
        {
            std::cerr << "[SKIP] " << name << ": build_group_operations failed\n";
            g_ok = false;
            return;
        }
        const int gorder = ops_res->order;

        // D2h sign-flip ops (today's production reduction).
        HartreeFock::Symmetry::update_integral_symmetry(calc);
        const std::size_t d2h_ops = calc._integral_symmetry_ops.size();
        const auto *d2h = calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr;

        // A real SCF density is CONTRAVARIANT in the (non-orthonormal) AO basis:
        // O_R P O_Rᵀ = P. This — not the covariant operator law O_Rᵀ P O_R = P that
        // symmetrize_matrix applies — is the contract _compute_2e_fock_symm requires,
        // because the Coulomb/exchange contraction Σ_λσ P_λσ (μν|λσ) is only symmetry-
        // invariant for a contravariant P. The two laws coincide for orthogonal O_R
        // (s,p shells) but differ for Cartesian d (and higher) under a non-monomial
        // operation (C₃/S₄/…). Build a contravariant symmetric density by averaging
        // O_R P_raw O_Rᵀ over the group.
        const Eigen::MatrixXd P_raw = random_symmetric(static_cast<Eigen::Index>(nb), 2024u);
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(nb), static_cast<Eigen::Index>(nb));
        for (const auto &op : ops_res->operations)
            P.noalias() += op.matrix * P_raw * op.matrix.transpose();
        P /= static_cast<double>(ops_res->operations.size());

        // Correctness guard: _symm must equal production (all engines) on this P,
        // or the timing would be meaningless. After the contravariant-density fix
        // this holds to ~1e-12 even for d-shells (was 0.07 with a covariant P).
        const Eigen::MatrixXd ref = HartreeFock::ObaraSaika::_compute_2e_fock(pairs, P, nb);
        auto chk = HartreeFock::ObaraSaika::_compute_2e_fock_symm(pairs, calc._shells, P, nb, *ops_res);
        const Eigen::MatrixXd ref_r = HartreeFock::RysQuad::_compute_2e_fock(pairs, P, nb);
        auto chk_r = HartreeFock::RysQuad::_compute_2e_fock_symm(pairs, calc._shells, P, nb, *ops_res);
        const Eigen::MatrixXd ref_h = HartreeFock::HeadGordonPople::_compute_2e_fock(pairs, P, nb);
        auto chk_h = HartreeFock::HeadGordonPople::_compute_2e_fock_symm(pairs, calc._shells, P, nb, *ops_res);
        const double diff_os = chk ? (*chk - ref).cwiseAbs().maxCoeff() : 1e9;
        const double diff_rys = chk_r ? (*chk_r - ref_r).cwiseAbs().maxCoeff() : 1e9;
        const double diff_hgp = chk_h ? (*chk_h - ref_h).cwiseAbs().maxCoeff() : 1e9;
        if (diff_os > 1e-9 || diff_rys > 1e-9 || diff_hgp > 1e-9)
        {
            std::printf("\n[FAIL] %s  (%s, |G|=%d, nbasis=%zu): _symm disagrees with "
                        "production (OS=%.2e, Rys=%.2e, HGP=%.2e) — benchmark skipped\n",
                        name.c_str(), calc._molecule._point_group.c_str(), gorder, nb,
                        diff_os, diff_rys, diff_hgp);
            g_ok = false;
            return;
        }

        // ── OS engine ────────────────────────────────────────────────────────────
        const double os_nosym = time_ms(reps, [&]
                                        { volatile double s = HartreeFock::ObaraSaika::_compute_2e_fock(pairs, P, nb)(0, 0); (void)s; });
        const double os_d2h = time_ms(reps, [&]
                                      { volatile double s = HartreeFock::ObaraSaika::_compute_2e_fock(pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, d2h)(0, 0); (void)s; });
        const double os_full = time_ms(reps, [&]
                                       { auto g = HartreeFock::ObaraSaika::_compute_2e_fock_symm(pairs, calc._shells, P, nb, *ops_res); volatile double s = (*g)(0, 0); (void)s; });

        // ── Rys engine ───────────────────────────────────────────────────────────
        const double ry_nosym = time_ms(reps, [&]
                                        { volatile double s = HartreeFock::RysQuad::_compute_2e_fock(pairs, P, nb)(0, 0); (void)s; });
        const double ry_d2h = time_ms(reps, [&]
                                      { volatile double s = HartreeFock::RysQuad::_compute_2e_fock(pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, d2h)(0, 0); (void)s; });
        const double ry_full = time_ms(reps, [&]
                                       { auto g = HartreeFock::RysQuad::_compute_2e_fock_symm(pairs, calc._shells, P, nb, *ops_res); volatile double s = (*g)(0, 0); (void)s; });

        // ── HGP engine ───────────────────────────────────────────────────────────
        const double hg_nosym = time_ms(reps, [&]
                                        { volatile double s = HartreeFock::HeadGordonPople::_compute_2e_fock(pairs, P, nb)(0, 0); (void)s; });
        const double hg_d2h = time_ms(reps, [&]
                                      { volatile double s = HartreeFock::HeadGordonPople::_compute_2e_fock(pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, d2h)(0, 0); (void)s; });
        const double hg_full = time_ms(reps, [&]
                                       { auto g = HartreeFock::HeadGordonPople::_compute_2e_fock_symm(pairs, calc._shells, P, nb, *ops_res); volatile double s = (*g)(0, 0); (void)s; });

        // ── Auto-dispatch (HGP for L_AB+L_CD>=2, Rys for the (ss|ss)/(sp|ss)/(ss|sp) tail) ──
        //
        // Auto routes through RysQuad::_compute_2e_fock_auto, which builds the
        // ERI tensor via the per-quartet HGP/Rys predicate (see
        // src/integrals/rys.cpp::_auto_prefers_rys). No symmetry-reduced
        // variant of the auto Fock build exists today, so the d2h and full
        // columns here delegate back to HGP — they exist only to keep the
        // table column-aligned with the per-engine rows.
        const double au_nosym = time_ms(reps, [&]
                                        { volatile double s = HartreeFock::RysQuad::_compute_2e_fock_auto(pairs, P, nb)(0, 0); (void)s; });
        const double au_d2h = time_ms(reps, [&]
                                      { volatile double s = HartreeFock::RysQuad::_compute_2e_fock_auto(pairs, P, nb, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10, d2h)(0, 0); (void)s; });
        // No native auto-symm path — reuse HGP-symm so the column has a value.
        const double au_full = hg_full;

        std::printf("\n%s  (%s, |G|=%d, D2h ops=%zu, nbasis=%zu)\n",
                    name.c_str(), calc._molecule._point_group.c_str(),
                    gorder, d2h_ops, nb);
        std::printf("  %-5s  %10s  %10s  %10s   %8s  %8s\n",
                    "eng", "nosym ms", "d2h ms", "full ms", "d2h x", "full x");
        std::printf("  %-5s  %10.3f  %10.3f  %10.3f   %7.2fx  %7.2fx\n",
                    "OS", os_nosym, os_d2h, os_full,
                    os_nosym / os_d2h, os_nosym / os_full);
        std::printf("  %-5s  %10.3f  %10.3f  %10.3f   %7.2fx  %7.2fx\n",
                    "Rys", ry_nosym, ry_d2h, ry_full,
                    ry_nosym / ry_d2h, ry_nosym / ry_full);
        std::printf("  %-5s  %10.3f  %10.3f  %10.3f   %7.2fx  %7.2fx\n",
                    "HGP", hg_nosym, hg_d2h, hg_full,
                    hg_nosym / hg_d2h, hg_nosym / hg_full);
        std::printf("  %-5s  %10.3f  %10.3f  %10.3f   %7.2fx  %7.2fx\n",
                    "Auto", au_nosym, au_d2h, au_full,
                    au_nosym / au_d2h, au_nosym / au_full);
    }
} // namespace

int main(int argc, char **argv)
{
    int reps = 7;
    if (argc > 1)
        reps = std::max(1, std::atoi(argv[1]));

    std::printf("Full-symmetry direct-Fock benchmark (median of %d reps; speedups vs nosym)\n", reps);
#ifdef USE_OPENMP
    std::printf("OpenMP: ENABLED\n");
#else
    std::printf("OpenMP: disabled\n");
#endif

    // C2v (|G|=4) — production already reduces here (Cs/D2h subgroup).
    bench("H2O / C2v",
          {8, 1, 1},
          {{{0.000000, 0.000000, 0.117176}},
           {{0.000000, 0.757200, -0.468704}},
           {{0.000000, -0.757200, -0.468704}}},
          0, 1, "STO-3G", reps);

    // C3v (|G|=6) — full group exceeds the D2h-monomial ceiling.
    bench("NH3 / C3v",
          {7, 1, 1, 1},
          {{{0.0000, 0.0000, 0.1173}},
           {{0.0000, 0.9377, -0.2738}},
           {{0.8121, -0.4689, -0.2738}},
           {{-0.8121, -0.4689, -0.2738}}},
          0, 1, "STO-3G", reps);

    // Td (|G|=24) — non-Abelian; the highest-leverage common case.
    bench("CH4 / Td",
          {6, 1, 1, 1, 1},
          {{{0.000000, 0.000000, 0.000000}},
           {{0.629118, 0.629118, 0.629118}},
           {{-0.629118, -0.629118, 0.629118}},
           {{-0.629118, 0.629118, -0.629118}},
           {{0.629118, -0.629118, -0.629118}}},
          0, 1, "STO-3G", reps);

    // ── Larger basis (6-31G**): more functions/primitives per shell shift cost
    //    toward integral compute, where the petite-list reduction pays off and the
    //    per-quartet orbit bookkeeping is amortized. This is where _symm is meant
    //    to win; STO-3G above is bookkeeping-dominated.
    bench("H2O / C2v  [6-31G**]",
          {8, 1, 1},
          {{{0.000000, 0.000000, 0.117176}},
           {{0.000000, 0.757200, -0.468704}},
           {{0.000000, -0.757200, -0.468704}}},
          0, 1, "6-31G**", reps);

    bench("NH3 / C3v  [6-31g]",
          {7, 1, 1, 1},
          {{{0.0000, 0.0000, 0.1173}},
           {{0.0000, 0.9377, -0.2738}},
           {{0.8121, -0.4689, -0.2738}},
           {{-0.8121, -0.4689, -0.2738}}},
          0, 1, "6-31g", reps);

    bench("NH3 / C3v  [6-31g*]",
          {7, 1, 1, 1},
          {{{0.0000, 0.0000, 0.1173}},
           {{0.0000, 0.9377, -0.2738}},
           {{0.8121, -0.4689, -0.2738}},
           {{-0.8121, -0.4689, -0.2738}}},
          0, 1, "6-31g*", reps);

    bench("NH3 / C3v  [6-31G**]",
          {7, 1, 1, 1},
          {{{0.0000, 0.0000, 0.1173}},
           {{0.0000, 0.9377, -0.2738}},
           {{0.8121, -0.4689, -0.2738}},
           {{-0.8121, -0.4689, -0.2738}}},
          0, 1, "6-31G**", reps);

    bench("CH4 / Td  [6-31G**]",
          {6, 1, 1, 1, 1},
          {{{0.000000, 0.000000, 0.000000}},
           {{0.629118, 0.629118, 0.629118}},
           {{-0.629118, -0.629118, 0.629118}},
           {{-0.629118, 0.629118, -0.629118}},
           {{0.629118, -0.629118, -0.629118}}},
          0, 1, "6-31G**", reps);

    if (!g_ok)
        std::cerr << "\nsymm_fock_benchmark: one or more cases failed setup/correctness\n";
    return g_ok ? 0 : 1;
}
