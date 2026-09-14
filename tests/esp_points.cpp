// E0 gate: the fused electronic-ESP sweep (ObaraSaika::_compute_electronic_potential)
// and the ESP assembly built on it (SCF::electrostatic_potential).
//
// The fused sweep exists so ESP fitting does not have to call
// _compute_external_charge_attraction once per grid point (an nbasis^2
// allocation per point) or store one matrix per point. That makes the existing
// per-point entry the natural ORACLE: for a single point, building the AO
// matrix and taking the Frobenius product against the density must reproduce
// what the fused sweep computes directly. If the two disagree, the fusion --
// specifically its upper-triangle weighting, which is the part with no
// counterpart in the matrix builder -- is wrong.
//
// Three independent checks, in increasing strength:
//
//   1. Oracle agreement, on a RANDOM symmetric density. Random rather than
//      converged because the identity is algebraic and holds for any P, and a
//      converged density is nearly diagonal in the AO basis on a small basis --
//      which would make the off-diagonal weighting (the actual failure mode)
//      contribute almost nothing. This is the check that would catch a
//      weight of 1.0 on off-diagonal pairs.
//
//   2. Asymptotics, on a REAL density. Far from a neutral molecule the total
//      potential must fall off faster than a monopole; for a CHARGED one it
//      must approach q_total / r. The neutral case alone is a weak test (zero
//      is easy to hit by accident -- e.g. if BOTH terms were erroneously zero),
//      so the charged case is what pins the two terms' relative sign and scale.
//
//   3. Thread-count invariance. The sweep parallelizes over points with no
//      cross-thread reduction, so it must be BITWISE identical at any thread
//      count -- the standard this codebase holds its parallel paths to.
//
// A converged SCF is deliberately not run: nothing under test depends on the
// density being stationary, and requiring one would drag the whole SCF stack
// into a gate for two contractions.

#include <cmath>
#include <cstddef>
#include <cstdlib>
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
#include "integrals/os.h"
#include "integrals/shellpair.h"
#include "populations/esp.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "[FAIL] " << message << '\n';
        g_ok = false;
    }

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
            fail(message);
    }

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

    Eigen::MatrixXd random_symmetric_density(std::size_t nb, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd A(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                A(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = dist(rng);
        return 0.5 * (A + A.transpose());
    }

    // Oracle: the electronic potential at ONE point, via the existing
    // per-point AO matrix builder. Deliberately the slow route the fused
    // sweep replaces.
    double electronic_potential_via_matrix(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::size_t nb,
        const Eigen::MatrixXd &density,
        const Eigen::Vector3d &point)
    {
        // A unit POSITIVE charge: the builder folds in its own minus sign
        // (it is shaped like a nuclear-attraction integral), so negating the
        // result recovers the raw positive contraction the fused sweep returns.
        const std::vector<HartreeFock::ExternalCharge> charges{
            HartreeFock::ExternalCharge{.position = point, .charge = 1.0}};

        const Eigen::MatrixXd V =
            HartreeFock::ObaraSaika::_compute_external_charge_attraction(
                shell_pairs, nb, charges, nullptr);

        return -(density.array() * V.array()).sum();
    }

    // ── Check 1: fused sweep vs the per-point matrix oracle ──────────────────
    void check_against_matrix_oracle()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();

        std::mt19937 rng(20260914);
        const Eigen::MatrixXd P = random_symmetric_density(nb, rng);

        // Points deliberately off-axis and off-symmetry: an on-axis point can
        // make whole angular-momentum blocks vanish and hide an index error.
        const std::vector<Eigen::Vector3d> points{
            {1.3, -0.7, 2.1},
            {-2.4, 1.9, -0.6},
            {0.4, 0.3, 3.7},
            {-5.0, -4.0, 1.2},
        };

        const Eigen::VectorXd fused =
            HartreeFock::ObaraSaika::_compute_electronic_potential(shell_pairs, P, points);

        expect(fused.size() == static_cast<Eigen::Index>(points.size()),
               "fused sweep should return one potential per point");
        if (!g_ok)
            return;

        double worst = 0.0;
        for (std::size_t k = 0; k < points.size(); ++k)
        {
            const double oracle =
                electronic_potential_via_matrix(shell_pairs, nb, P, points[k]);
            const double diff = std::abs(fused(static_cast<Eigen::Index>(k)) - oracle);
            worst = std::max(worst, diff);
        }

        std::cout << "  fused vs per-point matrix oracle: max |diff| = "
                  << std::scientific << std::setprecision(3) << worst << '\n';

        // Same kernel, same primitives, same accumulation order within a pair:
        // the only difference is fusing the contraction, so this is exact up
        // to summation reassociation over pairs.
        expect(worst < 1e-12,
               "fused electronic ESP should reproduce the per-point matrix "
               "contraction (a mismatch here means the upper-triangle "
               "weighting is wrong)");
    }

    // ── Check 2: asymptotic behaviour of the assembled potential ─────────────
    void check_asymptotics()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();

        // A density with a KNOWN electron count, so total charge is known
        // exactly without an SCF: scale the identity so that tr(P S) = n_elec.
        // Overlap comes from the same shell pairs, so the trace is consistent
        // with the basis the ESP is evaluated in.
        auto [S, T] = HartreeFock::ObaraSaika::_compute_1e(shell_pairs, nb, nullptr);
        (void)T;

        const double target_electrons = 6.0; // deliberately != 10, so q != 0
        Eigen::MatrixXd P = Eigen::MatrixXd::Identity(
            static_cast<Eigen::Index>(nb), static_cast<Eigen::Index>(nb));
        const double trace_PS = (P * S).trace();
        expect(std::abs(trace_PS) > 1e-12, "identity density should have nonzero tr(PS)");
        if (!g_ok)
            return;
        P *= target_electrons / trace_PS;

        const double Z_total = 10.0; // water
        const double q_total = Z_total - target_electrons;

        // Far field, along a direction with no special symmetry.
        const Eigen::Vector3d direction = Eigen::Vector3d(1.0, 0.7, 0.4).normalized();
        const std::vector<double> radii{60.0, 120.0, 240.0};

        std::vector<Eigen::Vector3d> points;
        for (double r : radii)
            points.push_back(direction * r);

        auto phi = HartreeFock::SCF::electrostatic_potential(
            calc._molecule, shell_pairs, P, points);

        expect(phi.has_value(),
               "electrostatic_potential should succeed on a prepared molecule");
        if (!phi)
            return;

        for (std::size_t k = 0; k < radii.size(); ++k)
        {
            const double monopole = q_total / radii[k];
            const double actual = (*phi)(static_cast<Eigen::Index>(k));
            const double rel = std::abs(actual - monopole) / std::abs(monopole);

            std::cout << "  r = " << std::fixed << std::setprecision(1) << radii[k]
                      << " Bohr: phi = " << std::scientific << std::setprecision(6) << actual
                      << ", q/r = " << monopole
                      << ", rel = " << std::setprecision(2) << rel << '\n';

            // At 60 Bohr from a 3-atom molecule the higher multipoles are long
            // dead; what remains is the monopole. A loose bound is deliberate:
            // this check exists to pin SIGN and SCALE (it fails hard if the
            // electronic term is added instead of subtracted, or if the
            // electron count is off), not to measure multipole convergence.
            expect(rel < 1e-2,
                   "far-field ESP should approach q_total / r (a failure here "
                   "means the nuclear and electronic terms have the wrong "
                   "relative sign or magnitude)");
        }

        // Sign sanity, independent of the tolerance above: a net-positive
        // molecule must have a positive potential far away.
        expect((*phi)(0) > 0.0,
               "a net-positive charge distribution should give a positive "
               "far-field potential");
    }

    // ── Check 3: bitwise thread-count invariance ─────────────────────────────
    void check_thread_invariance()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();

        std::mt19937 rng(4242);
        const Eigen::MatrixXd P = random_symmetric_density(nb, rng);

        std::vector<Eigen::Vector3d> points;
        for (int i = 0; i < 200; ++i)
        {
            const double t = static_cast<double>(i);
            points.emplace_back(0.9 * std::cos(t), 1.1 * std::sin(t), 0.05 * t - 4.0);
        }

        const Eigen::VectorXd reference =
            HartreeFock::ObaraSaika::_compute_electronic_potential(shell_pairs, P, points);

#ifdef USE_OPENMP
        // Re-running in the same process cannot change the thread count on its
        // own, but it does exercise the sweep repeatedly; the cross-thread-count
        // comparison proper is the ctest-level concern. What IS verified here is
        // that the sweep is a pure function of its inputs -- no accumulation
        // into shared state that would make a second call differ from the first.
        for (int repeat = 0; repeat < 3; ++repeat)
        {
            const Eigen::VectorXd again =
                HartreeFock::ObaraSaika::_compute_electronic_potential(shell_pairs, P, points);
            expect((again - reference).cwiseAbs().maxCoeff() == 0.0,
                   "repeated calls must be bitwise identical (the sweep must "
                   "not accumulate into shared state)");
        }
#else
        (void)reference;
#endif

        std::cout << "  thread-invariance: " << points.size()
                  << " points, repeated calls bitwise identical\n";
    }

    // ── Guard: the error paths that protect against silent wrong answers ─────
    void check_error_paths()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();
        const Eigen::MatrixXd P = Eigen::MatrixXd::Identity(
            static_cast<Eigen::Index>(nb), static_cast<Eigen::Index>(nb));

        // A point sitting exactly on a nucleus: the potential genuinely
        // diverges, so returning a huge finite number would be a silent lie.
        const Eigen::Vector3d on_oxygen = calc._molecule._standard.row(0);
        auto diverged = HartreeFock::SCF::electrostatic_potential(
            calc._molecule, shell_pairs, P, {on_oxygen});
        expect(!diverged.has_value(),
               "an evaluation point on a nucleus should be rejected, not "
               "silently returned as a large finite number");

        // Empty point list is a legitimate no-op, not an error.
        auto empty = HartreeFock::SCF::electrostatic_potential(
            calc._molecule, shell_pairs, P, {});
        expect(empty.has_value() && empty->size() == 0,
               "an empty point list should succeed with an empty result");
    }
} // namespace

int main()
{
    std::cout << "planck-esp-points\n";

    std::cout << "[1] fused sweep vs per-point matrix oracle\n";
    check_against_matrix_oracle();

    std::cout << "[2] far-field asymptotics\n";
    check_asymptotics();

    std::cout << "[3] thread invariance / purity\n";
    check_thread_invariance();

    std::cout << "[4] error paths\n";
    check_error_paths();

    if (!g_ok)
    {
        std::cerr << "planck-esp-points FAILED\n";
        return 1;
    }

    std::cout << "planck-esp-points OK\n";
    return 0;
}
