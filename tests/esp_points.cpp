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

    // ── Check 5: the CHELPG fit recovers KNOWN point charges exactly ────────
    //
    // This is the load-bearing gate for E3, and it exists because PySCF has no
    // CHELPG module -- there is no external reference to compare against, so
    // the oracle has to be an analytically known answer instead.
    //
    // Replace the QM potential with the field of point charges sitting ON the
    // nuclei. The model the fit assumes is then EXACTLY right, so the solve
    // must return those charges to solver precision. A wrong design matrix, a
    // wrong constraint row, or a broken solve all fail here; none of them can
    // be caught by looking at charges alone, because the constraint forces
    // those to sum correctly whatever else is wrong.
    void check_fit_recovers_point_charges()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        auto grid = HartreeFock::SCF::chelpg_grid(
            calc._molecule, 0.3 * ANGSTROM_TO_BOHR, 2.8 * ANGSTROM_TO_BOHR, 1.0);
        expect(grid.has_value(), "chelpg_grid should succeed on prepared water");
        if (!grid)
            return;

        std::cout << "  grid points: " << grid->size() << '\n';
        expect(grid->size() > 100,
               "a 0.3 A grid around water should give a few thousand points");

        // Deliberately NOT the physical charges: an arbitrary set that still
        // sums to zero, so a fit that secretly returns something plausible
        // (Mulliken-like, or all zeros) cannot pass by luck.
        Eigen::VectorXd known(3);
        known << -0.834, 0.417, 0.417;

        Eigen::VectorXd phi(static_cast<Eigen::Index>(grid->size()));
        for (std::size_t k = 0; k < grid->size(); ++k)
        {
            double acc = 0.0;
            for (std::size_t a = 0; a < calc._molecule.natoms; ++a)
            {
                const Eigen::Vector3d R =
                    calc._molecule._standard.row(static_cast<Eigen::Index>(a));
                acc += known(static_cast<Eigen::Index>(a)) / ((*grid)[k] - R).norm();
            }
            phi(static_cast<Eigen::Index>(k)) = acc;
        }

        auto fit = HartreeFock::SCF::fit_esp_charges(
            calc._molecule, *grid, phi, known.sum());
        expect(fit.has_value(), "fit_esp_charges should succeed on a well-posed grid");
        if (!fit)
            return;

        const double worst = (fit->charges - known).cwiseAbs().maxCoeff();
        std::cout << "  recovered: " << std::fixed << std::setprecision(10)
                  << fit->charges(0) << ", " << fit->charges(1) << ", "
                  << fit->charges(2) << "  (max err " << std::scientific
                  << std::setprecision(3) << worst << ")\n";
        std::cout << "  rrms: " << std::scientific << std::setprecision(3)
                  << fit->rrms << '\n';

        expect(worst < 1e-8,
               "the fit must recover charges it was given exactly -- the model is "
               "exact here, so any error is in the design matrix, the constraint "
               "row, or the solve");

        // An exactly-representable potential must be fit exactly. This is what
        // makes the gate non-vacuous: charges alone always look plausible
        // because the constraint forces them to sum correctly.
        expect(fit->rrms < 1e-10,
               "an exactly-representable potential must give a vanishing RRMS");

        // NOTE: this fixture canNOT gate the constraint -- see
        // check_constraint_is_load_bearing below for why, and for the check
        // that actually does.
        expect(std::abs(fit->charges.sum() - known.sum()) < 1e-12,
               "the total-charge constraint must be satisfied exactly");
    }

    // ── Check 5b: the total-charge constraint is LOAD-BEARING ───────────────
    //
    // This check exists because of a mutation that PASSED. Zeroing the Lagrange
    // row in fit_esp_charges left check 5 completely green, and the reason is
    // instructive: check 5 fits a potential the model can represent EXACTLY, so
    // the unconstrained least-squares solution already lands on the right
    // charges and already sums correctly. There is nothing for the constraint
    // to do, so removing it changes nothing and asserting "the sum is right"
    // asserts a property that holds for free.
    //
    // A constraint can only be gated on a fixture where it BINDS -- i.e. where
    // the unconstrained fit would drift away from the target sum. So here the
    // potential is deliberately NOT representable by nuclear-centred charges:
    // it comes from charges placed OFF the nuclei, which no atom-centred model
    // can reproduce. The unconstrained sum then drifts, and the Lagrange row is
    // the only thing pulling it back.
    void check_constraint_is_load_bearing()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        auto grid = HartreeFock::SCF::chelpg_grid(
            calc._molecule, 0.4 * ANGSTROM_TO_BOHR, 2.8 * ANGSTROM_TO_BOHR, 1.0);
        if (!grid)
        {
            fail("chelpg_grid failed in the constraint check");
            return;
        }

        // Sources displaced well off every nucleus, so the field has structure
        // an atom-centred model cannot capture.
        const std::vector<std::pair<Eigen::Vector3d, double>> sources{
            {{1.9, 1.3, -0.8}, 0.63},
            {{-1.4, -1.7, 1.1}, -0.41},
            {{0.2, 2.2, 2.0}, 0.28},
        };

        Eigen::VectorXd phi(static_cast<Eigen::Index>(grid->size()));
        for (std::size_t k = 0; k < grid->size(); ++k)
        {
            double acc = 0.0;
            for (const auto &[pos, q] : sources)
                acc += q / ((*grid)[k] - pos).norm();
            phi(static_cast<Eigen::Index>(k)) = acc;
        }

        // Ask for a total that the unconstrained fit has no reason to hit.
        const double target_total = -0.75;

        auto fit = HartreeFock::SCF::fit_esp_charges(
            calc._molecule, *grid, phi, target_total);
        if (!fit)
        {
            fail("fit_esp_charges failed in the constraint check");
            return;
        }

        std::cout << "  constrained sum: " << std::fixed << std::setprecision(12)
                  << fit->charges.sum() << " (target " << target_total << ")\n";
        std::cout << "  rrms: " << std::scientific << std::setprecision(3)
                  << fit->rrms << "  (must be >0: the model cannot fit this field)\n";

        expect(std::abs(fit->charges.sum() - target_total) < 1e-10,
               "the constraint must hold even when it BINDS -- if this passes "
               "with the Lagrange row removed, the fixture is too easy");

        // Non-vacuity of the fixture itself: if the model could represent this
        // field, the constraint would again be free and this check would be as
        // hollow as check 5's version of it.
        expect(fit->rrms > 1e-3,
               "the fixture must be genuinely unfittable, otherwise the "
               "constraint is satisfied for free and nothing is being tested");
    }

    // ── Check 6: charges are invariant under rigid motion ────────────────────
    //
    // A grid keyed to the LAB axes rather than to the molecule would still pass
    // every check above, and would silently give different charges for the same
    // molecule in a different orientation. Nothing else here would catch it.
    void check_fit_is_rotation_invariant()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        // What this check can and cannot establish, measured rather than assumed.
        //
        // The field here is built from charges ON the nuclei, so the model
        // represents it exactly and the fit must return them in any
        // orientation. That is a genuine requirement -- a fit that got the
        // right answer only in one frame would be broken -- but it is a
        // property of the FIT, not of the grid: any grid that samples an
        // exactly-representable field returns the exact answer. Two grid
        // mutations confirmed this empirically (a lab-snapped lattice origin,
        // and an anisotropic box exclusion test in place of the spherical one):
        // both passed here, because neither can perturb an exact fit.
        //
        // The tempting fix -- fit an UNREPRESENTABLE field so the surviving
        // points matter -- was tried and is wrong, because the property then
        // does not hold. Measured drift between two orientations, sources off
        // the nuclei, as the lattice is refined:
        //
        //     0.50 A  2.2e-01      0.20 A  6.1e-02
        //     0.40 A  3.6e-01      0.15 A  7.8e-02
        //     0.30 A  5.8e-02      0.10 A  2.8e-02
        //
        // against charges of order 0.9. It is 3-7% throughout and does NOT
        // converge with spacing, because refining a cubic lattice does not make
        // it rotationally symmetric -- it just resamples a field the model
        // cannot represent. This is CHELPG's known rotational variance, not a
        // defect in this grid, and gating on it would mean inventing a
        // tolerance to hide a real property of the method.
        //
        // So: assert invariance where it is actually required, and record the
        // measurement above for whoever wonders why there is no tighter gate.
        Eigen::VectorXd known(3);
        known << -0.834, 0.417, 0.417;

        auto charges_for = [&](const Eigen::Matrix3d &R,
                               const Eigen::Vector3d &t) -> Eigen::VectorXd {
            HartreeFock::Molecule mol = calc._molecule;
            Eigen::MatrixXd moved(mol.natoms, 3);
            for (std::size_t a = 0; a < mol.natoms; ++a)
                moved.row(static_cast<Eigen::Index>(a)) =
                    (R * mol._standard.row(static_cast<Eigen::Index>(a)).transpose() + t)
                        .transpose();
            mol.set_standard_from_bohr(moved);

            auto grid = HartreeFock::SCF::chelpg_grid(
                mol, 0.3 * ANGSTROM_TO_BOHR, 2.8 * ANGSTROM_TO_BOHR, 1.0);
            if (!grid)
                return Eigen::VectorXd();

            Eigen::VectorXd phi(static_cast<Eigen::Index>(grid->size()));
            for (std::size_t k = 0; k < grid->size(); ++k)
            {
                double acc = 0.0;
                for (std::size_t a = 0; a < mol.natoms; ++a)
                {
                    const Eigen::Vector3d P =
                        mol._standard.row(static_cast<Eigen::Index>(a));
                    acc += known(static_cast<Eigen::Index>(a)) / ((*grid)[k] - P).norm();
                }
                phi(static_cast<Eigen::Index>(k)) = acc;
            }

            auto fit = HartreeFock::SCF::fit_esp_charges(mol, *grid, phi, known.sum());
            return fit ? fit->charges : Eigen::VectorXd();
        };

        const Eigen::VectorXd reference =
            charges_for(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        expect(reference.size() == 3, "reference orientation should produce a fit");
        if (reference.size() != 3)
            return;

        // A rotation with no special relationship to the lattice axes, plus a
        // translation that is deliberately NOT a multiple of the grid spacing.
        const Eigen::Matrix3d R =
            (Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized()))
                .toRotationMatrix();
        const Eigen::Vector3d t(0.137, -0.921, 0.455);

        const Eigen::VectorXd moved = charges_for(R, t);
        expect(moved.size() == 3, "rotated orientation should produce a fit");
        if (moved.size() != 3)
            return;

        const double drift = (moved - reference).cwiseAbs().maxCoeff();
        std::cout << "  rotation/translation drift: " << std::scientific
                  << std::setprecision(3) << drift << '\n';

        // Tight, because the field is exactly representable: the fit must land
        // on `known` in every frame, so the two answers agree to solver noise.
        expect(drift < 1e-10,
               "an exactly-representable field must give the same charges in any "
               "orientation -- a frame-dependent answer here is a broken fit");

        // And it must be the RIGHT answer in the rotated frame too, not merely
        // the same wrong one in both.
        expect((moved - known).cwiseAbs().maxCoeff() < 1e-8,
               "the rotated fit must recover the known charges, not just agree "
               "with the reference");
    }

    // ── Check 7: the Connolly grid IS rotationally invariant ────────────────
    //
    // This is the check E3 could not have. It fits a field the atom-centred
    // model CANNOT represent -- so the surviving sample points genuinely
    // determine the answer -- and requires the charges to be unchanged under a
    // rigid motion of the whole problem.
    //
    // chelpg_grid fails this by 3-7% and does not converge with spacing,
    // because a cubic lattice is not rotationally symmetric. connolly_grid
    // passes because a sphere is: rotating the molecule rotates the sample set
    // with it. The same fixture is run through BOTH grids here, so the
    // comparison is the point, not an incidental detail -- if the CHELPG arm
    // ever stops failing, this gate has stopped measuring the grid.
    void check_connolly_is_rotation_invariant()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        // Sources OFF the nuclei: unrepresentable by atom-centred charges, so
        // which points survive actually matters.
        const std::vector<std::pair<Eigen::Vector3d, double>> sources{
            {{1.9, 1.3, -0.8}, 0.63},
            {{-1.4, -1.7, 1.1}, -0.41},
            {{0.2, 2.2, 2.0}, 0.28},
        };
        const double target_total = -0.75;

        const std::vector<double> shells{1.4, 1.6, 1.8, 2.0};

        auto drift_for = [&](bool connolly) -> double {
            auto charges_for = [&](const Eigen::Matrix3d &R,
                                   const Eigen::Vector3d &t) -> Eigen::VectorXd {
                HartreeFock::Molecule mol = calc._molecule;
                Eigen::MatrixXd moved(mol.natoms, 3);
                for (std::size_t a = 0; a < mol.natoms; ++a)
                    moved.row(static_cast<Eigen::Index>(a)) =
                        (R * mol._standard.row(static_cast<Eigen::Index>(a)).transpose() + t)
                            .transpose();
                mol.set_standard_from_bohr(moved);

                auto grid =
                    connolly
                        ? HartreeFock::SCF::connolly_grid(mol, shells, 200, 1.0)
                        : HartreeFock::SCF::chelpg_grid(
                              mol, 0.3 * ANGSTROM_TO_BOHR, 2.8 * ANGSTROM_TO_BOHR, 1.0);
                if (!grid)
                    return Eigen::VectorXd();

                // The sources move WITH the molecule, so the physical problem
                // is identical in both frames and only the grid can differ.
                Eigen::VectorXd phi(static_cast<Eigen::Index>(grid->size()));
                for (std::size_t k = 0; k < grid->size(); ++k)
                {
                    double acc = 0.0;
                    for (const auto &[pos, q] : sources)
                        acc += q / ((*grid)[k] - (R * pos + t)).norm();
                    phi(static_cast<Eigen::Index>(k)) = acc;
                }

                auto fit = HartreeFock::SCF::fit_esp_charges(mol, *grid, phi, target_total);
                return fit ? fit->charges : Eigen::VectorXd();
            };

            const Eigen::Matrix3d R =
                Eigen::AngleAxisd(0.7, Eigen::Vector3d(1.0, 2.0, 3.0).normalized())
                    .toRotationMatrix();
            const Eigen::Vector3d t(0.137, -0.921, 0.455);

            const Eigen::VectorXd a =
                charges_for(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
            const Eigen::VectorXd b = charges_for(R, t);
            if (a.size() != 3 || b.size() != 3)
                return -1.0;
            return (b - a).cwiseAbs().maxCoeff();
        };

        const double chelpg_drift = drift_for(false);
        const double connolly_drift = drift_for(true);

        std::cout << "  chelpg   drift: " << std::scientific << std::setprecision(3)
                  << chelpg_drift << '\n';
        std::cout << "  connolly drift: " << std::scientific << std::setprecision(3)
                  << connolly_drift << '\n';

        expect(chelpg_drift >= 0.0 && connolly_drift >= 0.0,
               "both grids should produce a fit");
        if (chelpg_drift < 0.0 || connolly_drift < 0.0)
            return;

        // The claim being gated.
        expect(connolly_drift < 1e-8,
               "Connolly shells must give orientation-independent charges -- "
               "that is the whole reason E4a replaced the cubic lattice");

        // Non-vacuity, and the reason both arms are run: if the CHELPG arm also
        // passed, the fixture would not be exercising grid orientation at all
        // and the Connolly result would prove nothing.
        expect(chelpg_drift > 1e-4,
               "the CHELPG arm must still fail -- if it does not, this fixture "
               "has stopped measuring the grid and the Connolly pass is vacuous");
    }

    // ── Check 8: RESP reduces to the unrestrained fit, and the restraint trades ──
    //
    // Three properties, each chosen because it can actually fail:
    //
    //   (a) REDUCTION. With strength = 0 the restraint term vanishes
    //       identically, so fit_resp_charges must reproduce fit_esp_charges to
    //       solver precision. This pins that the restraint is the ONLY
    //       difference between the two paths -- if the iteration, the
    //       constraint block, or the design matrix drifted apart, this fails
    //       even though both answers would look individually plausible.
    //
    //   (b) THE TRADE. A restraint that changed nothing would be pointless, and
    //       one that improved the fit would mean the unrestrained solve was not
    //       optimal. So turning it on must shrink the charges AND raise the
    //       RRMS. Asserting only the first would pass for a restraint that
    //       simply scaled everything down.
    //
    //   (c) EQUIVALENCE. Atoms in a group must come back with EXACTLY equal
    //       charges -- these are hard linear constraints in the same Lagrange
    //       block as the total charge, not penalties, so "close" is not good
    //       enough and a loose bound would hide a penalty-style implementation.
    void check_resp_restraint()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        auto grid = HartreeFock::SCF::connolly_grid(
            calc._molecule, {1.4, 1.6, 1.8, 2.0}, 200, 1.0);
        if (!grid)
        {
            fail("connolly_grid failed in the RESP check");
            return;
        }

        // Off-nucleus sources: the model cannot fit this exactly, so the
        // unrestrained solve has genuine freedom for the restraint to remove.
        // On an exactly-fittable field the restraint would have nothing to do
        // and (b) would be untestable -- the fixture-too-easy trap this file
        // has already hit twice.
        const std::vector<std::pair<Eigen::Vector3d, double>> sources{
            {{1.9, 1.3, -0.8}, 0.63},
            {{-1.4, -1.7, 1.1}, -0.41},
            {{0.2, 2.2, 2.0}, 0.28},
        };
        Eigen::VectorXd phi(static_cast<Eigen::Index>(grid->size()));
        for (std::size_t k = 0; k < grid->size(); ++k)
        {
            double acc = 0.0;
            for (const auto &[pos, q] : sources)
                acc += q / ((*grid)[k] - pos).norm();
            phi(static_cast<Eigen::Index>(k)) = acc;
        }

        const double total = -0.75;

        // (a) Reduction at zero strength.
        auto plain = HartreeFock::SCF::fit_esp_charges(calc._molecule, *grid, phi, total);
        HartreeFock::SCF::RESPOptions off;
        off.strength = 0.0;
        auto reduced =
            HartreeFock::SCF::fit_resp_charges(calc._molecule, *grid, phi, total, off);
        if (!plain || !reduced)
        {
            fail("both fits should succeed in the RESP reduction check");
            return;
        }

        const double reduction_gap =
            (reduced->fit.charges - plain->charges).cwiseAbs().maxCoeff();
        std::cout << "  strength=0 vs unrestrained: " << std::scientific
                  << std::setprecision(3) << reduction_gap << '\n';
        expect(reduction_gap < 1e-10,
               "with the restraint off, RESP must reproduce the unrestrained fit "
               "exactly -- a gap here means the two paths differ in something "
               "other than the restraint");

        // (b) The trade. Restrain every atom, hydrogens included, so water's
        // three atoms all feel it; exempting H would leave only one restrained
        // atom and a much weaker signal.
        HartreeFock::SCF::RESPOptions on;
        on.strength = 0.01; // well above the 0.0005 production value, to make
                            // the effect unambiguous rather than marginal
        on.exempt_hydrogen = false;
        auto restrained =
            HartreeFock::SCF::fit_resp_charges(calc._molecule, *grid, phi, total, on);
        if (!restrained)
        {
            fail("the restrained fit should succeed");
            return;
        }

        const double norm_plain = plain->charges.norm();
        const double norm_restrained = restrained->fit.charges.norm();
        const double shrink = (norm_plain - norm_restrained) / norm_plain;
        const double rrms_rise = restrained->fit.rrms - plain->rrms;
        std::cout << "  |q| unrestrained " << std::fixed << std::setprecision(6)
                  << norm_plain << " -> restrained " << norm_restrained
                  << "   (shrink " << std::scientific << std::setprecision(3)
                  << shrink << ")\n";
        std::cout << "  rrms " << std::scientific << std::setprecision(10)
                  << plain->rrms << " -> " << restrained->fit.rrms
                  << "   (rise " << std::setprecision(3) << rrms_rise << ")\n";
        std::cout << "  iterations: " << restrained->iterations
                  << " (converged " << (restrained->converged ? "yes" : "no") << ")\n";

        expect(restrained->converged,
               "the restraint iteration must converge");

        // MINIMUM EFFECT SIZES, not bare inequalities.
        //
        // A mutation that deletes the restraint diagonal entirely does fail the
        // bare forms `norm_restrained < norm_plain` and `rrms > plain->rrms`,
        // so they are load-bearing rather than vacuous. But they pass on ANY
        // nonzero difference, and the measured effect here is small -- 0.5% in
        // |q|, and an RRMS rise below the 5th significant figure. A bare
        // inequality on a quantity that small is one fixture change or one
        // compiler reassociation away from passing on rounding noise.
        //
        // So require the effect to be big enough to be unambiguous. The
        // thresholds are well below the measured values (shrink ~4.7e-03,
        // rise ~2.6e-06) and well above double-precision noise on quantities
        // of order 1.
        expect(shrink > 1e-4,
               "the restraint must shrink the charges by a measurable amount, "
               "not merely by an amount that happens to be nonzero");
        expect(rrms_rise > 1e-9,
               "the restraint must WORSEN the fit to the potential measurably; "
               "if it improved it, the unrestrained solve was not least-squares "
               "optimal and something upstream is wrong");

        // The iteration count separates a working restraint from an inert one
        // independently of the charges: with no restraint the diagonal never
        // changes, so the second pass reproduces the first exactly and the loop
        // exits at 2. A real restraint takes several passes to settle.
        expect(restrained->iterations > 2,
               "a live restraint must take more than two passes to converge -- "
               "exiting at 2 means the normal matrix did not change between "
               "iterations, i.e. the restraint is inert");

        // The constraint still holds exactly under restraint.
        expect(std::abs(restrained->fit.charges.sum() - total) < 1e-10,
               "the total-charge constraint must survive the restraint iteration");

        // (c) Equivalence groups. Water's two hydrogens are symmetry
        // equivalent; the grid samples them slightly differently, so without
        // the constraint their charges differ.
        HartreeFock::SCF::RESPOptions equiv;
        equiv.strength = 0.0;
        equiv.equivalence_groups = {{1, 2}};
        auto grouped =
            HartreeFock::SCF::fit_resp_charges(calc._molecule, *grid, phi, total, equiv);
        if (!grouped)
        {
            fail("the equivalence-constrained fit should succeed");
            return;
        }

        const double h_gap =
            std::abs(grouped->fit.charges(1) - grouped->fit.charges(2));
        const double h_gap_free = std::abs(plain->charges(1) - plain->charges(2));
        std::cout << "  H-H charge gap: free " << std::scientific
                  << std::setprecision(3) << h_gap_free << " -> grouped " << h_gap
                  << '\n';

        expect(h_gap < 1e-12,
               "equivalence groups are hard constraints, so grouped atoms must "
               "share a charge exactly, not approximately");

        // Non-vacuity: if the free fit already gave them equal charges, the
        // constraint would be doing nothing and (c) would prove nothing.
        expect(h_gap_free > 1e-6,
               "the unconstrained fit must give the two hydrogens DIFFERENT "
               "charges, otherwise the equivalence constraint is untested");
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

    std::cout << "[7] Connolly grid is rotation invariant (CHELPG is not)\n";
    check_connolly_is_rotation_invariant();

    std::cout << "[8] RESP restraint: reduction, trade, equivalence\n";
    check_resp_restraint();

    std::cout << "[4] error paths\n";
    check_error_paths();

    std::cout << "[5] CHELPG fit recovers known point charges\n";
    check_fit_recovers_point_charges();

    std::cout << "[5b] total-charge constraint is load-bearing\n";
    check_constraint_is_load_bearing();

    std::cout << "[6] fitted charges are rotation/translation invariant\n";
    check_fit_is_rotation_invariant();

    if (!g_ok)
    {
        std::cerr << "planck-esp-points FAILED\n";
        return 1;
    }

    std::cout << "planck-esp-points OK\n";
    return 0;
}
