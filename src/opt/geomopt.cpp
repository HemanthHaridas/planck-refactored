#include "geomopt.h"
#include "intcoords.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <format>

#include "io/logging.h"
#include "basis/basis.h"
#include "integrals/shellpair.h"
#include "integrals/base.h"
#include "scf/scf.h"
#include "gradient/gradient.h"
#include "post_hf/mp2.h"
#include "base/tables.h"
#include "symmetry/integral_symmetry.h"

// ─── Single-point helper ─────────────────────────────────────────────────────
//
// Re-runs basis construction, integrals, and SCF for the geometry currently
// stored in calc._molecule._standard.  Also computes the analytic gradient.
// Updates calc._total_energy, calc._gradient, and calc._nuclear_repulsion.
// Returns the flat (3*natoms) gradient vector.

static Eigen::VectorXd _run_sp_gradient(HartreeFock::Calculator& calc)
{
    const std::size_t natoms = calc._molecule.natoms;

    // Update input-frame coordinates from _standard (Bohr) for consistency
    calc._molecule._coordinates = calc._molecule._standard;
    calc._molecule.coordinates  = calc._molecule._standard / ANGSTROM_TO_BOHR;

    // Re-read basis from updated geometry (_standard used for shell centers)
    const std::string gbs_path = calc._basis._basis_path + "/" + calc._basis._basis_name;
    calc._shells = HartreeFock::BasisFunctions::read_gbs_basis(
        gbs_path, calc._molecule, calc._basis._basis);

    // Save converged density from the previous step to warm-start the next SCF.
    // (initialize() zeros the density, so we must capture it beforehand.)
    const Eigen::MatrixXd prev_alpha = calc._info._scf.alpha.density;
    const Eigen::MatrixXd prev_beta  = calc._info._scf.beta.density;
    const bool have_prev_density     = (prev_alpha.size() > 0);

    // Reset SCF data structures
    calc._info._scf = HartreeFock::DataSCF(calc._scf._scf == HartreeFock::SCFType::UHF);
    calc._info._scf.initialize(calc._shells.nbasis());
    calc._scf.set_scf_mode_auto(calc._shells.nbasis());
    calc._info._is_converged = false;
    calc._use_sao_blocking   = false;

    // Restore the saved density and tell SCF to use it as the initial guess.
    // Save and restore _scf._guess so this warm-start doesn't leak into subsequent
    // operations (e.g. Hessian SCF calls, post-opt symmetry SCF).
    const auto saved_guess = calc._scf._guess;
    if (have_prev_density)
    {
        calc._info._scf.alpha.density = prev_alpha;
        if (calc._scf._scf == HartreeFock::SCFType::UHF && prev_beta.size() > 0)
            calc._info._scf.beta.density = prev_beta;
        calc._scf._guess = HartreeFock::SCFGuess::ReadDensity;
    }

    // Nuclear repulsion
    calc._compute_nuclear_repulsion();

    // Shell pairs
    auto shell_pairs = build_shellpairs(calc._shells);
    HartreeFock::Symmetry::update_integral_symmetry(calc);

    // 1e integrals
    auto [S, T] = _compute_1e(shell_pairs, calc._shells.nbasis(), calc._integral._engine,
                              calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
    auto V = _compute_nuclear_attraction(shell_pairs, calc._shells.nbasis(),
                                         calc._molecule, calc._integral._engine,
                                         calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
    calc._overlap = S;
    calc._hcore   = T + V;

    // SCF
    std::expected<void, std::string> scf_res;
    if (calc._scf._scf == HartreeFock::SCFType::UHF)
        scf_res = HartreeFock::SCF::run_uhf(calc, shell_pairs);
    else
        scf_res = HartreeFock::SCF::run_rhf(calc, shell_pairs);

    // Restore the original guess mode so it doesn't contaminate subsequent SCF calls.
    calc._scf._guess = saved_guess;

    if (!scf_res)
        throw std::runtime_error("GeomOpt SCF failed: " + scf_res.error());

    // Post-HF correction / gradient
    Eigen::MatrixXd grad_mat;
    if (calc._correlation == HartreeFock::PostHF::RMP2)
    {
        if (auto corr_res = HartreeFock::Correlation::run_rmp2(calc, shell_pairs); !corr_res)
            throw std::runtime_error("GeomOpt RMP2 failed: " + corr_res.error());
        calc._total_energy += calc._correlation_energy;
        grad_mat = HartreeFock::Gradient::compute_rmp2_gradient(calc);
    }
    else if (calc._correlation == HartreeFock::PostHF::UMP2)
    {
        throw std::runtime_error("GeomOpt UMP2 gradient is not implemented");
    }
    else if (calc._scf._scf == HartreeFock::SCFType::UHF)
        grad_mat = HartreeFock::Gradient::compute_uhf_gradient(calc, shell_pairs);
    else
        grad_mat = HartreeFock::Gradient::compute_rhf_gradient(calc, shell_pairs);

    calc._gradient = grad_mat;

    // Return flat gradient vector (natoms*3) in atom-major order [x0,y0,z0, x1,y1,z1,...]
    // (consistent with how _standard is flattened: x[a*3+k])
    Eigen::VectorXd g_am(natoms * 3);
    for (std::size_t a = 0; a < natoms; ++a)
        for (int k = 0; k < 3; ++k)
            g_am[a * 3 + k] = grad_mat(a, k);
    return g_am;
}

// ─── Per-step geometry logger ─────────────────────────────────────────────────
//
// Logs the current geometry (Angstrom) using the same
// {Z:5d}{x:10.3f}{y:10.3f}{z:10.3f} format as "Input Coordinates" /
// "Standard Coordinates" in the driver, so the parser can read it.

static void _log_step_geometry(const HartreeFock::Calculator& calc)
{
    const std::size_t natoms = calc._molecule.natoms;
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Step Geometry :", "");
    for (std::size_t a = 0; a < natoms; ++a) {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
            std::format("{:5d}{:10.3f}{:10.3f}{:10.3f}",
                static_cast<int>(calc._molecule.atomic_numbers[a]),
                calc._molecule._standard(a, 0) * BOHR_TO_ANGSTROM,
                calc._molecule._standard(a, 1) * BOHR_TO_ANGSTROM,
                calc._molecule._standard(a, 2) * BOHR_TO_ANGSTROM));
    }
}

// ─── Two-loop L-BFGS search direction ────────────────────────────────────────
//
// Given current gradient g and stored (s_i, y_i, rho_i) history,
// returns the L-BFGS search direction p = -H_k * g.
static Eigen::VectorXd _lbfgs_direction(
    const Eigen::VectorXd& g,
    const std::vector<Eigen::VectorXd>& s_hist,
    const std::vector<Eigen::VectorXd>& y_hist,
    const std::vector<double>& rho_hist)
{
    const int m = static_cast<int>(s_hist.size());
    Eigen::VectorXd q = g;
    std::vector<double> alpha(m);

    // First loop: reverse order
    for (int i = m - 1; i >= 0; --i) {
        alpha[i] = rho_hist[i] * s_hist[i].dot(q);
        q -= alpha[i] * y_hist[i];
    }

    // Initial Hessian scaling: γ = (s_{k-1}·y_{k-1}) / (y_{k-1}·y_{k-1})
    double gamma = 1.0;
    if (m > 0) {
        const double sy = s_hist.back().dot(y_hist.back());
        const double yy = y_hist.back().squaredNorm();
        if (yy > 1e-30) gamma = sy / yy;
    }
    Eigen::VectorXd r = gamma * q;

    // Second loop: forward order
    for (int i = 0; i < m; ++i) {
        const double beta = rho_hist[i] * y_hist[i].dot(r);
        r += (alpha[i] - beta) * s_hist[i];
    }

    return -r;
}

// ─── Main geometry optimizer ──────────────────────────────────────────────────

HartreeFock::Opt::GeomOptResult HartreeFock::Opt::run_geomopt(
    HartreeFock::Calculator& calc)
{
    const std::size_t natoms = calc._molecule.natoms;
    const int max_iter       = calc._geomopt_max_iter;
    const double grad_tol    = calc._geomopt_grad_tol;
    const int lbfgs_m        = calc._geomopt_lbfgs_m;

    GeomOptResult result;
    result.energies.reserve(max_iter + 1);

    // Current geometry as flat Bohr vector (3*natoms)
    Eigen::VectorXd x(natoms * 3);
    for (std::size_t a = 0; a < natoms; ++a) {
        x[a*3 + 0] = calc._molecule._standard(a, 0);
        x[a*3 + 1] = calc._molecule._standard(a, 1);
        x[a*3 + 2] = calc._molecule._standard(a, 2);
    }

    // Run initial SCF + gradient
    Eigen::VectorXd g = _run_sp_gradient(calc);
    double E = calc._total_energy;
    result.energies.push_back(E);

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Opt Step 0 :",
        std::format("E = {:.10f} Eh   max|g| = {:.3e}   rms|g| = {:.3e}",
                    E,
                    g.cwiseAbs().maxCoeff(),
                    std::sqrt(g.squaredNorm() / static_cast<double>(g.size()))));
    _log_step_geometry(calc);

    // L-BFGS history
    std::vector<Eigen::VectorXd> s_hist, y_hist;
    std::vector<double> rho_hist;
    s_hist.reserve(lbfgs_m);
    y_hist.reserve(lbfgs_m);
    rho_hist.reserve(lbfgs_m);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Check convergence
        const double gmax = g.cwiseAbs().maxCoeff();
        if (gmax < grad_tol) {
            result.converged = true;
            break;
        }

        // Compute L-BFGS search direction
        const Eigen::VectorXd p = _lbfgs_direction(g, s_hist, y_hist, rho_hist);

        // ── Strong Wolfe line search ──────────────────────────────────────────────
        //
        // Strategy: try α=1 first (the natural quasi-Newton step).
        //   • If Armijo+curvature (strong Wolfe) both hold → accept, sy>0 guaranteed.
        //   • If only Armijo holds → accept; guard L-BFGS update with sy>0 check.
        //   • If Armijo fails at α=1 → zoom bisection on [0, 1] to find an Armijo
        //     point, then fall back to pure backtracking if zoom cannot converge.
        //
        // This design limits each iteration to O(zoom_iter + backtrack_iter) SCF
        // calls regardless of curvature, which is important for expensive systems.
        const double c1_w  = 1e-4;   // sufficient decrease (Armijo)
        const double c2_w  = 0.9;    // curvature (standard for L-BFGS)
        const double phi0  = E;
        const double dphi0 = p.dot(g);   // < 0 for a descent direction

        // State at one trial step
        struct LSState {
            double          alpha = 0.0;
            double          f     = 0.0;
            double          df    = 0.0;   // p · g (directional derivative)
            Eigen::VectorXd g;
            Eigen::VectorXd x;
        };

        // Run SCF+gradient at step size a; returns nullopt on SCF failure.
        auto ls_eval = [&](double a) -> std::optional<LSState>
        {
            Eigen::VectorXd x_try = x + a * p;
            for (std::size_t aa = 0; aa < natoms; ++aa) {
                calc._molecule._standard(aa, 0) = x_try[aa*3 + 0];
                calc._molecule._standard(aa, 1) = x_try[aa*3 + 1];
                calc._molecule._standard(aa, 2) = x_try[aa*3 + 2];
            }
            try {
                Eigen::VectorXd g_try = _run_sp_gradient(calc);
                return LSState{a, calc._total_energy, g_try.dot(p), g_try, x_try};
            } catch (...) { return std::nullopt; }
        };

        bool    step_accepted = false;
        bool    wolfe_curvature_met = false;
        LSState accepted_state;

        // ── Step 1: try α = 1 ─────────────────────────────────────────────────
        auto st1 = ls_eval(1.0);
        if (st1 && st1->f <= phi0 + c1_w * dphi0) {
            accepted_state    = *st1;
            step_accepted     = true;
            wolfe_curvature_met = (std::abs(st1->df) <= -c2_w * dphi0);
        }

        // ── Step 2: zoom bisection on (0, 1) when Armijo fails at α=1 ─────────
        if (!step_accepted) {
            LSState state_lo{0.0, phi0, dphi0, g, x};
            double  alpha_hi = 1.0;

            for (int j = 0; j < 20 && !step_accepted; ++j) {
                const double a_j = 0.5 * (state_lo.alpha + alpha_hi);
                auto st = ls_eval(a_j);
                if (!st) { alpha_hi = a_j; continue; }  // SCF failed → shrink hi

                if (st->f > phi0 + c1_w * a_j * dphi0 || st->f >= state_lo.f) {
                    alpha_hi = a_j;   // Armijo violated or not improving → shrink hi
                } else {
                    if (std::abs(st->df) <= -c2_w * dphi0) {
                        // Strong Wolfe — ideal
                        accepted_state      = *st;
                        step_accepted       = true;
                        wolfe_curvature_met = true;
                    } else {
                        // Armijo satisfied; accept even without curvature
                        accepted_state = *st;
                        step_accepted  = true;
                    }
                    if (st->df * (alpha_hi - state_lo.alpha) >= 0.0)
                        alpha_hi = state_lo.alpha;
                    state_lo = *st;
                }
            }
            // Fallback: accept best lo found in zoom if it satisfies Armijo
            if (!step_accepted && state_lo.alpha > 0.0 &&
                state_lo.f <= phi0 + c1_w * state_lo.alpha * dphi0)
            {
                accepted_state = state_lo;
                step_accepted  = true;
            }
        }

        // ── Step 3: pure Armijo backtracking if zoom also failed ──────────────
        if (!step_accepted) {
            for (double a = 0.5; a > 1e-10 && !step_accepted; a *= 0.5) {
                auto st = ls_eval(a);
                if (st && st->f <= phi0 + c1_w * a * dphi0) {
                    accepted_state = *st;
                    step_accepted  = true;
                }
            }
        }

        // ── Accept or give up ─────────────────────────────────────────────────
        if (step_accepted) {
            const Eigen::VectorXd s  = accepted_state.x - x;
            const Eigen::VectorXd y  = accepted_state.g - g;
            const double          sy = s.dot(y);

            x = accepted_state.x;
            g = accepted_state.g;
            E = accepted_state.f;
            result.energies.push_back(E);

            // Only update L-BFGS history when curvature is positive.
            // With strong Wolfe this is guaranteed; otherwise use the guard.
            if (wolfe_curvature_met || sy > 1e-30) {
                s_hist.push_back(s);
                y_hist.push_back(y);
                rho_hist.push_back(1.0 / sy);
                if (static_cast<int>(s_hist.size()) > lbfgs_m) {
                    s_hist.erase(s_hist.begin());
                    y_hist.erase(y_hist.begin());
                    rho_hist.erase(rho_hist.begin());
                }
            }

            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                std::format("Opt Step {} :", iter + 1),
                std::format("E = {:.10f} Eh   max|g| = {:.3e}   rms|g| = {:.3e}",
                            E,
                            g.cwiseAbs().maxCoeff(),
                            std::sqrt(g.squaredNorm() / static_cast<double>(g.size()))));
            _log_step_geometry(calc);
        } else {
            // All line search strategies failed — restore geometry and stop
            for (std::size_t a = 0; a < natoms; ++a) {
                calc._molecule._standard(a, 0) = x[a*3 + 0];
                calc._molecule._standard(a, 1) = x[a*3 + 1];
                calc._molecule._standard(a, 2) = x[a*3 + 2];
            }
            _run_sp_gradient(calc);
            break;
        }
    }

    // Final convergence check
    const double gmax = g.cwiseAbs().maxCoeff();
    if (gmax < grad_tol) result.converged = true;

    result.energy     = E;
    result.iterations = static_cast<int>(result.energies.size()) - 1;
    result.grad_max   = gmax;
    result.grad_rms   = std::sqrt(g.squaredNorm() / static_cast<double>(g.size()));
    result.gradient   = calc._gradient;

    // Store final coordinates (natoms × 3)
    result.final_coords = calc._molecule._standard;

    return result;
}

// ─── IC-based BFGS geometry optimizer ────────────────────────────────────────
//
// Algorithm:
//   1. Build redundant GIC system (bonds, bends, torsions).
//   2. At each step, compute IC gradient g_q = G⁺ B g_x.
//   3. Take BFGS quasi-Newton step Δq = -H_IC⁻¹ g_q with step limiting.
//   4. Back-transform Δq → Δx via microiterations.
//   5. Run SCF+gradient at new geometry; BFGS update of H_IC.
//   6. Convergence on max Cartesian gradient component.
//
// Initial diagonal Hessian (Schlegel 1984):
//   H_ii = 0.5 (stretch), 0.2 (bend), 0.1 (torsion)  [Ha/Bohr² or Ha/rad²]

HartreeFock::Opt::GeomOptResult HartreeFock::Opt::run_geomopt_ic(
    HartreeFock::Calculator& calc)
{
    const std::size_t natoms = calc._molecule.natoms;
    const int  max_iter  = calc._geomopt_max_iter;
    const double grad_tol = calc._geomopt_grad_tol;

    GeomOptResult result;
    result.energies.reserve(max_iter + 1);

    // ── Build IC system ───────────────────────────────────────────────────────
    auto ics = IntCoordSystem::build(calc._molecule._standard,
                                     calc._molecule.atomic_numbers);
    int nq = ics.nics();

    if (nq == 0) {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
            "IC System :", "No internal coordinates found — falling back to Cartesian optimizer");
        return run_geomopt(calc);
    }

    // Count each type for the log message
    int n_stretch = 0, n_bend = 0, n_torsion = 0;
    for (const auto& ic : ics.coords) {
        if (ic.type == ICType::Stretch)  ++n_stretch;
        else if (ic.type == ICType::Bend) ++n_bend;
        else                              ++n_torsion;
    }
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "IC System :",
        std::format("{} stretches, {} bends, {} torsions ({} total)",
                    n_stretch, n_bend, n_torsion, nq));

    // ── Constraint setup ──────────────────────────────────────────────────────
    std::vector<bool> ic_frozen(nq, false);
    std::vector<bool> atom_frozen(natoms, false);

    for (const auto& con : calc._constraints) {
        const int a0 = con.atoms[0] - 1;
        const int a1 = (con.atoms[1] >= 0) ? con.atoms[1] - 1 : -1;
        const int a2 = (con.atoms[2] >= 0) ? con.atoms[2] - 1 : -1;
        const int a3 = (con.atoms[3] >= 0) ? con.atoms[3] - 1 : -1;

        if (con.type == HartreeFock::GeomConstraint::Type::FrozenAtom) {
            if (a0 >= 0 && a0 < (int)natoms) atom_frozen[a0] = true;
            continue;
        }

        InternalCoord ic;
        if (con.type == HartreeFock::GeomConstraint::Type::Bond) {
            ic.type = ICType::Stretch; ic.atoms = {a0, a1, -1, -1};
        } else if (con.type == HartreeFock::GeomConstraint::Type::Angle) {
            ic.type = ICType::Bend; ic.atoms = {a0, a1, a2, -1};
        } else {
            ic.type = ICType::Torsion; ic.atoms = {a0, a1, a2, a3};
        }
        const int idx = ics.add_coord(ic);
        if (idx >= (int)ic_frozen.size())
            ic_frozen.resize(idx + 1, false);
        ic_frozen[idx] = true;
    }
    nq = ics.nics();

    if (!calc._constraints.empty()) {
        int n_frozen_ic = 0, n_frozen_atom = 0;
        for (bool f : ic_frozen)   if (f) ++n_frozen_ic;
        for (bool f : atom_frozen) if (f) ++n_frozen_atom;
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Constraints :",
            std::format("{} IC(s) frozen, {} atom(s) frozen", n_frozen_ic, n_frozen_atom));
    }

    // ── Initial SCF + gradient ────────────────────────────────────────────────
    Eigen::MatrixXd xyz = calc._molecule._standard;
    Eigen::VectorXd g_cart = _run_sp_gradient(calc);
    // Zero frozen-atom gradient contributions
    for (std::size_t a = 0; a < natoms; ++a)
        if (atom_frozen[a]) g_cart.segment(static_cast<int>(a) * 3, 3).setZero();
    double E = calc._total_energy;
    result.energies.push_back(E);

    Eigen::VectorXd g_ic = ics.cart_to_ic_grad(xyz, g_cart);
    const double g_rms = std::sqrt(g_cart.squaredNorm() /
                                   static_cast<double>(g_cart.size()));
    const double g_ic_rms = std::sqrt(g_ic.squaredNorm() /
                                      static_cast<double>(nq));

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Opt Step 0 :",
        std::format("E = {:.10f} Eh   max|g| = {:.3e}   rms|g| = {:.3e}   rms|g_ic| = {:.3e}",
                    E,
                    g_cart.cwiseAbs().maxCoeff(),
                    g_rms,
                    g_ic_rms));
    _log_step_geometry(calc);

    // ── Initial diagonal Hessian ──────────────────────────────────────────────
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nq, nq);
    for (int i = 0; i < nq; ++i) {
        switch (ics.coords[i].type) {
            case ICType::Stretch:  H(i, i) = 0.5; break;
            case ICType::Bend:     H(i, i) = 0.2; break;
            case ICType::Torsion:  H(i, i) = 0.1; break;
        }
    }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // ── Convergence check ─────────────────────────────────────────────────
        const double gmax = g_cart.cwiseAbs().maxCoeff();
        if (gmax < grad_tol) {
            result.converged = true;
            break;
        }

        // Zero constrained IC gradient components before solve
        for (int c = 0; c < nq; ++c)
            if (ic_frozen[c]) g_ic[c] = 0.0;

        // ── Quasi-Newton step in IC space ─────────────────────────────────────
        // Use LDLt for the (possibly indefinite after BFGS) matrix.
        // If LDLT fails or gives a bad direction, fall back to negative gradient.
        Eigen::VectorXd dq = -H.ldlt().solve(g_ic);

        // Guard: if step doesn't look like descent, use steepest descent in IC
        if (dq.dot(g_ic) > 0.0)
            dq = -g_ic;

        // ── Step limiting (trust-radius style) ────────────────────────────────
        // Max displacement: 0.3 Bohr for stretches, 0.3 rad for angles
        double scale = 1.0;
        for (int i = 0; i < nq; ++i) {
            const double limit = (ics.coords[i].type == ICType::Stretch)
                                 ? 0.3 : 0.3;
            if (std::abs(dq[i]) > limit)
                scale = std::max(scale, std::abs(dq[i]) / limit);
        }
        if (scale > 1.0) dq /= scale;

        // Zero constrained IC step components
        for (int c = 0; c < nq; ++c)
            if (ic_frozen[c]) dq[c] = 0.0;

        // ── Back-transform to Cartesian + energy-decrease line search ─────────
        // Halve the IC step until energy decreases, with a minimum-alpha fallback.
        const Eigen::VectorXd dq_full = dq;

        Eigen::MatrixXd xyz_new;
        Eigen::VectorXd g_cart_new;
        double E_new   = E + 1.0;
        bool   step_ok = false;
        double alpha   = 1.0;

        for (int ls = 0; ls < 20 && !step_ok; ++ls) {
            Eigen::VectorXd dq_try = alpha * dq_full;
            xyz_new = ics.ic_to_cart_step(xyz, dq_try);

            for (std::size_t a = 0; a < natoms; ++a) {
                calc._molecule._standard(a, 0) = xyz_new(a, 0);
                calc._molecule._standard(a, 1) = xyz_new(a, 1);
                calc._molecule._standard(a, 2) = xyz_new(a, 2);
            }
            try {
                g_cart_new = _run_sp_gradient(calc);
                // Zero frozen-atom gradient contributions
                for (std::size_t a = 0; a < natoms; ++a)
                    if (atom_frozen[a]) g_cart_new.segment(static_cast<int>(a) * 3, 3).setZero();
                E_new      = calc._total_energy;
                // Accept if energy decreases OR step is negligibly small
                if (E_new < E || alpha < 1e-6)
                    step_ok = true;
                else
                    alpha *= 0.5;
            } catch (...) {
                alpha *= 0.5;
            }
        }

        if (!step_ok) {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                std::format("Opt Step {} :", iter + 1), "line search failed; restoring geometry");
            for (std::size_t a = 0; a < natoms; ++a) {
                calc._molecule._standard(a, 0) = xyz(a, 0);
                calc._molecule._standard(a, 1) = xyz(a, 1);
                calc._molecule._standard(a, 2) = xyz(a, 2);
            }
            break;
        }

        Eigen::VectorXd g_ic_new = ics.cart_to_ic_grad(xyz_new, g_cart_new);
        // Zero constrained IC gradient components
        for (int c = 0; c < nq; ++c)
            if (ic_frozen[c]) g_ic_new[c] = 0.0;

        // ── BFGS Hessian update in IC space ───────────────────────────────────
        // s_bfgs = actual IC displacement; y_bfgs = gradient change
        Eigen::VectorXd q_old     = ics.values(xyz);
        Eigen::VectorXd q_new_val = ics.values(xyz_new);
        Eigen::VectorXd s_bfgs(nq);
        for (int i = 0; i < nq; ++i) {
            s_bfgs[i] = q_new_val[i] - q_old[i];
            if (ics.coords[i].type == ICType::Torsion) {
                while (s_bfgs[i] >  M_PI) s_bfgs[i] -= 2.0 * M_PI;
                while (s_bfgs[i] < -M_PI) s_bfgs[i] += 2.0 * M_PI;
            }
        }
        Eigen::VectorXd y_bfgs = g_ic_new - g_ic;
        // Zero constrained IC components before BFGS update
        for (int c = 0; c < nq; ++c) {
            if (ic_frozen[c]) { s_bfgs[c] = 0.0; y_bfgs[c] = 0.0; }
        }
        const double sy = s_bfgs.dot(y_bfgs);

        // BFGS update: H_new = H - (H s sᵀ H)/(sᵀ H s) + (y yᵀ)/(sᵀ y)
        // Only update when curvature condition holds (sy > 0)
        if (sy > 1e-10) {
            const Eigen::VectorXd Hs  = H * s_bfgs;
            const double sHs          = s_bfgs.dot(Hs);
            H += (y_bfgs * y_bfgs.transpose()) / sy
               - (Hs * Hs.transpose()) / sHs;
        }

        // ── Accept step ───────────────────────────────────────────────────────
        xyz    = xyz_new;
        g_cart = g_cart_new;
        g_ic   = g_ic_new;
        E      = E_new;
        result.energies.push_back(E);
        const double g_rms_step = std::sqrt(g_cart.squaredNorm() /
                                            static_cast<double>(g_cart.size()));
        const double g_ic_rms_step = std::sqrt(g_ic.squaredNorm() /
                                               static_cast<double>(nq));

        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
            std::format("Opt Step {} :", iter + 1),
            std::format("E = {:.10f} Eh   max|g| = {:.3e}   rms|g| = {:.3e}   rms|g_ic| = {:.3e}",
                        E,
                        g_cart.cwiseAbs().maxCoeff(),
                        g_rms_step,
                        g_ic_rms_step));
        _log_step_geometry(calc);
    }

    // Final convergence check
    const double gmax = g_cart.cwiseAbs().maxCoeff();
    if (gmax < grad_tol) result.converged = true;

    result.energy       = E;
    result.iterations   = static_cast<int>(result.energies.size()) - 1;
    result.grad_max     = gmax;
    result.grad_rms     = std::sqrt(g_cart.squaredNorm() /
                                    static_cast<double>(natoms * 3));
    result.gradient     = calc._gradient;
    result.final_coords = calc._molecule._standard;

    return result;
}
