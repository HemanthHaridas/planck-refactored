// PCM cavity-gradient gate (docs/PCM_GEOMETRY_SCOPE.md step 4b).
//
// For fixed random apparent charges q, the two geometry-only terms of the
// C-PCM gradient must match a five-point finite difference of
//   E_nuc(R)    = q . phi_nuc(R)            (pcm_grad_nuc)
//   E_solver(R) = q^T S(R) q / (2 f)        (pcm_grad_solver)
// where every displaced geometry rebuilds the cavity with build_pcm_state, so
// the tesserae, F_i and phi_nuc are recomputed from scratch. The FD shares no
// derivative code with the analytic terms. A displacement must keep the pruned
// tessera set; the test fails loudly if it does not.
//
// C1 H2O2, 302 points: low symmetry, so every gradient component is
// independent, and the O-O / O-H sphere overlaps put many tesserae in the
// switching region, which is where the F_i (diagonal) term lives.
// Also checks translational invariance (each term sums to zero over atoms).

#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "base/tables.h"
#include "base/types.h"
#include "io/logging.h"
#include "solvation/pcm.h"

namespace
{
    namespace S = HartreeFock::Solvation;

    HartreeFock::Calculator make_calc(const Eigen::MatrixXd &bohr)
    {
        HartreeFock::Calculator calc;
        auto &mol = calc._molecule;
        mol.natoms = 4;
        mol.atomic_numbers.resize(4);
        mol.atomic_numbers << 8, 8, 1, 1;
        mol.set_standard_from_bohr(bohr);
        calc._solvation._model = HartreeFock::SolvationModel::PCM;
        calc._solvation._dielectric = 78.3553;
        calc._solvation._cavity_scale = 1.2;
        calc._solvation._surface_points_per_atom = 302;
        return calc;
    }

    struct Energies
    {
        double nuc, solver;
        std::size_t npoints;
        std::vector<std::size_t> owners;
    };

    Energies energies(const Eigen::MatrixXd &bohr, const Eigen::VectorXd &q)
    {
        const HartreeFock::Logger::ScopedSilence quiet;
        const auto calc = make_calc(bohr);
        auto state = S::build_pcm_state(calc);
        if (!state)
        {
            std::cerr << "build_pcm_state failed: " << state.error() << "\n";
            std::exit(1);
        }
        Energies e{.nuc = 0.0, .solver = 0.0, .npoints = state->surface_points.size(), .owners = {}};
        for (const auto &p : state->surface_points)
            e.owners.push_back(p.atom_index);
        if (e.npoints != static_cast<std::size_t>(q.size()))
            return e;
        e.nuc = q.dot(state->nuclear_potential);
        e.solver = q.dot(S::pcm_influence_matrix(state->surface_points) * q) / (2.0 * state->dielectric_factor);
        return e;
    }
} // namespace

int main()
{
    Eigen::MatrixXd bohr(4, 3);
    bohr << 0.000000, 0.700000, -0.050000,
            0.020000, -0.710000, 0.040000,
            0.880000, 0.900000, 0.420000,
           -0.650000, -0.950000, 0.780000;
    bohr *= ANGSTROM_TO_BOHR;

    const auto calc = make_calc(bohr);
    const auto state = [&]
    {
        const HartreeFock::Logger::ScopedSilence quiet;
        return S::build_pcm_state(calc);
    }();
    if (!state)
    {
        std::cerr << "build_pcm_state failed: " << state.error() << "\n";
        return 1;
    }
    const auto n = static_cast<Eigen::Index>(state->surface_points.size());

    // Physical charges: the C-PCM response to the nuclear potential,
    // q = -f S^-1 phi_nuc. Random charges are the wrong fixture: on a tessera
    // that has nearly faded out (F_i -> 1e-16, S_ii -> 1e16), a real solve
    // gives q_i ~ F_i, but an O(1e-2) random q_i makes q_i^2 S_ii dominate
    // the energy and the finite difference.
    const Eigen::VectorXd q =
        state->influence_factor.solve(-state->dielectric_factor * state->nuclear_potential);

    const Eigen::MatrixXd g_nuc = S::pcm_grad_nuc(calc._molecule, *state, q);
    const Eigen::MatrixXd g_solver = S::pcm_grad_solver(calc._molecule, *state, q);
    const auto base = energies(bohr, q);

    // PCM_FD_H overrides the step (Bohr). A miss independent of h is a wrong
    // derivative; one that shrinks with h is FD error.
    const double h = std::getenv("PCM_FD_H") ? std::atof(std::getenv("PCM_FD_H")) : 1e-4;
    double worst_nuc = 0.0, worst_solver = 0.0;
    bool ok = true;
    for (Eigen::Index a = 0; a < 4; ++a)
        for (Eigen::Index k = 0; k < 3; ++k)
        {
            Energies e[4];
            const double steps[4] = {-2.0, -1.0, 1.0, 2.0};
            for (int s = 0; s < 4; ++s)
            {
                Eigen::MatrixXd displaced = bohr;
                displaced(a, k) += steps[s] * h;
                e[s] = energies(displaced, q);
                if (e[s].npoints != base.npoints || e[s].owners != base.owners)
                {
                    std::cerr << std::format("atom {} dir {} step {}: tessera set changed ({} -> {})\n",
                                             a, k, steps[s], base.npoints, e[s].npoints);
                    return 1;
                }
            }
            const auto fd = [&](double Energies::*m)
            { return (e[0].*m - 8.0 * (e[1].*m) + 8.0 * (e[2].*m) - e[3].*m) / (12.0 * h); };
            const double dn = std::abs(fd(&Energies::nuc) - g_nuc(a, k));
            const double ds = std::abs(fd(&Energies::solver) - g_solver(a, k));
            worst_nuc = std::max(worst_nuc, dn);
            worst_solver = std::max(worst_solver, ds);
            std::cout << std::format("atom {} dir {}  nuc {: .10e} (fd err {:.2e})  solver {: .10e} (fd err {:.2e})\n",
                                     a, k, g_nuc(a, k), dn, g_solver(a, k), ds);
        }

    const double tol = 1e-8;
    const double trans_nuc = g_nuc.colwise().sum().cwiseAbs().maxCoeff();
    const double trans_solver = g_solver.colwise().sum().cwiseAbs().maxCoeff();
    std::cout << std::format("tesserae {}, worst FD error: nuc {:.2e}, solver {:.2e} (tol {:.0e})\n",
                             n, worst_nuc, worst_solver, tol);
    std::cout << std::format("translation: nuc {:.2e}, solver {:.2e}\n", trans_nuc, trans_solver);
    ok = worst_nuc < tol && worst_solver < tol && trans_nuc < 1e-12 && trans_solver < 1e-12;
    // Non-vacuity: both terms must be far above the tolerance.
    ok = ok && g_nuc.cwiseAbs().maxCoeff() > 1e3 * tol && g_solver.cwiseAbs().maxCoeff() > 1e3 * tol;
    std::cout << (ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
