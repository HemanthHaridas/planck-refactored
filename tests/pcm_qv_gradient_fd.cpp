// PCM integral-gradient gate (docs/PCM_GEOMETRY.md step 4c).
//
// For a fixed random symmetric AO density P and fixed charges q on a subset of
// the real SWIG cavity's tesserae, pcm_grad_qv must match a five-point finite
// difference of E(R) = q . phi_el(R) = -sum_k q_k (P : V^k(R)), where each
// displaced geometry re-reads the basis (the AO centres move with their atom)
// and moves every tessera rigidly with its owning atom. The FD only calls the
// energy sweep _compute_electronic_potential, which shares the VRR with the
// gradient kernel but neither the angular-momentum shift nor the
// charge-centre (translation) term. Also checks that the result sums to zero
// over atoms.
//
// C1 H2O2 in 6-31G* (d shells) and cc-pVTZ (f shells), Cartesian.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include "base/basis.h"
#include "base/tables.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"
#include "io/logging.h"
#include "solvation/pcm.h"
#include "require_boys_library.h"

namespace
{
    namespace S = HartreeFock::Solvation;

    void set_geometry(HartreeFock::Calculator &calc, const Eigen::MatrixXd &bohr, const std::string &basis)
    {
        auto &mol = calc._molecule;
        mol.natoms = 4;
        mol.atomic_numbers.resize(4);
        mol.atomic_numbers << 8, 8, 1, 1;
        mol.set_standard_from_bohr(bohr);
        const auto gbs = std::filesystem::path(get_basis_path()) / basis;
        auto b = HartreeFock::BasisFunctions::read_gbs_basis(gbs.string(), mol, HartreeFock::BasisType::Cartesian);
        if (!b)
        {
            std::cerr << "basis read failed: " << b.error() << "\n";
            std::exit(1);
        }
        calc._shells = std::move(*b);
    }

    bool run(const std::string &basis)
    {
        Eigen::MatrixXd bohr(4, 3);
        bohr << 0.000000, 0.700000, -0.050000,
                0.020000, -0.710000, 0.040000,
                0.880000, 0.900000, 0.420000,
               -0.650000, -0.950000, 0.780000;
        bohr *= ANGSTROM_TO_BOHR;

        HartreeFock::Calculator calc;
        calc._solvation._model = HartreeFock::SolvationModel::PCM;
        calc._solvation._dielectric = 78.3553;
        calc._solvation._surface_points_per_atom = 302;
        set_geometry(calc, bohr, basis);

        auto state = [&]
        {
            const HartreeFock::Logger::ScopedSilence quiet;
            return S::build_pcm_state(calc);
        }();
        if (!state)
        {
            std::cerr << "build_pcm_state failed: " << state.error() << "\n";
            return false;
        }
        // Every 7th tessera keeps the FD cheap and still covers all four spheres.
        std::vector<S::PCMSurfacePoint> kept;
        for (std::size_t i = 0; i < state->surface_points.size(); i += 7)
            kept.push_back(state->surface_points[i]);
        state->surface_points = kept;
        const auto npts = static_cast<Eigen::Index>(kept.size());

        std::mt19937 rng(23);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        Eigen::VectorXd q(npts);
        for (Eigen::Index k = 0; k < npts; ++k)
            q(k) = 0.01 * u(rng);
        const auto nb = static_cast<Eigen::Index>(calc._shells.nbasis());
        Eigen::MatrixXd P(nb, nb);
        for (Eigen::Index i = 0; i < nb; ++i)
            for (Eigen::Index j = 0; j <= i; ++j)
                P(i, j) = P(j, i) = 0.2 * u(rng);

        std::vector<int> ao_atom;
        for (const auto &bf : calc._shells._basis_functions)
        {
            int best = 0;
            for (int a = 1; a < 4; ++a)
                if ((bohr.row(a).transpose() - bf._shell->_center).norm() <
                    (bohr.row(best).transpose() - bf._shell->_center).norm())
                    best = a;
            ao_atom.push_back(best);
        }

        const auto pairs = build_shellpairs(calc._shells);
        const auto g = S::pcm_grad_qv(calc, pairs, *state, P, q, ao_atom);
        if (!g)
        {
            std::cerr << "pcm_grad_qv failed: " << g.error() << "\n";
            return false;
        }

        std::vector<double> eta;
        for (const auto &p : kept)
            eta.push_back(p.exponent * p.exponent);
        const auto energy = [&](const Eigen::MatrixXd &geom)
        {
            HartreeFock::Calculator c;
            set_geometry(c, geom, basis);
            const auto sp = build_shellpairs(c._shells);
            std::vector<Eigen::Vector3d> pts;
            for (const auto &p : kept)
                pts.push_back(p.position + (geom.row(static_cast<Eigen::Index>(p.atom_index)) -
                                            bohr.row(static_cast<Eigen::Index>(p.atom_index))).transpose());
            return -q.dot(HartreeFock::ObaraSaika::_compute_electronic_potential(sp, P, pts, &eta));
        };

        const double h = std::getenv("PCM_FD_H") ? std::atof(std::getenv("PCM_FD_H")) : 1e-3;
        double worst = 0.0;
        for (Eigen::Index a = 0; a < 4; ++a)
            for (Eigen::Index k = 0; k < 3; ++k)
            {
                double e[4];
                const double steps[4] = {-2.0, -1.0, 1.0, 2.0};
                for (int s = 0; s < 4; ++s)
                {
                    Eigen::MatrixXd d = bohr;
                    d(a, k) += steps[s] * h;
                    e[s] = energy(d);
                }
                const double fd = (e[0] - 8.0 * e[1] + 8.0 * e[2] - e[3]) / (12.0 * h);
                worst = std::max(worst, std::abs(fd - (*g)(a, k)));
                std::cout << std::format("  {} atom {} dir {}  {: .10e}  fd err {:.2e}\n",
                                         basis, a, k, (*g)(a, k), std::abs(fd - (*g)(a, k)));
            }
        const double trans = g->colwise().sum().cwiseAbs().maxCoeff();
        const bool ok = worst < 1e-8 && trans < 1e-10 && g->cwiseAbs().maxCoeff() > 1e-4;
        std::cout << std::format("{}: {} AOs, {} tesserae, worst FD error {:.2e}, translation {:.2e} -> {}\n",
                                 basis, nb, npts, worst, trans, ok ? "ok" : "FAIL");
        return ok;
    }
} // namespace

int main()
{
    if (!require_boys_library())
        return 1;
    const bool ok = run("6-31g*") & run("cc-pVTZ");
    std::cout << (ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
