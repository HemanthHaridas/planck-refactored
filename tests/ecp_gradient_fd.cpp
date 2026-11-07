// ECP nuclear-derivative gate (docs/DEF2_ECP.md step 5a).
//
// For a fixed random symmetric density D, E(R) = tr(D U(R)) with U the ECP
// matrix from libecpint. The analytic gradient (_compute_ecp_gradient, from
// libecpint's shell-pair derivatives) must match a five-point finite difference
// of E, where every displaced geometry re-reads the basis so the shells and the
// ECP centre move with their atom. The FD shares no derivative code with
// libecpint. Also checks translational invariance (sum over atoms = 0), which
// is near-vacuous: libecpint builds the C derivative so the sum comes out ~0.
//
// ECP_FD_H overrides the step (default 1e-3 Bohr). Scan it when a case fails:
// a miss independent of h is a wrong derivative, one growing as h shrinks is
// non-smooth U(R). Stock libecpint at 95203c68 fails I2/def2-TZVP and
// HI/def2-QZVP here (rel 2.6e-7, 4.6e-5); upstream PR #81 passes all five.

#include <cmath>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/basis.h"
#include "base/tables.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/ecp.h"

namespace
{
    bool g_ok = true;

    struct Atom
    {
        int z;
        double x, y, z_coord; // Angstrom
    };

    struct Fixture
    {
        HartreeFock::Molecule mol;
        HartreeFock::Basis basis;
    };

    std::expected<Fixture, std::string> build(const std::vector<Atom> &atoms, const std::string &basis_name,
                                              const Eigen::MatrixXd &bohr)
    {
        Fixture f;
        HartreeFock::Molecule &mol = f.mol;
        mol.natoms = atoms.size();
        mol.atomic_numbers.resize(static_cast<Eigen::Index>(atoms.size()));
        for (std::size_t a = 0; a < atoms.size(); ++a)
            mol.atomic_numbers[static_cast<Eigen::Index>(a)] = atoms[a].z;
        mol.set_standard_from_bohr(bohr);
        const auto gbs = std::filesystem::path(get_basis_path()) / basis_name;
        auto b = HartreeFock::BasisFunctions::read_gbs_basis(gbs.string(), mol,
                                                             HartreeFock::BasisType::Cartesian);
        if (!b)
            return std::unexpected(b.error());
        f.basis = std::move(*b);
        return f;
    }

    std::vector<int> shell_atoms(const Fixture &f)
    {
        std::vector<int> map;
        for (const auto &sh : f.basis._shells)
        {
            int best = 0;
            for (std::size_t a = 1; a < f.mol.natoms; ++a)
                if ((f.mol._standard.row(static_cast<Eigen::Index>(a)).transpose() - sh._center).norm() <
                    (f.mol._standard.row(best).transpose() - sh._center).norm())
                    best = static_cast<int>(a);
            map.push_back(best);
        }
        return map;
    }

    void run_case(const std::string &name, const std::vector<Atom> &atoms, const std::string &basis_name)
    {
        Eigen::MatrixXd R(static_cast<Eigen::Index>(atoms.size()), 3);
        for (std::size_t a = 0; a < atoms.size(); ++a)
            R.row(static_cast<Eigen::Index>(a)) << atoms[a].x, atoms[a].y, atoms[a].z_coord;
        R *= ANGSTROM_TO_BOHR;

        auto f0 = build(atoms, basis_name, R);
        if (!f0)
        {
            std::cerr << "[FAIL] " << name << ": " << f0.error() << '\n';
            g_ok = false;
            return;
        }
        const std::size_t nb = f0->basis.nbasis();
        std::mt19937 rng(17);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd D(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                D(i, j) = dist(rng);
        D = 0.5 * (D + D.transpose()).eval();

        const Eigen::MatrixXd g =
            HartreeFock::ECP::_compute_ecp_gradient(f0->basis, shell_atoms(*f0), atoms.size(), D);

        const double h = std::getenv("ECP_FD_H") ? std::atof(std::getenv("ECP_FD_H")) : 1e-3;
        const auto energy = [&](const Eigen::MatrixXd &Rd)
        {
            auto f = build(atoms, basis_name, Rd);
            return (HartreeFock::ECP::_compute_ecp(f->basis).cwiseProduct(D)).sum();
        };
        double worst = 0.0, scale = 0.0;
        for (Eigen::Index a = 0; a < R.rows(); ++a)
            for (int q = 0; q < 3; ++q)
            {
                double e[4];
                const double steps[4] = {-2 * h, -h, h, 2 * h};
                for (int k = 0; k < 4; ++k)
                {
                    Eigen::MatrixXd Rd = R;
                    Rd(a, q) += steps[k];
                    e[k] = energy(Rd);
                }
                const double fd = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * h);
                worst = std::max(worst, std::abs(fd - g(a, q)));
                scale = std::max(scale, std::abs(fd));
            }
        const double trans = g.colwise().sum().cwiseAbs().maxCoeff();
        const double rel = worst / scale;
        const bool ok = rel < 1e-8 && trans < 1e-9 * scale;
        std::cout << std::format("[{}] {:<24} max|g| {:.3e}  max|fd-analytic| {:.3e} (rel {:.2e})  "
                                 "|sum_atoms g| {:.2e}\n",
                                 ok ? "PASS" : "FAIL", name, scale, worst, rel, trans);
        g_ok = g_ok && ok;
    }
} // namespace

int main()
{
    run_case("HI/def2-SVP", {{1, 0.0, 0.0, 0.0}, {53, 0.1, -0.05, 1.609}}, "def2-SVP");
    run_case("HI/def2-TZVP", {{1, 0.0, 0.0, 0.0}, {53, 0.1, -0.05, 1.609}}, "def2-TZVP");
    run_case("I2/def2-TZVP", {{53, 0.0, 0.0, 0.0}, {53, 0.2, 0.1, 2.666}}, "def2-TZVP");
    run_case("AuH.H2O/def2-SVP",
             {{79, 0.0, 0.0, 0.0},
              {1, 0.0, 0.0, 1.524},
              {8, 2.3, 0.4, -0.9},
              {1, 2.9, 1.1, -0.6},
              {1, 2.7, -0.4, -0.5}},
             "def2-SVP");
    run_case("HI/def2-QZVP", {{1, 0.0, 0.0, 0.0}, {53, 0.1, -0.05, 1.609}}, "def2-QZVP");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
