// Validates HartreeFock::SCF::rebuild_basis_dependent_state — the helper that
// will be wired into geomopt and freq in Phase 3 Step 2. Currently no live
// caller; this test is the contract.
//
// What we check:
//   1. Cartesian water/STO-3G: _overlap diagonal is 1.0 (basis normalization),
//      _overlap is symmetric, _hcore is symmetric, shape is nbasis × nbasis.
//   2. Spherical water/6-31g*: same checks at nbasis_sph × nbasis_sph. The
//      key property here is diag(_overlap) = 1 — the *whole point* of the
//      cart_to_sph row-normalization step inside the helper. If the helper
//      forgets the normalization, this assertion fails.
//   3. Idempotency in spherical mode: calling the helper twice in a row
//      yields the same _cart_to_sph (the second call sees the already-
//      normalized C and should leave it alone, since norm² = 1 already).
//      This is the property the geomopt inner loop relies on when the basis
//      object happens to be reused unchanged across a step.

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "scf/working_state.h"
#include "symmetry/symmetry.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &msg)
    {
        std::cerr << "[FAIL] " << msg << '\n';
        g_ok = false;
    }

    std::expected<HartreeFock::Calculator, std::string> make_water_calculator(
        const std::string &basis_name,
        HartreeFock::BasisType basis_type)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;

        // Water geometry matching the spherical gradient regression fixture.
        const std::vector<int> Z = {8, 1, 1};
        const std::vector<std::array<double, 3>> xyz = {
            {0.000000, 0.000000, 0.117176},
            {0.000000, 0.757200, -0.468704},
            {0.000000, -0.757200, -0.468704},
        };

        const std::size_t n = Z.size();
        mol.natoms = n;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(n);
        mol.atomic_masses.resize(n);
        mol.coordinates.resize(n, 3);
        for (std::size_t i = 0; i < n; ++i)
        {
            mol.atomic_numbers[static_cast<Eigen::Index>(i)] = Z[i];
            mol.atomic_masses[static_cast<Eigen::Index>(i)] = 1.0;
            for (int q = 0; q < 3; ++q)
                mol.coordinates(static_cast<Eigen::Index>(i), q) = xyz[i][q];
        }

        calc._basis._basis = basis_type;
        calc.prepare_coordinates();

        if (auto r = HartreeFock::Symmetry::detectSymmetry(mol, HartreeFock::Units::Angstrom); !r)
            return std::unexpected("detectSymmetry: " + r.error());

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis: " + basis_res.error());
        calc._shells = std::move(*basis_res);

        if (auto r = calc.initialize(); !r)
            return std::unexpected("initialize: " + r.error());

        return calc;
    }

    bool symmetric(const Eigen::MatrixXd &M, double tol)
    {
        if (M.rows() != M.cols())
            return false;
        return (M - M.transpose()).cwiseAbs().maxCoeff() < tol;
    }

    void check_cartesian()
    {
        auto calc_res = make_water_calculator("sto-3g", HartreeFock::BasisType::Cartesian);
        if (!calc_res)
        {
            fail("cartesian: setup failed: " + calc_res.error());
            return;
        }
        HartreeFock::Calculator calc = std::move(*calc_res);

        auto sp_res = HartreeFock::SCF::rebuild_basis_dependent_state(calc);
        if (!sp_res)
        {
            fail("cartesian: helper returned error: " + sp_res.error());
            return;
        }

        const auto nb = static_cast<Eigen::Index>(calc._shells.nbasis());
        if (calc._overlap.rows() != nb || calc._overlap.cols() != nb)
            fail("cartesian: _overlap shape is "
                 + std::to_string(calc._overlap.rows()) + "×"
                 + std::to_string(calc._overlap.cols()) + ", expected "
                 + std::to_string(nb) + "²");

        const double diag_err = (calc._overlap.diagonal() -
                                 Eigen::VectorXd::Ones(calc._overlap.rows())).cwiseAbs().maxCoeff();
        if (diag_err > 1e-12)
            fail("cartesian: diag(_overlap) deviates from 1 by " + std::to_string(diag_err));

        if (!symmetric(calc._overlap, 1e-12))
            fail("cartesian: _overlap is not symmetric");
        if (!symmetric(calc._hcore, 1e-12))
            fail("cartesian: _hcore is not symmetric");
        if (calc._hcore.rows() != nb || calc._hcore.cols() != nb)
            fail("cartesian: _hcore shape mismatch");
    }

    void check_spherical()
    {
        auto calc_res = make_water_calculator("6-31g*", HartreeFock::BasisType::Spherical);
        if (!calc_res)
        {
            fail("spherical: setup failed: " + calc_res.error());
            return;
        }
        HartreeFock::Calculator calc = std::move(*calc_res);

        if (!calc._shells._spherical)
        {
            fail("spherical: _shells._spherical not set after read_gbs_basis");
            return;
        }

        auto sp_res = HartreeFock::SCF::rebuild_basis_dependent_state(calc);
        if (!sp_res)
        {
            fail("spherical: helper returned error: " + sp_res.error());
            return;
        }

        const auto nb_sph = static_cast<Eigen::Index>(calc._shells.nbasis_sph());
        if (calc._overlap.rows() != nb_sph || calc._overlap.cols() != nb_sph)
            fail("spherical: _overlap shape is "
                 + std::to_string(calc._overlap.rows()) + "×"
                 + std::to_string(calc._overlap.cols()) + ", expected "
                 + std::to_string(nb_sph) + "² (nbasis_sph)");

        // The critical contract: diag(_overlap) = 1 after the helper runs.
        // This is what the row-normalization step inside the helper buys us.
        // Without normalization, d-shell entries drift to ~3/5 or similar.
        const double diag_err = (calc._overlap.diagonal() -
                                 Eigen::VectorXd::Ones(calc._overlap.rows())).cwiseAbs().maxCoeff();
        if (diag_err > 1e-12)
            fail("spherical: diag(_overlap) deviates from 1 by " + std::to_string(diag_err)
                 + " — the cart_to_sph row-normalization step may be missing");

        if (!symmetric(calc._overlap, 1e-12))
            fail("spherical: _overlap is not symmetric");
        if (!symmetric(calc._hcore, 1e-12))
            fail("spherical: _hcore is not symmetric");

        // _cart_to_sph still has the right outer shape [nbasis_sph × nbasis_cart].
        if (calc._shells._cart_to_sph.rows() != nb_sph ||
            calc._shells._cart_to_sph.cols() != static_cast<Eigen::Index>(calc._shells.nbasis()))
            fail("spherical: _cart_to_sph shape changed unexpectedly");

        // Idempotency: second call leaves _cart_to_sph and _overlap unchanged
        // (within floating-point reach), since the transform is already
        // normalized against the same S_cart.
        const Eigen::MatrixXd C_first = calc._shells._cart_to_sph;
        const Eigen::MatrixXd overlap_first = calc._overlap;
        auto sp2 = HartreeFock::SCF::rebuild_basis_dependent_state(calc);
        if (!sp2)
        {
            fail("spherical: second helper call failed: " + sp2.error());
            return;
        }
        const double dC = (calc._shells._cart_to_sph - C_first).cwiseAbs().maxCoeff();
        const double dS = (calc._overlap - overlap_first).cwiseAbs().maxCoeff();
        if (dC > 1e-14)
            fail("spherical: second call moved _cart_to_sph by " + std::to_string(dC)
                 + " — normalization is not idempotent");
        if (dS > 1e-14)
            fail("spherical: second call moved _overlap by " + std::to_string(dS));
    }
} // namespace

int main()
{
    check_cartesian();
    check_spherical();

    if (g_ok)
    {
        std::cout << "working_state_rebuild: OK\n";
        return 0;
    }
    std::cout << "working_state_rebuild: FAIL\n";
    return 1;
}
