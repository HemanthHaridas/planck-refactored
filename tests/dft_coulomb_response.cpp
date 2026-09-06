// D2.2.1 (docs/SOSCF_UHF_DFT_SCOPE.md): the Coulomb-response piece of the
// RKS SOSCF orbital-Hessian-vector product h_op.
//
// Unlike D2.0's XC-kernel piece, no new production function is written
// here: J is LINEAR in the density by construction
// (J[P](mu,nu) = sum (mu nu|lam sig) P(lam,sig)), so the induced Coulomb
// potential for a trial density dP, delta_J(dP), is exactly
// _compute_2e_j_direct(shell_pairs, dP, ...) -- the identical
// already-tested memory-direct builder the RKS loop's own per-iteration
// Coulomb term uses (src/dft/driver.cpp, assemble_current_ks_potential),
// just called on dP instead of the SCF's own converged density. There is
// nothing DFT-specific or SOSCF-specific to write: this file exists only
// to VERIFY that reuse is valid for an arbitrary (non-SCF) trial density,
// per D2.2.1's own scope note.
//
// Verification is a finite-difference check independent of D2.0/D2.1's
// machinery entirely, and independent of any orbital rotation: for
// P(kappa) = P0 + kappa*dP (a plain LINEAR density perturbation, not a
// Cayley-rotated one -- this piece has no orbital-rotation dependence at
// all, so testing it against a rotation would import machinery it does not
// need), E_Coulomb(kappa) = 0.5*Tr(P(kappa)*J(P(kappa))) is an exact
// quadratic in kappa (J is linear), so
//   d^2 E_Coulomb/d kappa^2 = Tr(dP . J(dP)) = Tr(dP . delta_J(dP))
// holds EXACTLY (no O(kappa) truncation, unlike D2.0's XC piece), which
// means a finite difference at any reasonably small h should already agree
// to near machine precision -- a much tighter bar than D2.0/D2.1's own
// checks, which is appropriate since this piece is pure linear algebra.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    // Same water/STO-3G-style fixture construction as
    // tests/fock_accumulate.cpp's make_water -- deliberately reproduced
    // rather than shared, since these test binaries are independent CMake
    // targets and this file has no other dependency on that one.
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

    Eigen::MatrixXd random_symmetric(std::size_t nb, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd M(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j <= i; ++j)
            {
                const double v = dist(rng);
                M(i, j) = v;
                M(j, i) = v;
            }
        return M;
    }

    void check_coulomb_response(unsigned seed)
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();

        std::mt19937 rng(seed);
        // P0: a plausible symmetric AO density (not converged -- the
        // identity under test holds for any P0/dP, exactly D2.2.1's own
        // "NOT necessarily an occ-virt rotation-generated one" note).
        const Eigen::MatrixXd P0 = random_symmetric(nb, rng);
        const Eigen::MatrixXd dP = random_symmetric(nb, rng);

        auto J = [&](const Eigen::MatrixXd &P) -> Eigen::MatrixXd
        {
            return _compute_2e_j_direct(
                shell_pairs, P, nb, HartreeFock::IntegralMethod::ObaraSaika,
                HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        };

        auto e_coulomb_at = [&](double h) -> double
        {
            const Eigen::MatrixXd P = P0 + h * dP;
            return 0.5 * (P.array() * J(P).array()).sum();
        };

        const double delta_J_dP_norm = J(dP).norm();
        if (delta_J_dP_norm == 0.0)
        {
            fail("delta_J(dP) is exactly zero -- fixture is degenerate, seed=" + std::to_string(seed));
            return;
        }

        const double tr_full = (dP.array() * J(dP).array()).sum();

        // E_Coulomb(kappa) is an EXACT quadratic in kappa (J is linear), so
        // the central second-difference recovers d^2E/dk^2 up to O(h^2)
        // truncation only -- tight tolerances are appropriate here, unlike
        // D2.0/D2.1's XC-kernel checks where P(kappa) itself is nonlinear.
        const double e0 = e_coulomb_at(0.0);
        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double ep = e_coulomb_at(h);
            const double em = e_coulomb_at(-h);
            const double h_fd = (ep + em - 2.0 * e0) / (h * h);
            const double diff = std::abs(h_fd - tr_full);
            // O(h^2) truncation on a quantity of this scale.
            const double tol = 1e-6 + 1.0 * h * h * std::abs(tr_full);
            if (diff > tol)
            {
                fail("seed=" + std::to_string(seed) + " h=" + std::to_string(h) +
                     ": h_fd_coulomb=" + std::to_string(h_fd) +
                     " Tr(dP.deltaJ(dP))=" + std::to_string(tr_full) +
                     " diff=" + std::to_string(diff) + " tol=" + std::to_string(tol));
            }
        }
    }
} // namespace

int main()
{
    check_coulomb_response(1);
    check_coulomb_response(2);
    check_coulomb_response(3);

    if (g_ok)
        std::cout << "OK  D2.2.1: Tr(dP . delta_J(dP)) matches FD of the Coulomb energy\n";

    return g_ok ? 0 : 1;
}
