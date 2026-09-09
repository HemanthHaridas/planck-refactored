// S2 (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md, N3.5.7.7): FD
// verification of DFT::Gradient::compute_dh_xc_pt2_gradient's LDA branch --
// Eq. 33 Term 1 of the double-hybrid PT2 gradient,
//
//   grad(A,q) = integral rho_P^(x)_{A,q}(r) * [ v3rho3(rho_P;r) * rho_D(r) ] dr
//
// where rho_P^(x) (Eq. 15) is the BASIS-function derivative of the density
// at a FIXED spatial point. Term 1 is a plain grid integral of that
// integrand -- NOT d/dR of an integral -- so there is no moving-grid
// correction, and the FD reference differentiates only the AO centers
// (grid, weights, and the frozen v3rho3*rho_D coefficient all held fixed).
//
// Reference: rebuild the basis with atom A's shell centers shifted by
// +-h along q, re-evaluate rho_P at the ORIGINAL grid points, central
// difference -> [d rho_P / d R_{A,q}]_basis at each point; contract with
// w * (v3rho3_x + v3rho3_c)(ground rho) * rho_D. This is an independent
// implementation of Term 1, not a call into the routine.
//
// He2/STO-3G: two atoms (so per-atom components are non-trivial), closed
// shell, small enough that a Normal grid resolves the FD. A strongly
// diagonally-dominant random P/D keeps rho_P well above the v3rho3 ~
// rho^(-5/3) singular regime on the whole quadrature support -- this is a
// math-correctness gate, not a physical-density one.
#include <array>
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
#include "dft/ao_grid.h"
#include "dft/base/grid.h"
#include "dft/base/wrapper.h"
#include "dft/dft_kernel_gradient.h"
#include "dft/xc_grid.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    HartreeFock::Calculator make_he2(double bond_angstrom)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        mol.natoms = 2;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(2);
        mol.atomic_numbers << 2, 2;
        mol.atomic_masses.resize(2);
        mol.atomic_masses << 4.0, 4.0;
        mol.coordinates.resize(2, 3);
        mol.coordinates << 0.0, 0.0, -0.5 * bond_angstrom,
            0.0, 0.0, 0.5 * bond_angstrom;

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / "sto-3g";
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
        {
            fail(std::string("read_gbs_basis failed: ") + basis_res.error());
            return calc;
        }
        calc._shells = std::move(*basis_res);
        return calc;
    }

    // Strongly diagonally-dominant symmetric matrix: the +4 diagonal keeps
    // rho_P well away from zero (v3rho3 ~ rho^(-5/3) blows up otherwise).
    Eigen::MatrixXd random_symmetric(Eigen::Index nb, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-0.2, 0.2);
        Eigen::MatrixXd M(nb, nb);
        for (Eigen::Index i = 0; i < nb; ++i)
            for (Eigen::Index j = 0; j <= i; ++j)
            {
                const double v = dist(rng);
                M(i, j) = v;
                M(j, i) = v;
            }
        for (Eigen::Index i = 0; i < nb; ++i)
            M(i, i) += 4.0;
        return M;
    }

    DFT::XC::Functional require_functional(const std::string &name)
    {
        auto id = DFT::XC::functional_id(name);
        if (!id)
        {
            fail("functional_id(" + name + "): " + id.error());
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Unpolarized).value();
        }
        auto f = DFT::XC::Functional::create(*id, DFT::XC::Spin::Unpolarized);
        if (!f)
        {
            fail("Functional::create(" + name + "): " + f.error());
            return DFT::XC::Functional::create(1, DFT::XC::Spin::Unpolarized).value();
        }
        return std::move(*f);
    }

    // rho(r) = phi^T P phi at every grid point.
    Eigen::VectorXd grid_rho(const DFT::AOGridEvaluation &ao, const Eigen::MatrixXd &P_sym)
    {
        const Eigen::Index npts = ao.npoints();
        Eigen::VectorXd rho(npts);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const Eigen::VectorXd phi = ao.values.row(p).transpose();
            rho(p) = phi.dot(P_sym * phi);
        }
        return rho;
    }

    // rho_P evaluated on `fixed_grid`'s points using a basis rebuilt with
    // atom `atom`'s shell centers shifted by `delta` along `q`. The grid
    // (points + weights) does NOT move -- only the AOs.
    Eigen::VectorXd rho_with_shifted_atom(
        const HartreeFock::Calculator &base,
        const DFT::MolecularGrid &fixed_grid,
        const Eigen::MatrixXd &P_sym,
        int atom, int q, double delta)
    {
        HartreeFock::Calculator c = base;
        c._molecule._coordinates(atom, q) += delta;
        c._molecule.set_standard_from_bohr(c._molecule._coordinates);
        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / "sto-3g";
        auto b = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), c._molecule, c._basis._basis);
        if (!b)
        {
            fail(std::string("read_gbs_basis (shifted): ") + b.error());
            return Eigen::VectorXd::Zero(fixed_grid.points.rows());
        }
        auto ao = DFT::evaluate_ao_basis_on_grid(*b, fixed_grid);
        if (!ao)
        {
            fail(std::string("evaluate_ao_basis_on_grid (shifted): ") + ao.error());
            return Eigen::VectorXd::Zero(fixed_grid.points.rows());
        }
        return grid_rho(*ao, P_sym);
    }

    void check_lda_term1(const std::string &x_name, const std::string &c_name,
                         unsigned seed, double h_ang)
    {
        HartreeFock::Calculator calc = make_he2(2.0);
        if (!g_ok)
            return;

        const Eigen::Index nb = calc._shells.nbasis();
        std::mt19937 rng(seed);
        const Eigen::MatrixXd P_sym = random_symmetric(nb, rng);
        const Eigen::MatrixXd D_sym = random_symmetric(nb, rng);

        auto x_func = require_functional(x_name);
        auto c_func = require_functional(c_name);
        if (!g_ok)
            return;

        auto grid = DFT::MakeMolecularGrid(calc._molecule, DFT::GridLevel::Normal);
        if (!grid) { fail("MakeMolecularGrid: " + grid.error()); return; }
        auto ao = DFT::evaluate_ao_basis_on_grid(calc._shells, *grid);
        if (!ao) { fail("evaluate_ao_basis_on_grid: " + ao.error()); return; }
        auto hess = DFT::evaluate_ao_hessian_on_grid(calc._shells, *grid);
        if (!hess) { fail("evaluate_ao_hessian_on_grid: " + hess.error()); return; }

        const Eigen::Index npts = grid->points.rows();
        const Eigen::VectorXd rho_ground = grid_rho(*ao, P_sym);
        const Eigen::VectorXd rho_relaxed = grid_rho(*ao, D_sym);

        std::vector<double> rho_vec(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
            rho_vec[static_cast<std::size_t>(p)] = rho_ground(p);

        std::vector<double> v3x, v3c;
        auto kx = x_func.evaluate_lda_kxc(rho_vec, static_cast<int>(npts), v3x);
        auto kc = c_func.evaluate_lda_kxc(rho_vec, static_cast<int>(npts), v3c);
        if (!kx || !kc) { fail("evaluate_lda_kxc"); return; }

        // Frozen per-point coefficient  w * (v3rho3_x + v3rho3_c) * rho_D,
        // with the same rho_P >= 1e-8 screen the routine applies.
        std::vector<double> coeff(static_cast<std::size_t>(npts), 0.0);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double w = grid->points(p, 3);
            if (w == 0.0 || rho_ground(p) < 1e-8)
                continue;
            coeff[static_cast<std::size_t>(p)] =
                w * (v3x[static_cast<std::size_t>(p)] + v3c[static_cast<std::size_t>(p)]) * rho_relaxed(p);
        }

        auto analytic = DFT::Gradient::compute_dh_xc_pt2_gradient(
            calc._molecule, calc._shells, *grid, *ao, *hess,
            P_sym, D_sym, x_func, c_func);
        if (!analytic) { fail("compute_dh_xc_pt2_gradient: " + analytic.error()); return; }

        const double h_bohr = h_ang * 1.8897259886;
        Eigen::MatrixXd fd = Eigen::MatrixXd::Zero(2, 3);
        for (int atom = 0; atom < 2; ++atom)
        {
            for (int q = 0; q < 3; ++q)
            {
                const Eigen::VectorXd rho_p =
                    rho_with_shifted_atom(calc, *grid, P_sym, atom, q, +h_bohr);
                const Eigen::VectorXd rho_m =
                    rho_with_shifted_atom(calc, *grid, P_sym, atom, q, -h_bohr);
                if (!g_ok)
                    return;
                double acc = 0.0;
                for (Eigen::Index p = 0; p < npts; ++p)
                {
                    const double drho = (rho_p(p) - rho_m(p)) / (2.0 * h_bohr);
                    acc += drho * coeff[static_cast<std::size_t>(p)];
                }
                fd(atom, q) = acc;
            }
        }

        double max_abs_err = 0.0;
        double max_ref = 0.0;
        for (int atom = 0; atom < 2; ++atom)
            for (int q = 0; q < 3; ++q)
            {
                max_abs_err = std::max(max_abs_err, std::abs((*analytic)(atom, q) - fd(atom, q)));
                max_ref = std::max(max_ref, std::abs(fd(atom, q)));
            }
        const double rel = max_abs_err / std::max(max_ref, 1e-30);
        const std::string tag = x_name + "/" + c_name + " seed=" + std::to_string(seed) +
                                " h=" + std::to_string(h_ang);
        if (rel > 2e-4)
        {
            std::cerr << "FAIL " << tag << "\n  analytic\n"
                      << *analytic << "\n  fd\n"
                      << fd << "\n  max|derr|=" << max_abs_err
                      << " max|ref|=" << max_ref << " rel=" << rel << '\n';
            g_ok = false;
        }
        else
        {
            std::cerr << "ok   " << tag << "  rel=" << rel << " max|ref|=" << max_ref << '\n';
        }

        // NB: Term 1 alone is NOT translationally invariant -- it is the
        // basis-derivative integrand of a response term, not a total dE/dR.
        // sum_A grad_A = 0 holds only for the full E_PT2^x (all terms +
        // the SCF gradient's own moving-grid treatment), checked end-to-end
        // by the water/STO-3G B2PLYP FD at S4, not here.
    }
} // namespace

int main()
{
    check_lda_term1("lda_x", "lda_c_pw", 1u, 1e-3);
    check_lda_term1("lda_x", "lda_c_pw", 1u, 5e-4);
    check_lda_term1("lda_x", "lda_c_pw", 7u, 1e-3);

    return g_ok ? 0 : 1;
}
