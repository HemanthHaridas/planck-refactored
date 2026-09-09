// S2/S3 (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md, N3.5.7.7): FD
// verification of DFT::Gradient::compute_dh_xc_pt2_gradient -- Eq. 33's XC
// contribution to the double-hybrid PT2 gradient. It is d/dR of
//
//   Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>
//          = integral w * { (df/drho)*rho_D + 2*(df/dgamma)*(grad_rho_P . grad_rho_D) } dr
//
// (V_xc = the SCF operator's XC part, Eq. 10, FIRST functional
// derivatives; D = relaxed PT2 difference density). The response part --
// rho_P / grad_rho_P move via their basis-function derivative at a FIXED
// spatial point (Eq. 15), rho_D / grad_rho_D frozen -- needs the SECOND
// functional derivative. NO third derivative (the paper's own text, p.6).
// Plain grid integrals, no moving-grid correction.
//
// LDA reference: coeff = w * (v2rho2_x + v2rho2_c)(ground rho) * rho_D,
// frozen; FD of rho_P w.r.t. shifted AO centers (grid/weights/coeff held
// fixed) contracted with it.
// GGA reference: central difference of Phi_XC itself, with the FIRST
// derivatives (vrho, vsigma) and grad_rho_P re-evaluated at the shifted-AO
// geometry and rho_D / grad_rho_D frozen at the base geometry. Linear in
// D. Independent of the routine's internals in both cases.
//
// He2/STO-3G: two atoms, closed shell, Normal grid. Strongly diagonally-
// dominant random P/D keeps rho_P above the v2rho2 ~ rho^(-2/3) regime on
// the whole quadrature support -- a math-correctness gate, not a physical-
// density one.
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
    // rho_P well away from zero (v2rho2 ~ rho^(-2/3) diverges otherwise).
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

    struct GridDensity
    {
        Eigen::VectorXd rho;
        Eigen::MatrixXd grad; // npts x 3
    };

    GridDensity grid_density(const DFT::AOGridEvaluation &ao, const Eigen::MatrixXd &P_sym)
    {
        const Eigen::Index npts = ao.npoints();
        GridDensity gd;
        gd.rho.resize(npts);
        gd.grad.resize(npts, 3);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const Eigen::VectorXd phi = ao.values.row(p).transpose();
            const Eigen::VectorXd Pphi = P_sym * phi;
            gd.rho(p) = phi.dot(Pphi);
            // grad rho = 2 * (grad phi)^T P phi  (evaluate_density_on_grid convention)
            gd.grad(p, 0) = 2.0 * ao.grad_x.row(p).dot(Pphi.transpose());
            gd.grad(p, 1) = 2.0 * ao.grad_y.row(p).dot(Pphi.transpose());
            gd.grad(p, 2) = 2.0 * ao.grad_z.row(p).dot(Pphi.transpose());
        }
        return gd;
    }

    // Basis rebuilt with atom `atom`'s shell centers shifted by `delta`
    // along `q`. The grid (points + weights) does NOT move -- only the AOs.
    std::expected<HartreeFock::Basis, std::string> shifted_basis(
        const HartreeFock::Calculator &base, int atom, int q, double delta)
    {
        HartreeFock::Calculator c = base;
        c._molecule._coordinates(atom, q) += delta;
        c._molecule.set_standard_from_bohr(c._molecule._coordinates);
        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / "sto-3g";
        return HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), c._molecule, c._basis._basis);
    }

    // -------- LDA --------------------------------------------------------
    void check_lda(unsigned seed, double h_ang)
    {
        HartreeFock::Calculator calc = make_he2(2.0);
        if (!g_ok) return;

        const Eigen::Index nb = calc._shells.nbasis();
        std::mt19937 rng(seed);
        const Eigen::MatrixXd P_sym = random_symmetric(nb, rng);
        const Eigen::MatrixXd D_sym = random_symmetric(nb, rng);

        auto x_func = require_functional("lda_x");
        auto c_func = require_functional("lda_c_pw");
        if (!g_ok) return;

        auto grid = DFT::MakeMolecularGrid(calc._molecule, DFT::GridLevel::Normal);
        if (!grid) { fail("MakeMolecularGrid: " + grid.error()); return; }
        auto ao = DFT::evaluate_ao_basis_on_grid(calc._shells, *grid);
        if (!ao) { fail("evaluate_ao_basis_on_grid: " + ao.error()); return; }
        auto hess = DFT::evaluate_ao_hessian_on_grid(calc._shells, *grid);
        if (!hess) { fail("evaluate_ao_hessian_on_grid: " + hess.error()); return; }

        const Eigen::Index npts = grid->points.rows();
        const GridDensity gp = grid_density(*ao, P_sym);
        const GridDensity gdd = grid_density(*ao, D_sym);

        std::vector<double> rho_vec(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
            rho_vec[static_cast<std::size_t>(p)] = gp.rho(p);

        std::vector<double> f2x, f2c;
        if (auto r = x_func.evaluate_lda_fxc(rho_vec, static_cast<int>(npts), f2x); !r) { fail("lda_fxc"); return; }
        if (auto r = c_func.evaluate_lda_fxc(rho_vec, static_cast<int>(npts), f2c); !r) { fail("lda_fxc"); return; }

        // Frozen coefficient  w * (v2rho2_x + v2rho2_c) * rho_D, same rho_P
        // >= 1e-8 screen the routine applies.
        std::vector<double> coeff(static_cast<std::size_t>(npts), 0.0);
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double w = grid->points(p, 3);
            if (w == 0.0 || gp.rho(p) < 1e-8)
                continue;
            coeff[static_cast<std::size_t>(p)] =
                w * (f2x[static_cast<std::size_t>(p)] + f2c[static_cast<std::size_t>(p)]) * gdd.rho(p);
        }

        auto analytic = DFT::Gradient::compute_dh_xc_pt2_gradient(
            calc._molecule, calc._shells, *grid, *ao, *hess, P_sym, D_sym, x_func, c_func);
        if (!analytic) { fail("compute_dh_xc_pt2_gradient: " + analytic.error()); return; }

        const double h_bohr = h_ang * 1.8897259886;
        Eigen::MatrixXd fd = Eigen::MatrixXd::Zero(2, 3);
        for (int atom = 0; atom < 2; ++atom)
        {
            for (int q = 0; q < 3; ++q)
            {
                auto bp = shifted_basis(calc, atom, q, +h_bohr);
                auto bm = shifted_basis(calc, atom, q, -h_bohr);
                if (!bp || !bm) { fail("shifted_basis (lda)"); return; }
                auto aop = DFT::evaluate_ao_basis_on_grid(*bp, *grid);
                auto aom = DFT::evaluate_ao_basis_on_grid(*bm, *grid);
                if (!aop || !aom) { fail("ao on shifted grid (lda)"); return; }
                double acc = 0.0;
                for (Eigen::Index p = 0; p < npts; ++p)
                {
                    const Eigen::VectorXd pp = aop->values.row(p).transpose();
                    const Eigen::VectorXd pm = aom->values.row(p).transpose();
                    const double drho = (pp.dot(P_sym * pp) - pm.dot(P_sym * pm)) / (2.0 * h_bohr);
                    acc += drho * coeff[static_cast<std::size_t>(p)];
                }
                fd(atom, q) = acc;
            }
        }

        double max_abs_err = 0.0, max_ref = 0.0;
        for (int a = 0; a < 2; ++a)
            for (int q = 0; q < 3; ++q)
            {
                max_abs_err = std::max(max_abs_err, std::abs((*analytic)(a, q) - fd(a, q)));
                max_ref = std::max(max_ref, std::abs(fd(a, q)));
            }
        const double rel = max_abs_err / std::max(max_ref, 1e-30);
        const std::string tag = "LDA seed=" + std::to_string(seed) + " h=" + std::to_string(h_ang);
        if (rel > 2e-4)
        {
            std::cerr << "FAIL " << tag << "\n  analytic\n"
                      << *analytic << "\n  fd\n"
                      << fd << "\n  rel=" << rel << " max|ref|=" << max_ref << '\n';
            g_ok = false;
        }
        else
            std::cerr << "ok   " << tag << "  rel=" << rel << " max|ref|=" << max_ref << '\n';
    }

    // -------- GGA ------------------------------------------------------------
    // Central difference of  Phi_XC = integral w * { vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D) }
    // with vrho, vsigma (FIRST derivatives) and grad_rho_P re-evaluated at
    // the shifted-AO geometry; rho_D, grad_rho_D frozen.
    double gga_phi_at_geometry(
        const HartreeFock::Basis &shifted,
        const DFT::MolecularGrid &fixed_grid,
        const Eigen::MatrixXd &P_sym,
        const GridDensity &relaxed_frozen,
        DFT::XC::Functional &x_func,
        DFT::XC::Functional &c_func,
        double rho_floor)
    {
        auto ao = DFT::evaluate_ao_basis_on_grid(shifted, fixed_grid);
        if (!ao) { fail("evaluate_ao_basis_on_grid (gga phi): " + ao.error()); return 0.0; }
        const Eigen::Index npts = fixed_grid.points.rows();
        const GridDensity gp = grid_density(*ao, P_sym);

        std::vector<double> rho_vec(static_cast<std::size_t>(npts)), sigma_vec(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            rho_vec[static_cast<std::size_t>(p)] = gp.rho(p);
            sigma_vec[static_cast<std::size_t>(p)] =
                gp.grad(p, 0) * gp.grad(p, 0) + gp.grad(p, 1) * gp.grad(p, 1) + gp.grad(p, 2) * gp.grad(p, 2);
        }

        std::vector<double> ex, vrx, vsx, ec, vrc, vsc;
        x_func.evaluate_gga_exc_vxc(rho_vec, sigma_vec, static_cast<int>(npts), ex, vrx, vsx);
        c_func.evaluate_gga_exc_vxc(rho_vec, sigma_vec, static_cast<int>(npts), ec, vrc, vsc);
        if (x_func.is_combined_exchange_correlation())
        {
            std::fill(vrc.begin(), vrc.end(), 0.0);
            std::fill(vsc.begin(), vsc.end(), 0.0);
        }
        if (!g_ok) return 0.0;

        double phi = 0.0;
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double w = fixed_grid.points(p, 3);
            if (w == 0.0 || gp.rho(p) < rho_floor)
                continue;
            const std::size_t pi = static_cast<std::size_t>(p);
            const double vrho = vrx[pi] + vrc[pi];
            const double vsigma = vsx[pi] + vsc[pi];
            const double d = relaxed_frozen.rho(p);
            const double gd = gp.grad(p, 0) * relaxed_frozen.grad(p, 0) +
                              gp.grad(p, 1) * relaxed_frozen.grad(p, 1) +
                              gp.grad(p, 2) * relaxed_frozen.grad(p, 2);
            phi += w * (vrho * d + 2.0 * vsigma * gd);
        }
        return phi;
    }

    void check_gga(unsigned seed, double h_ang)
    {
        HartreeFock::Calculator calc = make_he2(2.0);
        if (!g_ok) return;

        const Eigen::Index nb = calc._shells.nbasis();
        std::mt19937 rng(seed);
        const Eigen::MatrixXd P_sym = random_symmetric(nb, rng);
        const Eigen::MatrixXd D_sym = random_symmetric(nb, rng);

        auto x_func = require_functional("gga_x_pbe");
        auto c_func = require_functional("gga_c_pbe");
        if (!g_ok) return;

        auto grid = DFT::MakeMolecularGrid(calc._molecule, DFT::GridLevel::Normal);
        if (!grid) { fail("MakeMolecularGrid: " + grid.error()); return; }
        auto ao = DFT::evaluate_ao_basis_on_grid(calc._shells, *grid);
        if (!ao) { fail("evaluate_ao_basis_on_grid: " + ao.error()); return; }
        auto hess = DFT::evaluate_ao_hessian_on_grid(calc._shells, *grid);
        if (!hess) { fail("evaluate_ao_hessian_on_grid: " + hess.error()); return; }

        const GridDensity relaxed_frozen = grid_density(*ao, D_sym);

        auto analytic = DFT::Gradient::compute_dh_xc_pt2_gradient(
            calc._molecule, calc._shells, *grid, *ao, *hess, P_sym, D_sym, x_func, c_func);
        if (!analytic) { fail("compute_dh_xc_pt2_gradient (gga): " + analytic.error()); return; }

        const double h_bohr = h_ang * 1.8897259886;
        Eigen::MatrixXd fd = Eigen::MatrixXd::Zero(2, 3);
        for (int atom = 0; atom < 2; ++atom)
        {
            for (int q = 0; q < 3; ++q)
            {
                auto bp = shifted_basis(calc, atom, q, +h_bohr);
                auto bm = shifted_basis(calc, atom, q, -h_bohr);
                if (!bp || !bm) { fail("shifted_basis (gga)"); return; }
                const double phip = gga_phi_at_geometry(*bp, *grid, P_sym, relaxed_frozen, x_func, c_func, 1e-8);
                const double phim = gga_phi_at_geometry(*bm, *grid, P_sym, relaxed_frozen, x_func, c_func, 1e-8);
                if (!g_ok) return;
                fd(atom, q) = (phip - phim) / (2.0 * h_bohr);
            }
        }

        double max_abs_err = 0.0, max_ref = 0.0;
        for (int a = 0; a < 2; ++a)
            for (int q = 0; q < 3; ++q)
            {
                max_abs_err = std::max(max_abs_err, std::abs((*analytic)(a, q) - fd(a, q)));
                max_ref = std::max(max_ref, std::abs(fd(a, q)));
            }
        const double rel = max_abs_err / std::max(max_ref, 1e-30);
        const std::string tag = "GGA seed=" + std::to_string(seed) + " h=" + std::to_string(h_ang);
        if (rel > 5e-4)
        {
            std::cerr << "FAIL " << tag << "\n  analytic\n"
                      << *analytic << "\n  fd\n"
                      << fd << "\n  rel=" << rel << " max|ref|=" << max_ref << '\n';
            g_ok = false;
        }
        else
            std::cerr << "ok   " << tag << "  rel=" << rel << " max|ref|=" << max_ref << '\n';
    }
} // namespace

int main()
{
    check_lda(1u, 1e-3);
    check_lda(1u, 5e-4);
    check_lda(7u, 1e-3);

    check_gga(1u, 1e-3);
    check_gga(1u, 5e-4);
    check_gga(7u, 1e-3);

    return g_ok ? 0 : 1;
}
