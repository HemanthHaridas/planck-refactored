// N3.5.7.7 S5 (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md): FD
// verification of DFT::Gradient::compute_dh_xc_pt2_gradient -- Eq. 33's XC
// contribution to the double-hybrid PT2 gradient, the FULL geometry
// derivative of
//
//   Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>
//          = integral w * { vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D) } dr
//
// (V_xc = the SCF operator's XC part, Eq. 10, FIRST functional
// derivatives; D = relaxed PT2 difference density). d/dR has THREE pieces
// (Python/PySCF-validated to rel 3e-9 -- see the scope doc's S5 section):
//
//   XC_I   basis-function derivative of rho_D (drho_channel(D) /
//          dg_axis_spin(D)) against the FIRST XC derivatives  -- dominant
//          (~85% of the term)
//   XC_II  rho_P inside V_xc[rho_P] responds (drho_channel(P) /
//          dg_axis_spin(P)) against the SECOND XC derivatives  -- ~15%
//   XC_III grid quadrature moving frame: Becke partition weight response
//          + point translation (attributed to the owner atom)  -- ~0.1%,
//          but load-bearing for exact translational invariance
//
// Since Phi_XC is itself a grid integral, its d/dR is a TRUE geometry
// derivative -- basis functions move, grid points move with their owner,
// quadrature weights move. The FD reference below therefore rebuilds the
// molecule (grid + basis) at each displaced geometry and central-
// differences Phi_XC. sum_A grad_A = 0 is now a valid check.
//
// He2/STO-3G: two atoms, closed shell, Normal grid. Strongly diagonally-
// dominant random symmetric P/D keeps rho_P above the v2rho2 ~ rho^(-2/3)
// regime on the whole quadrature support -- a math-correctness gate, not a
// physical-density one.
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

    // Phi_XC at a geometry: rebuild grid + basis, evaluate rho_P / rho_D on
    // that geometry's own grid (both from the SAME frozen density matrices),
    // sum w * { vrho*rho_D + 2*vsigma*(grad_rho_P . grad_rho_D) }.
    // Eq. 33's (x) acts only on the SCF-operator side; P and D matrices are
    // frozen, but everything derived from them on the grid moves.
    std::expected<double, std::string> phi_xc_at_geometry(
        const HartreeFock::Calculator &base,
        const Eigen::MatrixXd &P_sym,
        const Eigen::MatrixXd &D_sym,
        DFT::XC::Functional &x_func,
        DFT::XC::Functional &c_func,
        const Eigen::Vector3d *displace_atom0,
        const Eigen::Vector3d *displace_atom1)
    {
        HartreeFock::Calculator calc = base;
        if (displace_atom0)
            calc._molecule._coordinates.row(0) += displace_atom0->transpose();
        if (displace_atom1)
            calc._molecule._coordinates.row(1) += displace_atom1->transpose();
        calc._molecule.set_standard_from_bohr(calc._molecule._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / "sto-3g";
        auto b = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), calc._molecule, calc._basis._basis);
        if (!b)
            return std::unexpected("read_gbs_basis (displaced): " + b.error());
        calc._shells = std::move(*b);

        auto grid = DFT::MakeMolecularGrid(calc._molecule, DFT::GridLevel::Normal);
        if (!grid)
            return std::unexpected("MakeMolecularGrid: " + grid.error());
        auto ao = DFT::evaluate_ao_basis_on_grid(calc._shells, *grid);
        if (!ao)
            return std::unexpected("evaluate_ao_basis_on_grid: " + ao.error());

        auto gp = DFT::evaluate_density_on_grid(*ao, P_sym);
        if (!gp)
            return std::unexpected("density_on_grid(P): " + gp.error());
        auto gd = DFT::evaluate_density_on_grid(*ao, D_sym);
        if (!gd)
            return std::unexpected("density_on_grid(D): " + gd.error());

        const Eigen::Index npts = grid->points.rows();
        std::vector<double> rho_vec(static_cast<std::size_t>(npts)), sigma_vec(static_cast<std::size_t>(npts));
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            rho_vec[static_cast<std::size_t>(p)] = gp->total.rho(p);
            sigma_vec[static_cast<std::size_t>(p)] =
                gp->total.grad_x(p) * gp->total.grad_x(p) +
                gp->total.grad_y(p) * gp->total.grad_y(p) +
                gp->total.grad_z(p) * gp->total.grad_z(p);
        }

        std::vector<double> ex, vrx, vsx, ec, vrc, vsc;
        const bool lda = x_func.is_lda_like();
        if (lda)
        {
            if (auto r = x_func.evaluate_lda_exc_vxc(rho_vec, static_cast<int>(npts), ex, vrx); !r)
                return std::unexpected(r.error());
            if (auto r = c_func.evaluate_lda_exc_vxc(rho_vec, static_cast<int>(npts), ec, vrc); !r)
                return std::unexpected(r.error());
            vsx.assign(static_cast<std::size_t>(npts), 0.0);
            vsc.assign(static_cast<std::size_t>(npts), 0.0);
        }
        else
        {
            if (auto r = x_func.evaluate_gga_exc_vxc(rho_vec, sigma_vec, static_cast<int>(npts), ex, vrx, vsx); !r)
                return std::unexpected(r.error());
            if (auto r = c_func.evaluate_gga_exc_vxc(rho_vec, sigma_vec, static_cast<int>(npts), ec, vrc, vsc); !r)
                return std::unexpected(r.error());
        }
        const bool combined = x_func.is_combined_exchange_correlation();

        double phi = 0.0;
        for (Eigen::Index p = 0; p < npts; ++p)
        {
            const double w = grid->points(p, 3);
            if (w == 0.0 || gp->total.rho(p) < 1e-8)
                continue;
            const std::size_t pi = static_cast<std::size_t>(p);
            const double vrho = vrx[pi] + (combined ? 0.0 : vrc[pi]);
            const double vsigma = vsx[pi] + (combined ? 0.0 : vsc[pi]);
            const double dloc = gd->total.rho(p);
            const double gd_dot = gp->total.grad_x(p) * gd->total.grad_x(p) +
                                  gp->total.grad_y(p) * gd->total.grad_y(p) +
                                  gp->total.grad_z(p) * gd->total.grad_z(p);
            phi += w * (vrho * dloc + 2.0 * vsigma * gd_dot);
        }
        return phi;
    }

    void check(const std::string &x_name, const std::string &c_name,
               unsigned seed, double h_ang)
    {
        HartreeFock::Calculator calc = make_he2(2.0);
        if (!g_ok) return;

        const Eigen::Index nb = calc._shells.nbasis();
        std::mt19937 rng(seed);
        const Eigen::MatrixXd P_sym = random_symmetric(nb, rng);
        const Eigen::MatrixXd D_sym = random_symmetric(nb, rng);

        auto x_func = require_functional(x_name);
        auto c_func = require_functional(c_name);
        if (!g_ok) return;

        auto grid = DFT::MakeMolecularGrid(calc._molecule, DFT::GridLevel::Normal);
        if (!grid) { fail("MakeMolecularGrid: " + grid.error()); return; }
        auto ao = DFT::evaluate_ao_basis_on_grid(calc._shells, *grid);
        if (!ao) { fail("evaluate_ao_basis_on_grid: " + ao.error()); return; }
        auto hess = DFT::evaluate_ao_hessian_on_grid(calc._shells, *grid);
        if (!hess) { fail("evaluate_ao_hessian_on_grid: " + hess.error()); return; }

        auto analytic = DFT::Gradient::compute_dh_xc_pt2_gradient(
            calc._molecule, calc._shells, *grid, *ao, *hess, P_sym, D_sym, x_func, c_func);
        if (!analytic) { fail("compute_dh_xc_pt2_gradient: " + analytic.error()); return; }

        const double h_bohr = h_ang * 1.8897259886;
        Eigen::MatrixXd fd = Eigen::MatrixXd::Zero(2, 3);
        for (int atom = 0; atom < 2; ++atom)
        {
            for (int q = 0; q < 3; ++q)
            {
                Eigen::Vector3d dvec = Eigen::Vector3d::Zero();
                dvec(q) = h_bohr;
                const Eigen::Vector3d *d0p = (atom == 0) ? &dvec : nullptr;
                const Eigen::Vector3d *d1p = (atom == 1) ? &dvec : nullptr;
                Eigen::Vector3d dneg = -dvec;
                const Eigen::Vector3d *d0m = (atom == 0) ? &dneg : nullptr;
                const Eigen::Vector3d *d1m = (atom == 1) ? &dneg : nullptr;

                auto phip = phi_xc_at_geometry(calc, P_sym, D_sym, x_func, c_func, d0p, d1p);
                auto phim = phi_xc_at_geometry(calc, P_sym, D_sym, x_func, c_func, d0m, d1m);
                if (!phip || !phim) { fail("phi_xc_at_geometry: " + (phip ? phim.error() : phip.error())); return; }
                fd(atom, q) = (*phip - *phim) / (2.0 * h_bohr);
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
        const std::string tag = x_name + "/" + c_name + " seed=" + std::to_string(seed) +
                                " h=" + std::to_string(h_ang);
        if (rel > 5e-4)
        {
            std::cerr << "FAIL " << tag << "\n  analytic\n"
                      << *analytic << "\n  fd\n"
                      << fd << "\n  rel=" << rel << " max|ref|=" << max_ref << '\n';
            g_ok = false;
        }
        else
            std::cerr << "ok   " << tag << "  rel=" << rel << " max|ref|=" << max_ref << '\n';

        // Translational invariance: the full geometry derivative of Phi_XC
        // must sum to zero over atoms (now XC_III is included).
        for (int q = 0; q < 3; ++q)
        {
            const double s = (*analytic)(0, q) + (*analytic)(1, q);
            const double scale = std::max({std::abs((*analytic)(0, q)),
                                           std::abs((*analytic)(1, q)), 1e-12});
            if (std::abs(s) / scale > 1e-4)
            {
                std::cerr << "FAIL " << tag << ": sum_A grad_A(" << q << ")/scale = "
                          << s / scale << '\n';
                g_ok = false;
            }
        }
    }
} // namespace

int main()
{
    // LDA (lda_x + lda_c_pw) and GGA (pbe x + pbe c) -- both non-combined
    // pairs so the combined-XC guard stays inert.
    check("lda_x", "lda_c_pw", 1u, 1e-3);
    check("lda_x", "lda_c_pw", 1u, 5e-4);
    check("lda_x", "lda_c_pw", 7u, 1e-3);

    check("gga_x_pbe", "gga_c_pbe", 1u, 1e-3);
    check("gga_x_pbe", "gga_c_pbe", 1u, 5e-4);
    check("gga_x_pbe", "gga_c_pbe", 7u, 1e-3);

    return g_ok ? 0 : 1;
}
