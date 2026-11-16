// The HF orbital Hessian operator (Correlation::OrbitalHessian,
// docs/SCF_ORBITAL_HESSIAN_OPERATOR.md) against an independent dense
// A_{bj,ai} = (e_a-e_i)δ + [4(ai|jb)-(ab|ji)-(aj|bi)] built here from
// transform_eri:
//   - exact operator (the SCF ran without RI-JK): equal to roundoff;
//   - fitted operator (scf_ri_jk): equal to it to fitting accuracy;
//   - UHF operator on a closed shell, [x, x]: equal to the RHF operator.
// Synthetic C/eps; the identities do not depend on a real SCF.

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "base/types.h"
#include "basis/basis.h"
#include "basis/rifit.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include "post_hf/integrals.h"
#include "post_hf/orbital_hessian.h"
#include "post_hf/ri/ri_eri.h"
#include "require_boys_library.h"

namespace
{
    bool g_ok = true;
    void fail(const std::string &m)
    {
        std::cerr << "FAIL: " << m << '\n';
        g_ok = false;
    }
    std::filesystem::path repo_root()
    {
        if (const char *env = std::getenv("BASIS_PATH"); env && *env)
            return std::filesystem::path(env).parent_path();
        return std::filesystem::current_path();
    }
}

int main()
{
    if (!require_boys_library()) return 1;

    using HartreeFock::BasisFunctions::read_gbs_basis;
    using HartreeFock::BasisFunctions::read_ri_basis;
    using HartreeFock::Correlation::OrbitalHessian;
    using HartreeFock::Correlation::transform_eri;
    using namespace HartreeFock::Correlation::RI;

    const auto root = repo_root();

    HartreeFock::Molecule mol;
    mol.natoms = 3;
    mol.atomic_numbers.resize(3);
    mol.atomic_numbers << 8, 1, 1;
    mol._standard.resize(3, 3);
    mol._standard << 0.0, 0.0, 0.0, 0.0, 1.43, 1.11, 0.0, -1.43, 1.11;
    mol._standard_is_bohr = true;

    HartreeFock::Calculator calc;
    calc._molecule = mol;
    calc._basis._basis_name = "sto-3g";
    calc._basis._basis_path = (root / "basis-sets").string();
    calc._integral._engine = HartreeFock::IntegralMethod::HeadGordonPople;
    calc._mp2.use_ri = true;
    calc._mp2.ri_basis_name = "cc-pVDZ-RIFIT";
    calc._mp2.ri_basis_path = (root / "basis-sets").string();
    calc._mp2.ri_lindep = 1e-7;
    calc._integral._tol_eri = 1e-14; // the exact comparison is to roundoff

    auto basis_res = read_gbs_basis((root / "basis-sets" / "sto-3g").string(),
                                    mol, HartreeFock::BasisType::Cartesian);
    if (!basis_res) { fail("read_gbs_basis: " + basis_res.error()); return 1; }
    calc._shells = std::move(*basis_res);
    auto aux_res = read_ri_basis((root / "basis-sets" / "cc-pVDZ-RIFIT").string(), mol);
    if (!aux_res) { fail("read_ri_basis: " + aux_res.error()); return 1; }
    calc._ri_aux_basis = std::make_shared<HartreeFock::AuxBasis>(std::move(*aux_res));

    if (auto p = ensure_ri_metric_ready(calc); !p) { fail(p.error()); return 1; }
    if (auto p = ensure_ri_3c_ready(calc); !p) { fail(p.error()); return 1; }

    const int nb = static_cast<int>(calc._shells.nbasis());
    const int nocc = 4, nvirt = nb - nocc;
    if (nvirt <= 0) { fail("need nvirt > 0"); return 1; }

    // Deterministic pseudo-orbitals and orbital energies.
    Eigen::MatrixXd C(nb, nb);
    for (int mu = 0; mu < nb; ++mu)
        for (int p = 0; p < nb; ++p)
            C(mu, p) = std::sin(0.5 * mu + 0.3 * p + 0.2 * (mu * p % 3));
    Eigen::VectorXd eps(nb);
    for (int p = 0; p < nb; ++p)
        eps(p) = -1.0 + 0.4 * p; // ascending, occ below virt

    const auto shell_pairs = build_shellpairs(calc._shells);


    // Dense reference A from transform_eri (full MO ERI).
    const std::vector<double> eri_ao = _compute_2e(
        build_shellpairs(calc._shells), static_cast<std::size_t>(nb),
        calc._integral._engine, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-12, nullptr);
    const std::vector<double> eri_mo = transform_eri(
        eri_ao, static_cast<std::size_t>(nb), C, C, C, C);
    auto idx = [nb](int p, int q, int r, int s) -> std::size_t
    {
        return ((static_cast<std::size_t>(p) * nb + q) * nb + r) * nb + s;
    };

    Eigen::MatrixXd A_dense = Eigen::MatrixXd::Zero(nvirt * nocc, nvirt * nocc);
    auto ai = [nocc](int a, int i) { return a * nocc + i; };
    for (int a = 0; a < nvirt; ++a)
        for (int i = 0; i < nocc; ++i)
        {
            const int aa = nocc + a;
            A_dense(ai(a, i), ai(a, i)) += eps(aa) - eps(i);
            for (int b = 0; b < nvirt; ++b)
                for (int j = 0; j < nocc; ++j)
                {
                    const int bb = nocc + b;
                    const double ai_jb = eri_mo[idx(aa, i, j, bb)];
                    const double ab_ji = eri_mo[idx(aa, bb, j, i)];
                    const double aj_bi = eri_mo[idx(aa, j, bb, i)];
                    A_dense(ai(b, j), ai(a, i)) += 4.0 * ai_jb - ab_ji - aj_bi;
                }
        }

    const double ref = std::max(A_dense.norm(), 1e-300);

    // Exact operator: the SCF ran without RI-JK.
    calc._scf._ri_jk = false;
    const Eigen::MatrixXd A_exact = OrbitalHessian::rhf(calc, shell_pairs, C, eps, nocc).materialize();
    const double rel_exact = (A_exact - A_dense).norm() / ref;
    std::cout << "exact operator vs dense A: ‖Δ‖/‖ref‖ = " << rel_exact << '\n';
    if (!(rel_exact < 1e-12))
        fail("exact orbital-Hessian operator differs from the dense A");

    // Fitted operator: the SCF ran RI-JK.
    calc._scf._ri_jk = true;
    const Eigen::MatrixXd A_fit = OrbitalHessian::rhf(calc, shell_pairs, C, eps, nocc).materialize();
    const double rel_fit = (A_fit - A_dense).norm() / ref;
    std::cout << "fitted operator vs dense A: ‖Δ‖/‖ref‖ = " << rel_fit << '\n';
    // Fitting accuracy on STO-3G / cc-pVDZ-RIFIT. A wrong block/formula is O(1)
    // relative; genuine RI fitting error is ~1e-3.
    if (!(rel_fit < 1e-2) || !(rel_fit > 1e-8))
        fail("fitted operator is not a fit of the dense A (expect 1e-8 < rel < 1e-2)");

    // UHF on a closed shell: [x, x] gives the RHF action in each spin block.
    calc._scf._ri_jk = false;
    {
        const OrbitalHessian r = OrbitalHessian::rhf(calc, shell_pairs, C, eps, nocc);
        const OrbitalHessian u = OrbitalHessian::uhf(calc, shell_pairs, C, C, eps, eps, nocc, nocc);
        Eigen::VectorXd x(nvirt * nocc);
        for (Eigen::Index k = 0; k < x.size(); ++k)
            x(k) = std::cos(0.7 * static_cast<double>(k) + 0.1);
        Eigen::VectorXd xx(2 * x.size());
        xx << x, x;
        const Eigen::VectorXd yr = r(x), yu = u(xx);
        const double d = std::max((yu.head(x.size()) - yr).cwiseAbs().maxCoeff(),
                                  (yu.tail(x.size()) - yr).cwiseAbs().maxCoeff());
        std::cout << "UHF [x,x] vs RHF x: max|Δ| = " << d << '\n';
        if (!(d < 1e-12 * std::max(1.0, yr.cwiseAbs().maxCoeff())))
            fail("UHF operator on a closed shell differs from the RHF operator");
    }

    if (g_ok)
        std::cout << "PASS: ri_cphf_matrix\n";
    return g_ok ? 0 : 1;
}
