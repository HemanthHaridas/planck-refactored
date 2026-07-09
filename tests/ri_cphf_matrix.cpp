// RI-MP2 gradient: RI-fitted RHF CPHF orbital Hessian (Step RG3.1).
//
// build_rhf_cphf_matrix_ri assembles A_{bj,ai} = (e_a-e_i)δ +
// [4(ai|jb)-(ab|ji)-(aj|bi)] from the RI 3-center factors instead of the nao⁴
// MO ERI tensor. Gate: RI A vs a dense A built from transform_eri, to fitting
// accuracy, on a fixed geometry. Synthetic C/eps — the identity is independent
// of whether they come from a real SCF.

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
#include "post_hf/rhf_response.h"
#include "post_hf/ri/ri_eri.h"

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
    using HartreeFock::BasisFunctions::read_gbs_basis;
    using HartreeFock::BasisFunctions::read_ri_basis;
    using HartreeFock::Correlation::build_rhf_cphf_matrix_ri;
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

    // RI A.
    auto A_ri_res = build_rhf_cphf_matrix_ri(calc, C, eps, nocc, nvirt);
    if (!A_ri_res) { fail("build_rhf_cphf_matrix_ri: " + A_ri_res.error()); return 1; }
    const Eigen::MatrixXd &A_ri = *A_ri_res;

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

    const double rel =
        (A_ri - A_dense).norm() / std::max(A_dense.norm(), 1e-300);
    std::cout << "RI CPHF A vs dense A: ‖Δ‖/‖ref‖ = " << rel << '\n';
    // Fitting accuracy on STO-3G / cc-pVDZ-RIFIT. A wrong block/formula is O(1)
    // relative; genuine RI fitting error is ~1e-3.
    if (rel > 1e-2)
        fail("RI CPHF A disagrees with dense beyond fitting accuracy (>1e-2)");

    if (g_ok)
        std::cout << "PASS: ri_cphf_matrix\n";
    return g_ok ? 0 : 1;
}
