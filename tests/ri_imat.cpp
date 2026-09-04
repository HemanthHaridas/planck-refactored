// RI-MP2 gradient: RI Lagrangian imat (Step RG3.3).
//
// build_ri_imat computes  imat(q,v) = Σ_{p,r,s} (pq|rs)·dm2buf[p,v,r,s]
// through the fitted ERI (pq|rs) = Σ_Q B[Q](p,q) B[Q](r,s), never forming the
// nao⁴ ERI. Gate: RI imat vs a dense imat built from _compute_2e, to fitting
// accuracy. dm2buf is a synthetic nao⁴ tensor — the contraction identity is
// independent of whether it comes from real T2.

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

    const int nao = static_cast<int>(calc._shells.nbasis());
    auto idx = [nao](int p, int q, int r, int s) -> std::size_t
    {
        return ((static_cast<std::size_t>(p) * nao + q) * nao + r) * nao + s;
    };

    // Synthetic nao⁴ dm2buf, row-major [p][v][r][s]. No symmetry assumed.
    std::vector<double> dm2buf(static_cast<std::size_t>(nao) * nao * nao * nao);
    for (int p = 0; p < nao; ++p)
        for (int v = 0; v < nao; ++v)
            for (int r = 0; r < nao; ++r)
                for (int s = 0; s < nao; ++s)
                    dm2buf[idx(p, v, r, s)] =
                        std::sin(0.4 * p - 0.3 * v + 0.6 * r + 0.2 * s + 0.5);

    // RI imat.
    const Eigen::MatrixXd imat_ri = build_ri_imat(calc, dm2buf, nao);

    // Dense imat from _compute_2e: imat(q,v) = Σ_prs (pq|rs) dm2buf[p,v,r,s].
    const std::vector<double> eri = _compute_2e(
        build_shellpairs(calc._shells), static_cast<std::size_t>(nao),
        calc._integral._engine, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-12, nullptr);
    Eigen::MatrixXd imat_dense = Eigen::MatrixXd::Zero(nao, nao);
    for (int q = 0; q < nao; ++q)
        for (int v = 0; v < nao; ++v)
        {
            double val = 0.0;
            for (int p = 0; p < nao; ++p)
                for (int r = 0; r < nao; ++r)
                    for (int s = 0; s < nao; ++s)
                        val += eri[idx(p, q, r, s)] * dm2buf[idx(p, v, r, s)];
            imat_dense(q, v) = val;
        }

    const double rel =
        (imat_ri - imat_dense).norm() / std::max(imat_dense.norm(), 1e-300);
    std::cout << "RI imat vs dense imat: ‖Δ‖/‖ref‖ = " << rel << '\n';
    if (rel > 1e-2)
        fail("RI imat disagrees with dense beyond fitting accuracy (>1e-2)");

    if (g_ok)
        std::cout << "PASS: ri_imat\n";
    return g_ok ? 0 : 1;
}
