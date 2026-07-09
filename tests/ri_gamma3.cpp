// RI-MP2 gradient fitted 3-index density (Step RG2.1).
//
// build_ri_gamma3_ov builds Γ3_{(ia),Q} = Σ_jb D_{(ia),(jb)} B_{(jb),Q}, the
// 3-index analog of the dense nao⁴ pair_dm2. The load-bearing check: contracting
// Γ3 back against the fitted factors reproduces the density-weighted MO ERIs to
// FITTING accuracy:
//
//     (Γ3 · B_ovᵀ)_{(ia),(jb)} = Σ_Q Γ3_{ia,Q} B_{jb,Q}
//                              = Σ_kc D_{ia,kc} (kc|jb)_RI
//   must match the DENSE   Σ_kc D_{ia,kc} (kc|jb)_dense   to fitting accuracy.
//
// This validates Γ3 as the correct fitted 2-particle density: when it later
// contracts against the derivative integrals (RG2.2) it produces the same 2e
// gradient the dense pair_dm2 does. D is a synthetic symmetric ov×ov density
// here — the identity is independent of whether D comes from real T2 amplitudes.

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
    using HartreeFock::Correlation::transform_eri;
    using HartreeFock::Correlation::RI::build_ri_gamma3_ov;
    using HartreeFock::Correlation::RI::build_ri_mo_block;
    using HartreeFock::Correlation::RI::build_ri_pair_factors;
    using HartreeFock::Correlation::RI::ensure_ri_3c_ready;
    using HartreeFock::Correlation::RI::ensure_ri_metric_ready;

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
    calc._basis._basis_name = "cc-pVDZ";
    calc._basis._basis_path = (root / "basis-sets").string();
    calc._integral._engine = HartreeFock::IntegralMethod::HeadGordonPople;
    calc._mp2.use_ri = true;
    calc._mp2.ri_basis_name = "cc-pVDZ-RIFIT";
    calc._mp2.ri_basis_path = (root / "basis-sets").string();
    calc._mp2.ri_lindep = 1e-7;

    auto basis_res = read_gbs_basis((root / "basis-sets" / "cc-pVDZ").string(),
                                    mol, HartreeFock::BasisType::Cartesian);
    if (!basis_res) { fail("read_gbs_basis: " + basis_res.error()); return 1; }
    calc._shells = std::move(*basis_res);
    auto aux_res = read_ri_basis((root / "basis-sets" / "cc-pVDZ-RIFIT").string(), mol);
    if (!aux_res) { fail("read_ri_basis: " + aux_res.error()); return 1; }
    calc._ri_aux_basis = std::make_shared<HartreeFock::AuxBasis>(std::move(*aux_res));

    if (auto p = ensure_ri_metric_ready(calc); !p) { fail(p.error()); return 1; }
    if (auto p = ensure_ri_3c_ready(calc); !p) { fail(p.error()); return 1; }

    const std::size_t nb = calc._shells.nbasis();

    // Deterministic occ/virt pseudo-orbital blocks.
    const int nocc = 3, nvirt = 4, nov = nocc * nvirt;
    Eigen::MatrixXd C_occ(nb, nocc), C_virt(nb, nvirt);
    for (std::size_t mu = 0; mu < nb; ++mu)
    {
        for (int i = 0; i < nocc; ++i)
            C_occ(static_cast<Eigen::Index>(mu), i) = std::sin(0.6 * mu + 1.1 * (i + 1));
        for (int a = 0; a < nvirt; ++a)
            C_virt(static_cast<Eigen::Index>(mu), a) = std::cos(0.4 * mu - 0.7 * (a + 1));
    }

    // Dense (ia|jb) MO block, laid out row-major i,a,j,b.
    const std::vector<double> ovov_dense = transform_eri(
        _compute_2e(build_shellpairs(calc._shells), nb, calc._integral._engine,
                    HartreeFock::ERIKernel::Coulomb, 0.0, 1e-12, nullptr),
        nb, C_occ, C_virt, C_occ, C_virt);

    // Reshape dense (ia|jb) into an [nov × nov] matrix M(ia,jb).
    Eigen::MatrixXd ovov_mat(nov, nov);
    for (int i = 0; i < nocc; ++i)
        for (int a = 0; a < nvirt; ++a)
            for (int j = 0; j < nocc; ++j)
                for (int b = 0; b < nvirt; ++b)
                    ovov_mat(i * nvirt + a, j * nvirt + b) =
                        ovov_dense[((static_cast<std::size_t>(i) * nvirt + a) * nocc + j) * nvirt + b];

    // Fitted ov factors b_ov [nov × naux].
    const Eigen::MatrixXd pf = build_ri_pair_factors(calc);
    const Eigen::MatrixXd b_ov = build_ri_mo_block(pf, C_occ, C_virt);

    // Synthetic symmetric ov×ov density D.
    Eigen::MatrixXd D(nov, nov);
    for (int p = 0; p < nov; ++p)
        for (int q = 0; q < nov; ++q)
            D(p, q) = std::sin(0.3 * p + 0.5 * q + 1.0);
    D = 0.5 * (D + D.transpose()).eval();

    // Γ3 = D · b_ov ; re-gram Γ3 · b_ovᵀ = D · (b_ov b_ovᵀ) = D · ovov_RI.
    const Eigen::MatrixXd gamma3 = build_ri_gamma3_ov(D, b_ov);
    if (gamma3.rows() != nov || gamma3.cols() != b_ov.cols())
    {
        fail("Γ3 has wrong shape");
        return 1;
    }
    const Eigen::MatrixXd regram = gamma3 * b_ov.transpose(); // D · ovov_RI

    // Reference: D · ovov_dense. Both are [nov × nov].
    const Eigen::MatrixXd ref = D * ovov_mat;

    const double rel = (regram - ref).norm() / std::max(ref.norm(), 1e-300);
    std::cout << "Γ3 regram vs dense D·ovov: ‖Δ‖/‖ref‖ = " << rel << '\n';
    // Fitting accuracy: a wrong Γ3 packing/contraction is O(1) relative; genuine
    // RI fitting error on this block is ~1e-2. 2e-2 separates them.
    if (rel > 2e-2)
        fail("Γ3 re-gram disagrees with dense beyond fitting accuracy (>2e-2) — "
             "indicates a wrong density contraction, not fitting error");

    if (g_ok)
        std::cout << "PASS: ri_gamma3\n";
    return g_ok ? 0 : 1;
}
