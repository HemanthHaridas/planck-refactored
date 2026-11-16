// RI-MP2 gradient: the factored pair density (RI::PairDensityBlock) and its
// two contractions, build_ri_imat_from_pair_blocks and
// build_ri_gamma3_from_pair_blocks, which replaced the nao⁴ AO buffer.
//
// Two synthetic blocks (unequal occ/virt counts, different bra and ket
// orbitals, weights 1 and 2) are expanded into the dense nao⁴ pair density by
// the documented formula, and each contraction is checked against the same
// contraction of that dense density through the same fitted factors (exact
// algebra, 1e-12). The imat is also checked against the exact-ERI imat, to
// fitting accuracy.

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

    // Synthetic orbitals and amplitudes; any values serve, the identity is
    // algebraic.
    const auto synth = [](Eigen::Index r, Eigen::Index c, double seed)
    {
        Eigen::MatrixXd m(r, c);
        for (Eigen::Index i = 0; i < r; ++i)
            for (Eigen::Index j = 0; j < c; ++j)
                m(i, j) = std::sin(seed + 0.37 * i - 0.53 * j + 0.11 * i * j);
        return m;
    };
    std::vector<PairDensityBlock> blocks;
    blocks.push_back({synth(nao, 2, 0.1), synth(nao, 5, 0.2), synth(nao, 2, 0.1), synth(nao, 5, 0.2),
                      synth(2 * 5, 2 * 5, 0.3), 1.0});
    blocks.push_back({synth(nao, 3, 0.4), synth(nao, 4, 0.5), synth(nao, 2, 0.6), synth(nao, 5, 0.7),
                      synth(3 * 4, 2 * 5, 0.8), 2.0});

    // Dense dm2[p,q,r,s] = Σ weight (B[p,q,r,s] + B[q,p,r,s]),
    // B[p,q,r,s] = Σ L_o(p,i) L_v(q,a) t[(ia),(jb)] R_v(r,b) R_o(s,j).
    std::vector<double> dm2(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
    for (const auto &b : blocks)
    {
        const Eigen::Index ni = b.bra_occ.cols(), na = b.bra_virt.cols();
        const Eigen::Index nj = b.ket_occ.cols(), nb = b.ket_virt.cols();
        for (int p = 0; p < nao; ++p)
            for (int q = 0; q < nao; ++q)
                for (int r = 0; r < nao; ++r)
                    for (int s = 0; s < nao; ++s)
                    {
                        double v = 0.0;
                        for (Eigen::Index i = 0; i < ni; ++i)
                            for (Eigen::Index a = 0; a < na; ++a)
                                for (Eigen::Index j = 0; j < nj; ++j)
                                    for (Eigen::Index c = 0; c < nb; ++c)
                                    {
                                        const double tv = b.t(i * na + a, j * nb + c) *
                                                          b.ket_virt(r, c) * b.ket_occ(s, j);
                                        v += (b.bra_occ(p, i) * b.bra_virt(q, a) +
                                              b.bra_occ(q, i) * b.bra_virt(p, a)) * tv;
                                    }
                        dm2[idx(p, q, r, s)] += b.weight * v;
                    }
    }

    // W[Q](p,v) = Σ_rs F[Q](r,s) dm2[p,v,r,s] for a symmetric factor F[Q].
    const auto dense_w = [&](const Eigen::MatrixXd &F)
    {
        Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nao, nao);
        for (int p = 0; p < nao; ++p)
            for (int v = 0; v < nao; ++v)
                for (int r = 0; r < nao; ++r)
                    for (int s = 0; s < nao; ++s)
                        W(p, v) += F(r, s) * dm2[idx(p, v, r, s)];
        return W;
    };
    const auto rel = [](const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
    { return (a - b).norm() / std::max(b.norm(), 1e-300); };

    // imat: Σ_Q B[Q]^T W[Q] through the same fitted factors.
    const std::vector<Eigen::MatrixXd> B = build_ri_3index_unpacked(calc);
    Eigen::MatrixXd imat_ref = Eigen::MatrixXd::Zero(nao, nao);
    for (const auto &BQ : B)
        imat_ref += BQ.transpose() * dense_w(BQ);
    const Eigen::MatrixXd imat_ri = build_ri_imat_from_pair_blocks(calc, blocks);
    std::cout << "imat, blocks vs dense pair density: " << rel(imat_ri, imat_ref) << '\n';
    if (rel(imat_ri, imat_ref) > 1e-12)
        fail("build_ri_imat_from_pair_blocks disagrees with the dense pair density");

    // gamma3: packed (μ≥ν) W(μν) + W(νμ) against X = J V^{-1}.
    const auto [gamma3, x_proj] = build_ri_gamma3_from_pair_blocks(calc, blocks);
    Eigen::MatrixXd gamma3_ref(gamma3.rows(), gamma3.cols());
    for (Eigen::Index Q = 0; Q < x_proj.cols(); ++Q)
    {
        Eigen::MatrixXd XQ(nao, nao);
        for (int mu = 0, pr = 0; mu < nao; ++mu)
            for (int nu = 0; nu <= mu; ++nu, ++pr)
                XQ(mu, nu) = XQ(nu, mu) = x_proj(pr, Q);
        const Eigen::MatrixXd W = dense_w(XQ);
        for (int mu = 0, pr = 0; mu < nao; ++mu)
            for (int nu = 0; nu <= mu; ++nu, ++pr)
                gamma3_ref(pr, Q) = (mu == nu) ? W(mu, nu) : W(mu, nu) + W(nu, mu);
    }
    std::cout << "gamma3, blocks vs dense pair density: " << rel(gamma3, gamma3_ref) << '\n';
    if (rel(gamma3, gamma3_ref) > 1e-12)
        fail("build_ri_gamma3_from_pair_blocks disagrees with the dense pair density");

    // Fitting accuracy: the RI imat against the exact-ERI imat.
    const std::vector<double> eri = _compute_2e(
        build_shellpairs(calc._shells), static_cast<std::size_t>(nao),
        calc._integral._engine, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-12, nullptr);
    Eigen::MatrixXd imat_dense = Eigen::MatrixXd::Zero(nao, nao);
    for (int q = 0; q < nao; ++q)
        for (int v = 0; v < nao; ++v)
            for (int p = 0; p < nao; ++p)
                for (int r = 0; r < nao; ++r)
                    for (int s = 0; s < nao; ++s)
                        imat_dense(q, v) += eri[idx(p, q, r, s)] * dm2[idx(p, v, r, s)];
    std::cout << "RI imat vs exact-ERI imat: " << rel(imat_ri, imat_dense) << '\n';
    if (rel(imat_ri, imat_dense) > 1e-2)
        fail("RI imat disagrees with the exact-ERI imat beyond fitting accuracy (>1e-2)");

    if (g_ok)
        std::cout << "PASS: ri_imat\n";
    return g_ok ? 0 : 1;
}
