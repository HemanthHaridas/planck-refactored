// RI-MP2 gradient two-electron term (Step RG2.2).
//
// build_ri_two_electron_gradient contracts the fitted 3-index 2-particle
// density against the RG1 derivative tensors (dJ = compute_3c_eri_deriv,
// dV = compute_2c_eri_deriv), producing the same per-atom two_e_terms the dense
// 4-center path builds — the load-bearing RG2 gate: one physical quantity, two
// routes.
//
//   dense:  E2(atom,c) = Σ_{p∈atom, q,r,s} d/dR(pq|rs)_c · Γ[p,q,r,s]
//   RI:     E2(atom,c) = 2·Σ_{(μν),Q} w·gamma3·dJ  −  Σ_{PQ} γ_{PQ}·dV
//
// with gamma3 = Γ·B (fitted proj), x_proj = Γ·X (V^{-1}-solved proj), and the
// metric charge γ_{PQ} folded from x_proj·gamma3. Γ is a SYNTHETIC symmetric
// AO-pair density here — the two-term structure and the −dV metric correction
// are what's under test (the derivation risk), independent of real T2.
// The dense side never fits, so the two agree only to RI fitting accuracy.
// The dense reference mirrors compute_rmp2_gradient EXACTLY: bra-p-leg-only
// derivative, p restricted to the atom's AOs, with the overall factor of 2 that
// PySCF/the dense path applies (dm2v = 2·dm2buf). Reproducing that convention is
// the point — a textbook Γ off by a leg/permutation factor would not match.

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
#include "integrals/hgp.h"
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
    // Packed index hi*(hi+1)/2 + lo for μ,ν (μ≥ν order).
    std::size_t packed(std::size_t mu, std::size_t nu)
    {
        const std::size_t hi = std::max(mu, nu), lo = std::min(mu, nu);
        return hi * (hi + 1) / 2 + lo;
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

    const std::size_t nb = calc._shells.nbasis();
    const std::size_t npair = nb * (nb + 1) / 2;
    const std::size_t natoms = calc._molecule.natoms;

    // Fitted pair factors B = J V^{-1/2} (packed pair × aux) and X = J V^{-1}.
    const Eigen::MatrixXd B_pf = build_ri_pair_factors(calc);
    const std::size_t naux = B_pf.cols();
    const auto &metric = *calc._ri_metric_factor;
    // X = B_pf · V^{-1/2}. For Cholesky V = L Lᵀ so V^{-1/2} is not L directly;
    // build V^{-1} explicitly and X = J V^{-1} to stay unambiguous.
    // J = _ri_j3c (packed pair × aux, raw 3-center).
    const Eigen::MatrixXd &J = calc._ri_j3c;
    Eigen::MatrixXd Vinv;
    {
        // Reconstruct V^{-1} from the factorization.
        Eigen::MatrixXd Vhalf_inv; // X_metric with X V Xᵀ = I  ⇒ V^{-1} = Xᵀ X
        if (metric.method == MetricFactorization::Method::Eigen)
            Vhalf_inv = metric.transform; // rows = kept modes
        else
        {
            // L Lᵀ = V ⇒ V^{-1} = L^{-T} L^{-1}; set Vhalf_inv = L^{-1} (lower solve on I).
            const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(J.cols(), J.cols());
            Vhalf_inv = metric.transform.triangularView<Eigen::Lower>().solve(I);
        }
        Vinv = Vhalf_inv.transpose() * Vhalf_inv;
    }
    const Eigen::MatrixXd X_pf = J * Vinv; // packed pair × aux

    // Synthetic symmetric AO-pair density Γ[μ,ν,λ,σ] = f(μν)·f(λσ) + f(λσ)·f(μν),
    // built from a per-pair vector so it is symmetric under (μν)↔(λσ) and under
    // μ↔ν / λ↔σ. Store packed: Gp[pairμν][pairλσ], symmetric.
    Eigen::VectorXd s(npair);
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu <= mu; ++nu)
            s(packed(mu, nu)) = std::sin(0.7 * mu - 0.4 * nu + 0.9);
    Eigen::MatrixXd Gp = s * s.transpose(); // npair × npair, symmetric

    // gamma3[pair,Q] = Σ_{λ≥σ} w_ket · Gp[pair, ketpair] · B[ketpair,Q].
    // x_proj[pair,P] same with X. w_ket = (λ==σ?1:2) folds the full λσ sum.
    Eigen::VectorXd wket(npair);
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu <= mu; ++nu)
            wket(packed(mu, nu)) = (mu == nu) ? 1.0 : 2.0;
    // Both gradient terms couple through X = J V^{-1}: gamma3 = Σ Γ·X, and the
    // raw X factors are the other leg of the metric fold. B_pf (V^{-1/2}) is not
    // used here.
    const Eigen::MatrixXd Gw = Gp * wket.asDiagonal(); // weight the ket
    const Eigen::MatrixXd gamma3 = Gw * X_pf;          // npair × naux
    const Eigen::MatrixXd x_proj = X_pf;               // npair × naux (raw)
    (void)B_pf;

    // Derivative tensors.
    auto dJ_res = compute_3c_eri_deriv(calc);
    if (!dJ_res) { fail("compute_3c_eri_deriv: " + dJ_res.error()); return 1; }
    auto dV_res = compute_2c_eri_deriv(calc);
    if (!dV_res) { fail("compute_2c_eri_deriv: " + dV_res.error()); return 1; }

    const Eigen::MatrixXd ri_two_e = build_ri_two_electron_gradient(
        gamma3, x_proj, *dJ_res, *dV_res, natoms, nb);

    // Dense reference: contract the SAME synthetic Γ against 4-center AO
    // derivatives. Γ_full[p,q,r,s] = Gp[packed(p,q)][packed(r,s)] (symmetric in
    // p↔q, r↔s, and (pq)↔(rs)). Only the p-leg (center A, comp 0..2) derivative
    // is summed, with p ∈ atom's AOs — same as compute_rmp2_gradient's two_e.
    std::vector<std::vector<int>> atom_aos(natoms);
    {
        const auto &bfs = calc._shells._basis_functions;
        for (std::size_t p = 0; p < nb; ++p)
            atom_aos[bfs[p]._shell->_atom_index].push_back(static_cast<int>(p));
    }
    const auto &bfs = calc._shells._basis_functions;
    Eigen::MatrixXd dense_two_e = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(natoms), 3);
    for (std::size_t atom = 0; atom < natoms; ++atom)
        for (int p : atom_aos[atom])
            for (std::size_t q = 0; q < nb; ++q)
                for (std::size_t r = 0; r < nb; ++r)
                    for (std::size_t s2 = 0; s2 < nb; ++s2)
                    {
                        const double g = Gp(packed(p, q), packed(r, s2));
                        if (g == 0.0)
                            continue;
                        const HartreeFock::ShellPair spAB(bfs[p], bfs[q]);
                        const HartreeFock::ShellPair spCD(bfs[r], bfs[s2]);
                        const auto dI =
                            HartreeFock::HeadGordonPople::_compute_eri_deriv_elem(
                                spAB, spCD);
                        // Match compute_rmp2_gradient: dm2v = 2·Γ, bra-p-leg only.
                        for (int c = 0; c < 3; ++c)
                            dense_two_e(static_cast<Eigen::Index>(atom), c) +=
                                dI[c] * (2.0 * g);
                    }

    const double rel =
        (ri_two_e - dense_two_e).norm() / std::max(dense_two_e.norm(), 1e-300);
    std::cout << "RI two_e vs dense two_e: ‖Δ‖/‖ref‖ = " << rel << '\n';
    std::cout << "  dense[0] = " << dense_two_e.row(0) << '\n';
    std::cout << "  RI   [0] = " << ri_two_e.row(0) << '\n';
    // Fitting accuracy on this block (STO-3G / cc-pVDZ-RIFIT). A wrong factor or
    // a missing metric term is O(1) relative; genuine fitting error is ~1e-2.
    if (rel > 3e-2)
        fail("RI two_e disagrees with dense beyond fitting accuracy (>3e-2) — "
             "wrong two-term contraction, factor, or metric correction");

    if (g_ok)
        std::cout << "PASS: ri_two_electron_gradient\n";
    return g_ok ? 0 : 1;
}
