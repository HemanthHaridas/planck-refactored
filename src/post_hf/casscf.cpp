// casscf.cpp — Second-order one-step CASSCF (Sun, Yang, Chan 2017)
//
// Modules:
//   1.  String generation & Slater–Condon rules
//   2.  CI solver (direct + Davidson)
//   3.  CI Hamiltonian builder (with symmetry / RAS screening)
//   4.  1-RDM, 2-RDM, and bilinear RDMs for first-order response
//   5.  Inactive Fock, Active Fock, core energy
//   6.  Q matrix
//   7.  Orbital gradient (generalized Fock)
//   8.  Orbital Hessian action H·R (diagonal approximation)
//   9.  CI response c¹ (all roots), first-order 2-RDM Γ¹, Q-matrix response Q¹
//  10.  Dressed gradient G̃ = G + H·κ + G^CI (fully consistent second-order)
//  11.  Augmented Hessian orbital step (RFO / AH)
//  12.  Orbital rotation via Cayley transform + re-orthogonalisation
//  13.  Macro/micro MCSCF loop
//  14.  Public API: run_casscf / run_rasscf

#include "post_hf/casscf.h"
#include "post_hf/integrals.h"
#include "integrals/os.h"
#include "io/logging.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace
{

// ─────────────────────────────────────────────────────────────────────────────
// Module 1: Slater determinant strings and matrix elements
// ─────────────────────────────────────────────────────────────────────────────

using CIString = uint64_t;

std::vector<CIString> generate_strings(int n_orb, int n_occ)
{
    std::vector<CIString> result;
    if (n_occ == 0) { result.push_back(0); return result; }
    if (n_occ > n_orb) return result;

    CIString v = (CIString(1) << n_occ) - 1;
    const CIString limit = CIString(1) << n_orb;
    while (v < limit)
    {
        result.push_back(v);
        CIString c = v & (-v);
        CIString r = v + c;
        v = (((r ^ v) >> 2) / c) | r;
    }
    return result;
}

inline int parity_between(CIString s, int lo, int hi)
{
    if (lo + 1 >= hi) return 1;
    CIString mask = ((CIString(1) << (hi - lo - 1)) - 1) << (lo + 1);
    return (std::popcount(s & mask) % 2 == 0) ? 1 : -1;
}

inline double g_act(const std::vector<double>& ga, int p, int q, int r, int s, int na)
{
    return ga[((p * na + q) * na + r) * na + s];
}

double slater_condon_element(
    CIString bra_a, CIString bra_b,
    CIString ket_a, CIString ket_b,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act)
{
    const int n_diff_a = std::popcount(bra_a ^ ket_a) / 2;
    const int n_diff_b = std::popcount(bra_b ^ ket_b) / 2;
    const int n_diff   = n_diff_a + n_diff_b;

    if (n_diff > 2) return 0.0;

    auto get_excitation = [](CIString bra, CIString ket) {
        std::vector<int> ann, cre;
        CIString d = bra ^ ket;
        while (d)
        {
            int k = std::countr_zero(d);
            if (bra & (CIString(1) << k)) ann.push_back(k);
            else                           cre.push_back(k);
            d &= d - 1;
        }
        return std::make_pair(ann, cre);
    };

    auto single_parity = [](CIString s, int p, int q) {
        return parity_between(s, std::min(p, q), std::max(p, q));
    };

    // Diagonal
    if (n_diff == 0)
    {
        double val = 0.0;
        for (int k = 0; k < n_act; ++k)
        {
            if (ket_a & (CIString(1) << k)) val += h_eff(k, k);
            if (ket_b & (CIString(1) << k)) val += h_eff(k, k);
        }
        for (int p = 0; p < n_act; ++p)
        {
            const bool pa = ket_a & (CIString(1) << p);
            const bool pb = ket_b & (CIString(1) << p);
            if (!pa && !pb) continue;
            for (int q = 0; q < n_act; ++q)
            {
                const bool qa = ket_a & (CIString(1) << q);
                const bool qb = ket_b & (CIString(1) << q);
                if (pa && qa && p < q)
                    val += g_act(ga, p, p, q, q, n_act) - g_act(ga, p, q, q, p, n_act);
                if (pb && qb && p < q)
                    val += g_act(ga, p, p, q, q, n_act) - g_act(ga, p, q, q, p, n_act);
                if (pa && qb)
                    val += g_act(ga, p, p, q, q, n_act);
            }
        }
        return val;
    }

    // Single alpha excitation
    if (n_diff_a == 1 && n_diff_b == 0)
    {
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        int p = ann_a[0], q = cre_a[0];
        int sgn = single_parity(ket_a, p, q);
        double val = h_eff(p, q) * sgn;
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_a & (CIString(1) << r)) || r == p) continue;
            val += sgn * (g_act(ga, p, q, r, r, n_act) - g_act(ga, p, r, r, q, n_act));
        }
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_b & (CIString(1) << r))) continue;
            val += sgn * g_act(ga, p, q, r, r, n_act);
        }
        return val;
    }

    // Single beta excitation
    if (n_diff_a == 0 && n_diff_b == 1)
    {
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        int p = ann_b[0], q = cre_b[0];
        int sgn = single_parity(ket_b, p, q);
        double val = h_eff(p, q) * sgn;
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_b & (CIString(1) << r)) || r == p) continue;
            val += sgn * (g_act(ga, p, q, r, r, n_act) - g_act(ga, p, r, r, q, n_act));
        }
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_a & (CIString(1) << r))) continue;
            val += sgn * g_act(ga, p, q, r, r, n_act);
        }
        return val;
    }

    // Double alpha excitation
    if (n_diff_a == 2 && n_diff_b == 0)
    {
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        int p1 = ann_a[0], p2 = ann_a[1]; if (p1 > p2) std::swap(p1, p2);
        int q1 = cre_a[0], q2 = cre_a[1]; if (q1 > q2) std::swap(q1, q2);
        CIString inter   = (ket_a ^ (CIString(1) << p1)) ^ (CIString(1) << p2);
        CIString after_p1 = ket_a ^ (CIString(1) << p1);
        int n1 = std::popcount(ket_a    & ((CIString(1) << p1) - 1));
        int n2 = std::popcount(after_p1 & ((CIString(1) << p2) - 1));
        int n3 = std::popcount(inter    & ((CIString(1) << q1) - 1));
        CIString after_q1 = inter | (CIString(1) << q1);
        int n4 = std::popcount(after_q1 & ((CIString(1) << q2) - 1));
        int sgn = ((n1 + n2 + n3 + n4) % 2 == 0) ? 1 : -1;
        return sgn * (g_act(ga, p1, q1, p2, q2, n_act) - g_act(ga, p1, q2, p2, q1, n_act));
    }

    // Double beta excitation
    if (n_diff_a == 0 && n_diff_b == 2)
    {
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        int p1 = ann_b[0], p2 = ann_b[1]; if (p1 > p2) std::swap(p1, p2);
        int q1 = cre_b[0], q2 = cre_b[1]; if (q1 > q2) std::swap(q1, q2);
        CIString inter    = (ket_b ^ (CIString(1) << p1)) ^ (CIString(1) << p2);
        CIString after_p1 = ket_b ^ (CIString(1) << p1);
        int n1 = std::popcount(ket_b    & ((CIString(1) << p1) - 1));
        int n2 = std::popcount(after_p1 & ((CIString(1) << p2) - 1));
        int n3 = std::popcount(inter    & ((CIString(1) << q1) - 1));
        CIString after_q1 = inter | (CIString(1) << q1);
        int n4 = std::popcount(after_q1 & ((CIString(1) << q2) - 1));
        int sgn = ((n1 + n2 + n3 + n4) % 2 == 0) ? 1 : -1;
        return sgn * (g_act(ga, p1, q1, p2, q2, n_act) - g_act(ga, p1, q2, p2, q1, n_act));
    }

    // One alpha + one beta excitation
    if (n_diff_a == 1 && n_diff_b == 1)
    {
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        int pa = ann_a[0], qa = cre_a[0];
        int pb = ann_b[0], qb = cre_b[0];
        return single_parity(ket_a, pa, qa)
             * single_parity(ket_b, pb, qb)
             * g_act(ga, pa, qa, pb, qb, n_act);
    }

    return 0.0;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 2: CI solver — direct diagonalisation + Davidson
// ─────────────────────────────────────────────────────────────────────────────

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
davidson(const Eigen::MatrixXd& H, int nroots, double tol = 1e-10, int max_iter = 1000)
{
    const int n = static_cast<int>(H.rows());
    const Eigen::VectorXd diag = H.diagonal();

    Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
    std::sort(idx.data(), idx.data() + n, [&](int a, int b){ return diag(a) < diag(b); });

    int nb_init = std::min(nroots * 4, n);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, nb_init);
    for (int k = 0; k < nb_init; ++k)
        V(idx(k % n), k) = 1.0;
    {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(V);
        V = qr.householderQ() * Eigen::MatrixXd::Identity(n, nb_init);
    }

    Eigen::VectorXd theta(nroots);
    Eigen::MatrixXd Y(n, nroots);

    for (int it = 0; it < max_iter; ++it)
    {
        int m = static_cast<int>(V.cols());
        Eigen::MatrixXd AV = H * V;
        Eigen::MatrixXd M  = V.transpose() * AV;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M);
        theta = eig.eigenvalues().head(nroots);
        Eigen::MatrixXd Yvec = eig.eigenvectors().leftCols(nroots);
        Y = V * Yvec;

        double max_res = 0.0;
        Eigen::MatrixXd new_vecs(n, nroots);
        int n_new = 0;
        for (int k = 0; k < nroots; ++k)
        {
            Eigen::VectorXd r = AV * Yvec.col(k) - theta(k) * Y.col(k);
            double rn = r.norm();
            max_res = std::max(max_res, rn);
            if (rn > tol)
            {
                for (int i = 0; i < n; ++i)
                {
                    double d = theta(k) - diag(i);
                    r(i) /= (std::abs(d) > 1e-14) ? d : -1e-14;
                }
                new_vecs.col(n_new++) = r;
            }
        }
        if (max_res < tol) break;

        const int max_sub = std::min(n, std::max(8 * nroots, m + n_new));
        if (m + n_new > max_sub)
            V = Y;
        else
        {
            V.conservativeResize(Eigen::NoChange, m + n_new);
            V.rightCols(n_new) = new_vecs.leftCols(n_new);
        }

        for (int k = 0; k < static_cast<int>(V.cols()); ++k)
        {
            for (int j = 0; j < k; ++j)
                V.col(k) -= V.col(j).dot(V.col(k)) * V.col(j);
            double nm = V.col(k).norm();
            if (nm < 1e-14) { V.col(k) = V.col(V.cols()-1); V.conservativeResize(Eigen::NoChange, V.cols()-1); --k; }
            else V.col(k) /= nm;
        }
        if (V.cols() == 0) break;
    }
    return {theta, Y};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
solve_ci(const Eigen::MatrixXd& H, int nroots, double tol = 1e-10)
{
    const int dim = static_cast<int>(H.rows());
    if (dim <= 500)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
        int nr = std::min(nroots, dim);
        return {eig.eigenvalues().head(nr), eig.eigenvectors().leftCols(nr)};
    }
    return davidson(H, nroots, tol, 1000);
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 3: Symmetry, RAS, CI string / Hamiltonian builder
// ─────────────────────────────────────────────────────────────────────────────

struct RASParams
{
    int nras1 = 0, nras2 = 0, nras3 = 0;
    int max_holes = 100, max_elec = 100;
    bool active = false;
};

std::vector<int> map_mo_irreps(const std::vector<std::string>& mo_sym,
                               const std::vector<std::string>& names)
{
    std::vector<int> idx(mo_sym.size(), -1);
    for (std::size_t i = 0; i < mo_sym.size(); ++i)
        for (std::size_t g = 0; g < names.size(); ++g)
            if (mo_sym[i] == names[g]) { idx[i] = static_cast<int>(g); break; }
    return idx;
}

int det_symmetry(CIString alpha, CIString beta, const std::vector<int>& irr_act)
{
    int sym = 0;
    for (int t = 0; t < static_cast<int>(irr_act.size()); ++t)
    {
        if (irr_act[t] < 0) continue;
        if (alpha & (CIString(1) << t)) sym ^= irr_act[t];
        if (beta  & (CIString(1) << t)) sym ^= irr_act[t];
    }
    return sym;
}

int resolve_target_irrep(const std::string& s, const std::vector<std::string>& names)
{
    if (s.empty()) return 0;
    for (std::size_t g = 0; g < names.size(); ++g)
        if (s == names[g]) return static_cast<int>(g);
    return 0;
}

// Generate alpha/beta strings with optional RAS and symmetry constraints.
void build_ci_strings(
    int n_act, int n_alpha, int n_beta,
    const RASParams& ras,
    const std::vector<int>& irr_act,
    bool use_sym, int target_irr,
    std::vector<CIString>& a_strs,
    std::vector<CIString>& b_strs)
{
    auto filter = [&](int n_occ) {
        auto all = generate_strings(n_act, n_occ);
        if (!ras.active) return all;
        const CIString m1 = (ras.nras1 > 0) ? ((CIString(1) << ras.nras1) - 1) : 0;
        const CIString m3 = (ras.nras3 > 0) ?
            (((CIString(1) << ras.nras3) - 1) << (ras.nras1 + ras.nras2)) : 0;
        std::vector<CIString> r;
        for (CIString s : all)
        {
            int holes = ras.nras1 - std::popcount(s & m1);
            int elec  = std::popcount(s & m3);
            if (holes <= ras.max_holes && elec <= ras.max_elec) r.push_back(s);
        }
        return r;
    };
    a_strs = filter(n_alpha);
    b_strs = filter(n_beta);
    // pair-level symmetry screening happens inside build_ci_hamiltonian_with_dets
}

// Build H_CI and return the (ia,ib) determinant list.
Eigen::MatrixXd build_ci_hamiltonian_with_dets(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act,
    const std::vector<int>& irr_act,
    bool use_sym, int target_irr,
    std::vector<std::pair<int,int>>& dets_out)
{
    const int na = static_cast<int>(a_strs.size());
    const int nb_s = static_cast<int>(b_strs.size());
    const bool do_sym = use_sym && !irr_act.empty();

    dets_out.clear();
    dets_out.reserve(na * nb_s);
    for (int ia = 0; ia < na; ++ia)
        for (int ib = 0; ib < nb_s; ++ib)
            if (!do_sym || det_symmetry(a_strs[ia], b_strs[ib], irr_act) == target_irr)
                dets_out.push_back({ia, ib});

    const int dim = static_cast<int>(dets_out.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    for (int I = 0; I < dim; ++I)
    {
        auto [ia, ib] = dets_out[I];
        for (int J = I; J < dim; ++J)
        {
            auto [ja, jb] = dets_out[J];
            double v = slater_condon_element(
                a_strs[ia], b_strs[ib],
                a_strs[ja], b_strs[jb],
                h_eff, ga, n_act);
            H(I, J) = H(J, I) = v;
        }
    }
    return H;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 4: 1-RDM and 2-RDM
// ─────────────────────────────────────────────────────────────────────────────

struct FermionOpResult { CIString det = 0; double phase = 0.0; bool valid = false; };

inline FermionOpResult apply_annihilation(CIString det, int orb)
{
    const CIString bit = CIString(1) << orb;
    if (!(det & bit)) return {};
    return {det ^ bit, (std::popcount(det & (bit - 1)) % 2 == 0) ? 1.0 : -1.0, true};
}

inline FermionOpResult apply_creation(CIString det, int orb)
{
    const CIString bit = CIString(1) << orb;
    if (det & bit) return {};
    return {det | bit, (std::popcount(det & (bit - 1)) % 2 == 0) ? 1.0 : -1.0, true};
}

std::vector<CIString> build_spin_dets(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    std::vector<CIString> sd; sd.reserve(dets.size());
    for (auto [ia, ib] : dets)
        sd.push_back(a_strs[ia] | (b_strs[ib] << n_act));
    return sd;
}

std::unordered_map<CIString, int> build_det_lookup(const std::vector<CIString>& sd)
{
    std::unordered_map<CIString, int> lut; lut.reserve(sd.size());
    for (int I = 0; I < static_cast<int>(sd.size()); ++I) lut.emplace(sd[I], I);
    return lut;
}

// State-averaged 1-RDM  γ[p,q] = Σ_σ <a†_{pσ} a_{qσ}>
Eigen::MatrixXd compute_1rdm(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    const int dim = static_cast<int>(ci_vecs.rows());
    const int nr  = static_cast<int>(ci_vecs.cols());
    const auto sd  = build_spin_dets(a_strs, b_strs, dets, n_act);
    const auto lut = build_det_lookup(sd);
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(n_act, n_act);

    for (int root = 0; root < nr; ++root)
    {
        const double w = weights(root);
        if (w == 0.0) continue;
        const auto& cv = ci_vecs.col(root);
        for (int J = 0; J < dim; ++J)
        {
            const double cJ = cv(J);
            if (std::abs(cJ) < 1e-15) continue;
            const CIString ket = sd[J];
            for (int q_so = 0; q_so < 2 * n_act; ++q_so)
            {
                auto ann = apply_annihilation(ket, q_so);
                if (!ann.valid) continue;
                const int spin_off = (q_so >= n_act) ? n_act : 0;
                const int q = q_so - spin_off;
                for (int p = 0; p < n_act; ++p)
                {
                    auto cre = apply_creation(ann.det, spin_off + p);
                    if (!cre.valid) continue;
                    auto it = lut.find(cre.det);
                    if (it == lut.end()) continue;
                    gamma(p, q) += w * ann.phase * cre.phase * cv(it->second) * cJ;
                }
            }
        }
    }
    return gamma;
}

// State-averaged 2-RDM  Γ[p,q,r,s] = Σ_{σσ'} <a†_{pσ}a†_{rσ'}a_{sσ'}a_{qσ}>
std::vector<double> compute_2rdm(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    const int dim = static_cast<int>(ci_vecs.rows());
    const int nr  = static_cast<int>(ci_vecs.cols());
    const auto sd  = build_spin_dets(a_strs, b_strs, dets, n_act);
    const auto lut = build_det_lookup(sd);
    const int na4 = n_act * n_act * n_act * n_act;
    std::vector<double> Gamma(na4, 0.0);

    auto idx4 = [&](int p, int q, int r, int s){ return ((p*n_act+q)*n_act+r)*n_act+s; };

    for (int root = 0; root < nr; ++root)
    {
        const double w = weights(root);
        if (w == 0.0) continue;
        const auto& cv = ci_vecs.col(root);
        for (int J = 0; J < dim; ++J)
        {
            const double cJ = cv(J);
            if (std::abs(cJ) < 1e-15) continue;
            const CIString ket = sd[J];
            for (int q_so = 0; q_so < 2*n_act; ++q_so)
            {
                auto ann_q = apply_annihilation(ket, q_so);
                if (!ann_q.valid) continue;
                const int qoff = (q_so >= n_act) ? n_act : 0;
                const int q = q_so - qoff;
                for (int s_so = 0; s_so < 2*n_act; ++s_so)
                {
                    auto ann_s = apply_annihilation(ann_q.det, s_so);
                    if (!ann_s.valid) continue;
                    const int soff = (s_so >= n_act) ? n_act : 0;
                    const int s = s_so - soff;
                    for (int r = 0; r < n_act; ++r)
                    {
                        auto cre_r = apply_creation(ann_s.det, soff + r);
                        if (!cre_r.valid) continue;
                        for (int p = 0; p < n_act; ++p)
                        {
                            auto cre_p = apply_creation(cre_r.det, qoff + p);
                            if (!cre_p.valid) continue;
                            auto it = lut.find(cre_p.det);
                            if (it == lut.end()) continue;
                            double ph = ann_q.phase * ann_s.phase * cre_r.phase * cre_p.phase;
                            Gamma[idx4(p,q,r,s)] += w * ph * cv(it->second) * cJ;
                        }
                    }
                }
            }
        }
    }
    return Gamma;
}


// Bilinear 2-RDM: Σ_r w_r <bra_r|a†_p a†_r a_s a_q|ket_r>
// Used to compute Γ¹ = bilinear(c¹,c⁰) + bilinear(c⁰,c¹).
std::vector<double> compute_2rdm_bilinear(
    const Eigen::MatrixXd& bra_vecs,
    const Eigen::MatrixXd& ket_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    const int dim = static_cast<int>(ket_vecs.rows());
    const int nr  = static_cast<int>(ket_vecs.cols());
    const auto sd  = build_spin_dets(a_strs, b_strs, dets, n_act);
    const auto lut = build_det_lookup(sd);
    const int na4 = n_act * n_act * n_act * n_act;
    std::vector<double> Gamma(na4, 0.0);

    auto idx4 = [&](int p, int q, int r, int s){ return ((p*n_act+q)*n_act+r)*n_act+s; };

    for (int root = 0; root < nr; ++root)
    {
        const double w = weights(root);
        if (w == 0.0) continue;
        const auto bra = bra_vecs.col(root);
        const auto ket = ket_vecs.col(root);
        for (int J = 0; J < dim; ++J)
        {
            const double ketJ = ket(J);
            if (std::abs(ketJ) < 1e-15) continue;
            const CIString det_J = sd[J];
            for (int q_so = 0; q_so < 2*n_act; ++q_so)
            {
                auto ann_q = apply_annihilation(det_J, q_so);
                if (!ann_q.valid) continue;
                const int qoff = (q_so >= n_act) ? n_act : 0;
                const int q = q_so - qoff;
                for (int s_so = 0; s_so < 2*n_act; ++s_so)
                {
                    auto ann_s = apply_annihilation(ann_q.det, s_so);
                    if (!ann_s.valid) continue;
                    const int soff = (s_so >= n_act) ? n_act : 0;
                    const int s = s_so - soff;
                    for (int r = 0; r < n_act; ++r)
                    {
                        auto cre_r = apply_creation(ann_s.det, soff + r);
                        if (!cre_r.valid) continue;
                        for (int p = 0; p < n_act; ++p)
                        {
                            auto cre_p = apply_creation(cre_r.det, qoff + p);
                            if (!cre_p.valid) continue;
                            auto it = lut.find(cre_p.det);
                            if (it == lut.end()) continue;
                            double ph = ann_q.phase * ann_s.phase * cre_r.phase * cre_p.phase;
                            Gamma[idx4(p,q,r,s)] += w * ph * bra(it->second) * ketJ;
                        }
                    }
                }
            }
        }
    }
    return Gamma;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 5: Inactive / Active Fock, core energy
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd build_inactive_fock_mo(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& H_core,
    const std::vector<double>& eri,
    int n_core, int nbasis)
{
    if (n_core == 0) return C.transpose() * H_core * C;
    Eigen::MatrixXd D = 2.0 * C.leftCols(n_core) * C.leftCols(n_core).transpose();
    return C.transpose() * (H_core + HartreeFock::ObaraSaika::_compute_fock_rhf(eri, D, nbasis)) * C;
}

Eigen::MatrixXd build_active_fock_mo(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& gamma,
    const std::vector<double>& eri,
    int n_core, int n_act, int nbasis)
{
    if (n_act == 0) return Eigen::MatrixXd::Zero(nbasis, nbasis);
    Eigen::MatrixXd C_act = C.middleCols(n_core, n_act);
    Eigen::MatrixXd D_act = C_act * gamma * C_act.transpose();
    return C.transpose() * HartreeFock::ObaraSaika::_compute_fock_rhf(eri, D_act, nbasis) * C;
}

// E_core = Σ_{i} (h_mo[i,i] + F^I_MO[i,i])
double compute_core_energy(const Eigen::MatrixXd& h_mo,
                           const Eigen::MatrixXd& F_I_mo,
                           int n_core)
{
    double e = 0.0;
    for (int i = 0; i < n_core; ++i) e += h_mo(i,i) + F_I_mo(i,i);
    return e;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 6: Q matrix  Q[p,t] = Σ_{uvw} Γ[t,u,v,w] (p u|v w)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd compute_Q_matrix(
    const std::vector<double>& eri,
    const Eigen::MatrixXd& C,
    const std::vector<double>& Gamma,
    int n_core, int n_act, int nbasis)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nbasis, n_act);
    if (n_act == 0) return Q;
    Eigen::MatrixXd C_act = C.middleCols(n_core, n_act);
    std::vector<double> T = HartreeFock::Correlation::transform_eri(eri, nbasis, C, C_act, C_act, C_act);
    const int na = n_act;
    for (int p = 0; p < nbasis; ++p)
    for (int t = 0; t < na; ++t)
    {
        double q_pt = 0.0;
        for (int u = 0; u < na; ++u)
        for (int v = 0; v < na; ++v)
        for (int w = 0; w < na; ++w)
            q_pt += Gamma[((t*na+u)*na+v)*na+w] * T[((p*na+u)*na+v)*na+w];
        Q(p, t) = q_pt;
    }
    return Q;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 7: Orbital gradient  G_{pq} = 2(F_gen[p,q] - F_gen[q,p])
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd compute_orbital_gradient(
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& gamma,
    int n_core, int n_act, int n_virt,
    const std::vector<int>& mo_irreps,
    bool use_sym)
{
    const int nb = n_core + n_act + n_virt;
    const Eigen::MatrixXd F_sum = F_I_mo + F_A_mo;

    // Generalised Fock F_gen[p,q]
    Eigen::MatrixXd F_gen = Eigen::MatrixXd::Zero(nb, nb);
    for (int p = 0; p < nb; ++p)
    {
        for (int i = 0; i < n_core; ++i)
            F_gen(p, i) = 2.0 * F_sum(p, i);
        for (int t = 0; t < n_act; ++t)
        {
            double val = 0.0;
            for (int u = 0; u < n_act; ++u)
                val += gamma(t, u) * F_I_mo(p, n_core + u);
            val += Q(p, t);
            F_gen(p, n_core + t) = val;
        }
        // Virtual columns remain zero
    }

    Eigen::MatrixXd g = 2.0 * (F_gen - F_gen.transpose());
    // Zero redundant (same-class) blocks
    g.topLeftCorner(n_core, n_core).setZero();
    g.block(n_core, n_core, n_act, n_act).setZero();
    g.bottomRightCorner(n_virt, n_virt).setZero();

    if (use_sym && !mo_irreps.empty())
        for (int p = 0; p < nb; ++p)
        for (int q = 0; q < nb; ++q)
        {
            if (p == q) continue;
            const int ip = (p < (int)mo_irreps.size()) ? mo_irreps[p] : -1;
            const int iq = (q < (int)mo_irreps.size()) ? mo_irreps[q] : -1;
            if (ip >= 0 && iq >= 0 && ip != iq) g(p, q) = 0.0;
        }

    return g;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 8: Orbital Hessian action H·R — diagonal approximation
//
// (H·R)_{pq} ≈ D_{pq} R_{pq}
// D_{pq} = 2 ( (F^I + F^A)[p,p] − (F^I + F^A)[q,q] )  (non-redundant pairs)
// ─────────────────────────────────────────────────────────────────────────────

// Returns the signed diagonal Hessian element for pair (p,q).
// Convention: H_{pq,pq} = 2*(F[q,q] - F[p,p]).
// For non-redundant pairs stored as (p<q) with p being the lower-energy
// (core/active) orbital and q the higher-energy (active/virtual) one, this
// is always positive.  The signed form also gives the correct antisymmetric
// gradient update in the full (p,q) loop of the FEP1 step.
inline double hess_diag(const Eigen::MatrixXd& F_sum, int p, int q)
{
    return 2.0 * (F_sum(q, q) - F_sum(p, p));
}

// Compute H·R (same shape as R) using diagonal approximation.
Eigen::MatrixXd hessian_action(
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core, int n_act, int n_virt)
{
    const int nb = n_core + n_act + n_virt;
    const Eigen::MatrixXd F_sum = F_I_mo + F_A_mo;
    Eigen::MatrixXd HR = Eigen::MatrixXd::Zero(nb, nb);

    auto cls = [&](int k) { return (k < n_core) ? 0 : (k < n_core + n_act) ? 1 : 2; };

    for (int p = 0; p < nb; ++p)
    for (int q = 0; q < nb; ++q)
    {
        if (cls(p) == cls(q)) continue;
        HR(p, q) = hess_diag(F_sum, p, q) * R(p, q);
    }
    return HR;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 9: CI response  c¹ ≈ −[diag(H_CI − E)]⁻¹ H^R c⁰
//
// H^R is the first-order change in the CI Hamiltonian due to orbital rotation R.
// Here we use only the one-body part: δh_eff = [κ^T F^I + F^I κ]_{act,act}
// and compute the σ-vector H^{δh} c⁰ by explicit matrix-vector product with
// the one-body CI Hamiltonian evaluated at δh_eff.
// ─────────────────────────────────────────────────────────────────────────────

// Compute the one-body sigma vector using the same one-body index convention as
// slater_condon_element and build_ci_hamiltonian_with_dets:
//   dh(p,q) multiplies the operator that annihilates orbital p and creates q.
// Using the same convention keeps the CI response aligned with the CI
// Hamiltonian builder already used elsewhere in this file.
Eigen::VectorXd ci_sigma_1body(
    const Eigen::MatrixXd& dh,          // n_act × n_act
    const Eigen::VectorXd& c,           // CI vector
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    const int dim = static_cast<int>(c.size());
    const auto sd  = build_spin_dets(a_strs, b_strs, dets, n_act);
    const auto lut = build_det_lookup(sd);
    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(dim);

    for (int J = 0; J < dim; ++J)
    {
        const double cJ = c(J);
        if (std::abs(cJ) < 1e-15) continue;
        const CIString ket = sd[J];
        for (int p_so = 0; p_so < 2 * n_act; ++p_so)
        {
            auto ann = apply_annihilation(ket, p_so);
            if (!ann.valid) continue;
            const int spin_off = (p_so >= n_act) ? n_act : 0;
            const int p = p_so - spin_off;
            for (int q = 0; q < n_act; ++q)
            {
                if (std::abs(dh(p, q)) < 1e-18) continue;
                auto cre = apply_creation(ann.det, spin_off + q);
                if (!cre.valid) continue;
                auto it = lut.find(cre.det);
                if (it == lut.end()) continue;
                sigma(it->second) += dh(p, q) * ann.phase * cre.phase * cJ;
            }
        }
    }
    return sigma;
}

// First-order change in h_eff due to rotation κ (nb×nb antisymmetric).
// δh_eff_{tu} = Σ_p [κ_{tp} F^I_{pu} + F^I_{tp} κ_{pu}] (active block, MO basis)
// = [κ F^I_MO + F^I_MO κ^T]_{act,act} = [κ F^I_MO - F^I_MO κ]_{act,act} (using κ^T = -κ)
Eigen::MatrixXd delta_h_eff(
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& F_I_mo,
    int n_core, int n_act)
{
    const int nb = static_cast<int>(F_I_mo.rows());
    // Full MO commutator [κ, F^I]_{act,act}
    Eigen::MatrixXd comm = kappa * F_I_mo - F_I_mo * kappa;
    return comm.block(n_core, n_core, n_act, n_act);
}

// Compute CI response c¹ ≈ -[diag(H_CI - E)]⁻¹ H^κ c⁰
// using a single preconditioning step (approximate Davidson step).
Eigen::VectorXd ci_response(
    const Eigen::VectorXd& c0,
    double E0,
    const Eigen::VectorXd& H_diag,    // diagonal of H_CI
    const Eigen::VectorXd& sigma,     // H^κ c⁰ (driving vector)
    double precond_floor = 1e-4)
{
    const int dim = static_cast<int>(c0.size());
    Eigen::VectorXd c1 = Eigen::VectorXd::Zero(dim);
    for (int I = 0; I < dim; ++I)
    {
        double denom = E0 - H_diag(I);
        if (std::abs(denom) < precond_floor)
            denom = (denom >= 0.0) ? precond_floor : -precond_floor;
        c1(I) = -sigma(I) / denom;
    }
    // Project out the ground state to maintain orthogonality
    c1 -= c0.dot(c1) * c0;
    return c1;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 10: Dressed gradient  G̃ = G + H·R  (FEP1 update)
//
// Within each microiteration, after computing orbital step R and CI response:
//   G̃_{pq} = G_{pq} + (H·R)_{pq} + CI contribution
// The diagonal Hessian approximation gives (H·R)_{pq} = D_{pq} R_{pq}.
// The CI contribution involves Γ¹ from the CI response.
// ─────────────────────────────────────────────────────────────────────────────

// Update gradient via FEP1: G ← G + H·R  (diagonal Hessian approximation)
Eigen::MatrixXd fep1_gradient_update(
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core, int n_act, int n_virt)
{
    const int nb = n_core + n_act + n_virt;
    const Eigen::MatrixXd F_sum = F_I_mo + F_A_mo;
    Eigen::MatrixXd G_new = G;

    auto cls = [&](int k){ return (k < n_core) ? 0 : (k < n_core+n_act) ? 1 : 2; };

    for (int p = 0; p < nb; ++p)
    for (int q = 0; q < nb; ++q)
    {
        if (cls(p) == cls(q) || std::abs(kappa(p, q)) < 1e-18) continue;
        G_new(p, q) += hess_diag(F_sum, p, q) * kappa(p, q);
    }
    return G_new;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 11: Augmented Hessian orbital step (RFO / AH)
//
// Solve  | 0   g^T | |1|   |1|
//         | g    D  | |x| = ε|x|
// for the lowest eigenvalue.  With diagonal D, the optimal step satisfies:
//   x_{pq} = -g_{pq} / (D_{pq} - ε)
// and ε is the lowest root of Σ g²_{pq}/(D_{pq} - ε)² = 1.
// The implementation below also builds a damped Newton candidate and keeps the
// step with the better local quadratic decrease. This makes the update less
// prone to stalling when the AH root is over-conservative.
// ─────────────────────────────────────────────────────────────────────────────

struct RotPair { int p = 0; int q = 0; };

double quadratic_model_delta(const Eigen::VectorXd& g_flat,
                             const Eigen::VectorXd& h_flat,
                             const Eigen::VectorXd& x)
{
    double dE = g_flat.dot(x);
    for (int k = 0; k < x.size(); ++k)
        dE += 0.5 * h_flat(k) * x(k) * x(k);
    return dE;
}

std::vector<RotPair> non_redundant_pairs(int n_core, int n_act, int n_virt)
{
    const int nb = n_core + n_act + n_virt;
    std::vector<RotPair> pairs;
    pairs.reserve(n_core*n_act + n_core*n_virt + n_act*n_virt);
    auto cls = [&](int k){ return (k < n_core) ? 0 : (k < n_core+n_act) ? 1 : 2; };
    for (int p = 0; p < nb; ++p)
    for (int q = p+1; q < nb; ++q)
        if (cls(p) != cls(q)) pairs.push_back({p, q});
    return pairs;
}

// Solve AH and return the antisymmetric step κ, capped to max_rot.
Eigen::MatrixXd augmented_hessian_step(
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core, int n_act, int n_virt,
    double level_shift,
    double max_rot,
    const std::vector<int>& mo_irreps,
    bool use_sym)
{
    const int nb = n_core + n_act + n_virt;
    const Eigen::MatrixXd F_sum = F_I_mo + F_A_mo;
    const auto pairs = non_redundant_pairs(n_core, n_act, n_virt);
    const int npairs = static_cast<int>(pairs.size());

    // No non-redundant pairs (e.g. full-valence active space with no core/virtual)
    if (npairs == 0)
        return Eigen::MatrixXd::Zero(nb, nb);

    // Flatten gradient and diagonal Hessian over non-redundant pairs.
    // h_flat is the physical diagonal Hessian; D_flat adds the LM shift used
    // by the AH solve.
    Eigen::VectorXd g_flat(npairs), h_flat(npairs), D_flat(npairs);
    for (int k = 0; k < npairs; ++k)
    {
        int p = pairs[k].p, q = pairs[k].q;
        if (use_sym && !mo_irreps.empty())
        {
            int ip = (p < (int)mo_irreps.size()) ? mo_irreps[p] : -1;
            int iq = (q < (int)mo_irreps.size()) ? mo_irreps[q] : -1;
            if (ip >= 0 && iq >= 0 && ip != iq)
                { g_flat(k) = 0.0; h_flat(k) = 1.0; D_flat(k) = 1.0; continue; }
        }
        // Use upper triangle: G is antisymmetric, take G[p,q] (p < q)
        g_flat(k) = G(p, q);
        h_flat(k) = hess_diag(F_sum, p, q);
        D_flat(k) = h_flat(k) + level_shift;
        // Guard the AH denominator but keep the physical Hessian sign in h_flat.
        if (std::abs(D_flat(k)) < 1e-4)
            D_flat(k) = (D_flat(k) >= 0.0) ? 1e-4 : -1e-4;
    }

    auto cap_step = [&](Eigen::VectorXd x) {
        if (!x.allFinite()) return x;
        const double max_x = x.cwiseAbs().maxCoeff();
        if (max_x > max_rot) x *= max_rot / max_x;
        return x;
    };

    auto damped_newton = [&](double lm_shift) {
        Eigen::VectorXd x(npairs);
        for (int k = 0; k < npairs; ++k)
        {
            double denom = h_flat(k) + lm_shift;
            if (std::abs(denom) < 1e-18)
                denom = (denom >= 0.0) ? 1e-18 : -1e-18;
            x(k) = -g_flat(k) / denom;
        }
        return x;
    };

    // Solve AH by finding ε via bisection on f(ε) = Σ g²/(D-ε)² - 1 = 0.
    // For the lowest eigenvalue ε < min(D), use the bracket [−∞, min(D) − eps].
    double D_min = D_flat.minCoeff();
    double eps_lo = D_min - 1.0 - g_flat.norm();
    double eps_hi = D_min - 1e-8;

    // f(ε) = Σ_k [g_k / (D_k - ε)]²  — equals 1 at the AH eigenvalue
    auto f = [&](double e) {
        double s = 0.0;
        for (int k = 0; k < npairs; ++k)
        {
            double d = D_flat(k) - e;
            s += (g_flat(k) / d) * (g_flat(k) / d);
        }
        return s - 1.0;
    };

    // Bisection (50 iterations is more than enough for 1e-12 precision)
    double eps_star = 0.0;
    if (f(eps_lo) * f(eps_hi) > 0.0)
    {
        // Gradient small or all D positive and large — use ε = 0 (pure Newton)
        eps_star = 0.0;
    }
    else
    {
        for (int bi = 0; bi < 50; ++bi)
        {
            eps_star = 0.5 * (eps_lo + eps_hi);
            if (f(eps_lo) * f(eps_star) <= 0.0) eps_hi = eps_star;
            else                                eps_lo = eps_star;
            if (eps_hi - eps_lo < 1e-12) break;
        }
    }

    // Extract AH step vector from the secular root.
    Eigen::VectorXd x_ah(npairs);
    for (int k = 0; k < npairs; ++k)
        x_ah(k) = -g_flat(k) / (D_flat(k) - eps_star);

    struct Candidate {
        Eigen::VectorXd x;
        double score = std::numeric_limits<double>::infinity();
    };

    auto score = [&](const Eigen::VectorXd& x) {
        if (!x.allFinite()) return std::numeric_limits<double>::infinity();
        return quadratic_model_delta(g_flat, h_flat, x);
    };

    Candidate best;
    auto consider = [&](Eigen::VectorXd x) {
        x = cap_step(std::move(x));
        const double s = score(x);
        if (s < best.score - 1e-12 ||
            (std::abs(s - best.score) <= 1e-12 &&
             x.norm() > best.x.norm() + 1e-12))
        {
            best.x = std::move(x);
            best.score = s;
        }
    };

    // Try a small trust-region family of Newton candidates before falling
    // back to the AH root. This is more aggressive in flat directions while
    // still respecting the max rotation cap applied by the caller.
    for (double lm_scale : {0.0, 0.25, 1.0, 4.0, 16.0})
        consider(damped_newton(lm_scale * std::max(level_shift, 1e-6)));
    consider(x_ah);

    Eigen::VectorXd x = best.x;
    if (!x.allFinite() || x.cwiseAbs().maxCoeff() < 1e-14)
        x = cap_step(damped_newton(std::max(level_shift, 1e-6)));

    // Build full antisymmetric κ matrix from the step vector
    Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nb, nb);
    for (int k = 0; k < npairs; ++k)
    {
        int p = pairs[k].p, q = pairs[k].q;
        kappa(p, q) =  x(k);
        kappa(q, p) = -x(k);
    }

    // Cap to max_rot
    double max_k = kappa.cwiseAbs().maxCoeff();
    if (max_k > max_rot) kappa *= max_rot / max_k;

    return kappa;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 12: Orbital rotation via Cayley transform + re-orthogonalisation
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd apply_orbital_rotation(
    const Eigen::MatrixXd& C_old,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& S)
{
    const int nb = static_cast<int>(C_old.rows());
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nb, nb);
    // Cayley approximation to exp(kappa): U = (I - κ/2)⁻¹ (I + κ/2)
    Eigen::MatrixXd C_new = C_old * (I - 0.5*kappa).colPivHouseholderQr().solve(I + 0.5*kappa);

    // Löwdin re-orthogonalisation: C_new ← C_new (C_new^T S C_new)^{-1/2}
    Eigen::MatrixXd ovlp = C_new.transpose() * S * C_new;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(ovlp);
    Eigen::VectorXd inv_sqrt = eig.eigenvalues().array().max(1e-12).sqrt().inverse();
    return C_new * (eig.eigenvectors() * inv_sqrt.asDiagonal() * eig.eigenvectors().transpose());
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 13: Macro/micro MCSCF loop
//
// Algorithm (Sun, Yang, Chan 2017):
//   Repeat macroiteration:
//     1. Full evaluation: F^I, transform ERIs, H_CI, CI solve, RDMs, gradient G
//     2. Convergence check
//     3. Repeat microiteration (nmicro times):
//        a. Augmented Hessian step → κ
//        b. First-order h_eff change δh = [κ, F^I]_{act,act}
//        c. CI response c¹_r for each root r
//        d. First-order 2-RDM Γ¹ = Σ_r w_r (|c¹_r><c⁰_r| + |c⁰_r><c¹_r|)
//        e. Q-matrix response Q¹_{pt} = Σ_{uvw} Γ¹_{tuvw} (pu|vw)
//        f. CI gradient G^CI_{pq} = 2(Q¹_{pq} - Q¹_{qp})
//        g. Dressed gradient G̃ ← G̃ + H·κ + G^CI
//        h. Accumulate κ_total
//     4. Apply accumulated rotation C ← C * exp(κ_total)
// ─────────────────────────────────────────────────────────────────────────────

std::expected<void, std::string> run_mcscf_loop(
    HartreeFock::Calculator& calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::string& tag,
    const RASParams& ras)
{
    using HartreeFock::Logger::logging;
    using HartreeFock::LogLevel;

    // ── Validate ──────────────────────────────────────────────────────────────
    if (!calc._info._is_converged)
        return std::unexpected(tag + ": requires a converged RHF reference.");
    if (calc._scf._scf != HartreeFock::SCFType::RHF)
        return std::unexpected(tag + ": only RHF reference supported.");

    const auto& as = calc._active_space;
    if (as.nactele <= 0) return std::unexpected(tag + ": nactele must be > 0.");
    if (as.nactorb <= 0) return std::unexpected(tag + ": nactorb must be > 0.");
    if (as.nactele > 2 * as.nactorb)
        return std::unexpected(tag + ": nactele > 2*nactorb is impossible.");

    const int nbasis = static_cast<int>(calc._shells.nbasis());
    const int n_total_elec =
        static_cast<int>(calc._molecule.atomic_numbers.cast<int>().sum()) - calc._molecule.charge;
    if ((n_total_elec - as.nactele) % 2 != 0)
        return std::unexpected(tag + ": (n_elec - nactele) must be even for RHF-based CASSCF.");

    const int n_core = (n_total_elec - as.nactele) / 2;
    const int n_act  = as.nactorb;
    const int n_virt = nbasis - n_core - n_act;
    if (n_core < 0) return std::unexpected(tag + ": nactele > total electrons.");
    if (n_virt < 0) return std::unexpected(tag + ": n_core + nactorb > nbasis.");
    if (ras.active && ras.nras1 + ras.nras2 + ras.nras3 != n_act)
        return std::unexpected(tag + ": nras1 + nras2 + nras3 must equal nactorb.");

    const int multiplicity = static_cast<int>(calc._molecule.multiplicity);
    const int n_alpha_act  = (as.nactele + (multiplicity - 1)) / 2;
    const int n_beta_act   = as.nactele - n_alpha_act;
    if (n_alpha_act < 0 || n_beta_act < 0 || n_alpha_act > n_act || n_beta_act > n_act)
        return std::unexpected(tag + ": invalid active-space electron count.");

    // CI size check
    auto nchoose = [](int n, int k) -> long long {
        if (k > n || k < 0) return 0; if (k == 0 || k == n) return 1;
        long long r = 1; for (int i = 0; i < k; ++i) r = r * (n-i) / (i+1); return r;
    };
    long long ci_dim_est = nchoose(n_act, n_alpha_act) * nchoose(n_act, n_beta_act);
    if (ci_dim_est > static_cast<long long>(as.ci_max_dim))
        return std::unexpected(std::format("{}: CI dim ({}) exceeds ci_max_dim ({}).",
                                           tag, ci_dim_est, as.ci_max_dim));

    // ── SA weights ────────────────────────────────────────────────────────────
    const int nroots = as.nroots;
    Eigen::VectorXd weights(nroots);
    if ((int)as.weights.size() == nroots)
        for (int k = 0; k < nroots; ++k) weights(k) = as.weights[k];
    else
        weights.setConstant(1.0 / nroots);
    weights /= weights.sum();

    // ── Symmetry ──────────────────────────────────────────────────────────────
    const bool have_sym = !calc._sao_irrep_names.empty()
                       && (int)calc._sao_irrep_names.size() <= 8;
    std::vector<int> irr_act, all_mo_irr;
    if (have_sym && !calc._info._scf.alpha.mo_symmetry.empty())
    {
        all_mo_irr = map_mo_irreps(calc._info._scf.alpha.mo_symmetry, calc._sao_irrep_names);
        irr_act.resize(n_act);
        for (int t = 0; t < n_act; ++t)
            irr_act[t] = (n_core + t < (int)all_mo_irr.size()) ? all_mo_irr[n_core + t] : -1;
    }
    const int  target_irr = resolve_target_irrep(as.target_irrep, calc._sao_irrep_names);
    const bool use_sym    = have_sym && !irr_act.empty();

    // ── Initial MO coefficients ───────────────────────────────────────────────
    Eigen::MatrixXd C = (calc._cas_mo_coefficients.rows() == nbasis &&
                         calc._cas_mo_coefficients.cols() == nbasis)
                       ? calc._cas_mo_coefficients
                       : calc._info._scf.alpha.mo_coefficients;
    if (C.rows() != nbasis || C.cols() != nbasis)
        return std::unexpected(tag + ": MO coefficient matrix has wrong size.");

    // ── ERI ───────────────────────────────────────────────────────────────────
    std::vector<double> eri_local;
    const std::vector<double>& eri = HartreeFock::Correlation::ensure_eri(
        calc, shell_pairs, eri_local, tag + " :");

    // ── CI strings ────────────────────────────────────────────────────────────
    std::vector<CIString> a_strs, b_strs;
    build_ci_strings(n_act, n_alpha_act, n_beta_act, ras,
                     irr_act, use_sym, target_irr, a_strs, b_strs);

    const unsigned int nmicro = std::max(1u, as.mcscf_micro_per_macro);

    logging(LogLevel::Info, tag + " :",
        std::format("Active space: ({:d}e, {:d}o)  n_core={:d}  n_virt={:d}  CI dim ≤ {:d}",
                    as.nactele, n_act, n_core, n_virt, ci_dim_est));
    logging(LogLevel::Info, tag + " :",
        std::format("Algorithm: macro/micro scaffold  nmicro={:d}", nmicro));
    if (use_sym)
        logging(LogLevel::Info, tag + " :",
            std::format("Target irrep: {}",
                as.target_irrep.empty() ? calc._sao_irrep_names[0] : as.target_irrep));
    if (nroots > 1)
        logging(LogLevel::Info, tag + " :",
            std::format("State-averaged over {:d} roots", nroots));
    HartreeFock::Logger::blank();
    HartreeFock::Logger::casscf_header();

    // ── State at the current geometry/orbitals ────────────────────────────────
    struct McscfState
    {
        Eigen::MatrixXd F_I_mo, F_A_mo, gamma, g_orb;
        std::vector<double> Gamma_vec;
        Eigen::MatrixXd H_CI;          // kept for CI response in micro loop
        Eigen::VectorXd H_CI_diag;
        Eigen::VectorXd ci_energies;
        Eigen::MatrixXd ci_vecs;
        std::vector<std::pair<int,int>> dets;
        double E_cas = 0.0, gnorm = 0.0;
    };

    const double max_rot = 0.20;

    auto evaluate = [&](const Eigen::MatrixXd& C_trial) -> std::expected<McscfState, std::string>
    {
        McscfState st;
        st.F_I_mo = build_inactive_fock_mo(C_trial, calc._hcore, eri, n_core, nbasis);
        Eigen::MatrixXd h_eff = st.F_I_mo.block(n_core, n_core, n_act, n_act);
        Eigen::MatrixXd C_act = C_trial.middleCols(n_core, n_act);
        std::vector<double> ga = HartreeFock::Correlation::transform_eri_internal(eri, nbasis, C_act);

        st.H_CI = build_ci_hamiltonian_with_dets(
            a_strs, b_strs, h_eff, ga, n_act,
            irr_act, use_sym, target_irr, st.dets);
        const int ci_dim = static_cast<int>(st.H_CI.rows());
        if (ci_dim == 0)
            return std::unexpected(tag + ": no CI determinants of target symmetry.");
        st.H_CI_diag = st.H_CI.diagonal();

        auto [E, V] = solve_ci(st.H_CI, std::min(nroots, ci_dim));
        const int nr_got = static_cast<int>(E.size());
        if (nr_got < nroots)
            return std::unexpected(
                std::format("{}: CI returned {:d} roots (wanted {:d}).", tag, nr_got, nroots));
        st.ci_energies = E;
        st.ci_vecs     = V;

        st.gamma     = compute_1rdm(V, weights, a_strs, b_strs, st.dets, n_act);
        st.Gamma_vec = compute_2rdm(V, weights, a_strs, b_strs, st.dets, n_act);

        const Eigen::MatrixXd h_mo = C_trial.transpose() * calc._hcore * C_trial;
        const double E_core = compute_core_energy(h_mo, st.F_I_mo, n_core);
        const double E_act  = weights.head(nroots).dot(E.head(nroots));
        st.E_cas = calc._nuclear_repulsion + E_core + E_act;

        st.F_A_mo = build_active_fock_mo(C_trial, st.gamma, eri, n_core, n_act, nbasis);
        Eigen::MatrixXd Q = compute_Q_matrix(eri, C_trial, st.Gamma_vec, n_core, n_act, nbasis);
        st.g_orb = compute_orbital_gradient(
            st.F_I_mo, st.F_A_mo, Q, st.gamma, n_core, n_act, n_virt,
            all_mo_irr, use_sym);
        st.gnorm = st.g_orb.cwiseAbs().maxCoeff();
        return st;
    };

    std::vector<RotPair> opt_pairs;
    opt_pairs.reserve(non_redundant_pairs(n_core, n_act, n_virt).size());
    for (const auto& pair : non_redundant_pairs(n_core, n_act, n_virt))
    {
        if (use_sym && !all_mo_irr.empty())
        {
            const int ip = (pair.p < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[pair.p] : -1;
            const int iq = (pair.q < static_cast<int>(all_mo_irr.size())) ? all_mo_irr[pair.q] : -1;
            if (ip >= 0 && iq >= 0 && ip != iq) continue;
        }
        opt_pairs.push_back(pair);
    }

    auto pack_pairs = [&](const Eigen::MatrixXd& M) {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(static_cast<int>(opt_pairs.size()));
        for (int k = 0; k < static_cast<int>(opt_pairs.size()); ++k)
            v(k) = M(opt_pairs[k].p, opt_pairs[k].q);
        return v;
    };

    auto unpack_pairs = [&](const Eigen::VectorXd& v) {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nbasis, nbasis);
        for (int k = 0; k < static_cast<int>(opt_pairs.size()); ++k)
        {
            M(opt_pairs[k].p, opt_pairs[k].q) =  v(k);
            M(opt_pairs[k].q, opt_pairs[k].p) = -v(k);
        }
        return M;
    };

    auto build_numeric_newton_step = [&](const McscfState& st_cur,
                                         const Eigen::MatrixXd& C_cur,
                                         double lm_shift) {
        const int npairs = static_cast<int>(opt_pairs.size());
        Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(nbasis, nbasis);
        if (npairs == 0 || npairs > 96) return zero;

        const Eigen::VectorXd g0 = pack_pairs(st_cur.g_orb);
        if (g0.cwiseAbs().maxCoeff() < 1e-10) return zero;

        constexpr double fd_step = 1e-3;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(npairs, npairs);
        for (int k = 0; k < npairs; ++k)
        {
            Eigen::VectorXd ek = Eigen::VectorXd::Zero(npairs);
            ek(k) = fd_step;
            Eigen::MatrixXd kappa_fd = unpack_pairs(ek);
            Eigen::MatrixXd C_fd = apply_orbital_rotation(C_cur, kappa_fd, calc._overlap);
            auto fd_res = evaluate(C_fd);
            if (!fd_res) return zero;
            H.col(k) = (pack_pairs(fd_res->g_orb) - g0) / fd_step;
        }

        H = 0.5 * (H + H.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
        if (eig.info() != Eigen::Success) return zero;

        Eigen::VectorXd evals = eig.eigenvalues();
        const double floor = std::max(1e-4, lm_shift);
        for (int i = 0; i < evals.size(); ++i)
            evals(i) = std::max(evals(i), floor);

        Eigen::VectorXd step = -eig.eigenvectors()
                             * evals.cwiseInverse().asDiagonal()
                             * eig.eigenvectors().transpose()
                             * g0;

        Eigen::MatrixXd kappa = unpack_pairs(step);
        const double max_elem = kappa.cwiseAbs().maxCoeff();
        if (max_elem > 0.20)
            kappa *= 0.20 / max_elem;

        const double trust_radius = 0.80;
        const double frob = kappa.norm();
        if (frob > trust_radius)
            kappa *= trust_radius / frob;

        return kappa;
    };

    auto build_gradient_fallback_step = [&](const Eigen::MatrixXd& G_trial) {
        Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);
        for (const auto& pair : opt_pairs)
        {
            const double step = -G_trial(pair.p, pair.q);
            kappa(pair.p, pair.q) = step;
            kappa(pair.q, pair.p) = -step;
        }

        const double max_elem = kappa.cwiseAbs().maxCoeff();
        if (max_elem > 0.20)
            kappa *= 0.20 / max_elem;

        const double trust_radius = 0.80;
        const double frob = kappa.norm();
        if (frob > trust_radius)
            kappa *= trust_radius / frob;

        return kappa;
    };

    // ── Main MCSCF loop ───────────────────────────────────────────────────────
    double E_prev = 0.0;
    bool converged = false;
    double level_shift = 0.2;   // initial Levenberg-Marquardt shift

    for (unsigned int macro = 1; macro <= as.mcscf_max_iter; ++macro)
    {
        // ── (1) Full evaluation ───────────────────────────────────────────────
        auto res = evaluate(C);
        if (!res) return std::unexpected(res.error());
        auto st_current = std::move(*res);

        // ── (2) Convergence ───────────────────────────────────────────────────
        const bool e_conv  = macro > 1 && std::abs(st_current.E_cas - E_prev) < as.tol_mcscf_energy;
        const bool g_conv  = st_current.gnorm < as.tol_mcscf_grad;
        // If the orbital gradient is exactly zero (no non-redundant pairs) the
        // system is already a pure FCI — converge immediately without requiring
        // a second macroiteration for the energy-change criterion.
        const bool no_orb_rot = (st_current.gnorm == 0.0);
        if ((e_conv && g_conv) || (g_conv && no_orb_rot)) { converged = true; break; }

        // ── (3) Microiterations ───────────────────────────────────────────────
        Eigen::MatrixXd G_curr  = st_current.g_orb;     // working gradient (updated by FEP1)
        Eigen::MatrixXd kappa_total = Eigen::MatrixXd::Zero(nbasis, nbasis);
        Eigen::MatrixXd kappa_first = Eigen::MatrixXd::Zero(nbasis, nbasis);
        const Eigen::MatrixXd kappa_newton = build_numeric_newton_step(st_current, C, level_shift);

        for (unsigned int micro = 0; micro < nmicro; ++micro)
        {
            // (a) AH orbital step from current gradient
            Eigen::MatrixXd kappa = augmented_hessian_step(
                G_curr, st_current.F_I_mo, st_current.F_A_mo,
                n_core, n_act, n_virt,
                level_shift, max_rot, all_mo_irr, use_sym);
            if (micro == 0)
                kappa_first = kappa;

            // (b) First-order h_eff change: δh = [κ, F^I]_{act,act}
            Eigen::MatrixXd dh = delta_h_eff(kappa, st_current.F_I_mo, n_core, n_act);

            // (c) CI response c¹_r for each root r
            const int ci_dim  = static_cast<int>(st_current.ci_vecs.rows());
            const int nr_used = static_cast<int>(st_current.ci_vecs.cols());
            Eigen::MatrixXd c1_vecs = Eigen::MatrixXd::Zero(ci_dim, nr_used);
            for (int r = 0; r < nr_used; ++r)
            {
                const Eigen::VectorXd& c0r = st_current.ci_vecs.col(r);
                Eigen::VectorXd sigma = ci_sigma_1body(
                    dh, c0r, a_strs, b_strs, st_current.dets, n_act);
                c1_vecs.col(r) = ci_response(
                    c0r, st_current.ci_energies(r), st_current.H_CI_diag, sigma);
            }

            // (d) First-order 2-RDM: Γ¹ = Σ_r w_r (|c¹_r><c⁰_r| + |c⁰_r><c¹_r|)
            std::vector<double> Gamma1 = compute_2rdm_bilinear(
                c1_vecs, st_current.ci_vecs, weights, a_strs, b_strs, st_current.dets, n_act);
            {
                auto Gamma1_T = compute_2rdm_bilinear(
                    st_current.ci_vecs, c1_vecs, weights, a_strs, b_strs, st_current.dets, n_act);
                for (std::size_t i = 0; i < Gamma1.size(); ++i)
                    Gamma1[i] += Gamma1_T[i];
            }

            // (e) Q-matrix response: Q¹_{pt} = Σ_{uvw} Γ¹_{tuvw} (pu|vw)
            Eigen::MatrixXd Q1 = compute_Q_matrix(eri, C, Gamma1, n_core, n_act, nbasis);

            // (f) CI gradient contribution: G^CI_{pq} = 2(Q¹_{pq} - Q¹_{qp})
            //     For q = n_core+t ∈ active: G^CI[p,q] = 2*Q¹[p,t], antisymmetric.
            Eigen::MatrixXd G_CI = Eigen::MatrixXd::Zero(nbasis, nbasis);
            for (int p = 0; p < nbasis; ++p)
            for (int t = 0; t < n_act; ++t)
            {
                const int q = n_core + t;
                G_CI(p, q) += 2.0 * Q1(p, t);
                G_CI(q, p) -= 2.0 * Q1(p, t);
            }
            G_CI.topLeftCorner(n_core, n_core).setZero();
            G_CI.block(n_core, n_core, n_act, n_act).setZero();
            G_CI.bottomRightCorner(n_virt, n_virt).setZero();

            // (g) Fully dressed gradient: G̃ ← G̃ + H·κ + G^CI
            G_curr = fep1_gradient_update(G_curr, kappa, st_current.F_I_mo, st_current.F_A_mo,
                                          n_core, n_act, n_virt);
            G_curr += G_CI;

            // (h) Accumulate rotation
            kappa_total += kappa;
        }

        // Cap total accumulated rotation
        double max_k = kappa_total.cwiseAbs().maxCoeff();
        if (max_k > max_rot) kappa_total *= max_rot / max_k;
        const Eigen::MatrixXd kappa_grad = build_gradient_fallback_step(st_current.g_orb);

        // ── (4) Apply accumulated rotation with energy-aware backtracking ─────
        bool accepted = false;
        McscfState accepted_state = st_current;
        double best_E = st_current.E_cas;
        double best_g = st_current.gnorm;
        const double merit_weight = 0.10;
        double best_merit = best_E + merit_weight * best_g * best_g;
        Eigen::MatrixXd C_best = C;
        std::vector<Eigen::MatrixXd> step_candidates;
        if (kappa_newton.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(kappa_newton);
        if (kappa_first.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(kappa_first);
        if (kappa_total.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(kappa_total);
        if (kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(kappa_grad);
        if (kappa_newton.cwiseAbs().maxCoeff() > 1e-12 &&
            kappa_total.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(0.5 * (kappa_newton + kappa_total));
        if (kappa_first.cwiseAbs().maxCoeff() > 1e-12 &&
            kappa_total.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(0.5 * (kappa_first + kappa_total));
        if (kappa_total.cwiseAbs().maxCoeff() > 1e-12 &&
            kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(0.5 * (kappa_total + kappa_grad));
        if (kappa_first.cwiseAbs().maxCoeff() > 1e-12 &&
            kappa_grad.cwiseAbs().maxCoeff() > 1e-12)
            step_candidates.push_back(0.5 * (kappa_first + kappa_grad));

        for (const Eigen::MatrixXd& step_base : step_candidates)
        {
            for (double scale : {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625})
            {
                Eigen::MatrixXd kappa_try = scale * step_base;
                if (kappa_try.cwiseAbs().maxCoeff() < 1e-12) continue;

                Eigen::MatrixXd C_trial = apply_orbital_rotation(C, kappa_try, calc._overlap);
                auto trial_res = evaluate(C_trial);
                if (!trial_res) continue;

                const auto& trial = *trial_res;
                const double trial_merit =
                    trial.E_cas + merit_weight * trial.gnorm * trial.gnorm;
                const bool merit_improved = trial_merit < best_merit - 1e-10;
                const double flat_energy_window =
                    std::max(1000.0 * as.tol_mcscf_energy, 1e-6);
                const bool gradient_reduced = trial.gnorm < best_g - 1e-12;
                const double gradient_worsen_window =
                    std::max(0.05 * std::max(best_g, 1e-8), 1e-6);
                const bool energy_improved =
                    trial.E_cas < best_E - 1e-10;
                const bool energy_improved_without_hurting_gradient =
                    energy_improved && trial.gnorm <= best_g + gradient_worsen_window;
                const bool stationary_but_better_grad =
                    std::abs(trial.E_cas - best_E) <= flat_energy_window && gradient_reduced;
                if (!energy_improved_without_hurting_gradient &&
                    !merit_improved &&
                    !stationary_but_better_grad) continue;

                accepted = true;
                best_E = trial.E_cas;
                best_g = trial.gnorm;
                best_merit = trial_merit;
                accepted_state = trial;
                C_best = std::move(C_trial);
            }
        }

        if (accepted)
        {
            C = C_best;
            st_current = std::move(accepted_state);
            level_shift = std::max(1e-3, level_shift * 0.7);
        }
        else
        {
            level_shift = std::min(20.0, level_shift * 2.0);
        }

        const double reported_gnorm = st_current.g_orb.cwiseAbs().maxCoeff();
        st_current.gnorm = reported_gnorm;

        const double dE = st_current.E_cas - E_prev;
        E_prev = st_current.E_cas;
        HartreeFock::Logger::casscf_iteration(macro, st_current.E_cas, dE,
                                              reported_gnorm, reported_gnorm,
                                              reported_gnorm, level_shift, 0.0);

        const bool e_conv_post = macro > 1 && std::abs(dE) < as.tol_mcscf_energy;
        const bool g_conv_post = reported_gnorm < as.tol_mcscf_grad;
        const bool no_orb_rot_post = (reported_gnorm == 0.0);
        if ((e_conv_post && g_conv_post) || (g_conv_post && no_orb_rot_post))
        {
            converged = true;
            break;
        }
    }

    if (!converged)
        return std::unexpected(
            std::format("{}: did not converge in {:d} iterations.", tag, as.mcscf_max_iter));

    HartreeFock::Logger::blank();
    logging(LogLevel::Info, tag + " :", "Converged.");

    // ── Post-convergence ──────────────────────────────────────────────────────
    auto final_res = evaluate(C);
    if (!final_res) return std::unexpected(final_res.error());
    const auto& fst = *final_res;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_g(fst.gamma);
    calc._cas_nat_occ         = eig_g.eigenvalues().reverse();
    calc._cas_mo_coefficients = C;
    calc._total_energy        = fst.E_cas;

    return {};
}

} // anonymous namespace


// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

namespace HartreeFock::Correlation
{

    std::expected<void, std::string> run_casscf(
        HartreeFock::Calculator&                   calc,
        const std::vector<HartreeFock::ShellPair>& shell_pairs)
    {
        RASParams ras;   // active = false → no RAS constraints
        return run_mcscf_loop(calc, shell_pairs, "CASSCF", ras);
    }

    std::expected<void, std::string> run_rasscf(
        HartreeFock::Calculator&                   calc,
        const std::vector<HartreeFock::ShellPair>& shell_pairs)
    {
        const auto& as = calc._active_space;
        RASParams ras;
        ras.nras1      = as.nras1;
        ras.nras2      = as.nras2;
        ras.nras3      = as.nras3;
        ras.max_holes  = as.max_holes;
        ras.max_elec   = as.max_elec;
        ras.active     = true;
        return run_mcscf_loop(calc, shell_pairs, "RASSCF", ras);
    }

} // namespace HartreeFock::Correlation
