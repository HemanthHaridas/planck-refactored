// casscf.cpp — Complete Active Space SCF implementation
//
// Modules:
//   1.  String generation & Slater–Condon rules
//   2.  CI solver (direct diagonalization + Davidson)
//   3.  CI Hamiltonian builder
//   4.  Symmetry screening for CI space and orbital rotations
//   5.  1-RDM and 2-RDM from CI eigenvectors
//   6.  Inactive Fock / Active Fock / core energy
//   7.  Q matrix (2-RDM contraction with partial MO-basis ERI)
//   8.  Orbital gradient (generalized Fock, non-redundant pairs)
//   9.  Orbital rotation via Cayley transform (augmented-Hessian step)
//  10.  Orbital DIIS acceleration (reuses DIISState from types.h)
//  11.  Main run_casscf / run_rasscf loops

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
#include <cassert>
#include <cmath>
#include <deque>
#include <format>
#include <numeric>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace // anonymous — all helpers are internal
{

// ─────────────────────────────────────────────────────────────────────────────
// Module 1: Slater determinant strings and matrix elements
// ─────────────────────────────────────────────────────────────────────────────

using CIString = uint64_t;

// Generate all n_occ-electron occupation strings (bitmasks) for n_orb orbitals.
// Bit k set means orbital k is occupied. Strings are returned in lexicographic
// (ascending integer) order.
std::vector<CIString> generate_strings(int n_orb, int n_occ)
{
    std::vector<CIString> result;
    if (n_occ == 0) { result.push_back(0); return result; }
    if (n_occ > n_orb) return result;

    // Walk through all n_orb-bit integers with exactly n_occ bits set
    // using Gosper's hack for bit-manipulation enumeration
    CIString v = (CIString(1) << n_occ) - 1;
    const CIString limit = CIString(1) << n_orb;
    while (v < limit)
    {
        result.push_back(v);
        // Gosper's hack: next permutation with same popcount
        CIString c = v & (-v);
        CIString r = v + c;
        v = (((r ^ v) >> 2) / c) | r;
    }
    return result;
}

// Sign from moving an electron at position `orb` through the occupied bits
// of string `s` that lie strictly between lo and hi (exclusive).
// Returns +1 or -1.
inline int parity_between(CIString s, int lo, int hi)
{
    if (lo + 1 >= hi) return 1;
    CIString mask = ((CIString(1) << (hi - lo - 1)) - 1) << (lo + 1);
    return (std::popcount(s & mask) % 2 == 0) ? 1 : -1;
}

// ERI index access (pq|rs) in flat nb^4 row-major array
// (same convention as ObaraSaika::_compute_2e)
inline double eri_get(const std::vector<double>& g, int p, int q, int r, int s, int n)
{
    return g[((p * n + q) * n + r) * n + s];
}

// Active-space ERI index (n_act^4 flat array from transform_eri_internal)
inline double g_act(const std::vector<double>& ga, int p, int q, int r, int s, int na)
{
    return ga[((p * na + q) * na + r) * na + s];
}

// Compute CI Hamiltonian matrix element <bra_a bra_b | H | ket_a ket_b>
// where bra/ket_a are alpha strings, bra/ket_b are beta strings.
// h_eff[n_act × n_act]: effective one-electron integrals in active MO basis
//   = F^I_MO[active block] (inactive Fock restricted to active indices)
// ga[n_act^4]: active-space two-electron integrals (tu|vw)
double slater_condon_element(
    CIString bra_a, CIString bra_b,
    CIString ket_a, CIString ket_b,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act)
{
    // Detect excitation level for alpha and beta separately
    const CIString da = bra_a ^ ket_a;
    const CIString db = bra_b ^ ket_b;
    const int n_diff_a = std::popcount(da) / 2;  // number of alpha excitations
    const int n_diff_b = std::popcount(db) / 2;  // number of beta excitations
    const int n_diff_total = n_diff_a + n_diff_b;

    if (n_diff_total > 2) return 0.0;

    // Helper: list of annihilated and created orbitals for one spin
    auto get_excitation = [](CIString bra, CIString ket) -> std::pair<std::vector<int>, std::vector<int>>
    {
        std::vector<int> ann, cre;
        CIString d = bra ^ ket;
        while (d)
        {
            int k = std::countr_zero(d);
            if (bra & (CIString(1) << k)) ann.push_back(k);  // in bra but not ket → annihilated in ket
            else                           cre.push_back(k);  // in ket but not bra → created in ket
            d &= d - 1;
        }
        return {ann, cre};
    };

    // Parity for a single excitation p→q in string s
    // (annihilate p from ket, create q into bra — order matters for sign)
    auto single_parity = [](CIString ket_str, int p_ann, int q_cre) -> int
    {
        int lo = std::min(p_ann, q_cre);
        int hi = std::max(p_ann, q_cre);
        return parity_between(ket_str, lo, hi);
    };

    // ── Diagonal (0 excitations total) ───────────────────────────────────────
    if (n_diff_total == 0)
    {
        double val = 0.0;
        // One-electron terms (alpha + beta)
        for (int k = 0; k < n_act; ++k)
        {
            if (ket_a & (CIString(1) << k)) val += h_eff(k, k);
            if (ket_b & (CIString(1) << k)) val += h_eff(k, k);
        }
        // Two-electron terms
        for (int p = 0; p < n_act; ++p)
        {
            const bool pa = ket_a & (CIString(1) << p);
            const bool pb = ket_b & (CIString(1) << p);
            if (!pa && !pb) continue;
            for (int q = 0; q < n_act; ++q)
            {
                const bool qa = ket_a & (CIString(1) << q);
                const bool qb = ket_b & (CIString(1) << q);
                // α-α: J - K (q > p only, then ×2 → skip, just use p<q half and double below)
                if (pa && qa && p < q)
                    val += g_act(ga, p, p, q, q, n_act) - g_act(ga, p, q, q, p, n_act);
                // β-β
                if (pb && qb && p < q)
                    val += g_act(ga, p, p, q, q, n_act) - g_act(ga, p, q, q, p, n_act);
                // α-β Coulomb only
                if (pa && qb)
                    val += g_act(ga, p, p, q, q, n_act);
            }
        }
        return val;
    }

    // ── One alpha excitation, zero beta ──────────────────────────────────────
    if (n_diff_a == 1 && n_diff_b == 0)
    {
        // bra_b == ket_b
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        int p = ann_a[0], q = cre_a[0];   // ket has p occupied; bra has q occupied
        int sgn = single_parity(ket_a, p, q);

        double val = h_eff(p, q) * sgn;
        // α-α two-electron: r runs over common alpha occupied
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_a & (CIString(1) << r))) continue;
            if (r == p) continue;  // p is being excited away
            val += sgn * (g_act(ga, p, q, r, r, n_act) - g_act(ga, p, r, r, q, n_act));
        }
        // α-β Coulomb: r runs over beta occupied (same in bra and ket)
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_b & (CIString(1) << r))) continue;
            val += sgn * g_act(ga, p, q, r, r, n_act);
        }
        return val;
    }

    // ── Zero alpha excitation, one beta ──────────────────────────────────────
    if (n_diff_a == 0 && n_diff_b == 1)
    {
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        int p = ann_b[0], q = cre_b[0];
        int sgn = single_parity(ket_b, p, q);

        double val = h_eff(p, q) * sgn;
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_b & (CIString(1) << r))) continue;
            if (r == p) continue;
            val += sgn * (g_act(ga, p, q, r, r, n_act) - g_act(ga, p, r, r, q, n_act));
        }
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_a & (CIString(1) << r))) continue;
            val += sgn * g_act(ga, p, q, r, r, n_act);
        }
        return val;
    }

    // ── Two alpha excitations, zero beta ─────────────────────────────────────
    if (n_diff_a == 2 && n_diff_b == 0)
    {
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        // Sort: p1 < p2 annihilated, q1 < q2 created
        int p1 = ann_a[0], p2 = ann_a[1];
        if (p1 > p2) std::swap(p1, p2);
        int q1 = cre_a[0], q2 = cre_a[1];
        if (q1 > q2) std::swap(q1, q2);

        // Combined parity: annihilate p1, p2 from ket; create q1, q2 into result
        // Standard: sgn = parity(ket, p1, p2) * parity(ket w/o p1, q1, q2) * ...
        // Simpler: use the combined formula
        // ε = (-1)^(pos(p1) + pos(p2) + pos(q1) + pos(q2)) in the string ordering
        // We compute step-by-step.
        CIString inter = (ket_a & ~(CIString(1) << p1)) & ~(CIString(1) << p2);
        // Parity for removing p1 from ket_a:
        int sp1 = parity_between(ket_a, std::min(p1, p2), std::max(p1, p2));
        // Remove p2 from ket_a (p2 > p1 after sort):
        // Count bits between p1 and p2 in ket_a (exclusive of p1 itself already):
        // Full parity = (-1)^(bits strictly below p1 in ket_a XOR p1_pos) * ...
        // Use the standard "string phase" calculation:
        // Phase = (-1)^k where k = number of occupied orbs strictly between each pair
        // For double excitation p1,p2 → q1,q2:
        // phase = parity(p1,ket_a) * parity(p2, ket_a \ {p1}) * parity(q1, inter) * parity(q2, inter \ {q1})
        // but this is complex. Use the equivalent:
        // count the number of "crossings" = occupied bits in range (p1,p2), (p1,q1), (p1,q2), (p2,q1), (p2,q2), (q1,q2) in appropriate strings
        // Standard implementation from MCSCF literature:
        auto count_between = [](CIString s, int a, int b) -> int {
            if (a > b) std::swap(a, b);
            if (a + 1 >= b) return 0;
            CIString mask = ((CIString(1) << (b - a - 1)) - 1) << (a + 1);
            return std::popcount(s & mask);
        };
        // Annihilate p1 and p2 from ket_a, create q1 and q2:
        // Phase from removing p1 from ket_a (position index of p1 in occ list):
        int n1 = count_between(ket_a, 0, p1);   // orbs below p1 in ket_a (0-indexed position)
        // After removing p1, remove p2:
        CIString after_p1 = ket_a ^ (CIString(1) << p1);
        // n2 = position of p2 in after_p1 minus 1 if p2 > p1
        int n2 = std::popcount(after_p1 & ((CIString(1) << p2) - 1));
        // Now add q1 to inter:
        int n3 = std::popcount(inter & ((CIString(1) << q1) - 1));
        // Add q2 to inter | q1:
        CIString after_q1 = inter | (CIString(1) << q1);
        int n4 = std::popcount(after_q1 & ((CIString(1) << q2) - 1));
        int total_phase = (n1 + n2 + n3 + n4) % 2;
        int sgn = (total_phase == 0) ? 1 : -1;

        // Matrix element: (p1 q1 | p2 q2) - (p1 q2 | p2 q1)
        return sgn * (g_act(ga, p1, q1, p2, q2, n_act) - g_act(ga, p1, q2, p2, q1, n_act));
    }

    // ── Zero alpha excitations, two beta ─────────────────────────────────────
    if (n_diff_a == 0 && n_diff_b == 2)
    {
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        int p1 = ann_b[0], p2 = ann_b[1];
        if (p1 > p2) std::swap(p1, p2);
        int q1 = cre_b[0], q2 = cre_b[1];
        if (q1 > q2) std::swap(q1, q2);

        CIString inter = (ket_b & ~(CIString(1) << p1)) & ~(CIString(1) << p2);
        CIString after_p1 = ket_b ^ (CIString(1) << p1);
        int n1 = std::popcount(ket_b   & ((CIString(1) << p1) - 1));
        int n2 = std::popcount(after_p1 & ((CIString(1) << p2) - 1));
        int n3 = std::popcount(inter   & ((CIString(1) << q1) - 1));
        CIString after_q1 = inter | (CIString(1) << q1);
        int n4 = std::popcount(after_q1 & ((CIString(1) << q2) - 1));
        int sgn = ((n1 + n2 + n3 + n4) % 2 == 0) ? 1 : -1;

        return sgn * (g_act(ga, p1, q1, p2, q2, n_act) - g_act(ga, p1, q2, p2, q1, n_act));
    }

    // ── One alpha + one beta excitation ──────────────────────────────────────
    if (n_diff_a == 1 && n_diff_b == 1)
    {
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        int pa = ann_a[0], qa = cre_a[0];
        int pb = ann_b[0], qb = cre_b[0];
        int sgn_a = single_parity(ket_a, pa, qa);
        int sgn_b = single_parity(ket_b, pb, qb);
        // No exchange between different spins
        return sgn_a * sgn_b * g_act(ga, pa, qa, pb, qb, n_act);
    }

    return 0.0;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 2: CI solver — direct diagonalization + Davidson
// ─────────────────────────────────────────────────────────────────────────────

// Davidson eigensolver: find lowest nroots eigenpairs of symmetric matrix H.
// Uses diagonal preconditioner and subspace collapse.
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
davidson(const Eigen::MatrixXd& H, int nroots, double tol = 1e-8, int max_iter = 500)
{
    const int n = static_cast<int>(H.rows());
    const Eigen::VectorXd diag = H.diagonal();

    // Initial guess: unit vectors at the nroots lowest diagonal positions
    Eigen::VectorXi idx = Eigen::VectorXi::LinSpaced(n, 0, n - 1);
    std::sort(idx.data(), idx.data() + n, [&](int a, int b){ return diag(a) < diag(b); });

    // Build initial subspace
    int nb = std::min(nroots * 4, n);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, nb);
    for (int k = 0; k < nb; ++k)
        V(idx(k % n), k) = 1.0;

    // Orthonormalize initial vectors
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(V);
    V = qr.householderQ() * Eigen::MatrixXd::Identity(n, nb);

    Eigen::VectorXd theta(nroots);
    Eigen::MatrixXd Y(n, nroots);

    for (int it = 0; it < max_iter; ++it)
    {
        int m = static_cast<int>(V.cols());
        Eigen::MatrixXd AV = H * V;              // H × subspace
        Eigen::MatrixXd M = V.transpose() * AV;  // subspace Hamiltonian

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M);
        theta = eig.eigenvalues().head(nroots);
        const Eigen::MatrixXd& Yvec = eig.eigenvectors().leftCols(nroots);

        // Ritz vectors
        Y = V * Yvec;

        // Residuals and convergence check
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
                // Diagonal preconditioner
                Eigen::VectorXd delta = r;
                for (int i = 0; i < n; ++i)
                {
                    double denom = theta(k) - diag(i);
                    delta(i) /= (std::abs(denom) > 1e-14) ? denom : -1e-14;
                }
                new_vecs.col(n_new++) = delta;
            }
        }

        if (max_res < tol) break;

        // Collapse subspace if too large (keep 2*nroots Ritz vectors + new)
        const int max_sub = std::min(n, std::max(8 * nroots, m + n_new));
        if (m + n_new > max_sub)
        {
            V = Y;  // restart from Ritz vectors
        }
        else
        {
            V.conservativeResize(Eigen::NoChange, m + n_new);
            V.rightCols(n_new) = new_vecs.leftCols(n_new);
        }

        // Orthonormalize enlarged subspace (MGS)
        for (int k = 0; k < static_cast<int>(V.cols()); ++k)
        {
            for (int j = 0; j < k; ++j)
                V.col(k) -= V.col(j).dot(V.col(k)) * V.col(j);
            double nm = V.col(k).norm();
            if (nm < 1e-14)
            {
                // Remove this vector (swap with last)
                V.col(k) = V.col(V.cols() - 1);
                V.conservativeResize(Eigen::NoChange, V.cols() - 1);
                --k;
            }
            else
                V.col(k) /= nm;
        }

        if (V.cols() == 0) break;
    }

    return {theta, Y};
}

// Unified CI solver: direct for dim ≤ 500, Davidson otherwise.
std::pair<Eigen::VectorXd, Eigen::MatrixXd>
solve_ci(const Eigen::MatrixXd& H_CI, int nroots, double tol = 1e-8, int max_iter = 500)
{
    const int dim = static_cast<int>(H_CI.rows());
    if (dim <= 500)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H_CI);
        const int nr = std::min(nroots, dim);
        return {eig.eigenvalues().head(nr), eig.eigenvectors().leftCols(nr)};
    }
    return davidson(H_CI, nroots, tol, max_iter);
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 4: Symmetry utilities
// ─────────────────────────────────────────────────────────────────────────────

// Map the vector of MO symmetry strings (from mo_symmetry) to integer irrep
// indices using the irrep_names table.  Returns -1 for unknown/unresolved.
std::vector<int> map_mo_irreps(
    const std::vector<std::string>& mo_symmetry,
    const std::vector<std::string>& irrep_names)
{
    std::vector<int> idx(mo_symmetry.size(), -1);
    for (std::size_t i = 0; i < mo_symmetry.size(); ++i)
    {
        for (std::size_t g = 0; g < irrep_names.size(); ++g)
        {
            if (mo_symmetry[i] == irrep_names[g])
            {
                idx[i] = static_cast<int>(g);
                break;
            }
        }
    }
    return idx;
}

// Get the irrep index of each active orbital (n_act orbitals starting at n_core).
// Returns empty vector if symmetry is unavailable.
std::vector<int> get_active_irreps(
    const HartreeFock::Calculator& calc, int n_core, int n_act)
{
    const auto& mo_sym = calc._info._scf.alpha.mo_symmetry;
    if (mo_sym.empty()) return {};

    const int nb = static_cast<int>(calc._shells.nbasis());
    if (n_core + n_act > nb) return {};

    const auto all_irreps = map_mo_irreps(mo_sym, calc._sao_irrep_names);
    std::vector<int> active(n_act);
    for (int t = 0; t < n_act; ++t)
        active[t] = all_irreps[n_core + t];
    return active;
}

// Get per-MO irrep indices for all nb MOs.
std::vector<int> get_all_mo_irreps(const HartreeFock::Calculator& calc)
{
    const auto& mo_sym = calc._info._scf.alpha.mo_symmetry;
    if (mo_sym.empty()) return {};
    return map_mo_irreps(mo_sym, calc._sao_irrep_names);
}

// Compute the symmetry (irrep index) of a determinant (α-string, β-string)
// using XOR of occupied orbital irreps.  Valid for Abelian groups (n_irreps ≤ 8).
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

// Resolve target_irrep string → irrep index.
// Empty string or unknown → 0 (totally symmetric).
int resolve_target_irrep(
    const std::string& target_str,
    const std::vector<std::string>& irrep_names)
{
    if (target_str.empty()) return 0;
    for (std::size_t g = 0; g < irrep_names.size(); ++g)
    {
        if (target_str == irrep_names[g])
            return static_cast<int>(g);
    }
    return 0;  // fallback to totally symmetric
}

// Check whether the group is Abelian (all common point groups with ≤ 8 irreps).
bool is_abelian_group(int n_irreps)
{
    return n_irreps > 0 && n_irreps <= 8;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 3: CI string generation with RAS / symmetry constraints
// ─────────────────────────────────────────────────────────────────────────────

struct RASParams
{
    int nras1 = 0, nras2 = 0, nras3 = 0;
    int max_holes = 100, max_elec = 100;  // 100 = no constraint (pure CAS)
    bool active = false;  // true when RAS constraints apply
};

struct StepCandidate
{
    Eigen::MatrixXd kappa;
    const char*     label = "";
};

struct RotationPair
{
    int p = 0;
    int q = 0;
};

// Generate strings with optional RAS constraints and symmetry screening.
// For CAS: params.active = false → no RAS restriction.
// irr_act: active orbital irrep indices (may be empty → no sym screening).
// target_irr: target irrep index (only used when !irr_act.empty() and is_abelian).
std::vector<CIString> generate_strings_filtered(
    int n_act, int n_occ,
    const RASParams& ras,
    const std::vector<int>& irr_act,
    bool use_symmetry, int target_irr,
    bool filter_by_symmetry)
{
    auto all = generate_strings(n_act, n_occ);

    if (!ras.active && (!filter_by_symmetry || irr_act.empty()))
        return all;

    std::vector<CIString> result;
    result.reserve(all.size());

    const CIString ras1_mask = (ras.nras1 > 0) ? ((CIString(1) << ras.nras1) - 1) : 0;
    const CIString ras3_mask = (ras.nras3 > 0)
        ? (((CIString(1) << ras.nras3) - 1) << (ras.nras1 + ras.nras2)) : 0;

    for (CIString s : all)
    {
        // RAS constraint
        if (ras.active)
        {
            int holes = ras.nras1 - std::popcount(s & ras1_mask);
            int elec  = std::popcount(s & ras3_mask);
            if (holes > ras.max_holes || elec > ras.max_elec) continue;
        }
        result.push_back(s);
    }
    return result;
}

// Build symmetry-filtered determinant list (pairs of α/β string indices).
// Returns (a_strs, b_strs) that together span a CI space of a specific
// target symmetry.  For a string-based CI we keep all string individually
// and screen pairs when building the Hamiltonian.
// Here we only pre-filter individual strings by their possible contribution
// to a determinant of target symmetry.  Full screening is in build_ci_hamiltonian.
void build_ci_strings(
    int n_act, int n_alpha, int n_beta,
    const RASParams& ras,
    const std::vector<int>& irr_act,
    bool use_symmetry, int target_irr,
    std::vector<CIString>& a_strs,
    std::vector<CIString>& b_strs)
{
    // Filter by symmetry at determinant level (need both strings)
    bool do_sym = use_symmetry && !irr_act.empty();

    auto all_a = generate_strings_filtered(n_act, n_alpha, ras, irr_act, use_symmetry, target_irr, false);
    auto all_b = generate_strings_filtered(n_act, n_beta,  ras, irr_act, use_symmetry, target_irr, false);

    if (!do_sym)
    {
        a_strs = std::move(all_a);
        b_strs = std::move(all_b);
        return;
    }

    // Remove alpha strings that cannot combine with ANY beta string to give target_irr
    // For efficiency we keep any alpha string whose own symmetry ⊕ some_beta_sym == target_irr.
    // Since we don't pre-filter betas here, just keep all strings for now and
    // screen pairs inside build_ci_hamiltonian.
    a_strs = std::move(all_a);
    b_strs = std::move(all_b);
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 3 (cont.): CI Hamiltonian builder
// ─────────────────────────────────────────────────────────────────────────────

// Build the full CI Hamiltonian matrix.
// When use_symmetry && !irr_act.empty(), only include (Iα,Iβ) determinants
// whose symmetry matches target_irr.
Eigen::MatrixXd build_ci_hamiltonian(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act,
    const std::vector<int>& irr_act,
    bool use_symmetry, int target_irr)
{
    const int na = static_cast<int>(a_strs.size());
    const int nb_s = static_cast<int>(b_strs.size());
    const bool do_sym = use_symmetry && !irr_act.empty();

    // Build determinant index mapping: (Ia, Ib) → flat index
    // Determine which (Ia,Ib) pairs are active
    std::vector<std::pair<int,int>> dets;
    dets.reserve(na * nb_s);
    for (int ia = 0; ia < na; ++ia)
        for (int ib = 0; ib < nb_s; ++ib)
            if (!do_sym || det_symmetry(a_strs[ia], b_strs[ib], irr_act) == target_irr)
                dets.push_back({ia, ib});

    const int dim = static_cast<int>(dets.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

    for (int I = 0; I < dim; ++I)
    {
        auto [ia, ib] = dets[I];
        for (int J = I; J < dim; ++J)
        {
            auto [ja, jb] = dets[J];
            double val = slater_condon_element(
                a_strs[ia], b_strs[ib],
                a_strs[ja], b_strs[jb],
                h_eff, ga, n_act);
            H(I, J) = val;
            H(J, I) = val;
        }
    }
    return H;
}

// Overload that also returns the filtered det list for RDM construction.
Eigen::MatrixXd build_ci_hamiltonian_with_dets(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act,
    const std::vector<int>& irr_act,
    bool use_symmetry, int target_irr,
    std::vector<std::pair<int,int>>& dets_out)
{
    const int na = static_cast<int>(a_strs.size());
    const int nb_s = static_cast<int>(b_strs.size());
    const bool do_sym = use_symmetry && !irr_act.empty();

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
            double val = slater_condon_element(
                a_strs[ia], b_strs[ib],
                a_strs[ja], b_strs[jb],
                h_eff, ga, n_act);
            H(I, J) = val;
            H(J, I) = val;
        }
    }
    return H;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 5: 1-RDM and 2-RDM from CI eigenvectors
// ─────────────────────────────────────────────────────────────────────────────

struct FermionOpResult
{
    CIString det = 0;
    double   phase = 0.0;
    bool     valid = false;
};

inline FermionOpResult apply_annihilation(CIString det, int orb)
{
    const CIString bit = CIString(1) << orb;
    if (!(det & bit)) return {};

    const CIString below = bit - 1;
    const int parity = std::popcount(det & below) % 2;
    return {det ^ bit, parity == 0 ? 1.0 : -1.0, true};
}

inline FermionOpResult apply_creation(CIString det, int orb)
{
    const CIString bit = CIString(1) << orb;
    if (det & bit) return {};

    const CIString below = bit - 1;
    const int parity = std::popcount(det & below) % 2;
    return {det | bit, parity == 0 ? 1.0 : -1.0, true};
}

std::vector<CIString> build_spin_determinants(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    std::vector<CIString> spin_dets;
    spin_dets.reserve(dets.size());
    for (auto [ia, ib] : dets)
        spin_dets.push_back(a_strs[ia] | (b_strs[ib] << n_act));
    return spin_dets;
}

std::unordered_map<CIString, int> build_det_lookup(const std::vector<CIString>& spin_dets)
{
    std::unordered_map<CIString, int> lookup;
    lookup.reserve(spin_dets.size());
    for (int I = 0; I < static_cast<int>(spin_dets.size()); ++I)
        lookup.emplace(spin_dets[I], I);
    return lookup;
}

// Compute state-averaged 1-RDM (n_act × n_act) in the active space.
// ci_vecs: [dim × nroots], each column is a CI eigenvector.
// dets: the (Ia, Ib) pair for each CI determinant index.
// gamma[p,q] = Σ_σ <a†_{pσ} a_{qσ}> (spin-summed)
Eigen::MatrixXd compute_1rdm(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    const int dim  = static_cast<int>(ci_vecs.rows());
    const int nr   = static_cast<int>(ci_vecs.cols());
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(n_act, n_act);

    for (int root = 0; root < nr; ++root)
    {
        const double w = weights(root);
        if (w == 0.0) continue;
        Eigen::MatrixXd gam_r = Eigen::MatrixXd::Zero(n_act, n_act);

        // Alpha contribution: γ^α_{pq} = Σ_{Iα,Jα,Iβ} C*_{Iα,Iβ,r} C_{Jα,Iβ,r} <Iα|a†_p a_q|Jα>
        // We loop over pairs of alpha strings that differ by exactly one orbital.
        for (int p = 0; p < n_act; ++p)
        for (int q = 0; q < n_act; ++q)
        {
            double sum_a = 0.0, sum_b = 0.0;
            for (int I = 0; I < dim; ++I)
            {
                auto [ia, ib] = dets[I];
                const CIString sa = a_strs[ia];
                const CIString sb = b_strs[ib];
                const double ci = ci_vecs(I, root);

                // Alpha: need J with a_strs[ja] = sa \ {p} ∪ {q}, b_strs[jb] = sb
                if (p == q)
                {
                    if (sa & (CIString(1) << p)) sum_a += ci * ci;
                    if (sb & (CIString(1) << p)) sum_b += ci * ci;
                }
                else
                {
                    // Alpha excitation q→p: ket has q occupied, bra has p occupied
                    if ((sa & (CIString(1) << q)) && !(sa & (CIString(1) << p)))
                    {
                        CIString target_a = (sa ^ (CIString(1) << q)) | (CIString(1) << p);
                        // Find J with a_strs[ja] = target_a, b_strs[jb] = sb
                        // Binary search in a_strs
                        auto it = std::lower_bound(a_strs.begin(), a_strs.end(), target_a);
                        if (it != a_strs.end() && *it == target_a)
                        {
                            int ja = static_cast<int>(it - a_strs.begin());
                            // Find J in dets with (ja, ib)
                            for (int J = 0; J < dim; ++J)
                            {
                                if (dets[J].first == ja && dets[J].second == ib)
                                {
                                    int lo = std::min(p, q), hi = std::max(p, q);
                                    int sgn = parity_between(sa, lo, hi);
                                    sum_a += sgn * ci * ci_vecs(J, root);
                                    break;
                                }
                            }
                        }
                    }
                    // Beta excitation q→p
                    if ((sb & (CIString(1) << q)) && !(sb & (CIString(1) << p)))
                    {
                        CIString target_b = (sb ^ (CIString(1) << q)) | (CIString(1) << p);
                        auto it = std::lower_bound(b_strs.begin(), b_strs.end(), target_b);
                        if (it != b_strs.end() && *it == target_b)
                        {
                            int jb = static_cast<int>(it - b_strs.begin());
                            for (int J = 0; J < dim; ++J)
                            {
                                if (dets[J].first == ia && dets[J].second == jb)
                                {
                                    int lo = std::min(p, q), hi = std::max(p, q);
                                    int sgn = parity_between(sb, lo, hi);
                                    sum_b += sgn * ci * ci_vecs(J, root);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            gam_r(p, q) += sum_a + sum_b;
        }
        gamma += w * gam_r;
    }
    return gamma;
}

// Exact state-averaged 1-RDM from the CI eigenvectors in the spin-orbital basis.
// gamma[p,q] = Σ_σ <a†_{pσ} a_{qσ}> (spin-summed).
Eigen::MatrixXd compute_1rdm_fast(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int,int>>& dets,
    int n_act)
{
    const int dim = static_cast<int>(ci_vecs.rows());
    const int nr  = static_cast<int>(ci_vecs.cols());
    const std::vector<CIString> spin_dets = build_spin_determinants(a_strs, b_strs, dets, n_act);
    const auto det_lookup = build_det_lookup(spin_dets);

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

            const CIString ket = spin_dets[J];
            for (int q_so = 0; q_so < 2 * n_act; ++q_so)
            {
                auto ann = apply_annihilation(ket, q_so);
                if (!ann.valid) continue;

                const int spin_offset = (q_so >= n_act) ? n_act : 0;
                const int q = q_so - spin_offset;

                for (int p = 0; p < n_act; ++p)
                {
                    auto cre = apply_creation(ann.det, spin_offset + p);
                    if (!cre.valid) continue;

                    auto it = det_lookup.find(cre.det);
                    if (it == det_lookup.end()) continue;

                    const int I = it->second;
                    gamma(p, q) += w * ann.phase * cre.phase * cv(I) * cJ;
                }
            }
        }
    }
    return gamma;
}

// Compute state-averaged 2-RDM Γ_{pqrs} (flat n_act^4, row-major pqrs).
// Convention: Γ_{pqrs} = Σ_{σσ'} <a†_{pσ}a†_{rσ'}a_{sσ'}a_{qσ}>
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
    const int na4 = n_act * n_act * n_act * n_act;
    const std::vector<CIString> spin_dets = build_spin_determinants(a_strs, b_strs, dets, n_act);
    const auto det_lookup = build_det_lookup(spin_dets);

    std::vector<double> Gamma(na4, 0.0);

    // Index helper
    auto idx4 = [&](int p, int q, int r, int s) -> int {
        return ((p * n_act + q) * n_act + r) * n_act + s;
    };

    for (int root = 0; root < nr; ++root)
    {
        const double w = weights(root);
        if (w == 0.0) continue;
        const auto& cv = ci_vecs.col(root);

        for (int J = 0; J < dim; ++J)
        {
            const double cJ = cv(J);
            if (std::abs(cJ) < 1e-15) continue;

            const CIString ket = spin_dets[J];
            for (int q_so = 0; q_so < 2 * n_act; ++q_so)
            {
                auto ann_q = apply_annihilation(ket, q_so);
                if (!ann_q.valid) continue;

                const int q_spin_offset = (q_so >= n_act) ? n_act : 0;
                const int q = q_so - q_spin_offset;

                for (int s_so = 0; s_so < 2 * n_act; ++s_so)
                {
                    auto ann_s = apply_annihilation(ann_q.det, s_so);
                    if (!ann_s.valid) continue;

                    const int s_spin_offset = (s_so >= n_act) ? n_act : 0;
                    const int s = s_so - s_spin_offset;

                    for (int r = 0; r < n_act; ++r)
                    {
                        auto cre_r = apply_creation(ann_s.det, s_spin_offset + r);
                        if (!cre_r.valid) continue;

                        for (int p = 0; p < n_act; ++p)
                        {
                            auto cre_p = apply_creation(cre_r.det, q_spin_offset + p);
                            if (!cre_p.valid) continue;

                            auto it = det_lookup.find(cre_p.det);
                            if (it == det_lookup.end()) continue;

                            const int I = it->second;
                            const double phase = ann_q.phase * ann_s.phase * cre_r.phase * cre_p.phase;
                            Gamma[idx4(p, q, r, s)] += w * phase * cv(I) * cJ;
                        }
                    }
                }
            }
        }
    }
    return Gamma;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 6: Inactive Fock, Active Fock, core energy
// ─────────────────────────────────────────────────────────────────────────────

// Inactive Fock matrix in MO basis.
// F^I_AO = H_core + _compute_fock_rhf(eri, D_inact, nb)
// D_inact = 2 * C_core * C_core^T
// F^I_MO = C^T * F^I_AO * C
Eigen::MatrixXd build_inactive_fock_mo(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& H_core,
    const std::vector<double>& eri,
    int n_core, int nbasis)
{
    if (n_core == 0)
        return C.transpose() * H_core * C;

    Eigen::MatrixXd C_core = C.leftCols(n_core);
    Eigen::MatrixXd D_inact = 2.0 * C_core * C_core.transpose();
    Eigen::MatrixXd G_inact = HartreeFock::ObaraSaika::_compute_fock_rhf(eri, D_inact, nbasis);
    Eigen::MatrixXd F_I_AO = H_core + G_inact;
    return C.transpose() * F_I_AO * C;
}

// Active Fock matrix in MO basis.
// F^A_AO = _compute_fock_rhf(eri, D_active, nb)
// D_active = C_act * gamma * C_act^T
// F^A_MO = C^T * F^A_AO * C
Eigen::MatrixXd build_active_fock_mo(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& gamma,
    const std::vector<double>& eri,
    int n_core, int n_act, int nbasis)
{
    if (n_act == 0)
        return Eigen::MatrixXd::Zero(nbasis, nbasis);

    Eigen::MatrixXd C_act = C.middleCols(n_core, n_act);
    Eigen::MatrixXd D_active = C_act * gamma * C_act.transpose();
    Eigen::MatrixXd G_active = HartreeFock::ObaraSaika::_compute_fock_rhf(eri, D_active, nbasis);
    return C.transpose() * G_active * C;
}

// Core energy: E_core = Σ_{i=0}^{n_core-1} (h_mo[i,i] + F^I_MO[i,i])
double compute_core_energy(
    const Eigen::MatrixXd& h_mo,
    const Eigen::MatrixXd& F_I_mo,
    int n_core)
{
    double e = 0.0;
    for (int i = 0; i < n_core; ++i)
        e += h_mo(i, i) + F_I_mo(i, i);
    return e;
}

// Active-space energy: Σ_{tu} F^I[t,u]*γ[t,u] + 0.5 Σ_{tuvw} g_act[t,u,v,w]*Γ[t,u,v,w]
double compute_active_energy(
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& gamma,
    const std::vector<double>& ga,
    const std::vector<double>& Gamma,
    int n_core, int n_act)
{
    double e = 0.0;
    // One-electron part: F^I restricted to active block
    for (int t = 0; t < n_act; ++t)
    for (int u = 0; u < n_act; ++u)
        e += F_I_mo(n_core + t, n_core + u) * gamma(t, u);

    // Two-electron part: 0.5 * Σ Γ_{tuvw} g_act_{tuvw}
    const int na4 = n_act * n_act * n_act * n_act;
    for (int k = 0; k < na4; ++k)
        e += 0.5 * Gamma[k] * ga[k];

    return e;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 7: Q matrix
// ─────────────────────────────────────────────────────────────────────────────

// Q_{pt} = Σ_{uvw} Γ_{tuvw} * T[p,u,v,w]
// T = transform_eri(eri, nb, C_all, C_act, C_act, C_act) → shape nb × n_act^3
// Returns Q as nb × n_act matrix.
Eigen::MatrixXd compute_Q_matrix(
    const std::vector<double>& eri,
    const Eigen::MatrixXd& C,
    const std::vector<double>& Gamma,
    int n_core, int n_act, int nbasis)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nbasis, n_act);
    if (n_act == 0) return Q;

    Eigen::MatrixXd C_act = C.middleCols(n_core, n_act);

    // T[p,u,v,w] = Σ_{μ,λ,σ,ρ} C[μ,p] C_act[λ,u] C_act[σ,v] C_act[ρ,w] eri[μλσρ]
    // Compute partial transform: nb × n_act × n_act × n_act
    std::vector<double> T = HartreeFock::Correlation::transform_eri(
        eri, nbasis, C, C_act, C_act, C_act);
    // T is flat [nb × n_act × n_act × n_act] row-major

    const int na = n_act;
    // Q[p,t] = Σ_{uvw} Γ_{tuvw} T[p,u,v,w]
    for (int p = 0; p < nbasis; ++p)
    for (int t = 0; t < na; ++t)
    {
        double q_pt = 0.0;
        for (int u = 0; u < na; ++u)
        for (int v = 0; v < na; ++v)
        for (int w = 0; w < na; ++w)
        {
            int idx_T = ((p * na + u) * na + v) * na + w;
            int idx_G = ((t * na + u) * na + v) * na + w;
            q_pt += Gamma[idx_G] * T[idx_T];
        }
        Q(p, t) = q_pt;
    }
    return Q;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 8: Orbital gradient
// ─────────────────────────────────────────────────────────────────────────────

// Compute the nb×nb antisymmetric orbital gradient matrix g[p,q] = 2*(F_gen[p,q] - F_gen[q,p])
// Non-redundant pairs: (inact,act), (inact,virt), (act,virt).
// Pairs with different irreps are zeroed when use_symmetry is true.
Eigen::MatrixXd compute_orbital_gradient(
    const Eigen::MatrixXd& F_I_mo,    // nb × nb
    const Eigen::MatrixXd& F_A_mo,    // nb × nb
    const Eigen::MatrixXd& Q,         // nb × n_act
    const Eigen::MatrixXd& gamma,     // n_act × n_act
    int n_core, int n_act, int n_virt,
    const std::vector<int>& mo_irreps,
    bool use_symmetry)
{
    const int nb = n_core + n_act + n_virt;
    const Eigen::MatrixXd FA_plus_FI = F_I_mo + F_A_mo;

    // Generalized Fock: F_gen[p,q]
    // q = inactive i: F_gen[p,i] = 2 * (F^I + F^A)[p,i]
    // q = active  t: F_gen[p,t] = Σ_u γ[t,u] * F^I[p,u+n_core] + Q[p, t-n_core]
    // The active Fock contribution for active columns is already represented by
    // the explicit Q contraction; including F^A here double counts it and
    // produces a spurious large orbital gradient.
    // q = virtual a: F_gen[p,a] = 0
    Eigen::MatrixXd F_gen = Eigen::MatrixXd::Zero(nb, nb);

    for (int p = 0; p < nb; ++p)
    {
        // Inactive column
        for (int i = 0; i < n_core; ++i)
            F_gen(p, i) = 2.0 * FA_plus_FI(p, i);

        // Active column t (global index n_core + t)
        for (int t = 0; t < n_act; ++t)
        {
            double val = 0.0;
            for (int u = 0; u < n_act; ++u)
                val += gamma(t, u) * F_I_mo(p, n_core + u);
            val += Q(p, t);
            F_gen(p, n_core + t) = val;
        }

        // Virtual column: F_gen[p, n_core+n_act+a] = 0 (already zero)
    }

    // Antisymmetric gradient g = 2*(F_gen - F_gen^T)
    Eigen::MatrixXd g = 2.0 * (F_gen - F_gen.transpose());

    // Zero redundant pairs (same orbital class–class rotations are redundant)
    // Zero inact-inact, act-act, virt-virt blocks
    g.topLeftCorner(n_core, n_core).setZero();
    g.block(n_core, n_core, n_act, n_act).setZero();
    g.bottomRightCorner(n_virt, n_virt).setZero();
    // Symmetry filtering: zero g[p,q] when irrep[p] ≠ irrep[q]
    if (use_symmetry && !mo_irreps.empty())
    {
        for (int p = 0; p < nb; ++p)
        for (int q = 0; q < nb; ++q)
        {
            if (p == q) continue;
            const int ip = (p < static_cast<int>(mo_irreps.size())) ? mo_irreps[p] : -1;
            const int iq = (q < static_cast<int>(mo_irreps.size())) ? mo_irreps[q] : -1;
            if (ip >= 0 && iq >= 0 && ip != iq)
                g(p, q) = 0.0;
        }
    }

    return g;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 9: Orbital rotation via Cayley transform + augmented-Hessian step
// ─────────────────────────────────────────────────────────────────────────────

// Augmented-Hessian preconditioned orbital rotation step.
// κ_{pq} = -g_{pq} / max(|ΔF_{pq}| + sigma, min_shift)
// where ΔF_{pq} = (F^I+F^A)[p,p] - (F^I+F^A)[q,q]
// U = Cayley transform of κ: U = (I + κ/2)^{-1} (I - κ/2)
// C_new = C_old * U
// Re-orthogonalize via C_new = QR decomposition
Eigen::MatrixXd apply_orbital_rotation(
    const Eigen::MatrixXd& C_old,
    const Eigen::MatrixXd& kappa,   // nb × nb antisymmetric rotation matrix
    const Eigen::MatrixXd& S)       // AO overlap (nb × nb)
{
    const int nb = static_cast<int>(C_old.rows());
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nb, nb);

    // Cayley transform: U = (I + κ/2)^{-1} (I - κ/2)
    // This is exactly orthogonal to first order and numerically stable.
    const Eigen::MatrixXd A = I + 0.5 * kappa;
    const Eigen::MatrixXd B = I - 0.5 * kappa;
    const Eigen::MatrixXd U = A.colPivHouseholderQr().solve(B);

    Eigen::MatrixXd C_new = C_old * U;

    // Re-orthogonalize C_new using Löwdin symmetric orthogonalization:
    // C_new^T S C_new should equal I.
    // Use X = (C^T S C)^{-1/2} and return C * X.
    Eigen::MatrixXd overlap_mo = C_new.transpose() * S * C_new;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(overlap_mo);
    Eigen::VectorXd vals = eig.eigenvalues();
    Eigen::MatrixXd vecs = eig.eigenvectors();

    // Build X = V * diag(1/sqrt(λ)) * V^T
    Eigen::VectorXd inv_sqrt = vals.array().max(1e-12).sqrt().inverse();
    Eigen::MatrixXd X = vecs * inv_sqrt.asDiagonal() * vecs.transpose();
    return C_new * X;
}

// Build the κ matrix from the gradient and the augmented-Hessian preconditioner.
// Returns the antisymmetric κ restricted to non-redundant pairs, capped to max_rot radians.
std::vector<RotationPair> build_rotation_pairs(int n_core, int n_act, int n_virt)
{
    const int nb = n_core + n_act + n_virt;
    std::vector<RotationPair> pairs;
    pairs.reserve(static_cast<std::size_t>(n_core * n_act + n_core * n_virt + n_act * n_virt));

    auto cls = [&](int k) -> int {
        if (k < n_core) return 0;
        if (k < n_core + n_act) return 1;
        return 2;
    };

    for (int p = 0; p < nb; ++p)
    for (int q = p + 1; q < nb; ++q)
    {
        if (cls(p) == cls(q)) continue;
        pairs.push_back({p, q});
    }
    return pairs;
}

Eigen::MatrixXd build_kappa(
    const Eigen::MatrixXd& g,          // nb × nb antisymmetric gradient
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core, int n_act, int n_virt,
    double level_shift = 0.1,          // diagonal LM shift for damped Newton
    double max_rot   = 0.10)           // maximum single-element rotation (radians)
{
    const int nb = n_core + n_act + n_virt;
    const Eigen::MatrixXd FA_plus_FI = F_I_mo + F_A_mo;
    Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nb, nb);
    constexpr double trust_radius = 0.80; // Frobenius norm cap for global step size

    auto cls = [&](int k) -> int {
        if (k < n_core) return 0;
        if (k < n_core + n_act) return 1;
        return 2;
    };

    for (int p = 0; p < nb; ++p)
    for (int q = 0; q < nb; ++q)
    {
        if (p == q || std::abs(g(p, q)) < 1e-18) continue;
        if (cls(p) == cls(q)) continue;  // only non-redundant pairs

        // Damped diagonal Newton step:
        //   step = -g / h  with LM regularization via h/(h^2 + mu^2).
        const double h = 2.0 * (FA_plus_FI(p, p) - FA_plus_FI(q, q));
        const double denom = h * h + level_shift * level_shift;
        kappa(p, q) = -g(p, q) * h / std::max(denom, 1e-18);
    }
    // Enforce antisymmetry
    kappa = 0.5 * (kappa - kappa.transpose());

    // Global trust-radius cap: scale down if any element exceeds max_rot
    const double max_k = kappa.cwiseAbs().maxCoeff();
    if (max_k > max_rot)
        kappa *= max_rot / max_k;

    // Additional global cap controls the full rotation amplitude and helps
    // keep Newton steps stable when many pair rotations are active.
    const double frob = kappa.norm();
    if (frob > trust_radius)
        kappa *= trust_radius / frob;

    return kappa;
}

double diagonal_newton_model_delta(
    const Eigen::MatrixXd& g_orb,
    const Eigen::MatrixXd& kappa,
    const Eigen::MatrixXd& F_I_mo,
    const Eigen::MatrixXd& F_A_mo,
    int n_core, int n_act, int n_virt)
{
    const int nb = n_core + n_act + n_virt;
    const Eigen::MatrixXd FA_plus_FI = F_I_mo + F_A_mo;

    auto cls = [&](int k) -> int {
        if (k < n_core) return 0;
        if (k < n_core + n_act) return 1;
        return 2;
    };

    double linear = 0.0;
    double quad = 0.0;
    for (int p = 0; p < nb; ++p)
    for (int q = p + 1; q < nb; ++q)
    {
        if (cls(p) == cls(q)) continue;
        const double step = kappa(p, q);
        if (std::abs(step) < 1e-18) continue;

        const double h = 2.0 * (FA_plus_FI(p, p) - FA_plus_FI(q, q));
        linear += g_orb(p, q) * step;
        quad += 0.5 * h * step * step;
    }
    return linear + quad;
}


// ─────────────────────────────────────────────────────────────────────────────
// Module 10: Orbital DIIS acceleration using DIISState from types.h
// ─────────────────────────────────────────────────────────────────────────────

// DIISState is defined in base/types.h.
// We use: fock_history = stored κ matrices, error_history = stored g (gradient) matrices.
// The extrapolated κ is used as the orbital rotation step.

// Pop the single oldest vector from a DIISState (divergence trimming).
void diis_pop_oldest(HartreeFock::DIISState& diis)
{
    if (!diis.fock_history.empty())
    {
        diis.fock_history.pop_front();
        diis.error_history.pop_front();
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Shared MCSCF loop (used by both run_casscf and run_rasscf)
// ─────────────────────────────────────────────────────────────────────────────

std::expected<void, std::string> run_mcscf_loop(
    HartreeFock::Calculator& calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const std::string& tag,      // "CASSCF" or "RASSCF"
    const RASParams& ras)
{
    using HartreeFock::Logger::logging;
    using HartreeFock::LogLevel;

    // ── Validate SCF convergence ──────────────────────────────────────────────
    if (!calc._info._is_converged)
        return std::unexpected(tag + ": requires a converged RHF reference.");
    if (calc._scf._scf != HartreeFock::SCFType::RHF)
        return std::unexpected(tag + ": only RHF reference is supported.");

    const auto& as = calc._active_space;
    const int nactele  = as.nactele;
    const int nactorb  = as.nactorb;
    const int nroots   = as.nroots;

    if (nactele <= 0)
        return std::unexpected(tag + ": nactele must be > 0.");
    if (nactorb <= 0)
        return std::unexpected(tag + ": nactorb must be > 0.");
    if (nactele > 2 * nactorb)
        return std::unexpected(tag + ": nactele > 2*nactorb is impossible.");

    const int nbasis = static_cast<int>(calc._shells.nbasis());

    // Total electrons
    const int n_total_elec = static_cast<int>(
        calc._molecule.atomic_numbers.cast<int>().sum()) - calc._molecule.charge;
    if ((n_total_elec - nactele) % 2 != 0)
        return std::unexpected(tag + ": (n_elec - nactele) must be even for RHF-based CASSCF.");

    const int n_core = (n_total_elec - nactele) / 2;
    const int n_act  = nactorb;
    const int n_virt = nbasis - n_core - n_act;

    if (n_core < 0)
        return std::unexpected(tag + ": nactele > total electrons.");
    if (n_virt < 0)
        return std::unexpected(tag + ": n_core + nactorb > nbasis.");
    if (ras.active && (ras.nras1 + ras.nras2 + ras.nras3 != n_act))
        return std::unexpected(tag + ": nras1 + nras2 + nras3 must equal nactorb.");

    // Active electron counts per spin
    const int multiplicity = static_cast<int>(calc._molecule.multiplicity);
    const int n_alpha_act  = (nactele + (multiplicity - 1)) / 2;
    const int n_beta_act   = nactele - n_alpha_act;
    if (n_alpha_act < 0 || n_beta_act < 0 || n_alpha_act > n_act || n_beta_act > n_act)
        return std::unexpected(tag + ": invalid active-space electron count.");

    // ── CI space size check ───────────────────────────────────────────────────
    // Rough estimate (no symmetry reduction)
    auto nchoose = [](int n, int k) -> long long {
        if (k > n || k < 0) return 0;
        if (k == 0 || k == n) return 1;
        long long r = 1;
        for (int i = 0; i < k; ++i) { r = r * (n - i) / (i + 1); }
        return r;
    };
    long long ci_dim_est = nchoose(n_act, n_alpha_act) * nchoose(n_act, n_beta_act);
    if (ci_dim_est > static_cast<long long>(as.ci_max_dim))
        return std::unexpected(std::format("{}: estimated CI dim ({}) exceeds ci_max_dim ({}).",
                                           tag, ci_dim_est, as.ci_max_dim));

    // ── State-averaging weights ───────────────────────────────────────────────
    Eigen::VectorXd weights(nroots);
    if (static_cast<int>(as.weights.size()) == nroots)
        for (int k = 0; k < nroots; ++k) weights(k) = as.weights[k];
    else
        weights.setConstant(1.0 / nroots);
    weights /= weights.sum();

    // ── Symmetry setup ────────────────────────────────────────────────────────
    const bool have_sym   = !calc._sao_irrep_names.empty()
                         && is_abelian_group(static_cast<int>(calc._sao_irrep_names.size()));
    const auto irr_act    = get_active_irreps(calc, n_core, n_act);
    const auto all_mo_irr = get_all_mo_irreps(calc);
    const int  target_irr = resolve_target_irrep(as.target_irrep, calc._sao_irrep_names);
    const bool use_sym    = have_sym && !irr_act.empty();

    // ── Initial MO coefficients ───────────────────────────────────────────────
    // Prefer a stored converged CASSCF orbital set from checkpoint restart when
    // available; otherwise start from the RHF canonical orbitals.
    Eigen::MatrixXd C = calc._info._scf.alpha.mo_coefficients;
    if (calc._cas_mo_coefficients.rows() == nbasis &&
        calc._cas_mo_coefficients.cols() == nbasis)
    {
        C = calc._cas_mo_coefficients;
        logging(LogLevel::Info, tag + " :",
            "Restarting orbital optimization from checkpoint CASSCF orbitals");
    }
    if (C.cols() != nbasis || C.rows() != nbasis)
        return std::unexpected(tag + ": MO coefficient matrix has wrong size.");

    // ── ERI ───────────────────────────────────────────────────────────────────
    std::vector<double> eri_local;
    const std::vector<double>& eri = HartreeFock::Correlation::ensure_eri(
        calc, shell_pairs, eri_local, tag + " :");

    // ── Build CI strings ──────────────────────────────────────────────────────
    std::vector<CIString> a_strs, b_strs;
    build_ci_strings(n_act, n_alpha_act, n_beta_act, ras, irr_act, use_sym, target_irr,
                     a_strs, b_strs);

    // Log header
    logging(LogLevel::Info, tag + " :",
        std::format("Active space: ({:d}e, {:d}o)  n_core={:d}  n_virt={:d}  CI dim ≤ {:d}",
                    nactele, nactorb, n_core, n_virt, ci_dim_est));
    if (use_sym)
        logging(LogLevel::Info, tag + " :",
            std::format("Target irrep: {}  (Abelian symmetry active)",
                        as.target_irrep.empty() ? calc._sao_irrep_names[0] : as.target_irrep));
    if (nroots > 1)
        logging(LogLevel::Info, tag + " :",
            std::format("State-averaged over {:d} roots", nroots));
    HartreeFock::Logger::blank();
    HartreeFock::Logger::casscf_header();

    // ── MCSCF iteration ───────────────────────────────────────────────────────
    HartreeFock::DIISState diis;
    diis.max_vecs = static_cast<std::size_t>(calc._scf._DIIS_dim);

    double E_prev      = 0.0;
    double best_gnorm  = 1e30;
    int    diis_stall  = 0;     // consecutive iterations with growing DIIS error
    bool   diis_active = false;
    bool   converged   = false;
    double newton_mu   = 0.20;  // Levenberg damping for diagonal Newton orbital steps

    std::vector<double> ga;               // active 4c ERI (cached)
    std::vector<std::pair<int,int>> dets; // CI determinant list (stable across macro-iters)
    Eigen::MatrixXd gamma;
    std::vector<double> Gamma_vec;

    struct McscfState
    {
        Eigen::MatrixXd F_I_mo;
        Eigen::MatrixXd F_A_mo;
        Eigen::MatrixXd gamma;
        std::vector<double> Gamma_vec;
        Eigen::MatrixXd g_orb;
        double E_cas = 0.0;
        double gnorm = 0.0;
    };

    auto evaluate_mcscf_state =
        [&](const Eigen::MatrixXd& C_trial) -> std::expected<McscfState, std::string>
    {
        McscfState state;

        state.F_I_mo = build_inactive_fock_mo(C_trial, calc._hcore, eri, n_core, nbasis);
        Eigen::MatrixXd h_eff = state.F_I_mo.block(n_core, n_core, n_act, n_act);

        Eigen::MatrixXd C_act = C_trial.middleCols(n_core, n_act);
        ga = HartreeFock::Correlation::transform_eri_internal(eri, nbasis, C_act);

        Eigen::MatrixXd H_CI = build_ci_hamiltonian_with_dets(
            a_strs, b_strs, h_eff, ga, n_act,
            irr_act, use_sym, target_irr, dets);

        const int ci_dim = static_cast<int>(H_CI.rows());
        if (ci_dim == 0)
            return std::unexpected(tag + ": no CI determinants of target symmetry found.");

        auto [ci_energies, ci_vecs] =
            solve_ci(H_CI, std::min(nroots, ci_dim), 1e-10, 1000);

        const int nr_got = static_cast<int>(ci_energies.size());
        if (nr_got < nroots)
            return std::unexpected(std::format("{}: CI solver returned only {:d} roots (wanted {:d}).",
                                               tag, nr_got, nroots));

        state.gamma = compute_1rdm_fast(ci_vecs, weights, a_strs, b_strs, dets, n_act);
        state.Gamma_vec = compute_2rdm(ci_vecs, weights, a_strs, b_strs, dets, n_act);

        const double E_nuc = calc._nuclear_repulsion;
        const Eigen::MatrixXd h_mo = C_trial.transpose() * calc._hcore * C_trial;
        const double E_core = compute_core_energy(h_mo, state.F_I_mo, n_core);
        const double E_act  = weights.head(nroots).dot(ci_energies.head(nroots));
        state.E_cas = E_nuc + E_core + E_act;

        state.F_A_mo = build_active_fock_mo(C_trial, state.gamma, eri, n_core, n_act, nbasis);
        Eigen::MatrixXd Q = compute_Q_matrix(eri, C_trial, state.Gamma_vec, n_core, n_act, nbasis);
        state.g_orb = compute_orbital_gradient(
            state.F_I_mo, state.F_A_mo, Q, state.gamma, n_core, n_act, n_virt,
            all_mo_irr, use_sym);
        state.gnorm = state.g_orb.cwiseAbs().maxCoeff();

        return state;
    };

    for (unsigned int iter = 1; iter <= as.mcscf_max_iter; ++iter)
    {
        auto state_res = evaluate_mcscf_state(C);
        if (!state_res) return std::unexpected(state_res.error());
        auto state = std::move(*state_res);

        gamma = state.gamma;
        Gamma_vec = state.Gamma_vec;
        const Eigen::MatrixXd& F_I_mo = state.F_I_mo;
        const Eigen::MatrixXd& F_A_mo = state.F_A_mo;
        const Eigen::MatrixXd& g_orb = state.g_orb;
        const double E_cas = state.E_cas;

        const double dE     = E_cas - E_prev;
        E_prev = E_cas;
        const double gnorm  = state.gnorm;

        // ── (k) Convergence check ─────────────────────────────────────────────
        const bool e_conv  = iter > 1 && std::abs(dE) < as.tol_mcscf_energy;
        const bool g_conv  = gnorm < as.tol_mcscf_grad;

        HartreeFock::Logger::casscf_iteration(iter, E_cas, dE, gnorm, gnorm,
            diis_active ? diis.error_norm() : gnorm, 0.0, 0.0);

        if (e_conv && g_conv)
        {
            converged = true;
            break;
        }

        // ── (l) Build damped diagonal-Newton step and optional DIIS blend ─────
        newton_mu = std::clamp(newton_mu, 1e-4, 20.0);
        Eigen::MatrixXd raw_kappa =
            build_kappa(g_orb, F_I_mo, F_A_mo, n_core, n_act, n_virt, newton_mu, 0.20);
        Eigen::MatrixXd diis_kappa = raw_kappa;

        if (calc._scf._use_DIIS && iter >= 2 && gnorm > 2.0 * as.tol_mcscf_grad)
            diis_active = true;
        else
        {
            diis.clear();
            diis_active = false;
            diis_stall = 0;
            best_gnorm = std::min(best_gnorm, gnorm);
        }

        if (diis_active)
        {
            diis.push(raw_kappa, g_orb);
            const double cur_err = diis.error_norm();
            if (cur_err > best_gnorm * calc._scf._diis_restart_factor &&
                calc._scf._diis_restart_factor > 0.0)
            {
                ++diis_stall;
                diis_pop_oldest(diis);
            }
            else
            {
                diis_stall = 0;
                best_gnorm = std::min(best_gnorm, cur_err);
            }

            if (diis_stall >= 3)
            {
                diis.clear();
                diis_stall = 0;
                diis_active = false;
            }

            if (diis_active && diis.ready())
                diis_kappa = diis.extrapolate();
        }

        // ── (m) Trust-region Newton line search ───────────────────────────────
        Eigen::MatrixXd C_best = C;
        double E_best = E_cas;
        double g_best = gnorm;
        double best_rho = -1.0;
        bool accepted = false;

        struct StepCandidate { Eigen::MatrixXd kappa; };
        std::vector<StepCandidate> candidates;
        candidates.push_back({raw_kappa});
        if (diis_active && diis.ready())
        {
            candidates.push_back({diis_kappa});
            candidates.push_back({0.7 * diis_kappa + 0.3 * raw_kappa});
        }

        const double merit_weight = (gnorm > 3.0 * as.tol_mcscf_grad) ? 0.002 : 0.05;
        const double energy_window = std::max(2e-5, 2e-2 * as.tol_mcscf_energy);
        const double ref_merit = E_cas + merit_weight * gnorm;
        double best_merit = ref_merit;

        for (const auto& cand : candidates)
        {
            for (double scale : {2.0, 1.0, 0.5, 0.25, 0.125, 0.0625})
            {
                Eigen::MatrixXd kappa_trial = scale * cand.kappa;
                const double pred = diagonal_newton_model_delta(
                    g_orb, kappa_trial, F_I_mo, F_A_mo, n_core, n_act, n_virt);
                if (pred >= -1e-12) continue;

                Eigen::MatrixXd C_trial =
                    apply_orbital_rotation(C, kappa_trial, calc._overlap);
                auto trial_res = evaluate_mcscf_state(C_trial);
                if (!trial_res) continue;

                const auto& trial = *trial_res;
                const double actual = trial.E_cas - E_cas;
                const double rho = actual / pred; // pred < 0

                const bool energy_ok = trial.E_cas <= E_cas + energy_window;
                const bool grad_win = trial.gnorm < gnorm - 0.1 * as.tol_mcscf_grad;
                if (!energy_ok && !grad_win) continue;

                const double trial_merit = trial.E_cas + merit_weight * trial.gnorm;
                if (trial_merit < best_merit - 1e-12 ||
                    (std::abs(trial_merit - best_merit) < 1e-12 &&
                     (trial.gnorm < g_best - 1e-12 ||
                      (std::abs(trial.gnorm - g_best) < 1e-12 && trial.E_cas < E_best))))
                {
                    C_best = std::move(C_trial);
                    E_best = trial.E_cas;
                    g_best = trial.gnorm;
                    best_merit = trial_merit;
                    best_rho = rho;
                }

                if (rho > 0.02 && trial_merit < ref_merit - 1e-10)
                    accepted = true;
            }
        }

        if (!accepted && E_best >= E_cas && g_best >= gnorm)
        {
            // A stalled line search is not sufficient for convergence on its
            // own. The main convergence gate above requires both energy and
            // orbital-gradient convergence; if we cannot find an improving
            // step here, reset DIIS and continue rather than declaring success
            // from an energy-only plateau.
            diis.clear();
            diis_active = false;
        }
        else
        {
            if (best_rho > 0.75)      newton_mu = std::max(1e-4, newton_mu * 0.7);
            else if (best_rho < 0.25) newton_mu = std::min(20.0, newton_mu * 1.8);
        }

        if (!accepted)
            newton_mu = std::min(20.0, newton_mu * 1.4);

        C = C_best;
    }

    if (!converged)
        return std::unexpected(std::format("{}: did not converge in {:d} iterations.",
                                           tag, as.mcscf_max_iter));

    HartreeFock::Logger::blank();
    logging(LogLevel::Info, tag + " :", "Converged.");

    // ── Post-convergence ──────────────────────────────────────────────────────
    // Natural occupation numbers from 1-RDM (eigenvalues, descending)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_gamma(gamma);
    Eigen::VectorXd nat_occ = eig_gamma.eigenvalues();
    // Reverse to descending order
    calc._cas_nat_occ        = nat_occ.reverse();
    calc._cas_mo_coefficients = C;
    calc._total_energy        = E_prev;

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
    RASParams ras;  // active = false → no RAS constraints
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
