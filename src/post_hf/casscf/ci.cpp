#include "post_hf/casscf/ci.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>

namespace HartreeFock::Correlation::CASSCF
{

namespace
{

inline double g_act(const std::vector<double>& ga, int p, int q, int r, int s, int na)
{
    // Active-space two-electron integrals are stored in a flat pqrs layout.
    return ga[((p * na + q) * na + r) * na + s];
}

std::vector<int> occupied_orbitals(CIString det, int n_orb)
{
    // Expand the bitstring into occupied orbitals in ascending index order.
    std::vector<int> occ;
    occ.reserve(n_orb);
    for (int i = 0; i < n_orb; ++i)
        if (det & CASSCFInternal::single_bit_mask(i))
            occ.push_back(i);
    return occ;
}

CIString pack_spin_det(CIString alpha, CIString beta, int n_act)
{
    // Pack alpha and beta strings into one key so mixed-spin determinants can
    // be hashed and looked up without storing a separate pair structure.
    return alpha | ((n_act >= CASSCFInternal::kCIStringBits) ? 0 : (beta << n_act));
}

std::vector<std::pair<int, int>> enumerate_ci_dets(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const RASParams& ras,
    const std::vector<int>& irr_act,
    const SymmetryContext* sym_ctx,
    int target_irr)
{
    // Enumerate only the RAS-admissible determinants, and optionally keep just
    // the target irrep when symmetry information is available.
    const int na = static_cast<int>(a_strs.size());
    const int nb = static_cast<int>(b_strs.size());
    const bool do_sym = sym_ctx != nullptr && !irr_act.empty();

    std::vector<std::pair<int, int>> dets;
    dets.reserve(na * nb);
    for (int ia = 0; ia < na; ++ia)
        for (int ib = 0; ib < nb; ++ib)
            if (CASSCFInternal::admissible_ras_pair(a_strs[ia], b_strs[ib], ras)
             && (!do_sym || CASSCFInternal::determinant_symmetry(a_strs[ia], b_strs[ib], irr_act, *sym_ctx) == target_irr))
                dets.push_back({ia, ib});
    return dets;
}

std::pair<std::vector<int>, std::vector<int>> get_excitation(CIString bra, CIString ket)
{
    // The symmetric difference identifies orbitals that changed occupation;
    // bits present in the bra are annihilations, the rest are creations.
    std::vector<int> ann, cre;
    CIString d = bra ^ ket;
    while (d)
    {
        const int k = std::countr_zero(d);
        if (bra & CASSCFInternal::single_bit_mask(k)) ann.push_back(k);
        else                                          cre.push_back(k);
        d &= d - 1;
    }
    return {ann, cre};
}

template <typename Apply>
std::pair<Eigen::VectorXd, Eigen::MatrixXd> davidson(
    int dim,
    int nroots,
    const Eigen::VectorXd& diag,
    Apply&& apply,
    double tol,
    int max_iter)
{
    if (dim <= 0 || nroots <= 0)
        return {Eigen::VectorXd(), Eigen::MatrixXd()};

    const int nr = std::min(nroots, dim);
    const int init_cols = std::min(dim, std::max(2 * nr, 4));

    // Start from the lowest diagonal guesses, then orthonormalize the initial
    // subspace before projecting and expanding it.
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(dim, init_cols);
    Eigen::VectorXi order = Eigen::VectorXi::LinSpaced(dim, 0, dim - 1);
    std::sort(order.data(), order.data() + dim, [&](int a, int b) { return diag(a) < diag(b); });
    for (int k = 0; k < init_cols; ++k)
        V(order(k), k) = 1.0;

    {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(V);
        V = qr.householderQ() * Eigen::MatrixXd::Identity(dim, init_cols);
    }

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(nr);
    Eigen::MatrixXd ritz   = Eigen::MatrixXd::Zero(dim, nr);

    for (int iter = 0; iter < max_iter; ++iter)
    {
        const int m = static_cast<int>(V.cols());
        Eigen::MatrixXd AV(dim, m);
        for (int k = 0; k < m; ++k)
        {
            Eigen::VectorXd sigma(dim);
            apply(V.col(k), sigma);
            AV.col(k) = sigma;
        }

        // Solve the small projected problem and form residuals for each root.
        Eigen::MatrixXd projected = V.transpose() * AV;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(projected);
        theta = eig.eigenvalues().head(nr);
        const Eigen::MatrixXd subspace = eig.eigenvectors().leftCols(nr);
        ritz = V * subspace;

        double max_residual = 0.0;
        std::vector<Eigen::VectorXd> corrections;
        corrections.reserve(nr);

        for (int root = 0; root < nr; ++root)
        {
            Eigen::VectorXd residual = AV * subspace.col(root) - theta(root) * ritz.col(root);
            const double residual_norm = residual.norm();
            max_residual = std::max(max_residual, residual_norm);
            if (residual_norm <= tol)
                continue;

            // Precondition by the diagonal and remove components already in the
            // current subspace or in previously accepted corrections.
            for (int i = 0; i < dim; ++i)
            {
                double denom = theta(root) - diag(i);
                if (std::abs(denom) < 1e-12)
                    denom = (denom >= 0.0) ? 1e-12 : -1e-12;
                residual(i) /= denom;
            }

            for (int k = 0; k < m; ++k)
                residual -= V.col(k).dot(residual) * V.col(k);
            for (int k = 0; k < static_cast<int>(corrections.size()); ++k)
                residual -= corrections[k].dot(residual) * corrections[k];

            const double norm = residual.norm();
            if (norm > 1e-12)
                corrections.push_back(residual / norm);
        }

        if (max_residual < tol)
            break;
        if (corrections.empty())
            break;

        const int old_cols = static_cast<int>(V.cols());
        V.conservativeResize(Eigen::NoChange, old_cols + static_cast<int>(corrections.size()));
        for (int i = 0; i < static_cast<int>(corrections.size()); ++i)
            V.col(old_cols + i) = corrections[i];

        // Reorthogonalize after subspace expansion so the next projection stays
        // numerically stable.
        for (int k = 0; k < static_cast<int>(V.cols()); ++k)
        {
            for (int j = 0; j < k; ++j)
                V.col(k) -= V.col(j).dot(V.col(k)) * V.col(j);
            const double norm = V.col(k).norm();
            if (norm < 1e-14)
            {
                if (k + 1 < static_cast<int>(V.cols()))
                    V.col(k) = V.col(V.cols() - 1);
                V.conservativeResize(Eigen::NoChange, V.cols() - 1);
                --k;
            }
            else
                V.col(k) /= norm;
        }
    }

    return {theta, ritz};
}

} // namespace

std::vector<std::pair<int, int>> build_ci_determinant_list(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const RASParams& ras,
    const std::vector<int>& irr_act,
    const SymmetryContext* sym_ctx,
    int target_irr)
{
    return enumerate_ci_dets(a_strs, b_strs, ras, irr_act, sym_ctx, target_irr);
}

Eigen::MatrixXd build_ci_hamiltonian_dense(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act)
{
    const int dim = static_cast<int>(dets.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    for (int i = 0; i < dim; ++i)
    {
        const auto [ia, ib] = dets[i];
        for (int j = i; j < dim; ++j)
        {
            const auto [ja, jb] = dets[j];
            const double v = slater_condon_element(
                a_strs[ia], b_strs[ib],
                a_strs[ja], b_strs[jb],
                h_eff, ga, n_act);
            H(i, j) = H(j, i) = v;
        }
    }
    return H;
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

    auto single_parity = [](CIString s, int p, int q) {
        return parity_between(s, std::min(p, q), std::max(p, q));
    };

    if (n_diff == 0)
    {
        // Diagonal matrix elements come from one-electron terms plus all active
        // pair interactions that remain unchanged between bra and ket.
        double val = 0.0;
        for (int k = 0; k < n_act; ++k)
        {
            if (ket_a & CASSCFInternal::single_bit_mask(k)) val += h_eff(k, k);
            if (ket_b & CASSCFInternal::single_bit_mask(k)) val += h_eff(k, k);
        }
        for (int p = 0; p < n_act; ++p)
        {
            const bool pa = ket_a & CASSCFInternal::single_bit_mask(p);
            const bool pb = ket_b & CASSCFInternal::single_bit_mask(p);
            if (!pa && !pb) continue;
            for (int q = 0; q < n_act; ++q)
            {
                const bool qa = ket_a & CASSCFInternal::single_bit_mask(q);
                const bool qb = ket_b & CASSCFInternal::single_bit_mask(q);
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

    if (n_diff_a == 1 && n_diff_b == 0)
    {
        // One alpha excitation: one-electron term plus Coulomb/exchange with the
        // untouched alpha and beta occupations.
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        const int p = ann_a[0], q = cre_a[0];
        const int sgn = single_parity(ket_a, p, q);
        double val = h_eff(p, q) * sgn;
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_a & CASSCFInternal::single_bit_mask(r)) || r == p) continue;
            val += sgn * (g_act(ga, p, q, r, r, n_act) - g_act(ga, p, r, r, q, n_act));
        }
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_b & CASSCFInternal::single_bit_mask(r))) continue;
            val += sgn * g_act(ga, p, q, r, r, n_act);
        }
        return val;
    }

    if (n_diff_a == 0 && n_diff_b == 1)
    {
        // One beta excitation is the same algebra with the spin channels swapped.
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        const int p = ann_b[0], q = cre_b[0];
        const int sgn = single_parity(ket_b, p, q);
        double val = h_eff(p, q) * sgn;
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_b & CASSCFInternal::single_bit_mask(r)) || r == p) continue;
            val += sgn * (g_act(ga, p, q, r, r, n_act) - g_act(ga, p, r, r, q, n_act));
        }
        for (int r = 0; r < n_act; ++r)
        {
            if (!(ket_a & CASSCFInternal::single_bit_mask(r))) continue;
            val += sgn * g_act(ga, p, q, r, r, n_act);
        }
        return val;
    }

    if (n_diff_a == 2 && n_diff_b == 0)
    {
        // Same-spin double excitations reduce to a pure antisymmetrized two-body
        // integral with the fermionic phase accumulated along the move.
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        int p1 = ann_a[0], p2 = ann_a[1]; if (p1 > p2) std::swap(p1, p2);
        int q1 = cre_a[0], q2 = cre_a[1]; if (q1 > q2) std::swap(q1, q2);
        const CIString inter   = (ket_a ^ CASSCFInternal::single_bit_mask(p1)) ^ CASSCFInternal::single_bit_mask(p2);
        const CIString after_p1 = ket_a ^ CASSCFInternal::single_bit_mask(p1);
        const int n1 = count_occupied_below(ket_a, p1);
        const int n2 = count_occupied_below(after_p1, p2);
        const int n3 = count_occupied_below(inter, q1);
        const CIString after_q1 = inter | CASSCFInternal::single_bit_mask(q1);
        const int n4 = count_occupied_below(after_q1, q2);
        const int sgn = ((n1 + n2 + n3 + n4) % 2 == 0) ? 1 : -1;
        return sgn * (g_act(ga, p1, q1, p2, q2, n_act) - g_act(ga, p1, q2, p2, q1, n_act));
    }

    if (n_diff_a == 0 && n_diff_b == 2)
    {
        // Beta-beta doubles follow the same pattern as the alpha-alpha branch.
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        int p1 = ann_b[0], p2 = ann_b[1]; if (p1 > p2) std::swap(p1, p2);
        int q1 = cre_b[0], q2 = cre_b[1]; if (q1 > q2) std::swap(q1, q2);
        const CIString inter   = (ket_b ^ CASSCFInternal::single_bit_mask(p1)) ^ CASSCFInternal::single_bit_mask(p2);
        const CIString after_p1 = ket_b ^ CASSCFInternal::single_bit_mask(p1);
        const int n1 = count_occupied_below(ket_b, p1);
        const int n2 = count_occupied_below(after_p1, p2);
        const int n3 = count_occupied_below(inter, q1);
        const CIString after_q1 = inter | CASSCFInternal::single_bit_mask(q1);
        const int n4 = count_occupied_below(after_q1, q2);
        const int sgn = ((n1 + n2 + n3 + n4) % 2 == 0) ? 1 : -1;
        return sgn * (g_act(ga, p1, q1, p2, q2, n_act) - g_act(ga, p1, q2, p2, q1, n_act));
    }

    if (n_diff_a == 1 && n_diff_b == 1)
    {
        // Mixed-spin singles have no exchange term, only the direct two-electron
        // contribution with one excitation in each spin channel.
        auto [ann_a, cre_a] = get_excitation(bra_a, ket_a);
        auto [ann_b, cre_b] = get_excitation(bra_b, ket_b);
        const int pa = ann_a[0], qa = cre_a[0];
        const int pb = ann_b[0], qb = cre_b[0];
        return single_parity(ket_a, pa, qa)
             * single_parity(ket_b, pb, qb)
             * g_act(ga, pa, qa, pb, qb, n_act);
    }

    return 0.0;
}

Eigen::MatrixXd build_ci_hamiltonian_with_dets(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const RASParams& ras,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act,
    const std::vector<int>& irr_act,
    const SymmetryContext* sym_ctx,
    int target_irr,
    std::vector<std::pair<int, int>>& dets_out)
{
    // Build the full determinant list once so later routines can reuse the same
    // ordering when constructing matrices or packed determinant keys.
    dets_out = enumerate_ci_dets(a_strs, b_strs, ras, irr_act, sym_ctx, target_irr);

    return build_ci_hamiltonian_dense(a_strs, b_strs, dets_out, h_eff, ga, n_act);
}

CIDeterminantSpace build_ci_space(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const RASParams& ras,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act,
    const std::vector<int>& irr_act,
    const SymmetryContext* sym_ctx,
    int target_irr,
    int dense_threshold)
{
    // Keep the determinant list, diagonal, packed keys, and optional dense
    // Hamiltonian together so CI solvers can choose the cheapest path later.
    CIDeterminantSpace space;
    space.dets = enumerate_ci_dets(a_strs, b_strs, ras, irr_act, sym_ctx, target_irr);
    space.diagonal = build_ci_diagonal(a_strs, b_strs, space.dets, h_eff, ga, n_act);
    space.spin_dets = build_spin_dets(a_strs, b_strs, space.dets, n_act);
    space.det_lookup = build_det_lookup(space.spin_dets);
    if (static_cast<int>(space.dets.size()) <= dense_threshold)
        space.dense_hamiltonian = build_ci_hamiltonian_dense(
            a_strs, b_strs, space.dets, h_eff, ga, n_act);
    return space;
}

void apply_ci_hamiltonian(
    const CIDeterminantSpace& space,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act,
    const Eigen::VectorXd& c,
    Eigen::VectorXd& sigma)
{
    const int dim = static_cast<int>(space.dets.size());
    sigma = Eigen::VectorXd::Zero(dim);
    if (dim == 0 || c.size() != dim)
        return;

    if (space.dense_hamiltonian.has_value())
    {
        // Small spaces can use the stored dense Hamiltonian directly.
        sigma = (*space.dense_hamiltonian) * c;
        return;
    }

    if (space.det_lookup.empty() || static_cast<int>(space.spin_dets.size()) != dim)
    {
        // Fallback path: rebuild each matrix element on demand when we do not
        // have a reliable packed-determinant lookup table.
        for (int j = 0; j < dim; ++j)
        {
            const double cj = c(j);
            if (std::abs(cj) < 1e-15)
                continue;
            const auto [ja, jb] = space.dets[j];
            for (int i = 0; i < dim; ++i)
            {
                const auto [ia, ib] = space.dets[i];
                const double hij = slater_condon_element(
                    a_strs[ia], b_strs[ib],
                    a_strs[ja], b_strs[jb],
                    h_eff, ga, n_act);
                if (std::abs(hij) < 1e-15)
                    continue;
                sigma(i) += hij * cj;
            }
        }
        return;
    }

    auto accumulate = [&](CIString bra_a, CIString bra_b, CIString ket_a, CIString ket_b, double coeff)
    {
        const auto it = space.det_lookup.find(pack_spin_det(bra_a, bra_b, n_act));
        if (it == space.det_lookup.end())
            return;

        const double hij = slater_condon_element(bra_a, bra_b, ket_a, ket_b, h_eff, ga, n_act);
        if (std::abs(hij) < 1e-15)
            return;
        sigma(it->second) += hij * coeff;
    };

    for (int j = 0; j < dim; ++j)
    {
        const double cj = c(j);
        if (std::abs(cj) < 1e-15)
            continue;
        const auto [ia, ib] = space.dets[j];
        const CIString ket_a = a_strs[ia];
        const CIString ket_b = b_strs[ib];
        const auto occ_alpha = occupied_orbitals(ket_a, n_act);
        const auto occ_beta = occupied_orbitals(ket_b, n_act);

        // Enumerate all determinants reachable by at most two annihilations and
        // two creations in the same active-space occupation pattern.
        accumulate(ket_a, ket_b, ket_a, ket_b, cj);

        // Alpha and beta excitations are handled separately so the spin labels
        // stay aligned with the Slater-Condon branch logic above.
        for (int q : occ_alpha)
            for (int p = 0; p < n_act; ++p)
            {
                if (ket_a & CASSCFInternal::single_bit_mask(p))
                    continue;
                auto ann = apply_annihilation(ket_a, q);
                if (!ann.valid) continue;
                auto cre = apply_creation(ann.det, p);
                if (!cre.valid) continue;
                accumulate(cre.det, ket_b, ket_a, ket_b, cj);
            }

        for (int q : occ_beta)
            for (int p = 0; p < n_act; ++p)
            {
                if (ket_b & CASSCFInternal::single_bit_mask(p))
                    continue;
                auto ann = apply_annihilation(ket_b, q);
                if (!ann.valid) continue;
                auto cre = apply_creation(ann.det, p);
                if (!cre.valid) continue;
                accumulate(ket_a, cre.det, ket_a, ket_b, cj);
            }

        // Same-spin doubles must preserve spin channel and ordering within each
        // pair of removed orbitals.
        for (std::size_t q1 = 0; q1 + 1 < occ_alpha.size(); ++q1)
            for (std::size_t q2 = q1 + 1; q2 < occ_alpha.size(); ++q2)
                for (int p1 = 0; p1 < n_act; ++p1)
                {
                    if (ket_a & CASSCFInternal::single_bit_mask(p1))
                        continue;
                    for (int p2 = p1 + 1; p2 < n_act; ++p2)
                    {
                        if (ket_a & CASSCFInternal::single_bit_mask(p2))
                            continue;
                        auto ann1 = apply_annihilation(ket_a, occ_alpha[q1]);
                        if (!ann1.valid) continue;
                        auto ann2 = apply_annihilation(ann1.det, occ_alpha[q2]);
                        if (!ann2.valid) continue;
                        auto cre1 = apply_creation(ann2.det, p1);
                        if (!cre1.valid) continue;
                        auto cre2 = apply_creation(cre1.det, p2);
                        if (!cre2.valid) continue;
                        accumulate(cre2.det, ket_b, ket_a, ket_b, cj);
                    }
                }

        for (std::size_t q1 = 0; q1 + 1 < occ_beta.size(); ++q1)
            for (std::size_t q2 = q1 + 1; q2 < occ_beta.size(); ++q2)
                for (int p1 = 0; p1 < n_act; ++p1)
                {
                    if (ket_b & CASSCFInternal::single_bit_mask(p1))
                        continue;
                    for (int p2 = p1 + 1; p2 < n_act; ++p2)
                    {
                        if (ket_b & CASSCFInternal::single_bit_mask(p2))
                            continue;
                        auto ann1 = apply_annihilation(ket_b, occ_beta[q1]);
                        if (!ann1.valid) continue;
                        auto ann2 = apply_annihilation(ann1.det, occ_beta[q2]);
                        if (!ann2.valid) continue;
                        auto cre1 = apply_creation(ann2.det, p1);
                        if (!cre1.valid) continue;
                        auto cre2 = apply_creation(cre1.det, p2);
                        if (!cre2.valid) continue;
                        accumulate(ket_a, cre2.det, ket_a, ket_b, cj);
                    }
                }

        // Mixed alpha-beta doubles are the remaining nonzero two-body couplings.
        for (int qa : occ_alpha)
            for (int qb : occ_beta)
                for (int pa = 0; pa < n_act; ++pa)
                {
                    if (ket_a & CASSCFInternal::single_bit_mask(pa))
                        continue;
                    for (int pb = 0; pb < n_act; ++pb)
                    {
                        if (ket_b & CASSCFInternal::single_bit_mask(pb))
                            continue;
                        auto ann_a = apply_annihilation(ket_a, qa);
                        if (!ann_a.valid) continue;
                        auto ann_b = apply_annihilation(ket_b, qb);
                        if (!ann_b.valid) continue;
                        auto cre_a = apply_creation(ann_a.det, pa);
                        if (!cre_a.valid) continue;
                        auto cre_b = apply_creation(ann_b.det, pb);
                        if (!cre_b.valid) continue;
                        accumulate(cre_a.det, cre_b.det, ket_a, ket_b, cj);
                    }
                }
    }
}

Eigen::VectorXd build_ci_diagonal(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act)
{
    // The diagonal is just the Hamiltonian expectation value of each determinant
    // with itself, so reuse the same Slater-Condon routine for consistency.
    Eigen::VectorXd diag(static_cast<int>(dets.size()));
    for (int I = 0; I < static_cast<int>(dets.size()); ++I)
    {
        const auto [ia, ib] = dets[I];
        diag(I) = slater_condon_element(
            a_strs[ia], b_strs[ib],
            a_strs[ia], b_strs[ib],
            h_eff, ga, n_act);
    }
    return diag;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_ci_dense(
    const Eigen::MatrixXd& H,
    int nroots,
    double tol)
{
    (void)tol;
    const int dim = static_cast<int>(H.rows());
    if (dim == 0 || nroots <= 0)
        return {Eigen::VectorXd(), Eigen::MatrixXd(dim, 0)};
    // Dense problems are solved exactly; `tol` is kept only for API symmetry.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
    const int nr = std::min(nroots, dim);
    return {eig.eigenvalues().head(nr), eig.eigenvectors().leftCols(nr)};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_ci(
    const Eigen::MatrixXd& H,
    int nroots,
    double tol,
    int dense_threshold)
{
    const int dim = static_cast<int>(H.rows());
    if (dim == 0 || nroots <= 0)
        return {Eigen::VectorXd(), Eigen::MatrixXd(dim, 0)};
    if (dim <= dense_threshold)
        // If the matrix is small enough, materialize it explicitly and diagonalize.
        return solve_ci_dense(H, nroots, tol);

    const Eigen::VectorXd diag = H.diagonal();
    auto apply = [&H](const Eigen::VectorXd& c, Eigen::VectorXd& sigma)
    {
        sigma = H * c;
    };
    return davidson(dim, nroots, diag, apply, tol, 1000);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> solve_ci(
    int dim,
    int nroots,
    const Eigen::VectorXd& diag,
    const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& sigma_apply,
    double tol,
    int max_iter,
    int dense_threshold)
{
    // Reconstruct a dense matrix from the sigma operator when the dimension is
    // still small, otherwise keep the iterative Davidson path.
    if (dim == 0 || nroots <= 0)
        return {Eigen::VectorXd(), Eigen::MatrixXd(dim, 0)};
    if (dim <= dense_threshold)
    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
        for (int j = 0; j < dim; ++j)
        {
            Eigen::VectorXd sigma(dim);
            sigma_apply(Eigen::VectorXd::Unit(dim, j), sigma);
            H.col(j) = sigma;
        }
        H = 0.5 * (H + H.transpose());
        return solve_ci_dense(H, nroots, tol);
    }
    return davidson(dim, nroots, diag, sigma_apply, tol, max_iter);
}

CISolveResult solve_ci(
    const CIDeterminantSpace& space,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const Eigen::MatrixXd& h_eff,
    const std::vector<double>& ga,
    int n_act,
    int nroots,
    double tol,
    int dense_threshold,
    int max_iter)
{
    CISolveResult result;
    result.diagonal = space.diagonal;
    result.dense_hamiltonian = space.dense_hamiltonian;

    if (space.dets.empty() || nroots <= 0)
        return result;

    if (space.dense_hamiltonian.has_value() && static_cast<int>(space.dense_hamiltonian->rows()) <= dense_threshold)
    {
        // Preserve the exact dense solution when the determinant space is small
        // enough to diagonalize outright.
        auto [energies, vectors] = solve_ci_dense(*space.dense_hamiltonian, nroots, tol);
        result.energies = std::move(energies);
        result.vectors = std::move(vectors);
        result.used_direct_sigma = false;
        return result;
    }

    auto sigma_apply = [&](const Eigen::VectorXd& c, Eigen::VectorXd& sigma)
    {
        // For larger spaces, apply the Hamiltonian through the packed lookup or
        // direct Slater-Condon expansion instead of forming H explicitly.
        apply_ci_hamiltonian(space, a_strs, b_strs, h_eff, ga, n_act, c, sigma);
    };
    auto [energies, vectors] = solve_ci(
        static_cast<int>(space.dets.size()),
        nroots,
        space.diagonal,
        sigma_apply,
        tol,
        max_iter,
        dense_threshold);

    result.energies = std::move(energies);
    result.vectors = std::move(vectors);
    result.used_direct_sigma = true;
    return result;
}

} // namespace HartreeFock::Correlation::CASSCF
