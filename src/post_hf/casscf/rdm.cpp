#include "post_hf/casscf/rdm.h"

#include "post_hf/casscf/strings.h"

#include <bit>

namespace
{

using HartreeFock::Correlation::CASSCFInternal::CIString;
using HartreeFock::Correlation::CASSCFInternal::low_bit_mask;
using HartreeFock::Correlation::CASSCFInternal::single_bit_mask;
using HartreeFock::Correlation::CASSCF::build_det_lookup;
using HartreeFock::Correlation::CASSCF::build_spin_dets;
using HartreeFock::Correlation::CASSCF::count_occupied_below;

struct FermionOpResult
{
    CIString det = 0;
    double phase = 0.0;
    bool valid = false;
};

// These helpers carry the fermionic phase from the number of occupied orbitals
// below the operator index, matching the determinant ordering used everywhere
// else in the CASSCF string utilities.
inline FermionOpResult apply_annihilation(CIString det, int orb)
{
    const CIString bit = single_bit_mask(orb);
    if (!(det & bit)) return {};
    return {det ^ bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
}

inline FermionOpResult apply_creation(CIString det, int orb)
{
    const CIString bit = single_bit_mask(orb);
    if (det & bit) return {};
    return {det | bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
}

template <typename Fn>
void for_each_set_bit(CIString bits, Fn&& fn)
{
    // Iterate occupied orbitals by repeatedly clearing the lowest set bit.
    while (bits)
    {
        const int bit = std::countr_zero(bits);
        fn(bit);
        bits &= bits - 1;
    }
}

template <typename Fn>
void for_each_clear_bit(CIString bits, int nbits, Fn&& fn)
{
    // The mask trims us to the active spin block so "clear" means "unoccupied"
    // within the selected alpha or beta sector.
    const CIString mask = low_bit_mask(nbits);
    for_each_set_bit((~bits) & mask, fn);
}

inline std::size_t idx4(int p, int q, int r, int s, int n_act)
{
    return static_cast<std::size_t>(((p * n_act + q) * n_act + r) * n_act + s);
}

Eigen::MatrixXd compute_1rdm_impl(
    const Eigen::MatrixXd& bra_vecs,
    const Eigen::MatrixXd& ket_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act,
    bool reference_mode)
{
    const int dim = static_cast<int>(ket_vecs.rows());
    const int nr = static_cast<int>(ket_vecs.cols());
    const auto sd = build_spin_dets(a_strs, b_strs, dets, n_act);
    const auto lut = build_det_lookup(sd);
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(n_act, n_act);

    for (int root = 0; root < nr; ++root)
    {
        const double w = weights(root);
        if (w == 0.0) continue;
        const auto bra = bra_vecs.col(root);
        const auto ket = ket_vecs.col(root);
        for (int j = 0; j < dim; ++j)
        {
            const double ket_j = ket(j);
            if (std::abs(ket_j) < 1e-15) continue;
            const CIString det_j = sd[j];

            if (reference_mode)
            {
                // Reference mode walks all spin orbitals explicitly so the
                // test helper can compare against the same algebra without
                // assuming the spin-blocked determinant layout.
                for (int q_so = 0; q_so < 2 * n_act; ++q_so)
                {
                    auto ann = apply_annihilation(det_j, q_so);
                    if (!ann.valid) continue;
                    const int spin_off = (q_so >= n_act) ? n_act : 0;
                    const int q = q_so - spin_off;
                    for (int p = 0; p < n_act; ++p)
                    {
                        auto cre = apply_creation(ann.det, spin_off + p);
                        if (!cre.valid) continue;
                        auto it = lut.find(cre.det);
                        if (it == lut.end()) continue;
                        gamma(p, q) += w * ann.phase * cre.phase * bra(it->second) * ket_j;
                    }
                }
                continue;
            }

            // In the production path we stay inside the alpha or beta block and
            // only enumerate occupied -> unoccupied moves that preserve spin.
            for (int spin_off : {0, n_act})
            {
                const CIString occ = (det_j >> spin_off) & low_bit_mask(n_act);
                for_each_set_bit(occ, [&](int q) {
                    auto ann = apply_annihilation(det_j, spin_off + q);
                    const CIString occ_after = (ann.det >> spin_off) & low_bit_mask(n_act);
                    for_each_clear_bit(occ_after, n_act, [&](int p) {
                        auto cre = apply_creation(ann.det, spin_off + p);
                        auto it = lut.find(cre.det);
                        if (it == lut.end()) return;
                        gamma(p, q) += w * ann.phase * cre.phase * bra(it->second) * ket_j;
                    });
                });
            }
        }
    }

    return gamma;
}

std::vector<double> compute_2rdm_impl(
    const Eigen::MatrixXd& bra_vecs,
    const Eigen::MatrixXd& ket_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act,
    bool reference_mode)
{
    const int dim = static_cast<int>(ket_vecs.rows());
    const int nr = static_cast<int>(ket_vecs.cols());
    const auto sd = build_spin_dets(a_strs, b_strs, dets, n_act);
    const auto lut = build_det_lookup(sd);
    std::vector<double> Gamma(static_cast<std::size_t>(n_act) * n_act * n_act * n_act, 0.0);

    for (int root = 0; root < nr; ++root)
    {
        const double w = weights(root);
        if (w == 0.0) continue;
        const auto bra = bra_vecs.col(root);
        const auto ket = ket_vecs.col(root);
        for (int j = 0; j < dim; ++j)
        {
            const double ket_j = ket(j);
            if (std::abs(ket_j) < 1e-15) continue;
            const CIString det_j = sd[j];

            if (reference_mode)
            {
                // The reference path keeps the operator order completely
                // explicit: q and s are annihilated first, then r and p are
                // created back onto the bra determinant.
                for (int q_so = 0; q_so < 2 * n_act; ++q_so)
                {
                    auto ann_q = apply_annihilation(det_j, q_so);
                    if (!ann_q.valid) continue;
                    const int qoff = (q_so >= n_act) ? n_act : 0;
                    const int q = q_so - qoff;
                    for (int s_so = 0; s_so < 2 * n_act; ++s_so)
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
                                const double phase =
                                    ann_q.phase * ann_s.phase * cre_r.phase * cre_p.phase;
                                Gamma[idx4(p, q, r, s, n_act)] +=
                                    w * phase * bra(it->second) * ket_j;
                            }
                        }
                    }
                }
                continue;
            }

            // The production path mirrors the spin-adapted determinant layout:
            // annihilate occupied spin orbitals, then fill the remaining holes
            // inside the same spin block.
            const CIString occ0 = det_j & low_bit_mask(2 * n_act);
            for_each_set_bit(occ0, [&](int q_so) {
                auto ann_q = apply_annihilation(det_j, q_so);
                const int qoff = (q_so >= n_act) ? n_act : 0;
                const int q = q_so - qoff;
                const CIString occ1 = ann_q.det & low_bit_mask(2 * n_act);
                for_each_set_bit(occ1, [&](int s_so) {
                    auto ann_s = apply_annihilation(ann_q.det, s_so);
                    const int soff = (s_so >= n_act) ? n_act : 0;
                    const int s = s_so - soff;
                    const CIString occ_after_s = ann_s.det & low_bit_mask(2 * n_act);
                    const CIString free_r = ((~occ_after_s) >> soff) & low_bit_mask(n_act);
                    for_each_set_bit(free_r, [&](int r) {
                        auto cre_r = apply_creation(ann_s.det, soff + r);
                        const CIString occ_after_r = cre_r.det & low_bit_mask(2 * n_act);
                        const CIString free_p = ((~occ_after_r) >> qoff) & low_bit_mask(n_act);
                        for_each_set_bit(free_p, [&](int p) {
                            auto cre_p = apply_creation(cre_r.det, qoff + p);
                            auto it = lut.find(cre_p.det);
                            if (it == lut.end()) return;
                            const double phase =
                                ann_q.phase * ann_s.phase * cre_r.phase * cre_p.phase;
                            Gamma[idx4(p, q, r, s, n_act)] +=
                                w * phase * bra(it->second) * ket_j;
                        });
                    });
                });
            });
        }
    }

    return Gamma;
}

} // namespace

namespace HartreeFock::Correlation::CASSCF
{

Eigen::MatrixXd compute_1rdm(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    return compute_1rdm_impl(ci_vecs, ci_vecs, weights, a_strs, b_strs, dets, n_act, false);
}

std::vector<double> compute_2rdm(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    return compute_2rdm_impl(ci_vecs, ci_vecs, weights, a_strs, b_strs, dets, n_act, false);
}

std::vector<double> compute_2rdm_bilinear(
    const Eigen::MatrixXd& bra_vecs,
    const Eigen::MatrixXd& ket_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    return compute_2rdm_impl(bra_vecs, ket_vecs, weights, a_strs, b_strs, dets, n_act, false);
}

Eigen::MatrixXd compute_1rdm_reference(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    return compute_1rdm_impl(ci_vecs, ci_vecs, weights, a_strs, b_strs, dets, n_act, true);
}

std::vector<double> compute_2rdm_reference(
    const Eigen::MatrixXd& ci_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    return compute_2rdm_impl(ci_vecs, ci_vecs, weights, a_strs, b_strs, dets, n_act, true);
}

std::vector<double> compute_2rdm_bilinear_reference(
    const Eigen::MatrixXd& bra_vecs,
    const Eigen::MatrixXd& ket_vecs,
    const Eigen::VectorXd& weights,
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    return compute_2rdm_impl(bra_vecs, ket_vecs, weights, a_strs, b_strs, dets, n_act, true);
}

} // namespace HartreeFock::Correlation::CASSCF
