#ifndef HF_POSTHF_CASSCF_INTERNAL_H
#define HF_POSTHF_CASSCF_INTERNAL_H

#include <Eigen/Core>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace HartreeFock::Correlation::CASSCFInternal
{

using CIString = std::uint64_t;

inline constexpr int kCIStringBits = std::numeric_limits<CIString>::digits;
inline constexpr int kMaxSeparateSpinOrbitals = kCIStringBits - 1;
inline constexpr int kMaxPackedSpatialOrbitals = (kCIStringBits - 1) / 2;

inline CIString single_bit_mask(int bit)
{
    if (bit < 0 || bit >= kCIStringBits) return 0;
    return CIString(1) << bit;
}

inline CIString low_bit_mask(int nbits)
{
    if (nbits <= 0) return 0;
    if (nbits >= kCIStringBits) return std::numeric_limits<CIString>::max();
    return (CIString(1) << nbits) - 1;
}

struct RASParams
{
    int nras1 = 0, nras2 = 0, nras3 = 0;
    int max_holes = 100, max_elec = 100;
    bool active = false;
};

// RASSCF constraints are enforced on the combined alpha+beta determinant.
// max_holes counts the total number of electrons missing from a doubly
// occupied RAS1 block; max_elec counts the total number of electrons in RAS3.
inline int ras1_holes(CIString alpha, CIString beta, const RASParams& ras)
{
    const CIString ras1_mask = low_bit_mask(ras.nras1);
    const int occ_alpha = std::popcount(alpha & ras1_mask);
    const int occ_beta = std::popcount(beta & ras1_mask);
    return 2 * ras.nras1 - (occ_alpha + occ_beta);
}

inline int ras3_electrons(CIString alpha, CIString beta, const RASParams& ras)
{
    const int ras3_offset = ras.nras1 + ras.nras2;
    const CIString ras3_mask = low_bit_mask(ras.nras3) << ras3_offset;
    return std::popcount(alpha & ras3_mask) + std::popcount(beta & ras3_mask);
}

inline bool admissible_ras_pair(CIString alpha, CIString beta, const RASParams& ras)
{
    if (!ras.active) return true;
    return ras1_holes(alpha, beta, ras) <= ras.max_holes
        && ras3_electrons(alpha, beta, ras) <= ras.max_elec;
}

struct SymmetryContext
{
    std::vector<std::string> names;
    std::vector<std::vector<int>> product;
    bool abelian_1d_only = false;
    int totally_symmetric_irrep = 0;
};

inline int determinant_symmetry(
    CIString alpha,
    CIString beta,
    const std::vector<int>& irr_act,
    const SymmetryContext& sym_ctx)
{
    int sym = sym_ctx.totally_symmetric_irrep;
    for (int t = 0; t < static_cast<int>(irr_act.size()); ++t)
    {
        if (irr_act[t] < 0) continue;
        if (alpha & single_bit_mask(t)) sym = sym_ctx.product[sym][irr_act[t]];
        if (beta  & single_bit_mask(t)) sym = sym_ctx.product[sym][irr_act[t]];
    }
    return sym;
}

inline Eigen::MatrixXd compute_root_overlap(
    const Eigen::MatrixXd& c_old,
    const Eigen::MatrixXd& c_new)
{
    if (c_old.rows() == 0 || c_new.rows() == 0 || c_old.rows() != c_new.rows())
        return Eigen::MatrixXd();
    return (c_old.adjoint() * c_new).cwiseAbs();
}

inline std::vector<int> match_roots_by_max_overlap(const Eigen::MatrixXd& overlaps)
{
    const int nold = static_cast<int>(overlaps.rows());
    const int nnew = static_cast<int>(overlaps.cols());
    const int nmatch = std::min(nold, nnew);

    std::vector<int> assignment(nmatch, -1);
    std::vector<bool> old_used(nold, false);
    std::vector<bool> new_used(nnew, false);

    for (int picked = 0; picked < nmatch; ++picked)
    {
        double best = -1.0;
        int best_old = -1;
        int best_new = -1;
        for (int i = 0; i < nold; ++i)
        {
            if (old_used[i]) continue;
            for (int j = 0; j < nnew; ++j)
            {
                if (new_used[j]) continue;
                const double ov = overlaps(i, j);
                if (ov > best)
                {
                    best = ov;
                    best_old = i;
                    best_new = j;
                }
            }
        }

        if (best_old < 0 || best_new < 0) break;
        assignment[best_old] = best_new;
        old_used[best_old] = true;
        new_used[best_new] = true;
    }

    int next_new = 0;
    for (int i = 0; i < nmatch; ++i)
    {
        if (assignment[i] >= 0) continue;
        while (next_new < nnew && new_used[next_new]) ++next_new;
        if (next_new >= nnew) break;
        assignment[i] = next_new;
        new_used[next_new] = true;
    }

    return assignment;
}

} // namespace HartreeFock::Correlation::CASSCFInternal

#endif // HF_POSTHF_CASSCF_INTERNAL_H
