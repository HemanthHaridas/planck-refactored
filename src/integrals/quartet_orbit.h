// quartet_orbit.h — the symmetry orbit of an ERI quartet, shared by the engines.
//
// Under integral symmetry (a set of signed AO permutations), one canonical
// quartet stands for a whole orbit of symmetry-equivalent quartets. The engines
// compute the ERI once at the orbit's representative and replicate it, with the
// accumulated AO sign, across the rest of the orbit.
//
// This was duplicated verbatim in os.cpp, hgp.cpp, and rys.cpp (QuartetOrbitElem
// is byte-identical in all three; build_quartet_orbit differs only in comments).
// The shared memory-direct Fock loop needs exactly one copy, so it lives here.
//
// Why the fused Fock build can use this directly — the dedup argument
// -----------------------------------------------------------------------------
// The scatter under symmetry is a NESTED orbit: the symmetry orbit, and then the
// 8-fold permutational orbit of each of its elements. Fusing that into a direct
// Fock accumulation needs the two to compose without double-counting. They do:
//
//   * build_quartet_orbit already returns a DEDUPLICATED set of distinct
//     canonical quartets (append_quartet_orbit rejects a repeat, and reports
//     forced_zero when the same quartet is reached with the opposite sign, i.e.
//     the integral is symmetry-forbidden and vanishes).
//   * fock_accumulate_{rhf,uhf} already deduplicates the 8-fold permutational
//     orbit of whatever quartet it is handed.
//
// So the two dedups are independent and compose: the fused rule is simply "for
// each symmetry-orbit element, call the existing accumulator with sign * val".
// No new dedup logic is required. Verified against a brute-force nb^4
// contraction of the production scatter for identity, sign-flip, AO-swap,
// swap+sign, and a C2v-like 4-operation group (nb = 2..6, 50 trials each;
// worst deviation 5.3e-15 — summation-order noise). Gated by
// planck-fock-accumulate.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "base/types.h"

namespace HartreeFock::Integrals
{
    struct QuartetOrbitElem
    {
        std::size_t i = 0;
        std::size_t j = 0;
        std::size_t k = 0;
        std::size_t l = 0;
        int sign = 1;
    };

    inline void canonicalize_orbit_pair(std::size_t &i, std::size_t &j) noexcept
    {
        if (i > j)
            std::swap(i, j);
    }

    // The canonical form used by the quartet loop's filter: i<=j, k<=l,
    // (i,j) <= (k,l) lexicographically.
    inline void canonicalize_orbit_quartet(std::size_t &i, std::size_t &j,
                                           std::size_t &k, std::size_t &l) noexcept
    {
        canonicalize_orbit_pair(i, j);
        canonicalize_orbit_pair(k, l);
        if (std::tie(i, j) > std::tie(k, l))
        {
            std::swap(i, k);
            std::swap(j, l);
        }
    }

    // Returns false if the quartet is already present with a CONFLICTING sign —
    // the orbit then cancels and the integral is symmetry-forbidden.
    inline bool append_quartet_orbit(std::vector<QuartetOrbitElem> &orbit,
                                     std::size_t i, std::size_t j,
                                     std::size_t k, std::size_t l, int sign)
    {
        for (const auto &elem : orbit)
        {
            if (elem.i == i && elem.j == j && elem.k == k && elem.l == l)
                return elem.sign == sign;
        }
        orbit.push_back({i, j, k, l, sign});
        return true;
    }

    // True iff canonical quartet (i,j,k,l) is its orbit's representative: no
    // operation maps it (after canonicalization) to a lexicographically smaller
    // quartet. Same answer as build_quartet_orbit(...).first.front() == (i,j,k,l)
    // whenever the orbit is not forced to zero, but with no allocation, dedup or
    // sort, and it returns at the first smaller image. Most canonical quartets
    // are not representatives (about 7 in 8 under D2h), so the fused Fock loop
    // calls this first and builds the full orbit only for representatives. A
    // forbidden non-representative is skipped either way; a forbidden
    // representative is still caught by build_quartet_orbit.
    inline bool is_orbit_representative(
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        const std::vector<HartreeFock::SignedAOSymOp> &sym_ops) noexcept
    {
        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            std::size_t kk = static_cast<std::size_t>(op.ao_map[k]);
            std::size_t ll = static_cast<std::size_t>(op.ao_map[l]);
            canonicalize_orbit_quartet(ii, jj, kk, ll);
            if (std::tie(ii, jj, kk, ll) < std::tie(i, j, k, l))
                return false;
        }
        return true;
    }

    // Skeleton weight (S1): the orbit size of canonical (i,j,k,l) if it is its
    // orbit's representative and not symmetry-forbidden, else 0. One pass over
    // the ops, no allocation. Needs sym_ops to be a GROUP (update_integral_symmetry
    // builds identity plus every valid axis flip, which is closed): then the ops
    // that fix the quartet are its stabilizer, the orbit has |G|/|Stab| distinct
    // members, and the quartet is forbidden iff some op fixes it with sign -1
    // (two ops reaching one image with opposite signs compose to exactly that).
    // Skipping a forbidden quartet only saves work: its contribution would be
    // cancelled exactly by the final symmetrization (the op fixing it with sign
    // -1 maps it to its own negative), so no value gate can see that test.
    inline double skeleton_orbit_weight(
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        const std::vector<HartreeFock::SignedAOSymOp> &sym_ops) noexcept
    {
        std::size_t stab = 0;
        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            std::size_t kk = static_cast<std::size_t>(op.ao_map[k]);
            std::size_t ll = static_cast<std::size_t>(op.ao_map[l]);
            canonicalize_orbit_quartet(ii, jj, kk, ll);
            const auto image = std::tie(ii, jj, kk, ll);
            const auto self = std::tie(i, j, k, l);
            if (image < self)
                return 0.0;
            if (image == self)
            {
                if (op.ao_sign[i] * op.ao_sign[j] * op.ao_sign[k] * op.ao_sign[l] < 0)
                    return 0.0;
                ++stab;
            }
        }
        return static_cast<double>(sym_ops.size()) / static_cast<double>(stab);
    }

    // True iff s(a) s(b) P(R(a), R(b)) == P(a, b) within `tol` for every op: the
    // density has the molecule's symmetry, which the skeleton scheme requires.
    inline bool density_has_symmetry(
        const Eigen::MatrixXd &P,
        const std::vector<HartreeFock::SignedAOSymOp> &sym_ops, double tol) noexcept
    {
        const Eigen::Index nb = P.rows();
        for (const auto &op : sym_ops)
            for (Eigen::Index b = 0; b < nb; ++b)
                for (Eigen::Index a = 0; a < nb; ++a)
                    if (std::abs(op.ao_sign[a] * op.ao_sign[b] * P(op.ao_map[a], op.ao_map[b]) -
                                 P(a, b)) > tol)
                        return false;
        return true;
    }

    // M <- (1/|G|) sum_R s(a) s(b) M(R(a), R(b)): turns a skeleton Fock matrix
    // into the full one (Dupuis-King). A gather; nb^2 |G|.
    inline void symmetrize_over_ops(
        Eigen::MatrixXd &M, const std::vector<HartreeFock::SignedAOSymOp> &sym_ops)
    {
        const Eigen::Index nb = M.rows();
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(nb, nb);
        for (const auto &op : sym_ops)
            for (Eigen::Index b = 0; b < nb; ++b)
                for (Eigen::Index a = 0; a < nb; ++a)
                    out(a, b) += op.ao_sign[a] * op.ao_sign[b] * M(op.ao_map[a], op.ao_map[b]);
        M = out / static_cast<double>(sym_ops.size());
    }

    // {orbit, forced_zero}. The orbit is sorted, so orbit.front() is the
    // representative: callers compute the ERI only when the incoming quartet IS
    // the representative, then replicate over the orbit.
    // Fill `orbit` (cleared first) with the deduplicated, sorted orbit of
    // (i,j,k,l); returns true if it is symmetry-forbidden (the orbit is then
    // partial). Takes the caller's vector so a loop can reuse one buffer per
    // thread instead of allocating per quartet: that allocation was ~13 % of a
    // symmetric fused Fock build after the representative pre-check.
    inline bool fill_quartet_orbit(
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        const std::vector<HartreeFock::SignedAOSymOp> &sym_ops,
        std::vector<QuartetOrbitElem> &orbit)
    {
        orbit.clear();
        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            std::size_t kk = static_cast<std::size_t>(op.ao_map[k]);
            std::size_t ll = static_cast<std::size_t>(op.ao_map[l]);
            const int sign = static_cast<int>(op.ao_sign[i]) * static_cast<int>(op.ao_sign[j]) *
                             static_cast<int>(op.ao_sign[k]) * static_cast<int>(op.ao_sign[l]);
            canonicalize_orbit_quartet(ii, jj, kk, ll);
            if (!append_quartet_orbit(orbit, ii, jj, kk, ll, sign))
                return true; // symmetry-forbidden
        }

        std::sort(orbit.begin(), orbit.end(),
                  [](const QuartetOrbitElem &a, const QuartetOrbitElem &b)
                  {
                      return std::tie(a.i, a.j, a.k, a.l) < std::tie(b.i, b.j, b.k, b.l);
                  });
        return false;
    }

    inline std::pair<std::vector<QuartetOrbitElem>, bool> build_quartet_orbit(
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        const std::vector<HartreeFock::SignedAOSymOp> &sym_ops)
    {
        std::vector<QuartetOrbitElem> orbit;
        orbit.reserve(sym_ops.size());
        const bool forced_zero = fill_quartet_orbit(i, j, k, l, sym_ops, orbit);
        return {std::move(orbit), forced_zero};
    }
}
