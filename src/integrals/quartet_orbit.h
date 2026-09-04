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
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

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

    // {orbit, forced_zero}. The orbit is sorted, so orbit.front() is the
    // representative: callers compute the ERI only when the incoming quartet IS
    // the representative, then replicate over the orbit.
    inline std::pair<std::vector<QuartetOrbitElem>, bool> build_quartet_orbit(
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        const std::vector<HartreeFock::SignedAOSymOp> &sym_ops)
    {
        std::vector<QuartetOrbitElem> orbit;
        orbit.reserve(sym_ops.size());

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
                return {orbit, true}; // symmetry-forbidden
        }

        std::sort(orbit.begin(), orbit.end(),
                  [](const QuartetOrbitElem &a, const QuartetOrbitElem &b)
                  {
                      return std::tie(a.i, a.j, a.k, a.l) < std::tie(b.i, b.j, b.k, b.l);
                  });
        return {orbit, false};
    }
}
