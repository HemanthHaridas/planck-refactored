// fock_accumulate.h — contract one canonical ERI quartet straight into the Fock
// matrix, without ever materializing the nb^4 tensor.
//
// This is the core of the memory-direct SCF Fock build. The two-phase builders
// (_compute_2e_fock and friends) scatter each canonical quartet's value into the
// 8 slots of its permutational orbit inside a full nb^4 array, then contract
// that array against the density in a second nb^4 sweep. These functions fuse
// the two: given the canonical quartet (i,j,k,l) and its contracted value, they
// apply the SAME orbit directly as accumulations into G (which is only nb^2).
//
// The rule — and why there are no degeneracy factors
// ---------------------------------------------------
// The quartet loop visits each canonical (i,j,k,l) exactly once (filter:
// j>=i, l>=k, (k,l) >=lex (i,j)). Its orbit under the 8-fold ERI symmetry is
//
//   (ij|kl) (ji|kl) (ij|lk) (ji|lk) (kl|ij) (lk|ij) (kl|ji) (lk|ji)
//
// When indices coincide (i==j, or k==l, or (ij)==(kl)) that orbit collapses and
// some of the 8 tuples are the SAME slot. The classic formulation handles this
// with hand-derived degeneracy weights, which is where these builds usually go
// wrong. We instead enumerate the orbit's DISTINCT tuples and apply an
// unweighted contribution per distinct tuple. Deduplication then handles every
// collapse case automatically, exactly reproducing what a full-tensor Phase 2
// would read back out — because Phase 2 reads each distinct slot once too.
//
// Verified against a brute-force nb^4 contraction on random 8-fold-symmetric
// tensors for nb = 1..7 (worst deviation 6e-15, pure summation-order noise) and
// against the production two-phase builder on real integrals — see
// tests/fock_accumulate.cpp.
//
// Accumulation, not store: unlike write_eri_permutations (which is store-only
// and therefore order-independent), these are read-modify-write reductions into
// G. Any threaded caller MUST give each thread its own G and sum the partials in
// a fixed thread-index order — never `omp atomic update`, never `omp critical`.
// See the DFT XC-reduction determinism note: an order-dependent reduction here
// reintroduces thread-count jitter.
#pragma once

#include <array>
#include <cstddef>

#include <Eigen/Dense>

namespace HartreeFock::Integrals
{
    // The distinct slots of the 8-fold orbit of canonical quartet (i,j,k,l).
    // Returns the count; fills `out` with that many (a,b,c,d) tuples.
    inline std::size_t distinct_eri_orbit(
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        std::array<std::array<std::size_t, 4>, 8> &out) noexcept
    {
        const std::array<std::array<std::size_t, 4>, 8> all = {{
            {i, j, k, l}, {j, i, k, l}, {i, j, l, k}, {j, i, l, k},
            {k, l, i, j}, {l, k, i, j}, {k, l, j, i}, {l, k, j, i},
        }};

        std::size_t n = 0;
        for (const auto &t : all)
        {
            bool seen = false;
            for (std::size_t m = 0; m < n; ++m)
                if (out[m] == t)
                {
                    seen = true;
                    break;
                }
            if (!seen)
                out[n++] = t;
        }
        return n;
    }

    // Which term(s) the loop accumulates.
    //
    // HF wants Combined (G = J - 0.5K) and is the only mode that existed when
    // this loop was written. DFT needs the single terms: J always, K alone
    // scaled by exact_exchange_coefficient, and for range-separated functionals
    // two K's at different omega added to one J.
    //
    // This is a mode on the existing loop rather than three copies of it. The
    // ~100 lines of block/component screening, canonical filtering, and orbit
    // handling between the loop head and the accumulate call are identical for
    // all three; forking them would mean keeping three copies of the trickiest
    // code in the build in sync. Only the terminal accumulate differs.
    //
    // CoulombOnly / ExchangeOnly emit RAW J and RAW K — the caller applies its
    // own coefficient. See the prefactor contract in fock_accumulate.h.
    enum class FusedTerm
    {
        Combined,     // G = J - 0.5K (RHF) / J - K (UHF) — the HF path
        CoulombOnly,  // J, raw
        ExchangeOnly, // K, raw
    };

    // ── Single-term accumulators (J-only / K-only) ───────────────────────────
    //
    // HF always wants the combined G = J - 0.5K, so the entries below fuse both
    // terms. DFT does not: it needs J alone, K alone scaled by
    // exact_exchange_coefficient, and — for range-separated functionals — two
    // K's at different omega added to one J. Hence the split.
    //
    // The orbit argument carries over verbatim: `distinct_eri_orbit` enumerates
    // the orbit's distinct tuples and each gets one unweighted contribution, so
    // collapse cases handle themselves. That is a property of the enumeration,
    // not of which term is accumulated, so it holds term-by-term.
    //
    // PREFACTOR CONTRACT: `exchange_accumulate` emits RAW, UNSCALED K —
    //
    //     K(a,c) += P(b,d) * val        (no 0.5, no sign)
    //
    // The 0.5 (RHF) and 1.0 (UHF) belong to the combined wrappers below, and
    // DFT applies its own coefficient downstream (-0.5 for RKS, -1 for UKS, on
    // top of full_range_/short_range_exchange_coefficient). Folding the 0.5 in
    // here would halve every RKS hybrid's exact exchange while leaving UKS
    // correct — a plausible-looking energy, no crash. Pinned by the round-trip
    // assertions in tests/fock_accumulate.cpp.

    // J(a,b) += P(c,d) * val, over the orbit.
    inline void coulomb_accumulate(
        Eigen::MatrixXd &J,
        const Eigen::MatrixXd &P,
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        double val) noexcept
    {
        std::array<std::array<std::size_t, 4>, 8> orbit;
        const std::size_t n = distinct_eri_orbit(i, j, k, l, orbit);
        for (std::size_t m = 0; m < n; ++m)
        {
            const std::size_t a = orbit[m][0], b = orbit[m][1];
            const std::size_t c = orbit[m][2], d = orbit[m][3];
            J(a, b) += P(c, d) * val;
        }
    }

    // K(a,c) += P(b,d) * val, over the orbit. Raw — see the contract above.
    inline void exchange_accumulate(
        Eigen::MatrixXd &K,
        const Eigen::MatrixXd &P,
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        double val) noexcept
    {
        std::array<std::array<std::size_t, 4>, 8> orbit;
        const std::size_t n = distinct_eri_orbit(i, j, k, l, orbit);
        for (std::size_t m = 0; m < n; ++m)
        {
            const std::size_t a = orbit[m][0], b = orbit[m][1];
            const std::size_t c = orbit[m][2], d = orbit[m][3];
            K(a, c) += P(b, d) * val;
        }
    }

    // ── Combined entries (the HF path) ───────────────────────────────────────
    //
    // Deliberately NOT rewritten as coulomb_accumulate + exchange_accumulate.
    // Both terms share one orbit enumeration and one traversal here; routing
    // through the split pair would call distinct_eri_orbit twice and reorder
    // UHF's accumulation. The bodies stay as they were, so the HF path is
    // byte-identical by construction and its gate cannot drift. The split
    // functions above are the same algebra, one term each — the round-trip
    // assertions verify the two forms agree.

    // RHF: G(mu,nu) = sum_{lam,sig} P(lam,sig) * [ (mu nu|lam sig)
    //                                              - 0.5 (mu lam|nu sig) ]
    //
    // Per distinct orbit tuple (a,b,c,d) carrying the value `val`:
    //   Coulomb : G(a,b) += P(c,d) * val
    //   Exchange: G(a,c) -= 0.5 * P(b,d) * val
    inline void fock_accumulate_rhf(
        Eigen::MatrixXd &G,
        const Eigen::MatrixXd &P,
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        double val) noexcept
    {
        std::array<std::array<std::size_t, 4>, 8> orbit;
        const std::size_t n = distinct_eri_orbit(i, j, k, l, orbit);
        for (std::size_t m = 0; m < n; ++m)
        {
            const std::size_t a = orbit[m][0], b = orbit[m][1];
            const std::size_t c = orbit[m][2], d = orbit[m][3];
            G(a, b) += P(c, d) * val;
            G(a, c) -= 0.5 * P(b, d) * val;
        }
    }

    // UHF: same orbit, spin-resolved.
    //   Ga(a,b) += Pt(c,d)*val ;  Ga(a,c) -= Pa(b,d)*val
    //   Gb(a,b) += Pt(c,d)*val ;  Gb(a,c) -= Pb(b,d)*val
    // (no 0.5 on exchange — the UHF exchange carries the full same-spin density,
    //  matching _compute_2e_fock_uhf's Phase 2.)
    inline void fock_accumulate_uhf(
        Eigen::MatrixXd &Ga, Eigen::MatrixXd &Gb,
        const Eigen::MatrixXd &Pt,
        const Eigen::MatrixXd &Pa, const Eigen::MatrixXd &Pb,
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        double val) noexcept
    {
        std::array<std::array<std::size_t, 4>, 8> orbit;
        const std::size_t n = distinct_eri_orbit(i, j, k, l, orbit);
        for (std::size_t m = 0; m < n; ++m)
        {
            const std::size_t a = orbit[m][0], b = orbit[m][1];
            const std::size_t c = orbit[m][2], d = orbit[m][3];
            const double coul = Pt(c, d) * val;
            Ga(a, b) += coul;
            Gb(a, b) += coul;
            Ga(a, c) -= Pa(b, d) * val;
            Gb(a, c) -= Pb(b, d) * val;
        }
    }
}
