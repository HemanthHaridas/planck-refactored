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
