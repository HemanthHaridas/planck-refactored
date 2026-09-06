#ifndef HARTREEFOCK_CI_SPAWN_ACCUMULATOR_H
#define HARTREEFOCK_CI_SPAWN_ACCUMULATOR_H

// H2.2 (docs/FCIQMC_PARALLEL_REWRITE_SCOPE.md): a reuse-stable accumulator for
// the FCIQMC spawn loop's per-bin output, to replace the per-bin
// `std::unordered_map<DetKey, Weight>` that H2.4 will drop.
//
// THE PROPERTY `unordered_map` CANNOT GIVE, and this one must:
//
//   summing the same multiset of (det, weight) additions produces a
//   BITWISE-IDENTICAL result on every call, regardless of
//     (a) the order the additions arrived in, and
//     (b) how many times the accumulator has been reused (`.reset()`-ed).
//
// (a) fails for `unordered_map` because a longer-than-2 run of same-key adds
// reassociates with insertion order (IEEE double + is commutative but not
// associative). (b) fails because `unordered_map::clear()` keeps the peak
// bucket count, so a later call with fewer entries iterates in a different
// order than a fresh map with the same contents -- the T2 threading work's
// invariant 3, and the reason R1/R2's reuse attempt had a reordering hazard.
//
// HOW THIS ONE GETS BOTH:
//
//   - inserts just append to a flat vector (cheap, order-immaterial);
//   - `finalize()` sorts the whole vector by a TOTAL order on
//     (alpha, beta, bit-pattern of weight), then folds consecutive equal-key
//     runs left to right. After the sort a key's run is in a fixed order that
//     depends only on the multiset, so the fold `((w_a + w_b) + w_c)` with
//     w_a <= w_b <= w_c by bits is a pure function of the multiset -- (a).
//   - `.reset()` is `vector::clear()` (keeps capacity, drops nothing that
//     affects a value) plus `finalized = false`; a `std::vector` genuinely
//     resets, unlike `unordered_map` -- (b).
//
// Header-only and Eigen-free so its test links nothing but this file.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "post_hf/ci/fciqmc.h"

namespace HartreeFock::Correlation::CI::QMC
{

    class SpawnAccumulator
    {
    public:
        // Add `w` to `det`. Cheap: just an append. Zero weights are kept here
        // (unlike WalkerPopulation::add) because filtering them changes nothing
        // about the finalized sum and keeps this a pure "record the multiset"
        // operation; the driver's compress() drops zeros downstream.
        void add(const DetKey &det, Weight w)
        {
            entries_.push_back({det, w});
            finalized_ = false;
        }

        // Sort + fold equal-key runs. Idempotent. Must be called before any
        // iteration; the sum is undefined (in the "not what you want" sense,
        // not UB) before it.
        void finalize()
        {
            if (finalized_)
                return;
            std::sort(entries_.begin(), entries_.end(), less_);

            std::size_t out = 0;
            std::size_t i = 0;
            while (i < entries_.size())
            {
                const DetKey key = entries_[i].first;
                Weight sum = entries_[i].second;
                ++i;
                while (i < entries_.size() && entries_[i].first == key)
                {
                    sum += entries_[i].second;
                    ++i;
                }
                entries_[out++] = {key, sum};
            }
            entries_.resize(out);
            finalized_ = true;
        }

        // Drop everything, keep capacity. A genuine reset -- a std::vector has
        // no bucket-layout memory to carry between calls.
        void reset() noexcept
        {
            entries_.clear();
            finalized_ = false;
        }

        [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
        [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

        // Iterate the finalized (key, summed-weight) pairs, in the canonical
        // (alpha, beta) order -- which is also the order a merge across bins
        // must consume them in for the merge itself to be deterministic.
        [[nodiscard]] auto begin() const noexcept { return entries_.begin(); }
        [[nodiscard]] auto end() const noexcept { return entries_.end(); }

    private:
        struct KeyWeightLess
        {
            bool operator()(const std::pair<DetKey, Weight> &a,
                            const std::pair<DetKey, Weight> &b) const noexcept
            {
                if (a.first.alpha != b.first.alpha)
                    return a.first.alpha < b.first.alpha;
                if (a.first.beta != b.first.beta)
                    return a.first.beta < b.first.beta;
                // Same key: order by the raw bit pattern of the weight so the
                // fold order is a pure function of the multiset. bit_cast to an
                // unsigned integer gives a total order that does not care about
                // -0.0 vs +0.0 or NaN payloads (neither occurs here, but the
                // order stays total regardless).
                return std::bit_cast<std::uint64_t>(a.second)
                       < std::bit_cast<std::uint64_t>(b.second);
            }
        };

        std::vector<std::pair<DetKey, Weight>> entries_;
        bool finalized_ = false;
        static constexpr KeyWeightLess less_{};
    };

    // H2.4/H2.5 (docs/FCIQMC_PARALLEL_REWRITE_SCOPE.md): the persistent
    // per-call scaffolding for propagate_stochastic, hoisted out of the
    // function so it is built ONCE (by the driver, or per test call) instead
    // of ~50,000 times. Holds the 64 per-bin accumulators, the 64 parent
    // buckets, AND (H2.5) the 64 per-bin RNG streams -- which are re-keyed in
    // place each call rather than reconstructed, now that RandomSource is
    // counter-based (xoshiro256**, O(1) reseed) instead of mt19937_64
    // (~600 ns state fill per stream, ~38 us/call for 64).
    //
    // ready() lazily sizes to `n_bins` on first use. reset_for_call() clears
    // every bin/bucket without freeing capacity -- SpawnAccumulator::reset()
    // and vector::clear() both genuinely reset (unlike unordered_map::clear),
    // so the result is a pure function of that call's inputs regardless of
    // how many prior calls the workspace served. The RNG streams are re-keyed
    // by rekey_streams(call_seed), a pure function of (call_seed, bin index)
    // -- the same derive() recipe the old fresh-construction path used, so
    // the per-bin stream a given (call_seed, bin) sees is unchanged by the
    // hoist.
    struct SpawnWorkspace
    {
        std::vector<SpawnAccumulator> bins;
        std::vector<std::vector<std::pair<DetKey, Weight>>> parents;
        std::vector<RandomSource> bin_rngs;

        void ready(std::size_t n_bins)
        {
            if (bins.size() != n_bins)
            {
                bins.assign(n_bins, SpawnAccumulator{});
                parents.assign(n_bins, {});
                bin_rngs.assign(n_bins, RandomSource{0});
            }
        }

        void reset_for_call()
        {
            for (auto &b : bins)
                b.reset();
            for (auto &p : parents)
                p.clear();
        }

        // Re-key the 64 streams from this call's seed, exactly as the pre-H2.5
        // `RandomSource call_source(call_seed); ... call_source.derive(b)` did
        // -- but in place, no 64 constructions.
        void rekey_streams(std::uint64_t call_seed)
        {
            RandomSource call_source(call_seed);
            for (std::size_t b = 0; b < bin_rngs.size(); ++b)
                bin_rngs[b] = call_source.derive(b);
        }
    };

} // namespace HartreeFock::Correlation::CI::QMC

#endif // HARTREEFOCK_CI_SPAWN_ACCUMULATOR_H
