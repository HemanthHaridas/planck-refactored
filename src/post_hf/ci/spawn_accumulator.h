#ifndef HARTREEFOCK_CI_SPAWN_ACCUMULATOR_H
#define HARTREEFOCK_CI_SPAWN_ACCUMULATOR_H

// H2.2 (docs/FCIQMC_PARALLELISM.md): a reuse-stable accumulator for
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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <queue>
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

        // Indexed read of finalized output for the canonical cross-bin merge.
        // Callers must finalize first, just as they must before begin()/end().
        [[nodiscard]] const std::pair<DetKey, Weight> &at(std::size_t i) const
        {
            return entries_.at(i);
        }

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

    // Literal oracle for the prospective contiguous population owner.  Each bin
    // is already sorted and locally folded by SpawnAccumulator::finalize().  This
    // 64-way merge emits globally determinant-key-sorted entries while preserving
    // the CURRENT map path's summation order for cross-bin duplicates: equal keys
    // are taken in ascending bin order, exactly as propagate_stochastic's
    // `for (bin = 0; ...) out.add(...)` loop does today.
    //
    // This is not yet a production population.  It is a typed boundary and an
    // oracle whose equality to the current map output must remain demonstrated
    // before any parent/RNG convention is redesigned around it.
    struct CanonicalSpawnMerge
    {
        std::vector<std::pair<DetKey, Weight>> entries;
    };

    struct CanonicalSpawnMergeProfile
    {
        double reserve_seconds = 0.0;
        double frontier_seconds = 0.0;
        double stream_seconds = 0.0;
        std::uint64_t emitted = 0;
    };

    // M1 records the merge-owned tagged-entry high-water mark separately from
    // the input bins and final canonical output.  A level merge necessarily
    // holds its input and output levels concurrently, but must not retain an
    // entire history of tree levels.  These counters make that bound testable.
    struct PairwiseSpawnMergeProfile
    {
        std::size_t levels = 0;
        std::size_t peak_live_entries = 0;
        std::size_t peak_live_bytes = 0;
        std::size_t tagged_entry_bytes = 0;
    };

    inline CanonicalSpawnMerge merge_finalized_spawn_bins(
        const std::vector<SpawnAccumulator> &bins,
        CanonicalSpawnMergeProfile *profile = nullptr)
    {
        struct Cursor
        {
            std::size_t bin = 0;
            std::size_t index = 0;
        };
        const auto less_key = [&bins](const Cursor &a, const Cursor &b) {
            const DetKey &ka = bins[a.bin].at(a.index).first;
            const DetKey &kb = bins[b.bin].at(b.index).first;
            if (ka.alpha != kb.alpha)
                return ka.alpha < kb.alpha;
            if (ka.beta != kb.beta)
                return ka.beta < kb.beta;
            return a.bin < b.bin;
        };
        // priority_queue exposes the largest element, hence reverse the total
        // key/bin order to obtain the next smallest determinant contribution.
        const auto later = [&less_key](const Cursor &a, const Cursor &b) {
            return less_key(b, a);
        };
        const auto reserve_start = profile ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
        CanonicalSpawnMerge out;
        std::size_t upper_bound = 0;
        for (const auto &bin : bins)
            upper_bound += bin.size();
        out.entries.reserve(upper_bound);
        if (profile)
            profile->reserve_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reserve_start).count();

        const auto frontier_start = profile ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        std::priority_queue<Cursor, std::vector<Cursor>, decltype(later)> queue(later);
        for (std::size_t bin = 0; bin < bins.size(); ++bin)
            if (!bins[bin].empty())
                queue.push({bin, 0});
        if (profile)
            profile->frontier_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - frontier_start).count();

        const auto stream_start = profile ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
        while (!queue.empty())
        {
            const Cursor first = queue.top();
            const DetKey key = bins[first.bin].at(first.index).first;
            Weight sum = 0.0;
            do
            {
                const Cursor current = queue.top();
                queue.pop();
                sum += bins[current.bin].at(current.index).second;
                const std::size_t next = current.index + 1;
                if (next < bins[current.bin].size())
                    queue.push({current.bin, next});
            }
            while (!queue.empty()
                   && queue.top().bin < bins.size()
                   && bins[queue.top().bin].at(queue.top().index).first == key);
            out.entries.push_back({key, sum});
        }
        if (profile)
        {
            profile->stream_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - stream_start).count();
            profile->emitted += out.entries.size();
        }
        return out;
    }

    // Production implementation for the canonical merge. A balanced pairwise
    // tree replaces the priority queue's O(log bins) maintenance on every
    // emitted record with sequential two-way merges. Every record carries its
    // source bin through the tree. Equal keys MUST NOT fold at an intermediate
    // level: for four bins, `(w0 + w1) + (w2 + w3)` changes the established
    // left-fold `((w0 + w1) + w2) + w3`. The final stream is ordered by
    // (key, bin) and folds only once, exactly matching the heap path.
    namespace detail
    {
        struct TaggedSpawnEntry
        {
            DetKey key;
            Weight weight = 0.0;
            std::size_t bin = 0;
        };

        inline bool less_tagged_spawn_entry(const TaggedSpawnEntry &a,
                                            const TaggedSpawnEntry &b) noexcept
        {
            if (a.key.alpha != b.key.alpha)
                return a.key.alpha < b.key.alpha;
            if (a.key.beta != b.key.beta)
                return a.key.beta < b.key.beta;
            return a.bin < b.bin;
        }

        inline std::vector<TaggedSpawnEntry>
        merge_tagged_spawn_streams(std::vector<TaggedSpawnEntry> left,
                                   std::vector<TaggedSpawnEntry> right)
        {
            std::vector<TaggedSpawnEntry> merged;
            merged.reserve(left.size() + right.size());
            std::size_t i = 0, j = 0;
            while (i < left.size() && j < right.size())
            {
                if (less_tagged_spawn_entry(left[i], right[j]))
                    merged.push_back(std::move(left[i++]));
                else
                    merged.push_back(std::move(right[j++]));
            }
            while (i < left.size())
                merged.push_back(std::move(left[i++]));
            while (j < right.size())
                merged.push_back(std::move(right[j++]));
            return merged;
        }

        inline CanonicalSpawnMerge merge_finalized_spawn_bins_pairwise_impl(
            const std::vector<SpawnAccumulator> &bins,
            bool parallel_levels,
            PairwiseSpawnMergeProfile *profile)
        {
            if (profile)
                profile->tagged_entry_bytes = sizeof(TaggedSpawnEntry);
            std::vector<std::vector<TaggedSpawnEntry>> level;
            level.reserve(bins.size());
            for (std::size_t bin = 0; bin < bins.size(); ++bin)
            {
                auto &entries = level.emplace_back();
                entries.reserve(bins[bin].size());
                for (const auto &[key, weight] : bins[bin])
                    entries.push_back({key, weight, bin});
            }
            while (level.size() > 1)
            {
                const std::size_t current_entries = [&level] {
                    std::size_t n = 0;
                    for (const auto &stream : level)
                        n += stream.size();
                    return n;
                }();
                const std::size_t next_count = (level.size() + 1) / 2;
                // Every merge is a permutation, so the next level has exactly
                // current_entries tagged records. The two simultaneously live
                // levels are the merge's complete tagged-entry high-water mark.
                if (profile)
                {
                    ++profile->levels;
                    profile->peak_live_entries = std::max(
                        profile->peak_live_entries, 2 * current_entries);
                    profile->peak_live_bytes = std::max(
                        profile->peak_live_bytes,
                        2 * current_entries * sizeof(TaggedSpawnEntry));
                }

                std::vector<std::vector<TaggedSpawnEntry>> next(next_count);
                const auto merge_pair = [&level, &next](std::size_t pair) {
                    const std::size_t left = 2 * pair;
                    if (left + 1 == level.size())
                        next[pair] = std::move(level[left]);
                    else
                        next[pair] = merge_tagged_spawn_streams(
                            std::move(level[left]), std::move(level[left + 1]));
                };
                if (parallel_levels)
                {
#pragma omp parallel for schedule(static)
                    for (std::ptrdiff_t pair = 0;
                         pair < static_cast<std::ptrdiff_t>(next_count); ++pair)
                        merge_pair(static_cast<std::size_t>(pair));
                }
                else
                {
                    for (std::size_t pair = 0; pair < next_count; ++pair)
                        merge_pair(pair);
                }
                level = std::move(next);
            }

            CanonicalSpawnMerge out;
            if (!level.empty())
            {
                const auto &stream = level.front();
                if (profile)
                {
                    profile->peak_live_entries = std::max(
                        profile->peak_live_entries, stream.size() * 2);
                    profile->peak_live_bytes = std::max(
                        profile->peak_live_bytes,
                        stream.size() * 2 * sizeof(TaggedSpawnEntry));
                }
                out.entries.reserve(stream.size());
                std::size_t i = 0;
                while (i < stream.size())
                {
                    const DetKey key = stream[i].key;
                    Weight sum = stream[i].weight;
                    ++i;
                    while (i < stream.size() && stream[i].key == key)
                        sum += stream[i++].weight;
                    out.entries.push_back({key, sum});
                }
            }
            return out;
        }
    } // namespace detail

    // Retained sequential tree oracle. It remains directly callable by tests
    // after the level-parallel implementation becomes production, so a future
    // change cannot silently alter the reduction association.
    inline CanonicalSpawnMerge merge_finalized_spawn_bins_pairwise(
        const std::vector<SpawnAccumulator> &bins,
        PairwiseSpawnMergeProfile *profile = nullptr)
    {
        return detail::merge_finalized_spawn_bins_pairwise_impl(
            bins, false, profile);
    }

    // M1 production tree. It merges independent pairs at a tree level in
    // parallel, but writes every predetermined pair result to its own slot.
    // Thus OpenMP schedule cannot alter either the merge tree or equal-key
    // addition association. The sequential entry point above is its oracle.
    inline CanonicalSpawnMerge merge_finalized_spawn_bins_pairwise_parallel(
        const std::vector<SpawnAccumulator> &bins,
        PairwiseSpawnMergeProfile *profile = nullptr)
    {
        return detail::merge_finalized_spawn_bins_pairwise_impl(
            bins, true, profile);
    }

    // H2.4/H2.5 (docs/FCIQMC_PARALLELISM.md): the persistent
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

        // H2.8.4-b: when true, `parents` was filled by the CALLER (the driver's
        // fused per-step pre-pass) and propagate_stochastic must NOT run its own
        // partition loop or clear `parents`. Reset to false by clear_output(),
        // so a caller that does not prefill gets the old self-contained
        // behaviour. The convenience overload never prefills.
        bool parents_prefilled = false;

        void ready(std::size_t n_bins)
        {
            if (bins.size() != n_bins)
            {
                bins.assign(n_bins, SpawnAccumulator{});
                parents.assign(n_bins, {});
                bin_rngs.assign(n_bins, RandomSource{0});
            }
        }

        // Clear only the spawn OUTPUT (per-bin accumulators). Leaves `parents`
        // untouched so a caller can fill it before calling propagate_stochastic.
        void clear_output()
        {
            for (auto &b : bins)
                b.reset();
            parents_prefilled = false;
        }

        // Clear everything, including `parents`. propagate_stochastic calls this
        // only when the caller did NOT prefill.
        void reset_for_call()
        {
            clear_output();
            clear_parents();
        }

        void clear_parents()
        {
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
