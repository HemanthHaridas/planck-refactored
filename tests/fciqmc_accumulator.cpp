// H2.2 (docs/FCIQMC_PARALLEL_REWRITE_SCOPE.md): unit test for SpawnAccumulator,
// the reuse-stable replacement for the FCIQMC spawn loop's per-bin
// std::unordered_map<DetKey, Weight>.
//
// The two properties it must have, and unordered_map does not:
//
//   1. same multiset of (det, weight) adds, ANY insertion order
//        -> BITWISE-identical finalized sum
//   2. BITWISE-identical after N reset()/refill cycles with the same multiset
//
// Both are checked against an independent std::map<...>-based reference that
// sums each key's contributions in sorted-bit-pattern order -- the same
// canonical fold SpawnAccumulator does, written a second, different way (a
// std::multimap grouped by key, not a sort-and-scan over a flat vector).
//
// Non-vacuity: a mutation-style negative test builds a "naive" reference that
// sums in insertion order and shows it DISAGREES on a >=3-long run, proving the
// canonical fold is doing real work.

#include "post_hf/ci/spawn_accumulator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <utility>
#include <vector>

using namespace HartreeFock::Correlation::CI::QMC;
using HartreeFock::Correlation::CASSCFInternal::CIString;

static int g_failures = 0;

static void check(bool cond, const char *what)
{
    if (!cond)
    {
        std::printf("  [FAIL] %s\n", what);
        ++g_failures;
    }
}

static DetKey det(CIString a, CIString b) { return DetKey{a, b}; }

// The independent reference: group by key with a std::map, and within each key
// fold contributions in ascending raw-bit-pattern order -- SpawnAccumulator's
// canonical order, arrived at by a different mechanism (a per-key vector that is
// sorted here, vs. a global sort-and-scan there).
struct RefKeyLess
{
    bool operator()(const DetKey &a, const DetKey &b) const noexcept
    {
        if (a.alpha != b.alpha)
            return a.alpha < b.alpha;
        return a.beta < b.beta;
    }
};

static std::vector<std::pair<DetKey, double>>
reference_canonical(const std::vector<std::pair<DetKey, double>> &adds)
{
    std::map<DetKey, std::vector<double>, RefKeyLess> grouped;
    for (const auto &[d, w] : adds)
        grouped[d].push_back(w);

    std::vector<std::pair<DetKey, double>> out;
    for (auto &[d, ws] : grouped)
    {
        std::sort(ws.begin(), ws.end(), [](double x, double y) {
            return std::bit_cast<std::uint64_t>(x) < std::bit_cast<std::uint64_t>(y);
        });
        double s = 0.0;
        for (double w : ws)
            s += w;
        out.push_back({d, s});
    }
    return out;
}

// The NAIVE reference: fold in insertion order. Used only to prove the
// canonical fold matters -- it must DISAGREE with the canonical one on a
// multiset containing a >=3-long same-key run whose insertion order is not
// already the bit-pattern order.
static std::vector<std::pair<DetKey, double>>
reference_insertion_order(const std::vector<std::pair<DetKey, double>> &adds)
{
    std::map<DetKey, double, RefKeyLess> grouped;
    for (const auto &[d, w] : adds)
        grouped[d] += w; // += in first-seen-per-key insertion order
    std::vector<std::pair<DetKey, double>> out(grouped.begin(), grouped.end());
    return out;
}

static bool bitwise_equal(const std::vector<std::pair<DetKey, double>> &a,
                          const std::vector<std::pair<DetKey, double>> &b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (!(a[i].first == b[i].first))
            return false;
        if (std::bit_cast<std::uint64_t>(a[i].second)
            != std::bit_cast<std::uint64_t>(b[i].second))
            return false;
    }
    return true;
}

static std::vector<std::pair<DetKey, double>>
drain(SpawnAccumulator &acc)
{
    acc.finalize();
    return {acc.begin(), acc.end()};
}

// Build a multiset with real structure: many distinct keys, several of them
// with runs of 3..6 same-key adds whose weights span a wide dynamic range (so
// reassociation is observable), plus exact-cancellation pairs (annihilation).
static std::vector<std::pair<DetKey, double>> make_multiset(std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::vector<std::pair<DetKey, double>> adds;

    auto rnd_key = [&]() {
        // Popcount-6 spin strings in the low 12 bits, like a real determinant.
        CIString a = 0, b = 0;
        int na = 0, nb = 0;
        while (na < 6) { int p = rng() % 12; if (!(a & (1ull << p))) { a |= 1ull << p; ++na; } }
        while (nb < 6) { int p = rng() % 12; if (!(b & (1ull << p))) { b |= 1ull << p; ++nb; } }
        return det(a, b);
    };

    for (int k = 0; k < 40; ++k)
    {
        const DetKey key = rnd_key();
        const int run = 1 + int(rng() % 6); // 1..6
        for (int r = 0; r < run; ++r)
        {
            // magnitudes from ~1 down to ~1e-9, mixed signs
            const double mag = std::pow(10.0, -9.0 * double(rng() % 1000) / 1000.0);
            const double w = (rng() & 1 ? 1.0 : -1.0) * mag;
            adds.push_back({key, w});
        }
        // an exact-cancellation pair on this key
        adds.push_back({key, 12345.678});
        adds.push_back({key, -12345.678});
    }
    return adds;
}

// Property 1: same multiset, any insertion order -> bitwise-identical sum.
static void test_insertion_order_invariance()
{
    std::mt19937_64 shuffler(0xABCDEF);
    for (std::uint64_t seed = 1; seed <= 20; ++seed)
    {
        auto base = make_multiset(seed);
        auto ref = reference_canonical(base);

        // permute the ADD ORDER five different ways, all the same multiset
        for (int perm = 0; perm < 5; ++perm)
        {
            auto shuffled = base;
            std::shuffle(shuffled.begin(), shuffled.end(), shuffler);

            SpawnAccumulator acc;
            for (const auto &[d, w] : shuffled)
                acc.add(d, w);
            const auto got = drain(acc);

            check(bitwise_equal(got, ref),
                  "finalized sum is bitwise-identical regardless of add order");
        }
    }
}

// Property 2: bitwise-identical after N reset()/refill cycles with the SAME
// multiset -- the property unordered_map fails because clear() keeps the peak
// bucket count. Also interleave a DIFFERENT, larger multiset between cycles so
// the accumulator's capacity grows and then a smaller refill follows (exactly
// the shape that made a reused unordered_map iterate differently).
static void test_reuse_stability()
{
    auto small = make_multiset(7);
    auto big = make_multiset(99);
    // make `big` genuinely bigger so capacity grows before a `small` refill
    auto big2 = make_multiset(100);
    big.insert(big.end(), big2.begin(), big2.end());

    const auto ref_small = reference_canonical(small);

    SpawnAccumulator acc;

    // cycle 0: baseline
    for (const auto &[d, w] : small) acc.add(d, w);
    const auto cycle0 = drain(acc);
    check(bitwise_equal(cycle0, ref_small), "reuse cycle 0 matches the reference");

    for (int cycle = 1; cycle <= 10; ++cycle)
    {
        // grow the accumulator with a bigger multiset...
        acc.reset();
        for (const auto &[d, w] : big) acc.add(d, w);
        (void)drain(acc);

        // ...then refill with the SMALL multiset and require the same bytes
        acc.reset();
        for (const auto &[d, w] : small) acc.add(d, w);
        const auto got = drain(acc);
        check(bitwise_equal(got, cycle0),
              "small refill after a large one is bitwise-identical to cycle 0");
    }
}

// finalize() is idempotent and does not change the sum on a second call.
static void test_finalize_idempotent()
{
    auto adds = make_multiset(3);
    SpawnAccumulator acc;
    for (const auto &[d, w] : adds) acc.add(d, w);

    acc.finalize();
    const std::vector<std::pair<DetKey, double>> once(acc.begin(), acc.end());
    acc.finalize();
    const std::vector<std::pair<DetKey, double>> twice(acc.begin(), acc.end());
    check(bitwise_equal(once, twice), "finalize() is idempotent");
}

// Non-vacuity: the canonical fold must actually differ from an insertion-order
// fold on a multiset with a >=3-long run. If make_multiset ever stopped
// producing such a run, this check would silently pass while property 1 became
// untestable -- so assert the disagreement exists.
static void test_canonical_fold_is_load_bearing()
{
    bool saw_disagreement = false;
    for (std::uint64_t seed = 1; seed <= 50 && !saw_disagreement; ++seed)
    {
        auto adds = make_multiset(seed);
        const auto canon = reference_canonical(adds);
        const auto naive = reference_insertion_order(adds);
        // same keys, but the summed weights differ on at least one key
        if (canon.size() == naive.size())
        {
            for (std::size_t i = 0; i < canon.size(); ++i)
            {
                if (std::bit_cast<std::uint64_t>(canon[i].second)
                    != std::bit_cast<std::uint64_t>(naive[i].second))
                {
                    saw_disagreement = true;
                    break;
                }
            }
        }
    }
    check(saw_disagreement,
          "canonical fold differs from insertion-order fold (the fold is load-bearing)");
}

// A tiny hand-built case: three adds to one key, out of bit-pattern order,
// checked against a hand-computed canonical sum.
static void test_hand_case()
{
    const DetKey d = det(0b111111, 0b111111);
    const double a = 1.0;
    const double b = 1e-16;   // lost entirely if added to `a` first
    const double c = -1.0;

    // canonical order by raw bits: for positive doubles bit pattern is
    // monotonic in value, so ascending-bits order is b(1e-16) < a(1.0), and c
    // is negative so its raw bit pattern (sign bit set) is the LARGEST unsigned.
    // Fold: (b + a) + c  ==  (1e-16 + 1.0) + (-1.0)  ==  1.0 + (-1.0) == 0.0
    // (the 1e-16 is lost in the first add).
    const double expected = (b + a) + c;

    SpawnAccumulator acc;
    acc.add(d, a);
    acc.add(d, b);
    acc.add(d, c);
    acc.finalize();
    check(acc.size() == 1, "one key after folding three adds");
    const double got = acc.begin()->second;
    check(std::bit_cast<std::uint64_t>(got) == std::bit_cast<std::uint64_t>(expected),
          "hand case folds in canonical (ascending-bit) order");
}

int main()
{
    std::printf("H2.2 -- SpawnAccumulator reuse-stable accumulator\n");
    test_insertion_order_invariance();
    test_reuse_stability();
    test_finalize_idempotent();
    test_canonical_fold_is_load_bearing();
    test_hand_case();

    if (g_failures == 0)
    {
        std::printf("All checks passed.\n");
        return 0;
    }
    std::printf("%d FAILURE(S).\n", g_failures);
    return 1;
}
