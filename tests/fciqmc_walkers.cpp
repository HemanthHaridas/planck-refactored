// F1 gate: the FCIQMC walker container and RNG policy.
//
// No dynamics here -- this tests the state layer only. The two properties that
// matter are that ANNIHILATION falls out of signed accumulation, and that the
// RNG honours its bitwise reproducibility contract.

#include "post_hf/ci/fciqmc.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <map>
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

static void check_close(double a, double b, double tol, const char *what)
{
    if (!(std::abs(a - b) <= tol))
    {
        std::printf("  [FAIL] %s (got %.12g, want %.12g)\n", what, a, b);
        ++g_failures;
    }
}

static DetKey det(CIString a, CIString b) { return DetKey{a, b}; }

static void test_population_basics()
{
    WalkerPopulation pop;
    check(pop.empty(), "a fresh population is empty");

    pop.add(det(0b0011, 0b0011), 3.0);
    pop.add(det(0b0101, 0b0011), -2.0);
    check(pop.size() == 2, "two distinct determinants are two entries");
    check_close(pop.weight_at(det(0b0011, 0b0011)), 3.0, 0.0, "weight round-trips");
    check_close(pop.weight_at(det(0b1111, 0b1111)), 0.0, 0.0, "absent determinant reads 0");

    // Adding zero must not materialize an entry -- otherwise every determinant
    // ever considered for a spawn would become resident.
    pop.add(det(0b1001, 0b0011), 0.0);
    check(pop.size() == 2, "adding zero weight creates no entry");

    check_close(pop.total_population(), 5.0, 1e-15, "total population is sum of |w|");
}

static void test_annihilation()
{
    // The core claim: annihilation is not a separate pass, it is what accumulating
    // signed weights into a determinant-keyed map already does.
    WalkerPopulation pop;
    const DetKey d = det(0b0011, 0b0011);

    pop.add(d, 5.0);
    pop.add(d, -3.0);
    check_close(pop.weight_at(d), 2.0, 1e-15, "opposite signs cancel partially");

    pop.add(d, -2.0);
    check_close(pop.weight_at(d), 0.0, 1e-15, "exact cancellation reaches zero");
    check(pop.size() == 1, "a cancelled determinant is still resident until compressed");

    const std::size_t removed = pop.compress();
    check(removed == 1, "compress removes the cancelled determinant");
    check(pop.empty(), "population is empty after compressing an exact zero");
}

static void test_compression_threshold()
{
    WalkerPopulation pop;
    pop.add(det(1, 1), 1.0);
    pop.add(det(2, 1), 1e-14);
    pop.add(det(3, 1), -1e-14);

    check(pop.compress(0.0) == 0, "threshold 0 keeps tiny but nonzero weights");
    check(pop.size() == 3, "nothing removed at threshold 0");

    check(pop.compress(1e-12) == 2, "threshold removes both tiny weights");
    check(pop.size() == 1, "only the significant determinant survives");
    check_close(pop.weight_at(det(1, 1)), 1.0, 0.0, "the survivor is untouched");
}

static void test_hash_spreads_determinants()
{
    // Determinant bitstrings have fixed popcount and are dense in the low bits.
    // std::hash<uint64_t> is the identity on libstdc++, so a naive combine piles
    // them into few buckets. Assert the hash actually distinguishes the cases that
    // an identity hash would collide or cluster.
    DetKeyHash h;
    check(h(det(0b0011, 0b0101)) != h(det(0b0101, 0b0011)),
          "hash is not symmetric in alpha/beta (a<->b must differ)");

    std::vector<std::size_t> hashes;
    for (CIString a = 1; a <= 64; ++a)
        hashes.push_back(h(det(a, a * 3 + 1)));
    std::map<std::size_t, int> seen;
    for (auto v : hashes)
        seen[v % 64]++;
    int worst = 0;
    for (const auto &[bucket, n] : seen)
        worst = std::max(worst, n);
    check(worst <= 6, "hash spreads 64 determinants across 64 buckets without heavy clustering");
}

static void test_rng_reproducibility()
{
    // The contract: same seed -> same sequence, bitwise.
    RandomSource a(12345), b(12345);
    for (int i = 0; i < 1000; ++i)
    {
        const double x = a.uniform();
        const double y = b.uniform();
        if (x != y)
        {
            check(false, "same seed must reproduce the draw sequence bitwise");
            break;
        }
    }

    RandomSource c(12345), d(54321);
    bool differs = false;
    for (int i = 0; i < 100; ++i)
        if (c.uniform() != d.uniform())
        {
            differs = true;
            break;
        }
    check(differs, "different seeds must give different sequences");

    // Derived streams are deterministic in the run seed and independent of how
    // many are created -- this is what keeps a threaded run thread-count-invariant.
    RandomSource root(999);
    RandomSource s2a = root.derive(2);
    RandomSource s2b = root.derive(2);
    check(s2a.seed() == s2b.seed(), "derive() is deterministic for a given index");
    check(root.derive(0).seed() != root.derive(1).seed(),
          "different shard indices give different streams");

    // Deriving shard 2 must not depend on whether shards 0,1,3 were also derived.
    RandomSource root2(999);
    (void)root2.derive(0);
    (void)root2.derive(1);
    check(root2.derive(2).seed() == s2a.seed(),
          "a shard's stream must not depend on how many shards were derived");
}

static void test_uniform_range()
{
    RandomSource rng(7);
    for (int i = 0; i < 10000; ++i)
    {
        const double u = rng.uniform();
        if (!(u >= 0.0 && u < 1.0))
        {
            check(false, "uniform() must lie in [0,1)");
            break;
        }
    }
    std::vector<int> counts(8, 0);
    for (int i = 0; i < 80000; ++i)
        counts[static_cast<std::size_t>(rng.uniform_int(8))]++;
    for (int k = 0; k < 8; ++k)
        check(counts[k] > 8000 && counts[k] < 12000, "uniform_int is roughly uniform");
}

static void test_stochastic_rounding_is_unbiased()
{
    // Stochastic rounding is what lets a fractional spawn be realized without
    // bias. Rounding to nearest would systematically discard small weights, which
    // biases the energy -- so the unbiasedness is load-bearing, not cosmetic.
    RandomSource rng(2024);
    const double x = 2.3;
    double sum = 0.0;
    const int n = 200000;
    for (int i = 0; i < n; ++i)
    {
        const double r = rng.stochastic_round(x);
        const double frac = r - std::floor(r);
        if (frac != 0.0)
        {
            check(false, "stochastic_round must return an integer");
            break;
        }
        if (r != 2.0 && r != 3.0)
        {
            check(false, "stochastic_round(2.3) must return 2 or 3");
            break;
        }
        sum += r;
    }
    check_close(sum / n, x, 0.01, "stochastic rounding is unbiased in expectation");

    // Negative values round toward the same expectation, preserving sign.
    double neg = 0.0;
    for (int i = 0; i < 200000; ++i)
        neg += rng.stochastic_round(-2.3);
    check_close(neg / 200000.0, -2.3, 0.01, "stochastic rounding is unbiased for negatives");

    // An exact integer must round to itself, always -- otherwise a full-weight
    // walker would randomly gain or lose weight for no reason.
    for (int i = 0; i < 1000; ++i)
        if (rng.stochastic_round(3.0) != 3.0)
        {
            check(false, "an exact integer must round to itself");
            break;
        }
}

// ---------------------------------------------------------------------------
// F2.1 -- the brute-force connection oracle
// ---------------------------------------------------------------------------

static DetKey make_det(int n_occ_a, int n_occ_b)
{
    CIString a = 0, b = 0;
    for (int i = 0; i < n_occ_a; ++i)
        a |= (CIString{1} << i);
    for (int i = 0; i < n_occ_b; ++i)
        b |= (CIString{1} << i);
    return DetKey{a, b};
}

static void test_oracle_counts()
{
    // Counts verified independently by brute-force enumeration in Python before
    // being written here (see docs/FCIQMC_F2_EXCITATION_SCOPE.md). A Slater-Condon
    // element vanishes beyond a double excitation, so these ARE all the
    // determinants the Hamiltonian connects to the parent.
    struct Case { int n_act, na, nb, expected; const char *name; };
    const Case cases[] = {
        {2, 1, 1, 3, "H2/STO-3G"},
        {7, 5, 5, 140, "water/STO-3G"},
        {10, 7, 7, 609, "N2/STO-3G"},
    };
    for (const auto &c : cases)
    {
        const auto conns = enumerate_connections(make_det(c.na, c.nb), c.n_act);
        if (static_cast<int>(conns.size()) != c.expected)
        {
            std::printf("  [FAIL] %s: %zu connections, expected %d\n",
                        c.name, conns.size(), c.expected);
            ++g_failures;
        }
    }
}

static void test_oracle_no_duplicates_and_excludes_parent()
{
    const DetKey parent = make_det(5, 5);
    const auto conns = enumerate_connections(parent, 7);

    std::map<std::pair<CIString, CIString>, int> seen;
    for (const auto &e : conns)
        seen[{e.det.alpha, e.det.beta}]++;

    check(seen.size() == conns.size(), "oracle emits no duplicate determinants");
    check(seen.find({parent.alpha, parent.beta}) == seen.end(),
          "oracle excludes the parent determinant itself");
}

static void test_oracle_preserves_particle_number()
{
    // Every connection must have the same alpha and beta electron count as the
    // parent: the Hamiltonian conserves particle number in each spin channel.
    const DetKey parent = make_det(5, 5);
    const auto conns = enumerate_connections(parent, 7);
    const int na = std::popcount(parent.alpha);
    const int nb = std::popcount(parent.beta);
    for (const auto &e : conns)
        if (std::popcount(e.det.alpha) != na || std::popcount(e.det.beta) != nb)
        {
            check(false, "every connection preserves per-spin electron count");
            break;
        }
}

static void test_oracle_excitation_rank()
{
    // Every connection differs from the parent by exactly 1 or 2 spin-orbitals.
    // Rank 0 would be the parent (excluded); rank > 2 cannot be connected.
    const DetKey parent = make_det(5, 5);
    const auto conns = enumerate_connections(parent, 7);
    for (const auto &e : conns)
    {
        const int rank = (std::popcount(e.det.alpha ^ parent.alpha)
                          + std::popcount(e.det.beta ^ parent.beta)) / 2;
        if (rank < 1 || rank > 2)
        {
            check(false, "every connection is a single or double excitation");
            break;
        }
    }
}

static void test_oracle_phase_is_a_sign()
{
    const auto conns = enumerate_connections(make_det(5, 5), 7);
    for (const auto &e : conns)
        if (e.phase != 1.0 && e.phase != -1.0)
        {
            check(false, "phase is exactly +1 or -1");
            break;
        }
    bool saw_neg = false;
    for (const auto &e : conns)
        if (e.phase < 0.0)
            saw_neg = true;
    check(saw_neg, "some connections carry a negative phase (else parity is ignored)");
}

static void test_operator_convention_matches_shared()
{
    // fciqmc.cpp reimplements the annihilate-then-create phase from the
    // header-inline bit helpers, to avoid dragging the symmetry/basis dependency
    // chain into this layer. That is only safe if it agrees EXACTLY with the
    // shared convention in strings.cpp -- so assert the parity rule directly here
    // rather than trusting the comment.
    //
    // Convention: the sign is -1 when an odd number of occupied orbitals lie
    // strictly below the acted-on orbital.
    for (int n_act = 2; n_act <= 6; ++n_act)
    {
        const DetKey parent = make_det(n_act / 2, n_act / 2);
        const auto conns = enumerate_connections(parent, n_act);
        for (const auto &e : conns)
        {
            // For a single alpha excitation i->a, the phase must equal
            // (-1)^(number of occupied orbitals strictly between i and a).
            //
            // The filter must require the BETA string to be unchanged as well.
            // An opposite-spin double changes alpha by two bits *and* beta by
            // two, so testing popcount(alpha_diff) == 2 alone would apply the
            // single-excitation rule to a double, whose phase is a product of
            // two such factors. That mis-filter produced a spurious failure
            // here before it was caught -- the implementation was right and the
            // test was wrong.
            if (e.det.beta != parent.beta)
                continue; // beta moved: not a pure alpha single
            const CIString diff = e.det.alpha ^ parent.alpha;
            if (std::popcount(diff) != 2)
                continue; // not a single alpha excitation
            const CIString removed = diff & parent.alpha;
            const CIString added = diff & e.det.alpha;
            const int lo = std::countr_zero(std::min(removed, added));
            const int hi = std::countr_zero(std::max(removed, added));
            CIString between = 0;
            for (int p = lo + 1; p < hi; ++p)
                between |= (CIString{1} << p);
            const int n_between = std::popcount(parent.alpha & between);
            const double want = (n_between % 2 == 0) ? 1.0 : -1.0;
            if (e.phase != want)
            {
                std::printf("  [FAIL] single-excitation phase convention "
                            "(n_act=%d, got %g, want %g)\n", n_act, e.phase, want);
                ++g_failures;
                return;
            }
        }
    }
}

int main()
{
    std::printf("F1 -- FCIQMC walker container and RNG policy\n");
    test_population_basics();
    test_annihilation();
    test_compression_threshold();
    test_hash_spreads_determinants();
    test_rng_reproducibility();
    test_uniform_range();
    test_stochastic_rounding_is_unbiased();

    std::printf("F2.1 -- brute-force connection oracle\n");
    test_oracle_counts();
    test_oracle_no_duplicates_and_excludes_parent();
    test_oracle_preserves_particle_number();
    test_oracle_excitation_rank();
    test_oracle_phase_is_a_sign();
    test_operator_convention_matches_shared();

    if (g_failures == 0)
    {
        std::printf("All checks passed.\n");
        return 0;
    }
    std::printf("%d FAILURE(S).\n", g_failures);
    return 1;
}
