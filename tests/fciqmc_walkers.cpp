// F1 gate: the FCIQMC walker container and RNG policy.
//
// No dynamics here -- this tests the state layer only. The two properties that
// matter are that ANNIHILATION falls out of signed accumulation, and that the
// RNG honours its bitwise reproducibility contract.

#include "post_hf/ci/fciqmc.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <algorithm>
#include <map>
#include <tuple>
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
    // being written here (see docs/FCIQMC_SAMPLING_AND_DYNAMICS.md). A Slater-Condon
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

// ---------------------------------------------------------------------------
// F2.2 -- the slow uniform generator
// ---------------------------------------------------------------------------

static void test_uniform_generator_support()
{
    // SUPPORT: the generator must reach every connection the oracle knows about,
    // and never produce one it does not.
    //
    // For THIS generator the support check is partly redundant: it is uniform, so
    // a support hole necessarily redistributes probability and the frequency test
    // sees it too (measured: dropping one of 140 connections shows up at ~54
    // sigma). The check is kept because its necessity appears at F2.3, where the
    // generator is WEIGHTED: a connection with p_gen ~ 1e-6 that is never
    // generated deviates by ~0.6 sigma over 400k draws, which no frequency test
    // will ever flag. Support and frequency are different failure modes, and the
    // difference only bites once p_gen stops being constant.
    const DetKey parent = make_det(5, 5);
    const int n_act = 7;
    const auto conns = enumerate_connections(parent, n_act);

    std::map<std::pair<CIString, CIString>, int> oracle;
    for (const auto &e : conns)
        oracle[{e.det.alpha, e.det.beta}] = 0;

    RandomSource rng(4242);
    std::map<std::pair<CIString, CIString>, int> seen;
    for (int i = 0; i < 200000; ++i)
    {
        const auto e = draw_uniform_excitation(parent, n_act, rng);
        if (!e.valid)
        {
            check(false, "generator returned an invalid excitation on a connected parent");
            return;
        }
        seen[{e.det.alpha, e.det.beta}]++;
    }

    check(seen.size() == oracle.size(),
          "generator reaches exactly as many distinct determinants as the oracle");
    for (const auto &[key, count] : seen)
        if (oracle.find(key) == oracle.end())
        {
            check(false, "generator produced a determinant the oracle says is unconnected");
            return;
        }
    for (const auto &[key, unused] : oracle)
        if (seen.find(key) == seen.end())
        {
            check(false, "generator never reached a determinant the oracle lists");
            return;
        }
}

static void test_uniform_generator_frequencies()
{
    // FREQUENCY: the empirical frequency of each connection must match its
    // returned p_gen. This is the assertion that catches a mis-reported p_gen --
    // the failure mode that produces a plausible, converged, WRONG energy.
    const DetKey parent = make_det(5, 5);
    const int n_act = 7;
    const auto conns = enumerate_connections(parent, n_act);
    const std::size_t n_conn = conns.size();

    RandomSource rng(31337);
    const int n_draws = 400000;
    std::map<std::pair<CIString, CIString>, int> counts;
    double p_gen_seen = -1.0;
    for (int i = 0; i < n_draws; ++i)
    {
        const auto e = draw_uniform_excitation(parent, n_act, rng);
        counts[{e.det.alpha, e.det.beta}]++;
        if (p_gen_seen < 0.0)
            p_gen_seen = e.p_gen;
        else if (e.p_gen != p_gen_seen)
        {
            check(false, "uniform generator must report a constant p_gen");
            return;
        }
    }

    check_close(p_gen_seen, 1.0 / static_cast<double>(n_conn), 1e-15,
                "reported p_gen equals 1/|connections|");

    // sum(p_gen) over the connection set is 1: the distribution is normalized.
    check_close(p_gen_seen * static_cast<double>(n_conn), 1.0, 1e-12,
                "p_gen sums to 1 over the connection set");

    // Each connection should appear n_draws * p_gen times, +/- sampling error.
    // Binomial sigma = sqrt(N p (1-p)); require every bin within 5 sigma, which
    // for 140 bins is a ~1-in-3000 false-failure rate overall.
    const double expected = n_draws * p_gen_seen;
    const double sigma = std::sqrt(n_draws * p_gen_seen * (1.0 - p_gen_seen));
    int worst_bin = 0;
    double worst_dev = 0.0;
    for (const auto &[key, c] : counts)
    {
        const double dev = std::abs(c - expected) / sigma;
        if (dev > worst_dev)
        {
            worst_dev = dev;
            worst_bin = c;
        }
    }
    if (!(worst_dev <= 5.0))
    {
        std::printf("  [FAIL] frequency does not match p_gen: worst bin %d, "
                    "expected %.1f +/- %.1f (%.1f sigma)\n",
                    worst_bin, expected, sigma, worst_dev);
        ++g_failures;
    }
}

static void test_uniform_generator_h2_exact()
{
    // H2/STO-3G has exactly 3 connections, so this case needs no statistical
    // argument at all: all three must appear, each about a third of the time.
    const DetKey parent = make_det(1, 1);
    const auto conns = enumerate_connections(parent, 2);
    check(conns.size() == 3, "H2/STO-3G parent has 3 connections");

    RandomSource rng(7);
    std::map<std::pair<CIString, CIString>, int> counts;
    const int n = 60000;
    bool p_gen_ok = true;
    for (int i = 0; i < n; ++i)
    {
        const auto e = draw_uniform_excitation(parent, 2, rng);
        // Report a p_gen mismatch ONCE, not once per draw: a check inside a
        // 60k-iteration loop turns a single defect into megabytes of output and
        // buries every other failure.
        if (p_gen_ok && std::abs(e.p_gen - 1.0 / 3.0) > 1e-15)
        {
            check_close(e.p_gen, 1.0 / 3.0, 1e-15, "H2 p_gen is exactly 1/3");
            p_gen_ok = false;
        }
        counts[{e.det.alpha, e.det.beta}]++;
    }
    check(counts.size() == 3, "all three H2 connections are reached");
    bool bins_ok = true;
    for (const auto &[key, c] : counts)
        if (!(c > n / 3 - 1500 && c < n / 3 + 1500))
            bins_ok = false;
    check(bins_ok, "each H2 connection appears about a third of the time");
}

static void test_uniform_generator_reproducible()
{
    // The generator draws through RandomSource, so it inherits F1's contract:
    // the same seed must reproduce the same sequence of excitations bitwise.
    const DetKey parent = make_det(5, 5);
    RandomSource a(999), b(999);
    for (int i = 0; i < 5000; ++i)
    {
        const auto ea = draw_uniform_excitation(parent, 7, a);
        const auto eb = draw_uniform_excitation(parent, 7, b);
        if (ea.det.alpha != eb.det.alpha || ea.det.beta != eb.det.beta
            || ea.phase != eb.phase || ea.p_gen != eb.p_gen)
        {
            check(false, "same seed must reproduce the excitation sequence bitwise");
            return;
        }
    }
}

static void test_uniform_generator_degenerate()
{
    // A one-determinant space has no connections; the generator must say so
    // rather than returning a garbage excitation.
    const DetKey parent = make_det(1, 1);
    RandomSource rng(1);
    const auto e = draw_uniform_excitation(parent, 1, rng);
    check(!e.valid, "a parent with no connections yields an invalid excitation");
}

// ---------------------------------------------------------------------------
// F2.3 -- the O(1) production generator
// ---------------------------------------------------------------------------

struct SampleStats
{
    std::map<std::pair<CIString, CIString>, int> counts;
    std::map<std::pair<CIString, CIString>, double> p_gen;
    int n_draws = 0;
    int n_invalid = 0;
};

static SampleStats sample_generator(const DetKey &parent, int n_act, int n_draws,
                                    std::uint64_t seed)
{
    SampleStats st;
    st.n_draws = n_draws;
    RandomSource rng(seed);
    for (int i = 0; i < n_draws; ++i)
    {
        const auto e = draw_excitation(parent, n_act, rng);
        if (!e.valid)
        {
            ++st.n_invalid;
            continue;
        }
        const auto key = std::make_pair(e.det.alpha, e.det.beta);
        st.counts[key]++;
        // Every draw of a given determinant must report the SAME p_gen. If two
        // draws of the same determinant disagree, the generator's probability
        // model is inconsistent with itself, which the frequency test would
        // average over and hide.
        const auto it = st.p_gen.find(key);
        if (it == st.p_gen.end())
            st.p_gen[key] = e.p_gen;
        else if (std::abs(it->second - e.p_gen) > 1e-15)
            st.p_gen[key] = -1.0; // poison: flagged by the caller
    }
    return st;
}

static void test_open_shell_support_and_frequency()
{
    // OPEN SHELL: na != nb. Every other fixture in this file is closed-shell, so
    // an index bug that only manifests when the alpha and beta counts differ
    // would go undetected. This was found by a mutation that swapped the
    // alpha-beta index split (ka = k/n_sb, kb = k%n_sb): with n_sa == n_sb both
    // forms are bijections onto the same product set, so the mutant is
    // EQUIVALENT and the gate correctly passed it -- but it revealed that
    // nothing here would catch the asymmetric case either.
    struct Case { int n_act, na, nb, expected; const char *name; };
    const Case cases[] = {
        {4, 3, 1, 15, "n_act=4 3a/1b"},
        {6, 4, 2, 92, "n_act=6 4a/2b"},
        {7, 5, 3, 170, "n_act=7 5a/3b"},
    };
    for (const auto &c : cases)
    {
        const DetKey parent = make_det(c.na, c.nb);
        const auto conns = enumerate_connections(parent, c.n_act);
        if (static_cast<int>(conns.size()) != c.expected)
        {
            std::printf("  [FAIL] %s: oracle gives %zu connections, expected %d\n",
                        c.name, conns.size(), c.expected);
            ++g_failures;
            continue;
        }

        std::map<std::pair<CIString, CIString>, int> oracle;
        for (const auto &e : conns)
            oracle[{e.det.alpha, e.det.beta}] = 0;

        const int n_draws = 800000;
        const auto st = sample_generator(parent, c.n_act, n_draws, 8675309);

        if (st.counts.size() != oracle.size())
        {
            std::printf("  [FAIL] %s: generator reached %zu, oracle has %zu\n",
                        c.name, st.counts.size(), oracle.size());
            ++g_failures;
            continue;
        }
        bool unconnected = false;
        for (const auto &[key, unused] : st.counts)
            if (oracle.find(key) == oracle.end())
                unconnected = true;
        if (unconnected)
        {
            std::printf("  [FAIL] %s: generated an unconnected determinant\n", c.name);
            ++g_failures;
            continue;
        }

        double worst = 0.0;
        for (const auto &[key, count] : st.counts)
        {
            const double p = st.p_gen.at(key);
            if (p < 0.0)
            {
                worst = 1e9;
                break;
            }
            const double expected = n_draws * p;
            const double sigma = std::sqrt(n_draws * p * (1.0 - p));
            worst = std::max(worst, std::abs(count - expected) / sigma);
        }
        if (!(worst <= 5.0))
        {
            std::printf("  [FAIL] %s: frequency vs p_gen off by %.1f sigma\n", c.name, worst);
            ++g_failures;
        }
    }
}

static void test_production_generator_support()
{
    // THE check that a frequency test cannot replace once p_gen is non-uniform.
    // A rare connection that is never generated deviates by ~0.6 sigma over 400k
    // draws -- invisible to frequencies, obvious here.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {2, 1, 1, "H2/STO-3G"},
        {7, 5, 5, "water/STO-3G"},
        {10, 7, 7, "N2/STO-3G"},
    };
    for (const auto &c : cases)
    {
        const DetKey parent = make_det(c.na, c.nb);
        const auto conns = enumerate_connections(parent, c.n_act);
        std::map<std::pair<CIString, CIString>, int> oracle;
        for (const auto &e : conns)
            oracle[{e.det.alpha, e.det.beta}] = 0;

        const auto st = sample_generator(parent, c.n_act, 400000, 20250831);

        if (st.counts.size() != oracle.size())
        {
            std::printf("  [FAIL] %s: generator reached %zu determinants, oracle has %zu\n",
                        c.name, st.counts.size(), oracle.size());
            ++g_failures;
            continue;
        }
        for (const auto &[key, unused] : st.counts)
            if (oracle.find(key) == oracle.end())
            {
                std::printf("  [FAIL] %s: generated a determinant the oracle says is unconnected\n",
                            c.name);
                ++g_failures;
                break;
            }
    }
}

static void test_production_generator_frequencies()
{
    // FREQUENCY against a NON-UNIFORM p_gen: each determinant must appear
    // n_draws * p_gen times. This is the assertion that catches a p_gen which
    // does not match the sampler's real distribution.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {7, 5, 5, "water/STO-3G"},
        {10, 7, 7, "N2/STO-3G"},
    };
    for (const auto &c : cases)
    {
        const DetKey parent = make_det(c.na, c.nb);
        const int n_draws = 2000000;
        const auto st = sample_generator(parent, c.n_act, n_draws, 12345);

        double worst_dev = 0.0;
        bool inconsistent = false;
        for (const auto &[key, count] : st.counts)
        {
            const double p = st.p_gen.at(key);
            if (p < 0.0)
            {
                inconsistent = true;
                break;
            }
            const double expected = n_draws * p;
            const double sigma = std::sqrt(n_draws * p * (1.0 - p));
            worst_dev = std::max(worst_dev, std::abs(count - expected) / sigma);
        }
        if (inconsistent)
        {
            std::printf("  [FAIL] %s: the same determinant was reported with two "
                        "different p_gen values\n", c.name);
            ++g_failures;
            continue;
        }
        // 5 sigma over ~600 bins is a ~1-in-3500 false-failure rate.
        if (!(worst_dev <= 5.0))
        {
            std::printf("  [FAIL] %s: frequency does not match p_gen (worst %.1f sigma)\n",
                        c.name, worst_dev);
            ++g_failures;
        }
    }
}

static void test_production_generator_p_gen_normalizes()
{
    // sum over the connection set of p_gen must be 1: the generator's probability
    // model must be a probability distribution. This is a necessary condition,
    // not a sufficient one -- a generator can normalize correctly and still
    // sample the wrong distribution, which is exactly why the frequency test
    // above exists and why the scope forbids gating on self-consistency alone.
    const DetKey parent = make_det(5, 5);
    const auto st = sample_generator(parent, 7, 400000, 777);
    double total = 0.0;
    for (const auto &[key, p] : st.p_gen)
        total += p;
    check_close(total, 1.0, 1e-9, "p_gen sums to 1 over the reachable set");
}

static void test_production_generator_is_non_uniform()
{
    // Guard against the generator silently becoming uniform -- which would make
    // it a slower rewrite of F2.2 and, more importantly, would make the support
    // check redundant again without anyone noticing.
    const auto st = sample_generator(make_det(7, 7), 10, 200000, 555);
    double lo = 1e9, hi = 0.0;
    for (const auto &[key, p] : st.p_gen)
    {
        lo = std::min(lo, p);
        hi = std::max(hi, p);
    }
    check(hi / lo > 5.0, "p_gen is genuinely non-uniform on N2/STO-3G (expect ~21x)");
}

static void test_production_matches_oracle_phases()
{
    // The fast index arithmetic must produce the same phase as the oracle's
    // explicit loops for the same determinant. A phase error here is a sign
    // error in the Hamiltonian, not a sampling issue.
    const DetKey parent = make_det(5, 5);
    const auto conns = enumerate_connections(parent, 7);
    std::map<std::pair<CIString, CIString>, double> oracle_phase;
    for (const auto &e : conns)
        oracle_phase[{e.det.alpha, e.det.beta}] = e.phase;

    RandomSource rng(2468);
    for (int i = 0; i < 100000; ++i)
    {
        const auto e = draw_excitation(parent, 7, rng);
        if (!e.valid)
            continue;
        const auto it = oracle_phase.find({e.det.alpha, e.det.beta});
        if (it == oracle_phase.end())
            continue; // support failure, reported elsewhere
        if (e.phase != it->second)
        {
            check(false, "generator phase disagrees with the oracle phase");
            return;
        }
    }
}

static void test_production_generator_reproducible()
{
    const DetKey parent = make_det(5, 5);
    RandomSource a(31415), b(31415);
    for (int i = 0; i < 10000; ++i)
    {
        const auto ea = draw_excitation(parent, 7, a);
        const auto eb = draw_excitation(parent, 7, b);
        if (ea.det.alpha != eb.det.alpha || ea.det.beta != eb.det.beta
            || ea.phase != eb.phase || ea.p_gen != eb.p_gen || ea.valid != eb.valid)
        {
            check(false, "same seed must reproduce the fast generator bitwise");
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// F2.4 -- verify the GATE, by running it against deliberately broken generators
// ---------------------------------------------------------------------------
//
// The scope asks for three injected defects that make the gate go red. Doing
// that by editing the generator proves it once, on the day it is done, and then
// the evidence evaporates -- a later change can weaken the gate and nothing
// notices. Instead the broken generators live here as fixtures and the gate's
// ABILITY TO FAIL is asserted on every run.
//
// This matters because a statistical gate that cannot fail is worse than no gate:
// it produces a green tick that means nothing. The codebase has been bitten by
// exactly that (ch4_rccsdt_sto3g sat green for its whole life while never running
// the kernel it protected).

// The defect each broken generator carries.
enum class Defect
{
    PGenOffByConstant,   // reports p_gen/2 -- the classic mis-report
    NoOppositeSpin,      // never generates ab doubles -- a support hole
    ClassProbMismatch,   // draws with one class probability, reports another
};

static Excitation draw_broken(const DetKey &parent, int n_act, RandomSource &rng,
                              Defect defect)
{
    if (defect == Defect::PGenOffByConstant)
    {
        auto e = draw_excitation(parent, n_act, rng);
        e.p_gen *= 0.5;
        return e;
    }

    if (defect == Defect::NoOppositeSpin)
    {
        // Redraw until the excitation is not an opposite-spin double. p_gen is
        // left as the unrestricted generator reported it, so the frequencies of
        // the remaining classes are inflated relative to their claimed p_gen --
        // but the headline failure is that the ab connections are unreachable.
        for (int attempt = 0; attempt < 1000; ++attempt)
        {
            const auto e = draw_excitation(parent, n_act, rng);
            if (!e.valid)
                return e;
            const bool alpha_moved = e.det.alpha != parent.alpha;
            const bool beta_moved = e.det.beta != parent.beta;
            if (!(alpha_moved && beta_moved))
                return e;
        }
        return {};
    }

    // ClassProbMismatch: draw singles far more often than reported. Implemented
    // by biasing the draw and leaving p_gen untouched.
    Excitation e = draw_excitation(parent, n_act, rng);
    if (rng.uniform() < 0.5)
    {
        // Force a single roughly half the time, without touching p_gen.
        for (int attempt = 0; attempt < 100; ++attempt)
        {
            const auto candidate = draw_excitation(parent, n_act, rng);
            if (!candidate.valid)
                break;
            const bool alpha_moved = candidate.det.alpha != parent.alpha;
            const bool beta_moved = candidate.det.beta != parent.beta;
            const int rank = (std::popcount(candidate.det.alpha ^ parent.alpha)
                              + std::popcount(candidate.det.beta ^ parent.beta)) / 2;
            if (rank == 1 && !(alpha_moved && beta_moved))
                return candidate;
        }
    }
    return e;
}

// Re-run the F2.3 checks against a broken generator and report whether the gate
// noticed. Returns true if the gate FAILED (which is the desired outcome here).
static bool gate_rejects(Defect defect, const DetKey &parent, int n_act, int n_draws,
                         bool &support_failed, bool &frequency_failed)
{
    const auto conns = enumerate_connections(parent, n_act);
    std::map<std::pair<CIString, CIString>, int> oracle;
    for (const auto &e : conns)
        oracle[{e.det.alpha, e.det.beta}] = 0;

    RandomSource rng(556677);
    std::map<std::pair<CIString, CIString>, int> counts;
    std::map<std::pair<CIString, CIString>, double> p_gen;
    for (int i = 0; i < n_draws; ++i)
    {
        const auto e = draw_broken(parent, n_act, rng, defect);
        if (!e.valid)
            continue;
        const auto key = std::make_pair(e.det.alpha, e.det.beta);
        counts[key]++;
        p_gen[key] = e.p_gen;
    }

    support_failed = (counts.size() != oracle.size());
    for (const auto &[key, unused] : counts)
        if (oracle.find(key) == oracle.end())
            support_failed = true;

    frequency_failed = false;
    for (const auto &[key, count] : counts)
    {
        const double p = p_gen.at(key);
        const double expected = n_draws * p;
        const double sigma = std::sqrt(n_draws * p * (1.0 - p));
        if (sigma > 0.0 && std::abs(count - expected) / sigma > 5.0)
            frequency_failed = true;
    }
    return support_failed || frequency_failed;
}

static void test_gate_rejects_broken_generators()
{
    const DetKey parent = make_det(5, 5);
    const int n_act = 7;
    const int n_draws = 400000;

    bool sup = false, freq = false;

    // 1. p_gen off by a constant -> the FREQUENCY comparison must go red.
    check(gate_rejects(Defect::PGenOffByConstant, parent, n_act, n_draws, sup, freq),
          "gate rejects a generator whose p_gen is off by a constant factor");
    check(freq, "  ...and it is the frequency check that catches it");

    // 2. no opposite-spin doubles -> the SUPPORT comparison must go red. This is
    //    the case the scope singles out: 100 of water's 140 connections are ab
    //    doubles, so their absence is a gaping support hole.
    check(gate_rejects(Defect::NoOppositeSpin, parent, n_act, n_draws, sup, freq),
          "gate rejects a generator that never produces opposite-spin doubles");
    check(sup, "  ...and it is the support check that catches it");

    // 3. class-probability mismatch -> frequencies must go red.
    check(gate_rejects(Defect::ClassProbMismatch, parent, n_act, n_draws, sup, freq),
          "gate rejects a generator whose draw probability differs from its p_gen");
    check(freq, "  ...and it is the frequency check that catches it");

    // NEGATIVE CONTROL: the same machinery must ACCEPT the real generator.
    // Without this, a gate_rejects() that always returned true would satisfy
    // every check above.
    {
        const auto st = sample_generator(parent, n_act, n_draws, 556677);
        bool ok = (st.counts.size() == enumerate_connections(parent, n_act).size());
        double worst = 0.0;
        for (const auto &[key, count] : st.counts)
        {
            const double p = st.p_gen.at(key);
            const double expected = n_draws * p;
            const double sigma = std::sqrt(n_draws * p * (1.0 - p));
            worst = std::max(worst, std::abs(count - expected) / sigma);
        }
        check(ok && worst <= 5.0,
              "the same gate ACCEPTS the real generator (else it rejects everything)");
    }
}

// ---------------------------------------------------------------------------
// F2.5 -- spin and symmetry constraints
// ---------------------------------------------------------------------------

static void test_particle_number_is_structural()
{
    // Per-spin particle number is conserved by CONSTRUCTION -- the generator
    // annihilates once and creates once per spin channel -- so this needs no
    // filter, only proof that the construction holds. Checked on open shell too,
    // where an index bug would be most likely to break it.
    struct Case { int n_act, na, nb; };
    const Case cases[] = {{7, 5, 5}, {10, 7, 7}, {7, 5, 3}, {6, 4, 2}};
    for (const auto &c : cases)
    {
        const DetKey parent = make_det(c.na, c.nb);
        RandomSource rng(90210);
        for (int i = 0; i < 50000; ++i)
        {
            const auto e = draw_excitation(parent, c.n_act, rng);
            if (!e.valid)
                continue;
            if (std::popcount(e.det.alpha) != c.na || std::popcount(e.det.beta) != c.nb)
            {
                check(false, "generator conserves per-spin electron count");
                return;
            }
        }
    }
}

static void test_in_space_rejection_sampling()
{
    // A synthetic "symmetry" restriction: keep only determinants whose alpha
    // string has even parity in the low 3 bits. Arbitrary, but it restricts the
    // space the way a target irrep does, and its acceptance rate is measurable.
    const DetKey parent = make_det(5, 5);
    const int n_act = 7;
    const auto in_space = [](const DetKey &d) {
        return (std::popcount(d.alpha & CIString{0b111}) % 2) == 0;
    };

    RandomSource probe(11111);
    const double p_accept = measure_acceptance_rate(parent, n_act, probe, in_space, 200000);
    check(p_accept > 0.05 && p_accept < 0.95,
          "the synthetic restriction actually restricts (else the test is vacuous)");

    // Every accepted draw must be in-space.
    RandomSource rng(22222);
    int n_valid = 0;
    for (int i = 0; i < 50000; ++i)
    {
        const auto e = draw_excitation_in_space(parent, n_act, rng, in_space, p_accept);
        if (!e.valid)
            continue;
        ++n_valid;
        if (!in_space(e.det))
        {
            check(false, "rejection sampling returns only in-space determinants");
            return;
        }
    }
    check(n_valid > 45000, "rejection sampling succeeds on most calls");
}

static void test_in_space_p_gen_is_corrected()
{
    // THE F2.5 assertion. A restricted draw happens LESS often than the
    // unrestricted p_gen claims, so p_gen must be divided by the acceptance rate.
    // Reporting the unrestricted value silently suppresses every spawn out of a
    // restricted space -- a plausible, converged, wrong energy.
    const DetKey parent = make_det(5, 5);
    const int n_act = 7;
    const auto in_space = [](const DetKey &d) {
        return (std::popcount(d.alpha & CIString{0b111}) % 2) == 0;
    };

    RandomSource probe(33333);
    const double p_accept = measure_acceptance_rate(parent, n_act, probe, in_space, 200000);

    // Sample and check frequency against the CORRECTED p_gen.
    RandomSource rng(44444);
    const int n_draws = 800000;
    std::map<std::pair<CIString, CIString>, int> counts;
    std::map<std::pair<CIString, CIString>, double> p_gen;
    int accepted = 0;
    for (int i = 0; i < n_draws; ++i)
    {
        const auto e = draw_excitation_in_space(parent, n_act, rng, in_space, p_accept);
        if (!e.valid)
            continue;
        ++accepted;
        counts[{e.det.alpha, e.det.beta}]++;
        p_gen[{e.det.alpha, e.det.beta}] = e.p_gen;
    }

    // Each in-space determinant appears accepted * p_gen_corrected times.
    double worst = 0.0;
    for (const auto &[key, count] : counts)
    {
        const double p = p_gen.at(key);
        const double expected = accepted * p;
        const double sigma = std::sqrt(accepted * p * (1.0 - p));
        if (sigma > 0.0)
            worst = std::max(worst, std::abs(count - expected) / sigma);
    }
    if (!(worst <= 5.0))
    {
        std::printf("  [FAIL] corrected p_gen does not match in-space frequency "
                    "(worst %.1f sigma)\n", worst);
        ++g_failures;
    }

    // The corrected p_gen must sum to 1 over the IN-SPACE set.
    double total = 0.0;
    for (const auto &[key, p] : p_gen)
        total += p;
    check_close(total, 1.0, 0.02, "corrected p_gen sums to 1 over the in-space set");

    // And it must be strictly LARGER than the unrestricted value -- the whole
    // point of the correction. A generator reporting the unrestricted p_gen
    // would fail this.
    check(p_accept < 1.0, "the restriction rejects some draws");
    bool all_larger = true;
    RandomSource unres(44444);
    for (int i = 0; i < 200; ++i)
    {
        const auto e = draw_excitation(parent, n_act, unres);
        if (!e.valid)
            continue;
        const auto it = p_gen.find({e.det.alpha, e.det.beta});
        if (it != p_gen.end() && !(it->second > e.p_gen * 1.001))
            all_larger = false;
    }
    check(all_larger, "corrected p_gen exceeds the unrestricted p_gen");
}

static void test_per_call_acceptance_estimate_is_biased()
{
    // A guard against the defect this step nearly shipped: estimating p_accept
    // from the attempt count of the call itself.
    //
    // That estimator is UNBIASED for p_gen -- E[p_gen * attempts] is exactly the
    // conditional probability -- but the spawn uses |H_ij| / p_gen, and
    // E[1/X] != 1/E[X]. Measured at p_accept = 0.3: the mean of p_gen is right,
    // and the mean of 1/p_gen is 1.72x too large. This test pins that so nobody
    // "simplifies" measure_acceptance_rate away.
    RandomSource rng(1234567);
    const double p_accept = 0.3;
    const double p_unres = 0.01;
    const int n = 200000;

    double sum_p = 0.0, sum_inv = 0.0;
    for (int i = 0; i < n; ++i)
    {
        int attempts = 1;
        while (rng.uniform() >= p_accept)
            ++attempts;
        const double per_call = p_unres * attempts;
        sum_p += per_call;
        sum_inv += 1.0 / per_call;
    }
    const double true_cond = p_unres / p_accept;

    check_close(sum_p / n, true_cond, true_cond * 0.05,
                "per-call estimator IS unbiased for p_gen (which is why it looks fine)");
    const double inv_ratio = (sum_inv / n) / (1.0 / true_cond);
    check(inv_ratio > 1.3,
          "per-call estimator is biased in 1/p_gen, which is what the spawn uses");
}

// ---------------------------------------------------------------------------
// F3.1 -- one deterministic iteration, checked against a dense matrix-vector
//         product to machine precision
// ---------------------------------------------------------------------------
//
// The reference Hamiltonian is built HERE, independently of the CI layer. A test
// that reused build_ci_hamiltonian_dense would check that the dynamics agree with
// the same matrix-element code they call -- consistency, not correctness. This
// way the only shared thing is the determinant convention.

// A small synthetic Hamiltonian over an explicitly enumerated determinant space.
struct ToyHamiltonian
{
    std::vector<DetKey> dets;
    std::map<std::pair<CIString, CIString>, int> index;
    std::vector<std::vector<double>> H;

    // Deterministic pseudo-random symmetric matrix: H_ij depends only on the
    // determinant pair, so it is reproducible and order-independent.
    static double element(const DetKey &a, const DetKey &b)
    {
        std::uint64_t x = a.alpha * 0x9e3779b97f4a7c15ULL + a.beta * 0xbf58476d1ce4e5b9ULL;
        std::uint64_t y = b.alpha * 0x9e3779b97f4a7c15ULL + b.beta * 0xbf58476d1ce4e5b9ULL;
        std::uint64_t s = x ^ y;   // symmetric in a,b
        s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
        s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
        s ^= (s >> 31);
        return (static_cast<double>(s % 2000) / 1000.0) - 1.0;   // in [-1, 1)
    }

    ToyHamiltonian(int n_act, int na, int nb)
    {
        // Enumerate the full determinant space by brute force.
        for (CIString a = 0; a < (CIString{1} << n_act); ++a)
        {
            if (std::popcount(a) != na)
                continue;
            for (CIString b = 0; b < (CIString{1} << n_act); ++b)
            {
                if (std::popcount(b) != nb)
                    continue;
                index[{a, b}] = static_cast<int>(dets.size());
                dets.push_back(DetKey{a, b});
            }
        }
        const std::size_t n = dets.size();
        H.assign(n, std::vector<double>(n, 0.0));
        // Fill ONLY connected pairs. A physical Hamiltonian is exactly zero
        // between determinants differing by more than a double excitation -- it
        // is a two-body operator. Filling every entry would not be a Hamiltonian
        // at all, and a reference matvec built from it disagrees with any correct
        // propagator (measured: 9 of 35 pairs unconnected at n_act=4, na=nb=2,
        // giving an 8.5e-2 discrepancy).
        for (std::size_t i = 0; i < n; ++i)
            for (const auto &e : enumerate_connections(dets[i], n_act))
            {
                const auto it = index.find({e.det.alpha, e.det.beta});
                if (it == index.end())
                    continue;
                H[i][static_cast<std::size_t>(it->second)] = element(dets[i], e.det);
            }
        // Diagonal made dominant and negative, as a real CI diagonal is.
        for (std::size_t i = 0; i < n; ++i)
            H[i][i] = -2.0 - 0.1 * static_cast<double>(i);
    }

    HamiltonianOps ops(int n_act) const
    {
        // Only elements between CONNECTED determinants are nonzero -- the
        // Hamiltonian is a two-body operator. The oracle decides connectivity, so
        // the ops agree with what propagate_deterministic will visit.
        auto off = [this, n_act](const DetKey &i, const DetKey &j) -> double {
            const auto ii = index.find({i.alpha, i.beta});
            const auto jj = index.find({j.alpha, j.beta});
            if (ii == index.end() || jj == index.end())
                return 0.0;
            if (ii->second == jj->second)
                return 0.0;
            return H[static_cast<std::size_t>(ii->second)][static_cast<std::size_t>(jj->second)];
        };
        auto diag = [this](const DetKey &i) -> double {
            const auto ii = index.find({i.alpha, i.beta});
            if (ii == index.end())
                return 0.0;
            return H[static_cast<std::size_t>(ii->second)][static_cast<std::size_t>(ii->second)];
        };
        return HamiltonianOps{off, diag};
    }
};

static void test_toy_hamiltonian_is_symmetric()
{
    // The toy H is filled by walking each row's connections. Reachability is
    // symmetric (established in the FCI sigma-build work), so the result should
    // be symmetric -- but assert it, because an asymmetric "Hamiltonian" would
    // make the matvec reference meaningless while every other check still passed.
    for (const auto &[n_act, na, nb] : std::vector<std::tuple<int, int, int>>{
             {4, 2, 2}, {5, 3, 2}})
    {
        const ToyHamiltonian toy(n_act, na, nb);
        const std::size_t n = toy.dets.size();
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                worst = std::max(worst, std::abs(toy.H[i][j] - toy.H[j][i]));
        check(worst < 1e-15, "the toy Hamiltonian is symmetric");
    }
}

static void test_deterministic_step_is_a_matvec()
{
    // The core F3.1 assertion. One propagate_deterministic call must equal
    // (1 - dt(H - S)) c computed by plain matrix arithmetic, to machine
    // precision. No statistics: if this does not hold exactly, nothing built on
    // top is worth debugging.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {2, 1, 1, "H2-like 2 orbitals"},
        {4, 2, 2, "4 orbitals closed shell"},
        {5, 3, 2, "5 orbitals OPEN shell"},   // F2's lesson: closed shell alone is blind
    };

    for (const auto &c : cases)
    {
        const ToyHamiltonian toy(c.n_act, c.na, c.nb);
        const auto ops = toy.ops(c.n_act);
        const std::size_t n = toy.dets.size();
        const double dt = 0.01;
        const double shift = -2.0;

        // A start vector with mixed signs, so annihilation is exercised.
        std::vector<double> c_vec(n);
        for (std::size_t i = 0; i < n; ++i)
            c_vec[i] = ((i % 3) == 0 ? 1.0 : -0.5) * (1.0 + 0.1 * static_cast<double>(i));

        WalkerPopulation pop;
        for (std::size_t i = 0; i < n; ++i)
            pop.add(toy.dets[i], c_vec[i]);

        const auto next = propagate_deterministic(pop, c.n_act, ops, dt, shift);

        // Reference: (1 - dt(H - S)) c, by hand.
        std::vector<double> want(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
        {
            double acc = c_vec[i] * (1.0 - dt * (toy.H[i][i] - shift));
            for (std::size_t j = 0; j < n; ++j)
            {
                if (j == i)
                    continue;
                // Only connected pairs contribute -- the ops return 0 otherwise,
                // and propagate_deterministic only visits connections.
                const double h = ops.off_diagonal(toy.dets[j], toy.dets[i]);
                if (h == 0.0)
                    continue;
                acc += -dt * h * c_vec[j];
            }
            want[i] = acc;
        }

        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::abs(next.weight_at(toy.dets[i]) - want[i]));

        if (!(worst < 1e-12))
        {
            std::printf("  [FAIL] %s: deterministic step differs from matvec by %.3e\n",
                        c.name, worst);
            ++g_failures;
        }
    }
}

static void test_deterministic_step_only_visits_connections()
{
    // A determinant unreachable in one step must receive nothing. If the spawn
    // visited unconnected determinants the matvec check above would still pass
    // whenever those elements happen to be zero -- so assert reachability
    // directly.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);

    WalkerPopulation pop;
    pop.add(toy.dets[0], 1.0);
    const auto next = propagate_deterministic(pop, 4, ops, 0.01, -2.0);

    std::map<std::pair<CIString, CIString>, int> allowed;
    allowed[{toy.dets[0].alpha, toy.dets[0].beta}] = 1;   // itself, via death
    for (const auto &e : enumerate_connections(toy.dets[0], 4))
        allowed[{e.det.alpha, e.det.beta}] = 1;

    for (const auto &[det, w] : next)
        if (w != 0.0 && allowed.find({det.alpha, det.beta}) == allowed.end())
        {
            check(false, "deterministic step touched an unconnected determinant");
            return;
        }
}

static void test_deterministic_step_annihilates()
{
    // Two parents spawning onto the same child with opposite signs must cancel.
    // This is the property that makes FCIQMC work rather than merely sample, and
    // it must survive the propagation, not just the container.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);

    WalkerPopulation a;
    a.add(toy.dets[0], 1.0);
    const auto from_a = propagate_deterministic(a, 4, ops, 0.01, -2.0);

    WalkerPopulation b;
    b.add(toy.dets[0], -1.0);
    const auto from_b = propagate_deterministic(b, 4, ops, 0.01, -2.0);

    // Propagating (+1) and (-1) separately must give exactly opposite results;
    // propagating their sum must give zero.
    WalkerPopulation both;
    both.add(toy.dets[0], 1.0);
    both.add(toy.dets[0], -1.0);
    const auto from_both = propagate_deterministic(both, 4, ops, 0.01, -2.0);

    double worst_opposite = 0.0, worst_zero = 0.0;
    for (const auto &[det, w] : from_a)
    {
        worst_opposite = std::max(worst_opposite, std::abs(w + from_b.weight_at(det)));
        worst_zero = std::max(worst_zero, std::abs(from_both.weight_at(det)));
    }
    check(worst_opposite < 1e-15, "propagating +c and -c gives exactly opposite populations");
    check(worst_zero < 1e-15, "a fully annihilated parent spawns nothing");
}

static void test_deterministic_step_linear()
{
    // The propagator is linear: P(a + b) == P(a) + P(b). A nonlinearity here
    // would mean the dynamics depend on how walkers are grouped, which no correct
    // implementation can.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    const double dt = 0.02, shift = -2.0;

    WalkerPopulation a, b, sum;
    a.add(toy.dets[0], 0.7);
    a.add(toy.dets[1], -0.3);
    b.add(toy.dets[1], 1.1);
    b.add(toy.dets[2], 0.5);
    for (const auto &[det, w] : a)
        sum.add(det, w);
    for (const auto &[det, w] : b)
        sum.add(det, w);

    const auto pa = propagate_deterministic(a, 4, ops, dt, shift);
    const auto pb = propagate_deterministic(b, 4, ops, dt, shift);
    const auto ps = propagate_deterministic(sum, 4, ops, dt, shift);

    double worst = 0.0;
    for (const auto &[det, w] : ps)
        worst = std::max(worst, std::abs(w - (pa.weight_at(det) + pb.weight_at(det))));
    check(worst < 1e-14, "the propagator is linear in the population");
}

// ---------------------------------------------------------------------------
// F3.2 -- stochastic spawning; the MEAN must reproduce F3.1 exactly
// ---------------------------------------------------------------------------

static void test_stochastic_mean_matches_deterministic()
{
    // THE F3.2 assertion. The stochastic spawn draws one connection instead of
    // visiting all, and reweights by 1/p_gen; its expectation is the deterministic
    // result exactly. This is a mean of a LINEAR quantity, so -- unlike the
    // projected energy -- it carries no ratio bias, which is what makes it a clean
    // test of the 1/p_gen division.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {4, 2, 2, "4 orbitals closed shell"},
        {5, 3, 2, "5 orbitals OPEN shell"},
    };

    for (const auto &c : cases)
    {
        const ToyHamiltonian toy(c.n_act, c.na, c.nb);
        const auto ops = toy.ops(c.n_act);
        const double dt = 0.01, shift = -2.0;

        WalkerPopulation start;
        start.add(toy.dets[0], 1.0);
        start.add(toy.dets[1], -0.6);

        const auto exact = propagate_deterministic(start, c.n_act, ops, dt, shift);

        // Average the stochastic step over many independent runs.
        const int n_runs = 200000;
        std::map<std::pair<CIString, CIString>, double> sum, sumsq;
        RandomSource rng(20250901);
        for (int r = 0; r < n_runs; ++r)
        {
            const auto step = propagate_stochastic(start, c.n_act, ops, dt, shift, rng);
            // Accumulate for EVERY determinant the deterministic step reaches, so
            // a component that is zero in this draw still contributes a zero to
            // its variance -- otherwise the standard error is computed from the
            // wrong sample count.
            for (const auto &[det, w_exact] : exact)
            {
                const double w = step.weight_at(det);
                sum[{det.alpha, det.beta}] += w;
                sumsq[{det.alpha, det.beta}] += w * w;
            }
        }

        // Compare against each component's own STANDARD ERROR, not a fixed
        // tolerance.
        //
        // A fixed absolute bound requires guessing the scale and getting it wrong
        // makes the test vacuous: a first version used 0.02, which happened to sit
        // right at the size of the effect from dropping 1/p_gen (spawn magnitudes
        // here span 0.005 to 0.4 across classes), so that mutation PASSED. A fixed
        // relative bound has the opposite problem -- it is dominated by sampling
        // noise on the smallest components. The standard error is the only scale
        // that is correct for every component at once.
        double worst_sigma = 0.0;
        for (const auto &[det, w_exact] : exact)
        {
            const auto key = std::make_pair(det.alpha, det.beta);
            const double mean = sum[key] / n_runs;
            const double var = std::max(0.0, sumsq[key] / n_runs - mean * mean);
            const double stderr_ = std::sqrt(var / n_runs);
            if (stderr_ <= 0.0)
            {
                // Deterministic component (the death term): must match exactly.
                worst_sigma = std::max(worst_sigma,
                                       std::abs(mean - w_exact) > 1e-12 ? 1e9 : 0.0);
                continue;
            }
            worst_sigma = std::max(worst_sigma, std::abs(mean - w_exact) / stderr_);
        }
        // Every determinant the stochastic step reached must be one the
        // deterministic step reaches too.
        for (const auto &[key, total] : sum)
        {
            if (std::abs(total / n_runs) < 1e-9)
                continue;
            if (exact.weight_at(DetKey{key.first, key.second}) == 0.0)
            {
                std::printf("  [FAIL] %s: stochastic step reached a determinant the "
                            "deterministic step does not\n", c.name);
                ++g_failures;
                break;
            }
        }
        if (!(worst_sigma <= 5.0))
        {
            std::printf("  [FAIL] %s: stochastic mean differs from deterministic by "
                        "%.1f sigma\n", c.name, worst_sigma);
            ++g_failures;
        }
    }
}

static void test_stochastic_variance_falls_with_attempts()
{
    // More spawn attempts per parent must reduce the variance of the step, as
    // 1/n_attempts. A spawn that is right in the mean but has wrong variance
    // scaling usually means p_gen is inconsistent with the sampling rather than
    // wrong by a constant factor -- so this catches a class the mean test does
    // not.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    const double dt = 0.01, shift = -2.0;

    WalkerPopulation start;
    start.add(toy.dets[0], 1.0);

    const auto target = toy.dets[3];   // watch one specific child

    std::vector<double> variances;
    for (int attempts : {1, 4, 16})
    {
        RandomSource rng(777);
        const int n_runs = 20000;
        double s1 = 0.0, s2 = 0.0;
        for (int r = 0; r < n_runs; ++r)
        {
            const auto step = propagate_stochastic(start, 4, ops, dt, shift, rng, attempts);
            const double w = step.weight_at(target);
            s1 += w;
            s2 += w * w;
        }
        const double mean = s1 / n_runs;
        variances.push_back(s2 / n_runs - mean * mean);
    }

    // variance(1 attempt) / variance(4 attempts) should be ~4, and /16 ~16.
    check(variances[0] > 0.0, "the stochastic step actually has variance");
    const double r4 = variances[0] / variances[1];
    const double r16 = variances[0] / variances[2];
    if (!(r4 > 2.5 && r4 < 6.0) || !(r16 > 9.0 && r16 < 25.0))
    {
        std::printf("  [FAIL] variance does not fall as 1/n_attempts "
                    "(ratios %.2f at 4, %.2f at 16; want ~4 and ~16)\n", r4, r16);
        ++g_failures;
    }
}

static void test_stochastic_attempts_do_not_rescale()
{
    // Raising n_spawn_attempts must not change the EXPECTED step -- only its
    // variance. If the per-attempt weight split were missing, more attempts would
    // silently multiply the off-diagonal part of H.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    const double dt = 0.01, shift = -2.0;

    WalkerPopulation start;
    start.add(toy.dets[0], 1.0);
    const auto exact = propagate_deterministic(start, 4, ops, dt, shift);

    for (int attempts : {1, 8})
    {
        RandomSource rng(31415);
        const int n_runs = 200000;
        std::map<std::pair<CIString, CIString>, double> sum, sumsq;
        for (int r = 0; r < n_runs; ++r)
        {
            const auto step = propagate_stochastic(start, 4, ops, dt, shift, rng, attempts);
            for (const auto &[det, w_exact] : exact)
            {
                const double w = step.weight_at(det);
                sum[{det.alpha, det.beta}] += w;
                sumsq[{det.alpha, det.beta}] += w * w;
            }
        }
        // Standard-error comparison, for the same reason as the mean test above.
        double worst = 0.0;
        for (const auto &[det, w_exact] : exact)
        {
            const auto key = std::make_pair(det.alpha, det.beta);
            const double mean = sum[key] / n_runs;
            const double var = std::max(0.0, sumsq[key] / n_runs - mean * mean);
            const double stderr_ = std::sqrt(var / n_runs);
            if (stderr_ <= 0.0)
                continue;
            worst = std::max(worst, std::abs(mean - w_exact) / stderr_);
        }
        if (!(worst <= 5.0))
        {
            std::printf("  [FAIL] n_spawn_attempts=%d changes the expected step "
                        "(off by %.1f sigma)\n", attempts, worst);
            ++g_failures;
        }
    }
}

static void test_stochastic_death_is_exact()
{
    // The death step is deterministic even in the stochastic propagator, so a
    // population with NO connections must evolve exactly. This isolates the
    // diagonal from the sampling.
    const ToyHamiltonian toy(2, 1, 1);
    // Build ops whose off-diagonal is identically zero: only death remains.
    HamiltonianOps death_only{
        [](const DetKey &, const DetKey &) { return 0.0; },
        toy.ops(2).diagonal};

    WalkerPopulation start;
    start.add(toy.dets[0], 3.0);
    RandomSource rng(1);
    const auto step = propagate_stochastic(start, 2, death_only, 0.01, -2.0, rng);

    const double diag = death_only.diagonal(toy.dets[0]);
    const double want = 3.0 * (1.0 - 0.01 * (diag - (-2.0)));
    check_close(step.weight_at(toy.dets[0]), want, 1e-15,
                "the death step is exact even in the stochastic propagator");
}

static void test_stochastic_reproducible()
{
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    WalkerPopulation start;
    start.add(toy.dets[0], 1.0);

    RandomSource a(2718), b(2718);
    for (int i = 0; i < 200; ++i)
    {
        const auto sa = propagate_stochastic(start, 4, ops, 0.01, -2.0, a);
        const auto sb = propagate_stochastic(start, 4, ops, 0.01, -2.0, b);
        for (const auto &[det, w] : sa)
            if (w != sb.weight_at(det))
            {
                check(false, "same seed must reproduce the stochastic step bitwise");
                return;
            }
    }
}

// ---------------------------------------------------------------------------
// F3.3 -- imaginary-time propagation reaches the ground state
// ---------------------------------------------------------------------------

// Power-iteration ground state of the toy H, computed independently of the
// propagator so the comparison is not circular. Uses the SAME (1 - dt(H-S))
// operator mathematically, but applied with plain matrix arithmetic.
static std::vector<double> exact_ground_state(const ToyHamiltonian &toy, double dt,
                                              double shift, int iters = 20000)
{
    const std::size_t n = toy.dets.size();
    std::vector<double> c(n, 0.0);
    c[0] = 1.0;
    for (int it = 0; it < iters; ++it)
    {
        std::vector<double> next(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
        {
            double acc = c[i] * (1.0 - dt * (toy.H[i][i] - shift));
            for (std::size_t j = 0; j < n; ++j)
                if (j != i)
                    acc += -dt * toy.H[i][j] * c[j];
            next[i] = acc;
        }
        // Renormalize every iteration: with a fixed shift the norm grows
        // exponentially (measured: 1e96 after 3000 steps on a 6-orbital toy), so
        // an unnormalized power iteration overflows long before it converges.
        double nrm = 0.0;
        for (double v : next)
            nrm += v * v;
        nrm = std::sqrt(nrm);
        for (std::size_t i = 0; i < n; ++i)
            c[i] = next[i] / nrm;
    }
    return c;
}

static void test_timestep_bound_is_computed()
{
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    const double shift = -2.0;

    WalkerPopulation pop;
    for (const auto &d : toy.dets)
        pop.add(d, 1.0);

    const double bound = max_stable_timestep(pop, ops, shift);

    // Check it against the definition, computed here from the diagonal directly.
    double worst = 0.0;
    for (std::size_t i = 0; i < toy.dets.size(); ++i)
        worst = std::max(worst, std::abs(toy.H[i][i] - shift));
    check_close(bound, 2.0 / worst, 1e-15, "max_stable_timestep matches 2/max|H_ii - S|");
}

// NOTE: there is deliberately NO "too large a timestep diverges" test here.
//
// The F3 scope asked for one, on the reasoning that dt above the stability bound
// must visibly break the propagation. Three formulations were tried and all three
// premises turned out to be false ON THIS HAMILTONIAN, measured:
//
//   1. "the population collapses onto one determinant" -- it does not; the shape
//      settles at max|component| = 0.0716 for every dt from 1.5x to 5x the
//      diagonal bound.
//   2. "the norm diverges above the bound and not below" -- the norm grows at
//      EVERY dt (1.22x per iteration at 0.05x the bound, 14.5x at 3x), because
//      with the shift below the ground-state energy exponential growth is what a
//      fixed shift produces by design.
//   3. "the converged shape is the wrong state" -- the overlap with the true
//      ground state is 1.000000 at every dt tried, from 0.05x to 5x the diagonal
//      bound and well past the true spectral bound of 0.2509.
//
// The reason is that this test renormalizes every iteration, which turns the
// propagation into a power iteration for the dominant eigenvector of
// (1 - dt(H - S)) -- and on this Hamiltonian the ground state remains dominant at
// every dt tested. A timestep that is "unstable" in the sense of the norm bound
// is therefore still projecting onto the right state here.
//
// Rather than construct a Hamiltonian contrived to fail, this records what was
// measured. max_stable_timestep is still gated (test_timestep_bound_is_computed)
// as an arithmetic identity, and its two documented caveats -- diagonal-only, and
// computed from the CURRENTLY occupied determinants -- are what a caller needs.
// A real divergence gate belongs with F4, where the population is controlled and
// an unstable dt has somewhere to show up.

static void test_propagation_reaches_the_ground_state()
{
    // The F3.3 assertion: iterating the deterministic propagator converges in
    // SHAPE to the ground-state eigenvector.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {4, 2, 2, "4 orbitals closed shell"},
        {5, 3, 2, "5 orbitals OPEN shell"},
    };

    for (const auto &c : cases)
    {
        const ToyHamiltonian toy(c.n_act, c.na, c.nb);
        const auto ops = toy.ops(c.n_act);
        const double shift = -2.0;

        WalkerPopulation whole;
        for (const auto &d : toy.dets)
            whole.add(d, 1.0);
        // 0.1x the FULL-SPACE diagonal bound: the diagonal bound alone does not
        // guarantee stability once off-diagonals are present (measured 2.28x too
        // large), and a bound from the seeded population alone would be infinite.
        const double dt = 0.1 * max_stable_timestep(whole, ops, shift);

        WalkerPopulation pop;
        pop.add(toy.dets[0], 1.0);

        for (int it = 0; it < 4000; ++it)
        {
            pop = propagate_deterministic(pop, c.n_act, ops, dt, shift);
            const double nrm = ordered_l1_norm(pop);
            if (nrm == 0.0 || !std::isfinite(nrm))
            {
                check(false, "population collapsed during propagation");
                return;
            }
            WalkerPopulation scaled;
            for (const auto &[det, w] : pop)
                scaled.add(det, w / nrm);
            pop = scaled;
        }

        const auto want = exact_ground_state(toy, dt, shift);
        // exact_ground_state renormalizes each iteration, so it converges for any
        // dt inside the true bound -- the same dt the propagator used.

        // Compare normalized shapes, fixing the overall sign (an eigenvector is
        // defined up to a sign, and both are).
        std::vector<double> got(toy.dets.size(), 0.0);
        for (std::size_t i = 0; i < toy.dets.size(); ++i)
            got[i] = pop.weight_at(toy.dets[i]);
        double gn = 0.0, wn = 0.0, dot = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i)
        {
            gn += got[i] * got[i];
            wn += want[i] * want[i];
            dot += got[i] * want[i];
        }
        gn = std::sqrt(gn);
        wn = std::sqrt(wn);
        const double overlap = std::abs(dot / (gn * wn));

        if (!(overlap > 0.9999))
        {
            std::printf("  [FAIL] %s: converged shape overlaps the ground state by "
                        "only %.6f\n", c.name, overlap);
            ++g_failures;
        }
    }
}

static void test_ordered_norm_is_deterministic()
{
    // The reported norm must not depend on insertion order -- the population is a
    // hash map, so a naive sum would.
    // The values must be chosen so that summing them in different orders GIVES a
    // different answer in floating point -- otherwise the test passes whether or
    // not the function sorts. A first version used 0.1*i, whose partial sums
    // happen to be order-insensitive here, and removing the sort passed it.
    //
    // Widely-separated magnitudes make the reassociation visible: adding a tiny
    // value to a large running total loses it, while summing the tiny ones first
    // does not.
    const ToyHamiltonian toy(4, 2, 2);
    // Graded magnitudes, spanning many orders. An alternating large/small pattern
    // is NOT enough -- with equal counts of each, forward and reverse sums agree
    // exactly, and the vacuity check below caught that fixture.
    std::vector<double> vals;
    for (std::size_t i = 0; i < toy.dets.size(); ++i)
        vals.push_back(std::pow(10.0, static_cast<double>(i) - 18.0));

    WalkerPopulation a, b;
    for (std::size_t i = 0; i < toy.dets.size(); ++i)
        a.add(toy.dets[i], vals[i]);
    for (std::size_t i = toy.dets.size(); i > 0; --i)
        b.add(toy.dets[i - 1], vals[i - 1]);

    check(ordered_l1_norm(a) == ordered_l1_norm(b),
          "ordered_l1_norm is independent of insertion order");

    // And confirm the fixture is not vacuous: these values DO reassociate.
    double fwd = 0.0, rev = 0.0;
    for (std::size_t i = 0; i < vals.size(); ++i)
        fwd += vals[i];
    for (std::size_t i = vals.size(); i > 0; --i)
        rev += vals[i - 1];
    check(fwd != rev, "the test values actually reassociate (else the check is vacuous)");
}

// ---------------------------------------------------------------------------
// F3.4 -- the projected energy, with its finite-population bias characterized
// ---------------------------------------------------------------------------

// Exact lowest eigenvalue of the toy H, by unshifted power iteration on
// (sigma*I - H), computed independently of the propagator.
static double exact_ground_energy(const ToyHamiltonian &toy)
{
    const std::size_t n = toy.dets.size();
    // Shift so the ground state becomes dominant in magnitude.
    double sigma = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        double row = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            row += std::abs(toy.H[i][j]);
        sigma = std::max(sigma, row);
    }
    std::vector<double> v(n, 0.0);
    v[0] = 1.0;
    double eig = 0.0;
    for (int it = 0; it < 20000; ++it)
    {
        std::vector<double> w(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
        {
            double acc = sigma * v[i];
            for (std::size_t j = 0; j < n; ++j)
                acc -= toy.H[i][j] * v[j];
            w[i] = acc;
        }
        double nrm = 0.0;
        for (double x : w)
            nrm += x * x;
        nrm = std::sqrt(nrm);
        for (std::size_t i = 0; i < n; ++i)
            v[i] = w[i] / nrm;
        eig = sigma - nrm;
    }
    // Rayleigh quotient for a clean final value.
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            acc += toy.H[i][j] * v[j];
        num += v[i] * acc;
        den += v[i] * v[i];
    }
    (void)eig;
    return num / den;
}

static void test_projected_energy_exact_on_converged_state()
{
    // On the converged DETERMINISTIC state there is no sampling noise, so the
    // projected energy must equal the exact ground-state energy. This isolates
    // the estimator's algebra from its statistics -- if this fails, no amount of
    // sampling analysis will help.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {4, 2, 2, "4 orbitals closed shell"},
        {5, 3, 2, "5 orbitals OPEN shell"},
    };

    for (const auto &c : cases)
    {
        const ToyHamiltonian toy(c.n_act, c.na, c.nb);
        const auto ops = toy.ops(c.n_act);
        const double shift = -2.0;

        WalkerPopulation whole;
        for (const auto &d : toy.dets)
            whole.add(d, 1.0);
        const double dt = 0.1 * max_stable_timestep(whole, ops, shift);

        WalkerPopulation pop;
        pop.add(toy.dets[0], 1.0);
        for (int it = 0; it < 4000; ++it)
        {
            pop = propagate_deterministic(pop, c.n_act, ops, dt, shift);
            const double nrm = ordered_l1_norm(pop);
            if (nrm == 0.0 || !std::isfinite(nrm))
                break;
            WalkerPopulation scaled;
            for (const auto &[det, w] : pop)
                scaled.add(det, w / nrm);
            pop = scaled;
        }

        const auto pe = projected_energy(pop, toy.dets[0], c.n_act, ops);
        check(pe.valid, "projected energy is valid on a converged population");
        const double exact = exact_ground_energy(toy);
        if (std::abs(pe.energy - exact) > 1e-8)
        {
            std::printf("  [FAIL] %s: projected energy %.10f vs exact %.10f\n",
                        c.name, pe.energy, exact);
            ++g_failures;
        }
    }
}

static void test_projected_energy_rejects_small_reference()
{
    // The reference weight is the denominator. If the population drifts off the
    // reference the ratio is noise over noise, and returning a number the caller
    // cannot distinguish from a good one is the most misleading possible output.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);

    WalkerPopulation pop;
    pop.add(toy.dets[1], 1.0);          // reference NOT occupied
    const auto pe = projected_energy(pop, toy.dets[0], 4, ops);
    check(!pe.valid, "projected energy is invalid when the reference is unoccupied");

    WalkerPopulation tiny;
    tiny.add(toy.dets[0], 1e-15);
    const auto pe2 = projected_energy(tiny, toy.dets[0], 4, ops);
    check(!pe2.valid, "projected energy is invalid when the reference weight is tiny");
}

static void test_projected_energy_bias_shrinks_with_population()
{
    // THE F3.4 assertion. The estimator is a ratio of stochastic quantities, so
    // it is biased at finite population. Gate the TREND, not a single value: a
    // population that happens to agree proves nothing, and the bias is negative
    // here, which is the direction that makes a result look more convincingly
    // variational than it is.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    const double exact = exact_ground_energy(toy);

    // Converged shape, to sample from.
    const double shift = -2.0;
    WalkerPopulation whole;
    for (const auto &d : toy.dets)
        whole.add(d, 1.0);
    const double dt = 0.1 * max_stable_timestep(whole, ops, shift);
    WalkerPopulation exact_state;
    exact_state.add(toy.dets[0], 1.0);
    for (int it = 0; it < 4000; ++it)
    {
        exact_state = propagate_deterministic(exact_state, 4, ops, dt, shift);
        const double nrm = ordered_l1_norm(exact_state);
        if (nrm == 0.0 || !std::isfinite(nrm))
            break;
        WalkerPopulation scaled;
        for (const auto &[det, w] : exact_state)
            scaled.add(det, w / nrm);
        exact_state = scaled;
    }

    // Sample walker populations of increasing size from that shape and measure
    // the mean projected energy at each size.
    // Population sizes are chosen so the REFERENCE is well occupied at every
    // point. At N=50 the reference carried c_0 = 1 -- a single walker -- and the
    // estimator, which divides by c_0, then swung between -5.7 and -6.9 against an
    // exact -10.0. That is the documented small-reference regime, not the
    // finite-population bias this test is about, and including it measures the
    // wrong thing while appearing to give a steeper trend.
    RandomSource rng(31337);
    std::vector<std::pair<double, double>> trend;   // (N, |bias|)
    for (int n_walkers : {800, 3200, 12800, 51200})
    {
        double sum_e = 0.0;
        int n_ok = 0;
        for (int trial = 0; trial < 3000; ++trial)
        {
            WalkerPopulation sampled;
            for (const auto &[det, w] : exact_state)
            {
                // Stochastic rounding to an integer walker count preserves the
                // expectation exactly -- rounding to nearest would bias it.
                const double target = w * n_walkers;
                const double n = rng.stochastic_round(target);
                if (n != 0.0)
                    sampled.add(det, n);
            }
            const auto pe = projected_energy(sampled, toy.dets[0], 4, ops);
            if (!pe.valid)
                continue;
            sum_e += pe.energy;
            ++n_ok;
            if (std::getenv("PLANCK_FCIQMC_VERBOSE") != nullptr && trial < 3)
                std::printf("        trial %d: c0=%.4f  E=%.6f\n",
                            trial, pe.reference_weight, pe.energy);
        }
        if (n_ok < 100)
        {
            check(false, "too few valid samples to measure the bias");
            return;
        }
        const double bias = std::abs(sum_e / n_ok - exact);
        trend.push_back({static_cast<double>(n_walkers), bias});
        if (std::getenv("PLANCK_FCIQMC_VERBOSE") != nullptr)
            std::printf("      N=%5d  bias=%.6e  bias*N=%.4f  (%d valid)\n",
                        n_walkers, bias, bias * n_walkers, n_ok);
    }

    // The bias must fall as the population grows. Require a clear decrease across
    // the 64x range rather than fitting an exponent -- the point is that it is a
    // bias that vanishes, not that it obeys a precise power law.
    const double first = trend.front().second;
    const double last = trend.back().second;
    if (!(last < first * 0.5))
    {
        std::printf("  [FAIL] projected-energy bias does not shrink with population "
                    "(%.3e at N=%.0f, %.3e at N=%.0f)\n",
                    first, trend.front().first, last, trend.back().first);
        ++g_failures;
    }

    // And it must reach the measurement floor -- a bias that shrinks but plateaus
    // far from the exact value would be a defect, not a bias.
    //
    // The threshold is the STANDARD ERROR of the measurement, not a constant: with
    // 3000 trials and a per-trial spread of ~0.1 the mean is only resolvable to
    // ~2e-3, so the largest population's apparent bias (measured 2.3e-4) is
    // already below what this many trials can distinguish from zero. Asserting a
    // tighter constant would be asserting noise.
    const double per_trial_sigma = 0.2;      // conservative, from observed spread
    const double resolution = per_trial_sigma / std::sqrt(3000.0);
    if (!(last < 3.0 * resolution))
    {
        std::printf("  [FAIL] projected-energy bias %.3e at N=%.0f exceeds the "
                    "measurement floor %.3e\n", last, trend.back().first,
                    3.0 * resolution);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// F3.5 -- fixed-seed reproducibility of a whole trajectory
// ---------------------------------------------------------------------------
//
// test_stochastic_reproducible (F3.2) checks single steps from a FIXED start.
// That cannot see a defect which accumulates: a trajectory feeds each step's
// output into the next, so state carried between iterations -- a stale buffer, a
// hash-order-dependent sum, a generator consulted a different number of times --
// drifts in over many steps while any single step still matches.
//
// This is the gate the FCIQMC research scope names as PRIMARY, because it is the
// one that survives at any system size: the statistical gate needs a deterministic
// reference and so only ever runs on small systems, while a fixed seed can be
// checked on Cr2 CAS(12,18) where no reference exists.

// Run a trajectory and digest it: every determinant weight at every recorded
// step, in a deterministic order. Returns a checksum over the raw bits.
static std::uint64_t trajectory_digest(const ToyHamiltonian &toy, int n_act,
                                       std::uint64_t seed, int n_steps,
                                       double dt, double shift)
{
    const auto ops = toy.ops(n_act);
    RandomSource rng(seed);
    WalkerPopulation pop;
    pop.add(toy.dets[0], 100.0);

    std::uint64_t h = 0xcbf29ce484222325ULL;   // FNV-1a offset basis
    auto mix_bits = [&h](double v) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(v), "double must be 64-bit");
        std::memcpy(&bits, &v, sizeof(bits));
        h ^= bits;
        h *= 0x100000001b3ULL;
    };

    for (int step = 0; step < n_steps; ++step)
    {
        pop = propagate_stochastic(pop, n_act, ops, dt, shift, rng);
        pop.compress(1e-14);

        // Digest in DETERMINANT order, not hash order -- otherwise the digest
        // itself would be non-reproducible and the test would fail for a reason
        // unrelated to the dynamics.
        for (const auto &d : toy.dets)
            mix_bits(pop.weight_at(d));
    }
    return h;
}

static void test_trajectory_is_reproducible()
{
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {4, 2, 2, "4 orbitals closed shell"},
        {5, 3, 2, "5 orbitals OPEN shell"},
    };

    for (const auto &c : cases)
    {
        const ToyHamiltonian toy(c.n_act, c.na, c.nb);
        const auto ops = toy.ops(c.n_act);
        WalkerPopulation whole;
        for (const auto &d : toy.dets)
            whole.add(d, 1.0);
        const double dt = 0.05 * max_stable_timestep(whole, ops, -2.0);

        // Same seed, twice: bitwise identical over a long trajectory.
        const auto d1 = trajectory_digest(toy, c.n_act, 12345, 300, dt, -2.0);
        const auto d2 = trajectory_digest(toy, c.n_act, 12345, 300, dt, -2.0);
        if (d1 != d2)
        {
            std::printf("  [FAIL] %s: same seed gave different trajectories\n", c.name);
            ++g_failures;
            continue;
        }

        // NEGATIVE CONTROL: a different seed must give a different trajectory.
        // Without this, a propagator that ignored its RNG entirely would pass the
        // check above trivially.
        const auto d3 = trajectory_digest(toy, c.n_act, 54321, 300, dt, -2.0);
        if (d1 == d3)
        {
            std::printf("  [FAIL] %s: different seeds gave the SAME trajectory "
                        "(is the RNG consulted at all?)\n", c.name);
            ++g_failures;
        }
    }
}

static void test_trajectory_digest_is_sensitive()
{
    // The digest must notice a single-ulp change in one weight at one step --
    // otherwise "identical trajectories" means only "similar enough", which is
    // exactly the tolerance this gate exists to avoid.
    std::uint64_t h1 = 0xcbf29ce484222325ULL, h2 = 0xcbf29ce484222325ULL;
    auto mix = [](std::uint64_t &h, double v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        h ^= bits;
        h *= 0x100000001b3ULL;
    };
    mix(h1, 1.0);
    mix(h1, 2.0);
    mix(h2, 1.0);
    mix(h2, std::nextafter(2.0, 3.0));   // one ulp away
    check(h1 != h2, "the trajectory digest distinguishes a one-ulp difference");
}

static void test_compression_does_not_break_reproducibility()
{
    // compress() erases from a hash map while iterating. If that ever became
    // order-dependent in a way that changed WHICH determinants survive, two runs
    // with the same seed could diverge. Pin it: the surviving set must be
    // identical regardless of insertion order.
    const ToyHamiltonian toy(4, 2, 2);

    WalkerPopulation a, b;
    for (std::size_t i = 0; i < toy.dets.size(); ++i)
        a.add(toy.dets[i], (i % 3 == 0) ? 1e-16 : 1.0);
    for (std::size_t i = toy.dets.size(); i > 0; --i)
        b.add(toy.dets[i - 1], ((i - 1) % 3 == 0) ? 1e-16 : 1.0);

    const std::size_t removed_a = a.compress(1e-14);
    const std::size_t removed_b = b.compress(1e-14);
    check(removed_a == removed_b, "compress removes the same count regardless of order");

    for (const auto &d : toy.dets)
        if (a.weight_at(d) != b.weight_at(d))
        {
            check(false, "compress leaves the same survivors regardless of order");
            return;
        }
}

// ---------------------------------------------------------------------------
// F4.1 -- shift control holds the population and finds the ground-state energy
// ---------------------------------------------------------------------------

struct ShiftRun
{
    double mean_shift = 0.0;
    double final_population = 0.0;
    double peak_ratio = 0.0;     // worst |N/target| seen after equilibration
    bool finite = true;
};

// Run the DETERMINISTIC propagator under shift control. Deterministic first, so
// the control loop is separated from sampling noise exactly as F3.1 separated
// the dynamics from the sampling.
static ShiftRun run_with_shift_control(const ToyHamiltonian &toy, int n_act,
                                       double dt, double zeta, double target,
                                       double start_population, int n_steps,
                                       int interval = 5)
{
    const auto ops = toy.ops(n_act);
    ShiftRun out;

    WalkerPopulation pop;
    pop.add(toy.dets[0], start_population);

    ShiftController ctl;
    ctl.shift = ops.diagonal(toy.dets[0]);
    ctl.target_population = target;
    ctl.zeta = zeta;
    ctl.interval = interval;

    double sum_shift = 0.0;
    int n_avg = 0;
    const int equilibrate = n_steps / 2;

    for (int step = 0; step < n_steps; ++step)
    {
        pop = propagate_deterministic(pop, n_act, ops, dt, ctl.shift);
        const double n = ordered_l1_norm(pop);
        if (!std::isfinite(n) || n <= 0.0)
        {
            out.finite = false;
            return out;
        }
        ctl.update(n, dt);

        if (step >= equilibrate)
        {
            sum_shift += ctl.shift;
            ++n_avg;
            out.peak_ratio = std::max(out.peak_ratio, n / target);
        }
        out.final_population = n;
    }
    out.mean_shift = (n_avg > 0) ? sum_shift / n_avg : 0.0;
    return out;
}

static void test_shift_control_converges_to_target()
{
    // Starting far off target in BOTH directions, the population must come back.
    // One direction alone would pass a controller that only ever shrinks (or only
    // grows) the population.
    const ToyHamiltonian toy(5, 3, 2);   // open shell, per F2's lesson
    const auto ops = toy.ops(5);
    WalkerPopulation whole;
    for (const auto &d : toy.dets)
        whole.add(d, 1.0);
    const double dt = 0.05 * max_stable_timestep(whole, ops, -2.0);
    const double target = 1000.0;

    // zeta = 1.5, from the band measured on this Hamiltonian (see
    // test_zeta_tradeoff_exists): at 0.3 the population is still 1200x high after
    // 6000 steps -- converging, but not converged, which is a statement about the
    // controller's time constant rather than about correctness.
    for (double start : {target * 10.0, target / 10.0})
    {
        const auto r = run_with_shift_control(toy, 5, dt, 1.5, target, start, 12000);
        check(r.finite, "shift-controlled run stays finite");
        if (!r.finite)
            continue;
        const double ratio = r.final_population / target;
        if (!(ratio > 0.2 && ratio < 5.0))
        {
            std::printf("  [FAIL] population did not return to target from %.0fx "
                        "(ended at %.2fx)\n", start / target, ratio);
            ++g_failures;
        }
    }
}

static void test_shift_converges_to_the_ground_state_energy()
{
    // The shift that holds the population steady IS an estimate of E0. This is a
    // second, independent estimator: it comes from the population growth rate, not
    // from any matrix element on the reference.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {4, 2, 2, "4 orbitals closed shell"},
        {5, 3, 2, "5 orbitals OPEN shell"},
    };

    for (const auto &c : cases)
    {
        const ToyHamiltonian toy(c.n_act, c.na, c.nb);
        const auto ops = toy.ops(c.n_act);
        WalkerPopulation whole;
        for (const auto &d : toy.dets)
            whole.add(d, 1.0);
        const double dt = 0.05 * max_stable_timestep(whole, ops, -2.0);

        const auto r = run_with_shift_control(toy, c.n_act, dt, 0.3, 1000.0, 1000.0, 8000);
        if (!r.finite)
        {
            check(false, "shift-controlled run stays finite");
            continue;
        }
        const double exact = exact_ground_energy(toy);
        if (std::abs(r.mean_shift - exact) > 1e-3)
        {
            std::printf("  [FAIL] %s: converged shift %.8f vs exact E0 %.8f\n",
                        c.name, r.mean_shift, exact);
            ++g_failures;
        }
    }
}

static void test_zeta_tradeoff_exists()
{
    // Assert the TRADEOFF, not a value -- and note WHAT is traded changed once the
    // target term was added.
    //
    // With only the growth-rate term, zeta traded shift accuracy against
    // population tightness. With both terms the xi term does the targeting, and
    // zeta becomes a STABILITY parameter: too little leaves the shift oscillating,
    // too much destabilises it. Measured on the 20-determinant scoping model
    // (xi = 0.05, start 20x above target):
    //
    //   zeta   peak/target   shift error   shift stdev
    //   0.0    20.36         4.6e-01       9.5e+00     (oscillates, never settles)
    //   0.1     1.000        1.6e-10       3.9e-10
    //   0.5     1.000        1.6e-10       3.9e-10
    //   2.0     1.98         3.8e+00       2.8e+01     (destabilised)
    //   5.0     DIVERGED
    //
    // So both ends fail, which is the evidence the feedback is real. A controller
    // that ignored zeta would give identical results at every setting.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    WalkerPopulation whole;
    for (const auto &d : toy.dets)
        whole.add(d, 1.0);
    const double dt = 0.05 * max_stable_timestep(whole, ops, -2.0);
    const double target = 1000.0;
    const double exact = exact_ground_energy(toy);

    const auto none = run_with_shift_control(toy, 4, dt, 0.0, target, target * 20.0, 8000);
    const auto good = run_with_shift_control(toy, 4, dt, 0.3, target, target * 20.0, 8000);
    const auto heavy = run_with_shift_control(toy, 4, dt, 4.0, target, target * 20.0, 8000);

    if (!none.finite || !good.finite)
    {
        check(false, "the zero- and mid-damping runs stay finite");
        return;
    }

    const double err_none = std::abs(none.mean_shift - exact);
    const double err_good = std::abs(good.mean_shift - exact);

    // Too little damping: the shift does not settle on the energy.
    if (!(err_none > err_good * 10.0))
    {
        std::printf("  [FAIL] zero damping did not degrade the shift "
                    "(none %.3e, good %.3e)\n", err_none, err_good);
        ++g_failures;
    }

    // Too much damping: the run either destabilises or its shift degrades. Accept
    // either, because which one happens depends on dt and the spectrum -- pinning
    // one would pin an accident.
    const bool heavy_degraded =
        !heavy.finite || std::abs(heavy.mean_shift - exact) > err_good * 10.0;
    if (!heavy_degraded)
    {
        std::printf("  [FAIL] heavy damping neither destabilised nor degraded the "
                    "shift (err %.3e vs %.3e)\n",
                    std::abs(heavy.mean_shift - exact), err_good);
        ++g_failures;
    }

    if (std::getenv("PLANCK_FCIQMC_VERBOSE") != nullptr)
        std::printf("      zeta=0: err=%.3e | zeta=0.3: err=%.3e | zeta=4: %s\n",
                    err_none, err_good,
                    heavy.finite ? "finite" : "diverged");
}

static void test_shift_update_has_correct_units()
{
    // The interval*dt denominator makes zeta DIMENSIONLESS: the log-ratio is a
    // growth over an interval, and dividing by the elapsed imaginary time turns it
    // into a rate, which is what has the units of an energy.
    //
    // Without it, the same zeta means a different feedback gain at every dt --
    // and the tradeoff tests cannot see that, because rescaling the denominator is
    // equivalent to rescaling zeta, which they deliberately do not pin. Measured:
    // dropping the denominator passed every other check in this file.
    //
    // So assert the scaling directly. Halving dt with interval fixed must halve
    // the shift correction for the same observed population ratio.
    auto correction_for = [](double dt, int interval) {
        ShiftController c;
        c.shift = 0.0;
        c.zeta = 1.0;
        c.xi = 0.0;                 // isolate the growth-rate term
        c.interval = interval;
        c.target_population = 1000.0;
        c.update(1000.0, dt);                    // prime
        for (int i = 0; i + 1 < interval; ++i)
            c.update(2000.0, dt);
        c.update(2000.0, dt);                    // triggers the update
        return c.shift;
    };

    const double s1 = correction_for(0.01, 5);
    const double s2 = correction_for(0.005, 5);
    check(std::abs(s2 - 2.0 * s1) < 1e-12 * std::abs(s2),
          "halving dt doubles the shift correction (zeta is a rate, not a step)");

    const double s3 = correction_for(0.01, 10);
    check(std::abs(s3 - 0.5 * s1) < 1e-12 * std::abs(s1),
          "doubling the interval halves the shift correction");
}

static void test_target_term_is_required()
{
    // The growth-rate term alone stabilises the RATE but never targets a
    // population: the final population comes out proportional to the starting one.
    // Measured with xi = 0 on the scoping model: 135.7x the target from every
    // start across a 1000x range. This pins that the target term is doing real
    // work, so nobody removes it as redundant.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    WalkerPopulation whole;
    for (const auto &d : toy.dets)
        whole.add(d, 1.0);
    const double dt = 0.05 * max_stable_timestep(whole, ops, -2.0);
    const double target = 1000.0;

    auto final_ratio = [&](double xi, double start) {
        WalkerPopulation pop;
        pop.add(toy.dets[0], start);
        ShiftController ctl;
        ctl.shift = ops.diagonal(toy.dets[0]);
        ctl.target_population = target;
        ctl.zeta = 0.3;
        ctl.xi = xi;
        ctl.interval = 5;
        for (int step = 0; step < 12000; ++step)
        {
            pop = propagate_deterministic(pop, 4, ops, dt, ctl.shift);
            const double n = ordered_l1_norm(pop);
            if (!std::isfinite(n) || n <= 0.0)
                return -1.0;
            ctl.update(n, dt);
        }
        return ordered_l1_norm(pop) / target;
    };

    // With the target term: same answer from both directions.
    const double hi_on = final_ratio(0.05, target * 20.0);
    const double lo_on = final_ratio(0.05, target / 20.0);
    check(hi_on > 0.5 && hi_on < 2.0, "with the target term, a high start lands on target");
    check(lo_on > 0.5 && lo_on < 2.0, "with the target term, a low start lands on target");

    // Without it: the final population tracks the start instead.
    const double hi_off = final_ratio(0.0, target * 20.0);
    const double lo_off = final_ratio(0.0, target / 20.0);
    if (hi_off > 0.0 && lo_off > 0.0 && !(hi_off > lo_off * 10.0))
    {
        std::printf("  [FAIL] without the target term the final population should "
                    "track the start (high %.2fx, low %.2fx)\n", hi_off, lo_off);
        ++g_failures;
    }
}

static void test_shift_controller_primes_before_updating()
{
    // The first interval only establishes a baseline. Updating from a zero or
    // absent reference population would send the shift to infinity on step one.
    ShiftController ctl;
    ctl.shift = -2.0;
    ctl.interval = 5;
    ctl.zeta = 0.5;

    check(!ctl.update(1000.0, 0.01), "the first call only primes, it does not update");
    check(ctl.shift == -2.0, "priming leaves the shift untouched");

    for (int i = 0; i < 4; ++i)
        check(!ctl.update(1000.0, 0.01), "no update before the interval elapses");
    check(ctl.update(1000.0, 0.01), "an update happens once the interval elapses");
    check(ctl.shift == -2.0, "a steady population leaves the shift unchanged");

    // A growing population must LOWER the shift (less growth next step).
    ShiftController g = ctl;
    g.update(2000.0, 0.01);
    for (int i = 0; i < ctl.interval; ++i)
        g.update(2000.0, 0.01);
    check(g.shift < -2.0, "a growing population lowers the shift");

    // Degenerate inputs must be ignored, not propagated as infinities.
    ShiftController d = ctl;
    const double before = d.shift;
    check(!d.update(0.0, 0.01), "a dead population produces no update");
    check(!d.update(std::numeric_limits<double>::infinity(), 0.01),
          "a diverged population produces no update");
    check(d.shift == before, "degenerate populations leave the shift untouched");
}

// ---------------------------------------------------------------------------
// F4.2 -- the shift energy, cross-checked against the projected energy
// ---------------------------------------------------------------------------

struct TwoEstimators
{
    double shift_energy = 0.0;
    double projected_energy_mean = 0.0;
    double shift_from_zero = 0.0;   // averaged WITHOUT discarding the transient
    int shift_samples = 0;
    bool finite = true;
};

static TwoEstimators run_both_estimators(const ToyHamiltonian &toy, int n_act,
                                         double dt, double target,
                                         double start_population, int n_steps)
{
    const auto ops = toy.ops(n_act);
    TwoEstimators out;

    WalkerPopulation pop;
    pop.add(toy.dets[0], start_population);

    ShiftController ctl;
    ctl.shift = ops.diagonal(toy.dets[0]);
    ctl.target_population = target;
    ctl.zeta = 0.3;
    ctl.xi = 0.05;
    ctl.interval = 5;

    const int equilibrate = n_steps / 2;
    ShiftAverager avg(equilibrate);
    ShiftAverager avg_from_zero(0);

    double sum_proj = 0.0;
    int n_proj = 0;

    for (int step = 0; step < n_steps; ++step)
    {
        pop = propagate_deterministic(pop, n_act, ops, dt, ctl.shift);
        const double n = ordered_l1_norm(pop);
        if (!std::isfinite(n) || n <= 0.0)
        {
            out.finite = false;
            return out;
        }
        ctl.update(n, dt);
        avg.add(ctl.shift);
        avg_from_zero.add(ctl.shift);

        if (step >= equilibrate)
        {
            const auto pe = projected_energy(pop, toy.dets[0], n_act, ops);
            if (pe.valid)
            {
                sum_proj += pe.energy;
                ++n_proj;
            }
        }
    }

    out.shift_energy = avg.mean();
    out.shift_from_zero = avg_from_zero.mean();
    out.shift_samples = avg.samples();
    out.projected_energy_mean = (n_proj > 0) ? sum_proj / n_proj : 0.0;
    return out;
}

static void test_two_estimators_agree()
{
    // THE self-validating check for this step. The shift energy comes from the
    // population growth rate; the projected energy from a ratio of walker weights
    // on the reference. They share no arithmetic, so agreement is evidence for
    // both -- a defect in one would have to be exactly mirrored in the other.
    //
    // Checked across a 100x range of target populations, because an estimator
    // that agreed only at one population would be fitting rather than measuring.
    struct Case { int n_act, na, nb; const char *name; };
    const Case cases[] = {
        {4, 2, 2, "4 orbitals closed shell"},
        {5, 3, 2, "5 orbitals OPEN shell"},
    };

    for (const auto &c : cases)
    {
        const ToyHamiltonian toy(c.n_act, c.na, c.nb);
        const auto ops = toy.ops(c.n_act);
        WalkerPopulation whole;
        for (const auto &d : toy.dets)
            whole.add(d, 1.0);
        const double dt = 0.05 * max_stable_timestep(whole, ops, -2.0);
        const double exact = exact_ground_energy(toy);

        for (double target : {100.0, 1000.0, 10000.0})
        {
            const auto r = run_both_estimators(toy, c.n_act, dt, target, target, 8000);
            if (!r.finite)
            {
                std::printf("  [FAIL] %s at target %.0f: run did not stay finite\n",
                            c.name, target);
                ++g_failures;
                continue;
            }

            const double gap = std::abs(r.shift_energy - r.projected_energy_mean);
            if (!(gap < 1e-4))
            {
                std::printf("  [FAIL] %s at target %.0f: shift %.10f vs projected "
                            "%.10f (gap %.3e)\n", c.name, target,
                            r.shift_energy, r.projected_energy_mean, gap);
                ++g_failures;
            }

            // Both must also be right, not merely equal to each other. Two
            // estimators can agree by sharing a common upstream defect -- here
            // the propagator -- so pin them to the exact energy too.
            if (std::abs(r.shift_energy - exact) > 1e-3)
            {
                std::printf("  [FAIL] %s at target %.0f: shift energy %.10f vs "
                            "exact %.10f\n", c.name, target, r.shift_energy, exact);
                ++g_failures;
            }

            if (std::getenv("PLANCK_FCIQMC_VERBOSE") != nullptr)
                std::printf("      %s target=%7.0f shift=%.8f proj=%.8f gap=%.2e\n",
                            c.name, target, r.shift_energy,
                            r.projected_energy_mean, gap);
        }
    }
}

static void test_equilibration_cut_matters()
{
    // Vacuity check on the fixture: if averaging from iteration 0 gives the same
    // answer as averaging after equilibration, the run reached equilibrium
    // immediately and the discard is not being tested at all.
    //
    // Start far off target so there IS a transient to discard.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    WalkerPopulation whole;
    for (const auto &d : toy.dets)
        whole.add(d, 1.0);
    const double dt = 0.05 * max_stable_timestep(whole, ops, -2.0);
    const double exact = exact_ground_energy(toy);

    const auto r = run_both_estimators(toy, 4, dt, 1000.0, 1000.0 * 50.0, 8000);
    check(r.finite, "the off-target run stays finite");
    if (!r.finite)
        return;

    const double err_cut = std::abs(r.shift_energy - exact);
    const double err_raw = std::abs(r.shift_from_zero - exact);

    if (!(err_raw > err_cut * 2.0))
    {
        std::printf("  [FAIL] discarding the transient did not improve the shift "
                    "energy (cut %.3e, raw %.3e) -- the fixture equilibrates too "
                    "fast to test the discard\n", err_cut, err_raw);
        ++g_failures;
    }

    if (std::getenv("PLANCK_FCIQMC_VERBOSE") != nullptr)
        std::printf("      equilibration: err with cut=%.3e, without=%.3e\n",
                    err_cut, err_raw);
}

static void test_shift_averager_semantics()
{
    ShiftAverager a(3);
    for (int i = 0; i < 3; ++i)
        a.add(100.0);            // discarded
    check(!a.valid(), "an averager with only discarded samples is not valid");
    check(a.discarded() == 3, "the averager reports how many it discarded");

    a.add(1.0);
    a.add(3.0);
    check(a.valid(), "two kept samples make it valid");
    check_close(a.mean(), 2.0, 1e-15, "the mean ignores the discarded transient");
    check(a.samples() == 2, "sample count excludes the discarded ones");

    // A constant series has zero standard error -- and must not produce NaN.
    ShiftAverager c(0);
    for (int i = 0; i < 10; ++i)
        c.add(-5.0);
    check_close(c.mean(), -5.0, 1e-15, "a constant series averages to itself");
    check(c.naive_standard_error() == 0.0, "a constant series has zero standard error");

    ShiftAverager one(0);
    one.add(1.0);
    check(!one.valid(), "a single sample is not enough for a standard error");
    check(std::isnan(one.naive_standard_error()), "one sample gives NaN, not 0");
}

// ---------------------------------------------------------------------------
// F4.3 -- the timestep divergence gate F3 could not construct
// ---------------------------------------------------------------------------
//
// F3 tried three times to assert that too large a dt breaks the propagation, and
// every premise turned out false: the population does not collapse onto one
// determinant, the norm grows at every dt by design, and the converged shape
// overlaps the true ground state even at 5x the bound. The cause was that
// renormalizing each iteration turns the propagation into a power iteration whose
// dominant eigenvector stays the ground state -- instability had nowhere to show.
//
// With the population CONTROLLED it does, and the boundary is sharp. Measured on
// the 36-determinant fixture (zeta = 0.3, xi = 0.05, target 1000, 8000 steps),
// as a fraction of the diagonal timestep bound:
//
//   dt/bound   outcome
//   0.10       settles, shift -9.971196
//   0.26       settles, shift -9.971196
//   0.30       DIVERGES
//   0.60       DIVERGES
//
// WHAT THIS ACTUALLY DETECTS is worth stating precisely: the boundary at
// 0.26-0.30 sits BELOW the true spectral stability limit of the propagator
// (~0.44x the diagonal bound, since the diagonal form is 2.28x too large). So the
// controller destabilises before the bare propagator would. That makes this a
// gate on the CONTROLLED dynamics -- which is the thing a real run uses -- not a
// measurement of the propagator's own stability limit. Do not quote the boundary
// found here as the propagator's bound.

// Returns the final population as a multiple of target, or infinity if the run
// blew up.
//
// Returning the VALUE rather than a bool was a deliberate change: a boolean
// predicate mutated to a constant made the "must not settle" assertions unable to
// fail, and that mutation passed the whole gate. With the ratio exposed the
// assertions compare numbers instead.
//
// HONEST LIMITATION, recorded rather than papered over: the two exit paths --
// the in-loop blow-up check and the final-ratio return -- are REDUNDANT here.
// Divergence on this fixture is violent enough to trip the in-loop check first,
// so mutating the final return alone changes nothing (measured: 0.40x still
// reports inf, from the early exit). Either path alone would catch the cases
// tested. That is fine, but it means no single-line mutation of this helper can
// demonstrate both paths are load-bearing -- do not read a passing mutation of
// one of them as evidence the gate is weak.
static double controlled_run_final_ratio(const ToyHamiltonian &toy, int n_act,
                                         double dt, double target, int n_steps)
{
    const auto ops = toy.ops(n_act);
    WalkerPopulation pop;
    pop.add(toy.dets[0], target);

    ShiftController ctl;
    ctl.shift = ops.diagonal(toy.dets[0]);
    ctl.target_population = target;
    ctl.zeta = 0.3;
    ctl.xi = 0.05;
    ctl.interval = 5;

    for (int step = 0; step < n_steps; ++step)
    {
        pop = propagate_deterministic(pop, n_act, ops, dt, ctl.shift);
        const double n = ordered_l1_norm(pop);
        if (!std::isfinite(n) || n <= 0.0 || n > target * 1e12)
            return std::numeric_limits<double>::infinity();
        ctl.update(n, dt);
    }
    const double final_n = ordered_l1_norm(pop);
    if (!std::isfinite(final_n))
        return std::numeric_limits<double>::infinity();
    return final_n / target;
}

// A run "settles" when the controlled population is within two orders of the
// target in either direction.
static bool settled(double ratio)
{
    return std::isfinite(ratio) && ratio > 0.01 && ratio < 100.0;
}

static void test_timestep_divergence_is_observable_under_control()
{
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    WalkerPopulation whole;
    for (const auto &d : toy.dets)
        whole.add(d, 1.0);
    const double bound = max_stable_timestep(whole, ops, -2.0);

    const double r_small = controlled_run_final_ratio(toy, 4, 0.10 * bound, 1000.0, 8000);
    const double r_mid_lo = controlled_run_final_ratio(toy, 4, 0.20 * bound, 1000.0, 8000);
    const double r_mid_hi = controlled_run_final_ratio(toy, 4, 0.40 * bound, 1000.0, 8000);
    const double r_large = controlled_run_final_ratio(toy, 4, 0.60 * bound, 1000.0, 8000);

    // Inside the stable region the controller holds the population AT target, not
    // merely somewhere finite. Asserting the value is what makes a constant-return
    // mutation of the helper detectable.
    check(std::abs(r_small - 1.0) < 0.5,
          "a small timestep holds the population at target");
    check(std::abs(r_mid_lo - 1.0) < 0.5,
          "0.20x the bound still holds the population at target");

    // Outside it, the population leaves that window entirely.
    check(!settled(r_mid_hi), "0.40x the bound fails to settle");
    check(!settled(r_large), "0.60x the bound fails to settle");

    // And the two regimes must be far apart, so the gate is measuring a
    // transition rather than two points that happen to differ slightly.
    check(!std::isfinite(r_large) || r_large > 100.0 * std::abs(r_small),
          "the unstable regime is qualitatively different, not marginally so");

    if (std::getenv("PLANCK_FCIQMC_VERBOSE") != nullptr)
        std::printf("      final N/target: 0.10x=%.3f 0.20x=%.3f 0.40x=%.3g "
                    "0.60x=%.3g\n", r_small, r_mid_lo, r_mid_hi, r_large);
}

static void test_divergence_gate_is_not_vacuous()
{
    // The gate above would pass trivially if EVERY run failed to settle -- so
    // confirm the stable case is genuinely stable, by checking it reaches the
    // right energy rather than merely staying finite.
    const ToyHamiltonian toy(4, 2, 2);
    const auto ops = toy.ops(4);
    WalkerPopulation whole;
    for (const auto &d : toy.dets)
        whole.add(d, 1.0);
    const double bound = max_stable_timestep(whole, ops, -2.0);
    const double exact = exact_ground_energy(toy);

    const auto r = run_both_estimators(toy, 4, 0.10 * bound, 1000.0, 1000.0, 8000);
    check(r.finite, "the stable-timestep run is finite");
    check(std::abs(r.shift_energy - exact) < 1e-3,
          "the stable-timestep run also reaches the right energy");
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

    std::printf("F2.2 -- slow uniform generator\n");
    test_uniform_generator_support();
    test_uniform_generator_frequencies();
    test_uniform_generator_h2_exact();
    test_uniform_generator_reproducible();
    test_uniform_generator_degenerate();

    std::printf("F2.3 -- O(1) production generator\n");
    test_production_generator_support();
    test_production_generator_frequencies();
    test_production_generator_p_gen_normalizes();
    test_production_generator_is_non_uniform();
    test_production_matches_oracle_phases();
    test_production_generator_reproducible();
    test_open_shell_support_and_frequency();

    std::printf("F2.4 -- the gate rejects broken generators\n");
    test_gate_rejects_broken_generators();

    std::printf("F2.5 -- spin and symmetry constraints\n");
    test_particle_number_is_structural();
    test_in_space_rejection_sampling();
    test_in_space_p_gen_is_corrected();
    test_per_call_acceptance_estimate_is_biased();

    std::printf("F3.1 -- deterministic propagation equals a matvec\n");
    test_toy_hamiltonian_is_symmetric();
    test_deterministic_step_is_a_matvec();
    test_deterministic_step_only_visits_connections();
    test_deterministic_step_annihilates();
    test_deterministic_step_linear();

    std::printf("F3.2 -- stochastic spawning reproduces the deterministic mean\n");
    test_stochastic_mean_matches_deterministic();
    test_stochastic_variance_falls_with_attempts();
    test_stochastic_attempts_do_not_rescale();
    test_stochastic_death_is_exact();
    test_stochastic_reproducible();

    std::printf("F3.3 -- propagation reaches the ground state\n");
    test_timestep_bound_is_computed();
    test_propagation_reaches_the_ground_state();
    test_ordered_norm_is_deterministic();

    std::printf("F3.4 -- projected energy and its finite-population bias\n");
    test_projected_energy_exact_on_converged_state();
    test_projected_energy_rejects_small_reference();
    test_projected_energy_bias_shrinks_with_population();

    std::printf("F3.5 -- fixed-seed trajectory reproducibility\n");
    test_trajectory_is_reproducible();
    test_trajectory_digest_is_sensitive();
    test_compression_does_not_break_reproducibility();

    std::printf("F4.1 -- shift control\n");
    test_shift_controller_primes_before_updating();
    test_shift_control_converges_to_target();
    test_shift_converges_to_the_ground_state_energy();
    test_zeta_tradeoff_exists();
    test_shift_update_has_correct_units();
    test_target_term_is_required();

    std::printf("F4.2 -- shift energy cross-checked against the projected energy\n");
    test_shift_averager_semantics();
    test_two_estimators_agree();
    test_equilibration_cut_matters();

    std::printf("F4.3 -- timestep divergence, observable under population control\n");
    test_timestep_divergence_is_observable_under_control();
    test_divergence_gate_is_not_vacuous();

    if (g_failures == 0)
    {
        std::printf("All checks passed.\n");
        return 0;
    }
    std::printf("%d FAILURE(S).\n", g_failures);
    return 1;
}
