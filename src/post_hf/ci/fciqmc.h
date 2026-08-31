#ifndef HF_POSTHF_CI_FCIQMC_H
#define HF_POSTHF_CI_FCIQMC_H

#include "post_hf/ci/strings.h"

#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

// Full Configuration Interaction Quantum Monte Carlo.
//
// FCIQMC represents the wavefunction as a population of signed walkers on
// determinants, evolved in imaginary time, instead of storing a CI vector. That
// is what lets it reach spaces deterministic FCI cannot: the target case is Cr2
// CAS(12,18), where one CI vector alone is 2.76 GB. See
// docs/FCIQMC_RESEARCH_SCOPE.md.
//
// This file is the walker state layer (scope step F1): the sparse population and
// the RNG policy. It deliberately contains no dynamics -- spawning, death and
// annihilation come later and build on these types.
//
// TEACHING NOTE. The three-line summary of the whole method:
//
//   1. SPAWN   -- each walker attempts to create children on connected
//                 determinants, with probability set by the off-diagonal
//                 Hamiltonian element H_ij.
//   2. DEATH   -- each walker survives with probability set by its own diagonal
//                 element H_ii relative to a shift S.
//   3. ANNIHILATE -- walkers of opposite sign on the SAME determinant cancel.
//
// Step 3 is the one that makes the method work rather than merely sample: the
// sign problem in FCIQMC is controlled because opposite-signed walkers meeting
// on a determinant destroy each other, which is only possible because the
// population is stored sparsely by determinant. That is why the container below
// is a map keyed on the determinant rather than a list of walkers.

namespace HartreeFock::Correlation::CI::QMC
{

    using HartreeFock::Correlation::CASSCFInternal::CIString;

    // A determinant key: the packed alpha/beta pair, same representation the CI
    // determinant layer already uses (see pack_spin_det in strings.h). Reusing it
    // means slater_condon_element and build_ci_diagonal apply unchanged.
    struct DetKey
    {
        CIString alpha = 0;
        CIString beta = 0;

        bool operator==(const DetKey &other) const noexcept
        {
            return alpha == other.alpha && beta == other.beta;
        }
    };

    struct DetKeyHash
    {
        std::size_t operator()(const DetKey &k) const noexcept
        {
            // splitmix64 finalizer on each half, then combine. The determinant
            // bitstrings are dense in the low bits and highly structured (fixed
            // popcount), so std::hash<uint64_t> -- identity on libstdc++ -- would
            // pile them into a narrow range of buckets.
            auto mix = [](std::uint64_t x) noexcept
            {
                x += 0x9e3779b97f4a7c15ULL;
                x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
                x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
                return x ^ (x >> 31);
            };
            return static_cast<std::size_t>(mix(k.alpha) ^ (mix(k.beta) << 1));
        }
    };

    // Walker weights are real-valued, not integer counts.
    //
    // The original method used integer walkers; real weights are the standard
    // modern choice because they remove the spawning discretization noise without
    // changing the algorithm's structure. The sign is what matters physically --
    // annihilation is cancellation of opposite signs.
    using Weight = double;

    // The sparse walker population: determinant -> signed weight.
    //
    // This is the piece the existing CI layer cannot supply. CIDeterminantSpace
    // indexes a FIXED, fully enumerated space (det_lookup maps into a dense array
    // built once); an FCIQMC population must grow and shrink every iteration, and
    // for the target case the enumerated space does not fit in memory at all.
    class WalkerPopulation
    {
    public:
        void clear() noexcept { _walkers.clear(); }

        std::size_t size() const noexcept { return _walkers.size(); }
        bool empty() const noexcept { return _walkers.empty(); }

        // Add weight to a determinant, creating it if absent.
        //
        // This IS the annihilation step: adding a negative weight to a
        // positive-weight determinant cancels it. There is no separate
        // annihilation pass -- it falls out of accumulating signed weights into a
        // map keyed by determinant.
        void add(const DetKey &det, Weight w)
        {
            if (w == 0.0)
                return;
            _walkers[det] += w;
        }

        Weight weight_at(const DetKey &det) const
        {
            const auto it = _walkers.find(det);
            return it == _walkers.end() ? 0.0 : it->second;
        }

        // Drop determinants whose weight has fallen below `threshold` in
        // magnitude, returning the number removed.
        //
        // Without this the map grows monotonically: annihilation leaves exact or
        // near-exact zeros behind, and every determinant ever touched by a single
        // spawn stays resident forever. Compression is what keeps the memory
        // footprint proportional to the OCCUPIED space rather than the visited one.
        std::size_t compress(Weight threshold = 0.0);

        // Sum of |weight| -- the walker number, the quantity population control
        // steers toward a target.
        Weight total_population() const noexcept;

        // Iteration order is unspecified (it is a hash map). Any reduction over
        // this must therefore be order-independent or explicitly sorted; see the
        // determinism discussion in docs/FCIQMC_RESEARCH_SCOPE.md.
        auto begin() const noexcept { return _walkers.begin(); }
        auto end() const noexcept { return _walkers.end(); }

    private:
        std::unordered_map<DetKey, Weight, DetKeyHash> _walkers;
    };

    // ------------------------------------------------------------------
    // Excitation layer (scope step F2)
    // ------------------------------------------------------------------

    // One excitation from a parent determinant.
    //
    // `phase` and `p_gen` are DIFFERENT quantities and must never be folded
    // together: `phase` is the fermionic sign from operator ordering (+1/-1),
    // `p_gen` is the probability that a draw produced this excitation. Conflating
    // them turns a sign error into something that looks like sampling bias.
    struct Excitation
    {
        DetKey det;
        double phase = 0.0;
        double p_gen = 0.0;
        bool valid = false;
    };

    // Enumerate EVERY determinant connected to `parent` by a single or double
    // excitation, with fermionic phases. This is the brute-force oracle.
    //
    // It is deliberately slow and obviously correct: nested loops over
    // occupied/virtual pairs, no index arithmetic to get wrong. Its job is to be
    // the measuring instrument for the sampling generator, so that the generator
    // is never the only implementation of "what is connected to what".
    //
    // Only singles and doubles appear because a Slater-Condon matrix element
    // vanishes beyond a double excitation -- the Hamiltonian is a two-body
    // operator, so it cannot connect determinants differing in more than two
    // orbitals. That is the fact the entire sparsity of the method rests on.
    //
    // Excludes the parent itself. `p_gen` is left at 0 -- enumeration is not
    // sampling, and filling it in here would invite exactly the self-consistency
    // check that F2's scope forbids.
    std::vector<Excitation> enumerate_connections(
        const DetKey &parent,
        int n_act);

    // Seeded RNG with an explicit reproducibility contract.
    //
    // The contract: a run with the same seed must reproduce its trajectory
    // BITWISE. That is the gate which survives at any system size -- unlike the
    // statistical gate, which needs a deterministic reference and so only ever
    // runs on small systems (tests/reproducibility.py).
    //
    // Consequently the draw order must not depend on thread count. A per-shard
    // generator seeded deterministically from the run seed satisfies this; drawing
    // from one shared generator across threads does not.
    class RandomSource
    {
    public:
        explicit RandomSource(std::uint64_t seed) noexcept : _engine(seed), _seed(seed) {}

        // Uniform in [0, 1).
        double uniform() noexcept
        {
            return std::generate_canonical<double, 53>(_engine);
        }

        // Uniform integer in [0, n).
        int uniform_int(int n) noexcept
        {
            return static_cast<int>(uniform() * static_cast<double>(n));
        }

        // Stochastic rounding: returns `floor(x)` or `ceil(x)` such that the
        // expectation is exactly x. This is how a fractional spawn or death is
        // realized without introducing bias -- rounding to nearest would
        // systematically discard small weights and bias the energy.
        double stochastic_round(double x) noexcept
        {
            const double base = std::floor(std::abs(x));
            const double frac = std::abs(x) - base;
            double mag = base + (uniform() < frac ? 1.0 : 0.0);
            return x < 0.0 ? -mag : mag;
        }

        std::uint64_t seed() const noexcept { return _seed; }

        // Derive an independent generator for shard `index`. Deterministic in the
        // run seed, so the set of streams does not depend on how many shards
        // (threads) are used -- only which one draws which.
        RandomSource derive(std::uint64_t index) const noexcept
        {
            std::uint64_t s = _seed + 0x9e3779b97f4a7c15ULL * (index + 1);
            s = (s ^ (s >> 30)) * 0xbf58476d1ce4e5b9ULL;
            s = (s ^ (s >> 27)) * 0x94d049bb133111ebULL;
            return RandomSource(s ^ (s >> 31));
        }

    private:
        std::mt19937_64 _engine;
        std::uint64_t _seed;
    };

    // Draw one excitation uniformly from the full connection set (scope step
    // F2.2).
    //
    // This is the SLOW, obviously-correct sampler: it enumerates every connection
    // on every call and picks one uniformly, so `p_gen = 1/|connections|`.
    // Production must not do this -- the target case has 7308 connections per
    // determinant and enumerating them per walker per iteration defeats the point
    // of the method. It exists as the reference distribution the fast generator
    // (F2.3) must reproduce, and as the thing a fast generator can be diffed
    // against when it disagrees with the oracle.
    //
    // Returns `valid = false` only when the parent has no connections at all,
    // which happens for a one-determinant space.
    //
    // TEACHING NOTE. `p_gen` is uniform HERE, and that is a property of this
    // particular sampler, not a requirement. A production generator deliberately
    // biases toward cheap or important excitations, so its p_gen varies -- by
    // 13.5x across connections on N2/STO-3G, for instance. Non-uniformity is not
    // the bug. The bug is a returned p_gen that disagrees with the sampler's
    // actual distribution, because the spawn divides by it: |H_ij| / p_gen. An
    // over-reported p_gen silently suppresses those spawns, an under-reported one
    // inflates them, and neither is visible in the trajectory.
    Excitation draw_uniform_excitation(
        const DetKey &parent,
        int n_act,
        RandomSource &rng);

    // Draw one excitation in O(1) -- the production generator (scope step F2.3).
    //
    // Picks one of five classes (alpha single, beta single, alpha-alpha double,
    // beta-beta double, alpha-beta double) with EQUAL probability among the
    // non-empty ones, then picks uniformly within the chosen class by index
    // arithmetic. No enumeration: cost is independent of the connection count,
    // which is the whole point -- the Cr2 target has 7308 connections per
    // determinant and F2.2's enumerate-per-call would dominate the run.
    //
    // The resulting `p_gen` is NON-UNIFORM: equal class probability over unequal
    // class sizes means a connection in a small class is far likelier than one in
    // a large class. Measured spread: 10x on water/STO-3G, 21x on N2/STO-3G. That
    // is legitimate -- what matters is that the returned p_gen equals the true
    // probability of this draw, which is what the F2.3 gate checks against
    // measured frequencies.
    //
    //   p_gen = (1 / n_live_classes) * (1 / class_size)
    //
    // This is also where the SUPPORT check starts earning its place. With a
    // uniform generator a support hole shows up in the frequencies anyway (~54
    // sigma for 1 missing connection in 140). With a non-uniform one, a rare
    // connection that is never generated deviates by ~0.6 sigma, which no
    // frequency test will flag -- only comparing the reachable set against the
    // oracle catches it.
    Excitation draw_excitation(
        const DetKey &parent,
        int n_act,
        RandomSource &rng);

    // Is `det` inside the sampled space? (scope step F2.5)
    //
    // The generator conserves particle number per spin channel structurally --
    // it annihilates and creates exactly once per spin -- so that constraint
    // needs no filter, only a gate asserting it holds.
    //
    // SYMMETRY is different, and is the trap this step exists for. When the CI
    // space is restricted (target irrep, or RAS occupancy limits) the generator
    // will happily produce determinants outside it. Discarding those is
    // legitimate -- but ONLY if the discard is reflected in the accepted
    // determinants' p_gen, because the spawn divides by p_gen.
    //
    // Concretely: if a fraction f of draws is discarded, the surviving draws
    // occur f-times less often than the unrestricted p_gen claims, so using the
    // unrestricted p_gen over-reports and silently suppresses those spawns. That
    // is a plausible, converged, WRONG energy -- the exact failure this scope
    // exists to prevent.
    //
    // Two ways to stay honest, and the choice belongs to the caller:
    //
    //   (a) REJECTION SAMPLING -- redraw until the excitation is in-space, and
    //       report p_gen / p_accept. Correct, and the acceptance rate is
    //       measurable, but the cost grows as the space gets more restricted.
    //   (b) DISCARD AND RENORMALIZE -- keep one draw, and let the discarded
    //       weight be absorbed by the shift. Cheaper, but it changes the
    //       normalization of the sampled operator, so it must be a deliberate
    //       decision rather than an accident.
    //
    // Neither is implemented here: F2.5 provides the PREDICATE and the gate, so
    // that whichever policy the spawn adopts can be checked. What is forbidden is
    // discarding silently while reporting the unrestricted p_gen.
    using InSpacePredicate = std::function<bool(const DetKey &)>;

    // Draw an in-space excitation by rejection sampling, reporting a p_gen that
    // accounts for the rejections (option (a) above).
    //
    // `p_accept` MUST be a fixed, pre-measured acceptance rate for this parent --
    // NOT estimated from the attempts this call happened to take.
    //
    // That distinction is not pedantic; getting it wrong is a 1.7x bias, measured.
    // Using the per-call attempt count gives an estimator that is unbiased for
    // p_gen itself, but the spawn uses |H_ij| / p_gen, and E[1/X] != 1/E[X]
    // (Jensen). With p_accept = 0.3 the per-call form over-weights spawns by 1.72x
    // while looking correct in every check of p_gen's mean. Measure the acceptance
    // rate once over many draws with measure_acceptance_rate() and pass it in.
    //
    // Returns valid = false if no in-space excitation was found within
    // `max_attempts`, which the caller must handle rather than treating as "no
    // connections".
    Excitation draw_excitation_in_space(
        const DetKey &parent,
        int n_act,
        RandomSource &rng,
        const InSpacePredicate &in_space,
        double p_accept,
        int max_attempts = 100);

    // Measure the fraction of draws from `parent` that land in-space.
    //
    // Run once per parent (or once per space, if the rate is uniform enough) and
    // feed the result to draw_excitation_in_space. Separating measurement from
    // sampling is what keeps the p_gen correction a constant rather than a
    // random variable -- see the Jensen note above.
    double measure_acceptance_rate(
        const DetKey &parent,
        int n_act,
        RandomSource &rng,
        const InSpacePredicate &in_space,
        int n_samples = 10000);

} // namespace HartreeFock::Correlation::CI::QMC

#endif
