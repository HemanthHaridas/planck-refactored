#ifndef HF_POSTHF_CI_FCIQMC_H
#define HF_POSTHF_CI_FCIQMC_H

#include "post_hf/ci/strings.h"

#include <cstdint>
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

} // namespace HartreeFock::Correlation::CI::QMC

#endif
