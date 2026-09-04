// C0 gate (caller side): seed_arbitrary_order_amplitudes must apply a seed's
// higher Sz sectors (ArbitraryOrderRCCAmplitudes::sectors), not just by_rank.
//
// Before this fix, a cc4 restart from a .ccamp sidecar carrying an
// independent (4, "aaabaaab") block silently left that sector at zero on the
// LIVE state -- the sidecar-side drop (cc_amplitude_checkpoint.cpp) is
// covered by tests/cc_amplitude_checkpoint.cpp; this test covers the
// separate defect on the seed-application side, which the file-format fix
// alone does not touch (the hook only ever read state.amplitudes.by_rank[r]
// = seed.by_rank[r], never state.amplitudes.sectors).
//
// Only seed_arbitrary_order_amplitudes's own contract is exercised here --
// state.amplitudes is populated directly (matching what ensure_amplitude_sectors
// would produce: pre-allocated zero sectors at the live state's own dims,
// before any seed is applied) rather than through a real SCF/CC solve, since
// the function reads and writes nothing else on the state.

#include <cassert>
#include <iostream>

#include "post_hf/cc/generated_arbitrary_runtime.h"

using namespace HartreeFock::Correlation::CC;

namespace
{
    TensorND make_tensor(std::vector<int> dims, double fill)
    {
        TensorND t(dims, 0.0);
        for (std::size_t i = 0; i < t.data.size(); ++i)
            t.data[i] = fill + 0.01 * static_cast<double>(i);
        return t;
    }
} // namespace

int main()
{
    // A minimal rank-4 state: by_rank[3] is the balanced t4 (dims [2,2,2,2,2,2,2,2]),
    // sectors carries the independent (4, "aaabaaab") block at the SAME
    // shape, pre-allocated at zero -- exactly ensure_amplitude_sectors' output
    // before any seeding happens.
    ArbitraryOrderTensorCCState state;
    state.max_excitation_rank = 4;
    for (int rank = 1; rank <= 4; ++rank)
        state.amplitudes.by_rank.push_back(
            TensorND(std::vector<int>(static_cast<std::size_t>(2 * rank), 2), 0.0));
    state.amplitudes.sectors.push_back(
        {{4, "aaabaaab"}, TensorND(std::vector<int>(8, 2), 0.0)});

    // The seed: converged by_rank values plus a converged, DISTINCT sector
    // value -- the whole point is that the sector is independent data the
    // balanced by_rank does not carry.
    ArbitraryOrderRCCAmplitudes seed;
    for (int rank = 1; rank <= 4; ++rank)
        seed.by_rank.push_back(
            make_tensor(std::vector<int>(static_cast<std::size_t>(2 * rank), 2),
                        static_cast<double>(rank)));
    seed.sectors.push_back({{4, "aaabaaab"}, make_tensor(std::vector<int>(8, 2), 99.0)});

    auto applied = seed_arbitrary_order_amplitudes(state, seed);
    assert(applied && "seeding a matching sector must succeed");

    // by_rank seeded as before (unchanged behavior).
    for (int rank = 1; rank <= 4; ++rank)
        assert(state.amplitudes.by_rank[static_cast<std::size_t>(rank - 1)].data ==
               seed.by_rank[static_cast<std::size_t>(rank - 1)].data);

    // The sector was applied, not left at zero.
    assert(state.amplitudes.sectors.size() == 1);
    assert(state.amplitudes.sectors[0].second.data == seed.sectors[0].second.data);
    // Sanity: it is genuinely the seed's data, not by_rank[3]'s (they were
    // built with different fill values above) -- catches an aliasing bug
    // where the sector application accidentally wrote into by_rank instead.
    assert(state.amplitudes.sectors[0].second.data != state.amplitudes.by_rank[3].data);

    // A seed sector with no live counterpart is skipped, not an error --
    // same degradation policy as an oversized by_rank seed.
    {
        ArbitraryOrderTensorCCState state2;
        state2.max_excitation_rank = 4;
        for (int rank = 1; rank <= 4; ++rank)
            state2.amplitudes.by_rank.push_back(
                TensorND(std::vector<int>(static_cast<std::size_t>(2 * rank), 2), 0.0));
        // No sectors on the live state -- e.g. a run whose kernel bundle
        // declares no higher sectors at this rank.

        ArbitraryOrderRCCAmplitudes seed2;
        for (int rank = 1; rank <= 4; ++rank)
            seed2.by_rank.push_back(
                TensorND(std::vector<int>(static_cast<std::size_t>(2 * rank), 2), 0.0));
        seed2.sectors.push_back({{4, "aaabaaab"}, make_tensor(std::vector<int>(8, 2), 5.0)});

        auto applied2 = seed_arbitrary_order_amplitudes(state2, seed2);
        assert(applied2 && "a seed sector with no live counterpart must not fail the restart");
        assert(state2.amplitudes.sectors.empty());
    }

    // A sector whose shape disagrees between seed and live state is skipped
    // (not applied, not fatal) -- the by_rank dim check stays the only fatal
    // mismatch, since by_rank determines whether this is the right rank/basis
    // at all; a lone sector shape mismatch should not abort an otherwise-good
    // restart.
    {
        ArbitraryOrderTensorCCState state3;
        state3.max_excitation_rank = 4;
        for (int rank = 1; rank <= 4; ++rank)
            state3.amplitudes.by_rank.push_back(
                TensorND(std::vector<int>(static_cast<std::size_t>(2 * rank), 2), 0.0));
        state3.amplitudes.sectors.push_back(
            {{4, "aaabaaab"}, TensorND(std::vector<int>(8, 2), 0.0)}); // dims all 2

        ArbitraryOrderRCCAmplitudes seed3;
        for (int rank = 1; rank <= 4; ++rank)
            seed3.by_rank.push_back(
                TensorND(std::vector<int>(static_cast<std::size_t>(2 * rank), 2), 0.0));
        seed3.sectors.push_back(
            {{4, "aaabaaab"}, TensorND(std::vector<int>(8, 3), 1.0)}); // dims all 3 -- mismatched

        auto applied3 = seed_arbitrary_order_amplitudes(state3, seed3);
        assert(applied3 && "a sector shape mismatch must not fail the whole restart");
        // Left at its original (zero) value, not overwritten with mismatched data.
        for (double v : state3.amplitudes.sectors[0].second.data)
            assert(v == 0.0);
    }

    std::cout << "cc_amplitude_sector_seed: all checks passed\n";
    return 0;
}
