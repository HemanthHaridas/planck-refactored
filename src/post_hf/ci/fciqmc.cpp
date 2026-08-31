#include "post_hf/ci/fciqmc.h"

#include <cmath>

namespace HartreeFock::Correlation::CI::QMC
{

    std::size_t WalkerPopulation::compress(Weight threshold)
    {
        std::size_t removed = 0;
        for (auto it = _walkers.begin(); it != _walkers.end();)
        {
            // <= so that threshold 0 still removes exact zeros, which is the
            // common case: annihilation of equal and opposite weights.
            if (std::abs(it->second) <= threshold)
            {
                it = _walkers.erase(it);
                ++removed;
            }
            else
            {
                ++it;
            }
        }
        return removed;
    }

    Weight WalkerPopulation::total_population() const noexcept
    {
        // Summed in hash order, which is not reproducible across rehashes. This is
        // a diagnostic and a population-control input, never an energy estimator;
        // if it ever feeds a reported quantity it must be sorted first. See the
        // determinism discussion in docs/FCIQMC_RESEARCH_SCOPE.md.
        Weight total = 0.0;
        for (const auto &[det, w] : _walkers)
            total += std::abs(w);
        return total;
    }

} // namespace HartreeFock::Correlation::CI::QMC
