#include "post_hf/ci/fciqmc.h"

#include <bit>
#include <cmath>
#include <vector>

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

    namespace
    {
        // Apply a->r within one spin string, returning the new string and phase.
        // Uses the shared fermionic operators so the sign convention is the one
        // the rest of the CI layer already uses -- a second convention here would
        // be a sign bug waiting for the first open-shell case.
        struct SpinExcitation
        {
            CIString det = 0;
            double phase = 0.0;
            bool valid = false;
        };

        // The fermionic sign: -1 when an odd number of occupied orbitals lie
        // strictly below `orb`. This is the SAME convention as
        // CI::apply_annihilation / apply_creation in strings.cpp, written here in
        // terms of the header-inline bit helpers so this layer does not drag the
        // symmetry/basis dependency chain into every consumer. The equivalence is
        // asserted by the F2 gate (test_operator_convention_matches_shared), so
        // the two cannot silently drift apart.
        inline double parity_phase(CIString det, int orb)
        {
            const int below = std::popcount(det & CASSCFInternal::low_bit_mask(orb));
            return (below % 2 == 0) ? 1.0 : -1.0;
        }

        SpinExcitation excite_one(CIString det, int from, int to)
        {
            const CIString from_bit = CASSCFInternal::single_bit_mask(from);
            if (!(det & from_bit))
                return {}; // not occupied: nothing to annihilate
            const double ph_ann = parity_phase(det, from);
            const CIString after_ann = det ^ from_bit;

            const CIString to_bit = CASSCFInternal::single_bit_mask(to);
            if (after_ann & to_bit)
                return {}; // already occupied: cannot create
            const double ph_cre = parity_phase(after_ann, to);

            return {after_ann | to_bit, ph_ann * ph_cre, true};
        }

        std::vector<int> occupied(CIString det, int n_act)
        {
            std::vector<int> occ;
            for (int p = 0; p < n_act; ++p)
                if (det & CASSCFInternal::single_bit_mask(p))
                    occ.push_back(p);
            return occ;
        }

        std::vector<int> virtuals(CIString det, int n_act)
        {
            std::vector<int> vir;
            for (int p = 0; p < n_act; ++p)
                if (!(det & CASSCFInternal::single_bit_mask(p)))
                    vir.push_back(p);
            return vir;
        }
    } // namespace

    std::vector<Excitation> enumerate_connections(const DetKey &parent, int n_act)
    {
        std::vector<Excitation> out;

        const std::vector<int> occ_a = occupied(parent.alpha, n_act);
        const std::vector<int> occ_b = occupied(parent.beta, n_act);
        const std::vector<int> vir_a = virtuals(parent.alpha, n_act);
        const std::vector<int> vir_b = virtuals(parent.beta, n_act);

        // --- singles, alpha
        for (int i : occ_a)
            for (int a : vir_a)
            {
                const auto e = excite_one(parent.alpha, i, a);
                if (e.valid)
                    out.push_back({DetKey{e.det, parent.beta}, e.phase, 0.0, true});
            }

        // --- singles, beta
        for (int i : occ_b)
            for (int a : vir_b)
            {
                const auto e = excite_one(parent.beta, i, a);
                if (e.valid)
                    out.push_back({DetKey{parent.alpha, e.det}, e.phase, 0.0, true});
            }

        // --- same-spin doubles, alpha. i<j and a<b so each unordered pair of
        // orbitals is visited exactly once; visiting both orderings would emit
        // duplicate determinants (with opposite phases, which is worse than
        // merely redundant).
        for (std::size_t ii = 0; ii + 1 < occ_a.size(); ++ii)
            for (std::size_t jj = ii + 1; jj < occ_a.size(); ++jj)
                for (std::size_t aa = 0; aa + 1 < vir_a.size(); ++aa)
                    for (std::size_t bb = aa + 1; bb < vir_a.size(); ++bb)
                    {
                        const auto e1 = excite_one(parent.alpha, occ_a[ii], vir_a[aa]);
                        if (!e1.valid)
                            continue;
                        const auto e2 = excite_one(e1.det, occ_a[jj], vir_a[bb]);
                        if (!e2.valid)
                            continue;
                        out.push_back({DetKey{e2.det, parent.beta}, e1.phase * e2.phase, 0.0, true});
                    }

        // --- same-spin doubles, beta
        for (std::size_t ii = 0; ii + 1 < occ_b.size(); ++ii)
            for (std::size_t jj = ii + 1; jj < occ_b.size(); ++jj)
                for (std::size_t aa = 0; aa + 1 < vir_b.size(); ++aa)
                    for (std::size_t bb = aa + 1; bb < vir_b.size(); ++bb)
                    {
                        const auto e1 = excite_one(parent.beta, occ_b[ii], vir_b[aa]);
                        if (!e1.valid)
                            continue;
                        const auto e2 = excite_one(e1.det, occ_b[jj], vir_b[bb]);
                        if (!e2.valid)
                            continue;
                        out.push_back({DetKey{parent.alpha, e2.det}, e1.phase * e2.phase, 0.0, true});
                    }

        // --- opposite-spin doubles: one alpha excitation and one beta. No
        // ordering restriction is needed because the two spin channels are
        // independent -- every (i->a, j->b) pair gives a distinct determinant.
        for (int i : occ_a)
            for (int a : vir_a)
            {
                const auto ea = excite_one(parent.alpha, i, a);
                if (!ea.valid)
                    continue;
                for (int j : occ_b)
                    for (int b : vir_b)
                    {
                        const auto eb = excite_one(parent.beta, j, b);
                        if (!eb.valid)
                            continue;
                        out.push_back({DetKey{ea.det, eb.det}, ea.phase * eb.phase, 0.0, true});
                    }
            }

        return out;
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
