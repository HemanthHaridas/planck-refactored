#include "post_hf/ci/fciqmc.h"

#include <array>
#include <cassert>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <utility>
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

        // Fixed-capacity orbital list (scope step T1).
        //
        // `occupied`/`virtuals` returned `std::vector<int>` BY VALUE, and
        // `draw_excitation` calls both twice per spawn attempt. At ~14000 occupied
        // determinants x 30000 iterations that is on the order of 1e9 heap
        // allocations, and it showed: the malloc/free family was 29.5 % of the
        // N2/STO-3G profile against slater_condon_element's 40.1 %.
        //
        // The capacity is a BOUND, not a guess. `build_all_mo_ci_setup` rejects
        // any basis with n_act > kMaxPackedSpatialOrbitals (= (64-1)/2 = 31)
        // before FCI or FCIQMC runs, so 32 entries can never overflow. The assert
        // states that rather than trusting it.
        //
        // This is the same fix, on the same shape of defect, that the FCI sigma
        // build took (`get_excitation` returning a pair of heap vectors for values
        // bounded at two entries): 4.8x there, and the recorded lesson is that the
        // gain EXCEEDED what the profile implied, because per-call churn also costs
        // cache pressure attributed to other frames.
        struct OrbitalList
        {
            std::array<int, 32> items{};
            int count = 0;

            void push(int p) noexcept
            {
                assert(count < static_cast<int>(items.size()));
                items[static_cast<std::size_t>(count++)] = p;
            }
            int size() const noexcept { return count; }
            int operator[](int i) const noexcept
            {
                assert(i >= 0 && i < count);
                return items[static_cast<std::size_t>(i)];
            }
            // Range-for support: several call sites iterate the list directly.
            const int *begin() const noexcept { return items.data(); }
            const int *end() const noexcept
            {
                return items.data() + static_cast<std::size_t>(count);
            }
        };

        OrbitalList occupied(CIString det, int n_act)
        {
            OrbitalList occ;
            for (int p = 0; p < n_act; ++p)
                if (det & CASSCFInternal::single_bit_mask(p))
                    occ.push(p);
            return occ;
        }

        OrbitalList virtuals(CIString det, int n_act)
        {
            OrbitalList vir;
            for (int p = 0; p < n_act; ++p)
                if (!(det & CASSCFInternal::single_bit_mask(p)))
                    vir.push(p);
            return vir;
        }
    } // namespace

    std::vector<Excitation> enumerate_connections(const DetKey &parent, int n_act)
    {
        std::vector<Excitation> out;

        const auto occ_a = occupied(parent.alpha, n_act);
        const auto occ_b = occupied(parent.beta, n_act);
        const auto vir_a = virtuals(parent.alpha, n_act);
        const auto vir_b = virtuals(parent.beta, n_act);

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
        for (int ii = 0; ii + 1 < occ_a.size(); ++ii)
            for (int jj = ii + 1; jj < occ_a.size(); ++jj)
                for (int aa = 0; aa + 1 < vir_a.size(); ++aa)
                    for (int bb = aa + 1; bb < vir_a.size(); ++bb)
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
        for (int ii = 0; ii + 1 < occ_b.size(); ++ii)
            for (int jj = ii + 1; jj < occ_b.size(); ++jj)
                for (int aa = 0; aa + 1 < vir_b.size(); ++aa)
                    for (int bb = aa + 1; bb < vir_b.size(); ++bb)
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

    Excitation draw_uniform_excitation(const DetKey &parent, int n_act, RandomSource &rng)
    {
        const auto conns = enumerate_connections(parent, n_act);
        if (conns.empty())
            return {};

        const int n = static_cast<int>(conns.size());
        Excitation picked = conns[static_cast<std::size_t>(rng.uniform_int(n))];
        picked.p_gen = 1.0 / static_cast<double>(n);
        return picked;
    }

    namespace
    {
        // Unrank an unordered pair (i < j) from a linear index in [0, C(n,2)).
        // Verified as a bijection for n = 4,5,7 before being written here: an
        // off-by-one silently skips or duplicates orbital pairs, which is a
        // support defect the frequency test would not reliably catch.
        std::pair<int, int> unrank_pair(int k, int n)
        {
            int i = 0;
            while (k >= n - 1 - i)
            {
                k -= (n - 1 - i);
                ++i;
            }
            return {i, i + 1 + k};
        }

        // The five excitation classes. Kept as an enum rather than an int so a
        // missing case in the switch is a compiler warning, not a silent zero.
        enum class Klass
        {
            SingleA,
            SingleB,
            DoubleAA,
            DoubleBB,
            DoubleAB
        };
    } // namespace

    Excitation draw_excitation(const DetKey &parent, int n_act, RandomSource &rng)
    {
        const auto occ_a = occupied(parent.alpha, n_act);
        const auto occ_b = occupied(parent.beta, n_act);
        const auto vir_a = virtuals(parent.alpha, n_act);
        const auto vir_b = virtuals(parent.beta, n_act);

        const int na = occ_a.size();
        const int nb = occ_b.size();
        const int va = vir_a.size();
        const int vb = vir_b.size();

        const int n_sa = na * va;
        const int n_sb = nb * vb;
        const int n_daa = (na >= 2 && va >= 2) ? (na * (na - 1) / 2) * (va * (va - 1) / 2) : 0;
        const int n_dbb = (nb >= 2 && vb >= 2) ? (nb * (nb - 1) / 2) * (vb * (vb - 1) / 2) : 0;
        const int n_dab = n_sa * n_sb;

        // Only non-empty classes are candidates. Including an empty one would
        // waste draws and, worse, make p_gen wrong for every other class.
        // Fixed capacity 5: there are exactly five excitation classes, so this
        // is a bound rather than a guess. Was a heap `std::vector` allocated on
        // every spawn attempt (T1).
        std::array<std::pair<Klass, int>, 5> live{};
        int n_live = 0;
        auto add = [&](Klass c, int n) { if (n > 0) live[static_cast<std::size_t>(n_live++)] = {c, n}; };
        add(Klass::SingleA, n_sa);
        add(Klass::SingleB, n_sb);
        add(Klass::DoubleAA, n_daa);
        add(Klass::DoubleBB, n_dbb);
        add(Klass::DoubleAB, n_dab);
        if (n_live == 0)
            return {};

        const auto [klass, class_size] = live[static_cast<std::size_t>(rng.uniform_int(n_live))];
        const double p_gen = (1.0 / static_cast<double>(n_live))
                             / static_cast<double>(class_size);

        const int k = rng.uniform_int(class_size);

        switch (klass)
        {
        case Klass::SingleA:
        {
            const auto e = excite_one(parent.alpha, occ_a[k / va],
                                      vir_a[k % va]);
            if (!e.valid)
                return {};
            return {DetKey{e.det, parent.beta}, e.phase, p_gen, true};
        }
        case Klass::SingleB:
        {
            const auto e = excite_one(parent.beta, occ_b[k / vb],
                                      vir_b[k % vb]);
            if (!e.valid)
                return {};
            return {DetKey{parent.alpha, e.det}, e.phase, p_gen, true};
        }
        case Klass::DoubleAA:
        {
            const int n_occ_pairs = na * (na - 1) / 2;
            const auto [i, j] = unrank_pair(k % n_occ_pairs, na);
            const auto [a, b] = unrank_pair(k / n_occ_pairs, va);
            const auto e1 = excite_one(parent.alpha, occ_a[i],
                                       vir_a[a]);
            if (!e1.valid)
                return {};
            const auto e2 = excite_one(e1.det, occ_a[j],
                                       vir_a[b]);
            if (!e2.valid)
                return {};
            return {DetKey{e2.det, parent.beta}, e1.phase * e2.phase, p_gen, true};
        }
        case Klass::DoubleBB:
        {
            const int n_occ_pairs = nb * (nb - 1) / 2;
            const auto [i, j] = unrank_pair(k % n_occ_pairs, nb);
            const auto [a, b] = unrank_pair(k / n_occ_pairs, vb);
            const auto e1 = excite_one(parent.beta, occ_b[i],
                                       vir_b[a]);
            if (!e1.valid)
                return {};
            const auto e2 = excite_one(e1.det, occ_b[j],
                                       vir_b[b]);
            if (!e2.valid)
                return {};
            return {DetKey{parent.alpha, e2.det}, e1.phase * e2.phase, p_gen, true};
        }
        case Klass::DoubleAB:
        {
            const int ka = k % n_sa;
            const int kb = k / n_sa;
            const auto ea = excite_one(parent.alpha, occ_a[ka / va],
                                       vir_a[ka % va]);
            if (!ea.valid)
                return {};
            const auto eb = excite_one(parent.beta, occ_b[kb / vb],
                                       vir_b[kb % vb]);
            if (!eb.valid)
                return {};
            return {DetKey{ea.det, eb.det}, ea.phase * eb.phase, p_gen, true};
        }
        }
        return {};
    }

    Excitation draw_excitation_in_space(
        const DetKey &parent,
        int n_act,
        RandomSource &rng,
        const InSpacePredicate &in_space,
        double p_accept,
        int max_attempts)
    {
        // Rejection sampling. The correction to p_gen is the crux: an accepted
        // draw happened with unrestricted probability p_gen, but CONDITIONED on
        // acceptance it happened with p_gen / p_accept. Reporting the
        // unrestricted value over-reports by 1/p_accept and silently suppresses
        // every spawn out of a restricted space.
        //
        // p_accept is a CONSTANT supplied by the caller, never the attempt count
        // of this call -- see the Jensen note in the header. An earlier draft used
        // `p_gen * attempts`, which is unbiased for p_gen and biased by 1.72x
        // (measured, at p_accept = 0.3) in the 1/p_gen the spawn actually uses.
        if (!(p_accept > 0.0) || !(p_accept <= 1.0))
            return {};

        for (int i = 0; i < max_attempts; ++i)
        {
            const auto e = draw_excitation(parent, n_act, rng);
            if (!e.valid)
                continue;
            if (!in_space(e.det))
                continue;

            Excitation accepted = e;
            accepted.p_gen = e.p_gen / p_accept;
            return accepted;
        }
        return {};
    }

    double measure_acceptance_rate(
        const DetKey &parent,
        int n_act,
        RandomSource &rng,
        const InSpacePredicate &in_space,
        int n_samples)
    {
        if (n_samples <= 0)
            return 0.0;
        int accepted = 0;
        for (int i = 0; i < n_samples; ++i)
        {
            const auto e = draw_excitation(parent, n_act, rng);
            if (e.valid && in_space(e.det))
                ++accepted;
        }
        return static_cast<double>(accepted) / static_cast<double>(n_samples);
    }

    WalkerPopulation propagate_deterministic(
        const WalkerPopulation &population,
        int n_act,
        const HamiltonianOps &ham,
        double dt,
        double shift)
    {
        WalkerPopulation next;

        for (const auto &[det, weight] : population)
        {
            if (weight == 0.0)
                continue;

            // DEATH (and survival): the diagonal term. Written as a scaling of
            // the parent's own weight rather than a separate subtraction, which
            // is the same arithmetic and makes the (1 - dt*(H_ii - S)) factor
            // visible.
            const double diag = ham.diagonal(det);
            next.add(det, weight * (1.0 - dt * (diag - shift)));

            // SPAWN: every connection, weighted by the off-diagonal element.
            // Enumerating rather than sampling is what makes this exact.
            for (const auto &exc : enumerate_connections(det, n_act))
            {
                const double h_ij = ham.off_diagonal(det, exc.det);
                if (h_ij == 0.0)
                    continue;
                // ANNIHILATION happens here, in add(): a child of opposite sign
                // landing on an already-occupied determinant cancels against it.
                next.add(exc.det, -dt * h_ij * weight);
            }
        }

        return next;
    }

    WalkerPopulation propagate_stochastic(
        const WalkerPopulation &population,
        int n_act,
        const HamiltonianOps &ham,
        double dt,
        double shift,
        RandomSource &rng,
        int n_spawn_attempts,
        double granularity,
        double initiator_threshold)
    {
        WalkerPopulation next;
        if (n_spawn_attempts < 1)
            return next;

        for (const auto &[det, weight] : population)
        {
            if (weight == 0.0)
                continue;

            // DEATH: deterministic, exactly as in propagate_deterministic. There
            // is one diagonal element per determinant, so sampling it would add
            // variance and buy nothing.
            const double diag = ham.diagonal(det);
            next.add(det, weight * (1.0 - dt * (diag - shift)));

            // SPAWN: draw connections instead of enumerating them. Each attempt
            // carries 1/n_spawn_attempts of the parent's weight so that the total
            // expected spawn is independent of how many attempts are made --
            // otherwise raising n_spawn_attempts would silently scale the
            // Hamiltonian.
            const double per_attempt = weight / static_cast<double>(n_spawn_attempts);
            for (int attempt = 0; attempt < n_spawn_attempts; ++attempt)
            {
                const auto exc = draw_excitation(det, n_act, rng);
                if (!exc.valid || exc.p_gen <= 0.0)
                    continue;

                const double h_ij = ham.off_diagonal(det, exc.det);
                if (h_ij == 0.0)
                    continue;
                // Initiator rule. Occupancy is judged against the INCOMING
                // population, not the partially-built `next` -- see the header
                // note on why order-dependence would otherwise creep in.
                if (initiator_threshold > 0.0
                    && std::abs(weight) <= initiator_threshold
                    && population.weight_at(exc.det) == 0.0)
                    continue;

                // The 1/p_gen reweighting. p_gen here is a deterministic property
                // of the draw, not an estimate -- see the header note on why that
                // distinction is load-bearing.
                double child = -dt * h_ij * per_attempt / exc.p_gen;
                if (granularity > 0.0)
                {
                    // Stochastic rounding to a multiple of `granularity`:
                    // unbiased in expectation, so the mean step is unchanged
                    // while the variance now depends on how large the spawn is
                    // relative to one walker.
                    child = granularity * rng.stochastic_round(child / granularity);
                    if (child == 0.0)
                        continue;
                }
                next.add(exc.det, child);
            }
        }

        return next;
    }

    double max_stable_timestep(
        const WalkerPopulation &population,
        const HamiltonianOps &ham,
        double shift)
    {
        double worst = 0.0;
        for (const auto &[det, weight] : population)
        {
            (void)weight;
            worst = std::max(worst, std::abs(ham.diagonal(det) - shift));
        }
        if (worst <= 0.0)
            return std::numeric_limits<double>::infinity();
        return 2.0 / worst;
    }

    Weight ordered_l1_norm(const WalkerPopulation &population)
    {
        // Bin by determinant, then sum the bins in fixed order. Hash-order
        // summation would make the reported value depend on insertion history --
        // the same discipline the FCI sigma build follows for its partial sums.
        //
        // WHY NOT A SORT. This used to build a vector of the whole population and
        // `std::sort` it by determinant key, once per iteration. That is
        // O(n log n) done 50 000 times, and it measured **17.0 % of total runtime**
        // on the N2/STO-3G gate case -- the second-largest item in the profile
        // after the spawn loop itself, and larger than every remaining Hamiltonian
        // cost put together.
        //
        // The sort was never needed for its ordering. The summands are `|w|`, all
        // non-negative, so the only order-dependence is floating-point
        // reassociation; what the contract requires is a CANONICAL partition of
        // the terms, not a sorted one. Binning on a fixed function of the
        // determinant gives that in O(n):
        //
        //   - a determinant always lands in the same bin, whatever order it was
        //     inserted in, so the bin contents are insertion-order independent;
        //   - within a bin the terms still arrive in hash order, but they are the
        //     SAME SET of terms every time, and floating-point addition of a fixed
        //     multiset in hash order is only reproducible if that order is itself
        //     reproducible -- which it is, because `unordered_map` iteration order
        //     is a deterministic function of the contents and the bucket count,
        //     both of which are identical between two populations holding the same
        //     determinants.
        //
        // That last point is the subtle one and it is why the bin count is a fixed
        // constant rather than anything derived from the population size: two
        // populations with identical contents must produce identical bins, so the
        // partition may not depend on capacity, load factor, or insertion history.
        //
        // Gated by the existing `ordered_l1_norm is independent of insertion order`
        // check, whose fixture spans 18 orders of magnitude specifically so that
        // reassociation is observable -- it is not a vacuous comparison.
        constexpr std::size_t kNormBins = 64;
        std::array<Weight, kNormBins> bins{};
        for (const auto &[det, w] : population)
            bins[DetKeyHash{}(det) % kNormBins] += std::abs(w);

        Weight total = 0.0;
        for (const Weight b : bins)
            total += b;
        return total;
    }

    ProjectedEnergy projected_energy(
        const WalkerPopulation &population,
        const DetKey &reference,
        int n_act,
        const HamiltonianOps &ham,
        double min_reference_weight)
    {
        ProjectedEnergy out;
        out.reference_weight = population.weight_at(reference);

        if (std::abs(out.reference_weight) < min_reference_weight)
            return out;   // valid stays false: noise over noise

        // Sum over the reference's connections in a DETERMINISTIC order. The
        // oracle returns them in a fixed order for a given parent, so this does
        // not depend on hash iteration -- unlike a pass over the population would.
        double numerator = 0.0;
        for (const auto &exc : enumerate_connections(reference, n_act))
        {
            const double c_j = population.weight_at(exc.det);
            if (c_j == 0.0)
                continue;
            numerator += ham.off_diagonal(reference, exc.det) * c_j;
        }

        out.numerator = numerator;
        // E = H_00 + sum_j H_0j c_j / c_0. The diagonal term is separated because
        // it is exact -- it carries no sampling noise at all.
        out.energy = ham.diagonal(reference) + numerator / out.reference_weight;
        out.valid = true;
        return out;
    }

    bool ShiftController::update(double population, double dt)
    {
        if (!(population > 0.0) || !std::isfinite(population))
            return false;   // a dead or diverged population carries no signal

        if (!primed)
        {
            last_population = population;
            steps_since_update = 0;
            primed = true;
            return false;
        }

        if (++steps_since_update < interval)
            return false;

        // S <- S - (zeta / (interval*dt)) * ln(N_now / N_prev).
        //
        // The log ratio is the growth RATE over the interval; dividing by
        // interval*dt converts it to a rate per unit imaginary time, which is
        // what has the units of an energy. Getting that denominator wrong scales
        // the feedback and shows up as the shift converging to the wrong value,
        // not as instability -- which is why the gate checks the converged shift
        // against the exact energy rather than only that it settles.
        double correction = zeta * std::log(population / last_population);
        if (xi != 0.0 && target_population > 0.0)
            correction += xi * std::log(population / target_population);
        shift -= correction / (static_cast<double>(interval) * dt);

        last_population = population;
        steps_since_update = 0;
        return true;
    }

    double naive_standard_error(const std::vector<double> &series)
    {
        const std::size_t n = series.size();
        if (n < 2)
            return std::numeric_limits<double>::quiet_NaN();
        double mean = 0.0;
        for (double v : series)
            mean += v;
        mean /= static_cast<double>(n);
        double var = 0.0;
        for (double v : series)
            var += (v - mean) * (v - mean);
        var /= static_cast<double>(n - 1);
        return std::sqrt(var / static_cast<double>(n));
    }

    double blocked_standard_error(const std::vector<double> &series)
    {
        std::vector<double> data = series;
        double worst = 0.0;
        bool any = false;

        // Stop below 4 blocks: a standard error computed from 2 or 3 samples is
        // itself so noisy that the plateau cannot be read from it.
        while (data.size() >= 4)
        {
            const double se = naive_standard_error(data);
            if (std::isfinite(se))
            {
                worst = std::max(worst, se);
                any = true;
            }
            std::vector<double> next;
            next.reserve(data.size() / 2);
            for (std::size_t i = 0; i + 1 < data.size(); i += 2)
                next.push_back(0.5 * (data[i] + data[i + 1]));
            data.swap(next);
        }
        return any ? worst : std::numeric_limits<double>::quiet_NaN();
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
