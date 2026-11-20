#include "post_hf/cc/ccsd_t.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <vector>

#include "base/types.h"
#include "io/logging.h"
#include "post_hf/cc/generated_arbitrary_runtime.h"

#if __has_include("generated/cc/ccsd_t_planck_generated.cpp")
#include "generated/cc/ccsd_t_planck_generated.cpp"
#define PLANCK_HAVE_CCSD_T_KERNELS 1
#else
#define PLANCK_HAVE_CCSD_T_KERNELS 0
#endif

namespace HartreeFock::Correlation::CC
{
#if PLANCK_HAVE_CCSD_T_KERNELS
    namespace
    {
        // The full-tensor oracle: W and V as o^3 v^3 tensors, then one serial sum.
        double full_tensor_triples_energy(const ArbitraryOrderTensorCCState &state)
        {
            const int no = state.reference.orbital_partition.n_occ;
            const int nv = state.reference.orbital_partition.n_virt;
            const Tensor6D w = compute_ccsd_t_W_residual(
                state.reference, state.mo_blocks, state.denominators, state.amplitudes);
            const Tensor6D v = compute_ccsd_t_V_residual(
                state.reference, state.mo_blocks, state.denominators, state.amplitudes);
            const Eigen::VectorXd &eo = state.reference.orbital_partition.eps_occ;
            const Eigen::VectorXd &ev = state.reference.orbital_partition.eps_virt;
            // Half the virtual antisymmetrizer: the closed-shell aaa block from aab.
            const auto aaa = [](const Tensor6D &x, int i, int j, int k, int a, int b, int c)
            {
                return 0.5 * (x(i, j, k, a, b, c) + x(i, j, k, b, c, a) + x(i, j, k, c, a, b) -
                              x(i, j, k, b, a, c) - x(i, j, k, a, c, b) - x(i, j, k, c, b, a));
            };
            double s_aab = 0.0;
            double s_aaa = 0.0;
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j)
                    for (int k = 0; k < no; ++k)
                        for (int a = 0; a < nv; ++a)
                            for (int b = 0; b < nv; ++b)
                                for (int c = 0; c < nv; ++c)
                                {
                                    const double d = eo(i) + eo(j) + eo(k) - ev(a) - ev(b) - ev(c);
                                    const double wx = w(i, j, k, a, b, c);
                                    s_aab += (wx + v(i, j, k, a, b, c)) * wx / d;
                                    const double wa = aaa(w, i, j, k, a, b, c);
                                    s_aaa += (wa + aaa(v, i, j, k, a, b, c)) * wa / d;
                                }
            return s_aaa / 18.0 + s_aab / 2.0;
        }

        // The six orderings of (a, b, c) as positions into it, with the
        // permutation sign: abc, bca, cab even; bac, acb, cba odd.
        constexpr std::array<std::array<int, 3>, 6> kPerms{{
            {0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {1, 0, 2}, {0, 2, 1}, {2, 1, 0}}};
        constexpr std::array<double, 6> kSigns{1.0, 1.0, 1.0, -1.0, -1.0, -1.0};

        // Batched: W and V for one unordered virtual triple a >= b >= c at a
        // time, o^3 each, never a full o^3 v^3 tensor. Each triple's energy goes
        // to its own slot and the slots are summed in triple order, so the result
        // does not depend on the thread count or schedule.
        double batched_triples_energy(const ArbitraryOrderTensorCCState &state)
        {
            const int no = state.reference.orbital_partition.n_occ;
            const int nv = state.reference.orbital_partition.n_virt;
            const Eigen::VectorXd &eo = state.reference.orbital_partition.eps_occ;
            const Eigen::VectorXd &ev = state.reference.orbital_partition.eps_virt;
            const std::size_t o3 = static_cast<std::size_t>(no) * no * no;

            std::vector<std::array<int, 3>> triples;
            for (int a = 0; a < nv; ++a)
                for (int b = 0; b <= a; ++b)
                    for (int c = 0; c <= b; ++c)
                        triples.push_back({a, b, c});
            std::vector<double> e_triple(triples.size(), 0.0);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
            {
                // Per-thread scratch: W and V for the six orderings, then aaa.
                std::vector<double> w(6 * o3), v(6 * o3), wa(o3), va(o3);
#ifdef USE_OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(triples.size()); ++t)
                {
                    const std::array<int, 3> abc = triples[static_cast<std::size_t>(t)];
                    std::array<std::array<int, 3>, 6> ord{};
                    for (int p = 0; p < 6; ++p)
                    {
                        ord[p] = {abc[kPerms[p][0]], abc[kPerms[p][1]], abc[kPerms[p][2]]};
                        compute_ccsd_t_W_block(ord[p][0], ord[p][1], ord[p][2], state.reference,
                                               state.mo_blocks, state.amplitudes, &w[p * o3]);
                        compute_ccsd_t_V_block(ord[p][0], ord[p][1], ord[p][2], state.reference,
                                               state.mo_blocks, state.amplitudes, &v[p * o3]);
                    }

                    // S_aab: every distinct ordering of the triple once.
                    double s_aab = 0.0;
                    for (int p = 0; p < 6; ++p)
                    {
                        bool repeat = false;
                        for (int q = 0; q < p; ++q)
                            repeat = repeat || ord[q] == ord[p];
                        if (repeat)
                            continue;
                        const double dv = ev(ord[p][0]) + ev(ord[p][1]) + ev(ord[p][2]);
                        for (int i = 0; i < no; ++i)
                            for (int j = 0; j < no; ++j)
                                for (int k = 0; k < no; ++k)
                                {
                                    const std::size_t x = p * o3 + (static_cast<std::size_t>(i) * no + j) * no + k;
                                    s_aab += (w[x] + v[x]) * w[x] / (eo(i) + eo(j) + eo(k) - dv);
                                }
                    }

                    // S_aaa: antisymmetric in (a, b, c), so zero unless all three
                    // differ, and then equal for all six orderings.
                    double s_aaa = 0.0;
                    if (abc[0] > abc[1] && abc[1] > abc[2])
                    {
                        std::fill(wa.begin(), wa.end(), 0.0);
                        std::fill(va.begin(), va.end(), 0.0);
                        for (int p = 0; p < 6; ++p)
                            for (std::size_t x = 0; x < o3; ++x)
                            {
                                wa[x] += 0.5 * kSigns[p] * w[p * o3 + x];
                                va[x] += 0.5 * kSigns[p] * v[p * o3 + x];
                            }
                        const double dv = ev(abc[0]) + ev(abc[1]) + ev(abc[2]);
                        for (int i = 0; i < no; ++i)
                            for (int j = 0; j < no; ++j)
                                for (int k = 0; k < no; ++k)
                                {
                                    const std::size_t x = (static_cast<std::size_t>(i) * no + j) * no + k;
                                    s_aaa += 6.0 * (wa[x] + va[x]) * wa[x] / (eo(i) + eo(j) + eo(k) - dv);
                                }
                    }
                    e_triple[static_cast<std::size_t>(t)] = s_aaa / 18.0 + s_aab / 2.0;
                }
            }

            double e_t = 0.0;
            for (const double e : e_triple)
                e_t += e;
            return e_t;
        }
    } // namespace
#endif

    std::expected<double, std::string> compute_rccsd_t_energy(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const ArbitraryOrderRCCAmplitudes &t1_t2)
    {
#if !PLANCK_HAVE_CCSD_T_KERNELS
        (void)calculator;
        (void)shell_pairs;
        (void)t1_t2;
        return std::unexpected(
            "CCSD(T): the generated (T) kernels were not built (ccsd_t_planck_generated.cpp "
            "is produced by ccgen, which needs Python 3 at configure time).");
#else
        auto state = prepare_generated_arbitrary_order_state(calculator, shell_pairs, 2, "CCSD(T) :");
        if (!state)
            return std::unexpected("CCSD(T): " + state.error());
        const int no = state->reference.orbital_partition.n_occ;
        const int nv = state->reference.orbital_partition.n_virt;
        if (t1_t2.by_rank.size() != 2 ||
            t1_t2.by_rank[0].dims != std::vector<int>{no, nv} ||
            t1_t2.by_rank[1].dims != std::vector<int>{no, no, nv, nv})
            return std::unexpected(std::format(
                "CCSD(T): amplitudes must be spatial t1 ({0},{1}) and t2 ({0},{0},{1},{1}).", no, nv));
        state->amplitudes = t1_t2;

        const double e_t = batched_triples_energy(*state);

        // Keep the full-tensor oracle exercised: at verbose output on a small
        // system, recompute E(T) with it and print the difference. Output only;
        // the returned energy is always the batched one.
        // ponytail: 2e6 elements = two 16 MB tensors; raise if the check is
        // wanted on larger fixtures.
        const double o3v3 = std::pow(static_cast<double>(no) * nv, 3);
        if (calculator._output._verbosity >= HartreeFock::Verbosity::Verbose && o3v3 <= 2.0e6)
        {
            const double e_full = full_tensor_triples_energy(*state);
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, "RCCSD(T) check :",
                std::format("|batched - full-tensor| E(T) = {:.3e}", std::abs(e_t - e_full)));
        }
        return e_t;
#endif
    }
} // namespace HartreeFock::Correlation::CC
