#include "post_hf/cc/ccsd_t.h"

#include <format>

#include "post_hf/cc/generated_arbitrary_runtime.h"

#if __has_include("generated/cc/ccsd_t_planck_generated.cpp")
#include "generated/cc/ccsd_t_planck_generated.cpp"
#define PLANCK_HAVE_CCSD_T_KERNELS 1
#else
#define PLANCK_HAVE_CCSD_T_KERNELS 0
#endif

namespace HartreeFock::Correlation::CC
{
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

        const Tensor6D w = compute_ccsd_t_W_residual(
            state->reference, state->mo_blocks, state->denominators, state->amplitudes);
        const Tensor6D v = compute_ccsd_t_V_residual(
            state->reference, state->mo_blocks, state->denominators, state->amplitudes);

        const Eigen::VectorXd &eo = state->reference.orbital_partition.eps_occ;
        const Eigen::VectorXd &ev = state->reference.orbital_partition.eps_virt;
        // Half the virtual antisymmetrizer: the closed-shell aaa block from aab.
        const auto aaa = [](const Tensor6D &x, int i, int j, int k, int a, int b, int c)
        {
            return 0.5 * (x(i, j, k, a, b, c) + x(i, j, k, b, c, a) + x(i, j, k, c, a, b) -
                          x(i, j, k, b, a, c) - x(i, j, k, a, c, b) - x(i, j, k, c, b, a));
        };

        // ponytail: serial full-tensor contraction (the oracle); the batched
        // kernel is where threading goes.
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
#endif
    }
} // namespace HartreeFock::Correlation::CC
