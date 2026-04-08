#include <algorithm>
#include <cstdint>
#include <format>

#include "integrals.h"
#include "integrals/base.h"
#include "io/logging.h"

namespace
{

    struct ActiveCacheScratch
    {
        std::vector<double> work_u_lam_sig;
        std::vector<double> work_u_v_sig;

        void ensure_capacity(std::size_t work_u_lam_sig_size,
                             std::size_t work_u_v_sig_size)
        {
            if (work_u_lam_sig.size() != work_u_lam_sig_size)
                work_u_lam_sig.resize(work_u_lam_sig_size);
            if (work_u_v_sig.size() != work_u_v_sig_size)
                work_u_v_sig.resize(work_u_v_sig_size);
        }
    };

    ActiveCacheScratch &active_cache_scratch(
        std::size_t work_u_lam_sig_size,
        std::size_t work_u_v_sig_size)
    {
        thread_local ActiveCacheScratch scratch;
        scratch.ensure_capacity(work_u_lam_sig_size, work_u_v_sig_size);
        return scratch;
    }

} // namespace

namespace HartreeFock::Correlation
{

    // ── ensure_eri ────────────────────────────────────────────────────────────────

    const std::vector<double> &ensure_eri(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::vector<double> &eri_local,
        const std::string &tag)
    {
        // Prefer the cached AO tensor when it is already present on the
        // calculator; only build into caller-owned storage when needed.
        if (!calc._eri.empty())
            return calc._eri;

        const std::size_t nb = calc._shells.nbasis();
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, tag,
                                     std::format("Building AO ERI tensor ({:.1f} MB)", nb * nb * nb * nb * 8.0 / 1e6));
        eri_local = _compute_2e(shell_pairs, nb, calc._integral._engine, 1e-10,
                                calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
        return eri_local;
    }

    // ── transform_eri ─────────────────────────────────────────────────────────────

    std::vector<double> transform_eri(
        const std::vector<double> &eri,
        std::size_t nb,
        const Eigen::MatrixXd &C1,
        const Eigen::MatrixXd &C2,
        const Eigen::MatrixXd &C3,
        const Eigen::MatrixXd &C4)
    {
        const std::size_t n1 = static_cast<std::size_t>(C1.cols());
        const std::size_t n2 = static_cast<std::size_t>(C2.cols());
        const std::size_t n3 = static_cast<std::size_t>(C3.cols());
        const std::size_t n4 = static_cast<std::size_t>(C4.cols());

        // Keep the intermediates as flat buffers so every quarter transform uses
        // the same row-major indexing convention as the source AO tensor.
        const std::size_t nb2 = nb * nb;
        const std::size_t nb3 = nb * nb2;

        // T1[i,ν,λ,σ] = Σ_μ C1(μ,i) * eri[μνλσ]   shape: n1 × nb × nb × nb
        std::vector<double> T1(n1 * nb * nb * nb, 0.0);
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        for (std::size_t mu = 0; mu < nb; ++mu)
                            T1[i * nb3 + nu * nb2 + lam * nb + sig] +=
                                C1(mu, i) * eri[mu * nb3 + nu * nb2 + lam * nb + sig];

        // T2[i,a,λ,σ] = Σ_ν C2(ν,a) * T1[i,ν,λ,σ]   shape: n1 × n2 × nb × nb
        std::vector<double> T2(n1 * n2 * nb * nb, 0.0);
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        for (std::size_t nu = 0; nu < nb; ++nu)
                            T2[i * n2 * nb * nb + a * nb * nb + lam * nb + sig] +=
                                C2(nu, a) * T1[i * nb3 + nu * nb2 + lam * nb + sig];

        T1.clear();
        T1.shrink_to_fit();

        // T3[i,a,j,σ] = Σ_λ C3(λ,j) * T2[i,a,λ,σ]   shape: n1 × n2 × n3 × nb
        std::vector<double> T3(n1 * n2 * n3 * nb, 0.0);
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t j = 0; j < n3; ++j)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        for (std::size_t lam = 0; lam < nb; ++lam)
                            T3[i * n2 * n3 * nb + a * n3 * nb + j * nb + sig] +=
                                C3(lam, j) * T2[i * n2 * nb * nb + a * nb * nb + lam * nb + sig];

        T2.clear();
        T2.shrink_to_fit();

        // out[i,a,j,b] = Σ_σ C4(σ,b) * T3[i,a,j,σ]   shape: n1 × n2 × n3 × n4
        std::vector<double> out(n1 * n2 * n3 * n4, 0.0);
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t j = 0; j < n3; ++j)
                    for (std::size_t b = 0; b < n4; ++b)
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            out[i * n2 * n3 * n4 + a * n3 * n4 + j * n4 + b] +=
                                C4(sig, b) * T3[i * n2 * n3 * nb + a * n3 * nb + j * nb + sig];

        return out;
    }

    // ── transform_eri_active_cache ──────────────────────────────────────────────

    std::vector<double> transform_eri_active_cache(
        const std::vector<double> &eri,
        std::size_t nb,
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &C_act)
    {
        const std::size_t n_act = static_cast<std::size_t>(C_act.cols());
        if (n_act == 0)
            return {};

        const std::size_t nb2 = nb * nb;
        const std::size_t nb3 = nb * nb2;
        const std::size_t act3 = n_act * n_act * n_act;
        const std::size_t work_u_lam_sig_size = n_act * nb * nb;
        const std::size_t work_u_v_sig_size = n_act * n_act * nb;

        std::vector<double> out(nb * act3, 0.0);

        auto build_p_block = [&](std::size_t p,
                                 std::vector<double> &work_u_lam_sig,
                                 std::vector<double> &work_u_v_sig)
        {
            std::fill(work_u_lam_sig.begin(), work_u_lam_sig.end(), 0.0);
            std::fill(work_u_v_sig.begin(), work_u_v_sig.end(), 0.0);
            double *out_p = out.data() + p * act3;

            // Contract the first two legs into a `u x λ x σ` slab for this `p`.
            for (std::size_t mu = 0; mu < nb; ++mu)
            {
                const double c_mu_p = C(mu, p);
                const std::size_t mu_offset = mu * nb3;
                for (std::size_t nu = 0; nu < nb; ++nu)
                {
                    const std::size_t nu_offset = mu_offset + nu * nb2;
                    for (std::size_t u = 0; u < n_act; ++u)
                    {
                        const double scale = c_mu_p * C_act(nu, u);
                        double *u_block = work_u_lam_sig.data() + u * nb2;
                        for (std::size_t lam = 0; lam < nb; ++lam)
                        {
                            const double *eri_row = eri.data() + nu_offset + lam * nb;
                            double *u_lam = u_block + lam * nb;
                            for (std::size_t sig = 0; sig < nb; ++sig)
                                u_lam[sig] += scale * eri_row[sig];
                        }
                    }
                }
            }

            // Contract λ -> v while keeping σ contiguous inside each `(u,v)` slab.
            for (std::size_t u = 0; u < n_act; ++u)
            {
                const double *u_block = work_u_lam_sig.data() + u * nb2;
                double *uv_block = work_u_v_sig.data() + u * n_act * nb;
                for (std::size_t lam = 0; lam < nb; ++lam)
                {
                    const double *u_lam = u_block + lam * nb;
                    for (std::size_t v = 0; v < n_act; ++v)
                    {
                        const double scale = C_act(lam, v);
                        double *uv = uv_block + v * nb;
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            uv[sig] += scale * u_lam[sig];
                    }
                }
            }

            // Final σ -> w contraction produces the row-major `(p,u,v,w)` block.
            for (std::size_t u = 0; u < n_act; ++u)
            {
                const double *uv_block = work_u_v_sig.data() + u * n_act * nb;
                double *uvw_block = out_p + u * n_act * n_act;
                for (std::size_t v = 0; v < n_act; ++v)
                {
                    const double *uv = uv_block + v * nb;
                    double *uvw = uvw_block + v * n_act;
                    for (std::size_t sig = 0; sig < nb; ++sig)
                    {
                        const double value = uv[sig];
                        for (std::size_t w = 0; w < n_act; ++w)
                            uvw[w] += value * C_act(sig, w);
                    }
                }
            }
        };

#ifdef USE_OPENMP
#pragma omp parallel
        {
            // Each thread owns whole `p` slabs of the `(p,u,v,w)` tensor, so no
            // reductions or shared scratch are needed; thread-local buffers are
            // reused across repeated active-cache builds to avoid allocator churn.
            ActiveCacheScratch &scratch =
                active_cache_scratch(work_u_lam_sig_size, work_u_v_sig_size);
#pragma omp for schedule(static)
            for (std::int64_t p_i = 0; p_i < static_cast<std::int64_t>(nb); ++p_i)
                build_p_block(static_cast<std::size_t>(p_i),
                              scratch.work_u_lam_sig,
                              scratch.work_u_v_sig);
        }
#else
        ActiveCacheScratch &scratch =
            active_cache_scratch(work_u_lam_sig_size, work_u_v_sig_size);
        for (std::size_t p = 0; p < nb; ++p)
            build_p_block(p, scratch.work_u_lam_sig, scratch.work_u_v_sig);
#endif

        return out;
    }

    // ── transform_eri_internal ────────────────────────────────────────────────────

    std::vector<double> transform_eri_internal(
        const std::vector<double> &eri,
        std::size_t nb,
        const Eigen::MatrixXd &C_int)
    {
        // The internal-space helper is just the generic quarter-transform with the
        // same matrix on all four legs.
        return transform_eri(eri, nb, C_int, C_int, C_int, C_int);
    }

} // namespace HartreeFock::Correlation
