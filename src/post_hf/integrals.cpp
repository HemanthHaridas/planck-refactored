#include <algorithm>
#include <format>
#include <limits>

#include "integrals.h"
#include "basis/spherical.h"
#include "integrals/base.h"
#include "io/logging.h"
#include "post_hf/ri/ri_eri.h"

namespace HartreeFock::Correlation
{

    // ── ensure_eri ────────────────────────────────────────────────────────────────

    const std::vector<double> &ensure_eri(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::vector<double> &eri_local,
        const std::string &tag)
    {
        // Spherical mode: every post-HF AO leg keys off working_nbasis(), so the
        // tensor must be the spherical one. A conventional SCF caches it in
        // calc._eri; a direct SCF (or the DFT driver) does not, so build the
        // Cartesian tensor and transform it, (pq|rs)_sph = C C C C (μν|λσ)_cart,
        // which is exact. Never return the bare Cartesian tensor: it would be
        // read at spherical indices.
        if (calc._shells._spherical)
        {
            if (!calc._eri.empty())
                return calc._eri;
            const std::size_t nb = calc._shells.nbasis();
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, tag,
                                         std::format("Building spherical AO ERI tensor from {} Cartesian AOs", nb));
            const std::vector<double> eri_cart =
                _compute_2e(shell_pairs, nb, calc._integral._engine, HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10,
                            calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
            // No size cap: the Cartesian n^4 tensor above is already the dominant
            // cost, and none of the ~20 callers survives an empty return.
            auto eri_sph = HartreeFock::BasisFunctions::transform_eri_cart_to_sph(
                eri_cart, calc._shells._cart_to_sph, nb, std::numeric_limits<std::size_t>::max());
            if (!eri_sph)
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, tag, eri_sph.error());
                eri_local.clear();
                return eri_local;
            }
            eri_local = std::move(*eri_sph);
            return eri_local;
        }

        // Prefer the cached AO tensor when it is already present on the
        // calculator; only build into caller-owned storage when needed.
        if (!calc._eri.empty())
            return calc._eri;

        const std::size_t nb = calc._shells.nbasis();
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, tag,
                                     std::format("Building AO ERI tensor ({:.1f} MB)", nb * nb * nb * nb * 8.0 / 1e6));
        eri_local = _compute_2e(shell_pairs, nb, calc._integral._engine,
                                HartreeFock::ERIKernel::Coulomb, 0.0, 1e-10,
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
        //
        // Each quarter transform is parallelized over its two leading indices,
        // which stride whole disjoint output slices. No thread shares a write
        // target and the inner accumulation order is unchanged, so the result is
        // identical to the serial version — this is a scheduling change only.
        // transform_eri is never called from inside an OpenMP parallel region
        // (all call sites are serial), so there is no nesting / over-subscription
        // concern.
        //
        // collapse(2) + schedule(dynamic): the leading index alone is the number
        // of occupied MOs (n1), which is O(10) — too few work units to fill 8+
        // threads without leaving cores idle at the barrier (profiled at ~38% idle
        // on the MP2 transform). Collapsing the first two loops widens the
        // iteration space (n1·n2, n1·nb, …) so dynamic scheduling can balance it.
        std::vector<double> T1(n1 * nb * nb * nb, 0.0);
#ifdef USE_OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        //
        // The contracted index sits outside the contiguous sigma loop, so both
        // tensors stream with unit stride; each element still sums over the
        // contracted index in ascending order, so the result is unchanged. With
        // mu innermost the AO read strode nb^3 and this quarter was 37 % of a
        // water/cc-pVTZ CASSCF run. Quarters 2 and 3 have the same shape.
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                {
                    double *t = T1.data() + i * nb3 + nu * nb2 + lam * nb;
                    for (std::size_t mu = 0; mu < nb; ++mu)
                    {
                        const double c = C1(mu, i);
                        const double *e = eri.data() + mu * nb3 + nu * nb2 + lam * nb;
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            t[sig] += c * e[sig];
                    }
                }

        // T2[i,a,λ,σ] = Σ_ν C2(ν,a) * T1[i,ν,λ,σ]   shape: n1 × n2 × nb × nb
        std::vector<double> T2(n1 * n2 * nb * nb, 0.0);
#ifdef USE_OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t lam = 0; lam < nb; ++lam)
                {
                    double *t = T2.data() + i * n2 * nb * nb + a * nb * nb + lam * nb;
                    for (std::size_t nu = 0; nu < nb; ++nu)
                    {
                        const double c = C2(nu, a);
                        const double *e = T1.data() + i * nb3 + nu * nb2 + lam * nb;
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            t[sig] += c * e[sig];
                    }
                }

        T1.clear();
        T1.shrink_to_fit();

        // T3[i,a,j,σ] = Σ_λ C3(λ,j) * T2[i,a,λ,σ]   shape: n1 × n2 × n3 × nb
        std::vector<double> T3(n1 * n2 * n3 * nb, 0.0);
#ifdef USE_OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t j = 0; j < n3; ++j)
                {
                    double *t = T3.data() + i * n2 * n3 * nb + a * n3 * nb + j * nb;
                    for (std::size_t lam = 0; lam < nb; ++lam)
                    {
                        const double c = C3(lam, j);
                        const double *e = T2.data() + i * n2 * nb * nb + a * nb * nb + lam * nb;
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            t[sig] += c * e[sig];
                    }
                }

        T2.clear();
        T2.shrink_to_fit();

        // out[i,a,j,b] = Σ_σ C4(σ,b) * T3[i,a,j,σ]   shape: n1 × n2 × n3 × n4
        std::vector<double> out(n1 * n2 * n3 * n4, 0.0);
#ifdef USE_OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t j = 0; j < n3; ++j)
                    for (std::size_t b = 0; b < n4; ++b)
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            out[i * n2 * n3 * n4 + a * n3 * n4 + j * n4 + b] +=
                                C4(sig, b) * T3[i * n2 * n3 * nb + a * n3 * nb + j * nb + sig];

        return out;
    }

    // ── transform_eri_ri ──────────────────────────────────────────────────────────

    std::expected<std::vector<double>, std::string> transform_eri_ri(
        HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &C1,
        const Eigen::MatrixXd &C2,
        const Eigen::MatrixXd &C3,
        const Eigen::MatrixXd &C4)
    {
        auto ri_ready = HartreeFock::Correlation::RI::ensure_ri_3c_ready(calculator);
        if (!ri_ready)
            return std::unexpected("transform_eri_ri: " + ri_ready.error());
        if (!calculator._ri_metric_factor)
            return std::unexpected("transform_eri_ri: RI metric factorization is missing.");

        // B_{(ia),Q} and B_{(jb),Q} share the same fitted AO-pair factors; the
        // two bra/ket blocks only differ in which MO coefficients project them.
        const Eigen::MatrixXd &pair_factors =
            HartreeFock::Correlation::RI::build_ri_pair_factors(calculator);
        const Eigen::MatrixXd b_bra =
            HartreeFock::Correlation::RI::build_ri_mo_block(pair_factors, C1, C2);
        const Eigen::MatrixXd b_ket =
            HartreeFock::Correlation::RI::build_ri_mo_block(pair_factors, C3, C4);

        // (ia|jb) = Σ_Q B_bra(ia,Q) B_ket(jb,Q); rows of b_bra are i*n2+a,
        // rows of b_ket are j*n4+b — exactly the transform_eri row-major layout.
        const Eigen::MatrixXd gram = HartreeFock::Correlation::RI::fitted_gram(b_bra, b_ket);

        const std::size_t n1 = static_cast<std::size_t>(C1.cols());
        const std::size_t n2 = static_cast<std::size_t>(C2.cols());
        const std::size_t n3 = static_cast<std::size_t>(C3.cols());
        const std::size_t n4 = static_cast<std::size_t>(C4.cols());
        std::vector<double> out(n1 * n2 * n3 * n4, 0.0);
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t j = 0; j < n3; ++j)
                    for (std::size_t b = 0; b < n4; ++b)
                        out[i * n2 * n3 * n4 + a * n3 * n4 + j * n4 + b] =
                            gram(static_cast<Eigen::Index>(i * n2 + a),
                                 static_cast<Eigen::Index>(j * n4 + b));
        return out;
    }

    // ── transform_eri_active_cache ──────────────────────────────────────────────

    std::vector<double> transform_eri_active_cache(
        const std::vector<double> &eri,
        std::size_t nb,
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &C_act)
    {
        const std::size_t na = static_cast<std::size_t>(C_act.cols());
        if (na == 0)
            return {};

        // (pu|vw) = (vw|up) under the AO tensor's 8-fold symmetry. Contracting
        // the three active legs first costs O(nb^4 na); putting the full-MO leg
        // first, as the generic leg order would, costs O(nb^5).
        const std::vector<double> vwup = transform_eri(eri, nb, C_act, C_act, C_act, C);

        std::vector<double> out(nb * na * na * na);
        for (std::size_t p = 0; p < nb; ++p)
            for (std::size_t u = 0; u < na; ++u)
                for (std::size_t v = 0; v < na; ++v)
                    for (std::size_t w = 0; w < na; ++w)
                        out[((p * na + u) * na + v) * na + w] = vwup[((v * na + w) * na + u) * nb + p];
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

    std::expected<std::vector<double>, std::string> transform_eri_internal_ri(
        HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &C_int)
    {
        return transform_eri_ri(calculator, C_int, C_int, C_int, C_int);
    }

    std::expected<std::vector<double>, std::string> transform_eri_active_cache_ri(
        HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &C,
        const Eigen::MatrixXd &C_act)
    {
        // _active_cache produces out[p,u,v,w] = (p u | v w), p full-MO, uvw
        // active — exactly transform_eri_ri with legs (C, C_act, C_act, C_act).
        return transform_eri_ri(calculator, C, C_act, C_act, C_act);
    }

} // namespace HartreeFock::Correlation
