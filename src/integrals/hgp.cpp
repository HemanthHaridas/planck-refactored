#include "hgp.h"

#include "boys.h"
#include "os.h"
#include "screening.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numbers>
#include <tuple>

namespace
{
    using SymOps = std::vector<HartreeFock::SignedAOSymOp>;

    struct PairOrbitElem
    {
        std::size_t i = 0;
        std::size_t j = 0;
        int sign = 1;
    };

    struct QuartetOrbitElem
    {
        std::size_t i = 0;
        std::size_t j = 0;
        std::size_t k = 0;
        std::size_t l = 0;
        int sign = 1;
    };

    static bool use_symmetry_ops(const SymOps *sym_ops)
    {
        return sym_ops != nullptr && sym_ops->size() > 1;
    }

    static void canonicalize_pair(std::size_t &i, std::size_t &j)
    {
        if (i > j)
            std::swap(i, j);
    }

    static void canonicalize_quartet(
        std::size_t &i, std::size_t &j,
        std::size_t &k, std::size_t &l)
    {
        canonicalize_pair(i, j);
        canonicalize_pair(k, l);
        if (std::tie(i, j) > std::tie(k, l))
        {
            std::swap(i, k);
            std::swap(j, l);
        }
    }

    static bool append_quartet_orbit(
        std::vector<QuartetOrbitElem> &orbit,
        std::size_t i, std::size_t j,
        std::size_t k, std::size_t l,
        int sign)
    {
        for (const auto &elem : orbit)
        {
            if (elem.i == i && elem.j == j &&
                elem.k == k && elem.l == l)
                return elem.sign == sign;
        }
        orbit.push_back({i, j, k, l, sign});
        return true;
    }

    static bool append_pair_orbit(
        std::vector<PairOrbitElem> &orbit,
        std::size_t i, std::size_t j, int sign)
    {
        for (const auto &elem : orbit)
        {
            if (elem.i == i && elem.j == j)
                return elem.sign == sign;
        }
        orbit.push_back({i, j, sign});
        return true;
    }

    static std::pair<std::vector<PairOrbitElem>, bool> build_pair_orbit(
        std::size_t i, std::size_t j, const SymOps &sym_ops)
    {
        std::vector<PairOrbitElem> orbit;
        orbit.reserve(sym_ops.size());

        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            const int sign =
                static_cast<int>(op.ao_sign[i]) *
                static_cast<int>(op.ao_sign[j]);
            canonicalize_pair(ii, jj);
            if (!append_pair_orbit(orbit, ii, jj, sign))
                return {orbit, true};
        }

        std::sort(
            orbit.begin(), orbit.end(),
            [](const PairOrbitElem &a, const PairOrbitElem &b)
            {
                return std::tie(a.i, a.j) < std::tie(b.i, b.j);
            });
        return {orbit, false};
    }

    static std::pair<std::vector<QuartetOrbitElem>, bool> build_quartet_orbit(
        std::size_t i, std::size_t j,
        std::size_t k, std::size_t l,
        const SymOps &sym_ops)
    {
        std::vector<QuartetOrbitElem> orbit;
        orbit.reserve(sym_ops.size());

        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            std::size_t kk = static_cast<std::size_t>(op.ao_map[k]);
            std::size_t ll = static_cast<std::size_t>(op.ao_map[l]);
            const int sign =
                static_cast<int>(op.ao_sign[i]) *
                static_cast<int>(op.ao_sign[j]) *
                static_cast<int>(op.ao_sign[k]) *
                static_cast<int>(op.ao_sign[l]);
            canonicalize_quartet(ii, jj, kk, ll);
            if (!append_quartet_orbit(orbit, ii, jj, kk, ll, sign))
                return {orbit, true};
        }

        std::sort(
            orbit.begin(), orbit.end(),
            [](const QuartetOrbitElem &a, const QuartetOrbitElem &b)
            {
                return std::tie(a.i, a.j, a.k, a.l) <
                       std::tie(b.i, b.j, b.k, b.l);
            });
        return {orbit, false};
    }

    static void write_eri_permutations(
        std::vector<double> &eri,
        std::size_t nb, std::size_t nb2, std::size_t nb3,
        std::size_t i, std::size_t j,
        std::size_t k, std::size_t l,
        double val)
    {
        auto write_slot = [&](std::size_t idx)
        {
#ifdef USE_OPENMP
#pragma omp atomic write
#endif
            eri[idx] = val;
        };

        write_slot(i * nb3 + j * nb2 + k * nb + l);
        write_slot(j * nb3 + i * nb2 + k * nb + l);
        write_slot(i * nb3 + j * nb2 + l * nb + k);
        write_slot(j * nb3 + i * nb2 + l * nb + k);
        write_slot(k * nb3 + l * nb2 + i * nb + j);
        write_slot(l * nb3 + k * nb2 + i * nb + j);
        write_slot(k * nb3 + l * nb2 + j * nb + i);
        write_slot(l * nb3 + k * nb2 + j * nb + i);
    }

    struct EriScratch
    {
        int ax_dim = 0;
        int ay_dim = 0;
        int az_dim = 0;
        int cx_dim = 0;
        int cy_dim = 0;
        int cz_dim = 0;
        int m_dim = 0;
        std::size_t ax_stride = 0;
        std::size_t ay_stride = 0;
        std::size_t az_stride = 0;
        std::size_t cx_stride = 0;
        std::size_t cy_stride = 0;
        std::size_t cz_stride = 0;
        std::size_t cd_block_size = 0;
        std::size_t spatial_size = 0;
        std::vector<double> vrr;
        std::vector<double> hrr;
        std::vector<double> a0c0_accum;
        double *vrr_data = nullptr;
        double *hrr_data = nullptr;
        double *a0c0_data = nullptr;

        void resize_for_quartet(
            int lABx, int lABy, int lABz,
            int lCDx, int lCDy, int lCDz,
            int mmax)
        {
            ax_dim = lABx + 1;
            ay_dim = lABy + 1;
            az_dim = lABz + 1;
            cx_dim = lCDx + 1;
            cy_dim = lCDy + 1;
            cz_dim = lCDz + 1;
            m_dim = mmax + 1;
            cz_stride = 1;
            cy_stride = static_cast<std::size_t>(cz_dim) * cz_stride;
            cx_stride = static_cast<std::size_t>(cy_dim) * cy_stride;
            cd_block_size = static_cast<std::size_t>(cx_dim) * cy_dim * cz_dim;
            az_stride = static_cast<std::size_t>(cx_dim) * cx_stride;
            ay_stride = static_cast<std::size_t>(az_dim) * az_stride;
            ax_stride = static_cast<std::size_t>(ay_dim) * ay_stride;

            spatial_size =
                static_cast<std::size_t>(ax_dim) * ay_dim * az_dim *
                cx_dim * cy_dim * cz_dim;
            const std::size_t vrr_size = spatial_size * static_cast<std::size_t>(m_dim);
            if (vrr.size() != vrr_size)
                vrr.resize(vrr_size);
            std::fill(vrr.begin(), vrr.end(), 0.0);
            if (hrr.size() != spatial_size)
                hrr.resize(spatial_size);
            if (a0c0_accum.size() != spatial_size)
                a0c0_accum.resize(spatial_size);
            std::fill(a0c0_accum.begin(), a0c0_accum.end(), 0.0);
            vrr_data = vrr.data();
            hrr_data = hrr.data();
            a0c0_data = a0c0_accum.data();
        }

        std::size_t spatial_index(
            int ax, int ay, int az,
            int cx, int cy, int cz) const
        {
            return static_cast<std::size_t>(ax) * ax_stride +
                   static_cast<std::size_t>(ay) * ay_stride +
                   static_cast<std::size_t>(az) * az_stride +
                   static_cast<std::size_t>(cx) * cx_stride +
                   static_cast<std::size_t>(cy) * cy_stride +
                   static_cast<std::size_t>(cz) * cz_stride;
        }

        double &v(
            int ax, int ay, int az,
            int cx, int cy, int cz,
            int m)
        {
            return vrr_data[spatial_index(ax, ay, az, cx, cy, cz) *
                                static_cast<std::size_t>(m_dim) +
                            static_cast<std::size_t>(m)];
        }

        double *v_ptr(
            int ax, int ay, int az,
            int cx, int cy, int cz)
        {
            return vrr_data + spatial_index(ax, ay, az, cx, cy, cz) *
                                  static_cast<std::size_t>(m_dim);
        }

        double &h(
            int ax, int ay, int az,
            int cx, int cy, int cz)
        {
            return hrr_data[spatial_index(ax, ay, az, cx, cy, cz)];
        }

        double *h_block_ptr(int ax, int ay, int az)
        {
            return hrr_data +
                   static_cast<std::size_t>(ax) * ax_stride +
                   static_cast<std::size_t>(ay) * ay_stride +
                   static_cast<std::size_t>(az) * az_stride;
        }
    };

    static thread_local EriScratch g_hgp_scratch;

    static constexpr int VRR_DIM = 2 * MAX_L + 1;

    struct ScreenedKernelData
    {
        double rho = 0.0;
        double prefactor_scale = 1.0;
        double boys_scale = 1.0;
    };

    enum class PrimitiveWeightCenter
    {
        None,
        A,
        B,
        C,
        D,
    };

    static ScreenedKernelData screened_kernel_data(
        double rho,
        HartreeFock::ERIKernel kernel,
        double omega) noexcept
    {
        if (kernel == HartreeFock::ERIKernel::Coulomb)
            return ScreenedKernelData{.rho = rho, .prefactor_scale = 1.0, .boys_scale = 1.0};

        if (omega <= 0.0)
        {
            return kernel == HartreeFock::ERIKernel::LongRange
                       ? ScreenedKernelData{.rho = 0.0, .prefactor_scale = 0.0, .boys_scale = 0.0}
                       : ScreenedKernelData{.rho = rho, .prefactor_scale = 1.0, .boys_scale = 1.0};
        }

        const double omega2 = omega * omega;
        const double lambda = omega2 / (omega2 + rho);
        return ScreenedKernelData{
            .rho = lambda * rho,
            .prefactor_scale = std::sqrt(lambda),
            .boys_scale = lambda};
    }

    static void hgp_hrr_cd(
        double *W,
        const int cx_dim, const int cy_dim, const int cz_dim,
        const int lCx, const int lCy, const int lCz,
        const int lDx, const int lDy, const int lDz,
        const double CDx, const double CDy, const double CDz) noexcept
    {
        const auto idx = [=](int cx, int cy, int cz) -> std::size_t
        {
            return (static_cast<std::size_t>(cx) * static_cast<std::size_t>(cy_dim) +
                    static_cast<std::size_t>(cy)) *
                       static_cast<std::size_t>(cz_dim) +
                   static_cast<std::size_t>(cz);
        };

        // Transfer angular momentum from the bra-side accumulation on C over to
        // D one axis at a time, matching the iterative HRR style already used
        // elsewhere in the codebase.
        for (int kz = 0; kz < lDz; ++kz)
            for (int cx = 0; cx <= lCx + lDx; ++cx)
                for (int cy = 0; cy <= lCy + lDy; ++cy)
                    for (int cz = 0; cz <= lCz + lDz - kz - 1; ++cz)
                    {
                        const std::size_t dst = idx(cx, cy, cz);
                        W[dst] = W[idx(cx, cy, cz + 1)] + CDz * W[dst];
                    }

        for (int ky = 0; ky < lDy; ++ky)
            for (int cx = 0; cx <= lCx + lDx; ++cx)
                for (int cy = 0; cy <= lCy + lDy - ky - 1; ++cy)
                    for (int cz = 0; cz <= lCz; ++cz)
                    {
                        const std::size_t dst = idx(cx, cy, cz);
                        W[dst] = W[idx(cx, cy + 1, cz)] + CDy * W[dst];
                    }

        for (int kx = 0; kx < lDx; ++kx)
            for (int cx = 0; cx <= lCx + lDx - kx - 1; ++cx)
                for (int cy = 0; cy <= lCy; ++cy)
                    for (int cz = 0; cz <= lCz; ++cz)
                    {
                        const std::size_t dst = idx(cx, cy, cz);
                        W[dst] = W[idx(cx + 1, cy, cz)] + CDx * W[dst];
                    }
    }

    static void hgp_hrr_ab(
        EriScratch &scratch,
        const int lAx, const int lAy, const int lAz,
        const int lBx, const int lBy, const int lBz,
        const int lCDx, const int lCDy, const int lCDz,
        const double ABx, const double ABy, const double ABz)
    {
        // The AB transfer operates on the whole six-dimensional quartet block so
        // every CD component shares the same recurrence sweep.
        const std::size_t block_size = scratch.cd_block_size;

        for (int kz = 0; kz < lBz; ++kz)
            for (int ax = 0; ax <= lAx + lBx; ++ax)
                for (int ay = 0; ay <= lAy + lBy; ++ay)
                    for (int az = 0; az <= lAz + lBz - kz - 1; ++az)
                    {
                        double *dst = scratch.h_block_ptr(ax, ay, az);
                        const double *src = scratch.h_block_ptr(ax, ay, az + 1);
                        for (std::size_t n = 0; n < block_size; ++n)
                            dst[n] = src[n] + ABz * dst[n];
                    }

        for (int ky = 0; ky < lBy; ++ky)
            for (int ax = 0; ax <= lAx + lBx; ++ax)
                for (int ay = 0; ay <= lAy + lBy - ky - 1; ++ay)
                    for (int az = 0; az <= lAz; ++az)
                    {
                        double *dst = scratch.h_block_ptr(ax, ay, az);
                        const double *src = scratch.h_block_ptr(ax, ay + 1, az);
                        for (std::size_t n = 0; n < block_size; ++n)
                            dst[n] = src[n] + ABy * dst[n];
                    }

        for (int kx = 0; kx < lBx; ++kx)
            for (int ax = 0; ax <= lAx + lBx - kx - 1; ++ax)
                for (int ay = 0; ay <= lAy; ++ay)
                    for (int az = 0; az <= lAz; ++az)
                    {
                        double *dst = scratch.h_block_ptr(ax, ay, az);
                        const double *src = scratch.h_block_ptr(ax + 1, ay, az);
                        for (std::size_t n = 0; n < block_size; ++n)
                            dst[n] = src[n] + ABx * dst[n];
                    }
    }

    static void hgp_vrr(
        const HartreeFock::PrimitivePair &ppAB,
        const HartreeFock::PrimitivePair &ppCD,
        const int lABx, const int lABy, const int lABz,
        const int lCDx, const int lCDy, const int lCDz,
        EriScratch &scratch,
        HartreeFock::ERIKernel kernel,
        double omega)
    {
        const Eigen::Vector3d &P = ppAB.center;
        const Eigen::Vector3d &Q = ppCD.center;
        const double zetaAB = ppAB.zeta;
        const double zetaCD = ppCD.zeta;
        const double delta = zetaAB + zetaCD;
        const double inv_2_zetaAB = 0.5 / zetaAB;
        const double inv_2_zetaCD = 0.5 / zetaCD;
        const double rho = zetaAB * zetaCD / delta;
        const auto screen = screened_kernel_data(rho, kernel, omega);
        // C-VRR bra/ket coupling term carries a λ = boys_scale factor for
        // screened kernels (OS matches: src/integrals/os.cpp inv_2_delta).
        const double inv_2_delta =
            (0.5 / delta) *
            ((kernel == HartreeFock::ERIKernel::Coulomb) ? 1.0 : screen.boys_scale);
        const double rho_over_zetaAB = screen.rho / zetaAB;
        const double rho_over_zetaCD = screen.rho / zetaCD;
        const double wpwq_scale = screen.rho / rho;

        const Eigen::Vector3d W = (zetaAB * P + zetaCD * Q) / delta;
        const double WPx = (W[0] - P[0]) * wpwq_scale;
        const double WPy = (W[1] - P[1]) * wpwq_scale;
        const double WPz = (W[2] - P[2]) * wpwq_scale;
        const double WQx = (W[0] - Q[0]) * wpwq_scale;
        const double WQy = (W[1] - Q[1]) * wpwq_scale;
        const double WQz = (W[2] - Q[2]) * wpwq_scale;

        const double PAx = ppAB.pA[0], PAy = ppAB.pA[1], PAz = ppAB.pA[2];
        const double QCx = ppCD.pA[0], QCy = ppCD.pA[1], QCz = ppCD.pA[2];

        const double PQx = P[0] - Q[0], PQy = P[1] - Q[1], PQz = P[2] - Q[2];
        const double T = screen.boys_scale * rho * (PQx * PQx + PQy * PQy + PQz * PQz);
        const int MMAX = lABx + lABy + lABz + lCDx + lCDy + lCDz;

        const double prefac =
            ppAB.prefactor * ppCD.prefactor * 2.0 * std::sqrt(rho / std::numbers::pi) *
            screen.prefactor_scale;

        // Build the primitive quartet with a VRR-first / HRR-second layout.
        // Phase 1 keeps this explicit and close to the OS structure so the HGP
        // path can be validated independently before deeper optimization.
        for (int m = 0; m <= MMAX; ++m)
            scratch.v(0, 0, 0, 0, 0, 0, m) = prefac * HartreeFock::Lookup::boys(m, T);

        for (int ax = 1; ax <= lABx; ++ax)
        {
            const int mlim = MMAX - ax;
            double *dst = scratch.v_ptr(ax, 0, 0, 0, 0, 0);
            double *prev = scratch.v_ptr(ax - 1, 0, 0, 0, 0, 0);
            double *prev2 = (ax > 1) ? scratch.v_ptr(ax - 2, 0, 0, 0, 0, 0) : nullptr;
            for (int m = 0; m <= mlim; ++m)
            {
                dst[m] = PAx * prev[m] + WPx * prev[m + 1];
                if (ax > 1)
                    dst[m] +=
                        (ax - 1) * inv_2_zetaAB *
                        (prev2[m] - rho_over_zetaAB * prev2[m + 1]);
            }
        }

        for (int ax = 0; ax <= lABx; ++ax)
        {
            for (int ay = 1; ay <= lABy; ++ay)
            {
                const int mlim = MMAX - ax - ay;
                if (mlim < 0)
                    continue;
                double *dst = scratch.v_ptr(ax, ay, 0, 0, 0, 0);
                double *prev = scratch.v_ptr(ax, ay - 1, 0, 0, 0, 0);
                double *prev2 = (ay > 1) ? scratch.v_ptr(ax, ay - 2, 0, 0, 0, 0) : nullptr;
                for (int m = 0; m <= mlim; ++m)
                {
                    dst[m] = PAy * prev[m] + WPy * prev[m + 1];
                    if (ay > 1)
                        dst[m] +=
                            (ay - 1) * inv_2_zetaAB *
                            (prev2[m] - rho_over_zetaAB * prev2[m + 1]);
                }
            }
        }

        for (int ax = 0; ax <= lABx; ++ax)
        {
            for (int ay = 0; ay <= lABy; ++ay)
            {
                for (int az = 1; az <= lABz; ++az)
                {
                    const int mlim = MMAX - ax - ay - az;
                    if (mlim < 0)
                        continue;
                    double *dst = scratch.v_ptr(ax, ay, az, 0, 0, 0);
                    double *prev = scratch.v_ptr(ax, ay, az - 1, 0, 0, 0);
                    double *prev2 = (az > 1) ? scratch.v_ptr(ax, ay, az - 2, 0, 0, 0) : nullptr;
                    for (int m = 0; m <= mlim; ++m)
                    {
                        dst[m] = PAz * prev[m] + WPz * prev[m + 1];
                        if (az > 1)
                            dst[m] +=
                                (az - 1) * inv_2_zetaAB *
                                (prev2[m] - rho_over_zetaAB * prev2[m + 1]);
                    }
                }
            }
        }

        for (int ax = 0; ax <= lABx; ++ax)
        {
            for (int ay = 0; ay <= lABy; ++ay)
            {
                for (int az = 0; az <= lABz; ++az)
                {
                    for (int cx = 1; cx <= lCDx; ++cx)
                    {
                        const int mlim = MMAX - ax - ay - az - cx;
                        if (mlim < 0)
                            continue;
                        double *dst = scratch.v_ptr(ax, ay, az, cx, 0, 0);
                        double *prev = scratch.v_ptr(ax, ay, az, cx - 1, 0, 0);
                        double *prev2 = (cx > 1) ? scratch.v_ptr(ax, ay, az, cx - 2, 0, 0) : nullptr;
                        double *cross = (ax > 0) ? scratch.v_ptr(ax - 1, ay, az, cx - 1, 0, 0) : nullptr;
                        for (int m = 0; m <= mlim; ++m)
                        {
                            dst[m] = QCx * prev[m] + WQx * prev[m + 1];
                            if (cx > 1)
                                dst[m] +=
                                    (cx - 1) * inv_2_zetaCD *
                                    (prev2[m] - rho_over_zetaCD * prev2[m + 1]);
                            if (ax > 0)
                                dst[m] += ax * inv_2_delta * cross[m + 1];
                        }
                    }
                }
            }
        }

        for (int ax = 0; ax <= lABx; ++ax)
        {
            for (int ay = 0; ay <= lABy; ++ay)
            {
                for (int az = 0; az <= lABz; ++az)
                {
                    for (int cx = 0; cx <= lCDx; ++cx)
                    {
                        for (int cy = 1; cy <= lCDy; ++cy)
                        {
                            const int mlim = MMAX - ax - ay - az - cx - cy;
                            if (mlim < 0)
                                continue;
                            double *dst = scratch.v_ptr(ax, ay, az, cx, cy, 0);
                            double *prev = scratch.v_ptr(ax, ay, az, cx, cy - 1, 0);
                            double *prev2 = (cy > 1) ? scratch.v_ptr(ax, ay, az, cx, cy - 2, 0) : nullptr;
                            double *cross = (ay > 0) ? scratch.v_ptr(ax, ay - 1, az, cx, cy - 1, 0) : nullptr;
                            for (int m = 0; m <= mlim; ++m)
                            {
                                dst[m] = QCy * prev[m] + WQy * prev[m + 1];
                                if (cy > 1)
                                    dst[m] +=
                                        (cy - 1) * inv_2_zetaCD *
                                        (prev2[m] - rho_over_zetaCD * prev2[m + 1]);
                                if (ay > 0)
                                    dst[m] += ay * inv_2_delta * cross[m + 1];
                            }
                        }
                    }
                }
            }
        }

        for (int ax = 0; ax <= lABx; ++ax)
        {
            for (int ay = 0; ay <= lABy; ++ay)
            {
                for (int az = 0; az <= lABz; ++az)
                {
                    for (int cx = 0; cx <= lCDx; ++cx)
                    {
                        for (int cy = 0; cy <= lCDy; ++cy)
                        {
                            for (int cz = 1; cz <= lCDz; ++cz)
                            {
                                const int mlim = MMAX - ax - ay - az - cx - cy - cz;
                                if (mlim < 0)
                                    continue;
                                double *dst = scratch.v_ptr(ax, ay, az, cx, cy, cz);
                                double *prev = scratch.v_ptr(ax, ay, az, cx, cy, cz - 1);
                                double *prev2 = (cz > 1) ? scratch.v_ptr(ax, ay, az, cx, cy, cz - 2) : nullptr;
                                double *cross = (az > 0) ? scratch.v_ptr(ax, ay, az - 1, cx, cy, cz - 1) : nullptr;
                                for (int m = 0; m <= mlim; ++m)
                                {
                                    dst[m] = QCz * prev[m] + WQz * prev[m + 1];
                                    if (cz > 1)
                                        dst[m] +=
                                            (cz - 1) * inv_2_zetaCD *
                                            (prev2[m] - rho_over_zetaCD * prev2[m + 1]);
                                    if (az > 0)
                                        dst[m] += az * inv_2_delta * cross[m + 1];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Runs VRR for one primitive pair and writes the m=0 (a0|c0) slice into
    // out_a0c0 (length scratch.spatial_size). Assumes scratch has already
    // been sized via resize_for_quartet so that out_a0c0 (typically pointing
    // into scratch storage) is not invalidated mid-call.
    // Does not touch scratch.hrr_data, so the same scratch can be reused
    // across primitive pairs for an outside-the-loop HRR.
    static void hgp_eri_primitive_vrr_only(
        const HartreeFock::PrimitivePair &ppAB,
        const HartreeFock::PrimitivePair &ppCD,
        const int lABx, const int lABy, const int lABz,
        const int lCDx, const int lCDy, const int lCDz,
        EriScratch &scratch,
        double *out_a0c0,
        HartreeFock::ERIKernel kernel,
        double omega)
    {
        hgp_vrr(ppAB, ppCD, lABx, lABy, lABz, lCDx, lCDy, lCDz, scratch, kernel, omega);

        const std::size_t m_stride = static_cast<std::size_t>(scratch.m_dim);
        for (std::size_t idx = 0; idx < scratch.spatial_size; ++idx)
            out_a0c0[idx] = scratch.vrr_data[idx * m_stride];
    }

    // Runs both HRR passes on scratch.hrr_data (which the caller has already
    // populated with the contracted (a0|c0; m=0) block) and returns the
    // scalar (ab|cd) ERI at the requested cartesian indices.
    static double hgp_hrr_finalize(
        EriScratch &scratch,
        const int lAx, const int lAy, const int lAz,
        const int lBx, const int lBy, const int lBz,
        const int lCx, const int lCy, const int lCz,
        const int lDx, const int lDy, const int lDz,
        const double ABx, const double ABy, const double ABz,
        const double CDx, const double CDy, const double CDz)
    {
        const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;

        hgp_hrr_ab(scratch, lAx, lAy, lAz, lBx, lBy, lBz, lCDx, lCDy, lCDz, ABx, ABy, ABz);

        double *cd_block = scratch.h_block_ptr(lAx, lAy, lAz);
        hgp_hrr_cd(
            cd_block, lCDx + 1, lCDy + 1, lCDz + 1,
            lCx, lCy, lCz, lDx, lDy, lDz, CDx, CDy, CDz);
        return cd_block[(static_cast<std::size_t>(lCx) * static_cast<std::size_t>(lCDy + 1) +
                         static_cast<std::size_t>(lCy)) *
                            static_cast<std::size_t>(lCDz + 1) +
                        static_cast<std::size_t>(lCz)];
    }

    static double hgp_contracted_eri_weighted_base(
        const HartreeFock::ShellPair &spAB,
        const HartreeFock::ShellPair &spCD,
        int lAx, int lAy, int lAz,
        int lBx, int lBy, int lBz,
        int lCx, int lCy, int lCz,
        int lDx, int lDy, int lDz,
        HartreeFock::ERIKernel kernel,
        double omega,
        PrimitiveWeightCenter weight_center)
    {
        const int lABx = lAx + lBx, lABy = lAy + lBy, lABz = lAz + lBz;
        const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;
        const int mmax = lABx + lABy + lABz + lCDx + lCDy + lCDz;

        // HGP loop reorder: contract VRR results across all primitive pairs into a
        // single (a0|c0) block, then run the two HRR passes once per shell quartet
        // instead of once per primitive pair. Derivative-weighted contractions keep
        // the same structure and only change the primitive prefactor per center.
        EriScratch &scratch = g_hgp_scratch;
        scratch.resize_for_quartet(lABx, lABy, lABz, lCDx, lCDy, lCDz, mmax);

        // hrr_data is only read inside hgp_hrr_finalize, which runs after the
        // accumulation loop completes — safe to use as per-pair VRR scratch in
        // the meantime, avoiding a per-quartet allocation.
        double *a0c0_pair = scratch.hrr_data;
        for (const auto &ppAB : spAB.primitive_pairs)
        {
            for (const auto &ppCD : spCD.primitive_pairs)
            {
                hgp_eri_primitive_vrr_only(
                    ppAB, ppCD, lABx, lABy, lABz, lCDx, lCDy, lCDz,
                    scratch, a0c0_pair, kernel, omega);

                double w = ppAB.coeff_product * ppCD.coeff_product;
                switch (weight_center)
                {
                case PrimitiveWeightCenter::None:
                    break;
                case PrimitiveWeightCenter::A:
                    w *= 2.0 * ppAB.alpha;
                    break;
                case PrimitiveWeightCenter::B:
                    w *= 2.0 * ppAB.beta;
                    break;
                case PrimitiveWeightCenter::C:
                    w *= 2.0 * ppCD.alpha;
                    break;
                case PrimitiveWeightCenter::D:
                    w *= 2.0 * ppCD.beta;
                    break;
                }

                for (std::size_t n = 0; n < scratch.spatial_size; ++n)
                    scratch.a0c0_data[n] += w * a0c0_pair[n];
            }
        }

        std::copy(scratch.a0c0_data,
                  scratch.a0c0_data + scratch.spatial_size,
                  scratch.hrr_data);

        const double ABx = spAB.R[0], ABy = spAB.R[1], ABz = spAB.R[2];
        const double CDx = spCD.R[0], CDy = spCD.R[1], CDz = spCD.R[2];
        return hgp_hrr_finalize(
            scratch,
            lAx, lAy, lAz, lBx, lBy, lBz,
            lCx, lCy, lCz, lDx, lDy, lDz,
            ABx, ABy, ABz, CDx, CDy, CDz);
    }

    static double hgp_contracted_eri_weighted(
        const HartreeFock::ShellPair &spAB,
        const HartreeFock::ShellPair &spCD,
        int lAx, int lAy, int lAz,
        int lBx, int lBy, int lBz,
        int lCx, int lCy, int lCz,
        int lDx, int lDy, int lDz,
        HartreeFock::ERIKernel kernel,
        double omega,
        PrimitiveWeightCenter weight_center)
    {
        if (kernel != HartreeFock::ERIKernel::ShortRange)
        {
            return hgp_contracted_eri_weighted_base(
                spAB, spCD,
                lAx, lAy, lAz, lBx, lBy, lBz,
                lCx, lCy, lCz, lDx, lDy, lDz,
                kernel, omega, weight_center);
        }

        if (omega <= 0.0)
            return 0.0;

        return hgp_contracted_eri_weighted_base(
                   spAB, spCD,
                   lAx, lAy, lAz, lBx, lBy, lBz,
                   lCx, lCy, lCz, lDx, lDy, lDz,
                   HartreeFock::ERIKernel::Coulomb, 0.0, weight_center) -
               hgp_contracted_eri_weighted_base(
                   spAB, spCD,
                   lAx, lAy, lAz, lBx, lBy, lBz,
                   lCx, lCy, lCz, lDx, lDy, lDz,
                   HartreeFock::ERIKernel::LongRange, omega, weight_center);
    }
} // namespace

double HartreeFock::HeadGordonPople::_contracted_eri_elem(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    int lAx, int lAy, int lAz,
    int lBx, int lBy, int lBz,
    int lCx, int lCy, int lCz,
    int lDx, int lDy, int lDz,
    HartreeFock::ERIKernel kernel,
    double omega)
{
    return hgp_contracted_eri_weighted(
        spAB, spCD,
        lAx, lAy, lAz, lBx, lBy, lBz,
        lCx, lCy, lCz, lDx, lDy, lDz,
        kernel, omega, PrimitiveWeightCenter::None);
}

double HartreeFock::HeadGordonPople::_contracted_eri_elem_native_test(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    int lAx, int lAy, int lAz,
    int lBx, int lBy, int lBz,
    int lCx, int lCy, int lCz,
    int lDx, int lDy, int lDz,
    HartreeFock::ERIKernel kernel,
    double omega)
{
    return hgp_contracted_eri_weighted(
        spAB, spCD,
        lAx, lAy, lAz, lBx, lBy, lBz,
        lCx, lCy, lCz, lDx, lDy, lDz,
        kernel, omega, PrimitiveWeightCenter::None);
}

double HartreeFock::HeadGordonPople::_contracted_eri_elem_weighted_native_test(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    int lAx, int lAy, int lAz,
    int lBx, int lBy, int lBz,
    int lCx, int lCy, int lCz,
    int lDx, int lDy, int lDz,
    int weight_center,
    HartreeFock::ERIKernel kernel,
    double omega)
{
    const auto center = [&]() -> PrimitiveWeightCenter
    {
        switch (weight_center)
        {
            case 0:
                return PrimitiveWeightCenter::A;
            case 1:
                return PrimitiveWeightCenter::B;
            case 2:
                return PrimitiveWeightCenter::C;
            case 3:
                return PrimitiveWeightCenter::D;
            default:
                throw std::runtime_error("invalid weighted centre for HGP ERI derivative test hook");
        }
    }();

    return hgp_contracted_eri_weighted(
        spAB, spCD,
        lAx, lAy, lAz, lBx, lBy, lBz,
        lCx, lCy, lCz, lDx, lDy, lDz,
        kernel, omega, center);
}

std::vector<double> HartreeFock::HeadGordonPople::_compute_2e(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    const bool use_sym = use_symmetry_ops(sym_ops);

    // Shared HGP-based Schwarz table; lives in src/integrals/screening.cpp
    // so that other engines (notably the Auto path) can share the cheap
    // implementation without duplicating it here.
    const std::vector<double> Q = HartreeFock::Screening::schwarz_table_hgp(
        shell_pairs, nb, sym_ops);
    std::vector<double> eri(nb * nb * nb * nb, 0.0);

    const std::size_t npairs = shell_pairs.size();
#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p)
    {
        const auto &spAB = shell_pairs[p];
        const std::size_t i = spAB.A._index;
        const std::size_t j = spAB.B._index;
        const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
        const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

        for (std::size_t q = p; q < npairs; ++q)
        {
            const auto &spCD = shell_pairs[q];
            const std::size_t k = spCD.A._index;
            const std::size_t l = spCD.B._index;
            std::vector<QuartetOrbitElem> orbit;
            if (Q[i * nb + j] * Q[k * nb + l] < tol_eri)
                continue;

            if (use_sym)
            {
                auto [orb, forced_zero] =
                    build_quartet_orbit(i, j, k, l, *sym_ops);
                if (forced_zero)
                    continue;
                orbit = std::move(orb);
                if (orbit.front().i != i || orbit.front().j != j ||
                    orbit.front().k != k || orbit.front().l != l)
                    continue;
            }

            const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
            const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];
            const double val = HartreeFock::HeadGordonPople::_contracted_eri_elem(
                spAB, spCD,
                lAx, lAy, lAz, lBx, lBy, lBz,
                lCx, lCy, lCz, lDx, lDy, lDz,
                kernel, omega);

            if (!use_sym)
            {
                write_eri_permutations(eri, nb, nb2, nb3, i, j, k, l, val);
                continue;
            }

            for (const auto &elem : orbit)
            {
                write_eri_permutations(
                    eri, nb, nb2, nb3,
                    elem.i, elem.j, elem.k, elem.l,
                    static_cast<double>(elem.sign) * val);
            }
        }
    }

    return eri;
}

Eigen::MatrixXd HartreeFock::HeadGordonPople::_compute_2e_fock(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &density,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    std::vector<double> eri = _compute_2e(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    G(mu, nu) += density(lam, sig) *
                                 (eri[mu * nb3 + nu * nb2 + lam * nb + sig] - 0.5 * eri[mu * nb3 + lam * nb2 + nu * nb + sig]);

    return G;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::HeadGordonPople::_compute_2e_fock_uhf(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &Pa,
    const Eigen::MatrixXd &Pb,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb * nb;
    const Eigen::MatrixXd Pt = Pa + Pb;
    std::vector<double> eri = _compute_2e(shell_pairs, nbasis, kernel, omega, tol_eri, sym_ops);
    Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);

#pragma omp parallel for schedule(static)
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                {
                    const double coulomb = eri[mu * nb3 + nu * nb2 + lam * nb + sig];
                    const double exch = eri[mu * nb3 + lam * nb2 + nu * nb + sig];
                    Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                    Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                }

    return {Ga, Gb};
}

std::array<double, 12> HartreeFock::HeadGordonPople::_compute_eri_deriv_elem(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    const HartreeFock::ERIKernel kernel,
    double omega)
{
    const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
    const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];
    const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
    const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

    std::array<double, 12> result{};
    for (int q = 0; q < 3; ++q)
    {
        const int axp = lAx + (q == 0), ayp = lAy + (q == 1), azp = lAz + (q == 2);
        const int bxp = lBx + (q == 0), byp = lBy + (q == 1), bzp = lBz + (q == 2);
        const int cxp = lCx + (q == 0), cyp = lCy + (q == 1), czp = lCz + (q == 2);
        const int dxp = lDx + (q == 0), dyp = lDy + (q == 1), dzp = lDz + (q == 2);

        result[q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            axp, ayp, azp, lBx, lBy, lBz,
            lCx, lCy, lCz, lDx, lDy, lDz,
            kernel, omega, PrimitiveWeightCenter::A);
        result[3 + q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            lAx, lAy, lAz, bxp, byp, bzp,
            lCx, lCy, lCz, lDx, lDy, lDz,
            kernel, omega, PrimitiveWeightCenter::B);
        result[6 + q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            lAx, lAy, lAz, lBx, lBy, lBz,
            cxp, cyp, czp, lDx, lDy, lDz,
            kernel, omega, PrimitiveWeightCenter::C);
        result[9 + q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            lAx, lAy, lAz, lBx, lBy, lBz,
            lCx, lCy, lCz, dxp, dyp, dzp,
            kernel, omega, PrimitiveWeightCenter::D);

        const int lAq = spAB.A._cartesian[q];
        const int lBq = spAB.B._cartesian[q];
        const int lCq = spCD.A._cartesian[q];
        const int lDq = spCD.B._cartesian[q];
        if (lAq > 0)
        {
            const int axm = lAx - (q == 0), aym = lAy - (q == 1), azm = lAz - (q == 2);
            result[q] -= static_cast<double>(lAq) *
                         _contracted_eri_elem(
                             spAB, spCD,
                             axm, aym, azm, lBx, lBy, lBz,
                             lCx, lCy, lCz, lDx, lDy, lDz,
                             kernel, omega);
        }
        if (lBq > 0)
        {
            const int bxm = lBx - (q == 0), bym = lBy - (q == 1), bzm = lBz - (q == 2);
            result[3 + q] -= static_cast<double>(lBq) *
                             _contracted_eri_elem(
                                 spAB, spCD,
                                 lAx, lAy, lAz, bxm, bym, bzm,
                                 lCx, lCy, lCz, lDx, lDy, lDz,
                                 kernel, omega);
        }
        if (lCq > 0)
        {
            const int cxm = lCx - (q == 0), cym = lCy - (q == 1), czm = lCz - (q == 2);
            result[6 + q] -= static_cast<double>(lCq) *
                             _contracted_eri_elem(
                                 spAB, spCD,
                                 lAx, lAy, lAz, lBx, lBy, lBz,
                                 cxm, cym, czm, lDx, lDy, lDz,
                                 kernel, omega);
        }
        if (lDq > 0)
        {
            const int dxm = lDx - (q == 0), dym = lDy - (q == 1), dzm = lDz - (q == 2);
            result[9 + q] -= static_cast<double>(lDq) *
                             _contracted_eri_elem(
                                 spAB, spCD,
                                 lAx, lAy, lAz, lBx, lBy, lBz,
                                 lCx, lCy, lCz, dxm, dym, dzm,
                                 kernel, omega);
        }
    }

    return result;
}

std::array<double, 12> HartreeFock::HeadGordonPople::_compute_eri_deriv_elem_native_test(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    const HartreeFock::ERIKernel kernel,
    double omega)
{
    const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
    const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];
    const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
    const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

    std::array<double, 12> result{};
    for (int q = 0; q < 3; ++q)
    {
        const int axp = lAx + (q == 0), ayp = lAy + (q == 1), azp = lAz + (q == 2);
        const int bxp = lBx + (q == 0), byp = lBy + (q == 1), bzp = lBz + (q == 2);
        const int cxp = lCx + (q == 0), cyp = lCy + (q == 1), czp = lCz + (q == 2);
        const int dxp = lDx + (q == 0), dyp = lDy + (q == 1), dzp = lDz + (q == 2);

        result[q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            axp, ayp, azp, lBx, lBy, lBz,
            lCx, lCy, lCz, lDx, lDy, lDz,
            kernel, omega, PrimitiveWeightCenter::A);
        result[3 + q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            lAx, lAy, lAz, bxp, byp, bzp,
            lCx, lCy, lCz, lDx, lDy, lDz,
            kernel, omega, PrimitiveWeightCenter::B);
        result[6 + q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            lAx, lAy, lAz, lBx, lBy, lBz,
            cxp, cyp, czp, lDx, lDy, lDz,
            kernel, omega, PrimitiveWeightCenter::C);
        result[9 + q] += hgp_contracted_eri_weighted(
            spAB, spCD,
            lAx, lAy, lAz, lBx, lBy, lBz,
            lCx, lCy, lCz, dxp, dyp, dzp,
            kernel, omega, PrimitiveWeightCenter::D);

        const int lAq = spAB.A._cartesian[q];
        const int lBq = spAB.B._cartesian[q];
        const int lCq = spCD.A._cartesian[q];
        const int lDq = spCD.B._cartesian[q];

        if (lAq > 0)
            result[q] -= lAq * _contracted_eri_elem_native_test(
                spAB, spCD,
                lAx - (q == 0), lAy - (q == 1), lAz - (q == 2), lBx, lBy, lBz,
                lCx, lCy, lCz, lDx, lDy, lDz,
                kernel, omega);
        if (lBq > 0)
            result[3 + q] -= lBq * _contracted_eri_elem_native_test(
                spAB, spCD,
                lAx, lAy, lAz, lBx - (q == 0), lBy - (q == 1), lBz - (q == 2),
                lCx, lCy, lCz, lDx, lDy, lDz,
                kernel, omega);
        if (lCq > 0)
            result[6 + q] -= lCq * _contracted_eri_elem_native_test(
                spAB, spCD,
                lAx, lAy, lAz, lBx, lBy, lBz,
                lCx - (q == 0), lCy - (q == 1), lCz - (q == 2), lDx, lDy, lDz,
                kernel, omega);
        if (lDq > 0)
            result[9 + q] -= lDq * _contracted_eri_elem_native_test(
                spAB, spCD,
                lAx, lAy, lAz, lBx, lBy, lBz,
                lCx, lCy, lCz, lDx - (q == 0), lDy - (q == 1), lDz - (q == 2),
                kernel, omega);
    }

    return result;
}
