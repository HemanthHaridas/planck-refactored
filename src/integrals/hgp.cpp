#include "hgp.h"

#include "boys.h"
#include "os.h"
#include "screening.h"
#include "quartet_layout.h"
#include "fused_fock.h"

#include <algorithm>
#include <array>
#include <cmath>
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

    // ─── Shell-quartet iteration support (H-10 step A2) ──────────────────────
    //
    // _compute_2e receives the per-Cartesian-AO shell_pairs list, not the Basis,
    // so to iterate at shell-quartet granularity we reconstruct the shell
    // grouping from the pair list itself. build_shellpairs emits every diagonal
    // pair (i,i), so each AO i appears as some pair's A side with A._index == i;
    // collecting those recovers the full AO -> ContractedView table. Runs of
    // AOs sharing the same Shell* (contiguous by construction) form the groups.
    // (Same approach as ObaraSaika::shell_groups_from_pairs; kept engine-local
    // because the helpers live in this translation unit's anonymous namespace.)
    struct HgpShellGroup
    {
        std::size_t first_ao = 0;    // _index of component 0
        std::size_t n_components = 0; // (L+1)(L+2)/2
    };

    static std::vector<HgpShellGroup> shell_groups_from_pairs(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::size_t nbasis,
        std::vector<const HartreeFock::ContractedView *> &ao_views)
    {
        ao_views.assign(nbasis, nullptr);
        for (const auto &sp : shell_pairs)
        {
            if (sp.A._index < nbasis)
                ao_views[sp.A._index] = &sp.A;
            if (sp.B._index < nbasis)
                ao_views[sp.B._index] = &sp.B;
        }

        std::vector<HgpShellGroup> groups;
        const HartreeFock::Shell *current = nullptr;
        for (std::size_t ao = 0; ao < nbasis; ++ao)
        {
            const HartreeFock::ContractedView *view = ao_views[ao];
            const HartreeFock::Shell *shell = view ? view->_shell : nullptr;
            if (groups.empty() || shell != current)
            {
                groups.push_back(HgpShellGroup{ao, 1});
                current = shell;
            }
            else
            {
                ++groups.back().n_components;
            }
        }
        return groups;
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
        // Six-axis dims/strides/index shared with OS and Rys (quartet_layout.h).
        HartreeFock::Integrals::SpatialQuartetLayout layout;
        int m_dim = 0;
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
            m_dim = mmax + 1;
            spatial_size =
                layout.configure(lABx, lABy, lABz, lCDx, lCDy, lCDz);
            cd_block_size = static_cast<std::size_t>(layout.cx_dim) *
                            layout.cy_dim * layout.cz_dim;
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
            return layout.spatial_index(ax, ay, az, cx, cy, cz);
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
                   static_cast<std::size_t>(ax) * layout.ax_stride +
                   static_cast<std::size_t>(ay) * layout.ay_stride +
                   static_cast<std::size_t>(az) * layout.az_stride;
        }
    };

    static thread_local EriScratch g_hgp_scratch;
    // Second scratch for the A4 hoisted block: g_hgp_scratch holds the shared
    // max-AM (a0|c0) accumulator for the whole quartet, while this one is sized
    // per Cartesian component for that component's HRR readout. Keeping them
    // separate lets the expensive contraction run once while each component HRRs
    // only its own sub-box (never the unreachable max-AM cube corners — see the
    // A4-pre finding in docs/SHELLPAIR_GRANULARITY_HANDOFF.md §5).
    static thread_local EriScratch g_hgp_hoist_comp_scratch;

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

    // Contract the per-primitive VRR (a0|c0; m=0) slices across all primitive
    // pairs into `scratch.a0c0_data`, at the AM ranges (lABx..lCDz) the caller
    // sized `scratch` to. This is the shared accumulation phase used by both the
    // per-component kernel (component AM ranges) and the H-10 step A4 hoisted
    // path (max AM ranges = L_A+L_B, L_C+L_D per axis). Because hgp_vrr is
    // strictly bottom-up, building at a larger max-AM box leaves every lower-AM
    // sub-block of a0c0_data bitwise-identical, and the primitive accumulation
    // order is unchanged — so a max-AM contraction's sub-block equals the
    // corresponding component-AM contraction exactly.
    static void hgp_contract_a0c0(
        const HartreeFock::ShellPair &spAB,
        const HartreeFock::ShellPair &spCD,
        EriScratch &scratch,
        int lABx, int lABy, int lABz,
        int lCDx, int lCDy, int lCDz,
        HartreeFock::ERIKernel kernel,
        double omega,
        PrimitiveWeightCenter weight_center)
    {
        const int mmax = lABx + lABy + lABz + lCDx + lCDy + lCDz;
        scratch.resize_for_quartet(lABx, lABy, lABz, lCDx, lCDy, lCDz, mmax);

        // hrr_data is only read after the accumulation loop, so it doubles as
        // the per-pair VRR scratch in the meantime (avoids a second buffer).
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

        // HGP loop reorder: contract VRR results across all primitive pairs into
        // a single (a0|c0) block, then run the two HRR passes once per shell
        // quartet instead of once per primitive pair.
        EriScratch &scratch = g_hgp_scratch;
        hgp_contract_a0c0(spAB, spCD, scratch,
                          lABx, lABy, lABz, lCDx, lCDy, lCDz,
                          kernel, omega, weight_center);

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

// Test hook (H-10 step A4-pre): contract the (a0|c0) block over all primitive
// pairs at a caller-specified AM box (lABx..lCDz per axis), then return the
// accumulated value at the logical coordinate (ax,ay,az,cx,cy,cz). This is the
// shared phase-1 accumulation the hoisted A4 path will run **once** per shell
// quartet at max AM (L_A+L_B / L_C+L_D per axis) and read per component. The
// gate validates the load-bearing §4 invariant: the value at every component's
// (a0|c0) coordinate inside the max-AM box equals the value from a per-component
// box sized exactly to that component. hgp_vrr is strictly bottom-up, so a
// larger box only adds higher-AM cells and leaves every lower coordinate
// bitwise-identical; this hook exposes the accumulator so the test can prove it
// on d-shells (the (dd|dd) case that NaN'd A4-1's dense-cube HRR readout).
// Build the whole (a0|c0) accumulator for one box ONCE and hand it back.
//
// `_contract_a0c0_at_native_test` below returns a single cell, but pays for a
// full contraction (every primitive pair, whole box) to produce it. Callers that
// want many cells of the same box — the box-invariance gates sweep every
// coordinate in it — must not call the per-cell entry in a loop: that rebuilds
// the box once per coordinate (a 5^6 = 15625x redundancy on a (dd|dd) quartet).
// Build once with this, then index `out` with `spatial_index`-equivalent
// row-major strides (cz fastest). Mirrors RysQuad::_build_sum_native_test.
void HartreeFock::HeadGordonPople::_build_a0c0_native_test(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    int lABx, int lABy, int lABz,
    int lCDx, int lCDy, int lCDz,
    HartreeFock::ERIKernel kernel,
    double omega,
    std::vector<double> &out)
{
    EriScratch &scratch = g_hgp_scratch;

    if (kernel == HartreeFock::ERIKernel::ShortRange)
    {
        // ShortRange = Coulomb - LongRange, so build both and subtract cellwise
        // (the per-cell entry does exactly this, one coordinate at a time).
        if (omega <= 0.0)
        {
            HartreeFock::Integrals::SpatialQuartetLayout probe;
            out.assign(probe.configure(lABx, lABy, lABz, lCDx, lCDy, lCDz), 0.0);
            return;
        }
        hgp_contract_a0c0(spAB, spCD, scratch,
                          lABx, lABy, lABz, lCDx, lCDy, lCDz,
                          HartreeFock::ERIKernel::Coulomb, 0.0,
                          PrimitiveWeightCenter::None);
        out.assign(scratch.a0c0_data, scratch.a0c0_data + scratch.spatial_size);
        hgp_contract_a0c0(spAB, spCD, scratch,
                          lABx, lABy, lABz, lCDx, lCDy, lCDz,
                          HartreeFock::ERIKernel::LongRange, omega,
                          PrimitiveWeightCenter::None);
        for (std::size_t n = 0; n < out.size(); ++n)
            out[n] -= scratch.a0c0_data[n];
        return;
    }

    hgp_contract_a0c0(spAB, spCD, scratch,
                      lABx, lABy, lABz, lCDx, lCDy, lCDz,
                      kernel, omega, PrimitiveWeightCenter::None);
    out.assign(scratch.a0c0_data, scratch.a0c0_data + scratch.spatial_size);
}

double HartreeFock::HeadGordonPople::_contract_a0c0_at_native_test(
    const HartreeFock::ShellPair &spAB,
    const HartreeFock::ShellPair &spCD,
    int lABx, int lABy, int lABz,
    int lCDx, int lCDy, int lCDz,
    int ax, int ay, int az,
    int cx, int cy, int cz,
    HartreeFock::ERIKernel kernel,
    double omega)
{
    EriScratch &scratch = g_hgp_scratch;
    if (kernel == HartreeFock::ERIKernel::ShortRange)
    {
        if (omega <= 0.0)
            return 0.0;
        hgp_contract_a0c0(spAB, spCD, scratch,
                          lABx, lABy, lABz, lCDx, lCDy, lCDz,
                          HartreeFock::ERIKernel::Coulomb, 0.0,
                          PrimitiveWeightCenter::None);
        const double coulomb =
            scratch.a0c0_data[scratch.spatial_index(ax, ay, az, cx, cy, cz)];
        hgp_contract_a0c0(spAB, spCD, scratch,
                          lABx, lABy, lABz, lCDx, lCDy, lCDz,
                          HartreeFock::ERIKernel::LongRange, omega,
                          PrimitiveWeightCenter::None);
        const double longrange =
            scratch.a0c0_data[scratch.spatial_index(ax, ay, az, cx, cy, cz)];
        return coulomb - longrange;
    }

    hgp_contract_a0c0(spAB, spCD, scratch,
                      lABx, lABy, lABz, lCDx, lCDy, lCDz,
                      kernel, omega, PrimitiveWeightCenter::None);
    return scratch.a0c0_data[scratch.spatial_index(ax, ay, az, cx, cy, cz)];
}

// ─── Shell-quartet block kernel (H-10 phase A, step A1) ──────────────────────
//
// Compute every Cartesian-component ERI of a shell quartet (A B | C D) in one
// call, filling a caller-provided flat buffer in [a][b][c][d] row-major order
// (d fastest). Like the OS step 2a block, this is an *iteration-shape* refactor:
// it loops the four shells' Cartesian components and calls the existing per-
// component _contracted_eri_elem once per (a,b,c,d), constructing each
// component's ShellPair from the real ContractedView entries in
// Basis::_basis_functions — exactly as build_shellpairs does, so the per-
// component _component_norm folded into PrimitivePair is identical and the
// values are bitwise-identical to the current per-AO path.
//
// It does NOT yet collapse HGP's per-component VRR-contract + HRR pipeline into
// one once-per-shell-quartet readout (that is step A4) and is NOT yet wired
// into any entry point (that is step A2/A3).
void HartreeFock::HeadGordonPople::_contracted_eri_block(
    const HartreeFock::Basis &basis,
    const ShellGroup &gA, const ShellGroup &gB,
    const ShellGroup &gC, const ShellGroup &gD,
    HartreeFock::ERIKernel kernel,
    double omega,
    double *block)
{
    const std::size_t nA = gA.n_components;
    const std::size_t nB = gB.n_components;
    const std::size_t nC = gC.n_components;
    const std::size_t nD = gD.n_components;
    const std::size_t nCD = nC * nD;

    for (std::size_t a = 0; a < nA; ++a)
    {
        const HartreeFock::ContractedView &cvA = basis._basis_functions[gA.first_ao + a];
        for (std::size_t b = 0; b < nB; ++b)
        {
            const HartreeFock::ContractedView &cvB = basis._basis_functions[gB.first_ao + b];
            const HartreeFock::ShellPair spAB(cvA, cvB);
            const int lAx = cvA._cartesian[0], lAy = cvA._cartesian[1], lAz = cvA._cartesian[2];
            const int lBx = cvB._cartesian[0], lBy = cvB._cartesian[1], lBz = cvB._cartesian[2];

            for (std::size_t c = 0; c < nC; ++c)
            {
                const HartreeFock::ContractedView &cvC = basis._basis_functions[gC.first_ao + c];
                for (std::size_t d = 0; d < nD; ++d)
                {
                    const HartreeFock::ContractedView &cvD = basis._basis_functions[gD.first_ao + d];
                    const HartreeFock::ShellPair spCD(cvC, cvD);
                    const int lCx = cvC._cartesian[0], lCy = cvC._cartesian[1], lCz = cvC._cartesian[2];
                    const int lDx = cvD._cartesian[0], lDy = cvD._cartesian[1], lDz = cvD._cartesian[2];

                    block[(a * nB + b) * nCD + (c * nD + d)] =
                        HartreeFock::HeadGordonPople::_contracted_eri_elem(
                            spAB, spCD,
                            lAx, lAy, lAz, lBx, lBy, lBz,
                            lCx, lCy, lCz, lDx, lDy, lDz,
                            kernel, omega);
                }
            }
        }
    }
}

// ─── Hoisted shell-quartet block kernel (H-10 step A4-1′) ────────────────────
//
// The A4 amortization: instead of re-running the per-primitive VRR + (a0|c0)
// contraction once per Cartesian component (what _contracted_eri_block above
// does via _contracted_eri_elem), contract the (a0|c0) block ONCE per shell
// quartet at the max AM box (lAB = L_A+L_B, lCD = L_C+L_D per axis), then HRR
// each component out of that shared accumulator. This is what turns the
// shell-quartet iteration shape (A1/A2) into an actual speedup: the expensive
// primitive loop runs once, the cheap per-component HRR runs n_components times.
//
// Two invariants, both established by A4-pre (planck-hgp-triangular-contract)
// and §3.2 of the handoff:
//   (1) Box-size invariance. hgp_vrr is strictly bottom-up, so a max-AM
//       contraction's value at any component's (a0|c0) coordinate is bitwise
//       equal to a component-AM contraction's. We therefore gather each
//       component's sub-box (0..lABx_comp, 0..lCDx_comp) out of the max
//       accumulator and HRR only that sub-box — never the unreachable max-AM
//       cube corners that NaN'd the original A4-1.
//   (2) Norm factoring. The shared contraction cannot carry per-component
//       _component_norm (it folds into PrimitivePair::coeff_product). So we
//       contract NORM-FREE (component_norm forced to 1 on the ShellPair's
//       views) and multiply each readout by normA·normB·normC·normD.
//
// Output layout matches _contracted_eri_block: [a][b][c][d] row-major, d fastest.
namespace
{
    // Gather one component's (a0|c0) sub-box out of the shared max-AM
    // accumulator `src_a0c0` (laid out with `src` strides) into `dst.hrr_data`
    // (laid out with `dst` strides, already sized to the component AM box), then
    // run both HRR passes and return the component ERI. `dst` is the per-thread
    // component scratch; `src` is the shared max-AM scratch.
    // Strides of the shared max-AM (a0|c0) accumulator, kept lightweight so the
    // readout can index a snapshot vector without copying the whole EriScratch.
    // Same stride convention as EriScratch — both now come from the shared
    // SpatialQuartetLayout, so they cannot drift.
    struct MaxBoxLayout
    {
        HartreeFock::Integrals::SpatialQuartetLayout layout;

        MaxBoxLayout() = default;
        MaxBoxLayout(int lABx, int lABy, int lABz, int lCDx, int lCDy, int lCDz)
        {
            layout.configure(lABx, lABy, lABz, lCDx, lCDy, lCDz);
        }
        std::size_t index(int ax, int ay, int az, int cx, int cy, int cz) const
        {
            return layout.spatial_index(ax, ay, az, cx, cy, cz);
        }
    };

    static double hgp_hoist_readout_component(
        const MaxBoxLayout &src, const double *src_a0c0,
        EriScratch &dst,
        int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
        int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
        double ABx, double ABy, double ABz,
        double CDx, double CDy, double CDz)
    {
        const int lABx = lAx + lBx, lABy = lAy + lBy, lABz = lAz + lBz;
        const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;
        const int mmax = lABx + lABy + lABz + lCDx + lCDy + lCDz;
        dst.resize_for_quartet(lABx, lABy, lABz, lCDx, lCDy, lCDz, mmax);

        // Copy the component's (a0|c0) sub-block, remapping max strides -> comp
        // strides. Both index the same logical (ax,ay,az,cx,cy,cz) coordinate;
        // only the stride layout differs.
        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                    for (int cx = 0; cx <= lCDx; ++cx)
                        for (int cy = 0; cy <= lCDy; ++cy)
                            for (int cz = 0; cz <= lCDz; ++cz)
                                dst.hrr_data[dst.spatial_index(ax, ay, az, cx, cy, cz)] =
                                    src_a0c0[src.index(ax, ay, az, cx, cy, cz)];

        return hgp_hrr_finalize(
            dst,
            lAx, lAy, lAz, lBx, lBy, lBz,
            lCx, lCy, lCz, lDx, lDy, lDz,
            ABx, ABy, ABz, CDx, CDy, CDz);
    }

    // Build a norm-free copy of a ContractedView: same shell (exponents/center),
    // but _component_norm = 1 so the folded coeff_product carries no per-
    // component norm. The cartesian AM is irrelevant to the contraction (which
    // takes explicit AM ranges) but we leave it as-is for clarity.
    static HartreeFock::ContractedView normfree_view(
        const HartreeFock::ContractedView &v)
    {
        HartreeFock::ContractedView out = v;
        out._component_norm = 1.0;
        return out;
    }

    // Holds the once-per-shell-quartet shared (a0|c0) contraction(s) so any
    // surviving Cartesian component can read its ERI out cheaply. Built norm-
    // free at the max AM box (invariants (1)/(2) above); component readout
    // applies normA·normB·normC·normD. Used by both _contracted_eri_block_hoisted
    // and _compute_2e (A4-2). The contraction is deferred to the first prepare()
    // call so a fully screened-out quartet pays nothing.
    struct HoistedQuartet
    {
        // Norm-free views must outlive spAB/spCD (which hold references).
        HartreeFock::ContractedView nfA, nfB, nfC, nfD;
        HartreeFock::ShellPair spAB, spCD;
        double ABx, ABy, ABz, CDx, CDy, CDz;
        int maxAB, maxCD;
        bool short_range;
        bool zero;        // ShortRange with omega<=0: identically zero
        bool prepared = false;
        std::vector<double> a0c0_primary;   // Coulomb (ShortRange) or `kernel`
        std::vector<double> a0c0_secondary; // LongRange (ShortRange only)
        std::size_t spatial_size = 0;
        // Max-AM stride layout so readout can index either snapshot independent
        // of the shared scratch's later per-component resizes.
        MaxBoxLayout layout;

        HoistedQuartet(
            const HartreeFock::ContractedView &cvA0,
            const HartreeFock::ContractedView &cvB0,
            const HartreeFock::ContractedView &cvC0,
            const HartreeFock::ContractedView &cvD0,
            HartreeFock::ERIKernel kernel, double omega)
            : nfA(normfree_view(cvA0)), nfB(normfree_view(cvB0)),
              nfC(normfree_view(cvC0)), nfD(normfree_view(cvD0)),
              spAB(nfA, nfB), spCD(nfC, nfD)
        {
            ABx = spAB.R[0]; ABy = spAB.R[1]; ABz = spAB.R[2];
            CDx = spCD.R[0]; CDy = spCD.R[1]; CDz = spCD.R[2];
            const int LA = static_cast<int>(cvA0._shell->_shell);
            const int LB = static_cast<int>(cvB0._shell->_shell);
            const int LC = static_cast<int>(cvC0._shell->_shell);
            const int LD = static_cast<int>(cvD0._shell->_shell);
            maxAB = LA + LB;
            maxCD = LC + LD;
            short_range = (kernel == HartreeFock::ERIKernel::ShortRange);
            zero = (short_range && omega <= 0.0);
            kernel_ = kernel;
            omega_ = omega;
        }

        void prepare()
        {
            if (prepared || zero)
                return;
            prepared = true;
            EriScratch &shared = g_hgp_scratch;
            const HartreeFock::ERIKernel primary =
                short_range ? HartreeFock::ERIKernel::Coulomb : kernel_;
            hgp_contract_a0c0(spAB, spCD, shared,
                              maxAB, maxAB, maxAB, maxCD, maxCD, maxCD,
                              primary, omega_, PrimitiveWeightCenter::None);
            spatial_size = shared.spatial_size;
            layout = MaxBoxLayout(maxAB, maxAB, maxAB, maxCD, maxCD, maxCD);
            a0c0_primary.assign(shared.a0c0_data, shared.a0c0_data + spatial_size);
            if (short_range)
            {
                hgp_contract_a0c0(spAB, spCD, shared,
                                  maxAB, maxAB, maxAB, maxCD, maxCD, maxCD,
                                  HartreeFock::ERIKernel::LongRange, omega_,
                                  PrimitiveWeightCenter::None);
                a0c0_secondary.assign(shared.a0c0_data,
                                      shared.a0c0_data + spatial_size);
            }
        }

        // ERI for one Cartesian component (norm applied by caller's `norm`).
        double readout(
            int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
            int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
            double norm)
        {
            if (zero)
                return 0.0;
            prepare();
            EriScratch &comp = g_hgp_hoist_comp_scratch;
            double value = hgp_hoist_readout_component(
                layout, a0c0_primary.data(), comp,
                lAx, lAy, lAz, lBx, lBy, lBz,
                lCx, lCy, lCz, lDx, lDy, lDz,
                ABx, ABy, ABz, CDx, CDy, CDz);
            if (short_range)
                value -= hgp_hoist_readout_component(
                    layout, a0c0_secondary.data(), comp,
                    lAx, lAy, lAz, lBx, lBy, lBz,
                    lCx, lCy, lCz, lDx, lDy, lDz,
                    ABx, ABy, ABz, CDx, CDy, CDz);
            return value * norm;
        }

    private:
        HartreeFock::ERIKernel kernel_;
        double omega_;
    };
} // namespace

void HartreeFock::HeadGordonPople::_contracted_eri_block_hoisted(
    const HartreeFock::Basis &basis,
    const ShellGroup &gA, const ShellGroup &gB,
    const ShellGroup &gC, const ShellGroup &gD,
    HartreeFock::ERIKernel kernel,
    double omega,
    double *block)
{
    // Basis stores components contiguously, so build per-shell pointer arrays
    // into _basis_functions and delegate to the pointer-array core (which the
    // Auto path also feeds, from its non-contiguous ao_views table).
    std::vector<const HartreeFock::ContractedView *> pA(gA.n_components);
    std::vector<const HartreeFock::ContractedView *> pB(gB.n_components);
    std::vector<const HartreeFock::ContractedView *> pC(gC.n_components);
    std::vector<const HartreeFock::ContractedView *> pD(gD.n_components);
    for (std::size_t a = 0; a < gA.n_components; ++a)
        pA[a] = &basis._basis_functions[gA.first_ao + a];
    for (std::size_t b = 0; b < gB.n_components; ++b)
        pB[b] = &basis._basis_functions[gB.first_ao + b];
    for (std::size_t c = 0; c < gC.n_components; ++c)
        pC[c] = &basis._basis_functions[gC.first_ao + c];
    for (std::size_t d = 0; d < gD.n_components; ++d)
        pD[d] = &basis._basis_functions[gD.first_ao + d];

    HartreeFock::HeadGordonPople::_contracted_eri_block_hoisted_views(
        pA.data(), gA.n_components, pB.data(), gB.n_components,
        pC.data(), gC.n_components, pD.data(), gD.n_components,
        kernel, omega, block);
}

void HartreeFock::HeadGordonPople::_contracted_eri_block_hoisted_views(
    const HartreeFock::ContractedView *const *viewsA, std::size_t nA,
    const HartreeFock::ContractedView *const *viewsB, std::size_t nB,
    const HartreeFock::ContractedView *const *viewsC, std::size_t nC,
    const HartreeFock::ContractedView *const *viewsD, std::size_t nD,
    HartreeFock::ERIKernel kernel,
    double omega,
    double *block)
{
    const std::size_t nCD = nC * nD;

    // One shared norm-free contraction at max AM for the whole quartet (uses
    // component 0 of each shell for the shell data — see invariants (1)/(2)).
    // Contraction is deferred to the first readout(), so this is cheap if the
    // caller only ends up needing a subset.
    HoistedQuartet quartet(*viewsA[0], *viewsB[0], *viewsC[0], *viewsD[0],
                           kernel, omega);

    for (std::size_t a = 0; a < nA; ++a)
    {
        const HartreeFock::ContractedView &cvA = *viewsA[a];
        for (std::size_t b = 0; b < nB; ++b)
        {
            const HartreeFock::ContractedView &cvB = *viewsB[b];
            const int lAx = cvA._cartesian[0], lAy = cvA._cartesian[1], lAz = cvA._cartesian[2];
            const int lBx = cvB._cartesian[0], lBy = cvB._cartesian[1], lBz = cvB._cartesian[2];
            const double normAB = cvA._component_norm * cvB._component_norm;

            for (std::size_t c = 0; c < nC; ++c)
            {
                const HartreeFock::ContractedView &cvC = *viewsC[c];
                for (std::size_t d = 0; d < nD; ++d)
                {
                    const HartreeFock::ContractedView &cvD = *viewsD[d];
                    const int lCx = cvC._cartesian[0], lCy = cvC._cartesian[1], lCz = cvC._cartesian[2];
                    const int lDx = cvD._cartesian[0], lDy = cvD._cartesian[1], lDz = cvD._cartesian[2];
                    const double norm = normAB * cvC._component_norm * cvD._component_norm;

                    block[(a * nB + b) * nCD + (c * nD + d)] = quartet.readout(
                        lAx, lAy, lAz, lBx, lBy, lBz,
                        lCx, lCy, lCz, lDx, lDy, lDz, norm);
                }
            }
        }
    }
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

    // H-10 step A2/A4-2: iterate at *shell-quartet* granularity instead of per
    // Cartesian-AO. Reconstruct the shell grouping from the per-AO shell_pairs
    // list and form the upper triangle of *shell* pairs. Per-component Schwarz
    // screening, symmetry orbit, and the 8-fold store-only scatter run in the
    // inner component loops unchanged. A4-2: the per-component
    // _contracted_eri_elem (which re-ran the per-primitive VRR + (a0|c0)
    // contraction every component) is replaced by one shared HoistedQuartet
    // contraction per shell quartet + a cheap per-component HRR readout. This is
    // NOT bitwise vs the per-AO build for d-shells: the hoist applies
    // _component_norm after HRR while the old path folded it into coeff_product
    // before contraction, so the two round differently at the last FP bit
    // (~1e-15). Gated at tight tolerance by planck-os-block-kernel (hoisted arm)
    // and planck-compute-2e (HGP-vs-OS). Store-only scatter keeps the tensor
    // independent of visitation order.
    std::vector<const HartreeFock::ContractedView *> ao_views;
    const std::vector<HgpShellGroup> groups =
        shell_groups_from_pairs(shell_pairs, nb, ao_views);
    const std::size_t ngroups = groups.size();

    // Bra shell pairs (sa <= sb) flattened for load balance. The bra-ket and AO
    // upper-triangle canonical restrictions are applied per component inside,
    // reproducing the old per-AO upper triangle + canonicalization exactly: the
    // per-component (k,l) >=_lex (i,j) check is the exact equivalent of the old
    // flat-pair q >= p ordering.
    struct GroupPair
    {
        std::size_t a;
        std::size_t b;
    };
    std::vector<GroupPair> group_pairs;
    group_pairs.reserve(ngroups * (ngroups + 1) / 2);
    for (std::size_t sa = 0; sa < ngroups; ++sa)
        for (std::size_t sb = sa; sb < ngroups; ++sb)
            group_pairs.push_back({sa, sb});

    const std::size_t ngp = group_pairs.size();

    // ponytail: HGP is NOT MPI-distributed — every rank builds the full tensor
    // (correct, just replicated work); no allreduce here. Only OS (the default
    // engine) is striped. Same upgrade path as the matching note in rys.cpp:
    // `bra % nranks == rank` + one Mpi::allreduce_inplace on the tensor.
#pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t bra = 0; bra < ngp; ++bra)
    {
        const HgpShellGroup &gA = groups[group_pairs[bra].a];
        const HgpShellGroup &gB = groups[group_pairs[bra].b];

        // Iterate every ket shell pair; the per-component bra-ket canonical
        // check ((k,l) >=_lex (i,j)) is the exact filter, so we must not prune
        // ket shell pairs by their flat index.
        for (std::size_t ket = 0; ket < ngp; ++ket)
        {
            const HgpShellGroup &gC = groups[group_pairs[ket].a];
            const HgpShellGroup &gD = groups[group_pairs[ket].b];

            // A4-2: one shared norm-free (a0|c0) contraction per shell quartet,
            // built lazily on the first surviving component (so a fully screened
            // quartet pays nothing). Each surviving component reads its ERI out
            // via a cheap per-component HRR + norm scaling.
            HoistedQuartet quartet(
                *ao_views[gA.first_ao], *ao_views[gB.first_ao],
                *ao_views[gC.first_ao], *ao_views[gD.first_ao],
                kernel, omega);

            for (std::size_t ca = 0; ca < gA.n_components; ++ca)
            {
                const HartreeFock::ContractedView &cvA = *ao_views[gA.first_ao + ca];
                const std::size_t i = cvA._index;
                const int lAx = cvA._cartesian[0], lAy = cvA._cartesian[1], lAz = cvA._cartesian[2];
                const double normA = cvA._component_norm;

                for (std::size_t cb = 0; cb < gB.n_components; ++cb)
                {
                    const HartreeFock::ContractedView &cvB = *ao_views[gB.first_ao + cb];
                    const std::size_t j = cvB._index;
                    if (j < i) // bra upper triangle: j >= i
                        continue;
                    const int lBx = cvB._cartesian[0], lBy = cvB._cartesian[1], lBz = cvB._cartesian[2];
                    const double normAB = normA * cvB._component_norm;

                    for (std::size_t cc = 0; cc < gC.n_components; ++cc)
                    {
                        const HartreeFock::ContractedView &cvC = *ao_views[gC.first_ao + cc];
                        const std::size_t k = cvC._index;
                        const int lCx = cvC._cartesian[0], lCy = cvC._cartesian[1], lCz = cvC._cartesian[2];
                        const double normABC = normAB * cvC._component_norm;

                        for (std::size_t cd = 0; cd < gD.n_components; ++cd)
                        {
                            const HartreeFock::ContractedView &cvD = *ao_views[gD.first_ao + cd];
                            const std::size_t l = cvD._index;
                            if (l < k) // ket upper triangle: l >= k
                                continue;
                            // bra-ket canonical: (k,l) >=_lex (i,j)
                            if (k < i || (k == i && l < j))
                                continue;

                            const int lDx = cvD._cartesian[0], lDy = cvD._cartesian[1], lDz = cvD._cartesian[2];

                            if (Q[i * nb + j] * Q[k * nb + l] < tol_eri)
                                continue;

                            std::vector<QuartetOrbitElem> orbit;
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

                            const double norm = normABC * cvD._component_norm;
                            const double val = quartet.readout(
                                lAx, lAy, lAz, lBx, lBy, lBz,
                                lCx, lCy, lCz, lDx, lDy, lDz, norm);

                            if (!use_sym)
                            {
                                write_eri_permutations(eri, nb, nb2, nb3, i, j, k, l, val);
                                continue;
                            }

                            for (const auto &elem : orbit)
                                write_eri_permutations(
                                    eri, nb, nb2, nb3,
                                    elem.i, elem.j, elem.k, elem.l,
                                    static_cast<double>(elem.sign) * val);
                        }
                    }
                }
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

// ─── Memory-direct Fock builders (no nb^4 tensor) ────────────────────────────
//
// Same result as _compute_2e_fock / _compute_2e_fock_uhf above, but driven by
// the shared fused loop (fused_fock.h): each canonical quartet is contracted
// straight into G, and the nb^4 tensor is never allocated. The two-phase
// builders above call _compute_2e, which materializes the full tensor on every
// SCF iteration. Only the per-quartet ERI callable differs between engines.
//
// sym_ops (integral symmetry) is delegated to the two-phase builder — see the
// note in fused_fock.h.

namespace
{
    // Schwarz table as an nb x nb Eigen view over the shared flat helper
    // (row-major nb*nb, Q[i*nb+j] == Q[j*nb+i]).
    inline Eigen::MatrixXd hgp_schwarz_matrix(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::size_t nb,
        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
    {
        const std::vector<double> flat =
            HartreeFock::Screening::schwarz_table_hgp(shell_pairs, nb, sym_ops);
        Eigen::MatrixXd Q(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                Q(i, j) = flat[i * nb + j];
        return Q;
    }

    // The HGP per-quartet contracted ERI, in the shape fused_fock_build wants.
    constexpr auto hgp_eri_elem =
        [](const HartreeFock::ShellPair &spAB, const HartreeFock::ShellPair &spCD,
           int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
           int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
           HartreeFock::ERIKernel k, double w)
    {
        return HartreeFock::HeadGordonPople::_contracted_eri_elem(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz,
            lCx, lCy, lCz, lDx, lDy, lDz, k, w);
    };
}

Eigen::MatrixXd HartreeFock::HeadGordonPople::_compute_2e_fock_direct(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &density,
    const std::size_t nbasis,
    const HartreeFock::ERIKernel kernel,
    const double omega,
    const double tol_eri,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    const Eigen::MatrixXd Q = hgp_schwarz_matrix(shell_pairs, nb, sym_ops);

    Eigen::MatrixXd G, Ga_unused, Gb_unused;
    HartreeFock::Integrals::FusedFockDensities dens;
    dens.P = &density;

    HartreeFock::Integrals::fused_fock_build(
        shell_pairs, nb, Q, kernel, omega, tol_eri,
        /*spin_resolved=*/false, dens, G, Ga_unused, Gb_unused, hgp_eri_elem,
        sym_ops);
    return G;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
HartreeFock::HeadGordonPople::_compute_2e_fock_uhf_direct(
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
    const Eigen::MatrixXd Q = hgp_schwarz_matrix(shell_pairs, nb, sym_ops);

    const Eigen::MatrixXd Pt = Pa + Pb;
    Eigen::MatrixXd G_unused, Ga, Gb;
    HartreeFock::Integrals::FusedFockDensities dens;
    dens.Pt = &Pt;
    dens.Pa = &Pa;
    dens.Pb = &Pb;

    HartreeFock::Integrals::fused_fock_build(
        shell_pairs, nb, Q, kernel, omega, tol_eri,
        /*spin_resolved=*/true, dens, G_unused, Ga, Gb, hgp_eri_elem,
        sym_ops);
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
