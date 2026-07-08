#include "ri_eri.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <numbers>
#include <vector>

#include "base/tables.h"
#include "basis/basis.h"        // BasisFunctions::_cartesian_shell_order
#include "lookup/elements.h"
#include "lookup/boys.h"        // HartreeFock::Lookup::boys
#include "integrals/shellpair.h"

namespace
{
    struct ThreeCenterScratch
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
        std::vector<double> data;

        ThreeCenterScratch(
            int lABx, int lABy, int lABz,
            int lCx, int lCy, int lCz,
            int mmax)
        {
            ax_dim = lABx + 1;
            ay_dim = lABy + 1;
            az_dim = lABz + 1;
            cx_dim = lCx + 1;
            cy_dim = lCy + 1;
            cz_dim = lCz + 1;
            m_dim = mmax + 1;

            cz_stride = 1;
            cy_stride = static_cast<std::size_t>(cz_dim) * cz_stride;
            cx_stride = static_cast<std::size_t>(cy_dim) * cy_stride;
            az_stride = static_cast<std::size_t>(cx_dim) * cx_stride;
            ay_stride = static_cast<std::size_t>(az_dim) * az_stride;
            ax_stride = static_cast<std::size_t>(ay_dim) * ay_stride;

            const std::size_t spatial =
                static_cast<std::size_t>(ax_dim) * ay_dim * az_dim *
                cx_dim * cy_dim * cz_dim;
            data.assign(spatial * static_cast<std::size_t>(m_dim), 0.0);
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
            return data[spatial_index(ax, ay, az, cx, cy, cz) *
                            static_cast<std::size_t>(m_dim) +
                        static_cast<std::size_t>(m)];
        }
    };

    struct TwoCenterScratch
    {
        int ax_dim = 0;
        int ay_dim = 0;
        int az_dim = 0;
        int bx_dim = 0;
        int by_dim = 0;
        int bz_dim = 0;
        int m_dim = 0;
        std::size_t ax_stride = 0;
        std::size_t ay_stride = 0;
        std::size_t az_stride = 0;
        std::size_t bx_stride = 0;
        std::size_t by_stride = 0;
        std::size_t bz_stride = 0;
        std::vector<double> data;

        TwoCenterScratch(
            int lAx, int lAy, int lAz,
            int lBx, int lBy, int lBz,
            int mmax)
        {
            ax_dim = lAx + 1;
            ay_dim = lAy + 1;
            az_dim = lAz + 1;
            bx_dim = lBx + 1;
            by_dim = lBy + 1;
            bz_dim = lBz + 1;
            m_dim = mmax + 1;

            bz_stride = 1;
            by_stride = static_cast<std::size_t>(bz_dim) * bz_stride;
            bx_stride = static_cast<std::size_t>(by_dim) * by_stride;
            az_stride = static_cast<std::size_t>(bx_dim) * bx_stride;
            ay_stride = static_cast<std::size_t>(az_dim) * az_stride;
            ax_stride = static_cast<std::size_t>(ay_dim) * ay_stride;

            const std::size_t spatial =
                static_cast<std::size_t>(ax_dim) * ay_dim * az_dim *
                bx_dim * by_dim * bz_dim;
            data.assign(spatial * static_cast<std::size_t>(m_dim), 0.0);
        }

        std::size_t spatial_index(
            int ax, int ay, int az,
            int bx, int by, int bz) const
        {
            return static_cast<std::size_t>(ax) * ax_stride +
                   static_cast<std::size_t>(ay) * ay_stride +
                   static_cast<std::size_t>(az) * az_stride +
                   static_cast<std::size_t>(bx) * bx_stride +
                   static_cast<std::size_t>(by) * by_stride +
                   static_cast<std::size_t>(bz) * bz_stride;
        }

        double &v(
            int ax, int ay, int az,
            int bx, int by, int bz,
            int m)
        {
            return data[spatial_index(ax, ay, az, bx, by, bz) *
                            static_cast<std::size_t>(m_dim) +
                        static_cast<std::size_t>(m)];
        }
    };

    double three_center_hrr_ab(
        const ThreeCenterScratch &scratch,
        const int lAx, const int lAy, const int lAz,
        const int lBx, const int lBy, const int lBz,
        const int lCx, const int lCy, const int lCz,
        const Eigen::Vector3d &AB)
    {
        const int lABx = lAx + lBx;
        const int lABy = lAy + lBy;
        const int lABz = lAz + lBz;

        if (lBx == 0 && lBy == 0 && lBz == 0)
            return scratch.data[scratch.spatial_index(lAx, lAy, lAz, lCx, lCy, lCz) *
                                    static_cast<std::size_t>(scratch.m_dim)];

        std::vector<double> work(
            static_cast<std::size_t>(lABx + 1) *
                static_cast<std::size_t>(lABy + 1) *
                static_cast<std::size_t>(lABz + 1),
            0.0);

        auto idx = [=](int ax, int ay, int az) {
            return (static_cast<std::size_t>(ax) * static_cast<std::size_t>(lABy + 1) +
                    static_cast<std::size_t>(ay)) *
                       static_cast<std::size_t>(lABz + 1) +
                   static_cast<std::size_t>(az);
        };

        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                    work[idx(ax, ay, az)] = scratch.data[
                        scratch.spatial_index(ax, ay, az, lCx, lCy, lCz) *
                        static_cast<std::size_t>(scratch.m_dim)];

        for (int kz = 0; kz < lBz; ++kz)
            for (int ax = 0; ax <= lABx; ++ax)
                for (int ay = 0; ay <= lABy; ++ay)
                    for (int az = 0; az <= lABz - kz - 1; ++az)
                        work[idx(ax, ay, az)] =
                            work[idx(ax, ay, az + 1)] + AB.z() * work[idx(ax, ay, az)];

        for (int ky = 0; ky < lBy; ++ky)
            for (int ax = 0; ax <= lABx; ++ax)
                for (int ay = 0; ay <= lABy - ky - 1; ++ay)
                    for (int az = 0; az <= lAz; ++az)
                        work[idx(ax, ay, az)] =
                            work[idx(ax, ay + 1, az)] + AB.y() * work[idx(ax, ay, az)];

        for (int kx = 0; kx < lBx; ++kx)
            for (int ax = 0; ax <= lABx - kx - 1; ++ax)
                for (int ay = 0; ay <= lAy; ++ay)
                    for (int az = 0; az <= lAz; ++az)
                        work[idx(ax, ay, az)] =
                            work[idx(ax + 1, ay, az)] + AB.x() * work[idx(ax, ay, az)];

        return work[idx(lAx, lAy, lAz)];
    }

    // Single-primitive 2-center ERI (P|Q):
    //
    //     (P|Q) = ∫∫ G_P(r1) [1/r12] G_Q(r2) dr1 dr2
    //
    // For unnormalized primitive Gaussians G_P on center A with exponent α and
    // G_Q on center B with exponent β, with angular components (lAx,lAy,lAz)
    // and (lBx,lBy,lBz) respectively:
    //
    //   T  = αβ/(α+β) · |A−B|²
    //   K  = 2π^(5/2) / (αβ √(α+β))
    //   seed: (s|s)^(m) = K · F_m(T)
    //
    // Then the OS-style VRR on the A-side reads (for axis q ∈ {x,y,z}):
    //
    //   (a+1_q, 0)^(m) = (A−A)_q · (a,0)^(m)
    //                  + (W−A)_q · (a,0)^(m+1)
    //                  + (a_q/(2α)) · [(a−1_q, 0)^(m) − (β/(α+β))(a−1_q, 0)^(m+1)]
    //
    // where W = (α·A + β·B)/(α+β). The first term vanishes because the
    // Gaussian center is A itself: (A−A)_q = 0. So this reduces to the same
    // recurrence shape as the nuclear-attraction VRR with PA→0 and PC→QA = A−W.
    //
    // After the A-side VRR is built, the B-side must retain the full ladder of
    // lower-A intermediates. The second-center recurrence contains the same
    // same-center lowering term as A plus a cross term proportional to the
    // angular momentum already present on A.
    //
    // Reference: Obara & Saika, J. Chem. Phys. 84 (1986) 3963 — eq. (19) of
    // the 2c reduction discussed in §III.C.
    double _2c_eri_primitive(
        const Eigen::Vector3d &A, double alpha,
        const Eigen::Vector3d &B, double beta,
        int lAx, int lAy, int lAz,
        int lBx, int lBy, int lBz)
    {
        const double zeta = alpha + beta;
        const double rho = alpha * beta / zeta;
        const Eigen::Vector3d AB = A - B;
        const double T = rho * AB.squaredNorm();

        // Seed prefactor K = 2π^(5/2) / (αβ √(α+β)). This is the
        // (s|s) ERI scaling for two Gaussians of unit contraction coefficient.
        // π^(5/2) = π² · √π — pre-computed once at function entry rather than
        // constexpr, since std::sqrt is not constexpr until C++26.
        static const double TWO_PI_TO_5_2 =
            2.0 * std::numbers::pi * std::numbers::pi *
            std::sqrt(std::numbers::pi);
        const double K = TWO_PI_TO_5_2 / (alpha * beta * std::sqrt(zeta));

        const int LA = lAx + lAy + lAz;
        const int LB = lBx + lBy + lBz;
        const int LT = LA + LB;

        // Phase 1: A-side VRR. Build V[ax][ay][az][m] over m ∈ [0, LT - axyz].
        // Indexing layout: m is the contiguous (innermost) axis, matching the
        // OS 4-center engine's scratch convention.
        // W = (αA + βB)/(α+β); we only need (W−A) which equals (β/(α+β))(B−A).
        // Materialize the displacement before scaling so editor linters don't
        // trip over Eigen's expression-template operator* here.
        Eigen::Vector3d WmA = B - A;
        WmA *= (beta / zeta);
        const double inv_2alpha = 0.5 / alpha;
        const double rho_over_alpha = rho / alpha; // (β/(α+β)) = ρ/α

        TwoCenterScratch scratch(lAx, lAy, lAz, lBx, lBy, lBz, LT);
        for (int m = 0; m <= LT; ++m)
            scratch.v(0, 0, 0, 0, 0, 0, m) = K * HartreeFock::Lookup::boys(m, T);

        // ── A-VRR x ───────────────────────────────────────────────────────
        // Term (PA)·(a-1,m) vanishes because PA = (A−A) = 0 in 2-center.
        for (int ax = 1; ax <= lAx; ++ax)
        {
            const int mlim = LT - ax;
            for (int m = 0; m <= mlim; ++m)
            {
                scratch.v(ax, 0, 0, 0, 0, 0, m) =
                    WmA.x() * scratch.v(ax - 1, 0, 0, 0, 0, 0, m + 1);
                if (ax > 1)
                    scratch.v(ax, 0, 0, 0, 0, 0, m) +=
                        (ax - 1) * inv_2alpha *
                        (scratch.v(ax - 2, 0, 0, 0, 0, 0, m) -
                         rho_over_alpha * scratch.v(ax - 2, 0, 0, 0, 0, 0, m + 1));
            }
        }

        // ── A-VRR y ───────────────────────────────────────────────────────
        for (int ax = 0; ax <= lAx; ++ax)
            for (int ay = 1; ay <= lAy; ++ay)
            {
                const int mlim = LT - ax - ay;
                if (mlim < 0) continue;
                for (int m = 0; m <= mlim; ++m)
                {
                    scratch.v(ax, ay, 0, 0, 0, 0, m) =
                        WmA.y() * scratch.v(ax, ay - 1, 0, 0, 0, 0, m + 1);
                    if (ay > 1)
                        scratch.v(ax, ay, 0, 0, 0, 0, m) +=
                            (ay - 1) * inv_2alpha *
                            (scratch.v(ax, ay - 2, 0, 0, 0, 0, m) -
                             rho_over_alpha * scratch.v(ax, ay - 2, 0, 0, 0, 0, m + 1));
                }
            }

        // ── A-VRR z ───────────────────────────────────────────────────────
        for (int ax = 0; ax <= lAx; ++ax)
            for (int ay = 0; ay <= lAy; ++ay)
                for (int az = 1; az <= lAz; ++az)
                {
                    const int mlim = LT - ax - ay - az;
                    if (mlim < 0) continue;
                    for (int m = 0; m <= mlim; ++m)
                    {
                        scratch.v(ax, ay, az, 0, 0, 0, m) =
                            WmA.z() * scratch.v(ax, ay, az - 1, 0, 0, 0, m + 1);
                        if (az > 1)
                            scratch.v(ax, ay, az, 0, 0, 0, m) +=
                                (az - 1) * inv_2alpha *
                                (scratch.v(ax, ay, az - 2, 0, 0, 0, m) -
                                 rho_over_alpha * scratch.v(ax, ay, az - 2, 0, 0, 0, m + 1));
                    }
                }

        if (LB == 0)
            return scratch.v(lAx, lAy, lAz, 0, 0, 0, 0);

        // Phase 2: B-side VRR. Unlike the A-only recurrence, this pass needs
        // the full ladder of lower-A intermediates because the OS recurrence
        // contains the cross term proportional to a_q when angular momentum is
        // transferred onto the second center.
        // Same linter-safe pattern as WmA: keep this as a concrete Vector3d
        // before applying the scalar factor.
        Eigen::Vector3d WmB = A - B;
        WmB *= (alpha / zeta);
        const double inv_2beta = 0.5 / beta;
        const double rho_over_beta = rho / beta;
        const double inv_2zeta = 0.5 / zeta;

        // Build B-side angular momentum against every lower-A intermediate so
        // the OS cross term can couple already-raised A quanta into B.
        for (int ax = 0; ax <= lAx; ++ax)
            for (int ay = 0; ay <= lAy; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int bx = 1; bx <= lBx; ++bx)
                    {
                        const int mlim = LT - ax - ay - az - bx;
                        if (mlim < 0) continue;
                        for (int m = 0; m <= mlim; ++m)
                        {
                            scratch.v(ax, ay, az, bx, 0, 0, m) =
                                WmB.x() * scratch.v(ax, ay, az, bx - 1, 0, 0, m + 1);
                            if (bx > 1)
                                scratch.v(ax, ay, az, bx, 0, 0, m) +=
                                    (bx - 1) * inv_2beta *
                                    (scratch.v(ax, ay, az, bx - 2, 0, 0, m) -
                                     rho_over_beta * scratch.v(ax, ay, az, bx - 2, 0, 0, m + 1));
                            if (ax > 0)
                                scratch.v(ax, ay, az, bx, 0, 0, m) +=
                                    ax * inv_2zeta *
                                    scratch.v(ax - 1, ay, az, bx - 1, 0, 0, m + 1);
                        }
                    }

        for (int ax = 0; ax <= lAx; ++ax)
            for (int ay = 0; ay <= lAy; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int bx = 0; bx <= lBx; ++bx)
                        for (int by = 1; by <= lBy; ++by)
                        {
                            const int mlim = LT - ax - ay - az - bx - by;
                            if (mlim < 0) continue;
                            for (int m = 0; m <= mlim; ++m)
                            {
                                scratch.v(ax, ay, az, bx, by, 0, m) =
                                    WmB.y() * scratch.v(ax, ay, az, bx, by - 1, 0, m + 1);
                                if (by > 1)
                                    scratch.v(ax, ay, az, bx, by, 0, m) +=
                                        (by - 1) * inv_2beta *
                                        (scratch.v(ax, ay, az, bx, by - 2, 0, m) -
                                         rho_over_beta * scratch.v(ax, ay, az, bx, by - 2, 0, m + 1));
                                if (ay > 0)
                                    scratch.v(ax, ay, az, bx, by, 0, m) +=
                                        ay * inv_2zeta *
                                        scratch.v(ax, ay - 1, az, bx, by - 1, 0, m + 1);
                            }
                        }

        for (int ax = 0; ax <= lAx; ++ax)
            for (int ay = 0; ay <= lAy; ++ay)
                for (int az = 0; az <= lAz; ++az)
                    for (int bx = 0; bx <= lBx; ++bx)
                        for (int by = 0; by <= lBy; ++by)
                            for (int bz = 1; bz <= lBz; ++bz)
                            {
                                const int mlim = LT - ax - ay - az - bx - by - bz;
                                if (mlim < 0) continue;
                                for (int m = 0; m <= mlim; ++m)
                                {
                                    scratch.v(ax, ay, az, bx, by, bz, m) =
                                        WmB.z() * scratch.v(ax, ay, az, bx, by, bz - 1, m + 1);
                                    if (bz > 1)
                                        scratch.v(ax, ay, az, bx, by, bz, m) +=
                                            (bz - 1) * inv_2beta *
                                            (scratch.v(ax, ay, az, bx, by, bz - 2, m) -
                                             rho_over_beta * scratch.v(ax, ay, az, bx, by, bz - 2, m + 1));
                                    if (az > 0)
                                        scratch.v(ax, ay, az, bx, by, bz, m) +=
                                            az * inv_2zeta *
                                            scratch.v(ax, ay, az - 1, bx, by, bz - 1, m + 1);
                                }
                            }

        return scratch.v(lAx, lAy, lAz, lBx, lBy, lBz, 0);
    }

    // (2n-1)!! with the (-1)!! = 1 convention. Inlined so this TU doesn't
    // need to pull in gaussian.cpp (which carries the orbital-basis loader).
    int dfact(int n)
    {
        if (n <= 0) return 1;
        int r = 1;
        while (n > 0) { r *= n; n -= 2; }
        return r;
    }

    // Cartesian-component normalization 1 / √((2lx-1)!! (2ly-1)!! (2lz-1)!!).
    // Matches BasisFunctions::component_norm with the standard df argument.
    double cartesian_norm(int lx, int ly, int lz)
    {
        return 1.0 / std::sqrt(
            static_cast<double>(dfact(2 * lx - 1) * dfact(2 * ly - 1) * dfact(2 * lz - 1)));
    }

    double _3c_eri_primitive(
        const HartreeFock::PrimitivePair &ppAB,
        const int lAx, const int lAy, const int lAz,
        const int lBx, const int lBy, const int lBz,
        const Eigen::Vector3d &AB,
        const Eigen::Vector3d &centerC, const double gamma,
        const int lCx, const int lCy, const int lCz)
    {
        const int lABx = lAx + lBx;
        const int lABy = lAy + lBy;
        const int lABz = lAz + lBz;
        const int LA = lABx + lABy + lABz;
        const int LC = lCx + lCy + lCz;
        const int LT = LA + LC;

        const double zetaAB = ppAB.zeta;
        const double delta = zetaAB + gamma;
        const double rho = zetaAB * gamma / delta;
        const double inv_2zetaAB = 0.5 / zetaAB;
        const double inv_2gamma = 0.5 / gamma;
        const double inv_2delta = 0.5 / delta;
        const double rho_over_zetaAB = rho / zetaAB;
        const double rho_over_gamma = rho / gamma;

        const Eigen::Vector3d &P = ppAB.center;
        const Eigen::Vector3d PC = P - centerC;

        const double T = rho * PC.squaredNorm();
        const double prefac =
            ppAB.prefactor *
            (2.0 * std::numbers::pi / gamma) *
            std::sqrt(zetaAB / delta);

        Eigen::Vector3d WmP = centerC - P;
        WmP *= (gamma / delta);
        Eigen::Vector3d WmC = P - centerC;
        WmC *= (zetaAB / delta);

        ThreeCenterScratch scratch(lABx, lABy, lABz, lCx, lCy, lCz, LT);
        for (int m = 0; m <= LT; ++m)
            scratch.v(0, 0, 0, 0, 0, 0, m) = prefac * HartreeFock::Lookup::boys(m, T);

        for (int ax = 1; ax <= lABx; ++ax)
        {
            const int mlim = LT - ax;
            for (int m = 0; m <= mlim; ++m)
            {
                scratch.v(ax, 0, 0, 0, 0, 0, m) =
                    ppAB.pA.x() * scratch.v(ax - 1, 0, 0, 0, 0, 0, m) +
                    WmP.x() * scratch.v(ax - 1, 0, 0, 0, 0, 0, m + 1);
                if (ax > 1)
                    scratch.v(ax, 0, 0, 0, 0, 0, m) +=
                        (ax - 1) * inv_2zetaAB *
                        (scratch.v(ax - 2, 0, 0, 0, 0, 0, m) -
                         rho_over_zetaAB * scratch.v(ax - 2, 0, 0, 0, 0, 0, m + 1));
            }
        }

        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 1; ay <= lABy; ++ay)
            {
                const int mlim = LT - ax - ay;
                if (mlim < 0) continue;
                for (int m = 0; m <= mlim; ++m)
                {
                    scratch.v(ax, ay, 0, 0, 0, 0, m) =
                        ppAB.pA.y() * scratch.v(ax, ay - 1, 0, 0, 0, 0, m) +
                        WmP.y() * scratch.v(ax, ay - 1, 0, 0, 0, 0, m + 1);
                    if (ay > 1)
                        scratch.v(ax, ay, 0, 0, 0, 0, m) +=
                            (ay - 1) * inv_2zetaAB *
                            (scratch.v(ax, ay - 2, 0, 0, 0, 0, m) -
                             rho_over_zetaAB * scratch.v(ax, ay - 2, 0, 0, 0, 0, m + 1));
                }
            }

        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 1; az <= lABz; ++az)
                {
                    const int mlim = LT - ax - ay - az;
                    if (mlim < 0) continue;
                    for (int m = 0; m <= mlim; ++m)
                    {
                        scratch.v(ax, ay, az, 0, 0, 0, m) =
                            ppAB.pA.z() * scratch.v(ax, ay, az - 1, 0, 0, 0, m) +
                            WmP.z() * scratch.v(ax, ay, az - 1, 0, 0, 0, m + 1);
                        if (az > 1)
                            scratch.v(ax, ay, az, 0, 0, 0, m) +=
                                (az - 1) * inv_2zetaAB *
                                (scratch.v(ax, ay, az - 2, 0, 0, 0, m) -
                                 rho_over_zetaAB * scratch.v(ax, ay, az - 2, 0, 0, 0, m + 1));
                    }
                }

        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                    for (int cx = 1; cx <= lCx; ++cx)
                    {
                        const int mlim = LT - ax - ay - az - cx;
                        if (mlim < 0) continue;
                        for (int m = 0; m <= mlim; ++m)
                        {
                            scratch.v(ax, ay, az, cx, 0, 0, m) =
                                WmC.x() * scratch.v(ax, ay, az, cx - 1, 0, 0, m + 1);
                            if (cx > 1)
                                scratch.v(ax, ay, az, cx, 0, 0, m) +=
                                    (cx - 1) * inv_2gamma *
                                    (scratch.v(ax, ay, az, cx - 2, 0, 0, m) -
                                     rho_over_gamma * scratch.v(ax, ay, az, cx - 2, 0, 0, m + 1));
                            if (ax > 0)
                                scratch.v(ax, ay, az, cx, 0, 0, m) +=
                                    ax * inv_2delta *
                                    scratch.v(ax - 1, ay, az, cx - 1, 0, 0, m + 1);
                        }
                    }

        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                    for (int cx = 0; cx <= lCx; ++cx)
                        for (int cy = 1; cy <= lCy; ++cy)
                        {
                            const int mlim = LT - ax - ay - az - cx - cy;
                            if (mlim < 0) continue;
                            for (int m = 0; m <= mlim; ++m)
                            {
                                scratch.v(ax, ay, az, cx, cy, 0, m) =
                                    WmC.y() * scratch.v(ax, ay, az, cx, cy - 1, 0, m + 1);
                                if (cy > 1)
                                    scratch.v(ax, ay, az, cx, cy, 0, m) +=
                                        (cy - 1) * inv_2gamma *
                                        (scratch.v(ax, ay, az, cx, cy - 2, 0, m) -
                                         rho_over_gamma * scratch.v(ax, ay, az, cx, cy - 2, 0, m + 1));
                                if (ay > 0)
                                    scratch.v(ax, ay, az, cx, cy, 0, m) +=
                                        ay * inv_2delta *
                                        scratch.v(ax, ay - 1, az, cx, cy - 1, 0, m + 1);
                            }
                        }

        for (int ax = 0; ax <= lABx; ++ax)
            for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                    for (int cx = 0; cx <= lCx; ++cx)
                        for (int cy = 0; cy <= lCy; ++cy)
                            for (int cz = 1; cz <= lCz; ++cz)
                            {
                                const int mlim = LT - ax - ay - az - cx - cy - cz;
                                if (mlim < 0) continue;
                                for (int m = 0; m <= mlim; ++m)
                                {
                                    scratch.v(ax, ay, az, cx, cy, cz, m) =
                                        WmC.z() * scratch.v(ax, ay, az, cx, cy, cz - 1, m + 1);
                                    if (cz > 1)
                                        scratch.v(ax, ay, az, cx, cy, cz, m) +=
                                            (cz - 1) * inv_2gamma *
                                            (scratch.v(ax, ay, az, cx, cy, cz - 2, m) -
                                             rho_over_gamma * scratch.v(ax, ay, az, cx, cy, cz - 2, m + 1));
                                    if (az > 0)
                                        scratch.v(ax, ay, az, cx, cy, cz, m) +=
                                            az * inv_2delta *
                                            scratch.v(ax, ay, az - 1, cx, cy, cz - 1, m + 1);
                                }
                            }

        return three_center_hrr_ab(
            scratch, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, AB);
    }

    Eigen::MatrixXd unpack_transform_repack_3c(
        const Eigen::MatrixXd &j3c_cart,
        const Eigen::MatrixXd &cart_to_sph)
    {
        const std::size_t n_cart = static_cast<std::size_t>(cart_to_sph.cols());
        const std::size_t n_sph = static_cast<std::size_t>(cart_to_sph.rows());
        const std::size_t naux = static_cast<std::size_t>(j3c_cart.cols());
        const std::size_t npair_sph = n_sph * (n_sph + 1) / 2;

        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(npair_sph, naux);
        Eigen::MatrixXd J_cart = Eigen::MatrixXd::Zero(n_cart, n_cart);

        for (std::size_t P = 0; P < naux; ++P)
        {
            std::size_t row = 0;
            for (std::size_t mu = 0; mu < n_cart; ++mu)
                for (std::size_t nu = 0; nu <= mu; ++nu, ++row)
                {
                    const double val = j3c_cart(static_cast<Eigen::Index>(row),
                                                static_cast<Eigen::Index>(P));
                    J_cart(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) = val;
                    J_cart(static_cast<Eigen::Index>(nu), static_cast<Eigen::Index>(mu)) = val;
                }

            const Eigen::MatrixXd J_sph = cart_to_sph * J_cart * cart_to_sph.transpose();
            row = 0;
            for (std::size_t p = 0; p < n_sph; ++p)
                for (std::size_t q = 0; q <= p; ++q, ++row)
                    out(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(P)) =
                        J_sph(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q));
        }

        return out;
    }
} // namespace

namespace HartreeFock::Correlation::RI
{
    std::expected<Eigen::MatrixXd, std::string> compute_2c_eri(const AuxBasis &aux)
    {
        const std::size_t n = aux.nfunctions;
        Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, n);

        if (n == 0)
            return V;

        // Pre-build per-shell Cartesian component orderings to avoid recomputing
        // inside the K-L double loop.
        const std::size_t nshells = aux.shells.size();
        std::vector<std::vector<Eigen::Vector3i>> shell_components(nshells);
        for (std::size_t s = 0; s < nshells; ++s)
        {
            const unsigned L = static_cast<unsigned>(aux.shells[s]._shell);
            shell_components[s] = BasisFunctions::_cartesian_shell_order(L);
        }

        // Symmetric build over shell pairs K ≤ L.
        for (std::size_t K = 0; K < nshells; ++K)
        {
            const auto &shP = aux.shells[K];
            const std::size_t off_K = aux.offsets[K];
            const auto &compP = shell_components[K];

            for (std::size_t L = K; L < nshells; ++L)
            {
                const auto &shQ = aux.shells[L];
                const std::size_t off_L = aux.offsets[L];
                const auto &compQ = shell_components[L];

                // Loop over Cartesian components of each shell.
                for (std::size_t p = 0; p < compP.size(); ++p)
                {
                    const int lAx = compP[p][0], lAy = compP[p][1], lAz = compP[p][2];
                    const double normA = cartesian_norm(lAx, lAy, lAz);

                    for (std::size_t q = 0; q < compQ.size(); ++q)
                    {
                        const int lBx = compQ[q][0], lBy = compQ[q][1], lBz = compQ[q][2];
                        const double normB = cartesian_norm(lBx, lBy, lBz);

                        // Contract over primitives. Each Shell._coefficients
                        // already has the contracted norm folded in (see Norm
                        // Factors gotcha); _normalizations holds the per-prim
                        // angular-dependent norm.
                        double val = 0.0;
                        const Eigen::VectorXd &expA = shP._primitives;
                        const Eigen::VectorXd &coefA = shP._coefficients;
                        const Eigen::VectorXd &normPA = shP._normalizations;
                        const Eigen::VectorXd &expB = shQ._primitives;
                        const Eigen::VectorXd &coefB = shQ._coefficients;
                        const Eigen::VectorXd &normPB = shQ._normalizations;

                        for (Eigen::Index ia = 0; ia < expA.size(); ++ia)
                        {
                            for (Eigen::Index ib = 0; ib < expB.size(); ++ib)
                            {
                                const double prim = _2c_eri_primitive(
                                    shP._center, expA(ia),
                                    shQ._center, expB(ib),
                                    lAx, lAy, lAz,
                                    lBx, lBy, lBz);
                                val += coefA(ia) * coefB(ib) *
                                       normPA(ia) * normPB(ib) *
                                       prim;
                            }
                        }
                        val *= normA * normB;

                        V(off_K + p, off_L + q) = val;
                        if (K != L)
                            V(off_L + q, off_K + p) = val;
                    }
                }
                // K == L: also need to mirror within the diagonal block
                // since the inner loop only wrote (p,q) not (q,p).
                if (K == L)
                {
                    const std::size_t sz = compP.size();
                    for (std::size_t p = 0; p < sz; ++p)
                        for (std::size_t q = p + 1; q < sz; ++q)
                            V(off_K + q, off_K + p) = V(off_K + p, off_K + q);
                }
            }
        }

        return V;
    }

    std::expected<MetricFactorization, std::string> factorize_2c_metric(
        const Eigen::MatrixXd &metric,
        double lindep)
    {
        if (metric.rows() != metric.cols())
            return std::unexpected("factorize_2c_metric: metric must be square.");
        if (!(lindep > 0.0))
            return std::unexpected("factorize_2c_metric: lindep must be positive.");

        if (metric.size() == 0)
            return MetricFactorization{};

        const Eigen::MatrixXd metric_asym = metric - metric.transpose();
        const Eigen::MatrixXd metric_abs = metric.cwiseAbs();
        const double asym = metric_asym.cwiseAbs().maxCoeff();
        const double scale = std::max(1.0, metric_abs.maxCoeff());
        if (asym > 1e-10 * scale)
            return std::unexpected("factorize_2c_metric: metric is not symmetric.");

        Eigen::LLT<Eigen::MatrixXd> llt(metric);
        if (llt.info() == Eigen::Success)
        {
            MetricFactorization out;
            out.method = MetricFactorization::Method::Cholesky;
            out.transform = llt.matrixL();
            return out;
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(metric);
        if (es.info() != Eigen::Success)
            return std::unexpected("factorize_2c_metric: eigenvalue decomposition failed.");

        const auto &evals = es.eigenvalues();
        const auto &evecs = es.eigenvectors();
        int nkeep = 0;
        for (Eigen::Index i = 0; i < evals.size(); ++i)
            if (evals(i) > lindep)
                ++nkeep;
        if (nkeep == 0)
            return std::unexpected("factorize_2c_metric: no auxiliary modes survived lindep threshold.");

        MetricFactorization out;
        out.method = MetricFactorization::Method::Eigen;
        out.transform.resize(nkeep, metric.rows());
        out.eigenvalues_kept.resize(nkeep);
        out.kept_indices.resize(nkeep);

        int row = 0;
        for (Eigen::Index i = 0; i < evals.size(); ++i)
        {
            if (evals(i) <= lindep)
                continue;
            out.eigenvalues_kept(row) = evals(i);
            out.kept_indices(row) = static_cast<int>(i);
            out.transform.row(row) = evecs.col(i).transpose() / std::sqrt(evals(i));
            ++row;
        }

        return out;
    }

    std::expected<void, std::string> ensure_ri_metric_ready(
        HartreeFock::Calculator &calculator)
    {
        const auto &opts = calculator._mp2;
        if (!opts.use_ri)
            return std::unexpected("ensure_ri_metric_ready: MP2 RI is disabled.");
        if (opts.ri_basis_name.empty())
            return std::unexpected("ensure_ri_metric_ready: mp2_ri_basis must be set when mp2_use_ri is true.");

        if (!calculator._ri_aux_basis)
        {
            const std::string file_name = opts.ri_basis_path + "/" + opts.ri_basis_name;
            auto aux_res = HartreeFock::BasisFunctions::read_ri_basis(
                file_name, calculator._molecule);
            if (!aux_res)
                return std::unexpected("ensure_ri_metric_ready: " + aux_res.error());
            calculator._ri_aux_basis =
                std::make_shared<HartreeFock::AuxBasis>(std::move(*aux_res));
        }

        if (calculator._ri_j2c.rows() == 0 || calculator._ri_j2c.cols() == 0)
        {
            auto metric_res = compute_2c_eri(*calculator._ri_aux_basis);
            if (!metric_res)
                return std::unexpected("ensure_ri_metric_ready: " + metric_res.error());
            calculator._ri_j2c = std::move(*metric_res);
        }

        if (!calculator._ri_metric_factor)
        {
            auto fac_res = factorize_2c_metric(calculator._ri_j2c, opts.ri_lindep);
            if (!fac_res)
                return std::unexpected("ensure_ri_metric_ready: " + fac_res.error());
            calculator._ri_metric_factor =
                std::make_shared<MetricFactorization>(std::move(*fac_res));
        }

        return {};
    }

    std::expected<Eigen::MatrixXd, std::string> compute_3c_eri(
        const HartreeFock::Calculator &calculator)
    {
        if (!calculator._mp2.use_ri)
            return std::unexpected("compute_3c_eri: MP2 RI is disabled.");
        if (!calculator._ri_aux_basis)
            return std::unexpected("compute_3c_eri: RI auxiliary basis is not loaded.");

        const std::size_t n_cart = calculator._shells.nbasis();
        const auto shell_pairs = build_shellpairs(calculator._shells);
        Eigen::MatrixXd j3c_cart(
            static_cast<Eigen::Index>(n_cart * (n_cart + 1) / 2),
            static_cast<Eigen::Index>(calculator._ri_aux_basis->nfunctions));
        j3c_cart.setZero();

        std::size_t aux_col = 0;
        for (std::size_t shell_idx = 0; shell_idx < calculator._ri_aux_basis->shells.size(); ++shell_idx)
        {
            const auto &shellC = calculator._ri_aux_basis->shells[shell_idx];
            const unsigned L = static_cast<unsigned>(shellC._shell);
            const auto components = HartreeFock::BasisFunctions::_cartesian_shell_order(L);

            for (const auto &amC : components)
            {
                const int lCx = amC[0], lCy = amC[1], lCz = amC[2];
                const double normC = cartesian_norm(lCx, lCy, lCz);

                for (const auto &spAB : shell_pairs)
                {
                    const std::size_t mu = spAB.A._index;
                    const std::size_t nu = spAB.B._index;
                    const std::size_t row = pair_index(nu, mu);
                    double value = 0.0;
                    for (const auto &ppAB : spAB.primitive_pairs)
                    {
                        for (Eigen::Index ic = 0; ic < shellC._primitives.size(); ++ic)
                        {
                            value += ppAB.coeff_product *
                                     shellC._coefficients(ic) *
                                     shellC._normalizations(ic) *
                                     normC *
                                     _3c_eri_primitive(
                                         ppAB,
                                         spAB.A._cartesian[0], spAB.A._cartesian[1], spAB.A._cartesian[2],
                                         spAB.B._cartesian[0], spAB.B._cartesian[1], spAB.B._cartesian[2],
                                         spAB.R,
                                         shellC._center, shellC._primitives(ic),
                                         lCx, lCy, lCz);
                        }
                    }
                    j3c_cart(static_cast<Eigen::Index>(row),
                             static_cast<Eigen::Index>(aux_col)) = value;
                }

                ++aux_col;
            }
        }

        if (!calculator._shells._spherical)
            return j3c_cart;
        if (calculator._shells._cart_to_sph.size() == 0)
            return std::unexpected("compute_3c_eri: spherical basis requested but cart_to_sph is empty.");
        return unpack_transform_repack_3c(j3c_cart, calculator._shells._cart_to_sph);
    }

    std::expected<void, std::string> ensure_ri_3c_ready(
        HartreeFock::Calculator &calculator)
    {
        auto metric_res = ensure_ri_metric_ready(calculator);
        if (!metric_res)
            return std::unexpected(metric_res.error());

        const std::size_t n_work = calculator.working_nbasis();
        const std::size_t npair = n_work * (n_work + 1) / 2;
        const std::size_t naux = calculator._ri_aux_basis ? calculator._ri_aux_basis->nfunctions : 0;
        if (calculator._ri_j3c.rows() == static_cast<Eigen::Index>(npair) &&
            calculator._ri_j3c.cols() == static_cast<Eigen::Index>(naux))
            return {};

        auto j3c_res = compute_3c_eri(calculator);
        if (!j3c_res)
            return std::unexpected("ensure_ri_3c_ready: " + j3c_res.error());
        calculator._ri_j3c = std::move(*j3c_res);
        return {};
    }

    Eigen::MatrixXd build_ri_pair_factors(const HartreeFock::Calculator &calculator)
    {
        const Eigen::MatrixXd &j3c = calculator._ri_j3c;
        const auto &metric = *calculator._ri_metric_factor;

        if (metric.method == MetricFactorization::Method::Eigen)
            return j3c * metric.transform.transpose();

        const Eigen::MatrixXd &L = metric.transform;
        Eigen::MatrixXd pair_factors_t = j3c.transpose();
        pair_factors_t =
            L.triangularView<Eigen::Lower>().solve(pair_factors_t);
        return pair_factors_t.transpose();
    }

    Eigen::MatrixXd build_ri_mo_block(
        const Eigen::MatrixXd &pair_factors,
        const Eigen::MatrixXd &C_row,
        const Eigen::MatrixXd &C_col)
    {
        const int nrow = static_cast<int>(C_row.cols());
        const int ncol = static_cast<int>(C_col.cols());
        const int nfit = static_cast<int>(pair_factors.cols());
        const int npq = nrow * ncol;

        Eigen::MatrixXd b_pq = Eigen::MatrixXd::Zero(npq, nfit);
        std::size_t pair_row = 0;
        for (Eigen::Index mu = 0; mu < C_row.rows(); ++mu)
        {
            const Eigen::RowVectorXd row_mu = C_row.row(mu);
            const Eigen::RowVectorXd col_mu = C_col.row(mu);
            for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pair_row)
            {
                const Eigen::RowVectorXd factors =
                    pair_factors.row(static_cast<Eigen::Index>(pair_row));
                const Eigen::RowVectorXd row_nu = C_row.row(nu);
                const Eigen::RowVectorXd col_nu = C_col.row(nu);

                int ia = 0;
                for (int i = 0; i < nrow; ++i)
                {
                    const double cmi = row_mu(i);
                    const double cni = row_nu(i);
                    for (int a = 0; a < ncol; ++a, ++ia)
                    {
                        double weight = cmi * col_nu(a);
                        if (mu != nu)
                            weight += cni * col_mu(a);
                        if (weight != 0.0)
                            b_pq.row(ia).noalias() += weight * factors;
                    }
                }
            }
        }
        return b_pq;
    }
} // namespace HartreeFock::Correlation::RI
