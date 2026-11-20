#include "ri_eri.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <numbers>
#include <vector>

#include "base/mpi_env.h"      // RI under MPI: docs/RI_MPI_SCOPE.md (M1)
#include "base/tables.h"
#include "basis/basis.h"        // BasisFunctions::_cartesian_shell_order
#include "basis/spherical.h"    // BasisFunctions::cart_to_sph_block (spherical aux)
#include "lookup/elements.h"
#include "lookup/boys.h"        // HartreeFock::Lookup::boys
#include "integrals/shellpair.h"

namespace
{
    // RI under MPI (docs/RI_MPI_SCOPE.md, M1): every RI contraction is a sum
    // over aux functions (or AO pairs), so each rank sums the indices it owns
    // (HartreeFock::Mpi::Stripe) and one allreduce completes it. The data stay
    // replicated. Serial builds own every index and the reduce is a no-op,
    // so the serial path is unchanged. Every rank must reach every reduce:
    // never call these from a rank-dependent branch or an OpenMP region.
    void sum_over_ranks(Eigen::MatrixXd &M)
    {
        HartreeFock::Mpi::allreduce_inplace(M.data(), static_cast<std::size_t>(M.size()));
    }
    void sum_over_ranks(Eigen::RowVectorXd &v)
    {
        HartreeFock::Mpi::allreduce_inplace(v.data(), static_cast<std::size_t>(v.size()));
    }

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
        // One grow-only buffer per thread, reused across primitive calls: the
        // per-call heap vector was ~20 % of the RI derivative loop (G6).
        // assign() still zero-fills, so the values are unchanged. Not
        // reentrant, and nothing calls it recursively.
        std::vector<double> &data = buffer();
        static std::vector<double> &buffer()
        {
            thread_local std::vector<double> b;
            return b;
        }

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
        std::vector<double> &data = buffer(); // as ThreeCenterScratch
        static std::vector<double> &buffer()
        {
            thread_local std::vector<double> b;
            return b;
        }

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
        double F[4 * MAX_L + 1]; // one vector Boys call, bitwise = per-order loop
        HartreeFock::Lookup::boys_vec(T, std::span<double>(F, LT + 1));
        for (int m = 0; m <= LT; ++m)
            scratch.v(0, 0, 0, 0, 0, 0, m) = K * F[m];

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
        double F[4 * MAX_L + 1]; // one vector Boys call, bitwise = per-order loop
        HartreeFock::Lookup::boys_vec(T, std::span<double>(F, LT + 1));
        for (int m = 0; m <= LT; ++m)
            scratch.v(0, 0, 0, 0, 0, 0, m) = prefac * F[m];

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

        // Symmetric build over shell pairs K ≤ L. Iteration K writes blocks
        // (K, L≥K) and their mirrors (L, K): disjoint across K, so threads
        // never share an element and the result is thread-count invariant.
#pragma omp parallel for schedule(dynamic)
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

        if (aux.cart_to_sph.size())
            return Eigen::MatrixXd(aux.cart_to_sph * V * aux.cart_to_sph.transpose());
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

    // Single invalidation point for every RI cache. All four (aux basis,
    // 2-center metric, its factorization, 3-center tensor) depend on the atom
    // positions, so when _molecule._standard no longer matches the geometry the
    // caches were built against, clear them all and restamp. ensure_ri_3c_ready
    // calls ensure_ri_metric_ready first, so placing this at the top of the
    // metric path covers both entry points. In the common case (geometry
    // unchanged) it is a cheap matrix compare and a no-op.
    void ri_invalidate_if_geometry_moved(HartreeFock::Calculator &calculator)
    {
        const Eigen::MatrixXd &geom = calculator._molecule._standard;
        const Eigen::MatrixXd &stamp = calculator._ri_cache_geometry;
        const bool same = stamp.rows() == geom.rows() &&
                          stamp.cols() == geom.cols() && stamp == geom;
        if (same)
            return;

        calculator._ri_aux_basis.reset();
        calculator._ri_j2c.resize(0, 0);
        calculator._ri_metric_factor.reset();
        calculator._ri_j3c.resize(0, 0);
        calculator._ri_pair_factors.resize(0, 0);
        calculator._ri_cache_geometry = geom;
    }

    std::expected<void, std::string> ensure_ri_metric_ready(
        HartreeFock::Calculator &calculator)
    {
        const auto &opts = calculator._mp2;
        // Two independent consumers can want the fitted cache: RI-MP2
        // (mp2_use_ri) and the RI-JK SCF/KS Fock build (scf_ri_jk). They share
        // one auxiliary basis, so either one is sufficient reason to prepare it.
        if (!opts.use_ri && !calculator._scf._ri_jk)
            return std::unexpected("ensure_ri_metric_ready: RI is disabled (set mp2_use_ri or scf_ri_jk).");
        if (opts.ri_basis_name.empty())
            return std::unexpected("ensure_ri_metric_ready: mp2_ri_basis must be set when mp2_use_ri or scf_ri_jk is true.");

        ri_invalidate_if_geometry_moved(calculator);

        if (!calculator._ri_aux_basis)
        {
            const std::string file_name = opts.ri_basis_path + "/" + opts.ri_basis_name;
            auto aux_res = HartreeFock::BasisFunctions::read_ri_basis(
                file_name, calculator._molecule);
            if (!aux_res)
                return std::unexpected("ensure_ri_metric_ready: " + aux_res.error());
            // The fitted aux follows the orbital basis (docs/RI_SPHERICAL_AUX.md).
            // The fit is invariant to the row scale of this map, so the bare
            // per-shell blocks serve; only their span (pure harmonics) matters.
            if (calculator._shells._spherical)
            {
                std::size_t n_sph = 0;
                for (const auto &sh : aux_res->shells)
                    n_sph += 2 * static_cast<std::size_t>(sh._shell) + 1;
                Eigen::MatrixXd T = Eigen::MatrixXd::Zero(
                    static_cast<Eigen::Index>(n_sph), static_cast<Eigen::Index>(aux_res->nfunctions));
                Eigen::Index row = 0;
                for (std::size_t K = 0; K < aux_res->shells.size(); ++K)
                {
                    const int L = static_cast<int>(aux_res->shells[K]._shell);
                    auto block = BasisFunctions::cart_to_sph_block(L);
                    if (!block)
                        return std::unexpected("ensure_ri_metric_ready: aux shell: " + block.error());
                    T.block(row, static_cast<Eigen::Index>(aux_res->offsets[K]),
                            block->rows(), block->cols()) = *block;
                    row += block->rows();
                }
                aux_res->cart_to_sph = std::move(T);
            }
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
        if (!calculator._mp2.use_ri && !calculator._scf._ri_jk)
            return std::unexpected("compute_3c_eri: RI is disabled (set mp2_use_ri or scf_ri_jk).");
        if (!calculator._ri_aux_basis)
            return std::unexpected("compute_3c_eri: RI auxiliary basis is not loaded.");

        const std::size_t n_cart = calculator._shells.nbasis();
        const auto shell_pairs = build_shellpairs(calculator._shells);
        Eigen::MatrixXd j3c_cart(
            static_cast<Eigen::Index>(n_cart * (n_cart + 1) / 2),
            static_cast<Eigen::Index>(calculator._ri_aux_basis->nfunctions));
        j3c_cart.setZero();

        // Each aux shell fills its own columns: threads never share an element.
        // Under MPI each rank fills its own shells and the sum below completes
        // the tensor (the other columns are still zero).
        const auto &aux = *calculator._ri_aux_basis;
        const HartreeFock::Mpi::Stripe stripe;
#pragma omp parallel for schedule(dynamic)
        for (std::size_t shell_idx = 0; shell_idx < aux.shells.size(); ++shell_idx)
        {
            if (!stripe.owns(shell_idx))
                continue;
            const auto &shellC = aux.shells[shell_idx];
            std::size_t aux_col = aux.offsets[shell_idx];
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
        sum_over_ranks(j3c_cart);

        const Eigen::MatrixXd &T_aux = calculator._ri_aux_basis->cart_to_sph;
        if (!calculator._shells._spherical)
            return T_aux.size() ? Eigen::MatrixXd(j3c_cart * T_aux.transpose()) : j3c_cart;
        if (calculator._shells._cart_to_sph.size() == 0)
            return std::unexpected("compute_3c_eri: spherical basis requested but cart_to_sph is empty.");
        Eigen::MatrixXd j3c = unpack_transform_repack_3c(j3c_cart, calculator._shells._cart_to_sph);
        return T_aux.size() ? Eigen::MatrixXd(j3c * T_aux.transpose()) : j3c;
    }

    std::array<double, 9> compute_3c_deriv_elem(
        const HartreeFock::ShellPair &spAB,
        int lAx, int lAy, int lAz,
        int lBx, int lBy, int lBz,
        const HartreeFock::Shell &shellC,
        int lCx, int lCy, int lCz)
    {
        std::array<double, 9> result{}; // zero-initialized

        // Which primitive exponent (if any) weights the contraction by 2ζ for a
        // derivative raise term: None (plain, the lower term), or the μ / ν / aux
        // primitive. The μ and ν exponents (alpha, beta) are per-pair; the aux
        // exponent varies per aux primitive, so its weight lives inside the ic
        // loop.
        enum class Weight { None, A, B, C };

        // Contracted (μν|Q) at explicit Cartesian momenta. Folds the aux
        // Cartesian norm (normC) the energy loop in compute_3c_eri applies; normC
        // depends only on (cx,cy,cz). The μ/ν orbital norms are NOT applied at
        // this level (they live in coeff_product / the OS primitive convention),
        // so the 2ζ raise acts on the bare angular part — exactly the
        // translational derivative identity. 3-center analog of the 4-center
        // nceri / wceri_{A,B,C} helpers.
        // The aux basis function's Cartesian normalization is fixed at its
        // ORIGINAL momenta (lCx,lCy,lCz) — it is the normalization of the
        // contracted d/f/… function, not of the raised/lowered angular part the
        // derivative recurrence walks through. Recomputing it from the raised
        // (cx,cy,cz) inside contract is wrong for the aux raise/lower (e.g. a
        // d-aux raised to f uses cartesian_norm(3,0,0) instead of the correct
        // cartesian_norm(2,0,0)) — it silently broke the Σ_center=0 identity for
        // p/d-aux elements. Fix it here, once.
        const double normC = cartesian_norm(lCx, lCy, lCz);

        auto contract = [&](int ax, int ay, int az,
                            int bx, int by, int bz,
                            int cx, int cy, int cz,
                            Weight w) -> double
        {
            double value = 0.0;
            for (const auto &ppAB : spAB.primitive_pairs)
            {
                const double wAB = (w == Weight::A)   ? (2.0 * ppAB.alpha)
                                   : (w == Weight::B) ? (2.0 * ppAB.beta)
                                                      : 1.0;
                for (Eigen::Index ic = 0; ic < shellC._primitives.size(); ++ic)
                {
                    const double wC = (w == Weight::C)
                                          ? (2.0 * shellC._primitives(ic))
                                          : 1.0;
                    value += wAB * wC * ppAB.coeff_product *
                             shellC._coefficients(ic) *
                             shellC._normalizations(ic) *
                             normC *
                             _3c_eri_primitive(
                                 ppAB, ax, ay, az, bx, by, bz, spAB.R,
                                 shellC._center, shellC._primitives(ic),
                                 cx, cy, cz);
                }
            }
            return value;
        };

        for (int q = 0; q < 3; ++q)
        {
            // Centre μ (A): d/dA_q = 2α·(lA+ê_q) − lAq·(lA−ê_q).
            {
                const int lAq = (q == 0 ? lAx : q == 1 ? lAy : lAz);
                double d = contract(lAx + (q == 0), lAy + (q == 1), lAz + (q == 2),
                                    lBx, lBy, lBz, lCx, lCy, lCz, Weight::A);
                if (lAq > 0)
                    d -= static_cast<double>(lAq) *
                         contract(lAx - (q == 0), lAy - (q == 1), lAz - (q == 2),
                                  lBx, lBy, lBz, lCx, lCy, lCz, Weight::None);
                result[0 * 3 + q] = d;
            }
            // Centre ν (B): d/dB_q = 2β·(lB+ê_q) − lBq·(lB−ê_q).
            {
                const int lBq = (q == 0 ? lBx : q == 1 ? lBy : lBz);
                double d = contract(lAx, lAy, lAz,
                                    lBx + (q == 0), lBy + (q == 1), lBz + (q == 2),
                                    lCx, lCy, lCz, Weight::B);
                if (lBq > 0)
                    d -= static_cast<double>(lBq) *
                         contract(lAx, lAy, lAz,
                                  lBx - (q == 0), lBy - (q == 1), lBz - (q == 2),
                                  lCx, lCy, lCz, Weight::None);
                result[1 * 3 + q] = d;
            }
            // Centre aux (C): d/dC_q = 2γ·(lC+ê_q) − lCq·(lC−ê_q).
            {
                const int lCq = (q == 0 ? lCx : q == 1 ? lCy : lCz);
                double d = contract(lAx, lAy, lAz, lBx, lBy, lBz,
                                    lCx + (q == 0), lCy + (q == 1), lCz + (q == 2),
                                    Weight::C);
                if (lCq > 0)
                    d -= static_cast<double>(lCq) *
                         contract(lAx, lAy, lAz, lBx, lBy, lBz,
                                  lCx - (q == 0), lCy - (q == 1), lCz - (q == 2),
                                  Weight::None);
                result[2 * 3 + q] = d;
            }
        }

        return result;
    }

    namespace
    {
        // The one loop over every 3-center derivative element (μν|Q)': each aux
        // Cartesian component × each AO pair (μ≥ν). fn(K, row, aux_col, atomA,
        // atomB, atomC, d) receives the aux shell K, the packed pair row, the
        // aux column, the atoms the three legs sit on and the 9 components
        // ([leg][axis]). compute_3c_eri_deriv scatters through it;
        // build_ri_two_electron_gradient contracts through it without storing
        // anything.
        //
        // Threaded over aux shells: fn runs concurrently for different K, so it
        // must write only K-owned storage (aux_col belongs to K; a reduction
        // keeps one partial per K and sums them in K order, which keeps the
        // result independent of the thread count).
        // Only shells stripe.owns(K) are visited: a caller passing the rank's
        // Stripe must sum its result over ranks; Stripe{0, 1} visits all.
        template <typename Fn>
        std::expected<void, std::string> for_each_3c_deriv(
            const HartreeFock::Calculator &calculator, const HartreeFock::Mpi::Stripe &stripe, Fn &&fn)
        {
            if (!calculator._ri_aux_basis)
                return std::unexpected("RI 3-center derivative: RI auxiliary basis is not loaded.");

            const auto shell_pairs = build_shellpairs(calculator._shells);
            const auto &aux = *calculator._ri_aux_basis;
#pragma omp parallel for schedule(dynamic)
            for (std::size_t K = 0; K < aux.shells.size(); ++K)
            {
                if (!stripe.owns(K))
                    continue;
                const auto &shellC = aux.shells[K];
                std::size_t aux_col = aux.offsets[K];
                const auto components = HartreeFock::BasisFunctions::_cartesian_shell_order(
                    static_cast<unsigned>(shellC._shell));
                const std::size_t atomC = shellC._atom_index;
                for (const auto &amC : components)
                {
                    for (const auto &spAB : shell_pairs)
                    {
                        const std::size_t mu = spAB.A._index;
                        const std::size_t nu = spAB.B._index;
                        const std::size_t hi = std::max(mu, nu), lo = std::min(mu, nu);
                        const auto d = compute_3c_deriv_elem(
                            spAB,
                            spAB.A._cartesian[0], spAB.A._cartesian[1], spAB.A._cartesian[2],
                            spAB.B._cartesian[0], spAB.B._cartesian[1], spAB.B._cartesian[2],
                            shellC, amC[0], amC[1], amC[2]);
                        fn(K, static_cast<Eigen::Index>(hi * (hi + 1) / 2 + lo),
                           static_cast<Eigen::Index>(aux_col),
                           spAB.A._shell->_atom_index, spAB.B._shell->_atom_index, atomC, d);
                    }
                    ++aux_col;
                }
            }
            return {};
        }

        // The one loop over every 2-center metric derivative element (P|Q)'.
        // fn(K, r, c, atomP, atomQ, d), d = [center][axis]; row r belongs to
        // aux shell K. Threaded over K with the same contract as
        // for_each_3c_deriv: fn writes only K-owned storage.
        template <typename Fn>
        std::expected<void, std::string> for_each_2c_deriv(
            const HartreeFock::Calculator &calculator, const HartreeFock::Mpi::Stripe &stripe, Fn &&fn)
        {
            if (!calculator._ri_aux_basis)
                return std::unexpected("RI 2-center derivative: RI auxiliary basis is not loaded.");
            // Cartesian components on both legs, even for a spherical aux basis:
            // callers lift to the fitted space (build_ri_two_electron_gradient).
            const HartreeFock::AuxBasis &aux = *calculator._ri_aux_basis;

            const std::size_t nshells = aux.shells.size();
#pragma omp parallel for schedule(dynamic)
            for (std::size_t K = 0; K < nshells; ++K)
            {
                if (!stripe.owns(K))
                    continue;
                const auto &shP = aux.shells[K];
                const auto compP = HartreeFock::BasisFunctions::_cartesian_shell_order(
                    static_cast<unsigned>(shP._shell));
                for (std::size_t Lsh = 0; Lsh < nshells; ++Lsh)
                {
                    const auto &shQ = aux.shells[Lsh];
                    const auto compQ = HartreeFock::BasisFunctions::_cartesian_shell_order(
                        static_cast<unsigned>(shQ._shell));
                    for (std::size_t p = 0; p < compP.size(); ++p)
                        for (std::size_t q = 0; q < compQ.size(); ++q)
                            fn(K, static_cast<Eigen::Index>(aux.offsets[K] + p),
                               static_cast<Eigen::Index>(aux.offsets[Lsh] + q),
                               shP._atom_index, shQ._atom_index,
                               compute_2c_deriv_elem(
                                   shP, compP[p][0], compP[p][1], compP[p][2],
                                   shQ, compQ[q][0], compQ[q][1], compQ[q][2]));
                }
            }
            return {};
        }
    }

    std::expected<std::vector<Eigen::MatrixXd>, std::string>
    compute_3c_eri_deriv(const HartreeFock::Calculator &calculator)
    {
        if (!calculator._ri_aux_basis)
            return std::unexpected("compute_3c_eri_deriv: RI auxiliary basis is not loaded.");
        const std::size_t n_cart = calculator._shells.nbasis();
        std::vector<Eigen::MatrixXd> dJ(
            calculator._molecule.natoms * 3,
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_cart * (n_cart + 1) / 2),
                                  static_cast<Eigen::Index>(calculator._ri_aux_basis->nfunctions)));
        // Scatter the 9 components to the atoms their legs sit on (atoms may
        // coincide; += accumulates, so translational invariance holds).
        auto ok = for_each_3c_deriv(
            calculator, HartreeFock::Mpi::Stripe{0, 1}, // whole tensor on every rank
            [&](std::size_t, Eigen::Index row, Eigen::Index col, std::size_t atomA,
                std::size_t atomB, std::size_t atomC, const std::array<double, 9> &d)
            {
                for (int q = 0; q < 3; ++q)
                {
                    dJ[atomA * 3 + q](row, col) += d[0 * 3 + q];
                    dJ[atomB * 3 + q](row, col) += d[1 * 3 + q];
                    dJ[atomC * 3 + q](row, col) += d[2 * 3 + q];
                }
            });
        if (!ok)
            return std::unexpected("compute_3c_eri_deriv: " + ok.error());
        return dJ;
    }

    std::array<double, 6> compute_2c_deriv_elem(
        const HartreeFock::Shell &shellP, int lPx, int lPy, int lPz,
        const HartreeFock::Shell &shellQ, int lQx, int lQy, int lQz)
    {
        std::array<double, 6> result{};

        enum class Weight { None, P, Q };

        // Both P and Q Cartesian norms are fixed at their ORIGINAL momenta — the
        // contracted aux functions' normalizations, not the raised angular parts
        // the derivative recurrence walks (the RG1a.3 normC lesson, on both legs).
        const double normP = cartesian_norm(lPx, lPy, lPz);
        const double normQ = cartesian_norm(lQx, lQy, lQz);

        auto contract = [&](int px, int py, int pz,
                            int qx, int qy, int qz,
                            Weight w) -> double
        {
            double value = 0.0;
            for (Eigen::Index ip = 0; ip < shellP._primitives.size(); ++ip)
            {
                const double wP = (w == Weight::P) ? (2.0 * shellP._primitives(ip)) : 1.0;
                for (Eigen::Index iq = 0; iq < shellQ._primitives.size(); ++iq)
                {
                    const double wQ = (w == Weight::Q) ? (2.0 * shellQ._primitives(iq)) : 1.0;
                    value += wP * wQ *
                             shellP._coefficients(ip) * shellQ._coefficients(iq) *
                             shellP._normalizations(ip) * shellQ._normalizations(iq) *
                             normP * normQ *
                             _2c_eri_primitive(
                                 shellP._center, shellP._primitives(ip),
                                 shellQ._center, shellQ._primitives(iq),
                                 px, py, pz, qx, qy, qz);
                }
            }
            return value;
        };

        for (int a = 0; a < 3; ++a)
        {
            // Centre P: d/dP_a = 2ζ_P·(lP+ê_a) − lPa·(lP−ê_a).
            {
                const int lPa = (a == 0 ? lPx : a == 1 ? lPy : lPz);
                double d = contract(lPx + (a == 0), lPy + (a == 1), lPz + (a == 2),
                                    lQx, lQy, lQz, Weight::P);
                if (lPa > 0)
                    d -= static_cast<double>(lPa) *
                         contract(lPx - (a == 0), lPy - (a == 1), lPz - (a == 2),
                                  lQx, lQy, lQz, Weight::None);
                result[0 * 3 + a] = d;
            }
            // Centre Q: d/dQ_a = 2ζ_Q·(lQ+ê_a) − lQa·(lQ−ê_a).
            {
                const int lQa = (a == 0 ? lQx : a == 1 ? lQy : lQz);
                double d = contract(lPx, lPy, lPz,
                                    lQx + (a == 0), lQy + (a == 1), lQz + (a == 2), Weight::Q);
                if (lQa > 0)
                    d -= static_cast<double>(lQa) *
                         contract(lPx, lPy, lPz,
                                  lQx - (a == 0), lQy - (a == 1), lQz - (a == 2), Weight::None);
                result[1 * 3 + a] = d;
            }
        }

        return result;
    }

    std::expected<std::vector<Eigen::MatrixXd>, std::string>
    compute_2c_eri_deriv(const HartreeFock::Calculator &calculator)
    {
        if (!calculator._ri_aux_basis)
            return std::unexpected("compute_2c_eri_deriv: RI auxiliary basis is not loaded.");
        const auto naux = static_cast<Eigen::Index>(calculator._ri_aux_basis->nfunctions);
        std::vector<Eigen::MatrixXd> dV(calculator._molecule.natoms * 3,
                                        Eigen::MatrixXd::Zero(naux, naux));
        auto ok = for_each_2c_deriv(
            calculator, HartreeFock::Mpi::Stripe{0, 1}, // whole tensor on every rank
            [&](std::size_t, Eigen::Index r, Eigen::Index c, std::size_t atomP,
                std::size_t atomQ, const std::array<double, 6> &d)
            {
                for (int a = 0; a < 3; ++a)
                {
                    dV[atomP * 3 + a](r, c) += d[0 * 3 + a];
                    dV[atomQ * 3 + a](r, c) += d[1 * 3 + a];
                }
            });
        if (!ok)
            return std::unexpected("compute_2c_eri_deriv: " + ok.error());
        return dV;
    }

    Eigen::MatrixXd build_ri_gamma3_ov(
        const Eigen::MatrixXd &D_ovov,
        const Eigen::MatrixXd &b_ov)
    {
        // Γ3_{(ia),Q} = Σ_{jb} D_{(ia),(jb)} B_{(jb),Q}. Both index the ov space
        // as i*nvirt+a; b_ov rows are that same packing (build_ri_mo_block with
        // C_occ, C_virt). Plain matrix product.
        return D_ovov * b_ov;
    }

    std::expected<Eigen::MatrixXd, std::string> build_ri_two_electron_gradient(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &gamma3,
        const Eigen::MatrixXd &x_proj,
        bool bra_prefolded)
    {
        // gamma3 and x_proj live in the working (fitted) space: working AO
        // pairs × fitted aux functions, spherical when the run is. The
        // derivative loops are Cartesian on both legs, so wg and γ_PQ are
        // lifted to Cartesian below (docs/RI_JK_ROUTINE.md step 16).
        const std::size_t natoms = calculator._molecule.natoms;
        const std::size_t nb = calculator.working_nbasis();
        const auto npair = static_cast<Eigen::Index>(nb * (nb + 1) / 2);
        if (!calculator._ri_aux_basis)
            return std::unexpected("build_ri_two_electron_gradient: RI auxiliary basis is not loaded.");
        if (gamma3.rows() != npair || x_proj.rows() != npair || gamma3.cols() != x_proj.cols())
            return std::unexpected("build_ri_two_electron_gradient: gamma3/x_proj shape does not "
                                   "match the working AO pair count.");

        // Bra pair weight. When gamma3's bra is a SINGLE ordering (μ≥ν only, e.g.
        // a bra-symmetric density), off-diagonal packed pairs stand in for both
        // (μν) and (νμ), so w=(μ==ν?1:2) recovers the full bra sum (build_ri_j
        // convention). When gamma3 is already the sum of both orderings
        // (bra_prefolded — the general non-bra-symmetric dm2buf case), w=1.
        Eigen::VectorXd w(npair);
        {
            Eigen::Index row = 0;
            for (std::size_t mu = 0; mu < nb; ++mu)
                for (std::size_t nu = 0; nu <= mu; ++nu, ++row)
                    w(row) = (bra_prefolded || mu == nu) ? 1.0 : 2.0;
        }

        // Both terms couple through V^{-1}, not V^{-1/2}: the fitted ERI is
        // (μν|λσ) = J V^{-1} Jᵀ, so d(μν|λσ) pairs dJ with (V^{-1}Jᵀ) and dV with
        // (J V^{-1})…(V^{-1}Jᵀ). gamma3 = Σ_{λσ} Γ·X (X = J V^{-1}) carries one
        // V^{-1}; x_proj is the raw X factors (the other leg of the metric fold).
        Eigen::MatrixXd wg = w.asDiagonal() * gamma3; // w · Σ Γ·X, bra fold

        // Metric-derivative charge γ_{PQ} = Σ_{(μν)} w·x_{(μν),P}·gamma3_{(μν),Q}.
        Eigen::MatrixXd gamma_pq = (w.asDiagonal() * x_proj).transpose() * gamma3;

        // Spherical orbital basis: Σ_{μν} G_Q(μν) d(μν|Q) with d_sph = C d_cart Cᵀ
        // equals Σ_{ab} (Cᵀ G_Q C)(ab) d_cart(ab|Q). Unpack each column to the
        // symmetric full G_Q (the packed sum w·gamma3 over μ≥ν, halved off the
        // diagonal), lift it, and repack with the Cartesian pair weight.
        if (calculator._shells._spherical)
        {
            const Eigen::MatrixXd &C = calculator._shells._cart_to_sph; // n_sph × n_cart
            const Eigen::Index nc = C.cols();
            Eigen::MatrixXd wg_cart(nc * (nc + 1) / 2, wg.cols());
            for (Eigen::Index Q = 0; Q < wg.cols(); ++Q)
            {
                Eigen::MatrixXd G(static_cast<Eigen::Index>(nb), static_cast<Eigen::Index>(nb));
                for (Eigen::Index mu = 0, r = 0; mu < static_cast<Eigen::Index>(nb); ++mu)
                    for (Eigen::Index nu = 0; nu <= mu; ++nu, ++r)
                        G(mu, nu) = G(nu, mu) = (mu == nu ? 1.0 : 0.5) * wg(r, Q);
                const Eigen::MatrixXd Gc = C.transpose() * G * C;
                for (Eigen::Index a = 0, r = 0; a < nc; ++a)
                    for (Eigen::Index b = 0; b <= a; ++b, ++r)
                        wg_cart(r, Q) = (a == b ? 1.0 : 2.0) * Gc(a, b);
            }
            wg = std::move(wg_cart);
        }
        // Spherical aux: d(·|P)_sph = Σ_c T_Pc d(·|c)_cart, dV_sph = T dV_cart Tᵀ.
        if (const Eigen::MatrixXd &T = calculator._ri_aux_basis->cart_to_sph; T.size())
        {
            wg = wg * T;
            gamma_pq = T.transpose() * gamma_pq * T;
        }

        // Streamed: each derivative element is contracted the moment it is
        // computed, so the 3·natoms × npair × naux derivative tensor is never
        // stored (docs/RI_JK_ROUTINE.md step 7). The loops run threaded
        // over aux shells; each shell accumulates into its own natoms × 3
        // partial, summed in shell order, so the gradient is the same at any
        // thread count (G6, docs/GRADIENT_INTEGRAL_PERFORMANCE.md).
        const std::size_t nshells = calculator._ri_aux_basis->shells.size();
        std::vector<Eigen::MatrixXd> part3(
            nshells, Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(natoms), 3));
        std::vector<Eigen::MatrixXd> part2 = part3;

        // 3-center: Σ_{(μν),Q} w·gamma3·dJ, each leg on its own atom.
        // Each rank visits its own aux shells; two_e is summed over ranks below.
        const HartreeFock::Mpi::Stripe stripe;
        auto ok3 = for_each_3c_deriv(
            calculator, stripe,
            [&](std::size_t K, Eigen::Index row, Eigen::Index col, std::size_t atomA,
                std::size_t atomB, std::size_t atomC, const std::array<double, 9> &d)
            {
                Eigen::MatrixXd &out = part3[K];
                const double g = wg(row, col);
                for (int q = 0; q < 3; ++q)
                {
                    out(static_cast<Eigen::Index>(atomA), q) += g * d[0 * 3 + q];
                    out(static_cast<Eigen::Index>(atomB), q) += g * d[1 * 3 + q];
                    out(static_cast<Eigen::Index>(atomC), q) += g * d[2 * 3 + q];
                }
            });
        if (!ok3)
            return std::unexpected(ok3.error());

        // 2-center metric correction: −½ Σ_{PQ} γ_{PQ}·dV. The ½ is real — it
        // balances the metric leg against the (unfactored) 3-center leg,
        // matching the dense two_e term to fitting accuracy (gated RG2.2).
        auto ok2 = for_each_2c_deriv(
            calculator, stripe,
            [&](std::size_t K, Eigen::Index r, Eigen::Index c, std::size_t atomP,
                std::size_t atomQ, const std::array<double, 6> &d)
            {
                Eigen::MatrixXd &out = part2[K];
                const double g = -0.5 * gamma_pq(r, c);
                for (int a = 0; a < 3; ++a)
                {
                    out(static_cast<Eigen::Index>(atomP), a) += g * d[0 * 3 + a];
                    out(static_cast<Eigen::Index>(atomQ), a) += g * d[1 * 3 + a];
                }
            });
        if (!ok2)
            return std::unexpected(ok2.error());

        Eigen::MatrixXd two_e = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(natoms), 3);
        for (const auto &p3 : part3)
            two_e += p3;
        for (const auto &p2 : part2)
            two_e += p2;
        sum_over_ranks(two_e);
        return two_e;
    }

    namespace
    {
        // X = J V^{-1} in packed (μ≥ν) pair × aux form. B = J V^{-1/2} is the
        // fitted pair factors; one more V^{-1/2} gives X. Built from the metric
        // factorization the same way build_ri_pair_factors applies V^{-1/2}.
        Eigen::MatrixXd build_ri_x_pair_factors(const HartreeFock::Calculator &calculator)
        {
            const Eigen::MatrixXd &J = calculator._ri_j3c;
            const auto &metric = *calculator._ri_metric_factor;
            if (metric.method == MetricFactorization::Method::Eigen)
            {
                // Pruned transform T with V^{-1} = Tᵀ T on the kept modes.
                const Eigen::MatrixXd &T = metric.transform;
                return (J * T.transpose()) * T;
            }
            // Cholesky V = L Lᵀ: Xᵀ = L^{-T} (L^{-1} Jᵀ), two triangular solves.
            // Never form V^{-1}: a Cartesian JKFIT metric has cond ~1e12 and the
            // explicit inverse put the RI-JK gradient 2e-6 Ha/Bohr off its own
            // energy FD (water, def2-universal-JKFIT); the solves match it.
            const auto L = metric.transform.triangularView<Eigen::Lower>();
            Eigen::MatrixXd Xt = L.solve(J.transpose());
            L.transpose().solveInPlace(Xt);
            return Xt.transpose(); // packed pair × aux
        }
    }

    namespace
    {
        // The shared core of the two pair-block contractions. For every aux
        // function Q and a symmetric factor F[Q] (nao × nao, from factor(Q, F)),
        //   W[Q](p,q) = Σ_{rs} dm2[p,q,r,s] F[Q](r,s)
        //             = Σ_blocks weight (M + M^T),  M = L_o G_Q L_v^T,
        //   G_Q(i,a)  = Σ_{jb} t[(ia),(jb)] (R_o^T F[Q] R_v)(j,b),
        // and sink(Q, W) receives it. Two passes over Q (the ket factors, one
        // GEMM per block, then W), each threaded over Q with Q-owned writes;
        // sink runs concurrently for different Q and must write only Q-owned
        // storage. Memory: (n_j·n_b + n_i·n_a) · naux per block. Under MPI only
        // this rank's Q (Mpi::Stripe) reach factor and sink; the caller sums
        // its result over ranks.
        template <typename Factor, typename Sink>
        void for_each_pair_block_w(const std::vector<PairDensityBlock> &blocks,
                                   Eigen::Index nao, Eigen::Index naux,
                                   Factor &&factor, Sink &&sink)
        {
            using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            const HartreeFock::Mpi::Stripe stripe;
            std::vector<Eigen::MatrixXd> ket(blocks.size());
            for (std::size_t k = 0; k < blocks.size(); ++k)
                ket[k].setZero(blocks[k].ket_occ.cols() * blocks[k].ket_virt.cols(), naux);
#pragma omp parallel for schedule(dynamic)
            for (Eigen::Index Q = 0; Q < naux; ++Q)
            {
                if (!stripe.owns(static_cast<std::size_t>(Q)))
                    continue;
                Eigen::MatrixXd F(nao, nao);
                factor(Q, F);
                for (std::size_t k = 0; k < blocks.size(); ++k)
                {
                    const RowMat f = blocks[k].ket_occ.transpose() * F * blocks[k].ket_virt; // (j, b)
                    ket[k].col(Q) = Eigen::Map<const Eigen::VectorXd>(f.data(), f.size());
                }
            }
            std::vector<Eigen::MatrixXd> G(blocks.size());
            for (std::size_t k = 0; k < blocks.size(); ++k)
                G[k] = blocks[k].t * ket[k]; // (ia) × Q
            ket.clear();
#pragma omp parallel for schedule(dynamic)
            for (Eigen::Index Q = 0; Q < naux; ++Q)
            {
                if (!stripe.owns(static_cast<std::size_t>(Q)))
                    continue;
                Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nao, nao);
                for (std::size_t k = 0; k < blocks.size(); ++k)
                {
                    const auto &b = blocks[k];
                    const Eigen::Map<const RowMat> GQ(G[k].col(Q).data(), b.bra_occ.cols(), b.bra_virt.cols());
                    const Eigen::MatrixXd M = b.bra_occ * GQ * b.bra_virt.transpose();
                    W += b.weight * (M + M.transpose());
                }
                sink(Q, W);
            }
        }
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_ri_gamma3_from_pair_blocks(
        const HartreeFock::Calculator &calculator,
        const std::vector<PairDensityBlock> &blocks)
    {
        const Eigen::MatrixXd x_pair = build_ri_x_pair_factors(calculator);
        const Eigen::Index naux = x_pair.cols();
        const auto n = static_cast<Eigen::Index>(calculator.working_nbasis());
        Eigen::MatrixXd gamma3 = Eigen::MatrixXd::Zero(n * (n + 1) / 2, naux);
        for_each_pair_block_w(
            blocks, n, naux,
            [&](Eigen::Index Q, Eigen::MatrixXd &F)
            {
                for (Eigen::Index mu = 0, pr = 0; mu < n; ++mu)
                    for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pr)
                        F(mu, nu) = F(nu, mu) = x_pair(pr, Q);
            },
            [&](Eigen::Index Q, const Eigen::MatrixXd &W)
            {
                // Packed bra folds μ<ν into μ≥ν: dm2[μν] + dm2[νμ].
                for (Eigen::Index mu = 0, pr = 0; mu < n; ++mu)
                    for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pr)
                        gamma3(pr, Q) = (mu == nu) ? W(mu, nu) : W(mu, nu) + W(nu, mu);
            });
        sum_over_ranks(gamma3);
        return {gamma3, x_pair};
    }

    std::expected<Eigen::MatrixXd, std::string> build_ri_scf_two_electron_gradient(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &Pa,
        const Eigen::MatrixXd &Pb,
        double exchange_coefficient)
    {
        // Working-basis densities: the fitted factors X are in the working
        // basis, and build_ri_two_electron_gradient lifts to Cartesian.
        const auto n = static_cast<Eigen::Index>(calculator.working_nbasis());
        const Eigen::Index npair = n * (n + 1) / 2;
        if (!calculator._ri_metric_factor || calculator._ri_j3c.rows() != npair)
            return std::unexpected("RI-JK gradient: the RI cache is not ready for this basis.");
        if (Pa.rows() != n || Pa.cols() != n || Pb.rows() != n || Pb.cols() != n)
            return std::unexpected("RI-JK gradient: density size does not match the AO basis.");

        // build_ri_two_electron_gradient returns ½ Σ G d(μν|λσ)_fit for the
        // three-index density of G; the SCF term is ¼ Σ Γ d, so G = Γ/2:
        //   G = Pt⊗Pt − k (Pa·Pa + Pb·Pb)
        //   gamma3_{μν,Q} = Σ_λσ G_{μν,λσ} X_{λσ,Q}
        //                 = Pt_μν c_Q − k (Pa X[Q] Pa + Pb X[Q] Pb)_μν,
        //   c_Q = Σ_λσ Pt_λσ X_{λσ,Q}  (full λσ sum: packed with weight 2).
        const Eigen::MatrixXd X = build_ri_x_pair_factors(calculator); // packed × naux
        const Eigen::Index naux = X.cols();
        const Eigen::MatrixXd Pt = Pa + Pb;
        const bool same_spin = (&Pa == &Pb) || (Pa.array() == Pb.array()).all();

        Eigen::VectorXd pt_packed(npair);
        for (Eigen::Index mu = 0, r = 0; mu < n; ++mu)
            for (Eigen::Index nu = 0; nu <= mu; ++nu, ++r)
                pt_packed(r) = (mu == nu ? 1.0 : 2.0) * Pt(mu, nu);
        const Eigen::RowVectorXd c = pt_packed.transpose() * X;

        // ponytail: Pσ X[Q] Pσ is nb³ per aux function; from C_occ it would be
        // nb²·n_occ (B2 in docs/ERI_FOCK_BLAS_SCOPE.md does that for RI-K).
        const HartreeFock::Mpi::Stripe stripe;
        Eigen::MatrixXd gamma3 = Eigen::MatrixXd::Zero(npair, naux);
#pragma omp parallel for schedule(dynamic)
        for (Eigen::Index Q = 0; Q < naux; ++Q)
        {
            if (!stripe.owns(static_cast<std::size_t>(Q)))
                continue;
            Eigen::MatrixXd XQ(n, n);
            for (Eigen::Index mu = 0, r = 0; mu < n; ++mu)
                for (Eigen::Index nu = 0; nu <= mu; ++nu, ++r)
                    XQ(mu, nu) = XQ(nu, mu) = X(r, Q);
            const Eigen::MatrixXd K = same_spin
                                          ? Eigen::MatrixXd(2.0 * Pa * XQ * Pa)
                                          : Eigen::MatrixXd(Pa * XQ * Pa + Pb * XQ * Pb);
            for (Eigen::Index mu = 0, r = 0; mu < n; ++mu)
                for (Eigen::Index nu = 0; nu <= mu; ++nu, ++r)
                    gamma3(r, Q) = Pt(mu, nu) * c(Q) - exchange_coefficient * K(mu, nu);
        }
        sum_over_ranks(gamma3);
        return build_ri_two_electron_gradient(calculator, gamma3, X);
    }

    Eigen::MatrixXd build_ri_imat_from_pair_blocks(
        const HartreeFock::Calculator &calculator,
        const std::vector<PairDensityBlock> &blocks)
    {
        const std::vector<Eigen::MatrixXd> B = build_ri_3index_unpacked(calculator);
        const auto n = static_cast<Eigen::Index>(calculator.working_nbasis());
        const auto naux = static_cast<Eigen::Index>(B.size());
        // imat = Σ_Q B[Q]^T W[Q]: one partial per bin (Q mod kBins, a function
        // of Q only), summed in bin order, so the result is the same at any
        // thread count.
        constexpr Eigen::Index kBins = 64;
        std::vector<Eigen::MatrixXd> part(kBins, Eigen::MatrixXd::Zero(n, n));
        std::vector<Eigen::MatrixXd> W_all(static_cast<std::size_t>(naux));
        for_each_pair_block_w(
            blocks, n, naux,
            [&](Eigen::Index Q, Eigen::MatrixXd &F) { F = B[static_cast<std::size_t>(Q)]; },
            [&](Eigen::Index Q, const Eigen::MatrixXd &W) { W_all[static_cast<std::size_t>(Q)] = W; });
        // Under MPI only this rank's Q are filled (B[Q] and W_all[Q] empty
        // otherwise); the sum over ranks completes imat.
#pragma omp parallel for schedule(static)
        for (Eigen::Index bin = 0; bin < kBins; ++bin)
            for (Eigen::Index Q = bin; Q < naux; Q += kBins)
                if (W_all[static_cast<std::size_t>(Q)].size())
                    part[bin].noalias() += B[static_cast<std::size_t>(Q)].transpose() *
                                           W_all[static_cast<std::size_t>(Q)];
        Eigen::MatrixXd imat = Eigen::MatrixXd::Zero(n, n);
        for (const auto &p : part)
            imat += p;
        sum_over_ranks(imat);
        return imat;
    }

    std::expected<void, std::string> ensure_ri_3c_ready(
        HartreeFock::Calculator &calculator)
    {
        auto metric_res = ensure_ri_metric_ready(calculator);
        if (!metric_res)
            return std::unexpected(metric_res.error());

        const std::size_t n_work = calculator.working_nbasis();
        const std::size_t npair = n_work * (n_work + 1) / 2;
        const std::size_t naux = calculator._ri_aux_basis ? calculator._ri_aux_basis->nfit() : 0;
        if (calculator._ri_j3c.rows() != static_cast<Eigen::Index>(npair) ||
            calculator._ri_j3c.cols() != static_cast<Eigen::Index>(naux))
        {
            auto j3c_res = compute_3c_eri(calculator);
            if (!j3c_res)
                return std::unexpected("ensure_ri_3c_ready: " + j3c_res.error());
            calculator._ri_j3c = std::move(*j3c_res);
            calculator._ri_pair_factors.resize(0, 0);
        }
        if (calculator._ri_pair_factors.rows() == calculator._ri_j3c.rows())
            return {};

        // B = J V^{-1/2}: the same metric factor the gradient's X uses.
        // Rows are independent, so under MPI each rank fits its own AO pairs
        // and the sum over ranks completes B; serial takes the whole matrix.
        const Eigen::MatrixXd &j3c = calculator._ri_j3c;
        const auto &metric = *calculator._ri_metric_factor;
        const auto fit = [&](const Eigen::MatrixXd &rows) -> Eigen::MatrixXd
        {
            if (metric.method == MetricFactorization::Method::Eigen)
                return rows * metric.transform.transpose();
            Eigen::MatrixXd t = rows.transpose();
            metric.transform.triangularView<Eigen::Lower>().solveInPlace(t);
            return t.transpose();
        };
        const HartreeFock::Mpi::Stripe stripe;
        if (stripe.size == 1)
            calculator._ri_pair_factors = fit(j3c);
        else
        {
            std::vector<Eigen::Index> mine;
            for (Eigen::Index r = 0; r < j3c.rows(); ++r)
                if (stripe.owns(static_cast<std::size_t>(r)))
                    mine.push_back(r);
            const Eigen::MatrixXd part = fit(j3c(mine, Eigen::all));
            Eigen::MatrixXd B = Eigen::MatrixXd::Zero(j3c.rows(), part.cols());
            B(mine, Eigen::all) = part;
            sum_over_ranks(B);
            calculator._ri_pair_factors = std::move(B);
        }
        return {};
    }

    const Eigen::MatrixXd &build_ri_pair_factors(const HartreeFock::Calculator &calculator)
    {
        return calculator._ri_pair_factors;
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

        // Under MPI each rank sums its own AO pairs; the sum over ranks below
        // completes b_pq.
        const HartreeFock::Mpi::Stripe stripe;
        Eigen::MatrixXd b_pq = Eigen::MatrixXd::Zero(npq, nfit);
        std::size_t pair_row = 0;
        for (Eigen::Index mu = 0; mu < C_row.rows(); ++mu)
        {
            const Eigen::RowVectorXd row_mu = C_row.row(mu);
            const Eigen::RowVectorXd col_mu = C_col.row(mu);
            for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pair_row)
            {
                if (!stripe.owns(pair_row))
                    continue;
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
        sum_over_ranks(b_pq);
        return b_pq;
    }

    Eigen::MatrixXd fitted_gram(const Eigen::MatrixXd &left, const Eigen::MatrixXd &right)
    {
        const HartreeFock::Mpi::Stripe stripe;
        if (stripe.size == 1)
            return left * right.transpose();
        std::vector<Eigen::Index> mine;
        for (Eigen::Index Q = 0; Q < left.cols(); ++Q)
            if (stripe.owns(static_cast<std::size_t>(Q)))
                mine.push_back(Q);
        Eigen::MatrixXd gram = left(Eigen::all, mine) * right(Eigen::all, mine).transpose();
        sum_over_ranks(gram);
        return gram;
    }

    Eigen::MatrixXd build_ri_j(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &D)
    {
        const Eigen::MatrixXd &pair_factors = build_ri_pair_factors(calculator);
        const Eigen::Index nb = D.rows();
        const Eigen::Index nfit = pair_factors.cols();

        // Under MPI each rank takes its own AO pairs in both passes; c and J
        // are each summed over ranks.
        const HartreeFock::Mpi::Stripe stripe;

        // c_Q = Σ_{μν} B_{(μν),Q} D_{μν}. The packed factors store μ≥ν only, so
        // off-diagonal pairs contribute twice (D symmetric): weight = (μ==ν?1:2).
        Eigen::RowVectorXd c = Eigen::RowVectorXd::Zero(nfit);
        std::size_t pair_row = 0;
        for (Eigen::Index mu = 0; mu < nb; ++mu)
            for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pair_row)
            {
                const double w = (mu == nu ? 1.0 : 2.0) * D(mu, nu);
                if (w != 0.0 && stripe.owns(pair_row))
                    c.noalias() += w * pair_factors.row(static_cast<Eigen::Index>(pair_row));
            }
        sum_over_ranks(c);

        // J_{μν} = Σ_Q B_{(μν),Q} c_Q; scatter to both (μ,ν) and (ν,μ).
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nb, nb);
        pair_row = 0;
        for (Eigen::Index mu = 0; mu < nb; ++mu)
            for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pair_row)
            {
                if (!stripe.owns(pair_row))
                    continue;
                const double val =
                    pair_factors.row(static_cast<Eigen::Index>(pair_row)).dot(c);
                J(mu, nu) = val;
                J(nu, mu) = val;
            }
        sum_over_ranks(J);
        return J;
    }

    std::vector<Eigen::MatrixXd> build_ri_3index_unpacked(
        const HartreeFock::Calculator &calculator)
    {
        const Eigen::MatrixXd &pair_factors = build_ri_pair_factors(calculator);
        const Eigen::Index nb =
            static_cast<Eigen::Index>(calculator.working_nbasis());
        const Eigen::Index naux = pair_factors.cols();

        // Under MPI only this rank's Q are allocated; the others stay empty
        // (0×0), so the list keeps its Q indexing and memory drops by 1/size.
        // Consumers skip empty entries and sum their result over ranks.
        const HartreeFock::Mpi::Stripe stripe;
        std::vector<Eigen::MatrixXd> B(static_cast<std::size_t>(naux));
        for (Eigen::Index Q = 0; Q < naux; ++Q)
            if (stripe.owns(static_cast<std::size_t>(Q)))
                B[static_cast<std::size_t>(Q)].setZero(nb, nb);

        std::size_t pair_row = 0;
        for (Eigen::Index mu = 0; mu < nb; ++mu)
            for (Eigen::Index nu = 0; nu <= mu; ++nu, ++pair_row)
            {
                const Eigen::RowVectorXd factors =
                    pair_factors.row(static_cast<Eigen::Index>(pair_row));
                for (Eigen::Index Q = 0; Q < naux; ++Q)
                {
                    if (B[static_cast<std::size_t>(Q)].size() == 0)
                        continue;
                    const double v = factors(Q);
                    B[static_cast<std::size_t>(Q)](mu, nu) = v;
                    B[static_cast<std::size_t>(Q)](nu, mu) = v;
                }
            }
        return B;
    }

    Eigen::MatrixXd build_ri_k(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &D,
        const std::vector<Eigen::MatrixXd> *unpacked)
    {
        std::vector<Eigen::MatrixXd> owned;
        if (!unpacked)
        {
            owned = build_ri_3index_unpacked(calculator);
            unpacked = &owned;
        }
        const auto &B = *unpacked;
        const Eigen::Index nb = D.rows();

        Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nb, nb);
        const double scale = D.cwiseAbs().maxCoeff();
        if (scale == 0.0)
            return K;

        // B2 (docs/ERI_FOCK_BLAS_SCOPE.md): factor D = X s Xᵀ over its
        // nonzero eigenvalues, so K = Σ_Q (B_Q X) s (B_Q X)ᵀ costs rank/nb of
        // the dense form. An SCF density has rank n_occ; any other density
        // (SAD guess, CASSCF, response trials) stays exact. Eigenvalues below
        // 1e-12 of the largest are roundoff (exact zeros for C_occ C_occᵀ).
        // A density that is not symmetric takes the dense form.
        const bool symmetric = (D - D.transpose()).cwiseAbs().maxCoeff() <= 1e-12 * scale;
        Eigen::Index rank = nb;
        Eigen::MatrixXd X;
        Eigen::VectorXd s;
        if (symmetric)
        {
            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(0.5 * (D + D.transpose()));
            const Eigen::VectorXd &lam = eig.eigenvalues();
            const double cut = 1e-12 * lam.cwiseAbs().maxCoeff();
            std::vector<Eigen::Index> kept;
            for (Eigen::Index k = 0; k < nb; ++k)
                if (std::abs(lam(k)) > cut)
                    kept.push_back(k);
            rank = static_cast<Eigen::Index>(kept.size());
            X.resize(nb, rank);
            s.resize(rank);
            for (Eigen::Index c = 0; c < rank; ++c)
            {
                X.col(c) = eig.eigenvectors().col(kept[c]);
                s(c) = lam(kept[c]);
            }
        }

        // Empty B[Q] belong to other ranks (build_ri_3index_unpacked); the
        // sum over ranks completes K.
        if (rank < nb)
        {
            for (const Eigen::MatrixXd &BQ : B)
            {
                if (BQ.size() == 0)
                    continue;
                const Eigen::MatrixXd Y = BQ * X;
                K.noalias() += Y * s.asDiagonal() * Y.transpose();
            }
        }
        else
        {
            for (const Eigen::MatrixXd &BQ : B)
            {
                if (BQ.size() == 0)
                    continue;
                // H = B[Q] D ; K += H B[Q]ᵀ  ⇒  K_{μν} += Σ_{λσ} B_{μλ} D_{λσ} B_{νσ}.
                const Eigen::MatrixXd H = BQ * D;
                K.noalias() += H * BQ.transpose();
            }
        }
        sum_over_ranks(K);
        return K;
    }

    Eigen::MatrixXd build_ri_fock_rhf(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &D)
    {
        // Share one unpacked tensor across J and K. J uses the packed factors
        // (via build_ri_j), K uses the unpacked ones; build the latter once.
        const std::vector<Eigen::MatrixXd> unpacked =
            build_ri_3index_unpacked(calculator);
        const Eigen::MatrixXd J = build_ri_j(calculator, D);
        const Eigen::MatrixXd K = build_ri_k(calculator, D, &unpacked);
        return J - 0.5 * K;
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_ri_fock_uhf(
        const HartreeFock::Calculator &calculator,
        const Eigen::MatrixXd &Pa,
        const Eigen::MatrixXd &Pb)
    {
        // G_sigma = J(Pa + Pb) - K(P_sigma). Coulomb from the TOTAL density,
        // exchange per-spin, no 1/2 (see the header note). One unpacked tensor
        // shared across both exchange builds.
        const std::vector<Eigen::MatrixXd> unpacked =
            build_ri_3index_unpacked(calculator);
        const Eigen::MatrixXd J = build_ri_j(calculator, Pa + Pb);
        const Eigen::MatrixXd Ka = build_ri_k(calculator, Pa, &unpacked);
        const Eigen::MatrixXd Kb = build_ri_k(calculator, Pb, &unpacked);
        return {J - Ka, J - Kb};
    }
} // namespace HartreeFock::Correlation::RI
