#ifndef HF_RYS_DETAIL_H
#define HF_RYS_DETAIL_H

// Rys pieces shared by the per-component path (rys.cpp) and the 2D Rys block
// kernel (scope step K2.1b, docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md): the
// screened-kernel map, the per-primitive-pair geometry and the 1D VRR.

#include <cmath>
#include <numbers>

#include "base/types.h"

namespace HartreeFock::RysQuad::detail
{
    // VRR_DIM = 2*MAX_L+1 = 13 (per-axis, matches os.cpp convention).
    inline constexpr int VRR_DIM = 2 * MAX_L + 1;

    struct ScreenedKernelData
    {
        double rho = 0.0;
        double prefactor_scale = 1.0;
        double boys_scale = 1.0;
    };

    inline ScreenedKernelData screened_kernel_data(
        double rho,
        HartreeFock::ERIKernel kernel,
        double omega) noexcept
    {
        // Range-separated kernels are implemented by replacing the Coulomb rho
        // parameter, prefactor, and Boys-function argument with their screened
        // equivalents. Keeping that map in one helper lets OS and Rys share the
        // same physics-level interpretation.
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

    // ─── 1D Rys VRR ───────────────────────────────────────────────────────────
    //
    // Fills vrr[a][c] for a in [0, lAB], c in [0, lCD] using the recurrences:
    //
    //   [0,0]     = 1.0                                       (seed)
    //   [a+1, 0]  = C00 * [a,0]  + a*B10 * [a-1,0]
    //   [a, c+1]  = D00 * [a,c]  + c*B01 * [a,c-1]  + a*B00 * [a-1,c]
    //
    // where for Rys root u = t_r^2:
    //   C00 = PA_q + u*WP_q,  D00 = QC_q + u*WQ_q
    //   B00 = u/(2*delta),  B10 = 1/(2*zeta) - u/(2*delta),  B01 = 1/(2*eta) - u/(2*delta)

    inline void _rys_vrr_1d(
        double vrr[VRR_DIM][VRR_DIM],
        const int lAB, const int lCD,
        const double C00, const double D00,
        const double B00, const double B10, const double B01) noexcept
    {
        vrr[0][0] = 1.0;

        // Build A-axis (c=0)
        for (int a = 1; a <= lAB; ++a)
        {
            vrr[a][0] = C00 * vrr[a - 1][0];
            if (a >= 2)
                vrr[a][0] += (a - 1) * B10 * vrr[a - 2][0];
        }

        // Build C-axis and mixed (increment c for each a)
        for (int c = 1; c <= lCD; ++c)
        {
            vrr[0][c] = D00 * vrr[0][c - 1];
            if (c >= 2)
                vrr[0][c] += (c - 1) * B01 * vrr[0][c - 2];

            for (int a = 1; a <= lAB; ++a)
            {
                vrr[a][c] = D00 * vrr[a][c - 1] + a * B00 * vrr[a - 1][c - 1];
                if (c >= 2)
                    vrr[a][c] += (c - 1) * B01 * vrr[a][c - 2];
            }
        }
    }

    struct RysPrimGeom
    {
        double PAx, PAy, PAz;
        double QCx, QCy, QCz;
        double WPx, WPy, WPz;
        double WQx, WQy, WQz;
        double T;
        double prefac;
        double inv_delta;
        double inv_zetaAB; // ppAB.inv_zeta
        double inv_zetaCD; // ppCD.inv_zeta
        double rho_over_zeta;
        double rho_over_eta;
        double b00_scale; // 1 for Coulomb, screen.boys_scale otherwise
    };

    // Per-primitive-pair derived quantities. Independent of how the bra/ket
    // angular momentum splits between A/B and C/D, so it is computed once per
    // primitive pair (component-independent).
    inline RysPrimGeom _rys_eri_prep(
        const HartreeFock::PrimitivePair &ppAB,
        const HartreeFock::PrimitivePair &ppCD,
        HartreeFock::ERIKernel kernel,
        double omega) noexcept
    {
        const double zeta = ppAB.zeta;
        const double eta = ppCD.zeta;
        const double delta = zeta + eta;
        const double inv_delta = 1.0 / delta;
        const double rho = zeta * eta * inv_delta;
        const ScreenedKernelData screen = screened_kernel_data(rho, kernel, omega);
        const double effective_rho = screen.rho;

        const double Px = ppAB.center[0], Py = ppAB.center[1], Pz = ppAB.center[2];
        const double Qx = ppCD.center[0], Qy = ppCD.center[1], Qz = ppCD.center[2];

        const double PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;

        const double Wx = (zeta * Px + eta * Qx) * inv_delta;
        const double Wy = (zeta * Py + eta * Qy) * inv_delta;
        const double Wz = (zeta * Pz + eta * Qz) * inv_delta;

        const double wpwq_scale =
            (kernel == HartreeFock::ERIKernel::Coulomb) ? 1.0 : screen.boys_scale;
        const double b00_scale =
            (kernel == HartreeFock::ERIKernel::Coulomb) ? 1.0 : screen.boys_scale;

        return RysPrimGeom{
            .PAx = ppAB.pA[0], .PAy = ppAB.pA[1], .PAz = ppAB.pA[2],
            .QCx = ppCD.pA[0], .QCy = ppCD.pA[1], .QCz = ppCD.pA[2],
            .WPx = (Wx - Px) * wpwq_scale,
            .WPy = (Wy - Py) * wpwq_scale,
            .WPz = (Wz - Pz) * wpwq_scale,
            .WQx = (Wx - Qx) * wpwq_scale,
            .WQy = (Wy - Qy) * wpwq_scale,
            .WQz = (Wz - Qz) * wpwq_scale,
            .T = screen.boys_scale * rho * (PQx * PQx + PQy * PQy + PQz * PQz),
            .prefac = ppAB.prefactor * ppCD.prefactor * 2.0 *
                      std::sqrt(rho / std::numbers::pi) * screen.prefactor_scale,
            .inv_delta = inv_delta,
            .inv_zetaAB = ppAB.inv_zeta,
            .inv_zetaCD = ppCD.inv_zeta,
            .rho_over_zeta = effective_rho * ppAB.inv_zeta,
            .rho_over_eta = effective_rho * ppCD.inv_zeta,
            .b00_scale = b00_scale,
        };
    }
} // namespace HartreeFock::RysQuad::detail

#endif
