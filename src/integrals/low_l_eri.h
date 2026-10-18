#ifndef HF_INTEGRALS_LOW_L_ERI_H
#define HF_INTEGRALS_LOW_L_ERI_H

#include <array>
#include <cmath>
#include <numbers>

#include "base/types.h"
#include "lookup/boys.h"

// Closed-form contracted Coulomb ERIs for shell quartets with
// L_A + L_B + L_C + L_D <= 1: (ss|ss) and one p on any of the four centers.
//
// The per-component engines repeat the primitive loop (one Boys call and one
// prefactor product per primitive quartet) once per Cartesian component. Here
// one pass yields every component, which is where the speedup comes from
// (docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md). Per primitive quartet:
//
//   (ss|ss)^(m) = K_AB K_CD 2 sqrt(rho/pi) F_m(T),  T = rho |P-Q|^2
//   (p_i s|ss)  = PA_i (ss|ss)^(0) + WP_i (ss|ss)^(1)       (B: PB_i, WP_i)
//   (ss|p_i s)  = QC_i (ss|ss)^(0) + WQ_i (ss|ss)^(1)       (D: QD_i, WQ_i)
//
// Coulomb only: the screened kernels scale T, the prefactor and W-P/W-Q
// (see hgp_vrr), and stay on the generic engines until a screened sweep
// covers them (scope step C5).
namespace HartreeFock::Integrals
{
    // spAB / spCD: the pairs of the FIRST component of each shell (all
    // components of an s or p shell share the primitive data and norm).
    // p_center: -1 for (ss|ss), else 0..3 = which of A, B, C, D is the p shell.
    // out[0] is the (ss|ss) value; with a p shell, out[axis] for x, y, z.
    inline void low_l_shell_eri(const HartreeFock::ShellPair &spAB,
                                const HartreeFock::ShellPair &spCD,
                                int p_center, std::array<double, 3> &out) noexcept
    {
        const double two_over_sqrt_pi = 2.0 / std::sqrt(std::numbers::pi);
        double v0 = 0.0, vx = 0.0, vy = 0.0, vz = 0.0;

        for (const auto &ab : spAB.primitive_pairs)
        {
            for (const auto &cd : spCD.primitive_pairs)
            {
                const double delta = ab.zeta + cd.zeta;
                const double rho = ab.zeta * cd.zeta / delta;
                const double PQx = ab.center[0] - cd.center[0];
                const double PQy = ab.center[1] - cd.center[1];
                const double PQz = ab.center[2] - cd.center[2];
                const double T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
                const double base = ab.coeff_product * cd.coeff_product *
                                    ab.prefactor * cd.prefactor *
                                    two_over_sqrt_pi * std::sqrt(rho);

                if (p_center < 0)
                {
                    v0 += base * HartreeFock::Lookup::boys(0, T);
                    continue;
                }

                std::array<double, 2> F;
                HartreeFock::Lookup::boys_vec(T, F);

                // W-P = (zeta_CD/delta)(Q-P) on the bra, W-Q = (zeta_AB/delta)(P-Q) on the ket.
                const bool bra = p_center < 2;
                const double s = bra ? -cd.zeta / delta : ab.zeta / delta;
                const Eigen::Vector3d &d = p_center == 0 ? ab.pA
                                         : p_center == 1 ? ab.pB
                                         : p_center == 2 ? cd.pA
                                                         : cd.pB;
                const double f0 = base * F[0], f1 = base * F[1] * s;
                vx += d[0] * f0 + PQx * f1;
                vy += d[1] * f0 + PQy * f1;
                vz += d[2] * f0 + PQz * f1;
            }
        }

        if (p_center < 0)
            out = {v0, 0.0, 0.0};
        else
            out = {vx, vy, vz};
    }
}

#endif // !HF_INTEGRALS_LOW_L_ERI_H
