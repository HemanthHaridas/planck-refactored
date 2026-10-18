#ifndef HF_INTEGRALS_LOW_L_ERI_H
#define HF_INTEGRALS_LOW_L_ERI_H

#include <array>
#include <cmath>
#include <numbers>

#include "base/types.h"
#include "lookup/boys.h"

// Closed-form contracted Coulomb ERIs for shell quartets with
// L_A + L_B + L_C + L_D <= 2.
//
// The per-component engines repeat the primitive loop (one Boys call and one
// prefactor product per primitive quartet) once per Cartesian component. Here
// one pass yields every component, which is where the speedup comes from
// (docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md). The angular momentum is a list of at
// most two "raises", each a center (0..3 = A, B, C, D) and an axis. Per
// primitive quartet, with theta_m = c K_AB K_CD 2 sqrt(rho/pi) F_m(T):
//
//   no raise:       theta_0
//   one raise:      R_i(m) = d1_i theta_m + w1_i theta_{m+1}           (m = 0)
//   two raises:     V_ij   = d2_j R_i(0) + w2_j R_i(1) + delta_ij kappa
//
// d is PA, PB, QC or QD for the raised center; w is W-P on the bra and W-Q on
// the ket. kappa is (theta_0 - (rho/zeta) theta_1)/(2 zeta) when both raises
// sit on the bra, the same with eta on the ket, and theta_1/(2(zeta+eta)) when
// they straddle (exact Obara-Saika). Coulomb only: the screened kernels scale
// T, the prefactor, W-P/W-Q and the straddling kappa (see hgp_vrr), and stay
// on the generic engines until a screened sweep covers them (scope step C5).
namespace HartreeFock::Integrals
{
    // spAB / spCD: the pairs of the FIRST component of each shell. The caller
    // rescales d components by their own norm (the first component's norm is
    // baked into coeff_product).
    // n_raise: 0, 1 or 2; c1 <= c2 the raised centers (unused beyond n_raise).
    // out: out[0] with no raise, out[i] with one, out[3 i + j] with two.
    inline void low_l_shell_eri(const HartreeFock::ShellPair &spAB,
                                const HartreeFock::ShellPair &spCD,
                                int n_raise, int c1, int c2,
                                std::array<double, 9> &out) noexcept
    {
        const double two_over_sqrt_pi = 2.0 / std::sqrt(std::numbers::pi);
        out.fill(0.0);

        for (const auto &ab : spAB.primitive_pairs)
        {
            for (const auto &cd : spCD.primitive_pairs)
            {
                const double delta = ab.zeta + cd.zeta;
                const double rho = ab.zeta * cd.zeta / delta;
                const double PQ[3] = {ab.center[0] - cd.center[0],
                                      ab.center[1] - cd.center[1],
                                      ab.center[2] - cd.center[2]};
                const double T = rho * (PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2]);
                const double base = ab.coeff_product * cd.coeff_product *
                                    ab.prefactor * cd.prefactor *
                                    two_over_sqrt_pi * std::sqrt(rho);

                if (n_raise == 0)
                {
                    out[0] += base * HartreeFock::Lookup::boys(0, T);
                    continue;
                }

                // W-P = (eta/delta)(Q-P) on the bra, W-Q = (zeta/delta)(P-Q) on the ket.
                const auto d_of = [&](int c) -> const Eigen::Vector3d &
                { return c == 0 ? ab.pA : c == 1 ? ab.pB : c == 2 ? cd.pA : cd.pB; };
                const auto w_scale = [&](int c)
                { return c < 2 ? -cd.zeta / delta : ab.zeta / delta; };
                const Eigen::Vector3d &d1 = d_of(c1);
                const double s1 = w_scale(c1);

                if (n_raise == 1)
                {
                    std::array<double, 2> F;
                    HartreeFock::Lookup::boys_vec(T, F);
                    const double f0 = base * F[0], f1 = base * F[1] * s1;
                    for (int i = 0; i < 3; ++i)
                        out[i] += d1[i] * f0 + PQ[i] * f1;
                    continue;
                }

                std::array<double, 3> F;
                HartreeFock::Lookup::boys_vec(T, F);
                const double t0 = base * F[0], t1 = base * F[1], t2 = base * F[2];
                const Eigen::Vector3d &d2 = d_of(c2);
                const double s2 = w_scale(c2);

                double R0[3], R1[3];
                for (int i = 0; i < 3; ++i)
                {
                    const double w1 = s1 * PQ[i];
                    R0[i] = d1[i] * t0 + w1 * t1;
                    R1[i] = d1[i] * t1 + w1 * t2;
                }
                const double kappa =
                    c2 < 2 ? (t0 - (rho / ab.zeta) * t1) * (0.5 / ab.zeta)
                    : c1 >= 2 ? (t0 - (rho / cd.zeta) * t1) * (0.5 / cd.zeta)
                              : t1 * (0.5 / delta);
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        out[3 * i + j] += d2[j] * R0[i] + s2 * PQ[j] * R1[i];
                out[0] += kappa;
                out[4] += kappa;
                out[8] += kappa;
            }
        }
    }
}

#endif // !HF_INTEGRALS_LOW_L_ERI_H
