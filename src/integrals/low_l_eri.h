#ifndef HF_INTEGRALS_LOW_L_ERI_H
#define HF_INTEGRALS_LOW_L_ERI_H

#include <array>
#include <cmath>
#include <numbers>

#include "base/types.h"
#include "block_eri.h"
#include "lookup/boys.h"

// Closed-form contracted Coulomb ERIs for shell quartets with
// L_A + L_B + L_C + L_D <= 3.
//
// The per-component engines repeat the primitive loop (one Boys call and one
// prefactor product per primitive quartet) once per Cartesian component. Here
// one pass yields every component, which is where the speedup comes from
// (docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md). The angular momentum is a list of at
// most three "raises", each a center (0..3 = A, B, C, D) and an axis, applied
// in center order. Per primitive quartet, with
// theta_m = c K_AB K_CD 2 sqrt(rho/pi) F_m(T):
//
//   no raise:      theta_0
//   one raise:     A_i(m)   = d1_i theta_m + w1_i theta_{m+1}
//   two raises:    B_ij(m)  = d2_j A_i(m) + w2_j A_i(m+1) + [i=j] cpl(1,2; theta)
//   three raises:  C_ijk    = d3_k B_ij(0) + w3_k B_ij(1)
//                             + [k=i] cpl(1,3; A2_j) + [k=j] cpl(2,3; A_i)
//
// d is PA, PB, QC or QD for the raised center; w is W-P on the bra and W-Q on
// the ket. A2_j is the second raise applied alone. cpl(y,x; X) is the
// Obara-Saika coupling to the earlier raise y that the new raise x lowers:
// (X(m) - (rho/zeta) X(m+1))/(2 zeta) when both sit on the bra, the same
// with eta on the ket, and X(m+1)/(2(zeta+eta)) when they straddle. Coulomb only: the screened kernels scale
// T, the prefactor, W-P/W-Q and the straddling kappa (see hgp_vrr), and stay
// on the generic engines until a screened sweep covers them (scope step C5).
namespace HartreeFock::Integrals
{
    // spAB / spCD: the pairs of the FIRST component of each shell. The caller
    // rescales d components by their own norm (the first component's norm is
    // baked into coeff_product).
    // n_raise: 0..3; c[0] <= c[1] <= c[2] the raised centers (unused beyond
    // n_raise). out: out[0] with no raise, out[i] with one, out[3 i + j] with
    // two, out[9 i + 3 j + k] with three.
    inline void low_l_shell_eri(const HartreeFock::ShellPair &spAB,
                                const HartreeFock::ShellPair &spCD,
                                int n_raise, const int (&c)[3],
                                std::array<double, 27> &out) noexcept
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
                const int c1 = c[0], c2 = c[1], c3 = c[2];
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

                const Eigen::Vector3d &d2 = d_of(c2);
                const double s2 = w_scale(c2);

                if (n_raise == 3)
                {
                    std::array<double, 4> F;
                    HartreeFock::Lookup::boys_vec(T, F);
                    double th[4];
                    for (int m = 0; m < 4; ++m)
                        th[m] = base * F[m];
                    const Eigen::Vector3d &d3 = d_of(c3);
                    const double s3 = w_scale(c3);
                    // Coupling of a new raise on x to an earlier one on y, at m.
                    const auto cpl = [&](int y, int x, const double *X, int m)
                    {
                        if (y < 2 && x < 2)
                            return (X[m] - (rho / ab.zeta) * X[m + 1]) * (0.5 / ab.zeta);
                        if (y >= 2 && x >= 2)
                            return (X[m] - (rho / cd.zeta) * X[m + 1]) * (0.5 / cd.zeta);
                        return X[m + 1] * (0.5 / delta);
                    };
                    double A[3][3], A2[3][2], B[3][3][2];
                    for (int i = 0; i < 3; ++i)
                    {
                        for (int m = 0; m < 3; ++m)
                            A[i][m] = d1[i] * th[m] + s1 * PQ[i] * th[m + 1];
                        for (int m = 0; m < 2; ++m)
                            A2[i][m] = d2[i] * th[m] + s2 * PQ[i] * th[m + 1];
                    }
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            for (int m = 0; m < 2; ++m)
                                B[i][j][m] = d2[j] * A[i][m] + s2 * PQ[j] * A[i][m + 1] +
                                             (i == j ? cpl(c1, c2, th, m) : 0.0);
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            for (int k = 0; k < 3; ++k)
                            {
                                double v = d3[k] * B[i][j][0] + s3 * PQ[k] * B[i][j][1];
                                if (k == i)
                                    v += cpl(c1, c3, A2[j], 0);
                                if (k == j)
                                    v += cpl(c2, c3, A[i], 0);
                                out[9 * i + 3 * j + k] += v;
                            }
                    continue;
                }

                std::array<double, 3> F;
                HartreeFock::Lookup::boys_vec(T, F);
                const double t0 = base * F[0], t1 = base * F[1], t2 = base * F[2];

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

    // low_l_shell_eri as a block kernel (block_eri.h): total L <= max_l (at
    // most 3), Coulomb, no primitive screening. The readout arithmetic is the
    // fused loop's original one, so G is bitwise unchanged by the interface.
    struct ClosedFormBlockKernel
    {
        int max_l = 3;

        explicit ClosedFormBlockKernel(int max_total_l = 3) noexcept : max_l(max_total_l) {}

        bool covers(int total_l, HartreeFock::ERIKernel k) const noexcept
        {
            return max_l > 0 && total_l <= (max_l < 3 ? max_l : 3) && k == HartreeFock::ERIKernel::Coulomb;
        }
        bool prepare(const BlockEriRequest &r) noexcept
        {
            const int lt = r.L[0] + r.L[1] + r.L[2] + r.L[3];
            if (max_l <= 0 || lt > (max_l < 3 ? max_l : 3) || !r.ab || !r.cd ||
                r.kernel != HartreeFock::ERIKernel::Coulomb || r.prim_eps != 0.0)
                return false;
            req_ = &r;
            n_raise_ = 0;
            for (int s = 0; s < 4; ++s)
                for (int u = 0; u < r.L[s]; ++u)
                    c_[n_raise_++] = s;
            low_l_shell_eri(*r.ab, *r.cd, n_raise_, c_, vals_);
            return true;
        }

        double value(std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd) const noexcept
        {
            const std::size_t comp[4] = {ca, cb, cc, cd};
            // Axes in the kernel's raise order: centers in order, each unit of
            // x, then y, then z.
            int axes[3] = {0, 0, 0}, n_ax = 0;
            std::size_t ao[4];
            for (int s = 0; s < 4; ++s)
            {
                const HartreeFock::ContractedView &v = *req_->views[s][comp[s]];
                ao[s] = v._index;
                for (int a = 0; a < 3 && n_ax < n_raise_; ++a)
                    for (int u = 0; u < v._cartesian[a]; ++u)
                        axes[n_ax++] = a;
            }
            const int slot = n_raise_ == 3   ? 9 * axes[0] + 3 * axes[1] + axes[2]
                             : n_raise_ == 2 ? 3 * axes[0] + axes[1]
                                             : axes[0];
            const double *nr = req_->norm_ratio;
            return vals_[slot] * nr[ao[0]] * nr[ao[1]] * nr[ao[2]] * nr[ao[3]];
        }

    private:
        const BlockEriRequest *req_ = nullptr;
        int n_raise_ = 0;
        int c_[3] = {0, 0, 0};
        std::array<double, 27> vals_{};
    };
}

#endif // !HF_INTEGRALS_LOW_L_ERI_H
