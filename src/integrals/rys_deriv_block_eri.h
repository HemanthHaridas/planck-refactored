// rys_deriv_block_eri.h — first nuclear derivatives of a whole shell quartet
// with the 2D Rys formulation (docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md, G3).
//
// libcint's route. A component is sum_r w_r I_x I_y I_z, and a derivative
// with respect to centre A on axis q changes only I_q:
//   d/dA_q I_q(a, b, c, d) = 2 alpha I_q(a+1, b, c, d) - a I_q(a-1, b, c, d)
// so per primitive quartet and root, the axis tables are built one angular
// step higher on A, B and C (rys_block::axis_table on (L_A+1, L_B+1 | L_C+1,
// L_D)), and each of the nine A/B/C derivatives is one root dot product with
// the differentiated axis swapped in. D follows from translational
// invariance, d/dD = -(d/dA + d/dB + d/dC). The exponent alpha is known per
// primitive quartet, so the weights 2 alpha, 2 beta, 2 gamma never need a
// separate contraction.
//
// Coulomb only, like the energy kernels.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rys_block_eri.h"

namespace HartreeFock::Integrals
{
    struct RysDerivBlockKernel
    {
        // n = (L + 1) / 2 + 1 roots for the raised total L.
        static constexpr int kMaxTotalL = 2 * (HartreeFock::Rys::RYS_MAX_ROOTS - 1) - 1;

        // Only the blocks where Rys2D beats HGP (same rule as RysBlockKernel's
        // crossover mode until measured for derivatives); a chain must put a
        // kernel that covers everything behind it.
        bool crossover = false;

        explicit RysDerivBlockKernel(bool only_crossover = false) noexcept : crossover(only_crossover) {}

        bool covers(int total_l, HartreeFock::ERIKernel k) const noexcept
        {
            return !crossover && total_l <= kMaxTotalL && k == HartreeFock::ERIKernel::Coulomb &&
                   total_l + 2 < HartreeFock::RysQuad::detail::VRR_DIM;
        }

        bool prepare(const BlockEriRequest &r)
        {
            using HartreeFock::RysQuad::detail::_rys_eri_prep;
            using HartreeFock::RysQuad::detail::_rys_vrr_1d;
            using HartreeFock::RysQuad::detail::VRR_DIM;
            const int LA = r.L[0], LB = r.L[1], LC = r.L[2], LD = r.L[3];
            const int lt = LA + LB + LC + LD;
            // Raised tables need L_A+1, L_B+1, L_C+1 <= MAX_L (axis_table's
            // fixed arrays) and L_A+L_B+2, L_C+L_D+1 < VRR_DIM.
            if (!r.ab || !r.cd || r.kernel != HartreeFock::ERIKernel::Coulomb || lt > kMaxTotalL ||
                std::max({LA, LB, LC}) + 1 > MAX_L || LA + LB + 2 >= VRR_DIM || LC + LD + 1 >= VRR_DIM)
                return false;
            if (crossover)
            {
                const std::size_t K = r.ab->primitive_pairs.size() * r.cd->primitive_pairs.size();
                if (K > 3 || (K > 1 && lt < 7) || lt < 4)
                    return false;
            }
            req_ = &r;

            const int n = (lt + 1) / 2 + 1;
            const int dA = LA + 2, dB = LB + 2, dC = LC + 2, dD = LD + 1;
            const std::size_t nt = static_cast<std::size_t>(dA) * dB * dC * dD * n;
            for (auto &t : I_)
                grow_to(t, nt);
            // Table strides for one step on A, B, C (in doubles).
            const std::size_t sA = static_cast<std::size_t>(dB) * dC * dD * n;
            const std::size_t sB = static_cast<std::size_t>(dC) * dD * n;
            const std::size_t sC = static_cast<std::size_t>(dD) * n;

            n_[0] = r.n[0], n_[1] = r.n[1], n_[2] = r.n[2], n_[3] = r.n[3];
            const std::size_t nc = n_[0] * n_[1] * n_[2] * n_[3];
            grow_to(out_, 9 * nc);
            std::fill_n(out_.begin(), 9 * nc, 0.0);

            // Per class: each component's row offset per axis and its exponents
            // (a_q, b_q, c_q) for the lowered terms.
            auto &cls = cls_[((LA * (MAX_L + 1) + LB) * (MAX_L + 1) + LC) * (MAX_L + 1) + LD];
            if (cls.empty())
            {
                cls.resize(nc);
                std::size_t o = 0;
                for (std::size_t ca = 0; ca < n_[0]; ++ca)
                    for (std::size_t cb = 0; cb < n_[1]; ++cb)
                        for (std::size_t cc = 0; cc < n_[2]; ++cc)
                            for (std::size_t cd = 0; cd < n_[3]; ++cd, ++o)
                            {
                                const auto &A = r.views[0][ca]->_cartesian, &B = r.views[1][cb]->_cartesian,
                                           &C = r.views[2][cc]->_cartesian, &D = r.views[3][cd]->_cartesian;
                                for (int q = 0; q < 3; ++q)
                                {
                                    cls[o].off[q] = static_cast<std::uint32_t>((((A[q] * dB + B[q]) * dC + C[q]) * dD + D[q]) * n);
                                    cls[o].e[0][q] = static_cast<std::uint8_t>(A[q]);
                                    cls[o].e[1][q] = static_cast<std::uint8_t>(B[q]);
                                    cls[o].e[2][q] = static_cast<std::uint8_t>(C[q]);
                                }
                            }
            }
            const std::size_t step[3] = {sA, sB, sC};

            double t2[HartreeFock::Rys::RYS_MAX_ROOTS], w[HartreeFock::Rys::RYS_MAX_ROOTS];
            double V[VRR_DIM][VRR_DIM];
            for (const PrimitivePair &p : r.ab->primitive_pairs)
                for (const PrimitivePair &q : r.cd->primitive_pairs)
                {
                    const auto g = _rys_eri_prep(p, q, HartreeFock::ERIKernel::Coulomb, 0.0);
                    HartreeFock::Rys::rys_roots_weights(n, g.T, t2, w);
                    const double scale = g.prefac * p.coeff_product * q.coeff_product;
                    const double PA[3] = {g.PAx, g.PAy, g.PAz}, QC[3] = {g.QCx, g.QCy, g.QCz};
                    const double WP[3] = {g.WPx, g.WPy, g.WPz}, WQ[3] = {g.WQx, g.WQy, g.WQz};
                    for (int k = 0; k < n; ++k)
                    {
                        const double u = t2[k];
                        const double B00 = 0.5 * g.inv_delta * u;
                        const double B10 = 0.5 * g.inv_zetaAB * (1.0 - g.rho_over_zeta * u);
                        const double B01 = 0.5 * g.inv_zetaCD * (1.0 - g.rho_over_eta * u);
                        for (int x = 0; x < 3; ++x)
                        {
                            _rys_vrr_1d(V, LA + LB + 2, LC + LD + 1, PA[x] + u * WP[x], QC[x] + u * WQ[x], B00, B10, B01);
                            rys_block::axis_table(V, LA + 1, LB + 1, LC + 1, LD, r.ab->R[x], r.cd->R[x], n, k, I_[x].data());
                        }
                        w[k] *= scale;
                    }
                    const double two_exp[3] = {2.0 * p.alpha, 2.0 * p.beta, 2.0 * q.alpha};
                    const double *T[3] = {I_[0].data(), I_[1].data(), I_[2].data()};
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        const Component &m = cls[c];
                        const double *ax[3] = {T[0] + m.off[0], T[1] + m.off[1], T[2] + m.off[2]};
                        double *o = &out_[9 * c];
                        for (int q = 0; q < 3; ++q)
                        {
                            const double *u = ax[(q + 1) % 3], *v = ax[(q + 2) % 3];
                            for (int s = 0; s < 3; ++s)
                            {
                                const double *up = ax[q] + step[s];
                                const int e = m.e[s][q];
                                double sum = 0.0;
                                if (e > 0)
                                {
                                    const double *dn = ax[q] - step[s];
                                    for (int k = 0; k < n; ++k)
                                        sum += w[k] * (two_exp[s] * up[k] - e * dn[k]) * u[k] * v[k];
                                }
                                else
                                    for (int k = 0; k < n; ++k)
                                        sum += w[k] * two_exp[s] * up[k] * u[k] * v[k];
                                o[3 * s + q] += sum;
                            }
                        }
                    }
                }
            return true;
        }

        // d(ca cb|cc cd) / d(centre s, axis q) at out[3 s + q], s = A, B, C, D.
        void deriv(std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd,
                   std::array<double, 12> &out) const
        {
            const BlockEriRequest &r = *req_;
            const double *nr = r.norm_ratio;
            const double scale = nr[r.views[0][ca]->_index] * nr[r.views[1][cb]->_index] *
                                 nr[r.views[2][cc]->_index] * nr[r.views[3][cd]->_index];
            const double *o = &out_[9 * (((ca * n_[1] + cb) * n_[2] + cc) * n_[3] + cd)];
            for (int i = 0; i < 9; ++i)
                out[i] = scale * o[i];
            for (int q = 0; q < 3; ++q)
                out[9 + q] = -(out[q] + out[3 + q] + out[6 + q]);
        }

    private:
        struct Component
        {
            std::uint32_t off[3];   // row offset per axis in the raised tables
            std::uint8_t e[3][3];   // exponent of centre A, B, C on each axis
        };

        const BlockEriRequest *req_ = nullptr;
        // Scratch, reused across prepare() calls (one kernel copy per thread).
        std::vector<double> I_[3], out_;
        std::array<std::vector<Component>, (MAX_L + 1) * (MAX_L + 1) * (MAX_L + 1) * (MAX_L + 1)> cls_;
        std::size_t n_[4] = {};
    };
}
