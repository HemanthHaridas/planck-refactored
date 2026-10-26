// rys_block_eri.h — the 2D Rys shell-quartet block kernel (K2.1b,
// docs/ERI_KERNEL_VS_LIBCINT_SCOPE.md).
//
// libcint's formulation. Per primitive quartet and per Rys root r (n =
// L/2 + 1 roots for total L):
//
//   VRR    per axis q, the 1D table I_q(e, f) for e <= L_A+L_B, f <= L_C+L_D
//          (detail::_rys_vrr_1d, shared with the per-component path);
//   HRR    per axis, in 1D: (a, b+1) = (a+1, b) + (A-B)_q (a, b) on the bra,
//          then the same with (C-D)_q on the ket, to I_q(a, b, c, d);
//   sum    every Cartesian component as sum_r w_r I_x I_y I_z.
//
// The per-component path instead sums the roots into a 6D box first and runs
// its HRR on that box, which is what made the hoisted Rys block 50-210x
// libcint (K2.0). Here the HRR runs on tables of (L_A+1)(L_B+1)(L_C+1)(L_D+1)
// entries per root and axis, and the assembly is the main cost.
//
// Tables are root-innermost: I_q[(((a dB + b) dC + c) dD + d) n + r], so a
// component is a length-n dot product over three contiguous rows.
//
// Coulomb only, like the other block kernels.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "base/types.h"
#include "block_eri.h"
#include "rys_detail.h"
#include "rys_roots.h"

namespace HartreeFock::Integrals
{
    namespace rys_block
    {
        using HartreeFock::RysQuad::detail::VRR_DIM;

        // Per-axis HRR of root r: V[e][f] -> I[(((a dB + b) dC + c) dD + d) n + r].
        inline void axis_table(const double V[VRR_DIM][VRR_DIM], int LA, int LB, int LC, int LD,
                               double ab, double cd, int n, int r, double *I) noexcept
        {
            const int LAB = LA + LB, LCD = LC + LD;
            double G[VRR_DIM][MAX_L + 1][VRR_DIM]; // [e][b][f]
            for (int e = 0; e <= LAB; ++e)
                for (int f = 0; f <= LCD; ++f)
                    G[e][0][f] = V[e][f];
            for (int b = 1; b <= LB; ++b)
                for (int e = 0; e <= LAB - b; ++e)
                    for (int f = 0; f <= LCD; ++f)
                        G[e][b][f] = G[e + 1][b - 1][f] + ab * G[e][b - 1][f];
            for (int a = 0; a <= LA; ++a)
                for (int b = 0; b <= LB; ++b)
                {
                    double H[VRR_DIM][MAX_L + 1]; // [c][d]
                    for (int c = 0; c <= LCD; ++c)
                        H[c][0] = G[a][b][c];
                    for (int d = 1; d <= LD; ++d)
                        for (int c = 0; c <= LCD - d; ++c)
                            H[c][d] = H[c + 1][d - 1] + cd * H[c][d - 1];
                    double *row = I + static_cast<std::size_t>((a * (LB + 1) + b) * (LC + 1)) * (LD + 1) * n + r;
                    for (int c = 0; c <= LC; ++c)
                        for (int d = 0; d <= LD; ++d)
                            row[static_cast<std::size_t>(c * (LD + 1) + d) * n] = H[c][d];
                }
        }
    }

    struct RysBlockKernel
    {
        static constexpr int kMaxTotalL = 2 * (HartreeFock::Rys::RYS_MAX_ROOTS - 1); // n = L/2 + 1 roots

        int min_l = 0; // handle total L in [min_l, max_l]
        int max_l = kMaxTotalL;

        explicit RysBlockKernel(int lo = 0, int hi = kMaxTotalL) noexcept
            : min_l(lo), max_l(std::min(hi, kMaxTotalL)) {}

        bool covers(int total_l, HartreeFock::ERIKernel k) const noexcept
        {
            return total_l >= min_l && total_l <= max_l && k == HartreeFock::ERIKernel::Coulomb;
        }
        bool prepare(const BlockEriRequest &r)
        {
            const int LA = r.L[0], LB = r.L[1], LC = r.L[2], LD = r.L[3];
            const int lt = LA + LB + LC + LD;
            if (lt < min_l || lt > max_l || !r.ab || !r.cd ||
                r.kernel != HartreeFock::ERIKernel::Coulomb || r.prim_eps != 0.0)
                return false;
            req_ = &r;

            using HartreeFock::RysQuad::detail::_rys_eri_prep;
            using HartreeFock::RysQuad::detail::_rys_vrr_1d;
            using rys_block::VRR_DIM;
            const int n = lt / 2 + 1, LAB = LA + LB, LCD = LC + LD;
            const int dB = LB + 1, dC = LC + 1, dD = LD + 1;
            const std::size_t nt = static_cast<std::size_t>(LA + 1) * dB * dC * dD * n;
            for (auto &t : I_)
                grow_to(t, nt);

            // Row offsets of each component in the three axis tables.
            n_[0] = r.n[0], n_[1] = r.n[1], n_[2] = r.n[2], n_[3] = r.n[3];
            const std::size_t nc = n_[0] * n_[1] * n_[2] * n_[3];
            grow_to(out_, nc);
            std::fill_n(out_.begin(), nc, 0.0);
            if (off_.size() < 3 * nc)
                off_.resize(3 * nc);
            std::size_t o = 0;
            for (std::size_t ca = 0; ca < n_[0]; ++ca)
                for (std::size_t cb = 0; cb < n_[1]; ++cb)
                    for (std::size_t cc = 0; cc < n_[2]; ++cc)
                        for (std::size_t cd = 0; cd < n_[3]; ++cd, ++o)
                        {
                            const auto &A = r.views[0][ca]->_cartesian, &B = r.views[1][cb]->_cartesian,
                                       &C = r.views[2][cc]->_cartesian, &D = r.views[3][cd]->_cartesian;
                            for (int q = 0; q < 3; ++q)
                                off_[3 * o + q] = static_cast<std::size_t>(((A[q] * dB + B[q]) * dC + C[q]) * dD + D[q]) * n;
                        }

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
                            _rys_vrr_1d(V, LAB, LCD, PA[x] + u * WP[x], QC[x] + u * WQ[x], B00, B10, B01);
                            rys_block::axis_table(V, LA, LB, LC, LD, r.ab->R[x], r.cd->R[x], n, k, I_[x].data());
                        }
                        w[k] *= scale;
                    }
                    const double *X = I_[0].data(), *Y = I_[1].data(), *Z = I_[2].data();
                    for (std::size_t c = 0; c < nc; ++c)
                    {
                        const double *x = X + off_[3 * c], *y = Y + off_[3 * c + 1], *z = Z + off_[3 * c + 2];
                        double s = 0.0;
                        for (int k = 0; k < n; ++k)
                            s += w[k] * x[k] * y[k] * z[k];
                        out_[c] += s;
                    }
                }
            return true;
        }

        double value(std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd) const noexcept
        {
            const double *nr = req_->norm_ratio;
            return out_[((ca * n_[1] + cb) * n_[2] + cc) * n_[3] + cd] *
                   nr[req_->views[0][ca]->_index] * nr[req_->views[1][cb]->_index] *
                   nr[req_->views[2][cc]->_index] * nr[req_->views[3][cd]->_index];
        }

    private:
        const BlockEriRequest *req_ = nullptr;
        // Scratch, reused across prepare() calls (one kernel copy per thread).
        std::vector<double> I_[3], out_;
        std::vector<std::size_t> off_;
        std::size_t n_[4] = {};
    };
}
