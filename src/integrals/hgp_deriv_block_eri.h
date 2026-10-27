// hgp_deriv_block_eri.h — first nuclear derivatives of a whole shell quartet
// from one primitive pass (docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md, G3).
//
// The per-component derivative (_compute_eri_deriv_elem) builds each of its 12
// outputs from up to two fresh contracted ERIs. Differentiating a Cartesian
// Gaussian raises or lowers one index:
//   d/dA_i (a b|c d) = 2 alpha (a+1_i b|c d) - a_i (a-1_i b|c d)
// and the same for B (2 beta) and C (2 gamma). D follows from translational
// invariance, d/dD = -(d/dA + d/dB + d/dC).
//
// So one VRR pass to (L_A+L_B+1 | L_C+L_D+1), contracted into four
// accumulators (weights 1, 2 alpha, 2 beta, 2 gamma), and an HRR
// (hgp_block::hrr) per raised or lowered shell quartet gives every component's
// 12 derivatives. The weights multiply the unraised pair's coefficients, so
// the component norm of the unraised function applies (as in the
// per-component path).
//
// Coulomb only, like the energy block kernel.
#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "hgp_block_eri.h"

namespace HartreeFock::Integrals
{
    struct HgpDerivBlockKernel
    {
        bool covers(int total_l, HartreeFock::ERIKernel k) const noexcept
        {
            return k == HartreeFock::ERIKernel::Coulomb && total_l + 2 <= 4 * MAX_L;
        }

        bool prepare(const BlockEriRequest &r)
        {
            using hgp_block::n_upto;
            const int LA = r.L[0], LB = r.L[1], LC = r.L[2], LD = r.L[3];
            const int LAB = LA + LB, LCD = LC + LD;
            if (!r.ab || !r.cd || r.kernel != HartreeFock::ERIKernel::Coulomb ||
                !covers(LAB + LCD, r.kernel) ||
                LAB + 1 > hgp_block::kMaxL || LCD + 1 > hgp_block::kMaxL)
                return false;
            req_ = &r;

            // Primitive pass over the raised grid, contracted four ways.
            const int Ne = n_upto(LAB + 1), Nf = n_upto(LCD + 1), Md = LAB + LCD + 3;
            const std::size_t nacc = static_cast<std::size_t>(Ne) * Nf;
            for (auto &a : acc_)
            {
                grow_to(a, nacc);
                std::fill_n(a.begin(), nacc, 0.0);
            }
            const auto &P = r.ab->primitive_pairs, &Q = r.cd->primitive_pairs;
            const std::size_t K = P.size() * Q.size();
            const auto pass = [&]<int B>()
            {
                grow_to(VL_, nacc * Md * B);
                const PrimitivePair *pa[B], *pc[B];
                double w[4][B];
                for (std::size_t k0 = 0; k0 < K; k0 += B)
                {
                    for (int l = 0; l < B; ++l)
                    {
                        // A padding lane repeats a valid quartet at weight 0.
                        const bool real = k0 + l < K;
                        const std::size_t k = real ? k0 + l : k0;
                        pa[l] = &P[k / Q.size()];
                        pc[l] = &Q[k % Q.size()];
                        const double c = real ? pa[l]->coeff_product * pc[l]->coeff_product : 0.0;
                        w[0][l] = c;
                        w[1][l] = 2.0 * pa[l]->alpha * c;
                        w[2][l] = 2.0 * pa[l]->beta * c;
                        w[3][l] = 2.0 * pc[l]->alpha * c;
                    }
                    hgp_block::vrr_lanes<B>(pa, pc, LAB + 1, LCD + 1, VL_.data());
                    for (std::size_t ef = 0; ef < nacc; ++ef)
                    {
                        const double *x = &VL_[ef * Md * B];
                        for (int s = 0; s < 4; ++s)
                        {
                            double sum = 0.0;
                            for (int l = 0; l < B; ++l)
                                sum += w[s][l] * x[l];
                            acc_[s][ef] += sum;
                        }
                    }
                }
            };
            if (K >= hgp_block::kMinLaneK)
                pass.template operator()<hgp_block::kLanes>();
            else
                pass.template operator()<1>();

            // Raised blocks (weighted accumulators) and lowered blocks (plain).
            const int L4[4] = {LA, LB, LC, LD};
            for (int c = 0; c < 3; ++c)
            {
                int Lr[4] = {LA, LB, LC, LD};
                ++Lr[c];
                block(acc_[c + 1], Nf, Lr, raised_[c]);
                if (L4[c] > 0)
                {
                    int Ll[4] = {LA, LB, LC, LD};
                    --Ll[c];
                    block(acc_[0], Nf, Ll, lowered_[c]);
                }
            }
            return true;
        }

        // d(ca cb|cc cd) / d(centre s, axis q) at out[3 s + q], s = A, B, C, D.
        void deriv(std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd,
                   std::array<double, 12> &out) const
        {
            const BlockEriRequest &r = *req_;
            const std::size_t c[4] = {ca, cb, cc, cd};
            std::array<int, 3> e[4];
            for (int s = 0; s < 4; ++s)
            {
                const auto &v = r.views[s][c[s]]->_cartesian;
                e[s] = {v[0], v[1], v[2]};
            }
            const double *nr = r.norm_ratio;
            const double scale = nr[r.views[0][ca]->_index] * nr[r.views[1][cb]->_index] *
                                 nr[r.views[2][cc]->_index] * nr[r.views[3][cd]->_index];
            for (int s = 0; s < 3; ++s)
                for (int q = 0; q < 3; ++q)
                {
                    std::array<int, 3> f[4] = {e[0], e[1], e[2], e[3]};
                    ++f[s][q];
                    double d = raised_[s][offset(s, +1, f)];
                    if (e[s][q] > 0)
                    {
                        f[s][q] -= 2;
                        d -= e[s][q] * lowered_[s][offset(s, -1, f)];
                    }
                    out[3 * s + q] = scale * d;
                }
            for (int q = 0; q < 3; ++q)
                out[9 + q] = -(out[q] + out[3 + q] + out[6 + q]);
        }

    private:
        // Every component of shell quartet L (L-major order within each shell)
        // into dst, via HRR from acc.
        void block(const std::vector<double> &acc, int acc_stride, const int (&L)[4],
                   std::vector<double> &dst)
        {
            using hgp_block::n_upto;
            std::size_t n[4];
            for (int s = 0; s < 4; ++s)
            {
                n[s] = static_cast<std::size_t>(n_upto(L[s]) - n_upto(L[s] - 1));
                ci_[s].resize(n[s]);
                for (std::size_t c = 0; c < n[s]; ++c)
                    ci_[s][c] = n_upto(L[s] - 1) + static_cast<int>(c);
            }
            const int *const idx[4] = {ci_[0].data(), ci_[1].data(), ci_[2].data(), ci_[3].data()};
            grow_to(dst, n[0] * n[1] * n[2] * n[3]);
            hgp_block::hrr(acc.data(), acc_stride, L[0], L[1], L[2], L[3],
                           req_->ab->R, req_->cd->R, idx, n, H_, K_, dst.data());
        }

        // Position of components f (one of them shifted by `shift` on shell s)
        // in a block() of that shifted quartet.
        std::size_t offset(int s, int shift, const std::array<int, 3> (&f)[4]) const
        {
            using hgp_block::n_upto;
            std::size_t pos = 0;
            for (int t = 0; t < 4; ++t)
            {
                const int L = req_->L[t] + (t == s ? shift : 0);
                const std::size_t n = static_cast<std::size_t>(n_upto(L) - n_upto(L - 1));
                pos = pos * n + static_cast<std::size_t>(hgp_block::index(f[t][0], f[t][1], f[t][2]) - n_upto(L - 1));
            }
            return pos;
        }

        const BlockEriRequest *req_ = nullptr;
        // Scratch, reused across prepare() calls; one kernel per thread.
        std::vector<double> VL_, H_, K_;
        std::vector<double> acc_[4];                      // weights 1, 2a, 2b, 2c
        std::vector<double> raised_[3], lowered_[3];      // per centre A, B, C
        std::vector<int> ci_[4];
    };
}
