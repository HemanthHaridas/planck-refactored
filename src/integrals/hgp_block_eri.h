// hgp_block_eri.h — the Head-Gordon–Pople shell-quartet block kernel.
//
// The per-component HGP entry reruns its whole primitive loop (Boys, VRR,
// contraction) once per Cartesian component, 27-78x per shell quartet
// wherever it runs (docs/ERI_LOW_L_CLOSED_FORM_SCOPE.md, D0). This kernel
// runs it once per shell quartet and returns every component
// (block_eri.h contract):
//
//   VRR    per primitive quartet: [e0|f0]^(m) for |e| <= L_A+L_B,
//          |f| <= L_C+L_D, Cartesian functions indexed L-major (a triangle,
//          not the per-axis cube box that made the earlier hoist slower);
//   sum    the m = 0 slice over primitive quartets, for |e| >= L_A and
//          |f| >= L_C only (what HRR reads);
//   HRR    once per shell quartet: (a b| from (e 0| on the bra, then (c d|
//          from (f 0| on the ket, for every component.
//
// Recurrences (Obara–Saika in the W form, as hgp_vrr):
//   [e+1_i|0]^m = PA_i [e]^m + WP_i [e]^{m+1}
//                 + e_i/(2 zeta) ([e-1_i]^m - rho/zeta [e-1_i]^{m+1})
//   [e|f+1_j]^m = QC_j [e|f]^m + WQ_j [e|f]^{m+1}
//                 + f_j/(2 eta) ([e|f-1_j]^m - rho/eta [e|f-1_j]^{m+1})
//                 + e_j/(2(zeta+eta)) [e-1_j|f]^{m+1}
//   (a, b+1_i| = (a+1_i, b| + (A-B)_i (a, b|        and the same on the ket.
//
// Coulomb only, like the closed forms: the screened scaling (scope step C5)
// is not mirrored yet, so screened kernels decline and stay on eri_elem.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "base/types.h"
#include "block_eri.h"
#include "lookup/boys.h"

namespace HartreeFock::Integrals
{
    namespace hgp_block
    {
        constexpr int kMaxL = 2 * MAX_L; // L_A + L_B

        // Number of Cartesian functions with total angular momentum <= L.
        constexpr int n_upto(int L) noexcept { return L < 0 ? 0 : (L + 1) * (L + 2) * (L + 3) / 6; }

        // L-major index: functions of lower total L first, then (x desc, z asc).
        constexpr int index(int x, int y, int z) noexcept
        {
            const int L = x + y + z;
            return n_upto(L - 1) + (L - x) * (L - x + 1) / 2 + z;
        }

        struct Cart
        {
            std::array<int, 3> e{};
            int L = 0;
            int axis = -1;                 // first axis with e > 0 (the build direction)
            std::array<int, 3> down{};     // index of e - 1_i, or -1
            std::array<int, 3> up{};       // index of e + 1_i, or -1 past kMaxL
        };

        inline const std::vector<Cart> &table()
        {
            static const std::vector<Cart> t = []
            {
                std::vector<Cart> v(n_upto(kMaxL));
                for (int L = 0; L <= kMaxL; ++L)
                    for (int x = L; x >= 0; --x)
                        for (int z = 0; z <= L - x; ++z)
                        {
                            Cart &c = v[index(x, L - x - z, z)];
                            c.e = {x, L - x - z, z};
                            c.L = L;
                            for (int i = 0; i < 3; ++i)
                            {
                                std::array<int, 3> d = c.e, u = c.e;
                                if (d[i] > 0)
                                {
                                    --d[i];
                                    c.down[i] = index(d[0], d[1], d[2]);
                                    if (c.axis < 0)
                                        c.axis = i;
                                }
                                else
                                    c.down[i] = -1;
                                ++u[i];
                                c.up[i] = L < kMaxL ? index(u[0], u[1], u[2]) : -1;
                            }
                        }
                return v;
            }();
            return t;
        }
    }

    struct HgpBlockKernel
    {
        int min_l = 0; // handle total L in [min_l, max_l]
        int max_l = 4 * MAX_L;

        explicit HgpBlockKernel(int lo = 0, int hi = 4 * MAX_L) noexcept : min_l(lo), max_l(hi) {}

        bool prepare(const BlockEriRequest &r)
        {
            const int LA = r.L[0], LB = r.L[1], LC = r.L[2], LD = r.L[3];
            const int lt = LA + LB + LC + LD;
            if (lt < min_l || lt > max_l || !r.ab || !r.cd ||
                r.kernel != HartreeFock::ERIKernel::Coulomb || r.prim_eps != 0.0)
                return false;
            req_ = &r;

            using hgp_block::n_upto;
            const auto &T = hgp_block::table();
            const int LAB = LA + LB, LCD = LC + LD, M = LAB + LCD;
            const int Ne = n_upto(LAB), Nf = n_upto(LCD), Md = M + 1;
            const int e0 = n_upto(LA - 1), f0 = n_upto(LC - 1);

            thread_local std::vector<double> V, acc, H, K;
            V.resize(static_cast<std::size_t>(Ne) * Nf * Md);
            acc.assign(static_cast<std::size_t>(Ne) * Nf, 0.0);
            const auto v = [&](int e, int f) { return &V[(static_cast<std::size_t>(e) * Nf + f) * Md]; };

            const double two_over_sqrt_pi = 2.0 / std::sqrt(std::numbers::pi);
            for (const auto &ab : r.ab->primitive_pairs)
                for (const auto &cd : r.cd->primitive_pairs)
                {
                    const double zeta = ab.zeta, eta = cd.zeta, delta = zeta + eta;
                    const double rho = zeta * eta / delta;
                    double WP[3], WQ[3], PQ2 = 0.0;
                    for (int x = 0; x < 3; ++x)
                    {
                        const double W = (zeta * ab.center[x] + eta * cd.center[x]) / delta;
                        WP[x] = W - ab.center[x];
                        WQ[x] = W - cd.center[x];
                        const double pq = ab.center[x] - cd.center[x];
                        PQ2 += pq * pq;
                    }
                    const double prefac = ab.prefactor * cd.prefactor * two_over_sqrt_pi * std::sqrt(rho);
                    double F[4 * MAX_L + 1]; // M <= 4*MAX_L: real totals, not per-axis maxima
                    HartreeFock::Lookup::boys_vec(rho * PQ2, std::span<double>(F, M + 1));
                    double *v00 = v(0, 0);
                    for (int m = 0; m <= M; ++m)
                        v00[m] = prefac * F[m];

                    const double i2z = 0.5 / zeta, i2e = 0.5 / eta, i2d = 0.5 / delta;
                    const double rz = rho / zeta, re = rho / eta;

                    for (int e = 1; e < Ne; ++e) // bra
                    {
                        const hgp_block::Cart &c = T[e];
                        const int i = c.axis, em = c.down[i], mt = M - c.L;
                        const double pa = ab.pA[i], wp = WP[i];
                        const double *a = v(em, 0);
                        double *o = v(e, 0);
                        for (int m = 0; m <= mt; ++m)
                            o[m] = pa * a[m] + wp * a[m + 1];
                        if (c.e[i] > 1)
                        {
                            const double *b = v(T[em].down[i], 0);
                            const double s = (c.e[i] - 1) * i2z;
                            for (int m = 0; m <= mt; ++m)
                                o[m] += s * (b[m] - rz * b[m + 1]);
                        }
                    }
                    for (int f = 1; f < Nf; ++f) // ket
                    {
                        const hgp_block::Cart &cf = T[f];
                        const int j = cf.axis, fm = cf.down[j];
                        const double qc = cd.pA[j], wq = WQ[j];
                        const int fmm = cf.e[j] > 1 ? T[fm].down[j] : -1;
                        const double sf = (cf.e[j] - 1) * i2e;
                        for (int e = 0; e < Ne; ++e)
                        {
                            const hgp_block::Cart &ce = T[e];
                            const int mt = M - ce.L - cf.L;
                            const double *a = v(e, fm);
                            double *o = v(e, f);
                            for (int m = 0; m <= mt; ++m)
                                o[m] = qc * a[m] + wq * a[m + 1];
                            if (fmm >= 0)
                            {
                                const double *b = v(e, fmm);
                                for (int m = 0; m <= mt; ++m)
                                    o[m] += sf * (b[m] - re * b[m + 1]);
                            }
                            if (ce.e[j] > 0)
                            {
                                const double *b = v(ce.down[j], fm);
                                const double s = ce.e[j] * i2d;
                                for (int m = 0; m <= mt; ++m)
                                    o[m] += s * b[m + 1];
                            }
                        }
                    }

                    const double w = ab.coeff_product * cd.coeff_product;
                    for (int e = e0; e < Ne; ++e)
                        for (int f = f0; f < Nf; ++f)
                            acc[static_cast<std::size_t>(e) * Nf + f] += w * v(e, f)[0];
                }

            // Bra HRR: H[a][b][f - f0] for |a| in [LA, LAB - |b|], |b| <= LB.
            const int Nb = n_upto(LB), Nd = n_upto(LD), nfk = Nf - f0;
            H.resize(static_cast<std::size_t>(Ne) * Nb * nfk);
            const auto h = [&](int a, int b) { return &H[(static_cast<std::size_t>(a) * Nb + b) * nfk]; };
            for (int a = e0; a < Ne; ++a)
                for (int f = 0; f < nfk; ++f)
                    h(a, 0)[f] = acc[static_cast<std::size_t>(a) * Nf + f0 + f];
            for (int b = 1; b < Nb; ++b)
            {
                const int i = T[b].axis, bm = T[b].down[i];
                const double ab_i = r.ab->R[i];
                for (int a = e0; a < n_upto(LAB - T[b].L); ++a)
                {
                    double *dst = h(a, b);
                    const double *s1 = h(T[a].up[i], bm), *s2 = h(a, bm);
                    for (int f = 0; f < nfk; ++f)
                        dst[f] = s1[f] + ab_i * s2[f];
                }
            }

            // Ket HRR per final (a, b), then the component readout.
            n_[0] = r.n[0], n_[1] = r.n[1], n_[2] = r.n[2], n_[3] = r.n[3];
            out_.resize(n_[0] * n_[1] * n_[2] * n_[3]);
            K.resize(static_cast<std::size_t>(Nf) * Nd);
            const auto idx = [&](int s, std::size_t c)
            {
                const auto &e = r.views[s][c]->_cartesian;
                return hgp_block::index(e[0], e[1], e[2]);
            };
            std::size_t o = 0;
            for (std::size_t ca = 0; ca < n_[0]; ++ca)
                for (std::size_t cb = 0; cb < n_[1]; ++cb)
                {
                    const double *g = h(idx(0, ca), idx(1, cb));
                    for (int c = f0; c < Nf; ++c)
                        K[static_cast<std::size_t>(c) * Nd] = g[c - f0];
                    for (int d = 1; d < Nd; ++d)
                    {
                        const int j = T[d].axis, dm = T[d].down[j];
                        const double cd_j = r.cd->R[j];
                        for (int c = f0; c < n_upto(LCD - T[d].L); ++c)
                            K[static_cast<std::size_t>(c) * Nd + d] =
                                K[static_cast<std::size_t>(T[c].up[j]) * Nd + dm] +
                                cd_j * K[static_cast<std::size_t>(c) * Nd + dm];
                    }
                    for (std::size_t cc = 0; cc < n_[2]; ++cc)
                        for (std::size_t cdc = 0; cdc < n_[3]; ++cdc)
                            out_[o++] = K[static_cast<std::size_t>(idx(2, cc)) * Nd + idx(3, cdc)];
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
        std::size_t n_[4] = {};
        std::vector<double> out_;
    };
}
