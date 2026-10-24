// hgp_block_eri.h — the Head-Gordon–Pople shell-quartet block kernel.
//
// The per-component HGP entry reruns its whole primitive loop (Boys, VRR,
// contraction) once per Cartesian component, 27-78x per shell quartet
// wherever it runs (docs/ERI_SHELL_QUARTET_FOCK_BUILD.md, D0). This kernel
// runs it once per shell quartet and returns every component
// (block_eri.h contract):
//
//   VRR    per primitive quartet: [e0|f0]^(m) for |e| <= L_A+L_B,
//          |f| <= L_C+L_D, Cartesian functions indexed L-major (a triangle,
//          not the per-axis cube box that made the earlier hoist slower);
//          in blocks with K >= 3 primitive quartets, four at a time across
//          SIMD lanes (hgp_block::vrr_lanes);
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

#include <algorithm>
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

        // Primitive quartets per pass, and the smallest block K that uses them.
        constexpr int kLanes = 4;
        constexpr std::size_t kMinLaneK = 3;

        // The VRR of the block kernel for B primitive quartets at once
        // (ab[l], cd[l]), lane-minor: out[((e * Nf + f) * (M + 1) + m) * B + l]
        // = [e0|f0]^(m) of lane l, |e| <= LAB, |f| <= LCD, M = LAB + LCD.
        // Weights and contraction are the caller's, so the block kernel and a
        // family pass (several member accumulators) share it. Every innermost
        // loop runs over the B lanes. out must hold n_upto(LAB) * n_upto(LCD) *
        // (M + 1) * B doubles.
        template <int B>
        inline void vrr_lanes(const HartreeFock::PrimitivePair *const (&ab)[B],
                              const HartreeFock::PrimitivePair *const (&cd)[B],
                              int LAB, int LCD, double *out)
        {
            const auto &T = table();
            const int M = LAB + LCD, Ne = n_upto(LAB), Nf = n_upto(LCD), Md = M + 1;
            const auto v = [&](int e, int f) { return out + (static_cast<std::size_t>(e) * Nf + f) * Md * B; };
            const double two_over_sqrt_pi = 2.0 / std::sqrt(std::numbers::pi);
            alignas(64) double pa[3][B], wp[3][B], qc[3][B], wq[3][B], i2z[B], rz[B], i2e[B], re[B], i2d[B],
                prefac[B], Tl[B];
            double *v00 = v(0, 0);
            for (int l = 0; l < B; ++l)
            {
                const HartreeFock::PrimitivePair &p = *ab[l], &q = *cd[l];
                const double zeta = p.zeta, eta = q.zeta, delta = zeta + eta;
                const double rho = zeta * eta / delta;
                double PQ2 = 0.0;
                for (int x = 0; x < 3; ++x)
                {
                    const double W = (zeta * p.center[x] + eta * q.center[x]) / delta;
                    wp[x][l] = W - p.center[x];
                    wq[x][l] = W - q.center[x];
                    pa[x][l] = p.pA[x];
                    qc[x][l] = q.pA[x];
                    const double pq = p.center[x] - q.center[x];
                    PQ2 += pq * pq;
                }
                prefac[l] = p.prefactor * q.prefactor * two_over_sqrt_pi * std::sqrt(rho);
                Tl[l] = rho * PQ2;
                i2z[l] = 0.5 / zeta, i2e[l] = 0.5 / eta, i2d[l] = 0.5 / delta;
                rz[l] = rho / zeta, re[l] = rho / eta;
            }
            // One batched Boys call for the B lanes (bitwise equal to a
            // boys_vec per lane).
            double F[B * (4 * MAX_L + 1)];
            HartreeFock::Lookup::boys_batch(std::span<const double>(Tl, B), M,
                                            std::span<double>(F, static_cast<std::size_t>(B) * Md));
            for (int l = 0; l < B; ++l)
                for (int m = 0; m <= M; ++m)
                    v00[m * B + l] = prefac[l] * F[l * Md + m];
            for (int e = 1; e < Ne; ++e)
            {
                const Cart &c = T[e];
                const int i = c.axis, em = c.down[i], mt = M - c.L;
                const double *a = v(em, 0);
                double *o = v(e, 0);
                for (int m = 0; m <= mt; ++m)
                    for (int l = 0; l < B; ++l)
                        o[m * B + l] = pa[i][l] * a[m * B + l] + wp[i][l] * a[(m + 1) * B + l];
                if (c.e[i] > 1)
                {
                    const double *b = v(T[em].down[i], 0);
                    const double s = c.e[i] - 1;
                    for (int m = 0; m <= mt; ++m)
                        for (int l = 0; l < B; ++l)
                            o[m * B + l] += s * i2z[l] * (b[m * B + l] - rz[l] * b[(m + 1) * B + l]);
                }
            }
            for (int f = 1; f < Nf; ++f)
            {
                const Cart &cf = T[f];
                const int j = cf.axis, fm = cf.down[j];
                const int fmm = cf.e[j] > 1 ? T[fm].down[j] : -1;
                const double sf = cf.e[j] - 1;
                for (int e = 0; e < Ne; ++e)
                {
                    const Cart &ce = T[e];
                    const int mt = M - ce.L - cf.L;
                    const double *a = v(e, fm);
                    double *o = v(e, f);
                    for (int m = 0; m <= mt; ++m)
                        for (int l = 0; l < B; ++l)
                            o[m * B + l] = qc[j][l] * a[m * B + l] + wq[j][l] * a[(m + 1) * B + l];
                    if (fmm >= 0)
                    {
                        const double *b = v(e, fmm);
                        for (int m = 0; m <= mt; ++m)
                            for (int l = 0; l < B; ++l)
                                o[m * B + l] += sf * i2e[l] * (b[m * B + l] - re[l] * b[(m + 1) * B + l]);
                    }
                    if (ce.e[j] > 0)
                    {
                        const double *b = v(ce.down[j], fm);
                        const double s = ce.e[j];
                        for (int m = 0; m <= mt; ++m)
                            for (int l = 0; l < B; ++l)
                                o[m * B + l] += s * i2d[l] * b[(m + 1) * B + l];
                    }
                }
            }
        }
    }

    struct HgpBlockKernel
    {
        int min_l = 0; // handle total L in [min_l, max_l]
        int max_l = 4 * MAX_L;

        explicit HgpBlockKernel(int lo = 0, int hi = 4 * MAX_L) noexcept : min_l(lo), max_l(hi) {}

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

            using hgp_block::n_upto;
            const int LAB = LA + LB, LCD = LC + LD, M = LAB + LCD;
            const int Ne = n_upto(LAB), Nf = n_upto(LCD), Md = M + 1;
            const int e0 = n_upto(LA - 1), f0 = n_upto(LC - 1);

            grow_to(acc, static_cast<std::size_t>(Ne) * Nf);
            std::fill_n(acc.begin(), static_cast<std::size_t>(Ne) * Nf, 0.0);

            // Blocks with K >= kMinLaneK primitive quartets run them kLanes at a
            // time; below that, one at a time. Both are hgp_block::vrr_lanes, so
            // the recurrence exists once. Measured
            // (docs/ERI_SHELL_QUARTET_FOCK_BUILD.md, 2b): 4 lanes beat 2 and 8,
            // and a K = 3 block is cheaper padded to 4 than run one at a time.
            const auto &P = r.ab->primitive_pairs, &Q = r.cd->primitive_pairs;
            const std::size_t K = P.size() * Q.size();
            const auto pass = [&]<int B>()
            {
                grow_to(VL, static_cast<std::size_t>(Ne) * Nf * Md * B);
                const PrimitivePair *pa[B], *pc[B];
                double w[B];
                for (std::size_t k0 = 0; k0 < K; k0 += B)
                {
                    for (int l = 0; l < B; ++l)
                    {
                        // A padding lane repeats a valid quartet at weight 0.
                        const std::size_t k = k0 + l < K ? k0 + l : k0;
                        pa[l] = &P[k / Q.size()];
                        pc[l] = &Q[k % Q.size()];
                        w[l] = k0 + l < K ? pa[l]->coeff_product * pc[l]->coeff_product : 0.0;
                    }
                    hgp_block::vrr_lanes<B>(pa, pc, LAB, LCD, VL.data());
                    for (int e = e0; e < Ne; ++e)
                        for (int f = f0; f < Nf; ++f)
                        {
                            const double *x = &VL[(static_cast<std::size_t>(e) * Nf + f) * Md * B];
                            double sum = 0.0;
                            for (int l = 0; l < B; ++l)
                                sum += w[l] * x[l];
                            acc[static_cast<std::size_t>(e) * Nf + f] += sum;
                        }
                }
            };
            if (K >= hgp_block::kMinLaneK)
                pass.template operator()<hgp_block::kLanes>();
            else
                pass.template operator()<1>();

            finish(r);
            return true;
        }

        // HRR and readout from `acc` ([e][f], |e| in [LA, LA+LB], |f| in
        // [LC, LC+LD], row stride n_upto(LC+LD)) for request r. prepare() calls
        // it after its primitive loop; a caller that fills acc itself (the D2a
        // family probe) calls it directly.
        void finish(const BlockEriRequest &r)
        {
            req_ = &r;
            using hgp_block::n_upto;
            const auto &T = hgp_block::table();
            const int LA = r.L[0], LB = r.L[1], LC = r.L[2], LD = r.L[3];
            const int LAB = LA + LB, LCD = LC + LD;
            const int Ne = n_upto(LAB), Nf = n_upto(LCD);
            const int e0 = n_upto(LA - 1), f0 = n_upto(LC - 1);

            // Bra HRR: H[a][b][f - f0] for |a| in [LA, LAB - |b|], |b| <= LB.
            const int Nb = n_upto(LB), Nd = n_upto(LD), nfk = Nf - f0;
            grow_to(H, static_cast<std::size_t>(Ne) * Nb * nfk);
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
            grow_to(out_, n_[0] * n_[1] * n_[2] * n_[3]);
            grow_to(K, static_cast<std::size_t>(Nf) * Nd);
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
        }

        // The contracted [e0|f0] that finish() reads (D2a probe fills it directly).
        std::vector<double> &acc_buffer() noexcept { return acc; }

        double value(std::size_t ca, std::size_t cb, std::size_t cc, std::size_t cd) const noexcept
        {
            const double *nr = req_->norm_ratio;
            return out_[((ca * n_[1] + cb) * n_[2] + cc) * n_[3] + cd] *
                   nr[req_->views[0][ca]->_index] * nr[req_->views[1][cb]->_index] *
                   nr[req_->views[2][cc]->_index] * nr[req_->views[3][cd]->_index];
        }

    private:
        std::vector<double> VL; // lane-minor VRR table for vrr_lanes

        const BlockEriRequest *req_ = nullptr;
        // Scratch, reused across prepare() calls; one kernel copy per thread
        // (fused_fock.h), so members rather than thread_local (emulated TLS on
        // macOS g++ was 2.5-4.7 % of d/f builds).
        std::vector<double> acc, H, K;
        std::size_t n_[4] = {};
        std::vector<double> out_;
    };
}
