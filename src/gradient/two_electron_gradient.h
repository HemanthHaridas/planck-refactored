// two_electron_gradient.h — the two-electron part of an analytic gradient,
// shared by every SCF reference (gradient.cpp) and the MP2 gradients
// (mp2_gradient.cpp). The caller supplies only Γ through gamma_fn; the result
// is ¼ Σ over ordered AO quartets of Γ(p,q,r,s) d(pq|rs)/dR
// (docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md).
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/types.h"
#include "gradient/gradient.h"
#include "integrals/auto_deriv_block_eri.h"
#include "integrals/fused_fock.h"

namespace HartreeFock::Gradient
{
    namespace detail
    {
        using HartreeFock::Gradient::compute_eri_deriv_dispatch;

        // Shell index -> atom index (shell centre within 1e-5 Bohr of the atom).
        std::expected<std::vector<int>, std::string> build_shell_atom_map(const HartreeFock::Calculator &calc);

        // Q(i, j) = sqrt|(ij|ij)| for every AO pair in shell_pairs.
        Eigen::MatrixXd build_pair_schwarz_table(const std::vector<HartreeFock::ShellPair> &shell_pairs,
                                                 std::size_t nbasis);

        // Γ summed over the (ij)/(kl) index swaps of one quartet. Every swap sends
        // each centre's derivative to that centre's own atom, so the swaps differ only
        // in their Γ weight and can share one dI.
        template <typename GammaFn>
        inline double pair_permuted_gamma(
            const GammaFn &gamma_fn,
            std::size_t ii,
            std::size_t jj,
            std::size_t kk,
            std::size_t ll)
        {
            double gamma = gamma_fn(ii, jj, kk, ll);
            if (kk != ll)
                gamma += gamma_fn(ii, jj, ll, kk);
            if (ii != jj)
                gamma += gamma_fn(jj, ii, kk, ll);
            if (ii != jj && kk != ll)
                gamma += gamma_fn(jj, ii, ll, kk);
            return gamma;
        }

        // One quartet's contribution: each centre's derivative goes to its own atom.
        inline void add_quartet_gradient(Eigen::MatrixXd &g, double gamma, const std::array<double, 12> &dI,
                                         int atom_A, int atom_B, int atom_C, int atom_D)
        {
            const double fac = 0.25 * gamma;
            for (int c = 0; c < 3; ++c)
            {
                g(atom_A, c) += fac * dI[c];
                g(atom_B, c) += fac * dI[3 + c];
                g(atom_C, c) += fac * dI[6 + c];
                g(atom_D, c) += fac * dI[9 + c];
            }
        }

        // Bra units (AO pairs, or shell-group pairs) are dealt into kBins fixed bins
        // (unit mod kBins), dynamically scheduled; each bin owns a gradient partial and
        // the partials are summed in bin order, so the result does not depend on the
        // thread count.
        inline constexpr int kGradientBins = 64;

        // The 2e gradient by shell quartet (docs/GRADIENT_INTEGRAL_PERFORMANCE_SCOPE.md,
        // G3): each block's derivatives come from one primitive pass of the Auto
        // derivative chain, and blocks it declines fall back to the per-component
        // derivative. Visits exactly the AO quartets of the AO loop below: shell-group
        // pairs Q >= P, within a group i <= j, and in a diagonal block (P == Q) each
        // unordered pair of AO pairs once.
        template <typename GammaFn>
        void accumulate_block_eri_gradient(
            const HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const std::vector<int> &shell_atom,
            const std::vector<int> &bf_shell,
            const Eigen::MatrixXd &schwarz_q,
            Eigen::MatrixXd &grad,
            const GammaFn &gamma_fn)
        {
            using HartreeFock::Integrals::BlockEriRequest;
            const std::size_t nb = calc._shells.nbasis();
            const double tol_eri = calc._integral._tol_eri;

            std::vector<const HartreeFock::ContractedView *> views;
            const auto groups = HartreeFock::Integrals::fused_shell_groups(shell_pairs, nb, views);
            const std::size_t ng = groups.size();

            // First-component pairs feed the kernel, whose outputs are rescaled by each
            // component's norm relative to the first (exactly 1 for s and p).
            std::vector<const HartreeFock::ShellPair *> pair_of(nb * nb, nullptr);
            for (const auto &sp : shell_pairs)
                pair_of[sp.A._index * nb + sp.B._index] = &sp;
            std::vector<int> group_L(ng), group_atom(ng);
            std::vector<double> norm_ratio(nb, 1.0);
            for (std::size_t g = 0; g < ng; ++g)
            {
                const auto &first = *views[groups[g].first_ao];
                group_L[g] = first._cartesian[0] + first._cartesian[1] + first._cartesian[2];
                group_atom[g] = shell_atom[bf_shell[groups[g].first_ao]];
                for (std::size_t c = 0; c < groups[g].n_components; ++c)
                    norm_ratio[groups[g].first_ao + c] =
                        views[groups[g].first_ao + c]->_component_norm / first._component_norm;
            }

            struct GroupPair
            {
                std::size_t a, b;
                double qmax;
            };
            std::vector<GroupPair> gpairs;
            for (std::size_t a = 0; a < ng; ++a)
                for (std::size_t b = a; b < ng; ++b)
                {
                    const auto &ga = groups[a], &gb = groups[b];
                    const double q = schwarz_q.block(ga.first_ao, gb.first_ao, ga.n_components, gb.n_components)
                                         .maxCoeff();
                    gpairs.push_back({a, b, q});
                }
            const std::size_t ngp = gpairs.size();

            std::vector<Eigen::MatrixXd> partial(kGradientBins, Eigen::MatrixXd::Zero(grad.rows(), 3));
        #pragma omp parallel
            {
                auto kernel = HartreeFock::Integrals::auto_deriv_kernel();
                BlockEriRequest r;
                r.norm_ratio = norm_ratio.data();
                std::array<double, 12> dI{};

        #pragma omp for schedule(dynamic, 1)
                for (int bin = 0; bin < kGradientBins; ++bin)
                {
                    Eigen::MatrixXd &g = partial[bin];
                    for (std::size_t P = bin; P < ngp; P += kGradientBins)
                        for (std::size_t Q = P; Q < ngp; ++Q)
                        {
                            if (gpairs[P].qmax * gpairs[Q].qmax < tol_eri)
                                continue;
                            const std::size_t s4[4] = {gpairs[P].a, gpairs[P].b, gpairs[Q].a, gpairs[Q].b};
                            r.ab = pair_of[groups[s4[0]].first_ao * nb + groups[s4[1]].first_ao];
                            r.cd = pair_of[groups[s4[2]].first_ao * nb + groups[s4[3]].first_ao];
                            for (int s = 0; s < 4; ++s)
                            {
                                r.views[s] = &views[groups[s4[s]].first_ao];
                                r.n[s] = groups[s4[s]].n_components;
                                r.L[s] = group_L[s4[s]];
                            }
                            bool prepared = false, block_ok = false;

                            for (std::size_t ca = 0; ca < r.n[0]; ++ca)
                                for (std::size_t cb = 0; cb < r.n[1]; ++cb)
                                {
                                    const std::size_t i = groups[s4[0]].first_ao + ca;
                                    const std::size_t j = groups[s4[1]].first_ao + cb;
                                    if (i > j)
                                        continue;
                                    for (std::size_t cc = 0; cc < r.n[2]; ++cc)
                                        for (std::size_t cd = 0; cd < r.n[3]; ++cd)
                                        {
                                            const std::size_t k = groups[s4[2]].first_ao + cc;
                                            const std::size_t l = groups[s4[3]].first_ao + cd;
                                            if (k > l || (P == Q && i * nb + j > k * nb + l))
                                                continue;
                                            if (schwarz_q(i, j) * schwarz_q(k, l) < tol_eri)
                                                continue;
                                            double gamma = pair_permuted_gamma(gamma_fn, i, j, k, l);
                                            if (i != k || j != l)
                                                gamma += pair_permuted_gamma(gamma_fn, k, l, i, j);
                                            if (std::abs(gamma) < 1e-14)
                                                continue;

                                            if (!prepared)
                                            {
                                                block_ok = kernel.prepare(r);
                                                prepared = true;
                                            }
                                            if (block_ok)
                                                kernel.deriv(ca, cb, cc, cd, dI);
                                            else
                                                dI = compute_eri_deriv_dispatch(
                                                    calc, *pair_of[i * nb + j], *pair_of[k * nb + l]);
                                            add_quartet_gradient(g, gamma, dI, group_atom[s4[0]], group_atom[s4[1]],
                                                                 group_atom[s4[2]], group_atom[s4[3]]);
                                        }
                                }
                        }
                }
            }

            for (const auto &g : partial)
                grad += g;
        }

        template <typename GammaFn>
        void accumulate_shell_pair_eri_gradient(
            const HartreeFock::Calculator &calc,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const std::vector<int> &shell_atom,
            const std::vector<int> &bf_shell,
            const Eigen::MatrixXd &schwarz_q,
            Eigen::MatrixXd &grad,
            const GammaFn &gamma_fn,
            HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
            double omega = 0.0)
        {
            // Auto + Coulomb takes the shell-quartet path; explicit engines and screened
            // kernels keep the per-component AO loop (so the cross-engine comparators
            // still compare engines).
            if (calc._integral._engine == HartreeFock::IntegralMethod::Auto &&
                kernel == HartreeFock::ERIKernel::Coulomb)
            {
                accumulate_block_eri_gradient(calc, shell_pairs, shell_atom, bf_shell, schwarz_q, grad, gamma_fn);
                return;
            }

            // shell_pairs holds each unordered AO pair once, so visiting q >= p covers
            // each unordered pair of pairs once; the bra-ket swap (q, p) shares dI and
            // only adds its Γ. Any Γ works, symmetric or not.
            constexpr int kBins = kGradientBins;
            const std::size_t npairs = shell_pairs.size();
            const double tol_eri = calc._integral._tol_eri;
            std::vector<Eigen::MatrixXd> partial(kBins, Eigen::MatrixXd::Zero(grad.rows(), 3));

        #pragma omp parallel for schedule(dynamic, 1)
            for (int bin = 0; bin < kBins; ++bin)
            {
                Eigen::MatrixXd &g = partial[bin];
                for (std::size_t p = bin; p < npairs; p += kBins)
                {
                    const auto &spAB = shell_pairs[p];
                    const std::size_t ii = spAB.A._index;
                    const std::size_t jj = spAB.B._index;
                    const int atom_A = shell_atom[bf_shell[ii]];
                    const int atom_B = shell_atom[bf_shell[jj]];

                    for (std::size_t q = p; q < npairs; ++q)
                    {
                        const auto &spCD = shell_pairs[q];
                        const std::size_t kk = spCD.A._index;
                        const std::size_t ll = spCD.B._index;

                        if (schwarz_q(ii, jj) * schwarz_q(kk, ll) < tol_eri)
                            continue;

                        double gamma = pair_permuted_gamma(gamma_fn, ii, jj, kk, ll);
                        if (q != p)
                            gamma += pair_permuted_gamma(gamma_fn, kk, ll, ii, jj);
                        if (std::abs(gamma) < 1e-14)
                            continue;

                        const int atom_C = shell_atom[bf_shell[kk]];
                        const int atom_D = shell_atom[bf_shell[ll]];
                        const auto dI = compute_eri_deriv_dispatch(
                            calc, spAB, spCD, kernel, omega);
                        add_quartet_gradient(g, gamma, dI, atom_A, atom_B, atom_C, atom_D);
                    }
                }
            }

            for (const auto &g : partial)
                grad += g;
        }
    }

    // ¼ Σ_pqrs Γ(p,q,r,s) d(pq|rs)/dR over ordered AO quartets, as a natoms x 3
    // matrix. Γ need not be symmetric.
    template <typename GammaFn>
    std::expected<Eigen::MatrixXd, std::string> two_electron_gradient(
        const HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const GammaFn &gamma_fn,
        HartreeFock::ERIKernel kernel = HartreeFock::ERIKernel::Coulomb,
        double omega = 0.0)
    {
        const auto &basis = calc._shells;
        const std::size_t nb = basis.nbasis();
        Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(calc._molecule.natoms, 3);

        auto shell_atom_res = detail::build_shell_atom_map(calc);
        if (!shell_atom_res)
            return std::unexpected(shell_atom_res.error());
        const std::vector<int> shell_atom = std::move(*shell_atom_res);
        const auto &shells = basis._shells;
        const auto &bfs = basis._basis_functions;
        const std::size_t nshells = shells.size();
        const Eigen::MatrixXd schwarz_q = detail::build_pair_schwarz_table(shell_pairs, nb);

        std::vector<int> bf_shell(nb, -1);
        for (std::size_t s = 0; s < nshells; ++s)
            for (std::size_t mu = 0; mu < nb; ++mu)
                if (bfs[mu]._shell == &shells[s])
                    bf_shell[mu] = static_cast<int>(s);

        detail::accumulate_shell_pair_eri_gradient(
            calc,
            shell_pairs,
            shell_atom,
            bf_shell,
            schwarz_q,
            grad,
            gamma_fn,
            kernel,
            omega);
        return grad;
    }
}
