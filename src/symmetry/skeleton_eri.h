#ifndef HF_SKELETON_ERI_H
#define HF_SKELETON_ERI_H

#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "base/types.h"
#include "integrals/shellpair.h"
#include "symmetry/group_operations.h"

// ─── Engine-agnostic skeleton ERI build for the full-symmetry direct Fock ───────
//
// The petite-list / orbit-multiplicity / 8-fold-scatter logic is identical for the
// Obara-Saika and Rys engines — they differ ONLY in the contracted-ERI primitive.
// This header factors out the shared machinery so os_symm.cpp and rys_symm.cpp stay
// thin (each just supplies its engine's ERI callable) while remaining separate
// translation units for A/B benchmarking. See docs/FULL_SYMMETRY_ERI_DESIGN.md.

namespace HartreeFock
{
    namespace Symmetry
    {
        namespace detail
        {
            // Canonicalize a shell quartet (sa,sb|sc,sd) under the 8-fold
            // permutational ERI symmetry to its lexicographic-minimum key.
            inline std::array<int, 4> canon_quartet(int sa, int sb, int sc, int sd)
            {
                if (sa < sb)
                    std::swap(sa, sb);
                if (sc < sd)
                    std::swap(sc, sd);
                if (std::tie(sa, sb) < std::tie(sc, sd))
                {
                    std::swap(sa, sc);
                    std::swap(sb, sd);
                }
                return {sa, sb, sc, sd};
            }

            // Representative iff no group image yields a strictly smaller canonical key.
            inline bool is_quartet_representative(int sa, int sb, int sc, int sd,
                                                 const GroupOperations &ops)
            {
                const std::array<int, 4> self = canon_quartet(sa, sb, sc, sd);
                for (const auto &op : ops.operations)
                {
                    const auto &P = op.shell_perm;
                    if (canon_quartet(P[sa], P[sb], P[sc], P[sd]) < self)
                        return false;
                }
                return true;
            }

            // Number of DISTINCT shell quartets in the orbit (each counted once).
            inline int orbit_multiplicity(int sa, int sb, int sc, int sd,
                                         const GroupOperations &ops)
            {
                std::vector<std::array<int, 4>> seen;
                seen.reserve(ops.operations.size());
                for (const auto &op : ops.operations)
                {
                    const auto &P = op.shell_perm;
                    const std::array<int, 4> img = canon_quartet(P[sa], P[sb], P[sc], P[sd]);
                    bool found = false;
                    for (const auto &e : seen)
                        if (e == img)
                        {
                            found = true;
                            break;
                        }
                    if (!found)
                        seen.push_back(img);
                }
                return static_cast<int>(seen.size());
            }

            // Store-only 8-fold permutational scatter into the dense ERI tensor.
            // Like the production path (os.cpp write_eri_permutations), every writer
            // stores the SAME value into a given canonical slot, so `atomic write`
            // (a plain store), not `atomic update` (a reduction), is the correct
            // guard when the outer petite-list loop is parallelized: distinct orbit
            // representatives own disjoint slots, so the only race is a benign
            // same-value store on the 8-fold images.
            inline void scatter8(std::vector<double> &eri, std::size_t nb,
                                std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                                double val)
            {
                const std::size_t nb2 = nb * nb, nb3 = nb * nb * nb;
                const std::array<std::array<std::size_t, 4>, 8> perms = {{{i, j, k, l},
                                                                         {j, i, k, l},
                                                                         {i, j, l, k},
                                                                         {j, i, l, k},
                                                                         {k, l, i, j},
                                                                         {l, k, i, j},
                                                                         {k, l, j, i},
                                                                         {l, k, j, i}}};
                for (const auto &pm : perms)
                {
                    const std::size_t idx = pm[0] * nb3 + pm[1] * nb2 + pm[2] * nb + pm[3];
#ifdef USE_OPENMP
#pragma omp atomic write
#endif
                    eri[idx] = val;
                }
            }
        } // namespace detail

        // Build the Schwarz screening table Q(i,j) = sqrt(|(ij|ij)|) from the shell
        // pairs, using the engine's own ERI primitive for the diagonal so the bound
        // matches the integrals it screens. Bounds any quartet by Cauchy-Schwarz:
        // |(ij|kl)| ≤ Q(i,j)·Q(k,l). Symmetry-independent — every pair (i,j) and its
        // transpose (j,i) is filled directly, so no orbit bookkeeping is needed here.
        template <typename EriFn>
        Eigen::MatrixXd build_schwarz_table(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            std::size_t nb,
            EriFn &&eri_fn)
        {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nb, nb);
            for (const auto &sp : shell_pairs)
            {
                const std::size_t i = sp.A._index;
                const std::size_t j = sp.B._index;
                const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
                const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];

                const double diag = eri_fn(sp, sp,
                                           lAx, lAy, lAz, lBx, lBy, lBz,
                                           lAx, lAy, lAz, lBx, lBy, lBz);
                const double q = std::sqrt(std::abs(diag));
                Q(i, j) = q;
                Q(j, i) = q;
            }
            return Q;
        }

        // Build the (orbit-weighted) skeleton ERI tensor over the petite list.
        // `eri_fn(spAB, spCD, lA.., lB.., lC.., lD..)` returns the contracted ERI for
        // one Cartesian-component quartet — supply ObaraSaika::_contracted_eri_elem
        // or RysQuad::_rys_contracted_eri. When use_sym is false every quartet is
        // computed once → skeleton == full tensor (engine-faithful, no reduction).
        //
        // Parallelism + screening (docs/FULL_SYMMETRY_ERI_DESIGN.md §8.4 item 5):
        //   * Schwarz: quartets with Q(i,j)·Q(k,l) < tol_eri are skipped before the
        //     ERI is computed (same bound the production path uses).
        //   * OpenMP: the outer petite-list pair loop is parallelized with dynamic
        //     scheduling. Distinct orbit representatives own disjoint tensor slots,
        //     so scatter8's same-value atomic stores carry no reduction hazard.
        template <typename EriFn>
        std::expected<std::vector<double>, std::string> build_skeleton_eri(
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const HartreeFock::Basis &basis,
            std::size_t nb,
            const GroupOperations &ops,
            bool use_sym,
            EriFn &&eri_fn,
            double tol_eri = 1e-10)
        {
            std::map<const HartreeFock::Shell *, int> shell_id;
            for (int s = 0; s < static_cast<int>(basis._shells.size()); ++s)
                shell_id[&basis._shells[s]] = s;

            const Eigen::MatrixXd Q = build_schwarz_table(shell_pairs, nb, eri_fn);

            std::vector<double> eri(nb * nb * nb * nb, 0.0);
            const std::size_t npairs = shell_pairs.size();

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (std::size_t p = 0; p < npairs; ++p)
            {
                const auto &spAB = shell_pairs[p];
                const int sa = shell_id.at(spAB.A._shell);
                const int sb = shell_id.at(spAB.B._shell);
                const std::size_t i = spAB.A._index;
                const std::size_t j = spAB.B._index;
                const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
                const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

                for (std::size_t q = p; q < npairs; ++q)
                {
                    const auto &spCD = shell_pairs[q];
                    const int sc = shell_id.at(spCD.A._shell);
                    const int sd = shell_id.at(spCD.B._shell);
                    const std::size_t k = spCD.A._index;
                    const std::size_t l = spCD.B._index;

                    // Schwarz screening: |(ij|kl)| ≤ Q(i,j)·Q(k,l).
                    if (Q(i, j) * Q(k, l) < tol_eri)
                        continue;

                    if (use_sym && !detail::is_quartet_representative(sa, sb, sc, sd, ops))
                        continue;

                    const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
                    const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

                    double val = eri_fn(spAB, spCD,
                                        lAx, lAy, lAz, lBx, lBy, lBz,
                                        lCx, lCy, lCz, lDx, lDy, lDz);

                    if (use_sym)
                        val *= static_cast<double>(detail::orbit_multiplicity(sa, sb, sc, sd, ops));

                    detail::scatter8(eri, nb, i, j, k, l, val);
                }
            }
            return eri;
        }

        // Contract a (skeleton or full) ERI tensor to the RHF Fock G = J − ½K.
        // Parallel over μ: thread μ owns row μ of G → no shared writes (mirrors the
        // production Phase-2 contraction in os.cpp).
        inline Eigen::MatrixXd contract_fock_rhf(const std::vector<double> &eri,
                                                std::size_t nb, const Eigen::MatrixXd &density)
        {
            const std::size_t nb2 = nb * nb, nb3 = nb * nb * nb;
            Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::size_t mu = 0; mu < nb; ++mu)
                for (std::size_t nu = 0; nu < nb; ++nu)
                    for (std::size_t lam = 0; lam < nb; ++lam)
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            G(mu, nu) += density(lam, sig) *
                                         (eri[mu * nb3 + nu * nb2 + lam * nb + sig] -
                                          0.5 * eri[mu * nb3 + lam * nb2 + nu * nb + sig]);
            return G;
        }

        // Contract to the UHF spin Focks {G_alpha, G_beta}.
        inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> contract_fock_uhf(
            const std::vector<double> &eri, std::size_t nb,
            const Eigen::MatrixXd &Pa, const Eigen::MatrixXd &Pb)
        {
            const std::size_t nb2 = nb * nb, nb3 = nb * nb * nb;
            const Eigen::MatrixXd Pt = Pa + Pb;
            Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
            Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::size_t mu = 0; mu < nb; ++mu)
                for (std::size_t nu = 0; nu < nb; ++nu)
                    for (std::size_t lam = 0; lam < nb; ++lam)
                        for (std::size_t sig = 0; sig < nb; ++sig)
                        {
                            const double coulomb = eri[mu * nb3 + nu * nb2 + lam * nb + sig];
                            const double exch = eri[mu * nb3 + lam * nb2 + nu * nb + sig];
                            Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                            Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                        }
            return {Ga, Gb};
        }
    } // namespace Symmetry
} // namespace HartreeFock

#endif // !HF_SKELETON_ERI_H
