// fused_fock.h — the shared memory-direct Fock driver.
//
// One shell-quartet loop, parameterized on the engine's per-quartet contracted-
// ERI callable. Each canonical quartet is contracted straight into the Fock
// matrix (fock_accumulate.h) instead of being scattered into an nb^4 tensor that
// a second nb^4 sweep then contracts. Nothing nb^4 is ever allocated.
//
// Why one loop instead of one per engine: OS, HGP, Rys, and Rys-Auto all had the
// identical two-phase Fock builder (build the full tensor, then contract it) and
// the identical shell-quartet traversal. They differ in exactly one expression —
// which per-quartet function computes the contracted ERI — and all four of those
// have the same argument list:
//
//   ObaraSaika::_contracted_eri        HeadGordonPople::_contracted_eri_elem
//   RysQuad::_rys_contracted_eri       (Auto: _auto_contracted_eri)
//
// So the engine enters as a callable and the loop is written once.
//
// Two invariants this loop depends on, both load-bearing:
//
//  1. Canonical filter (j>=i, l>=k, (k,l) >=lex (i,j)) — each canonical quartet
//     is visited exactly ONCE, which is what makes the unweighted 8-fold-orbit
//     accumulation in fock_accumulate.h correct.
//
//  2. schedule(static) + per-thread partials summed in FIXED thread-index order.
//     These accumulations are read-modify-write, unlike the store-only
//     write_eri_permutations of the tensor build. A critical/atomic reduction,
//     or schedule(dynamic), makes the result drift with thread count (measured);
//     see the note in the loop and src/dft/ks_matrix.cpp.
//
// Integral symmetry (sym_ops) is NOT handled here: that path replicates a
// symmetry orbit on top of the permutational orbit, and deduplicating across
// both needs its own correctness argument. Callers must delegate to the
// two-phase builder when symmetry ops are active.
#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "base/types.h"
#include "fock_accumulate.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace HartreeFock::Integrals
{
    // A run of AO components sharing one shell (the (L+1)(L+2)/2 Cartesian
    // components). Replaces the three near-identical Os/Hgp/RysShellGroup structs
    // for the fused path.
    struct FusedShellGroup
    {
        std::size_t first_ao = 0;     // _index of component 0
        std::size_t n_components = 0; // (L+1)(L+2)/2
    };

    inline std::vector<FusedShellGroup> fused_shell_groups(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::size_t nbasis,
        std::vector<const HartreeFock::ContractedView *> &ao_views)
    {
        ao_views.assign(nbasis, nullptr);
        for (const auto &sp : shell_pairs)
        {
            if (sp.A._index < nbasis)
                ao_views[sp.A._index] = &sp.A;
            if (sp.B._index < nbasis)
                ao_views[sp.B._index] = &sp.B;
        }

        std::vector<FusedShellGroup> groups;
        const HartreeFock::Shell *current = nullptr;
        for (std::size_t ao = 0; ao < nbasis; ++ao)
        {
            const HartreeFock::ContractedView *view = ao_views[ao];
            const HartreeFock::Shell *shell = view ? view->_shell : nullptr;
            if (groups.empty() || shell != current)
            {
                groups.push_back({ao, 1});
                current = shell;
            }
            else
            {
                ++groups.back().n_components;
            }
        }
        return groups;
    }

    // Densities the fused driver contracts against. For RHF only `P` is read;
    // for UHF only `Pt`/`Pa`/`Pb`.
    struct FusedFockDensities
    {
        const Eigen::MatrixXd *P = nullptr;  // RHF
        const Eigen::MatrixXd *Pt = nullptr; // Pa + Pb
        const Eigen::MatrixXd *Pa = nullptr;
        const Eigen::MatrixXd *Pb = nullptr;
    };

    // `eri_elem(spAB, spCD, lAx..lDz, kernel, omega) -> double` is the engine's
    // per-quartet contracted ERI. Fills G (RHF) or Ga/Gb (UHF).
    template <typename EriElem>
    void fused_fock_build(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::size_t nb,
        const Eigen::MatrixXd &Q, // Schwarz table
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri,
        bool spin_resolved,
        const FusedFockDensities &dens,
        Eigen::MatrixXd &G,
        Eigen::MatrixXd &Ga,
        Eigen::MatrixXd &Gb,
        EriElem &&eri_elem)
    {
        std::vector<const HartreeFock::ContractedView *> ao_views;
        const std::vector<FusedShellGroup> groups =
            fused_shell_groups(shell_pairs, nb, ao_views);
        const std::size_t ngroups = groups.size();

        struct GroupPair
        {
            std::size_t a;
            std::size_t b;
        };
        std::vector<GroupPair> group_pairs;
        group_pairs.reserve(ngroups * (ngroups + 1) / 2);
        for (std::size_t sa = 0; sa < ngroups; ++sa)
            for (std::size_t sb = sa; sb < ngroups; ++sb)
                group_pairs.push_back({sa, sb});

        const std::size_t ngp = group_pairs.size();

#ifdef USE_OPENMP
        const int n_threads = omp_get_max_threads();
#else
        const int n_threads = 1;
#endif
        const bool need_uhf = spin_resolved;
        std::vector<Eigen::MatrixXd> g_partials(
            need_uhf ? 0 : static_cast<std::size_t>(n_threads),
            Eigen::MatrixXd::Zero(nb, nb));
        std::vector<Eigen::MatrixXd> ga_partials(
            need_uhf ? static_cast<std::size_t>(n_threads) : 0,
            Eigen::MatrixXd::Zero(nb, nb));
        std::vector<Eigen::MatrixXd> gb_partials(
            need_uhf ? static_cast<std::size_t>(n_threads) : 0,
            Eigen::MatrixXd::Zero(nb, nb));

        // schedule(static), NOT dynamic — load-bearing, see the header note.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t bra = 0; bra < ngp; ++bra)
        {
#ifdef USE_OPENMP
            const int thread_id = omp_get_thread_num();
#else
            const int thread_id = 0;
#endif
            // In the RHF branch only g_partials is populated and only
            // fock_accumulate_rhf runs, so the beta reference is never read; it
            // aliases the alpha slot purely to keep both branches in one loop.
            Eigen::MatrixXd &G_local =
                need_uhf ? ga_partials[thread_id] : g_partials[thread_id];
            Eigen::MatrixXd &Gb_local =
                need_uhf ? gb_partials[thread_id] : g_partials[thread_id];

            const FusedShellGroup &gA = groups[group_pairs[bra].a];
            const FusedShellGroup &gB = groups[group_pairs[bra].b];

            for (std::size_t ket = 0; ket < ngp; ++ket)
            {
                const FusedShellGroup &gC = groups[group_pairs[ket].a];
                const FusedShellGroup &gD = groups[group_pairs[ket].b];

                for (std::size_t ca = 0; ca < gA.n_components; ++ca)
                {
                    const HartreeFock::ContractedView &cvA = *ao_views[gA.first_ao + ca];
                    const std::size_t i = cvA._index;
                    const int lAx = cvA._cartesian[0], lAy = cvA._cartesian[1], lAz = cvA._cartesian[2];

                    for (std::size_t cb = 0; cb < gB.n_components; ++cb)
                    {
                        const HartreeFock::ContractedView &cvB = *ao_views[gB.first_ao + cb];
                        const std::size_t j = cvB._index;
                        if (j < i) // bra upper triangle
                            continue;
                        const int lBx = cvB._cartesian[0], lBy = cvB._cartesian[1], lBz = cvB._cartesian[2];

                        const HartreeFock::ShellPair spAB(cvA, cvB);

                        for (std::size_t cc = 0; cc < gC.n_components; ++cc)
                        {
                            const HartreeFock::ContractedView &cvC = *ao_views[gC.first_ao + cc];
                            const std::size_t k = cvC._index;
                            const int lCx = cvC._cartesian[0], lCy = cvC._cartesian[1], lCz = cvC._cartesian[2];

                            for (std::size_t cd = 0; cd < gD.n_components; ++cd)
                            {
                                const HartreeFock::ContractedView &cvD = *ao_views[gD.first_ao + cd];
                                const std::size_t l = cvD._index;
                                if (l < k) // ket upper triangle
                                    continue;
                                if (k < i || (k == i && l < j)) // bra-ket canonical
                                    continue;

                                const int lDx = cvD._cartesian[0], lDy = cvD._cartesian[1], lDz = cvD._cartesian[2];

                                // Schwarz screening. With no tensor to fill, a
                                // screened quartet is simply never contracted —
                                // it contributes nothing to G, exactly as the
                                // stored zero it would have been.
                                if (Q(i, j) * Q(k, l) < tol_eri)
                                    continue;

                                const HartreeFock::ShellPair spCD(cvC, cvD);
                                const double val = eri_elem(
                                    spAB, spCD,
                                    lAx, lAy, lAz, lBx, lBy, lBz,
                                    lCx, lCy, lCz, lDx, lDy, lDz,
                                    kernel, omega);

                                if (need_uhf)
                                    fock_accumulate_uhf(G_local, Gb_local,
                                                        *dens.Pt, *dens.Pa, *dens.Pb,
                                                        i, j, k, l, val);
                                else
                                    fock_accumulate_rhf(G_local, *dens.P,
                                                        i, j, k, l, val);
                            }
                        }
                    }
                }
            }
        }

        // Fixed thread-index order — never completion order.
        if (need_uhf)
        {
            Ga.setZero(nb, nb);
            Gb.setZero(nb, nb);
            for (int t = 0; t < n_threads; ++t)
            {
                Ga += ga_partials[static_cast<std::size_t>(t)];
                Gb += gb_partials[static_cast<std::size_t>(t)];
            }
        }
        else
        {
            G.setZero(nb, nb);
            for (int t = 0; t < n_threads; ++t)
                G += g_partials[static_cast<std::size_t>(t)];
        }
    }
}
