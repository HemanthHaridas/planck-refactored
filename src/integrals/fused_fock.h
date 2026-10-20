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
// Integral symmetry (sym_ops) IS handled: the ERI is computed once at each
// symmetry-orbit representative and replicated across the orbit with the
// accumulated AO sign, exactly as the tensor build's scatter does. The two
// dedups compose without double-counting — see the dedup argument in
// quartet_orbit.h. When every density is itself symmetric, the loop instead
// accumulates each representative once, weighted by its orbit size, and
// symmetrizes G at the end (skeleton mode, S1). Passing sym_ops = nullptr (the
// default) disables both.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "base/mpi_env.h"
#include "base/types.h"
#include "block_eri.h"
#include "fock_accumulate.h"
#include "low_l_eri.h"
#include "quartet_orbit.h"

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


    // The terminal accumulation, the one thing that varies between the three
    // FusedTerm modes. Everything upstream of it in the quartet loop is shared.
    //
    // The single-term modes accumulate into `G` for RHF. For UHF they fill both
    // channels from their own spin density (Pa -> G, Pb -> Gb), so a UKS caller
    // gets K_alpha and K_beta from one sweep, matching how the combined mode
    // fills Ga/Gb. Coulomb is spin-independent and uses the total density in
    // both channels, so a UHF J call returns the same matrix twice.
    inline void accumulate_term(
        FusedTerm term,
        bool need_uhf,
        Eigen::MatrixXd &G,
        Eigen::MatrixXd &Gb,
        const FusedFockDensities &dens,
        std::size_t i, std::size_t j, std::size_t k, std::size_t l,
        double val) noexcept
    {
        switch (term)
        {
        case FusedTerm::RawCoulombExchange:
            // This mode is intentionally restricted: its second matrix is
            // the paired raw-K output, whereas the UHF slot convention is
            // alpha/beta. UHF retains its existing one-sweep path.
            if (!need_uhf)
            {
                coulomb_accumulate(G, *dens.P, i, j, k, l, val);
                exchange_accumulate(Gb, *dens.P, i, j, k, l, val);
            }
            return;

        case FusedTerm::CoulombOnly:
            if (need_uhf)
            {
                coulomb_accumulate(G, *dens.Pt, i, j, k, l, val);
                coulomb_accumulate(Gb, *dens.Pt, i, j, k, l, val);
            }
            else
                coulomb_accumulate(G, *dens.P, i, j, k, l, val);
            return;

        case FusedTerm::ExchangeOnly:
            if (need_uhf)
            {
                exchange_accumulate(G, *dens.Pa, i, j, k, l, val);
                exchange_accumulate(Gb, *dens.Pb, i, j, k, l, val);
            }
            else
                exchange_accumulate(G, *dens.P, i, j, k, l, val);
            return;

        case FusedTerm::Combined:
        default:
            if (need_uhf)
                fock_accumulate_uhf(G, Gb, *dens.Pt, *dens.Pa, *dens.Pb,
                                    i, j, k, l, val);
            else
                fock_accumulate_rhf(G, *dens.P, i, j, k, l, val);
            return;
        }
    }

    // `eri_elem(spAB, spCD, lAx..lDz, kernel, omega) -> double` is the engine's
    // per-quartet contracted ERI. Fills G (RHF) or Ga/Gb (UHF).
    //
    // `term` selects which term(s) land in the output; it defaults to Combined
    // so every existing HF call site is unchanged.
    //
    // `block_kernel` (block_eri.h) evaluates a whole shell quartet at once for
    // the classes it accepts; every other block uses eri_elem per component.
    // The default NoBlockKernel keeps an engine pure. A copy lives per bra.
    template <typename EriElem, typename BlockKernel = NoBlockKernel>
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
        EriElem &&eri_elem,
        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr,
        FusedTerm term = FusedTerm::Combined,
        BlockKernel block_kernel = {})
    {
        const bool use_sym = (sym_ops != nullptr) && !sym_ops->empty();

        // Skeleton mode (S1): when every density carries the molecule's
        // symmetry, accumulate each non-forbidden orbit representative ONCE,
        // weighted by its orbit size, and symmetrize G over the ops at the end
        // instead of replicating every value across its orbit. Exact for a
        // symmetric density; replication (below) is kept for any other, which
        // it reproduces exactly (PCM, symmetry-broken UHF, random test densities).
        // ponytail: absolute tolerance; a density off by <= 1e-10 is treated as
        // symmetric, i.e. G is built from its group average.
        constexpr double kSkeletonDensityTol = 1e-10;
        bool skeleton = use_sym;
        for (const Eigen::MatrixXd *D : {dens.P, dens.Pt, dens.Pa, dens.Pb})
            if (skeleton && D && !density_has_symmetry(*D, *sym_ops, kSkeletonDensityTol))
                skeleton = false;
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

        // first_pair_of[g]: index of the first group pair with .a == g (pairs
        // are ordered by .a, then .b). Groups are AO-contiguous, so a ket block
        // whose .a sorts before the bra's .a has k < i for every component and
        // holds no canonical quartet; the ket loop starts at first_pair_of[bra.a].
        // That skipped about half of all blocks surviving Schwarz, each of which
        // walked four component loops for nothing. Bitwise: the skipped blocks
        // contributed nothing, and the rest are visited in the same order.
        std::vector<std::size_t> first_pair_of(ngroups, 0);
        for (std::size_t p = ngp; p-- > 0;)
            first_pair_of[group_pairs[p].a] = p;

        // The caller's shell_pairs already holds ShellPair(cvA, cvB) for every
        // AO pair the canonical filter admits (j >= i, l >= k), built with the
        // same constructor. Look them up instead of rebuilding one per component
        // quartet — the rebuild (pow, exp, heap per primitive pair) was ~8 % of
        // the build. Keyed on _index, not list order. Bitwise-identical.
        std::vector<const HartreeFock::ShellPair *> pair_of(nb * nb, nullptr);
        for (const auto &sp : shell_pairs)
            if (sp.A._index < nb && sp.B._index < nb)
                pair_of[sp.A._index * nb + sp.B._index] = &sp;

        // norm_ratio: a block kernel is fed first-component pairs, whose
        // coeff_product carries the FIRST component's _component_norm. A d
        // shell's components differ (xx 1/sqrt(3), xy 1), so each output is
        // rescaled by its own ratio. Exactly 1.0 for s and p.
        std::vector<int> group_L(ngroups, 0);
        std::vector<double> norm_ratio(nb, 1.0);
        for (std::size_t g = 0; g < ngroups; ++g)
        {
            const HartreeFock::ContractedView &first = *ao_views[groups[g].first_ao];
            const auto &c = first._cartesian;
            group_L[g] = c[0] + c[1] + c[2];
            for (std::size_t comp = 0; comp < groups[g].n_components; ++comp)
                norm_ratio[groups[g].first_ao + comp] =
                    ao_views[groups[g].first_ao + comp]->_component_norm / first._component_norm;
        }

        // Shell-pair Schwarz bound: Qmax[p] = max over the components (ca,cb) of
        // shell-pair p of Q(i,j). Since every component quartet of the block
        // (A B | C D) obeys Q(i,j)*Q(k,l) <= Qmax[bra] * Qmax[ket], a block whose
        // product is already below tol_eri has EVERY component screened, so the
        // whole block can be skipped before entering the component loops.
        //
        // This is exact, not an approximation — it is the same Schwarz test,
        // hoisted. It matters now in a way it did not before: the two-phase
        // builder had to walk the components anyway to fill the tensor, but with
        // nothing to fill, a screened block costs literally nothing. Skipping at
        // the block level avoids up to 6^4 = 1296 component iterations (d shells)
        // and, more importantly, the ShellPair construction inside them.
        std::vector<double> qmax(ngp, 0.0);
        for (std::size_t p = 0; p < ngp; ++p)
        {
            const FusedShellGroup &ga = groups[group_pairs[p].a];
            const FusedShellGroup &gb = groups[group_pairs[p].b];
            double m = 0.0;
            for (std::size_t ca = 0; ca < ga.n_components; ++ca)
                for (std::size_t cb = 0; cb < gb.n_components; ++cb)
                {
                    const std::size_t i = ao_views[ga.first_ao + ca]->_index;
                    const std::size_t j = ao_views[gb.first_ao + cb]->_index;
                    m = std::max(m, Q(i, j));
                }
            qmax[p] = m;
        }

        // Block-level orbit rejection. A canonical quartet starts with its
        // smallest AO, so if an operation sends any of a block's four shell
        // groups below the bra's first group A, every component quartet of the
        // block maps to a smaller quartet and none is its orbit's
        // representative: skip the block without visiting a component. Needs
        // each operation to map a group onto one group (true for the signed AO
        // permutations update_integral_symmetry builds); checked here, and the
        // rejection is off if any operation splits a group. Bitwise: the
        // skipped blocks held no representatives. ~60 % of D2h naphthalene's
        // blocks, 1.07x on its build.
        std::vector<std::size_t> op_group; // op_group[o * ngroups + g]
        if (use_sym)
        {
            std::vector<std::size_t> group_of_ao(nb, 0);
            for (std::size_t g = 0; g < ngroups; ++g)
                for (std::size_t c = 0; c < groups[g].n_components; ++c)
                    group_of_ao[groups[g].first_ao + c] = g;
            op_group.resize(sym_ops->size() * ngroups);
            bool maps_groups = true;
            for (std::size_t o = 0; o < sym_ops->size() && maps_groups; ++o)
            {
                const std::vector<int> &map = (*sym_ops)[o].ao_map;
                for (std::size_t g = 0; g < ngroups && maps_groups; ++g)
                {
                    const std::size_t h = group_of_ao[map[groups[g].first_ao]];
                    for (std::size_t c = 1; c < groups[g].n_components; ++c)
                        maps_groups = maps_groups && group_of_ao[map[groups[g].first_ao + c]] == h;
                    op_group[o * ngroups + g] = h;
                }
            }
            if (!maps_groups)
                op_group.clear();
        }

#ifdef USE_OPENMP
        const int n_threads = omp_get_max_threads();
#else
        const int n_threads = 1;
#endif
        const bool need_uhf = spin_resolved;
        const bool raw_coulomb_exchange =
            !need_uhf && term == FusedTerm::RawCoulombExchange;
        std::vector<Eigen::MatrixXd> g_partials(
            need_uhf ? 0 : static_cast<std::size_t>(n_threads),
            Eigen::MatrixXd::Zero(nb, nb));
        std::vector<Eigen::MatrixXd> ga_partials(
            (need_uhf || raw_coulomb_exchange) ? static_cast<std::size_t>(n_threads) : 0,
            Eigen::MatrixXd::Zero(nb, nb));
        std::vector<Eigen::MatrixXd> gb_partials(
            need_uhf ? static_cast<std::size_t>(n_threads) : 0,
            Eigen::MatrixXd::Zero(nb, nb));

        // MPI: partition the bra shell-pair triangle across ranks; OpenMP splits
        // each rank's stripe within the node — the hybrid. Each canonical quartet
        // is produced by exactly one bra, so the ranks accumulate into disjoint
        // sets of quartets and the MPI_Allreduce(SUM) below merges them exactly.
        //
        // This is where the memory-direct build pays off a second time. The
        // tensor path (build_eri_tensor_shellwise) has to Allreduce the whole
        // nb^4 tensor — 267 MB at nb=76, 500 GB at nb=500 — which is not a
        // "cheap reduce" by any reading. Here the only thing reduced is the
        // nb^2 Fock matrix: 46 KB at nb=76, 2 MB at nb=500. That is the reduce
        // the HPC scope doc always assumed was happening.
        //
        // Serial builds get rank 0 / size 1, so the stride degrades to the full
        // loop and the reduce is a no-op — byte-identical to the non-MPI path.
        const int mpi_rank = HartreeFock::Mpi::rank();
        const int mpi_size = HartreeFock::Mpi::size();

        // The bra body is instantiated per symmetry mode (0 none, 1 orbit
        // replication, 2 skeleton), so the non-symmetric hot loop carries no
        // symmetry code. With the symmetry branches inline, merely moving the
        // orbit buffer swung the non-symmetric build by 2-5 % through code
        // generation.
        const auto bra_body = [&](std::size_t bra, auto mode_tag)
        {
            constexpr bool kSym = decltype(mode_tag)::value != 0;
            constexpr bool kSkel = decltype(mode_tag)::value == 2;
            if (mpi_size > 1 &&
                static_cast<int>(bra % static_cast<std::size_t>(mpi_size)) != mpi_rank)
                return;

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
            Eigen::MatrixXd &Gb_local = need_uhf ? gb_partials[thread_id]
                : raw_coulomb_exchange ? ga_partials[thread_id] : g_partials[thread_id];

            const FusedShellGroup &gA = groups[group_pairs[bra].a];

            BlockKernel block = block_kernel;
            BlockEriRequest request;
            request.norm_ratio = norm_ratio.data();
            request.kernel = kernel;
            request.omega = omega;

            // One orbit buffer per bra iteration (so per thread), reused for
            // every representative quartet: no per-quartet allocation.
            std::vector<QuartetOrbitElem> orbit;
            if constexpr (kSym && !kSkel)
                orbit.reserve(sym_ops->size());
            const FusedShellGroup &gB = groups[group_pairs[bra].b];

            for (std::size_t ket = first_pair_of[group_pairs[bra].a]; ket < ngp; ++ket)
            {
                // Block-level Schwarz: every component of this block is below
                // tolerance, so skip all of them without touching the component
                // loops or building a single ShellPair. Exact — same bound.
                if (qmax[bra] * qmax[ket] < tol_eri)
                    continue;

                const FusedShellGroup &gC = groups[group_pairs[ket].a];
                const FusedShellGroup &gD = groups[group_pairs[ket].b];

                if constexpr (kSym)
                {
                    if (!op_group.empty())
                    {
                        const std::size_t A = group_pairs[bra].a, B = group_pairs[bra].b;
                        const std::size_t C = group_pairs[ket].a, D = group_pairs[ket].b;
                        bool below = false;
                        for (std::size_t o = 0; o < sym_ops->size() && !below; ++o)
                        {
                            const std::size_t *m = &op_group[o * ngroups];
                            below = std::min({m[A], m[B], m[C], m[D]}) < A;
                        }
                        if (below)
                            continue;
                    }
                }

                // The block kernel is prepared on the first component quartet
                // that survives the filters: -1 not yet asked, 0 declined (use
                // eri_elem per component), 1 prepared.
                int block_state = -1;

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

                        // ponytail: a pair missing from a partial list is built
                        // on the spot, as before; no caller passes one today.
                        std::optional<HartreeFock::ShellPair> spAB_own;
                        const HartreeFock::ShellPair *spAB_ptr = pair_of[i * nb + j];
                        if (!spAB_ptr)
                            spAB_ptr = &spAB_own.emplace(cvA, cvB);
                        const HartreeFock::ShellPair &spAB = *spAB_ptr;

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

                                // Integral symmetry: compute the ERI only at the
                                // orbit representative, then replicate it (with
                                // the accumulated AO sign) across the orbit —
                                // exactly what the tensor build's scatter does.
                                //
                                // The two dedups compose without double-counting:
                                // build_quartet_orbit returns DISTINCT canonical
                                // quartets, and fock_accumulate_* dedups the
                                // 8-fold permutational orbit of each. See the
                                // dedup argument in quartet_orbit.h.
                                double weight = 1.0; // skeleton: orbit size
                                if constexpr (kSkel)
                                {
                                    weight = skeleton_orbit_weight(i, j, k, l, *sym_ops);
                                    if (weight == 0.0)
                                        continue; // not the representative, or forbidden
                                }
                                else if constexpr (kSym)
                                {
                                    // Cheap rejection first: most canonical
                                    // quartets are not their orbit's
                                    // representative and never need the orbit.
                                    if (!is_orbit_representative(i, j, k, l, *sym_ops))
                                        continue;
                                    if (fill_quartet_orbit(i, j, k, l, *sym_ops, orbit))
                                        continue; // symmetry-forbidden: vanishes
                                    // Only the representative computes the value.
                                    if (orbit.front().i != i || orbit.front().j != j ||
                                        orbit.front().k != k || orbit.front().l != l)
                                        continue;
                                }

                                if (block_state < 0)
                                {
                                    const FusedShellGroup *g4[4] = {&gA, &gB, &gC, &gD};
                                    const std::size_t s4[4] = {group_pairs[bra].a, group_pairs[bra].b,
                                                               group_pairs[ket].a, group_pairs[ket].b};
                                    request.ab = pair_of[gA.first_ao * nb + gB.first_ao];
                                    request.cd = pair_of[gC.first_ao * nb + gD.first_ao];
                                    for (int s = 0; s < 4; ++s)
                                    {
                                        request.views[s] = &ao_views[g4[s]->first_ao];
                                        request.n[s] = g4[s]->n_components;
                                        request.L[s] = group_L[s4[s]];
                                    }
                                    block_state = block.prepare(request) ? 1 : 0;
                                }

                                double val;
                                if (block_state == 1)
                                    val = block.value(ca, cb, cc, cd);
                                else
                                {
                                    std::optional<HartreeFock::ShellPair> spCD_own;
                                    const HartreeFock::ShellPair *spCD_ptr = pair_of[k * nb + l];
                                    if (!spCD_ptr)
                                        spCD_ptr = &spCD_own.emplace(cvC, cvD);
                                    val = eri_elem(
                                        spAB, *spCD_ptr,
                                        lAx, lAy, lAz, lBx, lBy, lBz,
                                        lCx, lCy, lCz, lDx, lDy, lDz,
                                        kernel, omega);
                                }

                                if constexpr (!kSym)
                                {
                                    accumulate_term(term, need_uhf, G_local, Gb_local,
                                                    dens, i, j, k, l, val);
                                }
                                else if constexpr (kSkel)
                                {
                                    accumulate_term(term, need_uhf, G_local, Gb_local,
                                                    dens, i, j, k, l, weight * val);
                                }
                                else
                                {
                                    for (const auto &e : orbit)
                                    {
                                        const double sv = static_cast<double>(e.sign) * val;
                                        accumulate_term(term, need_uhf, G_local, Gb_local,
                                                        dens, e.i, e.j, e.k, e.l, sv);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        // schedule(static), NOT dynamic — load-bearing, see the header note.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t bra = 0; bra < ngp; ++bra)
        {
            if (skeleton)
                bra_body(bra, std::integral_constant<int, 2>{});
            else if (use_sym)
                bra_body(bra, std::integral_constant<int, 1>{});
            else
                bra_body(bra, std::integral_constant<int, 0>{});
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
            // Merge the per-rank partial Focks. nb^2, not nb^4. No-op when serial.
            // Eigen's default storage is column-major and contiguous, so the raw
            // buffer is a plain nb*nb double array either way — the sum is
            // elementwise, so the layout convention does not matter as long as
            // every rank uses the same one (they do; same type, same dims).
            HartreeFock::Mpi::allreduce_inplace(Ga.data(), static_cast<std::size_t>(Ga.size()));
            HartreeFock::Mpi::allreduce_inplace(Gb.data(), static_cast<std::size_t>(Gb.size()));
            if (skeleton)
            {
                symmetrize_over_ops(Ga, *sym_ops);
                symmetrize_over_ops(Gb, *sym_ops);
            }
        }
        else
        {
            G.setZero(nb, nb);
            for (int t = 0; t < n_threads; ++t)
                G += g_partials[static_cast<std::size_t>(t)];
            HartreeFock::Mpi::allreduce_inplace(G.data(), static_cast<std::size_t>(G.size()));
            if (raw_coulomb_exchange)
            {
                Ga.setZero(nb, nb);
                for (int t = 0; t < n_threads; ++t)
                    Ga += ga_partials[static_cast<std::size_t>(t)];
                HartreeFock::Mpi::allreduce_inplace(Ga.data(), static_cast<std::size_t>(Ga.size()));
                if (skeleton)
                    symmetrize_over_ops(Ga, *sym_ops);
            }
            if (skeleton)
                symmetrize_over_ops(G, *sym_ops);
        }
    }
}
