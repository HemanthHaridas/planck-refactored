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
//  2. Partials keyed by BIN (bra mod n_bins, n_bins a function of nb only),
//     each bin's bras in increasing order, bins summed in FIXED bin order.
//     These accumulations are read-modify-write, unlike the store-only
//     write_eri_permutations of the tensor build. Keying partials by thread
//     made G depend on the thread count, and a critical/atomic reduction on
//     completion order (measured; see src/dft/ks_matrix.cpp). With bins, which
//     thread runs a bin cannot change G, so bins are scheduled dynamically.
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
#include <cassert>
#include <cstddef>
#include <optional>
#include <tuple>
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
    // The default NoBlockKernel keeps an engine pure. A copy lives per thread.
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
        BlockKernel block_kernel = {},
        bool dense_digest = true,
        bool density_screen = true)
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

        // K1: dense digestion per block (docs/ERI_SHELL_QUARTET_FOCK_BUILD.md).
        // Decided per build, never per block: the dense loop visits block pairs
        // p <= q, while the per-component loop's canonical filter is by AO index
        // and needs kets with q < p, so the two cannot share a build. Needs a
        // block kernel that covers every block of the basis (every total L up to
        // four times the largest shell L, for this kernel; the Auto chain covers
        // Coulomb through f shells), no integral symmetry or skeleton mode (K1.4;
        // replication mode keeps the per-component loop), and symmetric
        // densities: the final (G + G^T)/2 keeps only K of the density's
        // symmetric part.
        // ponytail: same 1e-10 absolute tolerance as skeleton mode.
        bool dense = dense_digest && (!use_sym || skeleton);
        for (int lt = 0; dense && lt <= 4 * *std::max_element(group_L.begin(), group_L.end()); ++lt)
            dense = block_kernel.covers(lt, kernel);
        for (const Eigen::MatrixXd *D : {dens.P, dens.Pt, dens.Pa, dens.Pb})
            if (dense && D && (*D - D->transpose()).cwiseAbs().maxCoeff() > kSkeletonDensityTol)
                dense = false;

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

        // Density-weighted block screen (S3, docs/SCF_VS_PYSCF.md):
        // skip a block when Qmax_bra * Qmax_ket * max(4|D_ab|, 4|D_cd|, |D_ac|,
        // |D_ad|, |D_bc|, |D_bd|) < tol_eri, with |D| the shell-block maximum
        // over every density passed in (Haser-Ahlrichs). This is what makes an
        // incremental build (D = P - P_prev) cheap. Off in replication mode: a
        // representative block is scattered onto images whose density differs.
        // The Schwarz table itself stays unscreened (invariant 8). The caller
        // can switch it off: the SCF tail does, since the skipped share of an
        // incremental G[dP] makes the energy random-walk at tight tolerances.
        density_screen = density_screen && (!use_sym || skeleton);
        Eigen::MatrixXd dblk = Eigen::MatrixXd::Zero(ngroups, ngroups);
        if (density_screen)
            for (const Eigen::MatrixXd *D : {dens.P, dens.Pt, dens.Pa, dens.Pb})
                if (D)
                    for (std::size_t a = 0; a < ngroups; ++a)
                        for (std::size_t b = 0; b < ngroups; ++b)
                            dblk(a, b) = std::max(dblk(a, b),
                                D->block(groups[a].first_ao, groups[b].first_ao,
                                         groups[a].n_components, groups[b].n_components)
                                    .cwiseAbs().maxCoeff());
        // The density test runs at tol_eri / 10. At tol_eri itself the skipped
        // set changes with dP every iteration, the SCF energy random-walks at
        // 1e-8 and DIIS stalls (12 H2O 6-31G*: 15 -> 24 iterations). At /10 the
        // energy is within 6e-10 of the full build, with the same iteration
        // count, and keeps 1.19-1.33x of the 1.28-1.46x (S3 scan).
        constexpr double kDensityScreenFactor = 0.1;
        const auto screened = [&](std::size_t bra, std::size_t ket)
        {
            const double qq = qmax[bra] * qmax[ket];
            if (qq < tol_eri)
                return true;
            if (!density_screen)
                return false;
            const std::size_t a = group_pairs[bra].a, b = group_pairs[bra].b;
            const std::size_t c = group_pairs[ket].a, d = group_pairs[ket].b;
            const double dm = std::max({4.0 * dblk(a, b), 4.0 * dblk(c, d), dblk(a, c),
                                        dblk(a, d), dblk(b, c), dblk(b, d)});
            return qq * dm < tol_eri * kDensityScreenFactor;
        };

        // op_group[o * ngroups + g]: the group op o maps group g onto, used by
        // the block-level orbit tests (below). Needs each
        // operation to map a group onto one group (true for the signed AO
        // permutations update_integral_symmetry builds); checked here, and the
        // block test is off if any operation splits a group.
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

        // The block request for shell groups (sA sB | sC sD): first-component
        // pairs, component views, counts and L. Shared by both traversals.
        const auto set_block = [&](BlockEriRequest &r, std::size_t sA, std::size_t sB,
                                   std::size_t sC, std::size_t sD)
        {
            const std::size_t s4[4] = {sA, sB, sC, sD};
            r.ab = pair_of[groups[sA].first_ao * nb + groups[sB].first_ao];
            r.cd = pair_of[groups[sC].first_ao * nb + groups[sD].first_ao];
            for (int s = 0; s < 4; ++s)
            {
                r.views[s] = &ao_views[groups[s4[s]].first_ao];
                r.n[s] = groups[s4[s]].n_components;
                r.L[s] = group_L[s4[s]];
            }
        };

        // K1.4b: an op whose AO map is the identity (the identity itself, or the
        // mirror through a planar molecule's plane) fixes every quartet. In the
        // dense skeleton weight it always counts toward the stabilizer and needs
        // only a sign test, never canonicalization. `sign_ops` holds those with
        // a negative sign somewhere (the identity needs no test at all).
        std::size_t n_identity_map_ops = 0;
        std::vector<std::size_t> sign_ops;
        std::vector<char> is_identity_map;
        if (use_sym)
        {
            is_identity_map.assign(sym_ops->size(), 0);
            for (std::size_t o = 0; o < sym_ops->size(); ++o)
            {
                const auto &op = (*sym_ops)[o];
                bool id = true, positive = true;
                for (std::size_t a = 0; a < nb && id; ++a)
                {
                    id = op.ao_map[a] == static_cast<int>(a);
                    positive = positive && op.ao_sign[a] > 0;
                }
                if (!id)
                    continue;
                is_identity_map[o] = 1;
                ++n_identity_map_ops;
                if (!positive)
                    sign_ops.push_back(o);
            }
        }

        // Two representative conventions, one per traversal. The skeleton (S1)
        // and replication schemes need exactly one representative per orbit;
        // any consistent choice works, but a traversal must not mix two.
        //
        // The per-component loop uses the AO-minimal canonical quartet
        // (is_orbit_representative, fill_quartet_orbit, skeleton_orbit_weight
        // over all ops). Its block test can only use what holds for every
        // component: each has its smallest AO in the block's smallest group
        // G0, so an op whose image's smallest group is below G0 sends every
        // component lower, and the block holds no representative. Comparing
        // whole shell quartets is NOT valid here: when both pairs start in G0,
        // the AO order is decided inside G0, so an AO-minimal representative
        // can sit in a block whose shell quartet some op sends lower.
        const auto ao_block_has_no_representative =
            [&](std::size_t sA, std::size_t sB, std::size_t sC, std::size_t sD)
        {
            const std::size_t g0 = std::min(sA, sC);
            for (std::size_t o = 0; o < sym_ops->size(); ++o)
            {
                const std::size_t *m = &op_group[o * ngroups];
                if (std::min({m[sA], m[sB], m[sC], m[sD]}) < g0)
                    return true;
            }
            return false;
        };

        // The dense loop (K1.4) uses the shell-first convention: the
        // representative lies in the orbit's lowest canonical shell quartet,
        // and within it is AO-minimal among the ops that map that shell
        // quartet onto itself. Unique per orbit (every other member's block is
        // sent lower by some op), so a block some op sends to a lower canonical
        // shell quartet holds no representative, and only the ops fixing the
        // block's shell quartet can matter to a component. op_pair[o * ngp + p]
        // is op o's canonical image of shell pair p; pairs are numbered
        // lexicographically, so the image of block {p, q} is (min, max) of two
        // entries. `fixing` collects the permuting ops that fix the block (the
        // identity-map ops are counted separately).
        std::vector<std::size_t> op_pair;
        if (use_sym && !op_group.empty())
        {
            op_pair.resize(sym_ops->size() * ngp);
            for (std::size_t o = 0; o < sym_ops->size(); ++o)
                for (std::size_t pp = 0; pp < ngp; ++pp)
                {
                    std::size_t x = op_group[o * ngroups + group_pairs[pp].a];
                    std::size_t y = op_group[o * ngroups + group_pairs[pp].b];
                    if (x > y)
                        std::swap(x, y);
                    op_pair[o * ngp + pp] = x * ngroups - x * (x - 1) / 2 + (y - x); // group_pairs order
                }
        }
        const auto shell_block_has_no_representative =
            [&](std::size_t p, std::size_t q, std::vector<std::size_t> &fixing)
        {
            fixing.clear();
            for (std::size_t o = 0; o < sym_ops->size(); ++o)
            {
                const std::size_t ip = op_pair[o * ngp + p], iq = op_pair[o * ngp + q];
                const std::size_t lo = std::min(ip, iq), hi = std::max(ip, iq);
                if (lo < p || (lo == p && hi < q))
                    return true;
                if (lo == p && hi == q && !is_identity_map[o])
                    fixing.push_back(o);
            }
            return false;
        };

#ifdef USE_OPENMP
        const int n_threads = omp_get_max_threads();
#else
        const int n_threads = 1;
#endif
        const bool need_uhf = spin_resolved;
        const bool raw_coulomb_exchange =
            !need_uhf && term == FusedTerm::RawCoulombExchange;
        // ponytail: 64 bins, fewer once 64 x nb^2 passes 256 MB per output (8 at
        // the least, ~1 GB at nb = 4000); a function of nb only, never of the
        // thread count, or G would depend on it again.
        const std::size_t n_bins = std::clamp<std::size_t>(
            (std::size_t{256} << 20) / (sizeof(double) * std::max<std::size_t>(nb * nb, 1)), 8, 64);
        std::vector<Eigen::MatrixXd> g_partials(
            need_uhf ? 0 : n_bins, Eigen::MatrixXd::Zero(nb, nb));
        std::vector<Eigen::MatrixXd> ga_partials(
            (need_uhf || raw_coulomb_exchange) ? n_bins : 0, Eigen::MatrixXd::Zero(nb, nb));
        std::vector<Eigen::MatrixXd> gb_partials(
            need_uhf ? n_bins : 0, Eigen::MatrixXd::Zero(nb, nb));
        // One block kernel per thread: it carries reusable scratch, so a copy per
        // bra would reallocate it every iteration.
        std::vector<BlockKernel> blocks(static_cast<std::size_t>(n_threads), block_kernel);

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
            const std::size_t bin = bra % n_bins;
            Eigen::MatrixXd &G_local =
                need_uhf ? ga_partials[bin] : g_partials[bin];
            Eigen::MatrixXd &Gb_local = need_uhf ? gb_partials[bin]
                : raw_coulomb_exchange ? ga_partials[bin] : g_partials[bin];

            const FusedShellGroup &gA = groups[group_pairs[bra].a];

            BlockKernel &block = blocks[static_cast<std::size_t>(thread_id)];
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
                if (screened(bra, ket))
                    continue;

                const FusedShellGroup &gC = groups[group_pairs[ket].a];
                const FusedShellGroup &gD = groups[group_pairs[ket].b];

                if constexpr (kSym)
                {
                    if (!op_group.empty() &&
                        ao_block_has_no_representative(group_pairs[bra].a, group_pairs[bra].b,
                                                       group_pairs[ket].a, group_pairs[ket].b))
                        continue;
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
                                    set_block(request, group_pairs[bra].a, group_pairs[bra].b,
                                              group_pairs[ket].a, group_pairs[ket].b);
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

        // Dense digestion (K1). Every component (a,b|c,d) of a visited block
        // pair p <= q above the component Schwarz bound is digested with
        // v' = deg * v, deg = (A=B ? 1 : 2)(C=D ? 1 : 2)(p=q ? 1 : 2):
        //   Coulomb   out_ab += cj P_cd v',  out_cd += cj P_ab v'
        //   exchange  out_ac += ck P_bd v',  out_bd += ck P_ac v',
        //             out_ad += ck P_bc v',  out_bc += ck P_ad v'
        // and out <- (out + out^T)/2 after the merge. Against the 8-fold orbit
        // sum this is exact for a symmetric density: cj = 1/2 for Coulomb,
        // ck = 1/4 for raw K, so -1/8 for RHF's -K/2 and -1/4 for UHF's -K.
        struct DenseChannel
        {
            const Eigen::MatrixXd *Pj = nullptr;
            double cj = 0.0;
            const Eigen::MatrixXd *Pk = nullptr;
            double ck = 0.0;
        };
        std::array<DenseChannel, 2> channel{}; // [0] -> G_local, [1] -> Gb_local
        if (need_uhf)
        {
            const bool coul = term == FusedTerm::Combined || term == FusedTerm::CoulombOnly;
            const double ck = term == FusedTerm::Combined ? -0.25
                              : term == FusedTerm::ExchangeOnly ? 0.25 : 0.0;
            for (int s = 0; s < 2; ++s)
            {
                if (coul)
                    channel[s].Pj = dens.Pt, channel[s].cj = 0.5;
                if (ck != 0.0)
                    channel[s].Pk = s == 0 ? dens.Pa : dens.Pb, channel[s].ck = ck;
            }
        }
        else if (term == FusedTerm::Combined)
            channel[0] = {dens.P, 0.5, dens.P, -0.125};
        else if (term == FusedTerm::CoulombOnly)
            channel[0] = {dens.P, 0.5, nullptr, 0.0};
        else if (term == FusedTerm::ExchangeOnly)
            channel[0] = {nullptr, 0.0, dens.P, 0.25};
        else // RawCoulombExchange
            channel[0] = {dens.P, 0.5, nullptr, 0.0}, channel[1] = {nullptr, 0.0, dens.P, 0.25};
        const int dense_mode = (channel[0].Pj ? 1 : 0) | (channel[0].Pk ? 2 : 0) |
                               (channel[1].Pj ? 4 : 0) | (channel[1].Pk ? 8 : 0);

        // Instantiated per active write group (bit 0: Coulomb into out[0],
        // 1: exchange into out[0], 2: Coulomb into out[1], 3: exchange into
        // out[1]), so the component loop carries no per-mode branch: the
        // generic two-channel loop cost 2-8 % against hard-coded RHF writes.
        //
        // Skeleton mode (kSkel, K1.4): a component carries the S1 weight of its
        // canonical form (orbit size, or 0 if it is not its orbit's
        // representative or is forbidden), and G is symmetrized over the ops at
        // the end as before. Groups are AO-contiguous, so an op that sends the
        // block's group quartet to a different canonical group quartet decides
        // every component the same way: a smaller one means no representative
        // in the block (skip it), a larger one is irrelevant. Only the ops that
        // map the block's group quartet onto itself are tested per component.
        //
        // dense_body(p, kets, ...) visits bra pair p against every ket q that
        // kets(f) hands to f; callers guarantee q >= p and that each unordered
        // block pair is handed over exactly once per build.
        std::vector<std::vector<std::size_t>> block_ops_of(static_cast<std::size_t>(n_threads));
        const auto dense_body = [&](std::size_t p, auto &&kets, auto mode_tag, auto skel_tag)
        {
            constexpr int kMode = decltype(mode_tag)::value;
            constexpr bool kSkel = decltype(skel_tag)::value;
#ifdef USE_OPENMP
            const int thread_id = omp_get_thread_num();
#else
            const int thread_id = 0;
#endif
            // the block's undecided permuting ops (per thread, reused)
            std::vector<std::size_t> &block_ops = block_ops_of[static_cast<std::size_t>(thread_id)];
            const std::size_t bin = p % n_bins;
            Eigen::MatrixXd *out[2] = {
                need_uhf ? &ga_partials[bin] : &g_partials[bin],
                need_uhf ? &gb_partials[bin]
                : raw_coulomb_exchange ? &ga_partials[bin] : &g_partials[bin]};
            // Column-major raw views: M(r, c) = M[c * nb + r].
            double *const o0 = out[0]->data(), *const o1 = out[1]->data();
            const auto raw = [](const Eigen::MatrixXd *M) { return M ? M->data() : nullptr; };
            const double *const j0 = raw(channel[0].Pj), *const k0 = raw(channel[0].Pk);
            const double *const j1 = raw(channel[1].Pj), *const k1 = raw(channel[1].Pk);
            const double cj0 = channel[0].cj, ck0 = channel[0].ck, cj1 = channel[1].cj, ck1 = channel[1].ck;

            BlockKernel &block = blocks[static_cast<std::size_t>(thread_id)];
            BlockEriRequest request;
            request.norm_ratio = norm_ratio.data();
            request.kernel = kernel;
            request.omega = omega;

            const std::size_t sA = group_pairs[p].a, sB = group_pairs[p].b;
            const FusedShellGroup &gA = groups[sA], &gB = groups[sB];

            kets([&](std::size_t q)
            {
                if (screened(p, q))
                    return;
                const std::size_t sC = group_pairs[q].a, sD = group_pairs[q].b;
                if constexpr (kSkel)
                {
                    if (!op_pair.empty() && shell_block_has_no_representative(p, q, block_ops))
                        return;
                }
                const FusedShellGroup &gC = groups[sC], &gD = groups[sD];
                // Prepared lazily at the first component that survives the
                // Schwarz mask and (under S1) has a nonzero weight, as in the
                // per-component loop: a block with no representative costs no
                // kernel work. The dense predicate admits only kernels that
                // cover every block of the basis.
                bool prepared = false;
                const auto ensure_prepared = [&]
                {
                    if (prepared)
                        return;
                    set_block(request, sA, sB, sC, sD);
                    prepared = block.prepare(request);
                    assert(prepared && "dense digestion: block kernel declined a block");
                };
                const double deg = (sA == sB ? 1.0 : 2.0) * (sC == sD ? 1.0 : 2.0) * (p == q ? 1.0 : 2.0);

                for (std::size_t ca = 0; ca < gA.n_components; ++ca)
                {
                    const std::size_t a = gA.first_ao + ca;
                    for (std::size_t cb = 0; cb < gB.n_components; ++cb)
                    {
                        const std::size_t b = gB.first_ao + cb;
                        const double qab = Q(a, b);
                        if (qab * qmax[q] < tol_eri)
                            continue;
                        for (std::size_t cc = 0; cc < gC.n_components; ++cc)
                        {
                            const std::size_t c = gC.first_ao + cc;
                            for (std::size_t cd = 0; cd < gD.n_components; ++cd)
                            {
                                const std::size_t d = gD.first_ao + cd;
                                // The component Schwarz mask: screening stays
                                // exactly today's.
                                if (qab * Q(c, d) < tol_eri)
                                    continue;
                                double weight = 1.0;
                                if constexpr (kSkel)
                                {
                                    // With the op tables, only the identity-map
                                    // ops (a sign test) and the permuting ops
                                    // fixing the block are applied per component.
                                    weight = op_pair.empty()
                                                 ? skeleton_orbit_weight(a, b, c, d, *sym_ops,
                                                                         std::views::iota(std::size_t{0}, sym_ops->size()),
                                                                         std::array<std::size_t, 0>{}, 0)
                                                 : skeleton_orbit_weight(a, b, c, d, *sym_ops, block_ops,
                                                                         sign_ops, n_identity_map_ops);
                                    if (weight == 0.0)
                                        continue;
                                }
                                ensure_prepared();
                                const double v = deg * weight * block.value(ca, cb, cc, cd);
                                const auto coul = [&](double *O, const double *P, double w)
                                {
                                    O[b * nb + a] += P[d * nb + c] * w;
                                    O[d * nb + c] += P[b * nb + a] * w;
                                };
                                const auto exch = [&](double *O, const double *P, double w)
                                {
                                    O[c * nb + a] += P[d * nb + b] * w;
                                    O[d * nb + b] += P[c * nb + a] * w;
                                    O[d * nb + a] += P[c * nb + b] * w;
                                    O[c * nb + b] += P[d * nb + a] * w;
                                };
                                if constexpr ((kMode & 1) != 0)
                                    coul(o0, j0, cj0 * v);
                                if constexpr ((kMode & 2) != 0)
                                    exch(o0, k0, ck0 * v);
                                if constexpr ((kMode & 4) != 0)
                                    coul(o1, j1, cj1 * v);
                                if constexpr ((kMode & 8) != 0)
                                    exch(o1, k1, ck1 * v);
                            }
                        }
                    }
                }
            });
        };

        const auto dense_unit = [&](std::size_t u, auto mode_tag, auto skel_tag)
        {
            if (mpi_size > 1 &&
                static_cast<int>(u % static_cast<std::size_t>(mpi_size)) != mpi_rank)
                return;
            dense_body(u, [&](auto &&f)
                       {
                           for (std::size_t q = u; q < ngp; ++q)
                               f(q);
                       },
                       mode_tag, skel_tag);
        };
        const std::size_t n_units = ngp;

        // One bin at a time, its bras round-robin (bin, bin + n_bins, ...) in
        // increasing order, so each bin's partial is fixed whoever runs it (see
        // the header note). Round-robin because bra p visits every ket q >= p:
        // contiguous ranges handed the first thread 44 % of the triangle (W1).
        // Dynamic because under S1 the work per bra is uneven: a static
        // round-robin left three of four threads waiting 10-17 % at the closing
        // barrier on naphthalene.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (std::size_t bin = 0; bin < n_bins; ++bin)
        for (std::size_t bra = bin; bra < n_units; bra += n_bins)
        {
            if (dense)
            {
                using std::integral_constant;
                const auto go = [&](auto skel)
                {
                    switch (dense_mode)
                    {
                    case 1: dense_unit(bra, integral_constant<int, 1>{}, skel); break;
                    case 2: dense_unit(bra, integral_constant<int, 2>{}, skel); break;
                    case 3: dense_unit(bra, integral_constant<int, 3>{}, skel); break;
                    case 5: dense_unit(bra, integral_constant<int, 5>{}, skel); break;
                    case 9: dense_unit(bra, integral_constant<int, 9>{}, skel); break;
                    case 10: dense_unit(bra, integral_constant<int, 10>{}, skel); break;
                    case 15: dense_unit(bra, integral_constant<int, 15>{}, skel); break;
                    default: break; // nothing to write (UHF RawCoulombExchange)
                    }
                };
                if (skeleton)
                    go(std::true_type{});
                else
                    go(std::false_type{});
            }
            else if (skeleton)
                bra_body(bra, std::integral_constant<int, 2>{});
            else if (use_sym)
                bra_body(bra, std::integral_constant<int, 1>{});
            else
                bra_body(bra, std::integral_constant<int, 0>{});
        }

        // Fixed bin order per element — never completion order. Columns are
        // independent, so the merge itself threads without changing a sum.
        const auto merge = [&](Eigen::MatrixXd &out, const std::vector<Eigen::MatrixXd> &parts)
        {
            out.setZero(nb, nb);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(nb); ++c)
                for (const Eigen::MatrixXd &part : parts)
                    out.col(c) += part.col(c);
        };
        if (need_uhf)
        {
            merge(Ga, ga_partials);
            merge(Gb, gb_partials);
            // Merge the per-rank partial Focks. nb^2, not nb^4. No-op when serial.
            // Eigen's default storage is column-major and contiguous, so the raw
            // buffer is a plain nb*nb double array either way — the sum is
            // elementwise, so the layout convention does not matter as long as
            // every rank uses the same one (they do; same type, same dims).
            HartreeFock::Mpi::allreduce_inplace(Ga.data(), static_cast<std::size_t>(Ga.size()));
            HartreeFock::Mpi::allreduce_inplace(Gb.data(), static_cast<std::size_t>(Gb.size()));
            if (dense)
            {
                Ga = (0.5 * (Ga + Ga.transpose())).eval();
                Gb = (0.5 * (Gb + Gb.transpose())).eval();
            }
            if (skeleton)
            {
                symmetrize_over_ops(Ga, *sym_ops);
                symmetrize_over_ops(Gb, *sym_ops);
            }
        }
        else
        {
            merge(G, g_partials);
            HartreeFock::Mpi::allreduce_inplace(G.data(), static_cast<std::size_t>(G.size()));
            if (raw_coulomb_exchange)
            {
                merge(Ga, ga_partials);
                HartreeFock::Mpi::allreduce_inplace(Ga.data(), static_cast<std::size_t>(Ga.size()));
                if (dense)
                    Ga = (0.5 * (Ga + Ga.transpose())).eval();
                if (skeleton)
                    symmetrize_over_ops(Ga, *sym_ops);
            }
            if (dense)
                G = (0.5 * (G + G.transpose())).eval();
            if (skeleton)
                symmetrize_over_ops(G, *sym_ops);
        }
    }
}
