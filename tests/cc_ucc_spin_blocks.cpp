// U3.1: the spin-blocked (UCC) MO ERI cache.
//
// The RCC cache stores seven blocks over one (occ, virt) partition. UCC cannot
// simply store three copies of that, for two independent reasons this gate pins:
//
//  1. The SPIN LIVES ON THE CHEMISTS' CHARGE-DENSITY PAIR, not on the physicist
//     block. Physicist <oovv>_abab is chemists (i_a a_a | j_b b_b) -- a genuinely
//     MIXED transform. It is not `ovov` of either pure spin relabeled, which is
//     the mistake that would produce a compiling, running, plausible wrong answer.
//
//  2. The BLOCK COUNT DIFFERS PER SPIN. Two of the four ERI permutations
//     (particle <qp|sr>, product <sr|qp>) are not symmetries of a mixed block --
//     they map `abab` to `baba` -- so its 8-fold orbit splits. Measured on the
//     CCSD UCC manifold: same-spin needs 6 stored arrays, mixed needs 10, and
//     three of the ten (oovo, vovo, vovv) have no RCC counterpart at all.
//
// ASYMMETRY IS LOAD-BEARING. The natural gate for this step -- an RHF-degenerate
// UHF reference (C_alpha == C_beta) must reproduce the RCC blocks bytewise -- is
// VACUOUS for the mixed block's pair ordering: with equal coefficients the (a|b)
// and (b|a) orderings coincide, so a swapped pair passes. Every case here
// therefore uses distinct alpha/beta coefficients AND noa != nob != nva != nvb,
// the same reasoning that made U2.1's noa=4 nva=3 nob=2 nvb=5 fixture
// load-bearing. The degeneracy check is kept as a separate case, clearly labelled
// for what it can and cannot see.

#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/tensor_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using HartreeFock::Correlation::CC::build_ucc_fock_blocks;
using HartreeFock::Correlation::CC::build_ucc_spin_block_cache_from_eri;
using HartreeFock::Correlation::CC::eri_permutation_preserves_block;
using HartreeFock::Correlation::CC::ucc_canonical_blocks;
using HartreeFock::Correlation::CC::Tensor4D;
using HartreeFock::Correlation::CC::TensorCCBlockCache;
using HartreeFock::Correlation::CC::UHFReference;

namespace
{
    int failures = 0;

    void check(bool ok, const std::string &what)
    {
        if (!ok)
        {
            std::printf("FAIL: %s\n", what.c_str());
            ++failures;
        }
    }

    // Deliberately asymmetric in every extent, so neither a transposed space slot
    // nor a swapped spin can land in bounds unnoticed.
    constexpr int NB = 7;
    constexpr int NOA = 3;
    constexpr int NVA = 4;
    constexpr int NOB = 2;
    constexpr int NVB = 5;

    std::size_t idx4(std::size_t p, std::size_t q, std::size_t r, std::size_t s)
    {
        return ((p * NB + q) * NB + r) * NB + s;
    }

    // A synthetic AO ERI carrying the full 8-fold real-orbital chemists symmetry.
    // Built by explicit symmetrization rather than a formula, so the fixture
    // cannot accidentally satisfy a symmetry the production path relies on.
    std::vector<double> make_eri()
    {
        std::vector<double> raw(NB * NB * NB * NB);
        unsigned int seed = 12345u;
        for (auto &value : raw)
        {
            seed = seed * 1103515245u + 12345u;
            value = static_cast<double>((seed >> 16) % 1000) / 500.0 - 1.0;
        }
        std::vector<double> eri(raw.size(), 0.0);
        for (std::size_t p = 0; p < NB; ++p)
            for (std::size_t q = 0; q < NB; ++q)
                for (std::size_t r = 0; r < NB; ++r)
                    for (std::size_t s = 0; s < NB; ++s)
                    {
                        const double v =
                            raw[idx4(p, q, r, s)] + raw[idx4(q, p, r, s)] +
                            raw[idx4(p, q, s, r)] + raw[idx4(q, p, s, r)] +
                            raw[idx4(r, s, p, q)] + raw[idx4(s, r, p, q)] +
                            raw[idx4(r, s, q, p)] + raw[idx4(s, r, q, p)];
                        eri[idx4(p, q, r, s)] = v / 8.0;
                    }
        return eri;
    }

    Eigen::MatrixXd make_coefficients(double scale, unsigned int seed)
    {
        Eigen::MatrixXd c(NB, NB);
        for (int mu = 0; mu < NB; ++mu)
            for (int p = 0; p < NB; ++p)
            {
                seed = seed * 1103515245u + 12345u;
                c(mu, p) = scale *
                           (static_cast<double>((seed >> 16) % 1000) / 1000.0 - 0.5);
            }
        return c;
    }

    UHFReference make_reference(bool degenerate)
    {
        UHFReference reference;
        reference.n_ao = NB;
        reference.n_mo = NB;
        reference.n_occ_alpha = NOA;
        reference.n_virt_alpha = NVA;
        reference.n_occ_beta = degenerate ? NOA : NOB;
        reference.n_virt_beta = degenerate ? NVA : NVB;
        reference.C_alpha = make_coefficients(1.0, 7u);
        reference.C_beta = degenerate ? reference.C_alpha
                                      : make_coefficients(0.8, 99u);
        return reference;
    }

    // Independent oracle: physicist <pq|rs> = chemists (pr|qs), each slot taking
    // its own space AND its own spin. Written from the definition rather than by
    // reusing the production helper, so a wrong pairing in the production code
    // cannot be mirrored here.
    double oracle(
        const std::vector<double> &eri,
        const UHFReference &ref,
        const std::string &space,
        const std::string &spin,
        int i0, int i1, int i2, int i3)
    {
        const auto column = [&](std::size_t slot, int index) {
            const bool alpha = spin[slot] == 'a';
            const Eigen::MatrixXd &c = alpha ? ref.C_alpha : ref.C_beta;
            const int occ = alpha ? ref.n_occ_alpha : ref.n_occ_beta;
            const int offset = space[slot] == 'o' ? 0 : occ;
            return c.col(offset + index);
        };
        const Eigen::VectorXd cp = column(0, i0);
        const Eigen::VectorXd cq = column(1, i1);
        const Eigen::VectorXd cr = column(2, i2);
        const Eigen::VectorXd cs = column(3, i3);

        double total = 0.0;
        for (std::size_t p = 0; p < NB; ++p)
            for (std::size_t q = 0; q < NB; ++q)
                for (std::size_t r = 0; r < NB; ++r)
                    for (std::size_t s = 0; s < NB; ++s)
                    {
                        // chemists (pr|qs): slot 0 pairs with slot 2, 1 with 3
                        total += eri[idx4(p, q, r, s)] *
                                 cp(static_cast<int>(p)) * cr(static_cast<int>(q)) *
                                 cq(static_cast<int>(r)) * cs(static_cast<int>(s));
                    }
        return total;
    }

    // The stored-block vocabulary, derived in U3.0 from the orbits of the space
    // patterns the manifold needs under the permutations valid for each tag.
    const std::vector<std::string> SAME_SPIN_BLOCKS{
        "oooo", "ooov", "oovv", "ovov", "ovvv", "vvvv"};
    const std::vector<std::string> MIXED_BLOCKS{
        "oooo", "ooov", "oovo", "oovv", "ovov",
        "ovvo", "ovvv", "vovo", "vovv", "vvvv"};
} // namespace

int main()
{
    const std::vector<double> eri = make_eri();
    const UHFReference reference = make_reference(/*degenerate=*/false);

    std::vector<std::pair<std::string, std::string>> requested;
    for (const auto &space : SAME_SPIN_BLOCKS)
    {
        requested.push_back({space, "aaaa"});
        requested.push_back({space, "bbbb"});
    }
    for (const auto &space : MIXED_BLOCKS)
        requested.push_back({space, "abab"});

    const auto cache = build_ucc_spin_block_cache_from_eri(
        eri, NB, reference, requested);
    check(cache.has_value(), "spin-block cache builds");
    if (!cache)
    {
        std::printf("  error: %s\n", cache.error().c_str());
        return 1;
    }

    // --- the block vocabulary ------------------------------------------------
    {
        check(cache->spin_blocks.size() == requested.size(),
              "cache holds one array per requested block");
        // Same-spin needs 6 and mixed needs 10 -- NOT three copies of seven.
        check(SAME_SPIN_BLOCKS.size() == 6, "same-spin vocabulary is 6 arrays");
        check(MIXED_BLOCKS.size() == 10, "mixed vocabulary is 10 arrays");
        // The three with no RCC counterpart at all.
        for (const char *space : {"oovo", "vovo", "vovv"})
            check(cache->spin_block(space, "abab").has_value(),
                  std::string("mixed block ") + space + " is stored (no RCC counterpart)");
    }

    // --- shapes follow each slot's own space AND spin ------------------------
    {
        const auto block = cache->spin_block("oovv", "abab");
        check(block.has_value(), "oovv_abab is stored");
        if (block)
        {
            // stored in chemists (p r | q s) order: (o_a, v_a, o_b, v_b)
            check((*block)->dim1 == NOA && (*block)->dim2 == NVA &&
                      (*block)->dim3 == NOB && (*block)->dim4 == NVB,
                  "oovv_abab is (noa, nva, nob, nvb) in chemists order");
        }

        const auto same = cache->spin_block("oovv", "aaaa");
        check(same.has_value() && (*same)->dim1 == NOA && (*same)->dim2 == NVA &&
                  (*same)->dim3 == NOA && (*same)->dim4 == NVA,
              "oovv_aaaa is all-alpha");
        const auto beta = cache->spin_block("oovv", "bbbb");
        check(beta.has_value() && (*beta)->dim1 == NOB && (*beta)->dim2 == NVB &&
                  (*beta)->dim3 == NOB && (*beta)->dim4 == NVB,
              "oovv_bbbb is all-beta");
    }

    // --- values against an independent oracle --------------------------------
    // This is the assertion that catches a wrong charge-density pairing: the
    // whole point is that slot 0 pairs with slot 2, not with slot 1.
    {
        struct Probe { const char *space; const char *spin; int i0, i1, i2, i3; };
        const Probe probes[] = {
            {"oovv", "abab", 1, 1, 0, 3},   // the mixed workhorse
            {"oovv", "aaaa", 2, 1, 0, 3},
            {"oovv", "bbbb", 1, 0, 1, 4},
            {"oovo", "abab", 0, 1, 1, 1},   // no RCC counterpart
            {"vovo", "abab", 2, 1, 0, 1},   // no RCC counterpart
            {"vovv", "abab", 3, 1, 1, 4},   // no RCC counterpart
            {"ovvo", "abab", 1, 2, 3, 0},
            {"ovov", "abab", 2, 0, 1, 2},
            {"vvvv", "abab", 1, 0, 3, 2},
            {"oooo", "abab", 2, 1, 0, 1},
        };
        for (const auto &probe : probes)
        {
            const auto block = cache->spin_block(probe.space, probe.spin);
            if (!block)
            {
                check(false, std::string("probe block ") + probe.space + "_" + probe.spin +
                                 " is stored");
                continue;
            }
            // Assert the probe indices are in range BEFORE reading. Tensor4D's
            // own check is a debug assert, compiled out in release, so an
            // out-of-range probe would otherwise read adjacent memory and report
            // a value mismatch -- pointing at the production code when the
            // fixture is what is wrong. That happened while writing this gate:
            // two probes used i1=2 against a beta-occupied extent of 2.
            {
                const int extents[4] = {
                    (*block)->dim1, (*block)->dim2, (*block)->dim3, (*block)->dim4};
                const int used[4] = {probe.i0, probe.i2, probe.i1, probe.i3};
                bool in_range = true;
                for (int k = 0; k < 4; ++k)
                    if (used[k] < 0 || used[k] >= extents[k])
                        in_range = false;
                if (!in_range)
                {
                    std::printf("FAIL: probe %s_%s indexes out of range\n",
                                probe.space, probe.spin);
                    ++failures;
                    continue;
                }
            }
            // stored chemists (p r | q s) -> index order (i0, i2, i1, i3)
            const double got = (**block)(probe.i0, probe.i2, probe.i1, probe.i3);
            const double want = oracle(
                eri, reference, probe.space, probe.spin,
                probe.i0, probe.i1, probe.i2, probe.i3);
            if (std::fabs(got - want) > 1e-10)
            {
                std::printf("FAIL: %s_%s value (got %.12g, want %.12g)\n",
                            probe.space, probe.spin, got, want);
                ++failures;
            }
        }
    }

    // --- a mixed block is NOT any pure-spin block relabeled -------------------
    // The falsifier for the "three copies of the RCC cache" model. If this ever
    // passes, the mixed transform is not actually mixed and U3.1 is unnecessary.
    {
        const auto mixed = cache->spin_block("oovv", "abab");
        const auto alpha = cache->spin_block("oovv", "aaaa");
        if (mixed && alpha)
        {
            bool identical = ((*mixed)->dim1 == (*alpha)->dim1 &&
                              (*mixed)->dim2 == (*alpha)->dim2 &&
                              (*mixed)->dim3 == (*alpha)->dim3 &&
                              (*mixed)->dim4 == (*alpha)->dim4 &&
                              (*mixed)->data == (*alpha)->data);
            check(!identical,
                  "oovv_abab differs from oovv_aaaa (it is a mixed transform)");
        }
    }

    // --- the RCC members are left empty on a UCC build ------------------------
    // So an RHF consumer reading them gets an obviously-empty tensor rather than
    // a plausible wrong one, and `spin_block` never silently falls back.
    {
        check(cache->oooo.data.empty() && cache->oovv.data.empty() &&
                  cache->vvvv.data.empty(),
              "the untagged RCC members stay empty on a UCC build");
        check(!cache->spin_block("oovv", "aabb").has_value(),
              "an unstored block errors rather than falling back");
    }

    // --- the memory report is labelled per spin block -------------------------
    {
        bool found = false;
        for (const auto &entry : cache->memory_report)
            if (entry.label == "oovv_abab")
                found = true;
        check(found, "memory report labels blocks by space and spin");
        check(cache->total_bytes > 0, "memory report accumulates bytes");
    }

    // --- RHF-degenerate: every spin block coincides ---------------------------
    // Kept because it catches a transposed SPACE index for free. It is explicitly
    // NOT sufficient: with C_alpha == C_beta the (a|b) and (b|a) pair orderings
    // coincide, so a swapped mixed pair passes it. The asymmetric cases above are
    // what actually guard that.
    {
        const UHFReference degenerate = make_reference(/*degenerate=*/true);
        const auto same = build_ucc_spin_block_cache_from_eri(
            eri, NB, degenerate,
            {{"oovv", "aaaa"}, {"oovv", "abab"}, {"oovv", "bbbb"}});
        check(same.has_value(), "degenerate cache builds");
        if (same)
        {
            const auto aa = same->spin_block("oovv", "aaaa");
            const auto ab = same->spin_block("oovv", "abab");
            const auto bb = same->spin_block("oovv", "bbbb");
            check(aa && ab && bb, "degenerate blocks all stored");
            if (aa && ab && bb)
            {
                check((*aa)->data == (*ab)->data && (*ab)->data == (*bb)->data,
                      "with C_alpha == C_beta every spin block coincides");
            }
        }
    }

    // --- malformed requests fail loudly --------------------------------------
    {
        check(!build_ucc_spin_block_cache_from_eri(
                   eri, NB, reference, {{"oov", "abab"}}).has_value(),
              "a space pattern that is not four slots is rejected");
        check(!build_ucc_spin_block_cache_from_eri(
                   eri, NB, reference, {{"oovv", "abx"}}).has_value(),
              "a spin tag that is not four slots is rejected");
        check(!build_ucc_spin_block_cache_from_eri(
                   eri, NB, reference, {{"oovx", "abab"}}).has_value(),
              "a non-o/v space character is rejected");
        check(!build_ucc_spin_block_cache_from_eri(
                   eri, NB, reference, {{"oovv", "abaX"}}).has_value(),
              "a non-a/b spin character is rejected");

        UHFReference empty = make_reference(false);
        empty.n_virt_beta = 0;
        check(!build_ucc_spin_block_cache_from_eri(
                   eri, NB, empty, {{"oovv", "abab"}}).has_value(),
              "an empty spin space is rejected");
    }

    // ======================================================================
    // U3.4 -- the open-shell MP2 limit.
    //
    // The scope framed this as "reproduce UMP2 from a single Jacobi step", which
    // would need the solver, a UHF reference threaded through the runtime, and an
    // SCF to produce one. None of that is necessary to answer the question this
    // step actually asks -- are the spin-blocked integrals and denominators right
    // TOGETHER -- because first-order MP2 amplitudes are t2 = <ij||ab> / D in
    // closed form. So the check assembles the UMP2 correlation energy directly
    // from U3.1's cache and U2.1's denominators and compares against the SAME
    // formula the production UMP2 kernel uses (mp2_ump2.cpp `canonical_kernel`),
    // evaluated here from an independent transform.
    //
    // This is a STRONGER gate than a Jacobi step and a cheaper one: it needs no
    // solver, no SCF and no PySCF, and it fails if EITHER the integrals or the
    // denominators are wrong, or if they are individually right but misaligned.
    //
    // The correspondence it rests on, verified against mp2_internal.cpp:
    //
    //   UMP2  ovOV = transform_eri(eri, nb, Ca_occ, Ca_virt, Cb_occ, Cb_virt)
    //   U3.1  oovv_abab: slots (o_a, o_b, v_a, v_b), chemists pairing
    //         (slot0,slot2)(slot1,slot3) = (Coa, Cva)(Cob, Cvb)
    //
    // i.e. the same four matrices in the same order -- so the mixed block must
    // agree with the production UMP2 mixed-spin ERI bitwise, not merely closely.
    // ======================================================================
    {
        // Orbital energies for the same asymmetric partition the cache uses.
        // Deliberately NOT degenerate between spins: an aa/bb mix-up must move
        // the energy rather than cancel.
        UHFReference mp2_ref = make_reference(/*degenerate=*/false);
        Eigen::VectorXd epsa(NOA + NVA), epsb(NOB + NVB);
        for (int p = 0; p < NOA + NVA; ++p)
            epsa(p) = -1.5 + 0.37 * static_cast<double>(p);
        for (int p = 0; p < NOB + NVB; ++p)
            epsb(p) = -1.1 + 0.29 * static_cast<double>(p);
        mp2_ref.eps_alpha = epsa;
        mp2_ref.eps_beta = epsb;

        const auto mp2_cache = build_ucc_spin_block_cache_from_eri(
            eri, NB, mp2_ref,
            {{"oovv", "aaaa"}, {"oovv", "abab"}, {"oovv", "bbbb"}});
        check(mp2_cache.has_value(), "mp2-limit cache builds");

        if (mp2_cache)
        {
            const auto aa = mp2_cache->spin_block("oovv", "aaaa");
            const auto ab = mp2_cache->spin_block("oovv", "abab");
            const auto bb = mp2_cache->spin_block("oovv", "bbbb");
            check(aa && ab && bb, "mp2-limit blocks stored");

            if (aa && ab && bb)
            {
                // Same-spin channels: E_ss = 1/2 sum t (g_ab - g_ba), with the
                // chemists block indexed (i,a,j,b) exactly as UMP2's ovov is.
                double e_ss = 0.0;
                for (int i = 0; i < NOA; ++i)
                    for (int j = 0; j < NOA; ++j)
                        for (int a = 0; a < NVA; ++a)
                            for (int b = 0; b < NVA; ++b)
                            {
                                const double gab = (**aa)(i, a, j, b);
                                const double gba = (**aa)(i, b, j, a);
                                const double d = epsa(i) + epsa(j)
                                                 - epsa(NOA + a) - epsa(NOA + b);
                                e_ss += 0.5 * (gab / d) * (gab - gba);
                            }
                for (int i = 0; i < NOB; ++i)
                    for (int j = 0; j < NOB; ++j)
                        for (int a = 0; a < NVB; ++a)
                            for (int b = 0; b < NVB; ++b)
                            {
                                const double gab = (**bb)(i, a, j, b);
                                const double gba = (**bb)(i, b, j, a);
                                const double d = epsb(i) + epsb(j)
                                                 - epsb(NOB + a) - epsb(NOB + b);
                                e_ss += 0.5 * (gab / d) * (gab - gba);
                            }

                // Mixed channel: no exchange partner, so it is opposite-spin only.
                double e_os = 0.0;
                for (int i = 0; i < NOA; ++i)
                    for (int j = 0; j < NOB; ++j)
                        for (int a = 0; a < NVA; ++a)
                            for (int b = 0; b < NVB; ++b)
                            {
                                const double g = (**ab)(i, a, j, b);
                                const double d = epsa(i) + epsb(j)
                                                 - epsa(NOA + a) - epsb(NOB + b);
                                e_os += (g / d) * g;
                            }

                // Independent oracle: the same three channels rebuilt from the
                // raw AO ERI by an explicit transform, never touching the cache.
                // If the cache mis-slices a block, or pairs the wrong coefficient
                // matrices, these disagree.
                const auto oracle_channel =
                    [&](const std::string &space, const std::string &spin,
                        int no1, int nv1, int no2, int nv2,
                        const Eigen::VectorXd &e1, const Eigen::VectorXd &e2,
                        bool exchange) {
                        double total = 0.0;
                        for (int i = 0; i < no1; ++i)
                            for (int j = 0; j < no2; ++j)
                                for (int a = 0; a < nv1; ++a)
                                    for (int b = 0; b < nv2; ++b)
                                    {
                                        const double gab = oracle(
                                            eri, mp2_ref, space, spin, i, j, a, b);
                                        const double d = e1(i) + e2(j)
                                                         - e1(no1 + a) - e2(no2 + b);
                                        if (!exchange)
                                        {
                                            total += (gab / d) * gab;
                                            continue;
                                        }
                                        const double gba = oracle(
                                            eri, mp2_ref, space, spin, i, j, b, a);
                                        total += 0.5 * (gab / d) * (gab - gba);
                                    }
                        return total;
                    };

                const double want_ss =
                    oracle_channel("oovv", "aaaa", NOA, NVA, NOA, NVA, epsa, epsa, true)
                    + oracle_channel("oovv", "bbbb", NOB, NVB, NOB, NVB, epsb, epsb, true);
                const double want_os =
                    oracle_channel("oovv", "abab", NOA, NVA, NOB, NVB, epsa, epsb, false);

                if (std::fabs(e_ss - want_ss) > 1e-10)
                {
                    std::printf("FAIL: mp2 same-spin channel (got %.12g, want %.12g)\n",
                                e_ss, want_ss);
                    ++failures;
                }
                if (std::fabs(e_os - want_os) > 1e-10)
                {
                    std::printf("FAIL: mp2 opposite-spin channel (got %.12g, want %.12g)\n",
                                e_os, want_os);
                    ++failures;
                }

                // Both channels must be genuinely non-zero, or the agreement
                // above is the agreement of two zeros. The mixed channel is the
                // one that only exists under UHF, so it is asserted separately.
                check(std::fabs(e_ss) > 1e-6, "same-spin MP2 channel is non-trivial");
                check(std::fabs(e_os) > 1e-6, "opposite-spin MP2 channel is non-trivial");
            }
        }
    }

    // --- U5.1a: the canonical UCC block set ---------------------------------
    //
    // NO METHOD IS INVOLVED, which is the design and the reason this is a closed
    // set rather than a vocabulary passed in from ccgen. RCC's
    // build_tensor_cc_block_cache takes no block list either -- the set IS its
    // struct's seven named members, built unconditionally, and it OVER-BUILDS
    // (measured: ccsd and ccsdt both read 6 of the 7; `ovvo` is never touched).
    // Nothing is negotiated with the emitter, so nothing can drift.
    //
    // The exact set is pinned against ccgen's `_canonical_eri_blocks_for` in
    // test_ucc_eri_symmetry.py, which is where a C++/Python divergence would be
    // caught. What is asserted HERE is the structure that makes the two agree:
    // the orbit rule, the counts it produces, and the canonical-member choice.
    {
        const auto blocks = ucc_canonical_blocks();
        check(blocks.size() == 24, "the canonical set is 24 stored arrays");

        std::map<std::string, std::vector<std::string>> by_tag;
        for (const auto &[space, tag] : blocks)
            by_tag[tag].push_back(space);

        // 7 / 10 / 7. The mixed block needs MORE because two of the four
        // physicist symmetries are not its symmetries (they map `abab` to
        // `baba`), so its orbits are smaller and more of them are needed to
        // cover the same sixteen patterns.
        check(by_tag.size() == 3, "exactly three ERI spin blocks are stored");
        check(by_tag["aaaa"].size() == 7, "aaaa folds sixteen patterns into 7");
        check(by_tag["bbbb"].size() == 7, "bbbb folds sixteen patterns into 7");
        check(by_tag["abab"].size() == 10, "abab folds sixteen patterns into 10");

        // `baba` must NOT be stored: it is `abab` under the particle swap, so
        // storing it costs ~33% more memory to avoid one explicit swap.
        check(!by_tag.contains("baba"), "baba is not stored (it is abab swapped)");

        // Every one of the sixteen o/v patterns must be REACHABLE in each tag,
        // or a kernel would ask for a block that was never built. This is the
        // property the counts above are a consequence of -- asserted directly so
        // a wrong count and a wrong orbit rule cannot cancel out.
        for (const auto &[tag, spaces] : by_tag)
        {
            std::set<std::string> reachable;
            for (const auto &space : spaces)
                for (const auto &perm : std::array<std::array<int, 4>, 4>{{
                         {{0, 1, 2, 3}}, {{1, 0, 3, 2}}, {{2, 3, 0, 1}}, {{3, 2, 1, 0}}}})
                {
                    if (!eri_permutation_preserves_block(tag, perm))
                        continue;
                    std::string image(4, 'o');
                    for (std::size_t slot = 0; slot < 4; ++slot)
                        image[slot] = space[static_cast<std::size_t>(perm[slot])];
                    reachable.insert(image);
                }
            check(reachable.size() == 16,
                  "every o/v pattern is reachable in block '" + tag + "'");
        }

        // The canonical member of each orbit is its LEXICOGRAPHIC MINIMUM ('o'
        // sorts before 'v'). The emitted kernels name these blocks, so the choice
        // is interface, not cosmetics -- ccgen picks the same member by walking
        // the patterns in sorted order, and if the two ever disagree the emitted
        // reads would name arrays the cache never built.
        //
        // Note this does NOT mean every member starts with 'o': `vovo` and `vovv`
        // are canonical because they are ALONE in their orbit under abab's
        // reduced symmetry group. An earlier version of this assertion assumed
        // occupied-first and failed on exactly those two -- the assumption was
        // wrong, not the code.
        for (const auto &[space, tag] : blocks)
        {
            std::string minimum = space;
            for (const auto &perm : std::array<std::array<int, 4>, 4>{{
                     {{0, 1, 2, 3}}, {{1, 0, 3, 2}}, {{2, 3, 0, 1}}, {{3, 2, 1, 0}}}})
            {
                if (!eri_permutation_preserves_block(tag, perm))
                    continue;
                std::string image(4, 'o');
                for (std::size_t slot = 0; slot < 4; ++slot)
                    image[slot] = space[static_cast<std::size_t>(perm[slot])];
                minimum = std::min(minimum, image);
            }
            check(space == minimum,
                  "'" + space + "' is the lexicographic minimum of its orbit in " + tag);
        }

        // Every block must be buildable from a real reference -- a set naming a
        // block the builder rejects would be a vocabulary that cannot be used.
        const auto full = build_ucc_spin_block_cache_from_eri(
            eri, NB, reference, blocks);
        check(full.has_value(), "every canonical block builds from a reference");
        if (full)
            check(full->spin_blocks.size() == 24, "all 24 are materialized");
    }

    // --- the permutation predicate mirrors ccgen's ---------------------------
    {
        constexpr std::array<int, 4> identity{{0, 1, 2, 3}};
        constexpr std::array<int, 4> particle{{1, 0, 3, 2}};
        constexpr std::array<int, 4> bra_ket{{2, 3, 0, 1}};
        constexpr std::array<int, 4> product{{3, 2, 1, 0}};

        for (const char *tag : {"aaaa", "bbbb"})
        {
            check(eri_permutation_preserves_block(tag, identity) &&
                      eri_permutation_preserves_block(tag, particle) &&
                      eri_permutation_preserves_block(tag, bra_ket) &&
                      eri_permutation_preserves_block(tag, product),
                  std::string("same-spin block '") + tag + "' keeps all four symmetries");
        }

        check(eri_permutation_preserves_block("abab", identity), "abab keeps identity");
        check(eri_permutation_preserves_block("abab", bra_ket), "abab keeps bra<->ket");
        check(!eri_permutation_preserves_block("abab", particle),
              "abab does NOT keep the particle swap (it maps to baba)");
        check(!eri_permutation_preserves_block("abab", product),
              "abab does NOT keep the product (it maps to baba)");

        check(!eri_permutation_preserves_block("aab", identity),
              "a malformed tag is rejected rather than silently accepted");
    }

    // --- U5.2b: the physicist rebind ----------------------------------------
    //
    // Every block is SELF-SOURCED: swap_mid_axes on the block stored under its
    // own (space, tag) key. No source map, no bra<->ket hop, no permuted tag.
    //
    // GETTING HERE TOOK TWO WRONG TURNS, both from treating a stored key as if it
    // named a CHEMISTS pattern. It does not -- U3.1 keys by the PHYSICIST (space,
    // spin) and applies the (p r | q s) pairing internally. Under that misreading
    // three mixed blocks appear to need a source that is not stored, and the spin
    // tag appears to need permuting (`abab` -> `aabb`). Both are convincing and
    // both are artifacts.
    //
    // WHY THEY SURVIVED: a same-spin check cannot tell the two hypotheses apart,
    // because `aaaa` is invariant under the tag permutation. The mixed-block
    // assertions below are the ones that discriminate, which is why they are here
    // rather than a same-spin spot check.
    {
        const auto blocks = ucc_canonical_blocks();
        const auto chem = build_ucc_spin_block_cache_from_eri(
            eri, NB, reference, blocks);
        check(chem.has_value(), "chemists cache builds for the rebind");

        if (chem)
        {
            const auto phys =
                HartreeFock::Correlation::CC::rebind_physicist_ucc(*chem);

            // The omission that makes the RCC rebind unusable here: it builds a
            // fresh cache from the seven NAMED members and never copies
            // spin_blocks, so it would return an empty cache that still looks
            // structurally valid.
            check(phys.spin_blocks.size() == 24,
                  "the rebind carries all 24 spin blocks");

            for (const auto &[space, tag] : blocks)
            {
                const auto src = chem->spin_block(space, tag);
                const auto out = phys.spin_block(space, tag);
                if (!src || !out)
                {
                    check(false, "block " + space + "_" + tag + " survives the rebind");
                    continue;
                }

                // swap_mid_axes: out(p,q,r,s) = in(p,r,q,s), dims (d1,d3,d2,d4).
                check((*out)->dim1 == (*src)->dim1 && (*out)->dim2 == (*src)->dim3 &&
                          (*out)->dim3 == (*src)->dim2 && (*out)->dim4 == (*src)->dim4,
                      "block " + space + "_" + tag + " has swapped middle dims");

                for (int p = 0; p < (*out)->dim1; ++p)
                    for (int q = 0; q < (*out)->dim2; ++q)
                        for (int r = 0; r < (*out)->dim3; ++r)
                            for (int t = 0; t < (*out)->dim4; ++t)
                                if ((**out)(p, q, r, t) != (**src)(p, r, q, t))
                                {
                                    check(false, "block " + space + "_" + tag +
                                                     " values follow swap_mid_axes");
                                    p = (*out)->dim1;
                                    q = (*out)->dim2;
                                    r = (*out)->dim3;
                                    break;
                                }
            }

            // THE DISCRIMINATING ASSERTION. A mixed block's middle dims differ
            // (noa != nob and nva != nvb in this fixture), so a rebind that
            // permuted the spin tag -- or read a different block as its source --
            // would produce the wrong SHAPE here, not merely wrong values. A
            // same-spin block cannot show this: its dims are symmetric under the
            // swap, which is exactly why the wrong hypothesis survived so long.
            const auto mixed = phys.spin_block("oovv", "abab");
            check(mixed.has_value(), "the mixed workhorse survives the rebind");
            if (mixed)
            {
                check((*mixed)->dim1 == NOA && (*mixed)->dim2 == NOB &&
                          (*mixed)->dim3 == NVA && (*mixed)->dim4 == NVB,
                      "oovv_abab rebinds to (noa, nob, nva, nvb) -- physicist order");
                check((*mixed)->dim2 != (*mixed)->dim3,
                      "the fixture's mixed block has ASYMMETRIC middle dims, so a "
                      "permuted tag would change its shape");
            }

            // The three blocks that a chemists-key misreading says need a hop.
            // They resolve from their own key like every other block.
            for (const char *space : {"oovo", "vovo", "vovv"})
                check(phys.spin_block(space, "abab").has_value(),
                      std::string("<") + space + "|_abab rebinds from its own key");
        }
    }

    // --- U5.2c: the MP2 limit survives the rebind ---------------------------
    //
    // U3.4 assembles the UMP2 correlation energy from the CHEMISTS cache, indexing
    // (i,a,j,b). This re-assembles the SAME energy from the REBOUND cache, indexing
    // physicist (i,j,a,b). The two index orders are different, the data is
    // different (mid axes swapped), and the answer must be identical -- which is
    // what "the rebind preserves the physics" means concretely.
    //
    // This is the first check that spans the whole U5.1/U5.2 chain: reference ->
    // spin-blocked transform -> rebind. A rebind that moved the right bytes to the
    // wrong key, or swapped the wrong axes, changes this number.
    {
        UHFReference mp2_ref = make_reference(/*degenerate=*/false);
        Eigen::VectorXd epsa(NOA + NVA), epsb(NOB + NVB);
        for (int p = 0; p < NOA + NVA; ++p)
            epsa(p) = -1.5 + 0.37 * static_cast<double>(p);
        for (int p = 0; p < NOB + NVB; ++p)
            epsb(p) = -1.1 + 0.29 * static_cast<double>(p);
        mp2_ref.eps_alpha = epsa;
        mp2_ref.eps_beta = epsb;

        const std::vector<std::pair<std::string, std::string>> mp2_blocks{
            {"oovv", "aaaa"}, {"oovv", "abab"}, {"oovv", "bbbb"}};
        const auto chem_cache =
            build_ucc_spin_block_cache_from_eri(eri, NB, mp2_ref, mp2_blocks);
        check(chem_cache.has_value(), "mp2 chemists cache builds");
        if (!chem_cache)
            return failures == 0 ? 0 : 1;

        const auto phys_cache =
            HartreeFock::Correlation::CC::rebind_physicist_ucc(*chem_cache);

        // Same-spin channel from a CHEMISTS block, indexed (i,a,j,b).
        const auto chem_ss = [&](const Tensor4D &g, int no, int nv,
                                 const Eigen::VectorXd &eps) {
            double total = 0.0;
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                        {
                            const double gab = g(i, a, j, b);
                            const double gba = g(i, b, j, a);
                            const double d =
                                eps(i) + eps(j) - eps(no + a) - eps(no + b);
                            total += 0.5 * (gab / d) * (gab - gba);
                        }
            return total;
        };
        // The same channel from the REBOUND block, indexed physicist (i,j,a,b).
        const auto phys_ss = [&](const Tensor4D &g, int no, int nv,
                                 const Eigen::VectorXd &eps) {
            double total = 0.0;
            for (int i = 0; i < no; ++i)
                for (int j = 0; j < no; ++j)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                        {
                            const double gab = g(i, j, a, b);
                            const double gba = g(i, j, b, a);
                            const double d =
                                eps(i) + eps(j) - eps(no + a) - eps(no + b);
                            total += 0.5 * (gab / d) * (gab - gba);
                        }
            return total;
        };

        const auto c_aa = chem_cache->spin_block("oovv", "aaaa");
        const auto c_ab = chem_cache->spin_block("oovv", "abab");
        const auto c_bb = chem_cache->spin_block("oovv", "bbbb");
        const auto p_aa = phys_cache.spin_block("oovv", "aaaa");
        const auto p_ab = phys_cache.spin_block("oovv", "abab");
        const auto p_bb = phys_cache.spin_block("oovv", "bbbb");
        check(c_aa && c_ab && c_bb && p_aa && p_ab && p_bb,
              "both caches hold all three mp2 blocks");

        if (c_aa && c_ab && c_bb && p_aa && p_ab && p_bb)
        {
            const double chem_total =
                chem_ss(**c_aa, NOA, NVA, epsa) + chem_ss(**c_bb, NOB, NVB, epsb);
            const double phys_total =
                phys_ss(**p_aa, NOA, NVA, epsa) + phys_ss(**p_bb, NOB, NVB, epsb);
            if (std::fabs(chem_total - phys_total) > 1e-10)
            {
                std::printf("FAIL: same-spin MP2 differs across the rebind "
                            "(chemists %.12g, physicist %.12g)\n",
                            chem_total, phys_total);
                ++failures;
            }

            // The mixed channel, which is the one that only exists under UHF and
            // the one a mis-keyed rebind would corrupt without changing shape on
            // the same-spin blocks.
            double chem_os = 0.0;
            for (int i = 0; i < NOA; ++i)
                for (int j = 0; j < NOB; ++j)
                    for (int a = 0; a < NVA; ++a)
                        for (int b = 0; b < NVB; ++b)
                        {
                            const double g = (**c_ab)(i, a, j, b);
                            const double d = epsa(i) + epsb(j)
                                             - epsa(NOA + a) - epsb(NOB + b);
                            chem_os += (g / d) * g;
                        }
            double phys_os = 0.0;
            for (int i = 0; i < NOA; ++i)
                for (int j = 0; j < NOB; ++j)
                    for (int a = 0; a < NVA; ++a)
                        for (int b = 0; b < NVB; ++b)
                        {
                            const double g = (**p_ab)(i, j, a, b);
                            const double d = epsa(i) + epsb(j)
                                             - epsa(NOA + a) - epsb(NOB + b);
                            phys_os += (g / d) * g;
                        }
            if (std::fabs(chem_os - phys_os) > 1e-10)
            {
                std::printf("FAIL: mixed MP2 differs across the rebind "
                            "(chemists %.12g, physicist %.12g)\n",
                            chem_os, phys_os);
                ++failures;
            }

            // Non-trivial, or the agreement above is two zeros agreeing.
            check(std::fabs(chem_total) > 1e-6, "the same-spin channel is non-trivial");
            check(std::fabs(chem_os) > 1e-6, "the mixed channel is non-trivial");
        }
    }

    // --- U3b.1: the four spin-resolved orbital counts ------------------------
    //
    // Every generated kernel takes its loop bounds and result shape from the
    // reference. Before this, a UCC reference carried only `orbital_partition` --
    // one (n_occ, n_virt) pair, left DEFAULT by build_ucc_fock_blocks -- so every
    // kernel allocated a (0,0) result and the first end-to-end `ucc2` run failed
    // with "sector residual shape mismatch at (rank 1, tag aa)".
    //
    // The four counts are ADDITIVE: `orbital_partition` is untouched, because the
    // RCC kernels read it (6 reads each of n_occ/n_virt in the rank-3 TU) and must
    // keep doing so byte-identically.
    {
        const UHFReference source = make_reference(/*degenerate=*/false);
        Eigen::MatrixXd fock_a = Eigen::MatrixXd::Zero(NOA + NVA, NOA + NVA);
        Eigen::MatrixXd fock_b = Eigen::MatrixXd::Zero(NOB + NVB, NOB + NVB);
        for (int p = 0; p < NOA + NVA; ++p)
            fock_a(p, p) = 1.0 + static_cast<double>(p);
        for (int p = 0; p < NOB + NVB; ++p)
            fock_b(p, p) = 100.0 + static_cast<double>(p);

        const auto ref = build_ucc_fock_blocks(source, fock_a, fock_b);
        check(ref.has_value(), "the UCC reference builds");

        if (ref)
        {
            check(ref->n_occ_alpha == NOA && ref->n_occ_beta == NOB &&
                      ref->n_virt_alpha == NVA && ref->n_virt_beta == NVB,
                  "the four counts come from the UHFReference");

            // The fixture is deliberately asymmetric in all four extents, so a
            // count wired to the wrong spin -- or to the wrong space -- changes
            // the value rather than coinciding.
            check(NOA != NOB && NVA != NVB && NOA != NVA,
                  "the fixture's four extents are all distinct");

            // Accessors, which is what the emitter will call per index.
            const auto oa = ref->occupied_count('a');
            const auto ob = ref->occupied_count('b');
            const auto va = ref->virtual_count('a');
            const auto vb = ref->virtual_count('b');
            check(oa && *oa == NOA, "occupied_count('a')");
            check(ob && *ob == NOB, "occupied_count('b')");
            check(va && *va == NVA, "virtual_count('a')");
            check(vb && *vb == NVB, "virtual_count('b')");

            // A bad spin errors rather than returning a plausible count: a
            // silently-wrong loop bound is precisely the defect U3b removes.
            check(!ref->occupied_count('x').has_value(),
                  "occupied_count rejects a non-spin character");
            check(!ref->virtual_count('\0').has_value(),
                  "virtual_count rejects a non-spin character");

            // orbital_partition stays DEFAULT on a UCC reference. Filling it with
            // either spin's counts would be a plausible wrong answer, and the
            // emitter must be made to stop reading it (U3b.2) rather than be fed
            // a value that happens to work for one spin.
            check(ref->orbital_partition.n_occ == 0 &&
                      ref->orbital_partition.n_virt == 0,
                  "orbital_partition is left default on a UCC reference");
        }
    }

    if (failures == 0)
        std::printf("cc_ucc_spin_blocks: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
