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

#include "post_hf/cc/tensor_backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using HartreeFock::Correlation::CC::build_ucc_spin_block_cache_from_eri;
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

    if (failures == 0)
        std::printf("cc_ucc_spin_blocks: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
