// U2: the spin-resolved (UCC) amplitude denominator.
//
// The RCC denominator draws every slot's orbital energy from one occ/virt pair.
// Under UCC a block's slots live in DIFFERENT spin spaces -- `t2_abab` has one
// alpha and one beta slot per half -- so the denominator must be resolved per
// slot, not per rank. This pins that.
//
// NON-SQUARE AND SPIN-ASYMMETRIC DIMENSIONS ARE THE POINT. With n_occ_alpha ==
// n_occ_beta a swapped spin index still lands in bounds and reads a plausible
// number; with n_occ == n_virt a transposed occ/vir slot does too. Every case
// below uses noa=4 nva=3 nob=2 nvb=5, so all four extents differ.
//
// The tag is per-slot spin in the tensor's own index order, which is OCC-FIRST
// then VIR (`rank_dims`). ccgen's UCC tags are bra(vir)-half-then-ket(occ), so a
// caller converts; the tag reaching this builder is always occ-half-first.

#include "post_hf/cc/amplitudes.h"
#include "post_hf/cc/common.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using HartreeFock::Correlation::CC::build_ucc_block_denominator;
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

    void check_close(double got, double want, double tol, const std::string &what)
    {
        if (!(std::fabs(got - want) <= tol))
        {
            std::printf("FAIL: %s (got %.12g, want %.12g)\n", what.c_str(), got, want);
            ++failures;
        }
    }

    // Deliberately asymmetric: noa != nob, nva != nvb, and n_occ != n_virt in
    // both spins, so no index confusion can survive silently.
    constexpr int NOA = 4;
    constexpr int NVA = 3;
    constexpr int NOB = 2;
    constexpr int NVB = 5;

    UHFReference make_reference()
    {
        UHFReference reference;
        reference.n_occ_alpha = NOA;
        reference.n_virt_alpha = NVA;
        reference.n_occ_beta = NOB;
        reference.n_virt_beta = NVB;
        reference.n_mo = NOA + NVA;

        // Distinct, easily-checked energies. Alpha and beta deliberately differ so
        // that reading the wrong spin's vector changes the answer.
        reference.eps_alpha = Eigen::VectorXd(NOA + NVA);
        for (int p = 0; p < NOA + NVA; ++p)
            reference.eps_alpha(p) = 1.0 + static_cast<double>(p);
        reference.eps_beta = Eigen::VectorXd(NOB + NVB);
        for (int p = 0; p < NOB + NVB; ++p)
            reference.eps_beta(p) = 100.0 + static_cast<double>(p);
        return reference;
    }
} // namespace

int main()
{
    const UHFReference reference = make_reference();

    // --- shapes come from each slot's own spin -------------------------------
    {
        const auto aa = build_ucc_block_denominator(reference, "aa");
        check(aa.has_value(), "t1_aa builds");
        if (aa)
        {
            check(aa->dims == std::vector<int>{NOA, NVA}, "t1_aa dims are (noa, nva)");
        }

        const auto bb = build_ucc_block_denominator(reference, "bb");
        check(bb.has_value() && bb->dims == std::vector<int>{NOB, NVB},
              "t1_bb dims are (nob, nvb)");

        // The block that makes this UCC rather than RCC: one slot per spin per half.
        const auto abab = build_ucc_block_denominator(reference, "abab");
        check(abab.has_value(), "t2_abab builds");
        if (abab)
        {
            check(abab->dims == std::vector<int>{NOA, NOB, NVA, NVB},
                  "t2_abab dims are (noa, nob, nva, nvb)");
        }

        const auto aaaa = build_ucc_block_denominator(reference, "aaaa");
        check(aaaa.has_value() && aaaa->dims == std::vector<int>{NOA, NOA, NVA, NVA},
              "t2_aaaa dims are all-alpha");
        const auto bbbb = build_ucc_block_denominator(reference, "bbbb");
        check(bbbb.has_value() && bbbb->dims == std::vector<int>{NOB, NOB, NVB, NVB},
              "t2_bbbb dims are all-beta");
    }

    // --- values: every slot uses ITS OWN spin's orbital energies -------------
    {
        // t1_aa[i,a] = eps_a(i) - eps_a(noa + a)
        const auto aa = build_ucc_block_denominator(reference, "aa");
        if (aa)
        {
            check_close((*aa)({2, 1}), reference.eps_alpha(2) - reference.eps_alpha(NOA + 1),
                        1e-12, "t1_aa value is alpha-only");
        }

        // t1_bb must read the BETA vector -- and beta's energies are ~100 apart
        // from alpha's, so a spin mix-up is a factor-100 error, not a rounding one.
        const auto bb = build_ucc_block_denominator(reference, "bb");
        if (bb)
        {
            check_close((*bb)({1, 3}), reference.eps_beta(1) - reference.eps_beta(NOB + 3),
                        1e-12, "t1_bb value is beta-only");
        }

        // The load-bearing case: t2_abab[i,J,a,B] mixes both spin sets in ONE
        // element. Slot order is occ(a), occ(b), vir(a), vir(b).
        const auto abab = build_ucc_block_denominator(reference, "abab");
        if (abab)
        {
            const double want = reference.eps_alpha(3) + reference.eps_beta(1)
                                - reference.eps_alpha(NOA + 0) - reference.eps_beta(NOB + 4);
            check_close((*abab)({3, 1, 0, 4}), want, 1e-12,
                        "t2_abab value draws each slot from its own spin");
        }
    }

    // --- the RCC limit: a spin-degenerate reference reproduces the RHF form ---
    {
        UHFReference degenerate = make_reference();
        degenerate.n_occ_beta = NOA;
        degenerate.n_virt_beta = NVA;
        degenerate.eps_beta = degenerate.eps_alpha;

        const auto aaaa = build_ucc_block_denominator(degenerate, "aaaa");
        const auto abab = build_ucc_block_denominator(degenerate, "abab");
        const auto bbbb = build_ucc_block_denominator(degenerate, "bbbb");
        check(aaaa && abab && bbbb, "degenerate reference builds every rank-2 block");
        if (aaaa && abab && bbbb)
        {
            // With alpha == beta every block is the same tensor. This is the free
            // regression the scope calls for: it catches a transposed spin index
            // immediately, because a transposition is only invisible when the two
            // spin spaces agree -- which is exactly this case.
            bool same = aaaa->dims == abab->dims && abab->dims == bbbb->dims;
            double worst = 0.0;
            if (same)
            {
                for (std::size_t k = 0; k < aaaa->data.size(); ++k)
                {
                    worst = std::max(worst, std::fabs(aaaa->data[k] - abab->data[k]));
                    worst = std::max(worst, std::fabs(aaaa->data[k] - bbbb->data[k]));
                }
            }
            check(same, "degenerate reference: all rank-2 blocks share a shape");
            check_close(worst, 0.0, 1e-14,
                        "degenerate reference: aaaa == abab == bbbb elementwise");
        }
    }

    // --- a block whose occ and vir spin patterns DIFFER ----------------------
    //
    // Every tag above has the same spin string in both halves ("abab" is occ
    // (a,b) and vir (a,b)), so reading a virtual slot's spin from the occupied
    // slot at the same position gives the right answer by accident. A tag like
    // "ab" + "ba" separates them, and it is a legitimate block: nothing requires
    // the two halves to agree slot-for-slot.
    {
        const auto skew = build_ucc_block_denominator(reference, "abba");
        check(skew.has_value(), "t2_abba builds");
        if (skew)
        {
            const std::vector<int> want_dims{NOA, NOB, NVB, NVA};
            check(skew->dims == want_dims, "t2_abba dims: occ (a,b) but vir (b,a)");
            // element (i=1 alpha, J=0 beta | B=4 beta, a=2 alpha) -- the last
            // index of each virtual space, so an off-by-one or a spin swap runs
            // off the end rather than reading a plausible neighbour.
            check_close((*skew)({1, 0, 4, 2}), reference.eps_alpha(1) + reference.eps_beta(0)
                            - reference.eps_beta(NOB + 4) - reference.eps_alpha(NOA + 2), 1e-12,
                        "t2_abba: each virtual slot uses ITS OWN spin, not the "
                        "occupied slot's at the same position");
        }
    }

    // --- rank 3, where a mixed block has an unequal split --------------------
    {
        const auto aab = build_ucc_block_denominator(reference, "aabaab");
        check(aab.has_value(), "t3_aabaab builds");
        if (aab)
        {
            check(aab->dims == std::vector<int>{NOA, NOA, NOB, NVA, NVA, NVB},
                  "t3_aabaab dims follow the tag slot by slot");
            const double want = reference.eps_alpha(0) + reference.eps_alpha(3)
                                + reference.eps_beta(1)
                                - reference.eps_alpha(NOA + 2) - reference.eps_alpha(NOA + 0)
                                - reference.eps_beta(NOB + 4);
            check_close((*aab)({0, 3, 1, 2, 0, 4}), want, 1e-12,
                        "t3_aabaab value is resolved per slot");
        }
    }

    // --- rejections, so a malformed tag fails loudly rather than silently ----
    {
        check(!build_ucc_block_denominator(reference, "").has_value(),
              "empty tag is rejected");
        check(!build_ucc_block_denominator(reference, "aab").has_value(),
              "odd-length tag is rejected");
        check(!build_ucc_block_denominator(reference, "axax").has_value(),
              "non-spin character is rejected");

        UHFReference truncated = make_reference();
        truncated.eps_beta = Eigen::VectorXd(1);
        truncated.eps_beta(0) = 0.0;
        check(!build_ucc_block_denominator(truncated, "abab").has_value(),
              "an eps vector too short for the partition is rejected");
    }

    if (failures == 0)
        std::printf("cc_ucc_denominator: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
