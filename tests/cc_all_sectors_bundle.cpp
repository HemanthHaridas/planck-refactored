// U4.0: the generated runtime accepts an ALL-SECTORS kernel bundle.
//
// The arbitrary-order runtime was built for RCC, where every excitation rank has
// one privileged "reference" residual (`residuals_by_rank[rank-1]`) and a rank-2n
// amplitude may carry EXTRA Sz sectors alongside it (the CCSDTQ `aaabaaab` case).
// `validate_kernel_bundle` encoded that as a hard requirement:
//
//     residuals_by_rank.size() == max_excitation_rank
//
// UCC breaks it. `ucc_adapt_equations` tags EVERY target -- `doubles_aaaa`, never
// a bare `doubles` -- so the emitter pushes zero per-rank residuals and five
// sector residuals, and the bundle was rejected before it could run. Measured on
// the emitted CCSD UCC TU: 0 pushes to residuals_by_rank, 5 to sector_residuals,
// max_excitation_rank = 2.
//
// WHY NOT PROMOTE ONE BLOCK PER RANK INTO THE REFERENCE SLOT. That was the
// obvious cheaper fix and it does not work: the slot is sized by `rank_dims`,
// which yields ONE shape per rank, while UCC blocks of a single rank have
// different shapes under UHF -- `aaaa` is (noa,noa,nva,nva) and `abab` is
// (noa,nob,nva,nvb). Promoting one would silently mis-size the other two, which
// is exactly the class of defect this whole UCC effort keeps finding.
//
// So `residuals_by_rank` becomes OPTIONAL: empty declares an all-sectors bundle.
// Empty-or-full, never partial -- a half-filled vector means a bundle lost a
// kernel, and that rank would otherwise evaluate as a silent zero contribution.
//
// Scope: this gate covers validation and evaluation only (U4.0). The Jacobi/DIIS
// update still requires matching ranks across amplitudes/residuals/denominators
// and is U4.1; that boundary is asserted here so it fails loudly rather than
// producing a wrong number if someone wires a solve too early.

#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/solver_arbitrary.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using HartreeFock::Correlation::CC::ArbitraryOrderDenominatorCache;
using HartreeFock::Correlation::CC::ArbitraryOrderRCCAmplitudes;
using HartreeFock::Correlation::CC::ArbitraryOrderTensorCCState;
using HartreeFock::Correlation::CC::evaluate_generated_arbitrary_order_residuals;
using HartreeFock::Correlation::CC::GeneratedArbitraryOrderKernels;
using HartreeFock::Correlation::CC::TensorND;

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

    // Asymmetric on purpose: a mixed block must not accidentally share a shape
    // with a same-spin one, or a mis-sized sector would still land in bounds.
    constexpr int NOA = 3, NVA = 4, NOB = 2, NVB = 5;

    std::vector<int> dims_for(const std::string &tag)
    {
        std::vector<int> dims;
        const int rank = static_cast<int>(tag.size()) / 2;
        for (int slot = 0; slot < rank; ++slot)
            dims.push_back(tag[static_cast<std::size_t>(slot)] == 'a' ? NOA : NOB);
        for (int slot = rank; slot < 2 * rank; ++slot)
            dims.push_back(tag[static_cast<std::size_t>(slot)] == 'a' ? NVA : NVB);
        return dims;
    }

    // A minimal UCC-shaped state: no reference blocks, one amplitude and one
    // denominator per spin block, mirroring what the UCC path will produce.
    ArbitraryOrderTensorCCState make_all_sectors_state(
        const std::vector<std::pair<int, std::string>> &blocks)
    {
        ArbitraryOrderTensorCCState state;
        state.max_excitation_rank = 2;
        for (const auto &[rank, tag] : blocks)
        {
            state.amplitudes.sectors.push_back({{rank, tag}, TensorND(dims_for(tag), 0.0)});
            state.denominators.sectors.push_back({{rank, tag}, TensorND(dims_for(tag), -1.0)});
        }
        return state;
    }

    GeneratedArbitraryOrderKernels make_all_sectors_bundle(
        const std::vector<std::pair<int, std::string>> &blocks)
    {
        GeneratedArbitraryOrderKernels kernels;
        kernels.max_excitation_rank = 2;
        kernels.energy = [](const auto &, const auto &, const auto &, const auto &)
        { return 0.0; };
        for (const auto &[rank, tag] : blocks)
        {
            kernels.sector_tags.push_back({rank, tag});
            const std::vector<int> dims = dims_for(tag);
            kernels.sector_residuals.push_back(
                {rank, tag,
                 [dims](const auto &, const auto &, const auto &, const auto &)
                 { return TensorND(dims, 0.25); }});
        }
        return kernels;
    }

    const std::vector<std::pair<int, std::string>> UCC_BLOCKS{
        {1, "aa"}, {1, "bb"}, {2, "aaaa"}, {2, "abab"}, {2, "bbbb"}};
} // namespace

int main()
{
    // --- the bundle reports its own mode ------------------------------------
    {
        const auto ucc = make_all_sectors_bundle(UCC_BLOCKS);
        check(ucc.is_all_sectors(), "a UCC bundle reports all-sectors");

        GeneratedArbitraryOrderKernels rcc;
        rcc.max_excitation_rank = 2;
        rcc.residuals_by_rank.resize(2);
        check(!rcc.is_all_sectors(), "a bundle with per-rank residuals does not");
    }

    // --- an all-sectors bundle EVALUATES ------------------------------------
    // The thing that was impossible before: this call used to fail validation.
    {
        const auto state = make_all_sectors_state(UCC_BLOCKS);
        const auto kernels = make_all_sectors_bundle(UCC_BLOCKS);

        const auto residuals =
            evaluate_generated_arbitrary_order_residuals(state, kernels);
        check(residuals.has_value(), "an all-sectors bundle evaluates");
        if (!residuals)
        {
            std::printf("  error: %s\n", residuals.error().c_str());
        }
        else
        {
            check(residuals->by_rank.empty(),
                  "an all-sectors evaluation leaves by_rank empty");
            check(residuals->sectors.size() == UCC_BLOCKS.size(),
                  "every declared sector produced a residual");

            // Each sector residual must keep ITS OWN shape -- the defect a shared
            // by_rank slot would have introduced.
            for (const auto &[key, tensor] : residuals->sectors)
            {
                check(tensor.dims == dims_for(key.second),
                      "sector residual '" + key.second + "' keeps its own shape");
            }
        }
    }

    // --- the mixed block really is shaped differently ------------------------
    // Guards the gate itself: if every block had the same dims, the shape
    // assertions above would pass vacuously.
    {
        check(dims_for("aaaa") != dims_for("abab"),
              "the fixture's mixed block differs in shape from the same-spin one");
        check(dims_for("aa") != dims_for("bb"),
              "the fixture's two singles blocks differ in shape");
    }

    // --- rejections: empty-or-full, never partial ----------------------------
    {
        // Partial means: fewer entries than max_excitation_rank, but every entry
        // PRESENT and callable. A `resize(1)` would leave a null std::function and
        // be caught by the missing-kernel guard instead -- which passes for the
        // wrong reason and cannot tell the two guards apart. Found by mutation:
        // disabling the size check left that version of this assertion green.
        auto partial = make_all_sectors_bundle(UCC_BLOCKS);
        partial.residuals_by_rank.push_back(
            [](const auto &, const auto &, const auto &, const auto &)
            { return TensorND(dims_for("aa"), 0.0); });   // 1 real kernel, need 2
        // The state must match the bundle's MODE, or an earlier guard fires and
        // the assertion passes for the wrong reason. A non-all-sectors bundle is
        // checked against reference-rank coverage first, so give it those ranks.
        // Found by mutation, then by printing which guard actually rejected.
        auto ref_state = make_all_sectors_state(UCC_BLOCKS);
        ref_state.amplitudes.by_rank.emplace_back(dims_for("aa"), 0.0);
        ref_state.amplitudes.by_rank.emplace_back(dims_for("aaaa"), 0.0);
        ref_state.denominators.by_rank.emplace_back(dims_for("aa"), -1.0);
        ref_state.denominators.by_rank.emplace_back(dims_for("aaaa"), -1.0);

        const auto partial_result =
            evaluate_generated_arbitrary_order_residuals(ref_state, partial);
        check(!partial_result.has_value(),
              "a PARTIALLY filled residuals_by_rank is rejected");
        if (!partial_result)
        {
            // Name the guard, so the assertion cannot be satisfied by a different
            // rejection reason.
            check(partial_result.error().find("residual kernel count") != std::string::npos,
                  "the partial bundle is rejected by the COUNT guard specifically");
        }

        // A bundle with no residuals of either kind would "converge" instantly at
        // the reference energy, which is the quietest possible wrong answer.
        GeneratedArbitraryOrderKernels empty;
        empty.max_excitation_rank = 2;
        empty.energy = [](const auto &, const auto &, const auto &, const auto &)
        { return 0.0; };
        const auto empty_state = make_all_sectors_state(UCC_BLOCKS);
        const auto empty_result =
            evaluate_generated_arbitrary_order_residuals(empty_state, empty);
        check(!empty_result.has_value(),
              "a bundle with no residual kernels at all is rejected");
        if (!empty_result)
        {
            check(empty_result.error().find("no residual kernels of either kind")
                      != std::string::npos,
                  "the empty bundle is rejected by the no-kernels guard specifically");
        }
    }

    // --- a declared sector with no amplitude block is still rejected ---------
    // The Gap B4 guard must survive the relaxation: it is what stops a sector's
    // contribution being silently dropped.
    {
        const auto kernels = make_all_sectors_bundle(UCC_BLOCKS);
        auto short_state = make_all_sectors_state(
            {{1, "aa"}, {1, "bb"}, {2, "aaaa"}});    // missing abab and bbbb
        check(!evaluate_generated_arbitrary_order_residuals(short_state, kernels).has_value(),
              "a declared sector with no amplitude block is rejected");
    }

    // --- U4.1 boundary, asserted so it fails loudly --------------------------
    // With by_rank empty the residuals report max_rank()==0 while the amplitudes
    // report 2, so the Jacobi/DIIS update's rank-coverage guard rejects. That is
    // the correct behavior TODAY: the update loop is U4.1 and is not written yet.
    // Asserting it here means wiring a solve too early fails loudly instead of
    // silently updating nothing.
    {
        const auto state = make_all_sectors_state(UCC_BLOCKS);
        const auto kernels = make_all_sectors_bundle(UCC_BLOCKS);
        const auto residuals =
            evaluate_generated_arbitrary_order_residuals(state, kernels);
        if (residuals)
        {
            check(residuals->max_rank() == 0,
                  "all-sectors residuals report max_rank 0 (no reference blocks)");
            check(state.amplitudes.max_rank() == 0,
                  "the all-sectors state likewise carries no reference amplitude ranks");
        }
    }

    // --- U4.1: the Jacobi/DIIS update drives an all-sectors state ----------
    //
    // MEASURED, NOT BUILT. U4.1 was scoped as work -- make pack/unpack and the
    // update loop tolerate an empty `by_rank`. Probing first showed they already
    // do, for two independent reasons that happen to compose:
    //
    //   - pack_amplitudes / unpack_amplitudes / pack_residuals iterate `by_rank`
    //     and then `sectors`. An empty `by_rank` simply contributes nothing, and
    //     the sector region starts at offset 0 instead of after the reference
    //     blocks. The packing is order-preserving either way.
    //   - the update's per-rank loop runs `rank <= max_rank()`, which is 0, so it
    //     is skipped entirely; the sector loop then does all the work, reading its
    //     denominators through `sector_tensor` (U2.2).
    //   - the rank-coverage guard compares 0 == 0 == 0 and passes.
    //
    // So the only thing that ever blocked an all-sectors solve was U4.0's
    // validation. This section exists to PIN that, because "it already works" is
    // exactly the kind of claim that silently stops being true.
    {
        auto state = make_all_sectors_state(UCC_BLOCKS);
        const auto kernels = make_all_sectors_bundle(UCC_BLOCKS);
        const auto residuals =
            evaluate_generated_arbitrary_order_residuals(state, kernels);
        check(residuals.has_value(), "all-sectors residuals evaluate for the update");

        if (residuals)
        {
            // Packing must round-trip with no reference blocks present. If the
            // offset bookkeeping assumed a non-empty `by_rank`, the sector region
            // would be written at the wrong place and this would not survive.
            const Eigen::VectorXd packed =
                HartreeFock::Correlation::CC::pack_amplitudes(state.amplitudes);
            std::size_t expected_elements = 0;
            for (const auto &[key, block] : state.amplitudes.sectors)
                expected_elements += block.size();
            check(static_cast<std::size_t>(packed.size()) == expected_elements,
                  "packing an all-sectors state covers exactly the sector blocks");

            HartreeFock::Correlation::CC::AmplitudeDIIS diis(8);
            const auto metrics =
                HartreeFock::Correlation::CC::update_amplitudes_with_jacobi_diis(
                    state.amplitudes, *residuals, state.denominators,
                    diis, 1.0, /*use_diis=*/false);
            check(metrics.has_value(), "the Jacobi update drives an all-sectors state");
            if (!metrics)
                std::printf("  error: %s\n", metrics.error().c_str());

            // EVERY block must have moved, and to its own value. The fixture uses
            // R = 0.25 and D = -1 throughout, so one undamped Jacobi step gives
            // t = R/D = -0.25. A block left at zero would mean the update silently
            // skipped it -- the failure mode an empty `by_rank` invites, since the
            // per-rank loop is what normally does the work.
            for (const auto &[key, block] : state.amplitudes.sectors)
            {
                bool all_stepped = !block.data.empty();
                for (const double value : block.data)
                    if (std::fabs(value - (-0.25)) > 1e-12)
                        all_stepped = false;
                check(all_stepped,
                      "sector '" + key.second + "' took its Jacobi step (t = R/D)");
            }
        }
    }

    if (failures == 0)
        std::printf("cc_all_sectors_bundle: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
