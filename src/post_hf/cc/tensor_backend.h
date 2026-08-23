#ifndef HF_POSTHF_CC_TENSOR_BACKEND_H
#define HF_POSTHF_CC_TENSOR_BACKEND_H

#include <cstddef>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "integrals/shellpair.h"
#include "post_hf/cc/amplitudes.h"
#include "post_hf/cc/common.h"

namespace HartreeFock::Correlation::CC
{
    enum class RCCSDTBackend
    {
        DeterminantPrototype,
        TensorProduction,
        TensorOptimized
    };

    struct TensorMemoryBlock
    {
        std::string label;
        std::size_t elements = 0;
        std::size_t bytes = 0;
    };

    // The production tensor path should not depend directly on the more
    // pedagogical RHF reference slices. This wrapper makes the canonical orbital
    // partition explicit and also snapshots the MO-basis Fock blocks that later
    // tensor kernels will reuse heavily.
    struct CanonicalRHFCCReference
    {
        RHFReference orbital_partition;
        Tensor2D f_oo;
        Tensor2D f_ov;
        Tensor2D f_vv;
    };

    // Unlike the teaching cache in mo_blocks.*, the production cache avoids the
    // full `(pq|rs)` tensor and stores only the blocks needed by tensor-based
    // CCSDT contractions. The memory report is kept with the cache so the
    // solver can print a clear pre-flight allocation summary.
    struct TensorCCBlockCache
    {
        Tensor4D oooo;
        Tensor4D ooov;
        Tensor4D oovv;
        Tensor4D ovov;
        Tensor4D ovvo;
        Tensor4D ovvv;
        Tensor4D vvvv;

        // U3.1: spin-blocked ERI blocks for the UCC path, keyed
        // (space pattern, spin tag) -- e.g. {"oovv", "abab"}. The seven named
        // members above stay the RCC storage and are untouched, so all 112
        // existing field accesses on the RHF path compile and behave unchanged;
        // a UCC run populates `spin_blocks` INSTEAD and leaves them empty.
        //
        // Not three copies of the seven above, and that is the point:
        //
        //  - Same-spin (`aaaa`/`bbbb`) needs SIX arrays, not seven -- `ovvo`
        //    folds into `ovov` under the particle swap, which is valid there.
        //  - Mixed (`abab`) needs TEN, because the particle swap and its product
        //    are NOT symmetries of a mixed block (they map it to `baba`), so the
        //    8-fold ERI orbit splits. Three of the ten (`oovo`, `vovo`, `vovv`)
        //    have no RCC counterpart at all.
        //
        // See `eri_permutations_for_block` in ccgen's emitter (U3.0) -- that
        // predicate and this block list are the same fact on the two sides of
        // the codegen boundary, so they must be changed together.
        std::vector<std::pair<std::pair<std::string, std::string>, Tensor4D>> spin_blocks;

        // The stored block for (space, tag), or an error naming what is missing.
        // Callers must NOT fall back to the untagged members: on a UCC run those
        // are empty, and on an RHF run reading them for a mixed tag would return
        // the wrong integral silently.
        [[nodiscard]] std::expected<const Tensor4D *, std::string> spin_block(
            const std::string &space,
            const std::string &tag) const;

        std::vector<TensorMemoryBlock> memory_report;
        std::size_t total_bytes = 0;
    };

    struct TensorTriplesWorkspace
    {
        RCCSDTAmplitudes amplitudes;
        Tensor6D r3;
        bool allocated = false;
        std::size_t storage_bytes = 0;
    };

    struct TensorRCCSDTState
    {
        CanonicalRHFCCReference reference;
        TensorCCBlockCache mo_blocks;
        DenominatorCache denominators;
        std::size_t estimated_t3_elements = 0;
        std::size_t estimated_t3_bytes = 0;
        TensorTriplesWorkspace triples;
        double warm_start_correlation_energy = 0.0;
        unsigned int warm_start_iterations = 0;
    };

    std::expected<CanonicalRHFCCReference, std::string> build_canonical_rhf_cc_reference(
        HartreeFock::Calculator &calculator);

    std::expected<TensorCCBlockCache, std::string> build_tensor_cc_block_cache(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const CanonicalRHFCCReference &reference,
        const std::string &tag);

    // U3.1 core: build the spin-blocked cache from an ALREADY-MATERIALIZED AO ERI
    // vector. Split out from build_ucc_spin_block_cache so the block/transform
    // logic -- which is where the defect risk is -- can be gated directly on a
    // synthetic ERI, without standing up a whole Calculator to reach ensure_eri.
    // The wrapper below is then only ERI acquisition.
    std::expected<TensorCCBlockCache, std::string> build_ucc_spin_block_cache_from_eri(
        const std::vector<double> &eri,
        std::size_t nb,
        const UHFReference &reference,
        const std::vector<std::pair<std::string, std::string>> &blocks);

    // U3.1: the spin-blocked (UCC) ERI cache. Populates `spin_blocks` and leaves
    // the seven untagged members empty, so an RHF consumer that reads them gets
    // an obviously-empty tensor rather than a plausible wrong one.
    //
    // The block list is derived, not hard-coded per spin: for each spin tag, one
    // stored array per orbit of the needed space patterns under the permutations
    // that are symmetries OF THAT TAG. Same-spin yields 6, mixed yields 10 -- see
    // TensorCCBlockCache::spin_blocks and ccgen's `eri_permutations_for_block`.
    //
    // The transform is the existing Correlation::transform_eri with per-slot
    // coefficient matrices; it already takes four independent ones. The spin sits
    // on the CHEMISTS' charge-density pair, not the physicist block: physicist
    // <oovv>_abab is chemists (i_a a_a | j_b b_b), a genuinely mixed transform
    // rather than any pure-spin RCC block relabeled.
    std::expected<TensorCCBlockCache, std::string> build_ucc_spin_block_cache(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UHFReference &reference,
        const std::vector<std::pair<std::string, std::string>> &blocks,
        const std::string &tag);

    [[nodiscard]] RCCSDTBackend choose_rccsdt_backend(
        const RHFReference &reference) noexcept;

    [[nodiscard]] std::string format_tensor_memory_summary(
        const TensorRCCSDTState &state);

    std::expected<void, std::string> allocate_dense_triples_workspace(
        TensorRCCSDTState &state);

    std::expected<TensorRCCSDTState, std::string> prepare_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_tensor_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_tensor_optimized_rccsdt(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_TENSOR_BACKEND_H
