#ifndef HF_POSTHF_CC_TENSOR_BACKEND_H
#define HF_POSTHF_CC_TENSOR_BACKEND_H

#include <array>
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

        // U3.3: spin-resolved Fock blocks for the UCC path, keyed (space, tag)
        // where tag is "aa" or "bb". The three untagged members above stay the
        // RCC storage and are untouched; a UCC build populates `spin_blocks`
        // instead and leaves them empty.
        //
        // Simpler than the ERI case and deliberately so: the Fock is two-index,
        // so both slots carry the SAME spin and there is no mixed block, no
        // permutation-validity question, and no orbit to enumerate. `f_ov` and
        // `f_vo` still collapse onto one stored block because the Fock is
        // symmetric -- that reorder is spin-safe here precisely because a
        // two-index tag cannot mix spins the way <ab|ab> does.
        //
        // This is where the spin resolution WITHDRAWN from U2's reference-variant
        // belongs: the kernels only ever touch f_oo/f_ov/f_vv and
        // orbital_partition, so resolving those three is the whole job, and it
        // costs no kernel-signature change.
        std::vector<std::pair<std::pair<std::string, std::string>, Tensor2D>> spin_blocks;

        // The stored Fock block for (space, tag). No fallback to the untagged
        // members, for the same reason TensorCCBlockCache::spin_block has none:
        // on a UCC build they are empty, and on an RHF build they are the
        // spin-free matrix, which is a different quantity.
        [[nodiscard]] std::expected<const Tensor2D *, std::string> spin_block(
            const std::string &space,
            const std::string &tag) const;
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

    // U3.3: the spin-resolved Fock blocks for the UCC path. Takes the MO-basis
    // alpha and beta Fock matrices (Ct F C, each in its own spin's MO order,
    // occupied columns first) and slices the oo/ov/vv blocks per spin.
    //
    // Deliberately takes the MO matrices rather than a Calculator: the slicing is
    // the part that can be wrong, and it is gateable without an SCF.
    std::expected<CanonicalRHFCCReference, std::string> build_ucc_fock_blocks(
        const UHFReference &reference,
        const Eigen::MatrixXd &fock_alpha_mo,
        const Eigen::MatrixXd &fock_beta_mo);

    // U5.2b: rebind a UCC (spin-blocked) chemists cache to the physicist <pq|rs|
    // the generated kernels index.
    //
    // EVERY BLOCK IS SELF-SOURCED: `swap_mid_axes` applied to the block stored
    // under its OWN (space, tag) key. No source map, no bra<->ket hop, and no
    // permutation of the spin tag. Verified numerically on all 24 blocks with a
    // non-degenerate reference (noa != nob != nva != nvb).
    //
    // That is simpler than it first appears, and the reason is worth stating:
    // U3.1 keys blocks by the PHYSICIST (space, spin) and applies the chemists
    // (p r | q s) pairing internally when building them. Treating a stored key as
    // if it named a CHEMISTS pattern produces a convincing false picture -- three
    // mixed blocks appear to need a source that is not stored, and the spin tag
    // appears to need permuting (`abab` -> `aabb`). Both are artifacts of that
    // misreading; neither survives a direct check against the physicist target.
    //
    // Unlike the RCC `rebind_physicist`, this copies `spin_blocks` -- that one
    // builds a fresh cache from the seven NAMED members and would silently return
    // a cache with all 24 UCC blocks discarded.
    [[nodiscard]] TensorCCBlockCache rebind_physicist_ucc(TensorCCBlockCache chem);

    // U5.1a: the ERI spin blocks a UCC run stores, as (space pattern, spin tag).
    //
    // NO METHOD IS INVOLVED, and that is the design. RCC's
    // build_tensor_cc_block_cache takes no block list either: its set IS the
    // struct's seven named members, built unconditionally, and it over-builds
    // (measured -- ccsd and ccsdt both read 6 of the 7, `ovvo` never touched).
    // Nothing is negotiated with the emitter, so nothing can drift.
    //
    // This is that property one level up. For each spin tag, one stored array per
    // ORBIT of the sixteen o/v patterns under THAT TAG's own symmetry group: the
    // four physicist symmetries of <pq|rs> are usable on a block only when they
    // map its spin string to itself, so a same-spin tag keeps all four and folds
    // 16 patterns into 7, while `abab` keeps only identity and bra<->ket (the
    // other two send it to `baba`) and folds into 10.
    //
    //     aaaa   7      abab  10      bbbb   7      = 24 stored arrays
    //
    // Fixed by the REFERENCE TYPE, not the method. It is the same rule ccgen's
    // `eri_permutations_for_block` / `_canonical_eri_blocks_for` encode, so the
    // two sets cannot disagree by construction -- gated per tag against the Python
    // side rather than trusted.
    //
    // If 24 blocks is ever too heavy, trim the stored set by REFERENCE SYMMETRY,
    // never by method: trimming per method is what reintroduces the coupling this
    // avoids.
    [[nodiscard]] std::vector<std::pair<std::string, std::string>>
    ucc_canonical_blocks();

    // Whether `perm` (a permutation of the four physicist slots) is a symmetry of
    // the spin block `tag` -- true iff it maps the tag to itself. The C++ mirror
    // of ccgen's `eri_permutation_preserves_block` (U3.0).
    [[nodiscard]] bool eri_permutation_preserves_block(
        const std::string &tag,
        const std::array<int, 4> &perm) noexcept;

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
