#ifndef HF_POSTHF_CC_RCCGEN_H
#define HF_POSTHF_CC_RCCGEN_H

#include <expected>
#include <string>

#include "integrals/shellpair.h"

namespace HartreeFock::Correlation::CC
{
    // U5.3b: the generated UNRESTRICTED CC entry (correlation ucc2/ucc3/ucc4).
    // Rank comes from OptionsSCF::_cc_generated_rank, as on the RCC path.
    // Declared beside run_rccgen because the aggregate post_hf/cc.h -- which the
    // driver includes -- already exposes this header.
    std::expected<void, std::string> run_uccgen(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<void, std::string> run_rccgen(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    // U5.3c: the method name for a generated RCC run of this rank, shared by the
    // solver's log tags and the driver's energy label so the two cannot disagree.
    //
    // RANK 4 MUST STAY "RCCSDTQ". Three consumers parse that exact string --
    // `tests/run_regressions.py:33`, the `be_rccsdtq_sto3g` case's `contains`
    // assertion, and `tests/ccsdtq_fci_acceptance.py` (the CCSDTQ==FCI gate) --
    // and all three exercise rank 4 only, so keeping its label leaves them
    // untouched and changes only the ranks that were previously mislabelled.
    [[nodiscard]] std::string rcc_method_label(int rank);
} // namespace HartreeFock::Correlation::CC

#endif // HF_POSTHF_CC_RCCGEN_H
