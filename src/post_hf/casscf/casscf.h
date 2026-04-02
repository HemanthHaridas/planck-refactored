#ifndef HF_POSTHF_CASSCF_DRIVER_H
#define HF_POSTHF_CASSCF_DRIVER_H

#include <expected>
#include <string>
#include <vector>

#include "integrals/shellpair.h"
#include "post_hf/casscf_internal.h"

namespace HartreeFock
{
    class Calculator;
}

namespace HartreeFock::Correlation::CASSCF
{
    // Shared macroiteration driver used by both CASSCF and RASSCF front-ends.
    // `tag` controls log labeling, while `ras` selects the determinant filter.
    std::expected<void, std::string> run_mcscf_loop(
        HartreeFock::Calculator&                   calc,
        const std::vector<HartreeFock::ShellPair>& shell_pairs,
        const std::string&                         tag,
        const CASSCFInternal::RASParams&           ras);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_DRIVER_H
