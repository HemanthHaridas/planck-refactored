#ifndef HF_DRIVER_H
#define HF_DRIVER_H

#include <expected>
#include <string>

#include "base/types.h"

// Hartree-Fock / post-HF top-level driver, the symmetric peer of
// DFT::Driver::run (src/dft/driver.h). It owns the full single-calculation
// workflow — coordinate prep, BSSE, symmetry/basis setup, SCF, post-HF,
// gradients, geomopt, frequencies, properties, and checkpoint/JSON output —
// operating on an already-parsed Calculator.
//
// Both the serial `hartree-fock` binary (src/driver.cpp) and the future
// unified `planck-mpi` binary dispatch to this after parsing:
//   calculator.is_dft_run() ? DFT::Driver::run(...) : HartreeFock::Driver::run(...)
//
// Returns the process exit code (EXIT_SUCCESS / EXIT_FAILURE) on success, or
// an error string if the workflow could not be started. Diagnostic logging for
// in-workflow failures is emitted at the failing call site (as before), and
// those paths return EXIT_FAILURE in the value channel — the string error is
// reserved for setup-level failures.
namespace HartreeFock::Driver
{
    std::expected<int, std::string> run(
        HartreeFock::Calculator &calculator,
        const std::string &input_file,
        const std::string &json_path);
} // namespace HartreeFock::Driver

#endif // HF_DRIVER_H
