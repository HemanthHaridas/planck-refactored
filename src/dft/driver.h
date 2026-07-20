#ifndef DFT_DRIVER_H
#define DFT_DRIVER_H

#include <expected>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/grid.h"
#include "base/types.h"
#include "base/wrapper.h"
#include "integrals/shellpair.h"
#include "ks_matrix.h"
#include "solvation/pcm.h"
#include "xc_grid.h"

namespace DFT::Driver
{

    struct Options
    {
        bool use_symmetry = true;
        bool use_sao_blocking = true;
        bool save_checkpoint = false;
        bool print_grid_summary = true;
    };

    struct PreparedSystem
    {
        std::vector<HartreeFock::ShellPair> shell_pairs;
        MolecularGrid molecular_grid;
        AOGridEvaluation ao_grid;
        GridPreset grid_preset;
        std::optional<HartreeFock::Solvation::PCMState> pcm;
    };

    struct Result
    {
        double total_energy = 0.0;
        double xc_energy = 0.0;
        double integrated_electrons = 0.0;
        double solvation_energy = 0.0;
        bool converged = false;
    };

    std::expected<XCGridEvaluation, std::string>
    evaluate_current_density_and_xc(
        const HartreeFock::Calculator &calculator,
        const PreparedSystem &prepared,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

    std::expected<KSPotentialMatrices, std::string>
    assemble_current_ks_potential(
        HartreeFock::Calculator &calculator,
        PreparedSystem &prepared,
        const XCGridEvaluation &xc_grid);

    std::expected<PreparedSystem, std::string>
    prepare(HartreeFock::Calculator &calculator, const Options &options = {});

    // Compute core: runs the KS-DFT workflow and returns the Result. Consumed by
    // the CLI entry below and directly by any caller that wants the structured
    // result rather than the process exit code.
    std::expected<Result, std::string>
    run(HartreeFock::Calculator &calculator, const Options &options = {});

    // CLI entry, the exact peer of HartreeFock::Driver::run: the DFT banner, the
    // compute core above, the energy / convergence / multipole report, timing,
    // and optional JSON dump. Returns the process exit code. Both planck-dft and
    // the unified planck-mpi reduce to a thin parse-then-dispatch shell that
    // calls this. `calculator` must already be parsed with its checkpoint path
    // set.
    std::expected<int, std::string> run(
        HartreeFock::Calculator &calculator,
        const std::string &input_file,
        const std::string &json_path);

} // namespace DFT::Driver

#endif // DFT_DRIVER_H
