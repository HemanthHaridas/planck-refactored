#ifndef DFT_DRIVER_H
#define DFT_DRIVER_H

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/types.h"
#include "base/grid.h"
#include "base/wrapper.h"
#include "integrals/shellpair.h"
#include "ks_matrix.h"
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
    };

    struct Result
    {
        double total_energy = 0.0;
        double xc_energy = 0.0;
        double integrated_electrons = 0.0;
        bool converged = false;
    };

    std::expected<XCGridEvaluation, std::string>
    evaluate_current_density_and_xc(
        const HartreeFock::Calculator& calculator,
        const PreparedSystem& prepared,
        const XC::Functional& exchange_functional,
        const XC::Functional& correlation_functional);

    std::expected<KSPotentialMatrices, std::string>
    assemble_current_ks_potential(
        HartreeFock::Calculator& calculator,
        const PreparedSystem& prepared,
        const XCGridEvaluation& xc_grid);

    std::expected<PreparedSystem, std::string>
    prepare(HartreeFock::Calculator& calculator, const Options& options = {});

    std::expected<Result, std::string>
    run(HartreeFock::Calculator& calculator, const Options& options = {});

} // namespace DFT::Driver

#endif // DFT_DRIVER_H
