#ifndef DFT_DRIVER_H
#define DFT_DRIVER_H

#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ao_grid.h"
#include "base/grid.h"
#include "base/types.h"
#include "base/wrapper.h"
#include "integrals/shellpair.h"
#include "ks_matrix.h"
#include "response_packing.h"
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

    // TDDFT excitation-space and finite-difference XC kernel machinery
    // (docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md, F3.1). These were originally
    // internal-linkage helpers inside driver.cpp's anonymous namespace; moved
    // here (definitions relocated in driver.cpp, unchanged otherwise) so F3's
    // own verification can call the existing FD-kernel oracle
    // (build_unrestricted_xc_kernel_blocks / build_closed_shell_xc_kernel_blocks)
    // directly from a standalone test binary, instead of only from inside the
    // TDDFT code path. ResponseExcitationSpace represents an arbitrary
    // occ-virt subset -- nothing here is TDDFT-specific, which is exactly why
    // F1's own scoping reused it as the SOSCF orbital-Hessian excitation
    // space rather than inventing a new type.
    struct ResponseExcitationSpace
    {
        std::string spin_label;
        int n_occ = 0;
        int n_virt = 0;
        int mo_offset = 0;
        Eigen::MatrixXd C_occ;
        Eigen::MatrixXd C_virt;
        Eigen::VectorXd occ_energies;
        Eigen::VectorXd virt_energies;
        std::vector<std::string> mo_symmetry;

        [[nodiscard]] int nov() const noexcept
        {
            return n_occ * n_virt;
        }

        [[nodiscard]] int flat_index(int i, int a) const noexcept
        {
            return i * n_virt + a;
        }
    };

    struct ResponseEigenpair
    {
        double omega = 0.0;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
    };

    Eigen::MatrixXd transition_density_matrix(
        const Eigen::Ref<const Eigen::VectorXd> &occupied,
        const Eigen::Ref<const Eigen::VectorXd> &virtual_orbital);

    std::expected<XCMatrixContribution, std::string> evaluate_xc_matrix_from_spin_densities(
        const PreparedSystem &prepared,
        const Eigen::Ref<const Eigen::MatrixXd> &alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &beta_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

    // The finite-difference XC kernel oracle (D1's chosen path,
    // docs/SOSCF_UHF_DFT_SCOPE.md). Perturbs the density along each
    // (occ,virt) direction in `spaces` and finite-differences the full
    // first-derivative XC potential -- O(n_occ*n_virt) grid passes, the
    // correctness-only reference every analytic Hessian-vector product in
    // F3 must be checked against.
    std::expected<std::vector<std::vector<Eigen::MatrixXd>>, std::string> build_unrestricted_xc_kernel_blocks(
        const PreparedSystem &prepared,
        const std::vector<ResponseExcitationSpace> &spaces,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_beta_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string> build_closed_shell_xc_kernel_blocks(
        const PreparedSystem &prepared,
        const ResponseExcitationSpace &space,
        const Eigen::Ref<const Eigen::MatrixXd> &restricted_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

    // slice_begin/slice_end restrict the density/XC evaluation to this rank's
    // grid-point slice (MPI). slice_end < 0 (default) = whole grid; the
    // gradient/TDDFT callers pass no slice and stay byte-identical. The returned
    // xc_grid's total_energy / integrated_electrons are then PARTIAL (slice
    // only) and must be scalar-reduced by the SCF caller before use.
    std::expected<XCGridEvaluation, std::string>
    evaluate_current_density_and_xc(
        const HartreeFock::Calculator &calculator,
        const PreparedSystem &prepared,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional,
        Eigen::Index slice_begin = 0,
        Eigen::Index slice_end = -1);

    // xc_point_begin/xc_point_end restrict XC assembly to this rank's slice and
    // reduce the nb^2 XC matrix internally (J/K reduce themselves). xc_point_end
    // < 0 (default) = whole grid, no reduce.
    std::expected<KSPotentialMatrices, std::string>
    assemble_current_ks_potential(
        HartreeFock::Calculator &calculator,
        PreparedSystem &prepared,
        const XCGridEvaluation &xc_grid,
        Eigen::Index xc_point_begin = 0,
        Eigen::Index xc_point_end = -1);

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
