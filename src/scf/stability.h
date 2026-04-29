#ifndef HF_SCF_STABILITY_H
#define HF_SCF_STABILITY_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::SCF
{
    // Result of one channel of the stability analysis.
    //
    // A negative `lowest_eigenvalue` (below `-tol`) means an instability has
    // been detected: an orbital rotation along `lowest_mode` lowers the energy.
    // `lowest_mode` is shaped [n_virt × n_occ] and is the eigenvector reshaped
    // back into the natural virtual-by-occupied layout.
    struct StabilityChannel
    {
        std::string label;                // human-readable channel name
        bool is_unstable = false;         // lowest_eigenvalue < -tol
        double lowest_eigenvalue = 0.0;   // signed lowest eigenvalue
        Eigen::MatrixXd lowest_mode;      // reshaped [n_virt x n_occ]
        // For UHF spin-flip: separate alpha→beta and beta→alpha blocks of the
        // mode are stored here; otherwise empty.
        Eigen::MatrixXd lowest_mode_beta;
    };

    struct StabilityReport
    {
        // RHF channels (populated when reference is RHF):
        //   internal_real    — RHF → another real RHF (lowest eig of A^S - B^S)
        //   internal_complex — RHF → complex orbital RHF (lowest eig of A^S + B^S)
        //   external_triplet — RHF → UHF (lowest eig of A^T - B^T)
        //
        // UHF channels (populated when reference is UHF):
        //   internal_uhf     — UHF → another UHF (full spin-conserving Hessian)
        //   external_ghf     — UHF → GHF (spin-flip / non-collinear)
        std::vector<StabilityChannel> channels;
        bool any_unstable = false;
    };

    // Analyze the stability of a converged RHF reference. Builds the closed-shell
    // singlet and triplet orbital Hessian blocks in the canonical-MO basis,
    // diagonalizes each, and returns the lowest eigenvalues + eigenvectors.
    //
    // Tolerance `tol` (default 1e-5 Hartree): an eigenvalue is reported as an
    // instability when it falls below `-tol`. Values in `(-tol, 0)` are treated
    // as numerical zeros and reported as stable.
    //
    // Cost: O((n_virt * n_occ)^2) memory for the dense Hessian per channel.
    std::expected<StabilityReport, std::string> analyze_rhf_stability(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        double tol = 1e-5);

    // Analyze the stability of a converged UHF reference. Builds the
    // spin-resolved orbital Hessian in the αα/ββ/αβ/βα blocks and reports the
    // lowest eigenvalues for the spin-conserving (internal) and spin-flip
    // (external → GHF) sectors.
    std::expected<StabilityReport, std::string> analyze_uhf_stability(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        double tol = 1e-5);

    // Apply the unstable rotation in `mode` to the current MO coefficients,
    // rebuild the density, clear DIIS, and re-run SCF. Returns the new
    // converged total energy. The SCF type may change: when following an
    // RHF → UHF (triplet) instability, the calculator is switched to UHF
    // before re-running.
    //
    // `mode` is the eigenvector returned from analyze_*_stability, reshaped
    // [n_virt x n_occ]. The rotation amplitude is taken as a fixed step
    // (`step_scale * mode`) — small enough to leave the unstable basin but
    // large enough to break orbital symmetry. Default 0.05 rad works for
    // every regression case we've seen.
    std::expected<double, std::string> follow_instability_and_rerun(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const StabilityChannel &channel,
        double step_scale = 0.05);

} // namespace HartreeFock::SCF

#endif // HF_SCF_STABILITY_H
