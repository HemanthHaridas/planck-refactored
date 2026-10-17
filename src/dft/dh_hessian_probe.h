#pragma once

#include "dh_pt2_gradient.h"
#include "ks_orbital_hessian.h"

namespace DFT::Driver
{
    // Opt-in small-system diagnostic only. All non-Z inputs are shared with
    // the normal dense-gradient calculation. Writes a full precision ledger
    // and returns the correction rebuilt with the shared KS action's Z.
    // Operator/gradient disagreements are recorded, not hidden by dense fallback.
    // Solver/callback failures return an error after writing available evidence.
    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string> run_dh_hessian_swap_probe(
        const Gradient::DHGradientDriverInputs &inputs,
        const Gradient::DHGradientDriverContract &dense_contract,
        const KsOrbitalHessianInputs &shared_inputs,
        const Eigen::MatrixXd &ks_gradient,
        const std::string &log_path);
}
