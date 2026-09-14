#ifndef DFT_UDH_KS_RESPONSE_H
#define DFT_UDH_KS_RESPONSE_H

#include <expected>
#include <functional>
#include <string>
#include <vector>

#include "analytic_hessian.h"
#include "base/types.h"

namespace DFT::Gradient
{
    // U2: physical spin AO matrices, NOT RKS total-density matrices. Trials
    // are symmetric but may be indefinite/non-number-conserving. A missing
    // occupied/virtual spin space is represented by a zero nao x nao trial,
    // never by an empty AO matrix. No occupations or PT2 factors enter here.
    struct UDHSpinMatrices
    {
        Eigen::MatrixXd alpha, beta;
    };

    using UDHMatrixResponseFn = std::function<std::expected<Eigen::MatrixXd, std::string>(
        const Eigen::Ref<const Eigen::MatrixXd> &)>;
    using UDHXCResponseFn = std::function<std::expected<UDHSpinMatrices, std::string>(
        const UDHSpinMatrices &)>;

    struct UDHKSResponseChannels
    {
        Eigen::MatrixXd coulomb;       // J[Qa+Qb], shared by both outputs
        UDHSpinMatrices exchange;      // -a_x K[Qa], -a_x K[Qb]; no cross K
        UDHSpinMatrices xc_from_alpha; // (f_aa[Qa], f_ba[Qa])
        UDHSpinMatrices xc_from_beta;  // (f_ab[Qb], f_bb[Qb])
        UDHSpinMatrices total;
    };

    struct UDHKSResponseOperator
    {
        Eigen::Index nbasis = 0;
        double exact_exchange = 0.0;
        UDHMatrixResponseFn coulomb, exchange; // raw J and K (no coefficients)
        UDHXCResponseFn xc;                  // physical polarized dVxc/dP

        [[nodiscard]] std::expected<UDHKSResponseChannels, std::string>
        apply_channels(const UDHSpinMatrices &trial) const;
        [[nodiscard]] std::expected<UDHSpinMatrices, std::string>
        apply(const UDHSpinMatrices &trial) const;
    };

    // Fixed-geometry linear callbacks. Owns the callback objects; expected
    // errors, exceptions, bad shapes and nonfinite/asymmetric outputs fail
    // closed. No silent zero substitution, symmetrization or factor 2/4.
    [[nodiscard]] std::expected<UDHKSResponseOperator, std::string>
    make_udh_ks_response_operator(Eigen::Index nbasis, double exact_exchange,
        UDHMatrixResponseFn coulomb, UDHMatrixResponseFn exchange, UDHXCResponseFn xc);

    // Direct J/K + existing analytic polarized LDA/GGA XC action. The ground
    // matrices are copied. All pointer targets (and the basis underlying the
    // shell pairs) must outlive the operator and remain unchanged. Rebuild on
    // geometry/grid/functional/density changes. No driver enablement in U2.
    struct UDHKSResponseInputs
    {
        const std::vector<HartreeFock::ShellPair> *shell_pairs = nullptr;
        const MolecularGrid *molecular_grid = nullptr;
        const AOGridEvaluation *ao_grid = nullptr;
        UDHSpinMatrices ground_density;
        const XC::Functional *exchange_functional = nullptr;
        const XC::Functional *correlation_functional = nullptr;
        double exact_exchange = 0.0;
        HartreeFock::IntegralMethod engine = HartreeFock::IntegralMethod::ObaraSaika;
        double tol_eri = 1e-10;
    };

    [[nodiscard]] std::expected<UDHKSResponseOperator, std::string>
    make_udh_direct_ks_response_operator(const UDHKSResponseInputs &inputs);
}
#endif
