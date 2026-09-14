#ifndef DFT_UDH_PT2_GRADIENT_H
#define DFT_UDH_PT2_GRADIENT_H

#include "post_hf/mp2.h"

namespace DFT::Gradient
{
    // U0 only: a detached, correction-only energy/storage boundary. This
    // does not authorize UKS gradients or establish KS orbital stationarity.
    struct UDHPT2Scope
    {
        HartreeFock::SCFType reference = HartreeFock::SCFType::UHF;
        HartreeFock::BasisType basis = HartreeFock::BasisType::Cartesian;
        bool range_separated = false;
        bool solvent = false;
        bool meta_gga = false;
        bool independent_spin_scaling = false;
    };

    // Direct (NOT antisymmetrized) (i_sigma a_sigma|j_tau b_tau) integrals,
    // repacked into amplitude order [i_sigma,j_tau,a_sigma,b_tau]. This is
    // deliberately NOT mp2_internal.h's [i,a,j,b] ovov storage.
    struct UDHPT2DirectIntegrals
    {
        std::vector<double> aa, ab, bb;
    };

    struct UDHPT2EnergyContract
    {
        int nocc_alpha = 0, nocc_beta = 0;
        int nvirt_alpha = 0, nvirt_beta = 0;
        double c_pt2 = 0.0;
        double aa = 0.0, ab = 0.0, bb = 0.0; // unscaled correlation only
        double same_spin = 0.0, opposite_spin = 0.0, unscaled_total = 0.0;
        double correction = 0.0; // c_pt2 * unscaled_total, exactly once
    };

    // Borrows inputs only during the call; returns an owned scalar ledger.
    // Validates a canonical, all-active, real UMP2 snapshot against supplied
    // direct MO integrals and AO overlap. Expected zero-extent spin arrays
    // are legal; missing nonzero-extent amplitudes are errors, even at c=0.
    // Response, Z, W and driver enablement belong to later steps.
    [[nodiscard]] std::expected<UDHPT2EnergyContract, std::string>
    build_udh_pt2_energy_contract(
        const HartreeFock::Correlation::UMP2Result &result,
        const UDHPT2DirectIntegrals &integrals,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,
        const HartreeFock::OptionsMP2 &options,
        double c_pt2,
        const UDHPT2Scope &scope = {});

    // U1 off-shell boundary: unscaled amplitudes in [i,j,a,b] order.
    // aa/bb are antisymmetric in both occupied and virtual indices; ab is
    // direct. No energies, denominators or cached SCF state are stored here.
    struct UDHPT2Amplitudes
    {
        int nocc_alpha = 0, nocc_beta = 0;
        int nvirt_alpha = 0, nvirt_beta = 0;
        std::vector<double> aa, ab, bb;
    };

    struct UDHPT2DPrime
    {
        Eigen::MatrixXd alpha_mo, beta_mo; // scaled once; oo/vv only
    };

    [[nodiscard]] std::expected<void, std::string>
    validate_udh_pt2_amplitudes(const UDHPT2Amplitudes &amplitudes);

    [[nodiscard]] std::expected<UDHPT2DPrime, std::string>
    build_udh_pt2_dprime(const UDHPT2Amplitudes &amplitudes, double c_pt2);

    struct UDHPT2AODPrime
    {
        Eigen::MatrixXd alpha_ao, beta_ao;
    };

    // Checked all-active C D' C^T adapter. Each spin uses its own C, with
    // C^T S C=I; Tr(D'_AO S)=Tr(D'_MO). Does not add the reference density.
    [[nodiscard]] std::expected<UDHPT2AODPrime, std::string>
    transform_udh_pt2_dprime_to_ao(const UDHPT2DPrime &density,
        const Eigen::Ref<const Eigen::MatrixXd> &c_alpha,
        const Eigen::Ref<const Eigen::MatrixXd> &c_beta,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao);

    struct UDHPT2StationaryScalar
    {
        UDHPT2DPrime dprime;
        double pair_aa = 0.0, pair_ab = 0.0, pair_bb = 0.0;
        double pair = 0.0;
        double dprime_f_alpha = 0.0, dprime_f_beta = 0.0;
        double dprime_f = 0.0, total = 0.0;
    };

    // H(T;g,F)=c*Taa:gaa+2c*Tab:gab+c*Tbb:gbb+sum_spin D'(T):F.
    // Accepts noncanonical, full symmetric MO Fock matrices and off-shell T.
    // Uses every oo/vv Fock entry, with no diagonalization or residual gate.
    // Amplitude stationarity is an assertion ONLY at a stationary solution.
    [[nodiscard]] std::expected<UDHPT2StationaryScalar, std::string>
    evaluate_udh_pt2_stationary_scalar(const UDHPT2Amplitudes &amplitudes,
        const UDHPT2DirectIntegrals &integrals,
        const Eigen::Ref<const Eigen::MatrixXd> &fock_alpha_mo,
        const Eigen::Ref<const Eigen::MatrixXd> &fock_beta_mo, double c_pt2);

    struct UDHPT2StationaryContract
    {
        UDHPT2EnergyContract energy;
        UDHPT2StationaryScalar stationary;
    };

    // Canonical U0-validated snapshot: certifies pair=2cE, D':eps=-cE,
    // H=cE. The detached evaluator above remains available off shell.
    [[nodiscard]] std::expected<UDHPT2StationaryContract, std::string>
    build_udh_pt2_stationary_contract(
        const HartreeFock::Correlation::UMP2Result &result,
        const UDHPT2DirectIntegrals &integrals,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,
        const HartreeFock::OptionsMP2 &options, double c_pt2,
        const UDHPT2Scope &scope = {});
}

#endif
