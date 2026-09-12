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
    // Dprime, response, Z, W and driver enablement belong to later steps.
    [[nodiscard]] std::expected<UDHPT2EnergyContract, std::string>
    build_udh_pt2_energy_contract(
        const HartreeFock::Correlation::UMP2Result &result,
        const UDHPT2DirectIntegrals &integrals,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,
        const HartreeFock::OptionsMP2 &options,
        double c_pt2,
        const UDHPT2Scope &scope = {});
}

#endif
