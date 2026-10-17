#ifndef DFT_UDH_OVERLAP_DENSITY_H
#define DFT_UDH_OVERLAP_DENSITY_H

#include "udh_relaxed_density.h"

namespace DFT::Gradient
{
    // U6: each matrix is a full spin-MO raw W with only an oo response
    // block. Physical U2 channels already include a_x/XC factors.
    struct UDHOverlapResponse
    {
        UDHOrbitalMatrices coulomb,exchange,xc_from_alpha,xc_from_beta,total;
    };

    struct UDHOverlapDensity
    {
        UDHOrbitalMatrices pair_aa,pair_ab,pair_bb;
        UDHOrbitalMatrices dprime_orbital,z_orbital;
        UDHOverlapResponse response_dprime,response_z;
        UDHOrbitalMatrices raw_mo,symmetric_mo;
        UDHSpinMatrices symmetric_ao;
        Eigen::MatrixXd total_ao;
    };

    // G:U -> B:S in the paper metric gauge Uoo=Uvv=-S/2,
    // Uov=-Sov, Uvo=0. Returns B_oo=-G_oo/2, B_vv=-G_vv/2,
    // B_ov=-G_ov, B_vo=0 without assuming permutation symmetry in G.
    [[nodiscard]] std::expected<UDHOrbitalMatrices,std::string>
    build_udh_pair_overlap_density(const UDHOrbitalMatrices &g,int nocc_alpha,int nocc_beta);

    // Detached canonical U6 boundary: inputs must describe the SAME frozen
    // PT2/KS state. Borrows synchronously, owns all outputs. U5 validates
    // the spin frames and raw/symmetric D. F_mo is physical and diagonal;
    // this routine cannot detect a hidden diagonal level shift or a callback
    // from another reference. U2 must be the joint-spin self-adjoint physical
    // derivative of that Fock model. No solve, implicit rescaling or driver
    // enablement. Two U2 calls keep the Dprime and Z response ledgers distinct.
    [[nodiscard]] std::expected<UDHOverlapDensity,std::string>
    build_udh_overlap_density(const UDHPairOrbitalGradient &pair,
        const UDHRelaxedDifferenceDensity &relaxed,const UDHOrbitalMatrices &mo_coeff,
        const UDHOrbitalMatrices &fock_mo,const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,
        const UDHKSResponseOperator &response);
}
#endif
