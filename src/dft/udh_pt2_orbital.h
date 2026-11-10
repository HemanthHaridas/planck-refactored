#ifndef DFT_UDH_PT2_ORBITAL_H
#define DFT_UDH_PT2_ORBITAL_H

#include <array>
#include "udh_pt2_gradient.h"
#include "udh_ks_response.h"
#include "dh_pt2_gradient.h"

namespace DFT::Gradient
{
    // U3 MO integrals: the five Eq. 40 sectors per spin pair, electron 1 in
    // the first spin and electron 2 in the second (aa, ab = alpha|beta, bb),
    // in the RKS DHMolecularERIBlocks layout. These are exactly the reads of
    // the literal four-coefficient derivative; no full MO tensor is formed.
    // Production builds them with DFT::Driver::build_dh_eq40_mo_eri_blocks_direct.
    struct UDHMOIntegrals { DHMolecularERIBlocks aa,ab,bb; };

    // Separate MO spaces (square G/F/C columns or rectangular vo blocks as
    // specified by the containing result); not U2's physical AO matrices.
    struct UDHOrbitalMatrices { Eigen::MatrixXd alpha,beta; };

    struct UDHPairOrbitalSector
    {
        // Slot order of (i_sigma a_sigma|j_tau b_tau): i,a,j,b.
        // G_pq multiplies U_pq for delta C = C U; no implicit transpose.
        std::array<UDHOrbitalMatrices,4> slots;
        UDHOrbitalMatrices g;
        UDHOrbitalMatrices external_ai; // +G_ai, occupied coefficient slots
        UDHOrbitalMatrices internal_ai; // -G_ia, virtual coefficient slots
        UDHOrbitalMatrices total_ai;
    };
    struct UDHPairOrbitalGradient
    {
        UDHPairOrbitalSector aa,ab,bb;
        UDHOrbitalMatrices g,total_ai;
    };

    [[nodiscard]] std::expected<UDHPairOrbitalGradient,std::string>
    build_udh_pair_orbital_gradient(const UDHPT2Amplitudes &amplitudes,
        const UDHMOIntegrals &integrals,double c_pt2);

    struct UDHOrbitalRHS
    {
        UDHPairOrbitalGradient pair;
        UDHPT2DPrime dprime;
        UDHPT2AODPrime dprime_ao;
        UDHKSResponseChannels response_ao;
        // Each is 2 Cv^T (physical U2 channel) Co, with no second PT2 scale.
        UDHOrbitalMatrices response_j,response_k,xc_from_alpha,xc_from_beta,response_ai;
        // Off-shell coefficient connection 2[(F D')_ai-(F D')_ia].
        // Zero for canonical F and block-diagonal D'; kept explicit so the
        // common scalar is also differentiable with noncanonical/off-shell F.
        UDHOrbitalMatrices fock_connection_ai,total_ai;
    };

    // Derivative of U1's H=P+sum D':F under spin-preserving vo/ov rotations.
    // U2 must be the physical, joint-spin self-adjoint derivative of THIS
    // Fock model at this geometry/density. This cannot be inferred from the
    // callback type; it is independently tested, not enforced by a fitted
    // factor. C^T S C=I is checked, callback errors are propagated. No Z solve
    // or UKS production enablement. Inputs are borrowed only during the call.
    [[nodiscard]] std::expected<UDHOrbitalRHS,std::string>
    build_udh_orbital_rhs(const UDHPT2Amplitudes &amplitudes,
        const UDHMOIntegrals &integrals,double c_pt2,
        const UDHOrbitalMatrices &mo_coeff,const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,
        const UDHOrbitalMatrices &fock_mo,const UDHKSResponseOperator &response);
}
#endif
