#ifndef DFT_UDH_RELAXED_DENSITY_H
#define DFT_UDH_RELAXED_DENSITY_H

#include "udh_pt2_orbital.h"

namespace DFT::Gradient
{
    // U5, Eq. 28 in the U4 multiplier convention L=H+sum_sigma z:F_vo.
    // Raw D has D' in oo/vv, z in vo and zero ov. Only the symmetric
    // adapter has z/2 and z^T/2. No reference density or second PT2 scale.
    struct UDHRelaxedDifferenceDensity
    {
        int nocc_alpha=0,nocc_beta=0;
        UDHOrbitalMatrices raw_mo,symmetric_mo;
    };

    // Detached algebra: does not certify that z solves the response system.
    // Occupations/virtual counts come from each z's (v,o) shape. Zero-size
    // spaces retain those shapes; finite, symmetric block-diagonal D' is
    // required. All returned matrices are owned, with no borrowed callbacks.
    [[nodiscard]] std::expected<UDHRelaxedDifferenceDensity,std::string>
    build_udh_relaxed_difference_density(const UDHPT2DPrime &dprime,
        const UDHOrbitalMatrices &z_ai);

    struct UDHRelaxedAODensity
    {
        UDHSpinMatrices spin; // C_sigma sym(D_sigma) C_sigma^T
        Eigen::MatrixXd total; // alpha+beta, for spin-independent operators
    };

    // Checked metric adapter. Different rectangular orthonormal spin frames
    // are legal for detached algebra (as in U4); empty MO frames produce a
    // zero nao-by-nao AO density. This does not widen U1/U3's all-active
    // handoff or enable any truncated-space/UKS gradient driver workflow.
    // Raw spin MO matrices must never be summed before AO transformation.
    [[nodiscard]] std::expected<UDHRelaxedAODensity,std::string>
    transform_udh_relaxed_density_to_ao(const UDHRelaxedDifferenceDensity &density,
        const UDHOrbitalMatrices &mo_coeff,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao);
}
#endif
