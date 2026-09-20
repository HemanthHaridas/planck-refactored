#ifndef DFT_UDH_XC_II_H
#define DFT_UDH_XC_II_H

#include "udh_ks_response.h"

namespace DFT::Gradient
{
    struct UDHXCIIInputs
    {
        const HartreeFock::Molecule *molecule=nullptr;
        const HartreeFock::Basis *basis=nullptr;
        const MolecularGrid *molecular_grid=nullptr;
        const AOGridEvaluation *ao_grid=nullptr;
        UDHSpinMatrices ground_density,relaxed_density; // physical spin AO matrices
        const XC::Functional *exchange_functional=nullptr;
        const XC::Functional *correlation_functional=nullptr;
    };

    struct UDHCompleteXCIIGradient
    {
        Eigen::MatrixXd p_side_fixed,d_side_ao,becke_partition,point_translation,total;
    };

    // U8a: complete fixed-AO-coefficient geometry derivative of
    // Q=sum_g w_g sum_sigma v_sigma(P) rho_Dsigma. All outputs are natoms x 3
    // in the supplied AO/molecular frame. P and signed, already scaled D
    // remain fixed; neither c_PT2 nor occupancy factors are inserted here.
    // LDA only, polarized functionals; no AO Hessians are required.
    // f_rr packs [aa,ab,bb], including both cross-spin chain-rule terms.
    // Uses the shared AO-center and owner-Becke derivative primitives.
    // No extra density cutoff/clamp. Libxc's configured thresholds apply.
    // A combined XC exchange slot suppresses the correlation slot entirely.
    // Inputs are borrowed synchronously; caller guarantees one unchanged
    // geometry/grid/AO/functional snapshot. Outputs own all matrices.
    // No XC-I, SCF contribution, Z solve, frame rotation or driver enablement.
    [[nodiscard]] std::expected<UDHCompleteXCIIGradient,std::string>
    build_udh_lda_complete_xc_ii_gradient(const UDHXCIIInputs &inputs);

    struct UDHGGAXCIIInputs
    {
        UDHXCIIInputs fixed;
        const AOGridHessian *ao_hessian=nullptr;
    };

    // U8b: Q=f_y.s_D, y=(rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb),
    // s_D=(rho_Da,rho_Db,2 u_a.v_a,u_a.v_b+u_b.v_a,2 u_b.v_b).
    // Both f_yy y^(x) and f_y s_D^(x) are retained. The ab invariant
    // appears once; differentiating its s_D entry produces four terms.
    // GGA functionals only (combined hybrid GGA is a semilocal GGA slot).
    // Requires all six spatial AO Hessian components from the same grid.
    // Same fixed-coefficient, scaling, lifetime and output contract as U8a.
    [[nodiscard]] std::expected<UDHCompleteXCIIGradient,std::string>
    build_udh_gga_complete_xc_ii_gradient(const UDHGGAXCIIInputs &inputs);
}
#endif
