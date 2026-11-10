#ifndef DFT_UDH_ZVECTOR_H
#define DFT_UDH_ZVECTOR_H

#include "udh_pt2_orbital.h"
#include "dh_zvector.h"

namespace DFT::Gradient
{
    // U4 canonical, physical (unshifted) Fock snapshot. Alpha/beta MO
    // counts and occupations may differ. Zero ov spaces retain their
    // rectangular shapes. C^T S C=I; each F_mo must be diagonal.
    struct UDHZVectorInputs
    {
        UDHOrbitalMatrices mo_coeff, fock_mo;
        Eigen::MatrixXd overlap_ao;
        int nocc_alpha=0,nocc_beta=0;
        UDHKSResponseOperator response;
    };

    struct UDHEq27Action
    {
        UDHSpinMatrices delta_density; // Cv X Co^T + transpose; NO RKS factor 2
        UDHOrbitalMatrices orbital, coulomb, exchange;
        UDHOrbitalMatrices xc_from_alpha, xc_from_beta, total;
    };

    [[nodiscard]] std::expected<UDHEq27Action,std::string>
    apply_udh_eq27_hessian(const UDHZVectorInputs &inputs,const UDHOrbitalMatrices &trial_ai);

    struct UDHZVectorProducts : DHZVectorStatistics
    {
        // Owned MO/F/S and RHS snapshot; callback captures retain U2's
        // documented borrowed geometry/functionals lifetimes. Never cache
        // this result across a changed geometry or response reference.
        UDHZVectorInputs snapshot;
        UDHOrbitalMatrices rhs_ai,z_ai;
    };

    // Pack alpha then beta, each a*nocc+i, and solve A z=-ell with the shared
    // RKS/UKS solver (solve_packed_zvector), using U3's total_ai unchanged (no
    // second c_PT2 or occupation factor). The joint Hessian is self-adjoint,
    // which the tests check against a dense oracle; production never builds it.
    [[nodiscard]] std::expected<UDHZVectorProducts,std::string>
    solve_udh_zvector(const UDHZVectorInputs &inputs,const UDHOrbitalMatrices &rhs_ai,
        const DHZVectorOptions &options={});
}
#endif
