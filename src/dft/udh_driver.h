#ifndef DFT_UDH_DRIVER_H
#define DFT_UDH_DRIVER_H
#include "udh_two_particle_density.h"
#include "udh_xc_ii.h"
#include "udh_zvector.h"

namespace DFT::Driver
{
    struct PreparedSystem;
    // Owned U10 ledger, in the prepared AO/standard molecular frame. No
    // borrowed response callbacks survive the call. Final requested-frame
    // rotation belongs to the ordinary driver, once, after KS+PT2 addition.
    struct UDHGradientResult
    {
        Gradient::UDHPT2EnergyContract energy;
        Gradient::UDHNonXCGradient non_xc;
        Gradient::UDHCompleteXCIIGradient xc;
        Gradient::UDHOrbitalMatrices rhs,z;
        Eigen::MatrixXd correction,ks,total;
        double z_residual=0,scaled_z_residual=0,physical_fock_error=0;
        bool assembled=false;
    };

    // Converged physical UKS and the exact UMP2 result used for energy.
    // All-active, conventional Cartesian global LDA/GGA, one scale, no
    // solvent/MPI. SCF iteration level shifts are legal only after removal;
    // the freshly rebuilt physical Fock is checked against PT2 orbitals.
    [[nodiscard]] std::expected<UDHGradientResult,std::string>
    build_udh_gradient_correction(HartreeFock::Calculator &calculator,
        PreparedSystem &prepared,const XC::Functional &exchange,
        const XC::Functional &correlation,
        const HartreeFock::Correlation::UMP2Result &pt2,double c_pt2);
}
#endif
