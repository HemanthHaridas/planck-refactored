#ifndef DFT_UDH_TWO_PARTICLE_DENSITY_H
#define DFT_UDH_TWO_PARTICLE_DENSITY_H

#include "udh_overlap_density.h"
#include "dh_pt2_gradient.h" // shared AO derivative contraction

namespace DFT::Gradient
{
    struct UDHTwoParticleDensity
    {
        int n_ao=0;
        UDHRelaxedAODensity relaxed_ao;
        UDHSpinMatrices ground_ao; // occupancy-one Co_sigma Co_sigma^T
        double c_pt2=0,exact_exchange=0;
    };

    // U7 Eq. 34/35 correction only. Checks U5 metric/layout and Dprime(T,c)
    // consistency; constructs P from the same occupied frames. D/Z already
    // carry c. No SCF P P, nuclear or XC contribution. Rectangular/empty spin
    // frames are legal. Caller owns common snapshot/a_x provenance.
    [[nodiscard]] std::expected<UDHTwoParticleDensity,std::string>
    build_udh_two_particle_density(const UDHPT2Amplitudes &amplitudes,double c_pt2,
        const UDHRelaxedDifferenceDensity &relaxed,const UDHOrbitalMatrices &mo_coeff,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,double exact_exchange);

    // Eq. 34: D P - a_x sum_sigma D_sigma(mu,kappa) P_sigma(nu,tau).
    // Eq. 35: four-coefficient (ia|jb) backtransform, weights c,2c,c, unscaled
    // by a_x. Exchange is ordered alpha, beta and pair aa, ab, bb. The source
    // borrows amplitudes and mo_coeff for the synchronous contraction only.
    [[nodiscard]] DHEq33TwoParticleSource make_udh_two_particle_source(
        const UDHTwoParticleDensity &two_particle,const UDHPT2Amplitudes &amplitudes,
        const UDHOrbitalMatrices &mo_coeff);
}
#endif
