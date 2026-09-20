#ifndef DFT_UDH_TWO_PARTICLE_DENSITY_H
#define DFT_UDH_TWO_PARTICLE_DENSITY_H

#include "udh_overlap_density.h"
#include "dh_pt2_gradient.h" // shared spin-independent AO derivative bundle

namespace DFT::Gradient
{
    // Chemists' [mu,nu,kappa,tau], n_ao^4 entries. The raw coefficient
    // need not have ERI symmetry; the adapter is an eightfold AVERAGE.
    struct UDHERICoefficients
    {
        std::vector<double> raw_ao,symmetric_ao;
    };

    struct UDHTwoParticleDensity
    {
        int n_ao=0;
        UDHRelaxedAODensity relaxed_ao;
        UDHSpinMatrices ground_ao; // occupancy-one Co_sigma Co_sigma^T
        UDHERICoefficients coulomb,exchange_alpha,exchange_beta;
        UDHERICoefficients pair_aa,pair_ab,pair_bb;
    };

    // U7 detached Eq. 34/35 correction only. Checks U5 metric/layout and
    // Dprime(T,c) consistency; constructs P from the same occupied frames.
    // Eq. 34: D P - a_x sum_sigma D_sigma(mu,kappa) P_sigma(nu,tau).
    // Eq. 35: four-coefficient (ia|jb) backtransform, weights c,2c,c.
    // D/Z already carry c; only the unscaled pair amplitudes are scaled here.
    // No a_x in the pair terms; no SCF P P, nuclear or XC contribution.
    // Four successive index transforms use O(N^5) arithmetic, O(N^4)
    // storage. This reference/acceptance boundary retains all raw/adapted
    // channels and is not a streaming or density-fitted implementation.
    // Rectangular/empty spin frames are legal detached; production remains
    // all-active and disabled. Caller owns common snapshot/a_x provenance.
    [[nodiscard]] std::expected<UDHTwoParticleDensity,std::string>
    build_udh_two_particle_density(const UDHPT2Amplitudes &amplitudes,double c_pt2,
        const UDHRelaxedDifferenceDensity &relaxed,const UDHOrbitalMatrices &mo_coeff,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,double exact_exchange);

    struct UDHNonXCGradient
    {
        Eigen::VectorXd one_electron,overlap;
        Eigen::VectorXd coulomb,exchange_alpha,exchange_beta,pair_aa,pair_ab,pair_bb;
        Eigen::VectorXd two_electron_separable,two_electron_nonseparable,two_electron,total;
    };

    // Consumes only U6's symmetric AO W and the U7 AO contract; never adds
    // unlike spin MO frames. All derivative arrays use one coordinate order.
    // Checks finite shapes, h/S symmetry, eightfold ERI symmetry and raw /
    // adapted coefficient consistency. Empty coordinate lists are allowed.
    // +D:h' + W:S' + Gamma:I', no additional signs or prefactors. No XC,
    // nuclear repulsion, KS reference gradient, coordinate rotation or solve.
    [[nodiscard]] std::expected<UDHNonXCGradient,std::string>
    build_udh_non_xc_gradient(const UDHTwoParticleDensity &two_particle,
        const UDHOverlapDensity &overlap,const DHEq33NonXCDerivatives &derivatives);
}
#endif
