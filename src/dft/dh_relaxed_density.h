#ifndef DFT_DH_RELAXED_DENSITY_H
#define DFT_DH_RELAXED_DENSITY_H

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/types.h"
#include "base/grid.h"
#include "base/wrapper.h"
#include "ao_grid.h"
#include "post_hf/mp2.h"
#include "post_hf/mp2_gradient.h"

// Double-hybrid PT2 orbital relaxation: solve the Z-vector against the KS
// orbital Hessian (not the HF CPHF matrix) and return the relaxed
// one-particle density. Split out of dh_gradient.h because it pulls in the
// DFT XC stack (base/wrapper.h -> libxc). See
// docs/DOUBLE_HYBRID_GRADIENT_SCOPE.md, N2.
namespace DFT::Gradient
{

    // `lagrangian` must already be c_PT2-scaled (from
    // build_pt2_mo_intermediates). RKS only. Returns
    // P_ao = C (2*I + gamma^1_diag + z_ov) C^T.
    struct PT2RelaxedDensityInputs
    {
        const HartreeFock::Correlation::RMP2Lagrangian *lagrangian = nullptr;
        const HartreeFock::Correlation::RMP2Result *result = nullptr;

        HartreeFock::Calculator *calculator = nullptr; // for build_veff_from_density
        const std::vector<HartreeFock::ShellPair> *shell_pairs = nullptr;
        const MolecularGrid *molecular_grid = nullptr;
        const AOGridEvaluation *ao_grid = nullptr;
        Eigen::MatrixXd density; // converged KS total density (AO)

        const XC::Functional *x_functional = nullptr;
        const XC::Functional *c_functional = nullptr;

        HartreeFock::IntegralMethod engine = HartreeFock::IntegralMethod::ObaraSaika;
        double tol_eri = 1e-10;
        const std::vector<HartreeFock::SignedAOSymOp> *sym_ops = nullptr;

        double full_range_exchange_coefficient = 0.0;
        double short_range_exchange_coefficient = 0.0;
        double range_separation_omega = 0.0;

        // KS mean-field response for vhf_s1occ (N3.5.5); same closure the
        // driver builds for build_pt2_mo_intermediates. Null => HF.
        HartreeFock::Correlation::KsVeffFn ks_veff = {};
    };

    struct PT2RelaxedDensity
    {
        Eigen::MatrixXd z;                    // Z-vector, n_virt x n_occ (KS-Hessian solve)
        Eigen::MatrixXd P_ao;                 // C (2*I + gamma^1_diag + z_ov) C^T
        Eigen::MatrixXd W_ao;                 // energy-weighted relaxed density
        Eigen::MatrixXd dm1_corr_relaxed_ao;  // P_ao - 2*C_occ C_occ^T
        Eigen::MatrixXd zeta_ao;              // for the s_zeta overlap term
        Eigen::MatrixXd imat_ao;              // symmetrized orbital Lagrangian (AO)
    };

    [[nodiscard]] std::expected<PT2RelaxedDensity, std::string>
    solve_pt2_relaxed_density(const PT2RelaxedDensityInputs &in);

} // namespace DFT::Gradient

#endif
