#ifndef HF_POSTHF_MP2_GRADIENT_H
#define HF_POSTHF_MP2_GRADIENT_H

#include <Eigen/Core>
#include <expected>
#include <functional>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"
#include "post_hf/mp2.h"

namespace HartreeFock::Correlation
{
    struct RMP2GradientIntermediates
    {
        Eigen::MatrixXd electronic_gradient;
        Eigen::MatrixXd P_mo;
        Eigen::MatrixXd P_ao;
        Eigen::MatrixXd W_ao;
        Eigen::MatrixXd P_total_ao;
        Eigen::MatrixXd P_gamma_ao;
        Eigen::MatrixXd im1_ao;
        Eigen::MatrixXd zeta_ao;
        Eigen::MatrixXd vhf_s1occ_ao;
        std::vector<double> Gamma_pair_ao;
        // Exposed for the DH B2 direct Eq. (46)/(47) equivalence gate.
        Eigen::MatrixXd two_e_nonseparable_gradient;
        Eigen::MatrixXd two_e_separable_gradient;
        Eigen::MatrixXd overlap_gradient;
    };

    // Orbital-basis half of the RMP2 gradient: the unrelaxed correlation
    // density, its Coulomb/exchange response, the orbital Lagrangian, and the
    // Z-vector right-hand side Xvo. None of this touches derivative integrals,
    // so it is shared between build_rmp2_gradient_intermediates (which then
    // does solve_rhf_cphf + the derivative sweeps) and the double-hybrid
    // gradient path (which solves Xvo against the KS orbital Hessian instead).
    // Everything is in whatever orbital basis `result` carries.
    struct RMP2Lagrangian
    {
        int n_occ = 0;
        int n_virt = 0;

        Eigen::MatrixXd doo;          // gamma^1 occ-occ block (depletion sign)
        Eigen::MatrixXd dvv;          // gamma^1 virt-virt block
        Eigen::MatrixXd dm1_corr_mo;  // symmetrized doo/dvv on the diagonal blocks
        Eigen::MatrixXd dm1_corr_ao;  // C dm1_corr_mo C^T
        Eigen::MatrixXd veff_corr_ao; // 2*(J - 1/2 K)[dm1_corr_ao]
        Eigen::MatrixXd imat_ao;      // orbital Lagrangian, AFTER the -1 flip
        Eigen::MatrixXd imat_mo;      // C^T imat_ao S C
        Eigen::MatrixXd Xvo;          // Z-vector RHS, n_virt x n_occ

        std::vector<double> dm2buf_full; // T2->AO 2-RDM buffer; RI 2e-grad term reuses it
    };

    // Mean-field response applied to an AO density, i.e. the quantity
    // build_veff_from_density returns for HF ( J[d] - 1/2 K[d] ). The
    // double-hybrid gradient path passes a closure computing the KS response
    // J[d] - 1/2 c_x K[d] + V_xc_response[d] instead. Null => HF (RMP2 path,
    // byte-identical). See docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md.
    using KsVeffFn = std::function<Eigen::MatrixXd(const Eigen::MatrixXd &density)>;

    std::expected<RMP2Lagrangian, std::string> build_rmp2_lagrangian(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result,
        const KsVeffFn &ks_veff = {});

    // Relaxed one-particle density and energy-weighted density, given the
    // Lagrangian and the solved Z-vector z (n_virt x n_occ). The Z-vector
    // operator is the caller's choice -- solve_rhf_cphf for RMP2, the KS
    // orbital Hessian for a double hybrid -- but everything downstream of it
    // (relaxed P, zeta, the symmetrized imat, vhf_s1occ, W) is identical, so
    // it lives here and both paths call it.
    struct RMP2RelaxedDensity
    {
        Eigen::MatrixXd P_mo;                 // 2*I + gamma^1_diag + z_ov
        Eigen::MatrixXd P_ao;                 // C P_mo C^T
        Eigen::MatrixXd dm1_corr_relaxed_ao;  // P_ao - 2*C_occ C_occ^T
        Eigen::MatrixXd zeta_ao;              // W_ref + C (zeta_w .* corr_relaxed_mo) C^T
        Eigen::MatrixXd imat_ao;              // symmetrized orbital Lagrangian, AO
        Eigen::MatrixXd vhf_s1occ_ao;         // occ-projected veff of the relaxed correction
        Eigen::MatrixXd W_ao;                 // 0.5*(zeta + zeta^T - imat - imat^T) + vhf_s1occ
    };

    // `ks_veff` (optional): same KS mean-field response as build_rmp2_lagrangian,
    // used for vhf_s1occ_ao. Null => HF (RMP2 path, byte-identical).
    std::expected<RMP2RelaxedDensity, std::string> build_rmp2_energy_weighted_density(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result,
        const RMP2Lagrangian &lagrangian,
        const Eigen::MatrixXd &z,
        const KsVeffFn &ks_veff = {});

    struct UMP2GradientIntermediates
    {
        Eigen::MatrixXd electronic_gradient;
        Eigen::MatrixXd P_alpha_ao;
        Eigen::MatrixXd P_beta_ao;
        Eigen::MatrixXd P_alpha_corr_ao;
        Eigen::MatrixXd P_beta_corr_ao;
        Eigen::MatrixXd P_total_ao;
        Eigen::MatrixXd W_ao;
        std::vector<double> Gamma_pair_ao;
    };

    // Optionally hand build_rmp2_gradient_intermediates a Lagrangian + Z-vector
    // solved elsewhere. The double-hybrid gradient path uses this to inject a
    // c_PT2-scaled Lagrangian and a Z-vector solved against the KS orbital
    // Hessian (not solve_rhf_cphf). Both pointers must be set or both null.
    struct RMP2PreSolved
    {
        const RMP2Lagrangian *lagrangian = nullptr;
        const Eigen::MatrixXd *z = nullptr; // n_virt x n_occ
        // KS mean-field response for vhf_s1occ in the internal
        // build_rmp2_energy_weighted_density call (N3.5.5). The injected
        // `lagrangian` already carries the KS veff in its veff_corr_ao/Xvo;
        // this covers the one veff site downstream of the Z-vector. Null => HF.
        KsVeffFn ks_veff = {};
    };

    // Public gradient-builder surface now consumes an explicit MP2 kernel
    // result instead of re-running MP2 internally.
    std::expected<RMP2GradientIntermediates, std::string> build_rmp2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result,
        RMP2PreSolved presolved = {});

    // Correction-only <D h^(x)> primitive shared by the direct double-hybrid
    // Eq. (33) assembly. `density` is not augmented with the HF reference.
    Eigen::MatrixXd contract_rmp2_one_electron_gradient(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density);

    std::expected<UMP2GradientIntermediates, std::string> build_ump2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UMP2Result &result);
} // namespace HartreeFock::Correlation

#endif
