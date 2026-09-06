#ifndef DFT_ANALYTIC_HESSIAN_H
#define DFT_ANALYTIC_HESSIAN_H

#include <expected>
#include <string>

#include <Eigen/Dense>

#include "base/grid.h"
#include "base/wrapper.h"
#include "xc_grid.h"

namespace DFT::Driver
{
    // D2.0 (docs/SOSCF_DFT.md): promotes the analytic XC
    // Hessian-vector product derived and point-level-verified in
    // docs/DFT_ANALYTIC_FXC_HESSIAN.md (F3.1 LDA, F3.3.1-F3.3.3 GGA
    // T1+T2+T3) into a real, callable production function. RKS, LDA and
    // GGA, unpolarized only -- UKS/polarized is D3's job (F3.4's own
    // T1..T5 algebra).
    //
    // Given the ground-state density P and a trial response density dP
    // (both AO-basis, dP built the same C_virt*x*C_occ^T + h.c. way U1/U2
    // and F3's own probes did), returns delta_V_xc in the AO basis:
    //
    //   LDA:  delta_V_xc = sum_r w_r * v2rho2(r) * drho(r) * phi(r)phi(r)^T
    //   GGA:  delta_V_xc = T1*AA + (T2+T3 bundled into one projected
    //         gradient term), exactly F3.3.3's own verified decomposition,
    //         built on the grid the same way accumulate_local_potential
    //         (src/dft/ks_matrix.cpp) builds the ordinary first-derivative
    //         V_xc: a rank-1 update from the coefficient term plus a
    //         symmetric rank-2 update from the gradient-coupling term.
    //
    // Kept out of driver.cpp, in its own small translation unit depending
    // only on xc_grid.h (Eigen + the functional wrapper + AO/grid types),
    // not the whole KS-loop driver -- the same reasoning response_packing.h
    // (F3.5) used, so a later test does not have to link SCF/post-HF/
    // gradient machinery it does not need.
    [[nodiscard]] std::expected<Eigen::MatrixXd, std::string> compute_analytic_xc_hessian_vector_product(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_density,
        const Eigen::Ref<const Eigen::MatrixXd> &trial_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

    // D2.2.0 (docs/SOSCF_DFT.md): the orbital-energy-difference
    // diagonal eps(n_occ+a) - eps(i), flattened into the SAME (a,i)
    // virtual-major order pack_hessian_vector_product_cphf_order (F3.5)
    // already uses -- idx(a,i) = a*n_occ + i. This is the RKS analogue of
    // the single line `A(ai,ai) += eps(a) - eps(i)` inside
    // HartreeFock::Correlation::build_rhf_cphf_matrix
    // (src/post_hf/rhf_response.cpp) -- it has no XC-functional dependence
    // at all (it is pure orbital-energy bookkeeping), so it is correct for
    // RHF and RKS alike given the same C/eps. Pure plumbing: no grid, no
    // functional, kept here only because D2.2's h_op composes it alongside
    // this file's other piece.
    [[nodiscard]] Eigen::VectorXd orbital_energy_difference_diagonal(
        const Eigen::Ref<const Eigen::VectorXd> &eps,
        int n_occ);

    // D3.0 (docs/SOSCF_DFT.md): the polarized (UKS) analogue of
    // compute_analytic_xc_hessian_vector_product above, promoting
    // tests/dft_gga_polarized_hessian_selfcheck.cpp's own verified T1..T5
    // point-level algebra (F3.4.2-F3.4.4,
    // docs/DFT_ANALYTIC_FXC_HESSIAN.md) into a real production
    // function. LDA and GGA polarized both handled (LDA path added
    // alongside GGA -- see below), same is_lda_like()/is_gga_like()
    // dispatch compute_analytic_xc_hessian_vector_product already uses.
    //
    // Given ground-state (P^a, P^b) and trial (dP^a, dP^b), both AO-basis,
    // returns (delta_V_xc^a, delta_V_xc^b) in the AO basis:
    //
    //   LDA: delta_V_xc^a = sum_r w_r * [v2rho2_aa(r)*drho_a(r) +
    //                                    v2rho2_ab(r)*drho_b(r)] * phi(r)phi(r)^T
    //        delta_V_xc^b = sum_r w_r * [v2rho2_ab(r)*drho_a(r) +
    //                                    v2rho2_bb(r)*drho_b(r)] * phi(r)phi(r)^T
    //        -- the polarized analogue of F3.1's LDA formula: no gradient
    //        terms at all (V_xc^sigma = vrho_sigma alone for LDA), so the
    //        only cross-spin coupling is the single v2rho2_ab slot shared
    //        by both channels' formulas (symmetric there, unlike GGA's
    //        distinct T4/T4' cross terms).
    //
    //   GGA: delta_V_xc^a = T1*AA + (T2+T3 bundled: SELF gradient term, grad_rho_a)
    //                             + (T4+T5 bundled: CROSS gradient term, grad_rho_b)
    //        delta_V_xc^b = T1'*AA + (T2'+T3' bundled: SELF term, grad_rho_b)
    //                              + (T4'+T5' bundled: CROSS term, grad_rho_a)
    //
    // exactly the test file's own check_mixed/check_beta decomposition,
    // ported into whole-molecule AO-projection form the same way D2.0's
    // restricted GGA branch was ported from F3.3.3's probe. The "aa"-rooted
    // v2rhosigma/v2sigma2 slots become "bb" for the beta channel; the "ab"
    // cross slot is READ FROM THE SAME libxc output for both channels (one
    // shared cross-spin sigma channel, not a mirrored pair) -- exactly the
    // test file's own header-comment warning.
    [[nodiscard]] std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    compute_analytic_xc_hessian_vector_product_polarized(
        const MolecularGrid &molecular_grid,
        const AOGridEvaluation &ao_grid,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_beta_density,
        const Eigen::Ref<const Eigen::MatrixXd> &trial_alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &trial_beta_density,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional);

} // namespace DFT::Driver

#endif // DFT_ANALYTIC_HESSIAN_H
