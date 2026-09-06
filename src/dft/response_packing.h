#ifndef DFT_RESPONSE_PACKING_H
#define DFT_RESPONSE_PACKING_H

#include <Eigen/Dense>

namespace DFT::Driver
{
    // F3.5 (docs/DFT_ANALYTIC_FXC_HESSIAN.md): pure plumbing -- project
    // an AO-basis induced XC potential into the (a,i) MO block and pack it
    // into the SAME flat layout HartreeFock::Correlation::build_rhf_cphf_matrix
    // / build_uhf_cphf_matrix already use for the orbital-Hessian linear
    // part: idx(a,i) = a*n_occ + i (VIRTUAL-major), NOT
    // ResponseExcitationSpace::flat_index's i*n_virt + a (OCCUPIED-major,
    // driver.h). These are genuinely different conventions in this
    // codebase -- confirmed by reading both implementations, not assumed --
    // so a future SOSCF/DFT wiring step that reads an FD-oracle
    // ResponseExcitationSpace block and hands it to solve_augmented_hessian
    // alongside a CPHF-convention gradient/Hessian must translate between
    // them explicitly, or every off-diagonal element silently lands in the
    // wrong row/column. This helper exists so that translation has exactly
    // one implementation.
    //
    // Kept in its own tiny translation unit (not driver.cpp) so a test that
    // only needs this pure index/projection utility does not have to link
    // the entire KS-loop driver and its transitive SCF/post-HF/gradient
    // dependencies.
    [[nodiscard]] Eigen::VectorXd pack_hessian_vector_product_cphf_order(
        const Eigen::Ref<const Eigen::MatrixXd> &delta_v_xc_ao,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt);

} // namespace DFT::Driver

#endif // DFT_RESPONSE_PACKING_H
