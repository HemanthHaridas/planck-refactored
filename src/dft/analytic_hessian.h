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
    // D2.0 (docs/SOSCF_UHF_DFT_SCOPE.md): promotes the analytic XC
    // Hessian-vector product derived and point-level-verified in
    // docs/SOSCF_DFT_ANALYTIC_FXC_SCOPE.md (F3.1 LDA, F3.3.1-F3.3.3 GGA
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

} // namespace DFT::Driver

#endif // DFT_ANALYTIC_HESSIAN_H
