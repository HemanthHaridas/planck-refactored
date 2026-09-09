#ifndef DFT_DH_GRADIENT_H
#define DFT_DH_GRADIENT_H

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/types.h"
#include "post_hf/mp2.h"
#include "post_hf/mp2_gradient.h"

// Double-hybrid analytic-gradient support.
//
// A double-hybrid gradient is NOT "KS-hybrid gradient + RMP2 gradient": the
// PT2 correction is evaluated on the *KS* orbitals, so its orbital-relaxation
// (Z-vector) term is solved against the KS orbital Hessian, not the HF CPHF
// matrix. See docs/DOUBLE_HYBRID_GRADIENT_SCOPE.md.
//
// This header carries only the MP2-side entry point (no libxc dependency).
// The Z-vector solve against the KS orbital Hessian lives in
// dh_relaxed_density.h, which pulls in the DFT XC stack.
namespace DFT::Gradient
{

    // Orbital-basis half of the PT2 gradient: gamma^1 blocks, unrelaxed
    // correlation density + veff, orbital Lagrangian, and the Z-vector RHS
    // Xvo. This is HartreeFock::Correlation::build_rmp2_lagrangian run on
    // whatever orbitals `result` carries -- KS orbitals from the double-hybrid
    // path -- with every gradient-linear quantity scaled by `pt2_scale`
    // (the double-hybrid c_PT2 coefficient; use 1.0 for plain MP2).
    // pt2_scale = 0 zeros the whole PT2 contribution, so the surrounding
    // gradient must reduce to the plain KS-hybrid gradient -- the cheap
    // correctness anchor for the feature.
    //
    // `result` must carry stored T2 amplitudes (options.with_t2) and a full
    // (non-frozen) orbital space.
    //
    // `ks_veff` (optional): the KS mean-field response
    // J[d] - 1/2 c_x K[d] + V_xc_response[d], applied to dm1_corr_ao inside
    // build_rmp2_lagrangian in place of the HF J - 1/2 K. Built by the driver
    // (it needs the grid + functionals); this file stays libxc-free. Null =>
    // HF veff (wrong for a real double hybrid -- see
    // docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md).
    [[nodiscard]] std::expected<HartreeFock::Correlation::RMP2Lagrangian, std::string>
    build_pt2_mo_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Correlation::RMP2Result &result,
        double pt2_scale,
        const HartreeFock::Correlation::KsVeffFn &ks_veff = {});

} // namespace DFT::Gradient

#endif
