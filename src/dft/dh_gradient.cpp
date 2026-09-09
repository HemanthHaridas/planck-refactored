#include "dh_gradient.h"

namespace DFT::Gradient
{

    std::expected<HartreeFock::Correlation::RMP2Lagrangian, std::string>
    build_pt2_mo_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Correlation::RMP2Result &result,
        double pt2_scale,
        const HartreeFock::Correlation::KsVeffFn &ks_veff)
    {
        // The orbital-basis half -- gamma^1, the unrelaxed correlation density
        // and its veff, the orbital Lagrangian, the Z-vector RHS Xvo -- is
        // exactly build_rmp2_lagrangian, run on whatever orbitals `result`
        // carries (converged KS orbitals from the double-hybrid path).
        // `ks_veff` (when set) swaps the HF J - 1/2 K for the KS mean-field
        // response in veff_corr_ao / Xvo -- N3.5.4.
        auto lag = HartreeFock::Correlation::build_rmp2_lagrangian(
            calculator, shell_pairs, result, ks_veff);
        if (!lag)
            return lag;

        // The double-hybrid energy functional is E_KS + c_PT2 * E_PT2, so the
        // gradient's PT2 contribution is c_PT2 times every PT2 quantity that
        // enters the gradient linearly:
        //   - Xvo is the Z-vector RHS; the operator is linear, so z (and the
        //     relaxed occ-virt density) scale with it.
        //   - doo/dvv/dm1_corr_* are the unrelaxed correlation density; they
        //     enter the relaxed density linearly.
        //   - veff_corr_ao is linear in dm1_corr_ao.
        //   - imat_ao/imat_mo (the Lagrangian) feed Xvo and the overlap term.
        //   - dm2buf_full is the T2->AO 2-particle buffer; it feeds both the
        //     conventional pair-density 2e derivative term
        //     (build_rmp2_gradient_intermediates, `dm2v = 2*dm2buf_full`) and
        //     the RI 2e-gradient term directly, at full magnitude. It must be
        //     scaled here too -- leaving it unscaled makes the whole PT2
        //     correction NON-linear in c_PT2 (verified: N3.3b linearity check
        //     failed at rel 1.16 before this).
        if (pt2_scale != 1.0)
        {
            lag->doo *= pt2_scale;
            lag->dvv *= pt2_scale;
            lag->dm1_corr_mo *= pt2_scale;
            lag->dm1_corr_ao *= pt2_scale;
            lag->veff_corr_ao *= pt2_scale;
            lag->imat_ao *= pt2_scale;
            lag->imat_mo *= pt2_scale;
            lag->Xvo *= pt2_scale;
            for (double &v : lag->dm2buf_full)
                v *= pt2_scale;
        }
        return lag;
    }

} // namespace DFT::Gradient
