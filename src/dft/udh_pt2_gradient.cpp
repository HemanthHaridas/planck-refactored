#include "udh_pt2_gradient.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
    constexpr double tolerance = 1e-10;
    bool near(double a, double b)
    {
        return std::isfinite(a) && std::isfinite(b) &&
            std::abs(a-b) <= tolerance * (1.0 + std::max(std::abs(a), std::abs(b)));
    }
    std::size_t index(int i, int j, int a, int b, int oj, int va, int vb)
    {
        return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b;
    }
    std::expected<std::size_t, std::string> extent(int oi, int oj, int va, int vb)
    {
        std::size_t n = 1;
        for (int d : {oi, oj, va, vb})
        {
            if (d < 0 || (d && n > std::numeric_limits<std::size_t>::max()/d))
                return std::unexpected("invalid or overflowing tensor extent");
            n *= static_cast<std::size_t>(d);
        }
        return n;
    }
    bool spin_space(int no, int nv, const Eigen::MatrixXd &c,
                    const Eigen::VectorXd &eps, const Eigen::VectorXd &occ,
                    const std::vector<int> &active, const Eigen::MatrixXd &s)
    {
        const Eigen::Index n = s.rows();
        if (no < 0 || nv <= 0 || static_cast<Eigen::Index>(no)+nv != n ||
            c.rows()!=n || c.cols()!=n || eps.size()!=n || occ.size()!=n ||
            active.size()!=static_cast<std::size_t>(n) || !c.allFinite() ||
            !eps.allFinite() || !occ.allFinite()) return false;
        for (Eigen::Index p=0; p<n; ++p)
            if (active[p]!=p || occ(p)!=(p<no ? 1.0 : 0.0)) return false;
        return (c.transpose()*s*c-Eigen::MatrixXd::Identity(n,n)).cwiseAbs().maxCoeff()
            <= tolerance;
    }

    std::expected<double, std::string> channel(
        const std::vector<double> &t, const std::vector<double> &g,
        const Eigen::VectorXd &ei, const Eigen::VectorXd &ej,
        int oi, int oj, int va, int vb, bool same_spin)
    {
        const auto n = extent(oi,oj,va,vb);
        if (!n) return std::unexpected(n.error());
        if (t.size()!=*n || g.size()!=*n)
            return std::unexpected("amplitude/direct-integral extent mismatch (missing stored amplitudes?)");
        double energy = 0.0;
        for (int i=0; i<oi; ++i) for (int j=0; j<oj; ++j)
            for (int a=0; a<va; ++a) for (int b=0; b<vb; ++b)
            {
                const auto k = index(i,j,a,b,oj,va,vb);
                double numerator = g[k];
                if (!std::isfinite(t[k]) || !std::isfinite(g[k]))
                    return std::unexpected("nonfinite amplitude or integral");
                if (same_spin)
                {
                    const auto swap_a = index(i,j,b,a,oj,va,vb);
                    const auto swap_i = index(j,i,a,b,oj,va,vb);
                    const auto pair = index(j,i,b,a,oj,va,vb);
                    if (!near(t[k],-t[swap_a]) || !near(t[k],-t[swap_i]) ||
                        !near(g[k],g[pair]))
                        return std::unexpected("same-spin antisymmetry or direct-integral pair symmetry violated");
                    numerator -= g[swap_a];
                }
                const double denominator = ei(i)+ej(j)-ei(oi+a)-ej(oj+b);
                if (!std::isfinite(denominator) || denominator >= -1e-12)
                    return std::unexpected("nonnegative or near-zero canonical PT2 denominator");
                if (!near(t[k]*denominator,numerator))
                    return std::unexpected("canonical amplitude residual mismatch");
                energy += (same_spin ? 0.25 : 1.0)*t[k]*numerator;
            }
        if (!std::isfinite(energy)) return std::unexpected("nonfinite channel energy");
        return energy;
    }
}

namespace DFT::Gradient
{
    std::expected<UDHPT2EnergyContract, std::string> build_udh_pt2_energy_contract(
        const HartreeFock::Correlation::UMP2Result &r,
        const UDHPT2DirectIntegrals &g,
        const Eigen::Ref<const Eigen::MatrixXd> &overlap_ao,
        const HartreeFock::OptionsMP2 &options, double c_pt2,
        const UDHPT2Scope &scope)
    {
        const auto error = [](const std::string &message)
            -> std::expected<UDHPT2EnergyContract,std::string> {
                return std::unexpected("UDH U0 contract: " + message);
            };
        if (scope.reference!=HartreeFock::SCFType::UHF ||
            scope.basis!=HartreeFock::BasisType::Cartesian || scope.range_separated ||
            scope.solvent || scope.meta_gga || scope.independent_spin_scaling)
            return error("requires global, real collinear UKS Cartesian LDA/GGA with one PT2 scale and no solvent");
        if (!options.frozen.empty() || options.use_ri || options.level_shift!=0.0 || !options.with_t2)
            return error("requires all-active conventional unshifted PT2 with stored amplitudes");
        if (!r.converged || r.n_iter!=0 || !std::isfinite(c_pt2))
            return error("requires finite scale and converged canonical UMP2 result");
        if (overlap_ao.rows()==0 || overlap_ao.rows()!=overlap_ao.cols() ||
            !overlap_ao.allFinite() ||
            (overlap_ao-overlap_ao.transpose()).cwiseAbs().maxCoeff()>tolerance)
            return error("invalid AO overlap");
        if (!spin_space(r.nocca,r.nvira,r.mo_coeff_alpha,r.mo_energy_alpha,r.mo_occ_alpha,
                        r.active_mo_alpha,overlap_ao) ||
            !spin_space(r.noccb,r.nvirb,r.mo_coeff_beta,r.mo_energy_beta,r.mo_occ_beta,
                        r.active_mo_beta,overlap_ao) || (r.nocca==0 && r.noccb==0))
            return error("invalid all-active spin space, occupation or C^T S C normalization");
        const auto aa=channel(r.t2_aa,g.aa,r.mo_energy_alpha,r.mo_energy_alpha,
                              r.nocca,r.nocca,r.nvira,r.nvira,true);
        if (!aa) return error("aa: "+aa.error());
        const auto ab=channel(r.t2_ab,g.ab,r.mo_energy_alpha,r.mo_energy_beta,
                              r.nocca,r.noccb,r.nvira,r.nvirb,false);
        if (!ab) return error("ab: "+ab.error());
        const auto bb=channel(r.t2_bb,g.bb,r.mo_energy_beta,r.mo_energy_beta,
                              r.noccb,r.noccb,r.nvirb,r.nvirb,true);
        if (!bb) return error("bb: "+bb.error());
        UDHPT2EnergyContract out;
        out.nocc_alpha=r.nocca; out.nocc_beta=r.noccb;
        out.nvirt_alpha=r.nvira; out.nvirt_beta=r.nvirb;
        out.c_pt2=c_pt2; out.aa=*aa; out.ab=*ab; out.bb=*bb;
        out.same_spin=*aa+*bb; out.opposite_spin=*ab;
        out.unscaled_total=out.same_spin+out.opposite_spin;
        out.correction=c_pt2*out.unscaled_total;
        if (!near(out.same_spin,r.e_corr_ss) || !near(out.opposite_spin,r.e_corr_os) ||
            !near(out.unscaled_total,r.e_corr) || !std::isfinite(out.correction))
            return error("cached correlation-energy ledger disagrees with amplitudes/direct integrals");
        return out;
    }
}
