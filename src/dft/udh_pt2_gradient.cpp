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

    bool symmetric_matrix(const Eigen::Ref<const Eigen::MatrixXd> &m, Eigen::Index n)
    {
        return m.rows()==n && m.cols()==n && m.allFinite() &&
            (n==0 || (m-m.transpose()).cwiseAbs().maxCoeff()<=tolerance);
    }

    std::expected<void,std::string> raw_channel(const std::vector<double> &t,
        int oi, int oj, int va, int vb, bool same_spin)
    {
        const auto n=extent(oi,oj,va,vb);
        if (!n) return std::unexpected(n.error());
        if (t.size()!=*n) return std::unexpected("amplitude extent mismatch");
        for (int i=0;i<oi;++i) for (int j=0;j<oj;++j)
            for (int a=0;a<va;++a) for (int b=0;b<vb;++b)
            {
                const auto k=index(i,j,a,b,oj,va,vb);
                if (!std::isfinite(t[k])) return std::unexpected("nonfinite amplitude");
                if (same_spin && (!near(t[k],-t[index(j,i,a,b,oj,va,vb)]) ||
                    !near(t[k],-t[index(i,j,b,a,oj,va,vb)])))
                    return std::unexpected("same-spin amplitude antisymmetry violated");
            }
        return {};
    }

    // Full spin-orbital -1/2 TT (oo), +1/2 TT (vv), restricted to one
    // same-spin sector. ab has two spin permutations and is handled below
    // with unit weight. No closed-shell occupation factor enters D'.
    void same_spin_density(Eigen::MatrixXd &d, const std::vector<double> &t,
                           int o, int v, double c)
    {
        if (t.empty()) return;
        const auto at=[&](int i,int j,int a,int b) { return t[index(i,j,a,b,o,v,v)]; };
        for (int i=0;i<o;++i) for (int j=0;j<o;++j)
            for (int k=0;k<o;++k) for (int a=0;a<v;++a) for (int b=0;b<v;++b)
                d(i,j)-=0.5*c*at(i,k,a,b)*at(j,k,a,b);
        for (int a=0;a<v;++a) for (int b=0;b<v;++b)
            for (int i=0;i<o;++i) for (int j=0;j<o;++j) for (int e=0;e<v;++e)
                d(o+a,o+b)+=0.5*c*at(i,j,a,e)*at(i,j,b,e);
    }

    std::expected<double,std::string> pair_channel(const std::vector<double> &t,
        const std::vector<double> &g, int oi,int oj,int va,int vb,bool same_spin,double scale)
    {
        if (g.size()!=t.size()) return std::unexpected("direct-integral extent mismatch");
        double value=0.0;
        for (int i=0;i<oi;++i) for (int j=0;j<oj;++j)
            for (int a=0;a<va;++a) for (int b=0;b<vb;++b)
            {
                const auto k=index(i,j,a,b,oj,va,vb);
                if (!std::isfinite(g[k])) return std::unexpected("nonfinite direct integral");
                if (same_spin && !near(g[k],g[index(j,i,b,a,oj,va,vb)]))
                    return std::unexpected("direct-integral pair symmetry violated");
                value+=scale*t[k]*g[k];
            }
        if (!std::isfinite(value)) return std::unexpected("nonfinite pair contraction");
        return value;
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

    std::expected<void,std::string> validate_udh_pt2_amplitudes(const UDHPT2Amplitudes &t)
    {
        const int oa=t.nocc_alpha, ob=t.nocc_beta, va=t.nvirt_alpha, vb=t.nvirt_beta;
        // Check sums before allocation as well as the tensor products.
        for (auto [o,v] : {std::pair{oa,va},std::pair{ob,vb}})
            if (o<0 || v<0 || o>std::numeric_limits<int>::max()-v)
                return std::unexpected("UDH U1 Dprime: invalid spin dimensions");
        const auto aa=raw_channel(t.aa,oa,oa,va,va,true);
        if (!aa) return std::unexpected("UDH U1 Dprime aa: "+aa.error());
        const auto ab=raw_channel(t.ab,oa,ob,va,vb,false);
        if (!ab) return std::unexpected("UDH U1 Dprime ab: "+ab.error());
        const auto bb=raw_channel(t.bb,ob,ob,vb,vb,true);
        if (!bb) return std::unexpected("UDH U1 Dprime bb: "+bb.error());
        return {};
    }

    std::expected<UDHPT2DPrime,std::string> build_udh_pt2_dprime(
        const UDHPT2Amplitudes &t,double c)
    {
        if (!std::isfinite(c)) return std::unexpected("UDH U1 Dprime: nonfinite PT2 scale");
        auto checked=validate_udh_pt2_amplitudes(t);
        if (!checked) return std::unexpected(checked.error());
        const int oa=t.nocc_alpha, ob=t.nocc_beta, va=t.nvirt_alpha, vb=t.nvirt_beta;
        UDHPT2DPrime out{Eigen::MatrixXd::Zero(oa+va,oa+va),
                         Eigen::MatrixXd::Zero(ob+vb,ob+vb)};
        same_spin_density(out.alpha_mo,t.aa,oa,va,c);
        same_spin_density(out.beta_mo,t.bb,ob,vb,c);
        const auto at=[&](int i,int j,int a,int b) { return t.ab[index(i,j,a,b,ob,va,vb)]; };
        for (int i=0;i<oa;++i) for (int j=0;j<oa;++j)
            for (int k=0;k<ob;++k) for (int a=0;a<va;++a) for (int b=0;b<vb;++b)
                out.alpha_mo(i,j)-=c*at(i,k,a,b)*at(j,k,a,b);
        for (int i=0;i<ob;++i) for (int j=0;j<ob;++j)
            for (int k=0;k<oa;++k) for (int a=0;a<va;++a) for (int b=0;b<vb;++b)
                out.beta_mo(i,j)-=c*at(k,i,a,b)*at(k,j,a,b);
        for (int a=0;a<va;++a) for (int b=0;b<va;++b)
            for (int i=0;i<oa;++i) for (int j=0;j<ob;++j) for (int e=0;e<vb;++e)
                out.alpha_mo(oa+a,oa+b)+=c*at(i,j,a,e)*at(i,j,b,e);
        for (int a=0;a<vb;++a) for (int b=0;b<vb;++b)
            for (int i=0;i<oa;++i) for (int j=0;j<ob;++j) for (int e=0;e<va;++e)
                out.beta_mo(ob+a,ob+b)+=c*at(i,j,e,a)*at(i,j,e,b);
        if (!out.alpha_mo.allFinite() || !out.beta_mo.allFinite())
            return std::unexpected("UDH U1 Dprime: nonfinite density contraction");
        return out;
    }

    std::expected<UDHPT2AODPrime,std::string> transform_udh_pt2_dprime_to_ao(
        const UDHPT2DPrime &d,const Eigen::Ref<const Eigen::MatrixXd> &ca,
        const Eigen::Ref<const Eigen::MatrixXd> &cb,
        const Eigen::Ref<const Eigen::MatrixXd> &s)
    {
        const auto n=s.rows();
        if (n==0 || !symmetric_matrix(s,n) || !symmetric_matrix(d.alpha_mo,n) ||
            !symmetric_matrix(d.beta_mo,n) || ca.rows()!=n || ca.cols()!=n ||
            cb.rows()!=n || cb.cols()!=n || !ca.allFinite() || !cb.allFinite())
            return std::unexpected("UDH U1 AO adapter: invalid all-active matrix shape or values");
        const Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(n,n);
        const Eigen::MatrixXd norm_a=ca.transpose()*s*ca-identity;
        const Eigen::MatrixXd norm_b=cb.transpose()*s*cb-identity;
        if (!norm_a.allFinite() || !norm_b.allFinite() ||
            norm_a.cwiseAbs().maxCoeff()>tolerance || norm_b.cwiseAbs().maxCoeff()>tolerance)
            return std::unexpected("UDH U1 AO adapter: C^T S C normalization violated");
        UDHPT2AODPrime out{ca*d.alpha_mo*ca.transpose(),cb*d.beta_mo*cb.transpose()};
        if (!out.alpha_ao.allFinite() || !out.beta_ao.allFinite())
            return std::unexpected("UDH U1 AO adapter: nonfinite transformed density");
        return out;
    }

    std::expected<UDHPT2StationaryScalar,std::string> evaluate_udh_pt2_stationary_scalar(
        const UDHPT2Amplitudes &t,const UDHPT2DirectIntegrals &g,
        const Eigen::Ref<const Eigen::MatrixXd> &fa,
        const Eigen::Ref<const Eigen::MatrixXd> &fb,double c)
    {
        auto d=build_udh_pt2_dprime(t,c);
        if (!d) return std::unexpected(d.error());
        if (!symmetric_matrix(fa,d->alpha_mo.rows()) || !symmetric_matrix(fb,d->beta_mo.rows()))
            return std::unexpected("UDH U1 scalar: invalid symmetric MO Fock matrix");
        const auto aa=pair_channel(t.aa,g.aa,t.nocc_alpha,t.nocc_alpha,t.nvirt_alpha,t.nvirt_alpha,true,c);
        if (!aa) return std::unexpected("UDH U1 scalar aa: "+aa.error());
        const auto ab=pair_channel(t.ab,g.ab,t.nocc_alpha,t.nocc_beta,t.nvirt_alpha,t.nvirt_beta,false,2*c);
        if (!ab) return std::unexpected("UDH U1 scalar ab: "+ab.error());
        const auto bb=pair_channel(t.bb,g.bb,t.nocc_beta,t.nocc_beta,t.nvirt_beta,t.nvirt_beta,true,c);
        if (!bb) return std::unexpected("UDH U1 scalar bb: "+bb.error());
        UDHPT2StationaryScalar out;
        out.dprime=std::move(*d);
        out.pair_aa=*aa; out.pair_ab=*ab; out.pair_bb=*bb;
        out.pair=*aa+*ab+*bb;
        out.dprime_f_alpha=out.dprime.alpha_mo.cwiseProduct(fa).sum();
        out.dprime_f_beta=out.dprime.beta_mo.cwiseProduct(fb).sum();
        out.dprime_f=out.dprime_f_alpha+out.dprime_f_beta;
        out.total=out.pair+out.dprime_f;
        if (!std::isfinite(out.pair) || !std::isfinite(out.dprime_f_alpha) ||
            !std::isfinite(out.dprime_f_beta) || !std::isfinite(out.total))
            return std::unexpected("UDH U1 scalar: nonfinite stationary contraction");
        return out;
    }

    std::expected<UDHPT2StationaryContract,std::string> build_udh_pt2_stationary_contract(
        const HartreeFock::Correlation::UMP2Result &r,const UDHPT2DirectIntegrals &g,
        const Eigen::Ref<const Eigen::MatrixXd> &s,const HartreeFock::OptionsMP2 &options,
        double c,const UDHPT2Scope &scope)
    {
        auto energy=build_udh_pt2_energy_contract(r,g,s,options,c,scope);
        if (!energy) return std::unexpected(energy.error());
        const UDHPT2Amplitudes t{r.nocca,r.noccb,r.nvira,r.nvirb,r.t2_aa,r.t2_ab,r.t2_bb};
        const Eigen::MatrixXd fa=r.mo_energy_alpha.asDiagonal(),fb=r.mo_energy_beta.asDiagonal();
        auto scalar=evaluate_udh_pt2_stationary_scalar(t,g,fa,fb,c);
        if (!scalar) return std::unexpected(scalar.error());
        if (!near(scalar->pair_aa,2*c*energy->aa) || !near(scalar->pair_ab,2*c*energy->ab) ||
            !near(scalar->pair_bb,2*c*energy->bb) || !near(scalar->pair,2*energy->correction) ||
            !near(scalar->dprime_f,-energy->correction) || !near(scalar->total,energy->correction))
            return std::unexpected("UDH U1 contract: canonical stationary identity mismatch");
        return UDHPT2StationaryContract{std::move(*energy),std::move(*scalar)};
    }
}
