#include "udh_two_particle_density.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace DFT::Gradient
{
    namespace
    {
        using M=Eigen::MatrixXd;
        std::size_t ti(int i,int j,int a,int b,int oj,int va,int vb)
        { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
        double maxabs(const M &m) { return m.size()?m.cwiseAbs().maxCoeff():0.0; }
        bool equal(const M &a,const M &b)
        { return a.rows()==b.rows() && a.cols()==b.cols() && a.allFinite() && b.allFinite() &&
            maxabs(a-b)<=1e-11*(1+std::max(maxabs(a),maxabs(b))); }
        DHPairChannel channel(const std::vector<double> &t,int os,int ot,int vs,int vt,
            const M &cs,const M &ct,double scale)
        {
            DHPairChannel out{os,vs,ot,vt,&cs,&ct,{}};
            out.fill_amplitudes=[&t,ot,vs,vt,scale](int i0,int count,double *dst)
            {
                for(int il=0;il<count;++il) for(int a=0;a<vs;++a) for(int j=0;j<ot;++j) for(int b=0;b<vt;++b)
                    dst[((static_cast<std::size_t>(il)*vs+a)*ot+j)*vt+b]=scale*t[ti(i0+il,j,a,b,ot,vs,vt)];
            };
            return out;
        }
    }

    std::expected<UDHTwoParticleDensity,std::string> build_udh_two_particle_density(
        const UDHPT2Amplitudes &t,double c,const UDHRelaxedDifferenceDensity &d,
        const UDHOrbitalMatrices &coeff,const Eigen::Ref<const M> &s,double ax)
    {
        if(s.rows()<=0 || s.rows()>std::numeric_limits<int>::max() || !std::isfinite(ax) || ax<0)
            return std::unexpected("UDH U7: invalid AO extent or exact-exchange coefficient.");
        // Bound spin matrix allocations before U1 constructs Dprime. Empty
        // amplitudes with a huge occupied/virtual extent are not a loophole.
        for(auto [o,v]:{std::pair{t.nocc_alpha,t.nvirt_alpha},std::pair{t.nocc_beta,t.nvirt_beta}})
            if(o<0 || v<0 || static_cast<long long>(o)+v>s.rows())
                return std::unexpected("UDH U7: spin extent exceeds the AO frame.");
        auto dp=build_udh_pt2_dprime(t,c);
        if(!dp) return std::unexpected("UDH U7: "+dp.error());
        auto ao=transform_udh_relaxed_density_to_ao(d,coeff,s);
        if(!ao) return std::unexpected("UDH U7: "+ao.error());
        const int oa=t.nocc_alpha,ob=t.nocc_beta,va=t.nvirt_alpha,vb=t.nvirt_beta;
        const auto same_prime=[](const M &raw,const M &prime,int o,int v)
        { return raw.rows()==o+v && equal(raw.topLeftCorner(o,o),prime.topLeftCorner(o,o)) &&
            equal(raw.bottomRightCorner(v,v),prime.bottomRightCorner(v,v)); };
        if(d.nocc_alpha!=oa || d.nocc_beta!=ob || !same_prime(d.raw_mo.alpha,dp->alpha_mo,oa,va) ||
            !same_prime(d.raw_mo.beta,dp->beta_mo,ob,vb))
            return std::unexpected("UDH U7: relaxed Dprime does not match amplitudes and PT2 scale.");
        UDHTwoParticleDensity out;
        out.n_ao=static_cast<int>(s.rows()); out.relaxed_ao=std::move(*ao);
        out.ground_ao={coeff.alpha.leftCols(oa)*coeff.alpha.leftCols(oa).transpose(),
                       coeff.beta.leftCols(ob)*coeff.beta.leftCols(ob).transpose()};
        out.c_pt2=c; out.exact_exchange=ax;
        return out;
    }

    DHEq33TwoParticleSource make_udh_two_particle_source(
        const UDHTwoParticleDensity &g,const UDHPT2Amplitudes &t,const UDHOrbitalMatrices &coeff)
    {
        const int oa=t.nocc_alpha,ob=t.nocc_beta,va=t.nvirt_alpha,vb=t.nvirt_beta;
        const double c=g.c_pt2,ax=g.exact_exchange;
        DHEq33TwoParticleSource out;
        out.n_ao=g.n_ao;
        out.coulomb_d=g.relaxed_ao.total;
        out.coulomb_p=g.ground_ao.alpha+g.ground_ao.beta;
        out.exchange.push_back({g.relaxed_ao.spin.alpha,g.ground_ao.alpha,-ax});
        out.exchange.push_back({g.relaxed_ao.spin.beta,g.ground_ao.beta,-ax});
        out.pair.push_back(channel(t.aa,oa,oa,va,va,coeff.alpha,coeff.alpha,c));
        out.pair.push_back(channel(t.ab,oa,ob,va,vb,coeff.alpha,coeff.beta,2*c));
        out.pair.push_back(channel(t.bb,ob,ob,vb,vb,coeff.beta,coeff.beta,c));
        return out;
    }
}
