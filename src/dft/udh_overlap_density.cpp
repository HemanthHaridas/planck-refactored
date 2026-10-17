#include "udh_overlap_density.h"

namespace DFT::Gradient
{
    namespace
    {
        using M=Eigen::MatrixXd;
        using Orb=UDHOrbitalMatrices;
        double maxabs(const M &x) { return x.size()?x.cwiseAbs().maxCoeff():0; }
        bool valid(const M &x,Eigen::Index n)
        { return x.rows()==n && x.cols()==n && x.allFinite(); }
        bool canonical(const M &f,Eigen::Index n)
        {
            if(!valid(f,n)) return false;
            const M diagonal=f.diagonal().asDiagonal();
            return maxabs(f-diagonal)<=1e-10*(1+maxabs(f));
        }
        M project(const M &g,int o)
        {
            const auto n=g.rows(),v=n-o;
            M b=M::Zero(n,n);
            b.topLeftCorner(o,o)=-0.5*g.topLeftCorner(o,o);
            b.bottomRightCorner(v,v)=-0.5*g.bottomRightCorner(v,v);
            b.topRightCorner(o,v)=-g.topRightCorner(o,v);
            return b;
        }
    }

    std::expected<UDHOrbitalMatrices,std::string> build_udh_pair_overlap_density(
        const UDHOrbitalMatrices &g,int oa,int ob)
    {
        if(!valid(g.alpha,g.alpha.rows()) || !valid(g.beta,g.beta.rows()) ||
            oa<0 || ob<0 || oa>g.alpha.rows() || ob>g.beta.rows())
            return std::unexpected("UDH U6 pair W: invalid coefficient gradients or occupations.");
        return Orb{project(g.alpha,oa),project(g.beta,ob)};
    }

    std::expected<UDHOverlapDensity,std::string> build_udh_overlap_density(
        const UDHPairOrbitalGradient &pair,const UDHRelaxedDifferenceDensity &d,
        const UDHOrbitalMatrices &c,const UDHOrbitalMatrices &f,
        const Eigen::Ref<const M> &s,const UDHKSResponseOperator &response)
    {
        auto density_ao=transform_udh_relaxed_density_to_ao(d,c,s);
        if(!density_ao) return std::unexpected("UDH U6: "+density_ao.error());
        const auto na=c.alpha.cols(),nb=c.beta.cols();
        const int oa=d.nocc_alpha,ob=d.nocc_beta;
        if(!canonical(f.alpha,na) || !canonical(f.beta,nb) || response.nbasis!=s.rows())
            return std::unexpected("UDH U6: invalid canonical Fock or response dimension.");
        for(const auto *g:{&pair.aa.g,&pair.ab.g,&pair.bb.g,&pair.g})
            if(!valid(g->alpha,na) || !valid(g->beta,nb))
                return std::unexpected("UDH U6: invalid pair sector dimensions or values.");
        const M ga=pair.aa.g.alpha+pair.ab.g.alpha+pair.bb.g.alpha;
        const M gb=pair.aa.g.beta+pair.ab.g.beta+pair.bb.g.beta;
        if(!ga.allFinite() || !gb.allFinite() || maxabs(ga-pair.g.alpha)>1e-11*(1+maxabs(ga)) ||
            maxabs(gb-pair.g.beta)>1e-11*(1+maxabs(gb)))
            return std::unexpected("UDH U6: inconsistent pair sector sum.");
        UDHOverlapDensity out;
        out.pair_aa=*build_udh_pair_overlap_density(pair.aa.g,oa,ob);
        out.pair_ab=*build_udh_pair_overlap_density(pair.ab.g,oa,ob);
        out.pair_bb=*build_udh_pair_overlap_density(pair.bb.g,oa,ob);
        UDHSpinMatrices prime_ao,z_ao;
        const auto orbital=[](const M &raw,const M &coeff,const M &fm,int o,
            M &prime,M &z,M &ao_prime,M &ao_z)
        {
            const auto n=raw.rows(),v=n-o;
            M dp=raw; dp.bottomLeftCorner(v,o).setZero();
            prime=M::Zero(n,n); z=M::Zero(n,n);
            for(Eigen::Index p=0;p<n;++p) for(Eigen::Index q=0;q<n;++q)
                prime(p,q)=-0.5*dp(p,q)*(fm(p,p)+fm(q,q));
            for(Eigen::Index a=0;a<v;++a) for(int i=0;i<o;++i)
                z(o+a,i)=-fm(i,i)*raw(o+a,i);
            const M dp_sym=0.5*dp+0.5*dp.transpose();
            ao_prime=coeff*dp_sym*coeff.transpose();
            const M z_raw=coeff.rightCols(v)*raw.bottomLeftCorner(v,o)*coeff.leftCols(o).transpose();
            ao_z=0.5*z_raw+0.5*z_raw.transpose();
        };
        orbital(d.raw_mo.alpha,c.alpha,f.alpha,oa,out.dprime_orbital.alpha,out.z_orbital.alpha,prime_ao.alpha,z_ao.alpha);
        orbital(d.raw_mo.beta,c.beta,f.beta,ob,out.dprime_orbital.beta,out.z_orbital.beta,prime_ao.beta,z_ao.beta);
        const auto response_w=[&](const UDHSpinMatrices &trial)->std::expected<UDHOverlapResponse,std::string>
        {
            auto applied=response.apply_channels(trial);
            if(!applied) return std::unexpected("UDH U6 response W: "+applied.error());
            const auto oo=[](const M &coeff,const M &q,int o)->M
            {
                M w=M::Zero(coeff.cols(),coeff.cols());
                w.topLeftCorner(o,o)=-coeff.leftCols(o).transpose()*q*coeff.leftCols(o);
                return w;
            };
            const auto spin=[&](const UDHSpinMatrices &q)->Orb
            { return {oo(c.alpha,q.alpha,oa),oo(c.beta,q.beta,ob)}; };
            UDHOverlapResponse w;
            w.coulomb=spin({applied->coulomb,applied->coulomb});
            w.exchange=spin(applied->exchange);
            w.xc_from_alpha=spin(applied->xc_from_alpha); w.xc_from_beta=spin(applied->xc_from_beta);
            w.total=spin(applied->total);
            for(const auto *channel:{&w.coulomb,&w.exchange,&w.xc_from_alpha,&w.xc_from_beta,&w.total})
                if(!channel->alpha.allFinite() || !channel->beta.allFinite())
                    return std::unexpected("UDH U6 response W: nonfinite projected channel.");
            return w;
        };
        auto dp_response=response_w(prime_ao); if(!dp_response) return std::unexpected(dp_response.error());
        auto z_response=response_w(z_ao); if(!z_response) return std::unexpected(z_response.error());
        out.response_dprime=std::move(*dp_response); out.response_z=std::move(*z_response);
        const auto assemble=[](const M &a,const M &b,const M &cc,const M &dp,const M &z,const M &r,const M &rz)->M
        { return a+b+cc+dp+z+r+rz; };
        out.raw_mo.alpha=assemble(out.pair_aa.alpha,out.pair_ab.alpha,out.pair_bb.alpha,out.dprime_orbital.alpha,
            out.z_orbital.alpha,out.response_dprime.total.alpha,out.response_z.total.alpha);
        out.raw_mo.beta=assemble(out.pair_aa.beta,out.pair_ab.beta,out.pair_bb.beta,out.dprime_orbital.beta,
            out.z_orbital.beta,out.response_dprime.total.beta,out.response_z.total.beta);
        out.symmetric_mo={0.5*out.raw_mo.alpha+0.5*out.raw_mo.alpha.transpose(),
                          0.5*out.raw_mo.beta+0.5*out.raw_mo.beta.transpose()};
        out.symmetric_ao={c.alpha*out.symmetric_mo.alpha*c.alpha.transpose(),
                          c.beta*out.symmetric_mo.beta*c.beta.transpose()};
        out.total_ao=out.symmetric_ao.alpha+out.symmetric_ao.beta;
        if(!out.raw_mo.alpha.allFinite() || !out.raw_mo.beta.allFinite() || !out.total_ao.allFinite())
            return std::unexpected("UDH U6: nonfinite overlap density.");
        return out;
    }
}
