#include "dft/udh_relaxed_density.h"
#include "dft/udh_zvector.h"
#include "dft/dh_pt2_gradient.h"
#include <Eigen/QR>
#include <cmath>
#include <iostream>
#include <limits>
#include <source_location>

namespace
{
    using namespace DFT::Gradient;
    using M=Eigen::MatrixXd;
    using Orb=UDHOrbitalMatrices;
    bool ok=true;
    void check(bool pass,std::source_location loc=std::source_location::current())
    { if(!pass) { ok=false; std::cerr<<"FAIL U5 line "<<loc.line()<<'\n'; } }
    double dot(const M &a,const M &b) { return (a.array()*b.array()).sum(); }
    double maxabs(const M &a) { return a.size()?a.cwiseAbs().maxCoeff():0; }
    void near(double a,double b,double tol=2e-11,std::source_location loc=std::source_location::current())
    { check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tol*(1+std::abs(b)),loc); }
    void equal(const M &a,const M &b,std::source_location loc=std::source_location::current())
    { check(a.rows()==b.rows() && a.cols()==b.cols(),loc);
      if(a.rows()==b.rows() && a.cols()==b.cols()) near(maxabs(a-b),0,2e-11,loc); }
    std::size_t idx(int i,int j,int a,int b,int oj,int va,int vb)
    { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
    UDHPT2Amplitudes amplitudes(int oa,int ob,int va,int vb)
    {
        UDHPT2Amplitudes t{oa,ob,va,vb,{},{},{}};
        const auto same=[](int o,int v)
        {
            std::vector<double> out(static_cast<std::size_t>(o)*o*v*v);
            for(int i=0;i<o;++i) for(int j=i+1;j<o;++j)
                for(int a=0;a<v;++a) for(int b=a+1;b<v;++b)
                {
                    const double x=.03*std::sin(1+i+3*j+5*a+7*b);
                    out[idx(i,j,a,b,o,v,v)]=out[idx(j,i,b,a,o,v,v)]=x;
                    out[idx(j,i,a,b,o,v,v)]=out[idx(i,j,b,a,o,v,v)]=-x;
                }
            return out;
        };
        t.aa=same(oa,va); t.bb=same(ob,vb);
        t.ab.resize(static_cast<std::size_t>(oa)*ob*va*vb);
        for(std::size_t k=0;k<t.ab.size();++k) t.ab[k]=.04*std::cos(.7*(k+1));
        return t;
    }
    M direction(int v,int o,double seed)
    {
        M z(v,o);
        for(int a=0;a<v;++a) for(int i=0;i<o;++i) z(a,i)=.13*std::sin(seed+a+2*i);
        return z;
    }
    M frame(int n,int m,double seed,const M &l)
    {
        M x=M::Identity(n,n);
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) x(p,q)+=.2*std::cos(seed+p+3*q);
        const M q=x.householderQr().householderQ()*M::Identity(n,n);
        const M full=l.transpose().triangularView<Eigen::Upper>().solve(q);
        return full.leftCols(m);
    }
    // Independent scalar, without either U5 matrix adapter. Explicit AO/MO
    // indices keep the one-sided z:F_ai multiplier separate from D':F.
    double scalar(const M &dp,const M &z,const M &c,const M &f)
    {
        double value=0;
        for(int p=0;p<dp.rows();++p) for(int q=0;q<dp.cols();++q)
            for(int mu=0;mu<c.rows();++mu) for(int nu=0;nu<c.rows();++nu)
                value+=dp(p,q)*c(mu,p)*f(mu,nu)*c(nu,q);
        for(int a=0;a<z.rows();++a) for(int i=0;i<z.cols();++i)
            for(int mu=0;mu<c.rows();++mu) for(int nu=0;nu<c.rows();++nu)
                value+=z(a,i)*c(mu,z.cols()+a)*f(mu,nu)*c(nu,i);
        return value;
    }
    void fixture(int oa,int ob,int va,int vb)
    {
        const int n=std::max({oa+va,ob+vb,1});
        const auto t=amplitudes(oa,ob,va,vb);
        const auto dp=build_udh_pt2_dprime(t,.27);
        check(dp.has_value()); if(!dp) return;
        Orb z{direction(va,oa,.3),direction(vb,ob,1.1)};
        const auto d=build_udh_relaxed_difference_density(*dp,z);
        check(d.has_value()); if(!d) return;
        for(int spin=0;spin<2;++spin)
        {
            const auto &raw=spin?d->raw_mo.beta:d->raw_mo.alpha;
            const auto &sym=spin?d->symmetric_mo.beta:d->symmetric_mo.alpha;
            const auto &prime=spin?dp->beta_mo:dp->alpha_mo;
            const auto &zz=spin?z.beta:z.alpha;
            const int o=zz.cols(),v=zz.rows();
            equal(raw.topLeftCorner(o,o),prime.topLeftCorner(o,o));
            equal(raw.bottomRightCorner(v,v),prime.bottomRightCorner(v,v));
            equal(raw.bottomLeftCorner(v,o),zz);
            equal(raw.topRightCorner(o,v),M::Zero(o,v));
            equal(sym.bottomLeftCorner(v,o),.5*zz);
            equal(sym.topRightCorner(o,v),.5*zz.transpose());
            near(raw.trace(),prime.trace()); near(sym.trace(),0);
            if(v && o) { near(raw(o,0),zz(0,0)); near(sym(o,0),.5*zz(0,0)); }
        }
        M l=M::Identity(n,n);
        for(int p=0;p<n;++p) { l(p,p)=1+.1*p; for(int q=0;q<p;++q) l(p,q)=.03*(1+p-q); }
        const M s=l*l.transpose();
        Orb c{frame(n,oa+va,.7,l),frame(n,ob+vb,1.9,l)};
        const auto ao=transform_udh_relaxed_density_to_ao(*d,c,s);
        check(ao.has_value()); if(!ao) return;
        near(dot(ao->spin.alpha,s),dp->alpha_mo.trace()); near(dot(ao->spin.beta,s),dp->beta_mo.trace());
        equal(ao->total,ao->spin.alpha+ao->spin.beta);
        // All independent symmetric operator basis directions, per spin and
        // common. Using different C_alpha/C_beta forbids premature MO sums.
        M f=M::Identity(n,n);
        double worst=0;
        for(int p=0;p<n;++p) for(int q=0;q<=p;++q)
        {
            M h=M::Zero(n,n); h(p,q)=h(q,p)=1;
            const double a=scalar(dp->alpha_mo,z.alpha,c.alpha,h);
            const double b=scalar(dp->beta_mo,z.beta,c.beta,h);
            near(dot(ao->spin.alpha,h),a); near(dot(ao->spin.beta,h),b);
            near(dot(ao->total,h),a+b);
            near(dot(c.alpha*d->raw_mo.alpha*c.alpha.transpose(),h),a);
            near(dot(c.beta*d->raw_mo.beta*c.beta.transpose(),h),b);
            for(double step : {1e-3,1e-4})
            {
                const double fd=(scalar(dp->alpha_mo,z.alpha,c.alpha,f+step*h)-
                    scalar(dp->alpha_mo,z.alpha,c.alpha,f-step*h)+
                    scalar(dp->beta_mo,z.beta,c.beta,f+step*h)-
                    scalar(dp->beta_mo,z.beta,c.beta,f-step*h))/(2*step);
                near(fd,dot(ao->total,h),2e-10);
                worst=std::max(worst,std::abs(fd-dot(ao->total,h)));
            }
        }
        const UDHPT2DPrime swapped{dp->beta_mo,dp->alpha_mo};
        const auto ds=build_udh_relaxed_difference_density(swapped,{z.beta,z.alpha});
        check(ds.has_value()); if(ds)
        {
            const auto as=transform_udh_relaxed_density_to_ao(*ds,{c.beta,c.alpha},s);
            check(as.has_value()); if(as) { equal(as->total,ao->total); equal(as->spin.alpha,ao->spin.beta); }
        }
        for(double scale : {0.0,-.6,2.3})
        {
            const auto scaled=build_udh_relaxed_difference_density({scale*dp->alpha_mo,scale*dp->beta_mo},
                {scale*z.alpha,scale*z.beta});
            check(scaled.has_value()); if(scaled)
            {
                equal(scaled->symmetric_mo.alpha,scale*d->symmetric_mo.alpha);
                equal(scaled->symmetric_mo.beta,scale*d->symmetric_mo.beta);
                const auto scaled_ao=transform_udh_relaxed_density_to_ao(*scaled,c,s);
                check(scaled_ao.has_value()); if(scaled_ao) equal(scaled_ao->total,scale*ao->total);
            }
        }
        // Outputs own their state; no coefficient/density/Z view escapes.
        const M frozen=d->raw_mo.alpha; z.alpha.setZero(); c.alpha.setZero();
        equal(d->raw_mo.alpha,frozen);
        std::cout<<"U5 spaces "<<oa<<'/'<<va<<" alpha "<<ob<<'/'<<vb<<" beta scalar FD="<<worst<<'\n';
    }
    void closed_shell()
    {
        constexpr int o=2,v=3,n=o+v;
        UDHPT2Amplitudes t{o,o,v,v,{},{},{}};
        HartreeFock::Correlation::RMP2Result r; r.n_occ=o; r.n_virt=v;
        const auto at=[](int i,int j,int a,int b)
        {
            const double g=std::cos(.3*(i+1)*(o+a+1))*std::cos(.3*(j+1)*(o+b+1));
            return g/(-3+.1*(i+j)-.2*(a+b));
        };
        for(int i=0;i<o;++i) for(int j=0;j<o;++j) for(int a=0;a<v;++a) for(int b=0;b<v;++b)
        { r.t2.push_back(at(i,j,a,b)); t.ab.push_back(at(i,j,a,b)); t.aa.push_back(at(i,j,a,b)-at(i,j,b,a)); }
        t.bb=t.aa;
        const auto dp=build_udh_pt2_dprime(t,.27);
        const auto rp=build_dh_pt2_amplitude_density(r,.27);
        check(dp && rp); if(!dp || !rp) return;
        equal(dp->alpha_mo+dp->beta_mo,rp->dprime_mo);
        auto op=make_udh_ks_response_operator(n,0,
            [=](const Eigen::Ref<const M>&)->std::expected<M,std::string>{return M::Zero(n,n);},{},
            [=](const UDHSpinMatrices&)->std::expected<UDHSpinMatrices,std::string>{return UDHSpinMatrices{M::Zero(n,n),M::Zero(n,n)};});
        check(op.has_value()); if(!op) return;
        UDHZVectorInputs in;
        in.mo_coeff={M::Identity(n,n),M::Identity(n,n)}; in.overlap_ao=M::Identity(n,n);
        in.fock_mo={M::Zero(n,n),M::Zero(n,n)};
        for(int a=0;a<v;++a) in.fock_mo.alpha(o+a,o+a)=1+.1*a;
        in.fock_mo.beta=in.fock_mo.alpha; in.nocc_alpha=in.nocc_beta=o; in.response=*op;
        const M zr=direction(v,o,.4);
        Orb rhs{M(v,o),M(v,o)};
        for(int a=0;a<v;++a) for(int i=0;i<o;++i) rhs.alpha(a,i)=-.5*(1+.1*a)*zr(a,i);
        rhs.beta=rhs.alpha;
        const auto solved=solve_udh_zvector(in,rhs);
        check(solved.has_value()); if(!solved) return;
        equal(solved->z_ai.alpha,.5*zr); equal(solved->z_ai.beta,.5*zr);
        const auto u=build_udh_relaxed_difference_density(*dp,solved->z_ai);
        const auto restricted=build_dh_relaxed_difference_density(*rp,zr);
        check(u && restricted); if(!u || !restricted) return;
        equal(u->raw_mo.alpha+u->raw_mo.beta,restricted->raw_mo);
        equal(u->symmetric_mo.alpha+u->symmetric_mo.beta,restricted->symmetric_mo);
        const auto ao=transform_udh_relaxed_density_to_ao(*u,in.mo_coeff,in.overlap_ao);
        check(ao.has_value()); if(ao) equal(ao->total,restricted->symmetric_mo);
        std::cout<<"U5 U1/U4 handoff and RKS half-spin Dprime/Z reduction PASS\n";
    }
    void errors()
    {
        UDHPT2DPrime dp{M::Zero(3,3),M::Zero(3,3)};
        Orb z{M::Ones(2,1),M::Ones(1,2)};
        const auto good=build_udh_relaxed_difference_density(dp,z);
        check(good.has_value()); if(!good) return;
        auto bad=dp; bad.alpha_mo(0,1)=1e-8; check(!build_udh_relaxed_difference_density(bad,z));
        bad=dp; bad.beta_mo(0,1)=1; check(!build_udh_relaxed_difference_density(bad,z));
        bad=dp; bad.alpha_mo(0,0)=std::numeric_limits<double>::infinity(); check(!build_udh_relaxed_difference_density(bad,z));
        check(!build_udh_relaxed_difference_density(dp,{M::Zero(1,1),z.beta}));
        auto zz=z; zz.beta(0,0)=std::numeric_limits<double>::quiet_NaN(); check(!build_udh_relaxed_difference_density(dp,zz));
        Orb c{M::Identity(3,3),M::Identity(3,3)}; M s=M::Identity(3,3);
        auto d=*good; d.symmetric_mo.beta(0,2)+=1; check(!transform_udh_relaxed_density_to_ao(d,c,s));
        d=*good; d.raw_mo.alpha(0,1)=1; check(!transform_udh_relaxed_density_to_ao(d,c,s));
        d=*good; d.nocc_alpha=-1; check(!transform_udh_relaxed_density_to_ao(d,c,s));
        auto cc=c; cc.beta(0,0)=2; check(!transform_udh_relaxed_density_to_ao(*good,cc,s));
        cc=c; cc.alpha.resize(3,2); check(!transform_udh_relaxed_density_to_ao(*good,cc,s));
        s(0,0)=-1; check(!transform_udh_relaxed_density_to_ao(*good,c,s));
        s=M::Identity(3,3); s(0,1)=.1; check(!transform_udh_relaxed_density_to_ao(*good,c,s));
        std::cout<<"U5 malformed density, metric, frame and Z rejection checks\n";
    }
}

int main()
{
    try
    {
        fixture(3,2,2,3); fixture(2,1,2,2); fixture(2,0,2,3); fixture(0,2,3,2);
        fixture(2,1,0,3); fixture(1,2,3,0); fixture(1,0,3,0); fixture(0,0,0,0);
        closed_shell(); errors();
    }
    catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; ok=false; }
    std::cout<<(ok?"PASS":"FAIL")<<" U5 Eq28 raw/symmetric relaxed spin densities\n";
    return ok?0:1;
}
