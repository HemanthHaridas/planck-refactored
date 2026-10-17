#include "dft/udh_overlap_density.h"
#include "dft/dh_pt2_gradient.h"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <source_location>
#include <stdexcept>

namespace
{
    using namespace DFT::Gradient;
    using M=Eigen::MatrixXd;
    using Orb=UDHOrbitalMatrices;
    using AO=UDHSpinMatrices;
    bool ok=true;
    void check(bool pass,std::source_location loc=std::source_location::current())
    { if(!pass) { ok=false; std::cerr<<"FAIL U6 line "<<loc.line()<<'\n'; } }
    double dot(const M &a,const M &b) { return (a.array()*b.array()).sum(); }
    double maxabs(const M &x) { return x.size()?x.cwiseAbs().maxCoeff():0; }
    void near(double a,double b,double tol=2e-9,std::source_location loc=std::source_location::current())
    { check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<tol*(1+std::abs(b)),loc); }
    void equal(const M &a,const M &b,double tol=2e-11)
    { check(a.rows()==b.rows() && a.cols()==b.cols()); if(a.rows()==b.rows() && a.cols()==b.cols()) near(maxabs(a-b),0,tol); }
    std::size_t idx(int i,int j,int a,int b,int oj,int va,int vb)
    { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
    M inverse_root(const M &s)
    {
        if(!s.rows()) return M(0,0);
        Eigen::SelfAdjointEigenSolver<M> eig(s);
        if(eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff()<=0) throw std::runtime_error("oracle metric failed");
        return eig.eigenvectors()*eig.eigenvalues().array().rsqrt().matrix().asDiagonal()*eig.eigenvectors().transpose();
    }
    // Exact metric-preserving occupied-first orthogonalization. At x=0 its
    // connection is Uoo=Uvv=-S'/2, Uov=-S'ov, Uvo=0. No W/U6 call here.
    M transport(const M &c,int o,const M &s)
    {
        const auto v=c.cols()-o;
        M co=c.leftCols(o)*inverse_root(c.leftCols(o).transpose()*s*c.leftCols(o));
        M cv=c.rightCols(v)-co*(co.transpose()*s*c.rightCols(v));
        cv=cv*inverse_root(cv.transpose()*s*cv);
        M out(c.rows(),c.cols()); out.leftCols(o)=co; out.rightCols(v)=cv;
        return out;
    }
    struct Model
    {
        M a,b,xa,xb;
        double eri(int p,int q,int r,int s) const { return .12*a(p,q)*a(r,s)+.08*b(p,q)*b(r,s); }
        // Independent scalar-side physical J/K loops. No U2 callback or W.
        std::array<AO,4> channels(const AO &d) const
        {
            const int n=a.rows();
            std::array<AO,4> out;
            for(auto &q:out) q={M::Zero(n,n),M::Zero(n,n)};
            for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            {
                out[0].alpha(p,q)+=eri(p,q,r,s)*(d.alpha(r,s)+d.beta(r,s));
                out[1].alpha(p,q)-=.53*eri(p,r,q,s)*d.alpha(r,s);
                out[1].beta(p,q)-=.53*eri(p,r,q,s)*d.beta(r,s);
            }
            out[0].beta=out[0].alpha;
            out[2]={.14*dot(xa,d.alpha)*xa,.03*dot(xa,d.alpha)*xb};
            out[3]={.03*dot(xb,d.beta)*xa,.14*dot(xb,d.beta)*xb};
            return out;
        }
        UDHKSResponseOperator response() const
        {
            auto result=make_udh_ks_response_operator(a.rows(),.53,
                [m=*this](const Eigen::Ref<const M> &d)->std::expected<M,std::string>
                { return (.12*dot(m.a,d)*m.a+.08*dot(m.b,d)*m.b).eval(); },
                [m=*this](const Eigen::Ref<const M> &d)->std::expected<M,std::string>
                { return (.12*m.a*d*m.a+.08*m.b*d*m.b).eval(); },
                [m=*this](const AO &d)->std::expected<AO,std::string>
                { return AO{(.14*dot(m.xa,d.alpha)+.03*dot(m.xb,d.beta))*m.xa,
                            (.03*dot(m.xa,d.alpha)+.14*dot(m.xb,d.beta))*m.xb}; });
            if(!result) throw std::runtime_error(result.error());
            return *result;
        }
    };
    struct Fixture
    {
        UDHPT2Amplitudes t;
        Orb c,f,z;
        M s;
        Model model;
    };
    Fixture make(int oa,int ob,int va,int vb)
    {
        Fixture x;
        const int n=std::max({oa+va,ob+vb,1});
        x.s=M::Identity(n,n);
        for(int i=0;i<n;++i) x.s(i,i)=1+.1*i;
        x.model.a=M(n,n); x.model.b=M(n,n); x.model.xa=M(n,n); x.model.xb=M(n,n);
        for(int p=0;p<n;++p) for(int q=0;q<n;++q)
        {
            x.model.a(p,q)=.3*std::cos(.21*(p+1)*(q+1)); x.model.b(p,q)=.2*std::sin(.4*(p+q+1));
            x.model.xa(p,q)=.13*std::cos(p+q+.2); x.model.xb(p,q)=.17*std::sin(p+q+.8);
        }
        for(int spin=0;spin<2;++spin)
        {
            const int o=spin?ob:oa,v=spin?vb:va,m=o+v;
            M trial=M::Identity(n,n);
            for(int p=0;p<n;++p) for(int q=0;q<n;++q) trial(p,q)+=.11*std::cos(.3+p+2*q+spin);
            const M q=trial.householderQr().householderQ()*M::Identity(n,n);
            M c=inverse_root(x.s)*q;
            (spin?x.c.beta:x.c.alpha)=c.leftCols(m);
            M f=M::Zero(m,m),z(v,o);
            for(int p=0;p<m;++p) f(p,p)=p<o?-1+.17*p:.3+.2*(p-o)+.07*spin;
            for(int a=0;a<v;++a) for(int i=0;i<o;++i) z(a,i)=.05*std::sin(.4+a+2*i+spin);
            (spin?x.f.beta:x.f.alpha)=f; (spin?x.z.beta:x.z.alpha)=z;
        }
        x.t={oa,ob,va,vb,{},{},{}};
        const auto same=[](int o,int v)
        {
            std::vector<double> t(static_cast<std::size_t>(o)*o*v*v);
            for(int i=0;i<o;++i) for(int j=i+1;j<o;++j) for(int a=0;a<v;++a) for(int b=a+1;b<v;++b)
            {
                const double value=.025*std::sin(1+i+2*j+3*a+5*b);
                t[idx(i,j,a,b,o,v,v)]=t[idx(j,i,b,a,o,v,v)]=value;
                t[idx(j,i,a,b,o,v,v)]=t[idx(i,j,b,a,o,v,v)]=-value;
            }
            return t;
        };
        x.t.aa=same(oa,va); x.t.bb=same(ob,vb); x.t.ab.resize(static_cast<std::size_t>(oa)*ob*va*vb);
        for(std::size_t k=0;k<x.t.ab.size();++k) x.t.ab[k]=.03*std::cos(.27*(k+1));
        return x;
    }
    UDHFullMOIntegrals integrals(const Fixture &x,const Orb &c)
    {
        UDHFullMOIntegrals g; g.n_alpha=c.alpha.cols(); g.n_beta=c.beta.cols();
        // Transform fixed AO rank-two ERIs; no amplitude/G/W expression.
        std::array<M,2> aa{c.alpha.transpose()*x.model.a*c.alpha,c.beta.transpose()*x.model.a*c.beta};
        std::array<M,2> bb{c.alpha.transpose()*x.model.b*c.alpha,c.beta.transpose()*x.model.b*c.beta};
        for(int sector=0;sector<3;++sector)
        {
            int s=sector==2,t=sector!=0;
            auto &out=sector==0?g.aa:sector==1?g.ab:g.bb;
            for(int p=0;p<aa[s].rows();++p) for(int q=0;q<aa[s].rows();++q)
                for(int r=0;r<aa[t].rows();++r) for(int u=0;u<aa[t].rows();++u)
                    out.push_back(.12*aa[s](p,q)*aa[t](r,u)+.08*bb[s](p,q)*bb[t](r,u));
        }
        return g;
    }
    // 13 independent scalar ledgers: aa/ab/bb pair, D' coefficient,
    // four D' induced-Fock channels, Z coefficient, four Z induced channels.
    std::array<double,13> scalar(const Fixture &x,const UDHPT2DPrime &dp,const Orb &cx)
    {
        std::array<double,13> out{};
        const auto g=integrals(x,cx);
        for(int sector=0;sector<3;++sector)
        {
            const int os=sector==2?x.t.nocc_beta:x.t.nocc_alpha,ot=sector==0?x.t.nocc_alpha:x.t.nocc_beta;
            const int vs=sector==2?x.t.nvirt_beta:x.t.nvirt_alpha,vt=sector==0?x.t.nvirt_alpha:x.t.nvirt_beta;
            const int ns=os+vs,nt=ot+vt;
            const auto &t=sector==0?x.t.aa:sector==1?x.t.ab:x.t.bb;
            const auto &gg=sector==0?g.aa:sector==1?g.ab:g.bb;
            for(int i=0;i<os;++i) for(int j=0;j<ot;++j) for(int a=0;a<vs;++a) for(int b=0;b<vt;++b)
                out[sector]+=.27*(sector==1?2:1)*t[idx(i,j,a,b,ot,vs,vt)]*
                    gg[((static_cast<std::size_t>(i)*ns+os+a)*nt+j)*nt+ot+b];
        }
        const int oa=x.t.nocc_alpha,ob=x.t.nocc_beta;
        AO delta{cx.alpha.leftCols(oa)*cx.alpha.leftCols(oa).transpose()-x.c.alpha.leftCols(oa)*x.c.alpha.leftCols(oa).transpose(),
                 cx.beta.leftCols(ob)*cx.beta.leftCols(ob).transpose()-x.c.beta.leftCols(ob)*x.c.beta.leftCols(ob).transpose()};
        const auto response=x.model.channels(delta);
        for(int spin=0;spin<2;++spin)
        {
            const auto &c=spin?cx.beta:cx.alpha,&c0=spin?x.c.beta:x.c.alpha,&f=spin?x.f.beta:x.f.alpha;
            const auto &d=spin?dp.beta_mo:dp.alpha_mo,&z=spin?x.z.beta:x.z.alpha;
            const M f0=x.s*c0*f*c0.transpose()*x.s;
            const M moved=c.transpose()*f0*c;
            out[3]+=dot(d,moved); out[8]+=dot(z,moved.bottomLeftCorner(z.rows(),z.cols()));
            for(int channel=0;channel<4;++channel)
            {
                const M induced=c.transpose()*(spin?response[channel].beta:response[channel].alpha)*c;
                // Under the joint adjoint, a source-D spin in K[D]
                // corresponds to a target-F spin in D:K[delta P]. Each
                // individual off-diagonal XC source block is NOT self-adjoint.
                const int dp_index=channel<2?4+channel:6+spin;
                const int z_index=channel<2?9+channel:11+spin;
                out[dp_index]+=dot(d,induced);
                out[z_index]+=dot(z,induced.bottomLeftCorner(z.rows(),z.cols()));
            }
        }
        return out;
    }
    std::array<const Orb*,13> channels(const UDHOverlapDensity &w)
    { return {&w.pair_aa,&w.pair_ab,&w.pair_bb,&w.dprime_orbital,&w.response_dprime.coulomb,
        &w.response_dprime.exchange,&w.response_dprime.xc_from_alpha,&w.response_dprime.xc_from_beta,
        &w.z_orbital,&w.response_z.coulomb,&w.response_z.exchange,&w.response_z.xc_from_alpha,&w.response_z.xc_from_beta}; }
    void audit(Fixture x)
    {
        const auto dp=build_udh_pt2_dprime(x.t,.27);
        const auto pair=build_udh_pair_orbital_gradient(x.t,integrals(x,x.c),.27);
        check(dp && pair); if(!dp || !pair) return;
        const auto d=build_udh_relaxed_difference_density(*dp,x.z);
        check(d.has_value()); if(!d) return;
        const auto w=build_udh_overlap_density(*pair,*d,x.c,x.f,x.s,x.model.response());
        check(w.has_value()); if(!w) { std::cerr<<w.error()<<'\n'; return; }
        const auto ledger=channels(*w);
        const auto zero_pair=build_udh_pair_orbital_gradient(x.t,integrals(x,x.c),0.0);
        const auto zero_dp=build_udh_pt2_dprime(x.t,0.0);
        check(zero_pair && zero_dp);
        if(zero_pair && zero_dp)
        {
            const auto zero_d=build_udh_relaxed_difference_density(*zero_dp,
                {M::Zero(x.z.alpha.rows(),x.z.alpha.cols()),M::Zero(x.z.beta.rows(),x.z.beta.cols())});
            check(zero_d.has_value()); if(zero_d)
            {
                const auto zero_w=build_udh_overlap_density(*zero_pair,*zero_d,x.c,x.f,x.s,x.model.response());
                check(zero_w.has_value()); if(zero_w) near(maxabs(zero_w->total_ao),0,1e-14);
            }
        }
        auto swapped=x;
        std::swap(swapped.c.alpha,swapped.c.beta); std::swap(swapped.f.alpha,swapped.f.beta);
        std::swap(swapped.z.alpha,swapped.z.beta); std::swap(swapped.model.xa,swapped.model.xb);
        swapped.t={x.t.nocc_beta,x.t.nocc_alpha,x.t.nvirt_beta,x.t.nvirt_alpha,x.t.bb,{},x.t.aa};
        for(int i=0;i<swapped.t.nocc_alpha;++i) for(int j=0;j<swapped.t.nocc_beta;++j)
            for(int a=0;a<swapped.t.nvirt_alpha;++a) for(int b=0;b<swapped.t.nvirt_beta;++b)
                swapped.t.ab.push_back(x.t.ab[idx(j,i,b,a,x.t.nocc_beta,x.t.nvirt_alpha,x.t.nvirt_beta)]);
        const auto sdp=build_udh_pt2_dprime(swapped.t,.27);
        const auto spair=build_udh_pair_orbital_gradient(swapped.t,integrals(swapped,swapped.c),.27);
        check(sdp && spair);
        if(sdp && spair)
        {
            const auto sd=build_udh_relaxed_difference_density(*sdp,swapped.z);
            check(sd.has_value()); if(sd)
            {
                const auto sw=build_udh_overlap_density(*spair,*sd,swapped.c,swapped.f,swapped.s,swapped.model.response());
                check(sw.has_value()); if(sw)
                { equal(sw->total_ao,w->total_ao); equal(sw->raw_mo.alpha,w->raw_mo.beta); }
            }
        }
        std::array<double,13> worst{};
        const int oa=x.t.nocc_alpha,ob=x.t.nocc_beta;
        // Each spin's oo/vv/ov basis perturbation defines ONE AO S'. The
        // other spin sees that same S', not an independently chosen MO S'.
        for(int anchor=0;anchor<2;++anchor)
        {
            const auto &ca=anchor?x.c.beta:x.c.alpha;
            for(int p=0;p<ca.cols();++p) for(int q=0;q<=p;++q)
            {
                M b=M::Zero(ca.cols(),ca.cols()); b(p,q)=b(q,p)=.2;
                const M ds=x.s*ca*b*ca.transpose()*x.s;
                const Orb sm{x.c.alpha.transpose()*ds*x.c.alpha,x.c.beta.transpose()*ds*x.c.beta};
                double analytic=0;
                for(const auto *block:ledger) analytic+=dot(block->alpha,sm.alpha)+dot(block->beta,sm.beta);
                near(analytic,dot(w->total_ao,ds),2e-11);
                near(analytic,dot(w->raw_mo.alpha,sm.alpha)+dot(w->raw_mo.beta,sm.beta),2e-11);
                for(double step : {1e-3,2e-4})
                {
                    const Orb plus{transport(x.c.alpha,oa,x.s+step*ds),transport(x.c.beta,ob,x.s+step*ds)};
                    const Orb minus{transport(x.c.alpha,oa,x.s-step*ds),transport(x.c.beta,ob,x.s-step*ds)};
                    equal(plus.alpha.transpose()*(x.s+step*ds)*plus.alpha,M::Identity(plus.alpha.cols(),plus.alpha.cols()));
                    equal(plus.beta.transpose()*(x.s+step*ds)*plus.beta,M::Identity(plus.beta.cols(),plus.beta.cols()));
                    const auto ep=scalar(x,*dp,plus),em=scalar(x,*dp,minus);
                    double fd_total=0;
                    for(int k=0;k<13;++k)
                    {
                        const double fd=(ep[k]-em[k])/(2*step);
                        const double ref=dot(ledger[k]->alpha,sm.alpha)+dot(ledger[k]->beta,sm.beta);
                        worst[k]=std::max(worst[k],std::abs(fd-ref)); near(fd,ref,2e-8);
                        fd_total+=fd;
                    }
                    near(fd_total,analytic,4e-8);
                }
            }
        }
        std::cout<<"U6 o/v "<<oa<<'/'<<x.t.nvirt_alpha<<" alpha "<<ob<<'/'<<x.t.nvirt_beta<<" beta FD errors:";
        for(double e:worst) std::cout<<' '<<e;
        std::cout<<'\n';
        // Orientation: pair has no vo, energy-times-Z has only vo.
        equal(w->pair_ab.alpha.bottomLeftCorner(x.t.nvirt_alpha,oa),M::Zero(x.t.nvirt_alpha,oa));
        equal(w->z_orbital.alpha.topRightCorner(oa,x.t.nvirt_alpha),M::Zero(oa,x.t.nvirt_alpha));
        auto failed=x.model.response();
        failed.xc=[](const AO&)->std::expected<AO,std::string>{return std::unexpected("U6 injected XC failure");};
        auto failure=build_udh_overlap_density(*pair,*d,x.c,x.f,x.s,failed);
        check(!failure && failure.error().find("U6 injected XC failure")!=std::string::npos);
        failed=x.model.response(); int calls=0;
        const auto j=failed.coulomb;
        failed.coulomb=[&](const Eigen::Ref<const M> &q)->std::expected<M,std::string>
        { if(++calls==2) return std::unexpected("U6 injected Z-response failure"); return j(q); };
        failure=build_udh_overlap_density(*pair,*d,x.c,x.f,x.s,failed);
        check(!failure && calls==2 && failure.error().find("U6 injected Z-response failure")!=std::string::npos);
        auto bad=*pair; if(bad.g.alpha.size())
        { bad.g.alpha(0,0)+=1; check(!build_udh_overlap_density(bad,*d,x.c,x.f,x.s,x.model.response())); }
        auto f=x.f; if(f.alpha.rows()>1)
        { f.alpha(0,1)=.01; check(!build_udh_overlap_density(*pair,*d,x.c,f,x.s,x.model.response())); }
    }
    void pair_orientation()
    {
        Orb g{M(4,4),M(3,3)};
        for(auto *m:{&g.alpha,&g.beta}) for(int p=0;p<m->rows();++p) for(int q=0;q<m->cols();++q) (*m)(p,q)=.1+p+3*q;
        auto w=build_udh_pair_overlap_density(g,2,1); check(w.has_value()); if(!w) return;
        equal(w->alpha.topLeftCorner(2,2),-.5*g.alpha.topLeftCorner(2,2));
        equal(w->alpha.bottomRightCorner(2,2),-.5*g.alpha.bottomRightCorner(2,2));
        equal(w->alpha.topRightCorner(2,2),-g.alpha.topRightCorner(2,2));
        equal(w->alpha.bottomLeftCorner(2,2),M::Zero(2,2));
        check(!build_udh_pair_overlap_density(g,-1,1));
        g.beta(0,0)=std::numeric_limits<double>::quiet_NaN(); check(!build_udh_pair_overlap_density(g,2,1));
    }
    void closed_shell()
    {
        auto x=make(2,2,2,2);
        x.c.beta=x.c.alpha; x.f.beta=x.f.alpha; x.z.beta=x.z.alpha; x.model.xb=x.model.xa;
        const auto g=integrals(x,x.c);
        constexpr int o=2,v=2,n=4;
        HartreeFock::Correlation::RMP2Result r; r.n_occ=o; r.n_virt=v;
        const auto at=[&](int i,int j,int a,int b)
        {
            return g.aa[((i*n+o+a)*n+j)*n+o+b]/
                (x.f.alpha(i,i)+x.f.alpha(j,j)-x.f.alpha(o+a,o+a)-x.f.alpha(o+b,o+b));
        };
        x.t.aa.clear(); x.t.ab.clear();
        for(int i=0;i<o;++i) for(int j=0;j<o;++j) for(int a=0;a<v;++a) for(int b=0;b<v;++b)
        { r.t2.push_back(at(i,j,a,b)); x.t.ab.push_back(at(i,j,a,b)); x.t.aa.push_back(at(i,j,a,b)-at(i,j,b,a)); }
        x.t.bb=x.t.aa;
        const auto dp=build_udh_pt2_dprime(x.t,.27);
        const auto pair=build_udh_pair_orbital_gradient(x.t,g,.27);
        const auto rp=build_dh_pt2_amplitude_density(r,.27);
        check(dp && pair && rp); if(!dp || !pair || !rp) return;
        equal(dp->alpha_mo+dp->beta_mo,rp->dprime_mo);
        const auto d=build_udh_relaxed_difference_density(*dp,x.z);
        const auto rd=build_dh_relaxed_difference_density(*rp,2*x.z.alpha);
        check(d && rd); if(!d || !rd) return;
        const auto op=x.model.response();
        const auto w=build_udh_overlap_density(*pair,*d,x.c,x.f,x.s,op);
        const auto rop=make_dh_eq41_response_operator(.53,op.coulomb,op.exchange,
            [h=x.model.xa](const Eigen::Ref<const M> &q)->std::expected<M,std::string>{return (.085*dot(h,q)*h).eval();});
        check(w && rop); if(!w || !rop) return;
        const Eigen::VectorXd eps=x.f.alpha.diagonal();
        auto diagonal=build_dh_eq42_43_energy_weighted_density(*rd,r,*rp,x.c.alpha,eps,g.aa,*rop,false);
        auto off=build_dh_eq44_45_energy_weighted_density(*rd,r,*rp,eps,g.aa,false);
        const auto metric=build_dh_eq47_pair_metric_overlap_density(r,*rp,g.aa);
        check(diagonal && off && metric); if(!diagonal || !off || !metric) return;
        diagonal->w_oo+=metric->w_oo; diagonal->w_vv+=metric->w_vv; off->w_ia=metric->w_ia;
        const auto rw=build_dh_overlap_density_adapter(*diagonal,*off);
        check(rw.has_value()); if(!rw) return;
        equal(w->raw_mo.alpha+w->raw_mo.beta,rw->raw_mo);
        equal(w->total_ao,x.c.alpha*rw->symmetric_mo*x.c.alpha.transpose());
        std::cout<<"U6 closed-shell raw W/AO reduction to RKS Eqs42-45\n";
        audit(x);
    }
}
int main()
{
    try
    {
        pair_orientation(); audit(make(2,1,2,3)); audit(make(1,2,2,2));
        audit(make(2,0,2,3)); audit(make(2,1,0,3)); audit(make(1,0,3,0)); audit(make(0,0,0,0));
        closed_shell();
    }
    catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; ok=false; }
    std::cout<<(ok?"PASS":"FAIL")<<" U6 fixed-AO metric pair/Dprime/Z overlap ledgers\n";
    return ok?0:1;
}
