// U9 common-functional integration oracle. Manufactured unequal-spin AO model,
// NOT a molecular SCF/gradient acceptance test. No production switches.
#include "dft/udh_two_particle_density.h"
#include "dft/udh_zvector.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
using namespace DFT::Gradient;
using M=Eigen::MatrixXd;
using V=Eigen::VectorXd;
using Orb=UDHOrbitalMatrices;
using AO=UDHSpinMatrices;
constexpr int n=5,oa=3,ob=2;
constexpr double scale=.27,ax=.53;
bool ok=true;
void check(bool pass,std::source_location loc=std::source_location::current())
{ if(!pass) { ok=false; std::cerr<<"FAIL U9 line "<<loc.line()<<'\n'; } }
double dot(const M &a,const M &b) { return a.cwiseProduct(b).sum(); }
double maxabs(const M &m) { return m.size()?m.cwiseAbs().maxCoeff():0.; }
void near(double a,double b,double tol=2e-9,std::source_location loc=std::source_location::current())
{
    if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tol*(1+std::abs(b)))
    { check(false,loc); std::cerr<<"  actual="<<a<<" expected="<<b<<" tolerance="<<tol<<'\n'; }
}
template<class T> T require(std::expected<T,std::string> value)
{ if(!value) throw std::runtime_error(value.error()); return std::move(*value); }
const M &spin(const Orb &m,int s) { return s?m.beta:m.alpha; }
const M &spin(const AO &m,int s) { return s?m.beta:m.alpha; }
std::size_t ti(int i,int j,int a,int b,int oj,int va,int vb)
{ return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
std::size_t gi(int p,int q,int r,int s)
{ return ((static_cast<std::size_t>(p)*n+q)*n+r)*n+s; }
M inverse_root(const M &s)
{
    Eigen::SelfAdjointEigenSolver<M> e(s);
    if(e.info()!=Eigen::Success || e.eigenvalues().minCoeff()<=0)
        throw std::runtime_error("U9 non-positive metric");
    return e.eigenvectors()*e.eigenvalues().array().rsqrt().matrix().asDiagonal()*e.eigenvectors().transpose();
}
// Occupied-first common metric gauge: Uoo=Uvv=-s/2, Uov=-sov, Uvo=0.
M transport(const M &c,int o,const M &s)
{
    M co=c.leftCols(o)*inverse_root(c.leftCols(o).transpose()*s*c.leftCols(o));
    M cv=c.rightCols(n-o)-co*(co.transpose()*s*c.rightCols(n-o));
    cv=cv*inverse_root(cv.transpose()*s*cv);
    M out(n,n); out.leftCols(o)=co; out.rightCols(n-o)=cv; return out;
}
Orb rotate(const Orb &c,const Orb &u,double step)
{
    Orb out;
    for(int s=0;s<2;++s)
    {
        int o=s?ob:oa; const M &x=spin(u,s);
        M generator=M::Zero(n,n);
        generator.bottomLeftCorner(n-o,o)=x;
        generator.topRightCorner(o,n-o)=-x.transpose();
        (s?out.beta:out.alpha)=spin(c,s)*(step*generator).exp();
    }
    return out;
}
AO density(const Orb &c)
{ return {c.alpha.leftCols(oa)*c.alpha.leftCols(oa).transpose(),
          c.beta.leftCols(ob)*c.beta.leftCols(ob).transpose()}; }
Orb zero_vo() { return {M::Zero(n-oa,oa),M::Zero(n-ob,ob)}; }
V pack(const Orb &q)
{
    V v((n-oa)*oa+(n-ob)*ob); int k=0;
    for(int s=0;s<2;++s) for(int a=0;a<spin(q,s).rows();++a)
        for(int i=0;i<spin(q,s).cols();++i) v(k++)=spin(q,s)(a,i);
    return v;
}
Orb unpack(const V &v)
{
    Orb q=zero_vo(); int k=0;
    for(M *m:{&q.alpha,&q.beta}) for(int a=0;a<m->rows();++a)
        for(int i=0;i<m->cols();++i) (*m)(a,i)=v(k++);
    return q;
}
struct Model
{
    std::array<M,3> factors;
    M xa,xb;
    AO h;
    M j(const M &d) const
    { M v=M::Zero(n,n); for(const auto &b:factors) v+=dot(b,d)*b; return v; }
    M k(const M &d) const
    { M v=M::Zero(n,n); for(const auto &b:factors) v+=b*d*b; return v; }
    // One nonlinear spin-coupled functional, used for both F and its kernel:
    // Exc=.05 za^3+zb^3/30+.13 za zb+.015 za^2 zb, z_sigma=X_sigma:P_sigma.
    AO xc(const AO &p) const
    {
        double a=dot(xa,p.alpha),b=dot(xb,p.beta);
        return {(.15*a*a+.13*b+.03*a*b)*xa,(.10*b*b+.13*a+.015*a*a)*xb};
    }
    std::array<AO,5> fock_channels(const AO &p) const
    {
        M jj=j(p.alpha+p.beta),zero=M::Zero(n,n);
        return {h,AO{jj,jj},AO{-ax*k(p.alpha),zero},AO{zero,-ax*k(p.beta)},xc(p)};
    }
    UDHKSResponseOperator response(const AO &p) const
    {
        double a=dot(xa,p.alpha),b=dot(xb,p.beta);
        return require(make_udh_ks_response_operator(n,ax,
            [m=*this](const Eigen::Ref<const M> &d)->std::expected<M,std::string>{return m.j(d);},
            [m=*this](const Eigen::Ref<const M> &d)->std::expected<M,std::string>{return m.k(d);},
            [m=*this,a,b](const AO &d)->std::expected<AO,std::string>
            {
                double da=dot(m.xa,d.alpha),db=dot(m.xb,d.beta);
                return AO{((.3*a+.03*b)*da+(.13+.03*a)*db)*m.xa,
                          ((.13+.03*a)*da+.2*b*db)*m.xb};
            }));
    }
};
struct Direction
{
    M h,s,xa,xb;
    std::array<M,3> factors;
};
Model move(const Model &base,const Direction &d,double x)
{
    Model m=base; m.h.alpha+=x*d.h; m.h.beta+=x*d.h;
    m.xa+=x*d.xa; m.xb+=x*d.xb;
    for(int l=0;l<3;++l) m.factors[l]+=x*d.factors[l];
    return m;
}
// Independent antisymmetric amplitude coordinates (i<j,a<b for aa/bb;
// all i,j,a,b for ab). Each coordinate has the same spin-orbital norm four.
struct Entry { int sector,i,j,a,b; };
std::vector<Entry> basis()
{
    std::vector<Entry> e;
    for(int sector=0;sector<3;++sector)
    {
        int os=sector==2?ob:oa,ot=sector==0?oa:ob,vs=n-os,vt=n-ot;
        for(int i=0;i<os;++i) for(int j=0;j<ot;++j)
            for(int a=0;a<vs;++a) for(int b=0;b<vt;++b)
                if(sector==1 || (i<j && a<b)) e.push_back({sector,i,j,a,b});
    }
    return e;
}
UDHPT2Amplitudes expand(const V &q)
{
    UDHPT2Amplitudes t{oa,ob,n-oa,n-ob,{},{},{}};
    t.aa.resize(oa*oa*(n-oa)*(n-oa)); t.ab.resize(oa*ob*(n-oa)*(n-ob));
    t.bb.resize(ob*ob*(n-ob)*(n-ob));
    auto entries=basis();
    for(int k=0;k<q.size();++k)
    {
        auto [sector,i,j,a,b]=entries[k];
        int os=sector==2?ob:oa,ot=sector==0?oa:ob,vs=n-os,vt=n-ot;
        auto &out=sector==0?t.aa:sector==1?t.ab:t.bb;
        out[ti(i,j,a,b,ot,vs,vt)]=q(k);
        if(sector!=1)
        {
            out[ti(j,i,a,b,ot,vs,vt)]=-q(k); out[ti(i,j,b,a,ot,vs,vt)]=-q(k);
            out[ti(j,i,b,a,ot,vs,vt)]=q(k);
        }
    }
    return t;
}
// Literal four-Fock-slot amplitude operator; no denominators, Dprime,
// production RHS, eigenvalue subtraction, or recanonicalization.
V amplitude_action(const Orb &f,const V &q)
{
    auto t=expand(q); auto entries=basis(); V out=V::Zero(q.size());
    for(int l=0;l<q.size();++l)
    {
        auto [sector,i,j,a,b]=entries[l];
        int s=sector==2,r=sector!=0,os=s?ob:oa,ot=r?ob:oa,vs=n-os,vt=n-ot;
        const M &fs=spin(f,s),&ft=spin(f,r);
        const auto &v=sector==0?t.aa:sector==1?t.ab:t.bb;
        auto at=[&](int ii,int jj,int aa,int bb){return v[ti(ii,jj,aa,bb,ot,vs,vt)];};
        for(int k=0;k<os;++k) out(l)+=fs(i,k)*at(k,j,a,b);
        for(int k=0;k<ot;++k) out(l)+=ft(j,k)*at(i,k,a,b);
        for(int e=0;e<vs;++e) out(l)-=fs(os+a,os+e)*at(i,j,e,b);
        for(int e=0;e<vt;++e) out(l)-=ft(ot+b,ot+e)*at(i,j,a,e);
    }
    return out;
}
M amplitude_matrix(const Orb &f)
{
    const int m=basis().size(); M a(m,m);
    for(int k=0;k<m;++k) a.col(k)=amplitude_action(f,V::Unit(m,k));
    near(maxabs(a-a.transpose()),0,2e-12); return a;
}
V solve(const M &a,const V &b)
{
    Eigen::FullPivLU<M> lu(a);
    if(lu.rank()!=a.rows() || lu.rcond()<1e-10) throw std::runtime_error("U9 singular oracle solve");
    V q=lu.solve(b); near(maxabs(a*q-b),0,2e-12); return q;
}
struct State
{
    Orb f;
    std::array<Orb,5> f_channels;
    UDHFullMOIntegrals full;
    UDHPT2DirectIntegrals direct;
    V b;
};
State state(const Model &m,const Orb &c)
{
    State out; out.f={M::Zero(n,n),M::Zero(n,n)};
    auto channels=m.fock_channels(density(c));
    for(int k=0;k<5;++k)
    {
        out.f_channels[k]={c.alpha.transpose()*channels[k].alpha*c.alpha,
                           c.beta.transpose()*channels[k].beta*c.beta};
        out.f.alpha+=out.f_channels[k].alpha; out.f.beta+=out.f_channels[k].beta;
    }
    out.full.n_alpha=n; out.full.n_beta=n;
    for(int sector=0;sector<3;++sector)
    {
        int s=sector==2,r=sector!=0,os=s?ob:oa,ot=r?ob:oa;
        auto &g=sector==0?out.full.aa:sector==1?out.full.ab:out.full.bb;
        g.resize(n*n*n*n);
        for(const auto &b:m.factors)
        {
            M l=spin(c,s).transpose()*b*spin(c,s),rr=spin(c,r).transpose()*b*spin(c,r);
            for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int u=0;u<n;++u) for(int v=0;v<n;++v)
                g[gi(p,q,u,v)]+=l(p,q)*rr(u,v);
        }
        auto &d=sector==0?out.direct.aa:sector==1?out.direct.ab:out.direct.bb;
        for(int i=0;i<os;++i) for(int j=0;j<ot;++j)
            for(int a=0;a<n-os;++a) for(int b=0;b<n-ot;++b) d.push_back(g[gi(i,os+a,j,ot+b)]);
    }
    auto entries=basis(); out.b=V(entries.size());
    for(int k=0;k<out.b.size();++k)
    {
        auto [sector,i,j,a,b]=entries[k]; int os=sector==2?ob:oa,ot=sector==0?oa:ob;
        const auto &g=sector==0?out.full.aa:sector==1?out.full.ab:out.full.bb;
        out.b(k)=g[gi(i,os+a,j,ot+b)];
        if(sector!=1) out.b(k)-=g[gi(i,os+b,j,ot+a)];
    }
    return out;
}
// Independent scalar ledger. First five entries: h,J,Ka,Kb,XC in D'F+zF.
// Last three: aa,ab,bb pair. Does not consume any production D,W,Gamma/RHS.
std::array<double,8> scalar(const State &s,const V &q,const Orb &z)
{
    std::array<double,8> e{};
    for(int k=0;k<5;++k)
    {
        e[k]=-scale*q.dot(amplitude_action(s.f_channels[k],q));
        e[k]+=dot(z.alpha,s.f_channels[k].alpha.bottomLeftCorner(n-oa,oa));
        e[k]+=dot(z.beta,s.f_channels[k].beta.bottomLeftCorner(n-ob,ob));
    }
    auto entries=basis();
    for(int k=0;k<q.size();++k) e[5+entries[k].sector]+=2*scale*q(k)*s.b(k);
    return e;
}
double total(const std::array<double,8> &e)
{ double sum=0; for(double v:e) sum+=v; return sum; }
Orb fvo(const State &s)
{ return {s.f.alpha.bottomLeftCorner(n-oa,oa),s.f.beta.bottomLeftCorner(n-ob,ob)}; }
struct Fixture { Model model; Orb c,f; M s; };
Fixture fixture()
{
    Fixture x; x.s=M::Identity(n,n);
    for(int p=0;p<n;++p) x.s(p,p)+=.08*p;
    M qa=M::Identity(n,n),qb=qa;
    qa.applyOnTheRight(0,4,Eigen::JacobiRotation<double>(std::cos(.31),std::sin(.31)));
    qb.applyOnTheRight(1,3,Eigen::JacobiRotation<double>(std::cos(.47),std::sin(.47)));
    x.c={inverse_root(x.s)*qa,inverse_root(x.s)*qb}; x.f={M::Zero(n,n),M::Zero(n,n)};
    for(int p=0;p<n;++p)
    {
        x.f.alpha(p,p)=p<oa?-1.4+.11*p:.4+.17*(p-oa);
        x.f.beta(p,p)=p<ob?-1.3+.09*p:.5+.13*(p-ob);
    }
    for(int l=0;l<3;++l)
    {
        x.model.factors[l]=M(n,n);
        for(int p=0;p<n;++p) for(int q=0;q<n;++q)
            x.model.factors[l](p,q)=.12*std::cos(.31*(l+1)*(p+1)*(q+1));
    }
    x.model.xa=.01*M::Ones(n,n); x.model.xb=.017*M::Ones(n,n);
    for(int p=0;p<n;++p) { x.model.xa(p,p)+=.11*(p+1); x.model.xb(p,p)+=.07*(n-p); }
    x.model.h={M::Zero(n,n),M::Zero(n,n)};
    auto potential=state(x.model,x.c).f;
    M ia=x.c.alpha.inverse(),ib=x.c.beta.inverse();
    // Manufactured spin-dependent fixed h, not a molecular Hamiltonian.
    x.model.h={ia.transpose()*(x.f.alpha-potential.alpha)*ia,
               ib.transpose()*(x.f.beta-potential.beta)*ib};
    return x;
}
Direction direction(int channel)
{
    Direction d{M::Zero(n,n),M::Zero(n,n),M::Zero(n,n),M::Zero(n,n),{}};
    for(auto &b:d.factors) b=M::Zero(n,n);
    for(int p=0;p<n;++p) for(int q=0;q<n;++q)
    {
        if(channel==0 || channel==4) d.h(p,q)=.13*std::cos(.4*(p+1)*(q+1));
        if(channel==1 || channel==4) for(int l=0;l<3;++l)
            d.factors[l](p,q)=.08*std::sin(.21*(l+1)*(p+q+1));
        if(channel==2 || channel==4)
        { d.xa(p,q)=.1*std::cos(.2*(p+q+1)); d.xb(p,q)=.07*std::sin(.3*(p+q+1)); }
        if(channel==3 || channel==4) d.s(p,q)=.11*std::cos(.27*(p+1)*(q+1));
    }
    return d;
}
Orb moved_coeff(const Fixture &x,const Direction &d,double h,const Orb &u)
{
    auto r=rotate(x.c,u,h);
    return {transport(r.alpha,oa,x.s+h*d.s),transport(r.beta,ob,x.s+h*d.s)};
}
AO xc_geometry(const Model &m,const Direction &d,const AO &p)
{
    double a=dot(m.xa,p.alpha),b=dot(m.xb,p.beta),da=dot(d.xa,p.alpha),db=dot(d.xb,p.beta);
    return {((.3*a+.03*b)*da+(.13+.03*a)*db)*m.xa+(.15*a*a+.13*b+.03*a*b)*d.xa,
            ((.13+.03*a)*da+.2*b*db)*m.xb+(.10*b*b+.13*a+.015*a*a)*d.xb};
}
DHEq33NonXCDerivatives derivatives(const Model &m,const Direction &d)
{
    DHEq33NonXCDerivatives out; out.hamiltonian_ao={d.h}; out.overlap_ao={d.s};
    std::vector<double> eri(n*n*n*n);
    for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
        for(int l=0;l<3;++l) eri[gi(p,q,r,s)]+=d.factors[l](p,q)*m.factors[l](r,s)+m.factors[l](p,q)*d.factors[l](r,s);
    out.eri_ao={std::move(eri)}; return out;
}
struct Contract { UDHTwoParticleDensity gamma; UDHOverlapDensity w; };
Contract contract(const Fixture &x,const V &q,const Orb &z)
{
    auto t=expand(q); auto dp=require(build_udh_pt2_dprime(t,scale));
    auto pair=require(build_udh_pair_orbital_gradient(t,state(x.model,x.c).full,scale));
    auto relaxed=require(build_udh_relaxed_difference_density(dp,z));
    return {require(build_udh_two_particle_density(t,scale,relaxed,x.c,x.s,ax)),
            require(build_udh_overlap_density(pair,relaxed,x.c,x.f,x.s,x.model.response(density(x.c))))};
}
std::array<double,9> analytic(const Fixture &x,const Direction &d,const Contract &c)
{
    auto g=require(build_udh_non_xc_gradient(c.gamma,c.w,derivatives(x.model,d)));
    auto vx=xc_geometry(x.model,d,density(x.c));
    return {g.one_electron(0),g.coulomb(0),g.exchange_alpha(0),g.exchange_beta(0),
            dot(c.gamma.relaxed_ao.spin.alpha,vx.alpha)+dot(c.gamma.relaxed_ao.spin.beta,vx.beta),
            g.pair_aa(0),g.pair_ab(0),g.pair_bb(0),g.overlap(0)};
}
double sum(const std::array<double,9> &e)
{ double v=0; for(double q:e) v+=q; return v; }

void amplitude_oracle(const Fixture &x)
{
    // Noncanonical oo/vv frames and a deliberately nonstationary amplitude.
    auto c=x.c;
    for(int s=0;s<2;++s)
    {
        int o=s?ob:oa; M r=M::Identity(n,n);
        r.applyOnTheRight(0,1,Eigen::JacobiRotation<double>(std::cos(.38),std::sin(.38)));
        r.applyOnTheRight(o,o+1,Eigen::JacobiRotation<double>(std::cos(.29),std::sin(.29)));
        (s?c.beta:c.alpha)=spin(x.c,s)*r;
    }
    auto base=state(x.model,c); M a=amplitude_matrix(base.f); V stationary=solve(a,base.b);
    check(std::abs(base.f.alpha(0,1))>1e-3);
    check(maxabs(a-M(a.diagonal().asDiagonal()))>1e-3);
    near(total(scalar(base,stationary,zero_vo())),scale*stationary.dot(base.b),2e-12);
    V q=1.17*stationary;
    for(int k=0;k<q.size();++k) q(k)+=.003*std::sin(k+.4);
    V residual=base.b-a*q;
    check(residual.norm()>1e-3);
    // A diagonal-denominator shortcut in these unchanged frames is invalid.
    check((a*(base.b.array()/a.diagonal().array()).matrix()-base.b).norm()>1e-5);
    auto typed=require(evaluate_udh_pt2_stationary_scalar(expand(q),base.direct,base.f.alpha,base.f.beta,scale));
    near(typed.total,total(scalar(base,q,zero_vo())),2e-12);
    double worst=0;
    for(int k=0;k<q.size();++k) for(double h:{1e-3,2e-4})
    {
        double fd=(total(scalar(base,q+h*V::Unit(q.size(),k),zero_vo()))-
                   total(scalar(base,q-h*V::Unit(q.size(),k),zero_vo())))/(2*h);
        near(fd,2*scale*residual(k),2e-11); worst=std::max(worst,std::abs(fd-2*scale*residual(k)));
    }
    Orb u=unpack(V::LinSpaced((n-oa)*oa+(n-ob)*ob,-.14,.19));
    Direction d=direction(4);
    // No metric perturbation here: isolates full nonmetric coefficient/geometry transfer.
    d.s.setZero();
    for(double h:{1e-3,2e-4})
    {
        auto plus=state(move(x.model,d,h),rotate(c,u,h));
        auto minus=state(move(x.model,d,-h),rotate(c,u,-h));
        M ap=amplitude_matrix(plus.f),am=amplitude_matrix(minus.f);
        M da=(ap-am)/(2*h); V db=(plus.b-minus.b)/(2*h);
        for(bool off:{false,true})
        {
            V center=off?q:stationary,r=base.b-a*center;
            V qp=solve(ap,plus.b-r),qm=solve(am,minus.b-r);
            V response=solve(a,V(db-da*center)),lambda=solve(M(a.transpose()),V(scale*base.b));
            near(maxabs((qp-qm)/(2*h)-response),0,2e-8);
            double explicit_energy=scale*center.dot(db);
            double response_energy=scale*response.dot(base.b);
            double multiplier=lambda.dot(db-da*center);
            near(maxabs(a.transpose()*lambda-scale*base.b),0,2e-12);
            near(response_energy,multiplier,2e-12);
            double fd=scale*(qp.dot(plus.b)-qm.dot(minus.b))/(2*h);
            near(fd,explicit_energy+multiplier,2e-9);
            double frozen_h=(total(scalar(plus,center,zero_vo()))-total(scalar(minus,center,zero_vo())))/(2*h);
            // Off shell H_T:T' is NOT zero; do not silently drop it.
            double live_h=(total(scalar(plus,qp,zero_vo()))-total(scalar(minus,qm,zero_vo())))/(2*h);
            near(live_h-frozen_h,2*scale*r.dot(response),2e-9);
            if(!off) near(fd,frozen_h,2e-9);
            std::cout<<"amplitude "<<(off?"off-shell":"stationary")<<" h="<<h
                     <<" residual="<<r.norm()<<" E_response="<<response_energy
                     <<" multiplier="<<multiplier<<" H_T:Tprime="<<2*scale*r.dot(response)
                     <<" transfer_error="<<fd-explicit_energy-multiplier<<'\n';
        }
    }
    std::cout<<"amplitude basis directions="<<q.size()<<" max FD error="<<worst<<'\n';
}

void integration(const Fixture &x)
{
    auto base=state(x.model,x.c); M amp=amplitude_matrix(base.f); V q=solve(amp,base.b);
    near(maxabs(base.f.alpha-x.f.alpha),0,2e-12); near(maxabs(base.f.beta-x.f.beta),0,2e-12);
    auto rhs=require(build_udh_orbital_rhs(expand(q),base.full,scale,x.c,x.s,x.f,x.model.response(density(x.c))));
    auto z=require(solve_udh_zvector({x.c,x.f,x.s,oa,ob,x.model.response(density(x.c))},rhs.total_ai));
    check(pack(rhs.total_ai).norm()>1e-6); check(pack(z.z_ai).norm()>1e-6);
    // All independent alpha/beta orbital directions; scalar FD, not analytic
    // RHS contractions, supplies the reference ell and KS Jacobian columns.
    double worst_rhs=0,worst_jacobian=0,worst_stationarity=0;
    const int m=pack(rhs.total_ai).size();
    V fd_rhs(m),fd_stationarity(m); M fd_jacobian(m,m);
    for(int k=0;k<m;++k) for(double h:{2e-4,1e-4})
    {
        auto u=unpack(V::Unit(m,k));
        auto plus=state(x.model,rotate(x.c,u,h)),minus=state(x.model,rotate(x.c,u,-h));
        double fd=(total(scalar(plus,q,zero_vo()))-total(scalar(minus,q,zero_vo())))/(2*h);
        double fd_l=(total(scalar(plus,q,z.z_ai))-total(scalar(minus,q,z.z_ai)))/(2*h);
        V column=(pack(fvo(plus))-pack(fvo(minus)))/(2*h);
        near(fd,pack(rhs.total_ai)(k),3e-9);
        near(maxabs(column-z.jacobian.col(k)),0,1e-7);
        near(fd_l,0,3e-9);
        worst_rhs=std::max(worst_rhs,std::abs(fd-pack(rhs.total_ai)(k)));
        worst_jacobian=std::max(worst_jacobian,maxabs(column-z.jacobian.col(k)));
        worst_stationarity=std::max(worst_stationarity,std::abs(fd_l));
        if(h==1e-4) { fd_rhs(k)=fd; fd_stationarity(k)=fd_l; fd_jacobian.col(k)=column; }
    }
    std::cout<<"orbital directions="<<m<<" RHS_FD="<<worst_rhs<<" Hessian_FD="<<worst_jacobian
             <<" L_stationarity="<<worst_stationarity<<" Z_residual="<<z.residual_max_abs<<'\n';
    std::cout<<"RHS analytic (alpha a-major, then beta):\n"<<pack(rhs.total_ai).transpose()
             <<"\nRHS scalar FD:\n"<<fd_rhs.transpose()
             <<"\nRHS error:\n"<<(fd_rhs-pack(rhs.total_ai)).transpose()
             <<"\nL orbital stationarity FD:\n"<<fd_stationarity.transpose()
             <<"\nKS Jacobian analytic:\n"<<z.jacobian
             <<"\nKS Jacobian FD error:\n"<<fd_jacobian-z.jacobian<<'\n';
    auto nonz=contract(x,q,zero_vo()),full=contract(x,q,z.z_ai);
    const std::array<const char*,5> names{"h","ERI","XC_geometry","metric","combined"};
    const std::array<const char*,9> labels{"h","J","Ka","Kb","XC-II","pair-aa","pair-ab","pair-bb","overlap"};
    for(int channel=0;channel<5;++channel)
    {
        auto d=direction(channel); auto an=analytic(x,d,nonz),af=analytic(x,d,full);
        for(int k=0;k<9;++k)
            std::cout<<names[channel]<<" analytic "<<labels[k]<<" nonZ="<<an[k]<<" Z="<<af[k]-an[k]<<'\n';
        for(double h:{1e-3,2e-4})
        {
            auto mp=move(x.model,d,h),mm=move(x.model,d,-h);
            // Fixed AO explicit and coefficient/metric derivatives separated.
            auto ep=state(mp,x.c),em=state(mm,x.c);
            auto cp=moved_coeff(x,d,h,zero_vo()),cm=moved_coeff(x,d,-h,zero_vo());
            auto plus=state(mp,cp),minus=state(mm,cm);
            auto enp=scalar(ep,q,zero_vo()),enm=scalar(em,q,zero_vo());
            auto efp=scalar(ep,q,z.z_ai),efm=scalar(em,q,z.z_ai);
            for(int k=0;k<8;++k)
            {
                double fdn=(enp[k]-enm[k])/(2*h),fdf=(efp[k]-efm[k])/(2*h);
                near(fdn,an[k],3e-9); near(fdf-fdn,af[k]-an[k],3e-9);
                std::cout<<names[channel]<<" h="<<h<<" explicit "<<labels[k]
                         <<" nonZ_error="<<fdn-an[k]<<" Z_error="<<fdf-fdn-af[k]+an[k]<<'\n';
            }
            double fdn=(total(scalar(plus,q,zero_vo()))-total(scalar(minus,q,zero_vo())))/(2*h);
            double fdf=(total(scalar(plus,q,z.z_ai))-total(scalar(minus,q,z.z_ai)))/(2*h);
            double explicit_n=(total(enp)-total(enm))/(2*h),explicit_f=(total(efp)-total(efm))/(2*h);
            near(fdn-explicit_n,an[8],3e-9);
            near(fdf-fdn-explicit_f+explicit_n,af[8]-an[8],3e-9);
            near(fdn,sum(an),3e-9); near(fdf-fdn,sum(af)-sum(an),3e-9);

            // Independent FD KS forcing. Solve the tangent of F_vo=0, then
            // follow this co-moving direction and re-solve the full amplitude
            // equations at BOTH endpoints. This is a first-order SCF path,
            // not an endpoint SCF or molecular total-energy FD.
            V forcing=(pack(fvo(plus))-pack(fvo(minus)))/(2*h);
            V u=solve(z.jacobian,V(-forcing));
            double nonz_response=pack(rhs.total_ai).dot(u);
            double z_explicit=pack(z.z_ai).dot(forcing);
            double z_response=pack(z.z_ai).dot(z.jacobian*u);
            near(nonz_response,z_explicit,2e-12); near(nonz_response+z_response,0,2e-12);
            auto livep=state(mp,moved_coeff(x,d,h,unpack(u)));
            auto livem=state(mm,moved_coeff(x,d,-h,unpack(u)));
            V qp=solve(amplitude_matrix(livep.f),livep.b),qm=solve(amplitude_matrix(livem.f),livem.b);
            double live_fd=scale*(qp.dot(livep.b)-qm.dot(livem.b))/(2*h);
            double live_frozen=(total(scalar(livep,q,zero_vo()))-total(scalar(livem,q,zero_vo())))/(2*h);
            double ks_tangent=maxabs((pack(fvo(livep))-pack(fvo(livem)))/(2*h));
            near(ks_tangent,0,2e-7); near(live_fd,live_frozen,3e-9);
            near(live_fd,sum(af),4e-9);
            std::cout<<names[channel]<<" h="<<h<<" stationary_nonZ="<<sum(an)<<" stationary_Z="<<sum(af)-sum(an)
                     <<" frozen_nonZ_error="<<fdn-sum(an)<<" frozen_Z_error="<<fdf-fdn-sum(af)+sum(an)
                     <<" orbital_nonZ="<<nonz_response<<" orbital_Z="<<z_response
                     <<" KS_tangent="<<ks_tangent<<" live_PT2_FD="<<live_fd
                     <<" amplitude_cancel_error="<<live_fd-live_frozen<<" Eq33_error="<<live_fd-sum(af)<<'\n';
        }
    }
}
}
int main()
{
    std::cout<<std::setprecision(14);
    try { auto x=fixture(); amplitude_oracle(x); integration(x); }
    catch(const std::exception &e) { std::cerr<<"FAIL U9 "<<e.what()<<'\n'; return 1; }
    std::cout<<(ok?"PASS":"FAIL")<<" U9 common stationary-transfer model (not molecular acceptance)\n";
    return ok?0:1;
}
