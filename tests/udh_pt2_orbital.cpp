// U3: literal coefficient FDs and an independent nonlinear stationary model.
// No SCF, Z solve, geometry derivative, or external chemistry code reference.
#include "dft/udh_pt2_orbital.h"
#include <Eigen/Jacobi>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>

namespace
{
    using namespace DFT::Gradient;
    using M=Eigen::MatrixXd;
    using Orb=UDHOrbitalMatrices;
    using AO=UDHSpinMatrices;
    bool ok=true;
    void check(bool pass,std::source_location at=std::source_location::current())
    { if (!pass) { std::cerr << "FAIL U3 line " << at.line() << '\n'; ok=false; } }
    double dot(const M &a,const M &b) { return a.cwiseProduct(b).sum(); }
    double dot(const AO &a,const AO &b) { return dot(a.alpha,b.alpha)+dot(a.beta,b.beta); }
    void near(double a,double b,double tol=2e-11,std::source_location at=std::source_location::current())
    { check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tol*(1+std::abs(b)),at); }
    void same(const M &a,const M &b,double tol=2e-11,std::source_location at=std::source_location::current())
    {
        check(a.rows()==b.rows() && a.cols()==b.cols(),at);
        if (a.rows()==b.rows() && a.cols()==b.cols() && a.size()) near((a-b).cwiseAbs().maxCoeff(),0,tol,at);
    }
    std::size_t ti(int i,int j,int a,int b,int oj,int va,int vb)
    { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
    std::size_t gi(int p,int q,int r,int s,int ns,int nt)
    { return ((static_cast<std::size_t>(p)*ns+q)*nt+r)*nt+s; }
    struct Model
    {
        std::vector<M> factors;
        M xa,xb;
        AO h;
        double ax=.53;
        M j(const M &q) const
        { M v=M::Zero(q.rows(),q.cols()); for (const auto &b:factors) v+=dot(b,q)*b; return v; }
        M k(const M &q) const
        { M v=M::Zero(q.rows(),q.cols()); for (const auto &b:factors) v+=b*q*b; return v; }
        // Independent potential from E=.3 za^3/6+.2 zb^3/6+.13 za zb+.03 za^2 zb/2.
        AO xc(const AO &p) const
        {
            const double a=dot(xa,p.alpha),b=dot(xb,p.beta);
            return {(.15*a*a+.13*b+.03*a*b)*xa,(.10*b*b+.13*a+.015*a*a)*xb};
        }
        AO fock(const AO &p) const
        {
            const auto v=xc(p); const M coulomb=j(p.alpha+p.beta);
            return {h.alpha+coulomb-ax*k(p.alpha)+v.alpha,h.beta+coulomb-ax*k(p.beta)+v.beta};
        }
        UDHKSResponseOperator response(const AO &ground) const
        {
            const auto fixed=*this;
            auto rawj=[fixed](const Eigen::Ref<const M> &q)->std::expected<M,std::string>{return fixed.j(q);};
            auto rawk=[fixed](const Eigen::Ref<const M> &q)->std::expected<M,std::string>{return fixed.k(q);};
            const double a=dot(xa,ground.alpha),b=dot(xb,ground.beta);
            auto kernel=[fixed,a,b](const AO &q)->std::expected<AO,std::string>
            {
                double da=dot(fixed.xa,q.alpha),db=dot(fixed.xb,q.beta);
                return AO{((.3*a+.03*b)*da+(.13+.03*a)*db)*fixed.xa,
                          ((.13+.03*a)*da+.2*b*db)*fixed.xb};
            };
            auto op=make_udh_ks_response_operator(xa.rows(),ax,rawj,rawk,kernel);
            if (!op) throw std::runtime_error(op.error());
            return *op;
        }
    };
    struct Fixture
    {
        UDHPT2Amplitudes t;
        UDHFullMOIntegrals g;
        Orb coeff,eps_matrix;
        M s;
        Model model;
    };
    AO density(const Fixture &f,const Orb &c)
    { return {c.alpha.leftCols(f.t.nocc_alpha)*c.alpha.leftCols(f.t.nocc_alpha).transpose(),
              c.beta.leftCols(f.t.nocc_beta)*c.beta.leftCols(f.t.nocc_beta).transpose()}; }
    Orb mo_fock(const Fixture &f,const Orb &c)
    {
        const auto ao=f.model.fock(density(f,c));
        return {c.alpha.transpose()*ao.alpha*c.alpha,c.beta.transpose()*ao.beta*c.beta};
    }
    UDHFullMOIntegrals integrals(const Model &model,const Orb &c)
    {
        int na=c.alpha.cols(),nb=c.beta.cols();
        UDHFullMOIntegrals out{na,nb,{},{},{}};
        for (int block=0;block<3;++block)
        {
            const auto &cs=block==2?c.beta:c.alpha,&ct=block==0?c.alpha:c.beta;
            auto &g=block==0?out.aa:(block==1?out.ab:out.bb);
            const int ns=cs.cols(),nt=ct.cols();
            g.assign(static_cast<std::size_t>(ns)*ns*nt*nt,0);
            for (const auto &b:model.factors)
            {
                const M left=cs.transpose()*b*cs,right=ct.transpose()*b*ct;
                for (int p=0;p<ns;++p) for (int q=0;q<ns;++q)
                    for (int r=0;r<nt;++r) for (int s=0;s<nt;++s)
                        g[gi(p,q,r,s,ns,nt)]+=left(p,q)*right(r,s);
            }
        }
        return out;
    }
    Fixture make(int oa=3,int ob=2,int n=5,bool closed=false)
    {
        Fixture f; f.t={oa,ob,n-oa,n-ob,{},{},{}};
        f.s=M::Identity(n,n); M root=f.s;
        for (int p=0;p<n;++p) { f.s(p,p)=1+.07*p; root(p,p)=1/std::sqrt(f.s(p,p)); }
        M qa=M::Identity(n,n),qb=qa;
        qa.applyOnTheRight(0,n-1,Eigen::JacobiRotation<double>(std::cos(.31),std::sin(.31)));
        qb.applyOnTheRight(1,n-1,Eigen::JacobiRotation<double>(std::cos(.47),std::sin(.47)));
        f.coeff={root*qa,root*(closed?qa:qb)};
        f.eps_matrix={M::Zero(n,n),M::Zero(n,n)};
        for (int p=0;p<n;++p)
        {
            f.eps_matrix.alpha(p,p)=p<oa?-1.4+.11*p:.4+.17*(p-oa);
            f.eps_matrix.beta(p,p)=closed?f.eps_matrix.alpha(p,p):(p<ob?-1.3+.09*p:.5+.13*(p-ob));
        }
        for (int l=0;l<4;++l)
        {
            M b(n,n);
            for (int p=0;p<n;++p) for (int q=0;q<n;++q) b(p,q)=.12*std::cos(.31*(l+1)*(p+1)*(q+1));
            f.model.factors.push_back(b);
        }
        f.model.xa=.01*M::Ones(n,n); f.model.xb=.017*M::Ones(n,n);
        for (int p=0;p<n;++p) { f.model.xa(p,p)+=.11*(p+1); f.model.xb(p,p)+=.07*(n-p); }
        f.model.h={M::Zero(n,n),M::Zero(n,n)};
        const auto potential=f.model.fock(density(f,f.coeff));
        const M ia=f.coeff.alpha.inverse(),ib=f.coeff.beta.inverse();
        // Fixed spin-dependent one-body matrices make this synthetic model
        // exactly canonical. It is an orbital algebra oracle, not a molecule.
        f.model.h={ia.transpose()*f.eps_matrix.alpha*ia-potential.alpha,
                   ib.transpose()*f.eps_matrix.beta*ib-potential.beta};
        f.g=integrals(f.model,f.coeff);
        for (int block=0;block<3;++block)
        {
            int os=block==2?ob:oa,ot=block==0?oa:ob,vs=n-os,vt=n-ot;
            auto &t=block==0?f.t.aa:(block==1?f.t.ab:f.t.bb);
            const auto &g=block==0?f.g.aa:(block==1?f.g.ab:f.g.bb);
            const auto &es=block==2?f.eps_matrix.beta:f.eps_matrix.alpha;
            const auto &et=block==0?f.eps_matrix.alpha:f.eps_matrix.beta;
            for (int i=0;i<os;++i) for (int j=0;j<ot;++j)
                for (int a=0;a<vs;++a) for (int b=0;b<vt;++b)
                {
                    double v=g[gi(i,os+a,j,ot+b,n,n)];
                    if (block!=1) v-=g[gi(i,os+b,j,ot+a,n,n)];
                    t.push_back(v/(es(i,i)+et(j,j)-es(os+a,os+a)-et(ot+b,ot+b)));
                }
        }
        return f;
    }

    // Pair scalar evaluated from common AO factors and FOUR independent
    // coefficient matrices. Does not use the stored full MO integrals or G.
    double scalar(const Fixture &f,int block,const std::array<M,4> &c,double scale=.27)
    {
        int os=block==2?f.t.nocc_beta:f.t.nocc_alpha,ot=block==0?f.t.nocc_alpha:f.t.nocc_beta;
        int vs=(block==2?f.t.nvirt_beta:f.t.nvirt_alpha),vt=block==0?f.t.nvirt_alpha:f.t.nvirt_beta;
        const auto &t=block==0?f.t.aa:(block==1?f.t.ab:f.t.bb);
        double out=0;
        for (const auto &factor:f.model.factors)
        {
            const M left=c[0].transpose()*factor*c[1],right=c[2].transpose()*factor*c[3];
            for (int i=0;i<os;++i) for (int j=0;j<ot;++j)
                for (int a=0;a<vs;++a) for (int b=0;b<vt;++b)
                    out+=(block==1?2:1)*scale*t[ti(i,j,a,b,ot,vs,vt)]*left(i,os+a)*right(j,ot+b);
        }
        return out;
    }
    double pair_scalar(const Fixture &f,const Orb &c)
    {
        return scalar(f,0,{c.alpha,c.alpha,c.alpha,c.alpha})+
               scalar(f,1,{c.alpha,c.alpha,c.beta,c.beta})+
               scalar(f,2,{c.beta,c.beta,c.beta,c.beta});
    }
    Fixture swap_pair_spins(const Fixture &f)
    {
        auto b=f;
        std::swap(b.t.nocc_alpha,b.t.nocc_beta); std::swap(b.t.nvirt_alpha,b.t.nvirt_beta);
        std::swap(b.t.aa,b.t.bb); std::swap(b.coeff.alpha,b.coeff.beta);
        b.t.ab.clear();
        for (int j=0;j<f.t.nocc_beta;++j) for (int i=0;i<f.t.nocc_alpha;++i)
            for (int v=0;v<f.t.nvirt_beta;++v) for (int a=0;a<f.t.nvirt_alpha;++a)
                b.t.ab.push_back(f.t.ab[ti(i,j,a,v,f.t.nocc_beta,f.t.nvirt_alpha,f.t.nvirt_beta)]);
        b.g=integrals(f.model,b.coeff);
        return b;
    }
    void pair_checks(const Fixture &f)
    {
        auto built=build_udh_pair_orbital_gradient(f.t,f.g,.27);
        if (!built) { std::cerr<<built.error()<<'\n'; ok=false; return; }
        const int na=f.g.n_alpha,nb=f.g.n_beta;
        for (int block=0;block<3;++block)
        {
            const auto &sector=block==0?built->aa:(block==1?built->ab:built->bb);
            const auto &cs=block==2?f.coeff.beta:f.coeff.alpha,&ct=block==0?f.coeff.alpha:f.coeff.beta;
            const std::array<M,4> legs{cs,cs,ct,ct};
            double worst=0;
            for (int slot=0;slot<4;++slot)
            {
                const bool beta=slot<2?block==2:block!=0;
                const auto &expected=beta?sector.slots[slot].beta:sector.slots[slot].alpha;
                near((beta?sector.slots[slot].alpha:sector.slots[slot].beta).norm(),0);
                for (int p=0;p<expected.rows();++p) for (int q=0;q<expected.cols();++q)
                    for (double h:{1e-3,1e-4})
                    {
                        auto plus=legs,minus=legs;
                        plus[slot].col(q)+=h*legs[slot].col(p); minus[slot].col(q)-=h*legs[slot].col(p);
                        double fd=(scalar(f,block,plus)-scalar(f,block,minus))/(2*h);
                        near(fd,expected(p,q)); worst=std::max(worst,std::abs(fd-expected(p,q)));
                    }
            }
            // Eq.22 independently reduced external/internal sums. Same-spin
            // coefficient 2 combines two identical legs; OS 2 is its pair weight.
            for (int spin=0;spin<2;++spin)
            {
                int o=spin?f.t.nocc_beta:f.t.nocc_alpha,v=spin?f.t.nvirt_beta:f.t.nvirt_alpha;
                M ext=M::Zero(v,o),in=M::Zero(v,o);
                if (block!=1 && spin==(block==2))
                {
                    const auto &t=spin?f.t.bb:f.t.aa,&g=spin?f.g.bb:f.g.aa;
                    const int n=o+v;
                    for (int a=0;a<v;++a) for (int i=0;i<o;++i)
                    {
                        for (int j=0;j<o;++j) for (int b=0;b<v;++b) for (int c=0;c<v;++c)
                            ext(a,i)+=2*.27*g[gi(o+a,o+c,j,o+b,n,n)]*t[ti(i,j,c,b,o,v,v)];
                        for (int j=0;j<o;++j) for (int k=0;k<o;++k) for (int b=0;b<v;++b)
                            in(a,i)-=2*.27*g[gi(k,i,j,o+b,n,n)]*t[ti(k,j,a,b,o,v,v)];
                    }
                }
                if (block==1)
                {
                    const int oa=f.t.nocc_alpha,ob=f.t.nocc_beta,va=f.t.nvirt_alpha,vb=f.t.nvirt_beta;
                    for (int a=0;a<v;++a) for (int i=0;i<o;++i)
                    {
                        if (spin==0)
                        {
                            for (int j=0;j<ob;++j) for (int b=0;b<vb;++b) for (int c=0;c<va;++c)
                                ext(a,i)+=.54*f.g.ab[gi(oa+a,oa+c,j,ob+b,na,nb)]*f.t.ab[ti(i,j,c,b,ob,va,vb)];
                            for (int j=0;j<ob;++j) for (int k=0;k<oa;++k) for (int b=0;b<vb;++b)
                                in(a,i)-=.54*f.g.ab[gi(k,i,j,ob+b,na,nb)]*f.t.ab[ti(k,j,a,b,ob,va,vb)];
                        }
                        else
                        {
                            for (int j=0;j<oa;++j) for (int b=0;b<va;++b) for (int c=0;c<vb;++c)
                                ext(a,i)+=.54*f.g.ab[gi(j,oa+b,ob+a,ob+c,na,nb)]*f.t.ab[ti(j,i,b,c,ob,va,vb)];
                            for (int j=0;j<oa;++j) for (int k=0;k<ob;++k) for (int b=0;b<va;++b)
                                in(a,i)-=.54*f.g.ab[gi(j,oa+b,k,i,na,nb)]*f.t.ab[ti(j,k,b,a,ob,va,vb)];
                        }
                    }
                }
                same(ext,spin?sector.external_ai.beta:sector.external_ai.alpha);
                same(in,spin?sector.internal_ai.beta:sector.internal_ai.alpha);
                same(ext+in,spin?sector.total_ai.beta:sector.total_ai.alpha);
            }
            std::cout<<"U3 pair sector="<<block<<" four-slot basis FD max="<<worst<<'\n';
        }
    }

    void rhs_checks(Fixture f,bool offshell=false)
    {
        if (offshell)
        {
            // Keep coefficients and the response's density fixed, but change
            // amplitudes and the off-diagonal one-body Fock blocks.
            for (auto *t:{&f.t.aa,&f.t.ab,&f.t.bb}) for (auto &v:*t) v*=1.19;
            f.model.h.alpha+=.023*M::Ones(f.s.rows(),f.s.rows());
            f.model.h.beta-=.037*M::Ones(f.s.rows(),f.s.rows());
        }
        const auto ground=density(f,f.coeff);
        auto op=f.model.response(ground);
        const auto fm=mo_fock(f,f.coeff);
        auto rhs=build_udh_orbital_rhs(f.t,f.g,.27,f.coeff,f.s,fm,op);
        if (!rhs) { std::cerr<<rhs.error()<<'\n'; ok=false; return; }
        if (!offshell) { near(rhs->fock_connection_ai.alpha.norm(),0); near(rhs->fock_connection_ai.beta.norm(),0); }
        else check(rhs->fock_connection_ai.alpha.norm()+rhs->fock_connection_ai.beta.norm()>1e-8);
        const AO d{rhs->dprime_ao.alpha_ao,rhs->dprime_ao.beta_ao};
        const AO f0=f.model.fock(ground);
        const auto connection_scalar=[&](const Orb &c)
        { return dot(rhs->dprime.alpha_mo,M(c.alpha.transpose()*f0.alpha*c.alpha))+
                 dot(rhs->dprime.beta_mo,M(c.beta.transpose()*f0.beta*c.beta)); };
        const auto total_scalar=[&](const Orb &c)
        {
            const auto fmoving=mo_fock(f,c);
            return pair_scalar(f,c)+dot(rhs->dprime.alpha_mo,fmoving.alpha)+dot(rhs->dprime.beta_mo,fmoving.beta);
        };
        const auto flipped=swap_pair_spins(f);
        auto flipped_op=op;
        flipped_op.xc=[fn=op.xc](const AO &q)->std::expected<AO,std::string>
        {
            auto v=fn({q.beta,q.alpha});
            if (!v) return std::unexpected(v.error());
            return AO{v->beta,v->alpha};
        };
        auto flipped_rhs=build_udh_orbital_rhs(flipped.t,flipped.g,.27,flipped.coeff,f.s,
                                              {fm.beta,fm.alpha},flipped_op);
        check(flipped_rhs.has_value());
        if (flipped_rhs)
        {
            same(flipped_rhs->pair.aa.g.alpha,rhs->pair.bb.g.beta);
            same(flipped_rhs->pair.ab.g.alpha,rhs->pair.ab.g.beta);
            same(flipped_rhs->total_ai.alpha,rhs->total_ai.beta);
            same(flipped_rhs->total_ai.beta,rhs->total_ai.alpha);
        }
        // A simultaneous mixed-spin rotation checks that both spin ledgers
        // compose into the same stationary scalar, not just basis columns.
        Orb mixed{M::Zero(f.g.n_alpha,f.g.n_alpha),M::Zero(f.g.n_beta,f.g.n_beta)};
        for (int spin=0;spin<2;++spin)
        {
            int o=spin?f.t.nocc_beta:f.t.nocc_alpha,v=spin?f.t.nvirt_beta:f.t.nvirt_alpha;
            auto &u=spin?mixed.beta:mixed.alpha;
            for (int a=0;a<v;++a) for (int i=0;i<o;++i)
            { u(o+a,i)=.2*std::sin((a+1)*(i+1)+spin); u(i,o+a)=-u(o+a,i); }
        }
        const double mixed_want=dot(rhs->total_ai.alpha,M(mixed.alpha.bottomLeftCorner(f.t.nvirt_alpha,f.t.nocc_alpha)))+
                                dot(rhs->total_ai.beta,M(mixed.beta.bottomLeftCorner(f.t.nvirt_beta,f.t.nocc_beta)));
        for (double h:{1e-4,5e-5})
        {
            const Orb plus{f.coeff.alpha+h*f.coeff.alpha*mixed.alpha,f.coeff.beta+h*f.coeff.beta*mixed.beta};
            const Orb minus{f.coeff.alpha-h*f.coeff.alpha*mixed.alpha,f.coeff.beta-h*f.coeff.beta*mixed.beta};
            near((total_scalar(plus)-total_scalar(minus))/(2*h),mixed_want,3e-8);
        }
        double worst=0,response_worst=0,connection_worst=0;
        int directions=0;
        for (int spin=0;spin<2;++spin)
        {
            int o=spin?f.t.nocc_beta:f.t.nocc_alpha,v=spin?f.t.nvirt_beta:f.t.nvirt_alpha;
            for (int a=0;a<v;++a) for (int i=0;i<o;++i)
            {
                for (double h:{1e-4,5e-5})
                {
                    auto plus=f.coeff,minus=f.coeff;
                    auto &cp=spin?plus.beta:plus.alpha,&cm=spin?minus.beta:minus.alpha;
                    const auto &c=spin?f.coeff.beta:f.coeff.alpha;
                    cp.col(i)+=h*c.col(o+a); cp.col(o+a)-=h*c.col(i);
                    cm.col(i)-=h*c.col(o+a); cm.col(o+a)+=h*c.col(i);
                    const double fd=(total_scalar(plus)-total_scalar(minus))/(2*h);
                    const double want=(spin?rhs->total_ai.beta:rhs->total_ai.alpha)(a,i);
                    near(fd,want,3e-8); worst=std::max(worst,std::abs(fd-want));
                    double connection=(connection_scalar(plus)-connection_scalar(minus))/(2*h);
                    const double cw=(spin?rhs->fock_connection_ai.beta:rhs->fock_connection_ai.alpha)(a,i);
                    near(connection,cw); connection_worst=std::max(connection_worst,std::abs(connection-cw));
                    const auto pp=density(f,plus),pm=density(f,minus);
                    const auto fp=f.model.fock(pp),fm=f.model.fock(pm);
                    const double response_fd=(dot(d,fp)-dot(d,fm))/(2*h);
                    const double rw=(spin?rhs->response_ai.beta:rhs->response_ai.alpha)(a,i);
                    near(response_fd,rw,3e-8); response_worst=std::max(response_worst,std::abs(response_fd-rw));
                    // Separate physical J/K/XC finite differences, paired
                    // with fixed D'_AO, never with the projected RHS itself.
                    const M jp=f.model.j(pp.alpha+pp.beta),jm=f.model.j(pm.alpha+pm.beta);
                    near(dot(d.alpha+d.beta,M((jp-jm)/(2*h))),
                         (spin?rhs->response_j.beta:rhs->response_j.alpha)(a,i),3e-8);
                    AO kp{-f.model.ax*f.model.k(pp.alpha),-f.model.ax*f.model.k(pp.beta)};
                    AO km{-f.model.ax*f.model.k(pm.alpha),-f.model.ax*f.model.k(pm.beta)};
                    near((dot(d,kp)-dot(d,km))/(2*h),
                         (spin?rhs->response_k.beta:rhs->response_k.alpha)(a,i),3e-8);
                    const auto xp=f.model.xc(pp),xm=f.model.xc(pm);
                    near((dot(d,xp)-dot(d,xm))/(2*h),
                         (spin?rhs->xc_from_alpha.beta:rhs->xc_from_alpha.alpha)(a,i)+
                         (spin?rhs->xc_from_beta.beta:rhs->xc_from_beta.alpha)(a,i),3e-8);
                }
                ++directions;
            }
        }
        std::cout<<"U3 "<<(offshell?"off-shell":"canonical")<<" directions="<<directions
                 <<" stationary FD="<<worst<<" response FD="<<response_worst
                 <<" coefficient connection FD="<<connection_worst<<'\n';
        for (double scale:{0.,.54,-.27})
        {
            auto scaled=build_udh_orbital_rhs(f.t,f.g,scale,f.coeff,f.s,fm,op);
            check(scaled.has_value());
            if (scaled) { same(scaled->total_ai.alpha,(scale/.27)*rhs->total_ai.alpha);
                          same(scaled->total_ai.beta,(scale/.27)*rhs->total_ai.beta); }
        }
        auto failure=op;
        failure.xc=[](const AO &)->std::expected<AO,std::string>{return std::unexpected("U3 sentinel");};
        auto rejected=build_udh_orbital_rhs(f.t,f.g,.27,f.coeff,f.s,fm,failure);
        check(!rejected && rejected.error().find("U3 sentinel")!=std::string::npos);
        auto badc=f.coeff; badc.alpha*=2;
        check(!build_udh_orbital_rhs(f.t,f.g,.27,badc,f.s,fm,op));
        auto badf=fm; badf.alpha(0,1)+=.2;
        check(!build_udh_orbital_rhs(f.t,f.g,.27,f.coeff,f.s,badf,op));
    }
}

int main()
{
    std::cout<<std::setprecision(12);
    try
    {
        const auto f=make();
        pair_checks(f); rhs_checks(f); rhs_checks(f,true);
        for (int keep=0;keep<3;++keep)
        {
            auto isolated=f;
            if (keep!=0) std::fill(isolated.t.aa.begin(),isolated.t.aa.end(),0);
            if (keep!=1) std::fill(isolated.t.ab.begin(),isolated.t.ab.end(),0);
            if (keep!=2) std::fill(isolated.t.bb.begin(),isolated.t.bb.end(),0);
            std::cout<<"U3 isolated stationary sector="<<keep<<'\n';
            rhs_checks(isolated);
        }
        // Reversing unequal spin spaces gives the complementary beta orientation.
        const auto reversed=make(2,3); pair_checks(reversed); rhs_checks(reversed);
        // Legitimate zero sectors: no beta occupation, no alpha virtuals.
        const auto empty=make(2,0,4); pair_checks(empty); rhs_checks(empty);
        const auto full=make(4,2,4); pair_checks(full); rhs_checks(full);
        const auto closed=make(2,2,5,true);
        auto cg=build_udh_pair_orbital_gradient(closed.t,closed.g,.27);
        check(cg.has_value());
        if (cg)
        {
            same(cg->total_ai.alpha,cg->total_ai.beta);
            // Independent spatial RKS pair functional: 2c (2t-t_exchange):g.
            auto spatial=closed;
            for (int i=0;i<2;++i) for (int j=0;j<2;++j)
                for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                    spatial.t.ab[ti(i,j,a,b,2,3,3)]=2*closed.t.ab[ti(i,j,a,b,2,3,3)]-closed.t.ab[ti(i,j,b,a,2,3,3)];
            for (int p=0;p<5;++p) for (int q=0;q<5;++q)
            {
                auto plus=closed.coeff.alpha,minus=plus;
                plus.col(q)+=1e-4*closed.coeff.alpha.col(p); minus.col(q)-=1e-4*closed.coeff.alpha.col(p);
                double fd=(scalar(spatial,1,{plus,plus,plus,plus})-scalar(spatial,1,{minus,minus,minus,minus}))/(2e-4);
                near(fd,cg->g.alpha(p,q)+cg->g.beta(p,q),3e-9);
            }
        }
        auto bad=f.g; bad.ab.pop_back(); check(!build_udh_pair_orbital_gradient(f.t,bad,0));
        bad=f.g; bad.aa[1]+=.1; check(!build_udh_pair_orbital_gradient(f.t,bad,.27));
        bad=f.g; bad.bb[0]=std::numeric_limits<double>::infinity(); check(!build_udh_pair_orbital_gradient(f.t,bad,0));
        auto t=f.t; t.aa[0]=1; check(!build_udh_pair_orbital_gradient(t,f.g,.27));
        check(!build_udh_pair_orbital_gradient(f.t,f.g,std::numeric_limits<double>::quiet_NaN()));
    }
    catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; ok=false; }
    std::cout<<(ok?"PASS":"FAIL")<<" U3 literal pair and stationary orbital RHS\n";
    return ok?0:1;
}
