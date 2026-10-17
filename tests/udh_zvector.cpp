// U4: independent co-moving Fock and stationary-scalar orbital FDs.
// Synthetic common AO ERIs/nonlinear two-spin XC, not a molecular DH test.
#include "dft/udh_zvector.h"
#include "dft/dh_pt2_gradient.h"
#include <Eigen/Jacobi>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <iostream>
#include <iomanip>
#include <source_location>
#include <stdexcept>

namespace
{
    using namespace DFT::Gradient;
    using M=Eigen::MatrixXd; using V=Eigen::VectorXd;
    using Orb=UDHOrbitalMatrices; using AO=UDHSpinMatrices;
    bool ok=true;
    void check(bool value,std::source_location at=std::source_location::current())
    { if (!value) { ok=false; std::cerr<<"FAIL U4 line "<<at.line()<<'\n'; } }
    double dot(const M &a,const M &b) { return a.cwiseProduct(b).sum(); }
    double maxabs(const M &m) { return m.size()?m.cwiseAbs().maxCoeff():0; }
    void near(const M &a,const M &b,double tol=2e-9,std::source_location at=std::source_location::current())
    { check(a.rows()==b.rows() && a.cols()==b.cols(),at); if(a.rows()==b.rows() && a.cols()==b.cols())
        check(a.allFinite() && b.allFinite() && maxabs(a-b)<=tol,at); }
    Orb sum(const Orb &a,const Orb &b) { return {a.alpha+b.alpha,a.beta+b.beta}; }
    Orb difference(const Orb &a,const Orb &b,double h)
    { return {(a.alpha-b.alpha)/(2*h),(a.beta-b.beta)/(2*h)}; }
    double pairing(const Orb &a,const Orb &b) { return dot(a.alpha,b.alpha)+dot(a.beta,b.beta); }
    std::size_t gi(int p,int q,int r,int s,int ns,int nt)
    { return ((static_cast<std::size_t>(p)*ns+q)*nt+r)*nt+s; }
    struct Model
    {
        std::vector<M> factors;
        M xa,xb;
        AO h;
        double ax=.53;
        double eri(int p,int q,int r,int s) const
        { double value=0; for(const auto &b:factors) value+=b(p,q)*b(r,s); return value; }
        // Independent explicit AO-index potential (not response callbacks).
        AO jk(const AO &p,bool exchange) const
        {
            const int n=xa.rows(); AO out{M::Zero(n,n),M::Zero(n,n)};
            for(int m=0;m<n;++m) for(int v=0;v<n;++v)
                for(int k=0;k<n;++k) for(int l=0;l<n;++l)
                    if(exchange)
                    { out.alpha(m,v)-=ax*eri(m,k,v,l)*p.alpha(k,l);
                      out.beta(m,v)-=ax*eri(m,k,v,l)*p.beta(k,l); }
                    else
                    { const double value=eri(m,v,k,l)*(p.alpha(k,l)+p.beta(k,l));
                      out.alpha(m,v)+=value; out.beta(m,v)+=value; }
            return out;
        }
        AO xc(const AO &p) const
        {
            const double a=dot(xa,p.alpha),b=dot(xb,p.beta);
            return {(.15*a*a+.13*b+.03*a*b)*xa,(.10*b*b+.13*a+.015*a*a)*xb};
        }
        AO fock(const AO &p) const
        { const auto j=jk(p,false),k=jk(p,true),v=xc(p);
          return {h.alpha+j.alpha+k.alpha+v.alpha,h.beta+j.beta+k.beta+v.beta}; }
        UDHKSResponseOperator response(const AO &ground) const
        {
            const auto bs=factors;
            auto j=[bs](const Eigen::Ref<const M> &q)->std::expected<M,std::string>
            { M out=M::Zero(q.rows(),q.cols()); for(const auto &b:bs) out+=dot(b,q)*b; return out; };
            auto k=[bs](const Eigen::Ref<const M> &q)->std::expected<M,std::string>
            { M out=M::Zero(q.rows(),q.cols()); for(const auto &b:bs) out+=b*q*b; return out; };
            auto kernel=[xa=xa,xb=xb,a=dot(xa,ground.alpha),b=dot(xb,ground.beta)](const AO &q)
                ->std::expected<AO,std::string>
            { const double da=dot(xa,q.alpha),db=dot(xb,q.beta);
              return AO{((.3*a+.03*b)*da+(.13+.03*a)*db)*xa,
                        ((.13+.03*a)*da+.2*b*db)*xb}; };
            auto out=make_udh_ks_response_operator(xa.rows(),ax,j,k,kernel);
            if(!out) throw std::runtime_error(out.error()); return *out;
        }
    };
    struct Fixture { UDHZVectorInputs in; Model model; };
    AO density(const UDHZVectorInputs &in,const Orb &c)
    { return {c.alpha.leftCols(in.nocc_alpha)*c.alpha.leftCols(in.nocc_alpha).transpose(),
              c.beta.leftCols(in.nocc_beta)*c.beta.leftCols(in.nocc_beta).transpose()}; }
    Orb project(const UDHZVectorInputs &in,const Orb &c,const AO &f)
    { return {c.alpha.rightCols(c.alpha.cols()-in.nocc_alpha).transpose()*f.alpha*c.alpha.leftCols(in.nocc_alpha),
              c.beta.rightCols(c.beta.cols()-in.nocc_beta).transpose()*f.beta*c.beta.leftCols(in.nocc_beta)}; }
    Fixture make(int oa=2,int ob=1,int nb=3)
    {
        constexpr int n=4; Fixture f; auto &in=f.in;
        in.nocc_alpha=oa; in.nocc_beta=ob; in.overlap_ao=M::Identity(n,n); M root=in.overlap_ao;
        for(int p=0;p<n;++p) { in.overlap_ao(p,p)=1+.13*p; root(p,p)=1/std::sqrt(1+.13*p); }
        M qa=M::Identity(n,n),qb=qa;
        qa.applyOnTheRight(0,3,Eigen::JacobiRotation<double>(std::cos(.31),std::sin(.31)));
        qb.applyOnTheRight(0,2,Eigen::JacobiRotation<double>(std::cos(.47),std::sin(.47)));
        in.mo_coeff={root*qa,root*qb.leftCols(nb)};
        in.fock_mo={M::Zero(n,n),M::Zero(nb,nb)};
        for(int p=0;p<n;++p) in.fock_mo.alpha(p,p)=p<oa?-1.3+.13*p:.4+.17*p;
        for(int p=0;p<nb;++p) in.fock_mo.beta(p,p)=p<ob?-1.1+.12*p:.6+.15*p;
        for(int l=0;l<3;++l)
        { M b(n,n); for(int p=0;p<n;++p) for(int q=0;q<n;++q) b(p,q)=.12*std::cos(.23*(l+1)*(p+1)*(q+1));
          f.model.factors.push_back(b); }
        f.model.xa=.03*M::Ones(n,n); f.model.xb=.04*M::Ones(n,n);
        for(int p=0;p<n;++p) { f.model.xa(p,p)+=.12*(p+1); f.model.xb(p,p)+=.1*(n-p); }
        f.model.h={M::Zero(n,n),M::Zero(n,n)};
        const auto potential=f.model.fock(density(in,in.mo_coeff));
        const auto desired=[&](const M &c,const M &fm)->M
        { return in.overlap_ao*c*fm*c.transpose()*in.overlap_ao; };
        f.model.h={desired(in.mo_coeff.alpha,in.fock_mo.alpha)-potential.alpha,
                   desired(in.mo_coeff.beta,in.fock_mo.beta)-potential.beta};
        in.response=f.model.response(density(in,in.mo_coeff)); return f;
    }
    Orb zero(const UDHZVectorInputs &in)
    { return {M::Zero(in.mo_coeff.alpha.cols()-in.nocc_alpha,in.nocc_alpha),
              M::Zero(in.mo_coeff.beta.cols()-in.nocc_beta,in.nocc_beta)}; }
    V pack(const Orb &x)
    { V v(x.alpha.size()+x.beta.size()); int p=0;
      for(const auto *m:{&x.alpha,&x.beta}) for(int a=0;a<m->rows();++a) for(int i=0;i<m->cols();++i) v(p++)=(*m)(a,i);
      return v; }
    Orb unit(const UDHZVectorInputs &in,int index)
    { auto x=zero(in); int p=0;
      for(auto *m:{&x.alpha,&x.beta}) for(int a=0;a<m->rows();++a) for(int i=0;i<m->cols();++i) (*m)(a,i)=p++==index?1:0;
      return x; }
    Orb rotate(const UDHZVectorInputs &in,const Orb &x,double step)
    {
        const auto move=[step](const M &c,const M &u,int no)->M
        { M k=M::Zero(c.cols(),c.cols()); k.bottomLeftCorner(u.rows(),no)=u;
          k.topRightCorner(no,u.rows())=-u.transpose(); const M eye=M::Identity(c.cols(),c.cols());
          const M rotation=(eye-.5*step*k).partialPivLu().solve(eye+.5*step*k); return c*rotation; };
        return {move(in.mo_coeff.alpha,x.alpha,in.nocc_alpha),move(in.mo_coeff.beta,x.beta,in.nocc_beta)};
    }
    UDHFullMOIntegrals full_integrals(const Model &model,const Orb &c)
    {
        const int na=c.alpha.cols(),nb=c.beta.cols(); UDHFullMOIntegrals g{na,nb,{},{},{}};
        for(int spin=0;spin<3;++spin)
        {
            const M &cs=spin==2?c.beta:c.alpha,&ct=spin==0?c.alpha:c.beta;
            const int ns=cs.cols(),nt=ct.cols(); auto &v=spin==0?g.aa:spin==1?g.ab:g.bb;
            v.assign(static_cast<std::size_t>(ns)*ns*nt*nt,0);
            for(const auto &b:model.factors)
            { const M left=cs.transpose()*b*cs,right=ct.transpose()*b*ct;
              for(int p=0;p<ns;++p) for(int q=0;q<ns;++q) for(int r=0;r<nt;++r) for(int s=0;s<nt;++s)
                  v[gi(p,q,r,s,ns,nt)]+=left(p,q)*right(r,s); }
        }
        return g;
    }
    UDHPT2DirectIntegrals direct(const UDHFullMOIntegrals &g,int oa,int ob)
    {
        UDHPT2DirectIntegrals out;
        for(int spin=0;spin<3;++spin)
        { const int os=spin==2?ob:oa,ot=spin==0?oa:ob,ns=spin==2?g.n_beta:g.n_alpha,nt=spin==0?g.n_alpha:g.n_beta;
          auto &v=spin==0?out.aa:spin==1?out.ab:out.bb; const auto &full=spin==0?g.aa:spin==1?g.ab:g.bb;
          for(int i=0;i<os;++i) for(int j=0;j<ot;++j) for(int a=os;a<ns;++a) for(int b=ot;b<nt;++b)
              v.push_back(full[gi(i,a,j,b,ns,nt)]); }
        return out;
    }
    void audit(Fixture f)
    {
        const auto &in=f.in; const int n=zero(in).alpha.size()+zero(in).beta.size();
        const auto ground=density(in,in.mo_coeff); const auto f0=f.model.fock(ground);
        near(in.mo_coeff.alpha.transpose()*f0.alpha*in.mo_coeff.alpha,in.fock_mo.alpha,2e-12);
        near(in.mo_coeff.beta.transpose()*f0.beta*in.mo_coeff.beta,in.fock_mo.beta,2e-12);
        M fd(n,n),analytic(n,n); double largest=0,channels=0;
        for(int col=0;col<n;++col)
        {
            const auto x=unit(in,col); const auto ax=apply_udh_eq27_hessian(in,x);
            check(ax.has_value()); if(!ax) throw std::runtime_error(ax.error());
            analytic.col(col)=pack(ax->total);
            for(double step:{1e-4,3e-5})
            {
                const auto cp=rotate(in,x,step),cm=rotate(in,x,-step);
                const auto pp=density(in,cp),pm=density(in,cm);
                const auto fp=f.model.fock(pp),fm=f.model.fock(pm);
                const auto total=difference(project(in,cp,fp),project(in,cm,fm),step);
                fd.col(col)=pack(total); largest=std::max(largest,maxabs(fd.col(col)-analytic.col(col)));
                near(fd.col(col),analytic.col(col),3e-8);
                const auto gap=difference(project(in,cp,f0),project(in,cm,f0),step);
                const auto j=difference(project(in,in.mo_coeff,f.model.jk(pp,false)),project(in,in.mo_coeff,f.model.jk(pm,false)),step);
                const auto k=difference(project(in,in.mo_coeff,f.model.jk(pp,true)),project(in,in.mo_coeff,f.model.jk(pm,true)),step);
                const auto xc=difference(project(in,in.mo_coeff,f.model.xc(pp)),project(in,in.mo_coeff,f.model.xc(pm)),step);
                for(const auto &pair:{std::pair{gap,ax->orbital},std::pair{j,ax->coulomb},std::pair{k,ax->exchange},
                    std::pair{xc,sum(ax->xc_from_alpha,ax->xc_from_beta)}})
                { const double error=maxabs(pack(pair.first)-pack(pair.second)); channels=std::max(channels,error);
                  near(pack(pair.first),pack(pair.second),3e-8); }
                // Each basis direction populates exactly one source-spin XC
                // pair, exposing aa/ba separately from ab/bb (including zero K).
                if(col<zero(in).alpha.size())
                { near(ax->xc_from_beta.alpha,M::Zero(ax->total.alpha.rows(),ax->total.alpha.cols()));
                  near(ax->xc_from_beta.beta,M::Zero(ax->total.beta.rows(),ax->total.beta.cols()));
                  near(ax->exchange.beta,M::Zero(ax->total.beta.rows(),ax->total.beta.cols())); }
                else
                { near(ax->xc_from_alpha.alpha,M::Zero(ax->total.alpha.rows(),ax->total.alpha.cols()));
                  near(ax->xc_from_alpha.beta,M::Zero(ax->total.beta.rows(),ax->total.beta.cols()));
                  near(ax->exchange.alpha,M::Zero(ax->total.alpha.rows(),ax->total.alpha.cols())); }
            }
        }
        near(analytic,analytic.transpose(),2e-12);
        // U3 -> U4 handoff, with canonical amplitudes from the common AO model.
        const auto g=full_integrals(f.model,in.mo_coeff);
        UDHPT2Amplitudes t{in.nocc_alpha,in.nocc_beta,static_cast<int>(in.mo_coeff.alpha.cols())-in.nocc_alpha,
            static_cast<int>(in.mo_coeff.beta.cols())-in.nocc_beta,{},{},{}};
        for(int spin=0;spin<3;++spin)
        { const int os=spin==2?t.nocc_beta:t.nocc_alpha,ot=spin==0?t.nocc_alpha:t.nocc_beta,
                    ns=spin==2?g.n_beta:g.n_alpha,nt=spin==0?g.n_alpha:g.n_beta;
          const M &es=spin==2?in.fock_mo.beta:in.fock_mo.alpha,&et=spin==0?in.fock_mo.alpha:in.fock_mo.beta;
          auto &v=spin==0?t.aa:spin==1?t.ab:t.bb; const auto &eri=spin==0?g.aa:spin==1?g.ab:g.bb;
          for(int i=0;i<os;++i) for(int j=0;j<ot;++j) for(int a=os;a<ns;++a) for(int b=ot;b<nt;++b)
          { double value=eri[gi(i,a,j,b,ns,nt)]; if(spin!=1) value-=eri[gi(i,b,j,a,ns,nt)];
            v.push_back(value/(es(i,i)+et(j,j)-es(a,a)-et(b,b))); } }
        auto rhs=build_udh_orbital_rhs(t,g,.27,in.mo_coeff,in.overlap_ao,in.fock_mo,in.response);
        if(in.mo_coeff.alpha.cols()!=in.overlap_ao.rows() || in.mo_coeff.beta.cols()!=in.overlap_ao.rows())
        {
            // U4 permits rectangular response spaces, but the current U1/U3
            // AO Dprime handoff is deliberately all-active and square. Test
            // that rejection; do not broaden the stationary contract here.
            check(!rhs && rhs.error().find("invalid all-active matrix shape")!=std::string::npos);
            auto rectangular_rhs=zero(in);
            rectangular_rhs.alpha.setConstant(.17); rectangular_rhs.beta.setConstant(-.23);
            const auto solved=solve_udh_zvector(in,rectangular_rhs);
            check(solved.has_value()); if(!solved) throw std::runtime_error(solved.error());
            near(solved->jacobian,analytic,2e-12);
            const V reference=fd.transpose().colPivHouseholderQr().solve(-pack(rectangular_rhs));
            near(pack(solved->z_ai),reference,2e-9);
            check(solved->rank==n && solved->action_count==1+2*n);
            std::cout<<"U4 rectangular pairs="<<n<<" Fock-column FD="<<largest<<" channel FD="<<channels
                <<" Z FD="<<maxabs(pack(solved->z_ai)-reference)<<" residual="<<solved->residual_max_abs
                <<"; U3 all-active rejection confirmed\n";
            return;
        }
        check(rhs.has_value()); if(!rhs) throw std::runtime_error(rhs.error());
        auto solved=solve_udh_zvector(in,rhs->total_ai);
        check(solved.has_value()); if(!solved) throw std::runtime_error(solved.error());
        check(solved->rank==n && solved->action_count==1+2*n && solved->reciprocal_condition>1e-12);
        near(solved->jacobian,analytic,2e-12);
        const V reference=fd.transpose().colPivHouseholderQr().solve(-pack(rhs->total_ai));
        near(pack(solved->z_ai),reference,2e-9);
        Orb mixed=zero(in); mixed.alpha.setConstant(.37); mixed.beta.setConstant(-.23);
        const auto am=apply_udh_eq27_hessian(in,mixed); check(am.has_value());
        if(am) near(pack(am->total),analytic*pack(mixed),2e-12);
        const auto mixed_plus=rotate(in,mixed,3e-5),mixed_minus=rotate(in,mixed,-3e-5);
        near(pack(difference(project(in,mixed_plus,f.model.fock(density(in,mixed_plus))),
            project(in,mixed_minus,f.model.fock(density(in,mixed_minus))),3e-5)),analytic*pack(mixed),3e-8);
        // Independent H(C) with fixed stationary amplitudes and full moving
        // F_mo. Its derivative plus z:F_vo must cancel in every basis direction.
        const auto scalar=[&](const Orb &c)
        { const auto p=density(in,c),fock=f.model.fock(p); const auto gs=direct(full_integrals(f.model,c),in.nocc_alpha,in.nocc_beta);
          auto value=evaluate_udh_pt2_stationary_scalar(t,gs,c.alpha.transpose()*fock.alpha*c.alpha,
              c.beta.transpose()*fock.beta*c.beta,.27);
          if(!value) throw std::runtime_error(value.error());
          return value->total+pairing(solved->z_ai,project(in,c,fock)); };
        double stationary_error=0;
        for(int col=0;col<=n;++col)
        { const auto x=col<n?unit(in,col):mixed; constexpr double h=3e-5;
          const double value=(scalar(rotate(in,x,h))-scalar(rotate(in,x,-h)))/(2*h);
          stationary_error=std::max(stationary_error,std::abs(value)); check(std::abs(value)<2e-9); }
        // Spin-label permutation is a reordering, not a different Hessian.
        auto swapped=in; std::swap(swapped.mo_coeff.alpha,swapped.mo_coeff.beta);
        std::swap(swapped.fock_mo.alpha,swapped.fock_mo.beta); std::swap(swapped.nocc_alpha,swapped.nocc_beta);
        const auto original=in.response.xc;
        swapped.response.xc=[original](const AO &q)->std::expected<AO,std::string>
        { auto v=original({q.beta,q.alpha}); if(!v) return std::unexpected(v.error()); return AO{v->beta,v->alpha}; };
        auto reverse=solve_udh_zvector(swapped,{rhs->total_ai.beta,rhs->total_ai.alpha}); check(reverse.has_value());
        if(reverse) { near(reverse->z_ai.alpha,solved->z_ai.beta,2e-12); near(reverse->z_ai.beta,solved->z_ai.alpha,2e-12); }
        std::cout<<"U4 pairs="<<n<<" Fock-column FD="<<largest<<" channel FD="<<channels
            <<" stationary FD="<<stationary_error<<" residual="<<solved->residual_max_abs
            <<" rcond="<<solved->reciprocal_condition<<'\n';
    }
    void controls()
    {
        auto f=make(); auto in=f.in; auto rhs=zero(in); rhs.alpha.setConstant(.3); rhs.beta.setConstant(-.2);
        auto bad=rhs; bad.beta.resize(1,1); check(!solve_udh_zvector(in,bad));
        bad=rhs; bad.alpha(0,0)=std::numeric_limits<double>::quiet_NaN(); check(!solve_udh_zvector(in,bad));
        auto invalid=in; invalid.mo_coeff.beta(0,0)+=.1; check(!solve_udh_zvector(invalid,rhs));
        invalid=in; invalid.fock_mo.alpha(0,1)=invalid.fock_mo.alpha(1,0)=.1;
        check(!solve_udh_zvector(invalid,rhs)); // no hidden recanonicalization
        invalid=in; invalid.overlap_ao(0,0)=-1; check(!solve_udh_zvector(invalid,rhs));
        invalid=in; invalid.nocc_beta=-1; check(!solve_udh_zvector(invalid,rhs));
        UDHZVectorOptions options; options.residual_tolerance=0; check(!solve_udh_zvector(in,rhs,options));
        options={}; options.minimum_rcond=.99; check(!solve_udh_zvector(in,rhs,options));
        const auto zero_rhs=solve_udh_zvector(in,zero(in)); check(zero_rhs.has_value());
        if(zero_rhs) near(pack(zero_rhs->z_ai),V::Zero(pack(rhs).size()),1e-14);
        for(int channel=0;channel<3;++channel)
        {
            invalid=in;
            UDHMatrixResponseFn error=[](const Eigen::Ref<const M>&)->std::expected<M,std::string>
            { return std::unexpected("injected U4 channel failure"); };
            if(channel==0) invalid.response.coulomb=error;
            if(channel==1) invalid.response.exchange=error;
            if(channel==2) invalid.response.xc=[](const AO&)->std::expected<AO,std::string>
            { throw std::runtime_error("injected U4 channel failure"); };
            const auto result=solve_udh_zvector(invalid,rhs);
            check(!result && result.error().find("injected U4 channel failure")!=std::string::npos);
        }
        invalid=in; invalid.response.xc=[](const AO&)->std::expected<AO,std::string>
        { return AO{M::Zero(1,1),M::Zero(1,1)}; }; check(!solve_udh_zvector(invalid,rhs));
        invalid=in;
        invalid.response.xc=[xa=f.model.xa,xb=f.model.xb](const AO &q)->std::expected<AO,std::string>
        { return AO{.9*dot(xb,q.beta)*xa,.1*dot(xa,q.alpha)*xb}; };
        const auto nonadjoint=solve_udh_zvector(invalid,rhs);
        check(!nonadjoint && nonadjoint.error().find("adjointness")!=std::string::npos);
        // Explicitly relaxed adjoint gate solely to test the transpose route
        // on a nonphysical operator; Az=-ell must NOT satisfy this reference.
        options={}; options.adjoint_tolerance=1;
        auto transposed=solve_udh_zvector(invalid,rhs,options); check(transposed.has_value());
        if(transposed)
        { near(transposed->jacobian.transpose()*pack(transposed->z_ai),-pack(rhs),2e-12);
          check(maxabs(transposed->jacobian*pack(transposed->z_ai)+pack(rhs))>1e-6); }
        invalid=in; int calls=0; const int n=pack(rhs).size(); const auto j=in.response.coulomb;
        invalid.response.coulomb=[&](const Eigen::Ref<const M> &q)->std::expected<M,std::string>
        { if(++calls>n+1) return std::unexpected("fresh transpose injection"); return j(q); };
        const auto stale=solve_udh_zvector(invalid,rhs);
        check(!stale && calls==n+2 && stale.error().find("fresh transpose injection")!=std::string::npos);
        // Rank and indefiniteness are distinct: zero is rejected; negative
        // nonzero gaps are solved without adding a level shift.
        auto no_response=[](const Eigen::Ref<const M> &q)->std::expected<M,std::string>
        { return M::Zero(q.rows(),q.cols()).eval(); };
        invalid=in; invalid.response.exact_exchange=0; invalid.response.coulomb=no_response;
        invalid.response.exchange={}; invalid.response.xc=[](const AO &q)->std::expected<AO,std::string>
        { return AO{M::Zero(q.alpha.rows(),q.alpha.cols()),M::Zero(q.beta.rows(),q.beta.cols())}; };
        invalid.fock_mo.alpha.setZero(); invalid.fock_mo.beta.setZero();
        check(!solve_udh_zvector(invalid,rhs));
        invalid.fock_mo.alpha=in.fock_mo.alpha; invalid.fock_mo.beta=-in.fock_mo.beta;
        const auto indefinite=solve_udh_zvector(invalid,rhs); check(indefinite.has_value());
        if(indefinite) near(indefinite->jacobian.transpose()*pack(indefinite->z_ai),-pack(rhs),2e-12);
        auto empty=make(4,3,3).in; const auto no_pairs=solve_udh_zvector(empty,zero(empty));
        check(no_pairs && no_pairs->rank==0 && no_pairs->action_count==1 && no_pairs->z_ai.alpha.rows()==0 &&
            no_pairs->z_ai.alpha.cols()==4 && no_pairs->z_ai.beta.rows()==0 && no_pairs->z_ai.beta.cols()==3);
        std::cout<<"U4 controls: transpose, rank, indefinite, errors, fresh action, empty spaces\n";
    }
    void closed_shell()
    {
        auto f=make(2,2,4); auto in=f.in;
        in.mo_coeff.beta=in.mo_coeff.alpha; in.fock_mo.beta=in.fock_mo.alpha;
        const M x=f.model.xa;
        in.response.xc=[x](const AO &q)->std::expected<AO,std::string>
        { const double a=dot(x,q.alpha),b=dot(x,q.beta);
          return AO{(.3*a+.2*b)*x,(.2*a+.3*b)*x}; };
        auto rks=make_dh_eq41_response_operator(in.response.exact_exchange,in.response.coulomb,in.response.exchange,
            [x](const Eigen::Ref<const M> &q)->std::expected<M,std::string> { return (.25*dot(x,q)*x).eval(); });
        check(rks.has_value()); if(!rks) return;
        const int no=in.nocc_alpha,nv=in.mo_coeff.alpha.cols()-no;
        M trial(nv,no); trial<<.13,.31,-.21,.17;
        const V eps=in.fock_mo.alpha.diagonal();
        const auto restricted=apply_dh_eq27_hessian(trial,in.mo_coeff.alpha.leftCols(no),
            in.mo_coeff.alpha.rightCols(nv),eps,*rks);
        const auto unrestricted=apply_udh_eq27_hessian(in,{trial,trial});
        check(restricted && unrestricted); if(!restricted || !unrestricted) return;
        near(unrestricted->total.alpha,restricted->total_ai,2e-12);
        near(unrestricted->total.beta,restricted->total_ai,2e-12);
        // Equal-spin ell=-A X/2 => each UKS z=X/2, sum reproduces RKS X.
        const auto solved=solve_udh_zvector(in,{-.5*restricted->total_ai,-.5*restricted->total_ai});
        check(solved.has_value()); if(solved)
        { near(solved->z_ai.alpha,.5*trial,2e-12); near(solved->z_ai.beta,.5*trial,2e-12); }
        std::cout<<"U4 closed-shell Eq27 action and half-Z normalization\n";
    }
}

int main()
{
    std::cout<<std::setprecision(12);
    try
    {
        audit(make()); audit(make(2,0,3)); audit(make(4,1,3)); // Rectangular U4-only spaces.
        audit(make(2,1,4)); audit(make(1,2,4)); // All-active U3 -> U4 stationary handoff.
        audit(make(2,0,4)); audit(make(4,1,4)); // Empty spin blocks within that contract.
        controls(); closed_shell();
    }
    catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; ok=false; }
    std::cout<<(ok?"PASS":"FAIL")<<" U4 coupled canonical KS Jacobian and transpose Z solve\n";
    return ok?0:1;
}
