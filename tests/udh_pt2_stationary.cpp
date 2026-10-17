// U1: independent full spin-orbital density and amplitude-residual oracles.
// No SCF, orbital solve, external chemistry code, or production driver.
#include "dft/udh_pt2_gradient.h"
#include <Eigen/Jacobi>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <source_location>

namespace
{
    using namespace DFT::Gradient;
    using Matrix=Eigen::MatrixXd;
    bool ok=true;
    void check(bool pass,std::source_location loc=std::source_location::current())
    { if (!pass) { std::cerr << "FAIL line " << loc.line() << '\n'; ok=false; } }
    void near(double a,double b,double tol=2e-12,
              std::source_location loc=std::source_location::current())
    { check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tol*(1+std::abs(b)),loc); }
    void matrix_near(const Matrix &a,const Matrix &b,double tol=2e-12,
                     std::source_location loc=std::source_location::current())
    {
        check(a.rows()==b.rows() && a.cols()==b.cols(),loc);
        if (a.rows()==b.rows() && a.cols()==b.cols() && a.size())
            near((a-b).cwiseAbs().maxCoeff(),0,tol,loc);
    }
    std::size_t idx(int i,int j,int a,int b,int oj,int va,int vb)
    { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }

    // Chemist-pair symmetry, with distinct spin orbital features.
    double eri(int s,int p,int q,int t,int r,int u,bool closed=false)
    {
        double g=0;
        for (int k=1;k<=4;++k)
            g+=std::cos(.23*k*(p+1)*(q+1)+(closed?0:.4*s))*
               std::cos(.23*k*(r+1)*(u+1)+(closed?0:.4*t))/(20*k);
        return g;
    }
    struct Fixture
    {
        UDHPT2Amplitudes t;
        UDHPT2DirectIntegrals g;
        std::array<Eigen::VectorXd,2> eps;
        std::array<Matrix,2> f;
    };
    Fixture make(int oa,int ob,int va,int vb,bool closed=false)
    {
        Fixture x;
        x.t={oa,ob,va,vb,{},{},{}};
        for (int s=0;s<2;++s)
        {
            const int o=s?ob:oa,v=s?vb:va;
            x.eps[s].resize(o+v);
            for (int p=0;p<o+v;++p)
                x.eps[s](p)=(p<o ? -1.4+.13*p : .3+.17*(p-o))+(closed?0:.09*s);
            x.f[s]=x.eps[s].asDiagonal();
        }
        for (int sector=0;sector<3;++sector)
        {
            const int si=sector==2,sj=sector!=0;
            const int oi=si?ob:oa,oj=sj?ob:oa,vi=si?vb:va,vj=sj?vb:va;
            auto &t=sector==0?x.t.aa:(sector==1?x.t.ab:x.t.bb);
            auto &g=sector==0?x.g.aa:(sector==1?x.g.ab:x.g.bb);
            for (int i=0;i<oi;++i) for (int j=0;j<oj;++j)
                for (int a=0;a<vi;++a) for (int b=0;b<vj;++b)
                {
                    double direct=eri(si,i,oi+a,sj,j,oj+b,closed);
                    double v=direct-(si==sj?eri(si,i,oi+b,sj,j,oj+a,closed):0);
                    g.push_back(direct);
                    t.push_back(v/(x.eps[si](i)+x.eps[sj](j)-x.eps[si](oi+a)-x.eps[sj](oj+b)));
                }
        }
        return x;
    }
    struct Orb { int spin,p; };
    struct SpinOrbital
    {
        std::vector<Orb> o,v;
        std::vector<double> t,w;
        Matrix fo,fv;
        int no() const { return static_cast<int>(o.size()); }
        int nv() const { return static_cast<int>(v.size()); }
        std::size_t q(int i,int j,int a,int b) const { return idx(i,j,a,b,no(),nv(),nv()); }
    };
    SpinOrbital expand(const Fixture &x)
    {
        SpinOrbital z;
        const int oa=x.t.nocc_alpha,ob=x.t.nocc_beta,va=x.t.nvirt_alpha,vb=x.t.nvirt_beta;
        for (int s=0;s<2;++s)
        {
            for (int i=0;i<(s?ob:oa);++i) z.o.push_back({s,i});
            for (int a=0;a<(s?vb:va);++a) z.v.push_back({s,a});
        }
        z.fo=Matrix::Zero(z.no(),z.no()); z.fv=Matrix::Zero(z.nv(),z.nv());
        for (int i=0;i<z.no();++i) for (int j=0;j<z.no();++j)
            if (z.o[i].spin==z.o[j].spin)
                z.fo(i,j)=x.f[z.o[i].spin](z.o[i].p,z.o[j].p);
        for (int a=0;a<z.nv();++a) for (int b=0;b<z.nv();++b)
            if (z.v[a].spin==z.v[b].spin)
            {
                int s=z.v[a].spin,o=s?ob:oa;
                z.fv(a,b)=x.f[s](o+z.v[a].p,o+z.v[b].p);
            }
        // Extend aa/ab/bb to a fully antisymmetric spin-orbital tensor.
        // Mixed spin has FOUR placements: ab/ba in each occupied/virtual pair.
        for (auto i:z.o) for (auto j:z.o) for (auto a:z.v) for (auto b:z.v)
        {
            double t=0,w=0;
            if (i.spin==j.spin)
            {
                if (i.spin==a.spin && j.spin==b.spin)
                {
                    int o=i.spin?ob:oa,v=i.spin?vb:va;
                    const auto &ts=i.spin?x.t.bb:x.t.aa;
                    const auto &gs=i.spin?x.g.bb:x.g.aa;
                    t=ts[idx(i.p,j.p,a.p,b.p,o,v,v)];
                    w=gs[idx(i.p,j.p,a.p,b.p,o,v,v)]-gs[idx(i.p,j.p,b.p,a.p,o,v,v)];
                }
            }
            else if (a.spin!=b.spin)
            {
                const auto ia=i.spin?j:i,jb=i.spin?i:j;
                const auto av=a.spin?b:a,bv=a.spin?a:b;
                const double sign=(i.spin?-1.:1.)*(a.spin?-1.:1.);
                t=sign*x.t.ab[idx(ia.p,jb.p,av.p,bv.p,ob,va,vb)];
                w=sign*x.g.ab[idx(ia.p,jb.p,av.p,bv.p,ob,va,vb)];
            }
            z.t.push_back(t); z.w.push_back(w);
        }
        return z;
    }
    // Independent -1/2/+1/2 spin-orbital TT density; no aa/ab weights.
    UDHPT2DPrime density_oracle(const Fixture &x,double c)
    {
        const auto z=expand(x);
        Matrix oo=Matrix::Zero(z.no(),z.no()),vv=Matrix::Zero(z.nv(),z.nv());
        for (int i=0;i<z.no();++i) for (int j=0;j<z.no();++j)
            for (int k=0;k<z.no();++k) for (int a=0;a<z.nv();++a) for (int b=0;b<z.nv();++b)
                oo(i,j)-=.5*c*z.t[z.q(i,k,a,b)]*z.t[z.q(j,k,a,b)];
        for (int a=0;a<z.nv();++a) for (int b=0;b<z.nv();++b)
            for (int i=0;i<z.no();++i) for (int j=0;j<z.no();++j) for (int e=0;e<z.nv();++e)
                vv(a,b)+=.5*c*z.t[z.q(i,j,a,e)]*z.t[z.q(i,j,b,e)];
        UDHPT2DPrime d{Matrix::Zero(x.f[0].rows(),x.f[0].rows()),Matrix::Zero(x.f[1].rows(),x.f[1].rows())};
        for (int s=0;s<2;++s)
        {
            auto &m=s?d.beta_mo:d.alpha_mo;
            int o=s?x.t.nocc_beta:x.t.nocc_alpha;
            for (int i=0;i<z.no();++i) for (int j=0;j<z.no();++j)
                if (z.o[i].spin==s && z.o[j].spin==s) m(z.o[i].p,z.o[j].p)=oo(i,j);
            for (int a=0;a<z.nv();++a) for (int b=0;b<z.nv();++b)
                if (z.v[a].spin==s && z.v[b].spin==s) m(o+z.v[a].p,o+z.v[b].p)=vv(a,b);
        }
        return d;
    }
    // H = c/2 T:V - c/4 T:A_F(T). This four-slot noncanonical amplitude
    // operator is independent of the implementation's density contractions.
    std::pair<double,double> residual_oracle(const SpinOrbital &z,double c,
                                            const std::vector<double> &dt={})
    {
        double value=0,derivative=0;
        for (int i=0;i<z.no();++i) for (int j=0;j<z.no();++j)
            for (int a=0;a<z.nv();++a) for (int b=0;b<z.nv();++b)
            {
                double action=0;
                for (int k=0;k<z.no();++k)
                    action+=z.fo(i,k)*z.t[z.q(k,j,a,b)]+z.fo(j,k)*z.t[z.q(i,k,a,b)];
                for (int e=0;e<z.nv();++e)
                    action-=z.fv(a,e)*z.t[z.q(i,j,e,b)]+z.fv(b,e)*z.t[z.q(i,j,a,e)];
                auto q=z.q(i,j,a,b);
                value+=.5*c*z.t[q]*z.w[q]-.25*c*z.t[q]*action;
                if (!dt.empty()) derivative+=.5*c*dt[q]*(z.w[q]-action);
            }
        return {value,derivative};
    }
    UDHPT2StationaryScalar eval(const Fixture &x,double c=.27)
    {
        auto s=evaluate_udh_pt2_stationary_scalar(x.t,x.g,x.f[0],x.f[1],c);
        if (!s) { std::cerr << s.error() << '\n'; ok=false; return {}; }
        return *s;
    }
    auto snapshot(const Fixture &x)
    {
        HartreeFock::Correlation::UMP2Result r;
        r.nocca=x.t.nocc_alpha; r.noccb=x.t.nocc_beta;
        r.nvira=x.t.nvirt_alpha; r.nvirb=x.t.nvirt_beta;
        r.t2_aa=x.t.aa; r.t2_ab=x.t.ab; r.t2_bb=x.t.bb;
        r.mo_energy_alpha=x.eps[0]; r.mo_energy_beta=x.eps[1];
        for (int s=0;s<2;++s)
        {
            int n=static_cast<int>(x.eps[s].size()),o=s?r.noccb:r.nocca;
            auto &c=s?r.mo_coeff_beta:r.mo_coeff_alpha; c=Matrix::Identity(n,n);
            auto &occ=s?r.mo_occ_beta:r.mo_occ_alpha; occ=Eigen::VectorXd::Zero(n); occ.head(o).setOnes();
            auto &active=s?r.active_mo_beta:r.active_mo_alpha;
            active.resize(n); std::iota(active.begin(),active.end(),0);
        }
        const auto z=expand(x);
        for (int i=0;i<z.no();++i) for (int j=0;j<z.no();++j)
            for (int a=0;a<z.nv();++a) for (int b=0;b<z.nv();++b)
            {
                auto q=z.q(i,j,a,b);
                (z.o[i].spin==z.o[j].spin?r.e_corr_ss:r.e_corr_os)+=.25*z.t[q]*z.w[q];
            }
        r.e_corr=r.e_corr_ss+r.e_corr_os;
        return r;
    }
    Fixture swap_spin(const Fixture &x)
    {
        auto y=x;
        std::swap(y.t.nocc_alpha,y.t.nocc_beta); std::swap(y.t.nvirt_alpha,y.t.nvirt_beta);
        std::swap(y.t.aa,y.t.bb); std::swap(y.g.aa,y.g.bb);
        std::swap(y.eps[0],y.eps[1]); std::swap(y.f[0],y.f[1]);
        y.t.ab.clear(); y.g.ab.clear();
        for (int j=0;j<x.t.nocc_beta;++j) for (int i=0;i<x.t.nocc_alpha;++i)
            for (int b=0;b<x.t.nvirt_beta;++b) for (int a=0;a<x.t.nvirt_alpha;++a)
            {
                auto q=idx(i,j,a,b,x.t.nocc_beta,x.t.nvirt_alpha,x.t.nvirt_beta);
                y.t.ab.push_back(x.t.ab[q]); y.g.ab.push_back(x.g.ab[q]);
            }
        return y;
    }

    void density_and_scalar_checks(const Fixture &x)
    {
        const auto base=eval(x);
        for (double c:{0.,.27,.54,1.,-.27})
        {
            const auto s=eval(x,c);
            const auto ref=density_oracle(x,c);
            matrix_near(s.dprime.alpha_mo,ref.alpha_mo);
            matrix_near(s.dprime.beta_mo,ref.beta_mo);
            matrix_near(s.dprime.alpha_mo,(c/.27)*base.dprime.alpha_mo);
            matrix_near(s.dprime.beta_mo,(c/.27)*base.dprime.beta_mo);
            near(s.total,(c/.27)*base.total);
            near(s.total,residual_oracle(expand(x),c).first);
            near(s.dprime.alpha_mo.trace(),0); near(s.dprime.beta_mo.trace(),0);
            matrix_near(s.dprime.alpha_mo,s.dprime.alpha_mo.transpose());
            matrix_near(s.dprime.beta_mo,s.dprime.beta_mo.transpose());
            matrix_near(s.dprime.alpha_mo.topRightCorner(x.t.nocc_alpha,x.t.nvirt_alpha),
                        Matrix::Zero(x.t.nocc_alpha,x.t.nvirt_alpha));
            matrix_near(s.dprime.beta_mo.topRightCorner(x.t.nocc_beta,x.t.nvirt_beta),
                        Matrix::Zero(x.t.nocc_beta,x.t.nvirt_beta));
        }
        const auto y=eval(swap_spin(x));
        matrix_near(base.dprime.alpha_mo,y.dprime.beta_mo);
        matrix_near(base.dprime.beta_mo,y.dprime.alpha_mo);
        near(base.pair_aa,y.pair_bb); near(base.pair_ab,y.pair_ab);
        near(base.dprime_f_alpha,y.dprime_f_beta); near(base.total,y.total);
    }

    void amplitude_checks(const Fixture &x,bool stationary)
    {
        double max_error=0,max_derivative=0;
        int directions=0;
        for (int sector=0;sector<3;++sector)
        {
            int oi=sector==2?x.t.nocc_beta:x.t.nocc_alpha;
            int oj=sector==0?x.t.nocc_alpha:x.t.nocc_beta;
            int va=sector==2?x.t.nvirt_beta:x.t.nvirt_alpha;
            int vb=sector==0?x.t.nvirt_alpha:x.t.nvirt_beta;
            for (int i=0;i<oi;++i) for (int j=0;j<oj;++j)
                for (int a=0;a<va;++a) for (int b=0;b<vb;++b)
                {
                    if (sector!=1 && (j<=i || b<=a)) continue;
                    auto direction=x;
                    for (auto *t:{&direction.t.aa,&direction.t.ab,&direction.t.bb})
                        std::fill(t->begin(),t->end(),0);
                    auto &dt=sector==0?direction.t.aa:(sector==1?direction.t.ab:direction.t.bb);
                    dt[idx(i,j,a,b,oj,va,vb)]=1;
                    if (sector!=1)
                    {
                        dt[idx(j,i,a,b,oj,va,vb)]=-1; dt[idx(i,j,b,a,oj,va,vb)]=-1;
                        dt[idx(j,i,b,a,oj,va,vb)]=1;
                    }
                    const double expected=residual_oracle(expand(x),.27,expand(direction).t).second;
                    max_derivative=std::max(max_derivative,std::abs(expected));
                    if (stationary) near(expected,0);
                    for (double h:{1e-3,1e-4})
                    {
                        auto plus=x,minus=x;
                        auto &tp=sector==0?plus.t.aa:(sector==1?plus.t.ab:plus.t.bb);
                        auto &tm=sector==0?minus.t.aa:(sector==1?minus.t.ab:minus.t.bb);
                        for (std::size_t q=0;q<dt.size();++q) { tp[q]+=h*dt[q]; tm[q]-=h*dt[q]; }
                        const double fd=(eval(plus).total-eval(minus).total)/(2*h);
                        max_error=std::max(max_error,std::abs(fd-expected));
                        near(fd,expected,2e-10);
                    }
                    ++directions;
                }
        }
        if (!stationary) check(max_derivative>1e-5);
        std::cout << (stationary?"canonical":"off-shell/noncanonical")
                  << " amplitude basis directions=" << directions << " max FD error=" << max_error
                  << " max residual pairing=" << max_derivative << '\n';
    }
}

int main()
{
    std::cout << std::setprecision(14);
    const auto x=make(3,2,3,4);
    density_and_scalar_checks(x);
    const auto r=snapshot(x);
    for (double c:{0.,.27,.54,1.,-.27})
    {
        auto s=build_udh_pt2_stationary_contract(r,x.g,Matrix::Identity(6,6),HartreeFock::OptionsMP2{},c);
        check(s.has_value());
        if (s)
        {
            near(s->stationary.pair,2*c*r.e_corr);
            near(s->stationary.dprime_f,-c*r.e_corr);
            near(s->stationary.total,c*r.e_corr);
            std::cout << "canonical c=" << c << " pair=" << s->stationary.pair
                      << " Dprime:F=" << s->stationary.dprime_f << " H=" << s->stationary.total << '\n';
            std::cout << "  pair aa/ab/bb=" << s->stationary.pair_aa << ' '
                      << s->stationary.pair_ab << ' ' << s->stationary.pair_bb
                      << " Dprime:F alpha/beta=" << s->stationary.dprime_f_alpha
                      << ' ' << s->stationary.dprime_f_beta << '\n';
        }
    }
    // Independent isolated sectors, unequal totals and zero occupied/virtual spaces.
    for (int keep=0;keep<3;++keep)
    {
        auto y=x;
        if (keep!=0) { std::fill(y.t.aa.begin(),y.t.aa.end(),0); std::fill(y.g.aa.begin(),y.g.aa.end(),0); }
        if (keep!=1) { std::fill(y.t.ab.begin(),y.t.ab.end(),0); std::fill(y.g.ab.begin(),y.g.ab.end(),0); }
        if (keep!=2) { std::fill(y.t.bb.begin(),y.t.bb.end(),0); std::fill(y.g.bb.begin(),y.g.bb.end(),0); }
        density_and_scalar_checks(y);
        const auto e=snapshot(y).e_corr;
        const auto s=eval(y); near(s.pair,2*.27*e); near(s.dprime_f,-.27*e);
    }
    for (const auto &y:{make(2,1,3,2),make(2,0,2,4),make(0,2,4,2),
                       make(2,2,0,3),make(2,2,3,0),make(1,0,3,0),make(0,0,0,0)})
        density_and_scalar_checks(y);
    const auto closed_fixture=make(2,2,3,3,true);
    const auto closed=eval(closed_fixture);
    matrix_near(closed.dprime.alpha_mo,closed.dprime.beta_mo);
    near(closed.pair_aa,closed.pair_bb);
    // Closed-shell reduction to spatial t*(2t-t_exchange), total density.
    Matrix restricted=Matrix::Zero(5,5);
    const auto at=[&](int i,int j,int a,int b) { return closed_fixture.t.ab[idx(i,j,a,b,2,3,3)]; };
    for (int i=0;i<2;++i) for (int j=0;j<2;++j)
        for (int k=0;k<2;++k) for (int a=0;a<3;++a) for (int b=0;b<3;++b)
            restricted(i,j)-=2*.27*at(i,k,a,b)*(2*at(j,k,a,b)-at(j,k,b,a));
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
        for (int i=0;i<2;++i) for (int j=0;j<2;++j) for (int e=0;e<3;++e)
            restricted(2+a,2+b)+=2*.27*at(i,j,a,e)*(2*at(i,j,b,e)-at(i,j,e,b));
    matrix_near(closed.dprime.alpha_mo+closed.dprime.beta_mo,restricted);

    amplitude_checks(x,true);
    auto off=x;
    for (auto *t:{&off.t.aa,&off.t.ab,&off.t.bb}) for (auto &v:*t) v*=1.17;
    for (int s=0;s<2;++s) for (int p=0;p<6;++p) for (int q=0;q<=p;++q)
    {
        double value=off.f[s](p,q)+.02*std::cos(.31*(p+1)*(q+1)+s);
        off.f[s](p,q)=off.f[s](q,p)=value;
    }
    density_and_scalar_checks(off);
    amplitude_checks(off,false);
    // Every symmetric MO Fock direction: independent four-slot functional
    // derivative must be D_pp (diagonal) or 2 D_pq (off diagonal), including
    // exactly zero ov/vo. This catches diagonal-only implementations.
    const auto os=eval(off);
    double fock_error=0;
    for (int spin=0;spin<2;++spin) for (int p=0;p<6;++p) for (int q=0;q<=p;++q)
    {
        const auto &d=spin?os.dprime.beta_mo:os.dprime.alpha_mo;
        double want=(p==q?1.:2.)*d(p,q);
        for (double h:{1e-3,1e-4})
        {
            auto plus=off,minus=off;
            plus.f[spin](p,q)+=h; minus.f[spin](p,q)-=h;
            if (p!=q) { plus.f[spin](q,p)+=h; minus.f[spin](q,p)-=h; }
            double fd=(residual_oracle(expand(plus),.27).first-residual_oracle(expand(minus),.27).first)/(2*h);
            near(fd,want,2e-10); near((eval(plus).total-eval(minus).total)/(2*h),want,2e-10);
            fock_error=std::max(fock_error,std::abs(fd-want));
        }
    }
    std::cout << "all symmetric Fock directions max spin-orbital FD error=" << fock_error << '\n';

    // Nonorthogonal AO metric, distinct alpha/beta rotations, trace and all
    // matrix entries from explicit four-index C D C^T contraction.
    Matrix s=Matrix::Zero(6,6),root=Matrix::Zero(6,6);
    for (int p=0;p<6;++p) { s(p,p)=1+.13*p; root(p,p)=1/std::sqrt(s(p,p)); }
    Matrix qa=Matrix::Identity(6,6),qb=qa;
    qa.applyOnTheRight(0,4,Eigen::JacobiRotation<double>(std::cos(.3),std::sin(.3)));
    qb.applyOnTheRight(1,5,Eigen::JacobiRotation<double>(std::cos(.6),std::sin(.6)));
    Matrix ca=root*qa,cb=root*qb;
    const auto d=eval(x).dprime;
    const auto ao=transform_udh_pt2_dprime_to_ao(d,ca,cb,s); check(ao.has_value());
    if (ao) for (int spin=0;spin<2;++spin)
    {
        const auto &c=spin?cb:ca; const auto &m=spin?d.beta_mo:d.alpha_mo;
        const auto &actual=spin?ao->beta_ao:ao->alpha_ao;
        Matrix literal=Matrix::Zero(6,6);
        for (int u=0;u<6;++u) for (int v=0;v<6;++v)
            for (int p=0;p<6;++p) for (int q=0;q<6;++q) literal(u,v)+=c(u,p)*m(p,q)*c(v,q);
        matrix_near(actual,literal); near(actual.cwiseProduct(s).sum(),m.trace());
        Matrix a=Matrix::Ones(6,6); a.diagonal().array()+=.3;
        near(actual.cwiseProduct(a).sum(),m.cwiseProduct(c.transpose()*a*c).sum());
    }

    // Malformed inputs fail even at c=0; raw off-shell data must NOT enter
    // the canonical handoff through a bypass of U0's amplitude residual.
    auto bad=x.t; bad.ab.pop_back(); check(!build_udh_pt2_dprime(bad,0));
    bad=x.t; bad.aa[0]=1; check(!build_udh_pt2_dprime(bad,.27));
    bad=x.t; bad.bb[0]=std::numeric_limits<double>::quiet_NaN(); check(!build_udh_pt2_dprime(bad,0));
    bad=x.t; bad.nocc_alpha=-1; check(!build_udh_pt2_dprime(bad,.27));
    bad=x.t; bad.nvirt_alpha=std::numeric_limits<int>::max(); check(!build_udh_pt2_dprime(bad,.27));
    check(!build_udh_pt2_dprime(x.t,std::numeric_limits<double>::infinity()));
    auto g=x.g; g.ab.pop_back(); check(!evaluate_udh_pt2_stationary_scalar(x.t,g,x.f[0],x.f[1],0));
    g=x.g; g.aa[1]+=.1; check(!evaluate_udh_pt2_stationary_scalar(x.t,g,x.f[0],x.f[1],.27));
    g=x.g; g.ab[0]=std::numeric_limits<double>::infinity();
    check(!evaluate_udh_pt2_stationary_scalar(x.t,g,x.f[0],x.f[1],0));
    Matrix f=x.f[0]; f(0,1)=1; check(!evaluate_udh_pt2_stationary_scalar(x.t,x.g,f,x.f[1],.27));
    f=Matrix::Zero(5,5); check(!evaluate_udh_pt2_stationary_scalar(x.t,x.g,f,x.f[1],.27));
    f=x.f[0]; f(0,0)=std::numeric_limits<double>::infinity();
    check(!evaluate_udh_pt2_stationary_scalar(x.t,x.g,f,x.f[1],.27));
    auto rr=r; rr.t2_ab[0]+=.1;
    check(!build_udh_pt2_stationary_contract(rr,x.g,Matrix::Identity(6,6),HartreeFock::OptionsMP2{},.27));
    auto scope=UDHPT2Scope{}; scope.solvent=true;
    check(!build_udh_pt2_stationary_contract(r,x.g,Matrix::Identity(6,6),HartreeFock::OptionsMP2{},.27,scope));
    Matrix invalid_c=2*ca; check(!transform_udh_pt2_dprime_to_ao(d,invalid_c,cb,s));
    auto invalid_d=d; invalid_d.alpha_mo(0,1)+=1; check(!transform_udh_pt2_dprime_to_ao(invalid_d,ca,cb,s));
    std::cout << (ok?"PASS":"FAIL") << " U1 Dprime/common stationary scalar\n";
    return ok?0:1;
}
