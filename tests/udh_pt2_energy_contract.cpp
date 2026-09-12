// U0: independent spin-orbital energy sum versus the typed aa/ab/bb ledger.
// No SCF, response solve, production gate, or external-code reference.
#include "dft/udh_pt2_gradient.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <source_location>

namespace
{
    using HartreeFock::Correlation::UMP2Result;
    using namespace DFT::Gradient;
    bool ok=true;
    void check(bool pass, std::source_location at=std::source_location::current())
    {
        if (!pass) { std::cerr << "FAIL " << at.line() << '\n'; ok=false; }
    }
    void near(double a, double b, std::source_location at=std::source_location::current())
    { check(std::isfinite(a) && std::abs(a-b)<1e-12*(1+std::abs(b)),at); }

    // Symmetric AO/MO-pair features define physical ERI permutation symmetry.
    // Spin-dependent synthetic integral data test algebra only (not the
    // AO-to-MO transform); the molecular fixtures exercise a common AO origin.
    // A common spin feature gives the Eq. 36 limit.
    double integral(int s, int p, int q, int t, int r, int u, bool closed)
    {
        double value=0;
        for (int k=1; k<=4; ++k)
            value += std::cos(0.23*k*(p+1)*(q+1)+(closed ? 0 : 0.4*s)) *
                     std::cos(0.23*k*(r+1)*(u+1)+(closed ? 0 : 0.4*t)) / (20*k);
        return value;
    }
    struct Fixture { UMP2Result r; UDHPT2DirectIntegrals g; Eigen::MatrixXd s; bool closed; };
    struct Orbital { int spin, column; double energy; };
    // Full spin-orbital antisymmetrized sum, independent of aa/ab/bb weights.
    std::array<double,3> spin_orbital_energy(const Fixture &f)
    {
        std::vector<Orbital> occ,virt;
        for (int s=0; s<2; ++s)
        {
            int no=s ? f.r.noccb : f.r.nocca;
            const auto &eps=s ? f.r.mo_energy_beta : f.r.mo_energy_alpha;
            for (int p=0; p<eps.size(); ++p)
                (p<no ? occ : virt).push_back({s,p,eps(p)});
        }
        std::array<double,3> energy{};
        for (const auto &i:occ) for (const auto &j:occ)
            for (const auto &a:virt) for (const auto &b:virt)
            {
                double v=0;
                if (i.spin==a.spin && j.spin==b.spin)
                    v+=integral(i.spin,i.column,a.column,j.spin,j.column,b.column,f.closed);
                if (i.spin==b.spin && j.spin==a.spin)
                    v-=integral(i.spin,i.column,b.column,j.spin,j.column,a.column,f.closed);
                int block=i.spin==j.spin ? (i.spin ? 2 : 0) : 1;
                energy[block]+=0.25*v*v/(i.energy+j.energy-a.energy-b.energy);
            }
        return energy;
    }
    Fixture make(int n, int oa, int ob, bool closed=false)
    {
        Fixture f;
        f.closed=closed; f.s=Eigen::MatrixXd::Identity(n,n);
        auto &r=f.r;
        r.nocca=oa; r.noccb=ob; r.nvira=n-oa; r.nvirb=n-ob;
        r.mo_coeff_alpha=r.mo_coeff_beta=f.s;
        r.mo_occ_alpha=Eigen::VectorXd::Zero(n); r.mo_occ_alpha.head(oa).setOnes();
        r.mo_occ_beta=Eigen::VectorXd::Zero(n); r.mo_occ_beta.head(ob).setOnes();
        r.active_mo_alpha.resize(n); std::iota(r.active_mo_alpha.begin(),r.active_mo_alpha.end(),0);
        r.active_mo_beta=r.active_mo_alpha;
        r.mo_energy_alpha.resize(n); r.mo_energy_beta.resize(n);
        for (int p=0;p<n;++p)
        {
            r.mo_energy_alpha(p)=p<oa ? -1.4+0.13*p : 0.3+0.17*(p-oa);
            r.mo_energy_beta(p)=closed ? r.mo_energy_alpha(p) :
                (p<ob ? -1.3+0.11*p : 0.4+0.19*(p-ob));
        }
        for (int block=0;block<3;++block)
        {
            int si=block==2, sj=block!=0;
            int oi=si ? ob : oa, oj=sj ? ob : oa;
            const auto &ei=si ? r.mo_energy_beta : r.mo_energy_alpha;
            const auto &ej=sj ? r.mo_energy_beta : r.mo_energy_alpha;
            auto &g=block==0 ? f.g.aa : (block==1 ? f.g.ab : f.g.bb);
            auto &t=block==0 ? r.t2_aa : (block==1 ? r.t2_ab : r.t2_bb);
            for (int i=0;i<oi;++i) for (int j=0;j<oj;++j)
                for (int a=oi;a<n;++a) for (int b=oj;b<n;++b)
                {
                    double direct=integral(si,i,a,sj,j,b,closed);
                    double exchange=si==sj ? integral(si,i,b,sj,j,a,closed) : 0;
                    g.push_back(direct);
                    t.push_back((direct-exchange)/(ei(i)+ej(j)-ei(a)-ej(b)));
                }
        }
        auto [aa,ab,bb]=spin_orbital_energy(f);
        r.e_corr_ss=aa+bb; r.e_corr_os=ab; r.e_corr=aa+ab+bb;
        return f;
    }
    auto build(const Fixture &f, double c=0.27)
    {
        return build_udh_pt2_energy_contract(f.r,f.g,f.s,HartreeFock::OptionsMP2{},c);
    }
    Fixture spin_swap(const Fixture &f)
    {
        Fixture b=f;
        auto &r=b.r;
        std::swap(r.nocca,r.noccb); std::swap(r.nvira,r.nvirb);
        std::swap(r.mo_coeff_alpha,r.mo_coeff_beta);
        std::swap(r.mo_energy_alpha,r.mo_energy_beta);
        std::swap(r.mo_occ_alpha,r.mo_occ_beta);
        std::swap(r.active_mo_alpha,r.active_mo_beta);
        std::swap(r.t2_aa,r.t2_bb); std::swap(b.g.aa,b.g.bb);
        r.t2_ab.clear(); b.g.ab.clear();
        for (int j=0;j<f.r.noccb;++j) for (int i=0;i<f.r.nocca;++i)
            for (int bvir=0;bvir<f.r.nvirb;++bvir) for (int a=0;a<f.r.nvira;++a)
            {
                auto q=((static_cast<std::size_t>(i)*f.r.noccb+j)*f.r.nvira+a)*f.r.nvirb+bvir;
                r.t2_ab.push_back(f.r.t2_ab[q]); b.g.ab.push_back(f.g.ab[q]);
            }
        return b;
    }
}

int main()
{
    const auto f=make(6,3,2);
    const auto e=build(f);
    if (!e) { std::cerr << e.error() << '\n'; return 1; }
    near(e->unscaled_total,f.r.e_corr);
    const auto independent=spin_orbital_energy(f);
    near(e->aa,independent[0]); near(e->ab,independent[1]); near(e->bb,independent[2]);
    check(e->aa!=0 && e->ab!=0 && e->bb!=0 && e->aa!=e->bb);
    near(e->correction,0.27*f.r.e_corr);
    for (double c : {0.0,1.0,0.54,-0.27})
    {
        const auto scaled=build(f,c); check(scaled.has_value());
        if (scaled) { near(scaled->correction,c*e->unscaled_total); near(scaled->aa,e->aa); }
    }
    const auto swapped=build(spin_swap(f)); check(swapped.has_value());
    if (swapped) { near(swapped->aa,e->bb); near(swapped->bb,e->aa); near(swapped->ab,e->ab); }

    // Closed-shell Eq. 36: compare directly with the spatial RMP2 expression.
    const auto closed=make(5,2,2,true);
    const auto ce=build(closed); check(ce.has_value());
    double rmp2=0,os=0;
    const auto &eps=closed.r.mo_energy_alpha;
    for (int i=0;i<2;++i) for (int j=0;j<2;++j)
        for (int a=2;a<5;++a) for (int b=2;b<5;++b)
        {
            double g=integral(0,i,a,0,j,b,true), gx=integral(0,i,b,0,j,a,true);
            double delta=eps(i)+eps(j)-eps(a)-eps(b);
            rmp2+=g*(2*g-gx)/delta; os+=g*g/delta;
            std::size_t q=((i*2+j)*3+(a-2))*3+(b-2);
            std::size_t exchanged=((i*2+j)*3+(b-2))*3+(a-2);
            near(closed.r.t2_aa[q],closed.r.t2_ab[q]-closed.r.t2_ab[exchanged]);
            near(closed.r.t2_bb[q],closed.r.t2_aa[q]);
        }
    if (ce) { near(ce->unscaled_total,rmp2); near(ce->opposite_spin,os); near(ce->same_spin,rmp2-os); }

    // Expected empty spin sectors versus missing nonzero-extent storage.
    for (const auto &empty : {make(4,1,0),spin_swap(make(4,1,0))})
    {
        const auto z=build(empty); check(z.has_value());
        if (z) near(z->correction,0);
    }
    check(!build(make(4,0,0)));

    // Each sector independently: use an integral-defined zero interaction
    // for the other sectors and matching cached spin-orbital channel sums.
    for (int keep=0;keep<3;++keep)
    {
        auto only=f;
        if (keep!=0) { std::fill(only.g.aa.begin(),only.g.aa.end(),0); std::fill(only.r.t2_aa.begin(),only.r.t2_aa.end(),0); }
        if (keep!=1) { std::fill(only.g.ab.begin(),only.g.ab.end(),0); std::fill(only.r.t2_ab.begin(),only.r.t2_ab.end(),0); }
        if (keep!=2) { std::fill(only.g.bb.begin(),only.g.bb.end(),0); std::fill(only.r.t2_bb.begin(),only.r.t2_bb.end(),0); }
        double target=independent[keep];
        only.r.e_corr_ss=keep==1 ? 0 : target; only.r.e_corr_os=keep==1 ? target : 0;
        only.r.e_corr=target;
        const auto isolated=build(only); check(isolated.has_value());
        if (isolated) near(isolated->unscaled_total,target);
    }
    // Orbital phase gauge: transform coefficients, amplitudes AND integrals.
    auto phased=f;
    phased.r.mo_coeff_alpha.col(0)*=-1;
    for (int block=0;block<2;++block)
    {
        auto &g=block ? phased.g.ab : phased.g.aa;
        auto &t=block ? phased.r.t2_ab : phased.r.t2_aa;
        std::size_t q=0;
        for (int i=0;i<f.r.nocca;++i) for (int j=0;j<(block ? f.r.noccb : f.r.nocca);++j)
            for (int a=0;a<f.r.nvira;++a) for (int b=0;b<(block ? f.r.nvirb : f.r.nvira);++b,++q)
            {
                int sign=(i==0 ? -1 : 1)*(!block && j==0 ? -1 : 1);
                g[q]*=sign; t[q]*=sign;
            }
    }
    const auto pe=build(phased); check(pe.has_value());
    if (pe) near(pe->correction,e->correction);

    // Independent relabeling inside occupied and virtual spin subspaces.
    auto permuted=f;
    std::array<std::vector<int>,2> maps;
    for (int spin=0;spin<2;++spin)
    {
        int no=spin ? f.r.noccb : f.r.nocca;
        auto &c=spin ? permuted.r.mo_coeff_beta : permuted.r.mo_coeff_alpha;
        auto &eps=spin ? permuted.r.mo_energy_beta : permuted.r.mo_energy_alpha;
        const auto &old_c=spin ? f.r.mo_coeff_beta : f.r.mo_coeff_alpha;
        const auto &old_eps=spin ? f.r.mo_energy_beta : f.r.mo_energy_alpha;
        int n=static_cast<int>(eps.size());
        for (int p=0;p<n;++p)
        {
            int old=p<no ? no-1-p : n+no-1-p;
            maps[spin].push_back(old); c.col(p)=old_c.col(old); eps(p)=old_eps(old);
        }
    }
    for (int block=0;block<3;++block)
    {
        int si=block==2, sj=block!=0;
        int oi=si ? f.r.noccb : f.r.nocca, oj=sj ? f.r.noccb : f.r.nocca;
        int vi=si ? f.r.nvirb : f.r.nvira, vj=sj ? f.r.nvirb : f.r.nvira;
        auto &g=block==0 ? permuted.g.aa : (block==1 ? permuted.g.ab : permuted.g.bb);
        auto &t=block==0 ? permuted.r.t2_aa : (block==1 ? permuted.r.t2_ab : permuted.r.t2_bb);
        const auto old_g=g, old_t=t;
        std::size_t q=0;
        for (int i=0;i<oi;++i) for (int j=0;j<oj;++j)
            for (int a=0;a<vi;++a) for (int b=0;b<vj;++b,++q)
            {
                std::size_t old=((static_cast<std::size_t>(maps[si][i])*oj+maps[sj][j])*vi+
                    maps[si][oi+a]-oi)*vj+maps[sj][oj+b]-oj;
                t[q]=old_t[old]; g[q]=old_g[old];
            }
    }
    const auto permuted_energy=build(permuted); check(permuted_energy.has_value());
    if (permuted_energy) near(permuted_energy->correction,e->correction);

    auto bad=f; bad.r.t2_ab.pop_back(); check(!build(bad));
    bad=f; bad.g.aa.pop_back(); check(!build(bad));
    bad=f; bad.r.t2_aa[0]=0.1; check(!build(bad));
    bad=f; bad.r.t2_ab[1]+=0.01; check(!build(bad));
    bad=f; std::swap(bad.g.ab[1],bad.g.ab[5]); check(!build(bad)); // wrong packing
    bad=f; bad.r.t2_bb[1]=std::numeric_limits<double>::quiet_NaN(); check(!build(bad));
    bad=f; bad.g.ab[1]=std::numeric_limits<double>::infinity(); check(!build(bad));
    bad=f; bad.r.e_corr_ss*=2; check(!build(bad));
    bad=f; bad.r.e_corr_os*=2; check(!build(bad));
    bad=f; bad.r.e_corr*=0.27; check(!build(bad)); // pre-scaled cached energy
    bad=f; bad.r.converged=false; check(!build(bad));
    bad=f; bad.r.n_iter=2; check(!build(bad));
    bad=f; bad.r.nocca=-1; check(!build(bad));
    bad=f; bad.r.nvira=std::numeric_limits<int>::max(); check(!build(bad));
    bad=f; bad.r.mo_occ_alpha(0)=2; check(!build(bad));
    bad=f; bad.r.mo_occ_beta(0)=0.5; check(!build(bad));
    bad=f; bad.r.active_mo_alpha[0]=1; check(!build(bad));
    bad=f; bad.r.mo_coeff_beta(0,0)=2; check(!build(bad));
    bad=f; bad.s(0,1)=0.2; check(!build(bad));
    bad=f; bad.r.mo_energy_alpha.setZero(); bad.r.mo_energy_beta.setZero(); check(!build(bad));
    check(!build(f,std::numeric_limits<double>::infinity()));
    bad=f; bad.r.t2_ab.clear(); check(!build(bad,0)); // c=0 must not bypass validation
    auto metric=f; metric.s*=4; metric.r.mo_coeff_alpha*=0.5; metric.r.mo_coeff_beta*=0.5;
    check(build(metric).has_value()); // actual S, not Euclidean normalization

    for (int kind=0;kind<5;++kind)
    {
        HartreeFock::OptionsMP2 options;
        if (kind==0) options.frozen={1};
        if (kind==1) options.use_ri=true;
        if (kind==2) options.level_shift=0.1;
        if (kind==3) options.with_t2=false;
        if (kind==4) options.level_shift=std::numeric_limits<double>::quiet_NaN();
        check(!build_udh_pt2_energy_contract(f.r,f.g,f.s,options,0.27));
    }
    for (int kind=0;kind<7;++kind)
    {
        UDHPT2Scope scope;
        if (kind==0) scope.reference=HartreeFock::SCFType::RHF;
        if (kind==1) scope.reference=HartreeFock::SCFType::ROHF;
        if (kind==2) scope.basis=HartreeFock::BasisType::Spherical;
        if (kind==3) scope.range_separated=true;
        if (kind==4) scope.solvent=true;
        if (kind==5) scope.meta_gga=true;
        if (kind==6) scope.independent_spin_scaling=true;
        check(!build_udh_pt2_energy_contract(f.r,f.g,f.s,HartreeFock::OptionsMP2{},0.27,scope));
    }
    if (ok) std::cout << "PASS UDH U0: spin-orbital energy, storage, Eq. 36 and scope invariants\n";
    return ok ? 0 : 1;
}
