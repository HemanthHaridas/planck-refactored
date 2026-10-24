// U7 independent fixed-contract scalars. No SCF, orbital-response solve,
// molecular total-energy gradient or external quantum-chemistry oracle.
#include "dft/udh_two_particle_density.h"
#include "base/basis.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include "require_boys_library.h"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <source_location>
#include <stdexcept>

namespace
{
    using namespace DFT::Gradient;
    using M=Eigen::MatrixXd;
    using V=std::vector<double>;
    bool ok=true;
    void check(bool value,std::source_location loc=std::source_location::current())
    { if(!value) { ok=false; std::cerr<<"FAIL U7 line "<<loc.line()<<'\n'; } }
    void near(double a,double b,double tol=2e-11,std::source_location loc=std::source_location::current())
    { check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tol*(1+std::abs(b)),loc); }
    template<class T> T require(std::expected<T,std::string> x)
    { if(!x) throw std::runtime_error(x.error()); return std::move(*x); }
    double dot(const M &a,const M &b) { return (a.array()*b.array()).sum(); }
    double dot(const V &a,const V &b) { return std::inner_product(a.begin(),a.end(),b.begin(),0.0); }
    std::size_t ix(int p,int q,int r,int s,int n)
    { return ((static_cast<std::size_t>(p)*n+q)*n+r)*n+s; }
    std::size_t ti(int i,int j,int a,int b,int oj,int va,int vb)
    { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
    void equal(const V &a,const V &b)
    { check(a.size()==b.size()); if(a.size()==b.size()) for(std::size_t k=0;k<a.size();++k) near(a[k],b[k]); }
    M inverse_root(const M &s)
    {
        Eigen::SelfAdjointEigenSolver<M> es(s);
        if(es.info()!=Eigen::Success || es.eigenvalues().minCoeff()<=0) throw std::runtime_error("metric failure");
        return es.eigenvectors()*es.eigenvalues().array().rsqrt().matrix().asDiagonal()*es.eigenvectors().transpose();
    }
    M symmetric(int n,double phase)
    {
        M m(n,n);
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) m(p,q)=.13*std::sin(phase+(p+1)*(q+1));
        return m;
    }
    struct Fixture
    {
        UDHPT2Amplitudes t;
        UDHOrbitalMatrices c,z;
        M s;
        double scale=.27,ax=.53;
    };
    Fixture fixture(int oa,int ob,int va,int vb,const M &metric)
    {
        Fixture x; x.s=metric; x.t={oa,ob,va,vb,{},{},{}};
        const int n=metric.rows();
        for(int spin=0;spin<2;++spin)
        {
            const int o=spin?ob:oa,v=spin?vb:va;
            M trial=M::Identity(n,n)+symmetric(n,.2+spin);
            const M q=trial.householderQr().householderQ()*M::Identity(n,n);
            (spin?x.c.beta:x.c.alpha)=inverse_root(metric)*q.leftCols(o+v);
            M z(v,o);
            for(int a=0;a<v;++a) for(int i=0;i<o;++i) z(a,i)=x.scale*.03*std::cos(1+spin+2*a+i);
            (spin?x.z.beta:x.z.alpha)=z;
        }
        const auto ss=[](int o,int v)
        {
            V t(static_cast<std::size_t>(o)*o*v*v);
            for(int i=0;i<o;++i) for(int j=i+1;j<o;++j) for(int a=0;a<v;++a) for(int b=a+1;b<v;++b)
            {
                const double u=.013*std::sin(1+i+2*j+3*a+5*b);
                t[ti(i,j,a,b,o,v,v)]=t[ti(j,i,b,a,o,v,v)]=u;
                t[ti(j,i,a,b,o,v,v)]=t[ti(i,j,b,a,o,v,v)]=-u;
            }
            return t;
        };
        x.t.aa=ss(oa,va); x.t.bb=ss(ob,vb);
        x.t.ab.resize(static_cast<std::size_t>(oa)*ob*va*vb);
        for(std::size_t k=0;k<x.t.ab.size();++k) x.t.ab[k]=.017*std::cos(.3+k);
        return x;
    }
    UDHRelaxedDifferenceDensity relaxed(const Fixture &x)
    { return require(build_udh_relaxed_difference_density(require(build_udh_pt2_dprime(x.t,x.scale)),x.z)); }
    UDHTwoParticleDensity gamma(const Fixture &x)
    { return require(build_udh_two_particle_density(x.t,x.scale,relaxed(x),x.c,x.s,x.ax)); }
    UDHOverlapDensity overlap(const Fixture &x)
    {
        // Independent frozen W: U6 already validates how W is built. U7
        // tests the AO consumer, not an unproved live stationary transfer.
        UDHOverlapDensity w;
        w.symmetric_mo={symmetric(x.c.alpha.cols(),.7),symmetric(x.c.beta.cols(),1.3)};
        w.raw_mo=w.symmetric_mo;
        w.symmetric_ao={x.c.alpha*w.symmetric_mo.alpha*x.c.alpha.transpose(),
                        x.c.beta*w.symmetric_mo.beta*x.c.beta.transpose()};
        w.total_ao=w.symmetric_ao.alpha+w.symmetric_ao.beta;
        return w;
    }
    std::array<const UDHERICoefficients*,6> channels(const UDHTwoParticleDensity &g)
    { return {&g.coulomb,&g.exchange_alpha,&g.exchange_beta,&g.pair_aa,&g.pair_ab,&g.pair_bb}; }
    std::array<const Eigen::VectorXd*,8> channels(const UDHNonXCGradient &g)
    { return {&g.one_electron,&g.overlap,&g.coulomb,&g.exchange_alpha,&g.exchange_beta,&g.pair_aa,&g.pair_ab,&g.pair_bb}; }
    // Literal eight-index AO coefficient oracle, independent of the staged
    // implementation. Deliberately no eightfold symmetry assumed here.
    V literal_pair(const Fixture &x,int sector)
    {
        const bool bs=sector==2,bt=sector!=0;
        const M &cs=bs?x.c.beta:x.c.alpha,&ct=bt?x.c.beta:x.c.alpha;
        const int os=bs?x.t.nocc_beta:x.t.nocc_alpha,ot=bt?x.t.nocc_beta:x.t.nocc_alpha;
        const int vs=bs?x.t.nvirt_beta:x.t.nvirt_alpha,vt=bt?x.t.nvirt_beta:x.t.nvirt_alpha,n=x.s.rows();
        const V &t=sector==0?x.t.aa:sector==1?x.t.ab:x.t.bb;
        V out(static_cast<std::size_t>(n)*n*n*n);
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            for(int i=0;i<os;++i) for(int j=0;j<ot;++j) for(int a=0;a<vs;++a) for(int b=0;b<vt;++b)
                out[ix(p,q,r,s,n)]+=(sector==1?2:1)*x.scale*t[ti(i,j,a,b,ot,vs,vt)]*
                    cs(p,i)*cs(q,os+a)*ct(r,j)*ct(s,ot+b);
        return out;
    }
    struct Integrals { M h,s; V eri; };
    // Independent scalar: build physical Fock J/K matrices and transformed
    // MO integrals, then contract D or unscaled amplitudes. No U7 tensor,
    // symmetrizer, consumer, U2 action or derivative routine is called.
    std::array<double,8> scalar(const Fixture &x,const UDHOverlapDensity &w,const Integrals &g)
    {
        const auto d=relaxed(x);
        const M da=x.c.alpha*d.symmetric_mo.alpha*x.c.alpha.transpose();
        const M db=x.c.beta*d.symmetric_mo.beta*x.c.beta.transpose();
        const M pa=x.c.alpha.leftCols(x.t.nocc_alpha)*x.c.alpha.leftCols(x.t.nocc_alpha).transpose();
        const M pb=x.c.beta.leftCols(x.t.nocc_beta)*x.c.beta.leftCols(x.t.nocc_beta).transpose();
        const int n=x.s.rows();
        M j=M::Zero(n,n),ka=j,kb=j;
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
        {
            j(p,q)+=g.eri[ix(p,q,r,s,n)]*(pa(r,s)+pb(r,s));
            ka(p,q)+=g.eri[ix(p,r,q,s,n)]*pa(r,s);
            kb(p,q)+=g.eri[ix(p,r,q,s,n)]*pb(r,s);
        }
        std::array<double,8> out{dot(da+db,g.h),dot(w.total_ao,g.s),dot(da+db,j),-x.ax*dot(da,ka),-x.ax*dot(db,kb),0,0,0};
        for(int spin=0;spin<3;++spin)
        {
            const M &cs=spin==2?x.c.beta:x.c.alpha,&ct=spin==0?x.c.alpha:x.c.beta;
            const int os=spin==2?x.t.nocc_beta:x.t.nocc_alpha,ot=spin==0?x.t.nocc_alpha:x.t.nocc_beta;
            const int vs=spin==2?x.t.nvirt_beta:x.t.nvirt_alpha,vt=spin==0?x.t.nvirt_alpha:x.t.nvirt_beta;
            const V &t=spin==0?x.t.aa:spin==1?x.t.ab:x.t.bb;
            for(int i=0;i<os;++i) for(int a=0;a<vs;++a) for(int jmo=0;jmo<ot;++jmo) for(int b=0;b<vt;++b)
            {
                double mo=0;
                for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
                    mo+=cs(p,i)*cs(q,os+a)*ct(r,jmo)*ct(s,ot+b)*g.eri[ix(p,q,r,s,n)];
                out[5+spin]+=(spin==1?2:1)*x.scale*t[ti(i,jmo,a,b,ot,vs,vt)]*mo;
            }
        }
        return out;
    }
    Integrals model(int n,double phase)
    {
        Integrals g{symmetric(n,phase),symmetric(n,phase+.2),V(static_cast<std::size_t>(n)*n*n*n)};
        const M a=symmetric(n,phase+.3),b=symmetric(n,phase+.8);
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            g.eri[ix(p,q,r,s,n)]=.7*a(p,q)*a(r,s)+.4*b(p,q)*b(r,s);
        return g;
    }
    Integrals displaced(const Integrals &g,const Integrals &dg,double step)
    {
        Integrals out{g.h+step*dg.h,g.s+step*dg.s,g.eri};
        for(std::size_t k=0;k<out.eri.size();++k) out.eri[k]+=step*dg.eri[k];
        return out;
    }
    void audit(const Fixture &x)
    {
        const auto g=gamma(x); const auto w=overlap(x); const int n=x.s.rows();
        for(int spin=0;spin<3;++spin) equal(channels(g)[3+spin]->raw_ao,literal_pair(x,spin));
        const auto base=model(n,.2),direction=model(n,1.1);
        const auto value=scalar(x,w,base);
        for(int ch=0;ch<6;++ch)
        {
            near(dot(channels(g)[ch]->raw_ao,base.eri),value[ch+2]);
            near(dot(channels(g)[ch]->symmetric_ao,base.eri),value[ch+2]);
        }
        const auto grad=require(build_udh_non_xc_gradient(g,w,{{direction.h},{direction.s},{direction.eri}}));
        const auto ledger=channels(grad);
        double worst=0;
        for(double step:{1e-3,2e-4})
        {
            const auto p=scalar(x,w,displaced(base,direction,step)),m=scalar(x,w,displaced(base,direction,-step));
            double total=0;
            for(int ch=0;ch<8;++ch)
            {
                const double fd=(p[ch]-m[ch])/(2*step);
                near((*ledger[ch])(0),fd,2e-9); total+=fd;
                worst=std::max(worst,std::abs((*ledger[ch])(0)-fd));
            }
            near(grad.total(0),total,2e-9);
        }
        // Every independent eightfold ERI basis direction. Assign each
        // orbit element ONCE, including repeated-index orbits of size <8.
        for(int p=0;p<n;++p) for(int q=0;q<=p;++q) for(int r=0;r<n;++r) for(int s=0;s<=r;++s)
        {
            if(p*(p+1)/2+q<r*(r+1)/2+s) continue;
            V unit(base.eri.size(),0);
            for(auto k:{ix(p,q,r,s,n),ix(q,p,r,s,n),ix(p,q,s,r,n),ix(q,p,s,r,n),
                        ix(r,s,p,q,n),ix(s,r,p,q,n),ix(r,s,q,p,n),ix(s,r,q,p,n)}) unit[k]=1;
            for(const auto *ch:channels(g)) near(dot(ch->raw_ao,unit),dot(ch->symmetric_ao,unit));
        }
        // a_x must touch only the two same-spin exchange channels.
        for(double ax:{0.,.17,.53,1.})
        {
            auto y=x; y.ax=ax; const auto gy=gamma(y);
            equal(gy.coulomb.raw_ao,g.coulomb.raw_ao);
            for(int ch=3;ch<6;++ch) equal(channels(gy)[ch]->raw_ao,channels(g)[ch]->raw_ao);
            for(int ch:{1,2}) for(std::size_t k=0;k<base.eri.size();++k)
                near(channels(gy)[ch]->raw_ao[k]*x.ax,channels(g)[ch]->raw_ao[k]*ax);
            const auto ref=scalar(y,w,direction);
            const auto got=require(build_udh_non_xc_gradient(gy,w,{{direction.h},{direction.s},{direction.eri}}));
            for(int ch=0;ch<8;++ch) near((*channels(got)[ch])(0),ref[ch]);
        }
        auto zero=x; zero.scale=0; zero.z.alpha.setZero(); zero.z.beta.setZero();
        const auto gzero=gamma(zero);
        for(const auto *ch:channels(gzero)) for(double v:ch->raw_ao) near(v,0);
        // Scaling the whole correction scales Dprime, Z, and pair once.
        auto twice=x; twice.scale*=2; twice.z.alpha*=2; twice.z.beta*=2;
        const auto g2=gamma(twice);
        for(int ch=0;ch<6;++ch) for(std::size_t k=0;k<base.eri.size();++k)
            near(channels(g2)[ch]->raw_ao[k],2*channels(g)[ch]->raw_ao[k]);
        auto dz=x; dz.z.alpha*=3; dz.z.beta*=3;
        const auto gz=gamma(dz);
        for(int ch=3;ch<6;++ch) equal(channels(gz)[ch]->raw_ao,channels(g)[ch]->raw_ao);
        // Relabeling ab transposes the electron pairs, not the raw tensor.
        auto sw=x; std::swap(sw.c.alpha,sw.c.beta); std::swap(sw.z.alpha,sw.z.beta);
        sw.t={x.t.nocc_beta,x.t.nocc_alpha,x.t.nvirt_beta,x.t.nvirt_alpha,x.t.bb,V(x.t.ab.size()),x.t.aa};
        for(int i=0;i<x.t.nocc_alpha;++i) for(int j=0;j<x.t.nocc_beta;++j)
            for(int a=0;a<x.t.nvirt_alpha;++a) for(int b=0;b<x.t.nvirt_beta;++b)
                sw.t.ab[ti(j,i,b,a,x.t.nocc_alpha,x.t.nvirt_beta,x.t.nvirt_alpha)]=
                    x.t.ab[ti(i,j,a,b,x.t.nocc_beta,x.t.nvirt_alpha,x.t.nvirt_beta)];
        const auto gs=gamma(sw);
        equal(gs.coulomb.raw_ao,g.coulomb.raw_ao); equal(gs.exchange_alpha.raw_ao,g.exchange_beta.raw_ao);
        equal(gs.exchange_beta.raw_ao,g.exchange_alpha.raw_ao); equal(gs.pair_aa.raw_ao,g.pair_bb.raw_ao);
        equal(gs.pair_bb.raw_ao,g.pair_aa.raw_ao); equal(gs.pair_ab.symmetric_ao,g.pair_ab.symmetric_ao);
        std::cout<<"U7 o/v "<<x.t.nocc_alpha<<'/'<<x.t.nvirt_alpha<<" alpha "<<x.t.nocc_beta<<'/'<<x.t.nvirt_beta
                 <<" beta fixed-contract FD max "<<worst<<'\n';
    }
    void closed_shell()
    {
        auto x=fixture(2,2,2,2,M::Identity(4,4)); x.c.beta=x.c.alpha; x.z.beta=x.z.alpha;
        HartreeFock::Correlation::RMP2Result r; r.n_occ=2; r.n_virt=2; r.t2.resize(16);
        // t_ijab = t_jiba is the spatial pair permutation, not separate
        // occupied/virtual symmetry. Spin adaptation follows paper Eq. 36.
        for(int i=0;i<2;++i) for(int j=0;j<2;++j) for(int a=0;a<2;++a) for(int b=0;b<2;++b)
            r.t2[ti(i,j,a,b,2,2,2)]=.02*std::sin((1+i+a*2)*(1+j+b*2));
        x.t.ab=r.t2;
        for(int i=0;i<2;++i) for(int j=0;j<2;++j) for(int a=0;a<2;++a) for(int b=0;b<2;++b)
            x.t.aa[ti(i,j,a,b,2,2,2)]=x.t.bb[ti(i,j,a,b,2,2,2)]=
                r.t2[ti(i,j,a,b,2,2,2)]-r.t2[ti(i,j,b,a,2,2,2)];
        const auto g=gamma(x); const auto a=require(build_dh_pt2_amplitude_density(r,x.scale));
        const auto d=require(build_dh_relaxed_difference_density(a,2*x.z.alpha));
        const M p=2*x.c.alpha.leftCols(2)*x.c.alpha.leftCols(2).transpose();
        const auto restricted=require(build_dh_eq46_47_two_particle_density(d,a,x.c.alpha,p,x.ax));
        V sep(256),pair(256);
        for(std::size_t k=0;k<sep.size();++k)
        {
            sep[k]=g.coulomb.raw_ao[k]+g.exchange_alpha.raw_ao[k]+g.exchange_beta.raw_ao[k];
            pair[k]=g.pair_aa.raw_ao[k]+g.pair_ab.raw_ao[k]+g.pair_bb.raw_ao[k];
        }
        equal(sep,restricted.separable_raw_ao); equal(pair,restricted.nonseparable_raw_ao);
        std::cout<<"U7 closed-shell raw Eq34/35 reduction to RKS Eq46/47\n";
    }
    void errors()
    {
        auto x=fixture(2,1,2,3,M::Identity(4,4)); const auto d=relaxed(x); auto g=gamma(x); auto w=overlap(x);
        const auto dg=model(4,.5); DHEq33NonXCDerivatives dx{{dg.h},{dg.s},{dg.eri}};
        check(!build_udh_two_particle_density(x.t,x.scale,d,x.c,x.s,-.1));
        check(!build_udh_two_particle_density(x.t,std::numeric_limits<double>::quiet_NaN(),d,x.c,x.s,x.ax));
        check(!build_udh_two_particle_density(x.t,2*x.scale,d,x.c,x.s,x.ax));
        auto t=x.t; t.ab.pop_back(); check(!build_udh_two_particle_density(t,x.scale,d,x.c,x.s,x.ax));
        t={std::numeric_limits<int>::max(),0,0,0,{},{},{}};
        check(!build_udh_two_particle_density(t,x.scale,d,x.c,x.s,x.ax));
        auto c=x.c; c.beta(0,0)+=.1; check(!build_udh_two_particle_density(x.t,x.scale,d,c,x.s,x.ax));
        auto bad_d=d; bad_d.symmetric_mo.alpha(0,0)+=.1;
        check(!build_udh_two_particle_density(x.t,x.scale,bad_d,x.c,x.s,x.ax));
        check(require(build_udh_non_xc_gradient(g,w,{})).total.size()==0);
        auto bad=dx; bad.overlap_ao.clear(); check(!build_udh_non_xc_gradient(g,w,bad));
        bad=dx; bad.hamiltonian_ao[0](0,1)+=.1; check(!build_udh_non_xc_gradient(g,w,bad));
        bad=dx; bad.eri_ao[0][1]+=.1; check(!build_udh_non_xc_gradient(g,w,bad));
        bad=dx; bad.eri_ao[0][0]=std::numeric_limits<double>::infinity(); check(!build_udh_non_xc_gradient(g,w,bad));
        auto bad_g=g; bad_g.pair_ab.symmetric_ao[0]+=.1; check(!build_udh_non_xc_gradient(bad_g,w,dx));
        bad_g=g; bad_g.pair_aa.raw_ao.pop_back(); check(!build_udh_non_xc_gradient(bad_g,w,dx));
        bad_g=g; bad_g.relaxed_ao.total(0,0)+=.1; check(!build_udh_non_xc_gradient(bad_g,w,dx));
        w.total_ao(0,0)+=.1; check(!build_udh_non_xc_gradient(g,w,dx));
    }

    HartreeFock::Calculator molecule(int atom=-1,int cart=0,double step=0)
    {
        // Non-symmetric water+ geometry, used only for physical AO integral
        // derivatives. Metric-orthonormal trial frames are NOT a converged UKS state.
        HartreeFock::Calculator calc; auto &m=calc._molecule;
        m.natoms=3; m.charge=1; m.multiplicity=2;
        m.atomic_numbers.resize(3); m.atomic_numbers<<8,1,1;
        m.atomic_masses.resize(3); m.atomic_masses<<16.,1.,1.;
        m.coordinates.resize(3,3); m.coordinates<<.12,-.07,.11, .91,.23,-.43, -.34,.86,.27;
        calc._basis._basis=HartreeFock::BasisType::Cartesian; calc.prepare_coordinates();
        if(atom>=0) m._coordinates(atom,cart)+=step;
        m.set_standard_from_bohr(m._coordinates);
        calc._shells=require(HartreeFock::BasisFunctions::read_gbs_basis(
            (std::filesystem::path(get_basis_path())/"sto-3g").string(),m,calc._basis._basis));
        return calc;
    }
    Integrals integrals(const HartreeFock::Calculator &calc)
    {
        const auto pairs=build_shellpairs(calc._shells);
        const auto n=calc._shells.nbasis();
        const auto engine=HartreeFock::IntegralMethod::ObaraSaika;
        const auto [s,t]=_compute_1e(pairs,n,engine);
        return {t+_compute_nuclear_attraction(pairs,n,calc._molecule,engine),s,
            _compute_2e(pairs,n,engine,HartreeFock::ERIKernel::Coulomb,0.,0.,nullptr)};
    }
    DHEq33NonXCDerivatives derivatives(const HartreeFock::Calculator &calc)
    {
        const int n=calc._shells.nbasis(); const auto &bfs=calc._shells._basis_functions;
        std::vector<int> owner(n,-1);
        for(int mu=0;mu<n;++mu) for(int a=0;a<3;++a)
            if((bfs[mu]._shell->_center-calc._molecule._standard.row(a).transpose()).squaredNorm()<1e-12) owner[mu]=a;
        if(std::find(owner.begin(),owner.end(),-1)!=owner.end()) throw std::runtime_error("AO owner not found");
        DHEq33NonXCDerivatives out;
        out.hamiltonian_ao.assign(9,M::Zero(n,n)); out.overlap_ao.assign(9,M::Zero(n,n));
        out.eri_ao.assign(9,V(static_cast<std::size_t>(n)*n*n*n));
        for(int mu=0;mu<n;++mu) for(int nu=0;nu<n;++nu)
        {
            const HartreeFock::ShellPair mn(bfs[mu],bfs[nu]),nm(bfs[nu],bfs[mu]);
            const auto dm=HartreeFock::ObaraSaika::_compute_1e_deriv_A(mn),dn=HartreeFock::ObaraSaika::_compute_1e_deriv_A(nm);
            const auto vm=HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(mn,calc._molecule);
            const auto vn=HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(nm,calc._molecule);
            for(int a=0;a<3;++a) for(int k=0;k<3;++k)
            {
                if(owner[mu]==a) { out.overlap_ao[3*a+k](mu,nu)+=dm[k]; out.hamiltonian_ao[3*a+k](mu,nu)+=dm[k+3]+vm[k]; }
                if(owner[nu]==a) { out.overlap_ao[3*a+k](mu,nu)+=dn[k]; out.hamiltonian_ao[3*a+k](mu,nu)+=dn[k+3]+vn[k]; }
                const Eigen::Vector3d center=calc._molecule._standard.row(a).transpose();
                out.hamiltonian_ao[3*a+k](mu,nu)+=HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(
                    mn,center,calc._molecule.atomic_numbers[a],k);
            }
            for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            {
                const auto dg=HartreeFock::ObaraSaika::_compute_eri_deriv_elem(mn,HartreeFock::ShellPair(bfs[r],bfs[s]));
                const int atoms[4]={owner[mu],owner[nu],owner[r],owner[s]};
                for(int a=0;a<4;++a) for(int k=0;k<3;++k) out.eri_ao[3*atoms[a]+k][ix(mu,nu,r,s,n)]+=dg[3*a+k];
            }
        }
        return out;
    }
    void physical()
    {
        const auto calc=molecule(); const auto base=integrals(calc); const int n=base.s.rows();
        auto x=fixture(5,4,n-5,n-4,base.s);
        const auto g=gamma(x); const auto w=overlap(x); const auto dx=derivatives(calc);
        const auto grad=require(build_udh_non_xc_gradient(g,w,dx));
        const auto values=channels(grad);
        // Separate direct Planck J, alpha K and beta K at the reference.
        const auto pairs=build_shellpairs(calc._shells);
        const auto j=_compute_2e_j_direct(pairs,g.ground_ao.alpha+g.ground_ao.beta,n,
            HartreeFock::IntegralMethod::ObaraSaika,HartreeFock::ERIKernel::Coulomb,0.,0.,nullptr);
        const auto ka=_compute_2e_k_direct(pairs,g.ground_ao.alpha,n,
            HartreeFock::IntegralMethod::ObaraSaika,HartreeFock::ERIKernel::Coulomb,0.,0.,nullptr);
        const auto kb=_compute_2e_k_direct(pairs,g.ground_ao.beta,n,
            HartreeFock::IntegralMethod::ObaraSaika,HartreeFock::ERIKernel::Coulomb,0.,0.,nullptr);
        near(dot(g.coulomb.raw_ao,base.eri),dot(g.relaxed_ao.total,j));
        near(dot(g.exchange_alpha.raw_ao,base.eri),-x.ax*dot(g.relaxed_ao.spin.alpha,ka));
        near(dot(g.exchange_beta.raw_ao,base.eri),-x.ax*dot(g.relaxed_ao.spin.beta,kb));
        for(double step:{1e-3,2e-4})
        {
            std::array<double,8> worst{}; double total_worst=0;
            for(int a=0;a<3;++a) for(int k=0;k<3;++k)
            {
                const auto p=scalar(x,w,integrals(molecule(a,k,step))),m=scalar(x,w,integrals(molecule(a,k,-step)));
                double sum=0;
                for(int ch=0;ch<8;++ch)
                {
                    const double fd=(p[ch]-m[ch])/(2*step),err=std::abs(fd-(*values[ch])(3*a+k));
                    worst[ch]=std::max(worst[ch],err); sum+=fd;
                    near(fd,(*values[ch])(3*a+k),2e-6);
                }
                total_worst=std::max(total_worst,std::abs(sum-grad.total(3*a+k)));
                near(sum,grad.total(3*a+k),4e-6);
            }
            std::cout<<"U7 nonsymmetric water+/STO-3G frozen-contract FD h="<<step<<" channel errors";
            for(double e:worst) std::cout<<' '<<e;
            std::cout<<" total "<<total_worst<<'\n';
        }
        double worst_translation=0;
        for(const auto *v:values) for(int k=0;k<3;++k)
        {
            const double sum=(*v)(k)+(*v)(3+k)+(*v)(6+k);
            near(sum,0.,2e-8); worst_translation=std::max(worst_translation,std::abs(sum));
        }
        for(int k=0;k<3;++k) near(grad.total(k)+grad.total(3+k)+grad.total(6+k),0.,2e-8);
        std::cout<<"U7 rigid-translation channel max "<<worst_translation<<'\n';
    }
}
int main()
{
    if (!require_boys_library())
        return 1;
    try
    {
        for(const auto dims:{std::array{2,2,2,2},std::array{2,1,2,3},std::array{1,2,2,2},
                             std::array{2,0,2,3},std::array{2,1,0,3},std::array{1,0,3,0},std::array{0,0,0,0}})
        {
            const int n=std::max({dims[0]+dims[2],dims[1]+dims[3],1});
            const M q=symmetric(n,.4); const M s=M::Identity(n,n)+q*q.transpose();
            audit(fixture(dims[0],dims[1],dims[2],dims[3],s));
        }
        closed_shell(); errors(); physical();
    }
    catch(const std::exception &e) { std::cerr<<"FAIL U7 "<<e.what()<<'\n'; return 1; }
    if(ok) std::cout<<"PASS U7 Eq34/35 tensors and fixed-contract non-XC Eq33\n";
    return ok?0:1;
}
