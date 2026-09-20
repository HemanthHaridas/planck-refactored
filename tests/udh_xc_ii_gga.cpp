#include "dft/udh_xc_ii.h"
#include "dft/udh_xc_ii_gga_detail.h"
#include "dft/dh_pt2_gradient.h"
#include "dft/ks_matrix.h"
#include "base/basis.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>

namespace
{
    using namespace DFT;
    using namespace DFT::Gradient;
    using M=Eigen::MatrixXd;
    using Spin=UDHSpinMatrices;
    bool ok=true;
    void check(bool pass,std::source_location loc=std::source_location::current())
    { if(!pass) { ok=false; std::cerr<<"FAIL U8b line "<<loc.line()<<'\n'; } }
    void near(double a,double b,double tol=2e-10,std::source_location loc=std::source_location::current())
    { check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tol*(1+std::abs(b)),loc); }
    template<class T> T require(std::expected<T,std::string> result)
    { if(!result) throw std::runtime_error(result.error()); return std::move(*result); }
    double dot(const M &a,const M &b) { return (a.array()*b.array()).sum(); }
    XC::Functional functional(const char *name,XC::Spin spin=XC::Spin::Polarized)
    { return require(XC::Functional::create(require(XC::functional_id(name)),spin)); }
    HartreeFock::Calculator molecule(int atom=-1,int axis=0,double step=0)
    {
        HartreeFock::Calculator calc; auto &m=calc._molecule;
        m.natoms=3; m.charge=1; m.multiplicity=2;
        m.atomic_numbers.resize(3); m.atomic_numbers<<8,1,1;
        m.atomic_masses.resize(3); m.atomic_masses<<16.,1.,1.;
        m.coordinates.resize(3,3); m.coordinates<<.12,-.07,.11, .91,.23,-.43, -.34,.86,.27;
        calc._basis._basis=HartreeFock::BasisType::Cartesian; calc.prepare_coordinates();
        if(atom>=0) m._coordinates(atom,axis)+=step;
        m.set_standard_from_bohr(m._coordinates);
        calc._shells=require(HartreeFock::BasisFunctions::read_gbs_basis(
            (std::filesystem::path(get_basis_path())/"sto-3g").string(),m,calc._basis._basis));
        return calc;
    }
    struct Fixture
    {
        HartreeFock::Calculator calc;
        MolecularGrid grid;
        AOGridEvaluation ao;
        AOGridHessian hess;
        Spin p,d;
    };
    Fixture fixture()
    {
        Fixture f; f.calc=molecule();
        f.grid=require(MakeMolecularGrid(f.calc._molecule,GridLevel::Coarse));
        f.ao=require(evaluate_ao_basis_on_grid(f.calc._shells,f.grid));
        f.hess=require(evaluate_ao_hessian_on_grid(f.calc._shells,f.grid));
        const auto pairs=build_shellpairs(f.calc._shells); const int n=f.ao.nbasis();
        const auto engine=HartreeFock::IntegralMethod::ObaraSaika;
        const auto [s,t]=_compute_1e(pairs,n,engine);
        Eigen::SelfAdjointEigenSolver<M> se(s);
        if(se.info()!=Eigen::Success || se.eigenvalues().minCoeff()<=0) throw std::runtime_error("metric failure");
        const M orth=se.operatorInverseSqrt();
        Eigen::SelfAdjointEigenSolver<M> he(orth*(t+_compute_nuclear_attraction(pairs,n,f.calc._molecule,engine))*orth);
        if(he.info()!=Eigen::Success) throw std::runtime_error("core orbitals failed");
        const M c=orth*he.eigenvectors();
        // Unequal 5/4 spin occupations. Signed non-diagonal D is a frozen
        // correction trial, not an SCF density or a live Z-vector solution.
        f.p={c.leftCols(5)*c.leftCols(5).transpose(),c.leftCols(4)*c.leftCols(4).transpose()};
        const auto cross=[&](int i,int j)->M { return c.col(i)*c.col(j).transpose()+c.col(j)*c.col(i).transpose(); };
        f.d={.03*cross(0,5)-.02*cross(1,4),-.025*cross(1,6)+.013*cross(2,3)};
        return f;
    }
    UDHGGAXCIIInputs inputs(const Fixture &f,const Spin &p,const Spin &d,const XC::Functional &x,const XC::Functional &c)
    { return {{&f.calc._molecule,&f.calc._shells,&f.grid,&f.ao,p,d,&x,&c},&f.hess}; }
    std::array<const M*,5> channels(const UDHCompleteXCIIGradient &q)
    { return {&q.p_side_fixed,&q.d_side_ao,&q.becke_partition,&q.point_translation,&q.total}; }
    void equal(const UDHCompleteXCIIGradient &a,const UDHCompleteXCIIGradient &b,double factor=1)
    { for(int ch=0;ch<5;++ch) near((*channels(a)[ch]-factor*(*channels(b)[ch])).norm(),0); }

    // Ordinary vxc only: does not call any XC-II, fxc, nuclear derivative,
    // density-response or partition-derivative implementation.
    double scalar(const MolecularGrid &weights,const AOGridEvaluation &p_ao,const AOGridEvaluation &d_ao,
        const Spin &p,const Spin &d,const XC::Functional &x,const XC::Functional &c)
    {
        const auto xc=require(evaluate_xc_on_grid(weights,p_ao,p.alpha,p.beta,x,c));
        const auto delta=require(evaluate_density_on_grid(d_ao,d.alpha,d.beta));
        double value=0;
        for(Eigen::Index k=0;k<weights.points.rows();++k)
        {
            const Eigen::Vector3d ua{xc.density.alpha.grad_x(k),xc.density.alpha.grad_y(k),xc.density.alpha.grad_z(k)};
            const Eigen::Vector3d ub{xc.density.beta.grad_x(k),xc.density.beta.grad_y(k),xc.density.beta.grad_z(k)};
            const Eigen::Vector3d va{delta.alpha.grad_x(k),delta.alpha.grad_y(k),delta.alpha.grad_z(k)};
            const Eigen::Vector3d vb{delta.beta.grad_x(k),delta.beta.grad_y(k),delta.beta.grad_z(k)};
            value+=weights.points(k,3)*(xc.vrho(k,0)*delta.alpha.rho(k)+xc.vrho(k,1)*delta.beta.rho(k)+
                2*xc.vsigma(k,0)*ua.dot(va)+xc.vsigma(k,1)*(ua.dot(vb)+ub.dot(va))+2*xc.vsigma(k,2)*ub.dot(vb));
        }
        return value;
    }
    // Monitor each active functional's density, diagonal/total sigma and
    // spin-saturation thresholds. Never silently screen a changed set.
    std::size_t crossings(const AOGridEvaluation &base,const AOGridEvaluation &changed,const Spin &p,
        const XC::Functional &f,std::array<std::size_t,3> &types)
    {
        const auto a=require(evaluate_density_on_grid(base,p.alpha,p.beta));
        const auto b=require(evaluate_density_on_grid(changed,p.alpha,p.beta));
        std::size_t count=0;
        for(Eigen::Index k=0;k<a.npoints();++k)
        {
            for(auto values:{std::pair{a.alpha.rho(k),b.alpha.rho(k)},std::pair{a.beta.rho(k),b.beta.rho(k)},
                             std::pair{a.total.rho(k),b.total.rho(k)}})
                if((values.first<=f.get()->dens_threshold)!=(values.second<=f.get()->dens_threshold)) { ++count; ++types[0]; }
            const auto sigma=[k](const DensityChannelOnGrid &d)
            { return d.grad_x(k)*d.grad_x(k)+d.grad_y(k)*d.grad_y(k)+d.grad_z(k)*d.grad_z(k); };
            for(auto values:{std::pair{sigma(a.alpha),sigma(b.alpha)},std::pair{sigma(a.beta),sigma(b.beta)},
                             std::pair{sigma(a.total),sigma(b.total)}})
                if((values.first<=f.get()->sigma_threshold)!=(values.second<=f.get()->sigma_threshold)) { ++count; ++types[1]; }
            if(a.total.rho(k)>f.get()->dens_threshold && b.total.rho(k)>f.get()->dens_threshold)
                for(auto values:{std::pair{a.alpha.rho(k),b.alpha.rho(k)},std::pair{a.beta.rho(k),b.beta.rho(k)}})
                    if((2*values.first/a.total.rho(k)<=f.get()->zeta_threshold)!=
                        (2*values.second/b.total.rho(k)<=f.get()->zeta_threshold)) { ++count; ++types[2]; }
        }
        return count;
    }
    void kernel_slots(const XC::Functional &f)
    {
        using DFT::Gradient::detail::GGA5;
        const std::vector<double> rho{.42,.17},sigma{.12,.025,.08};
        std::vector<double> rr,rs,ss;
        if(!f.evaluate_gga_fxc(rho,sigma,1,rr,rs,ss)) throw std::runtime_error("GGA kernel failed");
        const auto hessian=DFT::Gradient::detail::unpack_udh_gga_kernel(rr.data(),rs.data(),ss.data());
        std::array<double,25> worst{};
        for(int source=0;source<5;++source) for(double h:{1e-4,2e-5})
        {
            auto rp=rho,rm=rho,sp=sigma,sm=sigma;
            if(source<2) { rp[source]+=h; rm[source]-=h; }
            else { sp[source-2]+=h; sm[source-2]-=h; }
            std::vector<double> e,vrp,vsp,vrm,vsm;
            if(!f.evaluate_gga_exc_vxc(rp,sp,1,e,vrp,vsp) || !f.evaluate_gga_exc_vxc(rm,sm,1,e,vrm,vsm))
                throw std::runtime_error("GGA vxc direction failed");
            for(int target=0;target<5;++target)
            {
                const double fd=target<2?(vrp[target]-vrm[target])/(2*h):(vsp[target-2]-vsm[target-2])/(2*h);
                near(fd,hessian(target,source),3e-6);
                worst[5*target+source]=std::max(worst[5*target+source],std::abs(fd-hessian(target,source)));
            }
        }
        std::cout<<"U8b "<<f.name()<<" full 5x5 kernel FD errors";
        for(double e:worst) std::cout<<' '<<e;
        std::cout<<'\n';
    }
    void point_chain_rule()
    {
        using namespace DFT::Gradient::detail;
        // Nonzero, distinct entries in every packed slot. Check the packing
        // independently before differentiating an arbitrary quadratic f(y).
        const double rr[]{.11,.12,.13},rs[]{.21,.22,.23,.24,.25,.26},ss[]{.31,.32,.33,.34,.35,.36};
        const GGA55 h=unpack_udh_gga_kernel(rr,rs,ss);
        GGA55 expected;
        expected<<.11,.12,.21,.22,.23, .12,.13,.24,.25,.26,
            .21,.24,.31,.32,.33, .22,.25,.32,.34,.35, .23,.26,.33,.35,.36;
        near((h-expected).norm(),0,1e-14);
        struct Fields { Eigen::Vector2d r,d; SpinGradients u,v; };
        Fields base; base.r<<.4,.2; base.d<<-.03,.07;
        base.u<<.17,.12,-.23,.08,.14,-.19; base.v<<-.06,.04,.08,.01,.03,-.07;
        GGA5 linear; linear<<.3,-.2,.17,-.11,.27;
        const auto y=[](const Fields &s) { GGA5 out;
            out<<s.r(0),s.r(1),s.u.col(0).squaredNorm(),s.u.col(0).dot(s.u.col(1)),s.u.col(1).squaredNorm(); return out; };
        const auto value=[&](const Fields &s)
        {
            const GGA5 fy=linear+expected*y(s);
            double result=fy(0)*s.d(0)+fy(1)*s.d(1);
            for(int k=0;k<3;++k) result+=2*fy(2)*s.u(k,0)*s.v(k,0)+
                fy(3)*(s.u(k,0)*s.v(k,1)+s.u(k,1)*s.v(k,0))+2*fy(4)*s.u(k,1)*s.v(k,1);
            return result;
        };
        const auto partial=udh_gga_xc_ii_partials(base.d,base.u,base.v,linear+h*y(base),h);
        near(partial.scalar,value(base),1e-14);
        const auto component=[](Fields &s,int k)->double& {
            if(k<2) return s.r(k); if(k<4) return s.d(k-2);
            if(k<10) return s.u((k-4)%3,(k-4)/3); return s.v((k-10)%3,(k-10)/3);
        };
        double worst=0;
        for(int k=0;k<16;++k) for(double step:{1e-3,2e-4})
        {
            auto p=base,m=base; component(p,k)+=step; component(m,k)-=step;
            const double ref=k<2?partial.p_rho(k):k<4?partial.d_rho(k-2):
                k<10?partial.p_gradient((k-4)%3,(k-4)/3):partial.d_gradient((k-10)%3,(k-10)/3);
            const double fd=(value(p)-value(m))/(2*step);
            near(fd,ref,2e-7); worst=std::max(worst,std::abs(fd-ref));
        }
        std::cout<<"U8b polynomial local P/D chain-rule FD max "<<worst<<'\n';
    }
    void audit(Fixture &f,const XC::Functional &x,const XC::Functional &c)
    {
        const M zero=M::Zero(f.ao.nbasis(),f.ao.nbasis());
        const std::array<Spin,3> d{Spin{f.d.alpha,zero},Spin{zero,f.d.beta},f.d};
        std::array<UDHCompleteXCIIGradient,3> analytic;
        for(int trial=0;trial<3;++trial)
            analytic[trial]=require(build_udh_gga_complete_xc_ii_gradient(inputs(f,f.p,d[trial],x,c)));
        for(int ch=0;ch<5;++ch)
            near((*channels(analytic[2])[ch]-*channels(analytic[0])[ch]-*channels(analytic[1])[ch]).norm(),0);
        // Scalar agrees with the independently assembled ordinary XC matrix.
        const auto xc=require(evaluate_xc_on_grid(f.grid,f.ao,f.p.alpha,f.p.beta,x,c));
        const auto v=require(assemble_xc_matrix(f.grid,f.ao,xc));
        near(scalar(f.grid,f.ao,f.ao,f.p,f.d,x,c),dot(f.d.alpha,v.alpha)+dot(f.d.beta,v.beta));
        Spin ps{f.p.beta,f.p.alpha},ds{f.d.beta,f.d.alpha};
        equal(require(build_udh_gga_complete_xc_ii_gradient(inputs(f,ps,ds,x,c))),analytic[2]);
        Spin doubled{2*f.d.alpha,2*f.d.beta};
        equal(require(build_udh_gga_complete_xc_ii_gradient(inputs(f,f.p,doubled,x,c))),analytic[2],2);
        const auto z=require(build_udh_gga_complete_xc_ii_gradient(inputs(f,f.p,{zero,zero},x,c)));
        for(const auto *m:channels(z)) near(m->norm(),0);
        // The first rebuilt run found crossings at 1e-3 but none at 2e-4.
        // Retain the coarse step as a diagnostic. Acceptance still requires
        // TWO smooth steps, with unchanged channel-error bounds and masks.
        for(double h:{1e-3,2e-4,1e-4})
        {
            const bool acceptance_step=h<=2e-4;
            std::array<std::size_t,3> mask_types{};
            const auto count_crossings=[&](const AOGridEvaluation &changed)
            { return crossings(f.ao,changed,f.p,x,mask_types)+
                (x.is_combined_exchange_correlation()?0:crossings(f.ao,changed,f.p,c,mask_types)); };
            std::array<std::array<double,5>,3> worst{}; std::size_t changed_masks=0;
            for(int atom=0;atom<3;++atom) for(int axis=0;axis<3;++axis)
            {
                std::array<std::array<std::array<double,5>,3>,2> values{};
                for(int side=0;side<2;++side)
                {
                    const double step=side?-h:h; const auto displaced=molecule(atom,axis,step);
                    const auto full_grid=require(MakeMolecularGrid(displaced._molecule,GridLevel::Coarse));
                    check(full_grid.owner.size()==f.grid.owner.size());
                    check((full_grid.owner.array()==f.grid.owner.array()).all());
                    near((full_grid.atomic_weights-f.grid.atomic_weights).norm(),0);
                    const auto basis_ao=require(evaluate_ao_basis_on_grid(displaced._shells,f.grid));
                    auto point_grid=f.grid;
                    for(Eigen::Index k=0;k<point_grid.points.rows();++k)
                        if(point_grid.owner(k)==atom) point_grid.points(k,axis)+=step;
                    const auto point_ao=require(evaluate_ao_basis_on_grid(f.calc._shells,point_grid));
                    const auto full_ao=require(evaluate_ao_basis_on_grid(displaced._shells,full_grid));
                    changed_masks+=count_crossings(basis_ao)+count_crossings(point_ao)+count_crossings(full_ao);
                    for(int trial=0;trial<3;++trial)
                    {
                        auto &q=values[side][trial];
                        q[0]=scalar(f.grid,basis_ao,f.ao,f.p,d[trial],x,c); // move P AOs only
                        q[1]=scalar(f.grid,f.ao,basis_ao,f.p,d[trial],x,c); // move D AOs only
                        q[2]=scalar(full_grid,f.ao,f.ao,f.p,d[trial],x,c); // weight only, fields frozen
                        q[3]=scalar(f.grid,point_ao,point_ao,f.p,d[trial],x,c); // owner point only
                        q[4]=scalar(full_grid,full_ao,full_ao,f.p,d[trial],x,c); // full geometry
                    }
                }
                for(int trial=0;trial<3;++trial) for(int ch=0;ch<5;++ch)
                {
                    const double fd=(values[0][trial][ch]-values[1][trial][ch])/(2*h);
                    const double ref=(*channels(analytic[trial])[ch])(atom,axis);
                    near(fd,ref,ch==4?4e-6:2e-6);
                    worst[trial][ch]=std::max(worst[trial][ch],std::abs(fd-ref));
                }
            }
            for(int trial=0;trial<3;++trial)
            {
                std::cout<<"U8b "<<x.name()<<" D="<<trial<<" h="<<h<<" P/D/partition/point/total FD errors";
                for(double e:worst[trial]) std::cout<<' '<<e;
                std::cout<<'\n';
            }
            std::cout<<"U8b density/sigma/spin-threshold crossings h="<<h<<": "<<changed_masks<<'\n';
            std::cout<<"U8b crossing classes density="<<mask_types[0]<<" sigma="<<mask_types[1]<<" spin="<<mask_types[2]
                <<(acceptance_step?" acceptance step":" coarse diagnostic only")<<'\n';
            if(acceptance_step) check(changed_masks==0);
        }
        // AO-center motion cancels point motion under rigid translation.
        // Individual P/D channels need NOT sum to zero separately.
        for(int trial=0;trial<3;++trial) for(int axis=0;axis<3;++axis)
        {
            near(analytic[trial].becke_partition.col(axis).sum(),0,2e-9);
            near((analytic[trial].p_side_fixed+analytic[trial].d_side_ao+analytic[trial].point_translation).col(axis).sum(),0,2e-9);
            near(analytic[trial].total.col(axis).sum(),0,2e-9);
        }
        for(int ch=0;ch<4;++ch) check(channels(analytic[2])[ch]->norm()>1e-10);
    }
    void closed_shell(const Fixture &f)
    {
        auto ux=functional("gga_x_pbe"),uc=functional("gga_c_pbe");
        auto rx=functional("gga_x_pbe",XC::Spin::Unpolarized),rc=functional("gga_c_pbe",XC::Spin::Unpolarized);
        const M p=f.p.alpha+f.p.beta,d=f.d.alpha+f.d.beta;
        const auto u=require(build_udh_gga_complete_xc_ii_gradient(inputs(f,{.5*p,.5*p},{.5*d,.5*d},ux,uc)));
        const auto hess=require(evaluate_ao_hessian_on_grid(f.calc._shells,f.grid));
        DHEq33XCFixedDensityInputs in{&f.calc._molecule,&f.calc._shells,&f.grid,&f.ao,&hess,p,&rx,&rc};
        DHRelaxedDifferenceDensity rd; rd.symmetric_mo=d; rd.raw_mo=d;
        const auto r=require(build_dh_eq33_complete_xc_ii_gradient(rd,M::Identity(d.rows(),d.cols()),in));
        equal(u,{r.p_side_fixed,r.d_side_ao,r.becke_partition,r.point_translation,r.total});
        std::cout<<"U8b closed-shell reduction: all four channels and total\n";
    }
    void combined_and_errors(const Fixture &f)
    {
        auto x=functional("gga_x_pbe"),c=functional("gga_c_pbe"),both=functional("hyb_gga_xc_b2plyp");
        auto lda=functional("lda_x"),unpol=functional("gga_x_pbe",XC::Spin::Unpolarized);
        auto in=inputs(f,f.p,f.d,both,c);
        const auto reference=require(build_udh_gga_complete_xc_ii_gradient(in));
        in.fixed.correlation_functional=&lda;
        equal(require(build_udh_gga_complete_xc_ii_gradient(in)),reference);
        in=inputs(f,f.p,f.d,x,c); auto bad=in;
        bad.ao_hessian=nullptr; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        bad=in; bad.fixed.exchange_functional=&lda; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        bad=in; bad.fixed.exchange_functional=&unpol; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        bad=in; bad.fixed.relaxed_density.alpha(0,1)+=.1; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        auto h=f.hess; h.h_xy(0,0)=std::numeric_limits<double>::quiet_NaN();
        bad=in; bad.ao_hessian=&h; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        h=f.hess; h.h_zz.resize(0,0); check(!build_udh_gga_complete_xc_ii_gradient(bad));
        auto grid=f.grid; grid.owner(0)=3; bad=in; bad.fixed.molecular_grid=&grid;
        check(!build_udh_gga_complete_xc_ii_gradient(bad));
        grid=f.grid; grid.points(0,3)+=.1; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        auto ao=f.ao; ao.grad_x(0,0)=std::numeric_limits<double>::quiet_NaN();
        bad=in; bad.fixed.ao_grid=&ao; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        bad=in; bad.fixed.ground_density.alpha=-f.p.alpha; check(!build_udh_gga_complete_xc_ii_gradient(bad));
        grid=f.grid; grid.points.row(0).head<3>()=f.calc._molecule._standard.row(0);
        bad=in; bad.fixed.molecular_grid=&grid;
        const auto failure=build_udh_gga_complete_xc_ii_gradient(bad);
        check(!failure && failure.error().find("nucleus")!=std::string::npos);
    }
}
int main()
{
    try
    {
        auto f=fixture();
        auto x=functional("gga_x_pbe"),c=functional("gga_c_pbe"),both=functional("hyb_gga_xc_b2plyp");
        point_chain_rule(); kernel_slots(x); kernel_slots(c); kernel_slots(both);
        audit(f,x,c); audit(f,both,c); closed_shell(f); combined_and_errors(f);
    }
    catch(const std::exception &e) { std::cerr<<"FAIL U8b "<<e.what()<<'\n'; return 1; }
    if(ok) std::cout<<"PASS U8b complete polarized GGA fixed-coefficient XC-II\n";
    return ok?0:1;
}
