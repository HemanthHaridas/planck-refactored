// U2 only: independent fixed-AO-density oracles. No SCF, UMP2 amplitudes,
// orbital Hessian, Z solve, UKS derivative driver or external QC reference.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "base/basis.h"
#include "basis/basis.h"
#include "dft/ks_matrix.h"
#include "dft/udh_ks_response.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"

using namespace DFT::Gradient;
using Mat = Eigen::MatrixXd;
using Pair = UDHSpinMatrices;

namespace
{
    bool ok = true;
    void check(bool value, const std::string &label)
    { if (!value) { std::cerr << "FAIL U2 " << label << '\n'; ok = false; } }
    double error(const Mat &a, const Mat &b)
    {
        if (a.rows() != b.rows() || a.cols() != b.cols() || !a.allFinite() || !b.allFinite())
            return std::numeric_limits<double>::infinity();
        return a.size() ? (a-b).cwiseAbs().maxCoeff() : 0.0;
    }
    double error(const Pair &a, const Pair &b)
    { return std::max(error(a.alpha,b.alpha), error(a.beta,b.beta)); }
    double dot(const Pair &a, const Pair &b)
    { return (a.alpha.array()*b.alpha.array()).sum() + (a.beta.array()*b.beta.array()).sum(); }
    Pair add(const Pair &a, const Pair &q, double h)
    { return {a.alpha+h*q.alpha, a.beta+h*q.beta}; }
    Pair fd(const Pair &a, const Pair &b, double h)
    { return {(a.alpha-b.alpha)/(2*h), (a.beta-b.beta)/(2*h)}; }

    void detached_checks()
    {
        const Mat zero = Mat::Zero(2,2);
        Mat a(2,2), b(2,2); a << .3,.1,.1,-.2; b << -.4,.2,.2,.5;
        const UDHMatrixResponseFn j = [](const Eigen::Ref<const Mat> &q)
            -> std::expected<Mat,std::string> { return (3*q).eval(); };
        const UDHMatrixResponseFn k = [](const Eigen::Ref<const Mat> &q)
            -> std::expected<Mat,std::string> { return (5*q).eval(); };
        const UDHXCResponseFn xc = [](const Pair &q) -> std::expected<Pair,std::string>
        { return Pair{7*q.alpha+11*q.beta, 11*q.alpha+13*q.beta}; };
        auto op = make_udh_ks_response_operator(2,.53,j,k,xc);
        check(bool(op), "detached configuration");
        if (!op) return;
        auto r = op->apply_channels({a,b});
        check(bool(r), "detached action");
        if (r)
        {
            check(error(r->coulomb, (3*(a+b)).eval()) < 1e-14, "raw J, no occupancy factor");
            check(error(r->exchange, Pair{-2.65*a,-2.65*b}) < 1e-14, "separate -a_x K factors");
            check(error(r->xc_from_alpha, Pair{7*a,11*a}) < 1e-14, "XC aa/ba orientation");
            check(error(r->xc_from_beta, Pair{11*b,13*b}) < 1e-14, "XC ab/bb orientation");
            check(error(r->total, Pair{(3-2.65+7)*a+14*b, 14*a+(3-2.65+13)*b}) < 1e-14,
                  "physical total, no RKS factor 2 or adjoint factor 4");
        }
        auto pure = make_udh_ks_response_operator(2,0,j,{},xc);
        check(pure && pure->apply({a,zero}).has_value(), "zero exchange permits absent K");
        check(!make_udh_ks_response_operator(0,.5,j,k,xc), "zero AO dimension rejected");
        check(!make_udh_ks_response_operator(2,-.5,j,k,xc), "negative exchange rejected");
        check(!make_udh_ks_response_operator(2,std::numeric_limits<double>::quiet_NaN(),j,k,xc), "NaN coefficient");
        check(!make_udh_ks_response_operator(2,.5,{},k,xc), "missing J");
        check(!make_udh_ks_response_operator(2,.5,j,{},xc), "missing nonzero K");
        check(!make_udh_ks_response_operator(2,.5,j,k,{}), "missing XC");
        check(!op->apply({Mat(),b}), "missing spin AO matrix is not a zero spin trial");
        Mat bad = a; bad(0,1) += .1;
        check(!op->apply({bad,b}), "asymmetric input");
        bad = a; bad(0,0) = std::numeric_limits<double>::infinity();
        check(!op->apply({a,bad}), "nonfinite beta input");
        const UDHMatrixResponseFn broken = [](const Eigen::Ref<const Mat> &)
            -> std::expected<Mat,std::string> { return std::unexpected("injected"); };
        for (bool break_j : {true,false})
        {
            auto failure = make_udh_ks_response_operator(2,.5,break_j?broken:j,break_j?k:broken,xc);
            auto value = failure->apply({a,b});
            check(!value && value.error().find("injected") != std::string::npos, "J/K failure is an error");
        }
        auto beta_failure = *op;
        beta_failure.exchange = [a](const Eigen::Ref<const Mat> &q) -> std::expected<Mat,std::string>
        { if (error(q,a) < 1e-15) return q.eval(); return std::unexpected("beta K failure"); };
        auto bf = beta_failure.apply({a,b});
        check(!bf && bf.error().find("K-beta") != std::string::npos, "beta K failure label");
        for (int which = 0; which < 5; ++which)
        {
            auto failure = *op;
            failure.xc = [which,zero](const Pair &q) -> std::expected<Pair,std::string>
            {
                if (which == 0) return std::unexpected("XC injected");
                if (which == 1) throw std::runtime_error("XC exception");
                if (which == 2) return Pair{Mat::Zero(1,1),zero};
                if (which == 4) throw 42;
                if (q.beta.norm() != 0) return std::unexpected("second spin-only call");
                return Pair{zero,zero};
            };
            check(!failure.apply({a,b}), "XC failure/exception/dimension propagation");
        }
        auto nan_output = *op;
        nan_output.coulomb = [](const Eigen::Ref<const Mat> &q) -> std::expected<Mat,std::string>
        { return Mat::Constant(q.rows(),q.cols(),std::numeric_limits<double>::quiet_NaN()); };
        check(!nan_output.apply({a,b}), "nonfinite callback output");
        auto nonsymmetric = *op;
        nonsymmetric.exchange = [](const Eigen::Ref<const Mat> &q) -> std::expected<Mat,std::string>
        { Mat v=q; v(0,1)+=1; return v; };
        check(!nonsymmetric.apply({a,b}), "asymmetric callback output");
        auto zr = op->apply({zero,zero});
        check(zr && error(*zr,Pair{zero,zero}) == 0, "zero trial");
        // Callback copies survive destruction of the original operator.
        auto copy = *op; op = std::unexpected("discard original");
        check(copy.apply({a,b}).has_value(), "owned callback lifetime");
    }

    DFT::XC::Functional functional(const char *name, DFT::XC::Spin spin)
    {
        auto id = DFT::XC::functional_id(name);
        if (!id) throw std::runtime_error(id.error());
        auto f = DFT::XC::Functional::create(*id,spin);
        if (!f) throw std::runtime_error(f.error());
        return std::move(*f);
    }

    void physical_checks()
    {
        HartreeFock::Calculator calc;
        auto &mol = calc._molecule;
        mol.natoms=3; mol.charge=1; mol.multiplicity=2;
        mol.atomic_numbers.resize(3); mol.atomic_numbers << 8,1,1;
        mol.atomic_masses.resize(3); mol.atomic_masses << 16.,1.,1.;
        mol.coordinates.resize(3,3);
        mol.coordinates << .013,.021,-.011, .774,.032,.531, -.699,-.047,.481;
        calc._basis._basis=HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates(); mol.set_standard_from_bohr(mol._coordinates);
        auto basis = HartreeFock::BasisFunctions::read_gbs_basis(
            (std::filesystem::path(get_basis_path())/"sto-3g").string(),mol,calc._basis._basis);
        if (!basis) throw std::runtime_error(basis.error());
        calc._shells = std::move(*basis);
        const auto pairs=build_shellpairs(calc._shells);
        const auto n=static_cast<Eigen::Index>(calc._shells.nbasis());
        const auto engine=HartreeFock::IntegralMethod::ObaraSaika;
        auto [s,t]=_compute_1e(pairs,n,engine);
        const Mat hcore=t+_compute_nuclear_attraction(pairs,n,mol,engine);
        Eigen::SelfAdjointEigenSolver<Mat> se(s);
        if (se.info()!=Eigen::Success || se.eigenvalues().minCoeff()<=0) throw std::runtime_error("overlap");
        const Mat orth=se.operatorInverseSqrt();
        Eigen::SelfAdjointEigenSolver<Mat> he(orth*hcore*orth);
        if (he.info()!=Eigen::Success) throw std::runtime_error("core orbitals");
        const Mat c=orth*he.eigenvectors();
        const Pair p{c.leftCols(5)*c.leftCols(5).transpose(),c.leftCols(4)*c.leftCols(4).transpose()};
        check(std::abs((p.alpha.array()*s.array()).sum()-5.0)<1e-12,"one-electron alpha occupancy");
        check(std::abs((p.beta.array()*s.array()).sum()-4.0)<1e-12,"one-electron beta occupancy");
        const auto cross = [&c](int i,int j) -> Mat
        { return c.col(i)*c.col(j).transpose()+c.col(j)*c.col(i).transpose(); };
        // Signed, non-diagonal occupied-space directions keep both P +/- h Q
        // nonnegative on the real molecular grid, avoiding density-clamp FDs.
        const Pair q{.08*p.alpha+.03*cross(0,2),-.06*p.beta+.02*cross(1,3)};
        const Pair b{-.04*p.alpha+.01*cross(1,4),.07*p.beta-.03*cross(0,2)};
        const Mat zero=Mat::Zero(n,n);
        auto grid=DFT::MakeMolecularGrid(mol,DFT::GridLevel::Coarse);
        if (!grid) throw std::runtime_error(grid.error());
        auto ao=DFT::evaluate_ao_basis_on_grid(calc._shells,*grid);
        if (!ao) throw std::runtime_error(ao.error());
        const auto eri=_compute_2e(pairs,n,engine,HartreeFock::ERIKernel::Coulomb,0.0,0.0,nullptr);
        const auto idx=[n](int i,int j,int k,int l)
        { return ((static_cast<std::size_t>(i)*n+j)*n+k)*n+l; };
        // Independent full-AO-index contraction, not the direct J/K builders.
        const auto jk=[&](const Mat &density,bool exchange) -> Mat
        {
            Mat v=Mat::Zero(n,n);
            for (int i=0;i<n;++i) for (int j=0;j<n;++j)
                for (int k=0;k<n;++k) for (int l=0;l<n;++l)
                    v(i,j)+=density(k,l)*eri[exchange?idx(i,k,j,l):idx(i,j,k,l)];
            return v;
        };
        struct Case { const char *x; const char *c; double ax; };
        for (const auto item : {Case{"lda_x","lda_c_vwn",.31},
                                Case{"gga_x_pbe","gga_c_pbe",.53},
                                Case{"hyb_gga_xc_b2plyp","gga_c_pbe",.53}})
        {
            auto x=functional(item.x,DFT::XC::Spin::Polarized);
            auto corr=functional(item.c,DFT::XC::Spin::Polarized);
            UDHKSResponseInputs inputs{&pairs,&*grid,&*ao,p,&x,&corr,item.ax,engine,0.0};
            auto op=make_udh_direct_ks_response_operator(inputs);
            if (!op) throw std::runtime_error(op.error());
            // Ordinary first-derivative XC potential at independently changed
            // densities. It never calls the U2 or analytic-HVP implementation.
            const auto vxc=[&](const Pair &density) -> Pair
            {
                auto values=DFT::evaluate_xc_on_grid(*grid,*ao,density.alpha,density.beta,x,corr);
                if (!values) throw std::runtime_error(values.error());
                auto v=DFT::assemble_xc_matrix(*grid,*ao,*values);
                if (!v) throw std::runtime_error(v.error());
                return {v->alpha,v->beta};
            };
            double jerr=0,kaerr=0,kberr=0,aaerr=0,aberr=0,baerr=0,bberr=0,totalerr=0;
            for (const Pair &direction : {Pair{q.alpha,zero},Pair{zero,q.beta},q})
            {
                auto r=op->apply_channels(direction);
                if (!r) throw std::runtime_error(r.error());
                check(error(r->coulomb,jk((direction.alpha+direction.beta).eval(),false))<1e-11,"explicit J");
                check(error(r->exchange,Pair{-item.ax*jk(direction.alpha,true),-item.ax*jk(direction.beta,true)})<1e-11,"explicit spin K");
                if (direction.alpha.norm()==0) check(r->exchange.alpha.norm()==0,"no beta-to-alpha exchange");
                if (direction.beta.norm()==0) check(r->exchange.beta.norm()==0,"no alpha-to-beta exchange");
                for (double step : {1e-3,3e-4,1e-4})
                {
                    const auto plus=add(p,direction,step), minus=add(p,direction,-step);
                    const Mat dj=(jk((plus.alpha+plus.beta).eval(),false)-jk((minus.alpha+minus.beta).eval(),false))/(2*step);
                    const Pair dk{-item.ax*(jk(plus.alpha,true)-jk(minus.alpha,true))/(2*step),
                                  -item.ax*(jk(plus.beta,true)-jk(minus.beta,true))/(2*step)};
                    const Pair dx=fd(vxc(plus),vxc(minus),step);
                    jerr=std::max(jerr,error(r->coulomb,dj));
                    kaerr=std::max(kaerr,error(r->exchange.alpha,dk.alpha));
                    kberr=std::max(kberr,error(r->exchange.beta,dk.beta));
                    const Pair xa=fd(vxc(add(p,{direction.alpha,zero},step)),vxc(add(p,{direction.alpha,zero},-step)),step);
                    const Pair xb=fd(vxc(add(p,{zero,direction.beta},step)),vxc(add(p,{zero,direction.beta},-step)),step);
                    aaerr=std::max(aaerr,error(r->xc_from_alpha.alpha,xa.alpha));
                    baerr=std::max(baerr,error(r->xc_from_alpha.beta,xa.beta));
                    aberr=std::max(aberr,error(r->xc_from_beta.alpha,xb.alpha));
                    bberr=std::max(bberr,error(r->xc_from_beta.beta,xb.beta));
                    totalerr=std::max(totalerr,error(r->total,Pair{dj+dk.alpha+dx.alpha,dj+dk.beta+dx.beta}));
                }
            }
            std::cout << "U2 " << item.x << " density FD J=" << jerr << " K-alpha=" << kaerr << " K-beta=" << kberr
                      << " XC-aa=" << aaerr << " XC-ab=" << aberr << " XC-ba=" << baerr
                      << " XC-bb=" << bberr << " total=" << totalerr << '\n';
            check(jerr<1e-8 && kaerr<1e-8 && kberr<1e-8,"J/K density FD");
            check(std::max({aaerr,aberr,baerr,bberr,totalerr})<2e-7,"all XC blocks/total density FD");
            auto rq=op->apply(q), rb=op->apply(b);
            if (!rq || !rb) throw std::runtime_error("joint response");
            const double adj=std::abs(dot(b,*rq)-dot(q,*rb));
            check(adj<1e-10,"joint-spin adjointness");
            auto linear=op->apply(add(q,b,.7));
            check(linear && error(*linear,add(*rq,*rb,.7))<1e-11,"linearity");
            auto zr=op->apply({zero,zero});
            check(zr && error(*zr,Pair{zero,zero})<1e-14,"zero response");
            auto channels=op->apply_channels(q);
            check(channels && channels->xc_from_alpha.beta.norm()>1e-8 &&
                  channels->xc_from_beta.alpha.norm()>1e-8,"nonzero cross-spin XC");
            auto flipped=inputs; flipped.ground_density={p.beta,p.alpha};
            auto flip=make_udh_direct_ks_response_operator(flipped);
            if (!flip) throw std::runtime_error(flip.error());
            auto rf=flip->apply({q.beta,q.alpha});
            check(rf && error(*rf,Pair{rq->beta,rq->alpha})<1e-10,"spin swap");
            // The operator owns its ground-density snapshot, not Inputs storage.
            inputs.ground_density.alpha.setZero(); inputs.ground_density.beta.setZero();
            auto again=op->apply(q);
            check(again && error(*again,*rq)==0,"ground snapshot survives caller mutation");

            auto ux=functional(item.x,DFT::XC::Spin::Unpolarized);
            auto uc=functional(item.c,DFT::XC::Spin::Unpolarized);
            const Mat pt=p.alpha+p.beta, qt=q.alpha+q.beta;
            auto closed=flipped; closed.ground_density={.5*pt,.5*pt};
            auto cop=make_udh_direct_ks_response_operator(closed);
            if (!cop) throw std::runtime_error(cop.error());
            auto cr=cop->apply({.5*qt,.5*qt});
            auto xc_rks=DFT::Driver::compute_analytic_xc_hessian_vector_product(*grid,*ao,pt,qt,ux,uc);
            if (!cr || !xc_rks) throw std::runtime_error("closed-shell response");
            const Mat reference=jk(qt,false)-.5*item.ax*jk(qt,true)+*xc_rks;
            check(error(*cr,Pair{reference,reference})<1e-10,"physical RKS limit");
            std::cout << "U2 " << item.x << " joint_adjoint=" << adj
                      << " RKS_reduction=" << error(*cr,Pair{reference,reference}) << '\n';

            auto invalid=closed; invalid.exchange_functional=&ux;
            check(!make_udh_direct_ks_response_operator(invalid),"reject unpolarized binding");
            invalid=closed; invalid.tol_eri=-1;
            check(!make_udh_direct_ks_response_operator(invalid),"reject negative ERI cutoff");
            invalid=closed; invalid.molecular_grid=nullptr;
            check(!make_udh_direct_ks_response_operator(invalid),"reject missing grid");
        }
    }
}

int main()
{
    try { detached_checks(); physical_checks(); }
    catch (const std::exception &e) { check(false,e.what()); }
    std::cout << (ok?"PASS":"FAIL") << " U2 checked physical two-spin KS response\n";
    return ok?0:1;
}
