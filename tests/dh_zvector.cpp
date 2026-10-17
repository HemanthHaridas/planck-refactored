#include "dft/dh_zvector.h"
#include <iostream>
#include <stdexcept>

bool test_dh_zvector()
{
    using namespace DFT::Gradient;
    bool ok=true;
    const auto check=[&](bool passed,const char *name)
    { if (!passed) { ok=false; std::cerr<<"FAIL O3 "<<name<<'\n'; } };
    constexpr int no=2,nv=3,nao=5;
    const Eigen::MatrixXd c=Eigen::MatrixXd::Identity(nao,nao);
    const Eigen::MatrixXd co=c.leftCols(no),cv=c.rightCols(nv);
    Eigen::VectorXd eps(nao); eps<<-.8,-.4,.2,.6,1.1;
    Eigen::MatrixXd known(nv,no); known<<.12,-.23,.31,.17,-.28,.09;
    Eigen::MatrixXd j=Eigen::MatrixXd::Identity(nao,nao)*.12;
    Eigen::MatrixXd k=Eigen::MatrixXd::Identity(nao,nao)*.09;
    Eigen::MatrixXd f=Eigen::MatrixXd::Identity(nao,nao)*.07;
    j(0,1)=j(1,0)=.03; j(2,4)=j(4,2)=.04;
    k(0,1)=k(1,0)=-.02; k(2,3)=k(3,2)=.05;
    f(0,1)=f(1,0)=.04; f(3,4)=f(4,3)=-.03;
    const auto map=[](Eigen::MatrixXd m)->DHResponseFn
    { return [m=std::move(m)](const Eigen::Ref<const Eigen::MatrixXd> &q)
        ->std::expected<Eigen::MatrixXd,std::string> { return (m*q*m.transpose()).eval(); }; };
    const auto response=make_dh_eq41_response_operator(.53,map(j),map(k),map(f));
    if (!response) return false;
    // Independent channel expressions, before either solver is involved.
    const Eigen::MatrixXd raw=cv*known*co.transpose();
    const Eigen::MatrixXd dp=2*(raw+raw.transpose());
    Eigen::MatrixXd orbital(nv,no);
    for (int a=0;a<nv;++a) for(int i=0;i<no;++i) orbital(a,i)=(eps(no+a)-eps(i))*known(a,i);
    const Eigen::MatrixXd ej=cv.transpose()*j*dp*j.transpose()*co;
    const Eigen::MatrixXd ek=-.5*.53*cv.transpose()*k*dp*k.transpose()*co;
    const Eigen::MatrixXd ef=cv.transpose()*f*dp*f.transpose()*co;
    const auto applied=apply_dh_eq27_hessian(known,co,cv,eps,*response);
    check(applied && (applied->orbital_energy_ai-orbital).norm()<1e-14 &&
        (applied->coulomb_ai-ej).norm()<1e-14 && (applied->exchange_ai-ek).norm()<1e-14 &&
        (applied->xc_ai-ef).norm()<1e-14 &&
        (applied->total_ai-orbital-ej-ek-ef).norm()<1e-14,"independent Eq27 channels");
    const Eigen::MatrixXd rhs=-(orbital+ej+ek+ef);
    DHZVectorOptions dense_options; dense_options.backend=DHZVectorBackend::DenseReference;
    const auto dense=solve_dh_zvector(rhs,co,cv,eps,*response,dense_options);
    auto gmres_options=DHZVectorOptions{}; gmres_options.restart=2; gmres_options.tolerance=1e-13;
    const auto iterative=solve_dh_zvector(rhs,co,cv,eps,*response,gmres_options);
    check(dense && iterative && (dense->z_ai-known).norm()<1e-11 &&
        (iterative->z_ai-known).norm()<1e-11 && (iterative->z_ai-dense->z_ai).norm()<1e-11 &&
        iterative->restarts>0 && iterative->dense_matrix_bytes==0 && dense->krylov_matrix_bytes==0 &&
        dense->action_count==no*nv+1 && iterative->action_count==2+2*iterative->iterations &&
        iterative->residual_max_abs<=1e-13,"typed backends, packing, restart and fresh residual");
    gmres_options.max_iterations=1;
    const auto exhausted=solve_dh_zvector(rhs,co,cv,eps,*response,gmres_options);
    check(!exhausted && exhausted.error().find("no dense fallback")!=std::string::npos,"iteration exhaustion fails closed");
    const auto zero=solve_dh_zvector(Eigen::MatrixXd::Zero(nv,no),co,cv,eps,*response);
    check(zero && zero->iterations==0 && zero->action_count==2 && zero->z_ai.norm()==0,"zero RHS still final-checked");
    for (int channel=0;channel<3;++channel)
    {
        auto failed=*response;
        DHResponseFn error=[](const Eigen::Ref<const Eigen::MatrixXd>&)
            ->std::expected<Eigen::MatrixXd,std::string> { return std::unexpected("injected channel failure"); };
        (channel==0?failed.coulomb:channel==1?failed.exchange:failed.xc)=error;
        for (auto backend : {DHZVectorBackend::MatrixFreeGMRES,DHZVectorBackend::DenseReference})
        {
            DHZVectorOptions options; options.backend=backend;
            const auto value=solve_dh_zvector(rhs,co,cv,eps,failed,options);
            check(!value && value.error().find("injected channel failure")!=std::string::npos,"J/K/XC errors preserved");
        }
    }
    auto failed=*response;
    failed.xc=[](const Eigen::Ref<const Eigen::MatrixXd>&)->std::expected<Eigen::MatrixXd,std::string>
    { throw std::runtime_error("injected XC exception"); };
    const auto exception=solve_dh_zvector(rhs,co,cv,eps,failed);
    check(!exception && exception.error().find("injected XC exception")!=std::string::npos,"XC exception rejected");
    failed.xc=[](const Eigen::Ref<const Eigen::MatrixXd>&)->std::expected<Eigen::MatrixXd,std::string>
    { return Eigen::MatrixXd::Zero(1,1).eval(); };
    check(!solve_dh_zvector(rhs,co,cv,eps,failed),"wrong-sized XC rejected");
    failed.xc=[](const Eigen::Ref<const Eigen::MatrixXd>&q)->std::expected<Eigen::MatrixXd,std::string>
    { return Eigen::MatrixXd::Constant(q.rows(),q.cols(),std::numeric_limits<double>::quiet_NaN()).eval(); };
    check(!solve_dh_zvector(rhs,co,cv,eps,failed),"nonfinite XC rejected");
    // Negative and zero gaps are legal. A small gap affects preconditioning,
    // not the operator: this one-dimensional solve would expose any shift.
    const Eigen::MatrixXd one_c=Eigen::MatrixXd::Identity(2,2);
    auto scalar=*response; scalar.coulomb=map(Eigen::MatrixXd::Identity(2,2));
    scalar.exact_exchange=0; scalar.exchange={}; scalar.xc=map(Eigen::MatrixXd::Zero(2,2));
    for (double gap : {0.0,-3.0,1e-10})
    {
        Eigen::Vector2d energies; energies<<0,gap;
        const Eigen::MatrixXd l=Eigen::MatrixXd::Constant(1,1,-.3*(gap+2));
        const auto value=solve_dh_zvector(l,one_c.leftCols(1),one_c.rightCols(1),energies,scalar);
        check(value && std::abs(value->z_ai(0,0)-.3)<1e-12,"gap floor does not shift physical action");
    }
    int xc_calls=0;
    scalar.xc=[&](const Eigen::Ref<const Eigen::MatrixXd> &q)->std::expected<Eigen::MatrixXd,std::string>
    { ++xc_calls; return Eigen::MatrixXd::Constant(q.rows(),q.cols(),xc_calls>=4?.1:0).eval(); };
    const auto changed=solve_dh_zvector(Eigen::MatrixXd::Constant(1,1,-.6),
        one_c.leftCols(1),one_c.rightCols(1),Eigen::Vector2d::Zero(),scalar);
    check(!changed && xc_calls==4 && changed.error().find("fresh final residual")!=std::string::npos,
        "fresh residual detects changed callback after convergence");
    scalar.coulomb=map(Eigen::MatrixXd::Zero(2,2)); scalar.xc=scalar.coulomb;
    for (auto backend : {DHZVectorBackend::MatrixFreeGMRES,DHZVectorBackend::DenseReference})
    {
        DHZVectorOptions options; options.backend=backend;
        check(!solve_dh_zvector(Eigen::MatrixXd::Ones(1,1),one_c.leftCols(1),one_c.rightCols(1),
            Eigen::Vector2d::Zero(),scalar,options),"singular solve rejected by both backends");
    }
    auto invalid_options=DHZVectorOptions{}; invalid_options.restart=0;
    check(!solve_dh_zvector(rhs,co,cv,eps,*response,invalid_options),"invalid restart rejected");
    failed=*response; failed.xc={};
    check(!solve_dh_zvector(rhs,co,cv,eps,failed),"missing XC rejected");
    if (iterative) std::cout<<"O3 typed Eq27 actions="<<iterative->action_count<<" iterations="<<iterative->iterations
        <<" restarts="<<iterative->restarts<<" residual="<<iterative->residual_max_abs<<'\n';
    return ok;
}
