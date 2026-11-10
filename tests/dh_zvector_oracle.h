#ifndef PLANCK_TESTS_DH_ZVECTOR_ORACLE_H
#define PLANCK_TESTS_DH_ZVECTOR_ORACLE_H

// Test-only dense Eq. 27 Hessians, built one column per unit trial from the
// public production actions. Production solves matrix-free
// (solve_packed_zvector) and never forms these. Packing matches production:
// a*no+i, and for UKS alpha then beta.

#include "dft/dh_zvector.h"
#include "dft/udh_zvector.h"

#include <stdexcept>

namespace DFT::Gradient::TestOracle
{
    inline Eigen::MatrixXd dense_dh_hessian(const Eigen::MatrixXd &c_occ,const Eigen::MatrixXd &c_virt,
        const Eigen::VectorXd &eps,const DHEq41ResponseOperator &response)
    {
        const Eigen::Index no=c_occ.cols(),nv=c_virt.cols(),n=no*nv;
        Eigen::MatrixXd hessian(n,n);
        for (Eigen::Index col=0;col<n;++col)
        {
            Eigen::MatrixXd unit=Eigen::MatrixXd::Zero(nv,no); unit(col/no,col%no)=1;
            auto value=apply_dh_eq27_hessian(unit,c_occ,c_virt,eps,response);
            if (!value) throw std::runtime_error(value.error());
            for (Eigen::Index a=0;a<nv;++a) for (Eigen::Index i=0;i<no;++i) hessian(a*no+i,col)=value->total_ai(a,i);
        }
        return hessian;
    }

    // Dense A z = -rhs; throws on a rank-deficient Hessian.
    inline Eigen::MatrixXd dense_dh_zvector(const Eigen::MatrixXd &rhs_ai,const Eigen::MatrixXd &c_occ,
        const Eigen::MatrixXd &c_virt,const Eigen::VectorXd &eps,const DHEq41ResponseOperator &response)
    {
        const Eigen::Index no=c_occ.cols(),nv=c_virt.cols();
        const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(dense_dh_hessian(c_occ,c_virt,eps,response));
        if (qr.rank()!=no*nv) throw std::runtime_error("dense DH Z oracle: rank-deficient Hessian");
        Eigen::VectorXd b(no*nv);
        for (Eigen::Index a=0;a<nv;++a) for (Eigen::Index i=0;i<no;++i) b(a*no+i)=-rhs_ai(a,i);
        const Eigen::VectorXd z=qr.solve(b);
        Eigen::MatrixXd out(nv,no);
        for (Eigen::Index a=0;a<nv;++a) for (Eigen::Index i=0;i<no;++i) out(a,i)=z(a*no+i);
        return out;
    }

    inline Eigen::MatrixXd dense_udh_jacobian(const UDHZVectorInputs &in)
    {
        const Eigen::Index oa=in.nocc_alpha,ob=in.nocc_beta;
        const Eigen::Index va=in.mo_coeff.alpha.cols()-oa,vb=in.mo_coeff.beta.cols()-ob;
        const Eigen::Index na=oa*va,n=na+ob*vb;
        Eigen::MatrixXd jacobian(n,n);
        for (Eigen::Index col=0;col<n;++col)
        {
            UDHOrbitalMatrices unit{Eigen::MatrixXd::Zero(va,oa),Eigen::MatrixXd::Zero(vb,ob)};
            if (col<na) unit.alpha(col/oa,col%oa)=1; else unit.beta((col-na)/ob,(col-na)%ob)=1;
            auto value=apply_udh_eq27_hessian(in,unit);
            if (!value) throw std::runtime_error(value.error());
            for (Eigen::Index a=0;a<va;++a) for (Eigen::Index i=0;i<oa;++i) jacobian(a*oa+i,col)=value->total.alpha(a,i);
            for (Eigen::Index a=0;a<vb;++a) for (Eigen::Index i=0;i<ob;++i) jacobian(na+a*ob+i,col)=value->total.beta(a,i);
        }
        return jacobian;
    }
}

#endif
