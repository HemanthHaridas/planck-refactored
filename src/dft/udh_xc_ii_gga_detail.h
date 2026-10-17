#ifndef DFT_UDH_XC_II_GGA_DETAIL_H
#define DFT_UDH_XC_II_GGA_DETAIL_H

#include <Eigen/Dense>

namespace DFT::Gradient::detail
{
    using GGA5=Eigen::Matrix<double,5,1>;
    using GGA55=Eigen::Matrix<double,5,5>;
    using SpinGradients=Eigen::Matrix<double,3,2>;

    // Libxc rr=[aa,ab,bb], rs=[a-aa,a-ab,a-bb,b-aa,b-ab,b-bb],
    // ss=[aa-aa,aa-ab,aa-bb,ab-ab,ab-bb,bb-bb]. No multiplicities.
    inline GGA55 unpack_udh_gga_kernel(const double *rr,const double *rs,const double *ss)
    {
        GGA55 h=GGA55::Zero();
        h(0,0)=rr[0]; h(0,1)=h(1,0)=rr[1]; h(1,1)=rr[2];
        for(int r=0;r<2;++r) for(int s=0;s<3;++s) h(r,s+2)=h(s+2,r)=rs[3*r+s];
        int k=0;
        for(int r=0;r<3;++r) for(int s=r;s<3;++s) h(r+2,s+2)=h(s+2,r+2)=ss[k++];
        return h;
    }

    struct GGAXCIIPartials
    {
        double scalar=0;
        Eigen::Vector2d p_rho,d_rho;
        SpinGradients p_gradient,d_gradient;
    };

    // Exact local partials of Q=f_y(y(P)).s_D(P,D). This pure algebra
    // boundary has no libxc, AO, grid, density cutoff or nuclear convention.
    inline GGAXCIIPartials udh_gga_xc_ii_partials(const Eigen::Vector2d &d,
        const SpinGradients &u,const SpinGradients &v,const GGA5 &fy,const GGA55 &h)
    {
        GGA5 sd;
        sd<<d(0),d(1),2*u.col(0).dot(v.col(0)),
            u.col(0).dot(v.col(1))+u.col(1).dot(v.col(0)),2*u.col(1).dot(v.col(1));
        const GGA5 q=h*sd;
        GGAXCIIPartials out;
        out.scalar=fy.dot(sd); out.p_rho=q.head<2>(); out.d_rho=fy.head<2>();
        out.p_gradient.col(0)=2*q(2)*u.col(0)+q(3)*u.col(1)+2*fy(2)*v.col(0)+fy(3)*v.col(1);
        out.p_gradient.col(1)=2*q(4)*u.col(1)+q(3)*u.col(0)+2*fy(4)*v.col(1)+fy(3)*v.col(0);
        out.d_gradient.col(0)=2*fy(2)*u.col(0)+fy(3)*u.col(1);
        out.d_gradient.col(1)=2*fy(4)*u.col(1)+fy(3)*u.col(0);
        return out;
    }
}
#endif
