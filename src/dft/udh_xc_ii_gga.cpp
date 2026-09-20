#include "udh_xc_ii.h"
#include "udh_xc_ii_gga_detail.h"
#include "dft_gradient.h"
#include <array>
#include <cmath>
#include <limits>

namespace DFT::Gradient
{
    namespace
    {
        using M=Eigen::MatrixXd;
        using detail::SpinGradients;
        bool symmetric(const M &m,Eigen::Index n)
        { return m.rows()==n && m.cols()==n && m.allFinite() &&
            (m-m.transpose()).cwiseAbs().maxCoeff()<=1e-12*(1+m.cwiseAbs().maxCoeff()); }
        bool gga(const XC::Functional &f)
        { return f.is_gga_like() && f.spin()==XC::Spin::Polarized && f.get()->info &&
            (f.get()->info->flags & XC_FLAGS_HAVE_VXC) && (f.get()->info->flags & XC_FLAGS_HAVE_FXC); }
        Eigen::Vector3d grad(const DensityChannelOnGrid &d,Eigen::Index p)
        { return {d.grad_x(p),d.grad_y(p),d.grad_z(p)}; }
        bool finite(const DensityChannelOnGrid &d)
        { return d.rho.allFinite() && d.grad_x.allFinite() && d.grad_y.allFinite() && d.grad_z.allFinite(); }
        // Spatial derivative of grad(rho) when an owner point moves:
        // H_rho[k,q]=2(phi_kq^T D phi + phi_k^T D phi_q).
        Eigen::Vector3d spatial_column(const M &d,const AOGridEvaluation &ao,
            const AOGridHessian &h,Eigen::Index p,int q)
        {
            const std::array<const M*,3> g{&ao.grad_x,&ao.grad_y,&ao.grad_z};
            const std::array<std::array<const M*,3>,3> hm{{
                {&h.h_xx,&h.h_xy,&h.h_xz},{&h.h_xy,&h.h_yy,&h.h_yz},{&h.h_xz,&h.h_yz,&h.h_zz}}};
            const Eigen::VectorXd dphi=d*ao.values.row(p).transpose();
            const Eigen::VectorXd dg=d*g[q]->row(p).transpose();
            Eigen::Vector3d out;
            for(int k=0;k<3;++k) out(k)=2*(hm[k][q]->row(p).dot(dphi)+g[k]->row(p).dot(dg));
            return out;
        }
    }

    std::expected<UDHCompleteXCIIGradient,std::string>
    build_udh_gga_complete_xc_ii_gradient(const UDHGGAXCIIInputs &inputs)
    {
        const auto &in=inputs.fixed;
        if(!in.molecule || !in.basis || !in.molecular_grid || !in.ao_grid || !inputs.ao_hessian ||
            !in.exchange_functional || !in.correlation_functional)
            return std::unexpected("UDH U8b XC-II: missing input snapshot/Hessian.");
        const auto &x=*in.exchange_functional,&c=*in.correlation_functional;
        const bool combined=x.is_combined_exchange_correlation();
        if(!gga(x) || (!combined && !gga(c)))
            return std::unexpected("UDH U8b XC-II: requires polarized GGA vxc/fxc functionals.");
        const auto &mol=*in.molecule; const auto &grid=*in.molecular_grid;
        const auto &ao=*in.ao_grid; const auto &hess=*inputs.ao_hessian;
        const auto np=ao.npoints(),nb=ao.nbasis();
        if(mol.natoms==0 || mol.natoms>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            mol._standard.rows()!=static_cast<Eigen::Index>(mol.natoms) || mol._standard.cols()!=3 ||
            !mol._standard.allFinite() || mol.atomic_numbers.size()!=static_cast<Eigen::Index>(mol.natoms) ||
            (mol.atomic_numbers.array()<=0).any() || nb<=0 || in.basis->nbasis()!=static_cast<std::size_t>(nb) ||
            np<=0 || np>std::numeric_limits<int>::max()/6 || grid.points.rows()!=np || grid.points.cols()!=4 ||
            grid.owner.size()!=np || grid.atomic_weights.size()!=np || grid.partition_weights.size()!=np ||
            !grid.points.allFinite() || !grid.atomic_weights.allFinite() || !grid.partition_weights.allFinite())
            return std::unexpected("UDH U8b XC-II: invalid molecule, AO or grid dimensions/values.");
        for(const auto *m:{&ao.values,&ao.grad_x,&ao.grad_y,&ao.grad_z,
            &hess.h_xx,&hess.h_xy,&hess.h_xz,&hess.h_yy,&hess.h_yz,&hess.h_zz})
            if(m->rows()!=np || m->cols()!=nb || !m->allFinite())
                return std::unexpected("UDH U8b XC-II: invalid AO values/gradients/Hessians.");
        for(const auto *d:{&in.ground_density.alpha,&in.ground_density.beta,&in.relaxed_density.alpha,&in.relaxed_density.beta})
            if(!symmetric(*d,nb)) return std::unexpected("UDH U8b XC-II: densities must be finite symmetric spin AO matrices.");
        for(Eigen::Index p=0;p<np;++p)
        {
            const double expected=grid.atomic_weights(p)*grid.partition_weights(p);
            if(grid.owner(p)<0 || grid.owner(p)>=static_cast<int>(mol.natoms) ||
                grid.partition_weights(p)<0 || grid.partition_weights(p)>1 || !std::isfinite(expected) ||
                std::abs(grid.points(p,3)-expected)>1e-12*(1+std::abs(expected)))
                return std::unexpected("UDH U8b XC-II: inconsistent owner/partition/atomic weight metadata.");
        }
        const auto atoms=atom_bf_lists(mol,*in.basis);
        if(!atoms) return std::unexpected("UDH U8b XC-II: "+atoms.error());
        const auto ground=evaluate_density_on_grid(ao,in.ground_density.alpha,in.ground_density.beta);
        const auto difference=evaluate_density_on_grid(ao,in.relaxed_density.alpha,in.relaxed_density.beta);
        if(!ground || !difference) return std::unexpected("UDH U8b XC-II: density evaluation failed.");
        if(!finite(ground->alpha) || !finite(ground->beta) || !finite(difference->alpha) || !finite(difference->beta) ||
            ground->alpha.rho.minCoeff()<0 || ground->beta.rho.minCoeff()<0)
            return std::unexpected("UDH U8b XC-II: nonfinite fields or negative reference spin density.");
        std::vector<double> rho(2*np),sigma(3*np);
        for(Eigen::Index p=0;p<np;++p)
        {
            rho[2*p]=ground->alpha.rho(p); rho[2*p+1]=ground->beta.rho(p);
            const auto a=grad(ground->alpha,p),b=grad(ground->beta,p);
            sigma[3*p]=a.squaredNorm(); sigma[3*p+1]=a.dot(b); sigma[3*p+2]=b.squaredNorm();
        }
        for(double s:sigma) if(!std::isfinite(s)) return std::unexpected("UDH U8b XC-II: nonfinite sigma.");
        std::vector<double> exc,vr,vs,rr,rs,ss;
        const auto vx=x.evaluate_gga_exc_vxc(rho,sigma,static_cast<int>(np),exc,vr,vs);
        const auto fx=x.evaluate_gga_fxc(rho,sigma,static_cast<int>(np),rr,rs,ss);
        if(!vx) return std::unexpected("UDH U8b XC-II: "+vx.error());
        if(!fx) return std::unexpected("UDH U8b XC-II: "+fx.error());
        if(!combined)
        {
            std::vector<double> cr,cs,crr,crs,css;
            const auto cv=c.evaluate_gga_exc_vxc(rho,sigma,static_cast<int>(np),exc,cr,cs);
            const auto cf=c.evaluate_gga_fxc(rho,sigma,static_cast<int>(np),crr,crs,css);
            if(!cv) return std::unexpected("UDH U8b XC-II: "+cv.error());
            if(!cf) return std::unexpected("UDH U8b XC-II: "+cf.error());
            for(std::size_t k=0;k<vr.size();++k) vr[k]+=cr[k];
            for(std::size_t k=0;k<vs.size();++k) { vs[k]+=cs[k]; rr[k]+=crr[k]; }
            for(std::size_t k=0;k<rs.size();++k) { rs[k]+=crs[k]; ss[k]+=css[k]; }
        }
        for(const auto *a:{&vr,&vs,&rr,&rs,&ss}) for(double value:*a)
            if(!std::isfinite(value)) return std::unexpected("UDH U8b XC-II: nonfinite XC potential/kernel.");
        UDHCompleteXCIIGradient out;
        for(auto *m:{&out.p_side_fixed,&out.d_side_ao,&out.becke_partition,&out.point_translation})
            *m=M::Zero(static_cast<Eigen::Index>(mol.natoms),3);
        const std::array<const M*,2> pd{&in.ground_density.alpha,&in.ground_density.beta};
        const std::array<const M*,2> dd{&in.relaxed_density.alpha,&in.relaxed_density.beta};
        for(Eigen::Index p=0;p<np;++p)
        {
            if(grid.atomic_weights(p)==0) continue;
            SpinGradients u,v;
            u.col(0)=grad(ground->alpha,p); u.col(1)=grad(ground->beta,p);
            v.col(0)=grad(difference->alpha,p); v.col(1)=grad(difference->beta,p);
            detail::GGA5 fy; fy<<vr[2*p],vr[2*p+1],vs[3*p],vs[3*p+1],vs[3*p+2];
            const auto h=detail::unpack_udh_gga_kernel(rr.data()+3*p,rs.data()+6*p,ss.data()+6*p);
            const auto q=detail::udh_gga_xc_ii_partials({difference->alpha.rho(p),difference->beta.rho(p)},u,v,fy,h);
            const double weight=grid.points(p,3);
            const auto partition=becke_partition_owner_derivatives(grid,mol,p);
            if(!partition) return std::unexpected("UDH U8b XC-II: "+partition.error());
            for(std::size_t a=0;a<mol.natoms;++a) for(int k=0;k<3;++k)
            {
                for(int spin=0;spin<2;++spin)
                {
                    double dp=q.p_rho(spin)*drho_channel(*pd[spin],ao,p,a,k,*atoms);
                    double ddensity=q.d_rho(spin)*drho_channel(*dd[spin],ao,p,a,k,*atoms);
                    for(int g=0;g<3;++g)
                    {
                        dp+=q.p_gradient(g,spin)*dg_axis_spin(*pd[spin],ao,hess,p,g,a,k,*atoms);
                        ddensity+=q.d_gradient(g,spin)*dg_axis_spin(*dd[spin],ao,hess,p,g,a,k,*atoms);
                    }
                    out.p_side_fixed(a,k)+=weight*dp; out.d_side_ao(a,k)+=weight*ddensity;
                }
                out.becke_partition(a,k)+=grid.atomic_weights(p)*(*partition)(a,k)*q.scalar;
            }
            for(int k=0;k<3;++k) for(int spin=0;spin<2;++spin)
                out.point_translation(grid.owner(p),k)+=weight*(q.p_rho(spin)*u(k,spin)+q.d_rho(spin)*v(k,spin)+
                    q.p_gradient.col(spin).dot(spatial_column(*pd[spin],ao,hess,p,k))+
                    q.d_gradient.col(spin).dot(spatial_column(*dd[spin],ao,hess,p,k)));
        }
        out.total=out.p_side_fixed+out.d_side_ao+out.becke_partition+out.point_translation;
        for(const auto *m:{&out.p_side_fixed,&out.d_side_ao,&out.becke_partition,&out.point_translation,&out.total})
            if(!m->allFinite()) return std::unexpected("UDH U8b XC-II: nonfinite geometry channel.");
        return out;
    }
}
