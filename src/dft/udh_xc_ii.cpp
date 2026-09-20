#include "udh_xc_ii.h"
#include "dft_gradient.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace DFT::Gradient
{
    namespace
    {
        using M=Eigen::MatrixXd;
        bool symmetric(const M &m,Eigen::Index n)
        { return m.rows()==n && m.cols()==n && m.allFinite() &&
            (m-m.transpose()).cwiseAbs().maxCoeff()<=1e-12*(1+m.cwiseAbs().maxCoeff()); }
        bool lda(const XC::Functional &f)
        { return f.is_lda_like() && f.spin()==XC::Spin::Polarized && f.get()->info &&
            (f.get()->info->flags & XC_FLAGS_HAVE_VXC) && (f.get()->info->flags & XC_FLAGS_HAVE_FXC); }
        bool finite(const DensityChannelOnGrid &d)
        { return d.rho.allFinite() && d.grad_x.allFinite() && d.grad_y.allFinite() && d.grad_z.allFinite(); }
        double gradient(const DensityChannelOnGrid &d,Eigen::Index p,int k)
        { return k==0?d.grad_x(p):k==1?d.grad_y(p):d.grad_z(p); }
    }

    std::expected<UDHCompleteXCIIGradient,std::string>
    build_udh_lda_complete_xc_ii_gradient(const UDHXCIIInputs &in)
    {
        if(!in.molecule || !in.basis || !in.molecular_grid || !in.ao_grid ||
            !in.exchange_functional || !in.correlation_functional)
            return std::unexpected("UDH U8a XC-II: missing input snapshot.");
        const auto &x=*in.exchange_functional,&c=*in.correlation_functional;
        const bool combined=x.is_combined_exchange_correlation();
        if(!lda(x) || (!combined && !lda(c)))
            return std::unexpected("UDH U8a XC-II: requires polarized LDA vxc/fxc functionals.");
        const auto &mol=*in.molecule; const auto &grid=*in.molecular_grid; const auto &ao=*in.ao_grid;
        const auto np=ao.npoints(),nb=ao.nbasis();
        if(mol.natoms==0 || mol.natoms>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            mol._standard.rows()!=static_cast<Eigen::Index>(mol.natoms) || mol._standard.cols()!=3 ||
            !mol._standard.allFinite() || mol.atomic_numbers.size()!=static_cast<Eigen::Index>(mol.natoms) ||
            (mol.atomic_numbers.array()<=0).any() || nb<=0 || in.basis->nbasis()!=static_cast<std::size_t>(nb) ||
            np<=0 || np>std::numeric_limits<int>::max()/3 || grid.points.rows()!=np || grid.points.cols()!=4 ||
            grid.owner.size()!=np || grid.atomic_weights.size()!=np || grid.partition_weights.size()!=np ||
            !grid.points.allFinite() || !grid.atomic_weights.allFinite() || !grid.partition_weights.allFinite())
            return std::unexpected("UDH U8a XC-II: invalid molecule, AO or grid dimensions/values.");
        for(const auto *m:{&ao.values,&ao.grad_x,&ao.grad_y,&ao.grad_z})
            if(m->rows()!=np || m->cols()!=nb || !m->allFinite())
                return std::unexpected("UDH U8a XC-II: invalid AO values/first derivatives.");
        for(const auto *d:{&in.ground_density.alpha,&in.ground_density.beta,&in.relaxed_density.alpha,&in.relaxed_density.beta})
            if(!symmetric(*d,nb)) return std::unexpected("UDH U8a XC-II: densities must be finite symmetric spin AO matrices.");
        for(Eigen::Index p=0;p<np;++p)
        {
            const double expected=grid.atomic_weights(p)*grid.partition_weights(p);
            if(grid.owner(p)<0 || grid.owner(p)>=static_cast<int>(mol.natoms) ||
                grid.partition_weights(p)<0 || grid.partition_weights(p)>1 || !std::isfinite(expected) ||
                std::abs(grid.points(p,3)-expected)>1e-12*(1+std::abs(expected)))
                return std::unexpected("UDH U8a XC-II: inconsistent owner/partition/atomic weight metadata.");
        }
        const auto atoms=atom_bf_lists(mol,*in.basis);
        if(!atoms) return std::unexpected("UDH U8a XC-II: "+atoms.error());
        auto ground=evaluate_density_on_grid(ao,in.ground_density.alpha,in.ground_density.beta);
        auto difference=evaluate_density_on_grid(ao,in.relaxed_density.alpha,in.relaxed_density.beta);
        if(!ground || !difference) return std::unexpected("UDH U8a XC-II: density evaluation failed.");
        if(!finite(ground->alpha) || !finite(ground->beta) || !finite(difference->alpha) || !finite(difference->beta) ||
            ground->alpha.rho.minCoeff()<0 || ground->beta.rho.minCoeff()<0)
            return std::unexpected("UDH U8a XC-II: nonfinite fields or negative reference spin density.");
        // D is signed: it is never packed as a ground density for libxc.
        std::vector<double> rho(static_cast<std::size_t>(2*np));
        for(Eigen::Index p=0;p<np;++p) { rho[2*p]=ground->alpha.rho(p); rho[2*p+1]=ground->beta.rho(p); }
        std::vector<double> exc,v,rr;
        const auto vx=x.evaluate_lda_exc_vxc(rho,static_cast<int>(np),exc,v);
        const auto fx=x.evaluate_lda_fxc(rho,static_cast<int>(np),rr);
        if(!vx) return std::unexpected("UDH U8a XC-II: "+vx.error());
        if(!fx) return std::unexpected("UDH U8a XC-II: "+fx.error());
        if(!combined)
        {
            std::vector<double> vc,rc;
            const auto cv=c.evaluate_lda_exc_vxc(rho,static_cast<int>(np),exc,vc);
            const auto cf=c.evaluate_lda_fxc(rho,static_cast<int>(np),rc);
            if(!cv) return std::unexpected("UDH U8a XC-II: "+cv.error());
            if(!cf) return std::unexpected("UDH U8a XC-II: "+cf.error());
            for(std::size_t k=0;k<v.size();++k) v[k]+=vc[k];
            for(std::size_t k=0;k<rr.size();++k) rr[k]+=rc[k];
        }
        for(const auto *a:{&v,&rr}) for(double value:*a)
            if(!std::isfinite(value)) return std::unexpected("UDH U8a XC-II: nonfinite XC potential/kernel.");
        UDHCompleteXCIIGradient out;
        for(auto *m:{&out.p_side_fixed,&out.d_side_ao,&out.becke_partition,&out.point_translation})
            *m=M::Zero(static_cast<Eigen::Index>(mol.natoms),3);
        for(Eigen::Index p=0;p<np;++p)
        {
            if(grid.atomic_weights(p)==0) continue;
            const double weight=grid.points(p,3),da=difference->alpha.rho(p),db=difference->beta.rho(p);
            const double va=v[2*p],vb=v[2*p+1];
            // Coefficients of d(rho_Palpha), d(rho_Pbeta) in D^T f_rr dP.
            const double qa=da*rr[3*p]+db*rr[3*p+1];
            const double qb=da*rr[3*p+1]+db*rr[3*p+2];
            const double integrand=va*da+vb*db;
            const auto partition=becke_partition_owner_derivatives(grid,mol,p);
            if(!partition) return std::unexpected("UDH U8a XC-II: "+partition.error());
            for(std::size_t a=0;a<mol.natoms;++a) for(int k=0;k<3;++k)
            {
                out.p_side_fixed(a,k)+=weight*(qa*drho_channel(in.ground_density.alpha,ao,p,a,k,*atoms)+
                    qb*drho_channel(in.ground_density.beta,ao,p,a,k,*atoms));
                out.d_side_ao(a,k)+=weight*(va*drho_channel(in.relaxed_density.alpha,ao,p,a,k,*atoms)+
                    vb*drho_channel(in.relaxed_density.beta,ao,p,a,k,*atoms));
                out.becke_partition(a,k)+=grid.atomic_weights(p)*(*partition)(a,k)*integrand;
            }
            for(int k=0;k<3;++k)
                out.point_translation(grid.owner(p),k)+=weight*(qa*gradient(ground->alpha,p,k)+
                    qb*gradient(ground->beta,p,k)+va*gradient(difference->alpha,p,k)+vb*gradient(difference->beta,p,k));
        }
        out.total=out.p_side_fixed+out.d_side_ao+out.becke_partition+out.point_translation;
        for(const auto *m:{&out.p_side_fixed,&out.d_side_ao,&out.becke_partition,&out.point_translation,&out.total})
            if(!m->allFinite()) return std::unexpected("UDH U8a XC-II: nonfinite geometry channel.");
        return out;
    }
}
