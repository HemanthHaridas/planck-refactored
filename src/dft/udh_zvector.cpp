#include "udh_zvector.h"
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>

namespace DFT::Gradient
{
    namespace
    {
        using M=Eigen::MatrixXd;
        using V=Eigen::VectorXd;
        using Orb=UDHOrbitalMatrices;
        double maxabs(const M &m) { return m.size()?m.cwiseAbs().maxCoeff():0; }
        struct Layout { Eigen::Index oa,ob,va,vb,na,nb,n; };
        std::expected<Layout,std::string> layout(const UDHZVectorInputs &in)
        {
            const auto nao=in.overlap_ao.rows();
            if (nao<=0 || in.overlap_ao.cols()!=nao || !in.overlap_ao.allFinite() ||
                maxabs(in.overlap_ao-in.overlap_ao.transpose())>1e-12*(1+maxabs(in.overlap_ao)) ||
                Eigen::LLT<M>(in.overlap_ao).info()!=Eigen::Success)
                return std::unexpected("UDH U4: invalid positive-definite AO metric.");
            const auto valid=[nao,&in](const M &c,const M &f,int occupied)
            {
                if (c.rows()!=nao || c.cols()>nao || occupied<0 || occupied>c.cols() ||
                    f.rows()!=c.cols() || f.cols()!=c.cols() || !c.allFinite() || !f.allFinite()) return false;
                const M diagonal=f.diagonal().asDiagonal();
                return maxabs(c.transpose()*in.overlap_ao*c-M::Identity(c.cols(),c.cols()))<=1e-10 &&
                    maxabs(f-diagonal)<=1e-10*(1+maxabs(f));
            };
            const auto &op=in.response;
            if (!valid(in.mo_coeff.alpha,in.fock_mo.alpha,in.nocc_alpha) ||
                !valid(in.mo_coeff.beta,in.fock_mo.beta,in.nocc_beta) ||
                op.nbasis!=nao || !std::isfinite(op.exact_exchange) || op.exact_exchange<0 ||
                !op.coulomb || !op.xc || (op.exact_exchange!=0 && !op.exchange))
                return std::unexpected("UDH U4: invalid canonical orbitals, occupations or physical response.");
            Layout l{in.nocc_alpha,in.nocc_beta,in.mo_coeff.alpha.cols()-in.nocc_alpha,
                in.mo_coeff.beta.cols()-in.nocc_beta,0,0,0};
            const auto limit=std::numeric_limits<int>::max()/2;
            if ((l.va && l.oa>limit/l.va) || (l.vb && l.ob>limit/l.vb))
                return std::unexpected("UDH U4: orbital pair count overflow.");
            l.na=l.oa*l.va; l.nb=l.ob*l.vb; l.n=l.na+l.nb;
            return l;
        }
        bool valid_trial(const Orb &x,const Layout &l)
        { return x.alpha.rows()==l.va && x.alpha.cols()==l.oa && x.beta.rows()==l.vb &&
            x.beta.cols()==l.ob && x.alpha.allFinite() && x.beta.allFinite(); }
        V pack(const Orb &x,const Layout &l)
        {
            V v(l.n);
            for (Eigen::Index a=0;a<l.va;++a) for(Eigen::Index i=0;i<l.oa;++i) v(a*l.oa+i)=x.alpha(a,i);
            for (Eigen::Index a=0;a<l.vb;++a) for(Eigen::Index i=0;i<l.ob;++i) v(l.na+a*l.ob+i)=x.beta(a,i);
            return v;
        }
        Orb unpack(const V &v,const Layout &l)
        {
            Orb x{M(l.va,l.oa),M(l.vb,l.ob)};
            for (Eigen::Index a=0;a<l.va;++a) for(Eigen::Index i=0;i<l.oa;++i) x.alpha(a,i)=v(a*l.oa+i);
            for (Eigen::Index a=0;a<l.vb;++a) for(Eigen::Index i=0;i<l.ob;++i) x.beta(a,i)=v(l.na+a*l.ob+i);
            return x;
        }
        std::expected<UDHEq27Action,std::string> action(
            const UDHZVectorInputs &in,const Orb &x,const Layout &l)
        {
            if (!valid_trial(x,l)) return std::unexpected("UDH U4: invalid finite vo trial/RHS shapes.");
            const auto trial=[](const M &c,const M &u,Eigen::Index o)->M
            { const M raw=c.rightCols(u.rows())*u*c.leftCols(o).transpose(); return raw+raw.transpose(); };
            UDHEq27Action out;
            out.delta_density={trial(in.mo_coeff.alpha,x.alpha,l.oa),trial(in.mo_coeff.beta,x.beta,l.ob)};
            auto response=in.response.apply_channels(out.delta_density);
            if (!response) return std::unexpected(response.error());
            const auto project=[&in,&l](const UDHSpinMatrices &ao)->Orb
            { return {in.mo_coeff.alpha.rightCols(l.va).transpose()*ao.alpha*in.mo_coeff.alpha.leftCols(l.oa),
                in.mo_coeff.beta.rightCols(l.vb).transpose()*ao.beta*in.mo_coeff.beta.leftCols(l.ob)}; };
            out.coulomb=project({response->coulomb,response->coulomb});
            out.exchange=project(response->exchange);
            out.xc_from_alpha=project(response->xc_from_alpha);
            out.xc_from_beta=project(response->xc_from_beta);
            out.orbital={M::Zero(l.va,l.oa),M::Zero(l.vb,l.ob)};
            for (Eigen::Index a=0;a<l.va;++a) for(Eigen::Index i=0;i<l.oa;++i)
                out.orbital.alpha(a,i)=(in.fock_mo.alpha(l.oa+a,l.oa+a)-in.fock_mo.alpha(i,i))*x.alpha(a,i);
            for (Eigen::Index a=0;a<l.vb;++a) for(Eigen::Index i=0;i<l.ob;++i)
                out.orbital.beta(a,i)=(in.fock_mo.beta(l.ob+a,l.ob+a)-in.fock_mo.beta(i,i))*x.beta(a,i);
            out.total={out.orbital.alpha+out.coulomb.alpha+out.exchange.alpha+out.xc_from_alpha.alpha+out.xc_from_beta.alpha,
                out.orbital.beta+out.coulomb.beta+out.exchange.beta+out.xc_from_alpha.beta+out.xc_from_beta.beta};
            if (!valid_trial(out.total,l)) return std::unexpected("UDH U4: nonfinite orbital action.");
            return out;
        }
    }

    std::expected<UDHEq27Action,std::string> apply_udh_eq27_hessian(
        const UDHZVectorInputs &in,const UDHOrbitalMatrices &trial_ai)
    {
        auto l=layout(in); if (!l) return std::unexpected(l.error());
        return action(in,trial_ai,*l);
    }

    std::expected<UDHZVectorProducts,std::string> solve_udh_zvector(
        const UDHZVectorInputs &in,const UDHOrbitalMatrices &rhs,const UDHZVectorOptions &options)
    {
        auto dimensions=layout(in); if (!dimensions) return std::unexpected(dimensions.error());
        const auto l=*dimensions;
        if (!valid_trial(rhs,l) || !std::isfinite(options.residual_tolerance) || options.residual_tolerance<=0 ||
            !std::isfinite(options.adjoint_tolerance) || options.adjoint_tolerance<=0 ||
            !std::isfinite(options.minimum_rcond) || options.minimum_rcond<=0 || options.minimum_rcond>1)
            return std::unexpected("UDH U4: invalid RHS or solve controls.");
        UDHZVectorProducts out; out.snapshot=in; out.rhs_ai=rhs;
        const auto apply=[&](const V &v)->std::expected<V,std::string>
        {
            ++out.action_count;
            auto value=action(in,unpack(v,l),l);
            if (!value) return std::unexpected(value.error());
            return pack(value->total,l);
        };
        // Enforce homogeneous response, including the empty-pair boundary.
        auto zero=apply(V::Zero(l.n)); if (!zero) return std::unexpected(zero.error());
        if (maxabs(*zero)>options.residual_tolerance)
            return std::unexpected("UDH U4: nonzero action on zero trial.");
        out.jacobian=M(l.n,l.n);
        if (!l.n)
        { out.z_ai=unpack(V(0),l); out.residual_ai=out.z_ai; return out; }
        for (Eigen::Index col=0;col<l.n;++col)
        {
            V unit=V::Zero(l.n); unit(col)=1;
            auto value=apply(unit); if (!value) return std::unexpected(value.error());
            out.jacobian.col(col)=*value;
        }
        out.adjoint_max_abs=maxabs(out.jacobian-out.jacobian.transpose());
        if (out.adjoint_max_abs>options.adjoint_tolerance*std::max(1.0,maxabs(out.jacobian)))
            return std::unexpected("UDH U4: KS Jacobian violates joint-spin adjointness.");
        const Eigen::JacobiSVD<M> svd(out.jacobian);
        const V singular=svd.singularValues();
        if (!singular.allFinite() || singular(0)<=0)
            return std::unexpected("UDH U4: singular KS Jacobian.");
        out.reciprocal_condition=singular(l.n-1)/singular(0);
        if (out.reciprocal_condition<options.minimum_rcond)
            return std::unexpected("UDH U4: rank-deficient or ill-conditioned KS Jacobian.");
        const M transpose=out.jacobian.transpose();
        Eigen::ColPivHouseholderQR<M> qr(transpose); qr.setThreshold(options.minimum_rcond);
        out.rank=qr.rank();
        if (out.rank!=l.n) return std::unexpected("UDH U4: transpose solve is rank deficient.");
        const V b=-pack(rhs,l),z=qr.solve(b);
        if (!z.allFinite()) return std::unexpected("UDH U4: nonfinite Z solution.");
        // Fresh A^T z, not A z and not a residual against cached columns.
        V residual(l.n);
        for (Eigen::Index col=0;col<l.n;++col)
        {
            V unit=V::Zero(l.n); unit(col)=1;
            auto value=apply(unit); if (!value) return std::unexpected(value.error());
            residual(col)=z.dot(*value)-b(col);
        }
        out.residual_max_abs=maxabs(residual);
        const double limit=options.residual_tolerance*std::max(1.0,maxabs(b));
        if (!std::isfinite(limit) || !residual.allFinite() || out.residual_max_abs>limit)
            return std::unexpected("UDH U4: fresh transpose residual exceeds tolerance.");
        out.z_ai=unpack(z,l); out.residual_ai=unpack(residual,l);
        return out;
    }
}
