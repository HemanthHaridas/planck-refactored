#include "udh_two_particle_density.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace DFT::Gradient
{
    namespace
    {
        using M=Eigen::MatrixXd;
        std::size_t idx(int p,int q,int r,int s,int n)
        { return ((static_cast<std::size_t>(p)*n+q)*n+r)*n+s; }
        std::size_t ti(int i,int j,int a,int b,int oj,int va,int vb)
        { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
        double maxabs(const M &m) { return m.size()?m.cwiseAbs().maxCoeff():0.0; }
        bool close(double a,double b)
        { return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=1e-11*(1+std::max(std::abs(a),std::abs(b))); }
        bool equal(const M &a,const M &b)
        { return a.rows()==b.rows() && a.cols()==b.cols() && a.allFinite() && b.allFinite() &&
            maxabs(a-b)<=1e-11*(1+std::max(maxabs(a),maxabs(b))); }
        bool symmetric(const M &m,int n)
        { return m.rows()==n && m.cols()==n && equal(m,m.transpose()); }
        bool extent(Eigen::Index n,std::size_t &n4)
        {
            if(n<=0 || n>std::numeric_limits<int>::max()) return false;
            n4=1;
            for(int k=0;k<4;++k)
            {
                if(n4>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(n)) return false;
                n4*=static_cast<std::size_t>(n);
            }
            return n4<=std::vector<double>().max_size() && n4<=static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
        }
        double average(const std::vector<double> &v,int p,int q,int r,int s,int n)
        {
            // Scale before summing to avoid unnecessary overflow.
            return .125*v[idx(p,q,r,s,n)]+.125*v[idx(q,p,r,s,n)]+
                .125*v[idx(p,q,s,r,n)]+.125*v[idx(q,p,s,r,n)]+
                .125*v[idx(r,s,p,q,n)]+.125*v[idx(s,r,p,q,n)]+
                .125*v[idx(r,s,q,p,n)]+.125*v[idx(s,r,q,p,n)];
        }
        bool adapt(UDHERICoefficients &g,int n)
        {
            g.symmetric_ao.resize(g.raw_ao.size());
            for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            {
                const auto k=idx(p,q,r,s,n);
                if(!std::isfinite(g.raw_ao[k])) return false;
                g.symmetric_ao[k]=average(g.raw_ao,p,q,r,s,n);
                if(!std::isfinite(g.symmetric_ao[k])) return false;
            }
            return true;
        }
        bool valid(const UDHERICoefficients &g,int n,std::size_t n4)
        {
            if(g.raw_ao.size()!=n4 || g.symmetric_ao.size()!=n4) return false;
            for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
                if(!close(g.symmetric_ao[idx(p,q,r,s,n)],average(g.raw_ao,p,q,r,s,n))) return false;
            return true;
        }
        bool valid_eri(const std::vector<double> &v,int n,std::size_t n4)
        {
            if(v.size()!=n4) return false;
            for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            {
                const double x=v[idx(p,q,r,s,n)];
                if(!close(x,v[idx(q,p,r,s,n)]) || !close(x,v[idx(p,q,s,r,n)]) ||
                    !close(x,v[idx(r,s,p,q,n)])) return false;
            }
            return true;
        }
        // Same row-major, successive-index method as O5, with independent
        // alpha/beta coefficient matrices on the two electron pairs.
        std::vector<double> transform_axis(const std::vector<double> &in,
            std::array<std::size_t,4> &shape,int axis,const M &c)
        {
            std::size_t prefix=1,suffix=1;
            for(int k=0;k<axis;++k) prefix*=shape[k];
            for(int k=axis+1;k<4;++k) suffix*=shape[k];
            const auto old=shape[axis],n=static_cast<std::size_t>(c.rows());
            std::vector<double> out(prefix*n*suffix);
            using Row=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
            constexpr std::size_t tile=128;
            if(suffix==1)
            {
                const Eigen::Map<const Row> a(in.data(),prefix,old);
                Eigen::Map<Row> b(out.data(),prefix,n);
                for(std::size_t k=0;k<prefix;k+=tile)
                {
                    const auto width=std::min(tile,prefix-k);
                    b.middleRows(k,width).noalias()=a.middleRows(k,width)*c.transpose();
                }
            }
            else for(std::size_t p=0;p<prefix;++p)
            {
                const Eigen::Map<const Row> a(in.data()+p*old*suffix,old,suffix);
                Eigen::Map<Row> b(out.data()+p*n*suffix,n,suffix);
                for(std::size_t k=0;k<suffix;k+=tile)
                {
                    const auto width=std::min(tile,suffix-k);
                    b.middleCols(k,width).noalias()=c*a.middleCols(k,width);
                }
            }
            shape[axis]=n;
            return out;
        }
        std::vector<double> pair(const std::vector<double> &t,int os,int ot,int vs,int vt,
            const M &cs,const M &ct,double scale,std::size_t n4)
        {
            if(!os || !ot || !vs || !vt) return std::vector<double>(n4,0.0);
            std::array<std::size_t,4> shape{static_cast<std::size_t>(os),static_cast<std::size_t>(vs),
                static_cast<std::size_t>(ot),static_cast<std::size_t>(vt)};
            std::vector<double> v(t.size());
            for(int i=0;i<os;++i) for(int a=0;a<vs;++a) for(int j=0;j<ot;++j) for(int b=0;b<vt;++b)
                v[((static_cast<std::size_t>(i)*vs+a)*ot+j)*vt+b]=scale*t[ti(i,j,a,b,ot,vs,vt)];
            v=transform_axis(v,shape,3,ct.rightCols(vt));
            v=transform_axis(v,shape,2,ct.leftCols(ot));
            v=transform_axis(v,shape,1,cs.rightCols(vs));
            return transform_axis(v,shape,0,cs.leftCols(os));
        }
        std::array<const UDHERICoefficients*,6> channels(const UDHTwoParticleDensity &g)
        { return {&g.coulomb,&g.exchange_alpha,&g.exchange_beta,&g.pair_aa,&g.pair_ab,&g.pair_bb}; }
    }

    std::expected<UDHTwoParticleDensity,std::string> build_udh_two_particle_density(
        const UDHPT2Amplitudes &t,double c,const UDHRelaxedDifferenceDensity &d,
        const UDHOrbitalMatrices &coeff,const Eigen::Ref<const M> &s,double ax)
    {
        std::size_t n4=0;
        if(!extent(s.rows(),n4) || !std::isfinite(ax) || ax<0)
            return std::unexpected("UDH U7: invalid AO extent or exact-exchange coefficient.");
        // Bound spin matrix allocations before U1 constructs Dprime. Empty
        // amplitudes with a huge occupied/virtual extent are not a loophole.
        for(auto [o,v]:{std::pair{t.nocc_alpha,t.nvirt_alpha},std::pair{t.nocc_beta,t.nvirt_beta}})
            if(o<0 || v<0 || static_cast<long long>(o)+v>s.rows())
                return std::unexpected("UDH U7: spin extent exceeds the AO frame.");
        auto dp=build_udh_pt2_dprime(t,c);
        if(!dp) return std::unexpected("UDH U7: "+dp.error());
        auto ao=transform_udh_relaxed_density_to_ao(d,coeff,s);
        if(!ao) return std::unexpected("UDH U7: "+ao.error());
        const int oa=t.nocc_alpha,ob=t.nocc_beta,va=t.nvirt_alpha,vb=t.nvirt_beta;
        const auto same_prime=[](const M &raw,const M &prime,int o,int v)
        { return raw.rows()==o+v && equal(raw.topLeftCorner(o,o),prime.topLeftCorner(o,o)) &&
            equal(raw.bottomRightCorner(v,v),prime.bottomRightCorner(v,v)); };
        if(d.nocc_alpha!=oa || d.nocc_beta!=ob || !same_prime(d.raw_mo.alpha,dp->alpha_mo,oa,va) ||
            !same_prime(d.raw_mo.beta,dp->beta_mo,ob,vb))
            return std::unexpected("UDH U7: relaxed Dprime does not match amplitudes and PT2 scale.");
        UDHTwoParticleDensity out;
        out.n_ao=static_cast<int>(s.rows()); out.relaxed_ao=std::move(*ao);
        out.ground_ao={coeff.alpha.leftCols(oa)*coeff.alpha.leftCols(oa).transpose(),
                       coeff.beta.leftCols(ob)*coeff.beta.leftCols(ob).transpose()};
        const M p=out.ground_ao.alpha+out.ground_ao.beta;
        out.coulomb.raw_ao.resize(n4); out.exchange_alpha.raw_ao.resize(n4); out.exchange_beta.raw_ao.resize(n4);
        const int n=out.n_ao;
        for(int mu=0;mu<n;++mu) for(int nu=0;nu<n;++nu) for(int ka=0;ka<n;++ka) for(int ta=0;ta<n;++ta)
        {
            const auto k=idx(mu,nu,ka,ta,n);
            out.coulomb.raw_ao[k]=out.relaxed_ao.total(mu,nu)*p(ka,ta);
            out.exchange_alpha.raw_ao[k]=-ax*out.relaxed_ao.spin.alpha(mu,ka)*out.ground_ao.alpha(nu,ta);
            out.exchange_beta.raw_ao[k]=-ax*out.relaxed_ao.spin.beta(mu,ka)*out.ground_ao.beta(nu,ta);
        }
        out.pair_aa.raw_ao=pair(t.aa,oa,oa,va,va,coeff.alpha,coeff.alpha,c,n4);
        out.pair_ab.raw_ao=pair(t.ab,oa,ob,va,vb,coeff.alpha,coeff.beta,2*c,n4);
        out.pair_bb.raw_ao=pair(t.bb,ob,ob,vb,vb,coeff.beta,coeff.beta,c,n4);
        for(auto *g:{&out.coulomb,&out.exchange_alpha,&out.exchange_beta,&out.pair_aa,&out.pair_ab,&out.pair_bb})
            if(!adapt(*g,n)) return std::unexpected("UDH U7: nonfinite ERI coefficient.");
        return out;
    }

    std::expected<UDHNonXCGradient,std::string> build_udh_non_xc_gradient(
        const UDHTwoParticleDensity &g,const UDHOverlapDensity &w,const DHEq33NonXCDerivatives &dx)
    {
        std::size_t n4=0;
        const int n=g.n_ao;
        const auto nc=dx.hamiltonian_ao.size();
        if(!extent(n,n4) || nc>static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) ||
            dx.overlap_ao.size()!=nc || dx.eri_ao.size()!=nc ||
            !symmetric(g.relaxed_ao.spin.alpha,n) || !symmetric(g.relaxed_ao.spin.beta,n) ||
            !symmetric(g.relaxed_ao.total,n) || !equal(g.relaxed_ao.total,g.relaxed_ao.spin.alpha+g.relaxed_ao.spin.beta) ||
            !symmetric(w.symmetric_ao.alpha,n) || !symmetric(w.symmetric_ao.beta,n) ||
            !symmetric(w.total_ao,n) || !equal(w.total_ao,w.symmetric_ao.alpha+w.symmetric_ao.beta))
            return std::unexpected("UDH U7 non-XC: incompatible or nonfinite AO densities/coordinate counts.");
        for(const auto *q:channels(g)) if(!valid(*q,n,n4))
            return std::unexpected("UDH U7 non-XC: invalid raw/eightfold ERI coefficient contract.");
        UDHNonXCGradient out;
        out.one_electron=Eigen::VectorXd::Zero(nc); out.overlap=Eigen::VectorXd::Zero(nc);
        std::array<Eigen::VectorXd*,6> values{&out.coulomb,&out.exchange_alpha,&out.exchange_beta,
            &out.pair_aa,&out.pair_ab,&out.pair_bb};
        for(auto *v:values) *v=Eigen::VectorXd::Zero(nc);
        const auto tensors=channels(g);
        for(std::size_t x=0;x<nc;++x)
        {
            if(!symmetric(dx.hamiltonian_ao[x],n) || !symmetric(dx.overlap_ao[x],n) || !valid_eri(dx.eri_ao[x],n,n4))
                return std::unexpected("UDH U7 non-XC: invalid finite symmetric derivative at coordinate "+std::to_string(x)+".");
            out.one_electron(x)=(g.relaxed_ao.total.array()*dx.hamiltonian_ao[x].array()).sum();
            out.overlap(x)=(w.total_ao.array()*dx.overlap_ao[x].array()).sum();
            for(std::size_t k=0;k<n4;++k) for(int ch=0;ch<6;++ch)
                (*values[ch])(x)+=tensors[ch]->symmetric_ao[k]*dx.eri_ao[x][k];
        }
        out.two_electron_separable=out.coulomb+out.exchange_alpha+out.exchange_beta;
        out.two_electron_nonseparable=out.pair_aa+out.pair_ab+out.pair_bb;
        out.two_electron=out.two_electron_separable+out.two_electron_nonseparable;
        out.total=out.one_electron+out.overlap+out.two_electron;
        for(const auto *v:values) if(!v->allFinite()) return std::unexpected("UDH U7 non-XC: nonfinite ERI contraction.");
        if(!out.one_electron.allFinite() || !out.overlap.allFinite() || !out.two_electron_separable.allFinite() ||
            !out.two_electron_nonseparable.allFinite() || !out.two_electron.allFinite() || !out.total.allFinite())
            return std::unexpected("UDH U7 non-XC: nonfinite gradient channel.");
        return out;
    }
}
