#include "udh_relaxed_density.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <limits>

namespace DFT::Gradient
{
    namespace
    {
        using M=Eigen::MatrixXd;
        double maxabs(const M &x) { return x.size()?x.cwiseAbs().maxCoeff():0.0; }
        bool symmetric(const M &x)
        { return x.rows()==x.cols() && x.allFinite() &&
            maxabs(x-x.transpose())<=1e-12*(1+maxabs(x)); }
        bool diagonal_blocks(const M &d,Eigen::Index o)
        {
            if(!symmetric(d) || o<0 || o>d.rows()) return false;
            const auto v=d.rows()-o;
            return d.topRightCorner(o,v).isZero(0.0) && d.bottomLeftCorner(v,o).isZero(0.0);
        }
        bool raw_layout(const M &raw,const M &sym,int o)
        {
            if(raw.rows()!=raw.cols() || o<0 || o>raw.rows() || !raw.allFinite() ||
                sym.rows()!=raw.rows() || sym.cols()!=raw.cols() || !symmetric(sym)) return false;
            const auto v=raw.rows()-o;
            if(!raw.topRightCorner(o,v).isZero(0.0) ||
                !symmetric(raw.topLeftCorner(o,o)) || !symmetric(raw.bottomRightCorner(v,v))) return false;
            const M expected=0.5*raw+0.5*raw.transpose();
            return expected.allFinite() && maxabs(sym-expected)<=1e-12*(1+maxabs(expected));
        }
    }

    std::expected<UDHRelaxedDifferenceDensity,std::string>
    build_udh_relaxed_difference_density(const UDHPT2DPrime &dprime,const UDHOrbitalMatrices &z_ai)
    {
        const auto valid=[](const M &d,const M &z)
        {
            return z.rows()<=std::numeric_limits<int>::max() && z.cols()<=std::numeric_limits<int>::max() &&
                z.allFinite() && d.rows()==z.rows()+z.cols() && diagonal_blocks(d,z.cols());
        };
        if(!valid(dprime.alpha_mo,z_ai.alpha) || !valid(dprime.beta_mo,z_ai.beta))
            return std::unexpected("UDH U5: invalid Dprime blocks or finite vo Z shape.");
        UDHRelaxedDifferenceDensity out;
        out.nocc_alpha=static_cast<int>(z_ai.alpha.cols()); out.nocc_beta=static_cast<int>(z_ai.beta.cols());
        const auto assemble=[](const M &d,const M &z,M &raw,M &sym)
        {
            raw=d;
            raw.bottomLeftCorner(z.rows(),z.cols())=z;
            sym=0.5*raw+0.5*raw.transpose();
        };
        assemble(dprime.alpha_mo,z_ai.alpha,out.raw_mo.alpha,out.symmetric_mo.alpha);
        assemble(dprime.beta_mo,z_ai.beta,out.raw_mo.beta,out.symmetric_mo.beta);
        return out;
    }

    std::expected<UDHRelaxedAODensity,std::string>
    transform_udh_relaxed_density_to_ao(const UDHRelaxedDifferenceDensity &d,
        const UDHOrbitalMatrices &c,const Eigen::Ref<const M> &s)
    {
        const auto n=s.rows();
        if(n<=0 || !symmetric(s) || Eigen::LLT<M>(s).info()!=Eigen::Success)
            return std::unexpected("UDH U5 AO: invalid positive-definite metric.");
        const auto valid=[&](const M &coeff,const M &raw,const M &sym,int o)
        {
            if(!raw_layout(raw,sym,o) || coeff.rows()!=n || coeff.cols()!=raw.rows() ||
                coeff.cols()>n || !coeff.allFinite()) return false;
            const M gram=coeff.transpose()*s*coeff;
            return gram.allFinite() && maxabs(gram-M::Identity(coeff.cols(),coeff.cols()))<=1e-10;
        };
        if(!valid(c.alpha,d.raw_mo.alpha,d.symmetric_mo.alpha,d.nocc_alpha) ||
            !valid(c.beta,d.raw_mo.beta,d.symmetric_mo.beta,d.nocc_beta))
            return std::unexpected("UDH U5 AO: inconsistent raw/symmetric density or C^T S C frame.");
        UDHRelaxedAODensity out;
        out.spin.alpha=c.alpha*d.symmetric_mo.alpha*c.alpha.transpose();
        out.spin.beta=c.beta*d.symmetric_mo.beta*c.beta.transpose();
        out.total=out.spin.alpha+out.spin.beta;
        if(!out.spin.alpha.allFinite() || !out.spin.beta.allFinite() || !out.total.allFinite())
            return std::unexpected("UDH U5 AO: nonfinite transformed density.");
        return out;
    }
}
