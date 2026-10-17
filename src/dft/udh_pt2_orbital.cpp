#include "udh_pt2_orbital.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace DFT::Gradient
{
    namespace
    {
        std::size_t ti(int i,int j,int a,int b,int oj,int va,int vb)
        { return ((static_cast<std::size_t>(i)*oj+j)*va+a)*vb+b; }
        std::size_t gi(int p,int q,int r,int s,int ns,int nt)
        { return ((static_cast<std::size_t>(p)*ns+q)*nt+r)*nt+s; }
        bool near(double a,double b)
        { return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=1e-10*(1+std::max(std::abs(a),std::abs(b))); }
        bool valid_integrals(const std::vector<double> &g,int ns,int nt,bool same)
        {
            std::size_t n=1;
            for (int d:{ns,ns,nt,nt})
            {
                if (d<0 || (d && n>std::numeric_limits<std::size_t>::max()/d)) return false;
                n*=static_cast<std::size_t>(d);
            }
            if (g.size()!=n) return false;
            for (int p=0;p<ns;++p) for (int q=0;q<ns;++q)
                for (int r=0;r<nt;++r) for (int s=0;s<nt;++s)
                {
                    double v=g[gi(p,q,r,s,ns,nt)];
                    if (!near(v,g[gi(q,p,r,s,ns,nt)]) || !near(v,g[gi(p,q,s,r,ns,nt)]) ||
                        (same && !near(v,g[gi(r,s,p,q,ns,nt)]))) return false;
                }
            return true;
        }
        UDHOrbitalMatrices zeros(int ra,int ca,int rb,int cb)
        { return {Eigen::MatrixXd::Zero(ra,ca),Eigen::MatrixXd::Zero(rb,cb)}; }
        UDHOrbitalMatrices sum(const UDHOrbitalMatrices &a,const UDHOrbitalMatrices &b)
        { return {a.alpha+b.alpha,a.beta+b.beta}; }
        bool finite(const UDHOrbitalMatrices &a) { return a.alpha.allFinite() && a.beta.allFinite(); }
        bool symmetric(const Eigen::MatrixXd &a,int n)
        { return a.rows()==n && a.cols()==n && a.allFinite() && (n==0 || (a-a.transpose()).cwiseAbs().maxCoeff()<1e-10); }

        UDHPairOrbitalSector sector(const std::vector<double> &t,const std::vector<double> &g,
            const UDHPT2Amplitudes &dims,int spin_s,int spin_t,double scale)
        {
            const int oa=dims.nocc_alpha,ob=dims.nocc_beta,va=dims.nvirt_alpha,vb=dims.nvirt_beta;
            const int na=oa+va,nb=ob+vb;
            const int os=spin_s?ob:oa,ot=spin_t?ob:oa,vs=spin_s?vb:va,vt=spin_t?vb:va;
            const int ns=os+vs,nt=ot+vt;
            UDHPairOrbitalSector out;
            for (auto &slot:out.slots) slot=zeros(na,na,nb,nb);
            auto &g0=spin_s?out.slots[0].beta:out.slots[0].alpha;
            auto &g1=spin_s?out.slots[1].beta:out.slots[1].alpha;
            auto &g2=spin_t?out.slots[2].beta:out.slots[2].alpha;
            auto &g3=spin_t?out.slots[3].beta:out.slots[3].alpha;
            // Differentiate each coefficient once: replace its MO index by p
            // and accumulate in column equal to that coefficient's index.
            for (int i=0;i<os;++i) for (int j=0;j<ot;++j)
                for (int a=0;a<vs;++a) for (int b=0;b<vt;++b)
                {
                    const double w=scale*t[ti(i,j,a,b,ot,vs,vt)];
                    for (int p=0;p<ns;++p)
                    {
                        g0(p,i)+=w*g[gi(p,os+a,j,ot+b,ns,nt)];
                        g1(p,os+a)+=w*g[gi(i,p,j,ot+b,ns,nt)];
                    }
                    for (int p=0;p<nt;++p)
                    {
                        g2(p,j)+=w*g[gi(i,os+a,p,ot+b,ns,nt)];
                        g3(p,ot+b)+=w*g[gi(i,os+a,j,p,ns,nt)];
                    }
                }
            out.g=zeros(na,na,nb,nb);
            for (const auto &slot:out.slots) out.g=sum(out.g,slot);
            out.external_ai={out.g.alpha.bottomLeftCorner(va,oa),out.g.beta.bottomLeftCorner(vb,ob)};
            out.internal_ai={-out.g.alpha.topRightCorner(oa,va).transpose(),-out.g.beta.topRightCorner(ob,vb).transpose()};
            out.total_ai=sum(out.external_ai,out.internal_ai);
            return out;
        }
    }

    std::expected<UDHPairOrbitalGradient,std::string> build_udh_pair_orbital_gradient(
        const UDHPT2Amplitudes &t,const UDHFullMOIntegrals &g,double c)
    {
        auto checked=validate_udh_pt2_amplitudes(t);
        if (!checked) return std::unexpected(checked.error());
        if (!std::isfinite(c) || g.n_alpha!=t.nocc_alpha+t.nvirt_alpha || g.n_beta!=t.nocc_beta+t.nvirt_beta ||
            !valid_integrals(g.aa,g.n_alpha,g.n_alpha,true) ||
            !valid_integrals(g.ab,g.n_alpha,g.n_beta,false) || !valid_integrals(g.bb,g.n_beta,g.n_beta,true))
            return std::unexpected("UDH U3 pair: invalid scale, MO integral dimensions, finite values or chemist symmetry.");
        UDHPairOrbitalGradient out;
        out.aa=sector(t.aa,g.aa,t,0,0,c);
        out.ab=sector(t.ab,g.ab,t,0,1,2*c);
        out.bb=sector(t.bb,g.bb,t,1,1,c);
        out.g=sum(sum(out.aa.g,out.ab.g),out.bb.g);
        out.total_ai=sum(sum(out.aa.total_ai,out.ab.total_ai),out.bb.total_ai);
        for (const auto *s:{&out.aa,&out.ab,&out.bb})
        {
            for (const auto &slot:s->slots) if (!finite(slot)) return std::unexpected("UDH U3 pair: nonfinite slot.");
            if (!finite(s->g) || !finite(s->total_ai)) return std::unexpected("UDH U3 pair: nonfinite sector.");
        }
        if (!finite(out.g) || !finite(out.total_ai)) return std::unexpected("UDH U3 pair: nonfinite total.");
        return out;
    }

    std::expected<UDHOrbitalRHS,std::string> build_udh_orbital_rhs(
        const UDHPT2Amplitudes &t,const UDHFullMOIntegrals &g,double c,
        const UDHOrbitalMatrices &coeff,const Eigen::Ref<const Eigen::MatrixXd> &s,
        const UDHOrbitalMatrices &f,const UDHKSResponseOperator &response)
    {
        auto pair=build_udh_pair_orbital_gradient(t,g,c);
        if (!pair) return std::unexpected(pair.error());
        if (!symmetric(f.alpha,g.n_alpha) || !symmetric(f.beta,g.n_beta) || response.nbasis!=s.rows())
            return std::unexpected("UDH U3 RHS: invalid Fock matrix or response AO dimension.");
        auto d=build_udh_pt2_dprime(t,c);
        if (!d) return std::unexpected(d.error());
        auto ao=transform_udh_pt2_dprime_to_ao(*d,coeff.alpha,coeff.beta,s);
        if (!ao) return std::unexpected(ao.error());
        auto action=response.apply_channels({ao->alpha_ao,ao->beta_ao});
        if (!action) return std::unexpected("UDH U3 RHS: "+action.error());
        const int oa=t.nocc_alpha,ob=t.nocc_beta,va=t.nvirt_alpha,vb=t.nvirt_beta;
        const auto project=[&](const UDHSpinMatrices &q) -> UDHOrbitalMatrices
        { return {2.0*coeff.alpha.rightCols(va).transpose()*q.alpha*coeff.alpha.leftCols(oa),
                  2.0*coeff.beta.rightCols(vb).transpose()*q.beta*coeff.beta.leftCols(ob)}; };
        UDHOrbitalRHS out;
        out.pair=std::move(*pair); out.dprime=std::move(*d); out.dprime_ao=std::move(*ao);
        out.response_ao=std::move(*action);
        out.response_j=project({out.response_ao.coulomb,out.response_ao.coulomb});
        out.response_k=project(out.response_ao.exchange);
        out.xc_from_alpha=project(out.response_ao.xc_from_alpha);
        out.xc_from_beta=project(out.response_ao.xc_from_beta);
        out.response_ai=project(out.response_ao.total);
        const Eigen::MatrixXd fd_a=f.alpha*out.dprime.alpha_mo,fd_b=f.beta*out.dprime.beta_mo;
        out.fock_connection_ai={2.0*(fd_a.bottomLeftCorner(va,oa)-fd_a.topRightCorner(oa,va).transpose()),
                                2.0*(fd_b.bottomLeftCorner(vb,ob)-fd_b.topRightCorner(ob,vb).transpose())};
        out.total_ai=sum(sum(out.pair.total_ai,out.response_ai),out.fock_connection_ai);
        for (const auto *q:{&out.response_j,&out.response_k,&out.xc_from_alpha,&out.xc_from_beta,
                            &out.response_ai,&out.fock_connection_ai,&out.total_ai})
            if (!finite(*q)) return std::unexpected("UDH U3 RHS: nonfinite projected channel.");
        return out;
    }
}
