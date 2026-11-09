#ifndef PLANCK_TESTS_UDH_DENSE_ORACLE_H
#define PLANCK_TESTS_UDH_DENSE_ORACLE_H

// Test-only dense view of the UKS Eq. 34/35 source. The tensors are rebuilt
// from the production source, so literal-index oracles still exercise the
// streamed generator. Production keeps none of these n_ao^4 objects.

#include "dft/udh_two_particle_density.h"

#include <array>
#include <stdexcept>
#include <string>

namespace DFT::Gradient::TestOracle
{
    using Vec=std::vector<double>;
    struct ERICoefficients { Vec raw_ao,symmetric_ao; };

    struct Tensors
    {
        int n_ao=0;
        UDHRelaxedAODensity relaxed_ao;
        UDHSpinMatrices ground_ao;
        ERICoefficients coulomb,exchange_alpha,exchange_beta,pair_aa,pair_ab,pair_bb;
    };

    struct DenseNonXC
    {
        Eigen::VectorXd one_electron,overlap;
        Eigen::VectorXd coulomb,exchange_alpha,exchange_beta,pair_aa,pair_ab,pair_bb;
        Eigen::VectorXd total;
    };

    inline std::size_t oracle_index(int p,int q,int r,int s,int n)
    { return ((static_cast<std::size_t>(p)*n+q)*n+r)*n+s; }

    inline void adapt(ERICoefficients &g,int n)
    {
        const auto v=[&](int p,int q,int r,int s) { return .125*g.raw_ao[oracle_index(p,q,r,s,n)]; };
        g.symmetric_ao.resize(g.raw_ao.size());
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            g.symmetric_ao[oracle_index(p,q,r,s,n)]=v(p,q,r,s)+v(q,p,r,s)+v(p,q,s,r)+v(q,p,s,r)+
                v(r,s,p,q)+v(s,r,p,q)+v(r,s,q,p)+v(s,r,q,p);
    }

    inline Vec stream_channel(const DHPairChannel &channel,int n,int tile)
    {
        const std::size_t n4=static_cast<std::size_t>(n)*n*n*n;
        Vec out(n4,0.0);
        const auto streamed=for_each_dh_pair_ao_block(n,{channel},tile,tile,tile,tile,tile,
            [&](int m0,int mc,int n0,int nc,int k0,int kc,int t0,int tc,const Vec &block)
            -> std::expected<void,std::string>
            {
                for(int lm=0;lm<mc;++lm) for(int ln=0;ln<nc;++ln) for(int lk=0;lk<kc;++lk) for(int lt=0;lt<tc;++lt)
                    out[oracle_index(m0+lm,n0+ln,k0+lk,t0+lt,n)]=
                        block[((static_cast<std::size_t>(lm)*nc+ln)*kc+lk)*tc+lt];
                return {};
            });
        if(!streamed) throw std::runtime_error(streamed.error());
        return out;
    }

    inline Tensors materialize(const UDHTwoParticleDensity &g,const DHEq33TwoParticleSource &source)
    {
        Tensors out;
        const int n=g.n_ao;
        const std::size_t n4=static_cast<std::size_t>(n)*n*n*n;
        out.n_ao=n; out.relaxed_ao=g.relaxed_ao; out.ground_ao=g.ground_ao;
        out.coulomb.raw_ao.resize(n4); out.exchange_alpha.raw_ao.resize(n4); out.exchange_beta.raw_ao.resize(n4);
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
        {
            const auto k=oracle_index(p,q,r,s,n);
            out.coulomb.raw_ao[k]=source.coulomb_d(p,q)*source.coulomb_p(r,s);
            out.exchange_alpha.raw_ao[k]=source.exchange[0].factor*source.exchange[0].d(p,r)*source.exchange[0].p(q,s);
            out.exchange_beta.raw_ao[k]=source.exchange[1].factor*source.exchange[1].d(p,r)*source.exchange[1].p(q,s);
        }
        out.pair_aa.raw_ao=stream_channel(source.pair[0],n,2);
        out.pair_ab.raw_ao=stream_channel(source.pair[1],n,3);
        out.pair_bb.raw_ao=stream_channel(source.pair[2],n,1);
        for(auto *t:{&out.coulomb,&out.exchange_alpha,&out.exchange_beta,&out.pair_aa,&out.pair_ab,&out.pair_bb})
            adapt(*t,n);
        return out;
    }

    // The contraction the callbacks stand in for, one channel at a time.
    inline DenseNonXC dense_non_xc(const Tensors &g,const Eigen::MatrixXd &w_total,const DHEq33NonXCDerivatives &dx)
    {
        const auto nc=static_cast<Eigen::Index>(dx.hamiltonian_ao.size());
        DenseNonXC out;
        out.one_electron=out.overlap=Eigen::VectorXd::Zero(nc);
        const std::array<const ERICoefficients*,6> tensors{&g.coulomb,&g.exchange_alpha,&g.exchange_beta,
            &g.pair_aa,&g.pair_ab,&g.pair_bb};
        const std::array<Eigen::VectorXd*,6> values{&out.coulomb,&out.exchange_alpha,&out.exchange_beta,
            &out.pair_aa,&out.pair_ab,&out.pair_bb};
        for(auto *v:values) *v=Eigen::VectorXd::Zero(nc);
        for(Eigen::Index x=0;x<nc;++x)
        {
            out.one_electron(x)=(g.relaxed_ao.total.array()*dx.hamiltonian_ao[x].array()).sum();
            out.overlap(x)=(w_total.array()*dx.overlap_ao[x].array()).sum();
            for(std::size_t k=0;k<dx.eri_ao[x].size();++k) for(int ch=0;ch<6;++ch)
                (*values[ch])(x)+=tensors[ch]->symmetric_ao[k]*dx.eri_ao[x][k];
        }
        out.total=out.one_electron+out.overlap+out.coulomb+out.exchange_alpha+out.exchange_beta+
            out.pair_aa+out.pair_ab+out.pair_bb;
        return out;
    }

    // Production assembly against dense derivative data, for comparison with
    // dense_non_xc. Only the integrals are test-side; the source, its stream
    // and build_dh_eq33_direct_non_xc_gradient are the production code.
    inline DHEq33NonXCDerivatives callbacks_from_dense(const DHEq33NonXCDerivatives &dense)
    {
        DHEq33NonXCDerivatives out;
        out.direct_one_electron_overlap=[&dense](const Eigen::Ref<const Eigen::MatrixXd> &d,
            const Eigen::Ref<const Eigen::MatrixXd> &w)
            -> std::expected<DHEq33OneElectronOverlapGradient,std::string>
        {
            const auto nc=static_cast<Eigen::Index>(dense.hamiltonian_ao.size());
            DHEq33OneElectronOverlapGradient g{Eigen::VectorXd::Zero(nc),Eigen::VectorXd::Zero(nc)};
            for(Eigen::Index x=0;x<nc;++x)
            {
                g.one_electron(x)=(d.array()*dense.hamiltonian_ao[x].array()).sum();
                g.overlap(x)=(w.array()*dense.overlap_ao[x].array()).sum();
            }
            return g;
        };
        out.direct_two_electron=[&dense](const DHEq33TwoParticleSource &source)
            -> std::expected<DHEq33TwoElectronGradient,std::string>
        {
            const auto nc=static_cast<Eigen::Index>(dense.eri_ao.size());
            const int n=source.n_ao;
            DHEq33TwoElectronGradient g{Eigen::VectorXd::Zero(nc),Eigen::VectorXd::Zero(nc)};
            for(Eigen::Index x=0;x<nc;++x)
                for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
                    g.separable(x)+=source.separable(p,q,r,s)*dense.eri_ao[x][oracle_index(p,q,r,s,n)];
            const auto streamed=for_each_dh_pair_ao_block(n,source.pair,source.occupied_tile,source.mu_tile,
                source.nu_tile,source.kappa_tile,source.tau_tile,
                [&](int m0,int mc,int n0,int ncount,int k0,int kc,int t0,int tc,const Vec &block)
                -> std::expected<void,std::string>
                {
                    for(int lm=0;lm<mc;++lm) for(int ln=0;ln<ncount;++ln) for(int lk=0;lk<kc;++lk) for(int lt=0;lt<tc;++lt)
                        for(Eigen::Index x=0;x<nc;++x)
                            g.nonseparable(x)+=block[((static_cast<std::size_t>(lm)*ncount+ln)*kc+lk)*tc+lt]*
                                dense.eri_ao[x][oracle_index(m0+lm,n0+ln,k0+lk,t0+lt,n)];
                    return {};
                },source.workspace);
            if(!streamed) return std::unexpected(streamed.error());
            return g;
        };
        return out;
    }
}
#endif
