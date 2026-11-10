#ifndef PLANCK_TESTS_DH_DENSE_ORACLE_H
#define PLANCK_TESTS_DH_DENSE_ORACLE_H

// Test-only dense views of the double-hybrid gradient objects, for either
// spin treatment. Production keeps none of these n_ao^4 objects and has no
// dense derivative path: these are rebuilt here from the production
// producers, so the literal-index oracles still exercise the streams.

#include "dft/dh_pt2_gradient.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace DFT::Gradient::TestOracle
{
    using Vec=std::vector<double>;

    inline std::size_t oracle_index(int p,int q,int r,int s,int n)
    { return ((static_cast<std::size_t>(p)*n+q)*n+r)*n+s; }

    // Dense h/S/ERI derivatives, one entry per nuclear Cartesian coordinate.
    struct DenseDerivatives
    {
        std::vector<Eigen::MatrixXd> hamiltonian_ao;
        std::vector<Eigen::MatrixXd> overlap_ao;
        std::vector<Vec> eri_ao;
    };

    inline Vec eightfold_symmetrize(const Vec &raw,int n)
    {
        const auto v=[&](int p,int q,int r,int s) { return .125*raw[oracle_index(p,q,r,s,n)]; };
        Vec out(raw.size());
        for(int p=0;p<n;++p) for(int q=0;q<n;++q) for(int r=0;r<n;++r) for(int s=0;s<n;++s)
            out[oracle_index(p,q,r,s,n)]=v(p,q,r,s)+v(q,p,r,s)+v(p,q,s,r)+v(q,p,s,r)+
                v(r,s,p,q)+v(s,r,p,q)+v(r,s,q,p)+v(s,r,q,p);
        return out;
    }

    // The full raw AO tensor of a list of pair channels, from the production stream.
    inline Vec stream_channels(const std::vector<DHPairChannel> &channels,int n,int tile)
    {
        Vec out(static_cast<std::size_t>(n)*n*n*n,0.0);
        const auto streamed=for_each_dh_pair_ao_block(n,channels,tile,tile,tile,tile,tile,
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

    inline Vec stream_channel(const DHPairChannel &channel,int n,int tile)
    { return stream_channels({channel},n,tile); }

    // The six RKS Eq. 46/47 tensors the production density no longer stores.
    struct DenseGamma
    {
        int n_ao=0;
        Vec separable_raw_ao,nonseparable_raw_ao,total_raw_ao;
        Vec separable_symmetric_ao,nonseparable_symmetric_ao,total_symmetric_ao;
    };

    // Separable part from the stored D, P and a_x; Eq. 47 from the production
    // pair stream, so a wrong stream shows up in every dense comparison.
    inline DenseGamma dense_gamma(const DHEq46_47TwoParticleDensity &g,
        const DHPT2AmplitudeDensity &amplitudes,const Eigen::MatrixXd &mo_coeff)
    {
        DenseGamma out;
        const int n=g.n_ao;
        const Eigen::MatrixXd &d=g.relaxed_difference_ao, &p=g.ground_density_ao;
        out.n_ao=n;
        out.separable_raw_ao.resize(static_cast<std::size_t>(n)*n*n*n);
        for(int mu=0;mu<n;++mu) for(int nu=0;nu<n;++nu) for(int ka=0;ka<n;++ka) for(int ta=0;ta<n;++ta)
            out.separable_raw_ao[oracle_index(mu,nu,ka,ta,n)]=
                d(mu,nu)*p(ka,ta)-0.5*g.exact_exchange*d(mu,ka)*p(nu,ta);
        out.nonseparable_raw_ao=stream_channel(make_dh_eq47_pair_channel(amplitudes,mo_coeff),n,3);
        out.total_raw_ao=out.separable_raw_ao;
        for(std::size_t k=0;k<out.total_raw_ao.size();++k) out.total_raw_ao[k]+=out.nonseparable_raw_ao[k];
        out.separable_symmetric_ao=eightfold_symmetrize(out.separable_raw_ao,n);
        out.nonseparable_symmetric_ao=eightfold_symmetrize(out.nonseparable_raw_ao,n);
        out.total_symmetric_ao=eightfold_symmetrize(out.total_raw_ao,n);
        return out;
    }

    // Eq. 33's non-XC part contracted against dense derivatives, the
    // contraction the production callbacks stand in for.
    inline DHEq33NonXCGradient dense_non_xc(const Eigen::MatrixXd &d_ao,const Eigen::MatrixXd &w_ao,
        const DenseGamma &g,const DenseDerivatives &dx)
    {
        const auto nc=static_cast<Eigen::Index>(dx.hamiltonian_ao.size());
        DHEq33NonXCGradient out;
        out.one_electron=out.overlap=out.two_electron_separable=out.two_electron_nonseparable=
            Eigen::VectorXd::Zero(nc);
        for(Eigen::Index x=0;x<nc;++x)
        {
            out.one_electron(x)=(d_ao.array()*dx.hamiltonian_ao[x].array()).sum();
            out.overlap(x)=(w_ao.array()*dx.overlap_ao[x].array()).sum();
            for(std::size_t k=0;k<dx.eri_ao[x].size();++k)
            {
                out.two_electron_separable(x)+=g.separable_symmetric_ao[k]*dx.eri_ao[x][k];
                out.two_electron_nonseparable(x)+=g.nonseparable_symmetric_ao[k]*dx.eri_ao[x][k];
            }
        }
        out.two_electron=out.two_electron_separable+out.two_electron_nonseparable;
        out.total=out.one_electron+out.overlap+out.two_electron;
        return out;
    }

    // Production callbacks over dense derivative data. Only the integrals are
    // test-side; the source, its stream and the assembly are production code.
    // The callbacks hold `dense` by reference: it must outlive them.
    inline DHEq33NonXCDerivatives callbacks_from_dense(const DenseDerivatives &dense)
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

    // Stationary products from a dense all-MO ERI tensor with materialized
    // tilde amplitudes: the route the deleted mo_eri overload took.
    inline std::expected<DHStationaryProducts,std::string>
    stationary_from_mo_eri(const HartreeFock::Correlation::RMP2Result &pt2,double c_pt2,
        const Eigen::Ref<const Eigen::MatrixXd> &mo_coeff,const Vec &mo_eri,const DHEq41ResponseOperator &response)
    {
        auto blocks=build_dh_molecular_eri_blocks(mo_eri,pt2.n_occ,pt2.n_virt);
        if(!blocks) return std::unexpected(blocks.error());
        return build_dh_stationary_products(pt2,c_pt2,mo_coeff,response,*blocks,DHTildeAmplitudeStorage::Materialized);
    }

    // The deleted one-argument contract overload: stationary products from the
    // workspace's dense MO-ERI tensor, then the production assembly.
    inline std::expected<DHGradientDriverContract,std::string>
    reference_driver_contract(const DHGradientDriverInputs &inputs)
    {
        if(!inputs.pt2_result || !inputs.workspace)
            return std::unexpected("DH gradient reference contract: missing snapshot or workspace.");
        auto stationary=stationary_from_mo_eri(*inputs.pt2_result,inputs.c_pt2,inputs.mo_coeff,
            inputs.workspace->mo_eri,inputs.response_operator);
        if(!stationary) return std::unexpected(stationary.error());
        return build_dh_gradient_driver_contract(inputs,std::move(*stationary));
    }
}
#endif
