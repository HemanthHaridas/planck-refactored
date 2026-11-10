#ifndef PLANCK_TESTS_UDH_DENSE_ORACLE_H
#define PLANCK_TESTS_UDH_DENSE_ORACLE_H

// Test-only dense view of the UKS Eq. 34/35 source; the spin-agnostic
// pieces live in dh_dense_oracle.h. The tensors are rebuilt from the
// production source, so literal-index oracles still exercise the streamed
// generator. Production keeps none of these n_ao^4 objects.

#include "dft/udh_two_particle_density.h"
#include "dh_dense_oracle.h"

#include <array>
#include <stdexcept>
#include <string>

namespace DFT::Gradient::TestOracle
{
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

    inline void adapt(ERICoefficients &g,int n) { g.symmetric_ao=eightfold_symmetrize(g.raw_ao,n); }

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
    inline DenseNonXC dense_non_xc(const Tensors &g,const Eigen::MatrixXd &w_total,const DenseDerivatives &dx)
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

}
#endif
