#ifndef PLANCK_TESTS_UDH_MO_SECTOR_ORACLE_H
#define PLANCK_TESTS_UDH_MO_SECTOR_ORACLE_H

// Test-only full chemist MO tensors and the Eq. 40 sectors sliced from them.
// Production builds only the sectors (build_dh_eq40_mo_eri_blocks_direct);
// tests keep full tensors as an independent literal-index oracle and slice
// them here, so the sector layout itself is under test.

#include "dft/udh_pt2_orbital.h"

#include <vector>

namespace DFT::Gradient::TestOracle
{
    // aa/bb: [p,q,r,s] in one spin; ab: [p_alpha,q_alpha,r_beta,s_beta].
    // Offset ((p*n_sigma+q)*n_tau+r)*n_tau+s.
    struct FullMOIntegrals
    {
        int n_alpha=0,n_beta=0;
        std::vector<double> aa,ab,bb;
    };

    // Electron 1 has ns orbitals (os occupied), electron 2 nt (ot occupied).
    inline DHMolecularERIBlocks sectors(const std::vector<double> &g,int ns,int nt,int os,int ot)
    {
        const int vs=ns-os,vt=nt-ot;
        const auto at=[&](int p,int q,int r,int s)
        { return g[((static_cast<std::size_t>(p)*ns+q)*nt+r)*nt+s]; };
        DHMolecularERIBlocks b;
        b.n_occ=os; b.n_virt=vs; b.n_occ_2=ot; b.n_virt_2=vt;
        for (int i=0;i<os;++i) for (int a=0;a<vs;++a) for (int j=0;j<ot;++j) for (int c=0;c<vt;++c)
            b.ovov.push_back(at(i,os+a,j,ot+c));
        for (int a=0;a<vs;++a) for (int q=0;q<vs;++q) for (int j=0;j<ot;++j) for (int c=0;c<vt;++c)
            b.vvov.push_back(at(os+a,os+q,j,ot+c));
        for (int i=0;i<os;++i) for (int q=0;q<os;++q) for (int j=0;j<ot;++j) for (int c=0;c<vt;++c)
            b.ooov.push_back(at(i,q,j,ot+c));
        for (int i=0;i<os;++i) for (int a=0;a<vs;++a) for (int q=0;q<vt;++q) for (int c=0;c<vt;++c)
            b.ovvv.push_back(at(i,os+a,ot+q,ot+c));
        for (int i=0;i<os;++i) for (int a=0;a<vs;++a) for (int j=0;j<ot;++j) for (int q=0;q<ot;++q)
            b.ovoo.push_back(at(i,os+a,j,q));
        return b;
    }

    inline UDHMOIntegrals sectors(const FullMOIntegrals &g,int nocc_alpha,int nocc_beta)
    {
        return {sectors(g.aa,g.n_alpha,g.n_alpha,nocc_alpha,nocc_alpha),
                sectors(g.ab,g.n_alpha,g.n_beta,nocc_alpha,nocc_beta),
                sectors(g.bb,g.n_beta,g.n_beta,nocc_beta,nocc_beta)};
    }

    inline UDHMOIntegrals sectors(const FullMOIntegrals &g,const UDHPT2Amplitudes &t)
    { return sectors(g,t.nocc_alpha,t.nocc_beta); }
}

#endif
