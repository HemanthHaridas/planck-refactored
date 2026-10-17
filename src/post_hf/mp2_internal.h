#ifndef HF_POSTHF_MP2_INTERNAL_H
#define HF_POSTHF_MP2_INTERNAL_H

// Private MP2 helpers shared between mp2_rmp2.cpp and mp2_ump2.cpp.
//
// These types mirror PySCF's `_ChemistsERIs` (mp2.py / ump2.py) and the
// frozen-orbital book-keeping in `get_nocc / get_nmo / get_frozen_mask`.
// Nothing in this header is part of the public Planck API; outside of the two
// MP2 implementation files the public surface in `mp2.h` is the only contract.

#include <Eigen/Core>
#include <algorithm>
#include <cstddef>
#include <expected>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation::detail
{
    // RMP2 ChemistsERIs: matches PySCF mp2._ChemistsERIs.
    //
    // ovov[i,a,j,b] = (ia|jb) in chemist notation, stored row-major as
    //   index = ((i*nvir + a)*nocc + j)*nvir + b
    // mo_coeff is the active (frozen-removed) MO coefficient block in the AO
    // basis. fock is the active-block MO Fock matrix; for canonical references
    // it is diag(mo_energy).
    struct ChemistsERIs
    {
        Eigen::MatrixXd mo_coeff;     // nb × nmo (active MOs)
        int nocc = 0;
        Eigen::VectorXd mo_energy;    // nmo
        Eigen::MatrixXd fock;         // nmo × nmo
        std::vector<double> ovov;     // (nocc·nvir)² row-major
    };

    // Private transform-to-kernel seam for canonical RMP2.  A tile covers a
    // rectangular region of the ordered composite pair matrix p=(i,a),
    // q=(j,b), in lexicographic (p0,q0) order.  The full ordered domain is
    // emitted even when a backend can exploit pair symmetry, so consumers do
    // not inherit a backend-specific indexing convention.
    struct RMP2OVOVPairTile
    {
        int n_occ = 0, n_virt = 0;
        int p0 = 0, q0 = 0;
        int rows = 0, cols = 0;
        std::vector<double> values; // [p-p0,q-q0] row-major

        [[nodiscard]] double at(int i, int a, int j, int b) const
        {
            if (n_occ<=0 || n_virt<=0 || i<0 || a<0 || j<0 || b<0 ||
                i>=n_occ || j>=n_occ || a>=n_virt || b>=n_virt)
                return std::numeric_limits<double>::quiet_NaN();
            const int p=i*n_virt+a, q=j*n_virt+b;
            if (p<p0 || p>=p0+rows || q<q0 || q>=q0+cols ||
                values.size()!=static_cast<std::size_t>(rows)*cols)
                return std::numeric_limits<double>::quiet_NaN();
            return values[static_cast<std::size_t>(p-p0)*cols+(q-q0)];
        }
    };

    using RMP2OVOVPairTileConsumer = std::function<bool(const RMP2OVOVPairTile &)>;

    // Canonical restricted-MP2 consumer for an ordered OVOV tile source.
    // It owns only the pair-packed T2 result; callers select the dense,
    // conventional-direct, or RI-direct producer.  Keeping this seam here
    // makes the production consumer independently testable without exposing
    // any runtime test switch.
    struct RMP2CompactTileResult
    {
        std::vector<double> t2_pair_packed;
        double e_corr_ss = 0.0;
        double e_corr_os = 0.0;
    };

    [[nodiscard]] inline std::expected<RMP2CompactTileResult, std::string>
    contract_rmp2_ovov_tiles(
        int n_occ, int n_virt, const Eigen::Ref<const Eigen::VectorXd> &mo_energy,
        int pair_tile_width,
        const std::function<std::expected<void, std::string>(
            int, const RMP2OVOVPairTileConsumer &)> &producer)
    {
        const int npair=n_occ*n_virt;
        if (n_occ<=0 || n_virt<=0 || mo_energy.size()!=n_occ+n_virt ||
            pair_tile_width<=0 || !producer)
            return std::unexpected("RMP2 compact OVOV consumer: invalid dimensions or producer.");
        RMP2CompactTileResult out;
        out.t2_pair_packed.assign(static_cast<std::size_t>(npair)*(npair+1)/2,0.0);
        const auto packed_at=[&](int i,int j,int a,int b)
        {
            int p=i*n_virt+a, q=j*n_virt+b;
            if (p>q) std::swap(p,q);
            return out.t2_pair_packed[static_cast<std::size_t>(p)*npair-
                static_cast<std::size_t>(p)*(p-1)/2+(q-p)];
        };
        double direct=0.0;
        const auto consumed=producer(pair_tile_width,[&](const RMP2OVOVPairTile &tile)
        {
            if (tile.n_occ!=n_occ || tile.n_virt!=n_virt || tile.rows<=0 || tile.cols<=0 ||
                tile.p0<0 || tile.q0<0 || tile.p0+tile.rows>npair || tile.q0+tile.cols>npair ||
                tile.values.size()!=static_cast<std::size_t>(tile.rows)*tile.cols)
                return false;
            for(int ip=0;ip<tile.rows;++ip) for(int iq=0;iq<tile.cols;++iq)
            {
                const int p=tile.p0+ip, q=tile.q0+iq;
                const int i=p/n_virt, a=p%n_virt;
                const int j=q/n_virt, b=q%n_virt;
                const double denominator=mo_energy(i)+mo_energy(j)-
                    mo_energy(n_occ+a)-mo_energy(n_occ+b);
                const double integral=tile.values[static_cast<std::size_t>(ip)*tile.cols+iq];
                const double amplitude=integral/denominator;
                int upper_p=p, upper_q=q;
                if (upper_p>upper_q) std::swap(upper_p,upper_q);
                out.t2_pair_packed[static_cast<std::size_t>(upper_p)*npair-
                    static_cast<std::size_t>(upper_p)*(upper_p-1)/2+(upper_q-upper_p)]=amplitude;
                direct+=amplitude*integral;
            }
            return true;
        });
        if (!consumed) return std::unexpected(consumed.error());
        double exchange=0.0;
        for(int i=0;i<n_occ;++i) for(int j=0;j<n_occ;++j)
            for(int a=0;a<n_virt;++a) for(int b=0;b<n_virt;++b)
            {
                const double denominator=mo_energy(i)+mo_energy(j)-
                    mo_energy(n_occ+a)-mo_energy(n_occ+b);
                exchange-=packed_at(i,j,a,b)*packed_at(i,j,b,a)*denominator;
            }
        out.e_corr_os=direct;
        out.e_corr_ss=direct+exchange;
        return out;
    }

    // Dense reference producer for O7.6.4a.  Later direct conventional and
    // RI producers must obey this exact ordered-tile contract.
    [[nodiscard]] inline std::expected<void, std::string>
    for_each_rmp2_ovov_pair_tile(const ChemistsERIs &eris, int pair_tile_width,
                                 const RMP2OVOVPairTileConsumer &consumer)
    {
        const int nmo=static_cast<int>(eris.mo_coeff.cols());
        if (eris.nocc<=0 || nmo<=eris.nocc || eris.mo_energy.size()!=nmo)
            return std::unexpected("RMP2 OVOV tile stream: invalid canonical dimensions.");
        const int nvirt=nmo-eris.nocc;
        const int npair=eris.nocc*nvirt;
        if (eris.ovov.size()!=static_cast<std::size_t>(npair)*npair)
            return std::unexpected("RMP2 OVOV tile stream: invalid OVOV extent.");
        if (pair_tile_width<=0)
            return std::unexpected("RMP2 OVOV tile stream: pair tile width must be positive.");
        if (!consumer)
            return std::unexpected("RMP2 OVOV tile stream: consumer is empty.");
        for (int p0=0;p0<npair;p0+=pair_tile_width)
            for (int q0=0;q0<npair;q0+=pair_tile_width)
            {
                const int rows=std::min(pair_tile_width,npair-p0);
                const int cols=std::min(pair_tile_width,npair-q0);
                RMP2OVOVPairTile tile{eris.nocc,nvirt,p0,q0,rows,cols,
                    std::vector<double>(static_cast<std::size_t>(rows)*cols)};
                for (int ip=0;ip<rows;++ip)
                    for (int iq=0;iq<cols;++iq)
                        tile.values[static_cast<std::size_t>(ip)*cols+iq]=
                            eris.ovov[static_cast<std::size_t>(p0+ip)*npair+(q0+iq)];
                if (!consumer(tile)) return {};
            }
        return {};
    }

    // Conventional AO-ERI direct producer for the same contract.  It retains
    // the first occupied-leg quarter transform and bounded composite-pair
    // panels, but never constructs the complete (ov|ov) output.  The caller
    // still owns the AO ERI source; removing that separate cache is outside
    // this transform step.
    [[nodiscard]] inline std::expected<void, std::string>
    for_each_rmp2_ovov_conventional_tile(
        const std::vector<double> &eri, std::size_t n_ao,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt,
        int pair_tile_width, const RMP2OVOVPairTileConsumer &consumer)
    {
        const int no=static_cast<int>(c_occ.cols()), nv=static_cast<int>(c_virt.cols());
        const std::size_t n2=n_ao*n_ao, n3=n2*n_ao, n4=n3*n_ao;
        if (n_ao==0 || no<=0 || nv<=0 || c_occ.rows()!=static_cast<Eigen::Index>(n_ao) ||
            c_virt.rows()!=static_cast<Eigen::Index>(n_ao) || eri.size()!=n4)
            return std::unexpected("RMP2 OVOV conventional tile stream: invalid AO ERI or MO dimensions.");
        if (pair_tile_width<=0)
            return std::unexpected("RMP2 OVOV conventional tile stream: pair tile width must be positive.");
        if (!consumer)
            return std::unexpected("RMP2 OVOV conventional tile stream: consumer is empty.");
        const int npair=no*nv;
        // T1[i,nu,lambda,sigma] = sum_mu C_occ(mu,i) (mu nu|lambda sigma).
        std::vector<double> t1(static_cast<std::size_t>(no)*n3,0.0);
        for (int i=0;i<no;++i)
            for (std::size_t nu=0;nu<n_ao;++nu)
                for (std::size_t lambda=0;lambda<n_ao;++lambda)
                    for (std::size_t sigma=0;sigma<n_ao;++sigma)
                        for (std::size_t mu=0;mu<n_ao;++mu)
                            t1[static_cast<std::size_t>(i)*n3+nu*n2+lambda*n_ao+sigma]+=
                                c_occ(static_cast<Eigen::Index>(mu),i)*eri[mu*n3+nu*n2+lambda*n_ao+sigma];
        for (int p0=0;p0<npair;p0+=pair_tile_width)
        {
            const int rows=std::min(pair_tile_width,npair-p0);
            // T2[p,lambda,sigma] and T3[p,j,sigma] are bounded p-panels.
            std::vector<double> t2(static_cast<std::size_t>(rows)*n2,0.0);
            for (int ip=0;ip<rows;++ip)
            {
                const int p=p0+ip, i=p/nv, a=p%nv;
                for (std::size_t lambda=0;lambda<n_ao;++lambda)
                    for (std::size_t sigma=0;sigma<n_ao;++sigma)
                        for (std::size_t nu=0;nu<n_ao;++nu)
                            t2[static_cast<std::size_t>(ip)*n2+lambda*n_ao+sigma]+=
                                c_virt(static_cast<Eigen::Index>(nu),a)*
                                t1[static_cast<std::size_t>(i)*n3+nu*n2+lambda*n_ao+sigma];
            }
            std::vector<double> t3(static_cast<std::size_t>(rows)*no*n_ao,0.0);
            for (int ip=0;ip<rows;++ip)
                for (int j=0;j<no;++j)
                    for (std::size_t sigma=0;sigma<n_ao;++sigma)
                        for (std::size_t lambda=0;lambda<n_ao;++lambda)
                            t3[(static_cast<std::size_t>(ip)*no+j)*n_ao+sigma]+=
                                c_occ(static_cast<Eigen::Index>(lambda),j)*
                                t2[static_cast<std::size_t>(ip)*n2+lambda*n_ao+sigma];
            for (int q0=0;q0<npair;q0+=pair_tile_width)
            {
                const int cols=std::min(pair_tile_width,npair-q0);
                RMP2OVOVPairTile tile{no,nv,p0,q0,rows,cols,
                    std::vector<double>(static_cast<std::size_t>(rows)*cols,0.0)};
                for (int ip=0;ip<rows;++ip)
                    for (int iq=0;iq<cols;++iq)
                    {
                        const int q=q0+iq, j=q/nv, b=q%nv;
                        for (std::size_t sigma=0;sigma<n_ao;++sigma)
                            tile.values[static_cast<std::size_t>(ip)*cols+iq]+=
                                c_virt(static_cast<Eigen::Index>(sigma),b)*
                                t3[(static_cast<std::size_t>(ip)*no+j)*n_ao+sigma];
                    }
                if (!consumer(tile)) return {};
            }
        }
        return {};
    }

    // RI producer for the same contract.  b_ov is the factorized source
    // B[(ia),Q]; each callback receives only a pair-row/pair-column panel of
    // B B^T, never the complete pair-pair Gram matrix.
    [[nodiscard]] inline std::expected<void, std::string>
    for_each_rmp2_ovov_ri_tile(
        const Eigen::Ref<const Eigen::MatrixXd> &b_ov, int n_occ, int n_virt,
        int pair_tile_width, const RMP2OVOVPairTileConsumer &consumer)
    {
        const int npair=n_occ*n_virt;
        if (n_occ<=0 || n_virt<=0 || b_ov.rows()!=npair || b_ov.cols()<=0)
            return std::unexpected("RMP2 OVOV RI tile stream: invalid pair-factor dimensions.");
        if (pair_tile_width<=0)
            return std::unexpected("RMP2 OVOV RI tile stream: pair tile width must be positive.");
        if (!consumer)
            return std::unexpected("RMP2 OVOV RI tile stream: consumer is empty.");
        for (int p0=0;p0<npair;p0+=pair_tile_width)
            for (int q0=0;q0<npair;q0+=pair_tile_width)
            {
                const int rows=std::min(pair_tile_width,npair-p0);
                const int cols=std::min(pair_tile_width,npair-q0);
                RMP2OVOVPairTile tile{n_occ,n_virt,p0,q0,rows,cols,
                    std::vector<double>(static_cast<std::size_t>(rows)*cols,0.0)};
                for (int ip=0;ip<rows;++ip)
                    for (int iq=0;iq<cols;++iq)
                        for (Eigen::Index qaux=0;qaux<b_ov.cols();++qaux)
                            tile.values[static_cast<std::size_t>(ip)*cols+iq]+=
                                b_ov(p0+ip,qaux)*b_ov(q0+iq,qaux);
                if (!consumer(tile)) return {};
            }
        return {};
    }

    // UMP2 ChemistsERIs: matches PySCF ump2._ChemistsERIs.
    struct UChemistsERIs
    {
        Eigen::MatrixXd mo_coeff_a;   // nb × nmoa
        Eigen::MatrixXd mo_coeff_b;   // nb × nmob
        int nocca = 0;
        int noccb = 0;
        Eigen::VectorXd mo_energy_a;  // nmoa
        Eigen::VectorXd mo_energy_b;  // nmob
        Eigen::MatrixXd fock_a;       // nmoa × nmoa
        Eigen::MatrixXd fock_b;       // nmob × nmob
        std::vector<double> ovov;     // αα: (nocca·nvira)² row-major
        std::vector<double> OVOV;     // ββ: (noccb·nvirb)² row-major
        std::vector<double> ovOV;     // αβ: (nocca·nvira)·(noccb·nvirb) row-major
    };

    // RMP2 dimensions after applying the frozen mask. PySCF parity: frozen
    // empty → no orbitals frozen; frozen of length 1 → freeze the lowest N
    // orbitals; otherwise → explicit 0-based MO indices.
    struct RMP2Dims
    {
        int n_occ = 0;          // active occupied
        int n_virt = 0;         // active virtual
        int n_mo = 0;           // active total = n_occ + n_virt
        int n_occ_full = 0;     // total occupied including frozen
        int n_mo_full = 0;      // total MOs
        std::vector<int> active_mo;   // indices into the full MO list (size n_mo)
    };

    struct UMP2Dims
    {
        int nocca = 0;
        int noccb = 0;
        int nvira = 0;
        int nvirb = 0;
        int nmoa = 0;
        int nmob = 0;
        int nocca_full = 0;
        int noccb_full = 0;
        int nmoa_full = 0;
        int nmob_full = 0;
        std::vector<int> active_a;
        std::vector<int> active_b;
    };

    // Resolve PySCF-style frozen-orbital options into concrete active-MO masks.
    std::expected<RMP2Dims, std::string> resolve_rmp2_dims(
        const HartreeFock::Calculator &calculator,
        const HartreeFock::OptionsMP2 &options);

    std::expected<UMP2Dims, std::string> resolve_ump2_dims(
        const HartreeFock::Calculator &calculator,
        const HartreeFock::OptionsMP2 &options);

    // Build the RMP2 ChemistsERIs from the SCF reference, performing the
    // (μν|λσ) → (ia|jb) integral transformation. Mirrors PySCF mp2._make_eris.
    std::expected<ChemistsERIs, std::string> make_eris_rmp2(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Dims &dims,
        bool materialize_ovov = true);

    // Build a bounded OVOV source directly from the canonical reference in
    // `eris`.  This is deliberately separate from `make_eris_rmp2`: compact
    // RMP2 needs only the metadata carried by ChemistsERIs, whereas legacy
    // callers still require its materialized ovov reference tensor.
    [[nodiscard]] std::expected<void, std::string>
    for_each_rmp2_direct_ovov_pair_tile(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const ChemistsERIs &eris, int pair_tile_width,
        const RMP2OVOVPairTileConsumer &consumer);

    std::expected<UChemistsERIs, std::string> make_eris_ump2(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UMP2Dims &dims);

    // Flat-index helpers. Layout: [i, j, a, b] row-major when stored as t2;
    // [i, a, j, b] row-major when stored as ovov (chemist notation).
    inline std::size_t idx_t2(int i, int j, int a, int b, int n_occ, int n_virt) noexcept
    {
        return ((static_cast<std::size_t>(i) * n_occ + j) * n_virt + a) * n_virt + b;
    }
    inline std::size_t idx_ovov(int i, int a, int j, int b, int n_occ, int n_virt) noexcept
    {
        return ((static_cast<std::size_t>(i) * n_virt + a) * n_occ + j) * n_virt + b;
    }
    inline std::size_t idx_t2_ab(int i, int j, int a, int b,
                                 int nocca, int noccb, int nvira, int nvirb) noexcept
    {
        return ((static_cast<std::size_t>(i) * noccb + j) * nvira + a) * nvirb + b;
    }
    inline std::size_t idx_ovOV(int i, int a, int j, int b,
                                int nocca, int noccb, int nvira, int nvirb) noexcept
    {
        return ((static_cast<std::size_t>(i) * nvira + a) * noccb + j) * nvirb + b;
    }

} // namespace HartreeFock::Correlation::detail

#endif // HF_POSTHF_MP2_INTERNAL_H
