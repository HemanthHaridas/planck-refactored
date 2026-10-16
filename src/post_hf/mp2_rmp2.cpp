#include "mp2.h"
#include "mp2_internal.h"

#include <Eigen/Eigenvalues>
#include <expected>
#include <functional>
#include <string>

namespace
{
    using HartreeFock::Correlation::detail::ChemistsERIs;
    using HartreeFock::Correlation::detail::RMP2Dims;
    using HartreeFock::Correlation::detail::idx_ovov;
    using HartreeFock::Correlation::detail::idx_t2;
    using OVOVTileProducer = std::function<std::expected<void, std::string>(
        int, const HartreeFock::Correlation::detail::RMP2OVOVPairTileConsumer &)>;
    constexpr int compact_ovov_tile_width=32;

    std::expected<void, std::string> canonical_kernel(
        const ChemistsERIs &eris,
        int n_occ,
        int n_virt,
        bool with_t2,
        bool compact_t2,
        const OVOVTileProducer *direct_ovov,
        std::vector<double> &t2,
        std::vector<double> &packed_t2,
        double &e_ss,
        double &e_os)
    {
        e_ss = 0.0;
        e_os = 0.0;
        // The compact DH path consumes the ordered OVOV tile interface.  The
        // exchange contribution needs (ib|ja), which may live in another
        // tile.  After the first pass t_ij^ab is already in pair-packed
        // storage, so (ib|ja)=t_ij^ba*Delta_ij^ab can be recovered without
        // retaining or randomly indexing OVOV.
        if (with_t2 && compact_t2)
        {
            const auto producer=[&](int width,
                                    const HartreeFock::Correlation::detail::RMP2OVOVPairTileConsumer &consumer)
                -> std::expected<void,std::string>
            {
                return direct_ovov ? (*direct_ovov)(width,consumer) :
                    HartreeFock::Correlation::detail::for_each_rmp2_ovov_pair_tile(eris,width,consumer);
            };
            auto compact=HartreeFock::Correlation::detail::contract_rmp2_ovov_tiles(
                n_occ,n_virt,eris.mo_energy,compact_ovov_tile_width,producer);
            if (!compact) return std::unexpected(compact.error());
            packed_t2=std::move(compact->t2_pair_packed);
            e_ss=compact->e_corr_ss;
            e_os=compact->e_corr_os;
            return {};
        }
        if (with_t2 && !compact_t2)
            t2.assign(static_cast<std::size_t>(n_occ) * n_occ * n_virt * n_virt, 0.0);

        // Canonical RHF MP2 can be evaluated directly from the transformed ovov
        // integrals. We keep the same-spin and opposite-spin pieces separate so
        // the teaching output can report the familiar SCS-style decomposition.
        for (int i = 0; i < n_occ; ++i)
            for (int j = 0; j < n_occ; ++j)
                for (int a = 0; a < n_virt; ++a)
                    for (int b = 0; b < n_virt; ++b)
                    {
                        const double gijab = eris.ovov[idx_ovov(i, a, j, b, n_occ, n_virt)];
                        const double gijba = eris.ovov[idx_ovov(i, b, j, a, n_occ, n_virt)];
                        const double denom =
                            eris.mo_energy(i) + eris.mo_energy(j) -
                            eris.mo_energy(n_occ + a) - eris.mo_energy(n_occ + b);
                        const double t = gijab / denom;
                        const double edi = 2.0 * t * gijab;
                        const double exi = -t * gijba;
                        e_ss += 0.5 * edi + exi;
                        e_os += 0.5 * edi;
                        if (with_t2 && !compact_t2)
                            t2[idx_t2(i,j,a,b,n_occ,n_virt)] = t;
                    }
        return {};
    }

}

namespace HartreeFock::Correlation
{
    std::expected<RMP2Result, std::string> rmp2_kernel(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::OptionsMP2 &options)
    {
        auto dims_res = detail::resolve_rmp2_dims(calculator, options);
        if (!dims_res)
            return std::unexpected(dims_res.error());
        const RMP2Dims &dims = *dims_res;

        const bool compact_direct=options.with_t2 && options.compact_t2_storage;
        auto eris_res = detail::make_eris_rmp2(calculator, shell_pairs, dims,
                                                !compact_direct);
        if (!eris_res)
            return std::unexpected(eris_res.error());
        const ChemistsERIs &eris = *eris_res;

        RMP2Result out;
        out.n_occ = dims.n_occ;
        out.n_virt = dims.n_virt;
        out.active_mo = dims.active_mo;
        out.mo_coeff = eris.mo_coeff;
        out.mo_energy = eris.mo_energy;
        out.mo_occ = Eigen::VectorXd::Zero(dims.n_mo);
        out.mo_occ.head(dims.n_occ).setConstant(2.0);

        OVOVTileProducer direct_ovov;
        if (compact_direct)
        {
            direct_ovov=[&calculator,&shell_pairs,&eris](
                int width,const detail::RMP2OVOVPairTileConsumer &consumer)
            {
                return detail::for_each_rmp2_direct_ovov_pair_tile(
                    calculator,shell_pairs,eris,width,consumer);
            };
        }
        auto canonical=canonical_kernel(eris,dims.n_occ,dims.n_virt,options.with_t2,
                                        options.compact_t2_storage,
                                        compact_direct ? &direct_ovov : nullptr,
                                        out.t2,out.t2_pair_packed,
                                        out.e_corr_ss,out.e_corr_os);
        if (!canonical) return std::unexpected(canonical.error());
        if (compact_direct)
        {
            auto source=make_rmp2_scratch_t2_source(
                out.n_occ,out.n_virt,out.t2_pair_packed);
            if(!source) return std::unexpected(source.error());
            out.t2_source=std::move(*source);
            std::vector<double>().swap(out.t2_pair_packed);
        }
        out.e_corr = out.e_corr_ss + out.e_corr_os;
        out.converged = true;
        out.n_iter = 0;
        if (!options.with_t2)
            out.t2.clear();
        return out;
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    rmp2_gamma1_intermediates(const RMP2Result &result)
    {
        if (result.t2.empty())
            return std::unexpected("rmp2_gamma1_intermediates: T2 amplitudes not stored.");

        Eigen::MatrixXd doo = Eigen::MatrixXd::Zero(result.n_occ, result.n_occ);
        Eigen::MatrixXd dvv = Eigen::MatrixXd::Zero(result.n_virt, result.n_virt);

        // These occupied-occupied and virtual-virtual blocks are the standard
        // MP2 one-particle density intermediates. The relaxed occupied block is
        // returned with a minus sign because correlation depletes the reference
        // occupations.
        for (int i = 0; i < result.n_occ; ++i)
            for (int j = 0; j < result.n_occ; ++j)
                for (int a = 0; a < result.n_virt; ++a)
                    for (int b = 0; b < result.n_virt; ++b)
                    {
                        for (int c = 0; c < result.n_virt; ++c)
                        {
                            const double t_ijca = result.t2[idx_t2(i, j, c, a, result.n_occ, result.n_virt)];
                            const double t_ijcb = result.t2[idx_t2(i, j, c, b, result.n_occ, result.n_virt)];
                            const double t_ijbc = result.t2[idx_t2(i, j, b, c, result.n_occ, result.n_virt)];
                            dvv(a, b) += 2.0 * t_ijca * t_ijcb - t_ijca * t_ijbc;
                        }
                        for (int k = 0; k < result.n_occ; ++k)
                        {
                            const double t_ijab = result.t2[idx_t2(i, j, a, b, result.n_occ, result.n_virt)];
                            const double t_ikab = result.t2[idx_t2(i, k, a, b, result.n_occ, result.n_virt)];
                            const double t_ikba = result.t2[idx_t2(i, k, b, a, result.n_occ, result.n_virt)];
                            doo(j, k) += 2.0 * t_ijab * t_ikab - t_ijab * t_ikba;
                        }
                    }

        return std::make_pair(-doo, dvv);
    }

    std::expected<Eigen::MatrixXd, std::string>
    rmp2_make_rdm1(const RMP2Result &result, bool ao_repr)
    {
        auto gamma = rmp2_gamma1_intermediates(result);
        if (!gamma)
            return std::unexpected(gamma.error());
        const auto &[doo, dvv] = *gamma;

        const int nmo = result.n_occ + result.n_virt;
        Eigen::MatrixXd dm1 = Eigen::MatrixXd::Zero(nmo, nmo);
        // Closed-shell MP2 starts from a doubly occupied RHF reference, then
        // adds the correlated occupied/virtual corrections symmetrically.
        dm1.topLeftCorner(result.n_occ, result.n_occ) =
            2.0 * Eigen::MatrixXd::Identity(result.n_occ, result.n_occ) + doo + doo.transpose();
        dm1.bottomRightCorner(result.n_virt, result.n_virt) = dvv + dvv.transpose();

        if (ao_repr)
            return Eigen::MatrixXd(result.mo_coeff * dm1 * result.mo_coeff.transpose());
        return dm1;
    }

    std::expected<std::vector<double>, std::string>
    rmp2_make_rdm2(const RMP2Result &result, bool ao_repr)
    {
        if (result.t2.empty())
            return std::unexpected("rmp2_make_rdm2: T2 amplitudes not stored.");

        const int nocc = result.n_occ;
        const int nvirt = result.n_virt;
        const int nmo = nocc + nvirt;
        const std::size_t N = static_cast<std::size_t>(nmo);
        auto idx2 = [N](int p, int q, int r, int s) -> std::size_t
        {
            return ((static_cast<std::size_t>(p) * N + q) * N + r) * N + s;
        };

        std::vector<double> dm2(N * N * N * N, 0.0);
        // The explicit ovov block carries the genuine MP2 pair-correlation
        // contribution; the later loops fold in the disconnected reference and
        // one-particle-density terms to form a full spin-summed 2-RDM.
        for (int i = 0; i < nocc; ++i)
            for (int a = 0; a < nvirt; ++a)
                for (int j = 0; j < nocc; ++j)
                    for (int b = 0; b < nvirt; ++b)
                    {
                        const double tab = result.t2[idx_t2(i, j, a, b, nocc, nvirt)];
                        const double tba = result.t2[idx_t2(i, j, b, a, nocc, nvirt)];
                        const double dovov = 2.0 * (2.0 * tab - tba);
                        dm2[idx2(i, nocc + a, j, nocc + b)] = dovov;
                        dm2[idx2(nocc + a, i, nocc + b, j)] = dovov;
                    }

        auto rdm1_res = rmp2_make_rdm1(result, false);
        if (!rdm1_res)
            return std::unexpected(rdm1_res.error());
        Eigen::MatrixXd dm1 = *rdm1_res;
        for (int i = 0; i < nocc; ++i)
            dm1(i, i) -= 2.0;

        for (int i = 0; i < nocc; ++i)
            for (int p = 0; p < nmo; ++p)
                for (int q = 0; q < nmo; ++q)
                {
                    dm2[idx2(i, i, p, q)] += 2.0 * dm1(q, p);
                    dm2[idx2(p, q, i, i)] += 2.0 * dm1(q, p);
                    dm2[idx2(p, i, i, q)] -= dm1(q, p);
                    dm2[idx2(i, p, q, i)] -= dm1(p, q);
                }

        for (int i = 0; i < nocc; ++i)
            for (int j = 0; j < nocc; ++j)
            {
                dm2[idx2(i, i, j, j)] += 4.0;
                dm2[idx2(i, j, j, i)] -= 2.0;
            }

        if (!ao_repr)
            return dm2;

        const int nao = static_cast<int>(result.mo_coeff.rows());
        std::vector<double> ao(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                for (int la = 0; la < nao; ++la)
                    for (int si = 0; si < nao; ++si)
                    {
                        double val = 0.0;
                        for (int p = 0; p < nmo; ++p)
                            for (int q = 0; q < nmo; ++q)
                                for (int r = 0; r < nmo; ++r)
                                    for (int s = 0; s < nmo; ++s)
                                        val += result.mo_coeff(mu, p) * result.mo_coeff(nu, q) *
                                               result.mo_coeff(la, r) * result.mo_coeff(si, s) *
                                               dm2[idx2(p, q, r, s)];
                        ao[idx2(mu, nu, la, si)] = val;
                    }
        return ao;
    }
} // namespace HartreeFock::Correlation
