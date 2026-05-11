#include "mp2.h"
#include "mp2_internal.h"

#include <expected>
#include <string>

namespace
{
    using HartreeFock::Correlation::detail::UChemistsERIs;
    using HartreeFock::Correlation::detail::UMP2Dims;
    using HartreeFock::Correlation::detail::idx_ovOV;
    using HartreeFock::Correlation::detail::idx_ovov;
    using HartreeFock::Correlation::detail::idx_t2;
    using HartreeFock::Correlation::detail::idx_t2_ab;

    void canonical_kernel(
        const UChemistsERIs &eris,
        const UMP2Dims &dims,
        bool with_t2,
        std::vector<double> &t2aa,
        std::vector<double> &t2ab,
        std::vector<double> &t2bb,
        double &e_ss,
        double &e_os)
    {
        const int nocca = dims.nocca, noccb = dims.noccb;
        const int nvira = dims.nvira, nvirb = dims.nvirb;

        if (with_t2)
        {
            t2aa.assign(static_cast<std::size_t>(nocca) * nocca * nvira * nvira, 0.0);
            t2ab.assign(static_cast<std::size_t>(nocca) * noccb * nvira * nvirb, 0.0);
            t2bb.assign(static_cast<std::size_t>(noccb) * noccb * nvirb * nvirb, 0.0);
        }

        const Eigen::VectorXd &epsa = eris.mo_energy_a;
        const Eigen::VectorXd &epsb = eris.mo_energy_b;
        e_ss = 0.0;
        e_os = 0.0;

        for (int i = 0; i < nocca; ++i)
            for (int j = 0; j < nocca; ++j)
                for (int a = 0; a < nvira; ++a)
                    for (int b = 0; b < nvira; ++b)
                    {
                        const double gab = eris.ovov[idx_ovov(i, a, j, b, nocca, nvira)];
                        const double gba = eris.ovov[idx_ovov(i, b, j, a, nocca, nvira)];
                        const double denom = epsa(i) + epsa(j) - epsa(nocca + a) - epsa(nocca + b);
                        const double t = gab / denom;
                        e_ss += 0.5 * t * gab;
                        e_ss -= 0.5 * t * gba;
                        if (with_t2)
                            t2aa[idx_t2(i, j, a, b, nocca, nvira)] = (gab - gba) / denom;
                    }

        for (int i = 0; i < nocca; ++i)
            for (int j = 0; j < noccb; ++j)
                for (int a = 0; a < nvira; ++a)
                    for (int b = 0; b < nvirb; ++b)
                    {
                        const double g = eris.ovOV[idx_ovOV(i, a, j, b, nocca, noccb, nvira, nvirb)];
                        const double denom = epsa(i) + epsb(j) - epsa(nocca + a) - epsb(noccb + b);
                        const double t = g / denom;
                        e_os += t * g;
                        if (with_t2)
                            t2ab[idx_t2_ab(i, j, a, b, nocca, noccb, nvira, nvirb)] = t;
                    }

        for (int i = 0; i < noccb; ++i)
            for (int j = 0; j < noccb; ++j)
                for (int a = 0; a < nvirb; ++a)
                    for (int b = 0; b < nvirb; ++b)
                    {
                        const double gab = eris.OVOV[idx_ovov(i, a, j, b, noccb, nvirb)];
                        const double gba = eris.OVOV[idx_ovov(i, b, j, a, noccb, nvirb)];
                        const double denom = epsb(i) + epsb(j) - epsb(noccb + a) - epsb(noccb + b);
                        const double t = gab / denom;
                        e_ss += 0.5 * t * gab;
                        e_ss -= 0.5 * t * gba;
                        if (with_t2)
                            t2bb[idx_t2(i, j, a, b, noccb, nvirb)] = (gab - gba) / denom;
                    }
    }
}

namespace HartreeFock::Correlation
{
    std::expected<UMP2Result, std::string> ump2_kernel(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::OptionsMP2 &options)
    {
        auto dims_res = detail::resolve_ump2_dims(calculator, options);
        if (!dims_res)
            return std::unexpected(dims_res.error());
        const UMP2Dims &dims = *dims_res;

        auto eris_res = detail::make_eris_ump2(calculator, shell_pairs, dims);
        if (!eris_res)
            return std::unexpected(eris_res.error());
        const UChemistsERIs &eris = *eris_res;

        UMP2Result out;
        out.nocca = dims.nocca;
        out.noccb = dims.noccb;
        out.nvira = dims.nvira;
        out.nvirb = dims.nvirb;
        out.active_mo_alpha = dims.active_a;
        out.active_mo_beta = dims.active_b;
        out.mo_coeff_alpha = eris.mo_coeff_a;
        out.mo_coeff_beta = eris.mo_coeff_b;
        out.mo_energy_alpha = eris.mo_energy_a;
        out.mo_energy_beta = eris.mo_energy_b;
        out.mo_occ_alpha = Eigen::VectorXd::Zero(dims.nmoa);
        out.mo_occ_beta = Eigen::VectorXd::Zero(dims.nmob);
        out.mo_occ_alpha.head(dims.nocca).setOnes();
        out.mo_occ_beta.head(dims.noccb).setOnes();

        canonical_kernel(eris, dims, options.with_t2,
                         out.t2_aa, out.t2_ab, out.t2_bb,
                         out.e_corr_ss, out.e_corr_os);
        out.e_corr = out.e_corr_ss + out.e_corr_os;
        out.converged = true;
        out.n_iter = 0;
        if (!options.with_t2)
        {
            out.t2_aa.clear();
            out.t2_ab.clear();
            out.t2_bb.clear();
        }
        return out;
    }

    std::expected<
        std::pair<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>,
                  std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>,
        std::string>
    ump2_gamma1_intermediates(const UMP2Result &result)
    {
        if (result.t2_aa.empty() || result.t2_ab.empty() || result.t2_bb.empty())
            return std::unexpected("ump2_gamma1_intermediates: T2 amplitudes not stored.");

        Eigen::MatrixXd dooa = Eigen::MatrixXd::Zero(result.nocca, result.nocca);
        Eigen::MatrixXd doob = Eigen::MatrixXd::Zero(result.noccb, result.noccb);
        Eigen::MatrixXd dvva = Eigen::MatrixXd::Zero(result.nvira, result.nvira);
        Eigen::MatrixXd dvvb = Eigen::MatrixXd::Zero(result.nvirb, result.nvirb);

        for (int i = 0; i < result.nocca; ++i)
            for (int j = 0; j < result.nocca; ++j)
                for (int m = 0; m < result.nocca; ++m)
                    for (int e = 0; e < result.nvira; ++e)
                        for (int f = 0; f < result.nvira; ++f)
                        {
                            dooa(i, j) -= 0.5 *
                                result.t2_aa[idx_t2(i, m, e, f, result.nocca, result.nvira)] *
                                result.t2_aa[idx_t2(j, m, e, f, result.nocca, result.nvira)];
                        }
        for (int i = 0; i < result.nocca; ++i)
            for (int j = 0; j < result.nocca; ++j)
                for (int m = 0; m < result.noccb; ++m)
                    for (int e = 0; e < result.nvira; ++e)
                        for (int f = 0; f < result.nvirb; ++f)
                            dooa(i, j) -=
                                result.t2_ab[idx_t2_ab(i, m, e, f, result.nocca, result.noccb, result.nvira, result.nvirb)] *
                                result.t2_ab[idx_t2_ab(j, m, e, f, result.nocca, result.noccb, result.nvira, result.nvirb)];

        for (int i = 0; i < result.noccb; ++i)
            for (int j = 0; j < result.noccb; ++j)
                for (int m = 0; m < result.noccb; ++m)
                    for (int e = 0; e < result.nvirb; ++e)
                        for (int f = 0; f < result.nvirb; ++f)
                            doob(i, j) -= 0.5 *
                                result.t2_bb[idx_t2(i, m, e, f, result.noccb, result.nvirb)] *
                                result.t2_bb[idx_t2(j, m, e, f, result.noccb, result.nvirb)];
        for (int i = 0; i < result.noccb; ++i)
            for (int j = 0; j < result.noccb; ++j)
                for (int m = 0; m < result.nocca; ++m)
                    for (int e = 0; e < result.nvira; ++e)
                        for (int f = 0; f < result.nvirb; ++f)
                            doob(i, j) -=
                                result.t2_ab[idx_t2_ab(m, i, e, f, result.nocca, result.noccb, result.nvira, result.nvirb)] *
                                result.t2_ab[idx_t2_ab(m, j, e, f, result.nocca, result.noccb, result.nvira, result.nvirb)];

        for (int b = 0; b < result.nvira; ++b)
            for (int a = 0; a < result.nvira; ++a)
                for (int m = 0; m < result.nocca; ++m)
                    for (int n = 0; n < result.nocca; ++n)
                        for (int e = 0; e < result.nvira; ++e)
                            dvva(b, a) += 0.5 *
                                result.t2_aa[idx_t2(m, n, a, e, result.nocca, result.nvira)] *
                                result.t2_aa[idx_t2(m, n, b, e, result.nocca, result.nvira)];
        for (int b = 0; b < result.nvira; ++b)
            for (int a = 0; a < result.nvira; ++a)
                for (int m = 0; m < result.nocca; ++m)
                    for (int n = 0; n < result.noccb; ++n)
                        for (int e = 0; e < result.nvirb; ++e)
                            dvva(b, a) +=
                                result.t2_ab[idx_t2_ab(m, n, a, e, result.nocca, result.noccb, result.nvira, result.nvirb)] *
                                result.t2_ab[idx_t2_ab(m, n, b, e, result.nocca, result.noccb, result.nvira, result.nvirb)];

        for (int b = 0; b < result.nvirb; ++b)
            for (int a = 0; a < result.nvirb; ++a)
                for (int m = 0; m < result.noccb; ++m)
                    for (int n = 0; n < result.noccb; ++n)
                        for (int e = 0; e < result.nvirb; ++e)
                            dvvb(b, a) += 0.5 *
                                result.t2_bb[idx_t2(m, n, a, e, result.noccb, result.nvirb)] *
                                result.t2_bb[idx_t2(m, n, b, e, result.noccb, result.nvirb)];
        for (int b = 0; b < result.nvirb; ++b)
            for (int a = 0; a < result.nvirb; ++a)
                for (int m = 0; m < result.nocca; ++m)
                    for (int n = 0; n < result.noccb; ++n)
                        for (int e = 0; e < result.nvira; ++e)
                            dvvb(b, a) +=
                                result.t2_ab[idx_t2_ab(m, n, e, a, result.nocca, result.noccb, result.nvira, result.nvirb)] *
                                result.t2_ab[idx_t2_ab(m, n, e, b, result.nocca, result.noccb, result.nvira, result.nvirb)];

        return std::make_pair(std::make_pair(dooa, doob), std::make_pair(dvva, dvvb));
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    ump2_make_rdm1(const UMP2Result &result, bool ao_repr)
    {
        auto gamma = ump2_gamma1_intermediates(result);
        if (!gamma)
            return std::unexpected(gamma.error());
        const auto &[doo, dvv] = *gamma;
        const auto &[dooa, doob] = doo;
        const auto &[dvva, dvvb] = dvv;

        const int nmoa = result.nocca + result.nvira;
        const int nmob = result.noccb + result.nvirb;
        Eigen::MatrixXd dm1a = Eigen::MatrixXd::Zero(nmoa, nmoa);
        Eigen::MatrixXd dm1b = Eigen::MatrixXd::Zero(nmob, nmob);
        dm1a.topLeftCorner(result.nocca, result.nocca) =
            Eigen::MatrixXd::Identity(result.nocca, result.nocca) + dooa;
        dm1a.bottomRightCorner(result.nvira, result.nvira) = dvva;
        dm1b.topLeftCorner(result.noccb, result.noccb) =
            Eigen::MatrixXd::Identity(result.noccb, result.noccb) + doob;
        dm1b.bottomRightCorner(result.nvirb, result.nvirb) = dvvb;

        if (ao_repr)
            return std::make_pair(
                Eigen::MatrixXd(result.mo_coeff_alpha * dm1a * result.mo_coeff_alpha.transpose()),
                Eigen::MatrixXd(result.mo_coeff_beta * dm1b * result.mo_coeff_beta.transpose()));
        return std::make_pair(dm1a, dm1b);
    }
} // namespace HartreeFock::Correlation
