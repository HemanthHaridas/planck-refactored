#include "mp2_gradient.h"

#include <format>

#include "integrals/base.h"
#include "integrals/os.h"
#include "post_hf/integrals.h"
#include "post_hf/rhf_response.h"

namespace
{
inline std::size_t idx_iajb(int i, int a, int j, int b, int n_occ, int n_virt)
{
    return ((static_cast<std::size_t>(i) * n_virt + a) * n_occ + j) * n_virt + b;
}

inline std::size_t idx_pqrs(int p, int q, int r, int s, int nq, int nr, int ns)
{
    return ((static_cast<std::size_t>(p) * nq + q) * nr + r) * ns + s;
}

inline std::size_t idx_dm2(int p, int q, int r, int s, int nmo)
{
    return ((static_cast<std::size_t>(p) * nmo + q) * nmo + r) * nmo + s;
}

Eigen::MatrixXd build_full_mo_coeff(const HartreeFock::Correlation::RMP2AmplitudeData& amp)
{
    const int nao = static_cast<int>(amp.C_occ.rows());
    const int nmo = amp.n_occ + amp.n_virt;
    Eigen::MatrixXd C(nao, nmo);
    C.leftCols(amp.n_occ) = amp.C_occ;
    C.rightCols(amp.n_virt) = amp.C_virt;
    return C;
}

Eigen::MatrixXd build_rhf_reference_density_mo(int n_occ, int nmo)
{
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(nmo, nmo);
    P.topLeftCorner(n_occ, n_occ) = 2.0 * Eigen::MatrixXd::Identity(n_occ, n_occ);
    return P;
}

Eigen::MatrixXd build_rhf_reference_weighted_density_ao(
    const HartreeFock::Correlation::RMP2AmplitudeData& amp)
{
    return 2.0 * amp.C_occ * amp.eps_occ.asDiagonal() * amp.C_occ.transpose();
}

Eigen::MatrixXd contract_imat_from_pair_density(
    const std::vector<double>& eri,
    const std::vector<double>& pair_dm2,
    int nb)
{
    Eigen::MatrixXd imat = Eigen::MatrixXd::Zero(nb, nb);
    for (int p = 0; p < nb; ++p)
    for (int q = 0; q < nb; ++q)
    {
        double val = 0.0;
        for (int i = 0; i < nb; ++i)
        for (int r = 0; r < nb; ++r)
        for (int s = 0; s < nb; ++s)
        {
            const std::size_t iprs = ((static_cast<std::size_t>(i) * nb + p) * nb + r) * nb + s;
            const std::size_t iqrs = idx_dm2(i, q, r, s, nb);
            val += eri[iprs] * pair_dm2[iqrs];
        }
        imat(p, q) = val;
    }
    return imat;
}

Eigen::MatrixXd build_veff_from_density(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const Eigen::MatrixXd& density)
{
    return _compute_2e_fock(
        shell_pairs,
        density,
        calculator._shells.nbasis(),
        calculator._integral._engine,
        calculator._integral._tol_eri,
        calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
}

}

namespace HartreeFock::Correlation
{

std::expected<RMP2AmplitudeData, std::string> build_rmp2_amplitudes(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    if (calculator._info._scf.is_uhf || calculator._scf._scf != HartreeFock::SCFType::RHF)
        return std::unexpected("build_rmp2_amplitudes: RHF reference required.");
    if (!calculator._info._is_converged)
        return std::unexpected("build_rmp2_amplitudes: SCF not converged.");

    const std::size_t nb = calculator._shells.nbasis();
    const Eigen::MatrixXd& C   = calculator._info._scf.alpha.mo_coefficients;
    const Eigen::VectorXd& eps = calculator._info._scf.alpha.mo_energies;

    int n_electrons = 0;
    for (auto Z : calculator._molecule.atomic_numbers)
        n_electrons += static_cast<int>(Z);
    n_electrons -= calculator._molecule.charge;

    if (n_electrons % 2 != 0)
        return std::unexpected("build_rmp2_amplitudes: closed-shell RHF reference required.");

    RMP2AmplitudeData data;
    data.n_occ  = n_electrons / 2;
    data.n_virt = static_cast<int>(nb) - data.n_occ;

    if (data.n_occ <= 0 || data.n_virt <= 0)
        return std::unexpected("build_rmp2_amplitudes: no occupied or virtual orbitals.");

    data.C_occ   = C.leftCols(data.n_occ);
    data.C_virt  = C.middleCols(data.n_occ, data.n_virt);
    data.eps_occ = eps.head(data.n_occ);
    data.eps_virt= eps.tail(data.n_virt);

    std::vector<double> eri_local;
    const std::vector<double>& eri = ensure_eri(
        calculator, shell_pairs, eri_local, "RMP2 Gradient :");

    data.iajb = transform_eri(eri, nb, data.C_occ, data.C_virt, data.C_occ, data.C_virt);
    data.ibja.resize(data.iajb.size());
    data.t2.resize(data.iajb.size());
    data.tau.resize(data.iajb.size());

    for (int i = 0; i < data.n_occ; ++i)
    for (int a = 0; a < data.n_virt; ++a)
    for (int j = 0; j < data.n_occ; ++j)
    for (int b = 0; b < data.n_virt; ++b)
    {
        const std::size_t ijab = idx_iajb(i, a, j, b, data.n_occ, data.n_virt);
        const std::size_t ibja = idx_iajb(i, b, j, a, data.n_occ, data.n_virt);

        data.ibja[ijab] = data.iajb[ibja];

        const double denom = data.eps_occ(i) + data.eps_occ(j)
                           - data.eps_virt(a) - data.eps_virt(b);

        data.t2[ijab]  = data.iajb[ijab] / denom;
        data.tau[ijab] = (2.0 * data.iajb[ijab] - data.ibja[ijab]) / denom;
    }

    return data;
}

std::expected<RMP2UnrelaxedDensity, std::string> build_rmp2_unrelaxed_density(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto amp_res = build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res) return std::unexpected(amp_res.error());
    const auto& amp = *amp_res;

    RMP2UnrelaxedDensity dens;
    dens.P_occ  = Eigen::MatrixXd::Zero(amp.n_occ, amp.n_occ);
    dens.P_virt = Eigen::MatrixXd::Zero(amp.n_virt, amp.n_virt);

    // Closed-shell MP2 unrelaxed density in spin-summed spatial-orbital form.
    // These blocks are the standard second-order occupancy corrections and are
    // one of the ingredients of the final relaxed MP2 density.
    for (int i = 0; i < amp.n_occ; ++i)
    for (int j = 0; j < amp.n_occ; ++j)
    {
        double val = 0.0;
        for (int m = 0; m < amp.n_occ; ++m)
        for (int a = 0; a < amp.n_virt; ++a)
        for (int b = 0; b < amp.n_virt; ++b)
        {
            const double tau_imab = amp.tau[idx_iajb(i, a, m, b, amp.n_occ, amp.n_virt)];
            const double t_jmab   = amp.t2 [idx_iajb(j, a, m, b, amp.n_occ, amp.n_virt)];
            val -= tau_imab * t_jmab;
        }
        dens.P_occ(i, j) = val;
    }

    for (int a = 0; a < amp.n_virt; ++a)
    for (int b = 0; b < amp.n_virt; ++b)
    {
        double val = 0.0;
        for (int i = 0; i < amp.n_occ; ++i)
        for (int j = 0; j < amp.n_occ; ++j)
        for (int c = 0; c < amp.n_virt; ++c)
        {
            const double tau_ijac = amp.tau[idx_iajb(i, a, j, c, amp.n_occ, amp.n_virt)];
            const double t_ijbc   = amp.t2 [idx_iajb(i, b, j, c, amp.n_occ, amp.n_virt)];
            val += tau_ijac * t_ijbc;
        }
        dens.P_virt(a, b) = val;
    }

    return dens;
}

std::expected<Eigen::MatrixXd, std::string> build_rmp2_unrelaxed_density_ao(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto amp_res = build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res) return std::unexpected(amp_res.error());
    auto dens_res = build_rmp2_unrelaxed_density(calculator, shell_pairs);
    if (!dens_res) return std::unexpected(dens_res.error());

    const auto& amp  = *amp_res;
    const auto& dens = *dens_res;

    Eigen::MatrixXd P2 =
        amp.C_occ  * dens.P_occ  * amp.C_occ.transpose() +
        amp.C_virt * dens.P_virt * amp.C_virt.transpose();
    return P2;
}

std::expected<std::vector<double>, std::string> build_rmp2_unrelaxed_2pdm_mo(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto amp_res = build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res) return std::unexpected(amp_res.error());
    const auto& amp  = *amp_res;
    const int nmo = amp.n_occ + amp.n_virt;
    const int o = amp.n_occ;

    std::vector<double> dm2(static_cast<std::size_t>(nmo) * nmo * nmo * nmo, 0.0);

    for (int i = 0; i < amp.n_occ; ++i)
    for (int j = 0; j < amp.n_occ; ++j)
    for (int a = 0; a < amp.n_virt; ++a)
    for (int b = 0; b < amp.n_virt; ++b)
    {
        const double t_ijab = amp.t2[idx_iajb(i, a, j, b, amp.n_occ, amp.n_virt)];
        const double t_ijba = amp.t2[idx_iajb(i, b, j, a, amp.n_occ, amp.n_virt)];
        const double dovov = 4.0 * t_ijab - 2.0 * t_ijba;

        dm2[idx_dm2(i, o + a, j, o + b, nmo)] += dovov;
        dm2[idx_dm2(o + a, i, o + b, j, nmo)] += dovov;
    }

    return dm2;
}

std::expected<std::vector<double>, std::string> build_rmp2_unrelaxed_2pdm_ao(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto amp_res = build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res) return std::unexpected(amp_res.error());
    auto dm2_res = build_rmp2_unrelaxed_2pdm_mo(calculator, shell_pairs);
    if (!dm2_res) return std::unexpected(dm2_res.error());

    const auto& amp = *amp_res;
    const auto& dm2_mo = *dm2_res;
    const int nmo = amp.n_occ + amp.n_virt;
    const int nao = static_cast<int>(calculator._shells.nbasis());

    Eigen::MatrixXd C(nao, nmo);
    C.leftCols(amp.n_occ) = amp.C_occ;
    C.rightCols(amp.n_virt) = amp.C_virt;

    std::vector<double> dm2_ao(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
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
        {
            val += C(mu, p) * C(nu, q) * C(la, r) * C(si, s)
                 * dm2_mo[idx_dm2(p, q, r, s, nmo)];
        }
        dm2_ao[idx_dm2(mu, nu, la, si, nao)] = val;
    }

    return dm2_ao;
}

std::expected<Eigen::MatrixXd, std::string> build_rmp2_zvector_rhs(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto amp_res = build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res) return std::unexpected(amp_res.error());
    const auto& amp = *amp_res;

    std::vector<double> eri_local;
    const std::vector<double>& eri = ensure_eri(
        calculator, shell_pairs, eri_local, "RMP2 Z-Vector :");

    const std::size_t nb = calculator._shells.nbasis();

    // These mixed MO-ERI blocks are the ingredients needed to differentiate
    // the MP2 energy with respect to occupied-virtual orbital rotations.
    const auto vvov = transform_eri(eri, nb, amp.C_virt, amp.C_virt, amp.C_occ, amp.C_virt);
    const auto ooov = transform_eri(eri, nb, amp.C_occ,  amp.C_occ,  amp.C_occ, amp.C_virt);
    const auto ovvv = transform_eri(eri, nb, amp.C_occ,  amp.C_virt, amp.C_virt, amp.C_virt);
    const auto ovoo = transform_eri(eri, nb, amp.C_occ,  amp.C_virt, amp.C_occ, amp.C_occ);

    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(amp.n_virt, amp.n_occ);

    for (int i = 0; i < amp.n_occ; ++i)
    for (int a = 0; a < amp.n_virt; ++a)
    for (int j = 0; j < amp.n_occ; ++j)
    for (int b = 0; b < amp.n_virt; ++b)
    {
        const std::size_t ijab = idx_iajb(i, a, j, b, amp.n_occ, amp.n_virt);
        const double denom = amp.eps_occ(i) + amp.eps_occ(j)
                           - amp.eps_virt(a) - amp.eps_virt(b);

        const double coeff_main = (4.0 * amp.iajb[ijab] - amp.ibja[ijab]) / denom;
        const double coeff_exch = -amp.iajb[ijab] / denom;

        for (int c = 0; c < amp.n_virt; ++c)
        {
            rhs(c, i) += coeff_main * vvov[idx_pqrs(c, a, j, b, amp.n_virt, amp.n_occ, amp.n_virt)];
            rhs(c, j) += coeff_main * ovvv[idx_pqrs(i, a, c, b, amp.n_virt, amp.n_virt, amp.n_virt)];

            rhs(c, i) += coeff_exch * vvov[idx_pqrs(c, b, j, a, amp.n_virt, amp.n_occ, amp.n_virt)];
            rhs(c, j) += coeff_exch * ovvv[idx_pqrs(i, b, c, a, amp.n_virt, amp.n_virt, amp.n_virt)];
        }

        for (int k = 0; k < amp.n_occ; ++k)
        {
            rhs(a, k) -= coeff_main * ooov[idx_pqrs(i, k, j, b, amp.n_occ, amp.n_occ, amp.n_virt)];
            rhs(b, k) -= coeff_main * ovoo[idx_pqrs(i, a, j, k, amp.n_virt, amp.n_occ, amp.n_occ)];

            rhs(b, k) -= coeff_exch * ooov[idx_pqrs(i, k, j, a, amp.n_occ, amp.n_occ, amp.n_virt)];
            rhs(a, k) -= coeff_exch * ovoo[idx_pqrs(i, b, j, k, amp.n_virt, amp.n_occ, amp.n_occ)];
        }
    }

    return rhs;
}

std::expected<Eigen::MatrixXd, std::string> solve_rmp2_zvector(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto rhs_res = build_rmp2_zvector_rhs(calculator, shell_pairs);
    if (!rhs_res) return std::unexpected(rhs_res.error());

    return solve_rhf_cphf(calculator, shell_pairs, -(*rhs_res));
}

std::expected<RMP2RelaxedDensity, std::string> build_rmp2_relaxed_density(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto grad_res = build_rmp2_gradient_intermediates(calculator, shell_pairs);
    if (!grad_res) return std::unexpected(grad_res.error());

    RMP2RelaxedDensity relaxed;
    relaxed.P_mo = std::move(grad_res->P_mo);
    relaxed.P_ao = std::move(grad_res->P_ao);
    return relaxed;
}

std::expected<RMP2GradientIntermediates, std::string> build_rmp2_gradient_intermediates(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    auto amp_res = build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res) return std::unexpected(amp_res.error());
    auto dens_res = build_rmp2_unrelaxed_density(calculator, shell_pairs);
    if (!dens_res) return std::unexpected(dens_res.error());
    auto pair_res = build_rmp2_unrelaxed_2pdm_ao(calculator, shell_pairs);
    if (!pair_res) return std::unexpected(pair_res.error());

    const auto& amp  = *amp_res;
    const auto& dens = *dens_res;
    const auto& pair_dm2_ao = *pair_res;

    const int nb = amp.n_occ + amp.n_virt;
    const Eigen::MatrixXd C = build_full_mo_coeff(amp);

    Eigen::MatrixXd dm1_corr_mo = Eigen::MatrixXd::Zero(nb, nb);
    dm1_corr_mo.topLeftCorner(amp.n_occ, amp.n_occ) = dens.P_occ + dens.P_occ.transpose();
    dm1_corr_mo.bottomRightCorner(amp.n_virt, amp.n_virt) = dens.P_virt + dens.P_virt.transpose();
    const Eigen::MatrixXd dm1_corr_ao = C * dm1_corr_mo * C.transpose();

    std::vector<double> eri_local;
    const std::vector<double>& eri = ensure_eri(
        calculator, shell_pairs, eri_local, "RMP2 Gradient :");

    const Eigen::MatrixXd veff_corr_ao =
        2.0 * build_veff_from_density(calculator, shell_pairs, dm1_corr_ao);

    Eigen::MatrixXd imat_ao = contract_imat_from_pair_density(eri, pair_dm2_ao, nb);
    Eigen::MatrixXd imat_mo = -C.transpose() * imat_ao * calculator._overlap * C;

    Eigen::MatrixXd Xvo =
        amp.C_virt.transpose() * veff_corr_ao * amp.C_occ
        + imat_mo.topRightCorner(amp.n_occ, amp.n_virt).transpose()
        - imat_mo.bottomLeftCorner(amp.n_virt, amp.n_occ);

    auto z_res = solve_rhf_cphf(calculator, shell_pairs, -Xvo);
    if (!z_res) return std::unexpected(z_res.error());
    const auto& z = *z_res;

    Eigen::MatrixXd corr_relaxed_mo = dm1_corr_mo;
    corr_relaxed_mo.bottomLeftCorner(amp.n_virt, amp.n_occ) = z;
    corr_relaxed_mo.topRightCorner(amp.n_occ, amp.n_virt) = z.transpose();

    Eigen::MatrixXd P_mo = build_rhf_reference_density_mo(amp.n_occ, nb) + corr_relaxed_mo;
    Eigen::MatrixXd P_ao = C * P_mo * C.transpose();

    Eigen::MatrixXd zeta_weights = Eigen::MatrixXd::Zero(nb, nb);
    Eigen::VectorXd mo_energies(nb);
    mo_energies << amp.eps_occ, amp.eps_virt;

    for (int p = 0; p < nb; ++p)
    for (int q = 0; q < nb; ++q)
        zeta_weights(p, q) = 0.5 * (mo_energies(p) + mo_energies(q));

    for (int a = 0; a < amp.n_virt; ++a)
    for (int i = 0; i < amp.n_occ; ++i)
    {
        zeta_weights(amp.n_occ + a, i) = amp.eps_occ(i);
        zeta_weights(i, amp.n_occ + a) = amp.eps_occ(i);
    }

    Eigen::MatrixXd zeta_ao = build_rhf_reference_weighted_density_ao(amp)
        + C * zeta_weights.cwiseProduct(corr_relaxed_mo) * C.transpose();

    imat_mo.topRightCorner(amp.n_occ, amp.n_virt) =
        imat_mo.bottomLeftCorner(amp.n_virt, amp.n_occ).transpose();
    imat_ao = C * imat_mo * C.transpose();

    const Eigen::MatrixXd occ_projector = amp.C_occ * amp.C_occ.transpose();
    const Eigen::MatrixXd dm1_corr_relaxed_ao = P_ao - calculator._info._scf.alpha.density;
    const Eigen::MatrixXd vhf_s1occ =
        occ_projector
        * build_veff_from_density(
            calculator,
            shell_pairs,
            dm1_corr_relaxed_ao + dm1_corr_relaxed_ao.transpose())
        * occ_projector;

    RMP2GradientIntermediates out;
    out.P_mo = P_mo;
    out.P_ao = P_ao;
    out.W_ao = 0.5 * (zeta_ao + zeta_ao.transpose() - imat_ao - imat_ao.transpose()) + vhf_s1occ;
    out.P_total_ao = P_ao;
    out.P_gamma_ao = calculator._info._scf.alpha.density + 2.0 * dm1_corr_relaxed_ao;
    out.im1_ao = std::move(imat_ao);
    out.zeta_ao = std::move(zeta_ao);
    out.vhf_s1occ_ao = std::move(vhf_s1occ);
    out.Gamma_pair_ao = pair_dm2_ao;
    out.pair_dm2_ao = pair_dm2_ao;
    return out;
}

} // namespace HartreeFock::Correlation
