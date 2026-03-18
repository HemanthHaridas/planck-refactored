#include "mp2_gradient.h"

#include <format>

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
    auto amp_res = build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res) return std::unexpected(amp_res.error());
    auto dens_res = build_rmp2_unrelaxed_density(calculator, shell_pairs);
    if (!dens_res) return std::unexpected(dens_res.error());
    auto z_res = solve_rmp2_zvector(calculator, shell_pairs);
    if (!z_res) return std::unexpected(z_res.error());

    const auto& amp  = *amp_res;
    const auto& dens = *dens_res;
    const auto& z    = *z_res;

    const int nb = amp.n_occ + amp.n_virt;
    Eigen::MatrixXd P_mo = Eigen::MatrixXd::Zero(nb, nb);

    // RHF reference occupations.
    P_mo.topLeftCorner(amp.n_occ, amp.n_occ) = 2.0 * Eigen::MatrixXd::Identity(amp.n_occ, amp.n_occ);

    // Add the MP2 oo/vv corrections and the ov response block.
    P_mo.topLeftCorner(amp.n_occ, amp.n_occ) += dens.P_occ;
    P_mo.bottomRightCorner(amp.n_virt, amp.n_virt) = dens.P_virt;

    // The Z-vector lives in the virtual-occupied space. In the spin-summed
    // spatial density this enters as a symmetric occupied-virtual correction.
    P_mo.bottomLeftCorner(amp.n_virt, amp.n_occ) = 2.0 * z;
    P_mo.topRightCorner(amp.n_occ, amp.n_virt) = 2.0 * z.transpose();

    Eigen::MatrixXd C(nb, nb);
    C.leftCols(amp.n_occ) = amp.C_occ;
    C.rightCols(amp.n_virt) = amp.C_virt;

    RMP2RelaxedDensity relaxed;
    relaxed.P_mo = P_mo;
    relaxed.P_ao = C * P_mo * C.transpose();
    return relaxed;
}

} // namespace HartreeFock::Correlation
