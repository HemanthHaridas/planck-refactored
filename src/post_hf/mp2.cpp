#include "mp2.h"
#include "integrals.h"
#include "mp2_gradient.h"

#include <Eigen/Eigenvalues>
#include <algorithm>

// ─── RMP2 ────────────────────────────────────────────────────────────────────

std::expected<void, std::string>
HartreeFock::Correlation::run_rmp2(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    if (calculator._info._scf.is_uhf)
        return std::unexpected("run_rmp2: RMP2 requires a converged RHF; use UMP2 for UHF.");
    if (!calculator._info._is_converged)
        return std::unexpected("run_rmp2: SCF not converged.");

    auto amp_res = HartreeFock::Correlation::build_rmp2_amplitudes(calculator, shell_pairs);
    if (!amp_res)
        return std::unexpected("run_rmp2: " + amp_res.error());
    const auto& amp = *amp_res;

    // E_MP2 = Σ_{i,j ∈ occ, a,b ∈ virt}
    //         (ia|jb) * [2*(ia|jb) - (ib|ja)] / (ε_i + ε_j - ε_a - ε_b)
    double E_mp2 = 0.0;
    for (int i = 0; i < amp.n_occ; ++i)
        for (int j = 0; j < amp.n_occ; ++j)
            for (int a = 0; a < amp.n_virt; ++a)
                for (int b = 0; b < amp.n_virt; ++b)
                {
                    const std::size_t ijab = ((static_cast<std::size_t>(i) * amp.n_virt + a)
                                            * amp.n_occ + j) * amp.n_virt + b;
                    E_mp2 += amp.iajb[ijab] * amp.tau[ijab];
                }

    calculator._correlation_energy = E_mp2;
    return {};
}

// ─── UMP2 ────────────────────────────────────────────────────────────────────

std::expected<void, std::string>
HartreeFock::Correlation::run_ump2(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    if (!calculator._info._scf.is_uhf)
        return std::unexpected("run_ump2: UMP2 requires a converged UHF; use RMP2 for RHF.");
    if (!calculator._info._is_converged)
        return std::unexpected("run_ump2: SCF not converged.");

    const std::size_t nb = calculator._shells.nbasis();

    const Eigen::MatrixXd& Ca   = calculator._info._scf.alpha.mo_coefficients;
    const Eigen::VectorXd& epsa = calculator._info._scf.alpha.mo_energies;
    const Eigen::MatrixXd& Cb   = calculator._info._scf.beta.mo_coefficients;
    const Eigen::VectorXd& epsb = calculator._info._scf.beta.mo_energies;

    // Derive alpha/beta occupations
    int n_electrons = 0;
    for (auto Z : calculator._molecule.atomic_numbers)
        n_electrons += static_cast<int>(Z);
    n_electrons -= calculator._molecule.charge;

    const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
    const std::size_t n_alpha = static_cast<std::size_t>((n_electrons + n_unpaired) / 2);
    const std::size_t n_beta  = static_cast<std::size_t>((n_electrons - n_unpaired) / 2);
    const std::size_t nva = nb - n_alpha;
    const std::size_t nvb = nb - n_beta;

    if (n_alpha == 0 || nva == 0)
        return std::unexpected("run_ump2: no alpha occupied or virtual orbitals.");
    if (n_beta == 0 || nvb == 0)
        return std::unexpected("run_ump2: no beta occupied or virtual orbitals.");

    std::vector<double> eri_local;
    const std::vector<double>& eri = HartreeFock::Correlation::ensure_eri(calculator, shell_pairs, eri_local, "UMP2 :");

    const Eigen::MatrixXd Ca_occ  = Ca.leftCols(n_alpha);
    const Eigen::MatrixXd Ca_virt = Ca.middleCols(n_alpha, nva);
    const Eigen::MatrixXd Cb_occ  = Cb.leftCols(n_beta);
    const Eigen::MatrixXd Cb_virt = Cb.middleCols(n_beta, nvb);

    // Three MO-ERI blocks
    // mo_aa[i,a,j,b]: α occupied × α virtual × α occupied × α virtual
    // mo_bb[i,a,j,b]: β occupied × β virtual × β occupied × β virtual
    // mo_ab[i,a,j,b]: α occupied × α virtual × β occupied × β virtual
    const auto mo_aa = HartreeFock::Correlation::transform_eri(eri, nb, Ca_occ, Ca_virt, Ca_occ, Ca_virt);
    const auto mo_bb = HartreeFock::Correlation::transform_eri(eri, nb, Cb_occ, Cb_virt, Cb_occ, Cb_virt);
    const auto mo_ab = HartreeFock::Correlation::transform_eri(eri, nb, Ca_occ, Ca_virt, Cb_occ, Cb_virt);

    // ── Same-spin α-α term ────────────────────────────────────────────────────
    // E_αα = (1/4) Σ_{i,j,a,b} [(ia|jb) - (ib|ja)]² / (εᵢᵅ + εⱼᵅ - εₐᵅ - εᵦᵅ)
    double E_aa = 0.0;
    for (std::size_t i = 0; i < n_alpha; ++i)
        for (std::size_t j = 0; j < n_alpha; ++j)
            for (std::size_t a = 0; a < nva; ++a)
                for (std::size_t b = 0; b < nva; ++b)
                {
                    const double iajb = mo_aa[i*nva*n_alpha*nva + a*n_alpha*nva + j*nva + b];
                    const double ibja = mo_aa[i*nva*n_alpha*nva + b*n_alpha*nva + j*nva + a];
                    const double anti = iajb - ibja;
                    const double denom = epsa(i) + epsa(j) - epsa(n_alpha + a) - epsa(n_alpha + b);
                    E_aa += 0.25 * anti * anti / denom;
                }

    // ── Same-spin β-β term ────────────────────────────────────────────────────
    // E_ββ = (1/4) Σ_{i,j,a,b} [(ia|jb) - (ib|ja)]² / (εᵢᵝ + εⱼᵝ - εₐᵝ - εᵦᵝ)
    double E_bb = 0.0;
    for (std::size_t i = 0; i < n_beta; ++i)
        for (std::size_t j = 0; j < n_beta; ++j)
            for (std::size_t a = 0; a < nvb; ++a)
                for (std::size_t b = 0; b < nvb; ++b)
                {
                    const double iajb = mo_bb[i*nvb*n_beta*nvb + a*n_beta*nvb + j*nvb + b];
                    const double ibja = mo_bb[i*nvb*n_beta*nvb + b*n_beta*nvb + j*nvb + a];
                    const double anti = iajb - ibja;
                    const double denom = epsb(i) + epsb(j) - epsb(n_beta + a) - epsb(n_beta + b);
                    E_bb += 0.25 * anti * anti / denom;
                }

    // ── Opposite-spin α-β term ────────────────────────────────────────────────
    // E_αβ = Σ_{i(α),j(β),a(α),b(β)} (ia|jb)² / (εᵢᵅ + εⱼᵝ - εₐᵅ - εᵦᵝ)
    // No exchange term for opposite spins.
    double E_ab = 0.0;
    for (std::size_t i = 0; i < n_alpha; ++i)
        for (std::size_t j = 0; j < n_beta; ++j)
            for (std::size_t a = 0; a < nva; ++a)
                for (std::size_t b = 0; b < nvb; ++b)
                {
                    const double iajb = mo_ab[i*nva*n_beta*nvb + a*n_beta*nvb + j*nvb + b];
                    const double denom = epsa(i) + epsb(j) - epsa(n_alpha + a) - epsb(n_beta + b);
                    E_ab += iajb * iajb / denom;
                }

    calculator._correlation_energy = E_aa + E_bb + E_ab;
    return {};
}

std::expected<HartreeFock::Correlation::RMP2NaturalOrbitals, std::string>
HartreeFock::Correlation::compute_rmp2_natural_orbitals(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    if (calculator._info._scf.is_uhf)
        return std::unexpected("compute_rmp2_natural_orbitals: RHF reference required.");
    if (!calculator._info._is_converged)
        return std::unexpected("compute_rmp2_natural_orbitals: SCF not converged.");

    auto dens_res = build_rmp2_unrelaxed_density(calculator, shell_pairs);
    if (!dens_res)
        return std::unexpected("compute_rmp2_natural_orbitals: " + dens_res.error());
    const auto& dens = *dens_res;

    const int n_occ  = dens.P_occ.rows();
    const int n_virt = dens.P_virt.rows();
    const int nmo    = n_occ + n_virt;

    Eigen::MatrixXd dm1_mo = Eigen::MatrixXd::Zero(nmo, nmo);
    dm1_mo.topLeftCorner(n_occ, n_occ) =
        2.0 * Eigen::MatrixXd::Identity(n_occ, n_occ) + dens.P_occ + dens.P_occ.transpose();
    dm1_mo.bottomRightCorner(n_virt, n_virt) =
        dens.P_virt + dens.P_virt.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(dm1_mo);
    if (solver.info() != Eigen::Success)
        return std::unexpected("compute_rmp2_natural_orbitals: density diagonalization failed.");

    const Eigen::VectorXd occ_asc = solver.eigenvalues();
    const Eigen::MatrixXd coeff_asc = solver.eigenvectors();

    RMP2NaturalOrbitals result;
    result.occupations = Eigen::VectorXd(nmo);
    result.coefficients_mo = Eigen::MatrixXd(nmo, nmo);

    for (int i = 0; i < nmo; ++i)
    {
        const int src = nmo - 1 - i;
        result.occupations(i) = occ_asc(src);
        result.coefficients_mo.col(i) = coeff_asc.col(src);
    }

    const Eigen::MatrixXd& C_ao_mo = calculator._info._scf.alpha.mo_coefficients;
    result.coefficients_ao = C_ao_mo * result.coefficients_mo;
    return result;
}
