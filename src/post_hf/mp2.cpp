#include <format>

#include "mp2.h"
#include "integrals/os.h"
#include "io/logging.h"

// ─── Shared helper ────────────────────────────────────────────────────────────

// Perform a 4-index AO→MO transformation using successive quarter transforms.
//
// Returns the (i,a,j,b) tensor — shape n_occ1 × n_virt1 × n_occ2 × n_virt2 —
// where the mapping is:
//   (ia|jb) = Σ_{μνλσ} C1_occ(μ,i) C1_virt(ν,a) eri[μνλσ] C2_occ(λ,j) C2_virt(σ,b)
//
// Indices 1 transform μ,ν and indices 2 transform λ,σ. This allows mixed-spin
// blocks (α-β) by passing different coefficient matrices for the two index pairs.
static std::vector<double> _transform_eri(
    const std::vector<double>& eri,
    std::size_t nb,
    const Eigen::MatrixXd& C1_occ,   // nbasis × n_occ1
    const Eigen::MatrixXd& C1_virt,  // nbasis × n_virt1
    const Eigen::MatrixXd& C2_occ,   // nbasis × n_occ2
    const Eigen::MatrixXd& C2_virt)  // nbasis × n_virt2
{
    const std::size_t no1 = static_cast<std::size_t>(C1_occ.cols());
    const std::size_t nv1 = static_cast<std::size_t>(C1_virt.cols());
    const std::size_t no2 = static_cast<std::size_t>(C2_occ.cols());
    const std::size_t nv2 = static_cast<std::size_t>(C2_virt.cols());

    const std::size_t nb2 = nb * nb;
    const std::size_t nb3 = nb * nb2;

    // T1[i,ν,λ,σ] = Σ_μ C1_occ(μ,i) eri[μνλσ]   shape: no1 × nb × nb × nb
    std::vector<double> T1(no1 * nb * nb * nb, 0.0);
    for (std::size_t i = 0; i < no1; ++i)
        for (std::size_t nu = 0; nu < nb; ++nu)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    for (std::size_t mu = 0; mu < nb; ++mu)
                        T1[i*nb3 + nu*nb2 + lam*nb + sig] +=
                            C1_occ(mu, i) * eri[mu*nb3 + nu*nb2 + lam*nb + sig];

    // T2[i,a,λ,σ] = Σ_ν C1_virt(ν,a) T1[i,ν,λ,σ]   shape: no1 × nv1 × nb × nb
    std::vector<double> T2(no1 * nv1 * nb * nb, 0.0);
    for (std::size_t i = 0; i < no1; ++i)
        for (std::size_t a = 0; a < nv1; ++a)
            for (std::size_t lam = 0; lam < nb; ++lam)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    for (std::size_t nu = 0; nu < nb; ++nu)
                        T2[i*nv1*nb*nb + a*nb*nb + lam*nb + sig] +=
                            C1_virt(nu, a) * T1[i*nb3 + nu*nb2 + lam*nb + sig];

    T1.clear();
    T1.shrink_to_fit();

    // T3[i,a,j,σ] = Σ_λ C2_occ(λ,j) T2[i,a,λ,σ]   shape: no1 × nv1 × no2 × nb
    std::vector<double> T3(no1 * nv1 * no2 * nb, 0.0);
    for (std::size_t i = 0; i < no1; ++i)
        for (std::size_t a = 0; a < nv1; ++a)
            for (std::size_t j = 0; j < no2; ++j)
                for (std::size_t sig = 0; sig < nb; ++sig)
                    for (std::size_t lam = 0; lam < nb; ++lam)
                        T3[i*nv1*no2*nb + a*no2*nb + j*nb + sig] +=
                            C2_occ(lam, j) * T2[i*nv1*nb*nb + a*nb*nb + lam*nb + sig];

    T2.clear();
    T2.shrink_to_fit();

    // T4[i,a,j,b] = Σ_σ C2_virt(σ,b) T3[i,a,j,σ]   shape: no1 × nv1 × no2 × nv2
    std::vector<double> mo_eri(no1 * nv1 * no2 * nv2, 0.0);
    for (std::size_t i = 0; i < no1; ++i)
        for (std::size_t a = 0; a < nv1; ++a)
            for (std::size_t j = 0; j < no2; ++j)
                for (std::size_t b = 0; b < nv2; ++b)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        mo_eri[i*nv1*no2*nv2 + a*no2*nv2 + j*nv2 + b] +=
                            C2_virt(sig, b) * T3[i*nv1*no2*nb + a*no2*nb + j*nb + sig];

    return mo_eri;
}

// ─── Helper: ensure AO ERI is available ──────────────────────────────────────

static const std::vector<double>& _ensure_eri(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    std::vector<double>& eri_local,
    const std::string& tag)
{
    if (!calculator._eri.empty())
        return calculator._eri;

    const std::size_t nb = calculator._shells.nbasis();
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, tag,
        std::format("Building AO ERI tensor ({:.1f} MB)", nb * nb * nb * nb * 8.0 / 1e6));
    eri_local = HartreeFock::ObaraSaika::_compute_2e(shell_pairs, nb);
    return eri_local;
}

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

    const std::size_t nb  = calculator._shells.nbasis();
    const Eigen::MatrixXd& C   = calculator._info._scf.alpha.mo_coefficients;
    const Eigen::VectorXd& eps = calculator._info._scf.alpha.mo_energies;

    int n_electrons = 0;
    for (auto Z : calculator._molecule.atomic_numbers)
        n_electrons += static_cast<int>(Z);
    n_electrons -= calculator._molecule.charge;

    if (n_electrons % 2 != 0)
        return std::unexpected("run_rmp2: odd electron count; RMP2 requires closed-shell RHF.");

    const std::size_t n_occ  = static_cast<std::size_t>(n_electrons / 2);
    const std::size_t n_virt = nb - n_occ;

    if (n_occ == 0 || n_virt == 0)
        return std::unexpected("run_rmp2: no occupied or virtual orbitals.");

    std::vector<double> eri_local;
    const std::vector<double>& eri = _ensure_eri(calculator, shell_pairs, eri_local, "RMP2 :");

    const Eigen::MatrixXd C_occ  = C.leftCols(n_occ);
    const Eigen::MatrixXd C_virt = C.middleCols(n_occ, n_virt);

    const auto mo_eri = _transform_eri(eri, nb, C_occ, C_virt, C_occ, C_virt);

    // E_MP2 = Σ_{i,j ∈ occ, a,b ∈ virt}
    //         (ia|jb) * [2*(ia|jb) - (ib|ja)] / (ε_i + ε_j - ε_a - ε_b)
    double E_mp2 = 0.0;
    for (std::size_t i = 0; i < n_occ; ++i)
        for (std::size_t j = 0; j < n_occ; ++j)
            for (std::size_t a = 0; a < n_virt; ++a)
                for (std::size_t b = 0; b < n_virt; ++b)
                {
                    const double iajb = mo_eri[i*n_virt*n_occ*n_virt + a*n_occ*n_virt + j*n_virt + b];
                    const double ibja = mo_eri[i*n_virt*n_occ*n_virt + b*n_occ*n_virt + j*n_virt + a];
                    const double denom = eps(i) + eps(j) - eps(n_occ + a) - eps(n_occ + b);
                    E_mp2 += iajb * (2.0 * iajb - ibja) / denom;
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
    const std::vector<double>& eri = _ensure_eri(calculator, shell_pairs, eri_local, "UMP2 :");

    const Eigen::MatrixXd Ca_occ  = Ca.leftCols(n_alpha);
    const Eigen::MatrixXd Ca_virt = Ca.middleCols(n_alpha, nva);
    const Eigen::MatrixXd Cb_occ  = Cb.leftCols(n_beta);
    const Eigen::MatrixXd Cb_virt = Cb.middleCols(n_beta, nvb);

    // Three MO-ERI blocks
    // mo_aa[i,a,j,b]: α occupied × α virtual × α occupied × α virtual
    // mo_bb[i,a,j,b]: β occupied × β virtual × β occupied × β virtual
    // mo_ab[i,a,j,b]: α occupied × α virtual × β occupied × β virtual
    const auto mo_aa = _transform_eri(eri, nb, Ca_occ, Ca_virt, Ca_occ, Ca_virt);
    const auto mo_bb = _transform_eri(eri, nb, Cb_occ, Cb_virt, Cb_occ, Cb_virt);
    const auto mo_ab = _transform_eri(eri, nb, Ca_occ, Ca_virt, Cb_occ, Cb_virt);

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
