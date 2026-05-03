#include "gradient.h"

#include <array>
#include <cmath>
#include <format>
#include <vector>

#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"
#include "post_hf/mp2.h"
#include "post_hf/mp2_gradient.h"
#include "scf/scf.h"
#include "symmetry/integral_symmetry.h"

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Build a map: shell index in _shells._shells → atom index in _molecule.
// Matches shell._center ≈ _molecule._standard.row(a) within 1e-6 Bohr.
static std::expected<std::vector<int>, std::string> build_shell_atom_map(
    const HartreeFock::Calculator &calc)
{
    const auto &shells = calc._shells._shells;
    const auto &mol = calc._molecule;
    const std::size_t nshells = shells.size();

    std::vector<int> map(nshells, -1);
    for (std::size_t s = 0; s < nshells; ++s)
    {
        const Eigen::Vector3d &sc = shells[s]._center;
        for (std::size_t a = 0; a < mol.natoms; ++a)
        {
            const double dx = sc[0] - mol._standard(a, 0);
            const double dy = sc[1] - mol._standard(a, 1);
            const double dz = sc[2] - mol._standard(a, 2);
            if (dx * dx + dy * dy + dz * dz < 1e-10)
            { // 1e-5 Bohr tolerance squared
                map[s] = static_cast<int>(a);
                break;
            }
        }
        if (map[s] < 0)
            return std::unexpected(std::string("Gradient: shell does not match any atom"));
    }
    return map;
}

static std::size_t idx_dm2_grad(int p, int q, int r, int s, int nbf)
{
    return ((static_cast<std::size_t>(p) * nbf + q) * nbf + r) * nbf + s;
}

static Eigen::MatrixXd build_pair_schwarz_table(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    std::size_t nbasis)
{
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nbasis, nbasis);
    for (const auto &sp : shell_pairs)
    {
        const std::size_t i = sp.A._index;
        const std::size_t j = sp.B._index;
        const int lAx = sp.A._cartesian[0], lAy = sp.A._cartesian[1], lAz = sp.A._cartesian[2];
        const int lBx = sp.B._cartesian[0], lBy = sp.B._cartesian[1], lBz = sp.B._cartesian[2];
        const double diag = HartreeFock::ObaraSaika::_contracted_eri_elem(
            sp, sp, lAx, lAy, lAz, lBx, lBy, lBz, lAx, lAy, lAz, lBx, lBy, lBz);
        const double q = std::sqrt(std::abs(diag));
        Q(i, j) = q;
        Q(j, i) = q;
    }
    return Q;
}

template <typename GammaFn>
static void accumulate_eri_gradient_permutations(
    Eigen::MatrixXd &grad,
    const std::array<double, 12> &dI,
    GammaFn &&gamma_fn,
    std::size_t ii,
    std::size_t jj,
    std::size_t kk,
    std::size_t ll,
    int atom_A,
    int atom_B,
    int atom_C,
    int atom_D)
{
    const auto accumulate_perm = [&](double gamma,
                                     bool swap_ab,
                                     bool swap_cd)
    {
        if (std::abs(gamma) < 1e-14)
            return;

        const int deriv_a = swap_ab ? 1 : 0;
        const int deriv_b = swap_ab ? 0 : 1;
        const int deriv_c = swap_cd ? 3 : 2;
        const int deriv_d = swap_cd ? 2 : 3;

        const int atom_a = swap_ab ? atom_B : atom_A;
        const int atom_b = swap_ab ? atom_A : atom_B;
        const int atom_c = swap_cd ? atom_D : atom_C;
        const int atom_d = swap_cd ? atom_C : atom_D;

        const double fac = 0.25 * gamma;
        for (int q = 0; q < 3; ++q)
        {
            grad(atom_a, q) += fac * dI[deriv_a * 3 + q];
            grad(atom_b, q) += fac * dI[deriv_b * 3 + q];
            grad(atom_c, q) += fac * dI[deriv_c * 3 + q];
            grad(atom_d, q) += fac * dI[deriv_d * 3 + q];
        }
    };

    accumulate_perm(gamma_fn(ii, jj, kk, ll), false, false);
    if (kk != ll)
        accumulate_perm(gamma_fn(ii, jj, ll, kk), false, true);
    if (ii != jj)
        accumulate_perm(gamma_fn(jj, ii, kk, ll), true, false);
    if (ii != jj && kk != ll)
        accumulate_perm(gamma_fn(jj, ii, ll, kk), true, true);
}

template <typename GammaFn>
static std::expected<Eigen::MatrixXd, std::string> compute_closed_shell_gradient_from_density(
    const HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::MatrixXd &P,
    const Eigen::MatrixXd &W,
    GammaFn &&gamma_fn)
{
    const auto &mol = calc._molecule;
    const auto &basis = calc._shells;
    const std::size_t natoms = mol.natoms;
    const std::size_t nb = basis.nbasis();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natoms, 3);

    auto shell_atom_res = build_shell_atom_map(calc);
    if (!shell_atom_res)
        return std::unexpected(shell_atom_res.error());
    const std::vector<int> shell_atom = std::move(*shell_atom_res);
    const auto &shells = basis._shells;
    const auto &bfs = basis._basis_functions;
    const std::size_t nshells = shells.size();
    const Eigen::MatrixXd schwarz_q = build_pair_schwarz_table(shell_pairs, nb);

    std::vector<int> bf_shell(nb, -1);
    for (std::size_t s = 0; s < nshells; ++s)
    {
        for (std::size_t mu = 0; mu < nb; ++mu)
            if (bfs[mu]._shell == &shells[s])
                bf_shell[mu] = static_cast<int>(s);
    }

    for (const auto &sp : shell_pairs)
    {
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;
        const int atom_ii = shell_atom[bf_shell[ii]];
        const int atom_jj = shell_atom[bf_shell[jj]];

        const auto dST_A = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
        const auto dV_A = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp, mol);

        for (int q = 0; q < 3; ++q)
        {
            const double contrib = 2.0 * P(ii, jj) * (dST_A[q + 3] + dV_A[q]) - 2.0 * W(ii, jj) * dST_A[q];
            grad(atom_ii, q) += contrib;
        }

        if (ii != jj)
        {
            HartreeFock::ShellPair sp_rev(sp.B, sp.A);
            const auto dST_B = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp_rev);
            const auto dV_B = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp_rev, mol);

            for (int q = 0; q < 3; ++q)
            {
                const double contrib = 2.0 * P(jj, ii) * (dST_B[q + 3] + dV_B[q]) - 2.0 * W(jj, ii) * dST_B[q];
                grad(atom_jj, q) += contrib;
            }
        }
    }

    for (std::size_t atom_a = 0; atom_a < natoms; ++atom_a)
    {
        const double Z_A = static_cast<double>(mol.atomic_numbers[atom_a]);
        const Eigen::Vector3d C_A(mol._standard(atom_a, 0),
                                  mol._standard(atom_a, 1),
                                  mol._standard(atom_a, 2));

        for (int q = 0; q < 3; ++q)
        {
            double dV_sum = 0.0;
            for (const auto &sp : shell_pairs)
            {
                const std::size_t ii = sp.A._index;
                const std::size_t jj = sp.B._index;
                const double dv = HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(
                    sp, C_A, Z_A, q);
                if (ii == jj)
                    dV_sum += P(ii, jj) * dv;
                else
                    dV_sum += 2.0 * P(ii, jj) * dv;
            }
            grad(atom_a, q) += dV_sum;
        }
    }

    for (const auto &spAB : shell_pairs)
    {
        const std::size_t ii = spAB.A._index;
        const std::size_t jj = spAB.B._index;
        const int atom_A = shell_atom[bf_shell[ii]];
        const int atom_B = shell_atom[bf_shell[jj]];

        for (const auto &spCD : shell_pairs)
        {
            const std::size_t kk = spCD.A._index;
            const std::size_t ll = spCD.B._index;
            const int atom_C = shell_atom[bf_shell[kk]];
            const int atom_D = shell_atom[bf_shell[ll]];

            if (schwarz_q(ii, jj) * schwarz_q(kk, ll) < calc._integral._tol_eri)
                continue;

            const auto dI = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(spAB, spCD);
            accumulate_eri_gradient_permutations(
                grad,
                dI,
                gamma_fn,
                ii,
                jj,
                kk,
                ll,
                atom_A,
                atom_B,
                atom_C,
                atom_D);
        }
    }

    for (std::size_t a = 0; a < natoms; ++a)
    {
        for (std::size_t b = 0; b < natoms; ++b)
        {
            if (a == b)
                continue;
            const double Za = static_cast<double>(mol.atomic_numbers[a]);
            const double Zb = static_cast<double>(mol.atomic_numbers[b]);
            const double dx = mol._standard(a, 0) - mol._standard(b, 0);
            const double dy = mol._standard(a, 1) - mol._standard(b, 1);
            const double dz = mol._standard(a, 2) - mol._standard(b, 2);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double r3 = r * r * r;
            const double fac = Za * Zb / r3;
            grad(a, 0) -= fac * dx;
            grad(a, 1) -= fac * dy;
            grad(a, 2) -= fac * dz;
        }
    }

    return grad;
}

// ─── RHF Gradient ─────────────────────────────────────────────────────────────
//
// g[A,x] = 2 Σ_{μ∈A,ν} P_μν (dT_μν/dA_x + dV_μν/dA_x)  [1e GTO-centre]
//        + Σ_{μν}       P_μν dV_μν^{C=A}/dR_{A,x}         [nucleus-position V]
//        + ½ Σ_{μνλσ}   Γ_μνλσ d(μν|λσ)/dA_x              [2e ERI]
//        - 2 Σ_{μ∈A,ν}  W_μν dS_μν/dA_x                   [Pulay]
//        + Σ_{B≠A}      Z_A Z_B (R_A-R_B)/|R_A-R_B|³      [nuclear repulsion]

std::expected<Eigen::MatrixXd, std::string> HartreeFock::Gradient::compute_rhf_gradient(
    const HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    const Eigen::MatrixXd &P = calc._info._scf.alpha.density; // already has factor 2
    int n_elec = 0;
    for (std::size_t a = 0; a < calc._molecule.natoms; ++a)
        n_elec += calc._molecule.atomic_numbers[a];
    n_elec -= calc._molecule.charge;
    const int n_occ = n_elec / 2;

    const Eigen::MatrixXd C_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_occ);
    const Eigen::VectorXd eps = calc._info._scf.alpha.mo_energies.head(n_occ);
    const Eigen::MatrixXd W = 2.0 * C_occ * eps.asDiagonal() * C_occ.transpose();
    auto gamma_fn = [&P](std::size_t ii, std::size_t jj, std::size_t kk, std::size_t ll) -> double
    {
        return 2.0 * P(ii, jj) * P(kk, ll) - P(ii, kk) * P(jj, ll);
    };
    return compute_closed_shell_gradient_from_density(calc, shell_pairs, P, W, gamma_fn);
}

std::expected<Eigen::MatrixXd, std::string> HartreeFock::Gradient::compute_rks_gradient(
    const HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    double exact_exchange_coefficient)
{
    const Eigen::MatrixXd &P = calc._info._scf.alpha.density;
    int n_elec = 0;
    for (std::size_t a = 0; a < calc._molecule.natoms; ++a)
        n_elec += calc._molecule.atomic_numbers[a];
    n_elec -= calc._molecule.charge;
    const int n_occ = n_elec / 2;

    const Eigen::MatrixXd C_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_occ);
    const Eigen::VectorXd eps = calc._info._scf.alpha.mo_energies.head(n_occ);
    const Eigen::MatrixXd W = 2.0 * C_occ * eps.asDiagonal() * C_occ.transpose();
    const double cx = exact_exchange_coefficient;
    auto gamma_fn = [&P, cx](std::size_t ii, std::size_t jj, std::size_t kk, std::size_t ll) -> double
    {
        return 2.0 * P(ii, jj) * P(kk, ll) - cx * P(ii, kk) * P(jj, ll);
    };
    return compute_closed_shell_gradient_from_density(calc, shell_pairs, P, W, gamma_fn);
}

// ─── UHF Gradient ─────────────────────────────────────────────────────────────

std::expected<Eigen::MatrixXd, std::string> HartreeFock::Gradient::compute_uhf_gradient(
    const HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    const auto &mol = calc._molecule;
    const auto &basis = calc._shells;
    const std::size_t natoms = mol.natoms;
    const std::size_t nb = basis.nbasis();

    // UHF densities (already without factor 2)
    const Eigen::MatrixXd &P_a = calc._info._scf.alpha.density;
    const Eigen::MatrixXd &P_b = calc._info._scf.beta.density;
    const Eigen::MatrixXd P_t = P_a + P_b; // total density

    // Electron counts
    int n_elec = 0;
    for (std::size_t a = 0; a < natoms; ++a)
        n_elec += mol.atomic_numbers[a];
    n_elec -= mol.charge;
    const int n_unpaired = static_cast<int>(mol.multiplicity) - 1;
    const int n_alpha = (n_elec + n_unpaired) / 2;
    const int n_beta = (n_elec - n_unpaired) / 2;

    const Eigen::MatrixXd Ca_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_alpha);
    const Eigen::VectorXd ea = calc._info._scf.alpha.mo_energies.head(n_alpha);
    const Eigen::MatrixXd Cb_occ = calc._info._scf.beta.mo_coefficients.leftCols(n_beta);
    const Eigen::VectorXd eb = calc._info._scf.beta.mo_energies.head(n_beta);

    // Energy-weighted density (no factor 2 for UHF)
    const Eigen::MatrixXd W = Ca_occ * ea.asDiagonal() * Ca_occ.transpose() + Cb_occ * eb.asDiagonal() * Cb_occ.transpose();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natoms, 3);

    auto shell_atom_res = build_shell_atom_map(calc);
    if (!shell_atom_res)
        return std::unexpected(shell_atom_res.error());
    const std::vector<int> shell_atom = std::move(*shell_atom_res);
    const auto &shells = basis._shells;
    const auto &bfs = basis._basis_functions;
    const std::size_t nshells = shells.size();
    const Eigen::MatrixXd schwarz_q = build_pair_schwarz_table(shell_pairs, nb);

    std::vector<int> bf_shell(nb, -1);
    for (std::size_t s = 0; s < nshells; ++s)
        for (std::size_t mu = 0; mu < nb; ++mu)
            if (bfs[mu]._shell == &shells[s])
                bf_shell[mu] = static_cast<int>(s);

    // ── Term 1+Pulay (same structure as RHF but using P_t and W) ─────────────
    for (const auto &sp : shell_pairs)
    {
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;
        const int atom_ii = shell_atom[bf_shell[ii]];
        const int atom_jj = shell_atom[bf_shell[jj]];

        const auto dST_A = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
        const auto dV_A = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp, mol);

        for (int q = 0; q < 3; ++q)
        {
            const double contrib = 2.0 * P_t(ii, jj) * (dST_A[q + 3] + dV_A[q]) - 2.0 * W(ii, jj) * dST_A[q];
            grad(atom_ii, q) += contrib;
        }

        if (ii != jj)
        {
            HartreeFock::ShellPair sp_rev(sp.B, sp.A);
            const auto dST_B = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp_rev);
            const auto dV_B = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp_rev, mol);

            for (int q = 0; q < 3; ++q)
            {
                const double contrib = 2.0 * P_t(jj, ii) * (dST_B[q + 3] + dV_B[q]) - 2.0 * W(jj, ii) * dST_B[q];
                grad(atom_jj, q) += contrib;
            }
        }
    }

    // ── Term 2: nucleus-position V ────────────────────────────────────────────
    for (std::size_t atom_a = 0; atom_a < natoms; ++atom_a)
    {
        const double Z_A = static_cast<double>(mol.atomic_numbers[atom_a]);
        const Eigen::Vector3d C_A(mol._standard(atom_a, 0),
                                  mol._standard(atom_a, 1),
                                  mol._standard(atom_a, 2));

        for (int q = 0; q < 3; ++q)
        {
            double dV_sum = 0.0;
            for (const auto &sp : shell_pairs)
            {
                const std::size_t ii = sp.A._index;
                const std::size_t jj = sp.B._index;
                const double dv = HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(
                    sp, C_A, Z_A, q);
                if (ii == jj)
                    dV_sum += P_t(ii, jj) * dv;
                else
                    dV_sum += 2.0 * P_t(ii, jj) * dv;
            }
            grad(atom_a, q) += dV_sum;
        }
    }

    // ── Term 3: ERI gradient ──────────────────────────────────────────────────
    // Γ_μνλσ = 2*P_t_μν*P_t_λσ - 2*P_a_μλ*P_a_νσ - 2*P_b_μλ*P_b_νσ
    auto gamma_fn = [&P_t, &P_a, &P_b](std::size_t ii, std::size_t jj,
                                       std::size_t kk, std::size_t ll) -> double
    {
        return 2.0 * P_t(ii, jj) * P_t(kk, ll) -
               2.0 * P_a(ii, kk) * P_a(jj, ll) -
               2.0 * P_b(ii, kk) * P_b(jj, ll);
    };

    for (const auto &spAB : shell_pairs)
    {
        const std::size_t ii = spAB.A._index;
        const std::size_t jj = spAB.B._index;
        const int atom_A = shell_atom[bf_shell[ii]];
        const int atom_B = shell_atom[bf_shell[jj]];

        for (const auto &spCD : shell_pairs)
        {
            const std::size_t kk = spCD.A._index;
            const std::size_t ll = spCD.B._index;
            const int atom_C = shell_atom[bf_shell[kk]];
            const int atom_D = shell_atom[bf_shell[ll]];

            if (schwarz_q(ii, jj) * schwarz_q(kk, ll) < calc._integral._tol_eri)
                continue;

            const auto dI = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(spAB, spCD);
            accumulate_eri_gradient_permutations(
                grad,
                dI,
                gamma_fn,
                ii,
                jj,
                kk,
                ll,
                atom_A,
                atom_B,
                atom_C,
                atom_D);
        }
    }

    // ── Term 4: nuclear repulsion ─────────────────────────────────────────────
    for (std::size_t a = 0; a < natoms; ++a)
    {
        for (std::size_t b = 0; b < natoms; ++b)
        {
            if (a == b)
                continue;
            const double Za = static_cast<double>(mol.atomic_numbers[a]);
            const double Zb = static_cast<double>(mol.atomic_numbers[b]);
            const double dx = mol._standard(a, 0) - mol._standard(b, 0);
            const double dy = mol._standard(a, 1) - mol._standard(b, 1);
            const double dz = mol._standard(a, 2) - mol._standard(b, 2);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-24)
            {
                return std::unexpected(
                    std::format("Gradient: atoms {} and {} are coincident or too close for nuclear-repulsion differentiation",
                                static_cast<int>(a + 1),
                                static_cast<int>(b + 1)));
            }
            const double r3 = std::pow(r2, 1.5);
            grad(a, 0) -= Za * Zb * dx / r3;
            grad(a, 1) -= Za * Zb * dy / r3;
            grad(a, 2) -= Za * Zb * dz / r3;
        }
    }

    return grad;
}

std::expected<Eigen::MatrixXd, std::string> HartreeFock::Gradient::compute_uks_gradient(
    const HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    double exact_exchange_coefficient)
{
    const auto &mol = calc._molecule;
    const auto &basis = calc._shells;
    const std::size_t natoms = mol.natoms;
    const std::size_t nb = basis.nbasis();

    const Eigen::MatrixXd &P_a = calc._info._scf.alpha.density;
    const Eigen::MatrixXd &P_b = calc._info._scf.beta.density;
    const Eigen::MatrixXd P_t = P_a + P_b;

    int n_elec = 0;
    for (std::size_t a = 0; a < natoms; ++a)
        n_elec += mol.atomic_numbers[a];
    n_elec -= mol.charge;
    const int n_unpaired = static_cast<int>(mol.multiplicity) - 1;
    const int n_alpha = (n_elec + n_unpaired) / 2;
    const int n_beta = (n_elec - n_unpaired) / 2;

    const Eigen::MatrixXd Ca_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_alpha);
    const Eigen::VectorXd ea = calc._info._scf.alpha.mo_energies.head(n_alpha);
    const Eigen::MatrixXd Cb_occ = calc._info._scf.beta.mo_coefficients.leftCols(n_beta);
    const Eigen::VectorXd eb = calc._info._scf.beta.mo_energies.head(n_beta);

    const Eigen::MatrixXd W = Ca_occ * ea.asDiagonal() * Ca_occ.transpose() + Cb_occ * eb.asDiagonal() * Cb_occ.transpose();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natoms, 3);

    auto shell_atom_res = build_shell_atom_map(calc);
    if (!shell_atom_res)
        return std::unexpected(shell_atom_res.error());
    const std::vector<int> shell_atom = std::move(*shell_atom_res);
    const auto &shells = basis._shells;
    const auto &bfs = basis._basis_functions;
    const std::size_t nshells = shells.size();
    const Eigen::MatrixXd schwarz_q = build_pair_schwarz_table(shell_pairs, nb);

    std::vector<int> bf_shell(nb, -1);
    for (std::size_t s = 0; s < nshells; ++s)
        for (std::size_t mu = 0; mu < nb; ++mu)
            if (bfs[mu]._shell == &shells[s])
                bf_shell[mu] = static_cast<int>(s);

    for (const auto &sp : shell_pairs)
    {
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;
        const int atom_ii = shell_atom[bf_shell[ii]];
        const int atom_jj = shell_atom[bf_shell[jj]];

        const auto dST_A = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
        const auto dV_A = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp, mol);

        for (int q = 0; q < 3; ++q)
        {
            const double contrib = 2.0 * P_t(ii, jj) * (dST_A[q + 3] + dV_A[q]) - 2.0 * W(ii, jj) * dST_A[q];
            grad(atom_ii, q) += contrib;
        }

        if (ii != jj)
        {
            HartreeFock::ShellPair sp_rev(sp.B, sp.A);
            const auto dST_B = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp_rev);
            const auto dV_B = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp_rev, mol);

            for (int q = 0; q < 3; ++q)
            {
                const double contrib = 2.0 * P_t(jj, ii) * (dST_B[q + 3] + dV_B[q]) - 2.0 * W(jj, ii) * dST_B[q];
                grad(atom_jj, q) += contrib;
            }
        }
    }

    for (std::size_t atom_a = 0; atom_a < natoms; ++atom_a)
    {
        const double Z_A = static_cast<double>(mol.atomic_numbers[atom_a]);
        const Eigen::Vector3d C_A(mol._standard(atom_a, 0),
                                  mol._standard(atom_a, 1),
                                  mol._standard(atom_a, 2));

        for (int q = 0; q < 3; ++q)
        {
            double dV_sum = 0.0;
            for (const auto &sp : shell_pairs)
            {
                const std::size_t ii = sp.A._index;
                const std::size_t jj = sp.B._index;
                const double dv = HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(
                    sp, C_A, Z_A, q);
                if (ii == jj)
                    dV_sum += P_t(ii, jj) * dv;
                else
                    dV_sum += 2.0 * P_t(ii, jj) * dv;
            }
            grad(atom_a, q) += dV_sum;
        }
    }

    const double cx = exact_exchange_coefficient;
    auto gamma_fn = [&P_t, &P_a, &P_b, cx](std::size_t ii, std::size_t jj,
                                           std::size_t kk, std::size_t ll) -> double
    {
        return 2.0 * P_t(ii, jj) * P_t(kk, ll) -
               cx * (2.0 * P_a(ii, kk) * P_a(jj, ll) + 2.0 * P_b(ii, kk) * P_b(jj, ll));
    };

    for (const auto &spAB : shell_pairs)
    {
        const std::size_t ii = spAB.A._index;
        const std::size_t jj = spAB.B._index;
        const int atom_A = shell_atom[bf_shell[ii]];
        const int atom_B = shell_atom[bf_shell[jj]];

        for (const auto &spCD : shell_pairs)
        {
            const std::size_t kk = spCD.A._index;
            const std::size_t ll = spCD.B._index;
            const int atom_C = shell_atom[bf_shell[kk]];
            const int atom_D = shell_atom[bf_shell[ll]];

            if (schwarz_q(ii, jj) * schwarz_q(kk, ll) < calc._integral._tol_eri)
                continue;

            const auto dI = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(spAB, spCD);
            accumulate_eri_gradient_permutations(
                grad,
                dI,
                gamma_fn,
                ii,
                jj,
                kk,
                ll,
                atom_A,
                atom_B,
                atom_C,
                atom_D);
        }
    }

    for (std::size_t a = 0; a < natoms; ++a)
    {
        for (std::size_t b = 0; b < natoms; ++b)
        {
            if (a == b)
                continue;
            const double Za = static_cast<double>(mol.atomic_numbers[a]);
            const double Zb = static_cast<double>(mol.atomic_numbers[b]);
            const double dx = mol._standard(a, 0) - mol._standard(b, 0);
            const double dy = mol._standard(a, 1) - mol._standard(b, 1);
            const double dz = mol._standard(a, 2) - mol._standard(b, 2);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < 1e-24)
            {
                return std::unexpected(
                    std::format("Gradient: atoms {} and {} are coincident or too close for nuclear-repulsion differentiation",
                                static_cast<int>(a + 1),
                                static_cast<int>(b + 1)));
            }
            const double r3 = std::pow(r2, 1.5);
            grad(a, 0) -= Za * Zb * dx / r3;
            grad(a, 1) -= Za * Zb * dy / r3;
            grad(a, 2) -= Za * Zb * dz / r3;
        }
    }

    return grad;
}

std::expected<Eigen::MatrixXd, std::string> HartreeFock::Gradient::compute_rmp2_gradient(
    HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    if (calc._correlation != HartreeFock::PostHF::RMP2)
        return std::unexpected(std::string("RMP2 gradient requested without correlation = RMP2"));
    if (calc._scf._scf != HartreeFock::SCFType::RHF || calc._info._scf.is_uhf)
        return std::unexpected(std::string("RMP2 gradient requires an RHF reference"));

    auto grad_res = HartreeFock::Correlation::build_rmp2_gradient_intermediates(calc, shell_pairs);
    if (!grad_res)
        return std::unexpected(std::string("RMP2 gradient build failed: ") + grad_res.error());
    const auto &grad_data = *grad_res;

    const Eigen::MatrixXd &P_ref = calc._info._scf.alpha.density;
    const Eigen::MatrixXd P_corr = grad_data.P_gamma_ao - P_ref;
    const int nb = static_cast<int>(calc._shells.nbasis());
    const auto &gamma_pair = grad_data.Gamma_pair_ao;

    auto gamma_fn = [&P_ref, &P_corr, &gamma_pair, nb](std::size_t ii, std::size_t jj,
                                                       std::size_t kk, std::size_t ll) -> double
    {
        const double P0_ij = P_ref(ii, jj);
        const double P0_kl = P_ref(kk, ll);
        const double P0_ik = P_ref(ii, kk);
        const double P0_jl = P_ref(jj, ll);

        const double dP_ij = P_corr(ii, jj);
        const double dP_kl = P_corr(kk, ll);
        const double dP_ik = P_corr(ii, kk);
        const double dP_jl = P_corr(jj, ll);

        return 2.0 * P0_ij * P0_kl - P0_ik * P0_jl + 2.0 * (dP_ij * P0_kl + P0_ij * dP_kl) - (dP_ik * P0_jl + P0_ik * dP_jl) + gamma_pair[idx_dm2_grad(static_cast<int>(ii), static_cast<int>(jj), static_cast<int>(kk), static_cast<int>(ll), nb)];
    };

    return compute_closed_shell_gradient_from_density(calc, shell_pairs,
                                                      grad_data.P_ao, grad_data.W_ao, gamma_fn);
}

std::expected<Eigen::MatrixXd, std::string> HartreeFock::Gradient::compute_ump2_gradient(
    HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    if (calc._correlation != HartreeFock::PostHF::UMP2)
        return std::unexpected(std::string("UMP2 gradient requested without correlation = UMP2"));
    if (calc._scf._scf != HartreeFock::SCFType::UHF || !calc._info._scf.is_uhf)
        return std::unexpected(std::string("UMP2 gradient requires a UHF reference"));

    auto grad_res = HartreeFock::Correlation::build_ump2_gradient_intermediates(calc, shell_pairs);
    if (!grad_res)
        return std::unexpected(std::string("UMP2 gradient build failed: ") + grad_res.error());
    const auto &grad_data = *grad_res;

    const Eigen::MatrixXd &Pa_ref = calc._info._scf.alpha.density;
    const Eigen::MatrixXd &Pb_ref = calc._info._scf.beta.density;
    const Eigen::MatrixXd Pt_ref = Pa_ref + Pb_ref;
    const Eigen::MatrixXd &dPa = grad_data.P_alpha_corr_ao;
    const Eigen::MatrixXd &dPb = grad_data.P_beta_corr_ao;
    const Eigen::MatrixXd dPt = dPa + dPb;
    const int nb = static_cast<int>(calc._shells.nbasis());
    const auto &gamma_pair = grad_data.Gamma_pair_ao;

    auto gamma_fn = [&Pa_ref, &Pb_ref, &Pt_ref, &dPa, &dPb, &dPt, &gamma_pair, nb](std::size_t ii, std::size_t jj,
                                                                                   std::size_t kk, std::size_t ll) -> double
    {
        const double ref =
            2.0 * Pt_ref(ii, jj) * Pt_ref(kk, ll) -
            2.0 * Pa_ref(ii, kk) * Pa_ref(jj, ll) -
            2.0 * Pb_ref(ii, kk) * Pb_ref(jj, ll);

        const double linear =
            2.0 * (dPt(ii, jj) * Pt_ref(kk, ll) + Pt_ref(ii, jj) * dPt(kk, ll)) -
            2.0 * (dPa(ii, kk) * Pa_ref(jj, ll) + Pa_ref(ii, kk) * dPa(jj, ll)) -
            2.0 * (dPb(ii, kk) * Pb_ref(jj, ll) + Pb_ref(ii, kk) * dPb(jj, ll));

        return ref + linear +
               gamma_pair[idx_dm2_grad(static_cast<int>(ii), static_cast<int>(jj),
                                       static_cast<int>(kk), static_cast<int>(ll), nb)];
    };

    return compute_closed_shell_gradient_from_density(
        calc, shell_pairs, grad_data.P_total_ao, grad_data.W_ao, gamma_fn);
}
