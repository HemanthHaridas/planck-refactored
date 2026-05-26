#include "gradient.h"

#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <vector>

#include "basis/basis.h"
#include "basis/spherical.h"
#include "integrals/base.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"
#include "post_hf/mp2.h"
#include "post_hf/mp2_gradient.h"
#include "scf/scf.h"
#include "symmetry/integral_symmetry.h"

// ─── Helpers ─────────────────────────────────────────────────────────────────

// In spherical mode, the density / energy-weighted density that SCF produces
// live in the (2L+1)-per-shell spherical AO basis, but the derivative integral
// engine (integrals/os.cpp) emits Cartesian shell-pair blocks indexed by
// shell offsets. To keep the gradient kernel basis-agnostic, lift any AO matrix
// from the spherical basis back to the Cartesian one via M_cart = Cᵀ · M_sph · C.
// In Cartesian mode this is a no-op pass-through.
//
// See basis/spherical.h::lift_density_sph_to_cart for the energy-invariance
// contract that justifies this lift.
static std::expected<Eigen::MatrixXd, std::string> lift_ao_matrix_if_spherical(
    const HartreeFock::Calculator &calc, const Eigen::MatrixXd &M)
{
    if (!calc._shells._spherical)
        return M;
    return HartreeFock::BasisFunctions::lift_density_sph_to_cart(
        M, calc._shells._cart_to_sph);
}

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

static Eigen::MatrixXd compute_nuclear_repulsion_gradient(
    const HartreeFock::Calculator &calc)
{
    const auto &mol = calc._molecule;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(mol.natoms, 3);
    for (std::size_t a = 0; a < mol.natoms; ++a)
    {
        for (std::size_t b = 0; b < mol.natoms; ++b)
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
    GammaFn &&gamma_fn,
    bool assume_pair_exchange_symmetry = true)
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
    std::vector<std::unique_ptr<HartreeFock::ShellPair>> reversed_pairs(shell_pairs.size());

    std::vector<int> bf_shell(nb, -1);
    for (std::size_t s = 0; s < nshells; ++s)
    {
        for (std::size_t mu = 0; mu < nb; ++mu)
            if (bfs[mu]._shell == &shells[s])
                bf_shell[mu] = static_cast<int>(s);
    }

    for (const auto &sp : shell_pairs)
    {
        const std::size_t pair_index = static_cast<std::size_t>(&sp - shell_pairs.data());
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
            if (!reversed_pairs[pair_index])
                reversed_pairs[pair_index] = std::make_unique<HartreeFock::ShellPair>(sp.B, sp.A);
            const auto &sp_rev = *reversed_pairs[pair_index];
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

    if (assume_pair_exchange_symmetry)
    {
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
    }
    else
    {
        std::vector<HartreeFock::ShellPair> all_pairs;
        all_pairs.reserve(nb * nb);
        for (std::size_t ii = 0; ii < nb; ++ii)
            for (std::size_t jj = 0; jj < nb; ++jj)
                all_pairs.emplace_back(bfs[ii], bfs[jj]);

        for (const auto &spAB : all_pairs)
        {
            const std::size_t ii = spAB.A._index;
            const std::size_t jj = spAB.B._index;
            const int atom_A = shell_atom[bf_shell[ii]];
            const int atom_B = shell_atom[bf_shell[jj]];

            for (const auto &spCD : all_pairs)
            {
                const std::size_t kk = spCD.A._index;
                const std::size_t ll = spCD.B._index;
                const int atom_C = shell_atom[bf_shell[kk]];
                const int atom_D = shell_atom[bf_shell[ll]];

                if (schwarz_q(ii, jj) * schwarz_q(kk, ll) < calc._integral._tol_eri)
                    continue;

                const double gamma = gamma_fn(ii, jj, kk, ll);
                if (std::abs(gamma) < 1e-14)
                    continue;

                const auto dI = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(spAB, spCD);
                const double fac = 0.25 * gamma;
                for (int q = 0; q < 3; ++q)
                {
                    grad(atom_A, q) += fac * dI[q];
                    grad(atom_B, q) += fac * dI[3 + q];
                    grad(atom_C, q) += fac * dI[6 + q];
                    grad(atom_D, q) += fac * dI[9 + q];
                }
            }
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
    // In spherical mode the stored density lives in the (2L+1)-per-shell
    // spherical AO basis; lift it back to the Cartesian basis (Cᵀ P_sph C) so
    // the Cartesian derivative-integral kernel below can contract against it
    // with shell-pair offsets. In Cartesian mode this is a value copy of P.
    auto P_lifted = lift_ao_matrix_if_spherical(calc, calc._info._scf.alpha.density);
    if (!P_lifted)
        return std::unexpected(P_lifted.error());
    const Eigen::MatrixXd P = std::move(*P_lifted); // already has factor 2

    int n_elec = 0;
    for (std::size_t a = 0; a < calc._molecule.natoms; ++a)
        n_elec += calc._molecule.atomic_numbers[a];
    n_elec -= calc._molecule.charge;
    const int n_occ = n_elec / 2;

    const Eigen::MatrixXd C_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_occ);
    const Eigen::VectorXd eps = calc._info._scf.alpha.mo_energies.head(n_occ);
    // W is assembled from MO coefficients, so it inherits whichever AO basis
    // the stored MOs use (spherical when _spherical, Cartesian otherwise); the
    // same Cartesian lift applies.
    const Eigen::MatrixXd W_native = 2.0 * C_occ * eps.asDiagonal() * C_occ.transpose();
    auto W_lifted = lift_ao_matrix_if_spherical(calc, W_native);
    if (!W_lifted)
        return std::unexpected(W_lifted.error());
    const Eigen::MatrixXd W = std::move(*W_lifted);

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
    // nb is the Cartesian basis-function count (sized off shells), which is what
    // schwarz_q, bf_shell, and the derivative integral blocks are indexed by —
    // even in spherical mode, where the densities themselves are lifted below.
    const std::size_t nb = basis.nbasis();

    // UHF densities (already without factor 2). In spherical mode these come
    // from SCF in the (2L+1)-per-shell basis; lift each spin block back to the
    // Cartesian basis (Cᵀ P_sph C) so the Cartesian derivative kernel indexing
    // (sp.A._index, sp.B._index) lines up. In Cartesian mode the lift is a copy.
    auto Pa_lifted = lift_ao_matrix_if_spherical(calc, calc._info._scf.alpha.density);
    if (!Pa_lifted)
        return std::unexpected(Pa_lifted.error());
    auto Pb_lifted = lift_ao_matrix_if_spherical(calc, calc._info._scf.beta.density);
    if (!Pb_lifted)
        return std::unexpected(Pb_lifted.error());
    const Eigen::MatrixXd P_a = std::move(*Pa_lifted);
    const Eigen::MatrixXd P_b = std::move(*Pb_lifted);
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

    // Energy-weighted density (no factor 2 for UHF). Built from MOs in whichever
    // basis SCF used (spherical or Cartesian); lifted the same way as P_a/P_b.
    const Eigen::MatrixXd W_native = Ca_occ * ea.asDiagonal() * Ca_occ.transpose() + Cb_occ * eb.asDiagonal() * Cb_occ.transpose();
    auto W_lifted = lift_ao_matrix_if_spherical(calc, W_native);
    if (!W_lifted)
        return std::unexpected(W_lifted.error());
    const Eigen::MatrixXd W = std::move(*W_lifted);

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

    auto mp2_res = HartreeFock::Correlation::rmp2_kernel(calc, shell_pairs, calc._mp2);
    if (!mp2_res)
        return std::unexpected(std::string("RMP2 gradient MP2 kernel failed: ") + mp2_res.error());
    auto grad_res = HartreeFock::Correlation::build_rmp2_gradient_intermediates(calc, shell_pairs, *mp2_res);
    if (!grad_res)
        return std::unexpected(std::string("RMP2 gradient build failed: ") + grad_res.error());
    return grad_res->electronic_gradient + compute_nuclear_repulsion_gradient(calc);
}

std::expected<Eigen::MatrixXd, std::string> HartreeFock::Gradient::compute_ump2_gradient(
    HartreeFock::Calculator &calc,
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    if (calc._correlation != HartreeFock::PostHF::UMP2)
        return std::unexpected(std::string("UMP2 gradient requested without correlation = UMP2"));
    if (calc._scf._scf != HartreeFock::SCFType::UHF || !calc._info._scf.is_uhf)
        return std::unexpected(std::string("UMP2 gradient requires a UHF reference"));

    auto mp2_res = HartreeFock::Correlation::ump2_kernel(calc, shell_pairs, calc._mp2);
    if (!mp2_res)
        return std::unexpected(std::string("UMP2 gradient MP2 kernel failed: ") + mp2_res.error());
    auto grad_res = HartreeFock::Correlation::build_ump2_gradient_intermediates(calc, shell_pairs, *mp2_res);
    if (!grad_res)
        return std::unexpected(std::string("UMP2 gradient build failed: ") + grad_res.error());
    return grad_res->electronic_gradient + compute_nuclear_repulsion_gradient(calc);
}
