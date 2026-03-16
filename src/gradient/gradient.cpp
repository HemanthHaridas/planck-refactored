#include "gradient.h"

#include <cmath>
#include <vector>
#include <stdexcept>

#include "integrals/os.h"
#include "integrals/shellpair.h"

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Build a map: shell index in _shells._shells → atom index in _molecule.
// Matches shell._center ≈ _molecule._standard.row(a) within 1e-6 Bohr.
static std::vector<int> build_shell_atom_map(
    const HartreeFock::Calculator& calc)
{
    const auto& shells = calc._shells._shells;
    const auto& mol    = calc._molecule;
    const std::size_t nshells = shells.size();

    std::vector<int> map(nshells, -1);
    for (std::size_t s = 0; s < nshells; ++s)
    {
        const Eigen::Vector3d& sc = shells[s]._center;
        for (std::size_t a = 0; a < mol.natoms; ++a)
        {
            const double dx = sc[0] - mol._standard(a, 0);
            const double dy = sc[1] - mol._standard(a, 1);
            const double dz = sc[2] - mol._standard(a, 2);
            if (dx*dx + dy*dy + dz*dz < 1e-10) { // 1e-5 Bohr tolerance squared
                map[s] = static_cast<int>(a);
                break;
            }
        }
        if (map[s] < 0)
            throw std::runtime_error("Gradient: shell does not match any atom");
    }
    return map;
}

// ─── RHF Gradient ─────────────────────────────────────────────────────────────
//
// g[A,x] = 2 Σ_{μ∈A,ν} P_μν (dT_μν/dA_x + dV_μν/dA_x)  [1e GTO-centre]
//        + Σ_{μν}       P_μν dV_μν^{C=A}/dR_{A,x}         [nucleus-position V]
//        + ½ Σ_{μνλσ}   Γ_μνλσ d(μν|λσ)/dA_x              [2e ERI]
//        - 2 Σ_{μ∈A,ν}  W_μν dS_μν/dA_x                   [Pulay]
//        + Σ_{B≠A}      Z_A Z_B (R_A-R_B)/|R_A-R_B|³      [nuclear repulsion]

Eigen::MatrixXd HartreeFock::Gradient::compute_rhf_gradient(
    const HartreeFock::Calculator& calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    const auto& mol    = calc._molecule;
    const auto& basis  = calc._shells;
    const std::size_t natoms = mol.natoms;
    const std::size_t nb     = basis.nbasis();

    // Density and MO info
    const Eigen::MatrixXd& P = calc._info._scf.alpha.density;   // already has factor 2

    // Electron count
    int n_elec = 0;
    for (std::size_t a = 0; a < natoms; ++a) n_elec += mol.atomic_numbers[a];
    n_elec -= mol.charge;
    const int n_occ = n_elec / 2;

    const Eigen::MatrixXd C_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_occ);
    const Eigen::VectorXd eps   = calc._info._scf.alpha.mo_energies.head(n_occ);

    // Energy-weighted density: W_μν = 2 Σ_{i occ} ε_i C_μi C_νi
    const Eigen::MatrixXd W = 2.0 * C_occ * eps.asDiagonal() * C_occ.transpose();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natoms, 3);

    // Shell → atom map
    const std::vector<int> shell_atom = build_shell_atom_map(calc);

    // Build per-shell AO start index
    const auto& shells = basis._shells;
    const auto& bfs    = basis._basis_functions;
    const std::size_t nshells = shells.size();

    // Build ALL (ii,jj) basis-function pairs (including ii>jj) for the ERI loop.
    // Term 3 must iterate over ALL orderings, not just unique (ii≤jj) pairs,
    // matching the Python reference which iterates over all shell combinations.
    std::vector<HartreeFock::ShellPair> all_pairs;
    all_pairs.reserve(nb * nb);
    for (std::size_t ii = 0; ii < nb; ++ii)
        for (std::size_t jj = 0; jj < nb; ++jj)
            all_pairs.emplace_back(bfs[ii], bfs[jj]);

    // Map each basis function to its shell index
    std::vector<int> bf_shell(nb, -1);
    for (std::size_t s = 0; s < nshells; ++s)
    {
        for (std::size_t mu = 0; mu < nb; ++mu)
            if (bfs[mu]._shell == &shells[s])
                bf_shell[mu] = static_cast<int>(s);
    }

    // ── Term 1+Pulay: iterate over ALL (μ, ν) basis function pairs ────────────
    // For each pair, atom_a = atom of shell containing μ.
    // Contribution: grad[atom_a] += 2*P[μ,ν]*(dT+dV) - 2*W[μ,ν]*dS
    // Factor 2 comes from Hermitian symmetry of P (covers both μ∈A and ν∈A contributions
    // by iterating over all (μ,ν) including both orderings via the shell_pairs).
    //
    // For unique pairs (μ ≤ ν) we handle both the (μ,ν) contribution to atom(μ)
    // and the (ν,μ) contribution to atom(ν) by creating the reversed ShellPair.

    for (const auto& sp : shell_pairs)
    {
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;
        const int atom_ii = shell_atom[bf_shell[ii]];
        const int atom_jj = shell_atom[bf_shell[jj]];

        // A-side derivative: d<μ|..|ν>/dA contributes to grad[atom(μ)]
        const auto dST_A = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
        const auto dV_A  = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp, mol);

        for (int q = 0; q < 3; ++q) {
            const double contrib = 2.0 * P(ii, jj) * (dST_A[q+3] + dV_A[q])
                                 - 2.0 * W(ii, jj) * dST_A[q];
            grad(atom_ii, q) += contrib;
        }

        // B-side derivative (reversed pair): d<ν|..|μ>/dB contributes to grad[atom(ν)]
        // Only needed for off-diagonal pairs (ii != jj); use symmetry of matrix elements.
        if (ii != jj)
        {
            HartreeFock::ShellPair sp_rev(sp.B, sp.A);
            const auto dST_B = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp_rev);
            const auto dV_B  = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp_rev, mol);

            for (int q = 0; q < 3; ++q) {
                const double contrib = 2.0 * P(jj, ii) * (dST_B[q+3] + dV_B[q])
                                     - 2.0 * W(jj, ii) * dST_B[q];
                grad(atom_jj, q) += contrib;
            }
        }
    }

    // ── Term 2: nucleus-position V derivative ─────────────────────────────────
    // grad[A,q] += Σ_{μν} P[μ,ν] * dV_μν^{C=A}/dR_{A,q}
    for (std::size_t atom_a = 0; atom_a < natoms; ++atom_a)
    {
        const double Z_A = static_cast<double>(mol.atomic_numbers[atom_a]);
        const Eigen::Vector3d C_A(mol._standard(atom_a, 0),
                                  mol._standard(atom_a, 1),
                                  mol._standard(atom_a, 2));

        for (int q = 0; q < 3; ++q)
        {
            double dV_sum = 0.0;
            for (const auto& sp : shell_pairs)
            {
                const std::size_t ii = sp.A._index;
                const std::size_t jj = sp.B._index;
                const double dv = HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(
                                      sp, C_A, Z_A, q);
                // P is symmetric; unique pair (ii,jj) with ii≤jj covers both orderings
                if (ii == jj)
                    dV_sum += P(ii, jj) * dv;
                else
                    dV_sum += 2.0 * P(ii, jj) * dv;
            }
            grad(atom_a, q) += dV_sum;
        }
    }

    // ── Term 3: ERI gradient ──────────────────────────────────────────────────
    // contrib[cen,q] = 0.25 * Σ_{ALL ijkl} Gamma[i,j,k,l] * dI[i,j,k,l,cen,q]
    // Γ_μνλσ = 2*P_μν*P_λσ - P_μλ*P_νσ
    //
    // Must iterate over ALL (ii,jj) × ALL (kk,ll) combinations (not just unique pairs).
    // This matches the Python reference which iterates all shell combinations.
    for (const auto& spAB : all_pairs)
    {
        const std::size_t ii = spAB.A._index;
        const std::size_t jj = spAB.B._index;
        const int atom_A = shell_atom[bf_shell[ii]];
        const int atom_B = shell_atom[bf_shell[jj]];

        for (const auto& spCD : all_pairs)
        {
            const std::size_t kk = spCD.A._index;
            const std::size_t ll = spCD.B._index;
            const int atom_C = shell_atom[bf_shell[kk]];
            const int atom_D = shell_atom[bf_shell[ll]];

            // Two-particle density matrix element
            // Gamma = 2*P[i,j]*P[k,l] - P[i,k]*P[j,l]
            const double Gamma = 2.0 * P(ii, jj) * P(kk, ll) - P(ii, kk) * P(jj, ll);
            if (std::abs(Gamma) < 1e-14) continue;

            const auto dI = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(spAB, spCD);

            // contrib[cen,q] = 0.25 * Gamma * dI[cen*3+q]
            const double fac = 0.25 * Gamma;
            for (int q = 0; q < 3; ++q) {
                grad(atom_A, q) += fac * dI[0*3 + q];
                grad(atom_B, q) += fac * dI[1*3 + q];
                grad(atom_C, q) += fac * dI[2*3 + q];
                grad(atom_D, q) += fac * dI[3*3 + q];
            }
        }
    }

    // ── Term 4: nuclear repulsion gradient ───────────────────────────────────
    for (std::size_t a = 0; a < natoms; ++a)
    {
        for (std::size_t b = 0; b < natoms; ++b)
        {
            if (a == b) continue;
            const double Za = static_cast<double>(mol.atomic_numbers[a]);
            const double Zb = static_cast<double>(mol.atomic_numbers[b]);
            const double dx = mol._standard(a, 0) - mol._standard(b, 0);
            const double dy = mol._standard(a, 1) - mol._standard(b, 1);
            const double dz = mol._standard(a, 2) - mol._standard(b, 2);
            const double r  = std::sqrt(dx*dx + dy*dy + dz*dz);
            const double r3 = r * r * r;
            const double fac = Za * Zb / r3;
            grad(a, 0) -= fac * dx;
            grad(a, 1) -= fac * dy;
            grad(a, 2) -= fac * dz;
        }
    }

    return grad;
}

// ─── UHF Gradient ─────────────────────────────────────────────────────────────

Eigen::MatrixXd HartreeFock::Gradient::compute_uhf_gradient(
    const HartreeFock::Calculator& calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    const auto& mol    = calc._molecule;
    const auto& basis  = calc._shells;
    const std::size_t natoms = mol.natoms;
    const std::size_t nb     = basis.nbasis();

    // UHF densities (already without factor 2)
    const Eigen::MatrixXd& P_a = calc._info._scf.alpha.density;
    const Eigen::MatrixXd& P_b = calc._info._scf.beta.density;
    const Eigen::MatrixXd  P_t = P_a + P_b;   // total density

    // Electron counts
    int n_elec = 0;
    for (std::size_t a = 0; a < natoms; ++a) n_elec += mol.atomic_numbers[a];
    n_elec -= mol.charge;
    const int n_unpaired = static_cast<int>(mol.multiplicity) - 1;
    const int n_alpha = (n_elec + n_unpaired) / 2;
    const int n_beta  = (n_elec - n_unpaired) / 2;

    const Eigen::MatrixXd Ca_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_alpha);
    const Eigen::VectorXd ea     = calc._info._scf.alpha.mo_energies.head(n_alpha);
    const Eigen::MatrixXd Cb_occ = calc._info._scf.beta.mo_coefficients.leftCols(n_beta);
    const Eigen::VectorXd eb     = calc._info._scf.beta.mo_energies.head(n_beta);

    // Energy-weighted density (no factor 2 for UHF)
    const Eigen::MatrixXd W = Ca_occ * ea.asDiagonal() * Ca_occ.transpose()
                            + Cb_occ * eb.asDiagonal() * Cb_occ.transpose();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natoms, 3);

    const std::vector<int> shell_atom = build_shell_atom_map(calc);
    const auto& shells = basis._shells;
    const auto& bfs    = basis._basis_functions;
    const std::size_t nshells = shells.size();

    std::vector<int> bf_shell(nb, -1);
    for (std::size_t s = 0; s < nshells; ++s)
        for (std::size_t mu = 0; mu < nb; ++mu)
            if (bfs[mu]._shell == &shells[s])
                bf_shell[mu] = static_cast<int>(s);

    // All (ii,jj) basis-function pairs for the ERI gradient loop (see RHF notes)
    std::vector<HartreeFock::ShellPair> all_pairs;
    all_pairs.reserve(nb * nb);
    for (std::size_t ii = 0; ii < nb; ++ii)
        for (std::size_t jj = 0; jj < nb; ++jj)
            all_pairs.emplace_back(bfs[ii], bfs[jj]);

    // ── Term 1+Pulay (same structure as RHF but using P_t and W) ─────────────
    for (const auto& sp : shell_pairs)
    {
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;
        const int atom_ii = shell_atom[bf_shell[ii]];
        const int atom_jj = shell_atom[bf_shell[jj]];

        const auto dST_A = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
        const auto dV_A  = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp, mol);

        for (int q = 0; q < 3; ++q) {
            const double contrib = 2.0 * P_t(ii, jj) * (dST_A[q+3] + dV_A[q])
                                 - 2.0 * W(ii, jj)   * dST_A[q];
            grad(atom_ii, q) += contrib;
        }

        if (ii != jj)
        {
            HartreeFock::ShellPair sp_rev(sp.B, sp.A);
            const auto dST_B = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp_rev);
            const auto dV_B  = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp_rev, mol);

            for (int q = 0; q < 3; ++q) {
                const double contrib = 2.0 * P_t(jj, ii) * (dST_B[q+3] + dV_B[q])
                                     - 2.0 * W(jj, ii)   * dST_B[q];
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
            for (const auto& sp : shell_pairs)
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
    for (const auto& spAB : all_pairs)
    {
        const std::size_t ii = spAB.A._index;
        const std::size_t jj = spAB.B._index;
        const int atom_A = shell_atom[bf_shell[ii]];
        const int atom_B = shell_atom[bf_shell[jj]];

        for (const auto& spCD : all_pairs)
        {
            const std::size_t kk = spCD.A._index;
            const std::size_t ll = spCD.B._index;
            const int atom_C = shell_atom[bf_shell[kk]];
            const int atom_D = shell_atom[bf_shell[ll]];

            const double Gamma = 2.0 * P_t(ii, jj) * P_t(kk, ll)
                               - 2.0 * P_a(ii, kk)  * P_a(jj, ll)
                               - 2.0 * P_b(ii, kk)  * P_b(jj, ll);
            if (std::abs(Gamma) < 1e-14) continue;

            const auto dI = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(spAB, spCD);

            const double fac = 0.25 * Gamma;
            for (int q = 0; q < 3; ++q) {
                grad(atom_A, q) += fac * dI[0*3 + q];
                grad(atom_B, q) += fac * dI[1*3 + q];
                grad(atom_C, q) += fac * dI[2*3 + q];
                grad(atom_D, q) += fac * dI[3*3 + q];
            }
        }
    }

    // ── Term 4: nuclear repulsion ─────────────────────────────────────────────
    for (std::size_t a = 0; a < natoms; ++a)
    {
        for (std::size_t b = 0; b < natoms; ++b)
        {
            if (a == b) continue;
            const double Za = static_cast<double>(mol.atomic_numbers[a]);
            const double Zb = static_cast<double>(mol.atomic_numbers[b]);
            const double dx = mol._standard(a, 0) - mol._standard(b, 0);
            const double dy = mol._standard(a, 1) - mol._standard(b, 1);
            const double dz = mol._standard(a, 2) - mol._standard(b, 2);
            const double r3 = std::pow(dx*dx + dy*dy + dz*dz, 1.5);
            grad(a, 0) -= Za * Zb * dx / r3;
            grad(a, 1) -= Za * Zb * dy / r3;
            grad(a, 2) -= Za * Zb * dz / r3;
        }
    }

    return grad;
}
