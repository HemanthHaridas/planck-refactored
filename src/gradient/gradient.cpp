#include "gradient.h"

#include <cmath>
#include <vector>
#include <stdexcept>

#include "basis/basis.h"
#include "integrals/os.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include "post_hf/mp2.h"
#include "scf/scf.h"

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

// Rebuild the RHF + RMP2 total energy for the geometry currently stored in
// calc._molecule._standard (Bohr). Used by the semi-numerical RMP2 gradient.
static double run_sp_rmp2_energy(HartreeFock::Calculator& calc)
{
    calc._molecule._coordinates = calc._molecule._standard;
    calc._molecule.coordinates  = calc._molecule._standard / ANGSTROM_TO_BOHR;

    const std::string gbs_path =
        calc._basis._basis_path + "/" + calc._basis._basis_name;
    calc._shells = HartreeFock::BasisFunctions::read_gbs_basis(
        gbs_path, calc._molecule, calc._basis._basis);

    calc._info._scf = HartreeFock::DataSCF(false);
    calc._info._scf.initialize(calc._shells.nbasis());
    calc._scf.set_scf_mode_auto(calc._shells.nbasis());
    calc._info._is_converged   = false;
    calc._use_sao_blocking     = false;
    calc._correlation_energy   = 0.0;
    calc._eri.clear();

    calc._compute_nuclear_repulsion();

    auto shell_pairs = build_shellpairs(calc._shells);
    auto [S, T] = _compute_1e(shell_pairs, calc._shells.nbasis(), calc._integral._engine);
    auto V = _compute_nuclear_attraction(shell_pairs, calc._shells.nbasis(),
                                         calc._molecule, calc._integral._engine);
    calc._overlap = S;
    calc._hcore   = T + V;

    if (auto scf_res = HartreeFock::SCF::run_rhf(calc, shell_pairs); !scf_res)
        throw std::runtime_error("RMP2 gradient RHF failed: " + scf_res.error());

    if (auto mp2_res = HartreeFock::Correlation::run_rmp2(calc, shell_pairs); !mp2_res)
        throw std::runtime_error("RMP2 gradient MP2 failed: " + mp2_res.error());

    calc._total_energy += calc._correlation_energy;
    return calc._total_energy;
}

template <typename GammaFn>
static Eigen::MatrixXd compute_closed_shell_gradient_from_density(
    const HartreeFock::Calculator& calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    const Eigen::MatrixXd& P,
    const Eigen::MatrixXd& W,
    GammaFn&& gamma_fn)
{
    const auto& mol    = calc._molecule;
    const auto& basis  = calc._shells;
    const std::size_t natoms = mol.natoms;
    const std::size_t nb     = basis.nbasis();

    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natoms, 3);

    const std::vector<int> shell_atom = build_shell_atom_map(calc);
    const auto& shells = basis._shells;
    const auto& bfs    = basis._basis_functions;
    const std::size_t nshells = shells.size();

    std::vector<HartreeFock::ShellPair> all_pairs;
    all_pairs.reserve(nb * nb);
    for (std::size_t ii = 0; ii < nb; ++ii)
        for (std::size_t jj = 0; jj < nb; ++jj)
            all_pairs.emplace_back(bfs[ii], bfs[jj]);

    std::vector<int> bf_shell(nb, -1);
    for (std::size_t s = 0; s < nshells; ++s)
    {
        for (std::size_t mu = 0; mu < nb; ++mu)
            if (bfs[mu]._shell == &shells[s])
                bf_shell[mu] = static_cast<int>(s);
    }

    for (const auto& sp : shell_pairs)
    {
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;
        const int atom_ii = shell_atom[bf_shell[ii]];
        const int atom_jj = shell_atom[bf_shell[jj]];

        const auto dST_A = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
        const auto dV_A  = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp, mol);

        for (int q = 0; q < 3; ++q) {
            const double contrib = 2.0 * P(ii, jj) * (dST_A[q+3] + dV_A[q])
                                 - 2.0 * W(ii, jj) * dST_A[q];
            grad(atom_ii, q) += contrib;
        }

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
                    dV_sum += P(ii, jj) * dv;
                else
                    dV_sum += 2.0 * P(ii, jj) * dv;
            }
            grad(atom_a, q) += dV_sum;
        }
    }

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

            const double Gamma = gamma_fn(ii, jj, kk, ll);
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
    const Eigen::MatrixXd& P = calc._info._scf.alpha.density;   // already has factor 2
    int n_elec = 0;
    for (std::size_t a = 0; a < calc._molecule.natoms; ++a) n_elec += calc._molecule.atomic_numbers[a];
    n_elec -= calc._molecule.charge;
    const int n_occ = n_elec / 2;

    const Eigen::MatrixXd C_occ = calc._info._scf.alpha.mo_coefficients.leftCols(n_occ);
    const Eigen::VectorXd eps   = calc._info._scf.alpha.mo_energies.head(n_occ);
    const Eigen::MatrixXd W = 2.0 * C_occ * eps.asDiagonal() * C_occ.transpose();
    auto gamma_fn = [&P](std::size_t ii, std::size_t jj, std::size_t kk, std::size_t ll) -> double {
        return 2.0 * P(ii, jj) * P(kk, ll) - P(ii, kk) * P(jj, ll);
    };
    return compute_closed_shell_gradient_from_density(calc, shell_pairs, P, W, gamma_fn);
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

Eigen::MatrixXd HartreeFock::Gradient::compute_rmp2_gradient(
    const HartreeFock::Calculator& calc)
{
    if (calc._correlation != HartreeFock::PostHF::RMP2)
        throw std::runtime_error("RMP2 gradient requested without correlation = RMP2");
    if (calc._scf._scf != HartreeFock::SCFType::RHF || calc._info._scf.is_uhf)
        throw std::runtime_error("RMP2 gradient requires an RHF reference");

    const std::size_t natoms = calc._molecule.natoms;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(natoms, 3);

    // Central-difference step in Bohr. Chosen smaller than the Hessian default
    // because we are differentiating total energies directly.
    constexpr double step = 1e-3;

    for (std::size_t a = 0; a < natoms; ++a)
    {
        for (int q = 0; q < 3; ++q)
        {
            HartreeFock::Calculator calc_fwd = calc;
            HartreeFock::Calculator calc_bck = calc;

            calc_fwd._molecule._standard(a, q) += step;
            calc_bck._molecule._standard(a, q) -= step;

            const double e_fwd = run_sp_rmp2_energy(calc_fwd);
            const double e_bck = run_sp_rmp2_energy(calc_bck);
            grad(a, q) = (e_fwd - e_bck) / (2.0 * step);
        }
    }

    return grad;
}
