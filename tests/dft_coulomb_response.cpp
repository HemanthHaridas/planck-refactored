// D2.2.1 (docs/SOSCF_DFT.md): the Coulomb-response piece of the
// RKS SOSCF orbital-Hessian-vector product h_op.
//
// Unlike D2.0's XC-kernel piece, no new production function is written
// here: J is LINEAR in the density by construction
// (J[P](mu,nu) = sum (mu nu|lam sig) P(lam,sig)), so the induced Coulomb
// potential for a trial density dP, delta_J(dP), is exactly
// _compute_2e_j_direct(shell_pairs, dP, ...) -- the identical
// already-tested memory-direct builder the RKS loop's own per-iteration
// Coulomb term uses (src/dft/driver.cpp, assemble_current_ks_potential),
// just called on dP instead of the SCF's own converged density. There is
// nothing DFT-specific or SOSCF-specific to write: this file exists only
// to VERIFY that reuse is valid for an arbitrary (non-SCF) trial density,
// per D2.2.1's own scope note.
//
// Verification is a finite-difference check independent of D2.0/D2.1's
// machinery entirely, and independent of any orbital rotation: for
// P(kappa) = P0 + kappa*dP (a plain LINEAR density perturbation, not a
// Cayley-rotated one -- this piece has no orbital-rotation dependence at
// all, so testing it against a rotation would import machinery it does not
// need), E_Coulomb(kappa) = 0.5*Tr(P(kappa)*J(P(kappa))) is an exact
// quadratic in kappa (J is linear), so
//   d^2 E_Coulomb/d kappa^2 = Tr(dP . J(dP)) = Tr(dP . delta_J(dP))
// holds EXACTLY (no O(kappa) truncation, unlike D2.0's XC piece), which
// means a finite difference at any reasonably small h should already agree
// to near machine precision -- a much tighter bar than D2.0/D2.1's own
// checks, which is appropriate since this piece is pure linear algebra.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "dft/dft_gradient.h"
#include "dft/dh_pt2_gradient.h"
#include "dft/xc_grid.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    // Same water/STO-3G-style fixture construction as
    // tests/fock_accumulate.cpp's make_water -- deliberately reproduced
    // rather than shared, since these test binaries are independent CMake
    // targets and this file has no other dependency on that one.
    HartreeFock::Calculator make_water(const std::string &basis_name,
        int displaced_atom = -1, int cartesian = -1, double displacement_bohr = 0.0)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        mol.natoms = 3;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(3);
        mol.atomic_numbers << 8, 1, 1;
        mol.atomic_masses.resize(3);
        mol.atomic_masses << 16.0, 1.0, 1.0;
        mol.coordinates.resize(3, 3);
        mol.coordinates <<
            0.000000, 0.000000, 0.117176,
            0.000000, 0.757200, -0.468704,
            0.000000, -0.757200, -0.468704;

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        if (displaced_atom >= 0)
        {
            if (displaced_atom >= static_cast<int>(mol.natoms) || cartesian < 0 || cartesian >= 3)
            {
                fail("invalid water displacement requested");
                return calc;
            }
            mol._coordinates(displaced_atom, cartesian) += displacement_bohr;
        }
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
        {
            fail("read_gbs_basis failed (" + basis_name + "): " + basis_res.error());
            return calc;
        }
        calc._shells = std::move(*basis_res);
        return calc;
    }

    Eigen::MatrixXd random_symmetric(std::size_t nb, std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd M(nb, nb);
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j <= i; ++j)
            {
                const double v = dist(rng);
                M(i, j) = v;
                M(j, i) = v;
            }
        return M;
    }

    void check_coulomb_response(unsigned seed)
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok)
            return;

        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();

        std::mt19937 rng(seed);
        // P0: a plausible symmetric AO density (not converged -- the
        // identity under test holds for any P0/dP, exactly D2.2.1's own
        // "NOT necessarily an occ-virt rotation-generated one" note).
        const Eigen::MatrixXd P0 = random_symmetric(nb, rng);
        const Eigen::MatrixXd dP = random_symmetric(nb, rng);

        auto J = [&](const Eigen::MatrixXd &P) -> Eigen::MatrixXd
        {
            return _compute_2e_j_direct(
                shell_pairs, P, nb, HartreeFock::IntegralMethod::ObaraSaika,
                HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        };

        auto e_coulomb_at = [&](double h) -> double
        {
            const Eigen::MatrixXd P = P0 + h * dP;
            return 0.5 * (P.array() * J(P).array()).sum();
        };

        const double delta_J_dP_norm = J(dP).norm();
        if (delta_J_dP_norm == 0.0)
        {
            fail("delta_J(dP) is exactly zero -- fixture is degenerate, seed=" + std::to_string(seed));
            return;
        }

        const double tr_full = (dP.array() * J(dP).array()).sum();

        // E_Coulomb(kappa) is an EXACT quadratic in kappa (J is linear), so
        // the central second-difference recovers d^2E/dk^2 up to O(h^2)
        // truncation only -- tight tolerances are appropriate here, unlike
        // D2.0/D2.1's XC-kernel checks where P(kappa) itself is nonlinear.
        const double e0 = e_coulomb_at(0.0);
        for (double h : {1e-2, 1e-3, 1e-4})
        {
            const double ep = e_coulomb_at(h);
            const double em = e_coulomb_at(-h);
            const double h_fd = (ep + em - 2.0 * e0) / (h * h);
            const double diff = std::abs(h_fd - tr_full);
            // O(h^2) truncation on a quantity of this scale.
            const double tol = 1e-6 + 1.0 * h * h * std::abs(tr_full);
            if (diff > tol)
            {
                fail("seed=" + std::to_string(seed) + " h=" + std::to_string(h) +
                     ": h_fd_coulomb=" + std::to_string(h_fd) +
                     " Tr(dP.deltaJ(dP))=" + std::to_string(tr_full) +
                     " diff=" + std::to_string(diff) + " tol=" + std::to_string(tol));
            }
        }
    }

    // The Eq. (41) factory must use Planck's raw, memory-direct J and K
    // builders without importing the RHF 1/2 exchange convention.  This is a
    // real AO-integral fixture; the reference is assembled independently from
    // the two raw builders exposed by integrals/base.h.
    void check_eq41_direct_eri_binding()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok) return;
        const std::vector<HartreeFock::ShellPair> shell_pairs =
            build_shellpairs(calc._shells);
        const std::size_t nb = calc._shells._basis_functions.size();
        std::mt19937 rng(41);
        const Eigen::MatrixXd d = random_symmetric(nb, rng);
        const Eigen::MatrixXd e = random_symmetric(nb, rng);
        auto op = DFT::Gradient::make_dh_eq41_direct_eri_response_operator({
            &shell_pairs, nb, HartreeFock::IntegralMethod::ObaraSaika, 0.0,
            nullptr, 0.53,
            [](const Eigen::Ref<const Eigen::MatrixXd> &x)
                -> std::expected<Eigen::MatrixXd, std::string>
            { return 0.125 * x; }});
        if (!op)
        {
            fail("Eq. 41 direct-ERI factory rejected valid water/STO-3G inputs: " + op.error());
            return;
        }
        const auto r = op->apply(d);
        const auto re = op->apply(e);
        const auto rsum = op->apply(d + e);
        if (!r || !re || !rsum)
        {
            fail("Eq. 41 direct-ERI response application failed");
            return;
        }
        const Eigen::MatrixXd j = _compute_2e_j_direct(
            shell_pairs, d, nb, HartreeFock::IntegralMethod::ObaraSaika,
            HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        const Eigen::MatrixXd k = _compute_2e_k_direct(
            shell_pairs, d, nb, HartreeFock::IntegralMethod::ObaraSaika,
            HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        // Eq. (41) maps the symmetric closed-shell D' test density onto the
        // KS response adjoint.  The XC HVP receives the same total-density
        // factor four as the Coulomb channel, so this callback contributes
        // 4 * 0.125 D rather than its old unscaled value.
        const Eigen::MatrixXd expected = 4.0 * j - 0.53 * (k + k.transpose()) + 0.5 * d;
        const double max_diff = (*r - expected).cwiseAbs().maxCoeff();
        const double asym = (*r - r->transpose()).cwiseAbs().maxCoeff();
        const double linearity = (*rsum - *r - *re).cwiseAbs().maxCoeff();
        if (max_diff > 1e-10)
            fail("Eq. 41 direct-ERI response differs from independently assembled raw J/K, max|d|=" +
                 std::to_string(max_diff));
        if (asym > 1e-10)
            fail("Eq. 41 direct-ERI response is not symmetric for a symmetric density");
        if (linearity > 1e-10)
            fail("Eq. 41 direct-ERI response is not linear in its density");
    }

    // Eq. (46) uses Planck's total RKS density P = 2 C_occ C_occ^T.  This
    // physical water/STO-3G fixture verifies both coefficient sets against
    // independently built direct J/K contractions: the reference 1/2,-1/4
    // terms reproduce the closed-shell mean-field two-electron energy, and
    // the correction-only D P - 1/2 D P terms reproduce its first variation.
    void check_eq46_closed_shell_total_density()
    {
        HartreeFock::Calculator calc = make_water("sto-3g");
        if (!g_ok) return;
        const auto shell_pairs = build_shellpairs(calc._shells);
        const int nb = static_cast<int>(calc._shells.nbasis());
        const auto [s, t] = _compute_1e(shell_pairs, nb, HartreeFock::IntegralMethod::ObaraSaika);
        const Eigen::MatrixXd v = _compute_nuclear_attraction(
            shell_pairs, nb, calc._molecule, HartreeFock::IntegralMethod::ObaraSaika);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s_solver(s);
        if (s_solver.info() != Eigen::Success || s_solver.eigenvalues().minCoeff() <= 1e-8)
        {
            fail("Eq. 46 water/STO-3G overlap orthogonalization failed");
            return;
        }
        const Eigen::MatrixXd x = s_solver.eigenvectors() *
            s_solver.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
            s_solver.eigenvectors().transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> h_solver(x.transpose() * (t + v) * x);
        if (h_solver.info() != Eigen::Success)
        {
            fail("Eq. 46 water/STO-3G core-orbital diagonalization failed");
            return;
        }
        const Eigen::MatrixXd c = x * h_solver.eigenvectors();
        constexpr int no = 5; // Ten-electron closed-shell water.
        const int nv = nb - no;
        if (nv <= 0 || (c.transpose() * s * c - Eigen::MatrixXd::Identity(nb, nb)).norm() > 1e-10)
        {
            fail("Eq. 46 water/STO-3G orbital metric is invalid");
            return;
        }
        const Eigen::MatrixXd p = 2.0 * c.leftCols(no) * c.leftCols(no).transpose();
        if (std::abs((p * s).trace() - 10.0) > 1e-10)
        {
            fail("Eq. 46 fixture did not construct Planck's ten-electron total RKS density");
            return;
        }

        HartreeFock::Correlation::RMP2Result result;
        result.n_occ = no;
        result.n_virt = nv;
        result.t2.resize(static_cast<std::size_t>(no) * no * nv * nv);
        for (std::size_t q = 0; q < result.t2.size(); ++q)
            result.t2[q] = 1e-3 * static_cast<double>((q % 11) + 1);
        const auto amplitudes = DFT::Gradient::build_dh_pt2_amplitude_density(result, 0.27);
        const Eigen::MatrixXd z = Eigen::MatrixXd::Zero(nv, no);
        const auto relaxed = amplitudes
            ? DFT::Gradient::build_dh_relaxed_difference_density(*amplitudes, z)
            : std::expected<DFT::Gradient::DHRelaxedDifferenceDensity, std::string>(
                std::unexpected("Eq. 46 amplitude fixture setup failed"));
        const auto gamma = (amplitudes && relaxed)
            ? DFT::Gradient::build_dh_eq46_47_two_particle_density(*relaxed, *amplitudes, c, p, 1.0)
            : std::expected<DFT::Gradient::DHEq46_47TwoParticleDensity, std::string>(
                std::unexpected("Eq. 46 two-particle fixture setup failed"));
        if (!gamma)
        {
            fail("Eq. 46 two-particle density construction failed");
            return;
        }

        const Eigen::MatrixXd j = _compute_2e_j_direct(
            shell_pairs, p, nb, HartreeFock::IntegralMethod::ObaraSaika,
            HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        const Eigen::MatrixXd k = _compute_2e_k_direct(
            shell_pairs, p, nb, HartreeFock::IntegralMethod::ObaraSaika,
            HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        const double reference_jk = 0.5 * (p.array() * j.array()).sum() -
                                    0.25 * (p.array() * k.array()).sum();
        const Eigen::MatrixXd pa = 0.5 * p;
        const Eigen::MatrixXd ka = _compute_2e_k_direct(
            shell_pairs, pa, nb, HartreeFock::IntegralMethod::ObaraSaika,
            HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        const double spin_resolved = 0.5 * (p.array() * j.array()).sum() -
            0.5 * ((pa.array() * ka.array()).sum() + (pa.array() * ka.array()).sum());
        if (std::abs(reference_jk - spin_resolved) > 1e-10)
            fail("Eq. 46 reference 1/2,-1/4 factors disagree with alpha=beta=P/2 reconstruction");

        const std::vector<double> eri = _compute_2e(
            shell_pairs, nb, HartreeFock::IntegralMethod::ObaraSaika,
            HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
        double reference_gamma = 0.0, separable_gamma = 0.0, nonseparable_gamma = 0.0;
        for (int mu = 0; mu < nb; ++mu)
            for (int nu = 0; nu < nb; ++nu)
                for (int ka_idx = 0; ka_idx < nb; ++ka_idx)
                    for (int ta = 0; ta < nb; ++ta)
                    {
                        const std::size_t q =
                            ((static_cast<std::size_t>(mu) * nb + nu) * nb + ka_idx) * nb + ta;
                        reference_gamma += (0.5 * p(mu,nu) * p(ka_idx,ta) -
                            0.25 * p(mu,ka_idx) * p(nu,ta)) * eri[q];
                        separable_gamma += gamma->separable_symmetric_ao[q] * eri[q];
                        nonseparable_gamma += gamma->nonseparable_symmetric_ao[q] * eri[q];
                    }
        const Eigen::MatrixXd &d = gamma->relaxed_difference_ao;
        const double correction_jk = (d.array() * j.array()).sum() -
                                     0.5 * (d.array() * k.array()).sum();
        if (std::abs(reference_gamma - reference_jk) > 1e-9)
            fail("Eq. 46 reference tensor does not reproduce direct water/STO-3G J/K energy");
        if (std::abs(separable_gamma - correction_jk) > 1e-9)
            fail("Eq. 46 correction D P - 1/2 D P does not reproduce the total-density J/K first variation");

        // Eq. (47) oracle: backtransform the physical AO ERIs explicitly to
        // (ia|jb) and contract with (1+delta_ij) t_tilde. This has no use of
        // the AO two-particle builder or its eightfold-symmetry adapter.
        const auto tidx = [no, nv](int i, int j_idx, int a, int b)
        { return ((static_cast<std::size_t>(i) * no + j_idx) * nv + a) * nv + b; };
        double nonseparable_mo = 0.0;
        for (int i = 0; i < no; ++i)
            for (int j_idx = 0; j_idx < no; ++j_idx)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                    {
                        double ijab = 0.0;
                        for (int mu = 0; mu < nb; ++mu)
                            for (int nu = 0; nu < nb; ++nu)
                                for (int ka_idx = 0; ka_idx < nb; ++ka_idx)
                                    for (int ta = 0; ta < nb; ++ta)
                                    {
                                        const std::size_t q =
                                            ((static_cast<std::size_t>(mu) * nb + nu) * nb + ka_idx) * nb + ta;
                                        ijab += c(mu,i) * c(nu,no+a) * c(ka_idx,j_idx) * c(ta,no+b) * eri[q];
                                    }
                        nonseparable_mo += static_cast<double>(1 + (i == j_idx)) *
                            amplitudes->t_tilde[tidx(i,j_idx,a,b)] * ijab;
                    }
        if (std::abs(nonseparable_gamma - nonseparable_mo) > 1e-9)
            fail("Eq. 47 AO nonseparable tensor does not reproduce the independent water/STO-3G (ia|jb) contraction");

        // Physical Eq. (33) translation audit. Construct W from Eqs. (42)-
        // (45), then differentiate every AO integral explicitly with respect
        // to every water Cartesian coordinate. A simultaneous translation of
        // all three nuclei leaves h, S, and each ERI invariant, so each named
        // non-XC channel must have zero net force on its own.
        std::vector<double> mo_eri(static_cast<std::size_t>(nb) * nb * nb * nb, 0.0);
        const auto idx4 = [nb](int p_idx, int q_idx, int r_idx, int s_idx)
        { return ((static_cast<std::size_t>(p_idx) * nb + q_idx) * nb + r_idx) * nb + s_idx; };
        for (int p_idx = 0; p_idx < nb; ++p_idx)
            for (int q_idx = 0; q_idx < nb; ++q_idx)
                for (int r_idx = 0; r_idx < nb; ++r_idx)
                    for (int s_idx = 0; s_idx < nb; ++s_idx)
                        for (int mu = 0; mu < nb; ++mu)
                            for (int nu = 0; nu < nb; ++nu)
                                for (int ka_idx = 0; ka_idx < nb; ++ka_idx)
                                    for (int ta = 0; ta < nb; ++ta)
                                        mo_eri[idx4(p_idx,q_idx,r_idx,s_idx)] +=
                                            c(mu,p_idx) * c(nu,q_idx) * c(ka_idx,r_idx) * c(ta,s_idx) *
                                            eri[idx4(mu,nu,ka_idx,ta)];
        const auto zero_response = [](const Eigen::Ref<const Eigen::MatrixXd> &x)
            -> std::expected<Eigen::MatrixXd, std::string>
            { return Eigen::MatrixXd::Zero(x.rows(), x.cols()); };
        const auto response = DFT::Gradient::make_dh_eq41_direct_eri_response_operator({
            &shell_pairs, static_cast<std::size_t>(nb), HartreeFock::IntegralMethod::ObaraSaika,
            0.0, nullptr, 1.0, zero_response});
        const Eigen::VectorXd eps = h_solver.eigenvalues();
        // Water/STO-3G Eq. (27)/(40) sign and factor audit.  Build the
        // orbital Hessian independently from unit-vector actions, solve the
        // paper convention A Z = -L, then finite-difference Z under a signed
        // perturbation of L.
        const auto eq41_water = response
            ? DFT::Gradient::build_dh_eq41_response_density(*amplitudes, c, *response)
            : std::expected<DFT::Gradient::DHEq41ResponseDensity, std::string>(
                std::unexpected("Eq. 27/40 water response setup failed"));
        const auto eq40_water = eq41_water
            ? DFT::Gradient::build_dh_eq40_amplitude_rhs(result, *amplitudes, mo_eri, 0.27, true)
            : std::expected<DFT::Gradient::DHEq40AmplitudeRHS, std::string>(
                std::unexpected("Eq. 27/40 water amplitude setup failed"));
        const auto rhs_water = (eq41_water && eq40_water)
            ? DFT::Gradient::build_dh_lagrangian_rhs(
                *eq41_water, c.leftCols(no), c.rightCols(nv), *eq40_water,
                DFT::Gradient::DHRHSConvention::LegacyExternalPlusInternal)
            : std::expected<DFT::Gradient::DHLagrangianRHS, std::string>(
                std::unexpected("Eq. 27/40 water RHS setup failed"));
        if (!rhs_water || !response ||
            (rhs_water->total_ai - rhs_water->response_ai - rhs_water->amplitude_ai).norm() > 1e-13 ||
            (rhs_water->amplitude_ai - rhs_water->three_external_ai - rhs_water->three_internal_ai).norm() > 1e-13)
        {
            fail("Eq. 27/40 water RHS sign/factor assembly failed");
            return;
        }
        const int nov = no * nv;
        Eigen::MatrixXd hessian_water(nov, nov);
        for (int column = 0; column < nov; ++column)
        {
            Eigen::MatrixXd unit = Eigen::MatrixXd::Zero(nv, no);
            unit(column / no, column % no) = 1.0;
            const auto action = DFT::Gradient::apply_dh_eq27_hessian(
                unit, c.leftCols(no), c.rightCols(nv), eps, *response);
            if (!action)
            {
                fail("Eq. 27 water Hessian action failed");
                return;
            }
            for (int a = 0; a < nv; ++a)
                for (int i = 0; i < no; ++i)
                    hessian_water(a * no + i, column) = action->total_ai(a, i);
        }
        Eigen::VectorXd rhs_vector(nov), delta_rhs(nov);
        for (int a = 0; a < nv; ++a)
            for (int i = 0; i < no; ++i)
            {
                rhs_vector(a * no + i) = rhs_water->total_ai(a, i);
                delta_rhs(a * no + i) = 0.01 * static_cast<double>(1 + a * no + i);
            }
        const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> hessian_solver(hessian_water);
        const Eigen::VectorXd z_vector = hessian_solver.solve(-rhs_vector);
        const Eigen::VectorXd tangent = hessian_solver.solve(-delta_rhs);
        constexpr double rhs_step = 1e-5;
        const Eigen::VectorXd z_plus = hessian_solver.solve(-(rhs_vector + rhs_step * delta_rhs));
        const Eigen::VectorXd z_minus = hessian_solver.solve(-(rhs_vector - rhs_step * delta_rhs));
        if (!z_vector.allFinite() || !tangent.allFinite() ||
            (hessian_water * z_vector + rhs_vector).cwiseAbs().maxCoeff() > 1e-11 ||
            ((z_plus - z_minus) / (2.0 * rhs_step) - tangent).cwiseAbs().maxCoeff() > 1e-10)
        {
            fail("Eq. 27 water residual perturbation sign audit failed");
            return;
        }
        // Eqs. (42)-(45) at the *solved* water response.  Do not reuse the
        // zero-Z relaxed density used by the downstream frozen-contract
        // derivative test below: Eq. (42)'s raw R(D) and Eq. (45)'s Z term
        // must see the physical solution of this Eq. (27) fixture.
        Eigen::MatrixXd z_ai_water(nv, no);
        for (int a = 0; a < nv; ++a)
            for (int i = 0; i < no; ++i)
                z_ai_water(a, i) = z_vector(a * no + i);
        const auto relaxed_response = DFT::Gradient::build_dh_relaxed_difference_density(
            *amplitudes, z_ai_water);
        const auto diagonal_response = relaxed_response
            ? DFT::Gradient::build_dh_eq42_43_energy_weighted_density(
                *relaxed_response, result, *amplitudes, c, eps, mo_eri, *response)
            : std::expected<DFT::Gradient::DHEq42_43EnergyWeightedDensity, std::string>(
                std::unexpected("Eq. 42-43 solved-water relaxed density setup failed"));
        const auto off_diagonal_response = relaxed_response
            ? DFT::Gradient::build_dh_eq44_45_energy_weighted_density(
                *relaxed_response, result, *amplitudes, eps, mo_eri)
            : std::expected<DFT::Gradient::DHEq44_45EnergyWeightedDensity, std::string>(
                std::unexpected("Eq. 44-45 solved-water relaxed density setup failed"));
        const auto overlap_response = (diagonal_response && off_diagonal_response)
            ? DFT::Gradient::build_dh_overlap_density_adapter(
                *diagonal_response, *off_diagonal_response)
            : std::expected<DFT::Gradient::DHOverlapDensity, std::string>(
                std::unexpected("Eq. 42-45 solved-water overlap setup failed"));
        if (!relaxed_response || !diagonal_response || !off_diagonal_response || !overlap_response)
        {
            fail("Eq. 42-45 solved-water overlap construction failed");
            return;
        }
        // Eq. (42): independently apply the already direct-J/K-validated
        // Eq. (41) AO operator to Eq. (28)'s symmetric relaxed density, as
        // specified immediately before the paper's W-block equations, then
        // make its MO occupied block and -1/2 factor explicit.
        const Eigen::MatrixXd raw_response_ao =
            c * relaxed_response->symmetric_mo * c.transpose();
        const auto raw_response_fock = response->apply(raw_response_ao);
        const auto unsymmetrized_response_fock = response->apply(
            c * relaxed_response->raw_mo * c.transpose());
        Eigen::MatrixXd eq42_response_ref = Eigen::MatrixXd::Zero(no, no);
        if (raw_response_fock)
            eq42_response_ref = -0.5 * (c.transpose() * *raw_response_fock * c).topLeftCorner(no, no);
        Eigen::MatrixXd eq42_orbital_ref = Eigen::MatrixXd::Zero(no, no);
        Eigen::MatrixXd eq42_amplitude_ref = Eigen::MatrixXd::Zero(no, no);
        Eigen::MatrixXd eq43_orbital_ref = Eigen::MatrixXd::Zero(nv, nv);
        Eigen::MatrixXd eq43_amplitude_ref = Eigen::MatrixXd::Zero(nv, nv);
        Eigen::MatrixXd eq44_ref = Eigen::MatrixXd::Zero(no, nv);
        Eigen::MatrixXd eq45_ref = Eigen::MatrixXd::Zero(nv, no);
        for (int i = 0; i < no; ++i)
            for (int j_idx = 0; j_idx < no; ++j_idx)
            {
                eq42_orbital_ref(i,j_idx) = -0.5 * relaxed_response->raw_mo(i,j_idx) *
                    (eps(i) + eps(j_idx));
                for (int k = 0; k < no; ++k)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            eq42_amplitude_ref(i,j_idx) -= 0.5 *
                                amplitudes->t_tilde[tidx(j_idx,k,a,b)] *
                                mo_eri[idx4(i,no+a,k,no+b)];
            }
        for (int a = 0; a < nv; ++a)
            for (int b = 0; b < nv; ++b)
            {
                eq43_orbital_ref(a,b) = -0.5 * relaxed_response->raw_mo(no+a,no+b) *
                    (eps(no+a) + eps(no+b));
                for (int i = 0; i < no; ++i)
                    for (int j_idx = i; j_idx < no; ++j_idx)
                        for (int k = 0; k < nv; ++k)
                            eq43_amplitude_ref(a,b) -= mo_eri[idx4(i,no+a,j_idx,no+k)] *
                                amplitudes->t_tilde[tidx(i,j_idx,b,k)] /
                                static_cast<double>(1 + (i == j_idx));
            }
        for (int i = 0; i < no; ++i)
            for (int a = 0; a < nv; ++a)
            {
                for (int k = 0; k < no; ++k)
                    for (int j_idx = 0; j_idx < no; ++j_idx)
                        for (int b = 0; b < nv; ++b)
                            eq44_ref(i,a) -= amplitudes->t_tilde[tidx(k,j_idx,a,b)] *
                                mo_eri[idx4(k,i,j_idx,no+b)];
                eq45_ref(a,i) = -eps(i) * z_ai_water(a,i);
            }
        const double overlap_symmetry =
            (overlap_response->symmetric_mo - overlap_response->symmetric_mo.transpose()).norm();
        // The AO overlap itself is a nontrivial symmetric water matrix; its
        // contraction exercises the same raw-to-symmetric identity used for
        // every nuclear overlap derivative (which is likewise symmetric).
        const Eigen::MatrixXd water_overlap_derivative = c.transpose() * s * c;
        const double raw_overlap_contraction =
            (overlap_response->raw_mo.array() * water_overlap_derivative.array()).sum();
        const double symmetric_overlap_contraction =
            (overlap_response->symmetric_mo.array() * water_overlap_derivative.array()).sum();
        if (!raw_response_fock || !unsymmetrized_response_fock ||
            (*raw_response_fock - *unsymmetrized_response_fock).norm() > 1e-12 ||
            (diagonal_response->response_oo - eq42_response_ref).norm() > 1e-11 ||
            (diagonal_response->orbital_oo - eq42_orbital_ref).norm() > 1e-13 ||
            (diagonal_response->amplitude_oo - eq42_amplitude_ref).norm() > 1e-13 ||
            (diagonal_response->orbital_vv - eq43_orbital_ref).norm() > 1e-13 ||
            (diagonal_response->amplitude_vv - eq43_amplitude_ref).norm() > 1e-13 ||
            (off_diagonal_response->w_ia - eq44_ref).norm() > 1e-13 ||
            (off_diagonal_response->w_ai - eq45_ref).norm() > 1e-13 ||
            (overlap_response->raw_mo.topLeftCorner(no,no) - diagonal_response->w_oo).norm() > 1e-13 ||
            (overlap_response->raw_mo.bottomRightCorner(nv,nv) - diagonal_response->w_vv).norm() > 1e-13 ||
            (overlap_response->raw_mo.topRightCorner(no,nv) - off_diagonal_response->w_ia).norm() > 1e-13 ||
            (overlap_response->raw_mo.bottomLeftCorner(nv,no) - off_diagonal_response->w_ai).norm() > 1e-13 ||
            overlap_symmetry > 1e-13 ||
            std::abs(raw_overlap_contraction - symmetric_overlap_contraction) > 1e-12)
        {
            fail("Eq. 42-45 solved-water block, orientation, or symmetric-overlap audit failed");
            return;
        }
        const auto diagonal_w = response
            ? DFT::Gradient::build_dh_eq42_43_energy_weighted_density(
                *relaxed, result, *amplitudes, c, eps, mo_eri, *response)
            : std::expected<DFT::Gradient::DHEq42_43EnergyWeightedDensity, std::string>(
                std::unexpected("Eq. 33 physical response setup failed"));
        const auto off_diagonal_w = (diagonal_w)
            ? DFT::Gradient::build_dh_eq44_45_energy_weighted_density(
                *relaxed, result, *amplitudes, eps, mo_eri)
            : std::expected<DFT::Gradient::DHEq44_45EnergyWeightedDensity, std::string>(
                std::unexpected("Eq. 33 physical off-diagonal setup failed"));
        const auto overlap_w = (diagonal_w && off_diagonal_w)
            ? DFT::Gradient::build_dh_overlap_density_adapter(*diagonal_w, *off_diagonal_w)
            : std::expected<DFT::Gradient::DHOverlapDensity, std::string>(
                std::unexpected("Eq. 33 physical overlap setup failed"));
        if (!overlap_w)
        {
            fail("Eq. 33 physical water/STO-3G W construction failed");
            return;
        }

        std::vector<int> bf_atom(nb, -1);
        const auto &bfs = calc._shells._basis_functions;
        for (int mu = 0; mu < nb; ++mu)
            for (int atom = 0; atom < 3; ++atom)
            {
                const Eigen::Vector3d delta = bfs[mu]._shell->_center - calc._molecule._standard.row(atom).transpose();
                if (delta.squaredNorm() < 1e-12)
                    bf_atom[mu] = atom;
            }
        if (std::any_of(bf_atom.begin(), bf_atom.end(), [](int atom) { return atom < 0; }))
        {
            fail("Eq. 33 physical water/STO-3G basis-function atom map failed");
            return;
        }

        DFT::Gradient::DHEq33NonXCDerivatives derivatives;
        derivatives.hamiltonian_ao.assign(9, Eigen::MatrixXd::Zero(nb, nb));
        derivatives.overlap_ao.assign(9, Eigen::MatrixXd::Zero(nb, nb));
        derivatives.eri_ao.assign(9, std::vector<double>(static_cast<std::size_t>(nb) * nb * nb * nb, 0.0));
        for (int mu = 0; mu < nb; ++mu)
            for (int nu = 0; nu < nb; ++nu)
            {
                const HartreeFock::ShellPair sp_mu_nu(bfs[mu], bfs[nu]);
                const HartreeFock::ShellPair sp_nu_mu(bfs[nu], bfs[mu]);
                const auto d_mu = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp_mu_nu);
                const auto d_nu = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp_nu_mu);
                for (int atom = 0; atom < 3; ++atom)
                    for (int cart = 0; cart < 3; ++cart)
                    {
                        const int x_idx = 3 * atom + cart;
                        if (bf_atom[mu] == atom)
                        {
                            derivatives.overlap_ao[x_idx](mu,nu) += d_mu[cart];
                            derivatives.hamiltonian_ao[x_idx](mu,nu) += d_mu[cart + 3] +
                                HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp_mu_nu, calc._molecule)[cart];
                        }
                        if (bf_atom[nu] == atom)
                        {
                            derivatives.overlap_ao[x_idx](mu,nu) += d_nu[cart];
                            derivatives.hamiltonian_ao[x_idx](mu,nu) += d_nu[cart + 3] +
                                HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp_nu_mu, calc._molecule)[cart];
                        }
                        const Eigen::Vector3d center = calc._molecule._standard.row(atom).transpose();
                        derivatives.hamiltonian_ao[x_idx](mu,nu) +=
                            HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(
                                sp_mu_nu, center, calc._molecule.atomic_numbers[atom], cart);
                    }
            }
        for (int mu = 0; mu < nb; ++mu)
            for (int nu = 0; nu < nb; ++nu)
                for (int ka_idx = 0; ka_idx < nb; ++ka_idx)
                    for (int ta = 0; ta < nb; ++ta)
                    {
                        const HartreeFock::ShellPair sp_mu_nu(bfs[mu], bfs[nu]);
                        const HartreeFock::ShellPair sp_ka_ta(bfs[ka_idx], bfs[ta]);
                        const auto d_eri = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(sp_mu_nu, sp_ka_ta);
                        const int centers[4] = {bf_atom[mu], bf_atom[nu], bf_atom[ka_idx], bf_atom[ta]};
                        const std::size_t q = idx4(mu,nu,ka_idx,ta);
                        for (int center = 0; center < 4; ++center)
                            for (int cart = 0; cart < 3; ++cart)
                                derivatives.eri_ao[3 * centers[center] + cart][q] += d_eri[3 * center + cart];
                    }
        const auto eq33 = DFT::Gradient::build_dh_eq33_non_xc_gradient(
            *relaxed, *overlap_w, *gamma, c, derivatives);
        if (!eq33)
        {
            fail("Eq. 33 physical water/STO-3G channel assembly failed");
            return;
        }
        // Eq. (46) Z-only oracle.  The solved response changes the
        // separable D P - 1/2 D P tensor but cannot change Eq. (47), which is
        // amplitude-only.  Central-difference a freshly rebuilt AO ERI
        // contraction with that *fixed* tensor difference, independently of
        // the full Eq. (33) gradient assembler.
        const auto gamma_response = relaxed_response
            ? DFT::Gradient::build_dh_eq46_47_two_particle_density(
                *relaxed_response, *amplitudes, c, p, 1.0)
            : std::expected<DFT::Gradient::DHEq46_47TwoParticleDensity, std::string>(
                std::unexpected("Eq. 46 Z-only water response density setup failed"));
        if (!gamma_response ||
            gamma_response->nonseparable_symmetric_ao.size() != gamma->nonseparable_symmetric_ao.size())
        {
            fail("Eq. 46 Z-only water two-particle construction failed");
            return;
        }
        // Hybrid regression: independent direct-J/K first variation of
        // Q(P)=P:J[P]/2-a_x P:K[P]/4. Test D-prime, pure Z, and D-prime+Z
        // separately, so opposite errors cannot cancel in the relaxed sum.
        // This fixed-geometry density oracle does not use an SCF or PT2 FD.
        const auto mean_field_energy = [&](const Eigen::MatrixXd &density, double ax)
        {
            const Eigen::MatrixXd jd = _compute_2e_j_direct(
                shell_pairs, density, nb, HartreeFock::IntegralMethod::ObaraSaika,
                HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
            const Eigen::MatrixXd kd = _compute_2e_k_direct(
                shell_pairs, density, nb, HartreeFock::IntegralMethod::ObaraSaika,
                HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
            return 0.5 * (density.array() * jd.array()).sum() -
                   0.25 * ax * (density.array() * kd.array()).sum();
        };
        for (double ax : {0.0, 0.53, 1.0})
        {
            const auto base = DFT::Gradient::build_dh_eq46_47_two_particle_density(
                *relaxed, *amplitudes, c, p, ax);
            const auto with_z = DFT::Gradient::build_dh_eq46_47_two_particle_density(
                *relaxed_response, *amplitudes, c, p, ax);
            if (!base || !with_z)
            {
                fail("Hybrid Eq. 46 physical density construction failed");
                return;
            }
            if (base->nonseparable_raw_ao != gamma->nonseparable_raw_ao ||
                base->nonseparable_symmetric_ao != gamma->nonseparable_symmetric_ao ||
                with_z->nonseparable_raw_ao != gamma->nonseparable_raw_ao ||
                with_z->nonseparable_symmetric_ao != gamma->nonseparable_symmetric_ao)
                fail("Hybrid a_x or Z incorrectly scaled the Eq. 47 pair tensor");
            for (int direction = 0; direction < 3; ++direction)
            {
                Eigen::MatrixXd dd = direction == 0 ? base->relaxed_difference_ao :
                                                     with_z->relaxed_difference_ao;
                if (direction == 1) dd -= base->relaxed_difference_ao;
                double contraction = 0.0;
                for (std::size_t q = 0; q < eri.size(); ++q)
                {
                    double coefficient = direction == 0 ? base->separable_symmetric_ao[q] :
                                                          with_z->separable_symmetric_ao[q];
                    if (direction == 1) coefficient -= base->separable_symmetric_ao[q];
                    contraction += coefficient * eri[q];
                }
                const double direct = (dd.array() * (j - 0.5 * ax * k).array()).sum();
                constexpr double density_step = 1e-3;
                const double fd = (mean_field_energy(p + density_step * dd, ax) -
                                   mean_field_energy(p - density_step * dd, ax)) / (2.0 * density_step);
                if (std::abs(contraction - direct) > 1e-10 || std::abs(contraction - fd) > 1e-8)
                {
                    std::cerr << "Hybrid Eq46 ax=" << ax << " direction=" << direction
                              << " tensor=" << contraction << " direct=" << direct << " FD=" << fd << '\n';
                    fail("Hybrid Eq. 46 D-prime/Z density first variation failed");
                }
            }
        }
        std::vector<double> gamma_z_separable(gamma->separable_symmetric_ao.size());
        for (std::size_t q = 0; q < gamma_z_separable.size(); ++q)
        {
            gamma_z_separable[q] = gamma_response->separable_symmetric_ao[q] -
                gamma->separable_symmetric_ao[q];
            if (std::abs(gamma_response->nonseparable_symmetric_ao[q] -
                         gamma->nonseparable_symmetric_ao[q]) > 1e-13)
            {
                fail("Eq. 47 changed under a pure Z-vector response");
                return;
            }
        }
        const auto z_separable_scalar = [&gamma_z_separable](const HartreeFock::Calculator &displaced)
            -> std::expected<double, std::string>
        {
            const auto displaced_pairs = build_shellpairs(displaced._shells);
            const int displaced_nb = static_cast<int>(displaced._shells.nbasis());
            const std::vector<double> displaced_eri = _compute_2e(
                displaced_pairs, displaced_nb, HartreeFock::IntegralMethod::ObaraSaika,
                HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
            if (displaced_eri.size() != gamma_z_separable.size())
                return std::unexpected("Eq. 46 Z-only FD AO dimension changed");
            return std::inner_product(gamma_z_separable.begin(), gamma_z_separable.end(),
                displaced_eri.begin(), 0.0);
        };
        constexpr int z_fd_atom = 1;
        constexpr int z_fd_cart = 0;
        constexpr double z_fd_step = 1e-3;
        const auto z_separable_plus = z_separable_scalar(
            make_water("sto-3g", z_fd_atom, z_fd_cart, z_fd_step));
        const auto z_separable_minus = z_separable_scalar(
            make_water("sto-3g", z_fd_atom, z_fd_cart, -z_fd_step));
        double z_separable_analytic = 0.0;
        for (std::size_t q = 0; q < gamma_z_separable.size(); ++q)
            z_separable_analytic += gamma_z_separable[q] *
                derivatives.eri_ao[3 * z_fd_atom + z_fd_cart][q];
        if (!z_separable_plus || !z_separable_minus ||
            std::abs(((*z_separable_plus - *z_separable_minus) / (2.0 * z_fd_step)) -
                     z_separable_analytic) > 2e-6)
        {
            fail("Eq. 46 Z-only water separable ERI finite difference failed");
            return;
        }
        // Frozen-contract molecular finite-difference oracle for every
        // non-XC Eq. (33) scalar. D, W, and Gamma are frozen numerical AO
        // objects from the reference geometry; only the one- and two-electron
        // integrals are rebuilt at the displaced water geometry.
        const Eigen::MatrixXd d_frozen_ao = c * relaxed->symmetric_mo * c.transpose();
        const Eigen::MatrixXd w_frozen_ao = c * overlap_w->symmetric_mo * c.transpose();
        const auto frozen_non_xc_scalars = [&d_frozen_ao, &w_frozen_ao, &gamma](const HartreeFock::Calculator &displaced)
            -> std::expected<std::array<double, 4>, std::string>
        {
            const auto displaced_pairs = build_shellpairs(displaced._shells);
            const int displaced_nb = static_cast<int>(displaced._shells.nbasis());
            if (displaced_nb != d_frozen_ao.rows())
                return std::unexpected("frozen non-XC FD basis dimension changed");
            const auto [s_displaced, t_displaced] = _compute_1e(
                displaced_pairs, displaced_nb, HartreeFock::IntegralMethod::ObaraSaika);
            const Eigen::MatrixXd v_displaced = _compute_nuclear_attraction(
                displaced_pairs, displaced_nb, displaced._molecule, HartreeFock::IntegralMethod::ObaraSaika);
            const std::vector<double> eri_displaced = _compute_2e(
                displaced_pairs, displaced_nb, HartreeFock::IntegralMethod::ObaraSaika,
                HartreeFock::ERIKernel::Coulomb, 0.0, 0.0, nullptr);
            double separable = 0.0, nonseparable = 0.0;
            for (std::size_t q = 0; q < eri_displaced.size(); ++q)
            {
                separable += gamma->separable_symmetric_ao[q] * eri_displaced[q];
                nonseparable += gamma->nonseparable_symmetric_ao[q] * eri_displaced[q];
            }
            return std::array<double, 4>{
                (d_frozen_ao.array() * (t_displaced + v_displaced).array()).sum(),
                (w_frozen_ao.array() * s_displaced.array()).sum(),
                separable, nonseparable};
        };
        constexpr int frozen_fd_atom = 1;
        constexpr int frozen_fd_cart = 0;
        constexpr double frozen_fd_step = 1e-3;
        const auto frozen_plus = frozen_non_xc_scalars(
            make_water("sto-3g", frozen_fd_atom, frozen_fd_cart, frozen_fd_step));
        const auto frozen_minus = frozen_non_xc_scalars(
            make_water("sto-3g", frozen_fd_atom, frozen_fd_cart, -frozen_fd_step));
        if (!frozen_plus || !frozen_minus)
        {
            fail("Eq. 33 frozen non-XC water finite-difference setup failed");
            return;
        }
        const std::array<double, 4> analytic_non_xc = {
            eq33->one_electron(3 * frozen_fd_atom + frozen_fd_cart),
            eq33->overlap(3 * frozen_fd_atom + frozen_fd_cart),
            eq33->two_electron_separable(3 * frozen_fd_atom + frozen_fd_cart),
            eq33->two_electron_nonseparable(3 * frozen_fd_atom + frozen_fd_cart)};
        for (std::size_t channel = 0; channel < analytic_non_xc.size(); ++channel)
        {
            const double fd = ((*frozen_plus)[channel] - (*frozen_minus)[channel]) / (2.0 * frozen_fd_step);
            if (std::abs(fd - analytic_non_xc[channel]) > 2e-6)
            {
                fail("Eq. 33 frozen non-XC water finite difference failed");
                return;
            }
        }
        for (int cart = 0; cart < 3; ++cart)
        {
            const auto translation_sum = [cart](const Eigen::VectorXd &channel)
            { return channel(cart) + channel(3 + cart) + channel(6 + cart); };
            const double tol = 2e-8;
            if (std::abs(translation_sum(eq33->one_electron)) > tol ||
                std::abs(translation_sum(eq33->overlap)) > tol ||
                std::abs(translation_sum(eq33->two_electron_separable)) > tol ||
                std::abs(translation_sum(eq33->two_electron_nonseparable)) > tol ||
                std::abs(translation_sum(eq33->total)) > tol)
                fail("Eq. 33 physical water/STO-3G channel is not translationally invariant");
        }

        // Step 10: fixed-grid XC-II finite-difference oracle.  LDA keeps the
        // scalar oracle minimal: perturb only rho_P by the physical
        // basis-derivative r=rho_P^(x), hold D_relaxed and the grid fixed, and
        // finite-difference D_relaxed : V_XC[P].  No XC-I or grid motion is
        // present in either side.
        const auto grid = DFT::MakeMolecularGrid(calc._molecule, DFT::GridLevel::Coarse);
        const auto ao_grid = grid ? DFT::evaluate_ao_basis_on_grid(calc._shells, *grid)
            : std::expected<DFT::AOGridEvaluation, std::string>(std::unexpected("XC-II grid setup failed"));
        const auto ao_hessian = grid ? DFT::evaluate_ao_hessian_on_grid(calc._shells, *grid)
            : std::expected<DFT::AOGridHessian, std::string>(std::unexpected("XC-II Hessian setup failed"));
        const auto lda_id = DFT::XC::functional_id("lda_x");
        const auto lda_x = lda_id ? DFT::XC::Functional::create(*lda_id, DFT::XC::Spin::Unpolarized)
            : std::expected<DFT::XC::Functional, std::string>(std::unexpected("XC-II LDA lookup failed"));
        if (!grid || !ao_grid || !ao_hessian || !lda_x)
        {
            fail("Eq. 33 XC-II physical water/STO-3G grid or functional setup failed");
            return;
        }
        const DFT::Gradient::DHEq33XCFixedDensityInputs xc_inputs{
            &calc._molecule, &calc._shells, &*grid, &*ao_grid, &*ao_hessian, p, &*lda_x, &*lda_x};
        const auto xc_ii = DFT::Gradient::build_dh_eq33_xc_fixed_density_gradient(*relaxed, c, xc_inputs);
        auto half_relaxed = *relaxed;
        half_relaxed.symmetric_mo *= 0.5;
        const auto half_xc_ii = DFT::Gradient::build_dh_eq33_xc_fixed_density_gradient(half_relaxed, c, xc_inputs);
        if (!xc_ii || !half_xc_ii || xc_ii->gradient.rows() != 3 || xc_ii->gradient.cols() != 3 ||
            (half_xc_ii->gradient - 0.5 * xc_ii->gradient).norm() > 1e-11)
        {
            fail("Eq. 33 XC-II fixed-relaxed-density linearity failed");
            return;
        }
        const auto ground_grid = DFT::evaluate_density_on_grid(*ao_grid, p);
        const Eigen::MatrixXd d_ao = c * relaxed->symmetric_mo * c.transpose();
        const auto difference_grid = DFT::evaluate_density_on_grid(*ao_grid, d_ao);
        const auto atoms_bf = DFT::Gradient::atom_bf_lists(calc._molecule, calc._shells);
        if (!ground_grid || !difference_grid || !atoms_bf)
        {
            fail("Eq. 33 XC-II physical density setup failed");
            return;
        }
        const Eigen::Index npts = ao_grid->npoints();
        std::vector<double> rho(static_cast<std::size_t>(npts));
        for (Eigen::Index ip = 0; ip < npts; ++ip)
            rho[static_cast<std::size_t>(ip)] = ground_grid->total.rho(ip);
        for (int atom = 0; atom < 3; ++atom)
            for (int cart = 0; cart < 3; ++cart)
            {
                std::vector<double> r(static_cast<std::size_t>(npts));
                for (Eigen::Index ip = 0; ip < npts; ++ip)
                    r[static_cast<std::size_t>(ip)] = DFT::Gradient::drho_channel(
                        p, *ao_grid, ip, atom, cart, *atoms_bf);
                constexpr double step = 1e-4;
                std::vector<double> rho_plus = rho, rho_minus = rho, exc_plus, vrho_plus, exc_minus, vrho_minus;
                for (std::size_t ip = 0; ip < rho.size(); ++ip)
                {
                    rho_plus[ip] += step * r[ip];
                    rho_minus[ip] -= step * r[ip];
                }
                const auto vp = lda_x->evaluate_lda_exc_vxc(rho_plus, static_cast<int>(npts), exc_plus, vrho_plus);
                const auto vm = lda_x->evaluate_lda_exc_vxc(rho_minus, static_cast<int>(npts), exc_minus, vrho_minus);
                if (!vp || !vm)
                {
                    fail("Eq. 33 XC-II LDA finite-difference evaluation failed");
                    return;
                }
                double fd = 0.0;
                for (Eigen::Index ip = 0; ip < npts; ++ip)
                    fd += (*grid).points(ip, 3) * difference_grid->total.rho(ip) *
                        2.0 * (vrho_plus[static_cast<std::size_t>(ip)] - vrho_minus[static_cast<std::size_t>(ip)]) /
                        (2.0 * step);
                if (std::abs(xc_ii->gradient(atom, cart) - fd) > 3e-7)
                    fail("Eq. 33 XC-II fixed-grid LDA finite difference failed");
            }

        // The remaining LDA XC-II grid derivative is split into the derivative
        // of the owner Becke partition and translation of each owner point.
        // Hold P and D fixed as numerical AO matrices while rebuilding both
        // atom-centred quadrature and AO values at displaced water geometry.
        const auto moving = DFT::Gradient::build_dh_eq33_xc_lda_moving_grid_gradient(*relaxed, c, xc_inputs);
        const auto lda_d_ao = DFT::Gradient::build_dh_eq33_xc_lda_difference_ao_gradient(*relaxed, c, xc_inputs);
        const auto lda_complete = DFT::Gradient::build_dh_eq33_complete_xc_ii_gradient(*relaxed, c, xc_inputs);
        if (!moving || !lda_d_ao || !lda_complete ||
            (moving->total - moving->becke_partition - moving->point_translation).norm() > 1e-13 ||
            (lda_complete->total - xc_ii->gradient - lda_d_ao->gradient - moving->total).norm() > 1e-13 ||
            moving->becke_partition.norm() < 1e-11 || moving->point_translation.norm() < 1e-11)
        {
            fail("Eq. 33 LDA moving-grid channel partition failed");
            return;
        }
        const auto moving_scalar = [&p, &d_ao, &lda_x](const HartreeFock::Calculator &displaced)
            -> std::expected<double, std::string>
        {
            const auto displaced_grid = DFT::MakeMolecularGrid(displaced._molecule, DFT::GridLevel::Coarse);
            if (!displaced_grid) return std::unexpected(displaced_grid.error());
            const auto displaced_ao = DFT::evaluate_ao_basis_on_grid(displaced._shells, *displaced_grid);
            if (!displaced_ao) return std::unexpected(displaced_ao.error());
            const auto p_grid = DFT::evaluate_density_on_grid(*displaced_ao, p);
            const auto d_grid = DFT::evaluate_density_on_grid(*displaced_ao, d_ao);
            if (!p_grid || !d_grid) return std::unexpected("moving-grid density evaluation failed");
            const Eigen::Index n = displaced_ao->npoints();
            std::vector<double> displaced_rho(static_cast<std::size_t>(n)), exc, vrho;
            for (Eigen::Index ip = 0; ip < n; ++ip)
                displaced_rho[static_cast<std::size_t>(ip)] = p_grid->total.rho(ip);
            const auto vxc = lda_x->evaluate_lda_exc_vxc(displaced_rho, static_cast<int>(n), exc, vrho);
            if (!vxc) return std::unexpected(vxc.error());
            double value = 0.0;
            for (Eigen::Index ip = 0; ip < n; ++ip)
                value += (*displaced_grid).points(ip, 3) * 2.0 *
                    vrho[static_cast<std::size_t>(ip)] * d_grid->total.rho(ip);
            return value;
        };
        constexpr int fd_atom = 1;
        constexpr int fd_cart = 1;
        constexpr double grid_step = 1e-3;
        const auto plus = make_water("sto-3g", fd_atom, fd_cart, grid_step);
        const auto minus = make_water("sto-3g", fd_atom, fd_cart, -grid_step);
        const auto phi_plus = moving_scalar(plus);
        const auto phi_minus = moving_scalar(minus);
        if (!phi_plus || !phi_minus)
        {
            fail("Eq. 33 LDA moving-grid finite-difference setup failed");
            return;
        }
        const double total_fd = (*phi_plus - *phi_minus) / (2.0 * grid_step);
        const double analytic_total = lda_complete->total(fd_atom, fd_cart);
        if (std::abs(total_fd - analytic_total) > 3e-7)
            fail("Eq. 33 LDA moving-grid water finite difference failed");

        // GGA translation needs the spatial density Hessians.  Use independent
        // PBE exchange and correlation pieces, then translate only the points
        // owned by one atom to isolate this channel from AO-centre response.
        const auto pbe_x_id = DFT::XC::functional_id("gga_x_pbe");
        const auto pbe_c_id = DFT::XC::functional_id("gga_c_pbe");
        const auto pbe_x = pbe_x_id ? DFT::XC::Functional::create(*pbe_x_id, DFT::XC::Spin::Unpolarized)
            : std::expected<DFT::XC::Functional, std::string>(std::unexpected("PBE exchange lookup failed"));
        const auto pbe_c = pbe_c_id ? DFT::XC::Functional::create(*pbe_c_id, DFT::XC::Spin::Unpolarized)
            : std::expected<DFT::XC::Functional, std::string>(std::unexpected("PBE correlation lookup failed"));
        if (!pbe_x || !pbe_c)
        {
            fail("Eq. 33 GGA moving-grid functional setup failed");
            return;
        }
        const DFT::Gradient::DHEq33XCFixedDensityInputs gga_inputs{
            &calc._molecule, &calc._shells, &*grid, &*ao_grid, &*ao_hessian, p, &*pbe_x, &*pbe_c};
        const auto gga_fixed = DFT::Gradient::build_dh_eq33_xc_fixed_density_gradient(*relaxed, c, gga_inputs);
        const auto gga_moving = DFT::Gradient::build_dh_eq33_xc_gga_moving_grid_gradient(*relaxed, c, gga_inputs);
        const auto gga_d_ao = DFT::Gradient::build_dh_eq33_xc_gga_difference_ao_gradient(*relaxed, c, gga_inputs);
        const auto gga_complete = DFT::Gradient::build_dh_eq33_complete_xc_ii_gradient(*relaxed, c, gga_inputs);
        if (!gga_fixed || !gga_moving || !gga_d_ao || !gga_complete ||
            (gga_moving->total - gga_moving->becke_partition - gga_moving->point_translation).norm() > 1e-13 ||
            (gga_complete->total - gga_fixed->gradient - gga_d_ao->gradient - gga_moving->total).norm() > 1e-13 ||
            gga_moving->point_translation.norm() < 1e-11)
        {
            fail("Eq. 33 GGA moving-grid channel partition failed");
            return;
        }
        const auto gga_point_scalar = [&p, &d_ao, &pbe_x, &pbe_c, &calc](const DFT::MolecularGrid &translated)
            -> std::expected<double, std::string>
        {
            const auto translated_ao = DFT::evaluate_ao_basis_on_grid(calc._shells, translated);
            if (!translated_ao) return std::unexpected(translated_ao.error());
            const auto p_grid = DFT::evaluate_density_on_grid(*translated_ao, p);
            const auto d_grid = DFT::evaluate_density_on_grid(*translated_ao, d_ao);
            if (!p_grid || !d_grid) return std::unexpected("GGA point-translation density evaluation failed");
            const Eigen::Index n = translated_ao->npoints();
            std::vector<double> translated_rho(static_cast<std::size_t>(n)), sigma(static_cast<std::size_t>(n));
            for (Eigen::Index ip = 0; ip < n; ++ip)
            {
                translated_rho[static_cast<std::size_t>(ip)] = p_grid->total.rho(ip);
                sigma[static_cast<std::size_t>(ip)] = p_grid->total.gradient_squared()(ip);
            }
            std::vector<double> exc_x, exc_c, vrho_x, vrho_c, fs_x, fs_c;
            const auto vx = pbe_x->evaluate_gga_exc_vxc(
                translated_rho, sigma, static_cast<int>(n), exc_x, vrho_x, fs_x);
            const auto vc = pbe_c->evaluate_gga_exc_vxc(
                translated_rho, sigma, static_cast<int>(n), exc_c, vrho_c, fs_c);
            if (!vx || !vc) return std::unexpected("GGA point-translation vxc evaluation failed");
            double value = 0.0;
            for (Eigen::Index ip = 0; ip < n; ++ip)
            {
                const Eigen::Vector3d grad_p{p_grid->total.grad_x(ip), p_grid->total.grad_y(ip), p_grid->total.grad_z(ip)};
                const Eigen::Vector3d grad_d{d_grid->total.grad_x(ip), d_grid->total.grad_y(ip), d_grid->total.grad_z(ip)};
                value += translated.points(ip, 3) * (
                    (vrho_x[static_cast<std::size_t>(ip)] + vrho_c[static_cast<std::size_t>(ip)]) * d_grid->total.rho(ip) +
                    2.0 * (fs_x[static_cast<std::size_t>(ip)] + fs_c[static_cast<std::size_t>(ip)]) * grad_p.dot(grad_d));
            }
            return value;
        };
        DFT::MolecularGrid gga_point_plus = *grid, gga_point_minus = *grid;
        for (Eigen::Index ip = 0; ip < grid->points.rows(); ++ip)
        {
            if (grid->owner(ip) == fd_atom)
            {
                gga_point_plus.points(ip, fd_cart) += grid_step;
                gga_point_minus.points(ip, fd_cart) -= grid_step;
            }
        }
        const auto gga_point_plus_value = gga_point_scalar(gga_point_plus);
        const auto gga_point_minus_value = gga_point_scalar(gga_point_minus);
        if (!gga_point_plus_value || !gga_point_minus_value)
        {
            fail("Eq. 33 GGA point-translation finite-difference setup failed");
            return;
        }
        const double gga_point_fd = (*gga_point_plus_value - *gga_point_minus_value) / (2.0 * grid_step);
        if (std::abs(gga_point_fd - gga_moving->point_translation(fd_atom, fd_cart)) > 2e-7)
            fail("Eq. 33 GGA Hessian point-translation finite difference failed");
        const auto gga_geometry_scalar = [&p, &d_ao, &pbe_x, &pbe_c](const HartreeFock::Calculator &displaced)
            -> std::expected<double, std::string>
        {
            const auto displaced_grid = DFT::MakeMolecularGrid(displaced._molecule, DFT::GridLevel::Coarse);
            if (!displaced_grid) return std::unexpected(displaced_grid.error());
            const auto displaced_ao = DFT::evaluate_ao_basis_on_grid(displaced._shells, *displaced_grid);
            if (!displaced_ao) return std::unexpected(displaced_ao.error());
            const auto p_grid = DFT::evaluate_density_on_grid(*displaced_ao, p);
            const auto d_grid = DFT::evaluate_density_on_grid(*displaced_ao, d_ao);
            if (!p_grid || !d_grid) return std::unexpected("GGA geometry density evaluation failed");
            const Eigen::Index n = displaced_ao->npoints();
            std::vector<double> geometry_rho(static_cast<std::size_t>(n)), sigma(static_cast<std::size_t>(n));
            for (Eigen::Index ip = 0; ip < n; ++ip)
            {
                geometry_rho[static_cast<std::size_t>(ip)] = p_grid->total.rho(ip);
                sigma[static_cast<std::size_t>(ip)] = p_grid->total.gradient_squared()(ip);
            }
            std::vector<double> exc_x, exc_c, vrho_x, vrho_c, fs_x, fs_c;
            const auto vx = pbe_x->evaluate_gga_exc_vxc(geometry_rho, sigma, static_cast<int>(n), exc_x, vrho_x, fs_x);
            const auto vc = pbe_c->evaluate_gga_exc_vxc(geometry_rho, sigma, static_cast<int>(n), exc_c, vrho_c, fs_c);
            if (!vx || !vc) return std::unexpected("GGA geometry vxc evaluation failed");
            double value = 0.0;
            for (Eigen::Index ip = 0; ip < n; ++ip)
            {
                const Eigen::Vector3d grad_p{p_grid->total.grad_x(ip), p_grid->total.grad_y(ip), p_grid->total.grad_z(ip)};
                const Eigen::Vector3d grad_d{d_grid->total.grad_x(ip), d_grid->total.grad_y(ip), d_grid->total.grad_z(ip)};
                value += displaced_grid->points(ip, 3) * (
                    (vrho_x[static_cast<std::size_t>(ip)] + vrho_c[static_cast<std::size_t>(ip)]) * d_grid->total.rho(ip) +
                    2.0 * (fs_x[static_cast<std::size_t>(ip)] + fs_c[static_cast<std::size_t>(ip)]) * grad_p.dot(grad_d));
            }
            return value;
        };
        const auto gga_geometry_plus = gga_geometry_scalar(plus);
        const auto gga_geometry_minus = gga_geometry_scalar(minus);
        if (!gga_geometry_plus || !gga_geometry_minus)
        {
            fail("Eq. 33 GGA geometry finite-difference setup failed");
            return;
        }
        const double gga_geometry_fd = (*gga_geometry_plus - *gga_geometry_minus) / (2.0 * grid_step);
        const double gga_geometry_analytic = gga_complete->total(fd_atom, fd_cart);
        if (std::abs(gga_geometry_fd - gga_geometry_analytic) > 3e-7)
            fail("Eq. 33 complete GGA geometry-displacement finite difference failed");

        // The driver contract is the sole paper-object assembly boundary: it
        // must reproduce the independently built water objects without
        // rerunning an SCF, PT2 kernel, or Z-vector solve.
        const DFT::Gradient::DHGradientDriverInputs driver_inputs{
            &result, 0.27, c, eps, mo_eri, *response, z, derivatives, xc_inputs};
        const auto driver_contract = DFT::Gradient::build_dh_gradient_driver_contract(driver_inputs);
        // The coefficient must cross the driver boundary, not merely work
        // when passed directly to the low-level tensor builder. This is an
        // assembly test with fixed inputs, not a hybrid Z-stationarity test.
        auto hybrid_driver_inputs = driver_inputs;
        hybrid_driver_inputs.response_operator.exact_exchange = 0.53;
        const auto hybrid_driver_contract =
            DFT::Gradient::build_dh_gradient_driver_contract(hybrid_driver_inputs);
        const auto hybrid_gamma = DFT::Gradient::build_dh_eq46_47_two_particle_density(
            *relaxed, *amplitudes, c, p, 0.53);
        if (!hybrid_driver_contract || !hybrid_gamma ||
            hybrid_driver_contract->two_particle_density.separable_raw_ao != hybrid_gamma->separable_raw_ao ||
            hybrid_driver_contract->two_particle_density.separable_symmetric_ao != hybrid_gamma->separable_symmetric_ao ||
            hybrid_driver_contract->two_particle_density.nonseparable_symmetric_ao != gamma->nonseparable_symmetric_ao)
            fail("DH driver did not propagate hybrid exchange exclusively to Eq. 46");
        const auto literal_pair_metric = DFT::Gradient::build_dh_eq47_pair_metric_overlap_density(
            result, *amplitudes, mo_eri);
        if (!literal_pair_metric)
        {
            fail("Production literal pair overlap fixture failed");
            return;
        }
        auto production_diagonal = *diagonal_w;
        auto production_off_diagonal = *off_diagonal_w;
        production_diagonal.amplitude_oo = literal_pair_metric->w_oo;
        production_diagonal.amplitude_vv = literal_pair_metric->w_vv;
        production_diagonal.w_oo = production_diagonal.response_oo +
            production_diagonal.orbital_oo + production_diagonal.amplitude_oo;
        production_diagonal.w_vv = production_diagonal.orbital_vv + production_diagonal.amplitude_vv;
        production_off_diagonal.w_ia = literal_pair_metric->w_ia;
        const auto production_overlap = DFT::Gradient::build_dh_overlap_density_adapter(
            production_diagonal, production_off_diagonal);
        if (!production_overlap)
        {
            fail("Production overlap adapter fixture failed");
            return;
        }
        const auto production_non_xc = DFT::Gradient::build_dh_eq33_non_xc_gradient(
            *relaxed, *production_overlap, *gamma, c, derivatives);
        const auto pt2_correction = driver_contract
            ? DFT::Gradient::build_dh_eq33_pt2_correction_gradient(*driver_contract)
            : std::expected<DFT::Gradient::DHEq33PT2CorrectionGradient, std::string>(
                std::unexpected("DH driver contract setup failed"));
        if (!driver_contract || !production_non_xc || !pt2_correction ||
            (driver_contract->relaxed_density.symmetric_mo - relaxed->symmetric_mo).norm() > 1e-13 ||
            (driver_contract->overlap_density.symmetric_mo - production_overlap->symmetric_mo).norm() > 1e-13 ||
            (driver_contract->eq42_43_overlap.amplitude_oo - literal_pair_metric->w_oo).norm() > 1e-13 ||
            (driver_contract->eq42_43_overlap.amplitude_vv - literal_pair_metric->w_vv).norm() > 1e-13 ||
            (driver_contract->eq44_45_overlap.w_ia - literal_pair_metric->w_ia).norm() > 1e-13 ||
            (driver_contract->eq44_45_overlap.w_ai - off_diagonal_w->w_ai).norm() > 1e-13 ||
            (driver_contract->non_xc_gradient.total - production_non_xc->total).norm() > 1e-13 ||
            (driver_contract->xc_ii_gradient.total - lda_complete->total).norm() > 1e-13 ||
            (pt2_correction->non_xc.row(1) - production_non_xc->total.segment(3, 3).transpose()).norm() > 1e-13 ||
            (pt2_correction->total - pt2_correction->non_xc - pt2_correction->xc_ii).norm() > 1e-13 ||
            driver_contract->lagrangian_rhs.total_ai.rows() != nv ||
            driver_contract->lagrangian_rhs.total_ai.cols() != no)
            fail("DH gradient driver contract failed to reproduce paper-object assembly");
        if (driver_contract)
        {
            const auto &literal_rhs = driver_contract->lagrangian_rhs;
            const auto solver_rhs = DFT::Gradient::build_dh_lagrangian_rhs(
                driver_contract->eq41_response, c.leftCols(no), c.rightCols(nv),
                driver_contract->eq40_amplitude_rhs);
            if (!solver_rhs ||
                (solver_rhs->total_ai - literal_rhs.total_ai).norm() > 1e-13 ||
                literal_rhs.included_internal_ai.norm() != 0.0 ||
                driver_contract->eq40_amplitude_rhs.three_internal.norm() != 0.0 ||
                (literal_rhs.amplitude_ai - driver_contract->eq40_amplitude_rhs.three_external).norm() > 1e-13)
                fail("DH solver/contract literal RHS convention mismatch");
        }
    }
} // namespace

int main()
{
    check_coulomb_response(1);
    check_coulomb_response(2);
    check_coulomb_response(3);
    check_eq41_direct_eri_binding();
    check_eq46_closed_shell_total_density();

    if (g_ok)
        std::cout << "OK  D2.2.1: Tr(dP . delta_J(dP)) matches FD of the Coulomb energy\n";

    return g_ok ? 0 : 1;
}
