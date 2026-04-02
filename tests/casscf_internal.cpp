#include "post_hf/casscf_internal.h"
#include "post_hf/casscf/ci.h"
#include "post_hf/casscf/orbital.h"
#include "post_hf/casscf/rdm.h"
#include "post_hf/casscf/response.h"
#include "post_hf/casscf/strings.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{

    using namespace HartreeFock::Correlation::CASSCFInternal;
    using namespace HartreeFock::Correlation::CASSCF;

    bool expect(bool condition, const std::string &message)
    {
        if (condition)
            return true;
        std::cerr << message << '\n';
        return false;
    }

    std::vector<int> occupied_orbitals(CIString det, int n_orb)
    {
        std::vector<int> occ;
        for (int i = 0; i < n_orb; ++i)
            if (det & single_bit_mask(i))
                occ.push_back(i);
        return occ;
    }

    CIString pack_spin_det(CIString alpha, CIString beta, int n_act)
    {
        return alpha | ((n_act >= kCIStringBits) ? 0 : (beta << n_act));
    }

    Eigen::VectorXd ci_sigma_excitation_class(
        const CIDeterminantSpace &space,
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const Eigen::MatrixXd &h_eff,
        const std::vector<double> &ga,
        int n_act,
        const Eigen::VectorXd &c)
    {
        const int dim = static_cast<int>(space.dets.size());
        const std::vector<CIString> sd = build_spin_dets(a_strs, b_strs, space.dets, n_act);
        const auto lut = build_det_lookup(sd);
        Eigen::VectorXd sigma = Eigen::VectorXd::Zero(dim);

        for (int j = 0; j < dim; ++j)
        {
            const double cJ = c(j);
            if (std::abs(cJ) < 1e-15)
                continue;

            const auto [ia, ib] = space.dets[j];
            const CIString ket_a = a_strs[ia];
            const CIString ket_b = b_strs[ib];
            std::unordered_set<CIString> seen;

            auto accumulate = [&](CIString bra_a, CIString bra_b)
            {
                const CIString packed = pack_spin_det(bra_a, bra_b, n_act);
                if (!seen.insert(packed).second)
                    return;
                auto it = lut.find(packed);
                if (it == lut.end())
                    return;
                sigma(it->second) +=
                    slater_condon_element(bra_a, bra_b, ket_a, ket_b, h_eff, ga, n_act) * cJ;
            };

            accumulate(ket_a, ket_b);

            const auto occ_alpha = occupied_orbitals(ket_a, n_act);
            const auto occ_beta = occupied_orbitals(ket_b, n_act);

            for (int p : occ_alpha)
                for (int q = 0; q < n_act; ++q)
                {
                    if (ket_a & single_bit_mask(q))
                        continue;
                    auto ann = apply_annihilation(ket_a, p);
                    if (!ann.valid)
                        continue;
                    auto cre = apply_creation(ann.det, q);
                    if (!cre.valid)
                        continue;
                    accumulate(cre.det, ket_b);
                }

            for (int p : occ_beta)
                for (int q = 0; q < n_act; ++q)
                {
                    if (ket_b & single_bit_mask(q))
                        continue;
                    auto ann = apply_annihilation(ket_b, p);
                    if (!ann.valid)
                        continue;
                    auto cre = apply_creation(ann.det, q);
                    if (!cre.valid)
                        continue;
                    accumulate(ket_a, cre.det);
                }

            for (std::size_t i = 0; i + 1 < occ_alpha.size(); ++i)
                for (std::size_t k = i + 1; k < occ_alpha.size(); ++k)
                    for (int q = 0; q < n_act; ++q)
                    {
                        if (ket_a & single_bit_mask(q))
                            continue;
                        for (int s = q + 1; s < n_act; ++s)
                        {
                            if (ket_a & single_bit_mask(s))
                                continue;
                            auto d = ket_a;
                            auto ann1 = apply_annihilation(d, occ_alpha[i]);
                            if (!ann1.valid)
                                continue;
                            auto ann2 = apply_annihilation(ann1.det, occ_alpha[k]);
                            if (!ann2.valid)
                                continue;
                            auto cre1 = apply_creation(ann2.det, q);
                            if (!cre1.valid)
                                continue;
                            auto cre2 = apply_creation(cre1.det, s);
                            if (!cre2.valid)
                                continue;
                            accumulate(cre2.det, ket_b);
                        }
                    }

            for (std::size_t i = 0; i + 1 < occ_beta.size(); ++i)
                for (std::size_t k = i + 1; k < occ_beta.size(); ++k)
                    for (int q = 0; q < n_act; ++q)
                    {
                        if (ket_b & single_bit_mask(q))
                            continue;
                        for (int s = q + 1; s < n_act; ++s)
                        {
                            if (ket_b & single_bit_mask(s))
                                continue;
                            auto d = ket_b;
                            auto ann1 = apply_annihilation(d, occ_beta[i]);
                            if (!ann1.valid)
                                continue;
                            auto ann2 = apply_annihilation(ann1.det, occ_beta[k]);
                            if (!ann2.valid)
                                continue;
                            auto cre1 = apply_creation(ann2.det, q);
                            if (!cre1.valid)
                                continue;
                            auto cre2 = apply_creation(cre1.det, s);
                            if (!cre2.valid)
                                continue;
                            accumulate(ket_a, cre2.det);
                        }
                    }

            for (int pa : occ_alpha)
                for (int pb : occ_beta)
                    for (int qa = 0; qa < n_act; ++qa)
                    {
                        if (ket_a & single_bit_mask(qa))
                            continue;
                        for (int qb = 0; qb < n_act; ++qb)
                        {
                            if (ket_b & single_bit_mask(qb))
                                continue;
                            auto d_a = ket_a;
                            auto d_b = ket_b;
                            auto ann_a = apply_annihilation(d_a, pa);
                            if (!ann_a.valid)
                                continue;
                            auto ann_b = apply_annihilation(d_b, pb);
                            if (!ann_b.valid)
                                continue;
                            auto cre_a = apply_creation(ann_a.det, qa);
                            if (!cre_a.valid)
                                continue;
                            auto cre_b = apply_creation(ann_b.det, qb);
                            if (!cre_b.valid)
                                continue;
                            accumulate(cre_a.det, cre_b.det);
                        }
                    }
        }

        return sigma;
    }

    std::size_t idx4(int p, int q, int r, int s, int n)
    {
        return static_cast<std::size_t>(((p * n + q) * n + r) * n + s);
    }

    Eigen::MatrixXd rotate_one_body_tensor(
        const Eigen::MatrixXd &h,
        const Eigen::MatrixXd &U)
    {
        return U.transpose() * h * U;
    }

    std::vector<double> rotate_two_body_tensor(
        const std::vector<double> &ga,
        const Eigen::MatrixXd &U)
    {
        const int n_act = static_cast<int>(U.rows());
        std::vector<double> out(ga.size(), 0.0);
        auto idx4_local = [n_act](int p, int q, int r, int s)
        {
            return static_cast<std::size_t>(((p * n_act + q) * n_act + r) * n_act + s);
        };

        for (int p = 0; p < n_act; ++p)
            for (int q = 0; q < n_act; ++q)
                for (int r = 0; r < n_act; ++r)
                    for (int s = 0; s < n_act; ++s)
                    {
                        double value = 0.0;
                        for (int a = 0; a < n_act; ++a)
                            for (int b = 0; b < n_act; ++b)
                                for (int c = 0; c < n_act; ++c)
                                    for (int d = 0; d < n_act; ++d)
                                        value +=
                                            U(a, p) * U(b, q) * U(c, r) * U(d, s) *
                                            ga[idx4_local(a, b, c, d)];
                        out[idx4_local(p, q, r, s)] = value;
                    }

        return out;
    }

} // namespace

int main()
{
    bool ok = true;

    {
        ok &= expect(single_bit_mask(-1) == 0,
                     "single_bit_mask should guard negative bit indices");
        ok &= expect(single_bit_mask(kCIStringBits) == 0,
                     "single_bit_mask should guard out-of-range bit indices");
        ok &= expect(low_bit_mask(kCIStringBits) == std::numeric_limits<CIString>::max(),
                     "low_bit_mask should saturate instead of performing an undefined full-width shift");
    }

    {
        RASParams ras{1, 1, 1, 1, 2, true};
        const CIString alpha = single_bit_mask(1) | single_bit_mask(2);
        const CIString beta = single_bit_mask(1) | single_bit_mask(2);

        ok &= expect(ras1_holes(alpha, beta, ras) == 2,
                     "combined RAS1 holes should count alpha and beta together");
        ok &= expect(!admissible_ras_pair(alpha, beta, ras),
                     "pair-level RAS filtering should reject determinants with too many total RAS1 holes");
    }

    {
        RASParams ras{1, 1, 1, 2, 1, true};
        const CIString alpha = single_bit_mask(0) | single_bit_mask(2);
        const CIString beta = single_bit_mask(1) | single_bit_mask(2);

        ok &= expect(ras3_electrons(alpha, beta, ras) == 2,
                     "combined RAS3 electrons should count alpha and beta together");
        ok &= expect(!admissible_ras_pair(alpha, beta, ras),
                     "pair-level RAS filtering should reject determinants with too many total RAS3 electrons");
    }

    {
        SymmetryContext sym;
        sym.names = {"B1", "A1", "B2", "A2"};
        sym.product = {
            {1, 0, 3, 2},
            {0, 1, 2, 3},
            {3, 2, 1, 0},
            {2, 3, 0, 1},
        };
        sym.abelian_1d_only = true;
        sym.totally_symmetric_irrep = 1;
        const std::vector<int> irr_act = {0, 2};

        ok &= expect(determinant_symmetry(single_bit_mask(0), single_bit_mask(1), irr_act, sym) == 3,
                     "determinant symmetry should use the explicit product table");
        ok &= expect(determinant_symmetry(single_bit_mask(0), single_bit_mask(0), irr_act, sym) == 1,
                     "occupying the same irrep in alpha and beta should multiply back to the totally symmetric irrep");
    }

    {
        const std::vector<double> puvw = {2.0};
        const std::vector<double> gamma = {3.0};
        const Eigen::MatrixXd q = contract_q_matrix(puvw, gamma, 1, 1);

        ok &= expect(q.rows() == 1 && q.cols() == 1 && std::abs(q(0, 0) - 6.0) < 1e-12,
                     "cached Q contraction should reproduce the expected scalar contraction");
    }

    {
        Eigen::MatrixXd c_old = Eigen::MatrixXd::Zero(3, 2);
        c_old(0, 0) = 1.0;
        c_old(1, 1) = 1.0;

        Eigen::MatrixXd c_new = Eigen::MatrixXd::Zero(3, 2);
        c_new(1, 0) = 1.0;
        c_new(0, 1) = -1.0;

        const Eigen::MatrixXd overlaps = compute_root_overlap(c_old, c_new);
        const std::vector<int> match = match_roots_by_max_overlap(overlaps);

        ok &= expect(overlaps.rows() == 2 && overlaps.cols() == 2,
                     "root overlap matrix should have one row/column per tracked root");
        ok &= expect(match.size() == 2 && match[0] == 1 && match[1] == 0,
                     "root matching should preserve state identity across swapped eigenpairs");
        ok &= expect(std::abs(overlaps(0, 1)) > 0.999 && std::abs(overlaps(1, 0)) > 0.999,
                     "root overlaps should be computed from CI-vector inner products");
    }

    {
        Eigen::MatrixXd overlaps(3, 4);
        overlaps << 10.0, 9.0, 8.0, 0.0,
            9.0, 1.0, 0.0, 0.0,
            8.0, 0.0, 7.0, 0.0;

        const std::vector<int> match = match_roots_by_max_overlap(overlaps);
        const double total =
            overlaps(0, match[0]) +
            overlaps(1, match[1]) +
            overlaps(2, match[2]);

        ok &= expect(match.size() == 3 && match[0] == 1 && match[1] == 0 && match[2] == 2,
                     "root matching should find the globally optimal overlap assignment instead of a greedy local maximum");
        ok &= expect(std::abs(total - 25.0) < 1e-12,
                     "root matching should maximize the total overlap across all tracked roots");
    }

    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
        H(1, 1) = 2.0;
        H(1, 2) = 0.2;
        H(2, 1) = 0.2;
        H(2, 2) = 4.0;

        Eigen::VectorXd c0 = Eigen::VectorXd::Zero(3);
        c0(0) = 1.0;
        const double E0 = 0.0;
        const Eigen::VectorXd H_diag = H.diagonal();
        Eigen::VectorXd sigma(3);
        sigma << 0.7, 0.3, -0.2;

        const CIResponseResult response =
            solve_ci_response_iterative(H, c0, E0, H_diag, sigma, 1e-12, 16, 1e-6);

        const Eigen::Vector2d rhs = -project_orthogonal(sigma, c0).tail<2>();
        const Eigen::Matrix2d subblock = H.bottomRightCorner<2, 2>();
        Eigen::VectorXd exact = Eigen::VectorXd::Zero(3);
        exact.tail<2>() = subblock.colPivHouseholderQr().solve(rhs);

        ok &= expect(response.converged,
                     "iterative CI response should converge on a small projected linear problem");
        ok &= expect(response.iterations > 0,
                     "iterative CI response should report at least one iteration when solving a nonzero response");
        ok &= expect(response.residual_norm < 1e-10,
                     "iterative CI response should drive the projected residual norm small");
        ok &= expect(std::abs(c0.dot(response.c1)) < 1e-12,
                     "iterative CI response should preserve the orthogonality gauge");
        ok &= expect((response.c1 - exact).norm() < 1e-10,
                     "iterative CI response should match the dense projected solution on a small test problem");
    }

    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
        H(1, 1) = 1.5;
        H(1, 2) = 0.25;
        H(2, 1) = 0.25;
        H(2, 2) = 3.0;

        Eigen::VectorXd c0 = Eigen::VectorXd::Zero(3);
        c0(0) = 1.0;
        const double E0 = 0.0;
        const Eigen::VectorXd H_diag = H.diagonal();
        Eigen::VectorXd sigma(3);
        sigma << 0.4, -0.2, 0.5;

        const CIResponseResult response =
            ci_response_diag_precond_single_step(H, c0, E0, H_diag, sigma, 1e-6);

        ok &= expect(std::isfinite(response.residual_norm),
                     "single-step CI response should report a finite residual norm for a well-posed problem");
        ok &= expect(response.iterations == 1,
                     "single-step CI response should report exactly one preconditioned step");
        ok &= expect(!response.converged,
                     "single-step CI response should not claim convergence just because the residual is finite");
        ok &= expect(std::abs(c0.dot(response.c1)) < 1e-12,
                     "single-step CI response should preserve the orthogonality gauge");
    }

    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 4);
        H(1, 1) = 1.0;
        H(1, 2) = 0.18;
        H(1, 3) = -0.04;
        H(2, 1) = 0.18;
        H(2, 2) = 2.1;
        H(2, 3) = 0.26;
        H(3, 1) = -0.04;
        H(3, 2) = 0.26;
        H(3, 3) = 3.4;

        Eigen::VectorXd c0 = Eigen::VectorXd::Zero(4);
        c0(0) = 1.0;
        const double E0 = 0.0;
        const Eigen::VectorXd H_diag = H.diagonal();
        Eigen::VectorXd sigma(4);
        sigma << 0.5, -0.3, 0.4, -0.2;

        const auto apply = [&H](const Eigen::VectorXd &c, Eigen::VectorXd &sigma_vec)
        {
            sigma_vec = H * c;
        };

        const CIResponseResult one_iter = solve_ci_response_davidson(
            apply, c0, E0, H_diag, sigma, 1e-16, 1, 1e-6, 1);
        const CIResponseResult two_iter = solve_ci_response_davidson(
            apply, c0, E0, H_diag, sigma, 1e-16, 2, 1e-6, 1);

        const Eigen::VectorXd rhs = -project_orthogonal(sigma, c0);
        const double residual_one_iter =
            (project_orthogonal(rhs - (H * one_iter.c1 - E0 * one_iter.c1), c0)).norm();
        const double residual_two_iter =
            (project_orthogonal(rhs - (H * two_iter.c1 - E0 * two_iter.c1), c0)).norm();

        ok &= expect(!one_iter.converged && !two_iter.converged,
                     "truncated Davidson CI response runs should report converged=false");
        ok &= expect(two_iter.residual_norm <= one_iter.residual_norm + 1e-12,
                     "bounded-subspace CI response should report the best residual seen, not a worse later iterate");
        ok &= expect(std::abs(two_iter.residual_norm - residual_two_iter) < 1e-12,
                     "reported residual should match the returned CI response vector");
        ok &= expect(std::abs(one_iter.residual_norm - residual_one_iter) < 1e-12,
                     "single-iteration CI response should report the residual of its returned vector");
    }

    {
        ok &= expect(std::string(response_rhs_mode_name(ResponseRHSMode::CommutatorOnlyApproximate)) ==
                         "commutator-only approximate RHS",
                     "response RHS mode names should make the approximate shortcut explicit");
        ok &= expect(std::string(response_rhs_mode_name(ResponseRHSMode::ExactActiveSpaceOrbitalDerivative)) ==
                         "exact active-space orbital derivative RHS",
                     "response RHS mode names should expose the exact orbital-derivative path");
    }

    {
        auto default_rhs_mode = [](bool debug_commutator_rhs)
        {
            return debug_commutator_rhs
                ? ResponseRHSMode::CommutatorOnlyApproximate
                : ResponseRHSMode::ExactActiveSpaceOrbitalDerivative;
        };

        ok &= expect(default_rhs_mode(false) == ResponseRHSMode::ExactActiveSpaceOrbitalDerivative,
                     "CASSCF should default to the exact orbital-derivative response RHS");
        ok &= expect(default_rhs_mode(true) == ResponseRHSMode::CommutatorOnlyApproximate,
                     "the commutator-only response RHS should remain available only through the explicit debug switch");
    }

    {
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(2, 1, 1, a_strs, b_strs);

        Eigen::MatrixXd h_eff = Eigen::MatrixXd::Zero(2, 2);
        std::vector<double> ga(16, 0.0);
        auto idx4_tiny = [](int p, int q, int r, int s)
        {
            return ((p * 2 + q) * 2 + r) * 2 + s;
        };
        ga[idx4_tiny(0, 0, 0, 0)] = 0.72;
        ga[idx4_tiny(1, 1, 1, 1)] = 0.41;
        ga[idx4_tiny(0, 0, 1, 1)] = ga[idx4_tiny(1, 1, 0, 0)] = 0.19;
        ga[idx4_tiny(0, 1, 1, 0)] = ga[idx4_tiny(1, 0, 0, 1)] = 0.07;

        RASParams ras;
        const CIDeterminantSpace space =
            build_ci_space(a_strs, b_strs, ras, h_eff, ga, 2, {}, nullptr, 0, 8);
        ok &= expect(space.dets.size() == 4,
                     "tiny active-space test should build a 4-determinant CI space");

        Eigen::VectorXd c0(4);
        c0 << 0.50, -0.30, 0.40, -0.20;

        Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(2, 2);
        kappa(0, 1) = 0.11;
        kappa(1, 0) = -0.11;

        const Eigen::MatrixXd F_I = Eigen::MatrixXd::Zero(2, 2);
        const Eigen::VectorXd rhs_approx = build_ci_response_rhs(
            ResponseRHSMode::CommutatorOnlyApproximate,
            kappa, F_I, h_eff, ga, space, a_strs, b_strs, c0, 0, 2);
        const Eigen::VectorXd rhs_exact = build_ci_response_rhs(
            ResponseRHSMode::ExactActiveSpaceOrbitalDerivative,
            kappa, F_I, h_eff, ga, space, a_strs, b_strs, c0, 0, 2);

        constexpr double eps = 1e-6;
        const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        const Eigen::MatrixXd U_plus = I + eps * kappa;
        const Eigen::MatrixXd U_minus = I - eps * kappa;

        const Eigen::MatrixXd h_plus = rotate_one_body_tensor(h_eff, U_plus);
        const Eigen::MatrixXd h_minus = rotate_one_body_tensor(h_eff, U_minus);
        const std::vector<double> ga_plus = rotate_two_body_tensor(ga, U_plus);
        const std::vector<double> ga_minus = rotate_two_body_tensor(ga, U_minus);
        const Eigen::MatrixXd H_plus =
            build_ci_hamiltonian_dense(a_strs, b_strs, space.dets, h_plus, ga_plus, 2);
        const Eigen::MatrixXd H_minus =
            build_ci_hamiltonian_dense(a_strs, b_strs, space.dets, h_minus, ga_minus, 2);
        const Eigen::VectorXd rhs_fd = ((H_plus - H_minus) * c0) / (2.0 * eps);

        ok &= expect(rhs_approx.norm() < 1e-12,
                     "commutator-only RHS should stay zero when the inactive Fock block is zero");
        ok &= expect((rhs_exact - rhs_fd).norm() < 1e-10,
                     "exact orbital-derivative RHS should match a finite-difference orbital rotation");
        ok &= expect((rhs_exact - rhs_approx).norm() > 1e-6,
                     "exact orbital-derivative RHS should differ from the commutator-only shortcut when two-electron terms respond");
    }

    {
        Eigen::MatrixXd gamma(2, 2);
        gamma << 1.2, 0.3,
            0.3, 0.8;
        const NaturalOrbitalData natural = diagonalize_natural_orbitals(gamma);
        const Eigen::MatrixXd rebuilt =
            natural.rotation * natural.occupations.asDiagonal() * natural.rotation.transpose();

        ok &= expect(natural.occupations(0) >= natural.occupations(1),
                     "natural occupations should be returned in descending order");
        ok &= expect(std::abs(natural.occupations.sum() - gamma.trace()) < 1e-12,
                     "natural occupations should preserve the active electron count");
        ok &= expect((rebuilt - gamma).norm() < 1e-12,
                     "natural-orbital rotation should diagonalize the 1-RDM without changing it on reconstruction");
    }

    {
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(3, 1, 1, a_strs, b_strs);

        Eigen::MatrixXd dh(3, 3);
        dh << 0.30, -0.11, 0.07,
            -0.11, -0.20, 0.13,
            0.07, 0.13, 0.50;

        std::vector<double> ga_zero(81, 0.0);
        RASParams ras;
        const std::vector<std::pair<int, int>> dets =
            build_ci_determinant_list(a_strs, b_strs, ras, {}, nullptr, 0);
        const Eigen::MatrixXd one_body_matrix =
            build_ci_hamiltonian_dense(a_strs, b_strs, dets, dh, ga_zero, 3);

        Eigen::VectorXd c = Eigen::VectorXd::LinSpaced(static_cast<int>(dets.size()), -0.4, 0.5);
        const Eigen::VectorXd sigma_dense = one_body_matrix * c;
        const Eigen::VectorXd sigma_direct =
            ci_sigma_1body(dh, c, a_strs, b_strs, dets, 3);

        ok &= expect((one_body_matrix - one_body_matrix.transpose()).norm() < 1e-12,
                     "one-body CI matrix built from Slater-Condon rules should be Hermitian for Hermitian operators");
        ok &= expect((sigma_direct - sigma_dense).norm() < 1e-12,
                     "ci_sigma_1body should match explicit one-body CI matrix application under the shared ket-to-bra convention");
    }

    {
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(3, 1, 1, a_strs, b_strs);

        Eigen::MatrixXd h_eff = Eigen::MatrixXd::Zero(3, 3);
        h_eff << -1.2, 0.1, 0.0,
            0.1, -0.4, 0.05,
            0.0, 0.05, 0.3;

        std::vector<double> ga(81, 0.0);
        auto idx4 = [](int p, int q, int r, int s)
        {
            return ((p * 3 + q) * 3 + r) * 3 + s;
        };
        ga[idx4(0, 0, 0, 0)] = 0.70;
        ga[idx4(1, 1, 1, 1)] = 0.55;
        ga[idx4(2, 2, 2, 2)] = 0.40;
        ga[idx4(0, 0, 1, 1)] = ga[idx4(1, 1, 0, 0)] = 0.18;
        ga[idx4(0, 0, 2, 2)] = ga[idx4(2, 2, 0, 0)] = 0.08;
        ga[idx4(1, 1, 2, 2)] = ga[idx4(2, 2, 1, 1)] = 0.12;
        ga[idx4(0, 1, 1, 0)] = ga[idx4(1, 0, 0, 1)] = 0.04;
        ga[idx4(0, 2, 2, 0)] = ga[idx4(2, 0, 0, 2)] = 0.02;
        ga[idx4(1, 2, 2, 1)] = ga[idx4(2, 1, 1, 2)] = 0.03;

        RASParams ras;
        const CIDeterminantSpace space =
            build_ci_space(a_strs, b_strs, ras, h_eff, ga, 3, {}, nullptr, 0, 0);
        const Eigen::MatrixXd dense_h =
            build_ci_hamiltonian_dense(a_strs, b_strs, space.dets, h_eff, ga, 3);
        const auto dense = solve_ci_dense(dense_h, 2);
        const CISolveResult direct =
            solve_ci(space, a_strs, b_strs, h_eff, ga, 3, 2, 1e-10, 0, 128);

        const Eigen::MatrixXd dense_proj =
            dense.second.leftCols(2) * dense.second.leftCols(2).adjoint();
        const Eigen::MatrixXd direct_proj =
            direct.vectors.leftCols(2) * direct.vectors.leftCols(2).adjoint();

        ok &= expect(direct.used_direct_sigma,
                     "matrix-free CI solve should report that it used the direct sigma-vector path");
        ok &= expect((direct.energies - dense.first).cwiseAbs().maxCoeff() < 1e-10,
                     "direct sigma-vector Davidson should reproduce dense CI eigenvalues on a small problem");
        ok &= expect((direct_proj - dense_proj).norm() < 1e-8,
                     "direct sigma-vector Davidson should recover the same low-root invariant subspace as dense CI");
    }

    {
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(4, 2, 2, a_strs, b_strs);

        Eigen::MatrixXd h_eff(4, 4);
        h_eff << -1.40, 0.08, -0.02, 0.01,
            0.08, -0.90, 0.05, -0.03,
            -0.02, 0.05, -0.45, 0.04,
            0.01, -0.03, 0.04, 0.15;

        std::vector<double> ga(256, 0.0);
        auto idx4 = [](int p, int q, int r, int s)
        {
            return ((p * 4 + q) * 4 + r) * 4 + s;
        };
        ga[idx4(0, 0, 0, 0)] = 0.72;
        ga[idx4(1, 1, 1, 1)] = 0.55;
        ga[idx4(2, 2, 2, 2)] = 0.42;
        ga[idx4(3, 3, 3, 3)] = 0.31;
        ga[idx4(0, 0, 1, 1)] = ga[idx4(1, 1, 0, 0)] = 0.20;
        ga[idx4(0, 0, 2, 2)] = ga[idx4(2, 2, 0, 0)] = 0.11;
        ga[idx4(1, 1, 3, 3)] = ga[idx4(3, 3, 1, 1)] = 0.09;
        ga[idx4(0, 1, 1, 0)] = ga[idx4(1, 0, 0, 1)] = 0.04;
        ga[idx4(0, 2, 2, 0)] = ga[idx4(2, 0, 0, 2)] = 0.03;
        ga[idx4(1, 3, 3, 1)] = ga[idx4(3, 1, 1, 3)] = 0.05;
        ga[idx4(0, 3, 3, 0)] = ga[idx4(3, 0, 0, 3)] = 0.02;
        ga[idx4(1, 2, 2, 1)] = ga[idx4(2, 1, 1, 2)] = 0.06;

        RASParams ras;
        const CIDeterminantSpace space =
            build_ci_space(a_strs, b_strs, ras, h_eff, ga, 4, {}, nullptr, 0, 0);
        const Eigen::MatrixXd dense_h =
            build_ci_hamiltonian_dense(a_strs, b_strs, space.dets, h_eff, ga, 4);

        Eigen::VectorXd c = Eigen::VectorXd::LinSpaced(static_cast<int>(space.dets.size()), -0.45, 0.35);
        Eigen::VectorXd sigma_dense = dense_h * c;
        Eigen::VectorXd sigma_direct = Eigen::VectorXd::Zero(static_cast<int>(space.dets.size()));
        apply_ci_hamiltonian(space, a_strs, b_strs, h_eff, ga, 4, c, sigma_direct);
        Eigen::VectorXd sigma_class = ci_sigma_excitation_class(space, a_strs, b_strs, h_eff, ga, 4, c);

        ok &= expect((sigma_direct - sigma_dense).norm() < 1e-12,
                     "direct CI sigma application should match dense Hamiltonian multiplication on a 4-orbital test space");
        ok &= expect((sigma_class - sigma_dense).norm() < 1e-12,
                     "excitation-class CI sigma builder should match dense Hamiltonian multiplication on the same test space");
        ok &= expect((sigma_class - sigma_direct).norm() < 1e-12,
                     "excitation-class CI sigma builder should reproduce the current direct CI sigma path exactly");
    }

    {
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(2, 1, 1, a_strs, b_strs);

        Eigen::MatrixXd h_eff = Eigen::MatrixXd::Zero(2, 2);
        h_eff << -0.8, 0.07,
            0.07, 0.1;
        std::vector<double> ga(16, 0.0);
        auto idx4 = [](int p, int q, int r, int s)
        {
            return ((p * 2 + q) * 2 + r) * 2 + s;
        };
        ga[idx4(0, 0, 0, 0)] = 0.60;
        ga[idx4(1, 1, 1, 1)] = 0.45;
        ga[idx4(0, 0, 1, 1)] = ga[idx4(1, 1, 0, 0)] = 0.16;
        ga[idx4(0, 1, 1, 0)] = ga[idx4(1, 0, 0, 1)] = 0.05;

        RASParams ras;
        const CIDeterminantSpace space =
            build_ci_space(a_strs, b_strs, ras, h_eff, ga, 2, {}, nullptr, 0, 8);
        const CISolveResult ci = solve_ci(space, a_strs, b_strs, h_eff, ga, 2, 2);
        Eigen::VectorXd weights(2);
        weights << 0.7, 0.3;

        const Eigen::MatrixXd gamma_ref =
            compute_1rdm_reference(ci.vectors, weights, a_strs, b_strs, space.dets, 2);
        const Eigen::MatrixXd gamma_opt =
            compute_1rdm(ci.vectors, weights, a_strs, b_strs, space.dets, 2);
        const std::vector<double> gamma2_ref =
            compute_2rdm_reference(ci.vectors, weights, a_strs, b_strs, space.dets, 2);
        const std::vector<double> gamma2_opt =
            compute_2rdm(ci.vectors, weights, a_strs, b_strs, space.dets, 2);
        const std::vector<double> bilinear_ref =
            compute_2rdm_bilinear_reference(ci.vectors, ci.vectors, weights, a_strs, b_strs, space.dets, 2);
        const std::vector<double> bilinear_opt =
            compute_2rdm_bilinear(ci.vectors, ci.vectors, weights, a_strs, b_strs, space.dets, 2);
        std::vector<double> gamma2_split(gamma2_opt.size(), 0.0);

        for (int root = 0; root < 2; ++root)
        {
            Eigen::MatrixXd ket_root(ci.vectors.rows(), 1);
            ket_root.col(0) = ci.vectors.col(root);
            Eigen::VectorXd unit_weight(1);
            unit_weight(0) = 1.0;

            const std::vector<double> gamma2_root =
                compute_2rdm(ket_root, unit_weight, a_strs, b_strs, space.dets, 2);
            for (std::size_t i = 0; i < gamma2_split.size(); ++i)
                gamma2_split[i] += weights(root) * gamma2_root[i];
        }

        double gamma2_err = 0.0;
        for (std::size_t i = 0; i < gamma2_ref.size(); ++i)
            gamma2_err = std::max(gamma2_err, std::abs(gamma2_ref[i] - gamma2_opt[i]));

        double gamma2_split_err = 0.0;
        for (std::size_t i = 0; i < gamma2_split.size(); ++i)
            gamma2_split_err = std::max(gamma2_split_err, std::abs(gamma2_split[i] - gamma2_opt[i]));

        double bilinear_err = 0.0;
        for (std::size_t i = 0; i < bilinear_ref.size(); ++i)
            bilinear_err = std::max(bilinear_err, std::abs(bilinear_ref[i] - bilinear_opt[i]));

        ok &= expect((gamma_ref - gamma_opt).norm() < 1e-12,
                     "optimized 1-RDM accumulation should reproduce the reference operator loop");
        ok &= expect(gamma2_err < 1e-12,
                     "optimized 2-RDM accumulation should reproduce the reference operator loop");
        ok &= expect(gamma2_split_err < 1e-12,
                     "weighted multi-root 2-RDMs should equal an explicit weighted sum of per-root 2-RDMs");
        ok &= expect(bilinear_err < 1e-12,
                     "optimized bilinear 2-RDM accumulation should reproduce the reference operator loop");
    }

    {
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(2, 1, 1, a_strs, b_strs);

        Eigen::MatrixXd h_eff = Eigen::MatrixXd::Zero(2, 2);
        h_eff << -0.8, 0.07,
            0.07, 0.1;
        std::vector<double> ga(16, 0.0);
        auto idx4 = [](int p, int q, int r, int s)
        {
            return ((p * 2 + q) * 2 + r) * 2 + s;
        };
        ga[idx4(0, 0, 0, 0)] = 0.60;
        ga[idx4(1, 1, 1, 1)] = 0.45;
        ga[idx4(0, 0, 1, 1)] = ga[idx4(1, 1, 0, 0)] = 0.16;
        ga[idx4(0, 1, 1, 0)] = ga[idx4(1, 0, 0, 1)] = 0.05;

        RASParams ras;
        const CIDeterminantSpace space =
            build_ci_space(a_strs, b_strs, ras, h_eff, ga, 2, {}, nullptr, 0, 8);
        const CISolveResult ci = solve_ci(space, a_strs, b_strs, h_eff, ga, 2, 2);

        Eigen::VectorXd weights(2);
        weights << 0.7, 0.3;

        Eigen::MatrixXd bra_vecs = ci.vectors;
        bra_vecs.col(0) = 0.8 * ci.vectors.col(0) + 0.2 * ci.vectors.col(1);
        bra_vecs.col(1) = -0.3 * ci.vectors.col(0) + 0.7 * ci.vectors.col(1);

        const Eigen::MatrixXd gamma_weighted =
            compute_1rdm(ci.vectors, weights, a_strs, b_strs, space.dets, 2);
        const std::vector<double> bilinear_weighted =
            compute_2rdm_bilinear(bra_vecs, ci.vectors, weights, a_strs, b_strs, space.dets, 2);

        Eigen::MatrixXd gamma_sum = Eigen::MatrixXd::Zero(2, 2);
        std::vector<double> bilinear_sum(bilinear_weighted.size(), 0.0);
        for (int root = 0; root < 2; ++root)
        {
            Eigen::MatrixXd ket_root(ci.vectors.rows(), 1);
            ket_root.col(0) = ci.vectors.col(root);
            Eigen::MatrixXd bra_root(bra_vecs.rows(), 1);
            bra_root.col(0) = bra_vecs.col(root);
            Eigen::VectorXd unit_weight(1);
            unit_weight(0) = 1.0;

            gamma_sum += weights(root) *
                         compute_1rdm(ket_root, unit_weight, a_strs, b_strs, space.dets, 2);

            const std::vector<double> bilinear_root =
                compute_2rdm_bilinear(bra_root, ket_root, unit_weight, a_strs, b_strs, space.dets, 2);
            for (std::size_t i = 0; i < bilinear_sum.size(); ++i)
                bilinear_sum[i] += weights(root) * bilinear_root[i];
        }

        double bilinear_split_err = 0.0;
        for (std::size_t i = 0; i < bilinear_sum.size(); ++i)
            bilinear_split_err = std::max(bilinear_split_err, std::abs(bilinear_sum[i] - bilinear_weighted[i]));

        ok &= expect((gamma_sum - gamma_weighted).norm() < 1e-12,
                     "weighted multi-root 1-RDMs should equal an explicit weighted sum of per-root 1-RDMs");
        ok &= expect(bilinear_split_err < 1e-12,
                     "weighted multi-root bilinear 2-RDMs should equal an explicit weighted sum of per-root bilinear contributions");
    }

    {
        std::vector<CIString> a_strs;
        std::vector<CIString> b_strs;
        build_spin_strings_unfiltered(2, 1, 1, a_strs, b_strs);

        Eigen::MatrixXd h_eff = Eigen::MatrixXd::Zero(2, 2);
        h_eff << -0.70, 0.08,
            0.08, 0.05;

        std::vector<double> ga(16, 0.0);
        auto idx4 = [](int p, int q, int r, int s)
        {
            return ((p * 2 + q) * 2 + r) * 2 + s;
        };
        ga[idx4(0, 0, 0, 0)] = 0.65;
        ga[idx4(1, 1, 1, 1)] = 0.40;
        ga[idx4(0, 0, 1, 1)] = ga[idx4(1, 1, 0, 0)] = 0.14;
        ga[idx4(0, 1, 1, 0)] = ga[idx4(1, 0, 0, 1)] = 0.06;

        RASParams ras;
        const CIDeterminantSpace space =
            build_ci_space(a_strs, b_strs, ras, h_eff, ga, 2, {}, nullptr, 0, 8);
        const CISolveResult ci = solve_ci(space, a_strs, b_strs, h_eff, ga, 2, 2);
        const bool have_two_roots = (ci.vectors.cols() == 2);
        ok &= expect(have_two_roots,
                     "test problem should produce two CI roots for per-root orbital-intermediate reduction checks");
        if (!have_two_roots)
            return 1;

        Eigen::VectorXd weights(2);
        weights << 0.7, 0.3;

        auto linear_fock_like = [](const Eigen::MatrixXd &gamma)
        {
            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(2, 2);
            F(0, 0) = 0.30 * gamma(0, 0) + 0.20 * gamma(0, 1);
            F(0, 1) = -0.15 * gamma(0, 0) + 0.45 * gamma(1, 0);
            F(1, 0) = 0.10 * gamma(0, 1) - 0.25 * gamma(1, 1);
            F(1, 1) = 0.35 * gamma(1, 1) + 0.05 * gamma(1, 0);
            return F;
        };

        auto linear_q_like = [idx4](const std::vector<double> &Gamma)
        {
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2, 2);
            Q(0, 0) = 0.40 * Gamma[idx4(0, 0, 0, 0)] + 0.10 * Gamma[idx4(0, 1, 0, 1)];
            Q(0, 1) = -0.20 * Gamma[idx4(0, 0, 1, 0)] + 0.30 * Gamma[idx4(1, 0, 0, 1)];
            Q(1, 0) = 0.15 * Gamma[idx4(1, 1, 0, 0)] - 0.05 * Gamma[idx4(0, 1, 1, 0)];
            Q(1, 1) = 0.25 * Gamma[idx4(1, 0, 1, 0)] + 0.12 * Gamma[idx4(1, 1, 1, 1)];
            return Q;
        };

        auto linear_grad_like =
            [](const Eigen::MatrixXd &F_A, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &gamma)
        {
            Eigen::MatrixXd G = F_A + Q;
            G(0, 1) += 0.80 * gamma(0, 1);
            G(1, 0) -= 0.80 * gamma(1, 0);
            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(2, 2);
            grad(0, 0) = 0.0;
            grad(1, 1) = 0.0;
            grad(0, 1) = 2.0 * (G(0, 1) - G(1, 0));
            grad(1, 0) = -grad(0, 1);
            return grad;
        };

        Eigen::MatrixXd gamma_avg = Eigen::MatrixXd::Zero(2, 2);
        std::vector<double> Gamma_avg(16, 0.0);
        Eigen::MatrixXd F_A_avg = Eigen::MatrixXd::Zero(2, 2);
        Eigen::MatrixXd Q_avg = Eigen::MatrixXd::Zero(2, 2);
        Eigen::MatrixXd g_avg = Eigen::MatrixXd::Zero(2, 2);

        for (int root = 0; root < 2; ++root)
        {
            Eigen::MatrixXd root_vec(ci.vectors.rows(), 1);
            root_vec.col(0) = ci.vectors.col(root);
            Eigen::VectorXd unit_weight(1);
            unit_weight(0) = 1.0;

            const Eigen::MatrixXd gamma_root =
                compute_1rdm(root_vec, unit_weight, a_strs, b_strs, space.dets, 2);
            const std::vector<double> Gamma_root =
                compute_2rdm(root_vec, unit_weight, a_strs, b_strs, space.dets, 2);
            const Eigen::MatrixXd F_A_root = linear_fock_like(gamma_root);
            const Eigen::MatrixXd Q_root = linear_q_like(Gamma_root);
            const Eigen::MatrixXd g_root = linear_grad_like(F_A_root, Q_root, gamma_root);

            const double w = weights(root);
            gamma_avg += w * gamma_root;
            F_A_avg += w * F_A_root;
            Q_avg += w * Q_root;
            g_avg += w * g_root;
            for (std::size_t i = 0; i < Gamma_avg.size(); ++i)
                Gamma_avg[i] += w * Gamma_root[i];
        }

        const Eigen::MatrixXd F_A_from_avg = linear_fock_like(gamma_avg);
        const Eigen::MatrixXd Q_from_avg = linear_q_like(Gamma_avg);
        const Eigen::MatrixXd g_from_avg = linear_grad_like(F_A_from_avg, Q_from_avg, gamma_avg);
        ok &= expect((F_A_avg - F_A_from_avg).norm() < 1e-12,
                     "weighted per-root active Fock-like intermediates should match the averaged-1-RDM path");
        ok &= expect((Q_avg - Q_from_avg).norm() < 1e-12,
                     "weighted per-root Q-like intermediates should match the averaged-2-RDM path");
        ok &= expect((g_avg - g_from_avg).norm() < 1e-12,
                     "weighted per-root orbital-gradient-like intermediates should match the averaged reduced-density path when the maps are linear");
    }

    return ok ? 0 : 1;
}
