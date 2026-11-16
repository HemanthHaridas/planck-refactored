#include "rhf_response.h"

#include <cstdlib>
#include <format>
#include <iostream>
#include <iomanip>
#include <string_view>

#include "dft/response_gmres.h"
#include "post_hf/orbital_hessian.h"

namespace HartreeFock::Correlation
{
    namespace
    {
        bool debug_rhf_response()
        {
            const char *enabled = std::getenv("PLANCK_DEBUG_RHF_RESPONSE");
            return enabled != nullptr && std::string_view(enabled) == "1";
        }

        void maybe_print_rhf_response_matrix(const char *name, const Eigen::MatrixXd &mat)
        {
            if (!debug_rhf_response())
                return;

            std::cout << "PLANCK_RHF_RESPONSE " << name << " " << mat.rows() << " " << mat.cols() << "\n";
            std::cout << std::setprecision(16);
            for (Eigen::Index row = 0; row < mat.rows(); ++row)
                for (Eigen::Index col = 0; col < mat.cols(); ++col)
                    std::cout << "PLANCK_RHF_RESPONSE_ELEM "
                              << name << " "
                              << row << " "
                              << col << " "
                              << mat(row, col) << "\n";
        }
    }

    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &mo_coeff,
        const Eigen::VectorXd &mo_energy,
        const Eigen::MatrixXd &rhs)
    {
        if (calculator._scf._scf != HartreeFock::SCFType::RHF || calculator._info._scf.is_uhf)
            return std::unexpected("solve_rhf_cphf: RHF reference required.");
        const int n_electrons = calculator._molecule.total_nuclear_charge() - calculator._molecule.charge;
        if (n_electrons % 2 != 0)
            return std::unexpected("solve_rhf_cphf: closed-shell RHF reference required.");
        const int n_occ = n_electrons / 2;
        const int n_virt = static_cast<int>(mo_coeff.cols()) - n_occ;

        if (rhs.rows() != n_virt || rhs.cols() != n_occ)
        {
            return std::unexpected(std::format(
                "solve_rhf_cphf: RHS shape mismatch; expected {}x{}, got {}x{}.",
                n_virt, n_occ, rhs.rows(), rhs.cols()));
        }

        // The orbital Hessian of the SCF's own operator (exact, or fitted under
        // RI-JK), applied matrix-free: docs/SCF_ORBITAL_HESSIAN_OPERATOR.md.
        const OrbitalHessian hessian = OrbitalHessian::rhf(calculator, shell_pairs, mo_coeff, mo_energy, n_occ);
        if (debug_rhf_response())
            maybe_print_rhf_response_matrix("A", hessian.materialize());

        // MP2 gradients use the conventional A z = -rhs form; reshaping back to
        // a virtual-by-occupied matrix keeps the result aligned with the rest of
        // the relaxed-density code.
        Eigen::VectorXd rhs_vec(n_virt * n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
                rhs_vec(a * n_occ + i) = -rhs(a, i);
        maybe_print_rhf_response_matrix("rhs", rhs);
        maybe_print_rhf_response_matrix("rhs_vec", Eigen::Map<const Eigen::MatrixXd>(rhs_vec.data(), rhs_vec.size(), 1));

        // Same solver and tolerance as solve_uhf_cphf.
        const DFT::Response::Action action =
            [&hessian](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
        { return hessian(x); };
        auto solve = DFT::Response::gmres(action, rhs_vec, hessian.diagonal(), 1e-10);
        if (!solve)
            return std::unexpected("solve_rhf_cphf: " + solve.error());
        if (!solve->converged)
            return std::unexpected(std::format(
                "solve_rhf_cphf: GMRES did not converge (iterations={}, actions={}, residual={:.3e}).",
                solve->iterations, solve->action_count, solve->true_residuals.back()));

        Eigen::MatrixXd z = Eigen::MatrixXd::Zero(n_virt, n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
                z(a, i) = solve->x(a * n_occ + i);
        maybe_print_rhf_response_matrix("z", z);
        return z;
    }
} // namespace HartreeFock::Correlation
