#include "rhf_response.h"

#include <format>

#include "integrals/base.h"

namespace HartreeFock::Correlation
{
    std::expected<Eigen::MatrixXd, std::string> build_rhf_cphf_matrix(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (!calculator._info._is_converged)
            return std::unexpected("build_rhf_cphf_matrix: SCF not converged.");
        if (calculator._scf._scf != HartreeFock::SCFType::RHF || calculator._info._scf.is_uhf)
            return std::unexpected("build_rhf_cphf_matrix: RHF reference required.");

        const int nb = static_cast<int>(calculator._shells.nbasis());
        const Eigen::MatrixXd &C = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::VectorXd &eps = calculator._info._scf.alpha.mo_energies;

        int n_electrons = 0;
        for (auto Z : calculator._molecule.atomic_numbers)
            n_electrons += static_cast<int>(Z);
        n_electrons -= calculator._molecule.charge;
        if (n_electrons % 2 != 0)
            return std::unexpected("build_rhf_cphf_matrix: closed-shell RHF reference required.");

        const int n_occ = n_electrons / 2;
        const int n_virt = nb - n_occ;
        const Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        const Eigen::MatrixXd C_virt = C.middleCols(n_occ, n_virt);

        auto idx_ai = [n_occ](int a, int i) -> int
        {
            return a * n_occ + i;
        };

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_virt * n_occ, n_virt * n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
            {
                const int ai = idx_ai(a, i);
                A(ai, ai) = eps(n_occ + a) - eps(i);

                Eigen::MatrixXd trial = Eigen::MatrixXd::Zero(n_virt, n_occ);
                trial(a, i) = 1.0;
                const Eigen::MatrixXd dm1 = C_virt * trial * C_occ.transpose();
                const Eigen::MatrixXd veff = _compute_2e_fock(
                    shell_pairs,
                    dm1 + dm1.transpose(),
                    calculator._shells.nbasis(),
                    calculator._integral._engine,
                    HartreeFock::ERIKernel::Coulomb,
                    0.0,
                    calculator._integral._tol_eri,
                    calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
                const Eigen::MatrixXd vvo = 2.0 * (C_virt.transpose() * veff * C_occ);

                for (int b = 0; b < n_virt; ++b)
                    for (int j = 0; j < n_occ; ++j)
                        A(idx_ai(b, j), ai) += vvo(b, j);
            }
        return A;
    }

    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &rhs)
    {
        auto A_res = build_rhf_cphf_matrix(calculator, shell_pairs);
        if (!A_res)
            return std::unexpected(A_res.error());

        int n_electrons = 0;
        for (auto Z : calculator._molecule.atomic_numbers)
            n_electrons += static_cast<int>(Z);
        n_electrons -= calculator._molecule.charge;
        const int n_occ = n_electrons / 2;
        const int n_virt = static_cast<int>(calculator._shells.nbasis()) - n_occ;

        if (rhs.rows() != n_virt || rhs.cols() != n_occ)
        {
            return std::unexpected(std::format(
                "solve_rhf_cphf: RHS shape mismatch; expected {}x{}, got {}x{}.",
                n_virt, n_occ, rhs.rows(), rhs.cols()));
        }

        Eigen::VectorXd rhs_vec(n_virt * n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
                rhs_vec(a * n_occ + i) = rhs(a, i);

        const Eigen::VectorXd sol = A_res->colPivHouseholderQr().solve(rhs_vec);
        Eigen::MatrixXd z = Eigen::MatrixXd::Zero(n_virt, n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
                z(a, i) = sol(a * n_occ + i);
        return z;
    }
} // namespace HartreeFock::Correlation
