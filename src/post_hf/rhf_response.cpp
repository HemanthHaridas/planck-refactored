#include "rhf_response.h"

#include <cstdlib>
#include <format>
#include <iostream>
#include <iomanip>
#include <string_view>

#include "post_hf/integrals.h"

namespace HartreeFock::Correlation
{
    namespace
    {
        void maybe_print_rhf_response_matrix(const char *name, const Eigen::MatrixXd &mat)
        {
            const char *enabled = std::getenv("PLANCK_DEBUG_RHF_RESPONSE");
            if (enabled == nullptr || std::string_view(enabled) != "1")
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

    std::expected<Eigen::MatrixXd, std::string> build_rhf_cphf_matrix(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &mo_coeff,
        const Eigen::VectorXd &mo_energy)
    {
        if (!calculator._info._is_converged)
            return std::unexpected("build_rhf_cphf_matrix: SCF not converged.");
        if (calculator._scf._scf != HartreeFock::SCFType::RHF || calculator._info._scf.is_uhf)
            return std::unexpected("build_rhf_cphf_matrix: RHF reference required.");

        const int nb = static_cast<int>(calculator._shells.nbasis());
        const Eigen::MatrixXd &C = mo_coeff;
        const Eigen::VectorXd &eps = mo_energy;

        int n_electrons = 0;
        for (auto Z : calculator._molecule.atomic_numbers)
            n_electrons += static_cast<int>(Z);
        n_electrons -= calculator._molecule.charge;
        if (n_electrons % 2 != 0)
            return std::unexpected("build_rhf_cphf_matrix: closed-shell RHF reference required.");

        const int n_occ = n_electrons / 2;
        const int n_virt = nb - n_occ;

        auto idx_ai = [n_occ](int a, int i) -> int
        {
            return a * n_occ + i;
        };
        auto idx_eri = [nb](int p, int q, int r, int s) -> std::size_t
        {
            return ((static_cast<std::size_t>(p) * nb + q) * nb + r) * nb + s;
        };

        std::vector<double> eri_local;
        const std::vector<double> &eri_ao = ensure_eri(
            calculator, shell_pairs, eri_local, "RHF CPHF :");
        const std::vector<double> eri_mo = transform_eri(
            eri_ao,
            static_cast<std::size_t>(nb),
            C,
            C,
            C,
            C);

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_virt * n_occ, n_virt * n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
            {
                const int ai = idx_ai(a, i);
                const int aa = n_occ + a;
                A(ai, ai) += eps(aa) - eps(i);

                for (int b = 0; b < n_virt; ++b)
                    for (int j = 0; j < n_occ; ++j)
                    {
                        const int bb = n_occ + b;
                        const double ai_jb = eri_mo[idx_eri(aa, i, j, bb)];
                        const double ab_ji = eri_mo[idx_eri(aa, bb, j, i)];
                        const double aj_bi = eri_mo[idx_eri(aa, j, bb, i)];
                        // Standard RHF CPHF coupling: A_{ai,bj} = (e_a-e_i)d + [4(ai|jb)-(ab|ji)-(aj|bi)].
                        // The coupling adds (matching PySCF cphf.solve's fvind operator).
                        // solve_rhf_cphf solves A z = -rhs; the overall sign on z is chosen
                        // so the resulting vo block is phase-consistent with the doo/dvv
                        // blocks of the MP2 relaxed density (see mp2_gradient.cpp).
                        A(idx_ai(b, j), ai) += 4.0 * ai_jb - ab_ji - aj_bi;
                    }
            }
        maybe_print_rhf_response_matrix("A", A);
        return A;
    }

    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &mo_coeff,
        const Eigen::VectorXd &mo_energy,
        const Eigen::MatrixXd &rhs)
    {
        auto A_res = build_rhf_cphf_matrix(calculator, shell_pairs, mo_coeff, mo_energy);
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
                rhs_vec(a * n_occ + i) = -rhs(a, i);

        maybe_print_rhf_response_matrix("rhs", rhs);
        maybe_print_rhf_response_matrix("rhs_vec", Eigen::Map<const Eigen::MatrixXd>(rhs_vec.data(), rhs_vec.size(), 1));

        const Eigen::VectorXd sol = A_res->colPivHouseholderQr().solve(rhs_vec);
        Eigen::MatrixXd z = Eigen::MatrixXd::Zero(n_virt, n_occ);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
                z(a, i) = sol(a * n_occ + i);
        maybe_print_rhf_response_matrix("z", z);
        return z;
    }
} // namespace HartreeFock::Correlation
