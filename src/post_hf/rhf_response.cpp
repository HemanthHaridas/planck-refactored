#include "rhf_response.h"

#include <format>

#include "post_hf/integrals.h"

namespace HartreeFock::Correlation
{

    std::expected<Eigen::MatrixXd, std::string> build_rhf_cphf_matrix(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs)
    {
        // Reject anything except a converged closed-shell RHF reference; the
        // response matrix below is only defined in that setting.
        if (!calculator._info._is_converged)
            return std::unexpected("build_rhf_cphf_matrix: SCF not converged.");
        if (calculator._scf._scf != HartreeFock::SCFType::RHF ||
            calculator._info._scf.is_uhf)
            return std::unexpected("build_rhf_cphf_matrix: RHF reference required.");

        const std::size_t nb = calculator._shells.nbasis();
        const Eigen::MatrixXd& C   = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::VectorXd& eps = calculator._info._scf.alpha.mo_energies;

        int n_electrons = 0;
        for (auto Z : calculator._molecule.atomic_numbers)
            n_electrons += static_cast<int>(Z);
        n_electrons -= calculator._molecule.charge;
        if (n_electrons % 2 != 0)
            return std::unexpected("build_rhf_cphf_matrix: closed-shell RHF reference required.");

        const int n_occ  = n_electrons / 2;
        const int n_virt = static_cast<int>(nb) - n_occ;
        if (n_occ <= 0 || n_virt <= 0)
            return std::unexpected("build_rhf_cphf_matrix: no occupied or virtual orbitals.");

        std::vector<double> eri_local;
        const std::vector<double>& eri = ensure_eri(
            calculator, shell_pairs, eri_local, "RHF Response :");

        const Eigen::MatrixXd C_occ  = C.leftCols(n_occ);
        const Eigen::MatrixXd C_virt = C.middleCols(n_occ, n_virt);

        // (a i | b j)
        const auto mo_ai_bj = transform_eri(eri, nb, C_virt, C_occ, C_virt, C_occ);
        // (a b | i j)
        const auto mo_ab_ij = transform_eri(eri, nb, C_virt, C_virt, C_occ, C_occ);

        const int nov = n_occ * n_virt;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nov, nov);

        // Flatten the `(a,i)` response space in row-major order so the dense
        // matrix lines up with the vectorized solver interface.
        auto idx_ai = [n_occ](int a, int i) -> int {
            return a * n_occ + i;
        };
        auto idx_ai_bj = [n_occ, n_virt](int a, int i, int b, int j) -> std::size_t {
            return ((static_cast<std::size_t>(a) * n_occ + i) * n_virt + b) * n_occ + j;
        };
        auto idx_ab_ij = [n_virt, n_occ](int a, int b, int i, int j) -> std::size_t {
            return ((static_cast<std::size_t>(a) * n_virt + b) * n_occ + i) * n_occ + j;
        };

        for (int a = 0; a < n_virt; ++a)
        for (int i = 0; i < n_occ;  ++i)
        {
            const int ai = idx_ai(a, i);
            for (int b = 0; b < n_virt; ++b)
            for (int j = 0; j < n_occ;  ++j)
            {
                const int bj = idx_ai(b, j);
                double val = 0.0;
                if (a == b && i == j)
                    val += eps(n_occ + a) - eps(i);

                // RHF response is the orbital-energy gap plus the usual three
                // two-electron couplings from the CPHF/Z-vector equations.
                const double ai_bj = mo_ai_bj[idx_ai_bj(a, i, b, j)];
                const double ab_ij = mo_ab_ij[idx_ab_ij(a, b, i, j)];
                const double aj_bi = mo_ai_bj[idx_ai_bj(a, j, b, i)];

                val += 4.0 * ai_bj - ab_ij - aj_bi;
                A(ai, bj) = val;
            }
        }

        return A;
    }

    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs,
        const Eigen::MatrixXd& rhs)
    {
        // Build the response matrix once, then solve in the flattened
        // occupied-virtual space and reshape back into matrix form.
        auto A_res = build_rhf_cphf_matrix(calculator, shell_pairs);
        if (!A_res) return std::unexpected(A_res.error());

        const Eigen::MatrixXd& A = *A_res;

        int n_electrons = 0;
        for (auto Z : calculator._molecule.atomic_numbers)
            n_electrons += static_cast<int>(Z);
        n_electrons -= calculator._molecule.charge;

        const int n_occ  = n_electrons / 2;
        const int n_virt = static_cast<int>(calculator._shells.nbasis()) - n_occ;
        if (rhs.rows() != n_virt || rhs.cols() != n_occ)
        {
            return std::unexpected(std::format(
                "solve_rhf_cphf: RHS shape mismatch; expected {}x{}, got {}x{}.",
                n_virt, n_occ, rhs.rows(), rhs.cols()));
        }

        // Eigen's dense solver expects a vector, so map the 2-D response field
        // into the same `(a,i)` ordering used by the matrix builder.
        Eigen::VectorXd rhs_vec(Eigen::Map<const Eigen::VectorXd>(rhs.data(), rhs.size()));
        Eigen::VectorXd z_vec = A.colPivHouseholderQr().solve(rhs_vec);

        // Restore the natural virtual-by-occupied layout for the caller.
        Eigen::MatrixXd z = Eigen::Map<const Eigen::MatrixXd>(z_vec.data(), n_virt, n_occ);
        return z;
    }

} // namespace HartreeFock::Correlation
