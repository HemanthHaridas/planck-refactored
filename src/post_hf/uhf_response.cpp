#include "uhf_response.h"

#include <format>

#include "integrals/base.h"
#include "post_hf/ri/ri_eri.h"

namespace HartreeFock::Correlation
{
    // Coupled alpha/beta orbital-Hessian matrix (docs/SOSCF_UHF_DFT_SCOPE.md,
    // U1). Split out of solve_uhf_cphf so SOSCF can call it directly with the
    // CURRENT (not yet converged) MO coefficients/energies, mirroring
    // build_rhf_cphf_matrix's own convergence-guard relaxation. Its one
    // pre-SOSCF caller (solve_uhf_cphf, used by the UHF MP2 gradient) always
    // runs post-convergence anyway, so dropping the guard here is
    // behavior-neutral for it.
    std::expected<Eigen::MatrixXd, std::string> build_uhf_cphf_matrix(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &coeff_alpha,
        const Eigen::MatrixXd &coeff_beta,
        const Eigen::VectorXd &energy_alpha,
        const Eigen::VectorXd &energy_beta,
        int nocc_alpha,
        int nocc_beta)
    {
        if (calculator._scf._scf != HartreeFock::SCFType::UHF || !calculator._info._scf.is_uhf)
            return std::unexpected("build_uhf_cphf_matrix: UHF reference required.");

        const int nb = static_cast<int>(calculator._shells.nbasis());
        const int nva = static_cast<int>(coeff_alpha.cols()) - nocc_alpha;
        const int nvb = static_cast<int>(coeff_beta.cols()) - nocc_beta;
        if (nocc_alpha <= 0 || nocc_beta < 0 || nva <= 0 || nvb <= 0)
            return std::unexpected("build_uhf_cphf_matrix: invalid occupied/virtual dimensions.");

        const Eigen::MatrixXd Ca_occ = coeff_alpha.leftCols(nocc_alpha);
        const Eigen::MatrixXd Ca_virt = coeff_alpha.middleCols(nocc_alpha, nva);
        const Eigen::MatrixXd Cb_occ = coeff_beta.leftCols(nocc_beta);
        const Eigen::MatrixXd Cb_virt = coeff_beta.middleCols(nocc_beta, nvb);

        const int nova = nva * nocc_alpha;
        const int novb = nvb * nocc_beta;
        // The unrestricted response problem is one coupled alpha/beta linear
        // system. We lay it out as a single dense matrix so the block structure
        // stays visible in a debugger or when cross-checking with reference code.
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nova + novb, nova + novb);

        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < nocc_alpha; ++i)
                A(a * nocc_alpha + i, a * nocc_alpha + i) =
                    energy_alpha(nocc_alpha + a) - energy_alpha(i);
        for (int a = 0; a < nvb; ++a)
            for (int i = 0; i < nocc_beta; ++i)
                A(nova + a * nocc_beta + i, nova + a * nocc_beta + i) =
                    energy_beta(nocc_beta + a) - energy_beta(i);

        for (int col = 0; col < nova + novb; ++col)
        {
            Eigen::MatrixXd xa = Eigen::MatrixXd::Zero(nva, nocc_alpha);
            Eigen::MatrixXd xb = Eigen::MatrixXd::Zero(nvb, nocc_beta);
            if (col < nova)
                xa(col / nocc_alpha, col % nocc_alpha) = 1.0;
            else
            {
                const int local = col - nova;
                xb(local / nocc_beta, local % nocc_beta) = 1.0;
            }

            // Apply one trial orbital rotation, form the AO density response,
            // and ask the integral layer for the induced Coulomb/exchange
            // response. This mirrors the matrix-free view of CPHF, but here we
            // materialize the full Jacobian column by column for transparency.
            const Eigen::MatrixXd dm1a = Ca_virt * xa * Ca_occ.transpose();
            const Eigen::MatrixXd dm1b = Cb_virt * xb * Cb_occ.transpose();
            const Eigen::MatrixXd dm1a_sym = dm1a + dm1a.transpose();
            const Eigen::MatrixXd dm1b_sym = dm1b + dm1b.transpose();
            // RI-consistent CPHF operator under MP2 RI (Step RG4.2): the induced
            // Coulomb/exchange response is the same {J(Pa+Pb) - K(P_sigma)}
            // quantity, built from the 3-center factors instead of the dense ERI.
            const auto [va_ao, vb_ao] =
                calculator._mp2.use_ri
                    ? RI::build_ri_fock_uhf(calculator, dm1a_sym, dm1b_sym)
                    : _compute_2e_fock_uhf(
                          shell_pairs,
                          dm1a_sym,
                          dm1b_sym,
                          static_cast<std::size_t>(nb),
                          calculator._integral._engine,
                          HartreeFock::ERIKernel::Coulomb,
                          0.0,
                          calculator._integral._tol_eri,
                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

            const Eigen::MatrixXd va = Ca_virt.transpose() * va_ao * Ca_occ;
            const Eigen::MatrixXd vb = Cb_virt.transpose() * vb_ao * Cb_occ;
            for (int a = 0; a < nva; ++a)
                for (int i = 0; i < nocc_alpha; ++i)
                    A(a * nocc_alpha + i, col) += va(a, i);
            for (int a = 0; a < nvb; ++a)
                for (int i = 0; i < nocc_beta; ++i)
                    A(nova + a * nocc_beta + i, col) += vb(a, i);
        }

        return A;
    }

    std::expected<UHFCphfSolution, std::string> solve_uhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &coeff_alpha,
        const Eigen::MatrixXd &coeff_beta,
        const Eigen::VectorXd &energy_alpha,
        const Eigen::VectorXd &energy_beta,
        int nocc_alpha,
        int nocc_beta,
        const Eigen::MatrixXd &rhs_alpha,
        const Eigen::MatrixXd &rhs_beta)
    {
        if (!calculator._info._is_converged)
            return std::unexpected("solve_uhf_cphf: SCF not converged.");

        auto A_res = build_uhf_cphf_matrix(
            calculator, shell_pairs, coeff_alpha, coeff_beta,
            energy_alpha, energy_beta, nocc_alpha, nocc_beta);
        if (!A_res)
            return std::unexpected(A_res.error());
        const Eigen::MatrixXd &A = *A_res;

        const int nva = static_cast<int>(coeff_alpha.cols()) - nocc_alpha;
        const int nvb = static_cast<int>(coeff_beta.cols()) - nocc_beta;
        const int nova = nva * nocc_alpha;
        const int novb = nvb * nocc_beta;

        Eigen::VectorXd rhs(nova + novb);
        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < nocc_alpha; ++i)
                rhs(a * nocc_alpha + i) = -rhs_alpha(a, i);
        for (int a = 0; a < nvb; ++a)
            for (int i = 0; i < nocc_beta; ++i)
                rhs(nova + a * nocc_beta + i) = -rhs_beta(a, i);

        const Eigen::VectorXd sol = A.colPivHouseholderQr().solve(rhs);
        UHFCphfSolution out;
        out.alpha = Eigen::MatrixXd::Zero(nva, nocc_alpha);
        out.beta = Eigen::MatrixXd::Zero(nvb, nocc_beta);
        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < nocc_alpha; ++i)
                out.alpha(a, i) = sol(a * nocc_alpha + i);
        for (int a = 0; a < nvb; ++a)
            for (int i = 0; i < nocc_beta; ++i)
                out.beta(a, i) = sol(nova + a * nocc_beta + i);
        return out;
    }
} // namespace HartreeFock::Correlation
