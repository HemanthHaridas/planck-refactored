#include "uhf_response.h"

#include <format>
#include <limits>

#include "dft/response_gmres.h"
#include "post_hf/ri/ri_eri.h"
#include "scf/scf.h"

namespace HartreeFock::Correlation
{
    namespace
    {
        // The coupled alpha/beta orbital Hessian applied to one packed trial
        // rotation x = [x_alpha (a*nocc_alpha+i) | x_beta]. The dense builder
        // calls it on unit vectors; solve_uhf_cphf calls it from GMRES.
        struct UHFOrbitalHessian
        {
            HartreeFock::Calculator &calculator;
            const std::vector<HartreeFock::ShellPair> &shell_pairs;
            Eigen::MatrixXd Ca_occ, Ca_virt, Cb_occ, Cb_virt;
            Eigen::VectorXd diagonal;

            UHFOrbitalHessian(
                HartreeFock::Calculator &calc,
                const std::vector<HartreeFock::ShellPair> &pairs,
                const Eigen::MatrixXd &coeff_alpha,
                const Eigen::MatrixXd &coeff_beta,
                const Eigen::VectorXd &energy_alpha,
                const Eigen::VectorXd &energy_beta,
                int nocc_alpha,
                int nocc_beta)
                : calculator(calc), shell_pairs(pairs)
            {
                const int nva = static_cast<int>(coeff_alpha.cols()) - nocc_alpha;
                const int nvb = static_cast<int>(coeff_beta.cols()) - nocc_beta;
                Ca_occ = coeff_alpha.leftCols(nocc_alpha);
                Ca_virt = coeff_alpha.middleCols(nocc_alpha, nva);
                Cb_occ = coeff_beta.leftCols(nocc_beta);
                Cb_virt = coeff_beta.middleCols(nocc_beta, nvb);
                const int nova = nva * nocc_alpha;
                diagonal.resize(nova + nvb * nocc_beta);
                for (int a = 0; a < nva; ++a)
                    for (int i = 0; i < nocc_alpha; ++i)
                        diagonal(a * nocc_alpha + i) = energy_alpha(nocc_alpha + a) - energy_alpha(i);
                for (int a = 0; a < nvb; ++a)
                    for (int i = 0; i < nocc_beta; ++i)
                        diagonal(nova + a * nocc_beta + i) = energy_beta(nocc_beta + a) - energy_beta(i);
            }

            Eigen::VectorXd operator()(const Eigen::VectorXd &x) const
            {
                const int noa = static_cast<int>(Ca_occ.cols()), nva = static_cast<int>(Ca_virt.cols());
                const int nob = static_cast<int>(Cb_occ.cols()), nvb = static_cast<int>(Cb_virt.cols());
                const int nova = nva * noa;
                // x is packed a-major (a*nocc+i), i.e. the row-major (nv x nocc) block.
                using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                const Eigen::MatrixXd xa = Eigen::Map<const RowMajor>(x.data(), nva, noa);
                const Eigen::MatrixXd xb = Eigen::Map<const RowMajor>(x.data() + nova, nvb, nob);

                // Trial rotation -> AO density response -> induced Coulomb/exchange.
                const Eigen::MatrixXd dm1a = Ca_virt * xa * Ca_occ.transpose();
                const Eigen::MatrixXd dm1b = Cb_virt * xb * Cb_occ.transpose();
                const Eigen::MatrixXd dm1a_sym = dm1a + dm1a.transpose();
                const Eigen::MatrixXd dm1b_sym = dm1b + dm1b.transpose();
                // RI-consistent CPHF operator under RI-MP2 or RI-JK (Step RG4.2): the induced
                // Coulomb/exchange response is the same {J(Pa+Pb) - K(P_sigma)}
                // quantity, built from the 3-center factors instead of the dense ERI.
                const auto [va_ao, vb_ao] = [&]() -> std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
                {
                    if (RI::fitted_response(calculator))
                    {
                        auto [fa, fb] = RI::build_ri_fock_uhf(calculator, dm1a_sym, dm1b_sym);
                        return {std::move(fa), std::move(fb)};
                    }
                    // The densities and C are in the working basis.
                    auto g = SCF::two_electron_fock_uhf(calculator, shell_pairs, dm1a_sym, dm1b_sym);
                    if (!g)
                    {
                        // cannot happen with consistent C; poisons the step visibly
                        const Eigen::MatrixXd nan = Eigen::MatrixXd::Constant(
                            dm1a_sym.rows(), dm1a_sym.cols(), std::numeric_limits<double>::quiet_NaN());
                        return {nan, nan};
                    }
                    return std::move(*g);
                }();

                const RowMajor va = Ca_virt.transpose() * va_ao * Ca_occ;
                const RowMajor vb = Cb_virt.transpose() * vb_ao * Cb_occ;
                Eigen::VectorXd y = diagonal.cwiseProduct(x);
                y.head(nova) += Eigen::Map<const Eigen::VectorXd>(va.data(), nova);
                y.tail(nvb * nob) += Eigen::Map<const Eigen::VectorXd>(vb.data(), nvb * nob);
                return y;
            }
        };
    } // namespace

    // Coupled alpha/beta orbital-Hessian matrix (docs/SOSCF_UHF.md,
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

        const int nva = static_cast<int>(coeff_alpha.cols()) - nocc_alpha;
        const int nvb = static_cast<int>(coeff_beta.cols()) - nocc_beta;
        if (nocc_alpha <= 0 || nocc_beta < 0 || nva <= 0 || nvb <= 0)
            return std::unexpected("build_uhf_cphf_matrix: invalid occupied/virtual dimensions.");

        // Materialize the Hessian column by column from unit trial rotations.
        // One full UHF Fock build per column: use only where the dense matrix
        // itself is needed (SOSCF); solve_uhf_cphf stays matrix-free.
        const UHFOrbitalHessian hessian(
            calculator, shell_pairs, coeff_alpha, coeff_beta,
            energy_alpha, energy_beta, nocc_alpha, nocc_beta);
        const int n = static_cast<int>(hessian.diagonal.size());
        Eigen::MatrixXd A(n, n);
        for (int col = 0; col < n; ++col)
            A.col(col) = hessian(Eigen::VectorXd::Unit(n, col));
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

        const int nva = static_cast<int>(coeff_alpha.cols()) - nocc_alpha;
        const int nvb = static_cast<int>(coeff_beta.cols()) - nocc_beta;
        if (nocc_alpha <= 0 || nocc_beta < 0 || nva <= 0 || nvb <= 0)
            return std::unexpected("solve_uhf_cphf: invalid occupied/virtual dimensions.");
        const int nova = nva * nocc_alpha;
        const int novb = nvb * nocc_beta;

        Eigen::VectorXd rhs(nova + novb);
        for (int a = 0; a < nva; ++a)
            for (int i = 0; i < nocc_alpha; ++i)
                rhs(a * nocc_alpha + i) = -rhs_alpha(a, i);
        for (int a = 0; a < nvb; ++a)
            for (int i = 0; i < nocc_beta; ++i)
                rhs(nova + a * nocc_beta + i) = -rhs_beta(a, i);

        // Matrix-free: GMRES with the orbital-gap diagonal as preconditioner,
        // one Fock build per action instead of one per Hessian column.
        const UHFOrbitalHessian hessian(
            calculator, shell_pairs, coeff_alpha, coeff_beta,
            energy_alpha, energy_beta, nocc_alpha, nocc_beta);
        const DFT::Response::Action action =
            [&hessian](const Eigen::VectorXd &x) -> std::expected<Eigen::VectorXd, std::string>
        { return hessian(x); };
        // 1e-10 of max|rhs|: on 2 H2O+ / 6-31G* (n=541) this is 33 actions and
        // agrees with the dense QR to 1.2e-8 of max|z|; the dense reference's
        // own screening noise floors that at ~5e-10, and the printed gradient
        // is identical either way.
        auto solve = DFT::Response::gmres(action, rhs, hessian.diagonal, 1e-10);
        if (!solve)
            return std::unexpected("solve_uhf_cphf: " + solve.error());
        if (!solve->converged)
            return std::unexpected(std::format(
                "solve_uhf_cphf: GMRES did not converge (iterations={}, actions={}, residual={:.3e}).",
                solve->iterations, solve->action_count, solve->true_residuals.back()));
        const Eigen::VectorXd &sol = solve->x;
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
