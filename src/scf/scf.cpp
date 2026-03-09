#include <Eigen/Eigenvalues>
#include <chrono>
#include <format>

#include "scf.h"
#include "io/logging.h"

// ─── Orthogonalization ────────────────────────────────────────────────────────

std::expected<Eigen::MatrixXd, std::string>
HartreeFock::SCF::build_orthogonalizer(const Eigen::MatrixXd& S, double threshold)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    if (solver.info() != Eigen::Success)
        return std::unexpected("Overlap matrix diagonalization failed");

    const Eigen::VectorXd& evals = solver.eigenvalues();
    if (evals.minCoeff() < threshold)
        return std::unexpected(
            std::format("Overlap matrix is near-singular (min eigenvalue = {:.3e})", evals.minCoeff()));

    // X = U * s^{-1/2} * U^T
    const Eigen::MatrixXd& U = solver.eigenvectors();
    const Eigen::VectorXd  s_inv_sqrt = evals.array().rsqrt().matrix();
    return U * s_inv_sqrt.asDiagonal() * U.transpose();
}

// ─── Initial density ─────────────────────────────────────────────────────────

Eigen::MatrixXd HartreeFock::SCF::initial_density(const Eigen::MatrixXd& H,
                                                    const Eigen::MatrixXd& X,
                                                    std::size_t n_occ)
{
    // Transform H to orthonormal basis: H' = X^T * H * X
    const Eigen::MatrixXd Hprime = X.transpose() * H * X;

    // Diagonalize H'
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hprime);
    const Eigen::MatrixXd C = X * solver.eigenvectors();

    // P_μν = 2 * sum_{i=1}^{n_occ} C_{μi} * C_{νi}  (RHF: factor of 2 for closed shell)
    const Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    return 2.0 * C_occ * C_occ.transpose();
}

// ─── SCF iteration ───────────────────────────────────────────────────────────

std::expected<void, std::string>
HartreeFock::SCF::run_rhf(HartreeFock::Calculator& calculator)
{
    const Eigen::MatrixXd& S = calculator._overlap;
    const Eigen::MatrixXd& H = calculator._hcore;
    const std::size_t nbasis = calculator._shells.nbasis();

    // Number of occupied orbitals (closed shell singlet assumed)
    const int n_electrons = static_cast<int>(
        (calculator._molecule.atomic_numbers.cast<int>().sum()) - calculator._molecule.charge);

    if (n_electrons % 2 != 0)
        return std::unexpected("RHF requires an even number of electrons (closed shell)");

    const std::size_t n_occ = static_cast<std::size_t>(n_electrons / 2);

    // ── Orthogonalization matrix X = S^{-1/2} ────────────────────────────────
    auto X_result = build_orthogonalizer(S);
    if (!X_result)
        return std::unexpected(X_result.error());
    const Eigen::MatrixXd X = std::move(*X_result);

    // ── Initial density from core Hamiltonian ────────────────────────────────
    Eigen::MatrixXd P = initial_density(H, X, n_occ);

    const unsigned int max_iter = calculator._scf.get_max_cycles(nbasis);
    const double tol_energy  = calculator._scf._tol_energy;
    const double tol_density = calculator._scf._tol_density;

    double E_prev = 0.0;

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; iter++)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // ── Build two-electron contribution G = J - 0.5*K ────────────────────
        // TODO: implement ERI-based G build (conventional or direct)
        // For now G = 0 so this is purely a core-Hamiltonian calculation.
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nbasis, nbasis);

        // ── Fock matrix ───────────────────────────────────────────────────────
        const Eigen::MatrixXd F = H + G;

        // ── Electronic energy  E = 0.5 * tr(P * (H + F)) ────────────────────
        const double E_elec = 0.5 * (P.array() * (H + F).array()).sum();
        const double E_total = E_elec + calculator._nuclear_repulsion;

        // ── Transform F to orthonormal basis and diagonalize ─────────────────
        const Eigen::MatrixXd Fprime = X.transpose() * F * X;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Fprime);
        if (solver.info() != Eigen::Success)
            return std::unexpected(std::format("Fock diagonalization failed at iteration {}", iter));

        const Eigen::MatrixXd C     = X * solver.eigenvectors();
        const Eigen::VectorXd eps   = solver.eigenvalues();
        const Eigen::MatrixXd C_occ = C.leftCols(n_occ);

        // ── New density P_new ─────────────────────────────────────────────────
        const Eigen::MatrixXd P_new = 2.0 * C_occ * C_occ.transpose();

        // ── Convergence checks ────────────────────────────────────────────────
        const double delta_E       = std::abs(E_total - E_prev);
        const double delta_P_max   = (P_new - P).cwiseAbs().maxCoeff();
        const double delta_P_rms   = std::sqrt((P_new - P).squaredNorm() / (nbasis * nbasis));

        const double iter_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - iter_start).count();
        HartreeFock::Logger::scf_iteration(iter, E_total, delta_E, delta_P_rms, 0.0, 0.0, iter_time);

        P      = P_new;
        E_prev = E_total;

        // Store current SCF state
        calculator._info._scf.alpha.fock            = F;
        calculator._info._scf.alpha.density         = P;
        calculator._info._scf.alpha.mo_energies     = eps;
        calculator._info._scf.alpha.mo_coefficients = C;
        calculator._total_energy                    = E_total;

        if (iter > 1 && delta_E < tol_energy && delta_P_rms < tol_density)
        {
            calculator._info._scf.alpha.mo_energies     = eps;
            calculator._info._scf.alpha.mo_coefficients = C;
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                "SCF Converged :",
                std::format("E = {:.10f} Eh  after {} iterations", E_total, iter));
            return {};
        }
    }

    return std::unexpected(std::format("SCF did not converge in {} iterations", max_iter));
}
