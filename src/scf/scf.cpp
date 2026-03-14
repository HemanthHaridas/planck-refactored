#include <Eigen/Eigenvalues>
#include <chrono>
#include <format>
#include <limits>

#include "scf.h"
#include "integrals/base.h"
#include "io/logging.h"

// ─── Orthogonalization ────────────────────────────────────────────────────────

std::expected<Eigen::MatrixXd, std::string> HartreeFock::SCF::build_orthogonalizer(const Eigen::MatrixXd& S, double threshold)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    if (solver.info() != Eigen::Success)
        return std::unexpected("Overlap matrix diagonalization failed");

    const Eigen::VectorXd& evals = solver.eigenvalues();
    if (evals.minCoeff() < threshold)
        return std::unexpected(std::format("Overlap matrix is near-singular (min eigenvalue = {:.3e})", evals.minCoeff()));

    // X = U * s^{-1/2} * U^T
    const Eigen::MatrixXd& U = solver.eigenvectors();
    const Eigen::VectorXd  s_inv_sqrt = evals.array().rsqrt().matrix();
    return U * s_inv_sqrt.asDiagonal() * U.transpose();
}

// ─── Initial density ─────────────────────────────────────────────────────────

Eigen::MatrixXd HartreeFock::SCF::initial_density(const Eigen::MatrixXd& H, const Eigen::MatrixXd& X, std::size_t n_occ)
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

std::expected<void, std::string> HartreeFock::SCF::run_rhf(HartreeFock::Calculator& calculator, const std::vector<HartreeFock::ShellPair>& shell_pairs)
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

    // ── Initial density ───────────────────────────────────────────────────────
    // SCFGuess::Read: reuse density loaded from checkpoint.
    // The driver already reset _guess to HCore if the checkpoint load failed.
    Eigen::MatrixXd P = (calculator._scf._guess == HartreeFock::SCFGuess::Read)
        ? calculator._info._scf.alpha.density
        : initial_density(H, X, n_occ);

    const unsigned int max_iter = calculator._scf.get_max_cycles(nbasis);
    const double tol_energy  = calculator._scf._tol_energy;
    const double tol_density = calculator._scf._tol_density;

    // ── Conventional vs Direct ─────────────────────────────────────────────────
    // Conventional: ERI tensor built once; each iteration only contracts.
    // Direct: ERI recomputed from integrals every iteration.
    // Auto: conventional when nbasis ≤ _threshold, direct otherwise.
    // Only ObaraSaika supports precomputed ERI storage; other engines always direct.
    const bool use_conventional =
        calculator._integral._engine == HartreeFock::IntegralMethod::ObaraSaika &&
        (calculator._scf._mode == HartreeFock::SCFMode::Conventional ||
         (calculator._scf._mode == HartreeFock::SCFMode::Auto &&
          nbasis <= static_cast<std::size_t>(calculator._scf._threshold)));

    std::vector<double> eri;
    if (use_conventional)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :",
            std::format("Building ERI tensor ({:.1f} MB)", nbasis * nbasis * nbasis * nbasis * 8.0 / 1e6));
        eri = HartreeFock::ObaraSaika::_compute_2e(shell_pairs, nbasis, calculator._integral._tol_eri);
        calculator._eri = eri;  // persist for post-HF use
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    // ── DIIS state ────────────────────────────────────────────────────────────
    HartreeFock::DIISState diis;
    diis.max_vecs         = calculator._scf._DIIS_dim;
    const bool use_diis   = calculator._scf._use_DIIS;

    const double tol_eri = calculator._integral._tol_eri;
    double E_prev = 0.0;

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; iter++)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // ── Build two-electron contribution G = J - 0.5*K ────────────────────
        Eigen::MatrixXd G = use_conventional
            ? HartreeFock::ObaraSaika::_compute_fock_rhf(eri, P, nbasis)
            : _compute_2e_fock(shell_pairs, P, nbasis, calculator._integral._engine, tol_eri);

        // ── Fock matrix ───────────────────────────────────────────────────────
        const Eigen::MatrixXd F = H + G;

        // ── Electronic energy  E = 0.5 * tr(P * (H + F)) ────────────────────
        // Always computed from the raw (non-extrapolated) Fock matrix.
        const double E_elec  = 0.5 * (P.array() * (H + F).array()).sum();
        const double E_total = E_elec + calculator._nuclear_repulsion;

        // ── DIIS: compute Pulay error and push to subspace ────────────────────
        // Error matrix (orthonormal basis): e = X^T (FPS - SPF) X
        double diis_err = 0.0;
        if (use_diis)
        {
            const Eigen::MatrixXd e = X.transpose() * (F * P * S - S * P * F) * X;
            diis.push(F, e);
            diis_err = diis.error_norm();
        }

        // ── Select Fock matrix for diagonalization ────────────────────────────
        // Once DIIS has ≥2 vectors, use the extrapolated Fock; otherwise plain F.
        const bool do_diis           = use_diis && diis.ready();
        const Eigen::MatrixXd F_diag = do_diis ? diis.extrapolate() : F;

        // ── Transform to orthonormal basis and diagonalize ────────────────────
        const Eigen::MatrixXd Fprime = X.transpose() * F_diag * X;
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
        HartreeFock::Logger::scf_iteration(iter, E_total, delta_E, delta_P_rms, delta_P_max, diis_err, 0.0, iter_time);

        P      = P_new;
        E_prev = E_total;

        // Store current SCF state
        calculator._info._scf.alpha.fock            = F;
        calculator._info._scf.alpha.density         = P;
        calculator._info._scf.alpha.mo_energies     = eps;
        calculator._info._scf.alpha.mo_coefficients = C;
        calculator._info._energy                    = E_elec;
        calculator._info._delta_energy              = delta_E;
        calculator._info._delta_density_max         = delta_P_max;
        calculator._info._delta_density_rms         = delta_P_rms;
        
        calculator._total_energy                    = E_total;
        
        if (iter > 1 && delta_E < tol_energy && delta_P_rms < tol_density && delta_P_max < tol_density)
        {
            calculator._info._scf.alpha.mo_energies     = eps;
            calculator._info._scf.alpha.mo_coefficients = C;
            calculator._info._is_converged              = true;

            HartreeFock::Logger::scf_footer();
            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, std::format("SCF Converged after {} iterations", iter));
            HartreeFock::Logger::blank();
            return {};
        }
    }

    return std::unexpected(std::format("SCF did not converge in {} iterations", max_iter));
}

// ─── Spin contamination ───────────────────────────────────────────────────────

static void _log_spin_contamination(
    const Eigen::MatrixXd& Ca,
    const Eigen::MatrixXd& Cb,
    const Eigen::MatrixXd& S,
    int n_alpha, int n_beta,
    unsigned int multiplicity)
{
    // <S^2> = Sz*(Sz+1) + N_beta - ||C_alpha_occ^T S C_beta_occ||_F^2
    const double Sz = 0.5 * static_cast<double>(n_alpha - n_beta);
    const Eigen::MatrixXd OV = Ca.leftCols(n_alpha).transpose() * S * Cb.leftCols(n_beta);
    const double S2       = Sz * (Sz + 1.0) + static_cast<double>(n_beta) - OV.squaredNorm();
    const double S_exact  = 0.5 * static_cast<double>(multiplicity - 1);
    const double S2_exact = S_exact * (S_exact + 1.0);

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "<S^2> :",
        std::format("{:.6f}  (exact: {:.6f})", S2, S2_exact));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "<S>   :",
        std::format("{:.6f}", std::sqrt(std::max(0.0, S2))));
}

// ─── UHF SCF ─────────────────────────────────────────────────────────────────

std::expected<void, std::string> HartreeFock::SCF::run_uhf(
    HartreeFock::Calculator& calculator,
    const std::vector<HartreeFock::ShellPair>& shell_pairs)
{
    const Eigen::MatrixXd& S = calculator._overlap;
    const Eigen::MatrixXd& H = calculator._hcore;
    const std::size_t nbasis = calculator._shells.nbasis();

    const int n_electrons = static_cast<int>(
        calculator._molecule.atomic_numbers.cast<int>().sum()
        - calculator._molecule.charge);

    const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;

    if (n_unpaired < 0 || n_unpaired > n_electrons)
        return std::unexpected("Invalid multiplicity for given electron count");
    if ((n_electrons - n_unpaired) % 2 != 0)
        return std::unexpected("Multiplicity inconsistent with electron count parity");

    const int n_alpha = (n_electrons + n_unpaired) / 2;
    const int n_beta  = (n_electrons - n_unpaired) / 2;

    // ── Orthogonalization matrix X = S^{-1/2} ────────────────────────────────
    auto X_result = build_orthogonalizer(S);
    if (!X_result)
        return std::unexpected(X_result.error());
    const Eigen::MatrixXd X = std::move(*X_result);

    // ── Initial spin densities from core Hamiltonian ─────────────────────────
    // Factor 1.0 per spin (UHF), vs 2.0 in RHF.
    auto make_density_spin = [&](int n_occ) -> Eigen::MatrixXd
    {
        const Eigen::MatrixXd Hp = X.transpose() * H * X;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s(Hp);
        const Eigen::MatrixXd C = X * s.eigenvectors();
        return C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
    };

    // SCFGuess::Read: reuse densities loaded from checkpoint.
    const bool use_chk_uhf = (calculator._scf._guess == HartreeFock::SCFGuess::Read);

    Eigen::MatrixXd Pa = use_chk_uhf
        ? calculator._info._scf.alpha.density
        : make_density_spin(n_alpha);
    Eigen::MatrixXd Pb = use_chk_uhf
        ? calculator._info._scf.beta.density
        : make_density_spin(n_beta);

    const unsigned int max_iter  = calculator._scf.get_max_cycles(nbasis);
    const double tol_energy      = calculator._scf._tol_energy;
    const double tol_density     = calculator._scf._tol_density;
    const double level_shift     = calculator._scf._level_shift;
    const double restart_factor  = calculator._scf._diis_restart_factor;

    // ── Conventional vs Direct ────────────────────────────────────────────────
    const bool use_conventional =
        calculator._integral._engine == HartreeFock::IntegralMethod::ObaraSaika &&
        (calculator._scf._mode == HartreeFock::SCFMode::Conventional ||
         (calculator._scf._mode == HartreeFock::SCFMode::Auto &&
          nbasis <= static_cast<std::size_t>(calculator._scf._threshold)));

    std::vector<double> eri;
    if (use_conventional)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :",
            std::format("Building ERI tensor ({:.1f} MB)", nbasis * nbasis * nbasis * nbasis * 8.0 / 1e6));
        eri = HartreeFock::ObaraSaika::_compute_2e(shell_pairs, nbasis, calculator._integral._tol_eri);
        calculator._eri = eri;  // persist for post-HF use
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    // ── DIIS state per spin ───────────────────────────────────────────────────
    HartreeFock::DIISState diis_a, diis_b;
    diis_a.max_vecs = diis_b.max_vecs = calculator._scf._DIIS_dim;
    const bool use_diis = calculator._scf._use_DIIS;

    const double tol_eri = calculator._integral._tol_eri;
    double E_prev      = 0.0;
    double diis_err_prev = std::numeric_limits<double>::max();

    // Level-shift: save orthonormal MO coefficients from the previous iteration
    // to build the virtual projector Q = I - C_occ * C_occ^T  (in ortho basis).
    Eigen::MatrixXd Ca_orth_prev, Cb_orth_prev;
    const Eigen::MatrixXd I_nb = Eigen::MatrixXd::Identity(nbasis, nbasis);

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; ++iter)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // ── Two-electron Fock contributions ───────────────────────────────────
        auto [Ga, Gb] = use_conventional
            ? HartreeFock::ObaraSaika::_compute_fock_uhf(eri, Pa, Pb, nbasis)
            : _compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, calculator._integral._engine, tol_eri);

        const Eigen::MatrixXd Fa = H + Ga;
        const Eigen::MatrixXd Fb = H + Gb;

        // ── Electronic energy — always from the bare (unshifted) Fock ───────────
        const double E_elec  = 0.5 * ((Pa.array() * (H + Fa).array()).sum()
                                    +  (Pb.array() * (H + Fb).array()).sum());
        const double E_total = E_elec + calculator._nuclear_repulsion;

        // ── Level shift: build Fa_s/Fb_s before DIIS ─────────────────────────
        // The shift  F_s = F + λ·X·(I − P_occ^orth)·X^T  raises virtual orbital
        // energies, widening the gap and preventing occupied–virtual swapping.
        // DIIS stores and extrapolates the shifted Fock so the subspace is
        // self-consistent; energy is always reported from the bare Fock.
        const bool shift_active = (level_shift > 0.0 && Ca_orth_prev.cols() > 0);
        Eigen::MatrixXd Fa_s = Fa, Fb_s = Fb;
        if (shift_active)
        {
            const Eigen::MatrixXd shift_a =
                level_shift * X * (I_nb - Ca_orth_prev.leftCols(n_alpha) *
                                           Ca_orth_prev.leftCols(n_alpha).transpose()) * X.transpose();
            const Eigen::MatrixXd shift_b =
                level_shift * X * (I_nb - Cb_orth_prev.leftCols(n_beta)  *
                                           Cb_orth_prev.leftCols(n_beta).transpose())  * X.transpose();
            Fa_s += shift_a;
            Fb_s += shift_b;
        }

        // ── DIIS: Pulay errors from the shifted Fock ─────────────────────────
        double diis_err = 0.0;
        if (use_diis)
        {
            const Eigen::MatrixXd ea = X.transpose() * (Fa_s * Pa * S - S * Pa * Fa_s) * X;
            const Eigen::MatrixXd eb = X.transpose() * (Fb_s * Pb * S - S * Pb * Fb_s) * X;

            // RMS norm — same normalization as DIISState::error_norm()
            const auto rms_norm = [](const Eigen::MatrixXd& m) {
                return std::sqrt(m.squaredNorm() / static_cast<double>(m.size()));
            };
            const double cur_err = std::max(rms_norm(ea), rms_norm(eb));

            // ── DIIS restart ──────────────────────────────────────────────────
            if (restart_factor > 0.0 && iter > 2 && cur_err > diis_err_prev * restart_factor)
            {
                diis_a.clear();
                diis_b.clear();
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                    "DIIS :", std::format("Subspace restarted at iter {} (error grew {:.1f}×)",
                                          iter, cur_err / diis_err_prev));
            }

            diis_a.push(Fa_s, ea);
            diis_b.push(Fb_s, eb);
            diis_err = std::max(diis_a.error_norm(), diis_b.error_norm());
            diis_err_prev = cur_err;
        }

        // Extrapolated (shifted) Fock for diagonalization
        const Eigen::MatrixXd Fa_diag = (use_diis && diis_a.ready()) ? diis_a.extrapolate() : Fa_s;
        const Eigen::MatrixXd Fb_diag = (use_diis && diis_b.ready()) ? diis_b.extrapolate() : Fb_s;

        // ── Diagonalize alpha ─────────────────────────────────────────────────
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sa(X.transpose() * Fa_diag * X);
        if (sa.info() != Eigen::Success)
            return std::unexpected(std::format("Alpha Fock diagonalization failed at iter {}", iter));
        const Eigen::MatrixXd Ca  = X * sa.eigenvectors();

        // Subtract level shift from virtual eigenvalues to report physical energies
        Eigen::VectorXd epsa = sa.eigenvalues();
        if (shift_active)
            epsa.tail(nbasis - n_alpha).array() -= level_shift;

        // ── Diagonalize beta ──────────────────────────────────────────────────
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sb(X.transpose() * Fb_diag * X);
        if (sb.info() != Eigen::Success)
            return std::unexpected(std::format("Beta Fock diagonalization failed at iter {}", iter));
        const Eigen::MatrixXd Cb  = X * sb.eigenvectors();

        Eigen::VectorXd epsb = sb.eigenvalues();
        if (shift_active)
            epsb.tail(nbasis - n_beta).array() -= level_shift;

        // Save orthonormal eigenvectors for next iteration's virtual projector
        Ca_orth_prev = sa.eigenvectors();
        Cb_orth_prev = sb.eigenvectors();

        // ── New spin densities ────────────────────────────────────────────────
        const Eigen::MatrixXd Pa_new = Ca.leftCols(n_alpha) * Ca.leftCols(n_alpha).transpose();
        const Eigen::MatrixXd Pb_new = Cb.leftCols(n_beta)  * Cb.leftCols(n_beta).transpose();

        // ── Convergence on total density ──────────────────────────────────────
        const Eigen::MatrixXd dPt   = (Pa_new + Pb_new) - (Pa + Pb);
        const double delta_E        = std::abs(E_total - E_prev);
        const double delta_P_max    = dPt.cwiseAbs().maxCoeff();
        const double delta_P_rms    = std::sqrt(dPt.squaredNorm() / (nbasis * nbasis));

        const double iter_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - iter_start).count();
        HartreeFock::Logger::scf_iteration(iter, E_total, delta_E, delta_P_rms, delta_P_max, diis_err, 0.0, iter_time);

        Pa = Pa_new;  Pb = Pb_new;  E_prev = E_total;

        // ── Store current SCF state ───────────────────────────────────────────
        calculator._info._scf.alpha.fock            = Fa;
        calculator._info._scf.alpha.density         = Pa;
        calculator._info._scf.alpha.mo_energies     = epsa;
        calculator._info._scf.alpha.mo_coefficients = Ca;
        calculator._info._scf.beta.fock             = Fb;
        calculator._info._scf.beta.density          = Pb;
        calculator._info._scf.beta.mo_energies      = epsb;
        calculator._info._scf.beta.mo_coefficients  = Cb;
        calculator._info._energy                    = E_elec;
        calculator._info._delta_energy              = delta_E;
        calculator._info._delta_density_max         = delta_P_max;
        calculator._info._delta_density_rms         = delta_P_rms;
        calculator._total_energy                    = E_total;

        if (iter > 1 && delta_E < tol_energy && delta_P_rms < tol_density && delta_P_max < tol_density)
        {
            calculator._info._is_converged = true;

            HartreeFock::Logger::scf_footer();
            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "UHF Converged :",
                std::format("E = {:.10f} Eh  after {} iterations", E_total, iter));
            HartreeFock::Logger::blank();

            _log_spin_contamination(Ca, Cb, S, n_alpha, n_beta,
                                    calculator._molecule.multiplicity);
            HartreeFock::Logger::blank();

            return {};
        }
    }

    return std::unexpected(std::format("UHF SCF did not converge in {} iterations", max_iter));
}
