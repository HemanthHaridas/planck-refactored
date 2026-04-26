#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <format>
#include <limits>
#include <numeric>

#include "integrals/base.h"
#include "io/logging.h"
#include "sad.h"
#include "scf.h"

// ─── Orthogonalization ────────────────────────────────────────────────────────

std::expected<Eigen::MatrixXd, std::string> HartreeFock::SCF::build_orthogonalizer(const Eigen::MatrixXd &S, double threshold)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S);
    if (solver.info() != Eigen::Success)
        return std::unexpected("Overlap matrix diagonalization failed");

    const Eigen::VectorXd &evals = solver.eigenvalues();
    if (evals.minCoeff() < threshold)
        return std::unexpected(std::format("Overlap matrix is near-singular (min eigenvalue = {:.3e})", evals.minCoeff()));

    // X = U * s^{-1/2} * U^T
    const Eigen::MatrixXd &U = solver.eigenvectors();
    const Eigen::VectorXd s_inv_sqrt = evals.array().rsqrt().matrix();
    return U * s_inv_sqrt.asDiagonal() * U.transpose();
}

// ─── Initial density ─────────────────────────────────────────────────────────

Eigen::MatrixXd HartreeFock::SCF::initial_density(const Eigen::MatrixXd &H, const Eigen::MatrixXd &X, std::size_t n_occ)
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

Eigen::MatrixXd HartreeFock::SCF::initial_density_sao(
    const Eigen::MatrixXd &H,
    const Eigen::MatrixXd &U,
    const std::vector<int> &block_sizes,
    const std::vector<int> &block_offsets,
    std::size_t n_occ)
{
    const Eigen::Index nbasis = H.rows();
    if (U.rows() != nbasis || U.cols() != nbasis)
        return Eigen::MatrixXd::Zero(nbasis, nbasis);

    const Eigen::MatrixXd H_sao = U.transpose() * H * U;
    Eigen::VectorXd eps_sao(nbasis);
    Eigen::MatrixXd C_sao = Eigen::MatrixXd::Zero(nbasis, nbasis);

    for (int b = 0; b < static_cast<int>(block_sizes.size()); ++b)
    {
        const int off = block_offsets[static_cast<std::size_t>(b)];
        const int ni = block_sizes[static_cast<std::size_t>(b)];
        if (ni == 0)
            continue;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_sao.block(off, off, ni, ni));
        eps_sao.segment(off, ni) = solver.eigenvalues();
        C_sao.block(off, off, ni, ni) = solver.eigenvectors();
    }

    std::vector<int> order(static_cast<std::size_t>(nbasis));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b)
                     { return eps_sao[a] < eps_sao[b]; });

    Eigen::MatrixXd C_sao_sorted(nbasis, nbasis);
    for (int k = 0; k < static_cast<int>(nbasis); ++k)
        C_sao_sorted.col(k) = C_sao.col(order[static_cast<std::size_t>(k)]);

    const Eigen::MatrixXd C = U * C_sao_sorted;
    const Eigen::MatrixXd C_occ = C.leftCols(static_cast<Eigen::Index>(n_occ));
    return 2.0 * C_occ * C_occ.transpose();
}

HartreeFock::SCF::IterationMetrics HartreeFock::SCF::restricted_iteration_metrics(
    const Eigen::MatrixXd &previous_density,
    const Eigen::MatrixXd &next_density,
    double previous_total_energy,
    double total_energy)
{
    const Eigen::MatrixXd delta_density = next_density - previous_density;
    IterationMetrics metrics;
    metrics.delta_energy = std::abs(total_energy - previous_total_energy);
    metrics.delta_density_max = delta_density.cwiseAbs().maxCoeff();
    metrics.delta_density_rms = std::sqrt(
        delta_density.squaredNorm() /
        static_cast<double>(delta_density.rows() * delta_density.cols()));
    return metrics;
}

HartreeFock::SCF::IterationMetrics HartreeFock::SCF::unrestricted_iteration_metrics(
    const Eigen::MatrixXd &previous_alpha_density,
    const Eigen::MatrixXd &previous_beta_density,
    const Eigen::MatrixXd &next_alpha_density,
    const Eigen::MatrixXd &next_beta_density,
    double previous_total_energy,
    double total_energy)
{
    const Eigen::MatrixXd delta_density =
        (next_alpha_density + next_beta_density) -
        (previous_alpha_density + previous_beta_density);

    IterationMetrics metrics;
    metrics.delta_energy = std::abs(total_energy - previous_total_energy);
    metrics.delta_density_max = delta_density.cwiseAbs().maxCoeff();
    metrics.delta_density_rms = std::sqrt(
        delta_density.squaredNorm() /
        static_cast<double>(delta_density.rows() * delta_density.cols()));
    return metrics;
}

bool HartreeFock::SCF::is_converged(
    const HartreeFock::OptionsSCF &scf_options,
    const IterationMetrics &metrics,
    unsigned int iteration) noexcept
{
    return iteration > 1 &&
           metrics.delta_energy < scf_options._tol_energy &&
           metrics.delta_density_rms < scf_options._tol_density &&
           metrics.delta_density_max < scf_options._tol_density;
}

void HartreeFock::SCF::store_restricted_iteration(
    HartreeFock::Calculator &calculator,
    const RestrictedIterationData &iteration,
    const IterationMetrics &metrics)
{
    calculator._info._scf.alpha.fock = iteration.fock;
    calculator._info._scf.alpha.density = iteration.density;
    calculator._info._scf.alpha.mo_energies = iteration.mo_energies;
    calculator._info._scf.alpha.mo_coefficients = iteration.mo_coefficients;
    calculator._info._energy = iteration.electronic_energy;
    calculator._info._delta_energy = metrics.delta_energy;
    calculator._info._delta_density_max = metrics.delta_density_max;
    calculator._info._delta_density_rms = metrics.delta_density_rms;
    calculator._total_energy = iteration.total_energy;
}

void HartreeFock::SCF::store_unrestricted_iteration(
    HartreeFock::Calculator &calculator,
    const UnrestrictedIterationData &iteration,
    const IterationMetrics &metrics)
{
    calculator._info._scf.alpha.fock = iteration.alpha_fock;
    calculator._info._scf.alpha.density = iteration.alpha_density;
    calculator._info._scf.alpha.mo_energies = iteration.alpha_mo_energies;
    calculator._info._scf.alpha.mo_coefficients = iteration.alpha_mo_coefficients;
    calculator._info._scf.beta.fock = iteration.beta_fock;
    calculator._info._scf.beta.density = iteration.beta_density;
    calculator._info._scf.beta.mo_energies = iteration.beta_mo_energies;
    calculator._info._scf.beta.mo_coefficients = iteration.beta_mo_coefficients;
    calculator._info._energy = iteration.electronic_energy;
    calculator._info._delta_energy = metrics.delta_energy;
    calculator._info._delta_density_max = metrics.delta_density_max;
    calculator._info._delta_density_rms = metrics.delta_density_rms;
    calculator._total_energy = iteration.total_energy;
}

// ─── SCF iteration ───────────────────────────────────────────────────────────

std::expected<void, std::string> HartreeFock::SCF::run_rhf(
    HartreeFock::Calculator &calculator,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const HartreeFock::Solvation::PCMState *pcm)
{
    const Eigen::MatrixXd &S = calculator._overlap;
    const Eigen::MatrixXd &H = calculator._hcore;
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

    // ── SAO blocking setup ────────────────────────────────────────────────────
    // SAOs are orthonormal (U^T S U = I), so the orthogonalizer in SAO basis
    // is the identity — each block is diagonalized directly without an X step.
    const bool sao_active = calculator._use_sao_blocking &&
                            (calculator._sao_transform.rows() > 0);
    const Eigen::MatrixXd &U = calculator._sao_transform; // ref, no copy

    // ── Initial density ───────────────────────────────────────────────────────
    // ReadDensity / ReadFull: reuse density loaded from checkpoint.
    // The driver already reset _guess to HCore if the checkpoint load failed.
    const bool use_chk_density =
        (calculator._scf._guess == HartreeFock::SCFGuess::ReadDensity ||
         calculator._scf._guess == HartreeFock::SCFGuess::ReadFull);
    Eigen::MatrixXd P;
    if (use_chk_density)
    {
        P = calculator._info._scf.alpha.density;
    }
    else if (calculator._scf._guess == HartreeFock::SCFGuess::SAD)
    {
        auto sad_res = HartreeFock::SCF::compute_sad_guess_rhf(calculator);
        if (!sad_res)
            return std::unexpected("RHF SAD guess failed: " + sad_res.error());
        P = std::move(*sad_res);
    }
    else if (sao_active)
    {
        // When symmetry-blocked SCF is active, build the hcore guess in the
        // same SAO basis used for the production Fock diagonalization so the
        // initial density does not pick an arbitrary full-AO mixture from a
        // near-degenerate symmetry subspace.
        P = initial_density_sao(
            H,
            U,
            calculator._sao_block_sizes,
            calculator._sao_block_offsets,
            n_occ);
    }
    else
    {
        P = initial_density(H, X, n_occ);
    }

    const unsigned int max_iter = calculator._scf.get_max_cycles(nbasis);

    // ── Conventional vs Direct ─────────────────────────────────────────────────
    // Conventional: ERI tensor built once; each iteration only contracts.
    // Direct: ERI recomputed from integrals every iteration.
    // Auto: conventional when nbasis ≤ _threshold, direct otherwise.
    const bool use_conventional =
        (calculator._scf._mode == HartreeFock::SCFMode::Conventional ||
         (calculator._scf._mode == HartreeFock::SCFMode::Auto &&
          nbasis <= static_cast<std::size_t>(calculator._scf._threshold)));

    std::vector<double> eri;
    if (use_conventional)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :",
                                     std::format("Building ERI tensor ({:.1f} MB)", nbasis * nbasis * nbasis * nbasis * 8.0 / 1e6));
        eri = _compute_2e(shell_pairs, nbasis, calculator._integral._engine,
                          calculator._integral._tol_eri,
                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        calculator._eri = eri; // persist for post-HF use
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    // ── DIIS state ────────────────────────────────────────────────────────────
    HartreeFock::DIISState diis;
    diis.max_vecs = calculator._scf._DIIS_dim;
    const bool use_diis = calculator._scf._use_DIIS;

    const double tol_eri = calculator._integral._tol_eri;
    double E_prev = 0.0;

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; iter++)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // ── Build two-electron contribution G = J - 0.5*K ────────────────────
        Eigen::MatrixXd G = use_conventional
                                ? HartreeFock::ObaraSaika::_compute_fock_rhf(eri, P, nbasis)
                                : _compute_2e_fock(shell_pairs, P, nbasis, calculator._integral._engine, tol_eri,
                                                   calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

        Eigen::MatrixXd V_pcm = Eigen::MatrixXd::Zero(nbasis, nbasis);
        double pcm_energy = 0.0;
        if (pcm != nullptr && pcm->enabled())
        {
            auto pcm_result = HartreeFock::Solvation::evaluate_pcm_reaction_field(calculator, *pcm, P);
            if (!pcm_result)
                return std::unexpected("PCM build failed inside RHF iteration: " + pcm_result.error());
            V_pcm = std::move(pcm_result->reaction_potential);
            pcm_energy = pcm_result->solvation_energy;
        }

        // ── Fock matrix ───────────────────────────────────────────────────────
        const Eigen::MatrixXd F_gas = H + G;
        const Eigen::MatrixXd F = F_gas + V_pcm;

        // ── Electronic energy  E = E_gas + G_pcm ────────────────────────────
        const double E_gas = 0.5 * (P.array() * (H + F_gas).array()).sum();
        const double E_elec = E_gas + pcm_energy;
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
        const bool do_diis = use_diis && diis.ready();
        const Eigen::MatrixXd F_diag = do_diis ? diis.extrapolate() : F;

        // ── Diagonalize Fock matrix ───────────────────────────────────────────
        Eigen::MatrixXd C(nbasis, nbasis);
        Eigen::VectorXd eps(nbasis);

        if (!sao_active)
        {
            // ── Full AO diagonalization (original path) ───────────────────────
            const Eigen::MatrixXd Fprime = X.transpose() * F_diag * X;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Fprime);
            if (solver.info() != Eigen::Success)
                return std::unexpected(std::format("Fock diagonalization failed at iteration {}", iter));
            C = X * solver.eigenvectors();
            eps = solver.eigenvalues();
        }
        else
        {
            // ── SAO block-diagonal diagonalization ────────────────────────────
            // F in SAO basis is block-diagonal (one block per irrep).
            // Since SAOs are orthonormal, diagonalize each block directly.
            const Eigen::MatrixXd F_sao = U.transpose() * F_diag * U;
            const int n_blocks = static_cast<int>(calculator._sao_block_sizes.size());

            Eigen::VectorXd eps_sao(nbasis);
            Eigen::MatrixXd C_sao = Eigen::MatrixXd::Zero(nbasis, nbasis);
            std::vector<int> mo_irrep_idx(nbasis);

            for (int b = 0; b < n_blocks; ++b)
            {
                const int off = calculator._sao_block_offsets[b];
                const int ni = calculator._sao_block_sizes[b];
                if (ni == 0)
                    continue;

                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sb(
                    F_sao.block(off, off, ni, ni));
                if (sb.info() != Eigen::Success)
                    return std::unexpected(std::format(
                        "Block Fock diagonalization failed (block {}) at iteration {}", b, iter));

                eps_sao.segment(off, ni) = sb.eigenvalues();
                C_sao.block(off, off, ni, ni) = sb.eigenvectors();
                for (int k = 0; k < ni; ++k)
                    mo_irrep_idx[off + k] = calculator._sao_irrep_index[off + k];
            }

            // Sort all MOs globally by energy; MOs from different irreps interleave.
            std::vector<int> order(nbasis);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b)
                             { return eps_sao[a] < eps_sao[b]; });

            Eigen::VectorXd eps_sorted(nbasis);
            Eigen::MatrixXd C_sao_sorted(nbasis, nbasis);
            std::vector<std::string> mo_sym(nbasis);
            for (int k = 0; k < static_cast<int>(nbasis); ++k)
            {
                eps_sorted[k] = eps_sao[order[k]];
                C_sao_sorted.col(k) = C_sao.col(order[k]);
                mo_sym[k] = calculator._sao_irrep_names[mo_irrep_idx[order[k]]];
            }

            eps = eps_sorted;
            C = U * C_sao_sorted;
            // Write symmetry labels every iteration; the final write at convergence
            // will be the authoritative set.
            calculator._info._scf.alpha.mo_symmetry = std::move(mo_sym);
        }

        const Eigen::MatrixXd C_occ = C.leftCols(n_occ);

        // ── Next density ──────────────────────────────────────────────────────
        const Eigen::MatrixXd density_next = 2.0 * C_occ * C_occ.transpose();

        // ── Convergence checks ────────────────────────────────────────────────
        const IterationMetrics metrics =
            restricted_iteration_metrics(P, density_next, E_prev, E_total);

        const double iter_time = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - iter_start)
                                     .count();
        HartreeFock::Logger::scf_iteration(
            iter,
            E_total,
            metrics.delta_energy,
            metrics.delta_density_rms,
            metrics.delta_density_max,
            diis_err,
            0.0,
            iter_time);

        P = density_next;
        E_prev = E_total;

        store_restricted_iteration(
            calculator,
            RestrictedIterationData{
                .density = P,
                .fock = F,
                .mo_energies = eps,
                .mo_coefficients = C,
                .electronic_energy = E_elec,
                .total_energy = E_total},
            metrics);

        if (is_converged(calculator._scf, metrics, iter))
        {
            calculator._info._scf.alpha.mo_energies = eps;
            calculator._info._scf.alpha.mo_coefficients = C;
            calculator._info._is_converged = true;

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
    const Eigen::MatrixXd &Ca,
    const Eigen::MatrixXd &Cb,
    const Eigen::MatrixXd &S,
    int n_alpha, int n_beta,
    unsigned int multiplicity)
{
    // <S^2> = Sz*(Sz+1) + N_beta - ||C_alpha_occ^T S C_beta_occ||_F^2
    const double Sz = 0.5 * static_cast<double>(n_alpha - n_beta);
    const Eigen::MatrixXd OV = Ca.leftCols(n_alpha).transpose() * S * Cb.leftCols(n_beta);
    const double S2 = Sz * (Sz + 1.0) + static_cast<double>(n_beta) - OV.squaredNorm();
    const double S_exact = 0.5 * static_cast<double>(multiplicity - 1);
    const double S2_exact = S_exact * (S_exact + 1.0);

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "<S^2> :",
                                 std::format("{:.6f}  (exact: {:.6f})", S2, S2_exact));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "<S>   :",
                                 std::format("{:.6f}", std::sqrt(std::max(0.0, S2))));
}

// ─── UHF SCF ─────────────────────────────────────────────────────────────────

std::expected<void, std::string> HartreeFock::SCF::run_uhf(
    HartreeFock::Calculator &calculator,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const HartreeFock::Solvation::PCMState *pcm)
{
    const Eigen::MatrixXd &S = calculator._overlap;
    const Eigen::MatrixXd &H = calculator._hcore;
    const std::size_t nbasis = calculator._shells.nbasis();

    const int n_electrons = static_cast<int>(
        calculator._molecule.atomic_numbers.cast<int>().sum() - calculator._molecule.charge);

    const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;

    if (n_unpaired < 0 || n_unpaired > n_electrons)
        return std::unexpected("Invalid multiplicity for given electron count");
    if ((n_electrons - n_unpaired) % 2 != 0)
        return std::unexpected("Multiplicity inconsistent with electron count parity");

    const int n_alpha = (n_electrons + n_unpaired) / 2;
    const int n_beta = (n_electrons - n_unpaired) / 2;

    // ── Orthogonalization matrix X = S^{-1/2} ────────────────────────────────
    auto X_result = build_orthogonalizer(S);
    if (!X_result)
        return std::unexpected(X_result.error());
    const Eigen::MatrixXd X = std::move(*X_result);

    // ── SAO blocking setup ────────────────────────────────────────────────────
    // Same U used for both alpha and beta (basis and geometry are spin-independent).
    // When SAO is active, Ca_orth_prev/Cb_orth_prev are never filled, so
    // shift_active stays false throughout — level shift is incompatible with SAO.
    const bool sao_active_uhf = calculator._use_sao_blocking &&
                                (calculator._sao_transform.rows() > 0);
    const Eigen::MatrixXd &U_uhf = calculator._sao_transform; // ref, no copy

    // ── Initial spin densities from core Hamiltonian ─────────────────────────
    // Factor 1.0 per spin (UHF), vs 2.0 in RHF.
    auto make_density_spin = [&](int n_occ) -> Eigen::MatrixXd
    {
        const Eigen::MatrixXd Hp = X.transpose() * H * X;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s(Hp);
        const Eigen::MatrixXd C = X * s.eigenvectors();
        return C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
    };

    // ReadDensity / ReadFull: reuse densities loaded from checkpoint.
    const bool use_chk_uhf =
        (calculator._scf._guess == HartreeFock::SCFGuess::ReadDensity ||
         calculator._scf._guess == HartreeFock::SCFGuess::ReadFull);

    Eigen::MatrixXd Pa, Pb;
    if (use_chk_uhf)
    {
        Pa = calculator._info._scf.alpha.density;
        Pb = calculator._info._scf.beta.density;
    }
    else if (calculator._scf._guess == HartreeFock::SCFGuess::SAD)
    {
        auto sad_res = HartreeFock::SCF::compute_sad_guess_open_shell(
            calculator, n_alpha, n_beta);
        if (!sad_res)
            return std::unexpected("UHF SAD guess failed: " + sad_res.error());
        Pa = std::move(sad_res->first);
        Pb = std::move(sad_res->second);
    }
    else
    {
        Pa = make_density_spin(n_alpha);
        Pb = make_density_spin(n_beta);
    }

    const unsigned int max_iter = calculator._scf.get_max_cycles(nbasis);
    const double level_shift = calculator._scf._level_shift;
    const double restart_factor = calculator._scf._diis_restart_factor;

    // ── Conventional vs Direct ────────────────────────────────────────────────
    const bool use_conventional =
        (calculator._scf._mode == HartreeFock::SCFMode::Conventional ||
         (calculator._scf._mode == HartreeFock::SCFMode::Auto &&
          nbasis <= static_cast<std::size_t>(calculator._scf._threshold)));

    std::vector<double> eri;
    if (use_conventional)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :",
                                     std::format("Building ERI tensor ({:.1f} MB)", nbasis * nbasis * nbasis * nbasis * 8.0 / 1e6));
        eri = _compute_2e(shell_pairs, nbasis, calculator._integral._engine,
                          calculator._integral._tol_eri,
                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        calculator._eri = eri; // persist for post-HF use
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    // ── DIIS state per spin ───────────────────────────────────────────────────
    HartreeFock::DIISState diis_a, diis_b;
    diis_a.max_vecs = diis_b.max_vecs = calculator._scf._DIIS_dim;
    const bool use_diis = calculator._scf._use_DIIS;

    const double tol_eri = calculator._integral._tol_eri;
    double E_prev = 0.0;
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
                            : _compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, calculator._integral._engine, tol_eri,
                                                   calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

        const Eigen::MatrixXd P_total = Pa + Pb;
        Eigen::MatrixXd V_pcm = Eigen::MatrixXd::Zero(nbasis, nbasis);
        double pcm_energy = 0.0;
        if (pcm != nullptr && pcm->enabled())
        {
            auto pcm_result = HartreeFock::Solvation::evaluate_pcm_reaction_field(calculator, *pcm, P_total);
            if (!pcm_result)
                return std::unexpected("PCM build failed inside UHF iteration: " + pcm_result.error());
            V_pcm = std::move(pcm_result->reaction_potential);
            pcm_energy = pcm_result->solvation_energy;
        }

        const Eigen::MatrixXd Fa_gas = H + Ga;
        const Eigen::MatrixXd Fb_gas = H + Gb;
        const Eigen::MatrixXd Fa = Fa_gas + V_pcm;
        const Eigen::MatrixXd Fb = Fb_gas + V_pcm;

        // ── Electronic energy — always from the bare gas-phase Fock plus PCM ──
        const double E_gas = 0.5 * ((Pa.array() * (H + Fa_gas).array()).sum() + (Pb.array() * (H + Fb_gas).array()).sum());
        const double E_elec = E_gas + pcm_energy;
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
                level_shift * X * (I_nb - Ca_orth_prev.leftCols(n_alpha) * Ca_orth_prev.leftCols(n_alpha).transpose()) * X.transpose();
            const Eigen::MatrixXd shift_b =
                level_shift * X * (I_nb - Cb_orth_prev.leftCols(n_beta) * Cb_orth_prev.leftCols(n_beta).transpose()) * X.transpose();
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
            const auto rms_norm = [](const Eigen::MatrixXd &m)
            {
                return std::sqrt(m.squaredNorm() / static_cast<double>(m.size()));
            };
            const double cur_err = std::max(rms_norm(ea), rms_norm(eb));

            // ── DIIS restart ──────────────────────────────────────────────────
            if (restart_factor > 0.0 && iter > 2 && cur_err > diis_err_prev * restart_factor)
            {
                diis_a.clear();
                diis_b.clear();
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                                             "DIIS :", std::format("Subspace restarted at iter {} (error grew {:.1f}×)", iter, cur_err / diis_err_prev));
            }

            diis_a.push(Fa_s, ea);
            diis_b.push(Fb_s, eb);
            diis_err = std::max(diis_a.error_norm(), diis_b.error_norm());
            diis_err_prev = cur_err;
        }

        // Extrapolated (shifted) Fock for diagonalization
        const Eigen::MatrixXd Fa_diag = (use_diis && diis_a.ready()) ? diis_a.extrapolate() : Fa_s;
        const Eigen::MatrixXd Fb_diag = (use_diis && diis_b.ready()) ? diis_b.extrapolate() : Fb_s;

        // ── Diagonalize alpha and beta ────────────────────────────────────────
        Eigen::MatrixXd Ca(nbasis, nbasis), Cb(nbasis, nbasis);
        Eigen::VectorXd epsa(nbasis), epsb(nbasis);

        if (!sao_active_uhf)
        {
            // ── Full AO diagonalization (original path) ───────────────────────
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sa(X.transpose() * Fa_diag * X);
            if (sa.info() != Eigen::Success)
                return std::unexpected(std::format("Alpha Fock diagonalization failed at iter {}", iter));
            Ca = X * sa.eigenvectors();
            epsa = sa.eigenvalues();
            if (shift_active)
                epsa.tail(nbasis - n_alpha).array() -= level_shift;

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sb(X.transpose() * Fb_diag * X);
            if (sb.info() != Eigen::Success)
                return std::unexpected(std::format("Beta Fock diagonalization failed at iter {}", iter));
            Cb = X * sb.eigenvectors();
            epsb = sb.eigenvalues();
            if (shift_active)
                epsb.tail(nbasis - n_beta).array() -= level_shift;

            // Save orthonormal eigenvectors for next iteration's virtual projector
            Ca_orth_prev = sa.eigenvectors();
            Cb_orth_prev = sb.eigenvectors();
        }
        else
        {
            // ── SAO block-diagonal diagonalization ────────────────────────────
            // Helper lambda: block-diagonalize one spin's Fock in SAO basis.
            // Returns {C_AO [nb×nb], eps [nb]}.  Also fills mo_sym_out with
            // sorted irrep labels.
            auto sao_diag_spin = [&](const Eigen::MatrixXd &F_diag_spin,
                                     std::vector<std::string> &mo_sym_out,
                                     const std::string &spin_tag)
                -> std::expected<std::pair<Eigen::MatrixXd, Eigen::VectorXd>, std::string>
            {
                const Eigen::MatrixXd F_sao = U_uhf.transpose() * F_diag_spin * U_uhf;
                const int n_blocks = static_cast<int>(calculator._sao_block_sizes.size());

                Eigen::VectorXd eps_sao(nbasis);
                Eigen::MatrixXd C_sao = Eigen::MatrixXd::Zero(nbasis, nbasis);
                std::vector<int> mo_irrep_idx(nbasis);

                for (int b = 0; b < n_blocks; ++b)
                {
                    const int off = calculator._sao_block_offsets[b];
                    const int ni = calculator._sao_block_sizes[b];
                    if (ni == 0)
                        continue;

                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s(
                        F_sao.block(off, off, ni, ni));
                    if (s.info() != Eigen::Success)
                        return std::unexpected(std::format(
                            "{} block Fock diagonalization failed (block {}) at iter {}",
                            spin_tag, b, iter));

                    eps_sao.segment(off, ni) = s.eigenvalues();
                    C_sao.block(off, off, ni, ni) = s.eigenvectors();
                    for (int k = 0; k < ni; ++k)
                        mo_irrep_idx[off + k] = calculator._sao_irrep_index[off + k];
                }

                // Sort MOs globally by energy
                std::vector<int> order(nbasis);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b)
                                 { return eps_sao[a] < eps_sao[b]; });

                Eigen::VectorXd eps_sorted(nbasis);
                Eigen::MatrixXd C_sao_sorted(nbasis, nbasis);
                mo_sym_out.resize(nbasis);
                for (int k = 0; k < static_cast<int>(nbasis); ++k)
                {
                    eps_sorted[k] = eps_sao[order[k]];
                    C_sao_sorted.col(k) = C_sao.col(order[k]);
                    mo_sym_out[k] = calculator._sao_irrep_names[mo_irrep_idx[order[k]]];
                }

                return std::make_pair(U_uhf * C_sao_sorted, eps_sorted);
            };

            std::vector<std::string> mo_sym_a, mo_sym_b;

            auto res_a = sao_diag_spin(Fa_diag, mo_sym_a, "Alpha");
            if (!res_a)
                return std::unexpected(res_a.error());
            Ca = std::move(res_a->first);
            epsa = std::move(res_a->second);

            auto res_b = sao_diag_spin(Fb_diag, mo_sym_b, "Beta");
            if (!res_b)
                return std::unexpected(res_b.error());
            Cb = std::move(res_b->first);
            epsb = std::move(res_b->second);

            calculator._info._scf.alpha.mo_symmetry = std::move(mo_sym_a);
            calculator._info._scf.beta.mo_symmetry = std::move(mo_sym_b);
            // Ca_orth_prev / Cb_orth_prev intentionally NOT updated →
            // shift_active stays false (level shift incompatible with SAO blocking).
        }

        // ── Next spin densities ───────────────────────────────────────────────
        const Eigen::MatrixXd density_alpha_next =
            Ca.leftCols(n_alpha) * Ca.leftCols(n_alpha).transpose();
        const Eigen::MatrixXd density_beta_next =
            Cb.leftCols(n_beta) * Cb.leftCols(n_beta).transpose();

        // ── Convergence on total density ──────────────────────────────────────
        const IterationMetrics metrics = unrestricted_iteration_metrics(
            Pa, Pb, density_alpha_next, density_beta_next, E_prev, E_total);

        const double iter_time = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - iter_start)
                                     .count();
        HartreeFock::Logger::scf_iteration(
            iter,
            E_total,
            metrics.delta_energy,
            metrics.delta_density_rms,
            metrics.delta_density_max,
            diis_err,
            0.0,
            iter_time);

        Pa = density_alpha_next;
        Pb = density_beta_next;
        E_prev = E_total;

        store_unrestricted_iteration(
            calculator,
            UnrestrictedIterationData{
                .alpha_density = Pa,
                .beta_density = Pb,
                .alpha_fock = Fa,
                .beta_fock = Fb,
                .alpha_mo_energies = epsa,
                .beta_mo_energies = epsb,
                .alpha_mo_coefficients = Ca,
                .beta_mo_coefficients = Cb,
                .electronic_energy = E_elec,
                .total_energy = E_total},
            metrics);

        if (is_converged(calculator._scf, metrics, iter))
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

// ─── ROHF helpers ────────────────────────────────────────────────────────────

static Eigen::MatrixXd _rohf_effective_fock(
    const Eigen::MatrixXd &Fa,
    const Eigen::MatrixXd &Fb,
    const Eigen::MatrixXd &Pa,
    const Eigen::MatrixXd &Pb,
    const Eigen::MatrixXd &S)
{
    const Eigen::Index nbasis = S.rows();
    const Eigen::MatrixXd Fc = 0.5 * (Fa + Fb);
    const Eigen::MatrixXd Pc = Pb * S;
    const Eigen::MatrixXd Po = (Pa - Pb) * S;
    const Eigen::MatrixXd Pv = Eigen::MatrixXd::Identity(nbasis, nbasis) - Pa * S;

    Eigen::MatrixXd F = 0.5 * Pc.transpose() * Fc * Pc;
    F += 0.5 * Po.transpose() * Fc * Po;
    F += 0.5 * Pv.transpose() * Fc * Pv;
    F += Po.transpose() * Fb * Pc;
    F += Po.transpose() * Fa * Pv;
    F += Pv.transpose() * Fc * Pc;

    return F + F.transpose();
}

static Eigen::VectorXd _mo_energy_diagonal(
    const Eigen::MatrixXd &C,
    const Eigen::MatrixXd &F)
{
    return (C.transpose() * F * C).diagonal();
}

static void _reorder_rohf_orbitals(
    Eigen::MatrixXd &C,
    Eigen::VectorXd &eps,
    Eigen::VectorXd &eps_alpha,
    Eigen::VectorXd &eps_beta,
    std::vector<std::string> &mo_sym,
    int n_closed,
    int n_open)
{
    const int nmo = static_cast<int>(eps.size());
    if (n_open <= 0 || n_closed < 0 || n_closed + n_open > nmo)
        return;

    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(nmo));
    for (int i = 0; i < n_closed; ++i)
        order.push_back(i);

    std::vector<int> candidates;
    candidates.reserve(static_cast<std::size_t>(nmo - n_closed));
    for (int i = n_closed; i < nmo; ++i)
        candidates.push_back(i);

    std::stable_sort(candidates.begin(), candidates.end(),
                     [&](int a, int b)
                     { return eps_alpha[a] < eps_alpha[b]; });

    std::vector<char> selected(static_cast<std::size_t>(nmo), 0);
    for (int k = 0; k < n_open; ++k)
    {
        const int idx = candidates[static_cast<std::size_t>(k)];
        selected[static_cast<std::size_t>(idx)] = 1;
        order.push_back(idx);
    }

    std::vector<int> virtuals;
    for (int i = n_closed; i < nmo; ++i)
        if (!selected[static_cast<std::size_t>(i)])
            virtuals.push_back(i);
    std::stable_sort(virtuals.begin(), virtuals.end(),
                     [&](int a, int b)
                     { return eps[a] < eps[b]; });
    order.insert(order.end(), virtuals.begin(), virtuals.end());

    Eigen::MatrixXd C_sorted(C.rows(), C.cols());
    Eigen::VectorXd eps_sorted(nmo), epsa_sorted(nmo), epsb_sorted(nmo);
    std::vector<std::string> sym_sorted;
    if (!mo_sym.empty())
        sym_sorted.resize(static_cast<std::size_t>(nmo));

    for (int k = 0; k < nmo; ++k)
    {
        const int src = order[static_cast<std::size_t>(k)];
        C_sorted.col(k) = C.col(src);
        eps_sorted[k] = eps[src];
        epsa_sorted[k] = eps_alpha[src];
        epsb_sorted[k] = eps_beta[src];
        if (!mo_sym.empty() && src < static_cast<int>(mo_sym.size()))
            sym_sorted[static_cast<std::size_t>(k)] = mo_sym[static_cast<std::size_t>(src)];
    }

    C = std::move(C_sorted);
    eps = std::move(eps_sorted);
    eps_alpha = std::move(epsa_sorted);
    eps_beta = std::move(epsb_sorted);
    if (!mo_sym.empty())
        mo_sym = std::move(sym_sorted);
}

// ─── ROHF SCF ────────────────────────────────────────────────────────────────

std::expected<void, std::string> HartreeFock::SCF::run_rohf(
    HartreeFock::Calculator &calculator,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const HartreeFock::Solvation::PCMState *pcm)
{
    (void)pcm;
    const Eigen::MatrixXd &S = calculator._overlap;
    const Eigen::MatrixXd &H = calculator._hcore;
    const std::size_t nbasis = calculator._shells.nbasis();

    const int n_electrons = static_cast<int>(
        calculator._molecule.atomic_numbers.cast<int>().sum() - calculator._molecule.charge);
    const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;

    if (n_unpaired < 0 || n_unpaired > n_electrons)
        return std::unexpected("Invalid multiplicity for given electron count");
    if ((n_electrons - n_unpaired) % 2 != 0)
        return std::unexpected("Multiplicity inconsistent with electron count parity");

    const int n_alpha = (n_electrons + n_unpaired) / 2;
    const int n_beta = (n_electrons - n_unpaired) / 2;
    const int n_closed = n_beta;
    const int n_open = n_alpha - n_beta;

    if (n_open < 0)
        return std::unexpected("ROHF requires n_alpha >= n_beta");

    auto X_result = build_orthogonalizer(S);
    if (!X_result)
        return std::unexpected(X_result.error());
    const Eigen::MatrixXd X = std::move(*X_result);

    const bool sao_active = calculator._use_sao_blocking &&
                            (calculator._sao_transform.rows() > 0);
    const Eigen::MatrixXd &U = calculator._sao_transform;

    auto diagonalize_common = [&](const Eigen::MatrixXd &F_diag,
                                  std::vector<std::string> &mo_sym)
        -> std::expected<std::pair<Eigen::MatrixXd, Eigen::VectorXd>, std::string>
    {
        if (!sao_active)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(X.transpose() * F_diag * X);
            if (solver.info() != Eigen::Success)
                return std::unexpected("ROHF Fock diagonalization failed");
            return std::make_pair(X * solver.eigenvectors(), solver.eigenvalues());
        }

        const Eigen::MatrixXd F_sao = U.transpose() * F_diag * U;
        const int n_blocks = static_cast<int>(calculator._sao_block_sizes.size());

        Eigen::VectorXd eps_sao(nbasis);
        Eigen::MatrixXd C_sao = Eigen::MatrixXd::Zero(nbasis, nbasis);
        std::vector<int> mo_irrep_idx(nbasis);

        for (int b = 0; b < n_blocks; ++b)
        {
            const int off = calculator._sao_block_offsets[static_cast<std::size_t>(b)];
            const int ni = calculator._sao_block_sizes[static_cast<std::size_t>(b)];
            if (ni == 0)
                continue;

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sb(
                F_sao.block(off, off, ni, ni));
            if (sb.info() != Eigen::Success)
                return std::unexpected(std::format(
                    "ROHF block Fock diagonalization failed (block {})", b));

            eps_sao.segment(off, ni) = sb.eigenvalues();
            C_sao.block(off, off, ni, ni) = sb.eigenvectors();
            for (int k = 0; k < ni; ++k)
                mo_irrep_idx[off + k] = calculator._sao_irrep_index[off + k];
        }

        std::vector<int> order(nbasis);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b)
                         { return eps_sao[a] < eps_sao[b]; });

        Eigen::VectorXd eps_sorted(nbasis);
        Eigen::MatrixXd C_sao_sorted(nbasis, nbasis);
        mo_sym.resize(nbasis);
        for (int k = 0; k < static_cast<int>(nbasis); ++k)
        {
            eps_sorted[k] = eps_sao[order[k]];
            C_sao_sorted.col(k) = C_sao.col(order[k]);
            mo_sym[static_cast<std::size_t>(k)] =
                calculator._sao_irrep_names[mo_irrep_idx[order[k]]];
        }

        return std::make_pair(U * C_sao_sorted, eps_sorted);
    };

    const bool use_chk_density =
        (calculator._scf._guess == HartreeFock::SCFGuess::ReadDensity ||
         calculator._scf._guess == HartreeFock::SCFGuess::ReadFull);

    Eigen::MatrixXd Pa, Pb;
    if (use_chk_density)
    {
        Pa = calculator._info._scf.alpha.density;
        Pb = calculator._info._scf.beta.density;
        if (Pa.rows() != static_cast<Eigen::Index>(nbasis) ||
            Pb.rows() != static_cast<Eigen::Index>(nbasis))
            return std::unexpected("ROHF checkpoint density is missing alpha/beta spin channels");
    }
    else if (calculator._scf._guess == HartreeFock::SCFGuess::SAD)
    {
        auto sad_res = HartreeFock::SCF::compute_sad_guess_open_shell(
            calculator, n_alpha, n_beta);
        if (!sad_res)
            return std::unexpected("ROHF SAD guess failed: " + sad_res.error());
        Pa = std::move(sad_res->first);
        Pb = std::move(sad_res->second);
    }
    else
    {
        std::vector<std::string> initial_sym;
        auto init = diagonalize_common(H, initial_sym);
        if (!init)
            return std::unexpected(init.error());
        const Eigen::MatrixXd &C0 = init->first;
        Pa = C0.leftCols(n_alpha) * C0.leftCols(n_alpha).transpose();
        Pb = C0.leftCols(n_beta) * C0.leftCols(n_beta).transpose();
    }

    const unsigned int max_iter = calculator._scf.get_max_cycles(nbasis);

    const bool use_conventional =
        (calculator._scf._mode == HartreeFock::SCFMode::Conventional ||
         (calculator._scf._mode == HartreeFock::SCFMode::Auto &&
          nbasis <= static_cast<std::size_t>(calculator._scf._threshold)));

    std::vector<double> eri;
    if (use_conventional)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :",
                                     std::format("Building ERI tensor ({:.1f} MB)", nbasis * nbasis * nbasis * nbasis * 8.0 / 1e6));
        eri = _compute_2e(shell_pairs, nbasis, calculator._integral._engine,
                          calculator._integral._tol_eri,
                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        calculator._eri = eri;
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    HartreeFock::DIISState diis;
    diis.max_vecs = calculator._scf._DIIS_dim;
    const bool use_diis = calculator._scf._use_DIIS;

    const double tol_eri = calculator._integral._tol_eri;
    double E_prev = 0.0;

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; ++iter)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        auto [Ga, Gb] = use_conventional
                            ? HartreeFock::ObaraSaika::_compute_fock_uhf(eri, Pa, Pb, nbasis)
                            : _compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, calculator._integral._engine, tol_eri,
                                                   calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

        const Eigen::MatrixXd Fa = H + Ga;
        const Eigen::MatrixXd Fb = H + Gb;
        const Eigen::MatrixXd F_rohf = _rohf_effective_fock(Fa, Fb, Pa, Pb, S);

        const double E_elec = 0.5 * ((Pa.array() * (H + Fa).array()).sum() +
                                     (Pb.array() * (H + Fb).array()).sum());
        const double E_total = E_elec + calculator._nuclear_repulsion;

        double diis_err = 0.0;
        if (use_diis)
        {
            const Eigen::MatrixXd P_total = Pa + Pb;
            const Eigen::MatrixXd e = X.transpose() * (F_rohf * P_total * S - S * P_total * F_rohf) * X;
            diis.push(F_rohf, e);
            diis_err = diis.error_norm();
        }

        const Eigen::MatrixXd F_diag = (use_diis && diis.ready()) ? diis.extrapolate() : F_rohf;

        std::vector<std::string> mo_sym;
        auto diag = diagonalize_common(F_diag, mo_sym);
        if (!diag)
            return std::unexpected(std::format("{} at iteration {}", diag.error(), iter));

        Eigen::MatrixXd C = std::move(diag->first);
        Eigen::VectorXd eps = std::move(diag->second);
        Eigen::VectorXd epsa = _mo_energy_diagonal(C, Fa);
        Eigen::VectorXd epsb = _mo_energy_diagonal(C, Fb);

        _reorder_rohf_orbitals(C, eps, epsa, epsb, mo_sym, n_closed, n_open);

        const Eigen::MatrixXd density_alpha_next =
            C.leftCols(n_alpha) * C.leftCols(n_alpha).transpose();
        const Eigen::MatrixXd density_beta_next =
            C.leftCols(n_beta) * C.leftCols(n_beta).transpose();

        const IterationMetrics metrics = unrestricted_iteration_metrics(
            Pa, Pb, density_alpha_next, density_beta_next, E_prev, E_total);

        const double iter_time = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - iter_start)
                                     .count();
        HartreeFock::Logger::scf_iteration(
            iter,
            E_total,
            metrics.delta_energy,
            metrics.delta_density_rms,
            metrics.delta_density_max,
            diis_err,
            0.0,
            iter_time);

        Pa = density_alpha_next;
        Pb = density_beta_next;
        E_prev = E_total;

        calculator._info._scf.alpha.mo_symmetry = mo_sym;
        calculator._info._scf.beta.mo_symmetry = mo_sym;

        store_unrestricted_iteration(
            calculator,
            UnrestrictedIterationData{
                .alpha_density = Pa,
                .beta_density = Pb,
                .alpha_fock = Fa,
                .beta_fock = Fb,
                .alpha_mo_energies = eps,
                .beta_mo_energies = epsb,
                .alpha_mo_coefficients = C,
                .beta_mo_coefficients = C,
                .electronic_energy = E_elec,
                .total_energy = E_total},
            metrics);

        if (is_converged(calculator._scf, metrics, iter))
        {
            calculator._info._is_converged = true;

            HartreeFock::Logger::scf_footer();
            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "ROHF Converged :",
                                         std::format("E = {:.10f} Eh  after {} iterations", E_total, iter));
            HartreeFock::Logger::blank();

            _log_spin_contamination(C, C, S, n_alpha, n_beta,
                                    calculator._molecule.multiplicity);
            HartreeFock::Logger::blank();

            return {};
        }
    }

    return std::unexpected(std::format("ROHF SCF did not converge in {} iterations", max_iter));
}
