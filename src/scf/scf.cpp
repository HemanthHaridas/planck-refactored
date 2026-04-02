#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>

#include "integrals/base.h"
#include "io/logging.h"
#include "scf.h"

namespace
{
    constexpr double RHF_DEGENERACY_TOL = 1e-6;

    struct RestrictedOccupations
    {
        Eigen::VectorXd values;
        int frontier_start = -1; // inclusive, 0-based
        int frontier_end = -1;   // exclusive, 0-based

        [[nodiscard]] bool has_fractional_frontier() const noexcept
        {
            return frontier_start >= 0 && frontier_end > frontier_start;
        }
    };

    RestrictedOccupations build_restricted_occupations(const Eigen::VectorXd &eps, int n_electrons)
    {
        const int nmo = static_cast<int>(eps.size());

        RestrictedOccupations result;
        result.values = Eigen::VectorXd::Zero(nmo);

        int electrons_left = n_electrons;
        int first = 0;
        while (first < nmo && electrons_left > 0)
        {
            int last = first + 1;
            while (last < nmo &&
                   std::abs(eps(last) - eps(first)) <= RHF_DEGENERACY_TOL)
                ++last;

            const int block_size = last - first;
            const int block_capacity = 2 * block_size;
            if (electrons_left >= block_capacity)
            {
                result.values.segment(first, block_size).setConstant(2.0);
                electrons_left -= block_capacity;
            }
            else
            {
                result.values.segment(first, block_size)
                    .setConstant(static_cast<double>(electrons_left) / block_size);
                if (electrons_left > 0 && electrons_left < block_capacity)
                {
                    result.frontier_start = first;
                    result.frontier_end = last;
                }
                electrons_left = 0;
            }

            first = last;
        }

        return result;
    }

    Eigen::MatrixXd density_from_restricted_orbitals(const Eigen::MatrixXd &C,
                                                     const Eigen::VectorXd &occupations)
    {
        return C * occupations.asDiagonal() * C.transpose();
    }
} // namespace

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

Eigen::MatrixXd HartreeFock::SCF::initial_density(const Eigen::MatrixXd &H, const Eigen::MatrixXd &X, int n_electrons)
{
    // Transform H to orthonormal basis: H' = X^T * H * X
    const Eigen::MatrixXd Hprime = X.transpose() * H * X;

    // Diagonalize H'
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hprime);
    const Eigen::MatrixXd C = X * solver.eigenvectors();
    const RestrictedOccupations occupations =
        build_restricted_occupations(solver.eigenvalues(), n_electrons);
    return density_from_restricted_orbitals(C, occupations.values);
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

std::expected<void, std::string> HartreeFock::SCF::run_rhf(HartreeFock::Calculator &calculator, const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    const Eigen::MatrixXd &S = calculator._overlap;
    const Eigen::MatrixXd &H = calculator._hcore;
    const std::size_t nbasis = calculator._shells.nbasis();

    // Number of occupied orbitals (closed shell singlet assumed)
    const int n_electrons = static_cast<int>(
        (calculator._molecule.atomic_numbers.cast<int>().sum()) - calculator._molecule.charge);

    if (n_electrons % 2 != 0)
        return std::unexpected("RHF requires an even number of electrons (closed shell)");

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
    Eigen::MatrixXd P = use_chk_density
                            ? calculator._info._scf.alpha.density
                            : initial_density(H, X, n_electrons);

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
    bool logged_fractional_frontier = false;

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; iter++)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // ── Build two-electron contribution G = J - 0.5*K ────────────────────
        Eigen::MatrixXd G = use_conventional
                                ? HartreeFock::ObaraSaika::_compute_fock_rhf(eri, P, nbasis)
                                : _compute_2e_fock(shell_pairs, P, nbasis, calculator._integral._engine, tol_eri,
                                                   calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

        // ── Fock matrix ───────────────────────────────────────────────────────
        const Eigen::MatrixXd F = H + G;

        // ── Electronic energy  E = 0.5 * tr(P * (H + F)) ────────────────────
        // Always computed from the raw (non-extrapolated) Fock matrix.
        const double E_elec = 0.5 * (P.array() * (H + F).array()).sum();
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

        const RestrictedOccupations occupations =
            build_restricted_occupations(eps, n_electrons);
        if (occupations.has_fractional_frontier() && !logged_fractional_frontier)
        {
            const double frontier_electrons =
                occupations.values.segment(
                    occupations.frontier_start,
                    occupations.frontier_end - occupations.frontier_start)
                    .sum();
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "RHF Occupations :",
                std::format("Averaging {:.1f} electrons over degenerate frontier orbitals {}-{}",
                            frontier_electrons,
                            occupations.frontier_start + 1,
                            occupations.frontier_end));
            logged_fractional_frontier = true;
        }

        // ── New density P_new ─────────────────────────────────────────────────
        const Eigen::MatrixXd P_new =
            density_from_restricted_orbitals(C, occupations.values);

        // ── Convergence checks ────────────────────────────────────────────────
        const IterationMetrics metrics =
            restricted_iteration_metrics(P, P_new, E_prev, E_total);

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

        P = P_new;
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
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
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

    Eigen::MatrixXd Pa = use_chk_uhf
                             ? calculator._info._scf.alpha.density
                             : make_density_spin(n_alpha);
    Eigen::MatrixXd Pb = use_chk_uhf
                             ? calculator._info._scf.beta.density
                             : make_density_spin(n_beta);

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

        const Eigen::MatrixXd Fa = H + Ga;
        const Eigen::MatrixXd Fb = H + Gb;

        // ── Electronic energy — always from the bare (unshifted) Fock ───────────
        const double E_elec = 0.5 * ((Pa.array() * (H + Fa).array()).sum() + (Pb.array() * (H + Fb).array()).sum());
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

        // ── New spin densities ────────────────────────────────────────────────
        const Eigen::MatrixXd Pa_new = Ca.leftCols(n_alpha) * Ca.leftCols(n_alpha).transpose();
        const Eigen::MatrixXd Pb_new = Cb.leftCols(n_beta) * Cb.leftCols(n_beta).transpose();

        // ── Convergence on total density ──────────────────────────────────────
        const IterationMetrics metrics = unrestricted_iteration_metrics(
            Pa, Pb, Pa_new, Pb_new, E_prev, E_total);

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

        Pa = Pa_new;
        Pb = Pb_new;
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
