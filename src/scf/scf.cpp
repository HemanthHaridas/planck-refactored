#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <limits>
#include <numeric>
#include <string_view>
#include <tuple>

#include "base/mpi_env.h"
#include "basis/spherical.h"
#include "integrals/base.h"
#include "io/logging.h"
#include "post_hf/casscf/aug-hessian.h"
#include "post_hf/casscf/orbital.h"
#include "post_hf/rhf_response.h"
#include "post_hf/uhf_response.h"
#include "sad.h"
#include "scf.h"
#include "symmetry/fock_symmetrization.h"
#include "symmetry/hgp_symm.h"
#include "symmetry/os_symm.h"
#include "symmetry/rys_symm.h"
#include "symmetry/skeleton_eri.h" // contract_symm_fock_* (C1 persisted-skeleton path)

namespace
{
    // Direct-mode two-electron build in spherical mode.
    //
    // The per-quartet builder (_compute_2e_fock / _uhf) works in the Cartesian basis,
    // but in spherical mode the density and the Fock are spherical-dimensioned. We
    // back-project the spherical density to Cartesian (P_cart = C^T P_sph C), build the
    // Cartesian G, then forward-transform (G_sph = C G_cart C^T). G = J − ½K is linear
    // in the density and built from the same Cartesian ERIs the conventional path
    // contracts, so this reproduces the conventional spherical G exactly (validated
    // against the conventional spherical energy). The contamination subspace discarded
    // by C never enters the spherical G because both the density it is built from and
    // the result are projected through C. RHF helper; the UHF variant mirrors it inline.
    template <typename BuildCartFock>
    Eigen::MatrixXd spherical_direct_fock(
        const Eigen::MatrixXd &C,        // [n_sph × n_cart] S-normalized transform
        const Eigen::MatrixXd &P_sph,    // [n_sph × n_sph] spherical density
        BuildCartFock &&build_cart_fock) // (P_cart) -> G_cart [n_cart × n_cart]
    {
        const Eigen::MatrixXd P_cart = C.transpose() * P_sph * C;
        const Eigen::MatrixXd G_cart = build_cart_fock(P_cart);
        return C * G_cart * C.transpose();
    }

    // ── Full-symmetry direct Fock dispatch (SAO-blocked direct SCF) ───────────────
    // Routes the RHF / UHF skeleton+symmetrization Fock to the requested engine
    // (os_symm / rys_symm). See docs/FULL_SYMMETRY_ERI_DESIGN.md §8. The caller
    // gates on calculator._use_full_symmetry && sao_active && direct; this helper
    // then dispatches internally to the Cartesian or spherical full-symmetry path.
    std::expected<Eigen::MatrixXd, std::string> full_symmetry_fock_rhf(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Calculator &calc,
        const Eigen::MatrixXd &P, std::size_t nbasis, double tol_eri)
    {
        const auto engine = calc._integral._engine;
        if (calc._shells._spherical)
        {
            // Spherical mode (Step 2): skeleton built over Cartesian quartets, the
            // tensor transformed to spherical, contracted with the spherical density P,
            // symmetrized with the spherical O_R. `nbasis` here is the spherical AO
            // count; the engine needs the Cartesian count separately.
            const std::size_t nbasis_cart = calc._shells.nbasis();
            const Eigen::MatrixXd &C = calc._shells._cart_to_sph;
            if (engine == HartreeFock::IntegralMethod::RysQuadrature)
                return HartreeFock::RysQuad::_compute_2e_fock_symm_spherical(
                    shell_pairs, calc._shells, P, nbasis_cart, C, calc._group_operations,
                    HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
            if (engine == HartreeFock::IntegralMethod::HeadGordonPople)
                return HartreeFock::HeadGordonPople::_compute_2e_fock_symm_spherical(
                    shell_pairs, calc._shells, P, nbasis_cart, C, calc._group_operations,
                    HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
            return HartreeFock::ObaraSaika::_compute_2e_fock_symm_spherical(
                shell_pairs, calc._shells, P, nbasis_cart, C, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        }
        if (engine == HartreeFock::IntegralMethod::RysQuadrature)
            return HartreeFock::RysQuad::_compute_2e_fock_symm(
                shell_pairs, calc._shells, P, nbasis, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        if (engine == HartreeFock::IntegralMethod::HeadGordonPople)
            return HartreeFock::HeadGordonPople::_compute_2e_fock_symm(
                shell_pairs, calc._shells, P, nbasis, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        return HartreeFock::ObaraSaika::_compute_2e_fock_symm(
            shell_pairs, calc._shells, P, nbasis, calc._group_operations,
            HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    full_symmetry_fock_uhf(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Calculator &calc,
        const Eigen::MatrixXd &Pa, const Eigen::MatrixXd &Pb,
        std::size_t nbasis, double tol_eri)
    {
        const auto engine = calc._integral._engine;
        if (calc._shells._spherical)
        {
            const std::size_t nbasis_cart = calc._shells.nbasis();
            const Eigen::MatrixXd &C = calc._shells._cart_to_sph;
            if (engine == HartreeFock::IntegralMethod::RysQuadrature)
                return HartreeFock::RysQuad::_compute_2e_fock_uhf_symm_spherical(
                    shell_pairs, calc._shells, Pa, Pb, nbasis_cart, C, calc._group_operations,
                    HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
            if (engine == HartreeFock::IntegralMethod::HeadGordonPople)
                return HartreeFock::HeadGordonPople::_compute_2e_fock_uhf_symm_spherical(
                    shell_pairs, calc._shells, Pa, Pb, nbasis_cart, C, calc._group_operations,
                    HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
            return HartreeFock::ObaraSaika::_compute_2e_fock_uhf_symm_spherical(
                shell_pairs, calc._shells, Pa, Pb, nbasis_cart, C, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        }
        if (engine == HartreeFock::IntegralMethod::RysQuadrature)
            return HartreeFock::RysQuad::_compute_2e_fock_uhf_symm(
                shell_pairs, calc._shells, Pa, Pb, nbasis, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        if (engine == HartreeFock::IntegralMethod::HeadGordonPople)
            return HartreeFock::HeadGordonPople::_compute_2e_fock_uhf_symm(
                shell_pairs, calc._shells, Pa, Pb, nbasis, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        return HartreeFock::ObaraSaika::_compute_2e_fock_uhf_symm(
            shell_pairs, calc._shells, Pa, Pb, nbasis, calc._group_operations,
            HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
    }

    // ── C1: persisted-skeleton full-symmetry Fock (docs/FULL_SYMMETRY_PERF_SCOPE.md) ─
    // The skeleton ERI is density-independent, so it is built ONCE before the SCF loop
    // (full_symmetry_build_skeleton) and contracted each iteration against the current
    // density (full_symmetry_contract_{rhf,uhf}). This is bit-identical to calling
    // full_symmetry_fock_{rhf,uhf} every iteration (proven by planck-symm-fock-
    // equivalence) — only WHEN the skeleton is built changes. The skeleton is always
    // Cartesian-sized; in spherical mode the contraction does the cart→sph transform.

    // Returns the per-quartet count the skeleton will occupy: nbasis_cart⁴ doubles.
    std::size_t full_symmetry_skeleton_doubles(const HartreeFock::Calculator &calc)
    {
        const std::size_t nbc = calc._shells.nbasis(); // Cartesian AO count
        return nbc * nbc * nbc * nbc;
    }

    // Build the orbit-weighted Cartesian skeleton once (engine per _integral._engine).
    std::expected<std::vector<double>, std::string> full_symmetry_build_skeleton(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Calculator &calc, double tol_eri)
    {
        const auto engine = calc._integral._engine;
        const std::size_t nbasis_cart = calc._shells.nbasis();
        if (engine == HartreeFock::IntegralMethod::RysQuadrature)
            return HartreeFock::RysQuad::_build_skeleton_eri_symm(
                shell_pairs, calc._shells, nbasis_cart, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        if (engine == HartreeFock::IntegralMethod::HeadGordonPople)
            return HartreeFock::HeadGordonPople::_build_skeleton_eri_symm(
                shell_pairs, calc._shells, nbasis_cart, calc._group_operations,
                HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
        return HartreeFock::ObaraSaika::_build_skeleton_eri_symm(
            shell_pairs, calc._shells, nbasis_cart, calc._group_operations,
            HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri);
    }

    // Contract a persisted skeleton with the current RHF density (Cartesian or
    // spherical mode), matching full_symmetry_fock_rhf's dispatch.
    std::expected<Eigen::MatrixXd, std::string> full_symmetry_contract_rhf(
        const std::vector<double> &skeleton, const HartreeFock::Calculator &calc,
        const Eigen::MatrixXd &P, std::size_t nbasis)
    {
        const auto &ops = calc._group_operations;
        const bool use_sym = ops.valid && ops.operations.size() > 1;
        if (calc._shells._spherical)
        {
            const std::size_t nbasis_cart = calc._shells.nbasis();
            return HartreeFock::Symmetry::contract_symm_fock_rhf_spherical(
                skeleton, nbasis_cart, calc._shells._cart_to_sph, P, ops, use_sym);
        }
        return HartreeFock::Symmetry::contract_symm_fock_rhf(skeleton, nbasis, P, ops, use_sym);
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    full_symmetry_contract_uhf(
        const std::vector<double> &skeleton, const HartreeFock::Calculator &calc,
        const Eigen::MatrixXd &Pa, const Eigen::MatrixXd &Pb, std::size_t nbasis)
    {
        const auto &ops = calc._group_operations;
        const bool use_sym = ops.valid && ops.operations.size() > 1;
        if (calc._shells._spherical)
        {
            const std::size_t nbasis_cart = calc._shells.nbasis();
            return HartreeFock::Symmetry::contract_symm_fock_uhf_spherical(
                skeleton, nbasis_cart, calc._shells._cart_to_sph, Pa, Pb, ops, use_sym);
        }
        // Cartesian contract returns a bare pair (no failure mode); wrap as expected.
        return HartreeFock::Symmetry::contract_symm_fock_uhf(skeleton, nbasis, Pa, Pb, ops, use_sym);
    }

    // Verify a density is symmetry-adapted — the contract the skeleton+symmetrization
    // Fock requires. The density is a CONTRAVARIANT object in the (non-orthonormal)
    // AO basis, so the correct invariance is O_R P O_Rᵀ == P, NOT O_Rᵀ P O_R == P
    // (the covariant/operator law symmetrize_matrix applies). The two coincide only
    // for orthogonal O_R (s,p shells); for Cartesian d (and higher) under a non-
    // monomial operation (C₃, S₄, …) O_R is not orthogonal and the laws differ —
    // checking the covariant law there falsely rejects a perfectly symmetric SCF
    // density. Returns max deviation max_R ‖O_R P O_Rᵀ − P‖.
    //
    // The SAO basis (built in build_sao_basis with the correct AO representation —
    // metric-corrected in spherical mode) yields a density satisfying this contract
    // for the molecules whose ground state is fully symmetric. Used as a cheap loud
    // iteration-1 check so any violation fails visibly instead of silently corrupting
    // the energy.
    double density_symmetry_deviation(const Eigen::MatrixXd &P,
                                      const HartreeFock::Symmetry::GroupOperations &ops)
    {
        if (!ops.valid || ops.operations.empty())
            return std::numeric_limits<double>::infinity();
        double dev = 0.0;
        for (const auto &op : ops.operations)
        {
            if (op.matrix.rows() != P.rows() || op.matrix.cols() != P.cols())
                return std::numeric_limits<double>::infinity();
            dev = std::max(dev,
                           (op.matrix * P * op.matrix.transpose() - P).cwiseAbs().maxCoeff());
        }
        return dev;
    }

    Eigen::MatrixXd make_level_shift_matrix(
        const Eigen::MatrixXd &S,
        const Eigen::MatrixXd &density,
        double level_shift)
    {
        // Match PySCF's AO-metric level shift: raise the current virtual space
        // through S - S D S instead of building a projector from the previous
        // orthonormal orbitals. That keeps SAO and full-AO paths on the same
        // SCF surface and avoids branch-dependent basins of attraction.
        return level_shift * (S - S * density * S);
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
    // DIIS can extrapolate a Fock whose diagonalized density exactly
    // reproduces the previous one (ΔP → 0) while the DIIS residual FPS-SPF is
    // still large — a stalled step, not convergence. Gate on the DIIS error too
    // so we don't declare convergence in a wrong basin (seen with SAD guess on
    // lone closed-shell atoms). diis_error is 0 when DIIS is inactive, so this
    // is a no-op for non-DIIS runs. See SAD isolated-atom bug.
    const bool diis_residual_ok =
        metrics.diis_error <= 0.0 || metrics.diis_error < scf_options._tol_density;
    return iteration > 1 &&
           metrics.delta_energy < scf_options._tol_energy &&
           metrics.delta_density_rms < scf_options._tol_density &&
           metrics.delta_density_max < scf_options._tol_density &&
           diis_residual_ok;
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
    // Working AO dimension: spherical (2L+1 per shell) in spherical mode, else
    // Cartesian. S, H, and (below) the ERI tensor are all in this basis. The integral
    // engine still builds the ERI in the Cartesian basis (nbasis_cart) before it is
    // transformed. In Cartesian mode the two are equal.
    const std::size_t nbasis = calculator.working_nbasis();
    const std::size_t nbasis_cart = calculator._shells.nbasis();

    // Number of occupied orbitals (closed shell singlet assumed).
    // total_nuclear_charge() excludes ghost atoms (BSSE counterpoise), which
    // carry basis functions but no electrons.
    const int n_electrons =
        calculator._molecule.total_nuclear_charge() - calculator._molecule.charge;

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
        // The integral engine works in the Cartesian basis (nbasis_cart). In spherical
        // mode the tensor is then transformed with the same (S-normalized) C applied to
        // S/H, so every quantity SCF sees is consistently spherical. In Cartesian mode
        // nbasis_cart == nbasis and no transform is applied.
        eri = _compute_2e(shell_pairs, nbasis_cart, calculator._integral._engine,
                          HartreeFock::ERIKernel::Coulomb, 0.0,
                          calculator._integral._tol_eri,
                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

        if (calculator._shells._spherical)
        {
            auto eri_sph = HartreeFock::BasisFunctions::transform_eri_cart_to_sph(
                eri, calculator._shells._cart_to_sph, nbasis_cart);
            if (!eri_sph)
                return std::unexpected(eri_sph.error());
            eri = std::move(*eri_sph);
        }

        calculator._eri = eri; // persist for post-HF use
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    const double tol_eri = calculator._integral._tol_eri;

    // ── C1: build the full-symmetry skeleton ONCE (docs/FULL_SYMMETRY_PERF_SCOPE.md) ─
    // When the direct full-symmetry path is active, the density-independent skeleton
    // can be built before the loop and contracted each iteration. Memory gate: the
    // skeleton is nbasis_cart⁴ doubles, so persist only when the Cartesian basis size
    // is within the same cap the conventional path uses to decide an nb⁴ tensor fits
    // (`_threshold`). If not persisted, _symm_skeleton_eri stays empty and the loop
    // falls back to the per-iteration full_symmetry_fock_rhf build (unchanged path).
    calculator._symm_skeleton_eri.clear();
    if (!use_conventional && calculator._use_full_symmetry && sao_active &&
        calculator._shells.nbasis() <= static_cast<std::size_t>(calculator._scf._threshold))
    {
        auto skel = full_symmetry_build_skeleton(shell_pairs, calculator, tol_eri);
        if (!skel)
            return std::unexpected(skel.error());
        calculator._symm_skeleton_eri = std::move(*skel);
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "Full Symmetry :",
            std::format("skeleton ERI persisted across SCF iterations ({:.1f} MB)",
                        full_symmetry_skeleton_doubles(calculator) * 8.0 / 1e6));
    }

    // ── DIIS state ────────────────────────────────────────────────────────────
    HartreeFock::DIISState diis;
    diis.max_vecs = calculator._scf._DIIS_dim;
    const bool use_diis = calculator._scf._use_DIIS;
    double E_prev = 0.0;

    // SOSCF (docs/SOSCF.md, S2) reference orbitals, persisted across
    // iterations. Empty until SOSCF's first active iteration, then holds the
    // MO basis the NEXT iteration's orbital gradient/Hessian are expressed
    // in -- see the note at the SOSCF branch below for why this must be the
    // PREVIOUS iteration's C, not a fresh diagonalization of the current F.
    Eigen::MatrixXd C_soscf_prev;
    Eigen::VectorXd eps_soscf_prev;
    // S3: the iteration the SOSCF window actually started, once the
    // DIIS-error criterion fires (0 = not yet triggered). Needed because
    // with a criterion-based trigger the start iteration isn't known in
    // advance the way the fixed scf_soscf_start knob is.
    unsigned int soscf_window_start = 0;

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; iter++)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // ── Build two-electron contribution G = J - 0.5*K ────────────────────
        // Conventional: contract the (spherical, in spherical mode) ERI tensor.
        // Direct: the per-quartet builder works in Cartesian; in spherical mode the
        // density is back-projected and the result forward-transformed (see
        // spherical_direct_fock). In Cartesian mode the builder is called directly.
        Eigen::MatrixXd G;
        if (use_conventional)
        {
            G = HartreeFock::ObaraSaika::_compute_fock_rhf(eri, P, nbasis);
        }
        else if (calculator._use_full_symmetry && sao_active)
        {
            // Full point-group ERI reduction (supersedes the D2h sign-flip path:
            // full group ⊇ D2h, so the D2h ops are NOT also applied). Requires a
            // symmetry-adapted density (contravariant: O P O^T = P); the SAO basis
            // (built with the correct AO representation, Cartesian or spherical)
            // produces one. Works in BOTH Cartesian and spherical mode —
            // full_symmetry_fock_rhf dispatches to the spherical pipeline (skeleton →
            // cart→sph transform → contract → spherical symmetrize) when
            // _shells._spherical (Step 2). The iteration-1 assertion fails loudly if
            // the contract is ever violated rather than corrupting the energy.
            if (iter == 1)
            {
                const double dev = density_symmetry_deviation(P, calculator._group_operations);
                if (dev > 1e-8)
                    return std::unexpected(std::format(
                        "Full-symmetry SCF: initial density is not symmetry-adapted "
                        "(max |O P O^T - P| = {:.3e}); SAO blocking should guarantee this",
                        dev));
            }
            // C1: contract the persisted skeleton if built; else rebuild per-iteration.
            auto G_res = calculator._symm_skeleton_eri.empty()
                             ? full_symmetry_fock_rhf(shell_pairs, calculator, P, nbasis, tol_eri)
                             : full_symmetry_contract_rhf(calculator._symm_skeleton_eri, calculator, P, nbasis);
            if (!G_res)
                return std::unexpected(G_res.error());
            G = std::move(*G_res);
        }
        else if (calculator._shells._spherical)
        {
            G = spherical_direct_fock(
                calculator._shells._cart_to_sph, P,
                [&](const Eigen::MatrixXd &P_cart) {
                    return _compute_2e_fock(shell_pairs, P_cart, nbasis_cart,
                                            calculator._integral._engine,
                                            HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri,
                                            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
                });
        }
        else
        {
            G = _compute_2e_fock(shell_pairs, P, nbasis, calculator._integral._engine,
                                 HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri,
                                 calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        }

        // ponytail: phase timers, env-gated, RHF only -- the Amdahl probe for
        // MPI strong scaling (what fraction of the iteration is replicated).
        // Three clock reads; the print is behind PLANCK_PHASE_TIMING.
        const auto t_fock_end = std::chrono::steady_clock::now();

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
        // SOSCF is a transient accelerator, not a permanent replacement --
        // matching ORCA's own handoff: run a small fixed window of
        // second-order steps, then hand back to DIIS to finish. (Pure SOSCF
        // run to full convergence with no handoff DOES reach the same
        // minimum -- verified on water/6-31g and H2/6-31g -- but converges
        // linearly rather than DIIS's faster practical rate, so the
        // transient-window default remains faster in practice.)
        //
        // S3: the window's START is now criterion-based when
        // scf_soscf_diis_tol > 0 (the scope doc's explicit instruction: use
        // the DIIS error norm already computed above, not an invented
        // criterion), falling back to the fixed scf_soscf_start iteration
        // when the criterion is off (0). The window's DURATION stays the
        // fixed scf_soscf_cycles either way -- S3 only replaces the entry
        // trigger, not the transient-window design S2 established.
        const bool soscf_enabled =
            (calculator._scf._scf_soscf_diis_tol > 0.0 || calculator._scf._scf_soscf_start > 0) &&
            !sao_active && pcm == nullptr;
        if (soscf_enabled && soscf_window_start == 0)
        {
            const bool criterion_fires =
                calculator._scf._scf_soscf_diis_tol > 0.0
                    ? (use_diis && diis_err > 0.0 && diis_err < calculator._scf._scf_soscf_diis_tol &&
                       iter >= calculator._scf._scf_soscf_min_iter)
                    : (iter >= calculator._scf._scf_soscf_start);
            if (criterion_fires)
                soscf_window_start = iter;
        }
        const bool soscf_active =
            soscf_enabled && soscf_window_start > 0 &&
            iter < soscf_window_start + calculator._scf._scf_soscf_cycles &&
            C_soscf_prev.size() > 0;
        // The iteration right after the SOSCF window ends: DIIS's subspace
        // (if any survived from before SOSCF started) was built in a basis
        // several second-order steps removed from the current one, so its
        // stored (F, error) pairs are stale and would corrupt the
        // extrapolation. Clear it so DIIS restarts clean from the
        // SOSCF-accelerated point, exactly like a diis_restart trigger.
        if (soscf_window_start > 0 &&
            iter == soscf_window_start + calculator._scf._scf_soscf_cycles)
        {
            diis.clear();
        }
        const bool do_diis = use_diis && diis.ready() && !soscf_active;
        const Eigen::MatrixXd F_diag = do_diis ? diis.extrapolate() : F;

        const auto t_diis_end = std::chrono::steady_clock::now();

        // ── Diagonalize Fock matrix ───────────────────────────────────────────
        Eigen::MatrixXd C(nbasis, nbasis);
        Eigen::VectorXd eps(nbasis);

        if (soscf_active)
        {
            // ── SOSCF (docs/SOSCF.md, S2) ───────────────────────────────
            // RHF-only, fixed-iteration switch, no fallback logic, no
            // SAO/PCM coverage (S4+) -- S2's job is to prove the
            // augmented-Hessian step is correct, not to make the switch
            // smart (S3).
            //
            // This REPLACES diagonalization, it does not follow it. The
            // first two attempts at this step were both wrong, in ways only
            // an actual run caught:
            //   1. Diagonalizing F_diag (DIIS-extrapolated) for (C, eps) and
            //      then building the gradient from plain F: the two
            //      operators disagree, so the "gradient" never vanishes and
            //      the run oscillates forever.
            //   2. Diagonalizing plain F for (C, eps) and THEN computing
            //      Cᵀ F C as the gradient: Cᵀ F C is diagonal BY
            //      CONSTRUCTION (that is what diagonalizing F means), so
            //      the gradient measured this way is ~1e-14 from the very
            //      first SOSCF iteration regardless of how far from
            //      converged the run actually is -- confirmed by a real
            //      run where |g| hit machine epsilon at iteration 3 while
            //      the energy was still 5 Hartree from converged. The
            //      Newton step this produced was real but minuscule, so
            //      the run "worked" (same energy as DIIS) but took MORE
            //      iterations (39 vs 15) than plain DIIS, the opposite of
            //      the point.
            //
            // The orbital gradient/Hessian must be evaluated in the
            // PREVIOUS iteration's MO basis (C_soscf_prev/eps_soscf_prev),
            // against THIS iteration's Fock -- that pairing is what is
            // actually stationary at convergence, and it is what the
            // Newton step corrects. The result is the NEW C directly; nb
            // diagonalization of F happens on a SOSCF iteration at all.
            const int n_occ_i = static_cast<int>(n_occ);
            const int n_virt_i = static_cast<int>(nbasis) - n_occ_i;
            auto A_res = HartreeFock::Correlation::build_rhf_cphf_matrix(
                calculator, shell_pairs, C_soscf_prev, eps_soscf_prev);
            if (!A_res)
                return std::unexpected("SOSCF: " + A_res.error());
            const Eigen::MatrixXd &Amat = *A_res;

            // g_ai = F_mo(a,i), paired with Amat UNSCALED -- settled by a
            // direct finite-difference measurement against the ACTUAL RHF
            // energy E(kappa) (Step A of the systematic investigation,
            // PLANCK_SOSCF_FD_CHECK probe), not by re-deriving from PySCF's
            // source a third time. Measured, converged across h=1e-2/1e-3/1e-4:
            //   g_fd  / g(2*F_mo)     = 2.00  =>  g_true = 4*F_mo
            //   h_fd  / Amat_diagonal = 4.01  =>  H_true = 4*Amat
            // A Newton step depends only on the RATIO g/H, and 4*F_mo/(4*Amat)
            // = F_mo/Amat exactly -- so using g=F_mo against Amat unscaled
            // reproduces the true step at 1/4 the arithmetic, without ever
            // needing to touch build_rhf_cphf_matrix (which stays exactly
            // the form the MP2 Z-vector path already depends on). The two
            // earlier attempts (g=4*F_mo/Amat, ratio 4; g=2*F_mo/Amat, ratio
            // 2) were both wrong because they were derived by pattern-matching
            // PySCF's OWN (g,H) convention onto a DIFFERENT (unscaled) H --
            // matching PySCF's g alone, without also matching PySCF's H
            // scaling, does not preserve the ratio that actually matters.
            const Eigen::MatrixXd F_mo = C_soscf_prev.transpose() * F * C_soscf_prev;
            Eigen::VectorXd g(n_virt_i * n_occ_i);
            for (int a = 0; a < n_virt_i; ++a)
                for (int i = 0; i < n_occ_i; ++i)
                    g(a * n_occ_i + i) = F_mo(n_occ_i + a, i);

            // ponytail: debug probe (Step A of the systematic investigation) --
            // finite-difference verification of g/Amat against the ACTUAL RHF
            // energy E(kappa), independent of the AH solver, the trust-region
            // cap, DIIS, or iteration counting. Gated on PLANCK_SOSCF_FD_CHECK
            // so it never runs in a normal build. Only exercises the
            // conventional-ERI RHF path (this water/6-31g test case), since
            // that is the only Fock builder wired up here.
            if (std::getenv("PLANCK_SOSCF_FD_CHECK") && use_conventional)
            {
                auto energy_at_kappa = [&](const Eigen::MatrixXd &kap) -> double
                {
                    const Eigen::MatrixXd C_trial =
                        HartreeFock::Correlation::CASSCF::apply_orbital_rotation(
                            C_soscf_prev, kap, S);
                    const Eigen::MatrixXd C_occ_trial = C_trial.leftCols(n_occ_i);
                    const Eigen::MatrixXd P_trial = 2.0 * C_occ_trial * C_occ_trial.transpose();
                    const Eigen::MatrixXd G_trial =
                        HartreeFock::ObaraSaika::_compute_fock_rhf(eri, P_trial, nbasis);
                    const Eigen::MatrixXd F_gas_trial = H + G_trial;
                    return 0.5 * (P_trial.array() * (H + F_gas_trial).array()).sum();
                };

                const double E0 = energy_at_kappa(Eigen::MatrixXd::Zero(nbasis, nbasis));

                // Pick one random-ish (a,i) direction, not the full Newton
                // step -- isolates whether g/A themselves are right, before
                // asking anything about what the solver does with them.
                const int a_probe = 0, i_probe = 0;
                const int k_probe = a_probe * n_occ_i + i_probe;
                for (double h : {1e-2, 1e-3, 1e-4})
                {
                    Eigen::MatrixXd kap = Eigen::MatrixXd::Zero(nbasis, nbasis);
                    kap(n_occ_i + a_probe, i_probe) = h;
                    kap(i_probe, n_occ_i + a_probe) = -h;
                    const double Ep = energy_at_kappa(kap);
                    const double Em = energy_at_kappa(-kap);
                    const double g_fd = (Ep - Em) / (2.0 * h);
                    const double h_fd = (Ep - 2.0 * E0 + Em) / (h * h);
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info, "SOSCF[FD] :",
                        std::format(
                            "h={:.0e} 4*g_used={:.8f} g_fd={:.8f} diff={:.3e} | "
                            "4*A_used={:.8f} h_fd={:.8f} diff={:.3e}",
                            h, 4.0 * g(k_probe), g_fd, std::abs(4.0 * g(k_probe) - g_fd),
                            4.0 * Amat(k_probe, k_probe), h_fd,
                            std::abs(4.0 * Amat(k_probe, k_probe) - h_fd)));
                }
            }

            const auto h_op = [&Amat](const Eigen::VectorXd &x) -> Eigen::VectorXd
            { return Amat * x; };
            const auto g_op = [&g]() -> Eigen::VectorXd
            { return g; };

            // Step B of the linear-vs-quadratic-convergence investigation:
            // aug-hessian.h's ah_start_tol default (2.5, PySCF's own CASSCF
            // tuning) is a FIXED absolute residual threshold. CASSCF's own
            // orbital gradients run O(1-10), so 2.5 is a meaningful "not
            // converged enough yet" bar there; RHF SOSCF's |g| is typically
            // O(0.001-1), so the same constant is satisfied after exactly
            // ONE Krylov iteration every single time (measured directly:
            // ah_iters=1 at every SOSCF call on water/6-31g, regardless of
            // how far from converged the run was). That is why pure SOSCF
            // converged only LINEARLY (dE ratio ~0.89/iteration, ~290
            // iterations to match DIIS) instead of Newton's expected
            // superlinear rate: every "Newton step" was actually a
            // single-vector Krylov approximation, never refined. Scaling
            // the tolerance to |g| itself restores the expected rate --
            // measured, pure SOSCF now reaches DIIS's converged energy to
            // all 10 digits by iteration 8 instead of iteration ~290.
            HartreeFock::Correlation::CASSCF::AugHessianOptions ah_opts;
            ah_opts.ah_start_tol = std::max(1e-8, 0.1 * g.norm());
            Eigen::VectorXd x0 = -g;
            const double x0_norm = x0.norm();
            if (std::isfinite(x0_norm) && x0_norm > 0.0)
                x0 /= x0_norm;
            const HartreeFock::Correlation::CASSCF::AugHessianResult ah =
                HartreeFock::Correlation::CASSCF::solve_augmented_hessian(
                    h_op, g_op, nullptr, x0, ah_opts);

            // Trust-region cap: EVERY CASSCF caller of solve_augmented_hessian
            // caps the returned step (cap_packed_step / cap_step_norm at
            // max_rot) before applying it -- the AH solver's own docstring
            // notes the Krylov subspace can produce a large step where the
            // local quadratic model is a poor approximation to the true
            // surface, which is exactly early-iteration RHF far from
            // convergence. Omitting the cap here let iterations 6-10 take
            // unboundedly large uncapped rotations (|g| observed growing
            // 0.04 -> 29 across five iterations instead of shrinking) --
            // caught only by watching the eigenvalue estimate diverge more
            // negative every step, never by a static code read.
            constexpr double kSoscfMaxRot = 0.20; // matches CASSCF's mcscf_max_rot default
            Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);
            bool cap_fired = false;
            double raw_max_elem = 0.0;
            if (ah.x.size() == n_virt_i * n_occ_i && ah.x.allFinite())
            {
                Eigen::VectorXd step = ah.x;
                const double max_elem = step.cwiseAbs().maxCoeff();
                raw_max_elem = max_elem;
                if (max_elem > kSoscfMaxRot)
                {
                    step *= kSoscfMaxRot / max_elem;
                    cap_fired = true;
                }
                for (int a = 0; a < n_virt_i; ++a)
                    for (int i = 0; i < n_occ_i; ++i)
                    {
                        const double v = step(a * n_occ_i + i);
                        kappa(n_occ_i + a, i) = v;
                        kappa(i, n_occ_i + a) = -v;
                    }
            }
            C = HartreeFock::Correlation::CASSCF::apply_orbital_rotation(
                C_soscf_prev, kappa, S);
            if (!C.allFinite())
                return std::unexpected(std::format(
                    "SOSCF: orbital rotation produced non-finite coefficients at iteration {}", iter));

            // Semicanonicalize: block-diagonalize the occ-occ and virt-virt
            // blocks of Cᵀ F C separately and rotate C's occupied/virtual
            // column GROUPS accordingly (never mixing occ with virt here --
            // that mixing is exactly the Newton step already applied above).
            // This is a pure gauge freedom: rotating occupied orbitals among
            // themselves, or virtuals among themselves, changes neither the
            // density nor the energy, so it costs nothing physically.
            //
            // Required because reading eps off Cᵀ F C's raw diagonal (the
            // first version of this step) is not a real eigendecomposition,
            // so the next iteration's Hessian diagonal (eps(a)-eps(i)) is
            // built from a systematically approximate curvature estimate.
            // That approximation is fine for a handful of steps -- it is
            // what let the ORCA-style transient SOSCF window work -- but
            // running pure SOSCF (no DIIS handoff) indefinitely on top of it
            // plateaus at |g| ~ 0.09 well short of the true minimum, because
            // the error compounds over many consecutive steps instead of
            // being wiped out by DIIS's from-scratch diagonalization every
            // iteration. Confirmed by direct measurement on water/6-31g
            // before this fix.
            const Eigen::MatrixXd F_mo_new = C.transpose() * F * C;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> occ_solver(
                F_mo_new.topLeftCorner(n_occ_i, n_occ_i));
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> virt_solver(
                F_mo_new.bottomRightCorner(n_virt_i, n_virt_i));
            if (occ_solver.info() != Eigen::Success || virt_solver.info() != Eigen::Success)
                return std::unexpected(std::format(
                    "SOSCF: semicanonicalization eigensolve failed at iteration {}", iter));

            Eigen::MatrixXd C_canon(nbasis, nbasis);
            C_canon.leftCols(n_occ_i) = C.leftCols(n_occ_i) * occ_solver.eigenvectors();
            C_canon.rightCols(n_virt_i) = C.rightCols(n_virt_i) * virt_solver.eigenvectors();
            C = C_canon;

            eps.head(n_occ_i) = occ_solver.eigenvalues();
            eps.tail(n_virt_i) = virt_solver.eigenvalues();

            const double homo = eps.head(n_occ_i).maxCoeff();
            const double lumo = eps.tail(n_virt_i).minCoeff();
            const double offdiag_ov_rms =
                std::sqrt(F_mo_new.block(n_occ_i, 0, n_virt_i, n_occ_i).array().square().mean());
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, "SOSCF :",
                std::format(
                    "step at iter {}: |g|={:.3e} v0={:.4f} eig={:.4e} converged={} "
                    "ah_iters={} ah_residual={:.3e} "
                    "HOMO={:.4f} LUMO={:.4f} gap={:.4f} rms(F_mo_ov_after)={:.3e} "
                    "raw_max|kappa|={:.3e} cap_fired={}",
                    iter, g.norm(), ah.v0, ah.eigenvalue, ah.converged,
                    ah.iterations, ah.residual_norm,
                    homo, lumo, lumo - homo, offdiag_ov_rms,
                    raw_max_elem, cap_fired));
        }
        else if (!sao_active)
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

        const auto t_diag_end = std::chrono::steady_clock::now();

        const Eigen::MatrixXd C_occ = C.leftCols(n_occ);

        // ── Next density ──────────────────────────────────────────────────────
        const Eigen::MatrixXd density_next = 2.0 * C_occ * C_occ.transpose();

        // ── Convergence checks ────────────────────────────────────────────────
        IterationMetrics metrics =
            restricted_iteration_metrics(P, density_next, E_prev, E_total);
        metrics.diis_error = diis_err;

        const auto iter_end = std::chrono::steady_clock::now();
        const double iter_time = std::chrono::duration<double>(iter_end - iter_start).count();

        // ponytail: one line per iteration per rank, scraped by phase_bench.py.
        // "rest" = everything not Fock/DIIS/diag (density build, metrics, PCM):
        // by construction the four buckets sum to iter_s, so no phase hides.
        if (std::getenv("PLANCK_PHASE_TIMING"))
        {
            const auto sec = [](auto a, auto b)
            { return std::chrono::duration<double>(b - a).count(); };
            const double t_fock = sec(iter_start, t_fock_end);
            const double t_diis = sec(t_fock_end, t_diis_end);
            const double t_diag = sec(t_diis_end, t_diag_end);
            std::printf(
                "PLANCK_PHASE rank=%d iter=%u fock_s=%.6f diis_s=%.6f "
                "diag_s=%.6f rest_s=%.6f iter_s=%.6f\n",
                HartreeFock::Mpi::rank(), iter, t_fock, t_diis, t_diag,
                iter_time - t_fock - t_diis - t_diag, iter_time);
            std::fflush(stdout);
        }

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
        // SOSCF (S2): keep the reference basis current every iteration, not
        // just while SOSCF is active, so the switch iteration always has a
        // valid C_soscf_prev the moment it fires (rather than needing a
        // one-iteration warmup after scf_soscf_start).
        C_soscf_prev = C;
        eps_soscf_prev = eps;

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
    // Working AO dimension (spherical in spherical mode, else Cartesian); the ERI is
    // built in the Cartesian basis (nbasis_cart) and transformed. See run_rhf.
    const std::size_t nbasis = calculator.working_nbasis();
    const std::size_t nbasis_cart = calculator._shells.nbasis();

    // total_nuclear_charge() excludes ghost atoms (BSSE counterpoise).
    const int n_electrons =
        calculator._molecule.total_nuclear_charge() - calculator._molecule.charge;

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

    const unsigned int max_iter = std::max(calculator._scf.get_max_cycles(nbasis), 75u);
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
        // The integral engine works in the Cartesian basis (nbasis_cart). In spherical
        // mode the tensor is then transformed with the same (S-normalized) C applied to
        // S/H, so every quantity SCF sees is consistently spherical. In Cartesian mode
        // nbasis_cart == nbasis and no transform is applied.
        eri = _compute_2e(shell_pairs, nbasis_cart, calculator._integral._engine,
                          HartreeFock::ERIKernel::Coulomb, 0.0,
                          calculator._integral._tol_eri,
                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

        if (calculator._shells._spherical)
        {
            auto eri_sph = HartreeFock::BasisFunctions::transform_eri_cart_to_sph(
                eri, calculator._shells._cart_to_sph, nbasis_cart);
            if (!eri_sph)
                return std::unexpected(eri_sph.error());
            eri = std::move(*eri_sph);
        }

        calculator._eri = eri; // persist for post-HF use
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    // ── Combined-spin DIIS ────────────────────────────────────────────────────
    // Alpha and beta Fock are coupled through the shared Coulomb term, so they are
    // extrapolated together with a single coefficient vector (see UHFDIISState).
    HartreeFock::UHFDIISState diis;
    diis.max_vecs = calculator._scf._DIIS_dim;
    const bool use_diis = calculator._scf._use_DIIS;

    const double tol_eri = calculator._integral._tol_eri;
    double E_prev = 0.0;
    double diis_err_prev = std::numeric_limits<double>::max();

    // ── C1: build the full-symmetry skeleton ONCE (docs/FULL_SYMMETRY_PERF_SCOPE.md) ─
    // Mirror of the RHF path: density-independent skeleton built before the loop and
    // contracted each iteration. Same memory gate (nbasis_cart ≤ _threshold). Empty
    // ⇒ fall back to per-iteration full_symmetry_fock_uhf.
    calculator._symm_skeleton_eri.clear();
    if (!use_conventional && calculator._use_full_symmetry && sao_active_uhf &&
        calculator._shells.nbasis() <= static_cast<std::size_t>(calculator._scf._threshold))
    {
        auto skel = full_symmetry_build_skeleton(shell_pairs, calculator, tol_eri);
        if (!skel)
            return std::unexpected(skel.error());
        calculator._symm_skeleton_eri = std::move(*skel);
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "Full Symmetry :",
            std::format("skeleton ERI persisted across SCF iterations ({:.1f} MB)",
                        full_symmetry_skeleton_doubles(calculator) * 8.0 / 1e6));
    }

    HartreeFock::Logger::scf_header();

    // SOSCF (docs/SOSCF_UHF_DFT_SCOPE.md, U1/U2): the previous iteration's MO
    // basis, paired against the CURRENT iteration's Fock -- the same pairing
    // RHF SOSCF uses, and for the same reason: Ca^T Fa Ca is diagonal BY
    // CONSTRUCTION immediately after diagonalizing Fa, so building the
    // gradient there is vacuous (RHF SOSCF hit this trap first; U1's FD probe
    // hit it again here before this fix). U1 used this pairing only for its
    // FD-check probe; U2 promotes it to the actual SOSCF step's gradient/
    // Hessian source, exactly mirroring C_soscf_prev/eps_soscf_prev in run_rhf.
    Eigen::MatrixXd Ca_prev, Cb_prev;
    Eigen::VectorXd epsa_prev, epsb_prev;
    // U2: the iteration the SOSCF window actually started (0 = not yet
    // triggered). Fixed-iteration switch only for now (scf_soscf_start),
    // mirroring RHF's own S2 scope -- the DIIS-error criterion (RHF's S3) is
    // U4's job here, not U2's.
    unsigned int soscf_window_start = 0;

    for (unsigned int iter = 1; iter <= max_iter; ++iter)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // ── Two-electron Fock contributions ───────────────────────────────────
        // Conventional contracts the (spherical) ERI; direct builds per-quartet in
        // Cartesian, with spherical back-projection/forward-transform per spin channel.
        Eigen::MatrixXd Ga;
        Eigen::MatrixXd Gb;
        if (use_conventional)
        {
            std::tie(Ga, Gb) = HartreeFock::ObaraSaika::_compute_fock_uhf(eri, Pa, Pb, nbasis);
        }
        else if (calculator._use_full_symmetry && sao_active_uhf)
        {
            // Full point-group ERI reduction (supersedes the D2h sign-flip path).
            // Both spin densities must be symmetry-adapted (contravariant); SAO
            // blocking guarantees it, verified once on the total density. Dispatches
            // to the spherical pipeline when _shells._spherical (Step 2).
            if (iter == 1)
            {
                const double dev = density_symmetry_deviation(Pa + Pb, calculator._group_operations);
                if (dev > 1e-8)
                    return std::unexpected(std::format(
                        "Full-symmetry UHF: initial density is not symmetry-adapted "
                        "(max |O P O^T - P| = {:.3e}); SAO blocking should guarantee this",
                        dev));
            }
            // C1: contract the persisted skeleton if built; else rebuild per-iteration.
            auto G_res = calculator._symm_skeleton_eri.empty()
                             ? full_symmetry_fock_uhf(shell_pairs, calculator, Pa, Pb, nbasis, tol_eri)
                             : full_symmetry_contract_uhf(calculator._symm_skeleton_eri, calculator, Pa, Pb, nbasis);
            if (!G_res)
                return std::unexpected(G_res.error());
            std::tie(Ga, Gb) = std::move(*G_res);
        }
        else if (calculator._shells._spherical)
        {
            const Eigen::MatrixXd &C = calculator._shells._cart_to_sph;
            const Eigen::MatrixXd Pa_cart = C.transpose() * Pa * C;
            const Eigen::MatrixXd Pb_cart = C.transpose() * Pb * C;
            auto [Ga_cart, Gb_cart] =
                _compute_2e_fock_uhf(shell_pairs, Pa_cart, Pb_cart, nbasis_cart,
                                     calculator._integral._engine,
                                     HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri,
                                     calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
            Ga = C * Ga_cart * C.transpose();
            Gb = C * Gb_cart * C.transpose();
        }
        else
        {
            std::tie(Ga, Gb) =
                _compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, calculator._integral._engine,
                                     HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri,
                                     calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        }

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
        // Use the current AO densities in the overlap metric, matching PySCF:
        // F_s = F + λ (S - S D S). This raises the current virtual space
        // without relying on branch-specific cached orbitals.
        Eigen::MatrixXd Fa_s = Fa, Fb_s = Fb;
        if (level_shift > 0.0)
        {
            Fa_s += make_level_shift_matrix(S, Pa, level_shift);
            Fb_s += make_level_shift_matrix(S, Pb, level_shift);
        }

        // ── DIIS: combined-spin Pulay errors from the shifted Fock ───────────
        double diis_err = 0.0;
        Eigen::MatrixXd Fa_diag = Fa_s, Fb_diag = Fb_s;
        if (use_diis)
        {
            const Eigen::MatrixXd ea = X.transpose() * (Fa_s * Pa * S - S * Pa * Fa_s) * X;
            const Eigen::MatrixXd eb = X.transpose() * (Fb_s * Pb * S - S * Pb * Fb_s) * X;

            // Combined RMS error over both spins (matches UHFDIISState::error_norm).
            const double cur_err = std::sqrt(
                (ea.squaredNorm() + eb.squaredNorm()) /
                static_cast<double>(ea.size() + eb.size()));

            // ── DIIS restart ──────────────────────────────────────────────────
            if (restart_factor > 0.0 && iter > 2 && cur_err > diis_err_prev * restart_factor)
            {
                diis.clear();
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                                             "DIIS :", std::format("Subspace restarted at iter {} (error grew {:.1f}×)", iter, cur_err / diis_err_prev));
            }

            diis.push(Fa_s, Fb_s, ea, eb);
            diis_err = diis.error_norm();
            diis_err_prev = cur_err;

            // Extrapolate both spins with one shared coefficient vector.
            if (diis.ready())
                std::tie(Fa_diag, Fb_diag) = diis.extrapolate();
        }

        // ── SOSCF window selection (docs/SOSCF_UHF_DFT_SCOPE.md, U2/U3) ──────
        // Mirrors run_rhf's own soscf_enabled/soscf_active gating exactly
        // (S2's fixed-iteration switch only; U4 is the DIIS-error-criterion
        // follow-on, matching RHF's own S3). No SAO/PCM coverage yet, same
        // scope line RHF's S2 drew.
        //
        // U3's explicit decision: SOSCF and an active level shift are
        // mutually exclusive, not silently combined. The SOSCF gradient/
        // Hessian below reads the plain (unshifted) Fa/Fb -- level shift and
        // second-order Newton steps solve the same problem (raising the
        // virtual space to keep the aufbau ordering stable during early,
        // far-from-converged iterations), so running both is redundant, and
        // running SOSCF against the unshifted Fock while level_shift > 0 is
        // configured would silently ignore the user's own request on exactly
        // the iterations where they set it to matter. Disabling is simpler
        // and cheaper than threading the shift through the CPHF gradient/
        // Hessian construction (RHF has no level_shift feature at all, so
        // there is no existing pattern to thread it through).
        const bool soscf_enabled_uhf =
            (calculator._scf._scf_soscf_diis_tol > 0.0 || calculator._scf._scf_soscf_start > 0) &&
            !sao_active_uhf && pcm == nullptr && level_shift <= 0.0;
        if (soscf_enabled_uhf && soscf_window_start == 0)
        {
            const bool criterion_fires =
                calculator._scf._scf_soscf_diis_tol > 0.0
                    ? (use_diis && diis_err > 0.0 && diis_err < calculator._scf._scf_soscf_diis_tol &&
                       iter >= calculator._scf._scf_soscf_min_iter)
                    : (iter >= calculator._scf._scf_soscf_start);
            if (criterion_fires)
                soscf_window_start = iter;
        }
        const bool soscf_active_uhf =
            soscf_enabled_uhf && soscf_window_start > 0 &&
            iter < soscf_window_start + calculator._scf._scf_soscf_cycles &&
            Ca_prev.size() > 0;
        if (soscf_window_start > 0 &&
            iter == soscf_window_start + calculator._scf._scf_soscf_cycles)
        {
            diis.clear();
        }

        // ── Diagonalize alpha and beta ────────────────────────────────────────
        Eigen::MatrixXd Ca(nbasis, nbasis), Cb(nbasis, nbasis);
        Eigen::VectorXd epsa(nbasis), epsb(nbasis);
        std::vector<std::string> mo_sym_a, mo_sym_b;
        auto diagonalize_uhf_spin = [&](const Eigen::MatrixXd &F_spin,
                                        std::vector<std::string> *mo_sym_out,
                                        const std::string &spin_tag)
            -> std::expected<std::pair<Eigen::MatrixXd, Eigen::VectorXd>, std::string>
        {
            if (!sao_active_uhf)
            {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(X.transpose() * F_spin * X);
                if (solver.info() != Eigen::Success)
                    return std::unexpected(std::format(
                        "{} Fock diagonalization failed at iter {}", spin_tag, iter));
                return std::make_pair(X * solver.eigenvectors(), solver.eigenvalues());
            }

            const Eigen::MatrixXd F_sao = U_uhf.transpose() * F_spin * U_uhf;
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

            std::vector<int> order(nbasis);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b)
                             { return eps_sao[a] < eps_sao[b]; });

            Eigen::VectorXd eps_sorted(nbasis);
            Eigen::MatrixXd C_sao_sorted(nbasis, nbasis);
            std::vector<std::string> local_syms(nbasis);
            for (int k = 0; k < static_cast<int>(nbasis); ++k)
            {
                eps_sorted[k] = eps_sao[order[k]];
                C_sao_sorted.col(k) = C_sao.col(order[k]);
                local_syms[k] = calculator._sao_irrep_names[mo_irrep_idx[order[k]]];
            }

            if (mo_sym_out != nullptr)
                *mo_sym_out = std::move(local_syms);
            return std::make_pair(U_uhf * C_sao_sorted, eps_sorted);
        };

        if (soscf_active_uhf)
        {
            // ── SOSCF (docs/SOSCF_UHF_DFT_SCOPE.md, U2) ────────────────────
            // Mirrors run_rhf's SOSCF branch exactly, generalized to the
            // coupled alpha/beta step: this REPLACES diagonalization (both
            // spins), built from the PREVIOUS iteration's basis
            // (Ca_prev/Cb_prev/epsa_prev/epsb_prev) against the CURRENT
            // Fock (Fa/Fb) -- the pairing that is actually stationary at
            // convergence, and the only one RHF's own debugging found to
            // work (see the RHF branch's own comment for the two wrong
            // pairings that were tried and ruled out first).
            //
            // U1 measured the scale convention directly against the real
            // UHF E(kappa): g_true = 2*g_used and H_true = 2*Amat
            // (universal across a full index sweep, unlike RHF's 4x) --
            // since a Newton step depends only on the ratio g/H, using
            // g=F_mo against Amat UNSCALED reproduces the true step at
            // half the arithmetic, exactly RHF's own reasoning.
            const int n_virt_a_i = static_cast<int>(nbasis) - n_alpha;
            const int n_virt_b_i = static_cast<int>(nbasis) - n_beta;
            auto A_res = HartreeFock::Correlation::build_uhf_cphf_matrix(
                calculator, shell_pairs, Ca_prev, Cb_prev, epsa_prev, epsb_prev, n_alpha, n_beta);
            if (!A_res)
                return std::unexpected("SOSCF: " + A_res.error());
            const Eigen::MatrixXd &Amat = *A_res;
            const int nova = n_virt_a_i * n_alpha;
            const int novb = n_virt_b_i * n_beta;

            const Eigen::MatrixXd Fa_mo = Ca_prev.transpose() * Fa * Ca_prev;
            const Eigen::MatrixXd Fb_mo = Cb_prev.transpose() * Fb * Cb_prev;
            Eigen::VectorXd g(nova + novb);
            for (int a = 0; a < n_virt_a_i; ++a)
                for (int i = 0; i < n_alpha; ++i)
                    g(a * n_alpha + i) = Fa_mo(n_alpha + a, i);
            for (int a = 0; a < n_virt_b_i; ++a)
                for (int i = 0; i < n_beta; ++i)
                    g(nova + a * n_beta + i) = Fb_mo(n_beta + a, i);

            const auto h_op = [&Amat](const Eigen::VectorXd &x) -> Eigen::VectorXd
            { return Amat * x; };
            const auto g_op = [&g]() -> Eigen::VectorXd
            { return g; };

            // Same fixed-scale-to-|g| start tolerance RHF's own S2/Step-B fix
            // used (aug-hessian.h's default is CASSCF-tuned and satisfied
            // after exactly one Krylov iteration on a gradient this small).
            HartreeFock::Correlation::CASSCF::AugHessianOptions ah_opts;
            ah_opts.ah_start_tol = std::max(1e-8, 0.1 * g.norm());
            Eigen::VectorXd x0 = -g;
            const double x0_norm = x0.norm();
            if (std::isfinite(x0_norm) && x0_norm > 0.0)
                x0 /= x0_norm;
            const HartreeFock::Correlation::CASSCF::AugHessianResult ah =
                HartreeFock::Correlation::CASSCF::solve_augmented_hessian(
                    h_op, g_op, nullptr, x0, ah_opts);

            // Same trust-region cap RHF's own SOSCF branch uses, applied
            // per spin channel (not forked -- one shared step vector, two
            // kappa matrices).
            constexpr double kSoscfMaxRot = 0.20;
            Eigen::MatrixXd kappa_a = Eigen::MatrixXd::Zero(nbasis, nbasis);
            Eigen::MatrixXd kappa_b = Eigen::MatrixXd::Zero(nbasis, nbasis);
            bool cap_fired = false;
            double raw_max_elem = 0.0;
            if (ah.x.size() == nova + novb && ah.x.allFinite())
            {
                Eigen::VectorXd step = ah.x;
                const double max_elem = step.cwiseAbs().maxCoeff();
                raw_max_elem = max_elem;
                if (max_elem > kSoscfMaxRot)
                {
                    step *= kSoscfMaxRot / max_elem;
                    cap_fired = true;
                }
                for (int a = 0; a < n_virt_a_i; ++a)
                    for (int i = 0; i < n_alpha; ++i)
                    {
                        const double v = step(a * n_alpha + i);
                        kappa_a(n_alpha + a, i) = v;
                        kappa_a(i, n_alpha + a) = -v;
                    }
                for (int a = 0; a < n_virt_b_i; ++a)
                    for (int i = 0; i < n_beta; ++i)
                    {
                        const double v = step(nova + a * n_beta + i);
                        kappa_b(n_beta + a, i) = v;
                        kappa_b(i, n_beta + a) = -v;
                    }
            }
            Ca = HartreeFock::Correlation::CASSCF::apply_orbital_rotation(Ca_prev, kappa_a, S);
            Cb = HartreeFock::Correlation::CASSCF::apply_orbital_rotation(Cb_prev, kappa_b, S);
            if (!Ca.allFinite() || !Cb.allFinite())
                return std::unexpected(std::format(
                    "SOSCF: orbital rotation produced non-finite coefficients at iteration {}", iter));

            // Semicanonicalize each spin channel separately -- pure gauge
            // freedom (rotating occupied among themselves, or virtual among
            // themselves, changes neither density nor energy), required so
            // the NEXT iteration's Hessian diagonal is read off a genuine
            // eigendecomposition rather than Cᵀ F C's raw (non-diagonal)
            // blocks. Mirrors run_rhf's own semicanonicalization exactly,
            // once per spin.
            auto semicanonicalize = [&](const Eigen::MatrixXd &C_in,
                                        const Eigen::MatrixXd &F_in,
                                        int n_occ_s, int n_virt_s,
                                        const char *spin_tag)
                -> std::expected<std::pair<Eigen::MatrixXd, Eigen::VectorXd>, std::string>
            {
                const Eigen::MatrixXd F_mo_new = C_in.transpose() * F_in * C_in;
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> occ_solver(
                    F_mo_new.topLeftCorner(n_occ_s, n_occ_s));
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> virt_solver(
                    F_mo_new.bottomRightCorner(n_virt_s, n_virt_s));
                if (occ_solver.info() != Eigen::Success || virt_solver.info() != Eigen::Success)
                    return std::unexpected(std::format(
                        "SOSCF: {} semicanonicalization eigensolve failed at iteration {}", spin_tag, iter));

                Eigen::MatrixXd C_canon(nbasis, nbasis);
                C_canon.leftCols(n_occ_s) = C_in.leftCols(n_occ_s) * occ_solver.eigenvectors();
                C_canon.rightCols(n_virt_s) = C_in.rightCols(n_virt_s) * virt_solver.eigenvectors();

                Eigen::VectorXd eps_out(nbasis);
                eps_out.head(n_occ_s) = occ_solver.eigenvalues();
                eps_out.tail(n_virt_s) = virt_solver.eigenvalues();
                return std::make_pair(C_canon, eps_out);
            };

            // U3 (docs/SOSCF_UHF_DFT_SCOPE.md): re-measured directly rather
            // than assuming RHF's "harmless, keep it" verdict transfers.
            // Disabling this per spin (reading eps off the raw, non-
            // eigendecomposed Cᵀ F C diagonal) on a long pure-SOSCF window
            // (no DIIS handoff) converges to the SAME energy on both a
            // triplet (water/6-31g, 60 vs 64 iterations) and a doublet
            // (water-cation/STO-3G, 25 vs 32 iterations) -- no plateau, no
            // wrong-basin convergence, unlike the risk RHF's own note
            // describes for a genuinely long run. Kept anyway: it is correct
            // (pure gauge freedom -- rotating occupied or virtual orbitals
            // among themselves changes neither density nor energy) and cheap
            // (two small in-block eigendecompositions per spin, not a full
            // nbasis-size solve), so there is no reason to drop it even
            // though it measured as unnecessary here too.
            auto canon_a = semicanonicalize(Ca, Fa, n_alpha, n_virt_a_i, "alpha");
            if (!canon_a)
                return std::unexpected(canon_a.error());
            Ca = std::move(canon_a->first);
            epsa = std::move(canon_a->second);

            auto canon_b = semicanonicalize(Cb, Fb, n_beta, n_virt_b_i, "beta");
            if (!canon_b)
                return std::unexpected(canon_b.error());
            Cb = std::move(canon_b->first);
            epsb = std::move(canon_b->second);

            const double homo_a = epsa.head(n_alpha).maxCoeff();
            const double lumo_a = epsa.tail(n_virt_a_i).minCoeff();
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, "SOSCF :",
                std::format(
                    "step at iter {}: |g|={:.3e} v0={:.4f} eig={:.4e} converged={} "
                    "ah_iters={} ah_residual={:.3e} "
                    "HOMO(a)={:.4f} LUMO(a)={:.4f} gap(a)={:.4f} "
                    "raw_max|kappa|={:.3e} cap_fired={}",
                    iter, g.norm(), ah.v0, ah.eigenvalue, ah.converged,
                    ah.iterations, ah.residual_norm,
                    homo_a, lumo_a, lumo_a - homo_a,
                    raw_max_elem, cap_fired));
        }
        else
        {
            auto res_a = diagonalize_uhf_spin(Fa_diag, sao_active_uhf ? &mo_sym_a : nullptr, "Alpha");
            if (!res_a)
                return std::unexpected(res_a.error());
            Ca = std::move(res_a->first);
            epsa = std::move(res_a->second);

            auto res_b = diagonalize_uhf_spin(Fb_diag, sao_active_uhf ? &mo_sym_b : nullptr, "Beta");
            if (!res_b)
                return std::unexpected(res_b.error());
            Cb = std::move(res_b->first);
            epsb = std::move(res_b->second);

            if (sao_active_uhf)
            {
                calculator._info._scf.alpha.mo_symmetry = mo_sym_a;
                calculator._info._scf.beta.mo_symmetry = mo_sym_b;
            }
        }

        // ── SOSCF UHF Hessian FD check (docs/SOSCF_UHF_DFT_SCOPE.md, U1) ──────
        // Verifies build_uhf_cphf_matrix's gradient/Hessian pairing against the
        // ACTUAL UHF energy E(kappa), the same way RHF SOSCF's own
        // PLANCK_SOSCF_FD_CHECK probe verified build_rhf_cphf_matrix before any
        // SCF-loop wiring existed. Runs once, at iteration 2 (Ca/Cb/epsa/epsb
        // are already a real post-diagonalization basis by then), gated so it
        // never fires in a normal build. Only exercises the conventional-ERI
        // UHF path, matching the RHF probe's own scope.
        // SAD's per-element atomic UHF sub-solves (sad.cpp) recurse into this
        // same run_uhf on a lone atom, where a spin channel can have zero
        // virtuals (e.g. H's beta channel) -- excluded via atomic_numbers.size()
        // so the probe only ever fires on the real, multi-atom molecule.
        if (iter == 2 && std::getenv("PLANCK_SOSCF_FD_CHECK") && use_conventional &&
            calculator._molecule.atomic_numbers.size() > 1)
        {
            const int n_virt_a = static_cast<int>(nbasis) - n_alpha;

            auto energy_at_kappa = [&](const Eigen::MatrixXd &kap_a,
                                       const Eigen::MatrixXd &kap_b) -> double
            {
                const Eigen::MatrixXd Ca_trial =
                    HartreeFock::Correlation::CASSCF::apply_orbital_rotation(Ca_prev, kap_a, S);
                const Eigen::MatrixXd Cb_trial =
                    HartreeFock::Correlation::CASSCF::apply_orbital_rotation(Cb_prev, kap_b, S);
                const Eigen::MatrixXd Pa_trial =
                    Ca_trial.leftCols(n_alpha) * Ca_trial.leftCols(n_alpha).transpose();
                const Eigen::MatrixXd Pb_trial =
                    Cb_trial.leftCols(n_beta) * Cb_trial.leftCols(n_beta).transpose();
                const auto [Ga_trial, Gb_trial] =
                    HartreeFock::ObaraSaika::_compute_fock_uhf(eri, Pa_trial, Pb_trial, nbasis);
                const Eigen::MatrixXd Fa_trial = H + Ga_trial;
                const Eigen::MatrixXd Fb_trial = H + Gb_trial;
                return 0.5 * ((Pa_trial.array() * (H + Fa_trial).array()).sum() +
                              (Pb_trial.array() * (H + Fb_trial).array()).sum());
            };

            auto A_res = HartreeFock::Correlation::build_uhf_cphf_matrix(
                calculator, shell_pairs, Ca_prev, Cb_prev, epsa_prev, epsb_prev, n_alpha, n_beta);
            if (!A_res)
                return std::unexpected("SOSCF[FD]: " + A_res.error());
            const Eigen::MatrixXd &Amat = *A_res;
            const int nova = n_virt_a * n_alpha;

            const double E0 = energy_at_kappa(
                Eigen::MatrixXd::Zero(nbasis, nbasis), Eigen::MatrixXd::Zero(nbasis, nbasis));

            // g_ai = F_mo(a,i) per spin, evaluated in the PREVIOUS iteration's
            // basis against the CURRENT Fock -- Ca_prev does not diagonalize
            // Fa, so this is a genuine (nonzero) gradient, unlike Ca^T Fa Ca.
            // Packed [alpha block; beta block] the same way
            // build_uhf_cphf_matrix packs its rows/columns.
            const Eigen::MatrixXd Fa_mo = Ca_prev.transpose() * Fa * Ca_prev;
            const Eigen::MatrixXd Fb_mo = Cb_prev.transpose() * Fb * Cb_prev;

            // Measured (docs/SOSCF_UHF_DFT_SCOPE.md, U1): a full sweep over
            // EVERY (a,i) diagonal index on water/6-31g triplet (28 alpha +
            // 36 beta directions) found g_fd/g_used = 2.0000000 to 6 decimals
            // at every single index -- a universal, direction-independent
            // scale factor, exactly like RHF's own g_true=4*F_mo finding.
            // A_used (the raw diagonal Hessian element) does NOT reproduce
            // h_fd cleanly at most indices (ratios from 0.2 to 272 across the
            // sweep) -- but this is expected, not a bug: off-diagonal
            // orbital-Hessian curvature dominates a coupled multi-virtual
            // system, and a bare diagonal element was never meant to
            // reproduce a single-direction second derivative on its own (RHF
            // only saw a clean ratio because that probe direction happened to
            // be weakly coupled to the rest of the space). What actually
            // matters for a Newton step is the RATIO g/H, and since g and H
            // share the same unscaled convention (g_true=2*g_used,
            // H_true=2*Amat, verified together: g_used/Amat at the isolated
            // water/STO-3G triplet indices reproduces g_fd/h_fd to the same
            // few-percent residual RHF's own probe showed), using
            // g=F_mo against Amat unscaled reproduces the true step at 1/2
            // the arithmetic without ever touching build_uhf_cphf_matrix.
            struct Probe
            {
                const char *spin;
                int a, i, k; // k = packed index into g/Amat
            };
            const Probe probes[] = {
                {"alpha", 0, 0, 0 * n_alpha + 0},
                {"beta", 0, 0, nova + 0 * n_beta + 0},
            };

            for (const auto &p : probes)
            {
                const double g_used = std::string_view(p.spin) == "alpha"
                                          ? Fa_mo(n_alpha + p.a, p.i)
                                          : Fb_mo(n_beta + p.a, p.i);
                for (double h : {1e-2, 1e-3, 1e-4})
                {
                    Eigen::MatrixXd kap_a = Eigen::MatrixXd::Zero(nbasis, nbasis);
                    Eigen::MatrixXd kap_b = Eigen::MatrixXd::Zero(nbasis, nbasis);
                    Eigen::MatrixXd &kap = std::string_view(p.spin) == "alpha" ? kap_a : kap_b;
                    const int nocc = std::string_view(p.spin) == "alpha" ? n_alpha : n_beta;
                    kap(nocc + p.a, p.i) = h;
                    kap(p.i, nocc + p.a) = -h;
                    const double Ep = energy_at_kappa(kap_a, kap_b);
                    const double Em = energy_at_kappa(-kap_a, -kap_b);
                    const double g_fd = (Ep - Em) / (2.0 * h);
                    const double h_fd = (Ep - 2.0 * E0 + Em) / (h * h);
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info, "SOSCF[FD] :",
                        std::format(
                            "spin={} h={:.0e} 2*g_used={:.8f} g_fd={:.8f} diff={:.3e} | "
                            "2*A_used={:.8f} h_fd={:.8f} diff={:.3e}",
                            p.spin, h, 2.0 * g_used, g_fd, std::abs(2.0 * g_used - g_fd),
                            2.0 * Amat(p.k, p.k), h_fd, std::abs(2.0 * Amat(p.k, p.k) - h_fd)));
                }
            }
        }

        // ── Next spin densities ───────────────────────────────────────────────
        const Eigen::MatrixXd density_alpha_next =
            Ca.leftCols(n_alpha) * Ca.leftCols(n_alpha).transpose();
        const Eigen::MatrixXd density_beta_next =
            Cb.leftCols(n_beta) * Cb.leftCols(n_beta).transpose();

        // ── Convergence on total density ──────────────────────────────────────
        IterationMetrics metrics = unrestricted_iteration_metrics(
            Pa, Pb, density_alpha_next, density_beta_next, E_prev, E_total);
        metrics.diis_error = diis_err;

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

        // SOSCF UHF Hessian FD check (U1): keep the basis one iteration behind
        // for the next probe call, mirroring RHF SOSCF's C_soscf_prev.
        Ca_prev = Ca;
        Cb_prev = Cb;
        epsa_prev = epsa;
        epsb_prev = epsb;

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
            if (level_shift > 0.0)
            {
                // PySCF removes the level shift before returning the converged
                // orbitals. Post-HF methods need those unshifted canonical MO
                // energies in their denominators; keeping the shifted spectrum
                // artificially weakens UMP2 correlation.
                auto final_a = diagonalize_uhf_spin(Fa, sao_active_uhf ? &mo_sym_a : nullptr, "Alpha");
                if (!final_a)
                    return std::unexpected(final_a.error());
                Ca = std::move(final_a->first);
                epsa = std::move(final_a->second);

                auto final_b = diagonalize_uhf_spin(Fb, sao_active_uhf ? &mo_sym_b : nullptr, "Beta");
                if (!final_b)
                    return std::unexpected(final_b.error());
                Cb = std::move(final_b->first);
                epsb = std::move(final_b->second);

                Pa = Ca.leftCols(n_alpha) * Ca.leftCols(n_alpha).transpose();
                Pb = Cb.leftCols(n_beta) * Cb.leftCols(n_beta).transpose();

                if (sao_active_uhf)
                {
                    calculator._info._scf.alpha.mo_symmetry = std::move(mo_sym_a);
                    calculator._info._scf.beta.mo_symmetry = std::move(mo_sym_b);
                }
            }

            calculator._info._scf.alpha.density = Pa;
            calculator._info._scf.beta.density = Pb;
            calculator._info._scf.alpha.mo_coefficients = Ca;
            calculator._info._scf.beta.mo_coefficients = Cb;
            calculator._info._scf.alpha.mo_energies = epsa;
            calculator._info._scf.beta.mo_energies = epsb;
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
    // Working AO dimension: spherical (2L+1 per shell) in spherical mode, Cartesian
    // otherwise. The integral engine still builds in the Cartesian basis (nbasis_cart);
    // the ERI / direct Fock are transformed into the spherical basis. See run_rhf.
    const std::size_t nbasis = calculator.working_nbasis();
    const std::size_t nbasis_cart = calculator._shells.nbasis();

    // total_nuclear_charge() excludes ghost atoms (BSSE counterpoise).
    const int n_electrons =
        calculator._molecule.total_nuclear_charge() - calculator._molecule.charge;
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
        // Built in the Cartesian basis (nbasis_cart), then transformed to spherical
        // with the same S-normalized C used for S/H. Cartesian mode: no transform.
        eri = _compute_2e(shell_pairs, nbasis_cart, calculator._integral._engine,
                          HartreeFock::ERIKernel::Coulomb, 0.0,
                          calculator._integral._tol_eri,
                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        if (calculator._shells._spherical)
        {
            auto eri_sph = HartreeFock::BasisFunctions::transform_eri_cart_to_sph(
                eri, calculator._shells._cart_to_sph, nbasis_cart);
            if (!eri_sph)
                return std::unexpected(eri_sph.error());
            eri = std::move(*eri_sph);
        }
        calculator._eri = eri;
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "2e Integrals :", "ERI tensor ready");
        HartreeFock::Logger::blank();
    }

    HartreeFock::DIISState diis;
    diis.max_vecs = calculator._scf._DIIS_dim;
    const bool use_diis = calculator._scf._use_DIIS;
    const double tol_eri = calculator._integral._tol_eri;
    double E_prev = 0.0;

    // ── C1: build the full-symmetry skeleton ONCE (docs/FULL_SYMMETRY_PERF_SCOPE.md) ─
    // ROHF's two-electron build is the UHF (Ga, Gb) = f(Pa, Pb) path — all the Roothaan-
    // specific coupling happens downstream of the Fock build — so it reuses the same
    // density-independent skeleton: built before the loop, contracted each iteration.
    // Same memory gate (nbasis_cart ≤ _threshold). Empty ⇒ per-iteration rebuild.
    calculator._symm_skeleton_eri.clear();
    if (!use_conventional && calculator._use_full_symmetry && sao_active &&
        calculator._shells.nbasis() <= static_cast<std::size_t>(calculator._scf._threshold))
    {
        auto skel = full_symmetry_build_skeleton(shell_pairs, calculator, tol_eri);
        if (!skel)
            return std::unexpected(skel.error());
        calculator._symm_skeleton_eri = std::move(*skel);
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "Full Symmetry :",
            std::format("skeleton ERI persisted across SCF iterations ({:.1f} MB)",
                        full_symmetry_skeleton_doubles(calculator) * 8.0 / 1e6));
    }

    HartreeFock::Logger::scf_header();

    for (unsigned int iter = 1; iter <= max_iter; ++iter)
    {
        const auto iter_start = std::chrono::steady_clock::now();

        // Conventional contracts the (spherical) ERI; direct builds per-quartet in
        // Cartesian with spherical back-projection/forward-transform per spin channel.
        Eigen::MatrixXd Ga;
        Eigen::MatrixXd Gb;
        if (use_conventional)
        {
            std::tie(Ga, Gb) = HartreeFock::ObaraSaika::_compute_fock_uhf(eri, Pa, Pb, nbasis);
        }
        else if (calculator._use_full_symmetry && sao_active)
        {
            // Full point-group ERI reduction, reusing the UHF (Ga, Gb) machinery —
            // ROHF differs from UHF only AFTER the Fock build (the Roothaan effective
            // Fock below). Both spin densities must be symmetry-adapted (contravariant:
            // O P O^T = P); SAO blocking guarantees it, verified once on the total
            // density. A partially-occupied degenerate open shell breaks this and is
            // refused here rather than silently corrupting the energy.
            if (iter == 1)
            {
                const double dev = density_symmetry_deviation(Pa + Pb, calculator._group_operations);
                if (dev > 1e-8)
                    return std::unexpected(std::format(
                        "Full-symmetry ROHF: initial density is not symmetry-adapted "
                        "(max |O P O^T - P| = {:.3e}); SAO blocking should guarantee this "
                        "unless a degenerate open shell is partially occupied",
                        dev));
            }
            // C1: contract the persisted skeleton if built; else rebuild per-iteration.
            auto G_res = calculator._symm_skeleton_eri.empty()
                             ? full_symmetry_fock_uhf(shell_pairs, calculator, Pa, Pb, nbasis, tol_eri)
                             : full_symmetry_contract_uhf(calculator._symm_skeleton_eri, calculator, Pa, Pb, nbasis);
            if (!G_res)
                return std::unexpected(G_res.error());
            std::tie(Ga, Gb) = std::move(*G_res);
        }
        else if (calculator._shells._spherical)
        {
            const Eigen::MatrixXd &C = calculator._shells._cart_to_sph;
            const Eigen::MatrixXd Pa_cart = C.transpose() * Pa * C;
            const Eigen::MatrixXd Pb_cart = C.transpose() * Pb * C;
            auto [Ga_cart, Gb_cart] =
                _compute_2e_fock_uhf(shell_pairs, Pa_cart, Pb_cart, nbasis_cart,
                                     calculator._integral._engine,
                                     HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri,
                                     calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
            Ga = C * Ga_cart * C.transpose();
            Gb = C * Gb_cart * C.transpose();
        }
        else
        {
            std::tie(Ga, Gb) =
                _compute_2e_fock_uhf(shell_pairs, Pa, Pb, nbasis, calculator._integral._engine,
                                     HartreeFock::ERIKernel::Coulomb, 0.0, tol_eri,
                                     calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        }

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

        IterationMetrics metrics = unrestricted_iteration_metrics(
            Pa, Pb, density_alpha_next, density_beta_next, E_prev, E_total);
        metrics.diis_error = diis_err;

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
                // Store the canonical alpha-Fock diagonal (epsa), not the
                // effective Roothaan eigenvalues (eps). epsa is the physically
                // meaningful per-spin orbital energy and is what the MO-energy
                // printout should show; _reorder_rohf_orbitals already sorts the
                // columns by epsa, so the stored energies stay monotonic with
                // the column order and the downstream ordering consumers
                // (CASSCF/FCI active-space selection) are unaffected. The
                // effective eps was a convergence device only and is not read
                // after the reorder.
                .alpha_mo_energies = epsa,
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
