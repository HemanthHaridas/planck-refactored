// U5.3b: the generated UNRESTRICTED CC entry point.
//
// The UCC sibling of run_rccsdtq, and deliberately a separate file rather than a
// branch inside it: the two share the solver loop but nothing else. This one
// prepares an all-sectors state from a UHF reference (prepare_generated_ucc_state,
// U5.1b), reconciles the amplitude blocks the bundle declares, and drives the same
// Jacobi/DIIS iteration.
//
// Deliberately NOT carried over from the RCC path:
//   - warm start via in-memory recursion. It recurses to rank-1 through the
//     RCC registry, which returns restricted bundles; seeding an unrestricted
//     solve from them would be a silent reference mismatch. Revisit once a
//     UCC rank ladder exists.
// .ccamp persistence: the version-3 header (docs/CC_AMPLITUDE_CHECKPOINT.md)
// added an explicit by_rank count and the four UHF occupation counts
// specifically so a sectors-only UCC amplitude set is representable; the
// write site below and the restart site further down mirror rccgen.cpp's
// try_restart_from_sidecar with a reference_type check ahead of the
// occupation-count comparison.

#include "post_hf/cc/cc_amplitude_checkpoint.h"
#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/generated_kernel_registry.h"
#include "post_hf/cc/solver_arbitrary.h"
#include "io/logging.h"

#include <algorithm>
#include <filesystem>
#include <format>

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        // U3: the UCC sibling of rccgen.cpp's try_restart_from_sidecar. Any
        // problem (absent, stale, wrong reference kind, dim mismatch, corrupt)
        // logs and returns false so the caller cold-starts -- restart is an
        // optimization, never a correctness gate, exactly as for RCC.
        bool try_restart_ucc_from_sidecar(
            HartreeFock::Calculator &calculator,
            ArbitraryOrderTensorCCState &state,
            const std::string &tag)
        {
            const bool restart_requested =
                calculator._scf._guess == HartreeFock::SCFGuess::ReadDensity ||
                calculator._scf._guess == HartreeFock::SCFGuess::ReadFull;
            if (!restart_requested || calculator._checkpoint_path.empty())
                return false;

            const std::string ccamp_path =
                std::filesystem::path(calculator._checkpoint_path).replace_extension(".ccamp").string();
            if (!std::filesystem::exists(ccamp_path))
                return false;

            auto chk = load_cc_amplitudes(ccamp_path);
            if (!chk)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, tag,
                    std::format("Ignoring CC amplitude checkpoint '{}': {}", ccamp_path, chk.error()));
                return false;
            }

            // Reject a wrong reference kind BEFORE checking counts -- an RHF
            // sidecar's n_occ/n_virt could coincidentally match this run's
            // n_occ_alpha/n_virt_alpha the way C1's own motivating case was a
            // same-shape-different-basis coincidence, and reference_type is
            // exactly the field that exists to rule that out.
            if (chk->meta.reference_type != CCReferenceType::UHF)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, tag,
                    std::format(
                        "Ignoring CC amplitude checkpoint '{}': reference type is not UHF; "
                        "cold-starting.",
                        ccamp_path));
                return false;
            }
            if (chk->meta.basis_name != calculator._basis._basis_name)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, tag,
                    std::format(
                        "Ignoring CC amplitude checkpoint '{}': basis '{}' does not match "
                        "this run's basis '{}'; cold-starting.",
                        ccamp_path, chk->meta.basis_name, calculator._basis._basis_name));
                return false;
            }
            const auto &ref = state.reference;
            if (chk->meta.n_occ_alpha != static_cast<std::uint64_t>(ref.n_occ_alpha) ||
                chk->meta.n_occ_beta != static_cast<std::uint64_t>(ref.n_occ_beta) ||
                chk->meta.n_virt_alpha != static_cast<std::uint64_t>(ref.n_virt_alpha) ||
                chk->meta.n_virt_beta != static_cast<std::uint64_t>(ref.n_virt_beta))
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, tag,
                    std::format(
                        "Ignoring CC amplitude checkpoint '{}': occupation "
                        "{}/{}/{}/{} does not match this run's {}/{}/{}/{}; cold-starting.",
                        ccamp_path,
                        chk->meta.n_occ_alpha, chk->meta.n_occ_beta,
                        chk->meta.n_virt_alpha, chk->meta.n_virt_beta,
                        ref.n_occ_alpha, ref.n_occ_beta,
                        ref.n_virt_alpha, ref.n_virt_beta));
                return false;
            }

            auto applied = seed_arbitrary_order_amplitudes(state, chk->amplitudes);
            if (!applied)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, tag,
                    std::format("CC amplitude checkpoint '{}' does not fit this run ({}); cold-starting.",
                                ccamp_path, applied.error()));
                return false;
            }

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, tag,
                std::format(
                    "Warm-started from CC amplitude checkpoint '{}' ({} sector(s), method '{}').",
                    ccamp_path, chk->amplitudes.sectors.size(), chk->meta.method));
            return true;
        }
    } // namespace

    std::expected<void, std::string> run_uccgen(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected(
                "run_uccgen: the generated UCC path is currently available only for "
                "single-point calculations.");

        const int rank = calculator._scf._cc_generated_rank;
        if (rank < 2)
            return std::unexpected(std::format(
                "run_uccgen: the generated UCC path requires rank >= 2, got {}.", rank));

        // Fail here rather than at the registry, so the message names the build
        // switch instead of an internal lookup.
        if (!generated_ucc_kernels_available())
            return std::unexpected(
                "run_uccgen: this build carries no UCC kernels. Reconfigure with "
                "-DPLANCK_CC_UCC=ON.");

        auto kernels_res = make_generated_ucc_kernels(rank);
        if (!kernels_res)
            return std::unexpected("run_uccgen: " + kernels_res.error());

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "UCC :",
            std::format("Preparing generated unrestricted CC infrastructure (rank={}).", rank));

        auto state_res = prepare_generated_ucc_state(
            calculator, shell_pairs, rank, "UCC[GENERATED] :");
        if (!state_res)
            return std::unexpected("run_uccgen: " + state_res.error());

        // prepare runs before the bundle is known (as on the RCC path), so the
        // amplitude blocks are allocated here, sized from their own denominators.
        ensure_amplitude_sectors(*state_res, *kernels_res);

        // U3: sectors must already be allocated (zero-filled) before seeding,
        // since seed_arbitrary_order_amplitudes matches an incoming sector to
        // its live counterpart by (rank, tag) and skips one with none -- so
        // this runs after ensure_amplitude_sectors, not before.
        try_restart_ucc_from_sidecar(calculator, *state_res, "UCC[GENERATED] :");

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "UCC :",
            std::format("Running generated UCC tensor kernels (rank={}, {} spin blocks).",
                        rank, state_res->amplitudes.sectors.size()));

        // Same solver knobs the RCC path uses, validated the same way -- a
        // non-positive tolerance or an out-of-range damping would otherwise
        // surface as a non-convergence rather than a configuration error.
        const double tol_energy = calculator._scf._tol_energy;
        const double tol_residual = calculator._scf._tol_density;
        const double damping = calculator._scf._cc_damping;
        if (tol_energy <= 0.0)
            return std::unexpected("run_uccgen: tol_energy must be positive.");
        if (tol_residual <= 0.0)
            return std::unexpected("run_uccgen: tol_density must be positive.");
        if (damping < 0.0 || damping > 1.0)
            return std::unexpected("run_uccgen: cc_damping must be between 0 and 1.");

        auto solve_res = run_generated_arbitrary_order_iterations(
            std::move(*state_res), *kernels_res,
            std::max(calculator._scf.get_max_cycles(calculator._shells.nbasis()), 100u),
            tol_energy,
            tol_residual,
            damping,
            calculator._scf._use_DIIS,
            static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)),
            std::format("UCC[GENERATED] :(rank {})", rank));
        if (!solve_res)
            return std::unexpected("run_uccgen: " + solve_res.error());

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "UCC :",
            std::format(
                "Generated UCC iterations ran {} steps, E_corr={:.10f}, dE={:+.3e}, "
                "rms(res)={:.3e}, rms(step)={:.3e}.",
                solve_res->iterations, solve_res->correlation_energy,
                solve_res->energy_change, solve_res->metrics.residual_rms,
                solve_res->metrics.update_rms));

        if (!solve_res->converged)
            return std::unexpected(std::format(
                "Generated UCC kernels did not converge within {} iterations "
                "(last dE={:+.3e}, rms(res)={:.3e}).",
                solve_res->iterations, solve_res->energy_change,
                solve_res->metrics.residual_rms));

        calculator._correlation_energy = solve_res->correlation_energy;

        // U2: persist the converged sector amplitudes, mirroring run_rccgen's
        // write site exactly (same gate, same path derivation, same
        // warn-not-fail policy). by_rank stays empty for UCC -- the sector
        // block already carries every spin block -- which is exactly what
        // U0/U1's n_by_rank field exists to make representable.
        if (calculator._scf._save_checkpoint && !calculator._checkpoint_path.empty())
        {
            const std::string ccamp_path =
                std::filesystem::path(calculator._checkpoint_path).replace_extension(".ccamp").string();
            const auto &ref = solve_res->state.reference;
            CCAmplitudeCheckpointMeta meta{
                .max_rank = rank,
                .method = std::format("ucc{}", rank),
                .basis_name = calculator._basis._basis_name,
                .reference_type = CCReferenceType::UHF,
                .n_occ_alpha = static_cast<std::uint64_t>(ref.n_occ_alpha),
                .n_occ_beta = static_cast<std::uint64_t>(ref.n_occ_beta),
                .n_virt_alpha = static_cast<std::uint64_t>(ref.n_virt_alpha),
                .n_virt_beta = static_cast<std::uint64_t>(ref.n_virt_beta),
            };
            auto saved = save_cc_amplitudes(ccamp_path, solve_res->state.amplitudes, meta);
            if (!saved)
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, "UCC :",
                    std::format("Could not write CC amplitude checkpoint: {}", saved.error()));
            else
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, "UCC :",
                    std::format("Wrote CC amplitude checkpoint '{}' (rank {}).", ccamp_path, rank));
        }

        return {};
    }
} // namespace HartreeFock::Correlation::CC
