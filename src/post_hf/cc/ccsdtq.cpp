#include "post_hf/cc/ccsdtq.h"

#include <algorithm>
#include <filesystem>
#include <format>

#include "io/logging.h"
#include "post_hf/cc/cc_amplitude_checkpoint.h"
#include "post_hf/cc/generated_arbitrary_runtime.h"
#include "post_hf/cc/generated_kernel_registry.h"

#ifndef PLANCK_CC_ARBITRARY_LOWER_RANKS
#define PLANCK_CC_ARBITRARY_LOWER_RANKS 0
#endif

namespace HartreeFock::Correlation::CC
{
    namespace
    {
        // Solve the generated arbitrary-order RCC path at `rank`, returning the
        // full solve result (state carries the converged amplitudes). When
        // `warm_start` and a lower generated rank exists (the registry floors at
        // 4, so this fires for rank >= 5), recurse to rank-1 first and seed
        // T1..T(rank-1) from its converged amplitudes; the rank-`rank` loop then
        // mostly iterates the top block. cc4 (rank 4) always cold-starts: the
        // generated arbitrary registry has no rank-3 kernel to seed from.
        // ponytail: cc4-from-CCSDT seeding needs the rank-3 kernel re-emitted with
        // ArbitraryOrderRCCAmplitudes (today's ccsdt TU targets RCCSDTAmplitudes
        // for tensor_backend); wire that when cc4 cold-start time actually bites.
        // Try to seed `state` from a .ccamp sidecar next to the run's checkpoint.
        // Returns true iff a usable sidecar was found and applied. Any problem
        // (absent, stale, dim-mismatch, corrupt) logs and returns false so the
        // caller falls through to the cold / W6 path — a restart is an
        // optimization, never a correctness gate.
        bool try_restart_from_sidecar(
            HartreeFock::Calculator &calculator,
            ArbitraryOrderTensorCCState &state,
            int rank,
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

            // The seed hook validates per-rank dims; a sidecar carrying more ranks
            // than this solve would over-fill, so cap it. Fewer ranks (e.g. a
            // ccsdt sidecar seeding a cc4 run) is exactly the intended partial seed.
            if (chk->amplitudes.by_rank.size() > static_cast<std::size_t>(rank))
                chk->amplitudes.by_rank.resize(static_cast<std::size_t>(rank));

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
                std::format("Warm-started rank {} from CC amplitude checkpoint '{}' (seeded {} rank(s), method '{}').",
                            rank, ccamp_path, chk->amplitudes.by_rank.size(), chk->meta.method));
            return true;
        }

        std::expected<GeneratedArbitraryOrderSolveResult, std::string>
        solve_generated_rcc(
            HartreeFock::Calculator &calculator,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            int rank,
            bool warm_start,
            bool try_restart,
            const std::string &tag)
        {
            auto state_res = prepare_generated_arbitrary_order_state(
                calculator, shell_pairs, rank, tag);
            if (!state_res)
                return std::unexpected(state_res.error());

            auto kernels_res = make_generated_rcc_kernels(rank);
            if (!kernels_res)
                return std::unexpected(kernels_res.error());

            // Gap B4: allocate the higher Sz sector amplitude blocks the bundle
            // declares (prepare ran before the bundle was known). No-op for
            // methods without sectors (<= CCSDT).
            ensure_amplitude_sectors(*state_res, *kernels_res);

            // A sidecar restart supersedes the W6 in-memory recursion: if the file
            // already supplied the lower-rank amplitudes there is no point solving
            // rank-1 again in memory.
            const bool seeded_from_disk =
                try_restart && try_restart_from_sidecar(calculator, *state_res, rank, tag);

            // The lowest generated rank the registry can supply. With the rank-3
            // arbitrary companion built it is 3 (cc4 warm-starts from a generated
            // cc3); otherwise 4 (only cc5+ can recurse).
            constexpr int generated_floor = PLANCK_CC_ARBITRARY_LOWER_RANKS ? 3 : 4;
            if (!seeded_from_disk && warm_start && rank - 1 >= generated_floor)
            {
                auto seed_res = solve_generated_rcc(
                    calculator, shell_pairs, rank - 1, warm_start, false, tag);
                if (!seed_res)
                    return std::unexpected(std::format(
                        "warm-start rank-{} seed failed: {}", rank - 1, seed_res.error()));
                auto applied = seed_arbitrary_order_amplitudes(
                    *state_res, seed_res->state.amplitudes);
                if (!applied)
                    return std::unexpected(applied.error());
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, tag,
                    std::format("Warm-started rank {} from converged rank {} (seeded T1..T{}).",
                                rank, rank - 1, rank - 1));
            }

            const unsigned int max_iter =
                std::max(calculator._scf.get_max_cycles(calculator._shells.nbasis()), 100u);
            const double tol_energy = calculator._scf._tol_energy;
            const double tol_residual = calculator._scf._tol_density;
            const double damping = calculator._scf._cc_damping;
            if (tol_energy <= 0.0)
                return std::unexpected("tol_energy must be positive.");
            if (tol_residual <= 0.0)
                return std::unexpected("tol_density must be positive.");
            if (damping < 0.0 || damping > 1.0)
                return std::unexpected("cc_damping must be between 0 and 1.");

            return run_generated_arbitrary_order_iterations(
                std::move(*state_res),
                *kernels_res,
                max_iter,
                tol_energy,
                tol_residual,
                damping,
                calculator._scf._use_DIIS,
                static_cast<int>(std::max(2u, calculator._scf._DIIS_dim)),
                std::format("{}(rank {})", tag, rank));
        }
    } // namespace

    std::expected<void, std::string> run_rccsdtq(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected("run_rccsdtq: RCCSDTQ is currently available only for single-point calculations.");

        calculator._have_ccsd_reference_energy = false;
        calculator._ccsd_reference_correlation_energy = 0.0;

        // Excitation rank for the generated arbitrary-order RCC path (cc4=4,
        // cc5=5, cc6=6, and cc3/ccsdt_gen=3 when the rank-3 companion is built),
        // set from the %posthf `correlation` keyword. The runtime solver and the
        // registry are both rank-generic, so a single call site serves every rank
        // the build generated. Rank 3 through this path (Route B) is what lets a
        // ccsdt_gen run write a SPATIAL .ccamp that a later cc4 run seeds from.
        constexpr int generated_floor = PLANCK_CC_ARBITRARY_LOWER_RANKS ? 3 : 4;
        const int rank = calculator._scf._cc_generated_rank;
        if (rank < generated_floor)
            return std::unexpected(std::format(
                "run_rccsdtq: generated arbitrary-order RCC path requires rank >= {}, got {}.",
                generated_floor, rank));

        const bool warm_start = calculator._scf._cc_warm_start;
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDTQ :",
            std::format(
                "Running generated arbitrary-order RCC tensor kernels (rank={}, warm_start={}).",
                rank, warm_start ? "on" : "off"));

        auto solve_res = solve_generated_rcc(
            calculator, shell_pairs, rank, warm_start, /*try_restart=*/true, "RCCSDTQ[TENSOR] :");
        if (!solve_res)
            return std::unexpected("run_rccsdtq: " + solve_res.error());

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "RCCSDTQ :",
            std::format(
                "Generated RCCSDTQ iterations ran {} steps, E_corr={:.10f}, dE={:+.3e}, rms(res)={:.3e}, rms(step)={:.3e}.",
                solve_res->iterations,
                solve_res->correlation_energy,
                solve_res->energy_change,
                solve_res->metrics.residual_rms,
                solve_res->metrics.update_rms));

        if (!solve_res->converged)
        {
            return std::unexpected(
                std::format(
                    "Generated RCCSDTQ kernels did not converge within {} iterations (last dE={:+.3e}, rms(res)={:.3e}).",
                    solve_res->iterations,
                    solve_res->energy_change,
                    solve_res->metrics.residual_rms));
        }

        calculator._correlation_energy = solve_res->correlation_energy;

        // Persist the converged amplitudes to a .ccamp sidecar so a later run can
        // warm-start from them (X3). Gated on the same _save_checkpoint flag as the
        // SCF checkpoint; a write failure is a warning, never a run failure — the
        // energy is already in hand.
        if (calculator._scf._save_checkpoint && !calculator._checkpoint_path.empty())
        {
            const std::string ccamp_path =
                std::filesystem::path(calculator._checkpoint_path).replace_extension(".ccamp").string();
            CCAmplitudeCheckpointMeta meta{
                .max_rank = rank,
                .method = std::format("cc{}", rank),
                .basis_name = calculator._basis._basis_name,
                .n_occ = static_cast<std::uint64_t>(solve_res->state.reference.orbital_partition.n_occ),
                .n_virt = static_cast<std::uint64_t>(solve_res->state.reference.orbital_partition.n_virt),
            };
            auto saved = save_cc_amplitudes(ccamp_path, solve_res->state.amplitudes, meta);
            if (!saved)
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, "RCCSDTQ :",
                    std::format("Could not write CC amplitude checkpoint: {}", saved.error()));
            else
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, "RCCSDTQ :",
                    std::format("Wrote CC amplitude checkpoint '{}' (rank {}).", ccamp_path, rank));
        }

        return {};
    }
} // namespace HartreeFock::Correlation::CC
