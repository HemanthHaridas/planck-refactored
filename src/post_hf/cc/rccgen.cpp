#include "post_hf/cc/rccgen.h"

#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iomanip>

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

        // Fixture probe (docs/CCGEN_SPIN_ADAPT_DEFAULT.md): with
        // PLANCK_CC_FIXTURE_DIR set, load amplitudes from <dir>/t<r>.txt, seed
        // them, evaluate the generated residuals ONCE and report per-rank max
        // |R| with its index -- then stop. No iteration, no DIIS, no
        // denominators, so a non-zero element IS the defect: at a true fixed
        // point the correct residual is ~0 by construction.
        //
        // Fixture layout is the C++ one, (occ..., virt...) row-major -- NOT
        // ccgen's Python (vir..., occ...). The dumper transposes; a mismatch is
        // caught by seed_arbitrary_order_amplitudes' exact dims check.
        //
        // File format, whitespace-separated: rank, ndim, dims..., then values.
        // ponytail: plain text, not binary. These fixtures are <= a few MB and
        // being able to eyeball one is worth more than the parse time here.
        std::expected<TensorND, std::string>
        read_fixture_tensor(const std::filesystem::path &path, int expect_rank)
        {
            std::ifstream in(path);
            if (!in)
                return std::unexpected(std::format("cannot open fixture '{}'.", path.string()));

            int rank = 0;
            int ndim = 0;
            if (!(in >> rank >> ndim))
                return std::unexpected(std::format("fixture '{}': bad header.", path.string()));
            if (rank != expect_rank)
                return std::unexpected(std::format(
                    "fixture '{}': holds rank {} but rank {} was asked for.",
                    path.string(), rank, expect_rank));
            if (ndim != 2 * expect_rank)
                return std::unexpected(std::format(
                    "fixture '{}': ndim {} but rank {} needs {}.",
                    path.string(), ndim, expect_rank, 2 * expect_rank));

            std::vector<int> dims(static_cast<std::size_t>(ndim));
            std::size_t count = 1;
            for (int d = 0; d < ndim; ++d)
            {
                if (!(in >> dims[static_cast<std::size_t>(d)]) || dims[static_cast<std::size_t>(d)] <= 0)
                    return std::unexpected(std::format("fixture '{}': bad dim {}.", path.string(), d));
                count *= static_cast<std::size_t>(dims[static_cast<std::size_t>(d)]);
            }

            TensorND tensor(dims, 0.0);
            for (std::size_t i = 0; i < count; ++i)
            {
                if (!(in >> tensor.data[i]))
                    return std::unexpected(std::format(
                        "fixture '{}': ran out of values at {} of {}.", path.string(), i, count));
            }
            double trailing = 0.0;
            if (in >> trailing)
                return std::unexpected(std::format(
                    "fixture '{}': more values than the {} dims declare.", path.string(), count));
            return tensor;
        }

        // Returns true when the probe ran (caller must stop).
        bool run_fixture_probe(
            ArbitraryOrderTensorCCState &state,
            const GeneratedArbitraryOrderKernels &kernels,
            int rank,
            const std::string &tag)
        {
            const char *dir_env = std::getenv("PLANCK_CC_FIXTURE_DIR");
            if (dir_env == nullptr || *dir_env == '\0')
                return false;

            const std::filesystem::path dir(dir_env);
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, tag,
                std::format("R4.2b fixture probe: seeding from '{}', one residual "
                            "evaluation, no iteration.", dir.string()));

            ArbitraryOrderRCCAmplitudes seed;
            for (int r = 1; r <= rank; ++r)
            {
                auto tensor = read_fixture_tensor(dir / std::format("t{}.txt", r), r);
                if (!tensor)
                {
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Error, tag,
                        std::format("R4.2b fixture probe: {}", tensor.error()));
                    return true;
                }
                seed.by_rank.push_back(std::move(*tensor));
            }

            if (auto applied = seed_arbitrary_order_amplitudes(state, seed); !applied)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Error, tag,
                    std::format("R4.2b fixture probe: {}", applied.error()));
                return true;
            }

            auto residuals = evaluate_generated_arbitrary_order_residuals(state, kernels);
            if (!residuals)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Error, tag,
                    std::format("R4.2b fixture probe: {}", residuals.error()));
                return true;
            }

            // Per-rank max |R| AND its index: a single norm cannot localise the
            // defect, and localising is the entire point of R4.2c.
            for (int r = 1; r <= residuals->max_rank(); ++r)
            {
                auto view = residuals->tensor(r);
                if (!view)
                {
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Error, tag,
                        std::format("R4.2b fixture probe: rank {}: {}", r, view.error()));
                    continue;
                }
                const TensorND &t = residuals->by_rank[static_cast<std::size_t>(r - 1)];
                std::size_t argmax = 0;
                double best = 0.0;
                for (std::size_t i = 0; i < t.data.size(); ++i)
                {
                    const double mag = std::abs(t.data[i]);
                    if (mag > best)
                    {
                        best = mag;
                        argmax = i;
                    }
                }
                // Unflatten row-major so the index is readable as (i,j,..,a,b,..).
                std::string index;
                std::size_t remainder = argmax;
                std::vector<std::size_t> coords(t.dims.size());
                for (std::size_t d = t.dims.size(); d-- > 0;)
                {
                    coords[d] = remainder % static_cast<std::size_t>(t.dims[d]);
                    remainder /= static_cast<std::size_t>(t.dims[d]);
                }
                for (std::size_t d = 0; d < coords.size(); ++d)
                    index += std::format("{}{}", d == 0 ? "" : ",", coords[d]);

                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, tag,
                    std::format("R4.2b probe: rank {} max|R|={:.6e} at ({}) n={}",
                                r, best, index, t.data.size()));

                // R4.2c: a scalar max cannot tell "wrong values" from "right
                // values in a different index order". Dump the whole tensor so
                // the comparison can be elementwise, and so a permutation
                // hypothesis is testable rather than guessed at.
                const std::filesystem::path out = dir / std::format("r{}_cpp.txt", r);
                if (std::ofstream fh(out); fh)
                {
                    fh << r << ' ' << t.dims.size() << '\n';
                    for (std::size_t d = 0; d < t.dims.size(); ++d)
                        fh << t.dims[d] << (d + 1 == t.dims.size() ? '\n' : ' ');
                    fh << std::setprecision(17);
                    for (const double value : t.data)
                        fh << value << '\n';
                }
            }
            return true;
        }

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

            // C1: load_cc_amplitudes deliberately does not validate against a
            // live reference (its own header says so) -- it defers to the
            // seed hook's per-rank DIM check. That catches a wrong SHAPE, not
            // a wrong MEANING: two different bases (or two different
            // molecules/geometries under the same basis) can produce
            // identical n_occ/n_virt, in which case a stale-but-same-shaped
            // sidecar seeds a wrong basin SILENTLY -- the seed hook has no
            // way to tell that apart from a legitimate restart, and the
            // energy from a wrong-basin warm start is still a plausible
            // number, not an obvious failure.
            //
            // basis_name and n_occ/n_virt are exactly the metadata
            // save_cc_amplitudes already writes from the SAME live sources
            // compared here (calculator._basis._basis_name,
            // state.reference.orbital_partition.n_occ/n_virt in the write
            // site above) -- checking them is comparing a sidecar's own
            // declared provenance against the run it is about to seed, not
            // inventing new metadata. A mismatch degrades to cold-start with
            // a warning, the same policy every other sidecar problem here
            // uses -- restart is an optimization, never a correctness gate.
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
            if (chk->meta.n_occ != static_cast<std::uint64_t>(state.reference.orbital_partition.n_occ) ||
                chk->meta.n_virt != static_cast<std::uint64_t>(state.reference.orbital_partition.n_virt))
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, tag,
                    std::format(
                        "Ignoring CC amplitude checkpoint '{}': n_occ/n_virt "
                        "{}/{} does not match this run's {}/{}; cold-starting.",
                        ccamp_path, chk->meta.n_occ, chk->meta.n_virt,
                        state.reference.orbital_partition.n_occ,
                        state.reference.orbital_partition.n_virt));
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

            if (run_fixture_probe(*state_res, *kernels_res, rank, tag))
            {
                return std::unexpected(
                    "R4.2b fixture probe ran (PLANCK_CC_FIXTURE_DIR set); "
                    "no solve was attempted.");
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

    std::string rcc_method_label(int rank)
    {
        switch (rank)
        {
        case 2:
            return "RCCSD";
        case 3:
            return "RCCSDT";
        case 4:
            return "RCCSDTQ";
        default:
            return std::format("RCC{}", rank);
        }
    }

    std::expected<void, std::string> run_rccgen(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs)
    {
        if (calculator._calculation != HartreeFock::CalculationType::SinglePoint)
            return std::unexpected(
                "run_rccgen: the generated RCC path is currently available only for "
                "single-point calculations.");

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
                "run_rccgen: generated arbitrary-order RCC path requires rank >= {}, got {}.",
                generated_floor, rank));

        const std::string method = rcc_method_label(rank);
        const std::string tag = method + " :";

        const bool warm_start = calculator._scf._cc_warm_start;
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            tag,
            std::format(
                "Running generated arbitrary-order RCC tensor kernels (rank={}, warm_start={}).",
                rank, warm_start ? "on" : "off"));

        auto solve_res = solve_generated_rcc(
            calculator, shell_pairs, rank, warm_start, /*try_restart=*/true,
            method + "[TENSOR] :");
        if (!solve_res)
            return std::unexpected("run_rccgen: " + solve_res.error());

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            tag,
            std::format(
                "Generated {} iterations ran {} steps, E_corr={:.10f}, dE={:+.3e}, rms(res)={:.3e}, rms(step)={:.3e}.",
                method,
                solve_res->iterations,
                solve_res->correlation_energy,
                solve_res->energy_change,
                solve_res->metrics.residual_rms,
                solve_res->metrics.update_rms));

        if (!solve_res->converged)
        {
            return std::unexpected(
                std::format(
                    "Generated {} kernels did not converge within {} iterations (last dE={:+.3e}, rms(res)={:.3e}).",
                    method,
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
                    HartreeFock::LogLevel::Warning, tag,
                    std::format("Could not write CC amplitude checkpoint: {}", saved.error()));
            else
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, tag,
                    std::format("Wrote CC amplitude checkpoint '{}' (rank {}).", ccamp_path, rank));
        }

        return {};
    }
} // namespace HartreeFock::Correlation::CC
