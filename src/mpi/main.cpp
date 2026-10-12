// planck-mpi — unified hybrid MPI/OpenMP entry point.
//
// One binary for HPC campaigns. It parses the input once (on rank 0 when built
// with MPI), then dispatches on the method the input declared:
//   calculator.is_dft_run() ? DFT path : HartreeFock::Driver::run(...)
//
// Build modes:
//   * USE_MPI defined  -> real MPI_Init/Finalize, rank-0 parses + broadcasts,
//                         all ranks run the driver, rank 0 owns I/O.
//   * USE_MPI undefined -> compiles and runs as a plain serial binary (single
//                         "rank 0"), so the dispatch logic is testable without
//                         an MPI toolchain.
//
// Tier 3 scope (docs/HPC_MPI_EXECUTABLE_SCOPE.md): scaffolding + dispatch +
// rank-aware I/O. The distributed Fock build (Tier 1) plugs in underneath the
// existing Driver::run without changing this file.
//
// Also gates on a Boys-function license before any of the above: every rank
// verifies its own local license, rank 0 fetches the encrypted library blob
// once and MPI_Bcasts it to the rest (bounding Worker requests to one per
// job, not one per rank), and every rank decrypts + dlopens independently.

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "driver.h"
#include "dft/driver.h"

#include "base/types.h"
#include "io/io.h"
#include "io/logging.h"
#include "lookup/loader/license_path.h"
#include "lookup/loader/orchestrator.h"

namespace
{
    // Rank of this process (0 when built without MPI). Rank 0 is the only writer
    // of logs, checkpoints, and JSON — see the rank-aware I/O note in the scope.
    int mpi_rank()
    {
#ifdef USE_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
#else
        return 0;
#endif
    }
}

int main(int argc, const char *argv[])
{
#ifdef USE_MPI
    MPI_Init(&argc, const_cast<char ***>(&argv));
#endif
    const int rank = mpi_rank();

    // Gate all human-log / checkpoint output to rank 0 at the source: every
    // Logger function checks is_silenced(), which now also honors the rank. This
    // covers the driver's internal logging, not just main's messages, so
    // mpirun -n k no longer prints the log k times.
    HartreeFock::Logger::set_rank(rank);
    HartreeFock::Logger::banner();

    auto finalize = [](int code) -> int
    {
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return code;
    };

    // Gate on a valid license before doing anything else — no point parsing
    // an input file for a calculation that can't run boys() anyway.
    //
    // Under USE_MPI this cannot be a plain per-rank
    // initialize_boys_library() call: that would mean N independent network
    // fetches against the Worker's fixed free-tier monthly quota instead of
    // one per job. Split per the orchestrator's own design: rank 0 fetches
    // the encrypted blob once and MPI_Bcasts the bytes; every rank
    // (including rank 0) still verifies its own local license copy and
    // decrypts + dlopens independently, since a dlopen handle from one
    // process's address space is meaningless in another's.
    //
    // The broadcast is a collective, so every rank must call MPI_Bcast even
    // if rank 0's fetch failed — otherwise a rank-0 failure here hangs every
    // other rank waiting at the broadcast instead of exiting cleanly. Rank 0
    // signals failure by broadcasting a size of 0; every other rank checks
    // for that after the collective completes, before attempting to use the
    // (empty) buffer.
    {
        const std::string blob_url_template =
            std::string("https://boys-blob-worker.boys-core.workers.dev/{version}/planck_boys-") +
            HartreeFock::Loader::get_platform_string() + ".enc";

#ifdef USE_MPI
        // Every rank verifies its own local license copy independently
        // (cheap, no network — same reasoning as every rank re-reading the
        // same input file below). A bad license on ANY rank must fail the
        // whole job, not just that rank: silently continuing with fewer
        // ranks than requested is worse than a clean job-wide exit, and an
        // asymmetric per-rank exit here risks a deadlock at the next
        // collective the driver does (e.g. inside the distributed Fock
        // build). MPI_Allreduce with MPI_LAND turns "my license is fine"
        // into "every rank's license is fine" before anyone proceeds.
        auto local_license = HartreeFock::Loader::load_and_verify_license(
            HartreeFock::Loader::get_license_path());
        if (!local_license && rank == 0)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "License Error :",
                                         local_license.error());

        int local_license_ok = local_license ? 1 : 0;
        int all_licenses_ok = 0;
        MPI_Allreduce(&local_license_ok, &all_licenses_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (!all_licenses_ok)
            return finalize(EXIT_FAILURE);

        std::vector<std::uint8_t> blob;
        if (rank == 0)
        {
            // Every rank will independently decrypt whatever gets
            // broadcast below, so a stale/corrupt cache entry would
            // otherwise fail identically on all N ranks before the job
            // dies — wasteful, and it hides that the fix (retry past the
            // cache) only needed to happen once. Validate here instead,
            // with the side-effect-free decrypt check (NOT
            // initialize_boys_library_from_blob() — that sets the
            // process-global boys()/boys_vec() function pointers and
            // rejects being called a second time, which every rank
            // including this one still needs to do below after the
            // broadcast): on decrypt failure, evict and retry the fetch
            // once, exactly like initialize_boys_library()'s own retry.
            auto fetched = HartreeFock::Loader::fetch_boys_library_blob(*local_license, blob_url_template);
            if (fetched && !HartreeFock::Loader::boys_library_blob_decrypts(*local_license, *fetched))
            {
                HartreeFock::Loader::evict_boys_library_cache_entry(*local_license, blob_url_template);
                fetched = HartreeFock::Loader::fetch_boys_library_blob(*local_license, blob_url_template,
                                                                       /*force_network=*/true);
            }
            if (fetched)
                blob = std::move(*fetched);
            else
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "License Error :", fetched.error());
        }

        // Size broadcast (0 = rank 0's fetch failed), then the data
        // broadcast — the standard two-step pattern for a variable-length
        // MPI_Bcast. Every rank must reach both calls regardless of local
        // success/failure, which the Allreduce above guarantees for the
        // license check; the fetch itself only ever runs on rank 0, so its
        // failure mode is already symmetric (every non-zero rank sees the
        // same size-0 broadcast).
        std::uint64_t blob_size = blob.size();
        MPI_Bcast(&blob_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        if (blob_size == 0)
            return finalize(EXIT_FAILURE);
        blob.resize(blob_size);
        MPI_Bcast(blob.data(), static_cast<int>(blob_size), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (auto res = HartreeFock::Loader::initialize_boys_library_from_blob(*local_license, blob); !res)
        {
            if (rank == 0)
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "License Error :", res.error());
            return finalize(EXIT_FAILURE);
        }
#else
        // No MPI: single process, no quota concern, no collective to
        // coordinate — the plain atomic entry point used by the two serial
        // binaries is exactly right here too.
        if (auto res = HartreeFock::Loader::initialize_boys_library(
                HartreeFock::Loader::get_license_path(), blob_url_template);
            !res)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "License Error :", res.error());
            return finalize(EXIT_FAILURE);
        }
#endif
    }

    // Args: <input file> [--json <path>]. Only rank 0 emits usage errors.
    std::string input_file;
    std::string json_path;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc)
            json_path = argv[++i];
        else if (input_file.empty())
            input_file = arg;
        else
        {
            if (rank == 0)
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Usage :",
                                             std::format("{} <input file> [--json <path>]", argv[0]));
            return finalize(EXIT_FAILURE);
        }
    }
    if (input_file.empty())
    {
        if (rank == 0)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Usage :",
                                         std::format("{} <input file> [--json <path>]", argv[0]));
        return finalize(EXIT_FAILURE);
    }

    // Parse. Every rank reads the same input file (cheap, avoids a bespoke
    // Calculator broadcast for Tier 3); a broadcast replaces this when parsing
    // cost matters. All ranks reach the same Calculator, including is_dft_run().
    std::ifstream input_stream(input_file);
    if (!input_stream)
    {
        if (rank == 0)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Input Error :", "Failed to open input file");
        return finalize(EXIT_FAILURE);
    }

    HartreeFock::Calculator calculator{};
    if (auto res = HartreeFock::IO::parse_input(input_stream, calculator); !res)
    {
        if (rank == 0)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Input Parsing Failed :", res.error());
        return finalize(EXIT_FAILURE);
    }

    // Checkpoint path suffix follows the dispatched method, matching the two
    // serial binaries (.dftchk for DFT, .hfchk for HF).
    {
        std::filesystem::path inp(input_file);
        const std::string stem = (inp.parent_path() / inp.stem()).string();
        calculator._checkpoint_path =
            stem + (calculator.is_dft_run() ? ".dftchk" : ".hfchk");
    }

    // Single-writer checkpoint: only rank 0 saves. Non-zero ranks compute the
    // same converged result, so letting them all write the same file would just
    // race. Disable checkpoint saving on those ranks at the source (both the HF
    // and DFT save flags), rather than threading rank through the drivers.
    if (rank != 0)
    {
        calculator._scf._save_checkpoint = false;
        calculator._dft._save_checkpoint = false;
    }

    // Only rank 0 writes JSON; pass an empty path on the other ranks so the
    // driver's serializer stays a single-writer.
    const std::string effective_json = (rank == 0) ? json_path : std::string{};

    // Dispatch on the method the input declared — the whole point of the unified
    // binary. Both drivers are symmetric CLI peers returning the exit code.
    auto result = calculator.is_dft_run()
                      ? DFT::Driver::run(calculator, input_file, effective_json)
                      : HartreeFock::Driver::run(calculator, input_file, effective_json);
    if (!result)
    {
        if (rank == 0)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Driver Failed :", result.error());
        return finalize(EXIT_FAILURE);
    }
    return finalize(*result);
}
