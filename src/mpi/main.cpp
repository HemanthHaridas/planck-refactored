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

#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "driver.h"
#include "dft/driver.h"

#include "base/types.h"
#include "io/io.h"
#include "io/logging.h"

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

    auto finalize = [](int code) -> int
    {
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return code;
    };

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
