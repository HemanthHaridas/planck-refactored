#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include "driver.h"

#include "base/types.h"
#include "io/io.h"
#include "io/logging.h"
#include "lookup/loader/license_path.h"
#include "lookup/loader/orchestrator.h"

// Thin entry-point shell for the serial `hartree-fock` binary: parse args and
// input, then hand off to HartreeFock::Driver::run — the symmetric peer of
// DFT::Driver::run. The unified `planck-mpi` binary uses the same Driver::run
// after selecting HF vs DFT on calculator.is_dft_run().
int main(int argc, const char *argv[])
{
    // Gate on a valid license before doing anything else — no point parsing
    // an input file for a calculation that can't run boys() anyway.
    {
        const std::string blob_url_template =
            std::string("https://boys-blob-worker.boys-core.workers.dev/{version}/planck_boys-") +
            HartreeFock::Loader::get_platform_string() + ".enc";
        if (auto res = HartreeFock::Loader::initialize_boys_library(
                HartreeFock::Loader::get_license_path(), blob_url_template);
            !res)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "License Error :", res.error());
            return EXIT_FAILURE;
        }
    }

    // Args: <input file> [--json <path>]. --json writes machine-readable
    // results for the Python front end; the human log is unaffected.
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
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Usage :",
                                         std::format("{} <input file> [--json <path>]", argv[0]));
            return EXIT_FAILURE;
        }
    }
    if (input_file.empty())
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Usage :",
                                     std::format("{} <input file> [--json <path>]", argv[0]));
        return EXIT_FAILURE;
    }

    std::ifstream input_stream(input_file);
    if (!input_stream)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Input Error :", "Failed to open input file");
        return EXIT_FAILURE;
    }

    HartreeFock::Calculator calculator{};

    if (auto res = HartreeFock::IO::parse_input(input_stream, calculator); !res)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Input Parsing Failed :", res.error());
        return EXIT_FAILURE;
    }

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Input Parsing :", "Successful");

    // Derive checkpoint path: same directory + stem + ".hfchk"
    {
        std::filesystem::path inp(input_file);
        calculator._checkpoint_path =
            (inp.parent_path() / inp.stem()).string() + ".hfchk";
    }

    auto result = HartreeFock::Driver::run(calculator, input_file, json_path);
    if (!result)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Driver Failed :", result.error());
        return EXIT_FAILURE;
    }
    return *result;
}
