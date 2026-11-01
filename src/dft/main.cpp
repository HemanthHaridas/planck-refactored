#include <filesystem>
#include <format>
#include <fstream>
#include <string>

#include "driver.h"

#include "base/types.h"
#include "io/io.h"
#include "io/external_libraries.h"
#include "io/logging.h"
#include "lookup/loader/license_path.h"
#include "lookup/loader/orchestrator.h"

// Thin entry-point shell for the planck-dft binary: parse args and input, then
// hand off to DFT::Driver::run — the symmetric peer of HartreeFock::Driver::run.
// The unified planck-mpi binary uses the same DFT::Driver::run after selecting
// HF vs DFT on calculator.is_dft_run().
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
    HartreeFock::Logger::banner();
    HartreeFock::Logger::external_libraries();

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
            input_file.clear(); // force the usage error below
    }
    if (input_file.empty())
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Usage :",
                                     std::format("{} <input file> [--json <path>]", argv[0]));
        return EXIT_FAILURE;
    }

    HartreeFock::Logger::reading_input(input_file);
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

    // Derive checkpoint path: same directory + stem + ".dftchk"
    {
        std::filesystem::path inp(input_file);
        calculator._checkpoint_path =
            (inp.parent_path() / inp.stem()).string() + ".dftchk";
    }

    auto result = DFT::Driver::run(calculator, input_file, json_path);
    if (!result)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "DFT Driver Failed :", result.error());
        return EXIT_FAILURE;
    }
    return *result;
}
