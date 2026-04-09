#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "driver.h"
#include "integrals/os.h"
#include "io/io.h"
#include "io/logging.h"

using SystemClock = std::chrono::system_clock;

namespace
{

    std::string format_time(SystemClock::time_point tp)
    {
        const std::time_t t = SystemClock::to_time_t(tp);
        std::tm tm{};

#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif

        std::ostringstream os;
        os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return os.str();
    }

    std::string dft_reference_label(HartreeFock::SCFType scf_type)
    {
        switch (scf_type)
        {
        case HartreeFock::SCFType::RHF:
            return "RKS";
        case HartreeFock::SCFType::ROHF:
            return "ROKS";
        case HartreeFock::SCFType::UHF:
            return "UKS";
        }

        return "Unknown";
    }

    void log_multipole_report(const HartreeFock::Calculator &calculator)
    {
        auto shell_pairs = build_shellpairs(calculator._shells);
        auto moments = HartreeFock::ObaraSaika::_compute_multipole_moments(
            calculator,
            shell_pairs,
            Eigen::Vector3d::Zero());

        if (!moments)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Warning,
                "Multipole Moments :",
                "Unavailable: " + moments.error());
            HartreeFock::Logger::blank();
            return;
        }

        HartreeFock::Logger::multipole_moments(*moments);
        HartreeFock::Logger::blank();
    }

} // namespace

int main(int argc, const char *argv[])
{
    const auto program_start = SystemClock::now();

    if (argc != 2)
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Error,
            "Usage :",
            std::format("{} <input file>", argv[0]));
        return EXIT_FAILURE;
    }

    const std::string input_file = argv[1];
    std::ifstream input_stream(input_file);
    if (!input_stream)
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Error,
            "Input Error :",
            "Failed to open input file");
        return EXIT_FAILURE;
    }

    HartreeFock::Calculator calculator{};
    if (auto res = HartreeFock::IO::parse_input(input_stream, calculator); !res)
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Error,
            "Input Parsing Failed :",
            res.error());
        return EXIT_FAILURE;
    }

    {
        std::filesystem::path inp(input_file);
        calculator._checkpoint_path =
            (inp.parent_path() / inp.stem()).string() + ".dftchk";
    }

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Input Parsing :", "Successful");
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Calculation Type :", map_enum(calculator._calculation));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Theory :", "Kohn-Sham DFT");
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Reference :", dft_reference_label(calculator._scf._scf));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Basis :", calculator._basis._basis_name);
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "DFT Grid :", map_enum(calculator._dft._grid));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Exchange :", map_enum(calculator._dft._exchange));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Correlation :", map_enum(calculator._dft._correlation));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Charge :", calculator._molecule.charge);
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Multiplicity :", calculator._molecule.multiplicity);
    HartreeFock::Logger::blank();

    const auto result = DFT::Driver::run(calculator);
    if (!result)
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Error,
            "DFT Driver Failed :",
            result.error());
        return EXIT_FAILURE;
    }

    if (result->converged)
        log_multipole_report(calculator);

    HartreeFock::Logger::logging(
        HartreeFock::LogLevel::Info,
        "DFT Energy :",
        std::format("{:.10f} Eh", result->total_energy));
    HartreeFock::Logger::logging(
        HartreeFock::LogLevel::Info,
        "Converged :",
        result->converged ? "true" : "false");
    HartreeFock::Logger::logging(
        HartreeFock::LogLevel::Info,
        "Wall Time :",
        std::format(
            "{} ({} seconds)",
            format_time(SystemClock::now()),
            std::chrono::duration<double>(SystemClock::now() - program_start).count()));

    return EXIT_SUCCESS;
}
