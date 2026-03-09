#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <format>
#include <iomanip>
#include <sstream>
#include <string>
#include <sstream>

#include "base/types.h"
#include "io/io.h"
#include "io/logging.h"
#include "symmetry/symmetry.h"
#include "basis/basis.h"
#include "integrals/shellpair.h"

using SystemClock = std::chrono::system_clock;

static std::string format_time(SystemClock::time_point tp)
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

int main(int argc, const char* argv[])
{
    const auto program_start    = SystemClock::now();   // Start time
    
    if (argc != 2)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Usage :", std::format("{} <input file>", argv[0]));
        return EXIT_FAILURE;
    }
    
    const std::string input_file = argv[1];
    std::ifstream input_stream(input_file);
    
    if (!input_stream)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Input Error :", "Failed to open input file");
        return EXIT_FAILURE;
    }

    // Core objects
    HartreeFock::Calculator calculator{};

    if (auto res = HartreeFock::IO::parse_input(input_stream, calculator); !res)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Input Parsing Failed :", res.error());
        return EXIT_FAILURE;
    }

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Input Parsing :", "Successful");
    
    // Now log all input options
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Calculation Type :", map_enum(calculator._calculation));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Theory :",           map_enum(calculator._integral._engine));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Basis :",            calculator._basis._basis_name);

    // Detect Symmetry
    if (!calculator._geometry._use_symm)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Symmetry Detection :", "Symmetry detection is turned off by request");
        // If symmetry is not detected, set standar coordinates to input coordinates
        calculator._molecule._standard = calculator._molecule._coordinates;
    }

    else
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Symmetry Detection :", "We use libmsym library to detect point groups");
        
        if (auto res = HartreeFock::Symmetry::detectSymmetry(calculator._molecule); !res)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Symmetry Detection Failed :", res.error());
            return EXIT_FAILURE;
        }
        
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Symmetry Detection :", "Successful");
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Point Group :", calculator._molecule._point_group);
    }
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Input Coordinates :", "");

    // get input coordinates
    for (std::size_t index = 0; index < calculator._molecule.natoms; ++index)
    {
        std::string cstr;
        std::ostringstream astream;
        astream << std::setw(5) << std::right << calculator._molecule.atomic_numbers[index];
        cstr += astream.str();

        for (std::size_t cindex = 0; cindex < 3; ++cindex)
        {
            std::ostringstream oss;
            oss << std::setw(10) << std::setprecision(3) << std::fixed << calculator._molecule.coordinates(index, cindex);
            cstr += oss.str();
        }
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "", cstr);
    }

    // get reoriented coordinates
    if (calculator._molecule._symmetry && calculator._output._print_geometry)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Standard Coordinates :", "");
        for (std::size_t index = 0; index < calculator._molecule.natoms; ++index)
        {
            std::string cstr;
            std::ostringstream astream;
            astream << std::setw(5) << std::right << calculator._molecule.atomic_numbers[index];
            cstr += astream.str();

            for (std::size_t cindex = 0; cindex < 3; ++cindex)
            {
                std::ostringstream oss;
                oss << std::setw(10) << std::setprecision(3) << std::fixed << calculator._molecule._standard(index, cindex);
                cstr += oss.str();
            }
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "", cstr);
        }
    }

    // Now read basis set
    std::filesystem::path gbs_path = calculator._basis._basis_path + "/" + calculator._basis._basis_name;
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Reading Basis Set :", gbs_path.string());
    
    try
    {
        calculator._shells = HartreeFock::BasisFunctions::read_gbs_basis(gbs_path, calculator._molecule, calculator._basis._basis); // cartesian or pure
    }
    catch (const std::exception &e)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Basis Parsing Failed :", e.what());
        return EXIT_FAILURE;
    }
    
    // Now initialize SCF data structures
    calculator.initialize();
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Basis Construction :", std::format("Generated {} Shells and {} contracted functions", calculator._shells.nshells(), calculator._shells.nbasis()));
    
    // Now generate shell pairs
    std::vector <HartreeFock::ShellPair> shellpairs = build_shellpairs(calculator._shells);
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Number of Shell pairs :", shellpairs.size());
    
    // Initialize one-electron integrals
    // Initialize two-electron integrals
    const auto program_end      = SystemClock::now();   // End time
}
