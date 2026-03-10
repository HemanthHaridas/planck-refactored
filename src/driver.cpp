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
#include "integrals/base.h"
#include "scf/scf.h"

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

    // Convert input coordinates to Bohr immediately — must happen before symmetry
    // detection and basis reading, both of which need _coordinates in Bohr.
    calculator.prepare_coordinates();

    // Now log all input options
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Calculation Type :", map_enum(calculator._calculation));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Theory :",           map_enum(calculator._integral._engine));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Basis :",            calculator._basis._basis_name);
    HartreeFock::Logger::blank();

    // Detect Symmetry
    if (!calculator._geometry._use_symm)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Symmetry Detection :", "Symmetry detection is turned off by request");
        // No reorientation — standard frame equals input frame.
        calculator._molecule._standard = calculator._molecule._coordinates;
        HartreeFock::Logger::blank();
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
        HartreeFock::Logger::blank();
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
                oss << std::setw(10) << std::setprecision(3) << std::fixed << calculator._molecule.standard(index, cindex);
                cstr += oss.str();
            }
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "", cstr);
        }
    }
    HartreeFock::Logger::blank();

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
    HartreeFock::Logger::blank();

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "SCF Mode :", map_enum<HartreeFock::SCFMode>(calculator._scf._mode));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Nuclear Repulsion :", std::format("{:.10f} Eh", calculator._nuclear_repulsion));
    HartreeFock::Logger::blank();

    // ── One-electron integrals ────────────────────────────────────────────────
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing overlap and kinetic energy matrices");

    auto [S, T] = _compute_1e(shellpairs, calculator._shells.nbasis(), calculator._integral._engine);
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Overlap and kinetic done");

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing nuclear attraction matrix");
    Eigen::MatrixXd V = _compute_nuclear_attraction(shellpairs, calculator._shells.nbasis(), calculator._molecule, calculator._integral._engine);
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Nuclear attraction done");
    HartreeFock::Logger::blank();

    // ── Core Hamiltonian H = T + V ────────────────────────────────────────────
    calculator._overlap = S;
    calculator._hcore   = T + V;

    if (calculator._output._print_matrices)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Overlap Matrix S :", "");
        std::cout << S << "\n";
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Core Hamiltonian H :", "");
        std::cout << calculator._hcore << "\n";
        HartreeFock::Logger::blank();
    }

    // ── SCF ───────────────────────────────────────────────────────────────────
    if (calculator._scf._scf == HartreeFock::SCFType::RHF)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Begin SCF Cycles :", "");
        if (auto res = HartreeFock::SCF::run_rhf(calculator, shellpairs); !res)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "SCF Failed :", res.error());
            return EXIT_FAILURE;
        }
    }
    else
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "SCF :", "UHF not yet implemented");
        return EXIT_FAILURE;
    }

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Total Energy :", std::format("{:.10f} Eh", calculator._total_energy));

    const auto program_end = SystemClock::now();
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Wall Time :", std::format("{:.3f} s", std::chrono::duration<double>(program_end - program_start).count()));
}
