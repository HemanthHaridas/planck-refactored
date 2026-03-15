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
#include "io/checkpoint.h"
#include "io/logging.h"
#include "symmetry/symmetry.h"
#include "symmetry/mo_symmetry.h"
#include "basis/basis.h"
#include "integrals/shellpair.h"
#include "integrals/base.h"
#include "scf/scf.h"
#include "post_hf/mp2.h"

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

    // Derive checkpoint path: same directory + stem + ".hfchk"
    {
        std::filesystem::path inp(input_file);
        calculator._checkpoint_path =
            (inp.parent_path() / inp.stem()).string() + ".hfchk";
    }

    // Convert input coordinates to Bohr immediately — must happen before symmetry
    // detection and basis reading, both of which need _coordinates in Bohr.
    calculator.prepare_coordinates();

    // Now log all input options
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Calculation Type :", map_enum(calculator._calculation));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Theory :",           map_enum(calculator._scf._scf));
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

    // ── One-electron integrals (or load from checkpoint) ─────────────────────
    bool loaded_from_checkpoint = false;

    if (calculator._scf._guess == HartreeFock::SCFGuess::Read)
    {
        // ── Fast path: same-basis restart ─────────────────────────────────────
        if (auto res = HartreeFock::Checkpoint::load(calculator, calculator._checkpoint_path); res)
        {
            loaded_from_checkpoint = true;
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Checkpoint :",
                std::format("Loaded from {}", calculator._checkpoint_path));
            HartreeFock::Logger::blank();
        }
        else
        {
            // ── Cross-basis projection path ────────────────────────────────────
            auto mos_res = HartreeFock::Checkpoint::load_mos(calculator._checkpoint_path);

            if (mos_res && mos_res->nbasis != calculator._shells.nbasis())
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Checkpoint :",
                    std::format("Basis change detected ({} → {}); projecting density",
                                mos_res->basis_name, calculator._basis._basis_name));

                // 1e integrals must be computed in the large (current) basis
                const std::size_t large_nb = calculator._shells.nbasis();

                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing overlap and kinetic energy matrices");
                auto [S, T] = _compute_1e(shellpairs, large_nb, calculator._integral._engine);
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing nuclear attraction matrix");
                Eigen::MatrixXd V = _compute_nuclear_attraction(shellpairs, large_nb, calculator._molecule, calculator._integral._engine);
                HartreeFock::Logger::blank();

                calculator._overlap = S;
                calculator._hcore   = T + V;
                loaded_from_checkpoint = true;  // skip the unconditional 1e block below

                // Re-read the small basis for cross-overlap
                std::filesystem::path small_gbs =
                    calculator._basis._basis_path + "/" + mos_res->basis_name;

                bool projection_ok = false;
                try
                {
                    const HartreeFock::Basis small_shells =
                        HartreeFock::BasisFunctions::read_gbs_basis(
                            small_gbs, calculator._molecule, calculator._basis._basis);

                    auto X_res = HartreeFock::SCF::build_orthogonalizer(S);
                    if (X_res)
                    {
                        const Eigen::MatrixXd S_cross =
                            HartreeFock::ObaraSaika::_compute_cross_overlap(
                                calculator._shells, small_shells);

                        // Derive occupations from current molecule
                        int n_elec = 0;
                        for (auto z : calculator._molecule.atomic_numbers) n_elec += z;
                        n_elec -= calculator._molecule.charge;
                        const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
                        const int n_alpha    = (n_elec + n_unpaired) / 2;
                        const int n_beta     = (n_elec - n_unpaired) / 2;

                        const bool cur_uhf = (calculator._scf._scf == HartreeFock::SCFType::UHF);

                        if (mos_res->is_uhf)
                        {
                            calculator._info._scf.alpha.density =
                                HartreeFock::Checkpoint::project_density(
                                    *X_res, S_cross, mos_res->C_alpha.leftCols(n_alpha), 1.0);
                            calculator._info._scf.beta.density =
                                HartreeFock::Checkpoint::project_density(
                                    *X_res, S_cross, mos_res->C_beta.leftCols(n_beta), 1.0);
                        }
                        else
                        {
                            // RHF checkpoint
                            const double factor = cur_uhf ? 1.0 : 2.0;
                            calculator._info._scf.alpha.density =
                                HartreeFock::Checkpoint::project_density(
                                    *X_res, S_cross, mos_res->C_alpha.leftCols(n_alpha), factor);
                            if (cur_uhf)
                            {
                                // Use the same RHF MOs as the beta-spin initial guess
                                calculator._info._scf.beta.density =
                                    HartreeFock::Checkpoint::project_density(
                                        *X_res, S_cross, mos_res->C_alpha.leftCols(n_beta), 1.0);
                            }
                        }

                        projection_ok = true;
                        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Checkpoint :", "Density projection successful");
                        HartreeFock::Logger::blank();
                    }
                    else
                    {
                        HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning, "Checkpoint :",
                            std::format("Orthogonalizer failed: {} — using H_core guess", X_res.error()));
                    }
                }
                catch (const std::exception& e)
                {
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning, "Checkpoint :",
                        std::format("Projection failed: {} — using H_core guess", e.what()));
                }

                if (!projection_ok)
                    calculator._scf._guess = HartreeFock::SCFGuess::HCore;
            }
            else
            {
                // Same basis or checkpoint unreadable — full fallback to H_core
                calculator._scf._guess = HartreeFock::SCFGuess::HCore;
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning, "Checkpoint :",
                    std::format("Could not load '{}': {} — computing integrals from scratch",
                                calculator._checkpoint_path, res.error()));
            }
        }
    }

    if (!loaded_from_checkpoint)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing overlap and kinetic energy matrices");

        auto [S, T] = _compute_1e(shellpairs, calculator._shells.nbasis(), calculator._integral._engine);
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Overlap and kinetic done");

        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing nuclear attraction matrix");
        Eigen::MatrixXd V = _compute_nuclear_attraction(shellpairs, calculator._shells.nbasis(), calculator._molecule, calculator._integral._engine);
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Nuclear attraction done");
        HartreeFock::Logger::blank();

        calculator._overlap = S;
        calculator._hcore   = T + V;
    }

    // ── SAO basis for symmetry-blocked Fock diagonalization ──────────────────
    if (calculator._molecule._symmetry &&
        calculator._molecule._point_group != "C1" &&
        calculator._molecule._point_group.find("inf") == std::string::npos)
    {
        try
        {
            auto sao = HartreeFock::Symmetry::build_sao_basis(calculator);
            if (sao.valid)
            {
                calculator._sao_transform     = std::move(sao.transform);
                calculator._sao_irrep_index   = std::move(sao.sao_irrep_index);
                calculator._sao_irrep_names   = std::move(sao.irrep_names);
                calculator._sao_block_sizes   = std::move(sao.block_sizes);
                calculator._sao_block_offsets = std::move(sao.block_offsets);
                calculator._use_sao_blocking  = true;

                // Log irrep distribution, e.g. "A1(4)  B1(1)  B2(2)"
                std::string dist;
                for (std::size_t g = 0; g < calculator._sao_irrep_names.size(); ++g)
                {
                    if (g > 0) dist += "  ";
                    dist += calculator._sao_irrep_names[g] + "(" +
                            std::to_string(calculator._sao_block_sizes[g]) + ")";
                }
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "SAO Basis :", dist);
                HartreeFock::Logger::blank();
            }
        }
        catch (const std::exception& e)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                "SAO Basis :", std::format("Skipped: {}", e.what()));
        }
    }

    if (calculator._output._print_matrices)
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Overlap Matrix S :", "");
        std::cout << calculator._overlap << "\n";
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
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Begin UHF SCF Cycles :", "");
        if (auto res = HartreeFock::SCF::run_uhf(calculator, shellpairs); !res)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "SCF Failed :", res.error());
            return EXIT_FAILURE;
        }
    }

    // ── MO table (with optional symmetry labels) ──────────────────────────────
    if (calculator._info._is_converged)
    {
        // Assign irrep labels when molecule has non-trivial symmetry.
        // When SAO blocking is active, labels are already filled during SCF.
        if (calculator._molecule._symmetry && !calculator._use_sao_blocking)
        {
            try
            {
                HartreeFock::Symmetry::assign_mo_symmetry(calculator);
            }
            catch (const std::exception& e)
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                    "MO Symmetry :", std::format("Skipped: {}", e.what()));
            }
        }

        const bool have_symm = !calculator._info._scf.alpha.mo_symmetry.empty();
        int n_elec = 0;
        for (auto z : calculator._molecule.atomic_numbers) n_elec += z;
        n_elec -= calculator._molecule.charge;

        if (calculator._scf._scf == HartreeFock::SCFType::RHF)
        {
            HartreeFock::Logger::mo_header(have_symm);
            HartreeFock::Logger::mo_energies(
                calculator._info._scf.alpha.mo_energies,
                static_cast<std::size_t>(n_elec),
                calculator._info._scf.alpha.mo_symmetry);
            HartreeFock::Logger::blank();
        }
        else
        {
            const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
            const std::size_t n_alpha = static_cast<std::size_t>((n_elec + n_unpaired) / 2);
            const std::size_t n_beta  = static_cast<std::size_t>((n_elec - n_unpaired) / 2);

            const bool have_symm_b = !calculator._info._scf.beta.mo_symmetry.empty();

            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Alpha MOs :", "");
            HartreeFock::Logger::mo_header(have_symm);
            HartreeFock::Logger::mo_energies_uhf(
                calculator._info._scf.alpha.mo_energies, n_alpha,
                calculator._info._scf.alpha.mo_symmetry);
            HartreeFock::Logger::blank();

            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Beta MOs :", "");
            HartreeFock::Logger::mo_header(have_symm_b);
            HartreeFock::Logger::mo_energies_uhf(
                calculator._info._scf.beta.mo_energies, n_beta,
                calculator._info._scf.beta.mo_symmetry);
            HartreeFock::Logger::blank();
        }
    }

    // ── Save checkpoint ───────────────────────────────────────────────────────
    if (calculator._scf._save_checkpoint && calculator._info._is_converged)
    {
        if (auto res = HartreeFock::Checkpoint::save(calculator, calculator._checkpoint_path); res)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Checkpoint :",
                std::format("Saved to {}", calculator._checkpoint_path));
        else
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning, "Checkpoint :",
                std::format("Save failed: {}", res.error()));
    }

    HartreeFock::Logger::converged_energy(calculator._total_energy);

    // ── Post-HF correlation ───────────────────────────────────────────────────
    if (calculator._info._is_converged)
    {
        std::expected<void, std::string> corr_res;
        std::string corr_tag;

        if (calculator._correlation == HartreeFock::PostHF::RMP2)
        {
            corr_tag = "RMP2 :";
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, corr_tag, "Computing MP2 correlation energy");
            corr_res = HartreeFock::Correlation::run_rmp2(calculator, shellpairs);
        }
        else if (calculator._correlation == HartreeFock::PostHF::UMP2)
        {
            corr_tag = "UMP2 :";
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, corr_tag, "Computing MP2 correlation energy");
            corr_res = HartreeFock::Correlation::run_ump2(calculator, shellpairs);
        }

        if (corr_res.has_value() == false && !corr_tag.empty())
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error,
                corr_tag + " Failed :", corr_res.error());
            return EXIT_FAILURE;
        }

        if (!corr_tag.empty())
        {
            HartreeFock::Logger::blank();
            HartreeFock::Logger::correlation_energy(calculator._total_energy, calculator._correlation_energy);
        }
    }

    const auto program_end = SystemClock::now();
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Wall Time :", std::format("{:.3f} s", std::chrono::duration<double>(program_end - program_start).count()));
}
