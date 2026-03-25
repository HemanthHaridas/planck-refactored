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
#include "symmetry/integral_symmetry.h"
#include "basis/basis.h"
#include "integrals/shellpair.h"
#include "integrals/base.h"
#include "scf/scf.h"
#include "post_hf/mp2.h"
#include "post_hf/casscf.h"
#include "gradient/gradient.h"
#include "opt/geomopt.h"
#include "freq/hessian.h"

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

    // ── guess full: restore geometry from checkpoint before symmetry/basis setup ─
    //
    // When the user requests guess full, the molecule geometry, charge, and
    // multiplicity are taken from the checkpoint (e.g. an optimized geometry)
    // rather than the input file.  This must happen before detectSymmetry() and
    // read_gbs_basis() so that they operate on the checkpoint geometry.
    bool preserve_checkpoint_ao_frame = false;
    if (calculator._scf._guess == HartreeFock::SCFGuess::ReadFull)
    {
        if (auto geo = HartreeFock::Checkpoint::load_geometry(calculator._checkpoint_path); geo)
        {
            // Validate atom count matches
            if (geo->natoms != calculator._molecule.natoms)
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Checkpoint :",
                    std::format("Atom count mismatch: checkpoint has {}, input has {}",
                                geo->natoms, calculator._molecule.natoms));
                return EXIT_FAILURE;
            }

            // Override geometry, charge, and multiplicity
            calculator._molecule._standard     = geo->coords_bohr;
            calculator._molecule._coordinates  = geo->coords_bohr;
            calculator._molecule.coordinates   = geo->coords_bohr / ANGSTROM_TO_BOHR;
            calculator._molecule.charge        = geo->charge;
            calculator._molecule.multiplicity  = geo->multiplicity;
            calculator._molecule.atomic_numbers = geo->atomic_numbers;
            preserve_checkpoint_ao_frame = true;

            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Checkpoint :",
                std::format("Restoring {} geometry from {}{}",
                            geo->has_opt_coords ? "optimized" : "input",
                            calculator._checkpoint_path,
                            geo->has_opt_coords ? " (converged geomopt)" : ""));
        }
        else
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning, "Checkpoint :",
                std::format("Could not read geometry: {} — falling back to guess density",
                            geo.error()));
            // Downgrade to density-only restart
            calculator._scf._guess = HartreeFock::SCFGuess::ReadDensity;
        }
    }

    // Now log all input options
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Calculation Type :", map_enum(calculator._calculation));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Theory :",           map_enum(calculator._scf._scf));
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Basis :",            calculator._basis._basis_name);
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Charge :",           calculator._molecule.charge);
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Multiplicity :",     calculator._molecule.multiplicity);
    HartreeFock::Logger::blank();

    // Detect Symmetry
    if (preserve_checkpoint_ao_frame)
    {
        calculator._molecule._point_group = "C1";
        calculator._molecule._symmetry = false;
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Symmetry Detection :",
            "Skipped for guess full restart to preserve checkpoint AO frame");
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Symmetry Detection :",
            "Checkpoint density and 1e matrices are reused in the stored standard orientation");
        HartreeFock::Logger::blank();
    }

    else if (!calculator._geometry._use_symm)
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

    const bool want_checkpoint =
        (calculator._scf._guess == HartreeFock::SCFGuess::ReadDensity ||
         calculator._scf._guess == HartreeFock::SCFGuess::ReadFull);

    if (want_checkpoint)
    {
        // guess full:    load 1e matrices (geometry matches checkpoint) — skips integral recompute
        // guess density: load density only (geometry from input) — integrals recomputed below
        const bool load_1e = (calculator._scf._guess == HartreeFock::SCFGuess::ReadFull);

        if (auto res = HartreeFock::Checkpoint::load(
                calculator, calculator._checkpoint_path, load_1e); res)
        {
            // For guess full the 1e matrices are valid → skip recompute.
            // For guess density we still need to compute fresh integrals.
            loaded_from_checkpoint = load_1e;
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Checkpoint :",
                std::format("Loaded from {} ({})",
                            calculator._checkpoint_path,
                            load_1e ? "geometry + density" : "density only"));
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
                HartreeFock::Symmetry::update_integral_symmetry(calculator);

                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing overlap and kinetic energy matrices");
                auto [S, T] = _compute_1e(shellpairs, large_nb, calculator._integral._engine,
                                          calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing nuclear attraction matrix");
                Eigen::MatrixXd V = _compute_nuclear_attraction(shellpairs, large_nb, calculator._molecule,
                                                                calculator._integral._engine,
                                                                calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
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
        HartreeFock::Symmetry::update_integral_symmetry(calculator);
        if (calculator._use_integral_symmetry)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Integral Symmetry :",
                std::format("{} signed AO symmetry operations enabled",
                            calculator._integral_symmetry_ops.size()));
        }

        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing overlap and kinetic energy matrices");

        auto [S, T] = _compute_1e(shellpairs, calculator._shells.nbasis(), calculator._integral._engine,
                                  calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Overlap and kinetic done");

        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "1e Integrals :", "Computing nuclear attraction matrix");
        Eigen::MatrixXd V = _compute_nuclear_attraction(shellpairs, calculator._shells.nbasis(),
                                                        calculator._molecule, calculator._integral._engine,
                                                        calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
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

    HartreeFock::Logger::converged_energy(calculator._total_energy, calculator._nuclear_repulsion);

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
        else if (calculator._correlation == HartreeFock::PostHF::CASSCF)
        {
            corr_tag = "CASSCF :";
            calculator._casscf_rhf_energy = calculator._total_energy;
            corr_res = HartreeFock::Correlation::run_casscf(calculator, shellpairs);
        }
        else if (calculator._correlation == HartreeFock::PostHF::RASSCF)
        {
            corr_tag = "RASSCF :";
            calculator._casscf_rhf_energy = calculator._total_energy;
            corr_res = HartreeFock::Correlation::run_rasscf(calculator, shellpairs);
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
            if (calculator._correlation == HartreeFock::PostHF::CASSCF ||
                calculator._correlation == HartreeFock::PostHF::RASSCF)
            {
                HartreeFock::Logger::casscf_summary(
                    calculator._casscf_rhf_energy,
                    calculator._total_energy,
                    calculator._cas_nat_occ,
                    calculator._active_space.nroots,
                    calculator._active_space.nactorb);
            }
            else
            {
                HartreeFock::Logger::correlation_energy(calculator._total_energy, calculator._correlation_energy);
            }

            // Re-save after converged post-HF runs so restartable correlated
            // orbitals/energies land in the checkpoint.
            if (calculator._scf._save_checkpoint)
            {
                if (auto res = HartreeFock::Checkpoint::save(calculator, calculator._checkpoint_path); res)
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Checkpoint :",
                        std::format("Updated {}", calculator._checkpoint_path));
                else
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning, "Checkpoint :",
                        std::format("Post-HF save failed: {}", res.error()));
            }
        }
    }

    // ── Analytic gradient ─────────────────────────────────────────────────────
    if (calculator._info._is_converged &&
        (calculator._calculation == HartreeFock::CalculationType::Gradient ||
         calculator._calculation == HartreeFock::CalculationType::GeomOpt))
    {
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Gradient :", "Computing analytic nuclear gradient");

        Eigen::MatrixXd grad;
        if (calculator._correlation == HartreeFock::PostHF::RMP2)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Gradient :",
                "Using central-difference RMP2 total-energy gradient");
            calculator._total_energy += calculator._correlation_energy;
            grad = HartreeFock::Gradient::compute_rmp2_gradient(calculator);
        }
        else if (calculator._correlation == HartreeFock::PostHF::UMP2)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Gradient :",
                "UMP2 gradient is not implemented");
            return EXIT_FAILURE;
        }
        else if (calculator._info._scf.is_uhf)
            grad = HartreeFock::Gradient::compute_uhf_gradient(calculator, shellpairs);
        else
            grad = HartreeFock::Gradient::compute_rhf_gradient(calculator, shellpairs);
        calculator._gradient = grad;

        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Nuclear Gradient (Ha/Bohr) :", "");
        const std::size_t natoms_g = calculator._molecule.natoms;
        for (std::size_t a = 0; a < natoms_g; ++a)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                std::format("  Atom {:3d}: {:14.8f}  {:14.8f}  {:14.8f}",
                    static_cast<int>(a + 1),
                    grad(a, 0), grad(a, 1), grad(a, 2)));
        }
        const double gmax = grad.cwiseAbs().maxCoeff();
        const double grms = std::sqrt(grad.squaredNorm() / static_cast<double>(natoms_g * 3));
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Gradient max|g| :",
            std::format("{:.6e} Ha/Bohr", gmax));
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Gradient rms|g| :",
            std::format("{:.6e} Ha/Bohr", grms));
        HartreeFock::Logger::blank();
    }

    // ── Constraint validation ─────────────────────────────────────────────────
    if (!calculator._constraints.empty())
    {
        if (calculator._opt_coords != HartreeFock::OptCoords::Internal)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Constraints :",
                "Constrained optimization requires opt_coords internal");
            return EXIT_FAILURE;
        }
        if (calculator._geometry._type != HartreeFock::CoordType::ZMatrix)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error, "Constraints :",
                "Constrained optimization requires coord_type zmatrix");
            return EXIT_FAILURE;
        }
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Constraints :",
            std::format("{} constraint(s) active", calculator._constraints.size()));
    }

    // ── Imaginary Mode Follow: Hessian → find mode → displace → geomopt ─────
    bool imag_follow_armed = false;
    if (calculator._info._is_converged &&
        calculator._calculation == HartreeFock::CalculationType::ImaginaryFollow)
    {
        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
            "Imaginary Follow :", "Computing semi-numerical Hessian");
        try
        {
            auto freq_result = HartreeFock::Freq::compute_hessian(calculator);

            // Store for completeness
            calculator._hessian              = freq_result.hessian;
            calculator._frequencies          = freq_result.frequencies;
            calculator._normal_modes         = freq_result.normal_modes;
            calculator._vibrational_symmetry = freq_result.mode_symmetry;
            calculator._zpe                  = freq_result.zpe;

            if (freq_result.n_imaginary == 0)
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                    "Imaginary Follow :",
                    "No imaginary frequencies found — structure is a minimum; skipping optimization.");
            }
            else
            {
                // frequencies[] is sorted ascending; imaginary modes are negative and first.
                // Scan to find the one with the largest absolute value.
                int    imag_idx = 0;
                double max_abs  = std::abs(freq_result.frequencies[0]);
                for (int i = 1; i < freq_result.n_vib; ++i)
                {
                    if (freq_result.frequencies[i] >= 0.0) break;
                    const double a = std::abs(freq_result.frequencies[i]);
                    if (a > max_abs) { max_abs = a; imag_idx = i; }
                }

                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                    "Imaginary Follow :",
                    std::format("{} imaginary mode(s); following mode {} ({:.2f}i cm\u207b\u00b9), step {:.4f} Bohr",
                        freq_result.n_imaginary, imag_idx + 1,
                        -freq_result.frequencies[imag_idx],
                        calculator._imag_follow_step));

                // Displace _standard (Bohr) along the chosen mode column.
                // normal_modes is 3N×n_vib, unit-norm Cartesian columns, mass-unweighted.
                const std::size_t N_if = calculator._molecule.natoms;
                const double      stp  = calculator._imag_follow_step;
                for (std::size_t a = 0; a < N_if; ++a)
                    for (int d = 0; d < 3; ++d)
                        calculator._molecule._standard(a, d) +=
                            stp * freq_result.normal_modes(static_cast<int>(a) * 3 + d, imag_idx);

                // Keep all three coordinate frames in sync
                calculator._molecule._coordinates = calculator._molecule._standard;
                calculator._molecule.coordinates  = calculator._molecule._standard / ANGSTROM_TO_BOHR;

                // Log displaced geometry
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                    "Displaced Geometry (Angstrom) :", "");
                for (std::size_t a = 0; a < N_if; ++a)
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                        std::format("  Atom {:3d}:  {:14d}  {:14.8f}  {:14.8f}  {:14.8f}",
                            static_cast<int>(a + 1),
                            static_cast<int>(calculator._molecule.atomic_numbers[a]),
                            calculator._molecule.coordinates(a, 0),
                            calculator._molecule.coordinates(a, 1),
                            calculator._molecule.coordinates(a, 2)));
                HartreeFock::Logger::blank();

                imag_follow_armed = true;
            }
        }
        catch (const std::exception& e)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error,
                "Imaginary Follow :", e.what());
            return EXIT_FAILURE;
        }
    }

    // ── Geometry optimization ─────────────────────────────────────────────────
    if (calculator._info._is_converged &&
        (calculator._calculation == HartreeFock::CalculationType::GeomOpt ||
         calculator._calculation == HartreeFock::CalculationType::GeomOptFrequency ||
         (calculator._calculation == HartreeFock::CalculationType::ImaginaryFollow && imag_follow_armed)))
    {
        const bool use_ic = (calculator._opt_coords == HartreeFock::OptCoords::Internal);
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Geometry Optimization :",
            use_ic ? "Starting IC-BFGS optimizer" : "Starting L-BFGS optimizer");
        HartreeFock::Logger::blank();

        auto opt_result = use_ic
            ? HartreeFock::Opt::run_geomopt_ic(calculator)
            : HartreeFock::Opt::run_geomopt(calculator);

        HartreeFock::Logger::blank();
        if (opt_result.converged)
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Geometry Optimization :",
                std::format("Converged in {} steps", opt_result.iterations));
        else
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning, "Geometry Optimization :",
                std::format("Did NOT converge after {} steps", opt_result.iterations));

        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Final Energy :",
            std::format("{:.10f} Eh", opt_result.energy));
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Final max|g| :",
            std::format("{:.6e} Ha/Bohr", opt_result.grad_max));
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Optimized Geometry (Angstrom) :", "");
        for (std::size_t a = 0; a < calculator._molecule.natoms; ++a)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                std::format("  Atom {:3d}:  {:14d}  {:14.8f}  {:14.8f}  {:14.8f}",
                    static_cast<int>(a + 1),
                    static_cast<int>(calculator._molecule.atomic_numbers[a]),
                    opt_result.final_coords(a, 0) * BOHR_TO_ANGSTROM,
                    opt_result.final_coords(a, 1) * BOHR_TO_ANGSTROM,
                    opt_result.final_coords(a, 2) * BOHR_TO_ANGSTROM)
                    );
        }
        HartreeFock::Logger::blank();

        // ── Final SCF at optimized geometry with symmetry enabled ─────────────
        //
        // Run detectSymmetry on the converged structure, then rebuild
        // basis/integrals/SCF and print the point group and MO table.
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
            "Final Symmetry SCF :", "Re-running SCF at optimized geometry with symmetry");
        HartreeFock::Logger::blank();

        // Detect point group of the optimized structure
        if (auto res = HartreeFock::Symmetry::detectSymmetry(calculator._molecule); !res)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                "Symmetry Detection :", std::format("Failed: {} — skipping symmetry SCF", res.error()));
        }
        else
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                "Point Group :", calculator._molecule._point_group);
            HartreeFock::Logger::blank();

            // Rebuild basis from the symmetry-reoriented standard frame
            const std::string gbs_path_sym =
                calculator._basis._basis_path + "/" + calculator._basis._basis_name;
            calculator._shells = HartreeFock::BasisFunctions::read_gbs_basis(
                gbs_path_sym, calculator._molecule, calculator._basis._basis);

            // Reset SCF state
            calculator._info._scf = HartreeFock::DataSCF(
                calculator._scf._scf == HartreeFock::SCFType::UHF);
            calculator._info._scf.initialize(calculator._shells.nbasis());
            calculator._scf.set_scf_mode_auto(calculator._shells.nbasis());
            calculator._info._is_converged = false;
            calculator._use_sao_blocking   = false;

            calculator._compute_nuclear_repulsion();

            auto sp_sym  = build_shellpairs(calculator._shells);
            HartreeFock::Symmetry::update_integral_symmetry(calculator);
            auto [S_sym, T_sym] = _compute_1e(sp_sym, calculator._shells.nbasis(),
                                               calculator._integral._engine,
                                               calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
            auto V_sym = _compute_nuclear_attraction(sp_sym, calculator._shells.nbasis(),
                                                      calculator._molecule,
                                                      calculator._integral._engine,
                                                      calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
            calculator._overlap = S_sym;
            calculator._hcore   = T_sym + V_sym;

            // Try SAO symmetry blocking
            if (calculator._molecule._point_group != "C1" &&
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

                        std::string dist;
                        for (std::size_t g = 0; g < calculator._sao_irrep_names.size(); ++g)
                        {
                            if (g > 0) dist += "  ";
                            dist += calculator._sao_irrep_names[g] + "(" +
                                    std::to_string(calculator._sao_block_sizes[g]) + ")";
                        }
                        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                            "SAO Basis :", dist);
                        HartreeFock::Logger::blank();
                    }
                }
                catch (const std::exception& e)
                {
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                        "SAO Basis :", std::format("Skipped: {}", e.what()));
                }
            }

            // Run SCF
            std::expected<void, std::string> scf_sym_res;
            if (calculator._scf._scf == HartreeFock::SCFType::UHF)
                scf_sym_res = HartreeFock::SCF::run_uhf(calculator, sp_sym);
            else
                scf_sym_res = HartreeFock::SCF::run_rhf(calculator, sp_sym);

            if (!scf_sym_res)
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                    "Final Symmetry SCF :",
                    std::format("SCF failed: {}", scf_sym_res.error()));
            }
            else
            {
                // Assign MO symmetry labels (if not already set by SAO blocking)
                if (calculator._molecule._symmetry && !calculator._use_sao_blocking)
                {
                    try { HartreeFock::Symmetry::assign_mo_symmetry(calculator); }
                    catch (const std::exception& e)
                    {
                        HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                            "MO Symmetry :", std::format("Skipped: {}", e.what()));
                    }
                }

                // Print MO table
                const bool have_symm_f = !calculator._info._scf.alpha.mo_symmetry.empty();
                int n_elec_f = 0;
                for (auto z : calculator._molecule.atomic_numbers) n_elec_f += z;
                n_elec_f -= calculator._molecule.charge;

                if (calculator._scf._scf == HartreeFock::SCFType::RHF)
                {
                    HartreeFock::Logger::mo_header(have_symm_f);
                    HartreeFock::Logger::mo_energies(
                        calculator._info._scf.alpha.mo_energies,
                        static_cast<std::size_t>(n_elec_f),
                        calculator._info._scf.alpha.mo_symmetry);
                    HartreeFock::Logger::blank();
                }
                else
                {
                    const int n_unpaired_f = static_cast<int>(calculator._molecule.multiplicity) - 1;
                    const std::size_t n_alpha_f = static_cast<std::size_t>((n_elec_f + n_unpaired_f) / 2);
                    const std::size_t n_beta_f  = static_cast<std::size_t>((n_elec_f - n_unpaired_f) / 2);
                    const bool have_symm_b_f = !calculator._info._scf.beta.mo_symmetry.empty();

                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Alpha MOs :", "");
                    HartreeFock::Logger::mo_header(have_symm_f);
                    HartreeFock::Logger::mo_energies_uhf(
                        calculator._info._scf.alpha.mo_energies, n_alpha_f,
                        calculator._info._scf.alpha.mo_symmetry);
                    HartreeFock::Logger::blank();

                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Beta MOs :", "");
                    HartreeFock::Logger::mo_header(have_symm_b_f);
                    HartreeFock::Logger::mo_energies_uhf(
                        calculator._info._scf.beta.mo_energies, n_beta_f,
                        calculator._info._scf.beta.mo_symmetry);
                    HartreeFock::Logger::blank();
                }

                HartreeFock::Logger::converged_energy(calculator._total_energy, calculator._nuclear_repulsion);
                HartreeFock::Logger::blank();

                // ── Save checkpoint with optimized geometry ───────────────────
                // Re-save after the final symmetry SCF so the checkpoint holds
                // the converged geometry, the symmetry-frame density, and has
                // has_opt_coords = 1.  This allows "guess full" on a later run.
                if (calculator._scf._save_checkpoint)
                {
                    if (auto cres = HartreeFock::Checkpoint::save(
                            calculator, calculator._checkpoint_path); cres)
                        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                            "Checkpoint :", std::format("Updated with optimized geometry: {}",
                                                         calculator._checkpoint_path));
                    else
                        HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                            "Checkpoint :", std::format("Save failed: {}", cres.error()));
                }
            }
        }
    }

    // ── Vibrational frequency analysis ───────────────────────────────────────
    if (calculator._info._is_converged &&
        (calculator._calculation == HartreeFock::CalculationType::Frequency ||
         calculator._calculation == HartreeFock::CalculationType::GeomOptFrequency))
    {
        // Ensure gradient has been computed for the reference geometry.
        // (For a frequency-only run the analytic gradient was not computed above;
        //  the Hessian routine will call _run_sp_gradient_freq internally which
        //  also updates _gradient, so we don't need a separate gradient call here.)

        HartreeFock::Logger::blank();
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
            "Frequency :", "Computing semi-numerical Hessian (analytic gradients)");

        try
        {
            auto freq_result = HartreeFock::Freq::compute_hessian(calculator);

            // Store results on the calculator
            calculator._hessian      = freq_result.hessian;
            calculator._frequencies  = freq_result.frequencies;
            calculator._normal_modes = freq_result.normal_modes;
            calculator._vibrational_symmetry = freq_result.mode_symmetry;
            calculator._zpe          = freq_result.zpe;

            // ── Print frequency table ─────────────────────────────────────────
            HartreeFock::Logger::blank();
            const int n_vib   = freq_result.n_vib;
            const int n_tr    = static_cast<int>(calculator._molecule.natoms * 3) - n_vib;
            const std::string geo_label = freq_result.is_linear ? "linear" : "non-linear";

            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                "Vibrational Frequencies :", "");
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                std::format("  Molecule: {} ({} T+R modes removed, {} vibrational modes)",
                            geo_label, n_tr, n_vib));
            const bool have_mode_symmetry =
                freq_result.mode_symmetry.size() == static_cast<std::size_t>(n_vib) &&
                !freq_result.mode_symmetry.empty();
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                have_mode_symmetry
                    ? "  ─────────────────────────────────────────────────────"
                    : "  ──────────────────────────────────────────");
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                have_mode_symmetry
                    ? "    Mode    Symmetry    Frequency (cm⁻¹)"
                    : "    Mode    Frequency (cm⁻¹)");
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                have_mode_symmetry
                    ? "  ─────────────────────────────────────────────────────"
                    : "  ────────────────────────────────────────────");

            for (int i = 0; i < n_vib; ++i)
            {
                const double freq = freq_result.frequencies[i];
                if (freq < 0.0)
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                        have_mode_symmetry
                            ? std::format("  {:6d}  {:10s}  {:14.2f}i  (imaginary)",
                                          i + 1,
                                          freq_result.mode_symmetry[static_cast<std::size_t>(i)],
                                          -freq)
                            : std::format("  {:6d}  {:14.2f}i  (imaginary)", i + 1, -freq));
                else
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                        have_mode_symmetry
                            ? std::format("  {:6d}  {:10s}  {:14.2f}",
                                          i + 1,
                                          freq_result.mode_symmetry[static_cast<std::size_t>(i)],
                                          freq)
                            : std::format("  {:6d}  {:14.2f}", i + 1, freq));
            }

            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                have_mode_symmetry
                    ? "  ─────────────────────────────────────────────────────"
                    : "  ────────────────────────────────────────────");

            if (freq_result.n_imaginary > 0)
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Warning,
                    "Frequency :",
                    std::format("{} imaginary frequency(ies) — structure may be a saddle point",
                                freq_result.n_imaginary));

            const double zpe_kcal = freq_result.zpe * 627.509474;
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                "Zero-point energy :",
                std::format("{:.6f} Eh  ({:.2f} kcal/mol)",
                            freq_result.zpe, zpe_kcal));

            // ── Normal mode displacements (mass-unweighted, Cartesian-normalised) ──
            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                "Normal Mode Displacements :", "");
            for (int i = 0; i < n_vib; ++i)
            {
                HartreeFock::Logger::logging(HartreeFock::LogLevel::Info,
                    std::format("Normal Mode {:4d} :", i + 1),
                    have_mode_symmetry
                        ? freq_result.mode_symmetry[static_cast<std::size_t>(i)]
                        : std::string{});
                for (std::size_t a = 0; a < calculator._molecule.natoms; ++a)
                {
                    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "",
                        std::format("  {:4d}   {:12.8f}   {:12.8f}   {:12.8f}",
                                    static_cast<int>(a + 1),
                                    freq_result.normal_modes(static_cast<int>(a) * 3 + 0, i),
                                    freq_result.normal_modes(static_cast<int>(a) * 3 + 1, i),
                                    freq_result.normal_modes(static_cast<int>(a) * 3 + 2, i)));
                }
            }
            HartreeFock::Logger::blank();
        }
        catch (const std::exception& e)
        {
            HartreeFock::Logger::logging(HartreeFock::LogLevel::Error,
                "Frequency Failed :", e.what());
            return EXIT_FAILURE;
        }
    }

    const auto program_end = SystemClock::now();
    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Wall Time :", std::format("{:.3f} s", std::chrono::duration<double>(program_end - program_start).count()));
}
