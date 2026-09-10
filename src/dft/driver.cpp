#include "driver.h"

#include "analytic_hessian.h"
#include "ks_orbital_hessian.h"
#include "dh_gradient.h"
#include "dh_relaxed_density.h"
#include "post_hf/rhf_response.h"
#include "post_hf/casscf/aug-hessian.h"
#include "post_hf/casscf/orbital.h"

#include <Eigen/QR>

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

#include "base/mpi_env.h"
#include "base/wrapper.h"
#include "basis/basis.h"
#include "dft_gradient.h"
#include "dft_kernel_gradient.h"
#include "freq/hessian.h"
#include "gradient/gradient.h"
#include "integrals/base.h"
#include "integrals/os.h"
#include "io/checkpoint.h"
#include "io/logging.h"
#include "io/results_json.h"
#include "opt/geomopt.h"
#include "populations/multipole.h"
#include "post_hf/integrals.h"
#include "post_hf/mp2.h"
#include "scf/scf.h"
#include "symmetry/integral_symmetry.h"
#include "symmetry/mo_symmetry.h"
#include "symmetry/symmetry.h"

namespace DFT::Driver
{

    // MPI grid partition: this rank's contiguous slice [begin, end) of the
    // npoints grid points, split as evenly as possible. Serial builds get
    // rank 0 / size 1 => [0, npoints), i.e. all points, so the sliced grid
    // calls degrade to the whole-grid form. The remainder (npoints % size) is
    // spread one-per-rank across the low ranks so slices differ by at most one
    // point. Contiguous (not strided) keeps each rank's points spatially
    // coherent for AO screening; the reduce is a sum so order does not matter.
    inline std::pair<Eigen::Index, Eigen::Index>
    mpi_grid_slice(Eigen::Index npoints)
    {
        const Eigen::Index size = static_cast<Eigen::Index>(HartreeFock::Mpi::size());
        const Eigen::Index rank = static_cast<Eigen::Index>(HartreeFock::Mpi::rank());
        if (size <= 1)
            return {0, npoints};
        const Eigen::Index base = npoints / size;
        const Eigen::Index rem = npoints % size;
        const Eigen::Index begin = rank * base + std::min(rank, rem);
        const Eigen::Index end = begin + base + (rank < rem ? 1 : 0);
        return {begin, end};
    }

    // Sum the slice-local XC energy scalars across ranks so the full-grid totals
    // feed the SCF energy and the electron-count check. Per-point arrays inside
    // xc_grid stay slice-local (each rank only assembles its own points), so
    // ONLY these reduced scalars are touched. No-op when serial.
    inline void reduce_partial_xc_scalars(XCGridEvaluation &xc_grid)
    {
        if (!HartreeFock::Mpi::distributed())
            return;
        double scalars[4] = {
            xc_grid.total_energy, xc_grid.exchange_energy,
            xc_grid.correlation_energy, xc_grid.integrated_electrons};
        HartreeFock::Mpi::allreduce_inplace(scalars, 4);
        xc_grid.total_energy = scalars[0];
        xc_grid.exchange_energy = scalars[1];
        xc_grid.correlation_energy = scalars[2];
        xc_grid.integrated_electrons = scalars[3];
    }

    namespace
    {

        DFT::GridLevel to_grid_level(HartreeFock::DFTGridQuality quality)
        {
            switch (quality)
            {
            case HartreeFock::DFTGridQuality::Coarse:
                return DFT::GridLevel::Coarse;
            case HartreeFock::DFTGridQuality::Normal:
                return DFT::GridLevel::Normal;
            case HartreeFock::DFTGridQuality::Fine:
                return DFT::GridLevel::Fine;
            case HartreeFock::DFTGridQuality::UltraFine:
                return DFT::GridLevel::UltraFine;
            }

            return DFT::GridLevel::Normal;
        }

        constexpr double NUMERICAL_GRADIENT_STEP_BOHR = 1.0e-3;

        bool dft_gradient_debug_enabled()
        {
            const char *value = std::getenv("PLANCK_DFT_GRADIENT_DEBUG");
            return value != nullptr && std::string(value) != "0";
        }

        bool dft_allow_unvalidated_range_separated_workflows()
        {
            const char *value = std::getenv("PLANCK_DFT_ALLOW_RS_WORKFLOWS");
            return value != nullptr && std::string(value) != "0";
        }

        // Double-hybrid analytic gradient is in development
        // (docs/DOUBLE_HYBRID_GRADIENT_SCOPE.md, N3). Gated behind this flag
        // until the FD check (N3.4) passes; step 6 lifts the gate and removes
        // the flag.
        bool dft_allow_double_hybrid_gradient()
        {
            const char *value = std::getenv("PLANCK_DFT_DH_GRADIENT");
            return value != nullptr && std::string(value) != "0";
        }

        Eigen::MatrixXd rotate_gradient_rows(
            const Eigen::Ref<const Eigen::MatrixXd> &gradient,
            const Eigen::Matrix3d &rotation)
        {
            return (gradient * rotation).eval();
        }

        struct LinearResponseContribution
        {
            int occupied = 0;
            int virtual_orbital = 0;
            double weight = 0.0;
            std::string spin_label;
            std::string occupied_symmetry;
            std::string virtual_symmetry;
        };

        struct LinearResponseRoot
        {
            int root = 0;
            double excitation_energy = 0.0;
            double excitation_energy_ev = 0.0;
            double wavelength_nm = 0.0;
            Eigen::Vector3d transition_dipole = Eigen::Vector3d::Zero();
            double oscillator_strength = 0.0;
            std::vector<LinearResponseContribution> dominant_contributions;
        };

        struct UVVisSpectrumPoint
        {
            double energy_ev = 0.0;
            double wavelength_nm = 0.0;
            double intensity = 0.0;
        };

        struct DavidsonEigenpairResult
        {
            Eigen::VectorXd eigenvalues;
            Eigen::MatrixXd eigenvectors;
            Eigen::VectorXd residual_norms;
            int iterations = 0;
            bool converged = false;
        };

        template <typename Apply>
        std::expected<DavidsonEigenpairResult, std::string> davidson_lowest_eigenpairs(
            int dimension,
            int nroots,
            const Eigen::VectorXd &diagonal,
            Apply &&apply,
            double tolerance,
            int max_iterations,
            int max_subspace_dimension)
        {
            if (dimension <= 0 || nroots <= 0)
            {
                return DavidsonEigenpairResult{
                    .eigenvalues = Eigen::VectorXd(),
                    .eigenvectors = Eigen::MatrixXd(dimension, 0),
                    .residual_norms = Eigen::VectorXd(),
                    .iterations = 0,
                    .converged = true};
            }

            if (diagonal.size() != dimension)
                return std::unexpected("Davidson solver received a diagonal with inconsistent dimension");

            const int nr = std::min(nroots, dimension);
            const int initial_subspace = std::min(dimension, std::max(2 * nr, 4));
            const int subspace_limit = std::max(initial_subspace + 1, max_subspace_dimension);

            Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(dimension, initial_subspace);
            std::vector<int> order(static_cast<std::size_t>(dimension));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int lhs, int rhs)
                             { return diagonal(lhs) < diagonal(rhs); });
            for (int col = 0; col < initial_subspace; ++col)
                basis(order[static_cast<std::size_t>(col)], col) = 1.0;

            {
                Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
                basis = qr.householderQ() * Eigen::MatrixXd::Identity(dimension, initial_subspace);
            }

            DavidsonEigenpairResult result{
                .eigenvalues = Eigen::VectorXd::Zero(nr),
                .eigenvectors = Eigen::MatrixXd::Zero(dimension, nr),
                .residual_norms = Eigen::VectorXd::Constant(nr, std::numeric_limits<double>::infinity()),
                .iterations = 0,
                .converged = false};

            for (int iteration = 0; iteration < max_iterations; ++iteration)
            {
                const int m = static_cast<int>(basis.cols());
                Eigen::MatrixXd sigma_basis(dimension, m);
                for (int col = 0; col < m; ++col)
                    sigma_basis.col(col) = apply(basis.col(col));

                const Eigen::MatrixXd projected = basis.transpose() * sigma_basis;
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> projected_solver(projected);
                if (projected_solver.info() != Eigen::Success)
                    return std::unexpected("TDDFT / Davidson projected diagonalization failed");

                result.iterations = iteration + 1;
                result.eigenvalues = projected_solver.eigenvalues().head(nr);
                const Eigen::MatrixXd projected_vectors = projected_solver.eigenvectors().leftCols(nr);
                result.eigenvectors = basis * projected_vectors;

                double max_residual = 0.0;
                std::vector<Eigen::VectorXd> corrections;
                corrections.reserve(static_cast<std::size_t>(nr));

                for (int root = 0; root < nr; ++root)
                {
                    Eigen::VectorXd residual =
                        sigma_basis * projected_vectors.col(root) -
                        result.eigenvalues(root) * result.eigenvectors.col(root);
                    result.residual_norms(root) = residual.norm();
                    max_residual = std::max(max_residual, result.residual_norms(root));
                    if (result.residual_norms(root) <= tolerance)
                        continue;

                    for (int idx = 0; idx < dimension; ++idx)
                    {
                        double denom = result.eigenvalues(root) - diagonal(idx);
                        if (std::abs(denom) < 1e-10)
                            denom = (denom >= 0.0) ? 1e-10 : -1e-10;
                        residual(idx) /= denom;
                    }

                    for (int col = 0; col < m; ++col)
                        residual -= basis.col(col).dot(residual) * basis.col(col);
                    for (const Eigen::VectorXd &accepted : corrections)
                        residual -= accepted.dot(residual) * accepted;

                    const double norm = residual.norm();
                    if (norm > 1e-12)
                        corrections.push_back(residual / norm);
                }

                if (max_residual <= tolerance)
                {
                    result.converged = true;
                    return result;
                }

                if (corrections.empty())
                    return result;

                if (m + static_cast<int>(corrections.size()) > subspace_limit)
                {
                    basis = result.eigenvectors.leftCols(nr);
                    for (int col = 0; col < static_cast<int>(basis.cols()); ++col)
                    {
                        for (int prev = 0; prev < col; ++prev)
                            basis.col(col) -= basis.col(prev).dot(basis.col(col)) * basis.col(prev);
                        const double norm = basis.col(col).norm();
                        if (norm > 1e-12)
                            basis.col(col) /= norm;
                    }
                    const int restart_cols = static_cast<int>(basis.cols());
                    basis.conservativeResize(Eigen::NoChange, restart_cols + static_cast<int>(corrections.size()));
                    for (int idx = 0; idx < static_cast<int>(corrections.size()); ++idx)
                        basis.col(restart_cols + idx) = corrections[static_cast<std::size_t>(idx)];
                }
                else
                {
                    const int old_cols = m;
                    basis.conservativeResize(Eigen::NoChange, old_cols + static_cast<int>(corrections.size()));
                    for (int idx = 0; idx < static_cast<int>(corrections.size()); ++idx)
                        basis.col(old_cols + idx) = corrections[static_cast<std::size_t>(idx)];
                }

                for (int col = 0; col < static_cast<int>(basis.cols()); ++col)
                {
                    for (int prev = 0; prev < col; ++prev)
                        basis.col(col) -= basis.col(prev).dot(basis.col(col)) * basis.col(prev);
                    const double norm = basis.col(col).norm();
                    if (norm < 1e-12)
                    {
                        if (col + 1 < static_cast<int>(basis.cols()))
                            basis.col(col) = basis.col(basis.cols() - 1);
                        basis.conservativeResize(Eigen::NoChange, basis.cols() - 1);
                        --col;
                    }
                    else
                    {
                        basis.col(col) /= norm;
                    }
                }
            }

            return result;
        }

        std::expected<int, std::string> resolve_functional_id(
            HartreeFock::XCExchangeFunctional functional,
            int explicit_id)
        {
            // The input layer can name a functional symbolically (PBE, B3LYP,
            // ...) or pass a raw libxc id.  Normalize both cases here so the
            // rest of the driver only deals with resolved libxc identifiers.
            if (explicit_id > 0)
                return explicit_id;

            const char *functional_name = nullptr;
            switch (functional)
            {
            case HartreeFock::XCExchangeFunctional::Custom:
                return std::unexpected("No explicit libxc exchange functional id was provided for Custom exchange");
            case HartreeFock::XCExchangeFunctional::Slater:
                functional_name = "lda_x";
                break;
            case HartreeFock::XCExchangeFunctional::B88:
                functional_name = "gga_x_b88";
                break;
            case HartreeFock::XCExchangeFunctional::PW91:
                functional_name = "gga_x_pw91";
                break;
            case HartreeFock::XCExchangeFunctional::PBE:
                functional_name = "gga_x_pbe";
                break;
            case HartreeFock::XCExchangeFunctional::B3LYP:
                functional_name = "hyb_gga_xc_b3lyp";
                break;
            case HartreeFock::XCExchangeFunctional::PBE0:
                functional_name = "hyb_gga_xc_pbeh";
                break;
            }

            return DFT::XC::functional_id(functional_name);
        }

        std::expected<int, std::string> resolve_functional_id(
            HartreeFock::XCCorrelationFunctional functional,
            int explicit_id)
        {
            if (explicit_id > 0)
                return explicit_id;

            const char *functional_name = nullptr;
            switch (functional)
            {
            case HartreeFock::XCCorrelationFunctional::Custom:
                return std::unexpected("No explicit libxc correlation functional id was provided for Custom correlation");
            case HartreeFock::XCCorrelationFunctional::VWN5:
                // libxc names VWN5 plain "lda_c_vwn" (id 7); "lda_c_vwn_5"
                // does not exist in libxc's functional table at all (verified
                // against src/external/libxc/install/include/xc_funcs.h --
                // VWN1-4 are lda_c_vwn_{1,2,3,4}, VWN5 has no numeric suffix).
                // Found while building the F3.1 verification probe
                // (docs/DFT_ANALYTIC_FXC_HESSIAN.md): `correlation vwn5`
                // has never resolved to a real functional, and nothing in the
                // regression suite exercises VWN5 to have caught it.
                functional_name = "lda_c_vwn";
                break;
            case HartreeFock::XCCorrelationFunctional::LYP:
                functional_name = "gga_c_lyp";
                break;
            case HartreeFock::XCCorrelationFunctional::P86:
                functional_name = "gga_c_p86";
                break;
            case HartreeFock::XCCorrelationFunctional::PW91:
                functional_name = "gga_c_pw91";
                break;
            case HartreeFock::XCCorrelationFunctional::PBE:
                functional_name = "gga_c_pbe";
                break;
            }

            return DFT::XC::functional_id(functional_name);
        }

        std::expected<void, std::string> setup_symmetry(
            HartreeFock::Calculator &calculator,
            const Options &options,
            bool preserve_checkpoint_ao_frame)
        {
            // Full restarts are special: the checkpoint density and orbitals
            // live in the original AO frame used when they were written.  If
            // we re-standardized the molecule before loading them, every AO-
            // indexed quantity would silently refer to the wrong frame.
            if (preserve_checkpoint_ao_frame)
            {
                calculator._molecule._point_group = "C1";
                calculator._molecule._symmetry = false;
                calculator._molecule._symmetry_alignment_transform.setIdentity();
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Symmetry :",
                    "Skipped for guess full restart to preserve checkpoint AO frame");
                return {};
            }

            if (!options.use_symmetry || !calculator._geometry._use_symm)
            {
                calculator._molecule.set_standard_from_bohr(calculator._molecule._coordinates);
                calculator._molecule._symmetry = false;
                calculator._molecule._point_group = "C1";
                calculator._molecule._symmetry_alignment_transform.setIdentity();
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Symmetry :",
                    "Skipped; using input orientation");
                return {};
            }

            if (auto res = HartreeFock::Symmetry::detectSymmetry(
                    calculator._molecule,
                    calculator._geometry._units);
                !res)
                return std::unexpected("DFT symmetry detection failed: " + res.error());

            // Keep every downstream DFT subsystem in the same frame. Basis
            // construction and nuclear-derivative code use molecule._standard,
            // while grid generation prefers molecule._coordinates.
            calculator.sync_coordinate_frames_from_standard();

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Symmetry :",
                std::format("Detected point group {}", calculator._molecule._point_group));
            return {};
        }

        bool wants_checkpoint_restart(const HartreeFock::Calculator &calculator) noexcept
        {
            return calculator._scf._guess == HartreeFock::SCFGuess::ReadDensity ||
                   calculator._scf._guess == HartreeFock::SCFGuess::ReadFull;
        }

        struct RestartState
        {
            bool density_loaded = false;
            bool one_e_ready = false;
        };

        std::expected<bool, std::string> restore_geometry_for_full_restart(
            HartreeFock::Calculator &calculator)
        {
            if (calculator._scf._guess != HartreeFock::SCFGuess::ReadFull)
                return false;

            // ReadFull tries to restore the complete SCF state, including the
            // geometry that defined the checkpoint AO basis.  If geometry load
            // fails we degrade gracefully to ReadDensity, which can still reuse
            // a density matrix when the current geometry/basis are compatible.
            auto geometry = HartreeFock::Checkpoint::load_geometry(calculator._checkpoint_path);
            if (!geometry)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "DFT Checkpoint :",
                    std::format("Could not read geometry: {} - falling back to guess density",
                                geometry.error()));
                calculator._scf._guess = HartreeFock::SCFGuess::ReadDensity;
                return false;
            }

            if (geometry->natoms != calculator._molecule.natoms)
            {
                return std::unexpected(std::format(
                    "Checkpoint atom count mismatch: checkpoint has {}, input has {}",
                    geometry->natoms,
                    calculator._molecule.natoms));
            }

            calculator._molecule.set_standard_from_bohr(geometry->coords_bohr);
            calculator._molecule._coordinates = geometry->coords_bohr;
            calculator._molecule.coordinates = geometry->coords_bohr / ANGSTROM_TO_BOHR;
            calculator._molecule.charge = geometry->charge;
            calculator._molecule.multiplicity = geometry->multiplicity;
            calculator._molecule.atomic_numbers = geometry->atomic_numbers;

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Checkpoint :",
                std::format(
                    "Restoring {} geometry from {}{}",
                    geometry->has_opt_coords ? "optimized" : "input",
                    calculator._checkpoint_path,
                    geometry->has_opt_coords ? " (converged geomopt)" : ""));
            return true;
        }

        std::expected<void, std::string> read_basis_and_initialize(HartreeFock::Calculator &calculator)
        {
            const std::string gbs_path =
                calculator._basis._basis_path + "/" + calculator._basis._basis_name;
            auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
                gbs_path,
                calculator._molecule,
                calculator._basis._basis);
            if (!basis_res)
                return std::unexpected("DFT basis setup failed: " + basis_res.error());
            calculator._shells = std::move(*basis_res);
            calculator.initialize();
            return {};
        }

        void maybe_build_sao_basis(HartreeFock::Calculator &calculator, const Options &options)
        {
            // SAO blocking only helps when symmetry is both enabled and
            // non-trivial.  In C1 or linear infinite groups there is no useful
            // irrep partition to exploit, so keep the ordinary AO machinery.
            if (!calculator._dft._use_sao_blocking ||
                !options.use_sao_blocking ||
                !calculator._molecule._symmetry ||
                calculator._molecule._point_group == "C1" ||
                calculator._molecule._point_group.find("inf") != std::string::npos)
                return;

            // Suppressed under PCM: the cavity is tessellated with a Fibonacci
            // (golden-angle) sphere (src/solvation/pcm.cpp) and carries no
            // point-group symmetry, so V_pcm is not symmetry-adapted. Block
            // diagonalization reads only the diagonal irrep blocks of the KS
            // matrix and silently discards the off-block reaction-field elements,
            // converging to a symmetry-projected solution. Unlike HF (whose DIIS
            // gate catches it) the KS convergence test would report success on
            // the wrong energy: water/STO-3G/PBE/C-PCM gave -75.2062610742
            // (projected) vs the true -75.2062005342.
            if (calculator._solvation._model != HartreeFock::SolvationModel::None)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, "DFT SAO :",
                    "Disabled: the PCM cavity tessellation is not symmetry-adapted, "
                    "so symmetry-blocked diagonalization would project away part of "
                    "the reaction field. Running without SAO blocking.");
                return;
            }

            auto sao = HartreeFock::Symmetry::build_sao_basis(calculator);
            if (!sao)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "DFT SAO :",
                    std::format("Skipped: {}", sao.error()));
                return;
            }

            if (!sao->valid)
                return;

            calculator._sao_transform = std::move(sao->transform);
            calculator._sao_irrep_index = std::move(sao->sao_irrep_index);
            calculator._sao_irrep_names = std::move(sao->irrep_names);
            calculator._sao_block_sizes = std::move(sao->block_sizes);
            calculator._sao_block_offsets = std::move(sao->block_offsets);
            calculator._use_sao_blocking = true;
        }

        void reset_sao_state(HartreeFock::Calculator &calculator)
        {
            calculator._sao_transform.resize(0, 0);
            calculator._sao_irrep_index.clear();
            calculator._sao_irrep_names.clear();
            calculator._sao_block_sizes.clear();
            calculator._sao_block_offsets.clear();
            calculator._use_sao_blocking = false;
        }

        std::expected<void, std::string> compute_one_electron_terms(
            HartreeFock::Calculator &calculator,
            const std::vector<HartreeFock::ShellPair> &shell_pairs)
        {
            // Integral symmetry metadata depends on the current molecular
            // orientation and shell layout, so refresh it before every one-
            // electron build for the current geometry.
            HartreeFock::Symmetry::update_integral_symmetry(calculator);

            auto [S, T] = _compute_1e(
                shell_pairs,
                calculator._shells.nbasis(),
                calculator._integral._engine,
                calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

            const Eigen::MatrixXd V = _compute_nuclear_attraction(
                shell_pairs,
                calculator._shells.nbasis(),
                calculator._molecule,
                calculator._integral._engine,
                calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

            calculator._overlap = std::move(S);
            calculator._hcore = T + V;
            return {};
        }

        std::expected<void, std::string> initialize_ks_guess(HartreeFock::Calculator &calculator)
        {
            const auto X = HartreeFock::SCF::build_orthogonalizer(calculator._overlap);
            if (!X)
                return std::unexpected("DFT orthogonalizer build failed: " + X.error());

            int n_electrons = 0;
            for (auto z : calculator._molecule.atomic_numbers)
                n_electrons += z;
            n_electrons -= calculator._molecule.charge;

            const auto make_spin_density = [&calculator, &X](std::size_t n_occ, bool doubled_occupancy) -> Eigen::MatrixXd
            {
                // The fallback KS guess is the diagonalized core Hamiltonian in
                // the orthonormal AO basis, identical in spirit to the HCore
                // SCF guess used on the Hartree-Fock side.
                const Eigen::MatrixXd Hprime = X->transpose() * calculator._hcore * (*X);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Hprime);
                const Eigen::MatrixXd C = (*X) * solver.eigenvectors();
                const Eigen::MatrixXd C_occ = C.leftCols(n_occ);
                const double occupancy = doubled_occupancy ? 2.0 : 1.0;
                return (occupancy * C_occ * C_occ.transpose()).eval();
            };

            const std::size_t n_occ = static_cast<std::size_t>(std::max(0, n_electrons / 2));
            calculator._info._scf.alpha.density = make_spin_density(n_occ, true);

            if (calculator._scf._scf == HartreeFock::SCFType::UHF)
            {
                const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
                const std::size_t n_alpha = static_cast<std::size_t>((n_electrons + n_unpaired) / 2);
                const std::size_t n_beta = static_cast<std::size_t>((n_electrons - n_unpaired) / 2);
                calculator._info._scf.alpha.density = make_spin_density(n_alpha, false);
                calculator._info._scf.beta.density = make_spin_density(n_beta, false);
            }

            return {};
        }

        std::expected<bool, std::string> project_ks_restart_density(
            HartreeFock::Calculator &calculator,
            const std::vector<HartreeFock::ShellPair> &shell_pairs)
        {
            auto mos = HartreeFock::Checkpoint::load_mos(calculator._checkpoint_path);
            if (!mos)
                return std::unexpected(mos.error());

            if (mos->nbasis == calculator._shells.nbasis())
                return false;

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Checkpoint :",
                std::format("Basis change detected ({} -> {}); projecting density",
                            mos->basis_name,
                            calculator._basis._basis_name));

            if (auto one_e = compute_one_electron_terms(calculator, shell_pairs); !one_e)
                return std::unexpected("Current-basis 1e integral build failed before density projection: " + one_e.error());

            const std::string small_gbs =
                calculator._basis._basis_path + "/" + mos->basis_name;
            auto small_shells = HartreeFock::BasisFunctions::read_gbs_basis(
                small_gbs,
                calculator._molecule,
                calculator._basis._basis);
            if (!small_shells)
                return std::unexpected(small_shells.error());

            auto orthogonalizer = HartreeFock::SCF::build_orthogonalizer(calculator._overlap);
            if (!orthogonalizer)
                return std::unexpected("Orthogonalizer failed before density projection: " + orthogonalizer.error());

            const Eigen::MatrixXd cross_overlap =
                HartreeFock::ObaraSaika::_compute_cross_overlap(
                    calculator._shells,
                    *small_shells);

            int n_electrons = 0;
            for (auto z : calculator._molecule.atomic_numbers)
                n_electrons += z;
            n_electrons -= calculator._molecule.charge;

            const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
            const int n_alpha = (n_electrons + n_unpaired) / 2;
            const int n_beta = (n_electrons - n_unpaired) / 2;
            const bool current_unrestricted =
                calculator._scf._scf == HartreeFock::SCFType::UHF;

            if (mos->is_uhf)
            {
                calculator._info._scf.alpha.density =
                    HartreeFock::Checkpoint::project_density(
                        *orthogonalizer,
                        cross_overlap,
                        mos->C_alpha.leftCols(n_alpha),
                        1.0);
                calculator._info._scf.beta.density =
                    HartreeFock::Checkpoint::project_density(
                        *orthogonalizer,
                        cross_overlap,
                        mos->C_beta.leftCols(n_beta),
                        1.0);
            }
            else
            {
                const double alpha_factor = current_unrestricted ? 1.0 : 2.0;
                calculator._info._scf.alpha.density =
                    HartreeFock::Checkpoint::project_density(
                        *orthogonalizer,
                        cross_overlap,
                        mos->C_alpha.leftCols(n_alpha),
                        alpha_factor);

                if (current_unrestricted)
                {
                    calculator._info._scf.beta.density =
                        HartreeFock::Checkpoint::project_density(
                            *orthogonalizer,
                            cross_overlap,
                            mos->C_alpha.leftCols(n_beta),
                            1.0);
                }
            }

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Checkpoint :",
                "Density projection successful");
            return true;
        }

        std::expected<RestartState, std::string> load_ks_restart_state(
            HartreeFock::Calculator &calculator,
            const std::vector<HartreeFock::ShellPair> &shell_pairs)
        {
            if (!wants_checkpoint_restart(calculator))
                return RestartState{};

            const bool load_1e_matrices = (calculator._scf._guess == HartreeFock::SCFGuess::ReadFull);
            const auto load = HartreeFock::Checkpoint::load(
                calculator,
                calculator._checkpoint_path,
                load_1e_matrices);
            if (load)
            {
                calculator.initialize();
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Checkpoint :",
                    std::format(
                        "Loaded from {} ({})",
                        calculator._checkpoint_path,
                        load_1e_matrices ? "geometry + density" : "density only"));
                return RestartState{
                    .density_loaded = true,
                    .one_e_ready = load_1e_matrices};
            }

            auto projected = project_ks_restart_density(calculator, shell_pairs);
            if (projected && *projected)
            {
                return RestartState{
                    .density_loaded = true,
                    .one_e_ready = true};
            }

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Warning,
                "DFT Checkpoint :",
                std::format("Restart load failed: {} - using HCore guess",
                            load.error()));
            if (!projected)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "DFT Checkpoint :",
                    std::format("Density projection failed: {} - using HCore guess",
                                projected.error()));
            }

            calculator._scf._guess = HartreeFock::SCFGuess::HCore;
            return RestartState{};
        }
        double density_trace_product(
            const Eigen::Ref<const Eigen::MatrixXd> &density,
            const Eigen::Ref<const Eigen::MatrixXd> &matrix)
        {
            return (density.array() * matrix.array()).sum();
        }


        struct DiagonalizationResult
        {
            Eigen::MatrixXd coefficients;
            Eigen::VectorXd energies;
            std::vector<std::string> mo_symmetry;
        };

        std::expected<DiagonalizationResult, std::string> diagonalize_in_ao_basis(
            const HartreeFock::Calculator &calculator,
            const Eigen::Ref<const Eigen::MatrixXd> &orthogonalizer,
            const Eigen::Ref<const Eigen::MatrixXd> &fock,
            const std::string &label)
        {
            const Eigen::Index nbasis = static_cast<Eigen::Index>(calculator._shells.nbasis());
            const bool sao_active = calculator._use_sao_blocking &&
                                    calculator._sao_transform.rows() == nbasis &&
                                    calculator._sao_transform.cols() == nbasis &&
                                    !calculator._sao_block_sizes.empty();

            if (!sao_active)
            {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
                    orthogonalizer.transpose() * fock * orthogonalizer);
                if (solver.info() != Eigen::Success)
                    return std::unexpected(label + " Fock diagonalization failed");

                return DiagonalizationResult{
                    .coefficients = orthogonalizer * solver.eigenvectors(),
                    .energies = solver.eigenvalues(),
                    .mo_symmetry = {}};
            }

            const Eigen::MatrixXd fock_sao =
                calculator._sao_transform.transpose() * fock * calculator._sao_transform;
            const int n_blocks = static_cast<int>(calculator._sao_block_sizes.size());

            Eigen::VectorXd energies_sao(nbasis);
            Eigen::MatrixXd coefficients_sao = Eigen::MatrixXd::Zero(nbasis, nbasis);
            std::vector<int> mo_irrep_index(static_cast<std::size_t>(nbasis), 0);

            for (int block = 0; block < n_blocks; ++block)
            {
                const int offset = calculator._sao_block_offsets[block];
                const int size = calculator._sao_block_sizes[block];
                if (size == 0)
                    continue;

                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
                    fock_sao.block(offset, offset, size, size));
                if (solver.info() != Eigen::Success)
                    return std::unexpected(std::format(
                        "{} SAO block diagonalization failed (block {})",
                        label,
                        block));

                energies_sao.segment(offset, size) = solver.eigenvalues();
                coefficients_sao.block(offset, offset, size, size) = solver.eigenvectors();
                for (int column = 0; column < size; ++column)
                    mo_irrep_index[static_cast<std::size_t>(offset + column)] =
                        calculator._sao_irrep_index[static_cast<std::size_t>(offset + column)];
            }

            std::vector<int> order(static_cast<std::size_t>(nbasis));
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int left, int right)
                             { return energies_sao[left] < energies_sao[right]; });

            Eigen::VectorXd energies_sorted(nbasis);
            Eigen::MatrixXd coefficients_sao_sorted(nbasis, nbasis);
            std::vector<std::string> mo_symmetry(static_cast<std::size_t>(nbasis));

            for (Eigen::Index column = 0; column < nbasis; ++column)
            {
                const int source = order[static_cast<std::size_t>(column)];
                energies_sorted[column] = energies_sao[source];
                coefficients_sao_sorted.col(column) = coefficients_sao.col(source);
                mo_symmetry[static_cast<std::size_t>(column)] =
                    calculator._sao_irrep_names[static_cast<std::size_t>(mo_irrep_index[static_cast<std::size_t>(source)])];
            }

            return DiagonalizationResult{
                .coefficients = calculator._sao_transform * coefficients_sao_sorted,
                .energies = energies_sorted,
                .mo_symmetry = std::move(mo_symmetry)};
        }

        Eigen::MatrixXd density_from_orbitals(
            const Eigen::Ref<const Eigen::MatrixXd> &coefficients,
            std::size_t n_occ,
            double occupancy)
        {
            const Eigen::MatrixXd occupied = coefficients.leftCols(static_cast<Eigen::Index>(n_occ));
            return occupancy * occupied * occupied.transpose();
        }

        // ResponseExcitationSpace, ResponseEigenpair, transition_density_matrix,
        // evaluate_xc_matrix_from_spin_densities, build_unrestricted_xc_kernel_blocks,
        // and build_closed_shell_xc_kernel_blocks moved to driver.h/below the
        // anonymous namespace (F3.1, docs/DFT_ANALYTIC_FXC_HESSIAN.md) so
        // F3's verification can call the FD-kernel oracle from a standalone
        // test binary. driver.h is included at the top of this file, so their
        // declarations are visible here unchanged.

        std::string linear_response_method_label(HartreeFock::LinearResponseMethod method)
        {
            switch (method)
            {
            case HartreeFock::LinearResponseMethod::TDA:
                return "TDA";
            case HartreeFock::LinearResponseMethod::Casida:
                return "full Casida";
            }

            return "unknown";
        }

        std::string linear_response_spin_label(HartreeFock::LinearResponseSpin spin)
        {
            switch (spin)
            {
            case HartreeFock::LinearResponseSpin::Auto:
                return "auto";
            case HartreeFock::LinearResponseSpin::Singlet:
                return "singlet";
            case HartreeFock::LinearResponseSpin::Triplet:
                return "triplet";
            case HartreeFock::LinearResponseSpin::SpinConserving:
                return "spin-conserving";
            }

            return "unknown";
        }

        constexpr double EV_NM_CONVERSION = 1239.8419843320026;
        constexpr double UVVIS_BROADENING_EV = 0.15;
        constexpr int UVVIS_SPECTRUM_POINTS = 400;

        double energy_ev_to_wavelength_nm(double energy_ev)
        {
            if (energy_ev <= 1.0e-12)
                return 0.0;
            return EV_NM_CONVERSION / energy_ev;
        }

        std::vector<UVVisSpectrumPoint> build_uvvis_spectrum(
            const std::vector<LinearResponseRoot> &roots,
            double sigma_ev = UVVIS_BROADENING_EV,
            int npoints = UVVIS_SPECTRUM_POINTS)
        {
            std::vector<UVVisSpectrumPoint> spectrum;
            if (roots.empty() || sigma_ev <= 0.0 || npoints < 2)
                return spectrum;

            double min_energy = std::numeric_limits<double>::infinity();
            double max_energy = 0.0;
            for (const LinearResponseRoot &root : roots)
            {
                if (root.excitation_energy_ev <= 1.0e-8)
                    continue;
                min_energy = std::min(min_energy, root.excitation_energy_ev);
                max_energy = std::max(max_energy, root.excitation_energy_ev);
            }

            if (!std::isfinite(min_energy) || max_energy <= 0.0)
                return spectrum;

            const double start_ev = std::max(0.05, min_energy - 4.0 * sigma_ev);
            const double end_ev = std::max(start_ev + 8.0 * sigma_ev, max_energy + 4.0 * sigma_ev);
            const double step_ev = (end_ev - start_ev) / static_cast<double>(npoints - 1);

            spectrum.reserve(static_cast<std::size_t>(npoints));
            for (int point = 0; point < npoints; ++point)
            {
                const double energy_ev = start_ev + step_ev * static_cast<double>(point);
                double intensity = 0.0;
                for (const LinearResponseRoot &root : roots)
                {
                    if (root.excitation_energy_ev <= 1.0e-8)
                        continue;
                    const double delta = (energy_ev - root.excitation_energy_ev) / sigma_ev;
                    intensity += root.oscillator_strength * std::exp(-0.5 * delta * delta);
                }

                spectrum.push_back(
                    UVVisSpectrumPoint{
                        .energy_ev = energy_ev,
                        .wavelength_nm = energy_ev_to_wavelength_nm(energy_ev),
                        .intensity = intensity});
            }

            return spectrum;
        }

        std::vector<UVVisSpectrumPoint> extract_uvvis_peaks(
            const std::vector<UVVisSpectrumPoint> &spectrum,
            std::size_t max_peaks = 5,
            double min_separation_ev = UVVIS_BROADENING_EV / 4.0)
        {
            std::vector<UVVisSpectrumPoint> peaks;
            if (spectrum.size() < 3)
                return peaks;

            for (std::size_t i = 1; i + 1 < spectrum.size(); ++i)
            {
                if (spectrum[i].intensity >= spectrum[i - 1].intensity &&
                    spectrum[i].intensity >= spectrum[i + 1].intensity &&
                    spectrum[i].intensity > 1.0e-12)
                {
                    peaks.push_back(spectrum[i]);
                }
            }

            std::ranges::sort(
                peaks,
                [](const UVVisSpectrumPoint &lhs, const UVVisSpectrumPoint &rhs)
                {
                    return lhs.intensity > rhs.intensity;
                });

            std::vector<UVVisSpectrumPoint> unique_peaks;
            unique_peaks.reserve(std::min(max_peaks, peaks.size()));
            for (const UVVisSpectrumPoint &peak : peaks)
            {
                const bool too_close = std::ranges::any_of(
                    unique_peaks,
                    [&](const UVVisSpectrumPoint &accepted)
                    {
                        return std::abs(accepted.energy_ev - peak.energy_ev) < min_separation_ev;
                    });
                if (too_close)
                    continue;

                unique_peaks.push_back(peak);
                if (unique_peaks.size() >= max_peaks)
                    break;
            }
            std::ranges::sort(
                unique_peaks,
                [](const UVVisSpectrumPoint &lhs, const UVVisSpectrumPoint &rhs)
                {
                    return lhs.energy_ev < rhs.energy_ev;
                });
            return unique_peaks;
        }

        std::expected<std::filesystem::path, std::string> write_uvvis_spectrum_file(
            const HartreeFock::Calculator &calculator,
            const std::vector<UVVisSpectrumPoint> &spectrum,
            double sigma_ev = UVVIS_BROADENING_EV)
        {
            if (spectrum.empty())
                return std::unexpected("no UV-Vis spectrum points were generated");

            std::filesystem::path path = calculator._checkpoint_path;
            path.replace_extension(".uvvis.dat");
            std::ofstream out(path);
            if (!out)
                return std::unexpected("failed to open UV-Vis spectrum output file");

            out << "# Gaussian-broadened UV-Vis spectrum\n";
            out << std::format("# sigma_eV = {:.6f}\n", sigma_ev);
            out << "# Energy_eV Wavelength_nm Intensity_arb\n";
            out << std::fixed << std::setprecision(8);
            for (const UVVisSpectrumPoint &point : spectrum)
            {
                out << std::setw(14) << point.energy_ev
                    << std::setw(16) << point.wavelength_nm
                    << std::setw(18) << point.intensity
                    << "\n";
            }

            return path;
        }

        void print_uvvis_spectrum_report(
            const HartreeFock::Calculator &calculator,
            const std::vector<LinearResponseRoot> &roots)
        {
            const std::vector<UVVisSpectrumPoint> spectrum = build_uvvis_spectrum(roots);
            if (spectrum.empty())
                return;

            auto spectrum_path = write_uvvis_spectrum_file(calculator, spectrum);
            if (spectrum_path)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "UV-Vis Spectrum :",
                    std::format(
                        "Wrote {} Gaussian-broadened points to {} (sigma = {:.3f} eV)",
                        spectrum.size(),
                        spectrum_path->string(),
                        UVVIS_BROADENING_EV));
            }
            else
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "UV-Vis Spectrum :",
                    "Spectrum file write failed: " + spectrum_path.error());
            }

            const std::vector<UVVisSpectrumPoint> peaks = extract_uvvis_peaks(spectrum);
            if (peaks.empty())
                return;

            std::cout << std::string(66, '-') << "\n"
                      << std::setw(14) << "Peak (eV)"
                      << std::setw(16) << "Lambda (nm)"
                      << std::setw(18) << "Intensity (arb)"
                      << "\n"
                      << std::string(66, '-') << "\n";
            for (const UVVisSpectrumPoint &peak : peaks)
            {
                std::cout << std::setw(14) << std::fixed << std::setprecision(6) << peak.energy_ev
                          << std::setw(16) << std::fixed << std::setprecision(3) << peak.wavelength_nm
                          << std::setw(18) << std::fixed << std::setprecision(6) << peak.intensity
                          << "\n";
            }
            std::cout << std::string(66, '-') << "\n";
            HartreeFock::Logger::blank();
        }

        std::expected<std::vector<ResponseEigenpair>, std::string> solve_response_problem(
            const Eigen::Ref<const Eigen::MatrixXd> &A,
            const Eigen::Ref<const Eigen::MatrixXd> &B,
            HartreeFock::LinearResponseMethod method,
            int nroots)
        {
            if (A.rows() != A.cols() || B.rows() != B.cols() || A.rows() != B.rows())
                return std::unexpected("TDDFT response matrices must be square and dimension-matched");

            const int dimension = static_cast<int>(A.rows());
            const int roots_to_keep = std::min(std::max(nroots, 1), dimension);
            std::vector<ResponseEigenpair> roots;
            roots.reserve(static_cast<std::size_t>(roots_to_keep));

            if (method == HartreeFock::LinearResponseMethod::TDA)
            {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(0.5 * (A + A.transpose()));
                if (solver.info() != Eigen::Success)
                    return std::unexpected("TDA diagonalization failed");

                for (int root = 0; root < roots_to_keep; ++root)
                {
                    roots.push_back(
                        ResponseEigenpair{
                            .omega = solver.eigenvalues()(root),
                            .x = solver.eigenvectors().col(root),
                            .y = Eigen::VectorXd::Zero(dimension)});
                }
                return roots;
            }

            const Eigen::MatrixXd S = 0.5 * ((A - B) + (A - B).transpose());
            const Eigen::MatrixXd T = 0.5 * ((A + B) + (A + B).transpose());

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> s_solver(S);
            if (s_solver.info() != Eigen::Success)
                return std::unexpected("Casida metric diagonalization failed");

            const Eigen::VectorXd s_evals = s_solver.eigenvalues();
            if (s_evals.minCoeff() <= 1.0e-10)
            {
                return std::unexpected(std::format(
                    "Casida metric A-B is not positive definite (min eigenvalue = {:.3e})",
                    s_evals.minCoeff()));
            }

            const Eigen::MatrixXd s_vectors = s_solver.eigenvectors();
            const Eigen::VectorXd s_sqrt_diag = s_evals.array().sqrt();
            const Eigen::VectorXd s_inv_sqrt_diag = s_sqrt_diag.array().inverse();
            const Eigen::MatrixXd s_sqrt =
                s_vectors * s_sqrt_diag.asDiagonal() * s_vectors.transpose();
            const Eigen::MatrixXd s_inv_sqrt =
                s_vectors * s_inv_sqrt_diag.asDiagonal() * s_vectors.transpose();

            Eigen::MatrixXd casida = s_sqrt * T * s_sqrt;
            casida = 0.5 * (casida + casida.transpose());

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(casida);
            if (solver.info() != Eigen::Success)
                return std::unexpected("Full Casida diagonalization failed");

            int accepted = 0;
            for (int root = 0; root < dimension && accepted < roots_to_keep; ++root)
            {
                const double omega_sq = solver.eigenvalues()(root);
                if (omega_sq <= 1.0e-12)
                    continue;

                const double omega = std::sqrt(omega_sq);
                const Eigen::VectorXd f = solver.eigenvectors().col(root);
                Eigen::VectorXd x_plus_y = (s_sqrt * f) / std::sqrt(omega);
                Eigen::VectorXd x_minus_y = (s_inv_sqrt * f) * std::sqrt(omega);
                Eigen::VectorXd x = 0.5 * (x_plus_y + x_minus_y);
                Eigen::VectorXd y = 0.5 * (x_plus_y - x_minus_y);

                const double norm = x.squaredNorm() - y.squaredNorm();
                if (std::abs(norm) > 1.0e-12)
                {
                    const double scale = 1.0 / std::sqrt(std::abs(norm));
                    x *= scale;
                    y *= scale;
                }

                roots.push_back(ResponseEigenpair{.omega = omega, .x = std::move(x), .y = std::move(y)});
                ++accepted;
            }

            if (roots.empty())
                return std::unexpected("Full Casida solve did not yield any positive excitation energies");

            return roots;
        }

        void print_linear_response_report(
            const std::vector<LinearResponseRoot> &roots,
            const std::string &method_label,
            const std::string &spin_label,
            bool includes_semilocal_xc,
            double exact_exchange_coefficient)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "TDDFT / Linear Response :",
                std::format(
                    "{} {} solver with {:.6f} exact-exchange kernel coefficient",
                    method_label,
                    spin_label,
                    exact_exchange_coefficient));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "TDDFT / Linear Response :",
                includes_semilocal_xc
                    ? "Semilocal XC response kernels are included"
                    : "Semilocal XC response kernels are not included");
            HartreeFock::Logger::blank();

            std::cout << std::string(132, '-') << "\n"
                      << std::setw(6) << "Root"
                      << std::setw(16) << "Omega (Eh)"
                      << std::setw(14) << "Omega (eV)"
                      << std::setw(14) << "Lambda (nm)"
                      << std::setw(14) << "f"
                      << std::setw(16) << "mu_x (au)"
                      << std::setw(16) << "mu_y (au)"
                      << std::setw(16) << "mu_z (au)"
                      << std::setw(20) << "|mu| (Debye)"
                      << "\n"
                      << std::string(132, '-') << "\n";

            for (const LinearResponseRoot &root : roots)
            {
                const double mu_norm_debye = root.transition_dipole.norm() * AU_TO_DEBYE;
                std::cout << std::setw(6) << root.root
                          << std::setw(16) << std::fixed << std::setprecision(8) << root.excitation_energy
                          << std::setw(14) << std::fixed << std::setprecision(7) << root.excitation_energy_ev
                          << std::setw(14) << std::fixed << std::setprecision(3) << root.wavelength_nm
                          << std::setw(14) << std::fixed << std::setprecision(6) << root.oscillator_strength
                          << std::setw(16) << std::fixed << std::setprecision(6) << root.transition_dipole.x()
                          << std::setw(16) << std::fixed << std::setprecision(6) << root.transition_dipole.y()
                          << std::setw(16) << std::fixed << std::setprecision(6) << root.transition_dipole.z()
                          << std::setw(20) << std::fixed << std::setprecision(6) << mu_norm_debye
                          << "\n";
            }

            std::cout << std::string(132, '-') << "\n";

            for (const LinearResponseRoot &root : roots)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "",
                    std::format("  Root {:2d} dominant configurations:", root.root));

                for (const LinearResponseContribution &contribution : root.dominant_contributions)
                {
                    const std::string occupied_label =
                        contribution.occupied_symmetry.empty()
                            ? std::format("{} MO{}", contribution.spin_label, contribution.occupied)
                            : std::format("{} MO{} ({})", contribution.spin_label, contribution.occupied, contribution.occupied_symmetry);
                    const std::string virtual_label =
                        contribution.virtual_symmetry.empty()
                            ? std::format("{} MO{}", contribution.spin_label, contribution.virtual_orbital)
                            : std::format("{} MO{} ({})", contribution.spin_label, contribution.virtual_orbital, contribution.virtual_symmetry);

                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info,
                        "",
                        std::format("    {:<18} -> {:<18} weight = {:.6f}",
                                    occupied_label,
                                    virtual_label,
                                    contribution.weight));
                }
            }
            HartreeFock::Logger::blank();
        }

        std::expected<std::vector<LinearResponseRoot>, std::string> run_linear_response(
            HartreeFock::Calculator &calculator,
            const Options &options,
            const DFT::XC::Functional &exchange_functional,
            const DFT::XC::Functional &correlation_functional)
        {
            if (!calculator._info._is_converged)
                return std::unexpected("TDDFT / linear response requires a converged KS reference");

            PreparedSystem prepared;
            auto preset = grid_preset(to_grid_level(calculator._dft._grid));
            if (!preset)
                return std::unexpected("TDDFT grid preset resolution failed: " + preset.error());
            prepared.grid_preset = *preset;
            prepared.shell_pairs = build_shellpairs(calculator._shells);

            auto molecular_grid = MakeMolecularGrid(
                calculator._molecule,
                to_grid_level(calculator._dft._grid));
            if (!molecular_grid)
                return std::unexpected("TDDFT molecular grid construction failed: " + molecular_grid.error());
            prepared.molecular_grid = std::move(*molecular_grid);

            auto ao_grid = evaluate_ao_basis_on_grid(calculator._shells, prepared.molecular_grid);
            if (!ao_grid)
                return std::unexpected("TDDFT AO grid evaluation failed: " + ao_grid.error());
            prepared.ao_grid = std::move(*ao_grid);

            const bool unrestricted = calculator._scf._scf == HartreeFock::SCFType::UHF;
            if (!unrestricted && calculator._scf._scf != HartreeFock::SCFType::RHF)
                return std::unexpected("TDDFT / linear response currently supports RKS and UKS references only");

            HartreeFock::LinearResponseSpin spin_mode = calculator._dft._lr_spin;
            if (spin_mode == HartreeFock::LinearResponseSpin::Auto)
                spin_mode = unrestricted ? HartreeFock::LinearResponseSpin::SpinConserving
                                         : HartreeFock::LinearResponseSpin::Singlet;

            if (unrestricted &&
                (spin_mode == HartreeFock::LinearResponseSpin::Singlet ||
                 spin_mode == HartreeFock::LinearResponseSpin::Triplet))
            {
                return std::unexpected(
                    "UKS linear response currently supports spin-conserving roots only; singlet/triplet spin adaptation is restricted to closed-shell RKS");
            }

            const Eigen::Index nbasis = static_cast<Eigen::Index>(calculator._shells.nbasis());
            std::vector<ResponseExcitationSpace> spaces;
            spaces.reserve(unrestricted ? 2 : 1);

            if (!unrestricted)
            {
                const int n_electrons = static_cast<int>(
                    calculator._molecule.atomic_numbers.cast<int>().sum() - calculator._molecule.charge);
                if (n_electrons % 2 != 0)
                    return std::unexpected("Closed-shell RKS TDDFT / linear response requires an even number of electrons");

                const int n_occ = n_electrons / 2;
                const int n_virt = static_cast<int>(nbasis) - n_occ;
                if (n_occ <= 0 || n_virt <= 0)
                    return std::unexpected("TDDFT / linear response requires at least one occupied and one virtual orbital");

                const Eigen::MatrixXd &coefficients = calculator._info._scf.alpha.mo_coefficients;
                const Eigen::VectorXd &energies = calculator._info._scf.alpha.mo_energies;
                spaces.push_back(
                    ResponseExcitationSpace{
                        .spin_label = (spin_mode == HartreeFock::LinearResponseSpin::Triplet) ? "T" : "S",
                        .n_occ = n_occ,
                        .n_virt = n_virt,
                        .mo_offset = 0,
                        .C_occ = coefficients.leftCols(n_occ),
                        .C_virt = coefficients.middleCols(n_occ, n_virt),
                        .occ_energies = energies.head(n_occ),
                        .virt_energies = energies.tail(n_virt),
                        .mo_symmetry = calculator._info._scf.alpha.mo_symmetry});
            }
            else
            {
                const int n_electrons = static_cast<int>(
                    calculator._molecule.atomic_numbers.cast<int>().sum() - calculator._molecule.charge);
                const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
                const int n_alpha = (n_electrons + n_unpaired) / 2;
                const int n_beta = (n_electrons - n_unpaired) / 2;
                const int n_virt_alpha = static_cast<int>(nbasis) - n_alpha;
                const int n_virt_beta = static_cast<int>(nbasis) - n_beta;
                if (n_alpha <= 0 || n_beta < 0 || n_virt_alpha <= 0 || n_virt_beta <= 0)
                    return std::unexpected("UKS TDDFT / linear response requires occupied and virtual alpha/beta spaces");

                const Eigen::MatrixXd &coeff_alpha = calculator._info._scf.alpha.mo_coefficients;
                const Eigen::VectorXd &eps_alpha = calculator._info._scf.alpha.mo_energies;
                const Eigen::MatrixXd &coeff_beta = calculator._info._scf.beta.mo_coefficients;
                const Eigen::VectorXd &eps_beta = calculator._info._scf.beta.mo_energies;

                spaces.push_back(
                    ResponseExcitationSpace{
                        .spin_label = "alpha",
                        .n_occ = n_alpha,
                        .n_virt = n_virt_alpha,
                        .mo_offset = 0,
                        .C_occ = coeff_alpha.leftCols(n_alpha),
                        .C_virt = coeff_alpha.middleCols(n_alpha, n_virt_alpha),
                        .occ_energies = eps_alpha.head(n_alpha),
                        .virt_energies = eps_alpha.tail(n_virt_alpha),
                        .mo_symmetry = calculator._info._scf.alpha.mo_symmetry});
                spaces.push_back(
                    ResponseExcitationSpace{
                        .spin_label = "beta",
                        .n_occ = n_beta,
                        .n_virt = n_virt_beta,
                        .mo_offset = 0,
                        .C_occ = coeff_beta.leftCols(n_beta),
                        .C_virt = coeff_beta.middleCols(n_beta, n_virt_beta),
                        .occ_energies = eps_beta.head(n_beta),
                        .virt_energies = eps_beta.tail(n_virt_beta),
                        .mo_symmetry = calculator._info._scf.beta.mo_symmetry});
            }

            const int total_dimension = std::accumulate(
                spaces.begin(),
                spaces.end(),
                0,
                [](int acc, const ResponseExcitationSpace &space)
                { return acc + space.nov(); });
            if (total_dimension <= 0)
                return std::unexpected("TDDFT / linear response built an empty excitation space");

            int offset = 0;
            for (ResponseExcitationSpace &space : spaces)
            {
                space.mo_offset = offset;
                offset += space.nov();
            }

            const int requested_nroots = std::max(calculator._dft._lr_nstates, 1);
            const int requested_root = calculator._dft._lr_root;
            if (requested_root > 0 && requested_root > total_dimension)
            {
                return std::unexpected(std::format(
                    "Requested TDDFT root {} exceeds the excitation-space dimension {}",
                    requested_root,
                    total_dimension));
            }

            const int nroots = std::min(
                std::max(requested_nroots, std::max(requested_root, 1)),
                total_dimension);
            const auto shell_pairs = build_shellpairs(calculator._shells);
            std::vector<double> eri_local;
            const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
                calculator,
                shell_pairs,
                eri_local,
                "TDDFT / Linear Response :");

            const double exact_exchange_coefficient =
                exchange_functional.is_hybrid() ? exchange_functional.exact_exchange_coefficient() : 0.0;

            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(total_dimension, total_dimension);
            Eigen::MatrixXd B = Eigen::MatrixXd::Zero(total_dimension, total_dimension);

            const auto fill_diagonal_gap = [&](const ResponseExcitationSpace &space)
            {
                for (int i = 0; i < space.n_occ; ++i)
                    for (int a = 0; a < space.n_virt; ++a)
                        A(space.mo_offset + space.flat_index(i, a), space.mo_offset + space.flat_index(i, a)) =
                            space.virt_energies(a) - space.occ_energies(i);
            };

            if (!unrestricted)
            {
                const ResponseExcitationSpace &space = spaces.front();
                fill_diagonal_gap(space);

                const std::vector<double> j_a = HartreeFock::Correlation::transform_eri(
                    eri,
                    static_cast<std::size_t>(nbasis),
                    space.C_occ,
                    space.C_virt,
                    space.C_occ,
                    space.C_virt);
                const std::vector<double> j_b = HartreeFock::Correlation::transform_eri(
                    eri,
                    static_cast<std::size_t>(nbasis),
                    space.C_occ,
                    space.C_virt,
                    space.C_virt,
                    space.C_occ);
                const std::vector<double> k_a = HartreeFock::Correlation::transform_eri(
                    eri,
                    static_cast<std::size_t>(nbasis),
                    space.C_occ,
                    space.C_occ,
                    space.C_virt,
                    space.C_virt);
                const std::vector<double> k_b = HartreeFock::Correlation::transform_eri(
                    eri,
                    static_cast<std::size_t>(nbasis),
                    space.C_occ,
                    space.C_virt,
                    space.C_virt,
                    space.C_occ);

                auto response_exchange = DFT::XC::Functional::create(
                    calculator._dft._exchange_id,
                    DFT::XC::Spin::Polarized);
                if (!response_exchange)
                    return std::unexpected("TDDFT response exchange-functional initialization failed: " + response_exchange.error());

                auto response_correlation = DFT::XC::Functional::create(
                    calculator._dft._correlation_id,
                    DFT::XC::Spin::Polarized);
                if (!response_correlation)
                    return std::unexpected("TDDFT response correlation-functional initialization failed: " + response_correlation.error());

                auto kxc_blocks = build_closed_shell_xc_kernel_blocks(
                    prepared,
                    space,
                    calculator._info._scf.alpha.density,
                    *response_exchange,
                    *response_correlation);
                if (!kxc_blocks)
                    return std::unexpected(kxc_blocks.error());
                auto [kxc_same, kxc_cross] = *kxc_blocks;

                const Eigen::MatrixXd kxc =
                    (spin_mode == HartreeFock::LinearResponseSpin::Triplet)
                        ? (kxc_same - kxc_cross).eval()
                        : (kxc_same + kxc_cross).eval();
                const double coulomb_factor =
                    (spin_mode == HartreeFock::LinearResponseSpin::Triplet) ? 0.0 : 2.0;

                auto idx_j = [&](int i, int a, int j, int b) -> std::size_t
                {
                    return ((static_cast<std::size_t>(i) * space.n_virt + a) * space.n_occ + j) * space.n_virt + b;
                };
                // j_b / k_b are transform_eri(...,C_virt,C_occ) so they have shape
                // (nocc, nv, nv, nocc); index (i,a,b,j) uses nv as the third stride
                // and nocc as the fourth.
                auto idx_j_b = [&](int i, int a, int b, int j) -> std::size_t
                {
                    return ((static_cast<std::size_t>(i) * space.n_virt + a) * space.n_virt + b) * space.n_occ + j;
                };
                auto idx_k = [&](int i, int j, int a, int b) -> std::size_t
                {
                    return ((static_cast<std::size_t>(i) * space.n_occ + j) * space.n_virt + a) * space.n_virt + b;
                };

                for (int i = 0; i < space.n_occ; ++i)
                    for (int a = 0; a < space.n_virt; ++a)
                    {
                        const int ia = space.mo_offset + space.flat_index(i, a);
                        for (int j = 0; j < space.n_occ; ++j)
                            for (int b = 0; b < space.n_virt; ++b)
                            {
                                const int jb = space.mo_offset + space.flat_index(j, b);
                                A(ia, jb) += coulomb_factor * j_a[idx_j(i, a, j, b)];
                                B(ia, jb) += coulomb_factor * j_b[idx_j_b(i, a, b, j)];
                                A(ia, jb) -= exact_exchange_coefficient * k_a[idx_k(i, j, a, b)];
                                B(ia, jb) -= exact_exchange_coefficient * k_b[idx_j_b(i, a, b, j)];
                                A(ia, jb) += kxc(space.flat_index(i, a), space.flat_index(j, b));
                                B(ia, jb) += kxc(space.flat_index(i, a), space.flat_index(j, b));
                            }
                    }
            }
            else
            {
                for (const ResponseExcitationSpace &space : spaces)
                    fill_diagonal_gap(space);

                auto kxc_blocks = build_unrestricted_xc_kernel_blocks(
                    prepared,
                    spaces,
                    calculator._info._scf.alpha.density,
                    calculator._info._scf.beta.density,
                    exchange_functional,
                    correlation_functional);
                if (!kxc_blocks)
                    return std::unexpected(kxc_blocks.error());

                for (int target = 0; target < static_cast<int>(spaces.size()); ++target)
                {
                    const ResponseExcitationSpace &target_space = spaces[static_cast<std::size_t>(target)];
                    for (int source = 0; source < static_cast<int>(spaces.size()); ++source)
                    {
                        const ResponseExcitationSpace &source_space = spaces[static_cast<std::size_t>(source)];
                        const std::vector<double> j_a = HartreeFock::Correlation::transform_eri(
                            eri,
                            static_cast<std::size_t>(nbasis),
                            target_space.C_occ,
                            target_space.C_virt,
                            source_space.C_occ,
                            source_space.C_virt);
                        const std::vector<double> j_b = HartreeFock::Correlation::transform_eri(
                            eri,
                            static_cast<std::size_t>(nbasis),
                            target_space.C_occ,
                            target_space.C_virt,
                            source_space.C_virt,
                            source_space.C_occ);

                        std::vector<double> k_a;
                        std::vector<double> k_b;
                        if (target == source)
                        {
                            k_a = HartreeFock::Correlation::transform_eri(
                                eri,
                                static_cast<std::size_t>(nbasis),
                                target_space.C_occ,
                                target_space.C_occ,
                                target_space.C_virt,
                                target_space.C_virt);
                            k_b = HartreeFock::Correlation::transform_eri(
                                eri,
                                static_cast<std::size_t>(nbasis),
                                target_space.C_occ,
                                target_space.C_virt,
                                target_space.C_virt,
                                target_space.C_occ);
                        }

                        auto idx_j = [&](int i, int a, int j, int b) -> std::size_t
                        {
                            return ((static_cast<std::size_t>(i) * target_space.n_virt + a) * source_space.n_occ + j) * source_space.n_virt + b;
                        };
                        // j_b is transform_eri(C_occ_t, C_virt_t, C_virt_s, C_occ_s)
                        // so its shape is (nocc_t, nv_t, nv_s, nocc_s); index (i,a,b,j)
                        // uses nv_s as the third stride and nocc_s as the fourth.
                        auto idx_j_b = [&](int i, int a, int b, int j) -> std::size_t
                        {
                            return ((static_cast<std::size_t>(i) * target_space.n_virt + a) * source_space.n_virt + b) * source_space.n_occ + j;
                        };
                        auto idx_k = [&](int i, int j, int a, int b) -> std::size_t
                        {
                            return ((static_cast<std::size_t>(i) * source_space.n_occ + j) * target_space.n_virt + a) * source_space.n_virt + b;
                        };

                        for (int i = 0; i < target_space.n_occ; ++i)
                            for (int a = 0; a < target_space.n_virt; ++a)
                            {
                                const int ia = target_space.mo_offset + target_space.flat_index(i, a);
                                for (int j = 0; j < source_space.n_occ; ++j)
                                    for (int b = 0; b < source_space.n_virt; ++b)
                                    {
                                        const int jb = source_space.mo_offset + source_space.flat_index(j, b);
                                        A(ia, jb) += j_a[idx_j(i, a, j, b)];
                                        B(ia, jb) += j_b[idx_j_b(i, a, b, j)];
                                        if (target == source)
                                        {
                                            A(ia, jb) -= exact_exchange_coefficient * k_a[idx_k(i, j, a, b)];
                                            B(ia, jb) -= exact_exchange_coefficient * k_b[idx_j_b(i, a, b, j)];
                                        }
                                        A(ia, jb) += (*kxc_blocks)[static_cast<std::size_t>(target)][static_cast<std::size_t>(source)](
                                            target_space.flat_index(i, a),
                                            source_space.flat_index(j, b));
                                        B(ia, jb) += (*kxc_blocks)[static_cast<std::size_t>(target)][static_cast<std::size_t>(source)](
                                            target_space.flat_index(i, a),
                                            source_space.flat_index(j, b));
                                    }
                            }
                    }
                }
            }

            A = 0.5 * (A + A.transpose());
            B = 0.5 * (B + B.transpose());

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "TDDFT / Dense Solve :",
                std::format(
                    "Building {} {} response with {} solved roots in a {}-dimensional excitation space",
                    linear_response_method_label(calculator._dft._lr_method),
                    linear_response_spin_label(spin_mode),
                    nroots,
                    total_dimension));

            auto eigenpairs = solve_response_problem(A, B, calculator._dft._lr_method, nroots);
            if (!eigenpairs)
                return std::unexpected(eigenpairs.error());

            const HartreeFock::MultipoleMatrices multipoles =
                HartreeFock::ObaraSaika::_compute_multipole_matrices(
                    shell_pairs,
                    static_cast<std::size_t>(nbasis),
                    Eigen::Vector3d::Zero());

            std::vector<std::array<Eigen::MatrixXd, 3>> mo_dipole_ov;
            mo_dipole_ov.reserve(spaces.size());
            for (const ResponseExcitationSpace &space : spaces)
            {
                mo_dipole_ov.push_back(
                    {space.C_occ.transpose() * multipoles.dipole[0] * space.C_virt,
                     space.C_occ.transpose() * multipoles.dipole[1] * space.C_virt,
                     space.C_occ.transpose() * multipoles.dipole[2] * space.C_virt});
            }

            std::vector<LinearResponseRoot> roots;
            roots.reserve(eigenpairs->size());

            for (int root_index = 0; root_index < static_cast<int>(eigenpairs->size()); ++root_index)
            {
                const ResponseEigenpair &pair = (*eigenpairs)[static_cast<std::size_t>(root_index)];
                const Eigen::VectorXd transition_amplitudes = pair.x + pair.y;
                Eigen::Vector3d transition_dipole = Eigen::Vector3d::Zero();
                std::vector<LinearResponseContribution> contributions;
                contributions.reserve(static_cast<std::size_t>(total_dimension));

                for (std::size_t space_index = 0; space_index < spaces.size(); ++space_index)
                {
                    const ResponseExcitationSpace &space = spaces[space_index];
                    const auto &dipoles = mo_dipole_ov[space_index];
                    for (int i = 0; i < space.n_occ; ++i)
                        for (int a = 0; a < space.n_virt; ++a)
                        {
                            const int flat = space.mo_offset + space.flat_index(i, a);
                            const double amplitude = transition_amplitudes(flat);
                            if (!unrestricted)
                            {
                                if (spin_mode == HartreeFock::LinearResponseSpin::Triplet)
                                {
                                    // Electric dipole transitions vanish in the spin-adapted triplet block.
                                }
                                else
                                {
                                    transition_dipole.x() += std::sqrt(2.0) * amplitude * dipoles[0](i, a);
                                    transition_dipole.y() += std::sqrt(2.0) * amplitude * dipoles[1](i, a);
                                    transition_dipole.z() += std::sqrt(2.0) * amplitude * dipoles[2](i, a);
                                }
                            }
                            else
                            {
                                transition_dipole.x() += amplitude * dipoles[0](i, a);
                                transition_dipole.y() += amplitude * dipoles[1](i, a);
                                transition_dipole.z() += amplitude * dipoles[2](i, a);
                            }

                            const double contribution_weight = pair.x(flat) * pair.x(flat) + pair.y(flat) * pair.y(flat);
                            contributions.push_back(
                                LinearResponseContribution{
                                    .occupied = i + 1,
                                    .virtual_orbital = space.n_occ + a + 1,
                                    .weight = contribution_weight,
                                    .spin_label = space.spin_label,
                                    .occupied_symmetry =
                                        (static_cast<std::size_t>(i) < space.mo_symmetry.size()) ? space.mo_symmetry[static_cast<std::size_t>(i)] : "",
                                    .virtual_symmetry =
                                        (static_cast<std::size_t>(space.n_occ + a) < space.mo_symmetry.size()) ? space.mo_symmetry[static_cast<std::size_t>(space.n_occ + a)] : ""});
                        }
                }

                const double total_weight = std::accumulate(
                    contributions.begin(),
                    contributions.end(),
                    0.0,
                    [](double acc, const LinearResponseContribution &contribution)
                    { return acc + contribution.weight; });
                if (total_weight > 1.0e-12)
                {
                    for (LinearResponseContribution &contribution : contributions)
                        contribution.weight /= total_weight;
                }

                std::ranges::sort(
                    contributions,
                    [](const LinearResponseContribution &lhs, const LinearResponseContribution &rhs)
                    {
                        return lhs.weight > rhs.weight;
                    });
                if (contributions.size() > 3)
                    contributions.resize(3);

                const double oscillator_strength =
                    (spin_mode == HartreeFock::LinearResponseSpin::Triplet)
                        ? 0.0
                        : std::max(0.0, (2.0 / 3.0) * pair.omega * transition_dipole.squaredNorm());

                roots.push_back(
                    LinearResponseRoot{
                        .root = root_index + 1,
                        .excitation_energy = pair.omega,
                        .excitation_energy_ev = pair.omega * HARTREE_TO_EV,
                        .wavelength_nm = energy_ev_to_wavelength_nm(pair.omega * HARTREE_TO_EV),
                        .transition_dipole = transition_dipole,
                        .oscillator_strength = oscillator_strength,
                        .dominant_contributions = std::move(contributions)});
            }

            std::vector<LinearResponseRoot> reported_roots = roots;
            if (requested_root > 0)
            {
                if (requested_root > static_cast<int>(roots.size()))
                {
                    return std::unexpected(std::format(
                        "Requested TDDFT root {} was not found among the {} solved roots",
                        requested_root,
                        roots.size()));
                }

                reported_roots = {roots[static_cast<std::size_t>(requested_root - 1)]};
            }

            print_linear_response_report(
                reported_roots,
                linear_response_method_label(calculator._dft._lr_method),
                linear_response_spin_label(spin_mode),
                true,
                exact_exchange_coefficient);
            print_uvvis_spectrum_report(calculator, roots);
            return reported_roots;
        }

        std::expected<Result, std::string> run_ks_scf_scaffold(
            HartreeFock::Calculator &calculator,
            PreparedSystem &prepared,
            const DFT::XC::Functional &x_functional,
            const DFT::XC::Functional &c_functional)
        {
            // This routine is the shared KS-SCF engine used for the initial
            // single-point, displaced geometries in numerical derivatives, and
            // repeated geometry-optimization/frequency evaluations.
            if (prepared.ao_grid.npoints() != prepared.molecular_grid.points.rows())
                return std::unexpected("DFT KS-SCF scaffold reached with inconsistent AO/grid dimensions");

            if (prepared.ao_grid.nbasis() != static_cast<Eigen::Index>(calculator._shells.nbasis()))
                return std::unexpected("DFT KS-SCF scaffold reached with inconsistent AO/basis dimensions");

            const std::size_t nbasis = calculator._shells.nbasis();
            const bool unrestricted = calculator._scf._scf == HartreeFock::SCFType::UHF;
            const int n_electrons = static_cast<int>(
                calculator._molecule.atomic_numbers.cast<int>().sum() - calculator._molecule.charge);

            if (!unrestricted && (n_electrons % 2 != 0))
                return std::unexpected("RKS requires an even number of electrons; use UKS for open-shell systems");

            const auto orthogonalizer = HartreeFock::SCF::build_orthogonalizer(calculator._overlap);
            if (!orthogonalizer)
                return std::unexpected("DFT orthogonalizer build failed inside KS loop: " + orthogonalizer.error());
            const Eigen::MatrixXd X = *orthogonalizer;

            calculator._info._is_converged = false;

            const unsigned int max_iter = calculator._scf.get_max_cycles(nbasis);
            HartreeFock::Logger::scf_header();

            Result result;

            if (!unrestricted)
            {
                const std::size_t n_occ = static_cast<std::size_t>(n_electrons / 2);
                Eigen::MatrixXd density = calculator._info._scf.alpha.density;
                if (density.rows() != static_cast<Eigen::Index>(nbasis) ||
                    density.cols() != static_cast<Eigen::Index>(nbasis))
                    density = HartreeFock::SCF::initial_density(calculator._hcore, X, n_occ);

                HartreeFock::DIISState diis;
                diis.max_vecs = calculator._scf._DIIS_dim;
                const bool use_diis = calculator._scf._use_DIIS;
                double previous_total_energy = 0.0;

                // SOSCF (D2.2.3, docs/SOSCF_DFT.md): reference
                // orbitals persisted across iterations, mirroring RHF's
                // C_soscf_prev/eps_soscf_prev (src/scf/scf.cpp) exactly --
                // the orbital gradient/Hessian at a SOSCF iteration are
                // evaluated in the PREVIOUS iteration's MO basis against
                // THIS iteration's Fock, which is what is actually
                // stationary at convergence.
                Eigen::MatrixXd C_soscf_prev;
                Eigen::VectorXd eps_soscf_prev;
                unsigned int soscf_window_start = 0;
                // SOSCF_DFT.md invariant 3 (RKS hybrids): RKS hybrids -- global (B3LYP,
                // PBE0) and range-separated (HSE06, CAM-B3LYP) -- are now
                // supported. The exact-exchange (K) response is one more
                // linear-in-density term in h_op (c_fr*Coulomb + c_sr*ShortRange,
                // scaled -0.5 like the KS build); verified to ratio 1.000000 by
                // the scale probe (LDA + combined-XC hard assertion). The only
                // remaining RKS scope cuts are PCM and SAO, handled below.
                // D2.2.4: a user requesting SOSCF (either trigger keyword)
                // must be told when the request cannot be honored, rather
                // than silently running plain DIIS the whole time -- the
                // exact failure mode this step exists to close. Emitted
                // once, before the loop, so it is not spammy per-iteration.
                if ((calculator._scf._scf_soscf_diis_tol > 0.0 ||
                     calculator._scf._scf_soscf_start > 0))
                {
                    std::string reason;
                    if (prepared.pcm)
                        reason = "PCM solvation (not yet wired through DFT SOSCF)";
                    else if (calculator._use_sao_blocking)
                        reason = "SAO/symmetry blocking (not yet wired through DFT SOSCF)";
                    if (!reason.empty())
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Warning,
                            "DFT SOSCF :",
                            std::format(
                                "scf_soscf_start/scf_soscf_diis_tol requested but disabled for this "
                                "run: {} -- running with plain DIIS only",
                                reason));
                }

                for (unsigned int iter = 1; iter <= max_iter; ++iter)
                {
                    const auto iter_start = std::chrono::steady_clock::now();
                    calculator._info._scf.alpha.density = density;

                    // DFT replaces the HF Coulomb/exchange build with a grid
                    // loop that evaluates the current density, queries libxc
                    // for the semilocal derivatives, and assembles the AO-space
                    // KS potential for the present density matrix.
                    // MPI: this rank's grid-point slice, shared by the density/XC
                    // eval and the XC matrix assembly. Serial => whole grid.
                    const auto [xc_lo, xc_hi] =
                        mpi_grid_slice(prepared.molecular_grid.points.rows());

                    auto xc_grid = evaluate_current_density_and_xc(
                        calculator,
                        prepared,
                        x_functional,
                        c_functional,
                        xc_lo, xc_hi);
                    if (!xc_grid)
                        return std::unexpected("DFT density/XC evaluation failed: " + xc_grid.error());

                    // xc_grid's energy/electron scalars are this rank's slice
                    // only; sum them across ranks before they feed the SCF energy
                    // and the electron-count check below.
                    reduce_partial_xc_scalars(*xc_grid);

                    auto ks_potential =
                        assemble_current_ks_potential(calculator, prepared, *xc_grid, xc_lo, xc_hi);
                    if (!ks_potential)
                        return std::unexpected("DFT KS potential assembly failed: " + ks_potential.error());

                    Eigen::MatrixXd pcm_potential = Eigen::MatrixXd::Zero(nbasis, nbasis);
                    double pcm_energy = 0.0;
                    if (prepared.pcm && prepared.pcm->enabled())
                    {
                        auto pcm_result = HartreeFock::Solvation::evaluate_pcm_reaction_field(
                            calculator,
                            *prepared.pcm,
                            density);
                        if (!pcm_result)
                            return std::unexpected("DFT PCM reaction-field build failed: " + pcm_result.error());
                        pcm_potential = std::move(pcm_result->reaction_potential);
                        pcm_energy = pcm_result->solvation_energy;
                    }

                    const Eigen::MatrixXd fock = calculator._hcore + ks_potential->alpha + pcm_potential;
                    const double electronic_energy_gas =
                        (density.array() * calculator._hcore.array()).sum() +
                        0.5 * (density.array() * ks_potential->coulomb.array()).sum() +
                        xc_grid->total_energy +
                        ks_potential->exact_exchange_energy;
                    const double electronic_energy = electronic_energy_gas + pcm_energy;
                    const double total_energy = electronic_energy + calculator._nuclear_repulsion;

                    double diis_error = 0.0;
                    if (use_diis)
                    {
                        // Reuse the standard commutator error in the orthogonal
                        // AO basis so KS and HF share the same convergence
                        // acceleration machinery.
                        const Eigen::MatrixXd error =
                            X.transpose() *
                            (fock * density * calculator._overlap - calculator._overlap * density * fock) *
                            X;
                        diis.push(fock, error);
                        diis_error = diis.error_norm();
                    }

                    // SOSCF (D2.2.3): same criterion-or-fixed-iteration gate
                    // RHF/UHF already use (src/scf/scf.cpp), sharing the
                    // scf_soscf_* keywords. SAO-active and PCM are excluded
                    // exactly like RHF/UHF's own guard (neither is wired
                    // through this Hessian yet); hybrids are excluded per
                    // D2.2's own scope cut, enforced separately at D2.2.4.
                    const bool sao_active_rks = calculator._use_sao_blocking &&
                                                calculator._sao_transform.rows() ==
                                                    static_cast<Eigen::Index>(nbasis) &&
                                                calculator._sao_transform.cols() ==
                                                    static_cast<Eigen::Index>(nbasis) &&
                                                !calculator._sao_block_sizes.empty();
                    const bool soscf_enabled =
                        (calculator._scf._scf_soscf_diis_tol > 0.0 ||
                         calculator._scf._scf_soscf_start > 0) &&
                        !sao_active_rks && !prepared.pcm;
                    if (soscf_enabled && soscf_window_start == 0)
                    {
                        const bool criterion_fires =
                            calculator._scf._scf_soscf_diis_tol > 0.0
                                ? (use_diis && diis_error > 0.0 &&
                                   diis_error < calculator._scf._scf_soscf_diis_tol &&
                                   iter >= calculator._scf._scf_soscf_min_iter)
                                : (iter >= calculator._scf._scf_soscf_start);
                        if (criterion_fires)
                            soscf_window_start = iter;
                    }
                    const bool soscf_active =
                        soscf_enabled && soscf_window_start > 0 &&
                        iter < soscf_window_start + calculator._scf._scf_soscf_cycles &&
                        C_soscf_prev.size() > 0;
                    if (soscf_window_start > 0 &&
                        iter == soscf_window_start + calculator._scf._scf_soscf_cycles)
                    {
                        diis.clear();
                    }
                    const bool do_diis = use_diis && diis.ready() && !soscf_active;
                    const Eigen::MatrixXd fock_for_diagonalization =
                        do_diis ? diis.extrapolate() : fock;

                    Eigen::MatrixXd C_new;
                    Eigen::VectorXd eps_new;
                    if (soscf_active)
                    {
                        // ── SOSCF (D2.2.3) ──────────────────────────────────
                        // Closed-shell RKS. The exact Newton step is
                        // -H_true^-1 g_true with, in the kappa parametrization
                        // (occupancy-2 density, dP/dkappa = [R,P0] = 2*dP):
                        //   g_true      = 4 * F_mo(a,i)
                        //   H_true * x  = 4 * diag(eps_a-eps_i) .* x
                        //               + 8 * (J + V_xc + K)[dP]
                        // Dividing both by 4: g = F_mo (unscaled, as before) and
                        //   h_op(x) = diag .* x + 2 * (J + V_xc + K)[dP].
                        // The old code used a bare (J+V_xc), i.e. the kernel was
                        // half-weighted relative to the diagonal -- the Newton
                        // DIRECTION was slightly wrong (magnitude was fine because
                        // the diagonal dominates). See
                        // docs/SOSCF_DFT.md (invariant 2); LDA-verified
                        // to ratio 1.000000 by the S1 probe.
                        // (UKS is unaffected -- occupancy-1 makes diag and kernel
                        // scale identically there; do NOT add a 2x to UKS.)
                        const int n_occ_i = static_cast<int>(n_occ);
                        const int n_virt_i = static_cast<int>(nbasis) - n_occ_i;
                        const Eigen::MatrixXd C_occ_prev = C_soscf_prev.leftCols(n_occ_i);
                        const Eigen::MatrixXd C_virt_prev = C_soscf_prev.rightCols(n_virt_i);

                        const Eigen::MatrixXd F_mo = C_soscf_prev.transpose() * fock * C_soscf_prev;
                        Eigen::VectorXd g(n_virt_i * n_occ_i);
                        for (int a = 0; a < n_virt_i; ++a)
                            for (int i = 0; i < n_occ_i; ++i)
                                g(a * n_occ_i + i) = F_mo(n_occ_i + a, i);

                        // The KS orbital Hessian h_op is lifted into its own TU
                        // (src/dft/ks_orbital_hessian.{h,cpp}) so the analytic
                        // double-hybrid gradient's Z-vector solve uses the exact
                        // same operator. This branch is a pure caller: same
                        // diag + s*(J + xc + K) with s = 2 for RKS
                        // (docs/SOSCF_DFT.md invariant 2/3).
                        DFT::Driver::KsOrbitalHessianInputs h_in;
                        h_in.shell_pairs = &prepared.shell_pairs;
                        h_in.molecular_grid = &prepared.molecular_grid;
                        h_in.ao_grid = &prepared.ao_grid;
                        h_in.density = density;
                        h_in.C_occ = C_occ_prev;
                        h_in.C_virt = C_virt_prev;
                        h_in.eps = eps_soscf_prev;
                        h_in.x_functional = &x_functional;
                        h_in.c_functional = &c_functional;
                        h_in.engine = calculator._integral._engine;
                        h_in.tol_eri = calculator._integral._tol_eri;
                        h_in.sym_ops = calculator._use_integral_symmetry
                                           ? &calculator._integral_symmetry_ops
                                           : nullptr;
                        h_in.full_range_exchange_coefficient =
                            xc_grid->full_range_exchange_coefficient;
                        h_in.short_range_exchange_coefficient =
                            xc_grid->short_range_exchange_coefficient;
                        h_in.range_separation_omega = xc_grid->range_separation_omega;
                        h_in.kernel_scale = 2.0; // RKS
                        const auto h_op = DFT::Driver::build_ks_orbital_hessian_op(h_in);
                        const auto g_op = [&g]() -> Eigen::VectorXd
                        { return g; };

                        HartreeFock::Correlation::CASSCF::AugHessianOptions ah_opts;
                        ah_opts.ah_start_tol = std::max(1e-8, 0.1 * g.norm());
                        Eigen::VectorXd x0 = -g;
                        const double x0_norm = x0.norm();
                        if (std::isfinite(x0_norm) && x0_norm > 0.0)
                            x0 /= x0_norm;
                        const HartreeFock::Correlation::CASSCF::AugHessianResult ah =
                            HartreeFock::Correlation::CASSCF::solve_augmented_hessian(
                                h_op, g_op, nullptr, x0, ah_opts);

                        // Deadband: once the Newton step is at the density-tolerance
                        // scale, applying it just cycles the density at ~1e-10 (the
                        // AH solve returns a nonzero step for any g != 0, and a
                        // full-convergence SOSCF window has no DIIS to damp it).
                        // Pass the reference orbitals through unchanged so
                        // next_density == density and is_converged can fire. The
                        // DIIS-handoff mode never reaches this -- it hands back
                        // long before. See SOSCF_DFT.md invariant 2.
                        if (ah.x.allFinite() &&
                            ah.x.cwiseAbs().maxCoeff() < calculator._scf._tol_density)
                        {
                            C_new = C_soscf_prev;
                            eps_new = eps_soscf_prev;
                            HartreeFock::Logger::logging(
                                HartreeFock::LogLevel::Info, "DFT SOSCF :",
                                std::format("step at iter {}: |g|={:.3e} step below tol_density "
                                            "-- holding orbitals (deadband)",
                                            iter, g.norm()));
                        }
                        else
                        {

                        // Trust-region cap, same constant RHF/UHF SOSCF use.
                        constexpr double kSoscfMaxRot = 0.20;
                        Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(nbasis, nbasis);
                        bool cap_fired = false;
                        if (ah.x.size() == n_virt_i * n_occ_i && ah.x.allFinite())
                        {
                            Eigen::VectorXd step = ah.x;
                            const double max_elem = step.cwiseAbs().maxCoeff();
                            if (max_elem > kSoscfMaxRot)
                            {
                                step *= kSoscfMaxRot / max_elem;
                                cap_fired = true;
                            }
                            for (int a = 0; a < n_virt_i; ++a)
                                for (int i = 0; i < n_occ_i; ++i)
                                {
                                    const double v = step(a * n_occ_i + i);
                                    kappa(n_occ_i + a, i) = v;
                                    kappa(i, n_occ_i + a) = -v;
                                }
                        }
                        C_new = HartreeFock::Correlation::CASSCF::apply_orbital_rotation(
                            C_soscf_prev, kappa, calculator._overlap);
                        if (!C_new.allFinite())
                            return std::unexpected(std::format(
                                "DFT SOSCF: orbital rotation produced non-finite coefficients at "
                                "iteration {}",
                                iter));

                        // Semicanonicalize occ-occ/virt-virt blocks separately
                        // -- pure gauge freedom, same as RHF/UHF SOSCF.
                        const Eigen::MatrixXd F_mo_new = C_new.transpose() * fock * C_new;
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> occ_solver(
                            F_mo_new.topLeftCorner(n_occ_i, n_occ_i));
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> virt_solver(
                            F_mo_new.bottomRightCorner(n_virt_i, n_virt_i));
                        if (occ_solver.info() != Eigen::Success || virt_solver.info() != Eigen::Success)
                            return std::unexpected(std::format(
                                "DFT SOSCF: semicanonicalization eigensolve failed at iteration {}",
                                iter));

                        Eigen::MatrixXd C_canon(nbasis, nbasis);
                        C_canon.leftCols(n_occ_i) = C_new.leftCols(n_occ_i) * occ_solver.eigenvectors();
                        C_canon.rightCols(n_virt_i) = C_new.rightCols(n_virt_i) * virt_solver.eigenvectors();
                        C_new = C_canon;

                        eps_new.resize(nbasis);
                        eps_new.head(n_occ_i) = occ_solver.eigenvalues();
                        eps_new.tail(n_virt_i) = virt_solver.eigenvalues();

                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Info, "DFT SOSCF :",
                            std::format(
                                "step at iter {}: |g|={:.3e} v0={:.4f} eig={:.4e} converged={} "
                                "ah_iters={} ah_residual={:.3e} cap_fired={}",
                                iter, g.norm(), ah.v0, ah.eigenvalue, ah.converged, ah.iterations,
                                ah.residual_norm, cap_fired));
                        } // end deadband else
                    }
                    else
                    {
                        auto diagonalization = diagonalize_in_ao_basis(
                            calculator,
                            X,
                            fock_for_diagonalization,
                            "KS");
                        if (!diagonalization)
                            return std::unexpected(diagonalization.error());
                        C_new = diagonalization->coefficients;
                        eps_new = diagonalization->energies;
                        calculator._info._scf.alpha.mo_symmetry = diagonalization->mo_symmetry;
                    }

                    const Eigen::MatrixXd next_density = density_from_orbitals(C_new, n_occ, 2.0);
                    const auto metrics = HartreeFock::SCF::restricted_iteration_metrics(
                        density,
                        next_density,
                        previous_total_energy,
                        total_energy);

                    const double iter_time = std::chrono::duration<double>(
                                                 std::chrono::steady_clock::now() - iter_start)
                                                 .count();
                    HartreeFock::Logger::scf_iteration(
                        iter,
                        total_energy,
                        metrics.delta_energy,
                        metrics.delta_density_rms,
                        metrics.delta_density_max,
                        diis_error,
                        0.0,
                        iter_time);

                    density = next_density;
                    previous_total_energy = total_energy;
                    // SOSCF: keep the reference basis current every
                    // iteration (not just while active), so the switch
                    // iteration always has a valid C_soscf_prev the moment
                    // it fires -- same discipline RHF/UHF SOSCF use.
                    C_soscf_prev = C_new;
                    eps_soscf_prev = eps_new;

                    HartreeFock::SCF::store_restricted_iteration(
                        calculator,
                        HartreeFock::SCF::RestrictedIterationData{
                            .density = density,
                            .fock = fock,
                            .mo_energies = eps_new,
                            .mo_coefficients = C_new,
                            .electronic_energy = electronic_energy,
                            .total_energy = total_energy},
                        metrics);

                    result.total_energy = total_energy;
                    result.xc_energy = xc_grid->total_energy + ks_potential->exact_exchange_energy;
                    result.integrated_electrons = xc_grid->integrated_electrons;
                    // PLANCK_DEBUG_GRID_ACC: absolute grid-accuracy probe.
                    // int rho dr must equal the electron count EXACTLY, so the
                    // deviation is a reference-free measure of the quadrature's
                    // own error -- no PySCF, no FD, no fitting.
                    if (std::getenv("PLANCK_DEBUG_GRID_ACC"))
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Info, "Grid Accuracy :",
                            std::format("integrated_electrons = {:.10f}, deviation = {:.3e}",
                                        xc_grid->integrated_electrons,
                                        [&]{ int ne = 0;
                                             for (auto z : calculator._molecule.atomic_numbers) ne += static_cast<int>(z);
                                             ne -= calculator._molecule.charge;
                                             return std::abs(xc_grid->integrated_electrons - static_cast<double>(ne)); }()));
                    result.solvation_energy = pcm_energy;

                    if (HartreeFock::SCF::is_converged(calculator._scf, metrics, iter))
                    {
                        calculator._info._is_converged = true;
                        result.converged = true;
                        HartreeFock::Logger::scf_footer();
                        HartreeFock::Logger::blank();
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Info,
                            "RKS Converged :",
                            std::format("E = {:.10f} Eh after {} iterations", total_energy, iter));
                        HartreeFock::Logger::blank();

                        return result;
                    }
                }

                HartreeFock::Logger::scf_footer();
                return std::unexpected(std::format("RKS did not converge in {} iterations", max_iter));
            }

            const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;
            if (n_unpaired < 0 || n_unpaired > n_electrons)
                return std::unexpected("Invalid multiplicity for UKS");
            if ((n_electrons - n_unpaired) % 2 != 0)
                return std::unexpected("Multiplicity inconsistent with electron count parity for UKS");

            const std::size_t n_alpha = static_cast<std::size_t>((n_electrons + n_unpaired) / 2);
            const std::size_t n_beta = static_cast<std::size_t>((n_electrons - n_unpaired) / 2);

            Eigen::MatrixXd alpha_density = calculator._info._scf.alpha.density;
            Eigen::MatrixXd beta_density = calculator._info._scf.beta.density;
            const Eigen::MatrixXd hcore_prime = X.transpose() * calculator._hcore * X;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> hcore_solver(hcore_prime);
            if (hcore_solver.info() != Eigen::Success)
                return std::unexpected("UKS initial HCore diagonalization failed");
            const Eigen::MatrixXd hcore_coefficients = X * hcore_solver.eigenvectors();
            if (alpha_density.rows() != static_cast<Eigen::Index>(nbasis) ||
                alpha_density.cols() != static_cast<Eigen::Index>(nbasis))
                alpha_density = density_from_orbitals(hcore_coefficients, n_alpha, 1.0);
            if (beta_density.rows() != static_cast<Eigen::Index>(nbasis) ||
                beta_density.cols() != static_cast<Eigen::Index>(nbasis))
                beta_density = density_from_orbitals(hcore_coefficients, n_beta, 1.0);

            HartreeFock::DIISState diis_alpha, diis_beta;
            diis_alpha.max_vecs = diis_beta.max_vecs = calculator._scf._DIIS_dim;
            const bool use_diis = calculator._scf._use_DIIS;
            double previous_total_energy = 0.0;

            // SOSCF (D3.2, docs/SOSCF_DFT.md): the UKS analogue of
            // the RKS branch above, generalized to the coupled alpha/beta
            // step exactly the way U2 generalized S2 for UHF. Per-spin
            // reference orbitals persisted every iteration; the (a,i)
            // gradient/Hessian are evaluated in the PREVIOUS iteration's MO
            // basis against THIS iteration's Fock. Packing convention:
            // build_uhf_cphf_matrix's [0,nova) alpha + [nova,nova+novb)
            // beta, virtual-major (a*n_occ + i) within each block -- NOT a
            // third convention.
            Eigen::MatrixXd Ca_soscf_prev, Cb_soscf_prev;
            Eigen::VectorXd epsa_soscf_prev, epsb_soscf_prev;
            unsigned int soscf_window_start = 0;
            // SOSCF_DFT.md invariant 3 (UKS hybrids): UKS hybrids -- global and
            // range-separated -- are now supported. The polarized h_op gains
            // the spin-resolved K response (-1*(c_fr*K_C + c_sr*K_SR) per
            // spin, from _compute_2e_k_uhf_direct); UKS h_op stays uniform 1x
            // (no RKS-style 2x on the kernel). Only PCM and SAO remain as UKS
            // scope cuts.
            // D3.2.1: same one-time diagnostic D2.2.4 built for RKS -- a
            // user requesting SOSCF (either trigger keyword) is told when
            // the request cannot be honored rather than silently running
            // plain DIIS.
            if ((calculator._scf._scf_soscf_diis_tol > 0.0 ||
                 calculator._scf._scf_soscf_start > 0))
            {
                std::string reason;
                if (prepared.pcm)
                    reason = "PCM solvation (not yet wired through DFT SOSCF)";
                else if (calculator._use_sao_blocking)
                    reason = "SAO/symmetry blocking (not yet wired through DFT SOSCF)";
                if (!reason.empty())
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Warning,
                        "DFT SOSCF :",
                        std::format(
                            "scf_soscf_start/scf_soscf_diis_tol requested but disabled for this "
                            "run: {} -- running with plain DIIS only",
                            reason));
            }

            for (unsigned int iter = 1; iter <= max_iter; ++iter)
            {
                const auto iter_start = std::chrono::steady_clock::now();
                calculator._info._scf.alpha.density = alpha_density;
                calculator._info._scf.beta.density = beta_density;

                const auto [xc_lo, xc_hi] =
                    mpi_grid_slice(prepared.molecular_grid.points.rows());

                auto xc_grid = evaluate_current_density_and_xc(
                    calculator,
                    prepared,
                    x_functional,
                    c_functional,
                    xc_lo, xc_hi);
                if (!xc_grid)
                    return std::unexpected("DFT density/XC evaluation failed: " + xc_grid.error());

                reduce_partial_xc_scalars(*xc_grid);

                auto ks_potential =
                    assemble_current_ks_potential(calculator, prepared, *xc_grid, xc_lo, xc_hi);
                if (!ks_potential)
                    return std::unexpected("DFT KS potential assembly failed: " + ks_potential.error());

                const Eigen::MatrixXd total_density = alpha_density + beta_density;
                Eigen::MatrixXd pcm_potential = Eigen::MatrixXd::Zero(nbasis, nbasis);
                double pcm_energy = 0.0;
                if (prepared.pcm && prepared.pcm->enabled())
                {
                    auto pcm_result = HartreeFock::Solvation::evaluate_pcm_reaction_field(
                        calculator,
                        *prepared.pcm,
                        total_density);
                    if (!pcm_result)
                        return std::unexpected("DFT PCM reaction-field build failed: " + pcm_result.error());
                    pcm_potential = std::move(pcm_result->reaction_potential);
                    pcm_energy = pcm_result->solvation_energy;
                }

                const Eigen::MatrixXd fock_alpha = calculator._hcore + ks_potential->alpha + pcm_potential;
                const Eigen::MatrixXd fock_beta = calculator._hcore + ks_potential->beta + pcm_potential;

                const double electronic_energy_gas =
                    (total_density.array() * calculator._hcore.array()).sum() +
                    0.5 * (total_density.array() * ks_potential->coulomb.array()).sum() +
                    xc_grid->total_energy +
                    ks_potential->exact_exchange_energy;
                const double electronic_energy = electronic_energy_gas + pcm_energy;
                const double total_energy = electronic_energy + calculator._nuclear_repulsion;

                double diis_error = 0.0;
                if (use_diis)
                {
                    const Eigen::MatrixXd error_alpha =
                        X.transpose() *
                        (fock_alpha * alpha_density * calculator._overlap - calculator._overlap * alpha_density * fock_alpha) * X;
                    const Eigen::MatrixXd error_beta =
                        X.transpose() *
                        (fock_beta * beta_density * calculator._overlap - calculator._overlap * beta_density * fock_beta) * X;
                    diis_alpha.push(fock_alpha, error_alpha);
                    diis_beta.push(fock_beta, error_beta);
                    diis_error = std::max(diis_alpha.error_norm(), diis_beta.error_norm());
                }

                // ── SOSCF window selection (D3.2) ────────────────────────
                // Structural copy of the RKS branch's own gate, which is
                // itself a copy of RHF/UHF's -- same scf_soscf_* keywords,
                // same criterion-or-fixed-iteration logic, same three
                // exclusions (hybrid / PCM / SAO).
                const bool sao_active_uks = calculator._use_sao_blocking &&
                                            calculator._sao_transform.rows() ==
                                                static_cast<Eigen::Index>(nbasis) &&
                                            calculator._sao_transform.cols() ==
                                                static_cast<Eigen::Index>(nbasis) &&
                                            !calculator._sao_block_sizes.empty();
                const bool soscf_enabled =
                    (calculator._scf._scf_soscf_diis_tol > 0.0 ||
                     calculator._scf._scf_soscf_start > 0) &&
                    !sao_active_uks && !prepared.pcm;
                if (soscf_enabled && soscf_window_start == 0)
                {
                    const bool criterion_fires =
                        calculator._scf._scf_soscf_diis_tol > 0.0
                            ? (use_diis && diis_error > 0.0 &&
                               diis_error < calculator._scf._scf_soscf_diis_tol &&
                               iter >= calculator._scf._scf_soscf_min_iter)
                            : (iter >= calculator._scf._scf_soscf_start);
                    if (criterion_fires)
                        soscf_window_start = iter;
                }
                const bool soscf_active =
                    soscf_enabled && soscf_window_start > 0 &&
                    iter < soscf_window_start + calculator._scf._scf_soscf_cycles &&
                    Ca_soscf_prev.size() > 0;
                if (soscf_window_start > 0 &&
                    iter == soscf_window_start + calculator._scf._scf_soscf_cycles)
                {
                    diis_alpha.clear();
                    diis_beta.clear();
                }

                const bool do_diis_uks = use_diis && !soscf_active;
                const Eigen::MatrixXd fock_alpha_diag =
                    (do_diis_uks && diis_alpha.ready()) ? diis_alpha.extrapolate() : fock_alpha;
                const Eigen::MatrixXd fock_beta_diag =
                    (do_diis_uks && diis_beta.ready()) ? diis_beta.extrapolate() : fock_beta;

                Eigen::MatrixXd Ca_new, Cb_new;
                Eigen::VectorXd epsa_new, epsb_new;
                std::vector<std::string> mo_sym_a_new, mo_sym_b_new;
                if (soscf_active)
                {
                    // ── SOSCF (D3.2) ────────────────────────────────────
                    // g_ai^sigma = F_mo^sigma(a,i), paired with the composed
                    // polarized h_op UNSCALED. D3.1 measured the scale
                    // convention against the true UKS E(kappa):
                    // d2E_total/dkappa2 = 2 * H_bare_polarized with the
                    // UHF-convention unscaled dP = C_a*C_i^T + C_i*C_a^T
                    // (no closed-shell 2x). Since a Newton step depends only
                    // on the ratio g/H and g_true = 2*g_bare, using
                    // g = F_mo against the unscaled h_op reproduces the true
                    // step -- exactly UHF's own conclusion.
                    const int n_alpha_i = static_cast<int>(n_alpha);
                    const int n_beta_i = static_cast<int>(n_beta);
                    const int n_virt_a_i = static_cast<int>(nbasis) - n_alpha_i;
                    const int n_virt_b_i = static_cast<int>(nbasis) - n_beta_i;
                    const int nova = n_virt_a_i * n_alpha_i;
                    const int novb = n_virt_b_i * n_beta_i;

                    const Eigen::MatrixXd Ca_occ = Ca_soscf_prev.leftCols(n_alpha_i);
                    const Eigen::MatrixXd Ca_virt = Ca_soscf_prev.rightCols(n_virt_a_i);
                    const Eigen::MatrixXd Cb_occ = Cb_soscf_prev.leftCols(n_beta_i);
                    const Eigen::MatrixXd Cb_virt = Cb_soscf_prev.rightCols(n_virt_b_i);

                    const Eigen::MatrixXd Fa_mo =
                        Ca_soscf_prev.transpose() * fock_alpha * Ca_soscf_prev;
                    const Eigen::MatrixXd Fb_mo =
                        Cb_soscf_prev.transpose() * fock_beta * Cb_soscf_prev;
                    Eigen::VectorXd g(nova + novb);
                    for (int a = 0; a < n_virt_a_i; ++a)
                        for (int i = 0; i < n_alpha_i; ++i)
                            g(a * n_alpha_i + i) = Fa_mo(n_alpha_i + a, i);
                    for (int a = 0; a < n_virt_b_i; ++a)
                        for (int i = 0; i < n_beta_i; ++i)
                            g(nova + a * n_beta_i + i) = Fb_mo(n_beta_i + a, i);

                    const Eigen::VectorXd diag_a =
                        DFT::Driver::orbital_energy_difference_diagonal(epsa_soscf_prev, n_alpha_i);
                    const Eigen::VectorXd diag_b =
                        DFT::Driver::orbital_energy_difference_diagonal(epsb_soscf_prev, n_beta_i);

                    const auto h_op = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd
                    {
                        Eigen::MatrixXd xa_mat(n_virt_a_i, n_alpha_i);
                        for (int a = 0; a < n_virt_a_i; ++a)
                            for (int i = 0; i < n_alpha_i; ++i)
                                xa_mat(a, i) = x(a * n_alpha_i + i);
                        Eigen::MatrixXd xb_mat(n_virt_b_i, n_beta_i);
                        for (int a = 0; a < n_virt_b_i; ++a)
                            for (int i = 0; i < n_beta_i; ++i)
                                xb_mat(a, i) = x(nova + a * n_beta_i + i);

                        const Eigen::MatrixXd d1a = Ca_virt * xa_mat * Ca_occ.transpose();
                        const Eigen::MatrixXd dPa = d1a + d1a.transpose();
                        const Eigen::MatrixXd d1b = Cb_virt * xb_mat * Cb_occ.transpose();
                        const Eigen::MatrixXd dPb = d1b + d1b.transpose();

                        // J is built from the TOTAL trial density (same as
                        // UKS's own per-iteration Coulomb build), one call.
                        const Eigen::MatrixXd dJ = _compute_2e_j_direct(
                            prepared.shell_pairs, dPa + dPb, calculator._shells.nbasis(),
                            calculator._integral._engine, HartreeFock::ERIKernel::Coulomb,
                            0.0, calculator._integral._tol_eri,
                            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops
                                                              : nullptr);
                        const Eigen::VectorXd Ja_packed =
                            DFT::Driver::pack_hessian_vector_product_cphf_order(dJ, Ca_occ, Ca_virt);
                        const Eigen::VectorXd Jb_packed =
                            DFT::Driver::pack_hessian_vector_product_cphf_order(dJ, Cb_occ, Cb_virt);

                        const auto dV_xc = DFT::Driver::compute_analytic_xc_hessian_vector_product_polarized(
                            prepared.molecular_grid, prepared.ao_grid,
                            alpha_density, beta_density, dPa, dPb,
                            x_functional, c_functional);
                        if (!dV_xc)
                            return Eigen::VectorXd::Zero(x.size());
                        const Eigen::VectorXd xca_packed =
                            DFT::Driver::pack_hessian_vector_product_cphf_order(
                                dV_xc->first, Ca_occ, Ca_virt);
                        const Eigen::VectorXd xcb_packed =
                            DFT::Driver::pack_hessian_vector_product_cphf_order(
                                dV_xc->second, Cb_occ, Cb_virt);

                        // K response (SOSCF_DFT.md invariant 3 (UKS hybrids)): the UKS KS
                        // Fock carries -1*(c_fr*K_C + c_sr*K_SR) per spin from
                        // _compute_2e_k_uhf_direct (K_alpha from Pa, K_beta from
                        // Pb in ONE sweep per kernel) -- see
                        // assemble_current_ks_potential's UHF branch. K linear in
                        // the density like J, and UKS h_op is uniform 1x (no
                        // RKS-style 2x on the kernel -- occupancy-1 makes diag and
                        // kernel scale identically, docs/SOSCF_DFT.md invariant 2).
                        Eigen::VectorXd Ka_packed = Eigen::VectorXd::Zero(nova);
                        Eigen::VectorXd Kb_packed = Eigen::VectorXd::Zero(novb);
                        {
                            const double c_fr = xc_grid->full_range_exchange_coefficient;
                            const double c_sr = xc_grid->short_range_exchange_coefficient;
                            if (c_fr != 0.0 || c_sr != 0.0)
                            {
                                Eigen::MatrixXd dKa = Eigen::MatrixXd::Zero(nbasis, nbasis);
                                Eigen::MatrixXd dKb = Eigen::MatrixXd::Zero(nbasis, nbasis);
                                if (c_fr != 0.0)
                                {
                                    const auto [Ka, Kb] = _compute_2e_k_uhf_direct(
                                        prepared.shell_pairs, dPa, dPb, calculator._shells.nbasis(),
                                        calculator._integral._engine, HartreeFock::ERIKernel::Coulomb,
                                        0.0, calculator._integral._tol_eri,
                                        calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops
                                                                          : nullptr);
                                    dKa.noalias() += c_fr * Ka;
                                    dKb.noalias() += c_fr * Kb;
                                }
                                if (c_sr != 0.0)
                                {
                                    const auto [Ka, Kb] = _compute_2e_k_uhf_direct(
                                        prepared.shell_pairs, dPa, dPb, calculator._shells.nbasis(),
                                        calculator._integral._engine, HartreeFock::ERIKernel::ShortRange,
                                        xc_grid->range_separation_omega, calculator._integral._tol_eri,
                                        calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops
                                                                          : nullptr);
                                    dKa.noalias() += c_sr * Ka;
                                    dKb.noalias() += c_sr * Kb;
                                }
                                Ka_packed = DFT::Driver::pack_hessian_vector_product_cphf_order(
                                    -1.0 * dKa, Ca_occ, Ca_virt);
                                Kb_packed = DFT::Driver::pack_hessian_vector_product_cphf_order(
                                    -1.0 * dKb, Cb_occ, Cb_virt);
                            }
                        }

                        Eigen::VectorXd out(nova + novb);
                        out.head(nova) =
                            diag_a.cwiseProduct(x.head(nova)) + Ja_packed + xca_packed + Ka_packed;
                        out.tail(novb) =
                            diag_b.cwiseProduct(x.tail(novb)) + Jb_packed + xcb_packed + Kb_packed;
                        return out;
                    };
                    const auto g_op = [&g]() -> Eigen::VectorXd
                    { return g; };

                    HartreeFock::Correlation::CASSCF::AugHessianOptions ah_opts;
                    ah_opts.ah_start_tol = std::max(1e-8, 0.1 * g.norm());
                    Eigen::VectorXd x0 = -g;
                    const double x0_norm = x0.norm();
                    if (std::isfinite(x0_norm) && x0_norm > 0.0)
                        x0 /= x0_norm;
                    const HartreeFock::Correlation::CASSCF::AugHessianResult ah =
                        HartreeFock::Correlation::CASSCF::solve_augmented_hessian(
                            h_op, g_op, nullptr, x0, ah_opts);

                    constexpr double kSoscfMaxRot = 0.20;
                    Eigen::MatrixXd kappa_a = Eigen::MatrixXd::Zero(nbasis, nbasis);
                    Eigen::MatrixXd kappa_b = Eigen::MatrixXd::Zero(nbasis, nbasis);
                    bool cap_fired = false;
                    if (ah.x.size() == nova + novb && ah.x.allFinite())
                    {
                        Eigen::VectorXd step = ah.x;
                        const double max_elem = step.cwiseAbs().maxCoeff();
                        if (max_elem > kSoscfMaxRot)
                        {
                            step *= kSoscfMaxRot / max_elem;
                            cap_fired = true;
                        }
                        for (int a = 0; a < n_virt_a_i; ++a)
                            for (int i = 0; i < n_alpha_i; ++i)
                            {
                                const double v = step(a * n_alpha_i + i);
                                kappa_a(n_alpha_i + a, i) = v;
                                kappa_a(i, n_alpha_i + a) = -v;
                            }
                        for (int a = 0; a < n_virt_b_i; ++a)
                            for (int i = 0; i < n_beta_i; ++i)
                            {
                                const double v = step(nova + a * n_beta_i + i);
                                kappa_b(n_beta_i + a, i) = v;
                                kappa_b(i, n_beta_i + a) = -v;
                            }
                    }
                    Ca_new = HartreeFock::Correlation::CASSCF::apply_orbital_rotation(
                        Ca_soscf_prev, kappa_a, calculator._overlap);
                    Cb_new = HartreeFock::Correlation::CASSCF::apply_orbital_rotation(
                        Cb_soscf_prev, kappa_b, calculator._overlap);
                    if (!Ca_new.allFinite() || !Cb_new.allFinite())
                        return std::unexpected(std::format(
                            "DFT UKS SOSCF: orbital rotation produced non-finite coefficients at "
                            "iteration {}",
                            iter));

                    // Semicanonicalize each spin channel separately -- pure
                    // gauge freedom, same as RHF/UHF/RKS SOSCF.
                    auto semicanon = [&](const Eigen::MatrixXd &C_in, const Eigen::MatrixXd &F_in,
                                         int n_occ_s, int n_virt_s, const char *tag)
                        -> std::expected<std::pair<Eigen::MatrixXd, Eigen::VectorXd>, std::string>
                    {
                        const Eigen::MatrixXd F_mo_new = C_in.transpose() * F_in * C_in;
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> occ_solver(
                            F_mo_new.topLeftCorner(n_occ_s, n_occ_s));
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> virt_solver(
                            F_mo_new.bottomRightCorner(n_virt_s, n_virt_s));
                        if (occ_solver.info() != Eigen::Success || virt_solver.info() != Eigen::Success)
                            return std::unexpected(std::format(
                                "DFT UKS SOSCF: {} semicanonicalization eigensolve failed at "
                                "iteration {}",
                                tag, iter));
                        Eigen::MatrixXd C_canon(nbasis, nbasis);
                        C_canon.leftCols(n_occ_s) = C_in.leftCols(n_occ_s) * occ_solver.eigenvectors();
                        C_canon.rightCols(n_virt_s) = C_in.rightCols(n_virt_s) * virt_solver.eigenvectors();
                        Eigen::VectorXd eps_out(nbasis);
                        eps_out.head(n_occ_s) = occ_solver.eigenvalues();
                        eps_out.tail(n_virt_s) = virt_solver.eigenvalues();
                        return std::make_pair(C_canon, eps_out);
                    };
                    auto canon_a = semicanon(Ca_new, fock_alpha, n_alpha_i, n_virt_a_i, "alpha");
                    if (!canon_a)
                        return std::unexpected(canon_a.error());
                    Ca_new = std::move(canon_a->first);
                    epsa_new = std::move(canon_a->second);
                    auto canon_b = semicanon(Cb_new, fock_beta, n_beta_i, n_virt_b_i, "beta");
                    if (!canon_b)
                        return std::unexpected(canon_b.error());
                    Cb_new = std::move(canon_b->first);
                    epsb_new = std::move(canon_b->second);

                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info, "DFT UKS SOSCF :",
                        std::format(
                            "step at iter {}: |g|={:.3e} v0={:.4f} eig={:.4e} converged={} "
                            "ah_iters={} ah_residual={:.3e} cap_fired={}",
                            iter, g.norm(), ah.v0, ah.eigenvalue, ah.converged, ah.iterations,
                            ah.residual_norm, cap_fired));
                }
                else
                {
                    auto alpha_diagonalization = diagonalize_in_ao_basis(
                        calculator,
                        X,
                        fock_alpha_diag,
                        "Alpha KS");
                    if (!alpha_diagonalization)
                        return std::unexpected(alpha_diagonalization.error());
                    auto beta_diagonalization = diagonalize_in_ao_basis(
                        calculator,
                        X,
                        fock_beta_diag,
                        "Beta KS");
                    if (!beta_diagonalization)
                        return std::unexpected(beta_diagonalization.error());
                    Ca_new = std::move(alpha_diagonalization->coefficients);
                    epsa_new = std::move(alpha_diagonalization->energies);
                    Cb_new = std::move(beta_diagonalization->coefficients);
                    epsb_new = std::move(beta_diagonalization->energies);
                    mo_sym_a_new = std::move(alpha_diagonalization->mo_symmetry);
                    mo_sym_b_new = std::move(beta_diagonalization->mo_symmetry);
                }

                const Eigen::MatrixXd next_alpha_density =
                    density_from_orbitals(Ca_new, n_alpha, 1.0);
                const Eigen::MatrixXd next_beta_density =
                    density_from_orbitals(Cb_new, n_beta, 1.0);
                const auto metrics = HartreeFock::SCF::unrestricted_iteration_metrics(
                    alpha_density,
                    beta_density,
                    next_alpha_density,
                    next_beta_density,
                    previous_total_energy,
                    total_energy);

                const double iter_time = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() - iter_start)
                                             .count();
                HartreeFock::Logger::scf_iteration(
                    iter,
                    total_energy,
                    metrics.delta_energy,
                    metrics.delta_density_rms,
                    metrics.delta_density_max,
                    diis_error,
                    0.0,
                    iter_time);

                alpha_density = next_alpha_density;
                beta_density = next_beta_density;
                previous_total_energy = total_energy;
                // SOSCF (D3.2): keep the per-spin reference basis current
                // every iteration so the switch iteration always has a valid
                // Ca_soscf_prev/Cb_soscf_prev the moment it fires -- same
                // discipline RHF/UHF/RKS SOSCF use.
                Ca_soscf_prev = Ca_new;
                Cb_soscf_prev = Cb_new;
                epsa_soscf_prev = epsa_new;
                epsb_soscf_prev = epsb_new;

                HartreeFock::SCF::store_unrestricted_iteration(
                    calculator,
                    HartreeFock::SCF::UnrestrictedIterationData{
                        .alpha_density = alpha_density,
                        .beta_density = beta_density,
                        .alpha_fock = fock_alpha,
                        .beta_fock = fock_beta,
                        .alpha_mo_energies = epsa_new,
                        .beta_mo_energies = epsb_new,
                        .alpha_mo_coefficients = Ca_new,
                        .beta_mo_coefficients = Cb_new,
                        .electronic_energy = electronic_energy,
                        .total_energy = total_energy},
                    metrics);
                calculator._info._scf.alpha.mo_symmetry = mo_sym_a_new;
                calculator._info._scf.beta.mo_symmetry = mo_sym_b_new;

                result.total_energy = total_energy;
                result.xc_energy = xc_grid->total_energy + ks_potential->exact_exchange_energy;
                result.integrated_electrons = xc_grid->integrated_electrons;
                result.solvation_energy = pcm_energy;

                if (HartreeFock::SCF::is_converged(calculator._scf, metrics, iter))
                {
                    calculator._info._is_converged = true;
                    result.converged = true;
                    HartreeFock::Logger::scf_footer();
                    HartreeFock::Logger::blank();
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info,
                        "UKS Converged :",
                        std::format("E = {:.10f} Eh after {} iterations", total_energy, iter));
                    HartreeFock::Logger::blank();

                    return result;
                }
            }

            HartreeFock::Logger::scf_footer();
            return std::unexpected(std::format("UKS did not converge in {} iterations", max_iter));
        }

        std::expected<DFT::XC::Functional, std::string> initialize_functional(
            int functional_id,
            DFT::XC::Spin spin)
        {
            return DFT::XC::Functional::create(functional_id, spin);
        }

        struct InitializedFunctionals
        {
            DFT::XC::Functional exchange;
            DFT::XC::Functional correlation;
            double implemented_exact_exchange_coefficient = 0.0;
            double perturbative_correlation_coefficient = 0.0;
            bool has_range_separation = false;
            bool has_double_hybrid_pt2 = false;
        };

        std::string workflow_label(HartreeFock::CalculationType calculation)
        {
            switch (calculation)
            {
            case HartreeFock::CalculationType::SinglePoint:
                return "single-point energies";
            case HartreeFock::CalculationType::LinearResponse:
                return "linear-response / TDDFT";
            case HartreeFock::CalculationType::Gradient:
                return "analytic gradients";
            case HartreeFock::CalculationType::GeomOpt:
                return "geometry optimization";
            case HartreeFock::CalculationType::Frequency:
                return "frequency analysis";
            case HartreeFock::CalculationType::GeomOptFrequency:
                return "geometry optimization + frequency analysis";
            case HartreeFock::CalculationType::ImaginaryFollow:
                return "imaginary-mode following";
            }

            return "this workflow";
        }

        std::expected<void, std::string> validate_workflow_support(
            const HartreeFock::Calculator &calculator,
            const InitializedFunctionals &functionals)
        {
            if (calculator._calculation == HartreeFock::CalculationType::SinglePoint)
                return {};

            // Analytic gradients and gradient-driven workflows for
            // range-separated (non-double-hybrid) functionals are validated
            // end-to-end against PySCF on water/STO-3G HSE06:
            //   - Gradient components agree at <1e-6 Ha/Bohr for HF-2e-Exchange-LR
            //     (water STO-3G + 6-31G*); FD self-consistency <3e-7 Ha/Bohr.
            //   - Freq @ input geometry: max |Δ| = 2.17 cm^-1 vs PySCF analytic Hessian.
            //   - GeomOpt: final geometry within ~7e-4 Å, final energy within ~2e-5 Eh.
            //   - GeomOptFreq @ optimized geometry: max |Δ| = 6.78 cm^-1.
            // Double-hybrid PT2 paths and ImaginaryFollow / LinearResponse for
            // range-separated functionals remain unvalidated.
            if (functionals.has_range_separation &&
                !functionals.has_double_hybrid_pt2)
            {
                switch (calculator._calculation)
                {
                case HartreeFock::CalculationType::Gradient:
                case HartreeFock::CalculationType::Frequency:
                case HartreeFock::CalculationType::GeomOpt:
                case HartreeFock::CalculationType::GeomOptFrequency:
                    return {};
                default:
                    break;
                }
            }

            if (functionals.has_range_separation &&
                !functionals.has_double_hybrid_pt2 &&
                dft_allow_unvalidated_range_separated_workflows())
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "DFT Driver :",
                    "Debug override enabled: allowing unvalidated range-separated non-single-point workflow");
                return {};
            }

            if (!functionals.has_range_separation &&
                !functionals.has_double_hybrid_pt2)
                return {};

            // In-development double-hybrid analytic gradient (N3).
            if (functionals.has_double_hybrid_pt2 &&
                dft_allow_double_hybrid_gradient())
            {
                switch (calculator._calculation)
                {
                case HartreeFock::CalculationType::Gradient:
                case HartreeFock::CalculationType::Frequency:
                case HartreeFock::CalculationType::GeomOpt:
                case HartreeFock::CalculationType::GeomOptFrequency:
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Warning,
                        "DFT Driver :",
                        "PLANCK_DFT_DH_GRADIENT set: double-hybrid analytic gradient is "
                        "in development and unvalidated");
                    return {};
                default:
                    break;
                }
            }

            return std::unexpected(
                std::format(
                    "{} currently supports only single-point energies for range-separated and double-hybrid functionals",
                    workflow_label(calculator._calculation)));
        }

        std::expected<void, std::string> apply_post_ks_double_hybrid_correction(
            HartreeFock::Calculator &calculator,
            const std::vector<HartreeFock::ShellPair> &shell_pairs,
            const InitializedFunctionals &functionals,
            Result &result)
        {
            if (std::abs(functionals.perturbative_correlation_coefficient) <= 1.0e-14)
                return {};

            std::expected<void, std::string> correction =
                calculator._scf._scf == HartreeFock::SCFType::UHF
                    ? [&]() -> std::expected<void, std::string>
                      {
                          auto mp2_res = HartreeFock::Correlation::ump2_kernel(calculator, shell_pairs, calculator._mp2);
                          return mp2_res
                                     ? HartreeFock::Correlation::apply_ump2_result(calculator, *mp2_res)
                                     : std::unexpected(mp2_res.error());
                      }()
                    : [&]() -> std::expected<void, std::string>
                      {
                          auto mp2_res = HartreeFock::Correlation::rmp2_kernel(calculator, shell_pairs, calculator._mp2);
                          return mp2_res
                                     ? HartreeFock::Correlation::apply_rmp2_result(calculator, *mp2_res)
                                     : std::unexpected(mp2_res.error());
                      }();
            if (!correction)
            {
                return std::unexpected(
                    "DFT double-hybrid perturbative correction failed: " + correction.error());
            }

            const double bare_pt2 = calculator._correlation_energy;
            const double scaled_pt2 =
                functionals.perturbative_correlation_coefficient * bare_pt2;

            calculator._correlation_energy = scaled_pt2;
            calculator._correlated_total_energy = calculator._total_energy + scaled_pt2;
            calculator._have_correlated_total_energy = true;

            result.total_energy += scaled_pt2;
            result.xc_energy += scaled_pt2;

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Double Hybrid :",
                std::format(
                    "{} PT2 coefficient = {:.6f}; bare MP2-like correction = {:.10f} Eh; scaled contribution = {:.10f} Eh",
                    functionals.exchange.name(),
                    functionals.perturbative_correlation_coefficient,
                    bare_pt2,
                    scaled_pt2));

            return {};
        }

        std::expected<InitializedFunctionals, std::string> initialize_functionals(
            HartreeFock::Calculator &calculator)
        {
            const DFT::XC::Spin spin = (calculator._scf._scf == HartreeFock::SCFType::UHF)
                                           ? DFT::XC::Spin::Polarized
                                           : DFT::XC::Spin::Unpolarized;

            const auto exchange_id = resolve_functional_id(
                calculator._dft._exchange,
                calculator._dft._exchange_id);
            if (!exchange_id)
                return std::unexpected("DFT exchange functional resolution failed: " + exchange_id.error());
            calculator._dft._exchange_id = *exchange_id;

            const auto correlation_id = resolve_functional_id(
                calculator._dft._correlation,
                calculator._dft._correlation_id);
            if (!correlation_id)
                return std::unexpected("DFT correlation functional resolution failed: " + correlation_id.error());
            calculator._dft._correlation_id = *correlation_id;

            auto exchange = initialize_functional(calculator._dft._exchange_id, spin);
            if (!exchange)
                return std::unexpected("DFT exchange functional initialization failed: " + exchange.error());

            auto correlation = initialize_functional(calculator._dft._correlation_id, spin);
            if (!correlation)
                return std::unexpected("DFT correlation functional initialization failed: " + correlation.error());

            const bool has_range_separation = exchange->is_range_separated();
            const double implemented_exact_exchange_coefficient =
                exchange->fock_exchange_coefficient();
            const double perturbative_correlation_coefficient =
                exchange->perturbative_correlation_coefficient();

            if (exchange->is_global_hybrid())
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Hybrid XC :",
                    std::format("{} exact exchange coefficient = {:.6f}",
                                exchange->name(),
                                exchange->exact_exchange_coefficient()));
            }
            else if (exchange->is_range_separated())
            {
                const auto cam = exchange->cam_coefficients();
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Hybrid XC :",
                    std::format(
                        "{} range-separated exchange uses alpha(full-range) = {:.6f}, beta(short-range) = {:.6f}, omega = {:.6f}",
                        exchange->name(),
                        cam.alpha,
                        cam.beta,
                        cam.omega));
            }
            else if (std::abs(implemented_exact_exchange_coefficient) > 1.0e-14)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Hybrid XC :",
                    std::format("{} exact exchange coefficient = {:.6f}",
                                exchange->name(),
                                implemented_exact_exchange_coefficient));
            }

            if (std::abs(perturbative_correlation_coefficient) > 1.0e-14)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Double Hybrid :",
                    std::format("{} PT2 correlation coefficient = {:.6f}",
                                exchange->name(),
                                perturbative_correlation_coefficient));
            }

            if (exchange->is_combined_exchange_correlation())
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Correlation :",
                    std::format("Using {} as a combined XC functional; configured correlation is ignored",
                                exchange->name()));
            }

            return InitializedFunctionals{
                .exchange = std::move(*exchange),
                .correlation = std::move(*correlation),
                .implemented_exact_exchange_coefficient = implemented_exact_exchange_coefficient,
                .perturbative_correlation_coefficient = perturbative_correlation_coefficient,
                .has_range_separation = has_range_separation,
                .has_double_hybrid_pt2 = std::abs(perturbative_correlation_coefficient) > 1.0e-14};
        }

        std::expected<Eigen::MatrixXd, std::string> compute_analytic_ks_gradient(
            HartreeFock::Calculator &calculator,
            const PreparedSystem &prepared,
            const InitializedFunctionals &functionals);

        std::expected<PreparedSystem, std::string> prepare_current_geometry(
            HartreeFock::Calculator &calculator,
            bool preserve_previous_density)
        {
            calculator.sync_coordinate_frames_from_standard();
            calculator._molecule._symmetry = false;
            calculator._molecule._point_group = "C1";
            reset_sao_state(calculator);

            const Eigen::MatrixXd previous_alpha_density = calculator._info._scf.alpha.density;
            const Eigen::MatrixXd previous_beta_density = calculator._info._scf.beta.density;

            const std::string gbs_path =
                calculator._basis._basis_path + "/" + calculator._basis._basis_name;
            auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
                gbs_path,
                calculator._molecule,
                calculator._basis._basis);
            if (!basis_res)
                return std::unexpected("DFT basis setup failed: " + basis_res.error());
            calculator._shells = std::move(*basis_res);

            calculator._info._scf = HartreeFock::DataSCF(
                calculator._scf._scf == HartreeFock::SCFType::UHF);
            calculator._info._scf.initialize(calculator._shells.nbasis());
            calculator._scf.set_scf_mode_auto(calculator._shells.nbasis());
            calculator._scf.set_max_cycles_auto(calculator._shells.nbasis());
            calculator._info._is_converged = false;
            calculator._eri.clear();
            if (auto nuclear_repulsion = calculator.recompute_nuclear_repulsion(); !nuclear_repulsion)
                return std::unexpected("DFT geometry preparation failed: " + nuclear_repulsion.error());

            PreparedSystem prepared;
            auto preset = grid_preset(to_grid_level(calculator._dft._grid));
            if (!preset)
                return std::unexpected(preset.error());
            prepared.grid_preset = *preset;
            prepared.shell_pairs = build_shellpairs(calculator._shells);

            if (auto res = compute_one_electron_terms(calculator, prepared.shell_pairs); !res)
                return std::unexpected(res.error());

            auto molecular_grid = MakeMolecularGrid(
                calculator._molecule,
                to_grid_level(calculator._dft._grid));
            if (!molecular_grid)
                return std::unexpected("DFT molecular grid construction failed: " + molecular_grid.error());
            prepared.molecular_grid = std::move(*molecular_grid);

            auto ao_grid = evaluate_ao_basis_on_grid(calculator._shells, prepared.molecular_grid);
            if (!ao_grid)
                return std::unexpected("DFT AO grid evaluation failed: " + ao_grid.error());
            prepared.ao_grid = std::move(*ao_grid);

            if (calculator._solvation._model != HartreeFock::SolvationModel::None)
            {
                auto pcm = HartreeFock::Solvation::build_pcm_state(calculator, prepared.shell_pairs);
                if (!pcm)
                    return std::unexpected("DFT PCM setup failed: " + pcm.error());
                prepared.pcm = std::move(*pcm);
            }

            const Eigen::Index nbasis = static_cast<Eigen::Index>(calculator._shells.nbasis());
            const bool can_reuse_alpha =
                preserve_previous_density &&
                previous_alpha_density.rows() == nbasis &&
                previous_alpha_density.cols() == nbasis;
            const bool can_reuse_beta =
                preserve_previous_density &&
                previous_beta_density.rows() == nbasis &&
                previous_beta_density.cols() == nbasis;

            if (can_reuse_alpha)
                calculator._info._scf.alpha.density = previous_alpha_density;
            if (calculator._scf._scf == HartreeFock::SCFType::UHF && can_reuse_beta)
                calculator._info._scf.beta.density = previous_beta_density;

            if (!can_reuse_alpha ||
                (calculator._scf._scf == HartreeFock::SCFType::UHF && !can_reuse_beta))
            {
                if (auto res = initialize_ks_guess(calculator); !res)
                    return std::unexpected(res.error());
            }

            return prepared;
        }

        // Rebuild Lebedev/Becke quadrature and AO values without disturbing the SCF
        // wavefunction — required for analytic gradients after KS convergence.
        std::expected<PreparedSystem, std::string> prepare_quadrature_for_calculator(
            HartreeFock::Calculator &calculator)
        {
            PreparedSystem prepared;
            auto preset = grid_preset(to_grid_level(calculator._dft._grid));
            if (!preset)
                return std::unexpected(preset.error());
            prepared.grid_preset = *preset;
            prepared.shell_pairs = build_shellpairs(calculator._shells);

            auto molecular_grid = MakeMolecularGrid(
                calculator._molecule,
                to_grid_level(calculator._dft._grid));
            if (!molecular_grid)
                return std::unexpected("DFT molecular grid construction failed: " + molecular_grid.error());
            prepared.molecular_grid = std::move(*molecular_grid);

            auto ao_grid = evaluate_ao_basis_on_grid(calculator._shells, prepared.molecular_grid);
            if (!ao_grid)
                return std::unexpected("DFT AO grid evaluation failed: " + ao_grid.error());
            prepared.ao_grid = std::move(*ao_grid);

            if (calculator._solvation._model != HartreeFock::SolvationModel::None)
            {
                auto pcm = HartreeFock::Solvation::build_pcm_state(calculator, prepared.shell_pairs);
                if (!pcm)
                    return std::unexpected("DFT PCM setup failed: " + pcm.error());
                prepared.pcm = std::move(*pcm);
            }

            return prepared;
        }

        std::expected<Result, std::string> run_single_point_current_geometry(
            HartreeFock::Calculator &calculator,
            const InitializedFunctionals &functionals,
            bool preserve_previous_density)
        {
            // Geometry-derivative code paths call back into the driver many
            // times.  This helper rebuilds only geometry-dependent objects and
            // optionally seeds the next SCF with the density from the previous
            // nearby geometry for faster convergence.
            auto prepared = prepare_current_geometry(calculator, preserve_previous_density);
            if (!prepared)
                return std::unexpected(prepared.error());

            auto result = run_ks_scf_scaffold(
                calculator,
                *prepared,
                functionals.exchange,
                functionals.correlation);
            if (!result)
                return result;

            if (auto correction = apply_post_ks_double_hybrid_correction(
                    calculator,
                    prepared->shell_pairs,
                    functionals,
                    *result);
                !correction)
            {
                return std::unexpected(correction.error());
            }

            return result;
        }

        std::expected<Result, std::string> run_initial_single_point(
            HartreeFock::Calculator &calculator,
            const Options &options,
            const InitializedFunctionals &functionals)
        {
            auto prepared = prepare(calculator, options);
            if (!prepared)
                return std::unexpected(prepared.error());

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Driver :",
                "Preparation complete; entering KS-SCF");

            auto result = run_ks_scf_scaffold(
                calculator,
                *prepared,
                functionals.exchange,
                functionals.correlation);
            if (!result)
                return result;

            if (auto correction = apply_post_ks_double_hybrid_correction(
                    calculator,
                    prepared->shell_pairs,
                    functionals,
                    *result);
                !correction)
            {
                return std::unexpected(correction.error());
            }

            if ((calculator._dft._save_checkpoint || options.save_checkpoint) && result->converged)
            {
                if (auto save = HartreeFock::Checkpoint::save(calculator, calculator._checkpoint_path); !save)
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Warning,
                        "DFT Checkpoint :",
                        std::format("Save failed: {}", save.error()));
            }

            return result;
        }

        Eigen::VectorXd flatten_gradient_atom_major(const Eigen::Ref<const Eigen::MatrixXd> &gradient)
        {
            Eigen::VectorXd flattened(gradient.rows() * gradient.cols());
            for (Eigen::Index atom = 0; atom < gradient.rows(); ++atom)
                for (Eigen::Index axis = 0; axis < gradient.cols(); ++axis)
                    flattened(atom * gradient.cols() + axis) = gradient(atom, axis);
            return flattened;
        }

        std::expected<Eigen::MatrixXd, std::string> compute_numeric_gradient(
            HartreeFock::Calculator &calculator,
            const InitializedFunctionals &functionals,
            double step_bohr = NUMERICAL_GRADIENT_STEP_BOHR)
        {
            if (step_bohr <= 0.0)
                return std::unexpected("DFT numerical gradient step must be positive");

            const Eigen::MatrixXd reference_geometry = calculator._molecule._standard;
            const bool reference_symmetry = calculator._molecule._symmetry;
            const std::string reference_point_group = calculator._molecule._point_group;
            const std::size_t natoms = calculator._molecule.natoms;
            Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(natoms),
                3);

            auto energy_at = [&](const Eigen::MatrixXd &geometry) -> std::expected<double, std::string>
            {
                // Each displaced-point energy is a full DFT single-point at a
                // new geometry.  Silence the nested SCF logging so the user sees
                // the final derivative report rather than dozens of inner traces.
                calculator._molecule.set_standard_from_bohr(geometry);
                HartreeFock::Logger::ScopedSilence silence;
                auto result = run_single_point_current_geometry(
                    calculator,
                    functionals,
                    true);
                if (!result)
                    return std::unexpected(result.error());
                return result->total_energy;
            };

            for (std::size_t atom = 0; atom < natoms; ++atom)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    Eigen::MatrixXd geometry_plus = reference_geometry;
                    Eigen::MatrixXd geometry_minus = reference_geometry;
                    geometry_plus(static_cast<Eigen::Index>(atom), axis) += step_bohr;
                    geometry_minus(static_cast<Eigen::Index>(atom), axis) -= step_bohr;

                    auto energy_plus = energy_at(geometry_plus);
                    if (!energy_plus)
                        return std::unexpected("DFT +h energy evaluation failed: " + energy_plus.error());

                    auto energy_minus = energy_at(geometry_minus);
                    if (!energy_minus)
                        return std::unexpected("DFT -h energy evaluation failed: " + energy_minus.error());

                    gradient(static_cast<Eigen::Index>(atom), axis) =
                        (*energy_plus - *energy_minus) / (2.0 * step_bohr);
                }
            }

            calculator._molecule.set_standard_from_bohr(reference_geometry);
            {
                HartreeFock::Logger::ScopedSilence silence;
                auto reference = run_single_point_current_geometry(
                    calculator,
                    functionals,
                    true);
                if (!reference)
                    return std::unexpected("DFT reference energy restoration failed: " + reference.error());
            }

            calculator._molecule._symmetry = reference_symmetry;
            calculator._molecule._point_group = reference_point_group;
            calculator.sync_coordinate_frames_from_standard();

            calculator._gradient = gradient;
            return gradient;
        }

        void print_gradient_report(const Eigen::Ref<const Eigen::MatrixXd> &gradient)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Nuclear Gradient (Ha/Bohr) :",
                "");
            for (Eigen::Index atom = 0; atom < gradient.rows(); ++atom)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "",
                    std::format(
                        "  Atom {:3d}: {:14.8f}  {:14.8f}  {:14.8f}",
                        static_cast<int>(atom + 1),
                        gradient(atom, 0),
                        gradient(atom, 1),
                        gradient(atom, 2)));
            }

            const double gmax = gradient.cwiseAbs().maxCoeff();
            const double grms =
                std::sqrt(gradient.squaredNorm() / static_cast<double>(gradient.size()));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Gradient max|g| :",
                std::format("{:.6e} Ha/Bohr", gmax));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Gradient rms|g| :",
                std::format("{:.6e} Ha/Bohr", grms));
            HartreeFock::Logger::blank();
        }

        void print_gradient_component_report(
            const std::string &label,
            const Eigen::Ref<const Eigen::MatrixXd> &gradient)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Gradient Debug :",
                std::format("{} component (Ha/Bohr)", label));
            for (Eigen::Index atom = 0; atom < gradient.rows(); ++atom)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "",
                    std::format(
                        "  Atom {:3d}: {:14.8f}  {:14.8f}  {:14.8f}",
                        static_cast<int>(atom + 1),
                        gradient(atom, 0),
                        gradient(atom, 1),
                        gradient(atom, 2)));
            }

            const double gmax = gradient.cwiseAbs().maxCoeff();
            const double grms =
                std::sqrt(gradient.squaredNorm() / static_cast<double>(gradient.size()));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Gradient Debug :",
                std::format("{} max|g| = {:.6e} Ha/Bohr", label, gmax));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Gradient Debug :",
                std::format("{} rms|g| = {:.6e} Ha/Bohr", label, grms));
            HartreeFock::Logger::blank();
        }

        Eigen::MatrixXd rotate_gradient_to_requested_frame_if_needed(
            HartreeFock::Calculator &calculator,
            const Eigen::Ref<const Eigen::MatrixXd> &gradient_standard_frame,
            const Eigen::Ref<const Eigen::MatrixXd> &requested_frame_bohr)
        {
            if (!calculator._geometry._use_symm || !calculator._molecule._symmetry)
                return gradient_standard_frame;

            (void)requested_frame_bohr;
            return rotate_gradient_rows(
                gradient_standard_frame,
                calculator._molecule._symmetry_alignment_transform);
        }

        void store_frequency_result(
            HartreeFock::Calculator &calculator,
            const HartreeFock::Freq::HessianResult &freq_result)
        {
            calculator._hessian = freq_result.hessian;
            calculator._frequencies = freq_result.frequencies;
            calculator._normal_modes = freq_result.normal_modes;
            calculator._vibrational_symmetry = freq_result.mode_symmetry;
            calculator._zpe = freq_result.zpe;
        }

        void print_frequency_report(
            const HartreeFock::Calculator &calculator,
            const HartreeFock::Freq::HessianResult &freq_result)
        {
            HartreeFock::Logger::blank();
            const int n_vib = freq_result.n_vib;
            const int n_tr = static_cast<int>(calculator._molecule.natoms * 3) - n_vib;
            const std::string geo_label = freq_result.is_linear ? "linear" : "non-linear";

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Vibrational Frequencies :",
                "");
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "",
                std::format(
                    "  Molecule: {} ({} T+R modes removed, {} vibrational modes)",
                    geo_label,
                    n_tr,
                    n_vib));
            const bool have_mode_symmetry =
                freq_result.mode_symmetry.size() == static_cast<std::size_t>(n_vib) &&
                !freq_result.mode_symmetry.empty();
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "",
                have_mode_symmetry
                    ? "  ─────────────────────────────────────────────────────"
                    : "  ──────────────────────────────────────────");
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "",
                have_mode_symmetry
                    ? "    Mode    Symmetry    Frequency (cm⁻¹)"
                    : "    Mode    Frequency (cm⁻¹)");
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "",
                have_mode_symmetry
                    ? "  ─────────────────────────────────────────────────────"
                    : "  ────────────────────────────────────────────");

            for (int i = 0; i < n_vib; ++i)
            {
                const double freq = freq_result.frequencies[i];
                if (freq < 0.0)
                {
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info,
                        "",
                        have_mode_symmetry
                            ? std::format(
                                  "  {:6d}  {:10s}  {:14.2f}i  (imaginary)",
                                  i + 1,
                                  freq_result.mode_symmetry[static_cast<std::size_t>(i)],
                                  -freq)
                            : std::format(
                                  "  {:6d}  {:14.2f}i  (imaginary)",
                                  i + 1,
                                  -freq));
                }
                else
                {
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info,
                        "",
                        have_mode_symmetry
                            ? std::format(
                                  "  {:6d}  {:10s}  {:14.2f}",
                                  i + 1,
                                  freq_result.mode_symmetry[static_cast<std::size_t>(i)],
                                  freq)
                            : std::format("  {:6d}  {:14.2f}", i + 1, freq));
                }
            }

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "",
                have_mode_symmetry
                    ? "  ─────────────────────────────────────────────────────"
                    : "  ────────────────────────────────────────────");

            if (freq_result.n_imaginary > 0)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "Frequency :",
                    std::format(
                        "{} imaginary frequency(ies) — structure may be a saddle point",
                        freq_result.n_imaginary));
            }

            const double zpe_kcal = freq_result.zpe * HARTREE_TO_KCALMOL;
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Zero-point energy :",
                std::format("{:.6f} Eh  ({:.2f} kcal/mol)", freq_result.zpe, zpe_kcal));
            HartreeFock::Logger::blank();
        }

        std::expected<HartreeFock::Freq::HessianResult, std::string> run_frequency_analysis(
            HartreeFock::Calculator &calculator,
            const InitializedFunctionals &functionals)
        {
            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Frequency :",
                std::format(
                    "Computing numerical Hessian via analytic KS gradients (central differences, gradient step = {:.4f} Bohr)",
                    NUMERICAL_GRADIENT_STEP_BOHR));

            auto gradient_runner = [&](HartreeFock::Calculator &inner) -> std::expected<Eigen::MatrixXd, std::string>
            {
                auto ks = run_single_point_current_geometry(inner, functionals, true);
                if (!ks)
                    return std::unexpected(ks.error());
                if (!ks->converged)
                    return std::unexpected("KS-SCF did not converge during Hessian displacement");

                auto pq = prepare_quadrature_for_calculator(inner);
                if (!pq)
                    return std::unexpected(pq.error());

                return compute_analytic_ks_gradient(inner, *pq, functionals);
            };

            auto result = HartreeFock::Freq::compute_hessian(calculator, gradient_runner);
            if (!result)
                return std::unexpected(result.error());
            store_frequency_result(calculator, *result);
            print_frequency_report(calculator, *result);
            return result;
        }

        std::expected<Result, std::string> run_geometry_optimization(
            HartreeFock::Calculator &calculator,
            const InitializedFunctionals &functionals)
        {
            calculator.prepare_coordinates();
            calculator._molecule.set_standard_from_bohr(calculator._molecule._coordinates);

            if (!calculator._constraints.empty())
            {
                if (calculator._opt_coords != HartreeFock::OptCoords::Internal)
                {
                    return std::unexpected(
                        "Constrained optimization requires opt_coords internal");
                }
                if (calculator._geometry._type != HartreeFock::CoordType::ZMatrix)
                {
                    return std::unexpected(
                        "Constrained optimization requires coord_type zmatrix");
                }
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "Constraints :",
                    std::format("{} constraint(s) active", calculator._constraints.size()));
            }

            const bool use_internal_coordinates =
                calculator._opt_coords == HartreeFock::OptCoords::Internal;
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Geometry Optimization :",
                use_internal_coordinates
                    ? "Starting IC-BFGS optimizer with analytic KS gradients"
                    : "Starting L-BFGS optimizer with analytic KS gradients");
            HartreeFock::Logger::blank();

            auto gradient_runner = [&](HartreeFock::Calculator &inner) -> std::expected<Eigen::VectorXd, std::string>
            {
                auto ks = run_single_point_current_geometry(inner, functionals, true);
                if (!ks)
                    return std::unexpected(ks.error());
                if (!ks->converged)
                    return std::unexpected("KS-SCF did not converge during analytic gradient evaluation");

                auto pq = prepare_quadrature_for_calculator(inner);
                if (!pq)
                    return std::unexpected(pq.error());

                auto gradient = compute_analytic_ks_gradient(inner, *pq, functionals);
                if (!gradient)
                    return std::unexpected(gradient.error());
                return flatten_gradient_atom_major(*gradient);
            };

            auto opt_result = use_internal_coordinates
                                  ? HartreeFock::Opt::run_geomopt_ic(calculator, gradient_runner)
                                  : HartreeFock::Opt::run_geomopt(calculator, gradient_runner);
            if (!opt_result)
                return std::unexpected(opt_result.error());

            HartreeFock::Logger::blank();
            HartreeFock::Logger::logging(
                opt_result->converged ? HartreeFock::LogLevel::Info : HartreeFock::LogLevel::Warning,
                "Geometry Optimization :",
                opt_result->converged
                    ? std::format("Converged in {} steps", opt_result->iterations)
                    : std::format("Did NOT converge after {} steps", opt_result->iterations));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Final Energy :",
                std::format("{:.10f} Eh", opt_result->energy));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Final max|g| :",
                std::format("{:.6e} Ha/Bohr", opt_result->grad_max));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Optimized Geometry (Angstrom) :",
                "");
            for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "",
                    std::format(
                        "  Atom {:3d}:  {:14d}  {:14.8f}  {:14.8f}  {:14.8f}",
                        static_cast<int>(atom + 1),
                        static_cast<int>(calculator._molecule.atomic_numbers[atom]),
                        opt_result->final_coords(static_cast<Eigen::Index>(atom), 0) * BOHR_TO_ANGSTROM,
                        opt_result->final_coords(static_cast<Eigen::Index>(atom), 1) * BOHR_TO_ANGSTROM,
                        opt_result->final_coords(static_cast<Eigen::Index>(atom), 2) * BOHR_TO_ANGSTROM));
            }
            HartreeFock::Logger::blank();

            return Result{
                .total_energy = opt_result->energy,
                .xc_energy = 0.0,
                .integrated_electrons = 0.0,
                .converged = opt_result->converged};
        }

    } // namespace

    // Moved out of the anonymous namespace above (F3.1,
    // docs/DFT_ANALYTIC_FXC_HESSIAN.md) so F3's own Hessian-vector-
    // product verification can call the FD-kernel oracle directly from a
    // standalone test binary. Declared in driver.h; bodies unchanged from
    // their original internal-linkage form.
    Eigen::MatrixXd transition_density_matrix(
        const Eigen::Ref<const Eigen::VectorXd> &occupied,
        const Eigen::Ref<const Eigen::VectorXd> &virtual_orbital)
    {
        const Eigen::MatrixXd unsymmetrized = occupied * virtual_orbital.transpose();
        return (0.5 * (unsymmetrized + unsymmetrized.transpose())).eval();
    }

    std::expected<XCMatrixContribution, std::string> evaluate_xc_matrix_from_spin_densities(
        const PreparedSystem &prepared,
        const Eigen::Ref<const Eigen::MatrixXd> &alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &beta_density,
        const DFT::XC::Functional &exchange_functional,
        const DFT::XC::Functional &correlation_functional)
    {
        auto xc_grid = evaluate_xc_on_grid(
            prepared.molecular_grid,
            prepared.ao_grid,
            alpha_density,
            beta_density,
            exchange_functional,
            correlation_functional);
        if (!xc_grid)
            return std::unexpected(xc_grid.error());

        auto xc_matrix = assemble_xc_matrix(
            prepared.molecular_grid,
            prepared.ao_grid,
            *xc_grid);
        if (!xc_matrix)
            return std::unexpected(xc_matrix.error());

        return *xc_matrix;
    }

    std::expected<std::vector<std::vector<Eigen::MatrixXd>>, std::string> build_unrestricted_xc_kernel_blocks(
        const PreparedSystem &prepared,
        const std::vector<ResponseExcitationSpace> &spaces,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_alpha_density,
        const Eigen::Ref<const Eigen::MatrixXd> &ground_beta_density,
        const DFT::XC::Functional &exchange_functional,
        const DFT::XC::Functional &correlation_functional)
    {
        const int nspaces = static_cast<int>(spaces.size());
        std::vector<std::vector<Eigen::MatrixXd>> blocks(
            static_cast<std::size_t>(nspaces),
            std::vector<Eigen::MatrixXd>(static_cast<std::size_t>(nspaces)));

        for (int target = 0; target < nspaces; ++target)
            for (int source = 0; source < nspaces; ++source)
                blocks[static_cast<std::size_t>(target)][static_cast<std::size_t>(source)] =
                    Eigen::MatrixXd::Zero(spaces[static_cast<std::size_t>(target)].nov(),
                                          spaces[static_cast<std::size_t>(source)].nov());

        for (int source = 0; source < nspaces; ++source)
        {
            const ResponseExcitationSpace &source_space = spaces[static_cast<std::size_t>(source)];
            for (int j = 0; j < source_space.n_occ; ++j)
                for (int b = 0; b < source_space.n_virt; ++b)
                {
                    const Eigen::MatrixXd delta_density =
                        transition_density_matrix(source_space.C_occ.col(j), source_space.C_virt.col(b));
                    const double delta_scale = std::max(1.0, delta_density.cwiseAbs().maxCoeff());
                    const double step = 1.0e-5 / delta_scale;

                    Eigen::MatrixXd alpha_plus = ground_alpha_density;
                    Eigen::MatrixXd alpha_minus = ground_alpha_density;
                    Eigen::MatrixXd beta_plus = ground_beta_density;
                    Eigen::MatrixXd beta_minus = ground_beta_density;

                    if (source == 0)
                    {
                        alpha_plus += step * delta_density;
                        alpha_minus -= step * delta_density;
                    }
                    else
                    {
                        beta_plus += step * delta_density;
                        beta_minus -= step * delta_density;
                    }

                    auto plus = evaluate_xc_matrix_from_spin_densities(
                        prepared,
                        alpha_plus,
                        beta_plus,
                        exchange_functional,
                        correlation_functional);
                    if (!plus)
                        return std::unexpected("TDDFT XC kernel (+) evaluation failed: " + plus.error());

                    auto minus = evaluate_xc_matrix_from_spin_densities(
                        prepared,
                        alpha_minus,
                        beta_minus,
                        exchange_functional,
                        correlation_functional);
                    if (!minus)
                        return std::unexpected("TDDFT XC kernel (-) evaluation failed: " + minus.error());

                    const Eigen::MatrixXd delta_v_alpha =
                        (plus->alpha - minus->alpha) / (2.0 * step);
                    const Eigen::MatrixXd delta_v_beta =
                        (plus->beta - minus->beta) / (2.0 * step);

                    const int source_column = source_space.flat_index(j, b);
                    for (int target = 0; target < nspaces; ++target)
                    {
                        const ResponseExcitationSpace &target_space = spaces[static_cast<std::size_t>(target)];
                        const Eigen::MatrixXd &delta_v = (target == 0) ? delta_v_alpha : delta_v_beta;
                        const Eigen::MatrixXd projected =
                            target_space.C_occ.transpose() * delta_v * target_space.C_virt;

                        for (int i = 0; i < target_space.n_occ; ++i)
                            for (int a = 0; a < target_space.n_virt; ++a)
                                blocks[static_cast<std::size_t>(target)][static_cast<std::size_t>(source)](
                                    target_space.flat_index(i, a),
                                    source_column) = projected(i, a);
                    }
                }
        }

        return blocks;
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string> build_closed_shell_xc_kernel_blocks(
        const PreparedSystem &prepared,
        const ResponseExcitationSpace &space,
        const Eigen::Ref<const Eigen::MatrixXd> &restricted_density,
        const DFT::XC::Functional &exchange_functional,
        const DFT::XC::Functional &correlation_functional)
    {
        const Eigen::MatrixXd ground_alpha = 0.5 * restricted_density;
        const Eigen::MatrixXd ground_beta = 0.5 * restricted_density;
        const std::vector<ResponseExcitationSpace> duplicated_spaces = {space, space};

        auto blocks = build_unrestricted_xc_kernel_blocks(
            prepared,
            duplicated_spaces,
            ground_alpha,
            ground_beta,
            exchange_functional,
            correlation_functional);
        if (!blocks)
            return std::unexpected(blocks.error());

        return std::make_pair(
            (*blocks)[0][0],
            (*blocks)[0][1]);
    }

    std::expected<XCGridEvaluation, std::string>
    evaluate_current_density_and_xc(
        const HartreeFock::Calculator &calculator,
        const PreparedSystem &prepared,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional,
        Eigen::Index slice_begin,
        Eigen::Index slice_end)
    {
        if (prepared.ao_grid.npoints() != prepared.molecular_grid.points.rows())
            return std::unexpected("AO grid and molecular grid point counts do not match");

        if (prepared.ao_grid.nbasis() != static_cast<Eigen::Index>(calculator._shells.nbasis()))
            return std::unexpected("AO grid basis dimension does not match the calculator basis");

        const Eigen::Index nbasis = prepared.ao_grid.nbasis();
        const auto &alpha_density = calculator._info._scf.alpha.density;
        if (alpha_density.rows() != nbasis || alpha_density.cols() != nbasis)
            return std::unexpected("alpha density matrix is not initialized for the current basis size");

        if (calculator._scf._scf == HartreeFock::SCFType::UHF)
        {
            const auto &beta_density = calculator._info._scf.beta.density;
            if (beta_density.rows() != nbasis || beta_density.cols() != nbasis)
                return std::unexpected("beta density matrix is not initialized for the current basis size");

            return evaluate_xc_on_grid(
                prepared.molecular_grid,
                prepared.ao_grid,
                alpha_density,
                beta_density,
                exchange_functional,
                correlation_functional,
                slice_begin,
                slice_end);
        }

        return evaluate_xc_on_grid(
            prepared.molecular_grid,
            prepared.ao_grid,
            alpha_density,
            exchange_functional,
            correlation_functional,
            slice_begin,
            slice_end);
    }

    namespace
    {

        std::expected<Eigen::MatrixXd, std::string> compute_analytic_ks_gradient(
            HartreeFock::Calculator &calculator,
            const PreparedSystem &prepared,
            const InitializedFunctionals &functionals)
        {
            if (prepared.pcm && prepared.pcm->enabled())
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "DFT Gradient :",
                    "Analytic KS gradient omits PCM geometry response; use numerical gradient if solvation coupling is required.");
            }

            auto xc_grid = evaluate_current_density_and_xc(
                calculator,
                prepared,
                functionals.exchange,
                functionals.correlation);
            if (!xc_grid)
                return std::unexpected("DFT analytic gradient: XC evaluation failed: " + xc_grid.error());

            const Eigen::Index npoints = prepared.molecular_grid.points.rows();
            const Eigen::Index nbasis = prepared.ao_grid.nbasis();

            DFT::AOGridHessian hess;
            if (xc_grid->vsigma.cols() > 0)
            {
                auto hess_res = DFT::evaluate_ao_hessian_on_grid(calculator._shells, prepared.molecular_grid);
                if (!hess_res)
                    return std::unexpected("DFT analytic gradient: AO Hessian failed: " + hess_res.error());
                hess = std::move(*hess_res);
            }
            else
            {
                hess.h_xx.resize(npoints, nbasis);
                hess.h_xy.resize(npoints, nbasis);
                hess.h_xz.resize(npoints, nbasis);
                hess.h_yy.resize(npoints, nbasis);
                hess.h_yz.resize(npoints, nbasis);
                hess.h_zz.resize(npoints, nbasis);
                hess.h_xx.setZero();
                hess.h_xy.setZero();
                hess.h_xz.setZero();
                hess.h_yy.setZero();
                hess.h_yz.setZero();
                hess.h_zz.setZero();
            }

            const HartreeFock::Gradient::ExchangeGradientKernel exchange_kernel{
                .full_range_exchange_coefficient = xc_grid->full_range_exchange_coefficient,
                .short_range_exchange_coefficient = xc_grid->short_range_exchange_coefficient,
                .range_separation_omega = xc_grid->range_separation_omega};

            // Assemble the KS gradient as HF-like derivative terms plus the XC
            // grid derivative. Range separation only changes the exchange-kernel
            // metadata passed into the HF-like piece.
            auto wf_grad =
                calculator._scf._scf == HartreeFock::SCFType::UHF
                    ? HartreeFock::Gradient::compute_uks_gradient(calculator, prepared.shell_pairs, exchange_kernel)
                    : HartreeFock::Gradient::compute_rks_gradient(calculator, prepared.shell_pairs, exchange_kernel);
            if (!wf_grad)
                return std::unexpected("DFT Coulomb/exchange gradient failed: " + wf_grad.error());

            if (dft_gradient_debug_enabled())
            {
                if (const auto &breakdown = HartreeFock::Gradient::last_wavefunction_gradient_breakdown();
                    breakdown)
                {
                    print_gradient_component_report("HF-core+Pulay", breakdown->core_pulay);
                    print_gradient_component_report("HF-2e-Coulomb", breakdown->coulomb_two_electron);
                    print_gradient_component_report("HF-2e-Exchange-Full", breakdown->exchange_full_range);
                    print_gradient_component_report("HF-2e-Exchange-LR", breakdown->exchange_long_range_correction);
                    print_gradient_component_report("HF-2e-Exchange", breakdown->exchange_two_electron);
                    print_gradient_component_report("HF-2e", breakdown->two_electron);
                    print_gradient_component_report("HF-nuclear", breakdown->nuclear_repulsion);
                }
            }

            auto xc_grad =
                calculator._scf._scf == HartreeFock::SCFType::UHF
                    ? DFT::Gradient::compute_xc_nuclear_gradient_uks(
                          calculator._molecule,
                          calculator._shells,
                          prepared.molecular_grid,
                          prepared.ao_grid,
                          hess,
                          *xc_grid,
                          calculator._info._scf.alpha.density,
                          calculator._info._scf.beta.density)
                    : DFT::Gradient::compute_xc_nuclear_gradient_rks(
                          calculator._molecule,
                          calculator._shells,
                          prepared.molecular_grid,
                          prepared.ao_grid,
                          hess,
                          *xc_grid,
                          calculator._info._scf.alpha.density);
            if (!xc_grad)
                return std::unexpected("DFT XC nuclear gradient failed: " + xc_grad.error());

            if (dft_gradient_debug_enabled())
            {
                print_gradient_component_report("HF-like", *wf_grad);
                print_gradient_component_report("XC-grid", *xc_grad);
            }

            calculator._gradient = *wf_grad + *xc_grad;
            if (dft_gradient_debug_enabled())
                print_gradient_component_report("KS-total", calculator._gradient);

            // Double-hybrid PT2 orbital-relaxation contribution (N3). RKS
            // only, behind PLANCK_DFT_DH_GRADIENT until the FD check (N3.4).
            // N3.2 wires the chain and reports diagnostics; N3.3 contracts
            // the relaxed density into the gradient.
            if (functionals.has_double_hybrid_pt2 &&
                calculator._scf._scf != HartreeFock::SCFType::UHF &&
                dft_allow_double_hybrid_gradient())
            {
                const double c_pt2 = functionals.perturbative_correlation_coefficient;

                if (calculator._mp2.use_ri)
                    return std::unexpected(
                        "DFT double-hybrid gradient: RI is not supported for the PT2 "
                        "orbital-relaxation path yet (needs an RI KS mean-field response). "
                        "Disable RI for double-hybrid gradients.");

                auto rmp2_res = HartreeFock::Correlation::rmp2_kernel(
                    calculator, prepared.shell_pairs, calculator._mp2);
                if (!rmp2_res)
                    return std::unexpected(
                        "DFT double-hybrid gradient: RMP2 kernel on KS orbitals failed: " +
                        rmp2_res.error());

                // N3.5.4: the KS mean-field response
                //   R_KS[d] = J[d] - 0.5 c_x K[d] + V_xc_response[d]
                // applied to a trial AO density -- replaces the HF J - 0.5 K
                // that build_rmp2_lagrangian uses by default. Same pieces
                // build_ks_orbital_hessian_op composes; PySCF gets the
                // equivalent from mp._scf.get_veff dispatching to KS.
                const std::size_t nb = calculator._shells.nbasis();
                const double c_fr_veff = xc_grid->full_range_exchange_coefficient;
                const double c_sr_veff = xc_grid->short_range_exchange_coefficient;
                const double omega_veff = xc_grid->range_separation_omega;
                const Eigen::MatrixXd ground_density_veff =
                    calculator._info._scf.alpha.density; // KS total density (RKS)
                HartreeFock::Correlation::KsVeffFn ks_veff =
                    [&, nb, c_fr_veff, c_sr_veff, omega_veff](
                        const Eigen::MatrixXd &d) -> Eigen::MatrixXd
                {
                    Eigen::MatrixXd v = _compute_2e_j_direct(
                        prepared.shell_pairs, d, nb, calculator._integral._engine,
                        HartreeFock::ERIKernel::Coulomb, 0.0, calculator._integral._tol_eri,
                        calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops
                                                          : nullptr);
                    if (c_fr_veff != 0.0)
                        v.noalias() -= 0.5 * c_fr_veff * _compute_2e_k_direct(
                            prepared.shell_pairs, d, nb, calculator._integral._engine,
                            HartreeFock::ERIKernel::Coulomb, 0.0, calculator._integral._tol_eri,
                            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops
                                                              : nullptr);
                    if (c_sr_veff != 0.0)
                        v.noalias() -= 0.5 * c_sr_veff * _compute_2e_k_direct(
                            prepared.shell_pairs, d, nb, calculator._integral._engine,
                            HartreeFock::ERIKernel::ShortRange, omega_veff,
                            calculator._integral._tol_eri,
                            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops
                                                              : nullptr);
                    // PLANCK_DFT_DH_VEFF_XC selects the XC piece:
                    //   linear (default) -- the f_xc RESPONSE, d V_xc/d rho . d
                    //   nonlinear        -- V_xc[d] evaluated on d itself
                    // N3.5.5 tested passing ks_veff to the two HF-defaulted
                    // sites and found it made FD worse, but it tested the
                    // LINEAR form only -- and its own comment records that
                    // PySCF's get_veff there is the full nonlinear V_xc[dm].
                    // So "ks_veff makes it worse" was never tested for the
                    // form PySCF actually uses. This makes both reachable.
                    const char *veff_xc_mode = std::getenv("PLANCK_DFT_DH_VEFF_XC");
                    const bool use_nonlinear_xc =
                        veff_xc_mode != nullptr &&
                        std::string_view(veff_xc_mode) == "nonlinear";
                    if (use_nonlinear_xc)
                    {
                        auto xc_d = DFT::evaluate_xc_on_grid(
                            prepared.molecular_grid, prepared.ao_grid, d,
                            functionals.exchange, functionals.correlation);
                        if (xc_d)
                        {
                            auto vxc_d = DFT::assemble_xc_matrix(
                                prepared.molecular_grid, prepared.ao_grid, *xc_d);
                            if (vxc_d)
                                v.noalias() += vxc_d->alpha;
                        }
                        return v;
                    }
                    auto dvxc = DFT::Driver::compute_analytic_xc_hessian_vector_product(
                        prepared.molecular_grid, prepared.ao_grid,
                        ground_density_veff, d,
                        functionals.exchange, functionals.correlation);
                    if (dvxc)
                        v.noalias() += *dvxc;
                    else
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Warning, "DFT DH Gradient :",
                            "KS-veff XC response failed; falling back to J - 0.5 c_x K "
                            "(gradient will be less accurate): " + dvxc.error());
                    return v;
                };

                auto lag = DFT::Gradient::build_pt2_mo_intermediates(
                    calculator, prepared.shell_pairs, *rmp2_res, c_pt2, ks_veff);
                if (!lag)
                    return std::unexpected(
                        "DFT double-hybrid gradient: PT2 MO intermediates failed: " +
                        lag.error());

                DFT::Gradient::PT2RelaxedDensityInputs rd_in;
                rd_in.lagrangian = &*lag;
                rd_in.result = &*rmp2_res;
                rd_in.calculator = &calculator;
                rd_in.shell_pairs = &prepared.shell_pairs;
                rd_in.molecular_grid = &prepared.molecular_grid;
                rd_in.ao_grid = &prepared.ao_grid;
                rd_in.density = calculator._info._scf.alpha.density; // total for RKS
                rd_in.x_functional = &functionals.exchange;
                rd_in.c_functional = &functionals.correlation;
                rd_in.engine = calculator._integral._engine;
                rd_in.tol_eri = calculator._integral._tol_eri;
                rd_in.sym_ops = calculator._use_integral_symmetry
                                    ? &calculator._integral_symmetry_ops
                                    : nullptr;
                rd_in.full_range_exchange_coefficient =
                    xc_grid->full_range_exchange_coefficient;
                rd_in.short_range_exchange_coefficient =
                    xc_grid->short_range_exchange_coefficient;
                rd_in.range_separation_omega = xc_grid->range_separation_omega;
                // N3.5.5 (reverted): passing ks_veff for vhf_s1occ made the
                // B2PLYP FD error worse (1.88e-4 -> 2.86e-4). PySCF's
                // grad/mp2.py `mp._scf.get_veff` there is a full nonlinear
                // V_xc[dm_small], not the linear f_xc response, and pyscf has
                // no validated DH gradient to cross-check. vhf_s1occ stays HF.
                // PLANCK_DFT_DH_VEFF_SITES bitmask: 1 = vhf_s1occ
                // (relaxed-density path), 2 = the presolved gradient path.
                // Default 0 = both HF, i.e. byte-identical to N3.5.5's revert.
                unsigned veff_sites = 0;
                if (const char *e = std::getenv("PLANCK_DFT_DH_VEFF_SITES"))
                    veff_sites = static_cast<unsigned>(std::strtoul(e, nullptr, 10));
                if (veff_sites & 1u)
                    rd_in.ks_veff = ks_veff;
                else
                    rd_in.ks_veff = {};

                auto rd = DFT::Gradient::solve_pt2_relaxed_density(rd_in);
                if (!rd)
                    return std::unexpected(
                        "DFT double-hybrid gradient: PT2 relaxed-density solve failed: " +
                        rd.error());

                const Eigen::MatrixXd &S = calculator._overlap;
                const double tr_PS = (rd->P_ao * S).trace();
                const double asym = (rd->P_ao - rd->P_ao.transpose()).cwiseAbs().maxCoeff();
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT DH Gradient :",
                    std::format(
                        "PT2 relaxed density: c_PT2={:.6f}, tr(P S)={:.6f}, max|P - P^T|={:.3e}",
                        c_pt2, tr_PS, asym));

                // N3.3: c_PT2 * dE_MP2/dR
                //   = build_rmp2_gradient_intermediates(KS, {scaled lag, KS z}).electronic_gradient
                //     - compute_rhf_gradient(KS orbitals).electronic
                // (the KS reference dE_KS/dR is already in calculator._gradient
                //  from compute_rks_gradient on the main path). See
                //  docs/DOUBLE_HYBRID_GRADIENT_SCOPE.md N3.3.
                // Reference-only electronic gradient: the same contraction
                // build_rmp2_gradient_intermediates does, with a zero Lagrangian
                // and zero Z-vector. This is exactly the reference part folded
                // into electronic_gradient (2J - K on hf_dm1 + core + Pulay, no
                // Vnn), so subtracting it isolates c_PT2 * dE_MP2/dR without
                // re-deriving Vnn or matching compute_rhf_gradient's grouping.
                HartreeFock::Correlation::RMP2Lagrangian zero_lag = *lag;
                zero_lag.doo.setZero();
                zero_lag.dvv.setZero();
                zero_lag.dm1_corr_mo.setZero();
                zero_lag.dm1_corr_ao.setZero();
                zero_lag.veff_corr_ao.setZero();
                zero_lag.imat_ao.setZero();
                zero_lag.imat_mo.setZero();
                zero_lag.Xvo.setZero();
                std::fill(zero_lag.dm2buf_full.begin(), zero_lag.dm2buf_full.end(), 0.0);
                const Eigen::MatrixXd zero_z =
                    Eigen::MatrixXd::Zero(lag->n_virt, lag->n_occ);

                auto reference_electronic =
                    [&]() -> std::expected<Eigen::MatrixXd, std::string>
                {
                    HartreeFock::Correlation::RMP2PreSolved ps;
                    ps.lagrangian = &zero_lag;
                    ps.z = &zero_z;
                    // (N3.5.5 reverted: KS veff for vhf_s1occ worsened FD)
                    auto r = HartreeFock::Correlation::build_rmp2_gradient_intermediates(
                        calculator, prepared.shell_pairs, *rmp2_res, ps);
                    if (!r)
                        return std::unexpected(r.error());
                    return r->electronic_gradient;
                };

                auto ref_grad = reference_electronic();
                if (!ref_grad)
                    return std::unexpected(
                        "DFT double-hybrid gradient: reference electronic gradient failed: " +
                        ref_grad.error());

                auto pt2_correction_gradient =
                    [&](const HartreeFock::Correlation::RMP2Lagrangian &scaled_lag,
                        const Eigen::MatrixXd &ks_z)
                    -> std::expected<Eigen::MatrixXd, std::string>
                {
                    HartreeFock::Correlation::RMP2PreSolved ps;
                    ps.lagrangian = &scaled_lag;
                    ps.z = &ks_z;
                    // N3.5.5 tried ps.ks_veff = ks_veff here -- made FD WORSE
                    // (1.88e-4 -> 2.86e-4) with the LINEAR response, reverted.
                    // Reachable again via PLANCK_DFT_DH_VEFF_SITES bit 2, so the
                    // nonlinear form can be tested; default off.
                    if (veff_sites & 2u)
                        ps.ks_veff = ks_veff;
                    auto full = HartreeFock::Correlation::build_rmp2_gradient_intermediates(
                        calculator, prepared.shell_pairs, *rmp2_res, ps);
                    if (!full)
                        return std::unexpected(full.error());
                    return Eigen::MatrixXd(full->electronic_gradient - *ref_grad);
                };

                auto corr = pt2_correction_gradient(*lag, rd->z);
                if (!corr)
                    return std::unexpected(
                        "DFT double-hybrid gradient: PT2 correction contraction failed: " +
                        corr.error());

                calculator._gradient += *corr;

                // PLANCK_DEBUG_DH_CHANNELS: dump the KS and *corr halves of the
                // DH gradient separately. Without this the two can only be
                // separated by inference, and inference is what produced a
                // wrong localisation once already (a pure-B3LYP self-FD does
                // NOT isolate the DH run's KS channel -- B2PLYP's hybrid is a
                // different functional). Planck's own FD of each half is
                // directly comparable to these.
                if (std::getenv("PLANCK_DEBUG_DH_CHANNELS"))
                {
                    const Eigen::MatrixXd ks_half = calculator._gradient - *corr;
                    for (Eigen::Index a = 0; a < corr->rows(); ++a)
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Info, "DH Channels :",
                            std::format("atom {} KS {: .10f} {: .10f} {: .10f}"
                                        "  corr {: .10f} {: .10f} {: .10f}",
                                        a + 1,
                                        ks_half(a, 0), ks_half(a, 1), ks_half(a, 2),
                                        (*corr)(a, 0), (*corr)(a, 1), (*corr)(a, 2)));
                }

                // N3.5.7 S5 (docs/DOUBLE_HYBRID_GRADIENT_KS_VEFF_SCOPE.md):
                // Eq. 33's XC contribution to the PT2 gradient. The scalar
                //   Phi_XC = sum_munu D_munu <mu|V_xc[rho_P]|nu>
                // and its FULL geometry derivative XC_I + XC_II + XC_III are
                // now derived and Python/PySCF-validated to rel 3e-9
                // (DFT::Gradient::compute_dh_xc_pt2_gradient, FD-gated by
                // planck-dft-kernel-gradient-fd against a true moving-grid FD).
                // But wiring it here STILL does not close the ~1.9e-4 Ha/Bohr
                // water/STO-3G B2PLYP FD residual:
                //   full I+II+III   -> 3.5e-3   (XC_I, the rho_D^(x) piece,
                //                                is ~85% and clearly wrong here
                //                                -- the paper's "not the naive
                //                                D^(x)" is literal)
                //   XC_II only      -> 2.65e-4  (worse than the 1.88e-4 baseline)
                //   XC_II * c_pt2   -> ~1.4e-4  (a max-norm coincidence --
                //                                N3.5.7.8 showed no single
                //                                scale factor works)
                // Eq. 33's Term1+Term2 IS XC_II (basis-only rho_P^(x), no
                // moving grid, no rho_D^(x)). XC_II is FD-verified as
                // sum_munu D_munu R^XC[rho_P^(x)]_munu with the SOSCF-validated
                // total-density v2rho2. N3.5.7.8 measured the decomposition
                // end-to-end and eliminated three candidates: XC_III is ~1e-7
                // on real water (not just synthetic He2); the closed-shell
                // spin factor is fine (Sec. II vs polarized R^XC, 1e-17); and
                // restoring XC_II's translational invariance with its
                // point-translation companion (kXcIIt) makes the residual
                // WORSE. XC_II carries one component almost exactly and is
                // structurally missing the z-directional piece; the open
                // suspect is the `full - ref_grad` PT2 isolation below, built
                // for HF-MP2.
                // N3.5.7.10: Eq. 33's XC contribution to the PT2 gradient
                // is the XC_II piece of d/dR{Phi_XC} -- identified against an
                // exact FD target on the C1 H2O2 fixture (cos 0.946 at
                // coefficient 1.078; removes 69% of the residual). XC_I is not
                // in the answer and XC_III is ~1e-7, so kXcII alone is wired.
                // PLANCK_DFT_DH_XC_PARTS overrides the mask for probing; it
                // takes the bitmask 1=XC_I, 2=XC_II, 4=XC_III, and
                // PLANCK_DFT_DH_XC_SCALE an optional prefactor.
                {
                    unsigned parts = DFT::Gradient::kXcII;
                    if (const char *e = std::getenv("PLANCK_DFT_DH_XC_PARTS"))
                        parts = static_cast<unsigned>(std::strtoul(e, nullptr, 10));
                    double scale = 1.0;
                    if (const char *e = std::getenv("PLANCK_DFT_DH_XC_SCALE"))
                        scale = std::strtod(e, nullptr);
                    auto dh_xc_grad = DFT::Gradient::compute_dh_xc_pt2_gradient(
                        calculator._molecule, calculator._shells,
                        prepared.molecular_grid, prepared.ao_grid, hess,
                        calculator._info._scf.alpha.density,
                        rd->dm1_corr_relaxed_ao,
                        functionals.exchange, functionals.correlation, parts);
                    if (!dh_xc_grad)
                        return std::unexpected(
                            "DFT double-hybrid gradient: Eq. 33 XC term failed: " +
                            dh_xc_grad.error());
                    calculator._gradient += scale * (*dh_xc_grad);
                }

                // N3.5.1: is compute_xc_nuclear_gradient_rks linear in its
                // density argument? If g(Pa+Pb) == g(Pa) + g(Pb), the missing
                // Z-vector XC term is just one more call with the relaxed
                // correction density (N3.5.2). Also flag whether the Becke
                // moving-weight term (w_atomic * dpartition * exc_density)
                // rides along -- it should be density-independent, so a nonzero
                // g(0) means it does and must be handled.
                if (std::getenv("PLANCK_DFT_DH_GRADIENT_SELFCHECK"))
                {
                    const Eigen::MatrixXd &Pa = calculator._info._scf.alpha.density;
                    const Eigen::MatrixXd Pb = 0.37 * Pa + 0.11 * rd->dm1_corr_relaxed_ao;
                    auto ga = DFT::Gradient::compute_xc_nuclear_gradient_rks(
                        calculator._molecule, calculator._shells, prepared.molecular_grid,
                        prepared.ao_grid, hess, *xc_grid, Pa);
                    auto gb = DFT::Gradient::compute_xc_nuclear_gradient_rks(
                        calculator._molecule, calculator._shells, prepared.molecular_grid,
                        prepared.ao_grid, hess, *xc_grid, Pb);
                    auto gab = DFT::Gradient::compute_xc_nuclear_gradient_rks(
                        calculator._molecule, calculator._shells, prepared.molecular_grid,
                        prepared.ao_grid, hess, *xc_grid,
                        Eigen::MatrixXd(Pa + Pb));
                    auto g0 = DFT::Gradient::compute_xc_nuclear_gradient_rks(
                        calculator._molecule, calculator._shells, prepared.molecular_grid,
                        prepared.ao_grid, hess, *xc_grid,
                        Eigen::MatrixXd(Eigen::MatrixXd::Zero(Pa.rows(), Pa.cols())));
                    if (ga && gb && gab && g0)
                    {
                        const double add_err =
                            (*gab - (*ga + *gb)).cwiseAbs().maxCoeff();
                        const double g0_norm = g0->cwiseAbs().maxCoeff();
                        HartreeFock::Logger::logging(
                            HartreeFock::LogLevel::Info,
                            "DFT DH Gradient :",
                            std::format(
                                "N3.5.1 XC-grad linearity: max|g(Pa+Pb) - g(Pa) - g(Pb)| = {:.3e}, "
                                "max|g(0)| = {:.3e} (nonzero => Becke exc_density term rides along)",
                                add_err, g0_norm));
                    }
                }
            }

            return calculator._gradient;
        }

    } // namespace

    std::expected<KSPotentialMatrices, std::string>
    assemble_current_ks_potential(
        HartreeFock::Calculator &calculator,
        PreparedSystem &prepared,
        const XCGridEvaluation &xc_grid,
        Eigen::Index xc_point_begin,
        Eigen::Index xc_point_end)
    {
        if (prepared.ao_grid.nbasis() != static_cast<Eigen::Index>(calculator._shells.nbasis()))
            return std::unexpected("AO grid basis dimension does not match the calculator basis");

        // No ERI tensor is built here any more: the J and K builds below are
        // memory-direct. calculator._eri stays in the Calculator because TDDFT
        // still needs the dense tensor for its AO->MO transform, but it is no
        // longer populated for the SCF loop.

        // MPI grid partition: assemble only this rank's XC point slice, then
        // reduce the nb^2 XC contribution HERE -- before J/K are built and
        // combined below. J/K are already MPI-reduced inside their own direct
        // builders, so reducing the whole KS potential later would double-count
        // them; reducing XC in isolation at the point of assembly avoids that.
        // xc_point_end < 0 (default) = whole grid, so the gradient/TDDFT callers
        // that pass no slice are byte-identical.
        auto xc_matrix = assemble_xc_matrix(
            prepared.molecular_grid,
            prepared.ao_grid,
            xc_grid,
            xc_point_begin,
            xc_point_end);
        if (!xc_matrix)
            return std::unexpected(xc_matrix.error());

        if (xc_point_end >= 0 && HartreeFock::Mpi::distributed())
        {
            // Disjoint point slices => MPI_SUM reassembles the full XC matrix.
            // Same reduce pattern as the Fock build (fused_fock.h), already
            // gated bitwise across ranks. Both spin channels always exist
            // (beta == alpha shape for RKS), so reduce both unconditionally.
            HartreeFock::Mpi::allreduce_inplace(
                xc_matrix->alpha.data(), static_cast<std::size_t>(xc_matrix->alpha.size()));
            HartreeFock::Mpi::allreduce_inplace(
                xc_matrix->beta.data(), static_cast<std::size_t>(xc_matrix->beta.size()));
        }

        const Eigen::Index nbasis = prepared.ao_grid.nbasis();
        const auto &alpha_density = calculator._info._scf.alpha.density;
        if (alpha_density.rows() != nbasis || alpha_density.cols() != nbasis)
            return std::unexpected("alpha density matrix is not initialized for KS matrix assembly");

        Eigen::MatrixXd total_density = alpha_density;
        Eigen::MatrixXd beta_density;
        if (calculator._scf._scf == HartreeFock::SCFType::UHF)
        {
            beta_density = calculator._info._scf.beta.density;
            if (beta_density.rows() != nbasis || beta_density.cols() != nbasis)
                return std::unexpected("beta density matrix is not initialized for KS matrix assembly");
            total_density += beta_density;
        }

        // Memory-direct Coulomb: contract each canonical quartet straight into
        // J instead of sweeping the nb^4 tensor. Same loop the HF Fock build
        // uses, so this inherits block-level Schwarz, the fixed-order OpenMP
        // reduction, the MPI bra-stripe, and native sym_ops handling.
        //
        // Raw J — no coefficient. See the prefactor contract in
        // fock_accumulate.h.
        const Eigen::MatrixXd coulomb = _compute_2e_j_direct(
            prepared.shell_pairs,
            total_density,
            calculator._shells.nbasis(),
            calculator._integral._engine,
            HartreeFock::ERIKernel::Coulomb,
            0.0,
            calculator._integral._tol_eri,
            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);

        const double full_range_exchange_coefficient = xc_grid.full_range_exchange_coefficient;
        const double short_range_exchange_coefficient = xc_grid.short_range_exchange_coefficient;
        const double exact_exchange_coefficient =
            full_range_exchange_coefficient + short_range_exchange_coefficient;
        Eigen::MatrixXd exact_exchange_alpha;
        Eigen::MatrixXd exact_exchange_beta;
        double exact_exchange_energy = 0.0;

        // Range-separated functionals no longer build a SECOND nb^4 tensor at
        // the screened omega either — the short-range K calls below pass the
        // omega straight to the fused loop.

        if (std::abs(exact_exchange_coefficient) > 1.0e-14)
        {
            if (calculator._scf._scf == HartreeFock::SCFType::UHF)
            {
                Eigen::MatrixXd exchange_alpha = Eigen::MatrixXd::Zero(nbasis, nbasis);
                Eigen::MatrixXd exchange_beta = Eigen::MatrixXd::Zero(nbasis, nbasis);

                // Both spin channels from ONE quartet sweep per kernel — the UHF
                // entry fills K_alpha from Pa and K_beta from Pb together, so a
                // UKS hybrid does not pay the traversal twice.
                if (std::abs(full_range_exchange_coefficient) > 1.0e-14)
                {
                    const auto [Ka, Kb] = _compute_2e_k_uhf_direct(
                        prepared.shell_pairs, alpha_density, beta_density,
                        calculator._shells.nbasis(), calculator._integral._engine,
                        HartreeFock::ERIKernel::Coulomb, 0.0,
                        calculator._integral._tol_eri,
                        calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
                    exchange_alpha.noalias() += full_range_exchange_coefficient * Ka;
                    exchange_beta.noalias() += full_range_exchange_coefficient * Kb;
                }

                if (std::abs(short_range_exchange_coefficient) > 1.0e-14)
                {
                    const auto [Ka, Kb] = _compute_2e_k_uhf_direct(
                        prepared.shell_pairs, alpha_density, beta_density,
                        calculator._shells.nbasis(), calculator._integral._engine,
                        HartreeFock::ERIKernel::ShortRange,
                        xc_grid.range_separation_omega,
                        calculator._integral._tol_eri,
                        calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
                    exchange_alpha.noalias() += short_range_exchange_coefficient * Ka;
                    exchange_beta.noalias() += short_range_exchange_coefficient * Kb;
                }

                exact_exchange_alpha = -exchange_alpha;
                exact_exchange_beta = -exchange_beta;
                exact_exchange_energy =
                    -0.5 *
                    (density_trace_product(alpha_density, exchange_alpha) +
                     density_trace_product(beta_density, exchange_beta));
            }
            else
            {
                Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(nbasis, nbasis);
                if (std::abs(full_range_exchange_coefficient) > 1.0e-14)
                {
                    exchange.noalias() +=
                        full_range_exchange_coefficient *
                        _compute_2e_k_direct(
                            prepared.shell_pairs, alpha_density,
                            calculator._shells.nbasis(), calculator._integral._engine,
                            HartreeFock::ERIKernel::Coulomb, 0.0,
                            calculator._integral._tol_eri,
                            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
                }
                if (std::abs(short_range_exchange_coefficient) > 1.0e-14)
                {
                    exchange.noalias() +=
                        short_range_exchange_coefficient *
                        _compute_2e_k_direct(
                            prepared.shell_pairs, alpha_density,
                            calculator._shells.nbasis(), calculator._integral._engine,
                            HartreeFock::ERIKernel::ShortRange,
                            xc_grid.range_separation_omega,
                            calculator._integral._tol_eri,
                            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
                }

                exact_exchange_alpha = -0.5 * exchange;
                exact_exchange_beta = exact_exchange_alpha;
                exact_exchange_energy =
                    -0.25 *
                    density_trace_product(alpha_density, exchange);
            }
        }

        return combine_ks_potential(
            coulomb,
            *xc_matrix,
            exact_exchange_coefficient,
            exact_exchange_alpha,
            exact_exchange_beta,
            exact_exchange_energy);
    }

    std::expected<PreparedSystem, std::string>
    prepare(HartreeFock::Calculator &calculator, const Options &options)
    {
        // Central DFT rebuild path for a given geometry/orientation.
        // Everything that depends on nuclear positions is refreshed here before
        // any KS iterations begin.
        const GridLevel grid_level = to_grid_level(calculator._dft._grid);
        calculator.prepare_coordinates();
        calculator._eri.clear();
        reset_sao_state(calculator);

        auto preserve_checkpoint_ao_frame = restore_geometry_for_full_restart(calculator);
        if (!preserve_checkpoint_ao_frame)
            return std::unexpected("DFT checkpoint geometry restore failed: " + preserve_checkpoint_ao_frame.error());

        if (auto res = setup_symmetry(calculator, options, *preserve_checkpoint_ao_frame); !res)
            return std::unexpected(res.error());

        if (auto res = read_basis_and_initialize(calculator); !res)
            return std::unexpected(res.error());

        PreparedSystem prepared;
        auto preset = grid_preset(grid_level);
        if (!preset)
            return std::unexpected(preset.error());
        prepared.grid_preset = *preset;
        prepared.shell_pairs = build_shellpairs(calculator._shells);

        RestartState restart_state;
        if (wants_checkpoint_restart(calculator))
        {
            auto restart_loaded = load_ks_restart_state(calculator, prepared.shell_pairs);
            if (!restart_loaded)
                return std::unexpected("DFT checkpoint restart failed: " + restart_loaded.error());
            restart_state = *restart_loaded;
        }

        if (!restart_state.one_e_ready)
        {
            if (auto res = compute_one_electron_terms(calculator, prepared.shell_pairs); !res)
                return std::unexpected(res.error());
        }

        maybe_build_sao_basis(calculator, options);

        auto molecular_grid = MakeMolecularGrid(calculator._molecule, grid_level);
        if (!molecular_grid)
            return std::unexpected("DFT molecular grid construction failed: " + molecular_grid.error());
        prepared.molecular_grid = std::move(*molecular_grid);

        auto ao_grid = evaluate_ao_basis_on_grid(calculator._shells, prepared.molecular_grid);
        if (!ao_grid)
            return std::unexpected("DFT AO grid evaluation failed: " + ao_grid.error());
        prepared.ao_grid = std::move(*ao_grid);

        if (calculator._solvation._model != HartreeFock::SolvationModel::None)
        {
            auto pcm = HartreeFock::Solvation::build_pcm_state(calculator, prepared.shell_pairs);
            if (!pcm)
                return std::unexpected("DFT PCM setup failed: " + pcm.error());
            prepared.pcm = std::move(*pcm);
        }

        const bool restart_loaded = restart_state.density_loaded;
        if (!restart_loaded)
        {
            if (auto res = initialize_ks_guess(calculator); !res)
                return std::unexpected(res.error());
        }

        if (calculator._dft._print_grid_summary && options.print_grid_summary)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "DFT Grid :",
                std::format(
                    "{} points, {} basis functions, {} shell pairs",
                    prepared.molecular_grid.points.rows(),
                    prepared.ao_grid.nbasis(),
                    prepared.shell_pairs.size()));
        }

        return prepared;
    }

    std::expected<Result, std::string>
    run(HartreeFock::Calculator &calculator, const Options &options)
    {
        // Spherical-harmonic basis is not yet wired through the DFT path (KS matrix,
        // grid AO evaluation, exact-exchange assembly all assume Cartesian AO counts).
        // Reject at the entry point — before functional resolution or any basis load —
        // rather than risk a silent wrong answer. The HF driver supports spherical
        // single-point RHF/UHF energies.
        if (calculator._basis._basis == HartreeFock::BasisType::Spherical)
            return std::unexpected(
                "Spherical basis (basis_type spherical) is not supported by planck-dft; "
                "use basis_type cartesian, or run spherical single points through hartree-fock.");

        if (calculator._scf._scf == HartreeFock::SCFType::ROHF)
            return std::unexpected("ROKS/ROHF DFT references are not implemented; use UKS for open-shell DFT");

        auto functionals = initialize_functionals(calculator);
        if (!functionals)
            return std::unexpected(functionals.error());

        if (auto workflow_support = validate_workflow_support(calculator, *functionals); !workflow_support)
            return std::unexpected(workflow_support.error());

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "Libxc :",
            std::format(
                "Using {} with {} + {}",
                DFT::XC::version_string(),
                functionals->exchange.name(),
                functionals->correlation.name()));

        // All higher-level DFT workflows start from the same converged KS
        // reference.  Gradient, optimization, and frequency branches then build
        // their derivative-specific machinery on top of that shared entry step.
        switch (calculator._calculation)
        {
        case HartreeFock::CalculationType::SinglePoint:
            return run_initial_single_point(calculator, options, *functionals);

        case HartreeFock::CalculationType::LinearResponse:
        {
            auto result = run_initial_single_point(calculator, options, *functionals);
            if (!result)
                return std::unexpected(result.error());
            if (!result->converged)
                return *result;

            auto response = run_linear_response(
                calculator,
                options,
                functionals->exchange,
                functionals->correlation);
            if (!response)
                return std::unexpected(response.error());
            return *result;
        }

        case HartreeFock::CalculationType::Gradient:
        {
            const Eigen::MatrixXd requested_gradient_frame_bohr =
                calculator._molecule._coordinates;
            auto result = run_initial_single_point(calculator, options, *functionals);
            if (!result)
                return std::unexpected(result.error());
            if (!result->converged)
                return *result;

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Gradient :",
                "Computing analytic nuclear gradient (Kohn-Sham + grid XC)");

            auto prepared_grad = prepare_quadrature_for_calculator(calculator);
            if (!prepared_grad)
                return std::unexpected("DFT analytic gradient quadrature preparation failed: " +
                                       prepared_grad.error());

            auto gradient = compute_analytic_ks_gradient(calculator, *prepared_grad, *functionals);
            if (!gradient)
                return std::unexpected("DFT analytic gradient failed: " + gradient.error());

            calculator._gradient = rotate_gradient_to_requested_frame_if_needed(
                calculator,
                *gradient,
                requested_gradient_frame_bohr);
            print_gradient_report(calculator._gradient);
            return *result;
        }

        case HartreeFock::CalculationType::GeomOpt:
        {
            auto result = run_initial_single_point(calculator, options, *functionals);
            if (!result)
                return std::unexpected(result.error());
            if (!result->converged)
                return *result;

            return run_geometry_optimization(calculator, *functionals);
        }

        case HartreeFock::CalculationType::Frequency:
        {
            auto result = run_initial_single_point(calculator, options, *functionals);
            if (!result)
                return std::unexpected(result.error());
            if (!result->converged)
                return *result;

            auto frequency = run_frequency_analysis(calculator, *functionals);
            if (!frequency)
                return std::unexpected("DFT frequency analysis failed: " + frequency.error());
            return *result;
        }

        case HartreeFock::CalculationType::GeomOptFrequency:
        {
            auto result = run_initial_single_point(calculator, options, *functionals);
            if (!result)
                return std::unexpected(result.error());
            if (!result->converged)
                return *result;

            auto geomopt = run_geometry_optimization(calculator, *functionals);
            if (!geomopt)
                return std::unexpected(geomopt.error());

            auto frequency = run_frequency_analysis(calculator, *functionals);
            if (!frequency)
                return std::unexpected("DFT frequency analysis failed: " + frequency.error());
            return *geomopt;
        }

        case HartreeFock::CalculationType::ImaginaryFollow:
            return std::unexpected("DFT imaginary-mode following is not implemented yet");
        }

        return std::unexpected("Unsupported DFT calculation type");
    }

    // ── CLI entry (peer of HartreeFock::Driver::run) ──────────────────────────
    namespace
    {
        using CliSystemClock = std::chrono::system_clock;

        std::string cli_format_time(CliSystemClock::time_point tp)
        {
            const std::time_t t = CliSystemClock::to_time_t(tp);
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

        void log_multipole_report(HartreeFock::Calculator &calculator)
        {
            auto shell_pairs = build_shellpairs(calculator._shells);
            auto moments = HartreeFock::ObaraSaika::_compute_multipole_moments(
                calculator, shell_pairs, Eigen::Vector3d::Zero());
            if (!moments)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning, "Multipole Moments :",
                    "Unavailable: " + moments.error());
                HartreeFock::Logger::blank();
                return;
            }
            calculator._multipole = *moments; // cache for the JSON results dump
            calculator._have_multipole = true;
            HartreeFock::Logger::multipole_moments(*moments);
            HartreeFock::Logger::blank();
        }
    } // namespace

    std::expected<int, std::string> run(
        HartreeFock::Calculator &calculator,
        [[maybe_unused]] const std::string &input_file,
        const std::string &json_path)
    {
        const auto program_start = CliSystemClock::now();

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
        if (calculator._solvation._model != HartreeFock::SolvationModel::None)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, "Solvation :",
                std::format("PCM (epsilon = {:.4f}, points/atom = {})",
                            calculator._solvation._dielectric,
                            calculator._solvation._surface_points_per_atom));
        }
        HartreeFock::Logger::blank();

        const auto result = run(calculator);
        if (!result)
            return std::unexpected(result.error());

        if (result->converged)
            log_multipole_report(calculator);

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "DFT Energy :",
            std::format("{:.10f} Eh", result->total_energy));
        if (calculator._solvation._model != HartreeFock::SolvationModel::None)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info, "PCM Solvation Energy :",
                std::format("{:.10f} Eh", result->solvation_energy));
        }
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "Converged :",
            result->converged ? "true" : "false");
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "Wall Time :",
            std::format("{} ({} seconds)",
                        cli_format_time(CliSystemClock::now()),
                        std::chrono::duration<double>(CliSystemClock::now() - program_start).count()));

        if (!json_path.empty())
        {
            // The DFT total lives in the driver Result, not on the Calculator;
            // copy it in so the shared serializer reports the log's energy.
            calculator._total_energy = result->total_energy;
            if (auto res = HartreeFock::IO::dump_results_json(calculator, json_path); !res)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Error, "JSON Output Failed :", res.error());
                return EXIT_FAILURE;
            }
        }

        return EXIT_SUCCESS;
    }

} // namespace DFT::Driver
