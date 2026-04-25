#include "driver.h"

#include <Eigen/QR>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <numeric>
#include <string>

#include "base/wrapper.h"
#include "basis/basis.h"
#include "freq/hessian.h"
#include "integrals/base.h"
#include "integrals/os.h"
#include "io/checkpoint.h"
#include "io/logging.h"
#include "opt/geomopt.h"
#include "populations/multipole.h"
#include "post_hf/integrals.h"
#include "scf/scf.h"
#include "symmetry/integral_symmetry.h"
#include "symmetry/mo_symmetry.h"
#include "symmetry/symmetry.h"

namespace DFT::Driver
{

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

        struct LinearResponseContribution
        {
            int occupied = 0;
            int virtual_orbital = 0;
            double weight = 0.0;
            std::string occupied_symmetry;
            std::string virtual_symmetry;
        };

        struct LinearResponseRoot
        {
            int root = 0;
            double excitation_energy = 0.0;
            double excitation_energy_ev = 0.0;
            Eigen::Vector3d transition_dipole = Eigen::Vector3d::Zero();
            double oscillator_strength = 0.0;
            std::vector<LinearResponseContribution> dominant_contributions;
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
                functional_name = "lda_c_vwn_5";
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
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Symmetry :",
                    "Skipped; using input orientation");
                return {};
            }

            if (auto res = HartreeFock::Symmetry::detectSymmetry(calculator._molecule); !res)
                return std::unexpected("DFT symmetry detection failed: " + res.error());

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

        Eigen::MatrixXd build_coulomb_from_eri(
            const std::vector<double> &eri,
            const Eigen::Ref<const Eigen::MatrixXd> &density,
            std::size_t nbasis)
        {
            const std::size_t nb = nbasis;
            const std::size_t nb2 = nb * nb;
            const std::size_t nb3 = nb * nb * nb;

            Eigen::MatrixXd coulomb = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(nb),
                static_cast<Eigen::Index>(nb));

            for (std::size_t mu = 0; mu < nb; ++mu)
                for (std::size_t nu = 0; nu < nb; ++nu)
                    for (std::size_t lam = 0; lam < nb; ++lam)
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            coulomb(mu, nu) += density(lam, sig) * eri[mu * nb3 + nu * nb2 + lam * nb + sig];

            return coulomb;
        }

        Eigen::MatrixXd build_exchange_from_eri(
            const std::vector<double> &eri,
            const Eigen::Ref<const Eigen::MatrixXd> &density,
            std::size_t nbasis)
        {
            const std::size_t nb = nbasis;
            const std::size_t nb2 = nb * nb;
            const std::size_t nb3 = nb * nb * nb;

            Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(nb),
                static_cast<Eigen::Index>(nb));

            for (std::size_t mu = 0; mu < nb; ++mu)
                for (std::size_t nu = 0; nu < nb; ++nu)
                    for (std::size_t lam = 0; lam < nb; ++lam)
                        for (std::size_t sig = 0; sig < nb; ++sig)
                            exchange(mu, nu) += density(lam, sig) * eri[mu * nb3 + lam * nb2 + nu * nb + sig];

            return exchange;
        }

        double density_trace_product(
            const Eigen::Ref<const Eigen::MatrixXd> &density,
            const Eigen::Ref<const Eigen::MatrixXd> &matrix)
        {
            return (density.array() * matrix.array()).sum();
        }

        std::expected<void, std::string> ensure_eri_tensor(
            HartreeFock::Calculator &calculator,
            const PreparedSystem &prepared)
        {
            const std::size_t nbasis = calculator._shells.nbasis();
            const std::size_t expected_size = nbasis * nbasis * nbasis * nbasis;
            if (calculator._eri.size() == expected_size)
                return {};

            try
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT 2e Integrals :",
                    std::format("Building ERI tensor for KS Coulomb term ({:.1f} MB)",
                                expected_size * 8.0 / 1e6));
                calculator._eri = _compute_2e(
                    prepared.shell_pairs,
                    nbasis,
                    calculator._integral._engine,
                    calculator._integral._tol_eri,
                    calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
            }
            catch (const std::exception &e)
            {
                return std::unexpected("Failed to build ERI tensor for KS Coulomb term: " + std::string(e.what()));
            }

            return {};
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

        void print_linear_response_report(
            const std::vector<LinearResponseRoot> &roots,
            double exact_exchange_coefficient)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "TDDFT / Linear Response :",
                std::format(
                    "TDA-style singlet solver with {:.6f} exact-exchange kernel coefficient",
                    exact_exchange_coefficient));
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "TDDFT / Linear Response :",
                "Semilocal XC response kernels are not included in this initial implementation");
            HartreeFock::Logger::blank();

            std::cout << std::string(118, '-') << "\n"
                      << std::setw(6) << "Root"
                      << std::setw(16) << "Omega (Eh)"
                      << std::setw(14) << "Omega (eV)"
                      << std::setw(14) << "f"
                      << std::setw(16) << "mu_x (au)"
                      << std::setw(16) << "mu_y (au)"
                      << std::setw(16) << "mu_z (au)"
                      << std::setw(20) << "|mu| (Debye)"
                      << "\n"
                      << std::string(118, '-') << "\n";

            for (const LinearResponseRoot &root : roots)
            {
                const double mu_norm_debye = root.transition_dipole.norm() * AU_TO_DEBYE;
                std::cout << std::setw(6) << root.root
                          << std::setw(16) << std::fixed << std::setprecision(8) << root.excitation_energy
                          << std::setw(14) << std::fixed << std::setprecision(6) << root.excitation_energy_ev
                          << std::setw(14) << std::fixed << std::setprecision(6) << root.oscillator_strength
                          << std::setw(16) << std::fixed << std::setprecision(6) << root.transition_dipole.x()
                          << std::setw(16) << std::fixed << std::setprecision(6) << root.transition_dipole.y()
                          << std::setw(16) << std::fixed << std::setprecision(6) << root.transition_dipole.z()
                          << std::setw(20) << std::fixed << std::setprecision(6) << mu_norm_debye
                          << "\n";
            }

            std::cout << std::string(118, '-') << "\n";

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
                            ? std::format("MO{}", contribution.occupied)
                            : std::format("MO{} ({})", contribution.occupied, contribution.occupied_symmetry);
                    const std::string virtual_label =
                        contribution.virtual_symmetry.empty()
                            ? std::format("MO{}", contribution.virtual_orbital)
                            : std::format("MO{} ({})", contribution.virtual_orbital, contribution.virtual_symmetry);

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

        std::expected<std::vector<LinearResponseRoot>, std::string> run_linear_response_tda(
            HartreeFock::Calculator &calculator,
            const DFT::XC::Functional &exchange_functional)
        {
            if (!calculator._info._is_converged)
                return std::unexpected("TDDFT / linear response requires a converged RKS reference");
            if (calculator._scf._scf != HartreeFock::SCFType::RHF || calculator._info._scf.is_uhf)
                return std::unexpected("Initial TDDFT / linear response support currently requires a closed-shell RKS reference");

            const int n_electrons = static_cast<int>(
                calculator._molecule.atomic_numbers.cast<int>().sum() - calculator._molecule.charge);
            if (n_electrons % 2 != 0)
                return std::unexpected("Closed-shell RKS TDDFT / linear response requires an even number of electrons");

            const Eigen::Index nbasis = static_cast<Eigen::Index>(calculator._shells.nbasis());
            const int n_occ = n_electrons / 2;
            const int n_virt = static_cast<int>(nbasis) - n_occ;
            if (n_occ <= 0 || n_virt <= 0)
                return std::unexpected("TDDFT / linear response requires at least one occupied and one virtual orbital");

            const int nov = n_occ * n_virt;
            const int nroots = std::min(std::max(calculator._dft._lr_nstates, 1), nov);

            const Eigen::MatrixXd &coefficients = calculator._info._scf.alpha.mo_coefficients;
            const Eigen::VectorXd &energies = calculator._info._scf.alpha.mo_energies;
            const std::vector<std::string> &mo_symmetry = calculator._info._scf.alpha.mo_symmetry;

            const Eigen::MatrixXd C_occ = coefficients.leftCols(n_occ);
            const Eigen::MatrixXd C_virt = coefficients.middleCols(n_occ, n_virt);

            std::vector<double> eri_local;
            const auto shell_pairs = build_shellpairs(calculator._shells);
            const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
                calculator,
                shell_pairs,
                eri_local,
                "TDDFT / Linear Response :");

            const std::vector<double> mo_ia_jb =
                HartreeFock::Correlation::transform_eri(eri, static_cast<std::size_t>(nbasis), C_occ, C_virt, C_occ, C_virt);
            const std::vector<double> mo_ij_ab =
                HartreeFock::Correlation::transform_eri(eri, static_cast<std::size_t>(nbasis), C_occ, C_occ, C_virt, C_virt);

            auto idx_ia = [n_virt](int i, int a) -> int
            {
                return i * n_virt + a;
            };
            auto idx_ia_jb = [n_occ, n_virt](int i, int a, int j, int b) -> std::size_t
            {
                return ((static_cast<std::size_t>(i) * n_virt + a) * n_occ + j) * n_virt + b;
            };
            auto idx_ij_ab = [n_occ, n_virt](int i, int j, int a, int b) -> std::size_t
            {
                return ((static_cast<std::size_t>(i) * n_occ + j) * n_virt + a) * n_virt + b;
            };

            const double exact_exchange_coefficient =
                exchange_functional.is_hybrid() ? exchange_functional.exact_exchange_coefficient() : 0.0;

            Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(nov);
            for (int i = 0; i < n_occ; ++i)
                for (int a = 0; a < n_virt; ++a)
                {
                    const int ia = idx_ia(i, a);
                    diagonal(ia) =
                        energies(n_occ + a) - energies(i) +
                        2.0 * mo_ia_jb[idx_ia_jb(i, a, i, a)] -
                        exact_exchange_coefficient * mo_ij_ab[idx_ij_ab(i, i, a, a)];
                }

            const auto apply_tda_matrix = [&](const Eigen::Ref<const Eigen::VectorXd> &trial) -> Eigen::VectorXd
            {
                Eigen::VectorXd sigma = Eigen::VectorXd::Zero(nov);
                for (int i = 0; i < n_occ; ++i)
                    for (int a = 0; a < n_virt; ++a)
                    {
                        const int ia = idx_ia(i, a);
                        double value = (energies(n_occ + a) - energies(i)) * trial(ia);
                        for (int j = 0; j < n_occ; ++j)
                            for (int b = 0; b < n_virt; ++b)
                            {
                                const int jb = idx_ia(j, b);
                                value += 2.0 * mo_ia_jb[idx_ia_jb(i, a, j, b)] * trial(jb);
                                value -= exact_exchange_coefficient * mo_ij_ab[idx_ij_ab(i, j, a, b)] * trial(jb);
                            }
                        sigma(ia) = value;
                    }
                return sigma;
            };

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "TDDFT / Davidson :",
                std::format("Solving {} lowest roots in a {}-dimensional excitation space", nroots, nov));

            auto davidson = davidson_lowest_eigenpairs(
                nov,
                nroots,
                diagonal,
                apply_tda_matrix,
                1.0e-8,
                100,
                std::max(6 * nroots, 24));
            if (!davidson)
                return std::unexpected(davidson.error());

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "TDDFT / Davidson :",
                std::format(
                    "{} after {} iterations (max residual {:.3e})",
                    davidson->converged ? "Converged" : "Stopped",
                    davidson->iterations,
                    davidson->residual_norms.size() > 0 ? davidson->residual_norms.maxCoeff() : 0.0));
            HartreeFock::Logger::blank();

            const HartreeFock::MultipoleMatrices multipoles =
                HartreeFock::ObaraSaika::_compute_multipole_matrices(
                    shell_pairs,
                    static_cast<std::size_t>(nbasis),
                    Eigen::Vector3d::Zero());

            std::array<Eigen::MatrixXd, 3> mo_dipole_ov = {
                C_occ.transpose() * multipoles.dipole[0] * C_virt,
                C_occ.transpose() * multipoles.dipole[1] * C_virt,
                C_occ.transpose() * multipoles.dipole[2] * C_virt};

            std::vector<LinearResponseRoot> roots;
            roots.reserve(static_cast<std::size_t>(nroots));

            for (int root = 0; root < nroots; ++root)
            {
                const Eigen::VectorXd amplitudes = davidson->eigenvectors.col(root);
                Eigen::Vector3d transition_dipole = Eigen::Vector3d::Zero();

                for (int i = 0; i < n_occ; ++i)
                    for (int a = 0; a < n_virt; ++a)
                    {
                        const double amplitude = amplitudes(idx_ia(i, a));
                        transition_dipole.x() += std::sqrt(2.0) * amplitude * mo_dipole_ov[0](i, a);
                        transition_dipole.y() += std::sqrt(2.0) * amplitude * mo_dipole_ov[1](i, a);
                        transition_dipole.z() += std::sqrt(2.0) * amplitude * mo_dipole_ov[2](i, a);
                    }

                std::vector<LinearResponseContribution> contributions;
                contributions.reserve(static_cast<std::size_t>(nov));
                for (int i = 0; i < n_occ; ++i)
                    for (int a = 0; a < n_virt; ++a)
                    {
                        const double amplitude = amplitudes(idx_ia(i, a));
                        contributions.push_back(
                            LinearResponseContribution{
                                .occupied = i + 1,
                                .virtual_orbital = n_occ + a + 1,
                                .weight = amplitude * amplitude,
                                .occupied_symmetry =
                                    (static_cast<std::size_t>(i) < mo_symmetry.size()) ? mo_symmetry[static_cast<std::size_t>(i)] : "",
                                .virtual_symmetry =
                                    (static_cast<std::size_t>(n_occ + a) < mo_symmetry.size()) ? mo_symmetry[static_cast<std::size_t>(n_occ + a)] : ""});
                    }

                std::ranges::sort(
                    contributions,
                    [](const LinearResponseContribution &lhs, const LinearResponseContribution &rhs)
                    {
                        return lhs.weight > rhs.weight;
                    });
                if (contributions.size() > 3)
                    contributions.resize(3);

                const double excitation_energy = davidson->eigenvalues(root);
                const double oscillator_strength =
                    std::max(0.0, (2.0 / 3.0) * excitation_energy * transition_dipole.squaredNorm());

                roots.push_back(
                    LinearResponseRoot{
                        .root = root + 1,
                        .excitation_energy = excitation_energy,
                        .excitation_energy_ev = excitation_energy * HARTREE_TO_EV,
                        .transition_dipole = transition_dipole,
                        .oscillator_strength = oscillator_strength,
                        .dominant_contributions = std::move(contributions)});
            }

            print_linear_response_report(roots, exact_exchange_coefficient);
            return roots;
        }

        std::expected<Result, std::string> run_ks_scf_scaffold(
            HartreeFock::Calculator &calculator,
            const PreparedSystem &prepared,
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

                for (unsigned int iter = 1; iter <= max_iter; ++iter)
                {
                    const auto iter_start = std::chrono::steady_clock::now();
                    calculator._info._scf.alpha.density = density;

                    // DFT replaces the HF Coulomb/exchange build with a grid
                    // loop that evaluates the current density, queries libxc
                    // for the semilocal derivatives, and assembles the AO-space
                    // KS potential for the present density matrix.
                    auto xc_grid = evaluate_current_density_and_xc(
                        calculator,
                        prepared,
                        x_functional,
                        c_functional);
                    if (!xc_grid)
                        return std::unexpected("DFT density/XC evaluation failed: " + xc_grid.error());

                    auto ks_potential = assemble_current_ks_potential(calculator, prepared, *xc_grid);
                    if (!ks_potential)
                        return std::unexpected("DFT KS potential assembly failed: " + ks_potential.error());

                    const Eigen::MatrixXd fock = calculator._hcore + ks_potential->alpha;
                    const double electronic_energy =
                        (density.array() * calculator._hcore.array()).sum() +
                        0.5 * (density.array() * ks_potential->coulomb.array()).sum() +
                        xc_grid->total_energy +
                        ks_potential->exact_exchange_energy;
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

                    const Eigen::MatrixXd fock_for_diagonalization =
                        (use_diis && diis.ready()) ? diis.extrapolate() : fock;
                    auto diagonalization = diagonalize_in_ao_basis(
                        calculator,
                        X,
                        fock_for_diagonalization,
                        "KS");
                    if (!diagonalization)
                        return std::unexpected(diagonalization.error());

                    const Eigen::MatrixXd next_density =
                        density_from_orbitals(diagonalization->coefficients, n_occ, 2.0);
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

                    HartreeFock::SCF::store_restricted_iteration(
                        calculator,
                        HartreeFock::SCF::RestrictedIterationData{
                            .density = density,
                            .fock = fock,
                            .mo_energies = diagonalization->energies,
                            .mo_coefficients = diagonalization->coefficients,
                            .electronic_energy = electronic_energy,
                            .total_energy = total_energy},
                        metrics);
                    calculator._info._scf.alpha.mo_symmetry = diagonalization->mo_symmetry;

                    result.total_energy = total_energy;
                    result.xc_energy = xc_grid->total_energy + ks_potential->exact_exchange_energy;
                    result.integrated_electrons = xc_grid->integrated_electrons;

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

            for (unsigned int iter = 1; iter <= max_iter; ++iter)
            {
                const auto iter_start = std::chrono::steady_clock::now();
                calculator._info._scf.alpha.density = alpha_density;
                calculator._info._scf.beta.density = beta_density;

                auto xc_grid = evaluate_current_density_and_xc(
                    calculator,
                    prepared,
                    x_functional,
                    c_functional);
                if (!xc_grid)
                    return std::unexpected("DFT density/XC evaluation failed: " + xc_grid.error());

                auto ks_potential = assemble_current_ks_potential(calculator, prepared, *xc_grid);
                if (!ks_potential)
                    return std::unexpected("DFT KS potential assembly failed: " + ks_potential.error());

                const Eigen::MatrixXd fock_alpha = calculator._hcore + ks_potential->alpha;
                const Eigen::MatrixXd fock_beta = calculator._hcore + ks_potential->beta;
                const Eigen::MatrixXd total_density = alpha_density + beta_density;

                const double electronic_energy =
                    (total_density.array() * calculator._hcore.array()).sum() +
                    0.5 * (total_density.array() * ks_potential->coulomb.array()).sum() +
                    xc_grid->total_energy +
                    ks_potential->exact_exchange_energy;
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

                const Eigen::MatrixXd fock_alpha_diag =
                    (use_diis && diis_alpha.ready()) ? diis_alpha.extrapolate() : fock_alpha;
                const Eigen::MatrixXd fock_beta_diag =
                    (use_diis && diis_beta.ready()) ? diis_beta.extrapolate() : fock_beta;

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

                const Eigen::MatrixXd next_alpha_density =
                    density_from_orbitals(alpha_diagonalization->coefficients, n_alpha, 1.0);
                const Eigen::MatrixXd next_beta_density =
                    density_from_orbitals(beta_diagonalization->coefficients, n_beta, 1.0);
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

                HartreeFock::SCF::store_unrestricted_iteration(
                    calculator,
                    HartreeFock::SCF::UnrestrictedIterationData{
                        .alpha_density = alpha_density,
                        .beta_density = beta_density,
                        .alpha_fock = fock_alpha,
                        .beta_fock = fock_beta,
                        .alpha_mo_energies = alpha_diagonalization->energies,
                        .beta_mo_energies = beta_diagonalization->energies,
                        .alpha_mo_coefficients = alpha_diagonalization->coefficients,
                        .beta_mo_coefficients = beta_diagonalization->coefficients,
                        .electronic_energy = electronic_energy,
                        .total_energy = total_energy},
                    metrics);
                calculator._info._scf.alpha.mo_symmetry = alpha_diagonalization->mo_symmetry;
                calculator._info._scf.beta.mo_symmetry = beta_diagonalization->mo_symmetry;

                result.total_energy = total_energy;
                result.xc_energy = xc_grid->total_energy + ks_potential->exact_exchange_energy;
                result.integrated_electrons = xc_grid->integrated_electrons;

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
        };

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

            if (exchange->is_hybrid() && !exchange->is_global_hybrid())
            {
                return std::unexpected(
                    "DFT hybrid functional initialization failed: only global hybrids are supported; "
                    "range-separated and double-hybrid functionals are not yet implemented");
            }

            if (exchange->is_global_hybrid())
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info,
                    "DFT Hybrid XC :",
                    std::format("{} exact exchange coefficient = {:.6f}",
                                exchange->name(),
                                exchange->exact_exchange_coefficient()));
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
                .correlation = std::move(*correlation)};
        }

        std::expected<PreparedSystem, std::string> prepare_current_geometry(
            HartreeFock::Calculator &calculator,
            bool preserve_previous_density)
        {
            calculator._molecule._coordinates = calculator._molecule._standard;
            calculator._molecule.coordinates = calculator._molecule._standard / ANGSTROM_TO_BOHR;
            calculator._molecule.set_standard_from_bohr(calculator._molecule._standard);
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

            return run_ks_scf_scaffold(
                calculator,
                *prepared,
                functionals.exchange,
                functionals.correlation);
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
            calculator._molecule._coordinates = calculator._molecule._standard;
            calculator._molecule.coordinates = calculator._molecule._standard / ANGSTROM_TO_BOHR;
            calculator._molecule.set_standard_from_bohr(calculator._molecule._standard);

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
                    "Computing fully numerical Hessian (central differences, gradient step = {:.4f} Bohr)",
                    NUMERICAL_GRADIENT_STEP_BOHR));

            auto gradient_runner = [&](HartreeFock::Calculator &inner) -> std::expected<Eigen::MatrixXd, std::string>
            {
                return compute_numeric_gradient(
                    inner,
                    functionals,
                    NUMERICAL_GRADIENT_STEP_BOHR);
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
                    ? "Starting IC-BFGS optimizer with numerical KS gradients"
                    : "Starting L-BFGS optimizer with numerical KS gradients");
            HartreeFock::Logger::blank();

            auto gradient_runner = [&](HartreeFock::Calculator &inner) -> std::expected<Eigen::VectorXd, std::string>
            {
                auto gradient = compute_numeric_gradient(
                    inner,
                    functionals,
                    NUMERICAL_GRADIENT_STEP_BOHR);
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

    std::expected<XCGridEvaluation, std::string>
    evaluate_current_density_and_xc(
        const HartreeFock::Calculator &calculator,
        const PreparedSystem &prepared,
        const XC::Functional &exchange_functional,
        const XC::Functional &correlation_functional)
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
                correlation_functional);
        }

        return evaluate_xc_on_grid(
            prepared.molecular_grid,
            prepared.ao_grid,
            alpha_density,
            exchange_functional,
            correlation_functional);
    }

    std::expected<KSPotentialMatrices, std::string>
    assemble_current_ks_potential(
        HartreeFock::Calculator &calculator,
        const PreparedSystem &prepared,
        const XCGridEvaluation &xc_grid)
    {
        if (prepared.ao_grid.nbasis() != static_cast<Eigen::Index>(calculator._shells.nbasis()))
            return std::unexpected("AO grid basis dimension does not match the calculator basis");

        if (auto eri_ready = ensure_eri_tensor(calculator, prepared); !eri_ready)
            return std::unexpected(eri_ready.error());

        auto xc_matrix = assemble_xc_matrix(
            prepared.molecular_grid,
            prepared.ao_grid,
            xc_grid);
        if (!xc_matrix)
            return std::unexpected(xc_matrix.error());

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

        const Eigen::MatrixXd coulomb = build_coulomb_from_eri(
            calculator._eri,
            total_density,
            calculator._shells.nbasis());

        const double exact_exchange_coefficient = xc_grid.exact_exchange_coefficient;
        Eigen::MatrixXd exact_exchange_alpha;
        Eigen::MatrixXd exact_exchange_beta;
        double exact_exchange_energy = 0.0;

        if (std::abs(exact_exchange_coefficient) > 1.0e-14)
        {
            if (calculator._scf._scf == HartreeFock::SCFType::UHF)
            {
                const Eigen::MatrixXd exchange_alpha = build_exchange_from_eri(
                    calculator._eri,
                    alpha_density,
                    calculator._shells.nbasis());
                const Eigen::MatrixXd exchange_beta = build_exchange_from_eri(
                    calculator._eri,
                    beta_density,
                    calculator._shells.nbasis());
                exact_exchange_alpha = -exact_exchange_coefficient * exchange_alpha;
                exact_exchange_beta = -exact_exchange_coefficient * exchange_beta;
                exact_exchange_energy =
                    -0.5 * exact_exchange_coefficient *
                    (density_trace_product(alpha_density, exchange_alpha) +
                     density_trace_product(beta_density, exchange_beta));
            }
            else
            {
                const Eigen::MatrixXd exchange = build_exchange_from_eri(
                    calculator._eri,
                    alpha_density,
                    calculator._shells.nbasis());
                exact_exchange_alpha = -0.5 * exact_exchange_coefficient * exchange;
                exact_exchange_beta = exact_exchange_alpha;
                exact_exchange_energy =
                    -0.25 * exact_exchange_coefficient *
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
        if (calculator._scf._scf == HartreeFock::SCFType::ROHF)
            return std::unexpected("ROKS/ROHF DFT references are not implemented; use UKS for open-shell DFT");

        auto functionals = initialize_functionals(calculator);
        if (!functionals)
            return std::unexpected(functionals.error());

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

            auto response = run_linear_response_tda(calculator, functionals->exchange);
            if (!response)
                return std::unexpected(response.error());
            return *result;
        }

        case HartreeFock::CalculationType::Gradient:
        {
            auto result = run_initial_single_point(calculator, options, *functionals);
            if (!result)
                return std::unexpected(result.error());
            if (!result->converged)
                return *result;

            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Info,
                "Gradient :",
                std::format(
                    "Computing numerical nuclear gradient (central differences, h = {:.4f} Bohr)",
                    NUMERICAL_GRADIENT_STEP_BOHR));
            auto gradient = compute_numeric_gradient(calculator, *functionals);
            if (!gradient)
                return std::unexpected("DFT numerical gradient failed: " + gradient.error());
            print_gradient_report(*gradient);
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

} // namespace DFT::Driver
