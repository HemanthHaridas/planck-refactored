#ifndef HF_TYPES_H
#define HF_TYPES_H

#include <Eigen/Core>
#include <Eigen/QR>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <expected>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "basis.h"
#include "tables.h"

constexpr int MAX_L = 6;

namespace HartreeFock
{
    enum class BasisType
    {
        Cartesian, // Cartesian gaussians
        Spherical  // Spherical harmonics
    };

    enum class Units
    {
        Angstrom, // Angstrom
        Bohr      // Bohr
    };

    enum class CoordType
    {
        Cartesian, // Cartesian Coordinates
        ZMatrix    // ZMatrix
    };

    enum class ShellType
    {
        S, // L = 0
        P, // L = 1
        D, // L = 2
        F, // L = 3
        G, // L = 4
        H  // L = 5
    };

    enum class SCFType
    {
        RHF, // Restricted Hartree-Fock
        UHF  // Unrestricted Hartree-Fock
    };

    enum class PostHF
    {
        None,   // No Post-HF corrections
        RMP2,   // Restricted MP2
        UMP2,   // Unrestricted MP2
        CASSCF, // Complete active space SCF
        RASSCF  // Restricted active space SCF
    };

    enum class CalculationType
    {
        SinglePoint,      // Single point Energy Calculation
        Gradient,         // Analytic nuclear gradient
        GeomOpt,          // Geometry Optimization
        Frequency,        // Frequency Calculation
        GeomOptFrequency, // Geometry optimization followed by frequency calculation
        ImaginaryFollow   // Freq → displace along largest imaginary mode → geomopt
    };

    enum class SCFMode
    {
        Conventional, // Stores Matrices
        Direct,       // Recomputes Matrices at each step
        Auto          // Set automatically
    };

    enum class IntegralMethod
    {
        ObaraSaika,    // Obara-Saika recursion (default)
        RysQuadrature, // Rys quadrature for all quartets
        Auto           // OS for L<4, Rys for L>=4 per quartet
    };

    enum class DFTGridQuality
    {
        Coarse,
        Normal,
        Fine,
        UltraFine
    };

    enum class XCExchangeFunctional
    {
        Custom,
        Slater,
        B88,
        PW91,
        PBE
    };

    enum class XCCorrelationFunctional
    {
        Custom,
        VWN5,
        LYP,
        P86,
        PW91,
        PBE
    };

    enum class OptCoords
    {
        Cartesian, // Optimize in Cartesian coordinates (L-BFGS, default)
        Internal   // Optimize in generalized internal coordinates (BFGS)
    };

    // ── Geometry constraint ───────────────────────────────────────────────────
    //
    // Specifies one constraint for a constrained IC-BFGS geometry optimization.
    // Atom indices are 1-based (as given in the input file); unused slots = -1.
    //
    //   Bond       — fix the distance between atoms[0] and atoms[1]
    //   Angle      — fix the angle atoms[0]–atoms[1]–atoms[2]
    //   Dihedral   — fix the dihedral atoms[0]–atoms[1]–atoms[2]–atoms[3]
    //   FrozenAtom — hold all 3 Cartesian DOFs of atoms[0] fixed
    struct GeomConstraint
    {
        enum class Type
        {
            Bond,
            Angle,
            Dihedral,
            FrozenAtom
        };
        Type type;
        std::array<int, 4> atoms = {-1, -1, -1, -1}; // 1-based indices
    };

    enum class Verbosity
    {
        Silent,  // No output
        Minimal, // Minimal output (final results only)
        Normal,  // Normal output (iteration info)
        Verbose, // Verbose output (detailed info)
        Debug    // Debug output (everything)
    };

    struct Molecule
    {
        Eigen::VectorXi atomic_numbers = {}; // Atomic numbers
        Eigen::VectorXd atomic_masses = {};  // Atomic masses

        Eigen::MatrixXd coordinates;  // natoms × 3, in Angstrom
        Eigen::MatrixXd _coordinates; // natoms × 3, in Bohr (internal use)

        Eigen::MatrixXd standard;  // reoriented coordinates in Angstrom
        Eigen::MatrixXd _standard; // reoriented coordinates in Bohr

        std::string _point_group = "C1"; // Point group symmetry

        std::size_t natoms = 0;        // Number of atoms
        unsigned int multiplicity = 1; // Spin multiplicity
        signed int charge = 0;         // Molecular charge

        bool _symmetry = false; // Symmetry flag
        bool _is_bohr = false;

        void clear() noexcept
        {
            natoms = 0;
            charge = 0;
            multiplicity = 1;

            atomic_numbers.resize(0);
            atomic_masses.resize(0);

            coordinates.resize(0, 3);
            standard.resize(0, 3);

            _coordinates.resize(0, 3);
            _standard.resize(0, 3);

            _point_group = "C1";
            _symmetry = false;
            _is_bohr = false;
        }
    };

    struct Shell
    {
        Eigen::Vector3d _center = Eigen::Vector3d::Zero(); // Shell position (Bohr)
        ShellType _shell;
        unsigned int _atom_index = 0;

        Eigen::VectorXd _primitives;
        Eigen::VectorXd _coefficients;
        Eigen::VectorXd _normalizations;

        std::size_t nprimitives() const noexcept
        {
            return _primitives.size();
        }
    };

    struct ContractedView
    {
        const Shell *_shell = nullptr;
        std::size_t _index = 0;                               // position in Basis::_basis_functions
        double _component_norm = 1.0;                         // 1/sqrt((2lx-1)!! (2ly-1)!! (2lz-1)!!)
        Eigen::Vector3i _cartesian = Eigen::Vector3i::Zero(); // 4-byte tail padding follows

        std::span<const double> exponents() const noexcept
        {
            return std::span<const double>(_shell->_primitives.data(), _shell->_primitives.size());
        }

        std::span<const double> coefficients() const noexcept
        {
            return std::span<const double>(_shell->_coefficients.data(), _shell->_coefficients.size());
        }

        std::span<const double> normalizations() const noexcept
        {
            return std::span<const double>(_shell->_normalizations.data(), _shell->_normalizations.size());
        }

        const Eigen::Vector3d &center() const noexcept
        {
            return _shell->_center;
        }
    };

    struct Basis
    {
        std::vector<Shell> _shells;                  // Shells
        std::deque<ContractedView> _basis_functions; // Basis functions; deque keeps references stable across push_back

        std::size_t nshells() const noexcept
        {
            return _shells.size();
        }

        std::size_t nbasis() const noexcept
        {
            return _basis_functions.size();
        }

        void clear()
        {
            _shells.clear();
            _basis_functions.clear();
        }
    };

    enum class SCFGuess
    {
        HCore,       // Diagonalize core Hamiltonian (default)
        SAD,         // Superposition of Atomic Densities
        ReadDensity, // Load only the density matrix from checkpoint (geometry from input)
        ReadFull,    // Load geometry + charge + multiplicity + density from checkpoint
    };

    struct OptionsSCF
    {
        double _tol_energy = 1E-10;        // Energy tolerance
        double _tol_density = 1E-10;       // Density tolerance
        double _level_shift = 0.0;         // Virtual orbital level shift in Hartree (0 = off)
        double _diis_restart_factor = 2.0; // Restart DIIS when error grows by this factor (0 = off)

        SCFType _scf = SCFType::RHF;           // SCF Type (Default is RHF)
        SCFMode _mode = SCFMode::Conventional; // SCF Mode (Default is Conventional)
        SCFGuess _guess = SCFGuess::HCore;     // Initial guess

        unsigned int _max_cycles = 0;  // Maximum number of SCF Cycles
        unsigned int _threshold = 100; // Threshold before switching to Direct mode (Default is 100)
        unsigned int _DIIS_dim = 8;    // Dimension of DIIS Error Vector (Default is 8)

        bool _use_DIIS = true;        // Use DIIS (Default is true)
        bool _save_checkpoint = true; // Save checkpoint after convergence

        static unsigned int max_cycles_for_nbasis(std::size_t nbasis) noexcept
        {
            return (nbasis > 1000) ? 300 : (nbasis > 500) ? 200
                                       : (nbasis > 250)   ? 100
                                                          : 50;
        }

        // Automatic setter based on system size
        void set_max_cycles_auto(std::size_t nbasis) noexcept
        {
            _max_cycles = max_cycles_for_nbasis(nbasis);
        }

        // Getter (auto fallback if still 0)
        unsigned int get_max_cycles(std::size_t nbasis) const noexcept
        {
            if (_max_cycles != 0)
                return _max_cycles;

            return max_cycles_for_nbasis(nbasis);
        }

        // Resolve Auto mode based on system size; explicit Conventional/Direct are left unchanged.
        void set_scf_mode_auto(std::size_t nbasis)
        {
            if (_mode == SCFMode::Auto)
                _mode = (nbasis > _threshold) ? SCFMode::Direct : SCFMode::Conventional;
        }

        // Getter
        unsigned int get_threshold() const noexcept
        {
            return _threshold;
        }
    };

    struct OptionsBasis
    {
        std::string _basis_name;                    // Name of basis set
        std::string _basis_path = get_basis_path(); // Path to basis
        BasisType _basis = BasisType::Cartesian;    // Type of Basis (Only supports Cartesian)
    };

    struct OptionsGeometry
    {
        Units _units = Units::Angstrom;         // Coordinate units (Default is Angstrom)
        CoordType _type = CoordType::Cartesian; // Coordinate type (Default is Cartesian)
        bool _use_symm = true;                  // Detect point group symmetry
    };

    struct OptionsIntegral
    {
        double _tol_eri = 1E-10;                             // ERI tolerance for Schwarz screening
        IntegralMethod _engine = IntegralMethod::ObaraSaika; // Integral Engine
    };

    struct OptionsDFT
    {
        DFTGridQuality _grid = DFTGridQuality::Normal;
        XCExchangeFunctional _exchange = XCExchangeFunctional::PBE;
        XCCorrelationFunctional _correlation = XCCorrelationFunctional::PBE;
        int _exchange_id = 0;    // 0 => resolve from _exchange through libxc
        int _correlation_id = 0; // 0 => resolve from _correlation through libxc
        bool _use_sao_blocking = true;
        bool _print_grid_summary = true;
        bool _save_checkpoint = false;
    };

    // Optional symmetry-aware orbital selection metadata for active-space setup.
    // The parser accepts repeated "irrep count" pairs and stores them in the
    // order written by the input file.
    struct IrrepCount
    {
        std::string irrep;
        int count = 0;
    };

    // ── Active space specification (CASSCF / RASSCF) ─────────────────────────
    struct OptionsActiveSpace
    {
        // SA weights and symmetry filtering (24-byte heap types first)
        std::vector<double> weights; // SA weights (length nroots); empty → equal weights

        // Symmetry filtering: target CI state irrep (e.g. "A1", "B1g").
        // Empty string → use the totally-symmetric irrep of the detected point group.
        std::string target_irrep = "";

        // Optional symmetry-aware MO selection. If present, the parser records
        // explicit irrep quotas for the core and active blocks, and an optional
        // full MO permutation can override the automatic picker.
        std::vector<IrrepCount> core_irrep_counts;
        std::vector<IrrepCount> active_irrep_counts;
        std::vector<int> mo_permutation; // Full MO permutation as entered; selector normalizes 0- or 1-based input

        // MCSCF convergence tolerances
        double tol_mcscf_energy = 1e-8;
        double tol_mcscf_grad = 1e-5;

        // CASSCF / SA-CASSCF
        int nactele = 0; // number of active electrons
        int nactorb = 0; // number of active orbitals
        int nroots = 1;  // number of CI roots for state averaging (1 = single-state)

        // RASSCF extensions (ignored for plain CASSCF)
        int nras1 = 0;     // RAS1 orbital count (high-occ. restricted space)
        int nras2 = 0;     // RAS2 orbital count (full CAS subspace)
        int nras3 = 0;     // RAS3 orbital count (low-virt. restricted space)
        int max_holes = 2; // max electrons removed from RAS1
        int max_elec = 2;  // max electrons added to RAS3

        // MCSCF iteration limits
        unsigned int mcscf_max_iter = 100;
        unsigned int mcscf_micro_per_macro = 4;
        unsigned int ci_max_dim = 10000; // abort if CI space exceeds this

        bool mcscf_debug_numeric_newton = false; // debug-only numeric Newton fallback
        bool mcscf_debug_commutator_rhs = false; // debug-only approximate commutator-only CI-response RHS
    };

    struct OptionsOutput
    {
        Verbosity _verbosity = Verbosity::Minimal; // Default Verbosity is Minimal

        bool _print_orbitals = false;    // Print MO energies and coefficients
        bool _print_populations = false; // Print Mulliken populations
        bool _print_geometry = true;     // Print molecular geometry
        bool _print_basis_info = true;   // Print basis set information
        bool _print_matrices = false;    // Print SCF matrices (S, H, F, etc.)
        bool _write_molden = false;      // Write Molden format file
        bool _write_cube = false;        // Write cube files for orbitals

        // Automatic setter
        void set_output_options(Verbosity verbosity)
        {
            switch (verbosity)
            {
            case Verbosity::Silent:
            {
                _print_orbitals = false;
                _print_populations = false;
                _print_geometry = false;
                _print_basis_info = false;
                _print_matrices = false;
                _write_molden = false;
                _write_cube = false;
                break;
            }
            case Verbosity::Minimal:
            {
                _print_orbitals = false;
                _print_populations = false;
                _print_geometry = true;
                _print_basis_info = true;
                _print_matrices = false;
                _write_molden = false;
                _write_cube = false;
                break;
            }
            case Verbosity::Normal:
            {
                _print_orbitals = true;
                _print_populations = false;
                _print_geometry = true;
                _print_basis_info = true;
                _print_matrices = false;
                _write_molden = false;
                _write_cube = false;
                break;
            }
            case Verbosity::Verbose:
            {
                _print_orbitals = true;
                _print_populations = true;
                _print_geometry = true;
                _print_basis_info = true;
                _print_matrices = false;
                _write_molden = false;
                _write_cube = false;
                break;
            }

            case Verbosity::Debug:
            {
                _print_orbitals = true;
                _print_populations = true;
                _print_geometry = true;
                _print_basis_info = true;
                _print_matrices = true;
                _write_molden = false;
                _write_cube = false;
                break;
            }
            }
        }
    };

    struct PrimitivePair
    {
        double alpha;           // exponent of primitive on A
        double beta;            // exponent of primitive on B
        double zeta;            // alpha + beta
        double inv_zeta;        // 1 / (alpha + beta)
        double prefactor;       // (pi/zeta)^1.5 * exp(-alpha*beta/zeta * R^2)
        double coeff_product;   // c_i * c_j * N_i * N_j
        Eigen::Vector3d center; // Gaussian product center P
        Eigen::Vector3d pA;     // P - A
        Eigen::Vector3d pB;     // P - B
    };

    struct ShellPair
    {
        const ContractedView &A; // Shell A
        const ContractedView &B; // Shell B

        Eigen::Vector3d R; // R_AB = A - B
        double R2;         // |R_AB|^2
        double Rnorm;      // |R_AB|
        double screening;  // Schwarz screening

        std::vector<PrimitivePair> primitive_pairs;

        explicit ShellPair(const ContractedView &sA, const ContractedView &sB)
            : A(sA), B(sB)
        {
            R = A._shell->_center - B._shell->_center;
            R2 = R.squaredNorm();
            Rnorm = std::sqrt(R2);

            const std::size_t nA = A._shell->nprimitives();
            const std::size_t nB = B._shell->nprimitives();

            primitive_pairs.reserve(nA * nB);

            for (std::size_t i = 0; i < nA; ++i)
            {
                const double alpha = A._shell->_primitives[i];
                const double cA = A._shell->_coefficients[i];

                for (std::size_t j = 0; j < nB; ++j)
                {
                    const double beta = B._shell->_primitives[j];
                    const double cB = B._shell->_coefficients[j];

                    const double zeta = alpha + beta;
                    const double inv_zeta = 1.0 / zeta;
                    const double alpha_beta_over_zeta = alpha * beta * inv_zeta;

                    PrimitivePair pp;
                    pp.alpha = alpha;
                    pp.beta = beta;
                    pp.zeta = zeta;
                    pp.inv_zeta = inv_zeta;
                    pp.coeff_product = cA * cB * A._shell->_normalizations[i] * B._shell->_normalizations[j] * A._component_norm * B._component_norm;
                    pp.prefactor = std::pow(M_PI * inv_zeta, 1.5) * std::exp(-alpha_beta_over_zeta * R2);

                    // Gaussian product center
                    pp.center = (alpha * A._shell->_center + beta * B._shell->_center) * inv_zeta;

                    pp.pA = pp.center - A._shell->_center;
                    pp.pB = pp.center - B._shell->_center;

                    primitive_pairs.emplace_back(pp);
                }
            }
        }
    };

    struct SpinChannel
    {
        Eigen::MatrixXd density;
        Eigen::MatrixXd fock;
        Eigen::VectorXd mo_energies;
        Eigen::MatrixXd mo_coefficients;
        std::vector<std::string> mo_symmetry; // irrep labels (empty when symmetry is off)
    };

    struct DataSCF
    {
        SpinChannel alpha;
        SpinChannel beta;

        bool is_uhf = false;
        bool is_init = false;

        // Allow {} initialization
        DataSCF() = default;

        // Constructor
        explicit DataSCF(bool uhf)
            : is_uhf(uhf), is_init(false)
        {
            // Alpha channel always exists
            alpha.density = Eigen::MatrixXd{};
            alpha.fock = Eigen::MatrixXd{};
            alpha.mo_energies = Eigen::VectorXd{};
            alpha.mo_coefficients = Eigen::MatrixXd{};

            // Beta channel only meaningful for UHF
            if (is_uhf)
            {
                beta.density = Eigen::MatrixXd{};
                beta.fock = Eigen::MatrixXd{};
                beta.mo_energies = Eigen::VectorXd{};
                beta.mo_coefficients = Eigen::MatrixXd{};
            }
        }

        // Initialize the arrays
        void initialize(std::size_t nbasis)
        {
            alpha.density = Eigen::MatrixXd::Zero(nbasis, nbasis);
            alpha.fock = Eigen::MatrixXd::Zero(nbasis, nbasis);
            alpha.mo_energies = Eigen::VectorXd::Zero(nbasis);
            alpha.mo_coefficients = Eigen::MatrixXd::Zero(nbasis, nbasis);

            if (is_uhf)
            {
                beta.density = Eigen::MatrixXd::Zero(nbasis, nbasis);
                beta.fock = Eigen::MatrixXd::Zero(nbasis, nbasis);
                beta.mo_energies = Eigen::VectorXd::Zero(nbasis);
                beta.mo_coefficients = Eigen::MatrixXd::Zero(nbasis, nbasis);
            }

            is_init = true;
        }
    };

    struct InfoSCF
    {
        double _energy = 0;            // SCF Energy in Hartree
        double _delta_energy = 0;      // Difference in SCF energy
        double _delta_density_max = 0; // Difference in Max SCF Density
        double _delta_density_rms = 0; // Difference in RMS SCF Density
        bool _is_converged = false;    // Is convergence signalled
        DataSCF _scf;                  // SCF Data for current step
    };

    // ── DIIS working state for one spin channel ───────────────────────────────
    //
    // Implements Pulay's DIIS for SCF convergence acceleration.
    //
    // At each iteration, the caller supplies:
    //   F  — current Fock matrix (AO basis)
    //   e  — error matrix = X^T (F*P*S - S*P*F) X  (orthonormal basis)
    //        where X = S^{-1/2} is the orthogonalizer.
    //
    // Once at least 2 vectors are stored, extrapolate() returns a new Fock
    // matrix as a linear combination of the stored Fock matrices chosen to
    // minimise the DIIS error norm subject to sum(c_i)=1.
    struct DIISState
    {
        std::deque<Eigen::MatrixXd> fock_history;  // stored F matrices (AO basis)
        std::deque<Eigen::MatrixXd> error_history; // stored error matrices (orthonormal)
        std::size_t max_vecs = 8;                  // maximum subspace size

        // Append a new (F, e) pair, evicting the oldest if at capacity.
        void push(const Eigen::MatrixXd &F, const Eigen::MatrixXd &e)
        {
            fock_history.push_back(F);
            error_history.push_back(e);
            if (fock_history.size() > max_vecs)
            {
                fock_history.pop_front();
                error_history.pop_front();
            }
        }

        std::size_t size() const noexcept
        {
            return fock_history.size();
        }

        bool ready() const noexcept
        {
            return size() >= 2;
        }

        // Build the augmented B matrix and solve for DIIS coefficients, then
        // return the extrapolated Fock matrix.  Requires ready() == true.
        Eigen::MatrixXd extrapolate() const
        {
            const std::size_t m = size();

            // B_{ij} = Tr( e_i^T e_j )
            // Augmented system (Lagrange multiplier for sum(c)=1):
            //   [ B  -1 ] [ c   ]   [ 0 ]
            //   [-1   0 ] [ lam ] = [-1 ]
            Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m + 1, m + 1);
            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t j = i; j < m; ++j)
                {
                    const double bij = (error_history[i].array() * error_history[j].array()).sum();
                    B(i, j) = bij;
                    B(j, i) = bij;
                }
                B(i, m) = -1.0;
                B(m, i) = -1.0;
            }

            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1);
            rhs(m) = -1.0;

            const Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

            Eigen::MatrixXd F_extrap = Eigen::MatrixXd::Zero(fock_history[0].rows(),
                                                             fock_history[0].cols());
            for (std::size_t i = 0; i < m; ++i)
                F_extrap += c(i) * fock_history[i];

            return F_extrap;
        }

        // Return the DIIS error norm (RMS of the most recent error matrix).
        double error_norm() const
        {
            if (error_history.empty())
                return 0.0;
            const auto &e = error_history.back();
            return std::sqrt(e.squaredNorm() / static_cast<double>(e.size()));
        }

        void clear() noexcept
        {
            fock_history.clear();
            error_history.clear();
        }
    };

    struct SignedAOSymOp
    {
        std::vector<int> ao_map;     // mu -> nu under the symmetry operation
        std::vector<int8_t> ao_sign; // phase of the mapped Cartesian AO (+1 / -1)
    };

    struct MultipoleMatrices
    {
        std::array<Eigen::MatrixXd, 3> dipole;
        std::array<Eigen::MatrixXd, 6> quadrupole;
    };

    struct MultipoleMoments
    {
        Eigen::Vector3d origin = Eigen::Vector3d::Zero();

        Eigen::Vector3d electronic_dipole = Eigen::Vector3d::Zero();
        Eigen::Vector3d nuclear_dipole = Eigen::Vector3d::Zero();
        Eigen::Vector3d total_dipole = Eigen::Vector3d::Zero();

        Eigen::Matrix3d electronic_quadrupole = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d nuclear_quadrupole = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d total_quadrupole = Eigen::Matrix3d::Zero();
    };

    struct Calculator
    {
        OptionsSCF _scf;
        OptionsBasis _basis;
        OptionsGeometry _geometry;
        OptionsIntegral _integral;
        OptionsDFT _dft;
        OptionsOutput _output;
        InfoSCF _info;
        Molecule _molecule;
        Basis _shells;

        CalculationType _calculation = CalculationType::SinglePoint; // Default is Single point energy calculation
        PostHF _correlation = PostHF::None;                          // No Post HF corrections

        double _total_energy = 0;         // Total Energy (SCF + Nuclear Repulsion)
        double _nuclear_repulsion = 0;    // Nuclear Repulsion Energy (Bohr)
        double _correlation_energy = 0.0; // Post-HF correlation energy (0 if not computed)

        // CASSCF / RASSCF results
        OptionsActiveSpace _active_space;     // active space specification
        Eigen::VectorXd _cas_nat_occ;         // active natural occupation numbers
        Eigen::MatrixXd _cas_mo_coefficients; // converged CASSCF MO coefficients [nb×nb] in the optimization basis
        double _casscf_rhf_energy = 0.0;      // RHF reference energy (for ΔE printout)
        Eigen::VectorXd _cas_root_energies;   // per-root total CASSCF energies (length nroots; empty for SS-CASSCF)

        Eigen::MatrixXd _overlap; // Overlap matrix S
        Eigen::MatrixXd _hcore;   // Core Hamiltonian H = T + V
        std::vector<double> _eri; // ERI

        std::string _checkpoint_path; // Path to checkpoint file (set by driver)

        // Symmetry-adapted orbital (SAO) blocking — populated by build_sao_basis() pre-SCF.
        // When _use_sao_blocking is true, run_rhf/run_uhf use per-irrep block diagonalization.
        Eigen::MatrixXd _sao_transform;            // U [nb×nb]: AO→SAO unitary
        std::vector<int> _sao_irrep_index;         // irrep index per SAO column
        std::vector<std::string> _sao_irrep_names; // Mulliken name per irrep index
        std::vector<int> _sao_block_sizes;         // n_SAOs per irrep block
        std::vector<int> _sao_block_offsets;       // start offset per block in SAO ordering
        bool _use_sao_blocking = false;
        std::vector<SignedAOSymOp> _integral_symmetry_ops; // signed AO permutations used to reduce integral work
        bool _use_integral_symmetry = false;

        // Gradient and geometry optimization
        Eigen::MatrixXd _gradient;                    // natoms×3, Ha/Bohr; set by compute_rhf/uhf_gradient()
        double _geomopt_grad_tol = 3e-4;              // convergence: max |∂E/∂x_i| in Ha/Bohr
        int _geomopt_max_iter = 50;                   // maximum geometry steps
        int _geomopt_lbfgs_m = 10;                    // L-BFGS history size
        OptCoords _opt_coords = OptCoords::Cartesian; // coordinate system for optimization
        std::vector<GeomConstraint> _constraints;     // from %begin_constraints section

        // Hessian / frequency analysis
        Eigen::MatrixXd _hessian;                       // 3N×3N Cartesian Hessian, Ha/Bohr²
        Eigen::VectorXd _frequencies;                   // n_vib vibrational frequencies in cm⁻¹
        Eigen::MatrixXd _normal_modes;                  // 3N × n_vib mass-unweighted normal modes
        std::vector<std::string> _vibrational_symmetry; // n_vib Mulliken labels
        double _zpe = 0.0;                              // zero-point energy in Ha
        double _hessian_step = 5e-3;                    // finite-difference step in Bohr
        double _imag_follow_step = 0.2;                 // Cartesian displacement along imaginary mode, Bohr

        void _compute_nuclear_repulsion() noexcept
        {
            assert(_molecule._standard.rows() == static_cast<Eigen::Index>(_molecule.natoms) &&
                   _molecule._standard.cols() == 3 &&
                   "_compute_nuclear_repulsion requires molecule._standard to be initialized in Bohr");
            const std::size_t N = _molecule.natoms;
            double E_nuc = 0.0;
            for (std::size_t a = 0; a < N; a++)
            {
                for (std::size_t b = a + 1; b < N; b++)
                {
                    const double Za = static_cast<double>(_molecule.atomic_numbers[a]);
                    const double Zb = static_cast<double>(_molecule.atomic_numbers[b]);
                    const double dx = _molecule._standard(a, 0) - _molecule._standard(b, 0);
                    const double dy = _molecule._standard(a, 1) - _molecule._standard(b, 1);
                    const double dz = _molecule._standard(a, 2) - _molecule._standard(b, 2);
                    E_nuc += Za * Zb / std::sqrt(dx * dx + dy * dy + dz * dz);
                }
            }
            _nuclear_repulsion = E_nuc;
        }

    public:
        // Convert input coordinates to Bohr and store in _coordinates.
        // Must be called once, before detectSymmetry() and read_gbs_basis().
        void prepare_coordinates() noexcept
        {
            if (_molecule._is_bohr)
                return;
            if (_geometry._units == Units::Bohr)
            {
                _molecule._coordinates = _molecule.coordinates;
            }
            else
            {
                _molecule._coordinates = _molecule.coordinates * ANGSTROM_TO_BOHR;
            }
            _molecule._is_bohr = true;
        }

        // Getter
        Eigen::MatrixXd _bohr_to_angstrom() const noexcept
        {
            return _molecule._standard * 0.529177210903;
        }

        // Run Calculation
        std::expected<void, std::string> initialize()
        {
            if (!_info._scf.is_init)
            {
                // First set the spin channel information
                _info._scf = DataSCF(_scf._scf == SCFType::UHF);

                // Now initialize the matrices
                _info._scf.initialize(_shells.nbasis());

                // Set SCF Mode
                _scf.set_scf_mode_auto(_shells.nbasis());

                // Set Max SCF cycles (only if not explicitly set by the user)
                if (_scf._max_cycles == 0)
                    _scf.set_max_cycles_auto(_shells.nbasis());
            }

            // prepare_coordinates() must have been called before initialize().
            // Nothing to do here for coordinate conversion.

            _compute_nuclear_repulsion();

            return {};
        }
    };
} // namespace HartreeFock

#endif // !HF_TYPES_H
