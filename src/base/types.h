#ifndef HF_TYPES_H
#define HF_TYPES_H

#include <span>
#include <array>
#include <deque>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <expected>
#include <Eigen/Core>
#include <Eigen/QR>

#include "tables.h"
#include "basis.h"

constexpr int MAX_L = 6;

namespace HartreeFock
{
    enum class BasisType
    {
        Cartesian,  // Cartesian gaussians
        Spherical   // Spherical harmonics
    };

    enum class Units
    {
        Angstrom,   // Angstrom
        Bohr        // Bohr
    };

    enum class CoordType
    {
        Cartesian,  // Cartesian Coordinates
        ZMatrix     // ZMatrix
    };

    enum class ShellType
    {
        S,  // L = 0
        P,  // L = 1
        D,  // L = 2
        F,  // L = 3
        G,  // L = 4
        H   // L = 5
    };

    enum class SCFType
    {
        RHF,    // Restricted Hartee-Fock
        UHF    // Unrestricted Hartree-Fock
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
        SinglePoint,    // Single point Energy Calculation
        GeomOpt,        // Geometry Optimization
        Frequency       // Frequency Calculation
    };

    enum class SCFMode
    {
        Conventional,   // Stores Matrices
        Direct,         // Recomputes Matrices at each step
        Auto            // Set automatically
    };

    enum class IntegralMethod
    {
        ObaraSaika,        // Obara-Saika recursion (default)
        McMurchieDavidson, // McMurchie-Davidson (Hermite)
        Huzinaga           // Huzinaga method
    };

    enum class Verbosity
    {
        Silent,         // No output
        Minimal,        // Minimal output (final results only)
        Normal,         // Normal output (iteration info)
        Verbose,        // Verbose output (detailed info)
        Debug           // Debug output (everything)
    };

    struct Molecule
    {
        std::size_t     natoms = 0;         // Number of atoms
        unsigned int    multiplicity = 1;   // Spin multiplicity
        unsigned int    nelectrons = 0;     // Number of electrons
        signed int      charge = 0;         // Molecular charge
        
        Eigen::VectorXi atomic_numbers = {};   // Atomic numbers
        Eigen::VectorXd atomic_masses  = {};   // Atomic masses
        
        Eigen::MatrixXd coordinates;   // natoms × 3, in Angstrom
        Eigen::MatrixXd _coordinates;  // natoms × 3, in Bohr (internal use)
        
        Eigen::MatrixXd standard;      // reoriented coordinates in Angstrom
        Eigen::MatrixXd _standard;     // reoriented coordinates in Bohr
        
        std::string _point_group = "C1";   // Point group symmetry
        bool _symmetry  = false;            // Symmetry flag
        bool _is_bohr   = false;

        void clear() noexcept
        {
            natoms          = 0;
            charge          = 0;
            multiplicity    = 1;
            
            atomic_numbers.resize(0);
            atomic_masses.resize(0);
            
            coordinates.resize(0, 3);
            standard.resize(0, 3);
            
            _coordinates.resize(0, 3);
            _standard.resize(0, 3);
            
            _point_group    = "C1";
            _symmetry       = false;
            _is_bohr        = false;
        }
    };

    struct Shell
    {
        Eigen::Vector3d _center = Eigen::Vector3d::Zero();  // Shell position (Bohr)
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
        const Shell* _shell = nullptr;
        Eigen::Vector3i _cartesian = Eigen::Vector3i::Zero();
        std::size_t _index = 0;   // position in Basis::_basis_functions

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

        const Eigen::Vector3d& center() const noexcept
        {
            return _shell->_center;
        }
    };

    struct Basis
    {
        std::vector <Shell> _shells;                    // Shells
        std::vector <ContractedView> _basis_functions;  // Basis functions
        
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

    struct OptionsSCF
    {
        SCFType _scf  = SCFType::RHF;           // SCF Type (Default is RHF)
        SCFMode _mode = SCFMode::Conventional;  // SCF Mode (Default is Conventional)
        
        unsigned int _max_cycles    = 0;    // Maximum number of SCF Cycles
        unsigned int _threshold     = 100;  // Threshold before switching to Direct mode (Default is 100)
        
        double _tol_energy  = 1E-10;        // Energy tolerance
        double _tol_density = 1E-10;        // Density tolerance
        
        unsigned int _DIIS_dim = 8;         // Dimension of DIIS Error Vector (Default is 8)
        bool _use_DIIS = true;              // Use DIIS (Default is true)
        
        
        // Automatic setter based on system size
        void set_max_cycles_auto(std::size_t nbasis) noexcept
        {
            _max_cycles =
                (nbasis > 1000) ? 300 :
                (nbasis > 500)  ? 200 :
                (nbasis > 250)  ? 100 :
                50;
        }
        
        // Getter (auto fallback if still 0)
        unsigned int get_max_cycles(std::size_t nbasis) const noexcept
        {
            if (_max_cycles != 0)
                return _max_cycles;
            
            return (nbasis > 1000) ? 300 :
            (nbasis > 500)  ? 200 :
            (nbasis > 250)  ? 150 :
            50;
        }
        
        // Automatic setter for SCF Mode based on system size
        void set_scf_mode_auto(std::size_t nbasis)
        {
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
        std::string _basis_name;                        // Name of basis set
        std::string _basis_path = get_basis_path();     // Path to basis
        BasisType _basis = BasisType::Cartesian;        // Type of Basis (Only supports Cartesian)
    };

    struct OptionsGeometry
    {
        Units _units    = Units::Angstrom;      // Coordinate units (Default is Angstrom)
        CoordType _type = CoordType::Cartesian; // Coordinate type (Default is Cartesian)
        bool _use_symm  = true;                 // Detect point group symmetry
    };

    struct OptionsIntegral
    {
        IntegralMethod _engine = IntegralMethod::ObaraSaika;    // Integral Engine
        double _tol_eri = 1E-10;                                // ERI tolerance for Shwartz screening;
    };

    struct OptionsOutput
    {
        Verbosity _verbosity    = Verbosity::Minimal; // Default Verbosity is Minimal
        
        bool _print_orbitals    = false;    // Print MO energies and coefficients
        bool _print_populations = false;    // Print Mulliken populations
        bool _print_geometry    = true;     // Print molecular geometry
        bool _print_basis_info  = true;     // Print basis set information
        bool _print_matrices    = false;    // Print SCF matrices (S, H, F, etc.)
        bool _write_molden      = false;    // Write Molden format file
        bool _write_cube        = false;    // Write cube files for orbitals
        
        // Automatic setter
        void set_output_options(Verbosity verbosity)
        {
            switch (verbosity) {
                case Verbosity::Silent:
                {
                    _print_orbitals     = false;
                    _print_populations  = false;
                    _print_geometry     = false;
                    _print_basis_info   = false;
                    _print_matrices     = false;
                    _write_molden       = false;
                    _write_cube         = false;
                    break;
                }
                case Verbosity::Minimal:
                {
                    _print_orbitals     = false;
                    _print_populations  = false;
                    _print_geometry     = true;
                    _print_basis_info   = true;
                    _print_matrices     = false;
                    _write_molden       = false;
                    _write_cube         = false;
                    break;
                }
                case Verbosity::Normal:
                {
                    _print_orbitals     = true;
                    _print_populations  = false;
                    _print_geometry     = true;
                    _print_basis_info   = true;
                    _print_matrices     = false;
                    _write_molden       = false;
                    _write_cube         = false;
                    break;
                }
                case Verbosity::Verbose:
                {
                    _print_orbitals     = true;
                    _print_populations  = true;
                    _print_geometry     = true;
                    _print_basis_info   = true;
                    _print_matrices     = false;
                    _write_molden       = false;
                    _write_cube         = false;
                    break;
                }

                case Verbosity::Debug:
                {
                    _print_orbitals     = true;
                    _print_populations  = true;
                    _print_geometry     = true;
                    _print_basis_info   = true;
                    _print_matrices     = true;
                    _write_molden       = false;
                    _write_cube         = false;
                    break;
                }
                default:
                    throw std::runtime_error("Unknown verbosity level");
            }
        }
    };

    struct PrimitivePair
    {
        double alpha;               // exponent of primitive on A
        double beta;                // exponent of primitive on B
        double zeta;                // alpha + beta
        double inv_zeta;            // 1 / (alpha + beta)
        double prefactor;           // (pi/zeta)^1.5 * exp(-alpha*beta/zeta * R^2)
        double coeff_product;       // c_i * c_j * N_i * N_j
        Eigen::Vector3d center;     // Gaussian product center P
        Eigen::Vector3d pA;         // P - A
        Eigen::Vector3d pB;         // P - B
    };

    struct ShellPair
    {
        const ContractedView& A;   // Shell A
        const ContractedView& B;   // Shell B

        Eigen::Vector3d R;      // R_AB = A - B
        double R2;              // |R_AB|^2
        double Rnorm;           // |R_AB|
        double screening;       // Schwarz screening
        
        std::vector<PrimitivePair> primitive_pairs;

        explicit ShellPair(const ContractedView& sA, const ContractedView& sB)
            : A(sA), B(sB)
        {
            R     = A._shell->_center - B._shell->_center;
            R2    = R.squaredNorm();
            Rnorm = std::sqrt(R2);

            const std::size_t nA = A._shell->nprimitives();
            const std::size_t nB = B._shell->nprimitives();

            primitive_pairs.reserve(nA * nB);

            for (std::size_t i = 0; i < nA; ++i)
            {
                const double alpha = A._shell->_primitives[i];
                const double cA    = A._shell->_coefficients[i];

                for (std::size_t j = 0; j < nB; ++j)
                {
                    const double beta = B._shell->_primitives[j];
                    const double cB   = B._shell->_coefficients[j];

                    const double zeta      = alpha + beta;
                    const double inv_zeta  = 1.0 / zeta;
                    const double alpha_beta_over_zeta = alpha * beta * inv_zeta;

                    PrimitivePair pp;
                    pp.alpha        = alpha;
                    pp.beta         = beta;
                    pp.zeta         = zeta;
                    pp.inv_zeta     = inv_zeta;
                    pp.coeff_product= cA * cB
                                    * A._shell->_normalizations[i]
                                    * B._shell->_normalizations[j];
                    pp.prefactor    = std::pow(M_PI * inv_zeta, 1.5)
                                    * std::exp(-alpha_beta_over_zeta * R2);

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
    };

    struct DataSCF
    {
        SpinChannel alpha;
        SpinChannel beta;

        bool is_uhf  = false;
        bool is_init = false;

        // Allow {} initialization
        DataSCF() = default;
        
        // Constructor
        explicit DataSCF(bool uhf)
            : is_uhf(uhf), is_init(false)
        {
            // Alpha channel always exists
            alpha.density        = Eigen::MatrixXd{};
            alpha.fock           = Eigen::MatrixXd{};
            alpha.mo_energies    = Eigen::VectorXd{};
            alpha.mo_coefficients= Eigen::MatrixXd{};

            // Beta channel only meaningful for UHF
            if (is_uhf) {
                beta.density         = Eigen::MatrixXd{};
                beta.fock            = Eigen::MatrixXd{};
                beta.mo_energies     = Eigen::VectorXd{};
                beta.mo_coefficients = Eigen::MatrixXd{};
            }
        }
        
        // Initialize the arrays
        void initialize(std::size_t nbasis) {
            alpha.density         = Eigen::MatrixXd::Zero(nbasis, nbasis);
            alpha.fock            = Eigen::MatrixXd::Zero(nbasis, nbasis);
            alpha.mo_energies     = Eigen::VectorXd::Zero(nbasis);
            alpha.mo_coefficients = Eigen::MatrixXd::Zero(nbasis, nbasis);

            if (is_uhf) {
                beta.density         = Eigen::MatrixXd::Zero(nbasis, nbasis);
                beta.fock            = Eigen::MatrixXd::Zero(nbasis, nbasis);
                beta.mo_energies     = Eigen::VectorXd::Zero(nbasis);
                beta.mo_coefficients = Eigen::MatrixXd::Zero(nbasis, nbasis);
            }

            is_init = true;
        }

    };

    struct InfoSCF
    {
        double _energy              = 0;        // SCF Energy in Hartree
        double _delta_energy        = 0;        // Difference in SCF energy
        double _delta_density_max   = 0;        // Difference in Max SCF Density
        double _delta_density_rms   = 0;        // Difference in RMS SCF Density
        bool _is_converged          = false;    // Is convergence signalled
        DataSCF _scf;                           // SCF Data for current step
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
        std::deque<Eigen::MatrixXd> fock_history;   // stored F matrices (AO basis)
        std::deque<Eigen::MatrixXd> error_history;  // stored error matrices (orthonormal)
        std::size_t max_vecs = 8;                   // maximum subspace size

        // Append a new (F, e) pair, evicting the oldest if at capacity.
        void push(const Eigen::MatrixXd& F, const Eigen::MatrixXd& e)
        {
            fock_history.push_back(F);
            error_history.push_back(e);
            if (fock_history.size() > max_vecs)
            {
                fock_history.pop_front();
                error_history.pop_front();
            }
        }

        std::size_t size() const noexcept { return fock_history.size(); }

        bool ready() const noexcept { return size() >= 2; }

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
            if (error_history.empty()) return 0.0;
            const auto& e = error_history.back();
            return std::sqrt(e.squaredNorm() / static_cast<double>(e.size()));
        }

        void clear() noexcept
        {
            fock_history.clear();
            error_history.clear();
        }
    };

    struct Calculator
    {
        OptionsSCF      _scf;
        OptionsBasis    _basis;
        OptionsGeometry _geometry;
        OptionsIntegral _integral;
        OptionsOutput   _output;
        InfoSCF         _info;
        Molecule        _molecule;
        Basis           _shells;
        
        CalculationType _calculation = CalculationType::SinglePoint;    // Default is Single point energy calculation
        PostHF          _correlation = PostHF::None;                    // No Post HF corrections
        
        double          _total_energy      = 0;    // Total Energy (SCF + Nuclear Repulsion)
        double          _nuclear_repulsion = 0;    // Nuclear Repulsion Energy (Bohr)

        Eigen::MatrixXd _overlap;   // Overlap matrix S
        Eigen::MatrixXd _hcore;     // Core Hamiltonian H = T + V

        void _compute_nuclear_repulsion() noexcept
        {
            const std::size_t N = _molecule.natoms;
            double E_nuc = 0.0;
            for (std::size_t a = 0; a < N; a++) {
                for (std::size_t b = a + 1; b < N; b++) {
                    const double Za = static_cast<double>(_molecule.atomic_numbers[a]);
                    const double Zb = static_cast<double>(_molecule.atomic_numbers[b]);
                    const double dx = _molecule._standard(a, 0) - _molecule._standard(b, 0);
                    const double dy = _molecule._standard(a, 1) - _molecule._standard(b, 1);
                    const double dz = _molecule._standard(a, 2) - _molecule._standard(b, 2);
                    E_nuc += Za * Zb / std::sqrt(dx*dx + dy*dy + dz*dz);
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
        const Eigen::VectorXd _bohr_to_angstrom() const noexcept
        {
            return _molecule._standard * 0.529177210903;
        }
        
        // Run Calculation
        std::expected <void, std::string> initialize()
        {
            if (!_info._scf.is_init)
            {
                // First set the spin channel information
                _info._scf  = DataSCF(_scf._scf == SCFType::UHF);
                
                // Now initialize the matrices
                _info._scf.initialize(_shells.nbasis());
                
                // Set SCF Mode
                _scf.set_scf_mode_auto(_shells.nbasis());
                
                // Set Max SCF cycles
                _scf.set_max_cycles_auto(_shells.nbasis());
            }
            
            // prepare_coordinates() must have been called before initialize().
            // Nothing to do here for coordinate conversion.

            _compute_nuclear_repulsion();

            return {};
        }
    };
}

#endif // !HF_TYPES_H
