#ifndef HF_TYPES_H
#define HF_TYPES_H

#include <span>
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <expected>
#include <Eigen/Core>

#include "tables.h"
#include "basis.h"

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
        double zeta;                // gaussian exponent
        double inv_zeta;            // 1 / gaussian exponent
        double prefactor;           // gaussian integral
        double coeff_product;       // c_i * c_j
        Eigen::Vector3d center;     // Gaussian product center
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
                    pp.zeta         = zeta;
                    pp.inv_zeta     = inv_zeta;
                    pp.coeff_product= cA * cB;
                    pp.prefactor    = std::exp(-alpha_beta_over_zeta * R2);

                    // Gaussian product center
                    pp.center = (alpha * A._shell->_center + beta * B._shell->_center) * inv_zeta;

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
        unsigned int _iteration     = 0;        // Number of iterations
        double _energy              = 0;        // SCF Energy in Hartree
        double _delta_energy        = 0;        // Difference in SCF energy
        double _delta_density_max   = 0;        // Difference in Max SCF Density
        double _delta_density_rms   = 0;        // Difference in RMS SCF Density
        bool _is_converged          = false;    // Is convergence signalled
        DataSCF _scf;                           // SCF Data for current step
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
        
        double          _total_energy       = 0;    // Total Energy (SCF + Nuclear Repulsion)
        double          _nuclear_replusion  = 0;    // Nuclear Repulsion
        
    private:
        // Setter
        void _angstrom_to_bohr() noexcept
        {
            _molecule._coordinates = _molecule.coordinates * ANGSTROM_TO_BOHR;
        }
        
//        void _nuclear_repulsion() noexcept
//        {
//            // Compute distances in x, y and z directions
//            Eigen::MatrixXd dx = _molecule.standard.col(0).rowwise() - _molecule.standard.col(0).transpose();
//            Eigen::MatrixXd dy = _molecule.standard.col(1).rowwise() - _molecule.standard.col(1).transpose();
//            Eigen::MatrixXd dz = _molecule.standard.col(2).rowwise() - _molecule.standard.col(2).transpose();
//            
//            // Compute total distance as (N,N) matrix
//            Eigen::MatrixXd ds = (dx.array().square() + dy.array().square() + dz.array().square()).sqrt();
//            
//            // Set diagonal to infinity to avoid division by zero
//            ds.diagonal().setConstant(std::numeric_limits<double>::infinity());
//            
//            // Compute product of charges as (N,N) matrix
//            Eigen::MatrixXd zsq = _molecule.atomic_numbers * _molecule.atomic_numbers.transpose();
//            
//            // Sum the contributions and halve to avoid double counting
//            _nuclear_repulsion = (zsq.array() / ds.array()).sum() * 0.5;
//        }
        
//        std::expected <void, std::string> _hartree_fock()
//        {
//            // Compute Overlap
//            // Compute Kinetic
//            // Compute Electron - Nuclear Attraction
//            // Compute Electron - Electron Repulsion
//            // Populate Data
//        }
        
    public:
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
            }
            
            // Check if coordinates are in Bohr
            _molecule._is_bohr = (_geometry._units == Units::Bohr);
            
            if (!_molecule._is_bohr)
            {
                _angstrom_to_bohr();
                _molecule._is_bohr = true;
            }
            
            return {};
        }
    };
}

#endif // !HF_TYPES_H
