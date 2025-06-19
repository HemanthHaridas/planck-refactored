#pragma once

// Standard library includes for file I/O, data structures, and string processing
#include <unordered_map> // Hash map for efficient atomic data lookups
#include <string>        // Standard string class for atom names
#include <vector>        // Dynamic arrays for storing parsed data
#include <ranges>        //

// Planck-specific includes for molecule geometry
#include "planck_geometry.hpp"

/**
 * @namespace Planck::Interface
 * @brief Interface layer for the Planck quantum chemistry software package
 *
 * This namespace provides a hierarchical interface system for configuring and managing
 * quantum chemistry calculations. It implements a parameter management system that
 * allows users to specify calculation settings, control parameters, and molecular
 * geometry through a unified interface.
 */
namespace Planck::Interface
{
    /**
     * @namespace Defaults
     * @brief Default values and constants for quantum chemistry calculations
     *
     * Contains all default parameters used throughout the Planck software.
     * These values represent commonly used settings for quantum chemistry
     * calculations and can be overridden by user input.
     */
    namespace Defaults
    {
        // Unit conversion constants
        const std::double_t ANGTOBOHR = 1.8897259886; /// Conversion factor from Angstroms to Bohr radii (atomic units)
        const std::double_t BOHRTOANG = 0.52917725;   /// Conversion factor from Bohr radii (atomic units) to Angstroms

        // Iteration limits for convergence algorithms
        const std::uint64_t MAXSCF = 120;  /// Maximum number of Self-Consistent Field (SCF) iterations
        const std::uint64_t MAXITER = 120; /// Maximum number of general iterations (e.g., geometry optimization)
        const std::uint64_t DIIS_DIM = 10; /// Dimension of DIIS (Direct Inversion of Iterative Subspace) history

        // Convergence tolerance parameters
        const std::double_t TOLSCF = 1.0E-14; /// SCF convergence tolerance (energy change threshold)
        const std::double_t TOLERI = 1.0E-14; /// Electron repulsion integral screening tolerance

        // Default calculation settings
        const std::string DEFAULT_BASIS         = "sto-3g";  /// Default basis set (Slater-Type Orbital, 3 Gaussians)
        const std::string DEFAULT_THEORY        = "rhf";     /// Default theory level (Restricted Hartree-Fock)
        const std::string DEFAULT_CALC          = "energy";  /// Default calculation type (single-point energy)
        const std::string DEFAULT_COORD         = "ang";     /// Default coordinate units (Angstroms)
        const std::uint64_t DEFAULT_CHARGE      = 0;         /// Default charge (neutral)
        const std::int64_t DEFAULT_MULTIPLICITY = 1;         /// Default multiplicity (singlet)

        // Algorithm control flags
        const bool USE_DIIS = true; /// Enable DIIS acceleration for SCF convergence by default
        const bool USE_SYMM = true; /// Enable molecular symmetry detection and utilization by default

        // Mathematical function limits
        // NOTE: This constant defines the maximum supported argument for Boys function calculations
        // used in electron repulsion integral evaluation. Changing this value requires regenerating
        // the corresponding lookup tables.
        const std::uint64_t MAXM = 60; /// Maximum Boys function index supported by lookup tables
    }

    /**
     * @class BaseInterface
     * @brief Abstract base class for parameter management interfaces
     *
     * Provides a common foundation for all interface classes in the Planck system.
     * Implements a generic parameter storage and retrieval system using string-based
     * key-value pairs with type-safe conversion capabilities.
     *
     * Design Pattern: Template Method Pattern
     * - Defines the skeleton of parameter management operations
     * - Delegates specific parameter initialization to derived classes
     */
    class BaseInterface
    {
    protected:
        /// Internal storage for configuration parameters as key-value string pairs
        std::unordered_map<std::string, std::string> _options;

    public:
        /// Default constructor creates an empty parameter set
        BaseInterface() = default;

        /**
         * @brief Constructor with initial parameter set
         * @param options Pre-configured parameter map
         */
        explicit BaseInterface(const std::unordered_map<std::string, std::string> &options) : _options(options) {}

        /**
         * @brief Generic template method for type-safe parameter retrieval
         * @tparam T Target type for parameter conversion
         * @param key Parameter name to retrieve
         * @param default_value Value to return if parameter is not found
         * @return Parameter value converted to type T, or default_value if not found
         * @throws std::invalid_argument if parameter exists but cannot be converted to type T
         *
         * This template method provides automatic type conversion from string storage
         * to any type that supports stream extraction operator (>>).
         */
        template <typename T>
        T get_value(const std::string &key, const T &default_value) const
        {
            auto iterator_ = _options.find(key);
            if (iterator_ == _options.end())
                return default_value;

            std::istringstream buffer_(iterator_->second);
            T value;
            if (!(buffer_ >> value))
                throw std::invalid_argument("Failed to convert " + iterator_->second + " to target type");
            return value;
        }

        /**
         * @brief Template specialization for boolean parameter retrieval
         * @param key Parameter name to retrieve
         * @param default_value Default boolean value if parameter not found
         * @return Boolean value parsed from string representation
         * @throws std::invalid_argument if parameter exists but is not a valid boolean representation
         *
         * Recognizes the following as valid boolean representations:
         * - true, 1 → true
         * - false, 0 → false
         *
         * Note: The commented line suggests future support for case-insensitive parsing
         */
        template <>
        bool get_value(const std::string &key, const bool &default_value) const
        {
            auto iterator_ = _options.find(key);
            if (iterator_ == _options.end())
                return default_value;

            std::string value = iterator_->second;
            std::transform(value.begin(), value.end(), value.begin(), ::toupper);
            if (value == "true" || value == "1")
                return true;
            if (value == "false" || value == "0")
                return false;
            throw std::invalid_argument("Invalid Boolean argument for " + key);
        }

        /**
         * @brief Accessor for the complete parameter set
         * @return Copy of the internal parameter map
         *
         * Useful for debugging, serialization, or passing parameters to other components
         */
        std::unordered_map<std::string, std::string> get_input_parameters()
        {
            return _options;
        }

        /**
         * @brief Pure virtual method for parameter initialization
         *
         * Derived classes must implement this method to define how they extract
         * and validate their specific parameters from the generic parameter store.
         * This enforces a consistent initialization pattern across all interface types.
         */
        virtual void set_parameters_from_input() = 0;
        virtual void set_default_parameters() = 0;
    };

    /**
     * @class SetupInterface
     * @brief Interface for general calculation setup and configuration
     *
     * Manages high-level calculation parameters including:
     * - Calculation type (energy, optimization, frequency analysis, etc.)
     * - Quantum mechanical theory level (HF, DFT, MP2, etc.)
     * - Basis set specification
     * - Coordinate system and units
     * - Algorithm selection flags
     *
     * This class encapsulates the "what" and "how" of the calculation.
     */
    class SetupInterface : public BaseInterface
    {
    private:
        // Core calculation parameters
        std::string _calc_type; /// Type of calculation to perform (energy, optimization, frequency, etc.)
        std::string _theory;    /// Quantum mechanical theory level (rhf, uhf, dft, mp2, etc.)
        std::string _basis;     /// Basis set specification (sto-3g, 6-31g, cc-pvdz, etc.)
        std::string _coor_type; /// Coordinate system type (cartesian, internal, z-matrix, etc.)

        // Algorithm control flags
        bool _use_diis; /// Enable/disable DIIS convergence acceleration
        bool _use_symm; /// Enable/disable molecular symmetry utilization

        /// Initialization state tracking to prevent use before parameter setup
        bool _is_initialized_by_user = false;

    public:
        /// Default constructor - parameters must be set later via set_parameters_from_input()
        SetupInterface() = default;

        /**
         * @brief Constructor with parameter vectors
         * @param keys Vector of parameter names
         * @param values Vector of parameter values (must match keys length)
         *
         * Convenience constructor that builds the parameter map from parallel vectors.
         * Useful when parameters come from command-line arguments or file parsing.
         */
        explicit SetupInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values)
            : BaseInterface(build_map(keys, values)) {}

        /**
         * @brief Initialize setup parameters from the parameter store
         * @override BaseInterface::set_parameters_from_input()
         *
         * Extracts and validates all setup-related parameters from the internal
         * parameter map. Uses default values for any missing parameters.
         * Sets the initialization flag to indicate the object is ready for use.
         */
        void set_parameters_from_input() override
        {
            _calc_type = get_value<std::string>("CALC_TYPE", Defaults::DEFAULT_CALC);
            _theory    = get_value<std::string>("THEORY", Defaults::DEFAULT_THEORY);
            _basis     = get_value<std::string>("BASIS", Defaults::DEFAULT_BASIS);
            _coor_type = get_value<std::string>("COOR_TYPE", Defaults::DEFAULT_COORD);
            _use_diis  = get_value<bool>("USE_DIIS", Defaults::USE_DIIS);
            _use_symm  = get_value<bool>("USE_SYMM", Defaults::USE_SYMM);

            _is_initialized_by_user = true;
        }

        void set_default_parameters() override
        {
            _calc_type = Defaults::DEFAULT_CALC;
            _theory    = Defaults::DEFAULT_THEORY;
            _basis     = Defaults::DEFAULT_BASIS;
            _coor_type = Defaults::DEFAULT_COORD;
            _use_diis  = Defaults::USE_DIIS;
            _use_symm  = Defaults::USE_SYMM;
        }

    private:
        /**
         * @brief Utility function to construct parameter map from parallel vectors
         * @param keys Vector of parameter names
         * @param values Vector of parameter values
         * @return Unordered map of key-value pairs
         * @throws std::invalid_argument if vectors have different sizes
         *
         * Static helper function that validates input and constructs the parameter map.
         * Used by the vector-based constructor to ensure data consistency.
         */
        static std::unordered_map<std::string, std::string> build_map(const std::vector<std::string> &keys, const std::vector<std::string> &values)
        {
            if (keys.size() != values.size())
                throw std::invalid_argument("Mismatched number of keys and values");

            std::unordered_map<std::string, std::string> result;
            for (size_t i = 0; i < keys.size(); ++i)
                result[keys[i]] = values[i];
            return result;
        }
    };

    /**
     * @class ControlInterface
     * @brief Interface for numerical control and convergence parameters
     *
     * Manages the numerical aspects of quantum chemistry calculations:
     * - Iteration limits for various algorithms
     * - Convergence tolerance criteria
     * - Algorithm-specific parameters (e.g., DIIS history size)
     *
     * This class encapsulates the "how precisely" and "how long to try" aspects
     * of the calculation, providing fine-grained control over numerical behavior.
     */
    class ControlInterface : public BaseInterface
    {
    private:
        // Iteration control parameters
        std::uint64_t _max_scf;  /// Maximum SCF iterations before convergence failure
        std::uint64_t _max_iter; /// Maximum iterations for other iterative procedures
        std::uint64_t _diis_dim; /// Number of previous iterations to store for DIIS extrapolation

        // Convergence tolerance parameters
        std::double_t _tol_scf; /// SCF energy convergence threshold (Hartree)
        std::double_t _tol_eri; /// Electron repulsion integral screening threshold

        /// Initialization state tracking
        bool _is_initialized_by_user = false;

    public:
        /// Default constructor
        ControlInterface() = default;

        /**
         * @brief Constructor with parameter vectors
         * @param keys Vector of parameter names
         * @param values Vector of parameter values
         */
        explicit ControlInterface(const std::vector<std::string> &keys, const std::vector<std::string> &values)
            : BaseInterface(build_map(keys, values)) {}

        /**
         * @brief Initialize control parameters from the parameter store
         * @override BaseInterface::set_parameters_from_input()
         *
         * Extracts numerical control parameters and applies appropriate defaults.
         * All parameters are optional - defaults provide reasonable values for
         * most quantum chemistry calculations.
         */
        void set_parameters_from_input() override
        {
            _max_scf  = get_value<std::uint64_t>("MAX_SCF", Defaults::MAXSCF);
            _max_iter = get_value<std::uint64_t>("MAX_ITER", Defaults::MAXITER);
            _diis_dim = get_value<std::uint64_t>("DIIS_DIM", Defaults::DIIS_DIM);

            _tol_scf = get_value<std::double_t>("TOL_SCF", Defaults::TOLSCF);
            _tol_eri = get_value<std::double_t>("TOL_ERI", Defaults::TOLERI);

            _is_initialized_by_user = true;
        }

        void set_default_parameters() override
        {
            _max_scf  = Defaults::MAXSCF;
            _max_iter = Defaults::MAXITER;
            _diis_dim = Defaults::DIIS_DIM;
            _tol_scf  = Defaults::TOLSCF;
            _tol_eri  = Defaults::TOLERI;
        }

    private:
        /**
         * @brief Utility function to construct parameter map from parallel vectors
         * @param keys Vector of parameter names
         * @param values Vector of parameter values
         * @return Unordered map of key-value pairs
         * @throws std::invalid_argument if vectors have different sizes
         */
        static std::unordered_map<std::string, std::string> build_map(const std::vector<std::string> &keys, const std::vector<std::string> &values)
        {
            if (keys.size() != values.size())
                throw std::invalid_argument("Mismatched number of keys and values");

            std::unordered_map<std::string, std::string> result;
            for (size_t i = 0; i < keys.size(); ++i)
                result[keys[i]] = values[i];
            return result;
        }
    };

    /**
     * @class GeometryInterface
     * @brief Interface for molecular geometry and electronic structure parameters
     *
     * Manages molecular structure and electronic state specification:
     * - Molecular geometry (atomic positions and types)
     * - Electronic multiplicity (spin state)
     * - Molecular charge
     *
     * This class bridges the interface system with the geometry management
     * system, encapsulating the "what molecule" aspect of the calculation.
     */
    class GeometryInterface : public BaseInterface
    {
    private:
        // Electronic structure parameters
        std::uint64_t _multiplicity; /// Spin multiplicity (2S+1, where S is total spin)
        std::int64_t _charge;        /// Net molecular charge (can be negative, zero, or positive)

        /// Molecular geometry object containing atomic coordinates and types
        Planck::Geometry::Molecule _molecule;

    public:
        /// Default constructor - creates empty molecule
        GeometryInterface() = default;

        /**
         * @brief Constructor with molecular geometry data
         * @param keys Vector of parameter names
         * @param values Vector of parameter values
         * @param atoms Vector of atomic symbols/types
         * @param coords Vector of atomic coordinates (3D positions)
         *
         * Constructs both the parameter interface and the molecular geometry
         * object in a single operation. The geometry data is passed directly
         * to the Molecule constructor.
         */
        explicit GeometryInterface(const std::vector<std::string> &keys,
                                   const std::vector<std::string> &values,
                                   std::vector<std::string> &atoms,
                                   std::vector<Eigen::Vector3d> &coords)
            : BaseInterface(build_map(keys, values)), _molecule(atoms, coords) {}

        /**
         * @brief Initialize geometry-related parameters from the parameter store
         * @override BaseInterface::set_parameters_from_input()
         *
         * Extracts electronic structure parameters. The molecular geometry
         * itself is handled separately through the constructor or direct
         * manipulation of the _molecule member.
         */
        void set_parameters_from_input() override
        {
            _multiplicity = get_value<std::uint64_t>("MULTI", Defaults::DEFAULT_MULTIPLICITY); // Default: singlet state
            _charge = get_value<std::int64_t>("CHARGE", Defaults::DEFAULT_CHARGE);       // Default: neutral molecule
        }

        void set_default_parameters() override
        {
            _multiplicity = Defaults::DEFAULT_MULTIPLICITY;
            _charge       = Defaults::DEFAULT_CHARGE;
        }

        /**
         * @brief Returns a view of the current coordinates
         *
         * This method is essential for:
         * - Geometry optimization algorithms
         * - Molecular dynamics simulations
         */
        std::vector<std::tuple<std::string, Eigen::Vector3d>> get_coordinates() const
        {
            return _molecule.get_coordinates();
        }

        void generate_distance_matrix()
        {
            _molecule.create_connetivity_table();
        }

        Eigen::MatrixXd get_distance_matrix() const
        {
            return _molecule.get_connectity_table();
        }

        void convert_coords_to_bohr()
        {
            _molecule.convert_coords_to_bohr();
        }

        void convert_coords_to_angstrom()
        {
            _molecule.convert_coords_to_angstrom();
        }
        
    private:
        /**
         * @brief Utility function to construct parameter map from parallel vectors
         * @param keys Vector of parameter names
         * @param values Vector of parameter values
         * @return Unordered map of key-value pairs
         * @throws std::invalid_argument if vectors have different sizes
         */
        static std::unordered_map<std::string, std::string> build_map(const std::vector<std::string> &keys, const std::vector<std::string> &values)
        {
            if (keys.size() != values.size())
                throw std::invalid_argument("Mismatched number of keys and values");

            std::unordered_map<std::string, std::string> result;
            for (size_t i = 0; i < keys.size(); ++i)
                result[keys[i]] = values[i];
            return result;
        }
    };

};