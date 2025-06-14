#pragma once

// Standard library includes for exception handling and string manipulation
#include <exception>  // Base std::exception class for proper exception hierarchy
#include <string>     // String class for error message storage and manipulation

/**
 * @namespace Planck::Exceptions
 * @brief Comprehensive exception handling system for the Planck quantum chemistry package
 * 
 * This namespace provides a structured exception hierarchy designed specifically for
 * quantum chemistry applications. It categorizes errors by computational domain:
 * - IO: File operations, data parsing, input/output errors
 * - Geometry: Molecular structure, coordinate validation, atom property errors
 * - SCF: Self-Consistent Field convergence, orbital computation errors
 * - Optimization: Geometry optimization, energy minimization failures
 * 
 * The design follows the standard C++ exception hierarchy, inheriting from std::exception
 * to ensure compatibility with standard exception handling mechanisms.
 */
namespace Planck::Exceptions
{
    /**
     * @enum ExceptionTypes
     * @brief Categorizes different types of computational errors in quantum chemistry
     * 
     * This enumeration provides a systematic way to classify errors based on the
     * computational domain where they occur. This classification is useful for:
     * - Error reporting and logging systems
     * - Automated error recovery strategies
     * - Debugging and diagnostic tools
     * - User interface error message formatting
     */
    enum class ExceptionTypes
    {
        IO,   ///< Input/Output operations (file reading/writing, data parsing)
        Geom, ///< Geometry-related operations (coordinates, molecular structure)
        SCF,  ///< Self-Consistent Field calculations (convergence, orbital errors)
        Opt   ///< Optimization procedures (geometry optimization, energy minimization)
    };

    /**
     * @class BaseException
     * @brief Abstract base class for all Planck-specific exceptions
     * 
     * This class serves as the foundation for the entire exception hierarchy,
     * providing common functionality for all derived exception types:
     * 
     * Key Design Features:
     * - Inherits from std::exception for standard compatibility
     * - Stores both error messages and error categories
     * - Provides consistent error message formatting
     * - Enables polymorphic exception handling
     * 
     * The class uses composition to combine a descriptive error message with
     * a categorical error type, allowing for both human-readable diagnostics
     * and programmatic error classification.
     */
    class BaseException : public std::exception
    {
    private:
        std::string _error;        ///< Human-readable error message describing the specific problem
        ExceptionTypes _exception; ///< Categorical classification of the error type

    public:
        /**
         * @brief Default constructor
         * Creates an exception with uninitialized values - typically used by derived classes
         * that will set appropriate values through their own constructors
         */
        BaseException() = default; // default constructor

        /**
         * @brief Explicit constructor for complete exception initialization
         * @param message Descriptive error message explaining what went wrong
         * @param exception Category of error (IO, Geom, SCF, or Opt)
         * 
         * The explicit keyword prevents accidental implicit conversions from strings
         * to exceptions, which could mask programming errors. The message is moved
         * for efficiency when possible, avoiding unnecessary string copies.
         * 
         * @note This constructor is typically called by derived classes rather than
         *       directly by user code, as the derived classes provide domain-specific
         *       error prefixes and appropriate exception type categorization.
         */
        explicit BaseException(const std::string &message, ExceptionTypes exception) 
            : _error(std::move(message)), _exception(exception) {}

        /**
         * @brief Returns the error message as a C-style string
         * @return Null-terminated character array containing the error message
         * 
         * This method overrides std::exception::what() to provide the standard
         * interface for exception message retrieval. The noexcept specification
         * guarantees that this method will never throw an exception itself,
         * which is critical for exception safety in error handling code.
         * 
         * The returned pointer remains valid for the lifetime of the exception object.
         */
        const char *what() const noexcept override { return _error.c_str(); }
    };

    /**
     * @class IOException
     * @brief Handles exceptions related to input/output operations
     * 
     * This exception class is specialized for errors that occur during:
     * - File reading and writing operations
     * - Data format parsing (XYZ files, basis sets, etc.)
     * - Network communications (if applicable)
     * - Database access operations
     * - Configuration file processing
     * 
     * Common scenarios that trigger IOException:
     * - Missing or inaccessible input files
     * - Corrupted or malformed data files
     * - Insufficient disk space for output
     * - Network connectivity issues
     * - Permission denied errors
     * 
     * The class automatically prefixes error messages with "IOError : " to
     * clearly identify the error category in log files and user interfaces.
     */
    class IOException : public BaseException
    {
    public:
        /**
         * @brief Constructs an IO exception with descriptive message
         * @param message Specific description of the IO error that occurred
         * 
         * The constructor automatically:
         * - Prefixes the message with "IOError : " for clear categorization
         * - Sets the exception type to ExceptionTypes::IO
         * - Ensures noexcept guarantee for exception safety
         * 
         * Example usage:
         * @code
         * if (!file.is_open()) {
         *     throw IOException("Cannot open input file: " + filename);
         * }
         * @endcode
         */
        explicit IOException(const std::string &message) noexcept 
            : BaseException("IOError : " + message, ExceptionTypes::IO) {}
    };

    /**
     * @class GeomException
     * @brief Handles exceptions related to molecular geometry operations
     * 
     * This exception class is specialized for errors in molecular structure handling:
     * - Invalid atomic coordinates or atom types
     * - Geometric constraint violations
     * - Molecular structure validation failures
     * - Coordinate transformation errors
     * - Bond length/angle validation issues
     * 
     * Common scenarios that trigger GeomException:
     * - Unrecognized element symbols in input
     * - Atoms with undefined or invalid coordinates
     * - Geometric parameters outside physical bounds
     * - Symmetry detection failures
     * - Molecular fragmentation issues
     * 
     * This class is extensively used by the geometry classes in planck_geometry.hpp
     * to provide clear feedback when molecular structure operations fail.
     */
    class GeomException : public BaseException
    {
    public:
        /**
         * @brief Constructs a geometry exception with descriptive message
         * @param message Specific description of the geometry error that occurred
         * 
         * Automatically prefixes messages with "GeomError : " and categorizes
         * the exception as ExceptionTypes::Geom for consistent error handling.
         * 
         * Example usage:
         * @code
         * if (atoms.size() != coordinates.size()) {
         *     throw GeomException("Mismatch between atom count and coordinate count");
         * }
         * @endcode
         */
        explicit GeomException(const std::string &message) noexcept 
            : BaseException("GeomError : " + message, ExceptionTypes::Geom) {}
    };

    /**
     * @class SCFException
     * @brief Handles exceptions in Self-Consistent Field calculations
     * 
     * This exception class is specialized for errors in quantum chemical calculations:
     * - SCF convergence failures
     * - Orbital computation errors
     * - Basis set problems
     * - Integral evaluation failures
     * - Matrix diagonalization issues
     * - Density matrix problems
     * 
     * Common scenarios that trigger SCFException:
     * - Maximum iteration count exceeded without convergence
     * - Numerical instabilities in orbital calculations
     * - Inappropriate basis set for the molecular system
     * - Linear dependency in basis functions
     * - Memory allocation failures for large matrices
     * 
     * SCF calculations are often the computational bottleneck in quantum chemistry,
     * so robust error handling is critical for user experience and debugging.
     */
    class SCFException : public BaseException
    {
    public:
        /**
         * @brief Constructs an SCF exception with descriptive message
         * @param message Specific description of the SCF computational error
         * 
         * Automatically prefixes messages with "SCFError : " and categorizes
         * the exception as ExceptionTypes::SCF for systematic error handling.
         * 
         * Example usage:
         * @code
         * if (iteration_count > max_iterations) {
         *     throw SCFException("SCF failed to converge after " + 
         *                       std::to_string(max_iterations) + " iterations");
         * }
         * @endcode
         */
        explicit SCFException(const std::string &message) noexcept 
            : BaseException("SCFError : " + message, ExceptionTypes::SCF) {}
    };

    /**
     * @class OptException
     * @brief Handles exceptions in geometry optimization procedures
     * 
     * This exception class is specialized for errors in molecular optimization:
     * - Geometry optimization convergence failures
     * - Energy minimization problems
     * - Gradient calculation errors
     * - Step size determination issues
     * - Constraint violation problems
     * - Transition state search failures
     * 
     * Common scenarios that trigger OptException:
     * - Optimization stuck in local minima
     * - Gradient calculations producing NaN or infinite values
     * - Step sizes becoming too small to make progress
     * - Constraint satisfaction failures
     * - Hessian matrix computation problems
     * - Maximum optimization cycles exceeded
     * 
     * Geometry optimization is crucial for obtaining reliable molecular structures
     * and properties, making robust error handling essential for practical use.
     */
    class OptException : public BaseException
    {
    public:
        /**
         * @brief Constructs an optimization exception with descriptive message
         * @param message Specific description of the optimization error that occurred
         * 
         * Automatically prefixes messages with "OptError : " and categorizes
         * the exception as ExceptionTypes::Opt for consistent error classification.
         * 
         * Example usage:
         * @code
         * if (gradient_norm < convergence_threshold && cycle_count > max_cycles) {
         *     throw OptException("Geometry optimization failed to converge: " +
         *                       "gradient norm = " + std::to_string(gradient_norm));
         * }
         * @endcode
         */
        explicit OptException(const std::string &message) noexcept 
            : BaseException("OptError : " + message, ExceptionTypes::Opt) {}
    };
};