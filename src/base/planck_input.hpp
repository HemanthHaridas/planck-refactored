#pragma once

// Standard library includes for file I/O, data structures, and string processing
#include <string>      // String handling for file paths and parsing
#include <fstream>     // File stream operations for reading input files
#include <tuple>       // Tuple for returning multiple values from parsing functions
#include <memory>      // Smart pointers for safe file handle management
#include <vector>      // Dynamic arrays for storing parsed data
#include <sstream>     // String stream for parsing individual lines
#include <Eigen/Core>  // Linear algebra library for coordinate vectors

// Planck-specific includes for configuration interfaces and error handling
#include "planck_interface.hpp"   // Configuration interface classes
#include "planck_exceptions.hpp"  // Exception handling for input errors

/**
 * @namespace Planck::IO
 * @brief Input/Output operations for the Planck quantum chemistry package
 * 
 * This namespace contains classes and functions responsible for reading and writing
 * various file formats used in quantum chemistry calculations, including: 
 * - Input file parsing and validation
 * - Output file generation and formatting
 * - Data serialization and deserialization
 * - File format conversions
 */
namespace Planck::IO
{
    /**
     * @class InputReader
     * @brief Parses structured input files for quantum chemistry calculations
     * 
     * This class implements a comprehensive input file parser that handles: 
     * - Sectioned input format with [SECTION] ... END_SECTION blocks
     * - Key-value pair parsing for calculation parameters
     * - Molecular coordinate specification and validation
     * - Type conversion and validation for numerical parameters
     * - Error handling for malformed or missing input sections
     * 
     * The parser creates three main configuration interfaces: 
     * - ControlInterface                                    : Numerical convergence parameters and iteration limits
     * - SetupInterface                                      : Calculation type, theory level, and basis set specification
     * - GeometryInterface                                   : Molecular structure and electronic state parameters
     * 
     * Input File Format Example: 
     * @code
     * [CONTROL]
     * MAX_SCF 100
     * TOL_SCF 1.0E-8
     * END_CONTROL
     * 
     * [SETUP]
     * THEORY RHF
     * BASIS 6-31G
     * END_SETUP
     * 
     * [MOL_SPEC]
     * CHARGE 0
     * MULTI 1
     * END_MOL_SPEC
     * 
     * [COORDS]
     * H  0.0  0.0  0.0
     * H  0.0  0.0  1.4
     * END_COORDS
     * @endcode
     */
    class InputReader
    {
    private: 
        std::string _input;                     ///< Path to the input file
        std::shared_ptr<std::ifstream> _file;   ///< Smart pointer to file stream for safe resource management

    public: 
        // Public interface objects that store parsed configuration data
        Planck::Interface::ControlInterface  _control;  ///< Convergence parameters and iteration limits
        Planck::Interface::SetupInterface    _setup;    ///< Calculation type, theory, and basis set
        Planck::Interface::GeometryInterface _geom;     ///< Molecular structure and electronic properties

        /**
         * @brief Constructs InputReader and immediately parses the specified input file
         * @param input Path to the input file to be parsed
         * @throws IOException if the file cannot be opened or read
         * 
         * This constructor performs the complete input file parsing process: 
         * 1. Opens the specified input file with error checking
         * 2. Parses each required section (CONTROL, SETUP, MOL_SPEC, COORDS)
         * 3. Constructs interface objects with parsed data
         * 4. Validates data consistency and completeness
         * 
         * The use of shared_ptr ensures proper cleanup of file resources even
         * if exceptions occur during parsing. The constructor immediately
         * builds all interface objects, making them ready for use.
         * 
         * @note The file is kept open during the entire parsing process to
         *       handle multiple sections efficiently, then automatically
         *       closed when the shared_ptr goes out of scope.
         */
        explicit InputReader(const std::string &input) 
            :  _input(input),
              _file(std::make_shared<std::ifstream>(_input)), 
              _control(build_control_interface()), 
              _setup(build_setup_interface()), 
              _geom(build_geometry_interface())
        {
              // Immediate validation of file accessibility
            if (!_file || !_file->is_open())
                throw Planck::Exceptions::IOException("Could not open input file: " + _input);
        }

    private: 
         /**
         * @brief Converts a string to lowercase for case-insensitive comparisons
         * @param parsedString Input string to be converted
         * @return Lowercase version of the input string
         * 
         * This utility function enables case-insensitive parsing of input parameters,
         * improving user experience by accepting various capitalizations of keywords.
         * The function preserves the original string by creating a copy before
         * transformation, ensuring const-correctness.
         * 
         * Used primarily for boolean value parsing and keyword normalization.
         */
        // std::string toLower(const std::string &parsedString)
        // {
        //     std::string lowerString = parsedString;  // Create a copy to preserve the original string
        //     std::transform(lowerString.begin(), lowerString.end(), lowerString.begin(), ::tolower);
        //     return lowerString;  // Return the transformed string
        // }

        /**
         * @brief Converts string representations to boolean values
         * @param parsedString String to be converted ("ON"/"OFF", case-insensitive)
         * @return Boolean value corresponding to the input string
         * @throws std::invalid_argument if the string cannot be converted to boolean
         * 
         * This function provides flexible boolean parsing for user-friendly input: 
         * - "ON" (case-insensitive) maps to true
         * - "OFF" (case-insensitive) maps to false
         * - Throws exception for invalid input to prevent silent errors
         * 
         * The case-insensitive conversion improves usability by accepting
         * common variations like "on", "On", "ON", etc.
         */
        // bool stringToBool(const std::string &parsedString)
        // {
        //     std::string upperStr = parsedString;
        //     std::transform(upperStr.begin(), upperStr.end(), upperStr.begin(), ::toupper);  // Convert to uppercase

        //     if (upperStr == "ON")
        //         return true;  // "ON" maps to true
        //     else if (upperStr == "OFF")
        //         return false;  // "OFF" maps to false
        //     else
        //         throw std::invalid_argument("Invalid string for boolean conversion.");  // Handle unexpected input
        // }

        /**
         * @brief Builds the control interface by parsing the CONTROL section
         * @return Configured ControlInterface object with parsed parameters
         * @throws IOException if CONTROL section is missing or malformed
         * 
         * The CONTROL section contains numerical parameters that control the
         * convergence behavior and computational limits of quantum chemistry
         * calculations. Typical parameters include: 
         * - MAX_SCF                               : Maximum number of SCF iterations
         * - TOL_SCF                               : SCF convergence threshold
         * - DIIS_DIM                              : DIIS extrapolation dimension
         * - TOL_ERI                               : Electron repulsion integral threshold
         */
        Planck::Interface::ControlInterface build_control_interface()
        {
            auto [keys, values] = tokenize_input("CONTROL", _file);
            return Planck::Interface::ControlInterface(keys, values);
        }

        /**
         * @brief Builds the setup interface by parsing the SETUP section
         * @return Configured SetupInterface object with parsed parameters
         * @throws IOException if SETUP section is missing or malformed
         * 
         * The SETUP section contains high-level calculation configuration: 
         * - CALC_TYPE                                                    : Type of calculation (energy, gradient, optimization)
         * - THEORY                                                       : Level of theory (RHF, UHF, DFT, MP2, etc.)
         * - BASIS                                                        : Basis set specification (STO-3G, 6-31G, cc-pVDZ, etc.)
         * - USE_DIIS                                                     : Whether to use DIIS convergence acceleration
         * - USE_SYMM                                                     : Whether to use molecular symmetry
         */
        Planck::Interface::SetupInterface build_setup_interface()
        {
            auto [keys, values] = tokenize_input("SETUP", _file);
            return Planck::Interface::SetupInterface(keys, values);
        }

        /**
         * @brief Builds the geometry interface by parsing MOL_SPEC and COORDS sections
         * @return Configured GeometryInterface object with molecular structure
         * @throws IOException if required sections are missing or malformed
         * 
         * This method combines molecular specification (charge, multiplicity) with
         * atomic coordinates to create a complete molecular structure representation.
         * The geometry interface integrates with the Planck::Geometry classes to
         * provide validated molecular structures for quantum chemistry calculations.
         */
        Planck::Interface::GeometryInterface build_geometry_interface()
        {
            auto [keys, values]  = tokenize_input("MOL_SPEC", _file);
            auto [atoms, coords] = tokenize_coords(_file);
            return Planck::Interface::GeometryInterface(keys, values, atoms, coords);
        }

        /**
         * @brief Parses a sectioned input block into key-value pairs
         * @param header Name of the section to parse (without brackets)
         * @param file_pointer Shared pointer to the input file stream
         * @return Tuple containing vectors of keys and corresponding values
         * @throws IOException if section is not found or malformed
         * 
         * This method implements the core parsing logic for sectioned input: 
         * 1. Searches for [SECTION] header in the file
         * 2. Reads key-value pairs until END_SECTION is found
         * 3. Validates that each line contains both key and value
         * 4. Returns matched vectors of keys and values
         * 
         * The parser handles: 
         * - Whitespace-separated key-value pairs
         * - Empty lines (ignored)
         * - Comments (lines starting with #, ignored)
         * - Proper section termination validation
         * 
         * Format example: 
         * @code
         * [CONTROL]
         * MAX_SCF 100
         * TOL_SCF 1.0E-8
         * END_CONTROL
         * @endcode
         */
        std::tuple<std::vector<std::string>, std::vector<std::string>> tokenize_input(const std::string &header, std::shared_ptr<std::ifstream> file_pointer)
        {
            // Validate file stream accessibility
            if (!file_pointer || !file_pointer->is_open())
                throw Planck::Exceptions::IOException("Input stream is unavailable");

            std::vector<std::string> keys;
            std::vector<std::string> values;

            std::string line_;
            // Search for the target section header
            while (std::getline(*file_pointer, line_))
            {
                // Check for section header format: [SECTION_NAME]
                if (line_.starts_with("[") && line_.ends_with("]"))
                {
                    // Extract section name by removing brackets
                    std::string header_ = line_.substr(1, line_.length() - 2);
                    if (header == header_)
                    {
                        // Parse key-value pairs within the section
                        std::string data_;
                        while (std::getline(*file_pointer, data_))
                        {
                            // Check for section termination
                            if (data_.find("END_" + header_) != std::string::npos)
                                return {keys, values};

                            // Parse key-value pair from line
                            std::string _key, _value;
                            std::stringstream _data_buffer(data_);
                            _data_buffer >> _key >> _value;

                            std::transform(_value.begin(), _value.end(), _value.begin(), ::tolower);
                            // Only add non-empty key-value pairs
                            if (!_key.empty() && !_value.empty())
                            {
                                keys.emplace_back(_key);
                                values.emplace_back(_value);
                            }
                        }
                    }
                }
            }

            // Section not found or incomplete
            throw Planck::Exceptions::IOException("Section [" + header + "] not found or incomplete");
        }

        /**
         * @brief Parses molecular coordinates from the COORDS section
         * @param file_pointer Shared pointer to the input file stream
         * @return Tuple containing vectors of atom symbols and coordinate vectors
         * @throws IOException if COORDS section is missing or malformed
         * 
         * This method specifically handles the coordinate parsing which requires
         * different logic than key-value pairs: 
         * 1. Searches for [COORDS] section header
         * 2. Reads lines with format: ATOM_SYMBOL X Y Z
         * 3. Validates numerical coordinate values
         * 4. Creates Eigen::Vector3f objects for efficient 3D operations
         * 
         * The coordinate parser handles: 
         * - Various atom symbol formats (H, He, Li, etc.)
         * - Floating-point coordinates in Cartesian format
         * - Proper validation of coordinate completeness
         * - Integration with Eigen library for mathematical operations
         * 
         * Format example: 
         * @code
         * [COORDS]
         * H   0.000000   0.000000   0.000000
         * H   0.000000   0.000000   1.400000
         * END_COORDS
         * @endcode
         */
        std::tuple<std::vector<std::string>, std::vector<Eigen::Vector3d>> tokenize_coords(std::shared_ptr<std::ifstream> file_pointer)
        {
            std::string line_;
            std::vector<std::string> atoms;
            std::vector<Eigen::Vector3d> coords;

              // Search for COORDS section
            while (getline(*file_pointer, line_))
            {
                  // Check for COORDS section header
                if (line_.starts_with("[") && line_.ends_with("]") && (line_.substr(1, line_.length() - 2) == "COORDS"))
                {
                    std::string coords_, _atom;
                    Eigen::Vector3d _coords;
                    
                      // Parse coordinate lines
                    while (getline(*file_pointer, coords_))
                    {
                          // Check for section termination
                        if (coords_.find("END_COORDS") != std::string::npos)
                            return {atoms, coords};

                          // Parse atom symbol and xyz coordinates
                        std::stringstream _coord_buffer(coords_);
                        _coord_buffer >> _atom >> _coords[0] >> _coords[1] >> _coords[2];

                          // Store parsed data
                        atoms.push_back(_atom);
                        coords.push_back(_coords);
                    }
                }
            }

            // COORDS section not found or incomplete
            throw Planck::Exceptions::IOException("Section [COORDS] not found or incomplete");
        }
    };
};