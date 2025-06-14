#pragma once

// Standard library includes for mathematical operations, string handling, and optional values
#include <Eigen/Core>     // Eigen library for 3D vector operations and linear algebra
#include <string>         // Standard string class for atom names
#include <optional>       // Optional wrapper for coordinates that may not be set
#include <unordered_map>  // Hash map for efficient atomic data lookups

// Custom exception handling for geometry-related errors
#include "planck_exceptions.hpp"

/**
 * @namespace Planck::Geometry
 * @brief Contains classes for representing molecular geometry in quantum chemistry calculations
 * 
 * This namespace provides a hierarchical representation of molecular structures: 
 * - Atom                                                                       : Individual atomic entities with coordinates and properties
 * - Molecule                                                                   : Collections of atoms forming molecular structures
 */
namespace Planck::Geometry
{
    // Forward declaration to resolve circular dependency between Atom and Molecule classes
    class Molecule;  // forward declaration for Molecule class

      /**
     * @class Atom
     * @brief Represents a single atom with its spatial coordinates and chemical properties
     * 
     * This class encapsulates all the essential information about an individual atom: 
     * - 3D spatial coordinates (optional, as they may not always be set initially)
     * - Chemical identity (element symbol)
     * - Atomic number (number of protons)
     * - Atomic mass (standard atomic weight)
     * 
     * The class uses the PIMPL-like pattern with private data members and controlled access
     * through public methods, ensuring data integrity and encapsulation.
     */
    class Atom
    {
    private: 
        // Optional coordinates allow for atoms that haven't been positioned yet
        // Using Eigen::Vector3f for efficient 3D vector operations (single precision)
        std::optional<Eigen::Vector3f> _coordinates;  // atom coordinates in 3D space
        
        std::string   _atom;           // Element symbol (e.g., "H", "C", "O")
        std::uint64_t _atomic_number;  // Z - atomic number (number of protons)
        std::double_t _atomic_mass;    // Standard atomic mass in atomic mass units (amu)

        // Friend class declaration allows Molecule to access private members
        // This enables efficient construction and manipulation of atoms within molecules
        friend class Molecule;

    public: 
        /**
         * @brief Default constructor
         * Creates an atom with uninitialized values - typically used when atoms
         * will be configured later through setter methods or friend class access
         */
        Atom() = default;  // default constructor

        /**
         * @brief Explicit constructor for complete atom initialization
         * @param coord 3D coordinates of the atom
         * @param atom Element symbol (e.g., "H", "C", "N")
         * @param z Atomic number
         * @param mass Atomic mass in amu
         * 
         * Note: Parameters are passed by reference to avoid unnecessary copying,
         * but coordinates are moved for efficiency since Vector3f is relatively small.
         * The explicit keyword prevents implicit conversions.
         */
        explicit Atom(const Eigen::Vector3f &coord, std::string &atom, std::uint64_t &z, std::double_t &mass) 
            :  _coordinates(std::move(coord)), _atom(atom), _atomic_number(z), _atomic_mass(mass) {};

        /**
         * @brief Updates the atom's 3D coordinates
         * @param coord New coordinates for the atom
         * 
         * This method is essential for geometry optimization, molecular dynamics,
         * and other operations that modify atomic positions while preserving
         * chemical identity.
         */
        void set_coordinates(const Eigen::Vector3f coord) { _coordinates = coord; }

        /**
         * @brief Retrieves the atom's current coordinates
         * @return 3D coordinate vector
         * @throws GeomException if coordinates haven't been set
         * 
         * This method enforces that coordinates must be explicitly set before use,
         * preventing errors from uninitialized position data. The exception provides
         * clear feedback about which atom lacks coordinates.
         */
        Eigen::Vector3f coordinates() const
        {
            if (!_coordinates.has_value())
                throw Planck::Exceptions::GeomException("Atom " + _atom + " has no coordinates set");

            return _coordinates.value();
        }
    };

    /**
     * @class Molecule
     * @brief Represents a collection of atoms forming a molecular structure
     * 
     * This class manages groups of atoms and provides functionality for: 
     * - Constructing molecules from atom lists and coordinate arrays
     * - Updating atomic positions (useful for geometry optimization)
     * - Automatic lookup of atomic properties from element symbols
     * 
     * The class maintains internal consistency by ensuring atom names and
     * coordinates are properly matched and validated.
     */
    class Molecule
    {
    private: 
        // Vector container for efficient storage and iteration over atoms
        // Using std::vector provides contiguous memory layout and cache efficiency
        std::vector<Atom> _geometry;

    public: 
          /**
         * @brief Constructs a molecule from atom symbols and coordinates
         * @param atoms Vector of element symbols (e.g., {"H", "O", "H"} for water)
         * @param coords Vector of 3D coordinates corresponding to each atom
         * @throws GeomException if atom and coordinate vector sizes don't match
         * 
         * This constructor performs several important operations: 
         * 1. Validates input consistency (equal vector sizes)
         * 2. Looks up atomic numbers and masses from element symbols
         * 3. Creates properly initialized Atom objects
         * 4. Handles potential lookup failures with informative exceptions
         */
        explicit Molecule(const std::vector<std::string> atoms, const std::vector<Eigen::Vector3f> coords)
        {
            // Input validation - critical for preventing runtime errors
            if (atoms.size() != coords.size())
                throw Planck::Exceptions::GeomException("Mismatch between atom names and coordinate list size");

            // Construct each atom with complete chemical and spatial information
            for (size_t i = 0; i < atoms.size(); i++)
            {
                // Automatic lookup of atomic properties from element symbols
                std::uint64_t z    = get_atomic_number_from_name(atoms[i]);
                std::double_t mass = get_atomic_mass_from_name(atoms[i]);
                
                // const_cast needed due to constructor signature expecting non-const reference
                // This is a design limitation that could be improved in future versions
                Atom atom(coords[i], const_cast<std::string &>(atoms[i]), z, mass);
                _geometry.push_back(atom);
            }
        }

        /**
         * @brief Updates coordinates for all atoms in the molecule
         * @param coords New coordinate vector (must match current atom count)
         * 
         * This method is essential for: 
         * - Geometry optimization algorithms
         * - Molecular dynamics simulations  
         * - Loading new conformations from external sources
         * 
         * Note: No size validation is performed - calling code must ensure
         * the coordinate vector matches the number of atoms.
         */
        void update_coordinates(const std::vector<Eigen::Vector3f> &coords)
        {
            for (size_t i = 0; i < coords.size(); i++)
                _geometry[i].set_coordinates(coords[i]);  // update the coordinates of each atom;
        }

    private: 
        /**
         * @brief Internal lookup function for atomic numbers
         * @param atom Element symbol (case-sensitive)
         * @return Atomic number (number of protons)
         * @throws GeomException if element symbol is not recognized
         * 
         * Uses a static hash map for O(1) lookup performance. The static keyword
         * ensures the map is initialized only once and persists across function calls.
         * Covers elements 1-99 (Hydrogen through Einsteinium).
         */
        std::uint64_t get_atomic_number_from_name(const std::string &atom) const
        {
            // Static initialization ensures this expensive map construction happens only once
            static const std::unordered_map<std::string, std::uint64_t> atomicNumber = {
                {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10}, 
                {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20}, 
                {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30}, 
                {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, 
                {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50}, 
                {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}, 
                {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, 
                {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80}, 
                {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90}, 
                {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {"Am", 95}, {"Cm", 96}, {"Bk", 97}, {"Cf", 98}, {"Es", 99}
            };

            // Efficient hash map lookup with error handling
            auto iterator_ = atomicNumber.find(atom);
            if (iterator_ != atomicNumber.end())
                return iterator_->second;
            
            // Provide clear error message for debugging
            throw Planck::Exceptions::GeomException("No atom called " + atom);
        }

        /**
         * @brief Internal lookup function for atomic masses
         * @param atom Element symbol (case-sensitive)  
         * @return Standard atomic mass in atomic mass units (amu)
         * @throws GeomException if element symbol is not recognized
         * 
         * Provides standard atomic weights for elements 1-99. These values are
         * essential for molecular mass calculations, center of mass computations,
         * and moment of inertia calculations in quantum chemistry applications.
         */
        std::double_t get_atomic_mass_from_name(const std::string &atom) const
        {
            // Standard atomic masses from NIST/IUPAC data
            static const std::unordered_map<std::string, double> atomicMass = {
                {"H", 1.008}, {"He", 4.003}, {"Li", 7.000}, {"Be", 9.012}, {"B", 10.810}, {"C", 12.011}, {"N", 14.007}, {"O", 15.999}, {"F", 18.998}, {"Ne", 20.180}, 
                {"Na", 22.990}, {"Mg", 24.305}, {"Al", 26.982}, {"Si", 28.085}, {"P", 30.974}, {"S", 32.070}, {"Cl", 35.450}, {"Ar", 39.900}, {"K", 39.098}, {"Ca", 40.080}, 
                {"Sc", 44.956}, {"Ti", 47.867}, {"V", 50.942}, {"Cr", 51.996}, {"Mn", 54.938}, {"Fe", 55.840}, {"Co", 58.933}, {"Ni", 58.693}, {"Cu", 63.550}, {"Zn", 65.400}, 
                {"Ga", 69.723}, {"Ge", 72.630}, {"As", 74.922}, {"Se", 78.970}, {"Br", 79.900}, {"Kr", 83.800}, {"Rb", 85.468}, {"Sr", 87.620}, {"Y", 88.906}, {"Zr", 91.220}, 
                {"Nb", 92.906}, {"Mo", 95.950}, {"Tc", 96.906}, {"Ru", 101.100}, {"Rh", 102.906}, {"Pd", 106.420}, {"Ag", 107.868}, {"Cd", 112.410}, {"In", 114.818}, {"Sn", 118.710}, 
                {"Sb", 121.760}, {"Te", 127.600}, {"I", 126.905}, {"Xe", 131.290}, {"Cs", 132.905}, {"Ba", 137.330}, {"La", 138.906}, {"Ce", 140.116}, {"Pr", 140.908}, {"Nd", 144.240}, 
                {"Pm", 144.913}, {"Sm", 150.400}, {"Eu", 151.964}, {"Gd", 157.200}, {"Tb", 158.925}, {"Dy", 162.500}, {"Ho", 164.930}, {"Er", 167.260}, {"Tm", 168.934}, {"Yb", 173.050}, 
                {"Lu", 174.967}, {"Hf", 178.490}, {"Ta", 180.948}, {"W", 183.840}, {"Re", 186.207}, {"Os", 190.200}, {"Ir", 192.220}, {"Pt", 195.080}, {"Au", 196.967}, {"Hg", 200.590}, 
                {"Tl", 204.383}, {"Pb", 207.000}, {"Bi", 208.980}, {"Po", 208.982}, {"At", 209.987}, {"Rn", 222.018}, {"Fr", 223.020}, {"Ra", 226.025}, {"Ac", 227.028}, {"Th", 232.038}, 
                {"Pa", 231.036}, {"U", 238.029}, {"Np", 237.048}, {"Pu", 244.064}, {"Am", 243.061}, {"Cm", 247.070}, {"Bk", 247.070}, {"Cf", 251.080}, {"Es", 252.083}
            };

            // Efficient lookup with comprehensive error handling
            auto iterator_ = atomicMass.find(atom);
            if (iterator_ != atomicMass.end())
                return iterator_->second;
                
            // Consistent error reporting with atomic number lookup
            throw Planck::Exceptions::GeomException("No atom called " + atom);
        }
    };
}