#pragma once

// Standard library includes for mathematical operations, string handling, and optional values
#include <Eigen/Core>    // Eigen library for 3D vector operations and linear algebra
#include <Eigen/Dense>   // Eigen library for 3D vector operations and linear algebra
#include <string>        // Standard string class for atom names
#include <optional>      // Optional wrapper for coordinates that may not be set
#include <unordered_map> // Hash map for efficient atomic data lookups
#include <vector>        // Vector container for storing atoms
#include <tuple>         // Tuple for coordinate return type
#include <cstdint>       // Fixed-width integer types
#include <algorithm>     // Misc. Functions
#include <functional>    // Required for std::greater
#include <utility>       // Required for std::pair
#include <queue>         // Required for std::priority_queue

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
    class Molecule; // forward declaration for Molecule class

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
        std::optional<Eigen::Vector3d> _cartesian_coordinates; // atom coordinates in 3D space

        std::string _atom;            // Element symbol (e.g., "H", "C", "O")
        std::uint64_t _atomic_number; // Z - atomic number (number of protons)
        std::double_t _atomic_mass;   // Standard atomic mass in atomic mass units (amu)

        // Friend class declaration allows Molecule to access private members
        // This enables efficient construction and manipulation of atoms within molecules
        friend class Molecule;

    public:
        /**
         * @brief Default constructor
         * Creates an atom with uninitialized values - typically used when atoms
         * will be configured later through setter methods or friend class access
         */
        Atom() = default; // default constructor

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
        explicit Atom(const Eigen::Vector3d &coord, std::string &atom, std::uint64_t &z, std::double_t &mass)
            : _cartesian_coordinates(std::move(coord)), _atom(atom), _atomic_number(z), _atomic_mass(mass) {};

        /**
         * @brief Updates the atom's 3D coordinates
         * @param coord New coordinates for the atom
         *
         * This method is essential for geometry optimization, molecular dynamics,
         * and other operations that modify atomic positions while preserving
         * chemical identity.
         */
        void set_coordinates(const Eigen::Vector3d coord) { _cartesian_coordinates = coord; }

        /**
         * @brief Retrieves the atom's current coordinates
         * @return 3D coordinate vector
         * @throws GeomException if coordinates haven't been set
         *
         * This method enforces that coordinates must be explicitly set before use,
         * preventing errors from uninitialized position data. The exception provides
         * clear feedback about which atom lacks coordinates.
         */
        Eigen::Vector3d coordinates() const
        {
            if (!_cartesian_coordinates.has_value())
                throw Planck::Exceptions::GeomException("Atom " + _atom + " has no coordinates set");

            return _cartesian_coordinates.value();
        }

        /**
         * @brief Get the element symbol
         * @return Element symbol as string
         */
        const std::string &get_atom_symbol() const { return _atom; }

        /**
         * @brief Get the atomic number
         * @return Atomic number (Z)
         */
        std::uint64_t get_atomic_number() const { return _atomic_number; }

        /**
         * @brief Get the atomic mass
         * @return Atomic mass in amu
         */
        double get_atomic_mass() const { return _atomic_mass; }
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
        Eigen::MatrixXd _distance_matrix;

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
        Molecule() = default;
        explicit Molecule(const std::vector<std::string> atoms, const std::vector<Eigen::Vector3d> coords)
        {
            // Input validation - critical for preventing runtime errors
            if (atoms.size() != coords.size())
                throw Planck::Exceptions::GeomException("Mismatch between atom names and coordinate list size");

            // Construct each atom with complete chemical and spatial information
            for (std::uint64_t i = 0; i < atoms.size(); i++)
            {
                // Automatic lookup of atomic properties from element symbols
                std::uint64_t z = get_atomic_number_from_name(atoms[i]);
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
         * @throws GeomException if coordinate vector size doesn't match atom count
         *
         * This method is essential for:
         * - Geometry optimization algorithms
         * - Molecular dynamics simulations
         * - Loading new conformations from external sources
         */
        void update_coordinates(const std::vector<Eigen::Vector3d> &coords)
        {
            if (coords.size() != _geometry.size())
                throw Planck::Exceptions::GeomException("Coordinate vector size doesn't match number of atoms");

            for (std::uint64_t i = 0; i < coords.size(); i++)
                _geometry[i].set_coordinates(coords[i]);
        }

        /**
         * @brief Converts all atomic coordinates in the molecule from Angstrom to Bohr units
         *
         * This function iterates through all atoms in the molecular geometry and
         * converts their Cartesian coordinates from Angstrom units to Bohr units
         * (atomic units) by multiplying each coordinate by the conversion factor
         * 1.8897259886 Bohr/Angstrom.
         *
         * The conversion is performed in-place, modifying the original coordinate
         * values stored in each atom's _cartesian_coordinates member through the
         * Atom::set_coordinates() method.
         *
         * @note The conversion factor 1.8897259886 is the standard CODATA value
         *       for converting Angstrom to Bohr units (1/0.5291772109 Å/Bohr)
         * @note This operation preserves all other atomic properties (symbol, mass, etc.)
         *
         * @throws GeomException if any atom in the molecule has unset coordinates
         * @throws GeomException if the molecule is empty (no atoms)
         *
         * @see convert_coords_to_angstrom() for the inverse conversion
         * @see https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0 for CODATA values
         *
         * @warning Repeated conversions (Å→Bohr→Å) may introduce small numerical
         *          errors due to floating-point precision limitations
         */
        void convert_coords_to_bohr()
        {
            // Check if molecule contains any atoms
            if (_geometry.empty())
                throw Planck::Exceptions::GeomException("Cannot convert coordinates of an empty molecule");

            // Conversion factor: Angstrom to Bohr (1 Å = 1.8897259886 Bohr)
            constexpr std::double_t ANGSTROM_TO_BOHR = 1.8897259886;

            // Iterate through all atoms in the molecular geometry
            for (auto &atom : _geometry)
            {
                // Get current coordinates (this will throw if coordinates are not set)
                Eigen::Vector3d current_coords = atom.coordinates();

                // Convert coordinates by multiplying with conversion factor
                Eigen::Vector3d bohr_coords = current_coords * ANGSTROM_TO_BOHR;

                // Update the atom's coordinates with the converted values
                atom.set_coordinates(bohr_coords);
            }
        }

        /**
         * @brief Converts all atomic coordinates in the molecule from Bohr to Angstrom units
         *
         * This function iterates through all atoms in the molecular geometry and
         * converts their Cartesian coordinates from Bohr units (atomic units) to
         * Angstrom units by multiplying each coordinate by the conversion factor
         * 0.5291772109 Angstrom/Bohr.
         *
         * The conversion is performed in-place, modifying the original coordinate
         * values stored in each atom's _cartesian_coordinates member through the
         * Atom::set_coordinates() method.
         *
         * @note The conversion factor 0.5291772109 is the standard CODATA value
         *       for the Bohr radius (approximately 0.529177210903 Angstrom)
         * @note This operation preserves all other atomic properties (symbol, mass, etc.)
         *
         * @throws GeomException if any atom in the molecule has unset coordinates
         * @throws GeomException if the molecule is empty (no atoms)
         *
         * @see convert_coords_to_bohr() for the inverse conversion
         * @see https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0 for CODATA values
         *
         * @warning The conversion factor used is more precise than the truncated
         *          version (0.52917725) shown in the original example, reducing
         *          numerical errors in repeated conversions
         */
        void convert_coords_to_angstrom()
        {
            // Check if molecule contains any atoms
            if (_geometry.empty())
                throw Planck::Exceptions::GeomException("Cannot convert coordinates of an empty molecule");

            // Conversion factor: Bohr to Angstrom (1 Bohr = 0.5291772109 Å)
            // Using more precise CODATA value than the truncated 0.52917725
            constexpr std::double_t BOHR_TO_ANGSTROM = 0.5291772109;

            // Iterate through all atoms in the molecular geometry
            for (auto &atom : _geometry)
            {
                // Get current coordinates (this will throw if coordinates are not set)
                Eigen::Vector3d current_coords = atom.coordinates();

                // Convert coordinates by multiplying with conversion factor
                Eigen::Vector3d angstrom_coords = current_coords * BOHR_TO_ANGSTROM;

                // Update the atom's coordinates with the converted values
                atom.set_coordinates(angstrom_coords);
            }
        }

        /**
         * @brief Returns a view of the current coordinates
         * @return Vector of tuples containing element symbol and coordinates
         *
         * This method is essential for:
         * - Geometry optimization algorithms
         * - Molecular dynamics simulations
         */
        std::vector<std::tuple<std::string, Eigen::Vector3d>> get_coordinates() const
        {
            std::vector<std::tuple<std::string, Eigen::Vector3d>> coordinates;
            coordinates.reserve(_geometry.size());

            for (const auto &atom : _geometry)
                coordinates.emplace_back(atom.get_atom_symbol(), atom.coordinates());
            return coordinates;
        }

        /**
         * @brief Get the number of atoms in the molecule
         * @return Number of atoms
         */
        std::uint64_t size() const { return _geometry.size(); }

        /**
         * @brief Get atom by index
         * @param index Index of the atom
         * @return Const reference to the atom
         * @throws std::out_of_range if index is invalid
         */
        const Atom &get_atom(std::uint64_t index) const { return _geometry.at(index); }

        const Eigen::MatrixXd get_connectity_table() const { return _distance_matrix; }

        /**
         * @brief Structure to represent a Z-matrix _zmatrix_entry
         */
        struct ZMatrixEntry
        {
            std::string _atom_symbol;    // Element symbol
            std::int64_t _bond_atom;     // Index of atom for bond distance (-1 if not applicable)
            std::int64_t _angle_atom;    // Index of atom for bond angle (-1 if not applicable)
            std::int64_t _dihedral_atom; // Index of atom for dihedral angle (-1 if not applicable)
            std::double_t _bond;         // Bond distance in current units
            std::double_t _angle;        // Bond angle in degrees
            std::double_t _dihedral;     // Dihedral angle in degrees

            explicit ZMatrixEntry() : _atom_symbol(""), _bond_atom(-1), _angle_atom(-1), _dihedral_atom(-1), _bond(0.0), _angle(0.0), _dihedral(0.0) {};
        };

        /**
         * @brief Generates a Z-matrix using connectivity information for optimal atom ordering
         * @return Vector of Z-matrix entries with chemically meaningful internal coordinates
         *
         * This optimized version uses the connectivity table to:
         * 1. Prioritize connected atoms when selecting references
         * 2. Avoid problematic geometric arrangements (linear/coplanar)
         */
        std::vector<ZMatrixEntry> generate_zmatrix() const
        {
            if (_geometry.empty())
                throw Planck::Exceptions::GeomException("Cannot construct Z-Matrix for an empty molecule");

            // Ensure connectivity table is built
            if (_distance_matrix.rows() != static_cast<std::uint64_t>(_geometry.size()))
                throw Planck::Exceptions::GeomException("Connectivity table not initialized. Call create_connectivity_table() first.");

            // Generate a Minimum Spanning Tree from connetivity table
            // First find the node with maximum connections to use as the seed
            auto [_count, _root] = get_max_node();

            // Set up the MST
            std::vector<std::uint64_t> _parent(_geometry.size(), -1);
            std::vector<std::double_t> _keys(_geometry.size(), std::numeric_limits<std::double_t>::max());
            std::vector<bool> _inMST(_geometry.size(), false);

            // Use priority_queue for efficient access
            using _node = std::pair<std::double_t, std::uint64_t>;
            std::priority_queue<_node, std::vector<_node>, std::greater<_node>> _mst;

            // Set the key to the root node as 0.0
            // Because distance from root to root is 0.0
            _keys[_root] = 0.0;
            _mst.push({0.0, _root});

            // Now build MST
            while (!_mst.empty())
            {
                auto [_dist, _index] = _mst.top();
                _mst.pop();

                // Check if the currently visited node is already in MST
                if (_inMST[_index])
                    continue;

                // Place the node in MST
                _inMST[_index] = true;

                // Now loop over all nodes (atoms) in the molecule
                for (std::uint64_t i = 0; i < _geometry.size(); i++)
                {
                    // Check if the node is the current root or if it is already in MST
                    if (_index == i || _inMST[i])
                        continue;

                    // Get the distance from distance matrix
                    // Assign the node to MST based on the distance
                    auto _distance = _distance_matrix(_index, i);
                    if (_distance < _keys[i] && (_distance - 0.00) > 1.0e-6)
                    {
                        _keys[i] = _distance;
                        _parent[i] = _index;
                        _mst.push({_distance, i});
                    }
                }
            }
        }

        /**
         * @brief Constructs a pairwise atomic connectivity table based on interatomic distances.
         *
         * This function loops through all unique pairs of atoms in a molecular geometry
         * and checks if they are close enough (based on their van der Waals radii)
         * to be considered "connected" or potentially bonded. If so, it records the
         * distance in a symmetric distance matrix.
         *
         * The cutoff distance is defined as 1.2 times the sum of van der Waals radii
         * for the atom pair.
         */
        void create_connetivity_table()
        {
            // resize the connectivity matrix
            _distance_matrix.resize(_geometry.size(), _geometry.size());
            _distance_matrix.setZero();

            // Loop over all atoms in the geometry
            for (std::uint64_t i = 0; i < _geometry.size(); i++)
            {
                // Get coordinates of atom i
                auto _i_coords = _geometry[i].coordinates();

                // Loop only over atoms j > i to avoid redundant pairings (i,j) and (j,i)
                for (std::uint64_t j = i + 1; j < _geometry.size(); j++)
                {
                    // Get coordinates of atom j
                    auto _j_coords = _geometry[j].coordinates();

                    // Compute Euclidean distance between atom i and atom j
                    auto _distance = calculate_distance(_i_coords, _j_coords);

                    // Compute cutoff distance using van der Waals radii for atom i and atom j
                    auto _cut_off = 1.2 * (get_vdw_radii_from_name(_geometry[i].get_atom_symbol()) +
                                           get_vdw_radii_from_name(_geometry[j].get_atom_symbol()));

                    // If atoms are within the cutoff distance, register the connection in the distance matrix
                    if (_distance <= _cut_off)
                    {
                        _distance_matrix(i, j) = _distance;
                        _distance_matrix(j, i) = _distance; // ensure symmetry
                    }
                }
            }
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
                {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10}, {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30}, {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}, {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90}, {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {"Am", 95}, {"Cm", 96}, {"Bk", 97}, {"Cf", 98}, {"Es", 99}};

            // Efficient hash map lookup with error handling
            auto iterator_ = atomicNumber.find(atom);
            if (iterator_ != atomicNumber.end())
                return iterator_->second;

            // Provide clear error message for debugging
            throw Planck::Exceptions::GeomException("No atom called " + atom);
        }

        /**
         * @brief Internal lookup function for atomic radii
         * @param atom Element symbol (case-sensitive)
         * @return Standard atomic radii in angstroms (amu)
         * @throws GeomException if element symbol is not recognized
         *
         * Provides standard atomic radii for elements 1-96. These values are
         * essential for generation of connectivity tables in quantum chemistry
         * applications.
         */
        std::double_t get_vdw_radii_from_name(const std::string &atom) const
        {
            // Standard atomic radii from https://doi.org/10.1039/B801115J
            static const std::unordered_map<std::string, std::double_t> vdWRadii = {
                {"H", 0.31}, {"He", 0.28}, {"Li", 1.28}, {"Be", 0.96}, {"B", 0.84}, {"C", 0.76}, {"N", 0.71}, {"O", 0.66}, {"F", 0.57}, {"Ne", 0.58}, {"Na", 1.66}, {"Mg", 1.41}, {"Al", 1.21}, {"Si", 1.11}, {"P", 1.07}, {"S", 1.05}, {"Cl", 1.02}, {"Ar", 1.06}, {"K", 2.03}, {"Ca", 1.76}, {"Sc", 1.70}, {"Ti", 1.60}, {"V", 1.53}, {"Cr", 1.39}, {"Mn", 1.52}, {"Fe", 1.32}, {"Co", 1.26}, {"Ni", 1.24}, {"Cu", 1.32}, {"Zn", 1.22}, {"Ga", 1.22}, {"Ge", 1.20}, {"As", 1.19}, {"Se", 1.20}, {"Br", 1.20}, {"Kr", 1.16}, {"Rb", 2.20}, {"Sr", 1.95}, {"Y", 1.90}, {"Zr", 1.75}, {"Nb", 1.64}, {"Mo", 1.54}, {"Tc", 1.47}, {"Ru", 1.46}, {"Rh", 1.42}, {"Pd", 1.39}, {"Ag", 1.45}, {"Cd", 1.44}, {"In", 1.42}, {"Sn", 1.39}, {"Sb", 1.39}, {"Te", 1.38}, {"I", 1.39}, {"Xe", 1.40}, {"Cs", 2.44}, {"Ba", 2.15}, {"La", 2.07}, {"Ce", 2.04}, {"Pr", 2.03}, {"Nd", 2.01}, {"Pm", 1.99}, {"Sm", 1.98}, {"Eu", 1.98}, {"Gd", 1.96}, {"Tb", 1.94}, {"Dy", 1.92}, {"Ho", 1.92}, {"Er", 1.89}, {"Tm", 1.90}, {"Yb", 1.87}, {"Lu", 1.87}, {"Hf", 1.75}, {"Ta", 1.70}, {"W", 1.62}, {"Re", 1.51}, {"Os", 1.44}, {"Ir", 1.41}, {"Pt", 1.36}, {"Au", 1.36}, {"Hg", 1.32}, {"Tl", 1.45}, {"Pb", 1.46}, {"Bi", 1.48}, {"Po", 1.40}, {"At", 1.50}, {"Rn", 1.50}, {"Fr", 2.60}, {"Ra", 2.21}, {"Ac", 2.15}, {"Th", 2.06}, {"Pa", 2.00}, {"U", 1.96}, {"Np", 1.90}, {"Pu", 1.87}, {"Am", 1.80}, {"Cm", 1.69}};

            // Efficient lookup with comprehensive error handling
            auto iterator_ = vdWRadii.find(atom);
            if (iterator_ != vdWRadii.end())
                return iterator_->second;

            // Consistent error reporting with atomic number lookup
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
                {"H", 1.008}, {"He", 4.003}, {"Li", 7.000}, {"Be", 9.012}, {"B", 10.810}, {"C", 12.011}, {"N", 14.007}, {"O", 15.999}, {"F", 18.998}, {"Ne", 20.180}, {"Na", 22.990}, {"Mg", 24.305}, {"Al", 26.982}, {"Si", 28.085}, {"P", 30.974}, {"S", 32.070}, {"Cl", 35.450}, {"Ar", 39.900}, {"K", 39.098}, {"Ca", 40.080}, {"Sc", 44.956}, {"Ti", 47.867}, {"V", 50.942}, {"Cr", 51.996}, {"Mn", 54.938}, {"Fe", 55.840}, {"Co", 58.933}, {"Ni", 58.693}, {"Cu", 63.550}, {"Zn", 65.400}, {"Ga", 69.723}, {"Ge", 72.630}, {"As", 74.922}, {"Se", 78.970}, {"Br", 79.900}, {"Kr", 83.800}, {"Rb", 85.468}, {"Sr", 87.620}, {"Y", 88.906}, {"Zr", 91.220}, {"Nb", 92.906}, {"Mo", 95.950}, {"Tc", 96.906}, {"Ru", 101.100}, {"Rh", 102.906}, {"Pd", 106.420}, {"Ag", 107.868}, {"Cd", 112.410}, {"In", 114.818}, {"Sn", 118.710}, {"Sb", 121.760}, {"Te", 127.600}, {"I", 126.905}, {"Xe", 131.290}, {"Cs", 132.905}, {"Ba", 137.330}, {"La", 138.906}, {"Ce", 140.116}, {"Pr", 140.908}, {"Nd", 144.240}, {"Pm", 144.913}, {"Sm", 150.400}, {"Eu", 151.964}, {"Gd", 157.200}, {"Tb", 158.925}, {"Dy", 162.500}, {"Ho", 164.930}, {"Er", 167.260}, {"Tm", 168.934}, {"Yb", 173.050}, {"Lu", 174.967}, {"Hf", 178.490}, {"Ta", 180.948}, {"W", 183.840}, {"Re", 186.207}, {"Os", 190.200}, {"Ir", 192.220}, {"Pt", 195.080}, {"Au", 196.967}, {"Hg", 200.590}, {"Tl", 204.383}, {"Pb", 207.000}, {"Bi", 208.980}, {"Po", 208.982}, {"At", 209.987}, {"Rn", 222.018}, {"Fr", 223.020}, {"Ra", 226.025}, {"Ac", 227.028}, {"Th", 232.038}, {"Pa", 231.036}, {"U", 238.029}, {"Np", 237.048}, {"Pu", 244.064}, {"Am", 243.061}, {"Cm", 247.070}, {"Bk", 247.070}, {"Cf", 251.080}, {"Es", 252.083}};

            // Efficient lookup with comprehensive error handling
            auto iterator_ = atomicMass.find(atom);
            if (iterator_ != atomicMass.end())
                return iterator_->second;

            // Consistent error reporting with atomic number lookup
            throw Planck::Exceptions::GeomException("No atom called " + atom);
        }

        /**
         * @brief Calculates the Euclidean distance between two 3D points
         * @param p1 First point in 3D space
         * @param p2 Second point in 3D space
         * @return The distance between p1 and p2
         *
         * The Eigen::Vector3f::norm() method efficiently computes this Euclidean norm.
         * Used primarily for calculating bond lengths in molecular structures.
         */
        std::double_t calculate_distance(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) const
        {
            return (p1 - p2).norm();
        }

        /**
         * @brief Calculates the bond angle (in degrees) formed at a central atom
         * @param p1 First outer atom position
         * @param p2 Central atom position (vertex of the angle)
         * @param p3 Second outer atom position
         * @return The bond angle in degrees (0° to 180°)
         *
         * This function calculates the angle between two bonds meeting at p2:
         * - Bond 1: p2 -> p1
         * - Bond 2: p2 -> p3
         */
        std::double_t calculate_angle(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Eigen::Vector3d &p3) const
        {
            // Compute the vector from p2 to p1 and normalize it to obtain a unit direction vector
            // This represents one arm of the angle centered at p2
            // Vector points FROM the central atom TO the first outer atom
            auto arm_a = (p1 - p2).normalized();

            // Compute the vector from p2 to p3 and normalize it to obtain the second unit direction vector
            // This is the other arm of the angle centered at p2
            // Vector points FROM the central atom TO the second outer atom
            auto arm_b = (p3 - p2).normalized();

            // Compute the dot product between the two unit vectors
            auto cos_theta = arm_a.dot(arm_b);

            // Protect against floating point precision issues that may push cos_theta slightly outside [-1, 1]
            // This prevents domain errors in acos() function which only accepts values in [-1, 1]
            // Such errors can occur due to accumulated floating-point rounding in vector operations
            cos_theta = std::clamp(cos_theta, -1.0, 1.0);

            // Calculate the angle in radians using arccos of the clamped dot product
            std::double_t angle = std::acos(cos_theta) * (180.0 / M_PI);

            // Return the angle in degrees
            return angle;
        }

        /**
         * @brief Computes the unsigned dihedral angle (in degrees) defined by four points in 3D space
         * @param p1 First point (defines first plane with p2, p3)
         * @param p2 Second point (shared by both planes)
         * @param p3 Third point (shared by both planes, forms central bond with p2)
         * @param p4 Fourth point (defines second plane with p2, p3)
         * @return The unsigned dihedral angle in degrees (0° to 180°)
         *
         * This function returns only the magnitude (0° to 180°).
         * For distinguishing between clockwise/counterclockwise rotations,
         * use calculate_signed_dihedral_simple() instead.
         */
        std::double_t calculate_unsigned_dihedral(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                                                  const Eigen::Vector3d &p3, const Eigen::Vector3d &p4) const
        {
            // Vector from p2 to p1 (first bond arm)
            // This vector lies in the first plane (p1-p2-p3)
            auto arm_a = (p1 - p2).normalized();

            // Vector from p2 to p3 (shared bond between planes)
            // This is the central bond around which the dihedral rotation occurs
            // It serves as the common edge between the two planes
            auto arm_b = (p3 - p2).normalized();

            // Normal vector to the plane formed by (p1, p2, p3)
            auto norm_ab = arm_a.cross(arm_b).normalized();

            // Vector from p3 to p2 (opposite direction of shared bond)
            // This ensures consistent orientation when calculating the second plane's normal
            // We use p3 as the origin for the second plane's vectors
            auto arm_c = (p2 - p3).normalized();

            // Vector from p3 to p4 (second bond arm)
            // This vector lies in the second plane (p2-p3-p4)
            auto arm_d = (p4 - p3).normalized();

            // Normal vector to the plane formed by (p2, p3, p4)
            // The order matters for consistent orientation relative to the first normal
            auto norm_cd = arm_c.cross(arm_d).normalized();

            // Compute the cosine of the angle between the two plane normals
            // The angle between planes equals the angle between their normal vectors
            // (or the supplement, depending on normal vector directions)
            auto cos_phi = norm_ab.dot(norm_cd);

            // Clamp to [-1, 1] to prevent domain errors from floating-point inaccuracies
            // Similar to the angle calculation, floating-point operations can push
            // the dot product slightly outside the valid domain for acos()
            cos_phi = std::clamp(cos_phi, -1.0, 1.0);

            // Compute the angle in degrees (from arccosine of cosine value)
            std::double_t dihedral = std::acos(cos_phi) * (180.0 / M_PI);

            return dihedral;
        }

        /**
         * @brief Computes the signed dihedral angle (in degrees) with proper orientation
         * @param p1 First point (defines first plane with p2, p3)
         * @param p2 Second point (shared by both planes)
         * @param p3 Third point (shared by both planes, forms central bond with p2)
         * @param p4 Fourth point (defines second plane with p2, p3)
         * @return The signed dihedral angle in degrees (-180° to +180°)
         *
         * Unlike the unsigned version, this function preserves the rotational direction:
         * - POSITIVE (+): Clockwise rotation when looking down the p2→p3 bond
         * - NEGATIVE (-): Counter-clockwise rotation when looking down the p2→p3 bond
         *
         * - 0°: Cis/eclipsed conformation (p1 and p4 on same side)
         * - +60°: Gauche+ conformation
         * - -60°: Gauche- conformation
         * - +/-180°: Trans/anti conformation (p1 and p4 on opposite sides)
         */
        std::double_t calculate_signed_dihedral(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2,
                                                const Eigen::Vector3d &p3, const Eigen::Vector3d &p4) const
        {
            // Vectors defining the dihedral geometry
            // These vectors represent the "bonds" in the dihedral angle definition
            Eigen::Vector3d b1 = p2 - p1; // Vector from p1 to p2 (first bond)
            Eigen::Vector3d b2 = p3 - p2; // Central bond from p2 to p3 (rotation axis)
            Eigen::Vector3d b3 = p4 - p3; // Vector from p3 to p4 (second bond)

            // Normal vectors to the two planes defining the dihedral angle
            // Plane 1: spanned by vectors b1 and b2 (contains points p1, p2, p3)
            // Plane 2: spanned by vectors b2 and b3 (contains points p2, p3, p4)
            Eigen::Vector3d n1 = b1.cross(b2).normalized(); // Normal to plane 1
            Eigen::Vector3d n2 = b2.cross(b3).normalized(); // Normal to plane 2

            // Calculate signed dihedral angle using atan2 for robust quadrant determination
            // cos component: dot product of normal vectors gives cosine of angle between planes
            std::double_t cos_angle = n1.dot(n2);

            // Positive when (n1 × n2) points in same direction as b2, negative otherwise
            std::double_t sin_angle = n1.cross(n2).dot(b2.normalized());

            // This preserves the sign information lost in acos-based calculations
            return std::atan2(sin_angle, cos_angle) * (180.0 / M_PI);
        }

        // Returns the number of non-zero (i.e., connected) elements in the input vector.
        // Throws std::invalid_argument if there are no connections.
        std::uint64_t get_node_count(const Eigen::VectorXd &connectivity) const
        {
            // Create a boolean mask: true where the connectivity value is non-zero
            auto mask_ = (connectivity.array() != 0);

            // Count how many elements in the mask are true (i.e., non-zero in the input)
            const auto non_zero_count = mask_.count();

            // If no connections are found, throw an exception
            if (non_zero_count == 0)
                throw std::invalid_argument("The node has no connections.");

            // Return the number of connections as a standard std::uint64_t
            return static_cast<std::uint64_t>(non_zero_count);
        }

        // Returns a tuple with (maximum connection count, index of the node with that count)
        std::tuple<std::uint64_t, std::uint64_t> get_max_node() const
        {
            std::uint64_t max_count = 0;
            std::uint64_t max_node = 0;

            // Iterate over each row of the distance matrix
            for (std::uint64_t i = 0; i < _distance_matrix.rows(); ++i)
            {
                // Count non-zero entries (connections) in this node's row
                std::uint64_t node_count = get_node_count(_distance_matrix.row(i));

                // Update the maximum if this node has more connections
                if (node_count > max_count)
                    max_count = node_count;
                max_node = i;
            }

            return {max_count, max_node};
        }

        std::uint64_t get_best_neighbour(const std::uint64_t &node, const std::vector<bool> &placed) const
        {
            // Find all atoms connected to the given atom
            auto mask_ = (_distance_matrix.row(node).array() != 0);
            std::vector<int> indices;
            for (int i = 0; i < mask_.size(); ++i)
            {
                if (mask_(i))
                {
                    indices.push_back(i);
                }
            }

            // Now iterate over them to find the closest atom
            std::uint64_t _best_node;
            auto _min_dist = std::numeric_limits<std::double_t>::max();
            bool _connected = false;

            for (auto _index : indices)
            {
                auto _distance = _distance_matrix(node, _index);
                if (_distance <= _min_dist && !placed[_index])
                {
                    _min_dist = _distance;
                    _best_node = _index;
                    _connected = true;
                }
            }

            // Return the best connected atom, if any
            if (_connected)
            {
                return _best_node;
            }

            // Fall back if no connected atoms
            for (std::uint64_t i = 0; i < _geometry.size(); i++)
            {
                if (i == node)
                    continue;

                auto distance_ = calculate_distance(_geometry[node].coordinates(), _geometry[i].coordinates());
                if (distance_ < _min_dist && !placed[i])
                {
                    _min_dist = distance_;
                    _best_node = i;
                    _connected = true;
                }
            }

            // Return the best connected atom, if any
            if (_connected)
            {
                return _best_node;
            }

            // Throw error if no atom was found
            throw Planck::Exceptions::GeomException("Unable to find any neighbouring atoms for " + std::to_string(node));
        }
    };
}