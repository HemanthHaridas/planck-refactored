#pragma once

#include <Eigen/Core>
#include <string>
#include <optional>
#include <unordered_map>

#include "planck_exceptions.hpp"

namespace Planck::Geometry
{

    class Molecule; // forward declaration for Molecule class

    // class to hold a single atom
    class Atom
    {
    private:
        std::optional<Eigen::Vector3f> _coordinates; // atom coordinates
        std::string _atom;                           // atom name
        std::uint64_t _atomic_number;                // z - atomic number
        std::double_t _atomic_mass;                  // atomic mass

        friend class Molecule;

    public:
        Atom() = default; // default constructor

        // explicit constructor to set the atom name and atom coordinates
        explicit Atom(const Eigen::Vector3f &coord, std::string &atom, std::uint64_t &z, std::double_t &mass) : _coordinates(std::move(coord)), _atom(atom), _atomic_number(z), _atomic_mass(mass) {};

        // setter method to update the geometry
        void set_coordinates(const Eigen::Vector3f coord) { _coordinates = coord; }

        // getter method to get coordinates
        Eigen::Vector3f coordinates() const
        {
            if (!_coordinates.has_value())
                throw Planck::Exceptions::GeomException("Atom " + _atom + " has no coordinates set");

            return _coordinates.value();
        }
    };

    // class to hold a collection of atoms (can be one molecule or n molecules)
    class Molecule
    {
    private:
        std::vector<Atom> _geometry;

    public:
        explicit Molecule(const std::vector<std::string> atoms, const std::vector<Eigen::Vector3f> coords)
        {
            if (atoms.size() != coords.size())
                throw Planck::Exceptions::GeomException("Mismatch between atom names and coordinate list size");

            for (size_t i = 0; i < atoms.size(); i++)
            {
                std::uint64_t z = get_atomic_number_from_name(atoms[i]);
                std::double_t mass = get_atomic_mass_from_name(atoms[i]);
                Atom atom(coords[i], const_cast<std::string &>(atoms[i]), z, mass);
                _geometry.push_back(atom);
            }
        }

        void update_coordinates(const std::vector<Eigen::Vector3f> &coords)
        {
            for (size_t i = 0; i < coords.size(); i++)
            {
                _geometry[i].set_coordinates(coords[i]);    // update the coordinates of each atom;
            }
        }

    private:
        // private API to get atomic number
        std::uint64_t get_atomic_number_from_name(const std::string &atom) const
        {
            static const std::unordered_map<std::string, std::uint64_t> atomicNumber = {
                {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5}, {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10}, {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15}, {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20}, {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25}, {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30}, {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35}, {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40}, {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45}, {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50}, {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55}, {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60}, {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65}, {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70}, {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75}, {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80}, {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85}, {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90}, {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {"Am", 95}, {"Cm", 96}, {"Bk", 97}, {"Cf", 98}, {"Es", 99}};

            auto iterator_ = atomicNumber.find(atom);
            if (iterator_ != atomicNumber.end())
            {
                return iterator_->second;
            }
            throw Planck::Exceptions::GeomException("No atom called " + atom);
        }

        // private API to get atomic mass
        std::double_t get_atomic_mass_from_name(const std::string &atom) const
        {
            static const std::unordered_map<std::string, double> atomicMass = {
                {"H", 1.008}, {"He", 4.003}, {"Li", 7.000}, {"Be", 9.012}, {"B", 10.810}, {"C", 12.011}, {"N", 14.007}, {"O", 15.999}, {"F", 18.998}, {"Ne", 20.180}, {"Na", 22.990}, {"Mg", 24.305}, {"Al", 26.982}, {"Si", 28.085}, {"P", 30.974}, {"S", 32.070}, {"Cl", 35.450}, {"Ar", 39.900}, {"K", 39.098}, {"Ca", 40.080}, {"Sc", 44.956}, {"Ti", 47.867}, {"V", 50.942}, {"Cr", 51.996}, {"Mn", 54.938}, {"Fe", 55.840}, {"Co", 58.933}, {"Ni", 58.693}, {"Cu", 63.550}, {"Zn", 65.400}, {"Ga", 69.723}, {"Ge", 72.630}, {"As", 74.922}, {"Se", 78.970}, {"Br", 79.900}, {"Kr", 83.800}, {"Rb", 85.468}, {"Sr", 87.620}, {"Y", 88.906}, {"Zr", 91.220}, {"Nb", 92.906}, {"Mo", 95.950}, {"Tc", 96.906}, {"Ru", 101.100}, {"Rh", 102.906}, {"Pd", 106.420}, {"Ag", 107.868}, {"Cd", 112.410}, {"In", 114.818}, {"Sn", 118.710}, {"Sb", 121.760}, {"Te", 127.600}, {"I", 126.905}, {"Xe", 131.290}, {"Cs", 132.905}, {"Ba", 137.330}, {"La", 138.906}, {"Ce", 140.116}, {"Pr", 140.908}, {"Nd", 144.240}, {"Pm", 144.913}, {"Sm", 150.400}, {"Eu", 151.964}, {"Gd", 157.200}, {"Tb", 158.925}, {"Dy", 162.500}, {"Ho", 164.930}, {"Er", 167.260}, {"Tm", 168.934}, {"Yb", 173.050}, {"Lu", 174.967}, {"Hf", 178.490}, {"Ta", 180.948}, {"W", 183.840}, {"Re", 186.207}, {"Os", 190.200}, {"Ir", 192.220}, {"Pt", 195.080}, {"Au", 196.967}, {"Hg", 200.590}, {"Tl", 204.383}, {"Pb", 207.000}, {"Bi", 208.980}, {"Po", 208.982}, {"At", 209.987}, {"Rn", 222.018}, {"Fr", 223.020}, {"Ra", 226.025}, {"Ac", 227.028}, {"Th", 232.038}, {"Pa", 231.036}, {"U", 238.029}, {"Np", 237.048}, {"Pu", 244.064}, {"Am", 243.061}, {"Cm", 247.070}, {"Bk", 247.070}, {"Cf", 251.080}, {"Es", 252.083}};

            auto iterator_ = atomicMass.find(atom);
            if (iterator_ != atomicMass.end())
            {
                return iterator_->second;
            }
            throw Planck::Exceptions::GeomException("No atom called " + atom);
        }
    };

}