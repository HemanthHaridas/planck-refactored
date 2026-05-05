#include "symmetry/symmetry.h"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace
{
    bool expect(bool condition, const std::string &message)
    {
        if (condition)
            return true;
        std::cerr << message << '\n';
        return false;
    }

    bool near(double actual, double expected, double tol)
    {
        return std::abs(actual - expected) <= tol;
    }

    double bond_length_bohr(const Eigen::MatrixXd &coords_bohr)
    {
        return (coords_bohr.row(1) - coords_bohr.row(0)).norm();
    }

    HartreeFock::Molecule make_h2()
    {
        HartreeFock::Molecule molecule;
        molecule.natoms = 2;
        molecule.atomic_numbers.resize(2);
        molecule.atomic_numbers << 1, 1;
        molecule.atomic_masses.resize(2);
        molecule.atomic_masses << 1.0, 1.0;
        return molecule;
    }

    bool test_bohr_input_ignores_stale_raw_coordinates()
    {
        auto molecule = make_h2();
        molecule._is_bohr = true;

        molecule.coordinates.resize(2, 3);
        molecule.coordinates <<
            0.0, 0.0, -0.70,
            0.0, 0.0, 0.70;

        molecule._coordinates.resize(2, 3);
        molecule._coordinates <<
            0.0, 0.0, -1.20,
            0.0, 0.0, 1.20;

        const double expected_bond = bond_length_bohr(molecule._coordinates);

        const auto result = HartreeFock::Symmetry::detectSymmetry(
            molecule, HartreeFock::Units::Bohr);
        if (!expect(static_cast<bool>(result), "Bohr-input symmetry detection should succeed"))
            return false;

        const double actual_bond = bond_length_bohr(molecule._standard);
        std::ostringstream bohr_message;
        bohr_message.setf(std::ios::scientific);
        bohr_message.precision(16);
        bohr_message << "Bohr-input symmetry detection should use the internal Bohr geometry, not stale raw coordinates"
                     << " (expected bond " << expected_bond << " bohr, got " << actual_bond << " bohr)";

        return expect(
                   near(actual_bond, expected_bond, 5.0e-7),
                   bohr_message.str()) &&
               expect(
                   molecule._point_group != "C1",
                   "Bohr-input H2 should still be recognized as symmetric");
    }

    bool test_angstrom_input_prefers_raw_angstrom_coordinates()
    {
        auto molecule = make_h2();
        molecule._is_bohr = true;

        molecule.coordinates.resize(2, 3);
        molecule.coordinates <<
            0.0, 0.0, -0.37,
            0.0, 0.0, 0.37;

        molecule._coordinates.resize(2, 3);
        molecule._coordinates <<
            0.0, 0.0, -1.50,
            0.0, 0.0, 1.50;

        const double expected_bond = 0.74 * ANGSTROM_TO_BOHR;

        const auto result = HartreeFock::Symmetry::detectSymmetry(
            molecule, HartreeFock::Units::Angstrom);
        if (!expect(static_cast<bool>(result), "Angstrom-input symmetry detection should succeed"))
            return false;

        return expect(
                   near(bond_length_bohr(molecule._standard), expected_bond, 5.0e-7),
                   "Angstrom-input symmetry detection should preserve the raw Angstrom frame") &&
               expect(
                   molecule._point_group != "C1",
                   "Angstrom-input H2 should still be recognized as symmetric");
    }
} // namespace

int main()
{
    bool ok = true;
    ok &= test_bohr_input_ignores_stale_raw_coordinates();
    ok &= test_angstrom_input_prefers_raw_angstrom_coordinates();
    return ok ? 0 : 1;
}
