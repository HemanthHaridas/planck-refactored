#include <algorithm>
#include <iostream>
#include <iomanip>
#include <set>
#include <string.h>

#include "symmetry.h"
#include "wrapper.h"
#include "base/tables.h"

// Full implementation of detectSymmetry
std::expected<void, std::string> HartreeFock::Symmetry::detectSymmetry(HartreeFock::Molecule &molecule)
{
    try
    {
        HartreeFock::Symmetry::SymmetryContext ctx;

        msym_thresholds_t tight_thresholds = {
            0.08,   // zero
            0.1,    // geometry
            0.1,    // angle
            0.06,   // equivalence
            1.0e-1, // permutation
            1.0e-3, // eigfact
            0.1     // orthogonalization
        };

        msymSetThresholds(ctx.get(), &tight_thresholds);

        HartreeFock::Symmetry::SymmetryElements atoms(molecule.natoms);
        for (size_t i = 0; i < molecule.natoms; ++i)
        {
            atoms.data()[i].m = molecule.atomic_masses[i];
            atoms.data()[i].n = molecule.atomic_numbers[i];
            atoms.data()[i].v[0] = molecule.coordinates(i, 0);
            atoms.data()[i].v[1] = molecule.coordinates(i, 1);
            atoms.data()[i].v[2] = molecule.coordinates(i, 2);
        }

        if (MSYM_SUCCESS != msymSetElements(ctx.get(), atoms.size(), atoms.data()))
        {
            return std::unexpected("Unable to set elements.");
        }

        if (MSYM_SUCCESS != msymFindSymmetry(ctx.get()))
        {
            // Symmetry detection failed — fall back to input geometry (already in Bohr).
            molecule._point_group = "C1";
            molecule._standard    = molecule._coordinates;
            molecule._symmetry    = false;
            return {};
        }

        char point_group[32];
        if (MSYM_SUCCESS != msymGetPointGroupName(ctx.get(), sizeof(point_group), point_group))
        {
            return std::unexpected("Unable to get point group name.");
        }
        molecule._point_group = point_group;

        if (point_group[1] == '0')
        {
            molecule._point_group.replace(1, 1, "inf");
        }

        double symm_error = 0.0;
        if (MSYM_SUCCESS != msymSymmetrizeElements(ctx.get(), &symm_error))
        {
            return std::unexpected("Unable to symmetrize the molecule.");
        }

        int new_n_atoms = 0;
        msym_element_t *new_geometry = nullptr;
        if (MSYM_SUCCESS != msymGetElements(ctx.get(), &new_n_atoms, &new_geometry))
        {
            return std::unexpected("Unable to get symmetry elements.");
        }

        if (MSYM_SUCCESS != msymAlignAxes(ctx.get()))
        {
            return std::unexpected("Unable to align symmetry axes.");
        }

        // libmsym returns coordinates in the same units as input (Angstrom).
        // Store both the Angstrom and the Bohr versions.
        molecule.standard.resize(molecule.natoms, 3);
        molecule._standard.resize(molecule.natoms, 3);
        for (size_t i = 0; i < molecule.natoms; ++i)
        {
            molecule.standard(i, 0) = new_geometry[i].v[0];
            molecule.standard(i, 1) = new_geometry[i].v[1];
            molecule.standard(i, 2) = new_geometry[i].v[2];
        }
        molecule._standard = molecule.standard * ANGSTROM_TO_BOHR;
        molecule._symmetry = true;

        return {};
    }
    catch (const std::exception &e)
    {
        return std::unexpected(e.what());
    }
}

