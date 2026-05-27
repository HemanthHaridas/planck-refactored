#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>
#include <string.h>

#include "base/tables.h"
#include "symmetry.h"
#include "wrapper.h"

// Full implementation of detectSymmetry
std::expected<void, std::string> HartreeFock::Symmetry::detectSymmetry(
    HartreeFock::Molecule &molecule,
    HartreeFock::Units input_units)
{
    const auto expected_rows = static_cast<Eigen::Index>(molecule.natoms);
    const bool have_raw_coords =
        molecule.coordinates.rows() == expected_rows &&
        molecule.coordinates.cols() == 3;
    const bool have_bohr_coords =
        molecule._is_bohr &&
        molecule._coordinates.rows() == expected_rows &&
        molecule._coordinates.cols() == 3;
    const bool have_standard_bohr =
        molecule._standard_is_bohr &&
        molecule._standard.rows() == expected_rows &&
        molecule._standard.cols() == 3;

    Eigen::MatrixXd bohr_coords;
    Eigen::MatrixXd coords_angstrom;

    if (have_standard_bohr)
    {
        bohr_coords = molecule._standard;
    }
    else if (have_bohr_coords)
    {
        bohr_coords = molecule._coordinates;
    }
    else if (have_raw_coords)
    {
        bohr_coords = (input_units == HartreeFock::Units::Bohr)
                          ? molecule.coordinates
                          : molecule.coordinates * ANGSTROM_TO_BOHR;
    }
    else
    {
        return std::unexpected("Symmetry detection requires initialized molecular coordinates.");
    }

    if (have_raw_coords && input_units == HartreeFock::Units::Angstrom)
    {
        // Preserve the declared Angstrom frame exactly instead of converting
        // Bohr -> Angstrom and back through floating-point roundoff.
        coords_angstrom = molecule.coordinates;
    }
    else
    {
        // For Bohr inputs, or when only a Bohr frame is available, build the
        // libmsym input explicitly from the authoritative Bohr geometry.
        coords_angstrom = bohr_coords * BOHR_TO_ANGSTROM;
    }

    // ── Single atom → Kh ─────────────────────────────────────────────────────────
    // A lone atom has the full spherical symmetry group Kh (K_h, the O(3) point
    // group). libmsym returns C1 for it because its detection is operation-driven
    // and a single point at the origin generates no finite symmetry operations
    // (findSymmetrySpherical only emits an axis for an OFF-centre atom). We label it
    // Kh directly. The molecule is placed at the origin (its own centre of mass) as
    // the standard frame. No finite operation set is exposed, so the SAO-blocking
    // and full-symmetry-ERI machinery treats Kh like the linear groups (skipped at
    // the driver gates) — a one-atom system has no symmetry-equivalent atoms to
    // reduce over anyway, so there is nothing to gain there.
    if (molecule.natoms == 1)
    {
        molecule._point_group = "Kh";
        Eigen::MatrixXd origin = Eigen::MatrixXd::Zero(1, 3);
        molecule.set_standard_from_angstrom(origin);
        molecule._symmetry = true;
        molecule._symmetry_alignment_transform.setIdentity();
        return {};
    }

    auto ctx_result = HartreeFock::Symmetry::SymmetryContext::create();
    if (!ctx_result)
        return std::unexpected(ctx_result.error());
    HartreeFock::Symmetry::SymmetryContext ctx = std::move(*ctx_result);

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
        atoms.data()[i].v[0] = coords_angstrom(i, 0);
        atoms.data()[i].v[1] = coords_angstrom(i, 1);
        atoms.data()[i].v[2] = coords_angstrom(i, 2);
    }

    if (MSYM_SUCCESS != msymSetElements(ctx.get(), atoms.size(), atoms.data()))
    {
        return std::unexpected("Unable to set elements.");
    }

    if (MSYM_SUCCESS != msymFindSymmetry(ctx.get()))
    {
        // Symmetry detection failed — fall back to input geometry (already in Bohr).
        molecule._point_group = "C1";
        molecule.set_standard_from_bohr(bohr_coords);
        molecule._symmetry = false;
        molecule._symmetry_alignment_transform.setIdentity();
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

    double alignment_transform[3][3];
    if (MSYM_SUCCESS != msymGetAlignmentTransform(ctx.get(), alignment_transform))
    {
        return std::unexpected("Unable to get alignment transform.");
    }

    if (MSYM_SUCCESS != msymAlignAxes(ctx.get()))
    {
        return std::unexpected("Unable to align symmetry axes.");
    }

    // libmsym returns coordinates in the same units as input (Angstrom).
    // Store both the Angstrom and the Bohr versions.
    Eigen::MatrixXd standard_coords(molecule.natoms, 3);
    for (size_t i = 0; i < molecule.natoms; ++i)
    {
        standard_coords(i, 0) = new_geometry[i].v[0];
        standard_coords(i, 1) = new_geometry[i].v[1];
        standard_coords(i, 2) = new_geometry[i].v[2];
    }
    molecule.set_standard_from_angstrom(standard_coords);
    molecule._symmetry = true;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            molecule._symmetry_alignment_transform(row, col) = alignment_transform[row][col];

    return {};
}
