// Unit test for src/symmetry/group_operations.{h,cpp} — Phase 0 of the full-symmetry
// ERI reduction (docs/FULL_SYMMETRY_ERI_DESIGN.md).
//
// build_ao_transform's angular/permutation math is already validated for the
// Abelian subgroup via build_sao_basis (energy-transparent regressions). What is
// NEW here and must be proven independently is (a) pulling the FULL operation list
// (msymGetSymmetryOperations) rather than character-table class reps, and (b)
// including non-Abelian operations (C3, C2', sigma_v, S4, ...) that build_sao_basis
// never exercises. So the test deliberately uses non-Abelian groups (C3v: NH3,
// Td: CH4) and checks the O_R form a valid orthogonal representation of the group:
//   - count == |G|
//   - exactly one identity (O_R == I)
//   - every O_R is orthogonal (O_R^T O_R == I)
//   - the set is closed under multiplication (O_R O_S == some O_T): this is the
//     decisive group-representation invariant and catches transcription / ordering
//     / wrong-operation bugs without needing reference integrals.
//
// (The physical "O_R^T S O_R == S" check is deferred to the Phase 1 regression
// where symmetry-on == symmetry-off energies, per the design note.)

#include "symmetry/group_operations.h"
#include "symmetry/symmetry.h"
#include "basis/basis.h"
#include "base/basis.h"
#include "base/types.h"

#include <Eigen/Core>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    bool g_ok = true;

    void fail(const std::string &msg)
    {
        std::cerr << "[FAIL] " << msg << '\n';
        g_ok = false;
    }

    // Build a fully-initialized Calculator (molecule + symmetry + basis) the same
    // way the driver does, so build_group_operations sees a real basis.
    std::expected<HartreeFock::Calculator, std::string> make_calculator(
        const std::vector<int> &Z,
        const std::vector<std::array<double, 3>> &xyz_angstrom,
        int charge, int multiplicity,
        const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;

        const std::size_t n = Z.size();
        mol.natoms = n;
        mol.charge = charge;
        mol.multiplicity = multiplicity;
        mol.atomic_numbers.resize(n);
        mol.atomic_masses.resize(n);
        mol.coordinates.resize(n, 3);
        for (std::size_t i = 0; i < n; ++i)
        {
            mol.atomic_numbers[static_cast<Eigen::Index>(i)] = Z[i];
            mol.atomic_masses[static_cast<Eigen::Index>(i)] = 1.0; // mass-independent for op matrices
            mol.coordinates(static_cast<Eigen::Index>(i), 0) = xyz_angstrom[i][0];
            mol.coordinates(static_cast<Eigen::Index>(i), 1) = xyz_angstrom[i][1];
            mol.coordinates(static_cast<Eigen::Index>(i), 2) = xyz_angstrom[i][2];
        }

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();

        if (auto r = HartreeFock::Symmetry::detectSymmetry(mol, HartreeFock::Units::Angstrom); !r)
            return std::unexpected("detectSymmetry: " + r.error());

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis: " + basis_res.error());
        calc._shells = std::move(*basis_res);

        if (auto r = calc.initialize(); !r)
            return std::unexpected("initialize: " + r.error());

        return calc;
    }

    bool is_identity(const Eigen::MatrixXd &M, double tol)
    {
        return (M - Eigen::MatrixXd::Identity(M.rows(), M.cols())).cwiseAbs().maxCoeff() < tol;
    }

    void check_group(const std::string &name,
                     const std::vector<int> &Z,
                     const std::vector<std::array<double, 3>> &xyz,
                     int charge, int mult,
                     const std::string &basis_name,
                     const std::string &expected_pg,
                     int expected_order)
    {
        constexpr double tol = 1e-9;

        auto calc_res = make_calculator(Z, xyz, charge, mult, basis_name);
        if (!calc_res)
        {
            fail(name + ": setup failed: " + calc_res.error());
            return;
        }
        HartreeFock::Calculator calc = std::move(*calc_res);

        if (calc._molecule._point_group != expected_pg)
        {
            fail(name + ": detected point group " + calc._molecule._point_group +
                 ", expected " + expected_pg);
            return;
        }

        auto ops_res = HartreeFock::Symmetry::build_group_operations(calc);
        if (!ops_res)
        {
            fail(name + ": build_group_operations error: " + ops_res.error());
            return;
        }
        const auto &ops = *ops_res;

        if (!ops.valid)
        {
            fail(name + ": expected valid group operations");
            return;
        }
        if (ops.order != expected_order ||
            static_cast<int>(ops.operations.size()) != expected_order)
        {
            fail(name + ": |G| = " + std::to_string(ops.order) +
                 ", expected " + std::to_string(expected_order));
            return;
        }

        // Exactly one identity matrix among the O_R.
        int n_identity = 0;
        for (const auto &op : ops.operations)
            if (is_identity(op.matrix, tol))
                ++n_identity;
        if (n_identity != 1)
            fail(name + ": expected exactly one identity O_R, found " + std::to_string(n_identity));

        // Every O_R orthogonal: O_R^T O_R == I.
        const Eigen::Index nb = ops.operations.front().matrix.rows();
        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nb, nb);
        for (const auto &op : ops.operations)
        {
            if (op.matrix.rows() != nb || op.matrix.cols() != nb)
            {
                fail(name + ": O_R (" + op.label + ") wrong shape");
                continue;
            }
            if ((op.matrix.transpose() * op.matrix - I).cwiseAbs().maxCoeff() > 1e-8)
                fail(name + ": O_R (" + op.label + ") is not orthogonal");
        }

        // Closure: for every pair (R,S) the product O_R O_S equals some O_T in the
        // set. This is the decisive representation-validity check.
        for (std::size_t r = 0; r < ops.operations.size(); ++r)
            for (std::size_t s = 0; s < ops.operations.size(); ++s)
            {
                const Eigen::MatrixXd prod = ops.operations[r].matrix * ops.operations[s].matrix;
                bool found = false;
                for (const auto &op : ops.operations)
                    if ((prod - op.matrix).cwiseAbs().maxCoeff() < 1e-7)
                    {
                        found = true;
                        break;
                    }
                if (!found)
                {
                    fail(name + ": product O[" + ops.operations[r].label + "] * O[" +
                         ops.operations[s].label + "] is not in the group (closure violated)");
                    return; // one report is enough
                }
            }
    }
} // namespace

int main()
{
    // NH3 — C3v, |G| = 6. First non-Abelian group (C3, sigma_v); build_sao_basis
    // would only ever see its Cs/C... Abelian subgroup, so this exercises the new
    // full-operation path.
    check_group(
        "NH3/C3v",
        {7, 1, 1, 1},
        {{{0.0000, 0.0000, 0.1173}},
         {{0.0000, 0.9377, -0.2738}},
         {{0.8121, -0.4689, -0.2738}},
         {{-0.8121, -0.4689, -0.2738}}},
        0, 1, "STO-3G", "C3v", 6);

    // CH4 — Td, |G| = 24. Non-Abelian with S4 and C3; the largest common case and
    // the strongest closure test.
    check_group(
        "CH4/Td",
        {6, 1, 1, 1, 1},
        {{{0.0000, 0.0000, 0.0000}},
         {{0.6276, 0.6276, 0.6276}},
         {{-0.6276, -0.6276, 0.6276}},
         {{-0.6276, 0.6276, -0.6276}},
         {{0.6276, -0.6276, -0.6276}}},
        0, 1, "STO-3G", "Td", 24);

    if (g_ok)
        std::cout << "group_operations: all checks passed\n";
    return g_ok ? 0 : 1;
}
