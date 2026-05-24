// Cross-validation for the full-symmetry direct Fock (_symm) vs the production
// (D2h / no-symmetry) Fock — Phase 2 of the full-symmetry ERI reduction
// (docs/FULL_SYMMETRY_ERI_DESIGN.md).
//
// The decisive correctness gate: for a fixed (arbitrary but valid) density P, the
// skeleton+symmetrization Fock from os_symm must equal the full direct Fock from
// os.cpp to ~1e-10, on real molecules of nontrivial symmetry. If the petite-list
// representative test or the orbit-multiplicity weighting is wrong, the two differ.
//
// Tested on water (C2v, where the production engine also reduces, so this also
// confirms agreement with the existing path) and NH3 (C3v, where production only
// gets the Cs subgroup but _symm uses the full group — the new capability).

#include "integrals/os.h"
#include "symmetry/os_symm.h"
#include "integrals/shellpair.h"
#include "symmetry/group_operations.h"
#include "symmetry/fock_symmetrization.h"
#include "symmetry/symmetry.h"
#include "basis/basis.h"
#include "base/basis.h"
#include "base/types.h"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
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

    std::expected<HartreeFock::Calculator, std::string> make_calculator(
        const std::vector<int> &Z,
        const std::vector<std::array<double, 3>> &xyz,
        int charge, int multiplicity, const std::string &basis_name)
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
            mol.atomic_masses[static_cast<Eigen::Index>(i)] = 1.0;
            for (int c = 0; c < 3; ++c)
                mol.coordinates(static_cast<Eigen::Index>(i), c) = xyz[i][static_cast<std::size_t>(c)];
        }
        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        if (auto r = HartreeFock::Symmetry::detectSymmetry(mol, HartreeFock::Units::Angstrom); !r)
            return std::unexpected("detectSymmetry: " + r.error());
        const std::filesystem::path gbs = std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis: " + basis_res.error());
        calc._shells = std::move(*basis_res);
        if (auto r = calc.initialize(); !r)
            return std::unexpected("initialize: " + r.error());
        return calc;
    }

    Eigen::MatrixXd random_symmetric(Eigen::Index n, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd A(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                A(i, j) = dist(rng);
        return 0.5 * (A + A.transpose());
    }

    void check(const std::string &name,
               const std::vector<int> &Z,
               const std::vector<std::array<double, 3>> &xyz,
               int charge, int mult,
               const std::string &basis_name, const std::string &expected_pg)
    {
        auto calc_res = make_calculator(Z, xyz, charge, mult, basis_name);
        if (!calc_res)
        {
            fail(name + ": setup: " + calc_res.error());
            return;
        }
        HartreeFock::Calculator calc = std::move(*calc_res);
        if (calc._molecule._point_group != expected_pg)
        {
            fail(name + ": point group " + calc._molecule._point_group + " != " + expected_pg);
            return;
        }

        const std::size_t nb = calc._shells.nbasis();
        const auto pairs = build_shellpairs(calc._shells);

        auto ops_res = HartreeFock::Symmetry::build_group_operations(calc);
        if (!ops_res || !ops_res->valid)
        {
            fail(name + ": build_group_operations failed");
            return;
        }

        // The skeleton+symmetrization scheme reproduces F(P) only for a
        // SYMMETRY-ADAPTED density (O_R^T P O_R == P), which is what a converged SCF
        // produces. A generic random P is not group-invariant, so project it first
        // with the same operator the Fock build uses. Both the reference and the
        // symm Fock are then computed from this group-invariant P_sym.
        const Eigen::MatrixXd P_raw = random_symmetric(static_cast<Eigen::Index>(nb), 2024u);
        auto P_sym_res = HartreeFock::Symmetry::symmetrize_matrix(P_raw, *ops_res);
        if (!P_sym_res)
        {
            fail(name + ": failed to symmetrize test density: " + P_sym_res.error());
            return;
        }
        const Eigen::MatrixXd P = *P_sym_res;

        // Reference: production full direct Fock, no symmetry (sym_ops = nullptr).
        const Eigen::MatrixXd G_ref =
            HartreeFock::ObaraSaika::_compute_2e_fock(pairs, P, nb);

        // Full-symmetry skeleton + symmetrization.
        auto G_symm = HartreeFock::ObaraSaika::_compute_2e_fock_symm(
            pairs, calc._shells, P, nb, *ops_res);
        if (!G_symm)
        {
            fail(name + ": _compute_2e_fock_symm error: " + G_symm.error());
            return;
        }

        const double diff = (*G_symm - G_ref).cwiseAbs().maxCoeff();
        if (diff > 1e-9)
        {
            fail(name + ": symm Fock differs from production by " + std::to_string(diff) +
                 " (|G| = " + std::to_string(ops_res->order) + ")");
            return;
        }

        // UHF path: split P into Pa, Pb and compare both spin Focks.
        const Eigen::MatrixXd Pa = 0.6 * P;
        const Eigen::MatrixXd Pb = 0.4 * P;
        const auto [Ga_ref, Gb_ref] =
            HartreeFock::ObaraSaika::_compute_2e_fock_uhf(pairs, Pa, Pb, nb);
        auto uhf = HartreeFock::ObaraSaika::_compute_2e_fock_uhf_symm(
            pairs, calc._shells, Pa, Pb, nb, *ops_res);
        if (!uhf)
        {
            fail(name + ": _compute_2e_fock_uhf_symm error: " + uhf.error());
            return;
        }
        const double da = (uhf->first - Ga_ref).cwiseAbs().maxCoeff();
        const double db = (uhf->second - Gb_ref).cwiseAbs().maxCoeff();
        if (da > 1e-9 || db > 1e-9)
            fail(name + ": UHF symm Fock differs (da=" + std::to_string(da) +
                 ", db=" + std::to_string(db) + ")");
    }
} // namespace

int main()
{
    // Water — C2v (|G| = 4). Production also reduces here, so this confirms the new
    // path agrees on a case the existing engine already handles.
    check("H2O/C2v",
          {8, 1, 1},
          {{{0.000000, 0.000000, 0.117176}},
           {{0.000000, 0.757200, -0.468704}},
           {{0.000000, -0.757200, -0.468704}}},
          0, 1, "STO-3G", "C2v");

    // NH3 — C3v (|G| = 6). The full group exceeds the D2h-monomial ceiling: the new
    // capability. C3 makes this the real test of the petite list + multiplicity.
    check("NH3/C3v",
          {7, 1, 1, 1},
          {{{0.0000, 0.0000, 0.1173}},
           {{0.0000, 0.9377, -0.2738}},
           {{0.8121, -0.4689, -0.2738}},
           {{-0.8121, -0.4689, -0.2738}}},
          0, 1, "STO-3G", "C3v");

    if (g_ok)
        std::cout << "symm_fock_equivalence: all checks passed\n";
    return g_ok ? 0 : 1;
}
