// RI geometry-cache invalidation (Step G0 → G1).
//
// The RI caches (aux basis, 2-center metric, its factorization, 3-center
// tensor) all depend on atom positions, but the ensure_* guards key off
// dimensions / null-ness, never coordinates. So re-running ensure_ri_3c_ready
// on the SAME Calculator after the geometry moves returns the stale
// first-geometry tensor.
//
// This test builds the RI cache at geometry A, mutates _molecule._standard in
// place to geometry B (same atom count and basis, so npair × naux is
// unchanged — the exact case the dimension guard misses), rebuilds, and asserts
// the 3-center tensor reflects geometry B.
//
//   G0: with the dimension-only guard, this FAILS (stale tensor returned) —
//       proving the bug is real.
//   G1: with geometry-keyed invalidation, this PASSES.
//
// It must reuse the same Calculator across both geometries (mutate in place); a
// fresh Calculator would rebuild for the wrong reason and pass spuriously.

#include <Eigen/Dense>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "base/types.h"
#include "basis/basis.h"
#include "basis/rifit.h"
#include "post_hf/ri/ri_eri.h"

namespace
{
    bool g_ok = true;
    void fail(const std::string &m)
    {
        std::cerr << "FAIL: " << m << '\n';
        g_ok = false;
    }

    std::filesystem::path repo_root()
    {
        if (const char *env = std::getenv("BASIS_PATH"); env && *env)
            return std::filesystem::path(env).parent_path();
        return std::filesystem::current_path();
    }

    // Build a water Calculator with the RI inputs set, at the given standard
    // (Bohr) geometry. Basis + aux are read fresh each call so the only thing
    // that changes between the two builds in main() is the coordinate mutation.
    HartreeFock::Calculator make_calc(const std::filesystem::path &root,
                                      const Eigen::MatrixXd &standard_bohr)
    {
        HartreeFock::Molecule mol;
        mol.natoms = 3;
        mol.atomic_numbers.resize(3);
        mol.atomic_numbers << 8, 1, 1;
        mol._standard = standard_bohr;
        mol._standard_is_bohr = true;

        HartreeFock::Calculator calc;
        calc._molecule = mol;
        calc._basis._basis_name = "cc-pVDZ";
        calc._basis._basis_path = (root / "basis-sets").string();
        calc._mp2.use_ri = true;
        calc._mp2.ri_basis_name = "cc-pVDZ-RIFIT";
        calc._mp2.ri_basis_path = (root / "basis-sets").string();
        calc._mp2.ri_lindep = 1e-7;

        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            (root / "basis-sets" / "cc-pVDZ").string(), mol, HartreeFock::BasisType::Cartesian);
        if (basis_res)
            calc._shells = std::move(*basis_res);
        auto aux_res = HartreeFock::BasisFunctions::read_ri_basis(
            (root / "basis-sets" / "cc-pVDZ-RIFIT").string(), mol);
        if (aux_res)
            calc._ri_aux_basis =
                std::make_shared<HartreeFock::AuxBasis>(std::move(*aux_res));
        return calc;
    }
}

int main()
{
    using HartreeFock::Correlation::RI::ensure_ri_3c_ready;

    const auto root = repo_root();

    Eigen::MatrixXd geomA(3, 3);
    geomA << 0.0, 0.0, 0.117176,
        0.0, 0.757005, -0.468704,
        0.0, -0.757005, -0.468704;
    geomA *= 1.8897259886; // Angstrom -> Bohr

    // Geometry B: same atoms/basis, O pushed along z — npair × naux unchanged,
    // so the dimension guard cannot tell A from B.
    Eigen::MatrixXd geomB = geomA;
    geomB(0, 2) += 0.30; // move O by 0.30 Bohr

    // Build the cache at geometry A on a Calculator we will keep.
    HartreeFock::Calculator calc = make_calc(root, geomA);
    if (auto r = ensure_ri_3c_ready(calc); !r)
    {
        fail("ensure_ri_3c_ready(A) failed: " + r.error());
        return 1;
    }
    const Eigen::MatrixXd j3c_A = calc._ri_j3c;
    if (j3c_A.size() == 0)
    {
        fail("j3c at geometry A is empty");
        return 1;
    }

    // Move the geometry IN PLACE on the same Calculator, then rebuild.
    calc._molecule._standard = geomB;
    if (auto r = ensure_ri_3c_ready(calc); !r)
    {
        fail("ensure_ri_3c_ready(B) failed: " + r.error());
        return 1;
    }
    const Eigen::MatrixXd &j3c_B = calc._ri_j3c;

    if (j3c_A.rows() != j3c_B.rows() || j3c_A.cols() != j3c_B.cols())
    {
        fail("j3c dimensions changed between geometries — test premise broken");
        return 1;
    }

    const double max_change = (j3c_B - j3c_A).cwiseAbs().maxCoeff();
    std::cout << "j3c dims " << j3c_B.rows() << "x" << j3c_B.cols()
              << "  max|B-A| = " << max_change << '\n';

    // The 3-center integrals genuinely differ between A and B (O moved 0.3 Bohr),
    // so a correctly-invalidated cache produces a materially different tensor.
    // A stale cache returns A verbatim → max_change == 0.
    if (max_change < 1e-6)
        fail("RI 3-center tensor did NOT change after the geometry moved — the "
             "geometry-stale cache was returned (dimension-only guard).");

    if (g_ok)
        std::cout << "PASS: ri_cache_invalidation\n";
    return g_ok ? 0 : 1;
}
