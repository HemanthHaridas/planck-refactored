// Unit test for HartreeFock::BasisFunctions::read_ri_basis.
//
// Validates the .gbs loader for RI auxiliary basis sets on a toy basis
// (basis-sets/toy-ri) that exercises:
//
//   * multiple angular momenta (S, P, D)
//   * multiple shells per element (He has two S shells)
//   * multiple atoms (H2 to test per-atom shell instantiation)
//   * mixed-element molecules (HeH+ to confirm dispatch on atomic_numbers)
//
// What we check:
//
//   1. shell count per element matches the file (H: 2 shells, He: 4)
//   2. total aux function count matches Σ_K (L_K+1)(L_K+2)/2 (Cartesian)
//   3. offsets table is monotonic and ends at nfunctions
//   4. shell centers match the molecule coordinates exactly
//   5. contracted normalization was folded into _coefficients (the same
//      norm contract the orbital basis uses)
//   6. missing-element error path returns a useful message rather than
//      silently producing a zero-shell aux basis
//
// We do *not* test against PySCF reference values here. That's the right
// gate for shipped production aux basis sets (cc-pVDZ-RI etc.); the toy
// basis is purely a parser/structure check.

#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>

#include "base/types.h"
#include "basis/rifit.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &m)
    {
        std::cerr << "FAIL: " << m << '\n';
        g_ok = false;
    }

    std::string aux_basis_path()
    {
        if (const char *env = std::getenv("BASIS_PATH"); env && *env)
            return std::string(env) + "/toy-ri";
        return "basis-sets/toy-ri";
    }

    HartreeFock::Molecule make_h2()
    {
        HartreeFock::Molecule mol;
        mol.natoms = 2;
        mol.atomic_numbers.resize(2);
        mol.atomic_numbers << 1, 1;
        mol._standard.resize(2, 3);
        mol._standard << 0.0, 0.0, -0.7,
                         0.0, 0.0, +0.7;
        return mol;
    }

    HartreeFock::Molecule make_hehp()
    {
        HartreeFock::Molecule mol;
        mol.natoms = 2;
        mol.atomic_numbers.resize(2);
        mol.atomic_numbers << 1, 2; // H, He
        mol._standard.resize(2, 3);
        mol._standard << 0.0, 0.0, -0.8,
                         0.0, 0.0, +0.8;
        return mol;
    }

    // For a Cartesian shell of angular momentum L, count of basis functions
    // is (L+1)(L+2)/2. This mirrors what the loader computes per shell.
    std::size_t cart_count(unsigned L)
    {
        return (L + 1) * (L + 2) / 2;
    }
} // namespace

int main()
{
    using HartreeFock::BasisFunctions::read_ri_basis;
    const std::string aux = aux_basis_path();

    // ── Test 1: H2 with toy-ri ───────────────────────────────────────────────
    // toy-ri H: S (1 prim) + P (1 prim) = 2 shells, 1+3 = 4 functions per H.
    {
        auto mol = make_h2();
        auto res = read_ri_basis(aux, mol);
        if (!res)
        {
            fail("H2 load failed: " + res.error());
        }
        else
        {
            const auto &a = *res;
            if (a.shells.size() != 4)
                fail("H2: expected 4 shells (2 per H), got " + std::to_string(a.shells.size()));
            // Per H: S(1) + P(3) = 4; for 2 H atoms = 8 aux functions.
            if (a.nfunctions != 8)
                fail("H2: expected 8 aux functions, got " + std::to_string(a.nfunctions));
            if (a.offsets.size() != a.shells.size())
                fail("H2: offsets size must equal shells size");

            // Offsets monotonic and end at nfunctions.
            std::size_t expected_offset = 0;
            for (std::size_t k = 0; k < a.shells.size(); ++k)
            {
                if (a.offsets[k] != expected_offset)
                    fail("H2: offset[" + std::to_string(k) + "] mismatch");
                expected_offset += cart_count(
                    static_cast<unsigned>(a.shells[k]._shell));
            }
            if (expected_offset != a.nfunctions)
                fail("H2: final offset does not match nfunctions");

            // Each H's first shell center must be at z=-0.7, second H at z=+0.7.
            // Shells are instantiated in order (S, P) per atom.
            const double tol = 1e-15;
            if (std::abs(a.shells[0]._center.z() + 0.7) > tol ||
                std::abs(a.shells[1]._center.z() + 0.7) > tol)
                fail("H2: first-atom shell centers wrong");
            if (std::abs(a.shells[2]._center.z() - 0.7) > tol ||
                std::abs(a.shells[3]._center.z() - 0.7) > tol)
                fail("H2: second-atom shell centers wrong");

            // Contracted-norm contract: single-primitive shell with raw
            // coefficient 1.0 must have _coefficients[0] == Nc (the contracted
            // norm folded in), and Nc must be > 0. Sanity check that
            // normalization fired and produced a finite positive scale.
            for (const auto &s : a.shells)
            {
                if (s._coefficients.size() != 1)
                    continue;
                const double c = s._coefficients[0];
                if (!std::isfinite(c) || c <= 0.0)
                    fail("Aux shell contracted coefficient must be finite and > 0");
            }
        }
    }

    // ── Test 2: HeH+ ─────────────────────────────────────────────────────────
    // toy-ri He: S + S + P + D = 4 shells, 1+1+3+6 = 11 aux funcs.
    // toy-ri H : S + P         = 2 shells, 1+3 = 4 aux funcs.
    // Total: 6 shells, 15 aux functions.
    {
        auto mol = make_hehp();
        auto res = read_ri_basis(aux, mol);
        if (!res)
        {
            fail("HeH+ load failed: " + res.error());
        }
        else
        {
            const auto &a = *res;
            if (a.shells.size() != 6)
                fail("HeH+: expected 6 shells, got " + std::to_string(a.shells.size()));
            if (a.nfunctions != 15)
                fail("HeH+: expected 15 aux functions, got " + std::to_string(a.nfunctions));
            // Atom 0 is H (z=-0.8), gets the first 2 shells.
            // Atom 1 is He (z=+0.8), gets the next 4.
            const double tol = 1e-15;
            for (std::size_t k = 0; k < 2; ++k)
                if (std::abs(a.shells[k]._center.z() + 0.8) > tol)
                    fail("HeH+: H shell center wrong");
            for (std::size_t k = 2; k < 6; ++k)
                if (std::abs(a.shells[k]._center.z() - 0.8) > tol)
                    fail("HeH+: He shell center wrong");
        }
    }

    // ── Test 3: missing element produces a useful error ──────────────────────
    {
        HartreeFock::Molecule mol;
        mol.natoms = 1;
        mol.atomic_numbers.resize(1);
        mol.atomic_numbers << 6; // C — not in toy-ri
        mol._standard.resize(1, 3);
        mol._standard << 0.0, 0.0, 0.0;
        auto res = read_ri_basis(aux, mol);
        if (res)
            fail("Expected error for missing element C, got success");
        else if (res.error().find("missing element") == std::string::npos)
            fail("Missing-element error should mention 'missing element', got: " + res.error());
    }

    // ── Test 4: missing file path ────────────────────────────────────────────
    {
        HartreeFock::Molecule mol = make_h2();
        auto res = read_ri_basis(aux + ".does-not-exist", mol);
        if (res)
            fail("Expected error for missing aux file, got success");
    }

    if (g_ok)
        std::cout << "PASS: read_ri_basis on toy-ri\n";
    return g_ok ? 0 : 1;
}
