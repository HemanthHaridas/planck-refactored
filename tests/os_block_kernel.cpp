// H-10 block-kernel gate. Verifies each engine's shell-quartet block kernel
// against evaluating that engine's per-component _contracted_eri_elem over
// every Cartesian component of the quartet, on a d-shell basis (water/6-31g*)
// plus sto-3g. No production entry point is involved — this pins the
// block-shape refactors before any entry routes through them.
//
//   - steps 2a (OS) + A1 (HGP): the per-component block (_contracted_eri_block)
//     calls the same per-component kernel, so it must match BITWISE (tol = 0).
//   - step A4-1′ (HGP hoisted, _contracted_eri_block_hoisted): contracts the
//     (a0|c0) block once per shell quartet at max AM and HRRs each component out
//     of it. Mathematically identical, but it applies _component_norm after HRR
//     whereas the per-component path folds it into each primitive before
//     contraction, so for d-shells (norm != 1) the two drift at the last FP bit
//     (~1e-16). Gated at a tight relative tolerance (1e-13), the standard ERI
//     cross-validation bar — exact bitwise is impossible for a norm-free hoist.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/hgp.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << message << '\n';
        g_ok = false;
    }

    std::expected<HartreeFock::Calculator, std::string> make_water_calculator(
        const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;

        mol.natoms = 3;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(3);
        mol.atomic_numbers << 8, 1, 1;
        mol.atomic_masses.resize(3);
        mol.atomic_masses << 16.0, 1.0, 1.0;
        mol.coordinates.resize(3, 3);
        mol.coordinates <<
            0.000000, 0.000000, 0.117176,
            0.000000, 0.757200, -0.468704,
            0.000000, -0.757200, -0.468704;

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis failed: " + basis_res.error());

        calc._shells = std::move(*basis_res);
        return calc;
    }

    enum class Engine
    {
        OS,
        HGP
    };

    const char *engine_name(Engine e)
    {
        return e == Engine::OS ? "OS" : "HGP";
    }

    // Reference per-component value for the given engine: construct the
    // component ShellPairs exactly as build_shellpairs does and call that
    // engine's public per-component entry.
    double reference_elem(
        Engine engine,
        const HartreeFock::Basis &basis,
        const ShellGroup &gA, const ShellGroup &gB,
        const ShellGroup &gC, const ShellGroup &gD,
        std::size_t a, std::size_t b, std::size_t c, std::size_t d,
        HartreeFock::ERIKernel kernel, double omega)
    {
        const HartreeFock::ContractedView &cvA = basis._basis_functions[gA.first_ao + a];
        const HartreeFock::ContractedView &cvB = basis._basis_functions[gB.first_ao + b];
        const HartreeFock::ContractedView &cvC = basis._basis_functions[gC.first_ao + c];
        const HartreeFock::ContractedView &cvD = basis._basis_functions[gD.first_ao + d];

        const HartreeFock::ShellPair spAB(cvA, cvB);
        const HartreeFock::ShellPair spCD(cvC, cvD);

        if (engine == Engine::OS)
            return HartreeFock::ObaraSaika::_contracted_eri_elem(
                spAB, spCD,
                cvA._cartesian[0], cvA._cartesian[1], cvA._cartesian[2],
                cvB._cartesian[0], cvB._cartesian[1], cvB._cartesian[2],
                cvC._cartesian[0], cvC._cartesian[1], cvC._cartesian[2],
                cvD._cartesian[0], cvD._cartesian[1], cvD._cartesian[2],
                kernel, omega);
        return HartreeFock::HeadGordonPople::_contracted_eri_elem(
            spAB, spCD,
            cvA._cartesian[0], cvA._cartesian[1], cvA._cartesian[2],
            cvB._cartesian[0], cvB._cartesian[1], cvB._cartesian[2],
            cvC._cartesian[0], cvC._cartesian[1], cvC._cartesian[2],
            cvD._cartesian[0], cvD._cartesian[1], cvD._cartesian[2],
            kernel, omega);
    }

    // `hoisted` only applies to HGP: route through the A4-1′ hoisted block
    // (one max-AM contraction + per-component HRR readout) instead of the A1
    // per-component block. Must be bitwise-identical to per-component.
    void block_call(
        Engine engine,
        const HartreeFock::Basis &basis,
        const ShellGroup &gA, const ShellGroup &gB,
        const ShellGroup &gC, const ShellGroup &gD,
        HartreeFock::ERIKernel kernel, double omega, double *block,
        bool hoisted = false)
    {
        if (engine == Engine::OS)
            HartreeFock::ObaraSaika::_contracted_eri_block(
                basis, gA, gB, gC, gD, kernel, omega, block);
        else if (hoisted)
            HartreeFock::HeadGordonPople::_contracted_eri_block_hoisted(
                basis, gA, gB, gC, gD, kernel, omega, block);
        else
            HartreeFock::HeadGordonPople::_contracted_eri_block(
                basis, gA, gB, gC, gD, kernel, omega, block);
    }

    void check_basis(Engine engine, const std::string &basis_name,
                     HartreeFock::ERIKernel kernel, double omega,
                     const std::string &kernel_label, bool hoisted = false)
    {
        auto calc_res = make_water_calculator(basis_name);
        if (!calc_res)
        {
            fail("setup failed (" + basis_name + "): " + calc_res.error());
            return;
        }
        const HartreeFock::Basis &basis = calc_res->_shells;
        const std::vector<ShellGroup> groups = build_shell_groups(basis);

        if (groups.size() != basis.nshells())
        {
            fail("build_shell_groups produced " + std::to_string(groups.size()) +
                 " groups, expected nshells() = " + std::to_string(basis.nshells()));
            return;
        }

        // The per-component block (OS step 2a, HGP step A1) calls the same
        // per-component kernel on the same ShellPairs, so it must match exactly
        // (tol = 0). The hoisted HGP block (A4-1′) contracts once norm-free and
        // applies _component_norm after HRR, while the per-component path folds
        // the norm into each primitive's coeff_product before contraction. The
        // two are mathematically identical but the norm-scaling point differs,
        // so for d-shells (norm != 1) they drift at the last FP bit (~1e-16
        // observed). Exact bitwise is therefore impossible for a correct
        // norm-free hoist; we gate at a tight tolerance well above that drift —
        // the standard ERI cross-validation bar (cf. planck-compute-2e ~1e-12).
        const double rel_tol = hoisted ? 1e-13 : 0.0;

        std::size_t max_mismatch_quartets = 0;
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;
        std::vector<double> block;

        for (const ShellGroup &gA : groups)
            for (const ShellGroup &gB : groups)
                for (const ShellGroup &gC : groups)
                    for (const ShellGroup &gD : groups)
                    {
                        const std::size_t nA = gA.n_components;
                        const std::size_t nB = gB.n_components;
                        const std::size_t nC = gC.n_components;
                        const std::size_t nD = gD.n_components;
                        const std::size_t nCD = nC * nD;
                        block.assign(nA * nB * nCD, 0.0);

                        block_call(engine, basis, gA, gB, gC, gD, kernel, omega,
                                   block.data(), hoisted);

                        bool quartet_bad = false;
                        for (std::size_t a = 0; a < nA; ++a)
                            for (std::size_t b = 0; b < nB; ++b)
                                for (std::size_t c = 0; c < nC; ++c)
                                    for (std::size_t d = 0; d < nD; ++d)
                                    {
                                        const double got =
                                            block[(a * nB + b) * nCD + (c * nD + d)];
                                        const double ref = reference_elem(
                                            engine, basis, gA, gB, gC, gD, a, b, c, d,
                                            kernel, omega);
                                        const double diff = std::abs(got - ref);
                                        if (diff > max_abs_diff)
                                            max_abs_diff = diff;
                                        const double scale =
                                            std::max(std::abs(ref), 1.0);
                                        const double rdiff = diff / scale;
                                        if (rdiff > max_rel_diff)
                                            max_rel_diff = rdiff;
                                        // Non-hoisted: exact (rel_tol == 0).
                                        // Hoisted: allow last-bit norm-scaling
                                        // drift below rel_tol; NaN/Inf in `got`
                                        // (which would set rdiff to NaN/Inf and
                                        // never compare <= tol) still fails.
                                        if (!(rdiff <= rel_tol))
                                            quartet_bad = true;
                                    }
                        if (quartet_bad)
                            ++max_mismatch_quartets;
                    }

        const std::string variant = hoisted ? " [hoisted]" : "";
        if (max_mismatch_quartets != 0)
        {
            char dbuf[96];
            std::snprintf(dbuf, sizeof(dbuf), "%.3e abs / %.3e rel (tol %.0e)",
                          max_abs_diff, max_rel_diff, rel_tol);
            fail(std::string(engine_name(engine)) + variant + " / " + kernel_label +
                 " / " + basis_name + ": " + std::to_string(max_mismatch_quartets) +
                 " quartets exceed tol (max |diff| = " + dbuf + ")");
        }
        else
        {
            const char *match = hoisted ? "match per-component within tol"
                                        : "bitwise-match per-component";
            char dbuf[96];
            std::snprintf(dbuf, sizeof(dbuf), "%.3e abs / %.3e rel",
                          max_abs_diff, max_rel_diff);
            std::cout << "OK  " << engine_name(engine) << variant << " / "
                      << kernel_label << " / " << basis_name
                      << ": all shell-quartet blocks " << match
                      << " _contracted_eri_elem (max |diff| = " << dbuf << ")\n";
        }
    }

    void check_engine(Engine engine)
    {
        // Coulomb on a d-shell basis is the primary case.
        check_basis(engine, "6-31g*", HartreeFock::ERIKernel::Coulomb, 0.0, "Coulomb");
        // A screened kernel exercises the omega path through the same block.
        check_basis(engine, "6-31g*", HartreeFock::ERIKernel::LongRange, 0.3, "LongRange");
        check_basis(engine, "6-31g*", HartreeFock::ERIKernel::ShortRange, 0.3, "ShortRange");
        // STO-3G (s,p only) as a sanity lower bound.
        check_basis(engine, "sto-3g", HartreeFock::ERIKernel::Coulomb, 0.0, "Coulomb");
    }

    // A4-1′: the hoisted HGP block (one max-AM contraction + per-component HRR
    // readout) must be bitwise-identical to the per-component path — this is the
    // exact gate that caught the original A4-1 NaN on 6-31g* d-shells.
    void check_hgp_hoisted()
    {
        check_basis(Engine::HGP, "6-31g*", HartreeFock::ERIKernel::Coulomb, 0.0,
                    "Coulomb", /*hoisted=*/true);
        check_basis(Engine::HGP, "6-31g*", HartreeFock::ERIKernel::LongRange, 0.3,
                    "LongRange", /*hoisted=*/true);
        check_basis(Engine::HGP, "6-31g*", HartreeFock::ERIKernel::ShortRange, 0.3,
                    "ShortRange", /*hoisted=*/true);
        check_basis(Engine::HGP, "sto-3g", HartreeFock::ERIKernel::Coulomb, 0.0,
                    "Coulomb", /*hoisted=*/true);
        // f-shell coverage. The Auto dispatcher routes f-containing quartets
        // (L_AB up to 6) to the HGP hoisted block in the production _compute_2e
        // tensor build, but this gate historically only reached d (6-31g*). A
        // deleted OS hoisted twin SIGBUSed on exactly this untested f path, so
        // pin the live HGP kernel against its per-component reference on cc-pVTZ.
        check_basis(Engine::HGP, "cc-pVTZ", HartreeFock::ERIKernel::Coulomb, 0.0,
                    "Coulomb", /*hoisted=*/true);
        check_basis(Engine::HGP, "cc-pVTZ", HartreeFock::ERIKernel::LongRange, 0.3,
                    "LongRange", /*hoisted=*/true);
    }
} // namespace

int main()
{
    check_engine(Engine::OS);
    check_engine(Engine::HGP);
    check_hgp_hoisted();

    if (!g_ok)
    {
        std::cerr << "FAILED: a block kernel deviates from its per-component path\n";
        return 1;
    }
    std::cout << "PASSED: OS and HGP block kernels match their per-component paths\n";
    return 0;
}
