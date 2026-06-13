// H-10 block-kernel gate (steps 2a + A1): verify each engine's shell-quartet
// block kernel (_contracted_eri_block) is bitwise-identical to evaluating that
// engine's per-component _contracted_eri_elem over every Cartesian component of
// the quartet. Covers both ObaraSaika (step 2a) and HeadGordonPople (step A1).
// Exercises a d-shell basis (water/6-31g*) so the multi-component blocks are
// non-trivial. No production entry point is involved — this only pins the
// block-shape refactor before any entry routes through it.

#include <cmath>
#include <cstddef>
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

    void block_call(
        Engine engine,
        const HartreeFock::Basis &basis,
        const ShellGroup &gA, const ShellGroup &gB,
        const ShellGroup &gC, const ShellGroup &gD,
        HartreeFock::ERIKernel kernel, double omega, double *block)
    {
        if (engine == Engine::OS)
            HartreeFock::ObaraSaika::_contracted_eri_block(
                basis, gA, gB, gC, gD, kernel, omega, block);
        else
            HartreeFock::HeadGordonPople::_contracted_eri_block(
                basis, gA, gB, gC, gD, kernel, omega, block);
    }

    void check_basis(Engine engine, const std::string &basis_name,
                     HartreeFock::ERIKernel kernel, double omega,
                     const std::string &kernel_label)
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

        std::size_t max_mismatch_quartets = 0;
        double max_abs_diff = 0.0;
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
                                   block.data());

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
                                        // Bitwise: the block calls the same
                                        // per-component kernel on the same per-
                                        // component ShellPairs, so equality is
                                        // exact.
                                        if (got != ref)
                                            quartet_bad = true;
                                    }
                        if (quartet_bad)
                            ++max_mismatch_quartets;
                    }

        if (max_mismatch_quartets != 0)
        {
            fail(std::string(engine_name(engine)) + " / " + kernel_label + " / " +
                 basis_name + ": " + std::to_string(max_mismatch_quartets) +
                 " quartets mismatched (max |diff| = " +
                 std::to_string(max_abs_diff) + ")");
        }
        else
        {
            std::cout << "OK  " << engine_name(engine) << " / " << kernel_label
                      << " / " << basis_name
                      << ": all shell-quartet blocks bitwise-match per-component "
                         "_contracted_eri_elem (max |diff| = "
                      << max_abs_diff << ")\n";
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
} // namespace

int main()
{
    check_engine(Engine::OS);
    check_engine(Engine::HGP);

    if (!g_ok)
    {
        std::cerr << "FAILED: a block kernel deviates from its per-component path\n";
        return 1;
    }
    std::cout << "PASSED: OS and HGP block kernels match their per-component paths\n";
    return 0;
}
