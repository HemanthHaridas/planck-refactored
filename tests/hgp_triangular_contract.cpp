// H-10 step A4-pre gate: validate that contracting HGP's (a0|c0) block ONCE per
// shell quartet at the max AM box (lAB = L_A+L_B, lCD = L_C+L_D per axis) yields,
// at every Cartesian component's (a0|c0) sub-block, exactly the values a
// per-component-AM contraction produces — bitwise.
//
// This is the load-bearing, riskiest piece of A4 (handoff §5): the hoisted path
// runs one contraction at max AM and reads each component's sub-block out of it,
// instead of re-running the per-primitive VRR+contract once per component. The
// §4 root cause that NaN'd A4-1 was the dense-cube HRR readout reaching cells no
// component needs; this gate proves the underlying *contraction* (the part A4
// actually shares) is invariant to box size on d-shells, which is the property
// the hoist relies on.
//
// hgp_vrr is strictly bottom-up, so a larger AM box only adds higher-AM cells
// and leaves every lower coordinate identical. We assert that explicitly on
// water/6-31g* (d-shells; the (dd|dd) quartets are the ones A4-1 broke) plus
// sto-3g as an s,p lower bound, for Coulomb / LongRange / ShortRange.
//
// Norm note (handoff §3.2): a ShellPair folds each component's _component_norm
// into PrimitivePair::coeff_product, so the (a0|c0) accumulator carries
// normA·normB·normC·normD — which differs per component. To isolate the
// box-size-invariance property (the only thing A4-pre's contraction shares),
// every build here uses ShellPairs constructed from the SAME component-0
// ContractedViews, holding the norm fixed; only the AM box and readout
// coordinate vary. The hoisted A4 path will likewise contract norm-free once
// and apply the per-component norm product at readout, exactly mirroring this.

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

    const char *kernel_label(HartreeFock::ERIKernel k)
    {
        switch (k)
        {
        case HartreeFock::ERIKernel::Coulomb:
            return "Coulomb";
        case HartreeFock::ERIKernel::LongRange:
            return "LongRange";
        case HartreeFock::ERIKernel::ShortRange:
            return "ShortRange";
        }
        return "?";
    }

    // For shell quartet (gA gB | gC gD), build the (a0|c0) accumulator ONCE at
    // the max AM box and compare every coordinate inside each component's own
    // box against a contraction sized to that single component.
    void check_basis(const std::string &basis_name,
                     HartreeFock::ERIKernel kernel, double omega)
    {
        auto calc_res = make_water_calculator(basis_name);
        if (!calc_res)
        {
            fail("setup failed (" + basis_name + "): " + calc_res.error());
            return;
        }
        const HartreeFock::Basis &basis = calc_res->_shells;
        const std::vector<ShellGroup> groups = build_shell_groups(basis);

        std::size_t mismatched = 0;
        std::size_t coords_checked = 0;
        bool saw_nonfinite = false;

        for (const ShellGroup &gA : groups)
        for (const ShellGroup &gB : groups)
        for (const ShellGroup &gC : groups)
        for (const ShellGroup &gD : groups)
        {
            const auto &cvA0 = basis._basis_functions[gA.first_ao];
            const auto &cvB0 = basis._basis_functions[gB.first_ao];
            const auto &cvC0 = basis._basis_functions[gC.first_ao];
            const auto &cvD0 = basis._basis_functions[gD.first_ao];

            // Max AM box: total L of the shell on each side, on every axis.
            const int LA = static_cast<int>(cvA0._shell->_shell);
            const int LB = static_cast<int>(cvB0._shell->_shell);
            const int LC = static_cast<int>(cvC0._shell->_shell);
            const int LD = static_cast<int>(cvD0._shell->_shell);
            const int maxAB = LA + LB;
            const int maxCD = LC + LD;

            // Hold the norm fixed: every build uses the component-0 ShellPairs,
            // so coeff_product (and thus the accumulator scale) is constant. We
            // vary only the AM box and the readout coordinate. This is what lets
            // the comparison be bitwise — a per-component ShellPair would fold a
            // different _component_norm and confound the box-invariance test.
            const HartreeFock::ShellPair spAB(cvA0, cvB0);
            const HartreeFock::ShellPair spCD(cvC0, cvD0);

            for (std::size_t a = 0; a < gA.n_components; ++a)
            for (std::size_t b = 0; b < gB.n_components; ++b)
            for (std::size_t c = 0; c < gC.n_components; ++c)
            for (std::size_t d = 0; d < gD.n_components; ++d)
            {
                const auto &cvA = basis._basis_functions[gA.first_ao + a];
                const auto &cvB = basis._basis_functions[gB.first_ao + b];
                const auto &cvC = basis._basis_functions[gC.first_ao + c];
                const auto &cvD = basis._basis_functions[gD.first_ao + d];

                const int lABx = cvA._cartesian[0] + cvB._cartesian[0];
                const int lABy = cvA._cartesian[1] + cvB._cartesian[1];
                const int lABz = cvA._cartesian[2] + cvB._cartesian[2];
                const int lCDx = cvC._cartesian[0] + cvD._cartesian[0];
                const int lCDy = cvC._cartesian[1] + cvD._cartesian[1];
                const int lCDz = cvC._cartesian[2] + cvD._cartesian[2];

                // Every coordinate inside this component's box must be reachable
                // and equal between the max build and a build sized to exactly
                // this component's AM.
                for (int ax = 0; ax <= lABx; ++ax)
                for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                for (int cx = 0; cx <= lCDx; ++cx)
                for (int cy = 0; cy <= lCDy; ++cy)
                for (int cz = 0; cz <= lCDz; ++cz)
                {
                    const double from_max =
                        HartreeFock::HeadGordonPople::_contract_a0c0_at_native_test(
                            spAB, spCD,
                            maxAB, maxAB, maxAB, maxCD, maxCD, maxCD,
                            ax, ay, az, cx, cy, cz, kernel, omega);

                    const double from_comp =
                        HartreeFock::HeadGordonPople::_contract_a0c0_at_native_test(
                            spAB, spCD,
                            lABx, lABy, lABz, lCDx, lCDy, lCDz,
                            ax, ay, az, cx, cy, cz, kernel, omega);

                    ++coords_checked;
                    if (!std::isfinite(from_max) || !std::isfinite(from_comp))
                        saw_nonfinite = true;
                    if (from_max != from_comp)
                        ++mismatched;
                }
            }
        }

        if (mismatched != 0 || saw_nonfinite)
        {
            fail(std::string("HGP / ") + kernel_label(kernel) + " / " + basis_name +
                 ": " + std::to_string(mismatched) + " coordinate mismatches" +
                 (saw_nonfinite ? " (non-finite values present)" : "") +
                 " over " + std::to_string(coords_checked) + " checked");
        }
        else
        {
            std::cout << "OK  HGP / " << kernel_label(kernel) << " / " << basis_name
                      << ": max-AM (a0|c0) contraction bitwise-equals per-component "
                         "build at all "
                      << coords_checked << " component coordinates\n";
        }
    }
} // namespace

int main()
{
    // d-shell basis: the (dd|dd) quartets are the ones A4-1's dense readout broke.
    check_basis("6-31g*", HartreeFock::ERIKernel::Coulomb, 0.0);
    check_basis("6-31g*", HartreeFock::ERIKernel::LongRange, 0.3);
    check_basis("6-31g*", HartreeFock::ERIKernel::ShortRange, 0.3);
    // s,p lower bound (max box == per-component box, so this must trivially pass).
    check_basis("sto-3g", HartreeFock::ERIKernel::Coulomb, 0.0);

    if (!g_ok)
    {
        std::cerr << "FAILED: max-AM (a0|c0) contraction deviates from per-component build\n";
        return 1;
    }
    std::cout << "PASSED: HGP max-AM (a0|c0) contraction is box-size invariant\n";
    return 0;
}
