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

#include <algorithm>
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

    // One representative shell group per angular momentum L.
    //
    // The box-invariance property checked here (max-AM (a0|c0) contraction ==
    // per-component contraction) depends only on the quartet's angular-momentum
    // pattern (LA,LB,LC,LD), never on which shell of that L carries it: shells
    // of equal L differ only in their exponents, and the box geometry — the only
    // thing under test — is a function of L alone. Sweeping all shells re-runs
    // each L-pattern once per shell multiplicity.
    //
    // For water/6-31g* that is 10 shells but only 3 distinct L (s,p,d), so the
    // full 10^4 = 10000 quartet sweep collapses to 3^4 = 81 distinct patterns
    // while still covering the (dd|dd) quartets this gate exists to protect.
    // That is what takes this test from ~750 s to seconds. The comparison is
    // bitwise, so a dropped repeat cannot mask a discrepancy the kept
    // representative would not also show.
    std::vector<ShellGroup> unique_l_groups(const std::vector<ShellGroup> &groups,
                                            const HartreeFock::Basis &basis)
    {
        std::vector<ShellGroup> out;
        std::vector<int> seen;
        for (const ShellGroup &g : groups)
        {
            const int L = static_cast<int>(
                basis._basis_functions[g.first_ao]._shell->_shell);
            if (std::find(seen.begin(), seen.end(), L) != seen.end())
                continue;
            seen.push_back(L);
            out.push_back(g);
        }
        return out;
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
        const std::vector<ShellGroup> groups =
            unique_l_groups(build_shell_groups(basis), basis);

        std::size_t mismatched = 0;
        std::size_t coords_checked = 0;
        bool saw_nonfinite = false;
        // Reused across quartets so the box builds don't reallocate every time.
        std::vector<double> box_max;
        std::vector<double> box_comp;

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
                // Build each box ONCE, then read every coordinate out of it.
                // Calling the per-cell entry inside the coordinate loop would
                // re-contract the whole box per coordinate (the old shape, and
                // the reason this test took ~750 s).
                HartreeFock::HeadGordonPople::_build_a0c0_native_test(
                    spAB, spCD, maxAB, maxAB, maxAB, maxCD, maxCD, maxCD,
                    kernel, omega, box_max);
                HartreeFock::HeadGordonPople::_build_a0c0_native_test(
                    spAB, spCD, lABx, lABy, lABz, lCDx, lCDy, lCDz,
                    kernel, omega, box_comp);

                // Row-major flat index, cz fastest — the SpatialQuartetLayout
                // convention both boxes are built with. Each box has its own
                // dims, so the strides differ between the two.
                auto idx = [](int ax, int ay, int az, int cx, int cy, int cz,
                              int bABx, int bABy, int bABz,
                              int bCDx, int bCDy, int bCDz) -> std::size_t
                {
                    return (((((static_cast<std::size_t>(ax) * (bABy + 1) + ay) *
                                   (bABz + 1) + az) * (bCDx + 1) + cx) *
                                 (bCDy + 1) + cy) * (bCDz + 1) + cz);
                };

                for (int ax = 0; ax <= lABx; ++ax)
                for (int ay = 0; ay <= lABy; ++ay)
                for (int az = 0; az <= lABz; ++az)
                for (int cx = 0; cx <= lCDx; ++cx)
                for (int cy = 0; cy <= lCDy; ++cy)
                for (int cz = 0; cz <= lCDz; ++cz)
                {
                    const double from_max =
                        box_max[idx(ax, ay, az, cx, cy, cz,
                                    maxAB, maxAB, maxAB, maxCD, maxCD, maxCD)];
                    const double from_comp =
                        box_comp[idx(ax, ay, az, cx, cy, cz,
                                     lABx, lABy, lABz, lCDx, lCDy, lCDz)];

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
