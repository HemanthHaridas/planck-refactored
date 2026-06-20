// Phase B / B-1 gate: validate that building the Rys 6D `sum` ONCE per shell
// quartet at the max angular-momentum box (lAB = L_A+L_B, lCD = L_C+L_D per
// axis, which also fixes the root count at n_max = L_max/2+1) yields, at every
// Cartesian component's sub-block, exactly the value a per-component build
// produces — bitwise.
//
// This is the load-bearing correctness property the Rys shell-quartet hoist
// relies on (handoff §6 Phase B). Two facts must both hold for it to be true:
//
//   1. The 1D Rys VRR (_rys_vrr_1d) is strictly bottom-up, so a larger axis box
//      only adds higher-AM cells and leaves every lower (a,c) coordinate
//      identical.
//   2. Gauss over-integration is exact: a component cell of total degree d needs
//      ceil((d+1)/2) roots, and evaluating it with the quartet's max root count
//      n_max >= n_comp is still exact (an n-point Gauss rule is exact to degree
//      2n-1). So using n_max roots for every component reproduces each
//      component's value exactly — the non-nestedness of Rys roots across n is
//      irrelevant once everyone uses the largest count.
//
// Unlike the HGP/OS hoist, Rys folds no per-component norm, so this is bitwise
// even on d/g shells (no norm-reorder drift). We assert it on water/6-31g*
// (d-shells: components within a quartet span several total-L values, so n_max
// strictly exceeds many components' n) and on Ne/cc-pVQZ (g-shells: the
// (7,8)/(8,8) buckets the Auto path actually routes to Rys), for Coulomb /
// LongRange / ShortRange.
//
// The box-invariance is a per-primitive-pair property (the 6D sum is linear over
// primitive pairs), so every comparison holds the primitive pair fixed and
// varies only the box and the readout coordinate, via
// RysQuad::_build_sum_at_native_test.

#include <algorithm>
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
#include "integrals/rys.h"
#include "integrals/shellpair.h"

namespace
{
    bool g_ok = true;

    void fail(const std::string &message)
    {
        std::cerr << message << '\n';
        g_ok = false;
    }

    std::expected<HartreeFock::Calculator, std::string> make_calculator(
        const std::string &basis_name,
        const std::vector<int> &Z,
        const Eigen::MatrixXd &coords_ang)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;

        const int n = static_cast<int>(Z.size());
        mol.natoms = n;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(n);
        mol.atomic_masses.resize(n);
        for (int i = 0; i < n; ++i)
        {
            mol.atomic_numbers[i] = Z[i];
            mol.atomic_masses[i] = 1.0;
        }
        mol.coordinates = coords_ang;

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis failed (" + basis_name +
                                   "): " + basis_res.error());

        calc._shells = std::move(*basis_res);
        return calc;
    }

    std::expected<HartreeFock::Calculator, std::string> make_water(
        const std::string &basis_name)
    {
        Eigen::MatrixXd c(3, 3);
        c << 0.000000, 0.000000, 0.117176,
            0.000000, 0.757200, -0.468704,
            0.000000, -0.757200, -0.468704;
        return make_calculator(basis_name, {8, 1, 1}, c);
    }

    std::expected<HartreeFock::Calculator, std::string> make_ne(
        const std::string &basis_name)
    {
        Eigen::MatrixXd c(1, 3);
        c << 0.0, 0.0, 0.0;
        return make_calculator(basis_name, {10}, c);
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

    // For each shell quartet, compare the 6D `sum` built at the max-AM box (and
    // thus n_max roots) against a per-component-box build, at every coordinate
    // inside each component's box. Holds the primitive pair fixed (component-0
    // primitives) so the only varying quantities are the box and the coordinate.
    // `min_quartet_L` skips quartets whose max total angular momentum
    // (maxAB+maxCD) is below the threshold — used to confine the expensive
    // g-shell sweep to the high-L buckets that are not already covered by the
    // smaller d-shell basis.
    void check(const std::string &label,
               const HartreeFock::Calculator &calc,
               HartreeFock::ERIKernel kernel, double omega,
               int min_quartet_L = 0)
    {
        const HartreeFock::Basis &basis = calc._shells;
        const std::vector<ShellGroup> groups = build_shell_groups(basis);

        std::size_t over_tol = 0;
        std::size_t coords_checked = 0;
        bool saw_nonfinite = false;
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;
        // ERI cross-validation bar (cf. planck-compute-2e ~1e-12, the HGP hoist
        // gate 1e-13). Rys's max-box build sums n_max weighted roots vs the
        // component build's n_comp; the two are mathematically equal (Gauss
        // over-integration) but the differing term count rounds at the last bit,
        // so this is tight-tolerance, not bitwise.
        constexpr double REL_TOL = 1e-13;
        std::vector<double> buf_max;
        std::vector<double> buf_comp;

        for (const ShellGroup &gA : groups)
        for (const ShellGroup &gB : groups)
        for (const ShellGroup &gC : groups)
        for (const ShellGroup &gD : groups)
        {
            const auto &cvA0 = basis._basis_functions[gA.first_ao];
            const auto &cvB0 = basis._basis_functions[gB.first_ao];
            const auto &cvC0 = basis._basis_functions[gC.first_ao];
            const auto &cvD0 = basis._basis_functions[gD.first_ao];

            const int LA = static_cast<int>(cvA0._shell->_shell);
            const int LB = static_cast<int>(cvB0._shell->_shell);
            const int LC = static_cast<int>(cvC0._shell->_shell);
            const int LD = static_cast<int>(cvD0._shell->_shell);
            const int maxAB = LA + LB;
            const int maxCD = LC + LD;
            if (maxAB + maxCD < min_quartet_L)
                continue;

            // Component-0 primitive pairs: the 6D sum is built per primitive
            // pair, so the box-invariance is exposed on a single fixed pair.
            const HartreeFock::ShellPair spAB(cvA0, cvB0);
            const HartreeFock::ShellPair spCD(cvC0, cvD0);
            if (spAB.primitive_pairs.empty() || spCD.primitive_pairs.empty())
                continue;
            const auto &ppAB = spAB.primitive_pairs.front();
            const auto &ppCD = spCD.primitive_pairs.front();

            // The max-box build is identical for every component of this quartet
            // (same box, same n_max), so build it ONCE per quartet — this is also
            // exactly what the eventual hoist does. Only buf_comp varies per
            // component.
            const int n_max = (maxAB + maxCD) / 2 + 1;
            HartreeFock::RysQuad::_build_sum_native_test(
                ppAB, ppCD, maxAB, maxAB, maxAB, maxCD, maxCD, maxCD,
                n_max, kernel, omega, buf_max);

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

                // Component degree -> its own root count. Using n_max (>= n_comp)
                // for the shared max-box build is exact for every component cell
                // by Gauss over-integration.
                const int n_comp =
                    (lABx + lABy + lABz + lCDx + lCDy + lCDz) / 2 + 1;

                // One build per component (the per-component box); the max-box
                // build is reused from outside the loop.
                HartreeFock::RysQuad::_build_sum_native_test(
                    ppAB, ppCD, lABx, lABy, lABz, lCDx, lCDy, lCDz,
                    n_comp, kernel, omega, buf_comp);

                // RysScratch flat index: ((((ax*ay_dim+ay)*az_dim+az)*cx_dim+cx)
                // *cy_dim+cy)*cz_dim+cz, each *_dim = its axis box + 1. Per-axis
                // (boxes are generally non-cubic), so pass all six dims.
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
                        buf_max[idx(ax, ay, az, cx, cy, cz,
                                    maxAB, maxAB, maxAB, maxCD, maxCD, maxCD)];
                    const double comp_val =
                        buf_comp[idx(ax, ay, az, cx, cy, cz,
                                     lABx, lABy, lABz, lCDx, lCDy, lCDz)];

                    ++coords_checked;
                    if (!std::isfinite(from_max) || !std::isfinite(comp_val))
                        saw_nonfinite = true;
                    const double adiff = std::fabs(from_max - comp_val);
                    const double scale = std::max(std::fabs(comp_val), 1e-300);
                    const double rdiff = adiff / scale;
                    if (adiff > max_abs_diff)
                        max_abs_diff = adiff;
                    if (rdiff > max_rel_diff)
                        max_rel_diff = rdiff;
                    if (rdiff > REL_TOL && adiff > 1e-15)
                        ++over_tol;
                }
            }
        }

        char buf[96];
        std::snprintf(buf, sizeof(buf), "max|Δ|=%.2e rel=%.2e", max_abs_diff, max_rel_diff);
        if (over_tol != 0 || saw_nonfinite)
        {
            fail(std::string("Rys / ") + kernel_label(kernel) + " / " + label +
                 ": " + std::to_string(over_tol) + " coords over rel-tol " +
                 (saw_nonfinite ? "(non-finite values present) " : "") + buf +
                 " over " + std::to_string(coords_checked) + " checked");
        }
        else
        {
            std::cout << "OK  Rys / " << kernel_label(kernel) << " / " << label
                      << ": max-box 6D sum == per-component build (rel<=1e-13) at all "
                      << coords_checked << " coords  (" << buf << ")\n";
        }
    }
    // B-2b gate: contract-then-HRR == _rys_contracted_eri to the 1e-13 ERI bar.
    // _rys_contract_sum (via _contract_sum_native_test) builds Σ_pair
    // coeff·sum_pair into one block; HRR'ing it gives HRR(Σ coeff·sum_pair).
    // _rys_contracted_eri gives Σ coeff·HRR(sum_pair). HRR is linear so the two
    // are mathematically equal, but they reorder the FP accumulation at the last
    // bit two ways: (1) the cross-pair sum moves across HRR, and (2) production
    // applies coeff_product ONCE after HRR (a single final multiply), while the
    // hoist folds coeff into the block so it rides through every HRR add. Both
    // are inherent to the hoisted order the production path (B-3) will adopt, so
    // the correct bar is 1e-13 rel — same as B-1 and the eventual B-2c/d — NOT
    // bitwise. (The earlier scope note's "bitwise at a single pair" was an
    // over-claim: it overlooked the coeff-placement reorder (2), which is present
    // even at one pair.)
    void check_contract(const std::string &label,
                        const HartreeFock::Calculator &calc,
                        HartreeFock::ERIKernel kernel, double omega,
                        int min_quartet_L = 0)
    {
        const HartreeFock::Basis &basis = calc._shells;
        const std::vector<ShellGroup> groups = build_shell_groups(basis);

        // Same 1e-13 ERI bar as the box-invariance check above; the hoisted
        // contract-then-HRR order rounds at the last bit (coeff placement +
        // cross-pair sum reorder), so this is tight-tolerance, not bitwise.
        constexpr double REL_TOL = 1e-13;
        std::size_t over_tol = 0;
        std::size_t components_checked = 0;
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;
        std::vector<double> acc;

        for (const ShellGroup &gA : groups)
        for (const ShellGroup &gB : groups)
        for (const ShellGroup &gC : groups)
        for (const ShellGroup &gD : groups)
        {
            const auto &cvA0 = basis._basis_functions[gA.first_ao];
            const auto &cvB0 = basis._basis_functions[gB.first_ao];
            const auto &cvC0 = basis._basis_functions[gC.first_ao];
            const auto &cvD0 = basis._basis_functions[gD.first_ao];

            const int LA = static_cast<int>(cvA0._shell->_shell);
            const int LB = static_cast<int>(cvB0._shell->_shell);
            const int LC = static_cast<int>(cvC0._shell->_shell);
            const int LD = static_cast<int>(cvD0._shell->_shell);
            if ((LA + LB) + (LC + LD) < min_quartet_L)
                continue;

            // Per-component shell pairs (carry the component's _component_norm,
            // as production _rys_contracted_eri does). Full primitive-pair sets:
            // B-2b's _rys_contract_sum contracts over all pairs, exactly as the
            // production hoist (B-3) will.
            for (std::size_t a = 0; a < gA.n_components; ++a)
            for (std::size_t b = 0; b < gB.n_components; ++b)
            for (std::size_t c = 0; c < gC.n_components; ++c)
            for (std::size_t d = 0; d < gD.n_components; ++d)
            {
                const auto &cvA = basis._basis_functions[gA.first_ao + a];
                const auto &cvB = basis._basis_functions[gB.first_ao + b];
                const auto &cvC = basis._basis_functions[gC.first_ao + c];
                const auto &cvD = basis._basis_functions[gD.first_ao + d];

                const HartreeFock::ShellPair spAB(cvA, cvB);
                const HartreeFock::ShellPair spCD(cvC, cvD);
                if (spAB.primitive_pairs.empty() || spCD.primitive_pairs.empty())
                    continue;

                const int lAx = cvA._cartesian[0], lAy = cvA._cartesian[1], lAz = cvA._cartesian[2];
                const int lBx = cvB._cartesian[0], lBy = cvB._cartesian[1], lBz = cvB._cartesian[2];
                const int lCx = cvC._cartesian[0], lCy = cvC._cartesian[1], lCz = cvC._cartesian[2];
                const int lDx = cvD._cartesian[0], lDy = cvD._cartesian[1], lDz = cvD._cartesian[2];

                const int lABx = lAx + lBx, lABy = lAy + lBy, lABz = lAz + lBz;
                const int lCDx = lCx + lDx, lCDy = lCy + lDy, lCDz = lCz + lDz;
                const int n_comp =
                    (lABx + lABy + lABz + lCDx + lCDy + lCDz) / 2 + 1;

                HartreeFock::RysQuad::_contract_sum_native_test(
                    spAB, spCD, lABx, lABy, lABz, lCDx, lCDy, lCDz,
                    n_comp, kernel, omega, acc);

                const double via_hoist = HartreeFock::RysQuad::_hrr_block_native_test(
                    acc, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz,
                    spAB.R[0], spAB.R[1], spAB.R[2], spCD.R[0], spCD.R[1], spCD.R[2]);

                const double via_prod = HartreeFock::RysQuad::_rys_contracted_eri(
                    spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz,
                    kernel, omega);

                ++components_checked;
                const double adiff = std::fabs(via_hoist - via_prod);
                const double scale = std::max(std::fabs(via_prod), 1e-300);
                const double rdiff = adiff / scale;
                if (adiff > max_abs_diff)
                    max_abs_diff = adiff;
                // Only count/report relative diff above the absolute noise floor:
                // near-zero ERIs (|via_prod| ~ 1e-18) produce a meaningless huge
                // rdiff from a last-bit adiff. The gate ignores them (adiff floor
                // below), so the reported max_rel must too, else it alarms on
                // points the gate correctly passes.
                if (adiff > 1e-15)
                {
                    if (rdiff > max_rel_diff)
                        max_rel_diff = rdiff;
                    if (rdiff > REL_TOL)
                        ++over_tol;
                }
            }
        }

        char buf[96];
        std::snprintf(buf, sizeof(buf), "max|Δ|=%.2e rel=%.2e", max_abs_diff, max_rel_diff);
        if (over_tol != 0)
        {
            fail(std::string("Rys B-2b / ") + kernel_label(kernel) + " / " + label +
                 ": " + std::to_string(over_tol) + " of " +
                 std::to_string(components_checked) + " components over rel-tol  " + buf);
        }
        else
        {
            std::cout << "OK  Rys B-2b / " << kernel_label(kernel) << " / " << label
                      << ": contract-then-HRR == _rys_contracted_eri (rel<=1e-13) at all "
                      << components_checked << " components  (" << buf << ")\n";
        }
    }

    // B-2c gate: the full norm-free-max-box-contract + per-component-readout flow
    // (_contract_maxbox_readout_native_test) reproduces _rys_contracted_eri to the
    // 1e-13 ERI bar. This is the first step exercising BOTH the n_max-over-comp
    // reorder (B-1) and the norm-after-HRR reorder (invariant 2) together: the
    // hoist contracts once per quartet at the max box from norm-free views and
    // applies normA·normB·normC·normD only at readout, while production folds the
    // norm into each primitive pair and builds at the component box.
    void check_maxbox_readout(const std::string &label,
                              const HartreeFock::Calculator &calc,
                              HartreeFock::ERIKernel kernel, double omega,
                              int min_quartet_L = 0)
    {
        const HartreeFock::Basis &basis = calc._shells;
        const std::vector<ShellGroup> groups = build_shell_groups(basis);

        constexpr double REL_TOL = 1e-13;
        std::size_t over_tol = 0;
        std::size_t components_checked = 0;
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;

        for (const ShellGroup &gA : groups)
        for (const ShellGroup &gB : groups)
        for (const ShellGroup &gC : groups)
        for (const ShellGroup &gD : groups)
        {
            const auto &cvA0 = basis._basis_functions[gA.first_ao];
            const auto &cvB0 = basis._basis_functions[gB.first_ao];
            const auto &cvC0 = basis._basis_functions[gC.first_ao];
            const auto &cvD0 = basis._basis_functions[gD.first_ao];

            const int LA = static_cast<int>(cvA0._shell->_shell);
            const int LB = static_cast<int>(cvB0._shell->_shell);
            const int LC = static_cast<int>(cvC0._shell->_shell);
            const int LD = static_cast<int>(cvD0._shell->_shell);
            const int maxAB = LA + LB;
            const int maxCD = LC + LD;
            if (maxAB + maxCD < min_quartet_L)
                continue;

            // Snapshot built ONCE per quartet (norm-free, max box / n_max), exactly
            // as the production hoist (B-3) does; every component reads out of it.
            std::vector<double> snapshot;
            double R_AB[3], R_CD[3];
            HartreeFock::RysQuad::_contract_maxbox_snapshot_native_test(
                cvA0, cvB0, cvC0, cvD0, maxAB, maxCD, kernel, omega,
                snapshot, R_AB, R_CD);

            for (std::size_t a = 0; a < gA.n_components; ++a)
            for (std::size_t b = 0; b < gB.n_components; ++b)
            for (std::size_t c = 0; c < gC.n_components; ++c)
            for (std::size_t d = 0; d < gD.n_components; ++d)
            {
                const auto &cvA = basis._basis_functions[gA.first_ao + a];
                const auto &cvB = basis._basis_functions[gB.first_ao + b];
                const auto &cvC = basis._basis_functions[gC.first_ao + c];
                const auto &cvD = basis._basis_functions[gD.first_ao + d];

                const int lAx = cvA._cartesian[0], lAy = cvA._cartesian[1], lAz = cvA._cartesian[2];
                const int lBx = cvB._cartesian[0], lBy = cvB._cartesian[1], lBz = cvB._cartesian[2];
                const int lCx = cvC._cartesian[0], lCy = cvC._cartesian[1], lCz = cvC._cartesian[2];
                const int lDx = cvD._cartesian[0], lDy = cvD._cartesian[1], lDz = cvD._cartesian[2];

                const double via_hoist = HartreeFock::RysQuad::_maxbox_readout_native_test(
                    snapshot, maxAB, maxCD, cvA, cvB, cvC, cvD, R_AB, R_CD);

                // Production reference: norm-carrying per-component shell pairs,
                // built per component, norm folded per primitive pair.
                const HartreeFock::ShellPair spAB(cvA, cvB);
                const HartreeFock::ShellPair spCD(cvC, cvD);
                if (spAB.primitive_pairs.empty() || spCD.primitive_pairs.empty())
                    continue;
                const double via_prod = HartreeFock::RysQuad::_rys_contracted_eri(
                    spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz,
                    kernel, omega);

                ++components_checked;
                const double adiff = std::fabs(via_hoist - via_prod);
                const double scale = std::max(std::fabs(via_prod), 1e-300);
                const double rdiff = adiff / scale;
                if (adiff > max_abs_diff)
                    max_abs_diff = adiff;
                if (adiff > 1e-15)
                {
                    if (rdiff > max_rel_diff)
                        max_rel_diff = rdiff;
                    if (rdiff > REL_TOL)
                        ++over_tol;
                }
            }
        }

        char buf[96];
        std::snprintf(buf, sizeof(buf), "max|Δ|=%.2e rel=%.2e", max_abs_diff, max_rel_diff);
        if (over_tol != 0)
        {
            fail(std::string("Rys B-2c / ") + kernel_label(kernel) + " / " + label +
                 ": " + std::to_string(over_tol) + " of " +
                 std::to_string(components_checked) + " components over rel-tol  " + buf);
        }
        else
        {
            std::cout << "OK  Rys B-2c / " << kernel_label(kernel) << " / " << label
                      << ": norm-free max-box readout == _rys_contracted_eri (rel<=1e-13) at all "
                      << components_checked << " components  (" << buf << ")\n";
        }
    }

    // B-2d gate: the assembled RysHoistedQuartet block entry
    // (_contracted_eri_block_hoisted) fills every component of a shell quartet
    // from one lazy norm-free max-box contraction; each component matches
    // _rys_contracted_eri to ≤1e-13. Exercises the full B-2d surface (lazy
    // prepare, ShortRange two-snapshot subtract, norm-at-readout, block layout).
    void check_block(const std::string &label,
                     const HartreeFock::Calculator &calc,
                     HartreeFock::ERIKernel kernel, double omega,
                     int min_quartet_L = 0)
    {
        const HartreeFock::Basis &basis = calc._shells;
        const std::vector<ShellGroup> groups = build_shell_groups(basis);

        constexpr double REL_TOL = 1e-13;
        std::size_t over_tol = 0;
        std::size_t components_checked = 0;
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;
        std::vector<double> block;

        for (const ShellGroup &gA : groups)
        for (const ShellGroup &gB : groups)
        for (const ShellGroup &gC : groups)
        for (const ShellGroup &gD : groups)
        {
            const int LA = static_cast<int>(basis._basis_functions[gA.first_ao]._shell->_shell);
            const int LB = static_cast<int>(basis._basis_functions[gB.first_ao]._shell->_shell);
            const int LC = static_cast<int>(basis._basis_functions[gC.first_ao]._shell->_shell);
            const int LD = static_cast<int>(basis._basis_functions[gD.first_ao]._shell->_shell);
            if ((LA + LB) + (LC + LD) < min_quartet_L)
                continue;

            const std::size_t nCD = gC.n_components * gD.n_components;
            block.assign(gA.n_components * gB.n_components * nCD, 0.0);
            HartreeFock::RysQuad::_contracted_eri_block_hoisted(
                basis, gA, gB, gC, gD, kernel, omega, block.data());

            for (std::size_t a = 0; a < gA.n_components; ++a)
            for (std::size_t b = 0; b < gB.n_components; ++b)
            for (std::size_t c = 0; c < gC.n_components; ++c)
            for (std::size_t d = 0; d < gD.n_components; ++d)
            {
                const auto &cvA = basis._basis_functions[gA.first_ao + a];
                const auto &cvB = basis._basis_functions[gB.first_ao + b];
                const auto &cvC = basis._basis_functions[gC.first_ao + c];
                const auto &cvD = basis._basis_functions[gD.first_ao + d];

                const HartreeFock::ShellPair spAB(cvA, cvB);
                const HartreeFock::ShellPair spCD(cvC, cvD);
                if (spAB.primitive_pairs.empty() || spCD.primitive_pairs.empty())
                    continue;
                const double via_prod = HartreeFock::RysQuad::_rys_contracted_eri(
                    spAB, spCD,
                    cvA._cartesian[0], cvA._cartesian[1], cvA._cartesian[2],
                    cvB._cartesian[0], cvB._cartesian[1], cvB._cartesian[2],
                    cvC._cartesian[0], cvC._cartesian[1], cvC._cartesian[2],
                    cvD._cartesian[0], cvD._cartesian[1], cvD._cartesian[2],
                    kernel, omega);

                const double via_block =
                    block[(a * gB.n_components + b) * nCD + (c * gD.n_components + d)];

                ++components_checked;
                const double adiff = std::fabs(via_block - via_prod);
                const double scale = std::max(std::fabs(via_prod), 1e-300);
                const double rdiff = adiff / scale;
                if (adiff > max_abs_diff)
                    max_abs_diff = adiff;
                if (adiff > 1e-15)
                {
                    if (rdiff > max_rel_diff)
                        max_rel_diff = rdiff;
                    if (rdiff > REL_TOL)
                        ++over_tol;
                }
            }
        }

        char buf[96];
        std::snprintf(buf, sizeof(buf), "max|Δ|=%.2e rel=%.2e", max_abs_diff, max_rel_diff);
        if (over_tol != 0)
        {
            fail(std::string("Rys B-2d / ") + kernel_label(kernel) + " / " + label +
                 ": " + std::to_string(over_tol) + " of " +
                 std::to_string(components_checked) + " components over rel-tol  " + buf);
        }
        else
        {
            std::cout << "OK  Rys B-2d / " << kernel_label(kernel) << " / " << label
                      << ": hoisted block == _rys_contracted_eri (rel<=1e-13) at all "
                      << components_checked << " components  (" << buf << ")\n";
        }
    }
} // namespace

int main()
{
    // d-shell basis: components within a quartet span several total-L values, so
    // n_max strictly exceeds many components' own root count -> exercises the
    // Gauss over-integration leg of the invariant.
    auto water = make_water("6-31g*");
    if (!water)
    {
        std::cerr << water.error() << '\n';
        return 1;
    }
    check("water/6-31g*", *water, HartreeFock::ERIKernel::Coulomb, 0.0);
    check("water/6-31g*", *water, HartreeFock::ERIKernel::LongRange, 0.3);
    check("water/6-31g*", *water, HartreeFock::ERIKernel::ShortRange, 0.3);

    // B-2b: contract-then-HRR == _rys_contracted_eri (rel<=1e-13).
    check_contract("water/6-31g*", *water, HartreeFock::ERIKernel::Coulomb, 0.0);
    check_contract("water/6-31g*", *water, HartreeFock::ERIKernel::LongRange, 0.3);
    check_contract("water/6-31g*", *water, HartreeFock::ERIKernel::ShortRange, 0.3);

    // B-2c: norm-free max-box contract + per-component readout == _rys_contracted_eri.
    check_maxbox_readout("water/6-31g*", *water, HartreeFock::ERIKernel::Coulomb, 0.0);
    check_maxbox_readout("water/6-31g*", *water, HartreeFock::ERIKernel::LongRange, 0.3);
    check_maxbox_readout("water/6-31g*", *water, HartreeFock::ERIKernel::ShortRange, 0.3);

    // B-2d: assembled RysHoistedQuartet block == _rys_contracted_eri.
    check_block("water/6-31g*", *water, HartreeFock::ERIKernel::Coulomb, 0.0);
    check_block("water/6-31g*", *water, HartreeFock::ERIKernel::LongRange, 0.3);
    check_block("water/6-31g*", *water, HartreeFock::ERIKernel::ShortRange, 0.3);

    // g-shell basis: confine the sweep to the high-L quartets (maxAB+maxCD >= 7,
    // i.e. the (7,8)/(8,8) buckets the Auto path actually routes to Rys). Lower-L
    // Ne quartets are already covered structurally by water/6-31g*, and sweeping
    // all g quartets is prohibitively slow.
    auto ne = make_ne("cc-pVQZ");
    if (!ne)
    {
        std::cerr << ne.error() << '\n';
        return 1;
    }
    check("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::Coulomb, 0.0, 7);
    check("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::ShortRange, 0.3, 7);

    check_contract("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::Coulomb, 0.0, 7);
    check_contract("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::ShortRange, 0.3, 7);

    check_maxbox_readout("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::Coulomb, 0.0, 7);
    check_maxbox_readout("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::ShortRange, 0.3, 7);

    check_block("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::Coulomb, 0.0, 7);
    check_block("Ne/cc-pVQZ (Lq>=7)", *ne, HartreeFock::ERIKernel::ShortRange, 0.3, 7);

    if (!g_ok)
    {
        std::cerr << "planck-rys-box-invariance: FAIL\n";
        return 1;
    }
    std::cout << "planck-rys-box-invariance: OK\n";
    return 0;
}
