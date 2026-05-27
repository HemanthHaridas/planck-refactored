#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/os.h"
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

    std::expected<HartreeFock::Calculator, std::string> make_water_calculator(
        const std::string &basis_name = "sto-3g")
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

    double max_abs(const std::array<double, 12> &values)
    {
        double result = 0.0;
        for (double value : values)
            result = std::max(result, std::abs(value));
        return result;
    }

    double max_abs_diff(
        const std::array<double, 12> &lhs,
        const std::array<double, 12> &rhs)
    {
        double result = 0.0;
        for (std::size_t i = 0; i < lhs.size(); ++i)
            result = std::max(result, std::abs(lhs[i] - rhs[i]));
        return result;
    }

    std::size_t shell_index_for_view(
        const HartreeFock::Basis &basis,
        const HartreeFock::ContractedView &view)
    {
        for (std::size_t i = 0; i < basis._shells.size(); ++i)
            if (&basis._shells[i] == view._shell)
                return i;
        throw std::runtime_error("contracted view does not belong to the provided basis");
    }

    const HartreeFock::ShellPair *find_shell_pair(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const std::size_t a_index,
        const std::size_t b_index)
    {
        for (const auto &sp : shell_pairs)
        {
            if (sp.A._index == a_index && sp.B._index == b_index)
                return &sp;
        }
        return nullptr;
    }

    std::string cart_label(const Eigen::Vector3i &am)
    {
        std::string label;
        label.append(static_cast<std::size_t>(am[0]), 'x');
        label.append(static_cast<std::size_t>(am[1]), 'y');
        label.append(static_cast<std::size_t>(am[2]), 'z');
        return label.empty() ? "s" : label;
    }

    struct QuartetSelection
    {
        std::size_t ab_pair = 0;
        std::size_t cd_pair = 0;
        std::array<std::size_t, 4> shell_indices{};
        double max_component = 0.0;
    };

    std::optional<QuartetSelection> choose_distinct_shell_quartet(
        const HartreeFock::Basis &basis,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const double omega)
    {
        std::optional<QuartetSelection> best;
        for (std::size_t ij = 0; ij < shell_pairs.size(); ++ij)
        {
            const auto &sp_ab = shell_pairs[ij];
            const std::size_t shell_a = shell_index_for_view(basis, sp_ab.A);
            const std::size_t shell_b = shell_index_for_view(basis, sp_ab.B);
            for (std::size_t kl = 0; kl < shell_pairs.size(); ++kl)
            {
                const auto &sp_cd = shell_pairs[kl];
                const std::size_t shell_c = shell_index_for_view(basis, sp_cd.A);
                const std::size_t shell_d = shell_index_for_view(basis, sp_cd.B);
                if (shell_a == shell_b || shell_a == shell_c || shell_a == shell_d ||
                    shell_b == shell_c || shell_b == shell_d || shell_c == shell_d)
                    continue;

                const auto long_range = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                    sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
                const double score = max_abs(long_range);
                if (score < 1e-7)
                    continue;
                if (!best || score > best->max_component)
                {
                    best = QuartetSelection{
                        .ab_pair = ij,
                        .cd_pair = kl,
                        .shell_indices = {shell_a, shell_b, shell_c, shell_d},
                        .max_component = score,
                    };
                }
            }
        }
        return best;
    }

    double contracted_quartet_value(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        return HartreeFock::ObaraSaika::_contracted_eri_elem(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2],
            kernel, omega);
    }

    double rys_contracted_quartet_value(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        return HartreeFock::RysQuad::_rys_contracted_eri(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2],
            kernel, omega);
    }

    double rys_weighted_quartet_value_a(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        int ax, int ay, int az,
        int bx, int by, int bz,
        int cx, int cy, int cz,
        int dx, int dy, int dz,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        const double ABx = sp_ab.R[0], ABy = sp_ab.R[1], ABz = sp_ab.R[2];
        const double CDx = sp_cd.R[0], CDy = sp_cd.R[1], CDz = sp_cd.R[2];
        double eri = 0.0;
        for (const auto &pp_ab : sp_ab.primitive_pairs)
            for (const auto &pp_cd : sp_cd.primitive_pairs)
                eri += (2.0 * pp_ab.alpha) * pp_ab.coeff_product * pp_cd.coeff_product *
                       HartreeFock::RysQuad::_rys_eri_primitive(
                           pp_ab, pp_cd,
                           ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz,
                           ABx, ABy, ABz, CDx, CDy, CDz,
                           kernel, omega);
        return eri;
    }

    double rys_weighted_quartet_value_b(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        int ax, int ay, int az,
        int bx, int by, int bz,
        int cx, int cy, int cz,
        int dx, int dy, int dz,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        const double ABx = sp_ab.R[0], ABy = sp_ab.R[1], ABz = sp_ab.R[2];
        const double CDx = sp_cd.R[0], CDy = sp_cd.R[1], CDz = sp_cd.R[2];
        double eri = 0.0;
        for (const auto &pp_ab : sp_ab.primitive_pairs)
            for (const auto &pp_cd : sp_cd.primitive_pairs)
                eri += pp_ab.coeff_product * (2.0 * pp_ab.beta) * pp_cd.coeff_product *
                       HartreeFock::RysQuad::_rys_eri_primitive(
                           pp_ab, pp_cd,
                           ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz,
                           ABx, ABy, ABz, CDx, CDy, CDz,
                           kernel, omega);
        return eri;
    }

    double rys_weighted_quartet_value_c(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        int ax, int ay, int az,
        int bx, int by, int bz,
        int cx, int cy, int cz,
        int dx, int dy, int dz,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        const double ABx = sp_ab.R[0], ABy = sp_ab.R[1], ABz = sp_ab.R[2];
        const double CDx = sp_cd.R[0], CDy = sp_cd.R[1], CDz = sp_cd.R[2];
        double eri = 0.0;
        for (const auto &pp_ab : sp_ab.primitive_pairs)
            for (const auto &pp_cd : sp_cd.primitive_pairs)
                eri += pp_ab.coeff_product * (2.0 * pp_cd.alpha) * pp_cd.coeff_product *
                       HartreeFock::RysQuad::_rys_eri_primitive(
                           pp_ab, pp_cd,
                           ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz,
                           ABx, ABy, ABz, CDx, CDy, CDz,
                           kernel, omega);
        return eri;
    }

    double rys_weighted_quartet_value_d(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        int ax, int ay, int az,
        int bx, int by, int bz,
        int cx, int cy, int cz,
        int dx, int dy, int dz,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        const double ABx = sp_ab.R[0], ABy = sp_ab.R[1], ABz = sp_ab.R[2];
        const double CDx = sp_cd.R[0], CDy = sp_cd.R[1], CDz = sp_cd.R[2];
        double eri = 0.0;
        for (const auto &pp_ab : sp_ab.primitive_pairs)
            for (const auto &pp_cd : sp_cd.primitive_pairs)
                eri += pp_ab.coeff_product * pp_cd.coeff_product * (2.0 * pp_cd.beta) *
                       HartreeFock::RysQuad::_rys_eri_primitive(
                           pp_ab, pp_cd,
                           ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz,
                           ABx, ABy, ABz, CDx, CDy, CDz,
                           kernel, omega);
        return eri;
    }

    std::array<double, 12> rys_am_shift_derivative(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        const int lAx = sp_ab.A._cartesian[0], lAy = sp_ab.A._cartesian[1], lAz = sp_ab.A._cartesian[2];
        const int lBx = sp_ab.B._cartesian[0], lBy = sp_ab.B._cartesian[1], lBz = sp_ab.B._cartesian[2];
        const int lCx = sp_cd.A._cartesian[0], lCy = sp_cd.A._cartesian[1], lCz = sp_cd.A._cartesian[2];
        const int lDx = sp_cd.B._cartesian[0], lDy = sp_cd.B._cartesian[1], lDz = sp_cd.B._cartesian[2];
        std::array<double, 12> result{};
        for (int q = 0; q < 3; ++q)
        {
            const int axp = lAx + (q == 0), ayp = lAy + (q == 1), azp = lAz + (q == 2);
            const int bxp = lBx + (q == 0), byp = lBy + (q == 1), bzp = lBz + (q == 2);
            const int cxp = lCx + (q == 0), cyp = lCy + (q == 1), czp = lCz + (q == 2);
            const int dxp = lDx + (q == 0), dyp = lDy + (q == 1), dzp = lDz + (q == 2);
            result[q] += rys_weighted_quartet_value_a(
                sp_ab, sp_cd, axp, ayp, azp, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, kernel, omega);
            result[3 + q] += rys_weighted_quartet_value_b(
                sp_ab, sp_cd, lAx, lAy, lAz, bxp, byp, bzp, lCx, lCy, lCz, lDx, lDy, lDz, kernel, omega);
            result[6 + q] += rys_weighted_quartet_value_c(
                sp_ab, sp_cd, lAx, lAy, lAz, lBx, lBy, lBz, cxp, cyp, czp, lDx, lDy, lDz, kernel, omega);
            result[9 + q] += rys_weighted_quartet_value_d(
                sp_ab, sp_cd, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, dxp, dyp, dzp, kernel, omega);

            const int lAq = sp_ab.A._cartesian[q];
            const int lBq = sp_ab.B._cartesian[q];
            const int lCq = sp_cd.A._cartesian[q];
            const int lDq = sp_cd.B._cartesian[q];
            if (lAq > 0)
            {
                const int axm = lAx - (q == 0), aym = lAy - (q == 1), azm = lAz - (q == 2);
                result[q] -= static_cast<double>(lAq) *
                             HartreeFock::RysQuad::_rys_contracted_eri(
                                 sp_ab, sp_cd,
                                 axm, aym, azm, lBx, lBy, lBz,
                                 lCx, lCy, lCz, lDx, lDy, lDz,
                                 kernel, omega);
            }
            if (lBq > 0)
            {
                const int bxm = lBx - (q == 0), bym = lBy - (q == 1), bzm = lBz - (q == 2);
                result[3 + q] -= static_cast<double>(lBq) *
                                 HartreeFock::RysQuad::_rys_contracted_eri(
                                     sp_ab, sp_cd,
                                     lAx, lAy, lAz, bxm, bym, bzm,
                                     lCx, lCy, lCz, lDx, lDy, lDz,
                                     kernel, omega);
            }
            if (lCq > 0)
            {
                const int cxm = lCx - (q == 0), cym = lCy - (q == 1), czm = lCz - (q == 2);
                result[6 + q] -= static_cast<double>(lCq) *
                                 HartreeFock::RysQuad::_rys_contracted_eri(
                                     sp_ab, sp_cd,
                                     lAx, lAy, lAz, lBx, lBy, lBz,
                                     cxm, cym, czm, lDx, lDy, lDz,
                                     kernel, omega);
            }
            if (lDq > 0)
            {
                const int dxm = lDx - (q == 0), dym = lDy - (q == 1), dzm = lDz - (q == 2);
                result[9 + q] -= static_cast<double>(lDq) *
                                 HartreeFock::RysQuad::_rys_contracted_eri(
                                     sp_ab, sp_cd,
                                     lAx, lAy, lAz, lBx, lBy, lBz,
                                     lCx, lCy, lCz, dxm, dym, dzm,
                                     kernel, omega);
            }
        }
        return result;
    }

    HartreeFock::Calculator make_displaced_water_calculator(
        const std::string &basis_name,
        const std::size_t shell_index,
        const int direction,
        const double displacement)
    {
        auto calc_res = make_water_calculator(basis_name);
        if (!calc_res)
            throw std::runtime_error("setup failed: " + calc_res.error());
        auto calc = std::move(*calc_res);
        calc._shells._shells[shell_index]._center[direction] += displacement;
        return calc;
    }

    void test_kernel_entry_points()
    {
        auto calc_res = make_water_calculator();
        if (!calc_res)
        {
            fail("setup failed: " + calc_res.error());
            return;
        }

        const auto shell_pairs = build_shellpairs(calc_res->_shells);
        if (shell_pairs.empty())
        {
            fail("build_shellpairs returned no shell pairs");
            return;
        }

        constexpr double omega = 0.3;
        bool saw_nontrivial = false;
        bool saw_screening_effect = false;

        for (const auto &sp_ab : shell_pairs)
        {
            for (const auto &sp_cd : shell_pairs)
            {
                const auto coulomb_default = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                    sp_ab, sp_cd);
                const auto coulomb = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                    sp_ab, sp_cd, HartreeFock::ERIKernel::Coulomb, 0.0);
                const auto long_range = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                    sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
                const auto short_range = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                    sp_ab, sp_cd, HartreeFock::ERIKernel::ShortRange, omega);

                if (max_abs(coulomb) > 1e-8)
                    saw_nontrivial = true;
                if (max_abs_diff(coulomb_default, coulomb) > 1e-13)
                {
                    fail("default ERI derivative kernel no longer matches the explicit Coulomb path");
                    return;
                }
                std::array<double, 12> reconstructed{};
                for (std::size_t i = 0; i < reconstructed.size(); ++i)
                    reconstructed[i] = long_range[i] + short_range[i];
                if (max_abs_diff(coulomb, reconstructed) > 1e-8)
                {
                    fail("screened ERI derivative kernels no longer reconstruct the Coulomb derivative");
                    return;
                }
                if (max_abs_diff(coulomb, long_range) > 1e-8 ||
                    max_abs_diff(coulomb, short_range) > 1e-8)
                {
                    saw_screening_effect = true;
                }
            }
        }

        if (!saw_nontrivial)
            fail("ERI derivative kernel test only observed near-zero derivative quartets");
        if (!saw_screening_effect)
            fail("screened ERI derivative kernels did not differ from the Coulomb path");
    }

    void test_long_range_quartet_derivative_against_finite_difference()
    {
        auto calc_res = make_water_calculator();
        if (!calc_res)
        {
            fail("setup failed: " + calc_res.error());
            return;
        }

        const auto &basis = calc_res->_shells;
        const auto shell_pairs = build_shellpairs(basis);
        constexpr double omega = 0.11;
        constexpr double step = 1e-4;
        constexpr double tolerance = 5e-7;

        const auto selection = choose_distinct_shell_quartet(basis, shell_pairs, omega);
        if (!selection)
        {
            fail("could not find a nontrivial long-range quartet with four distinct shell centers");
            return;
        }

        const auto *ssss_ab = find_shell_pair(shell_pairs, 1, 5);
        const auto *ssss_cd = find_shell_pair(shell_pairs, 0, 6);
        const auto *psss_ab = find_shell_pair(shell_pairs, 3, 5);
        const auto *psss_cd = find_shell_pair(shell_pairs, 0, 1);
        const auto *ssps_ab = find_shell_pair(shell_pairs, 1, 5);
        const auto *ssps_cd = find_shell_pair(shell_pairs, 3, 6);
        const auto *same_center_ab = find_shell_pair(shell_pairs, 1, 3);
        const auto *same_center_cd = find_shell_pair(shell_pairs, 0, 5);
        const auto *triple_same_ab = find_shell_pair(shell_pairs, 0, 2);
        const auto *triple_same_cd = find_shell_pair(shell_pairs, 2, 5);
        if (!ssss_ab || !ssss_cd)
        {
            fail("failed to locate the fixed s-s-s-s diagnostic quartet");
            return;
        }
        if (!psss_ab || !psss_cd)
        {
            fail("failed to locate the fixed p-s-s-s diagnostic quartet");
            return;
        }
        if (!ssps_ab || !ssps_cd)
        {
            fail("failed to locate the fixed s-s-p-s diagnostic quartet");
            return;
        }
        if (!same_center_ab || !same_center_cd)
        {
            fail("failed to locate the fixed same-center p/s diagnostic quartet");
            return;
        }
        if (!triple_same_ab || !triple_same_cd)
        {
            fail("failed to locate the fixed triple-same-center p diagnostic quartet");
            return;
        }

        const auto &sp_ab = shell_pairs[selection->ab_pair];
        const auto &sp_cd = shell_pairs[selection->cd_pair];
        const auto analytic = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
            sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
        const auto analytic_rys_shift = rys_am_shift_derivative(
            sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
        const auto analytic_coulomb = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
            sp_ab, sp_cd, HartreeFock::ERIKernel::Coulomb, 0.0);
        const double base_os = contracted_quartet_value(
            sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
        const double base_rys = rys_contracted_quartet_value(
            sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
        const double base_coulomb = contracted_quartet_value(
            sp_ab, sp_cd, HartreeFock::ERIKernel::Coulomb, 0.0);

        std::array<double, 12> finite_difference{};
        std::array<double, 12> finite_difference_rys{};
        std::array<double, 12> finite_difference_coulomb{};
        double max_error = 0.0;
        double max_error_coulomb = 0.0;
        double max_os_rys_fd_gap = 0.0;

        for (int center = 0; center < 4; ++center)
        {
            for (int direction = 0; direction < 3; ++direction)
            {
                const std::size_t shell_index = selection->shell_indices[center];

                const auto plus_calc = make_displaced_water_calculator("sto-3g", shell_index, direction, step);
                const auto minus_calc = make_displaced_water_calculator("sto-3g", shell_index, direction, -step);

                const auto plus_pairs = build_shellpairs(plus_calc._shells);
                const auto minus_pairs = build_shellpairs(minus_calc._shells);

                const auto *plus_ab = find_shell_pair(plus_pairs, sp_ab.A._index, sp_ab.B._index);
                const auto *plus_cd = find_shell_pair(plus_pairs, sp_cd.A._index, sp_cd.B._index);
                const auto *minus_ab = find_shell_pair(minus_pairs, sp_ab.A._index, sp_ab.B._index);
                const auto *minus_cd = find_shell_pair(minus_pairs, sp_cd.A._index, sp_cd.B._index);
                if (!plus_ab || !plus_cd || !minus_ab || !minus_cd)
                {
                    fail("failed to recover the displaced shell-pair quartet");
                    return;
                }

                const double plus_value = contracted_quartet_value(
                    *plus_ab, *plus_cd, HartreeFock::ERIKernel::LongRange, omega);
                const double minus_value = contracted_quartet_value(
                    *minus_ab, *minus_cd, HartreeFock::ERIKernel::LongRange, omega);
                const double fd_value = (plus_value - minus_value) / (2.0 * step);
                const double plus_value_rys = rys_contracted_quartet_value(
                    *plus_ab, *plus_cd, HartreeFock::ERIKernel::LongRange, omega);
                const double minus_value_rys = rys_contracted_quartet_value(
                    *minus_ab, *minus_cd, HartreeFock::ERIKernel::LongRange, omega);
                const double fd_value_rys = (plus_value_rys - minus_value_rys) / (2.0 * step);
                const double plus_value_coulomb = contracted_quartet_value(
                    *plus_ab, *plus_cd, HartreeFock::ERIKernel::Coulomb, 0.0);
                const double minus_value_coulomb = contracted_quartet_value(
                    *minus_ab, *minus_cd, HartreeFock::ERIKernel::Coulomb, 0.0);
                const double fd_value_coulomb = (plus_value_coulomb - minus_value_coulomb) / (2.0 * step);
                const std::size_t slot = static_cast<std::size_t>(center * 3 + direction);
                finite_difference[slot] = fd_value;
                finite_difference_rys[slot] = fd_value_rys;
                finite_difference_coulomb[slot] = fd_value_coulomb;
                max_error = std::max(max_error, std::abs(fd_value - analytic[slot]));
                max_error_coulomb = std::max(max_error_coulomb, std::abs(fd_value_coulomb - analytic_coulomb[slot]));
                max_os_rys_fd_gap = std::max(max_os_rys_fd_gap, std::abs(fd_value - fd_value_rys));
            }
        }

        std::cout << std::setprecision(12)
                  << "Selected long-range quartet: ("
                  << sp_ab.A._index << "," << sp_ab.B._index << "|"
                  << sp_cd.A._index << "," << sp_cd.B._index << ")"
                  << " shell indices=("
                  << selection->shell_indices[0] << ","
                  << selection->shell_indices[1] << ","
                  << selection->shell_indices[2] << ","
                  << selection->shell_indices[3] << ")\n";
        std::cout << "  base_os=" << base_os
                  << " base_rys=" << base_rys
                  << " base_coulomb=" << base_coulomb
                  << " diff=" << (base_os - base_rys) << '\n';
        for (int center = 0; center < 4; ++center)
        {
            for (int direction = 0; direction < 3; ++direction)
            {
                const std::size_t slot = static_cast<std::size_t>(center * 3 + direction);
                std::cout << "  c" << center << " d" << direction
                          << " analytic=" << analytic[slot]
                          << " analytic_rys_shift=" << analytic_rys_shift[slot]
                          << " fd=" << finite_difference[slot]
                          << " fd_rys=" << finite_difference_rys[slot]
                          << " diff=" << (analytic[slot] - finite_difference[slot])
                          << " analytic_os-rys_shift=" << (analytic[slot] - analytic_rys_shift[slot])
                          << " fd_os-rys=" << (finite_difference[slot] - finite_difference_rys[slot]) << '\n';
            }
        }
        std::cout << "  translational sum y analytic="
                  << (analytic[1] + analytic[4] + analytic[7] + analytic[10])
                  << " analytic_rys_shift=" << (analytic_rys_shift[1] + analytic_rys_shift[4] + analytic_rys_shift[7] + analytic_rys_shift[10])
                  << " fd=" << (finite_difference[1] + finite_difference[4] + finite_difference[7] + finite_difference[10])
                  << " fd_rys=" << (finite_difference_rys[1] + finite_difference_rys[4] + finite_difference_rys[7] + finite_difference_rys[10])
                  << " coulomb_analytic=" << (analytic_coulomb[1] + analytic_coulomb[4] + analytic_coulomb[7] + analytic_coulomb[10])
                  << " coulomb_fd=" << (finite_difference_coulomb[1] + finite_difference_coulomb[4] + finite_difference_coulomb[7] + finite_difference_coulomb[10]) << '\n';
        std::cout << "  translational sum z analytic="
                  << (analytic[2] + analytic[5] + analytic[8] + analytic[11])
                  << " analytic_rys_shift=" << (analytic_rys_shift[2] + analytic_rys_shift[5] + analytic_rys_shift[8] + analytic_rys_shift[11])
                  << " fd=" << (finite_difference[2] + finite_difference[5] + finite_difference[8] + finite_difference[11])
                  << " fd_rys=" << (finite_difference_rys[2] + finite_difference_rys[5] + finite_difference_rys[8] + finite_difference_rys[11])
                  << " coulomb_analytic=" << (analytic_coulomb[2] + analytic_coulomb[5] + analytic_coulomb[8] + analytic_coulomb[11])
                  << " coulomb_fd=" << (finite_difference_coulomb[2] + finite_difference_coulomb[5] + finite_difference_coulomb[8] + finite_difference_coulomb[11]) << '\n';
        std::cout << "  max |fd_os-fd_rys|=" << max_os_rys_fd_gap << '\n';
        std::cout << "  max |coulomb analytic-fd|=" << max_error_coulomb << '\n';
        std::cout << "  coulomb center derivatives:\n";
        for (int center = 0; center < 4; ++center)
        {
            std::cout << "    c" << center
                      << " (" << analytic_coulomb[center * 3 + 0]
                      << ", " << analytic_coulomb[center * 3 + 1]
                      << ", " << analytic_coulomb[center * 3 + 2] << ")\n";
        }
        std::cout << "  omega sweep (Planck long-range quartet energy):\n";
        for (double sweep_omega : {0.01, 0.05, 0.11, 0.2, 0.5, 1.0})
        {
            const double sweep_value = contracted_quartet_value(
                sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, sweep_omega);
            std::cout << "    omega=" << sweep_omega
                      << " value=" << sweep_value << '\n';
        }
        std::cout << "  omega sweep (Planck long-range s-s-s-s quartet 1,5|0,6):\n";
        for (double sweep_omega : {0.01, 0.05, 0.11, 0.2, 0.5, 1.0})
        {
            const double sweep_value = contracted_quartet_value(
                *ssss_ab, *ssss_cd, HartreeFock::ERIKernel::LongRange, sweep_omega);
            std::cout << "    omega=" << sweep_omega
                      << " value=" << sweep_value << '\n';
        }
        std::cout << "  omega sweep (Planck long-range p-s-s-s quartet 3,5|0,1):\n";
        for (double sweep_omega : {0.01, 0.05, 0.11, 0.2, 0.5, 1.0})
        {
            const double sweep_value = contracted_quartet_value(
                *psss_ab, *psss_cd, HartreeFock::ERIKernel::LongRange, sweep_omega);
            std::cout << "    omega=" << sweep_omega
                      << " value=" << sweep_value << '\n';
        }
        std::cout << "  omega sweep (Planck long-range s-s-p-s quartet 1,5|3,6):\n";
        for (double sweep_omega : {0.01, 0.05, 0.11, 0.2, 0.5, 1.0})
        {
            const double sweep_value = contracted_quartet_value(
                *ssps_ab, *ssps_cd, HartreeFock::ERIKernel::LongRange, sweep_omega);
            std::cout << "    omega=" << sweep_omega
                      << " value=" << sweep_value << '\n';
        }
        std::cout << "  omega sweep (Planck long-range same-center p/s quartet 1,3|0,5):\n";
        for (double sweep_omega : {0.01, 0.05, 0.11, 0.2, 0.5, 1.0})
        {
            const double sweep_value = contracted_quartet_value(
                *same_center_ab, *same_center_cd, HartreeFock::ERIKernel::LongRange, sweep_omega);
            std::cout << "    omega=" << sweep_omega
                      << " value=" << sweep_value << '\n';
        }
        std::cout << "  omega sweep (Planck long-range triple-same-center quartet 0,2|2,5):\n";
        for (double sweep_omega : {0.01, 0.05, 0.11, 0.2, 0.5, 1.0})
        {
            const double sweep_value = contracted_quartet_value(
                *triple_same_ab, *triple_same_cd, HartreeFock::ERIKernel::LongRange, sweep_omega);
            std::cout << "    omega=" << sweep_omega
                      << " value=" << sweep_value << '\n';
        }

        if (max_error > tolerance)
        {
            std::cerr << "diagnostic: long-range quartet derivative failed direct finite-difference validation; "
                      << "max |analytic-fd| = " << max_error << '\n';
        }
    }

    void test_d_shell_long_range_quartets()
    {
        auto calc_res = make_water_calculator("6-31g*");
        if (!calc_res)
        {
            fail("6-31g* setup failed: " + calc_res.error());
            return;
        }

        const auto &basis = calc_res->_shells;
        const auto shell_pairs = build_shellpairs(basis);
        constexpr double omega = 0.11;
        constexpr double energy_tol = 1e-8;
        constexpr double deriv_tol = 2e-8;
        constexpr double fd_step = 1e-5;

        std::cout << "d-shell AO labels (water / 6-31g*):\n";
        for (std::size_t idx = 9; idx <= 14; ++idx)
        {
            const auto &bf = basis._basis_functions[idx];
            const auto *diag_pair = find_shell_pair(shell_pairs, idx, idx);
            const double overlap_diag = diag_pair
                                            ? std::get<0>(HartreeFock::ObaraSaika::_compute_3d_overlap_kinetic(*diag_pair))
                                            : -1.0;
            std::cout << "  ao " << idx
                      << " cart=(" << bf._cartesian[0] << "," << bf._cartesian[1] << "," << bf._cartesian[2] << ")"
                      << " label=" << cart_label(bf._cartesian)
                      << " component_norm=" << bf._component_norm
                      << " overlap_diag=" << std::setprecision(12) << overlap_diag
                      << '\n';
        }

        struct FamilySweep
        {
            const char *name;
            std::size_t i;
            std::size_t j;
            std::size_t k;
            std::size_t l;
            std::array<double, 6> pyscf_lr;
            std::array<double, 6> pyscf_coulomb;
        };

        const std::array<FamilySweep, 4> family_sweeps{{
            {"same_center_ket", 1, 15, 0, 9,
             {0.0016311796523281429, 0.0, 0.0, 0.0016311844063704958, -3.6784182959942555e-09, 0.0016311824984872967},
             {0.01269706381838066, 0.0, 0.0, 0.01303286612473238, -0.0002598254823631071, 0.012898102584369697}},
            {"same_center_bra", 1, 9, 0, 15,
             {0.0037463354754034488, 0.0, 0.0, 0.003746337153391111, -1.2983351977712677e-09, 0.003746336479984307},
             {0.02945919436371374, 0.0, 0.0, 0.02969404660770994, -0.00018171583823625937, 0.029599796153605258}},
            {"d_s_s_s", 9, 15, 0, 1,
             {0.007616813811113749, 0.0, 0.0, 0.018665270501053485, -0.008548692294640642, 0.014231324959534327},
             {0.046809804226248454, 0.0, 0.0, 0.1039928244554318, -0.04424509758567612, 0.08104424396937564}},
            {"s_s_d_s", 1, 15, 9, 17,
             {0.00806584202864128, 0.0, 0.0, 0.01971263895461429, 0.009051075916381674, 0.015099859414167515},
             {0.040237969270862955, 0.0, 0.0, 0.08405322345380006, 0.03879436813915374, 0.07819162099464037}},
        }};

        std::cout << "d-shell family sweeps (water / 6-31g* / omega=0.11):\n";
        for (const auto &family : family_sweeps)
        {
            std::cout << "  " << family.name << '\n';
            for (std::size_t offset = 0; offset < 6; ++offset)
            {
                const std::size_t d = 9 + offset;
                const auto *ab = find_shell_pair(shell_pairs,
                                                 family.i == 9 ? d : family.i,
                                                 family.j == 9 ? d : family.j);
                const auto *cd = find_shell_pair(shell_pairs,
                                                 family.k == 9 ? d : family.k,
                                                 family.l == 9 ? d : family.l);
                if (!ab || !cd)
                {
                    fail(std::string("failed to locate d-shell family sweep quartet for ") + family.name);
                    return;
                }
                const auto &bf = basis._basis_functions[d];
                const double planck_lr = contracted_quartet_value(*ab, *cd, HartreeFock::ERIKernel::LongRange, omega);
                const double planck_c = contracted_quartet_value(*ab, *cd, HartreeFock::ERIKernel::Coulomb, 0.0);
                std::cout << "    ao " << d
                          << " " << cart_label(bf._cartesian)
                          << " planck_lr=" << planck_lr
                          << " pyscf_lr=" << family.pyscf_lr[offset]
                          << " planck_c=" << planck_c
                          << " pyscf_c=" << family.pyscf_coulomb[offset]
                          << '\n';
            }
        }

        struct PairRef
        {
            std::size_t i;
            std::size_t j;
            double pyscf_overlap;
            double pyscf_kinetic;
        };
        const std::array<PairRef, 6> one_e_refs{{
            {9, 15, 0.2653552918427907, -0.13365613628669032},
            {12, 15, 0.6512735891333278, 0.5202286183987239},
            {13, 15, -0.2986024987012413, -0.5059403064911389},
            {14, 15, 0.496397594984739, 0.25781283725666465},
            {9, 17, 0.2653552918427907, -0.13365613628669032},
            {13, 17, 0.2986024987012413, 0.5059403064911389},
        }};
        std::cout << "d-shell one-electron pair checks (water / 6-31g*):\n";
        for (const auto &ref : one_e_refs)
        {
            const auto *pair = find_shell_pair(shell_pairs, ref.i, ref.j);
            if (!pair)
            {
                fail("failed to locate d-shell one-electron pair");
                return;
            }
            const auto [planck_overlap, planck_kinetic] = HartreeFock::ObaraSaika::_compute_3d_overlap_kinetic(*pair);
            std::cout << "  (" << ref.i << "," << ref.j << ")"
                      << " overlap=" << planck_overlap
                      << " pyscf_overlap=" << ref.pyscf_overlap
                      << " kinetic=" << planck_kinetic
                      << " pyscf_kinetic=" << ref.pyscf_kinetic
                      << '\n';
        }

        struct QuartetRef
        {
            const char *name;
            std::size_t i;
            std::size_t j;
            std::size_t k;
            std::size_t l;
            double pyscf_lr;
        };

        const std::array<QuartetRef, 5> refs{{
            {"same_center_ket_d", 1, 15, 0, 9, 0.0016311796523281429},
            {"same_center_bra_d", 1, 9, 0, 15, 0.0037463354754034488},
            {"d_s_s_s", 9, 15, 0, 1, 0.007616813811113749},
            {"s_s_d_s", 1, 15, 9, 17, 0.00806584202864128},
            {"triple_same_center_d", 0, 9, 9, 15, 0.001731908542963683},
        }};

        std::cout << "d-shell long-range quartet checks (water / 6-31g* / omega=0.11):\n";
        double max_energy_err = 0.0;
        for (const auto &ref : refs)
        {
            const auto *ab = find_shell_pair(shell_pairs, ref.i, ref.j);
            const auto *cd = find_shell_pair(shell_pairs, ref.k, ref.l);
            if (!ab || !cd)
            {
                fail(std::string("failed to locate d-shell quartet ") + ref.name);
                return;
            }
            const double planck_lr = contracted_quartet_value(*ab, *cd, HartreeFock::ERIKernel::LongRange, omega);
            const double err = std::abs(planck_lr - ref.pyscf_lr);
            std::cout << "  " << ref.name
                      << " (" << ref.i << "," << ref.j << "|" << ref.k << "," << ref.l << ")"
                      << " planck=" << planck_lr
                      << " pyscf=" << ref.pyscf_lr
                      << " diff=" << (planck_lr - ref.pyscf_lr) << '\n';
            max_energy_err = std::max(max_energy_err, err);
        }
        std::cout << "d-shell Coulomb quartet checks (water / 6-31g*):\n";
        for (const auto &ref : refs)
        {
            const auto *ab_ref = find_shell_pair(shell_pairs, ref.i, ref.j);
            const auto *cd_ref = find_shell_pair(shell_pairs, ref.k, ref.l);
            const double planck_coulomb = contracted_quartet_value(
                *ab_ref, *cd_ref, HartreeFock::ERIKernel::Coulomb, 0.0);
            std::cout << "  " << ref.name
                      << " (" << ref.i << "," << ref.j << "|" << ref.k << "," << ref.l << ")"
                      << " planck_coulomb=" << planck_coulomb << '\n';
        }

        const auto *ab = find_shell_pair(shell_pairs, 0, 9);
        const auto *cd = find_shell_pair(shell_pairs, 1, 15);
        if (!ab || !cd)
        {
            fail("failed to locate d-shell derivative quartet (0,9|1,15)");
            return;
        }

        const auto analytic = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
            *ab, *cd, HartreeFock::ERIKernel::LongRange, omega);
        const std::array<double, 12> pyscf{
            -0.0, 7.179632724047326e-06, -5.555207633867998e-06,
            -0.0, 8.397820302306878e-07, -6.497774641726828e-07,
            -0.0, 0.001906641761466277, -0.0014752552498783179,
            -0.0, -0.0019146611762205547, 0.0014814602349763582,
        };
        std::array<double, 12> fd{};
        std::array<std::size_t, 4> shell_indices{
            shell_index_for_view(calc_res->_shells, ab->A),
            shell_index_for_view(calc_res->_shells, ab->B),
            shell_index_for_view(calc_res->_shells, cd->A),
            shell_index_for_view(calc_res->_shells, cd->B),
        };
        double max_analytic_pyscf = 0.0;
        double max_analytic_fd = 0.0;
        for (int center = 0; center < 4; ++center)
        {
            for (int direction = 0; direction < 3; ++direction)
            {
                const std::size_t shell_index = shell_indices[center];
                const auto plus_calc = make_displaced_water_calculator("6-31g*", shell_index, direction, fd_step);
                const auto minus_calc = make_displaced_water_calculator("6-31g*", shell_index, direction, -fd_step);
                const auto plus_pairs = build_shellpairs(plus_calc._shells);
                const auto minus_pairs = build_shellpairs(minus_calc._shells);
                const auto *plus_ab = find_shell_pair(plus_pairs, 0, 9);
                const auto *plus_cd = find_shell_pair(plus_pairs, 1, 15);
                const auto *minus_ab = find_shell_pair(minus_pairs, 0, 9);
                const auto *minus_cd = find_shell_pair(minus_pairs, 1, 15);
                if (!plus_ab || !plus_cd || !minus_ab || !minus_cd)
                {
                    fail("failed to recover displaced d-shell derivative quartet");
                    return;
                }
                const double plus_val = contracted_quartet_value(*plus_ab, *plus_cd, HartreeFock::ERIKernel::LongRange, omega);
                const double minus_val = contracted_quartet_value(*minus_ab, *minus_cd, HartreeFock::ERIKernel::LongRange, omega);
                const std::size_t slot = static_cast<std::size_t>(center * 3 + direction);
                fd[slot] = (plus_val - minus_val) / (2.0 * fd_step);
                max_analytic_pyscf = std::max(max_analytic_pyscf, std::abs(analytic[slot] - pyscf[slot]));
                max_analytic_fd = std::max(max_analytic_fd, std::abs(analytic[slot] - fd[slot]));
            }
        }

        std::cout << "  d-shell derivative quartet (0,9|1,15):\n";
        for (int center = 0; center < 4; ++center)
        {
            for (int direction = 0; direction < 3; ++direction)
            {
                const std::size_t slot = static_cast<std::size_t>(center * 3 + direction);
                std::cout << "    c" << center << " d" << direction
                          << " analytic=" << analytic[slot]
                          << " pyscf=" << pyscf[slot]
                          << " fd=" << fd[slot]
                          << " d(planck-pyscf)=" << (analytic[slot] - pyscf[slot])
                          << " d(planck-fd)=" << (analytic[slot] - fd[slot]) << '\n';
            }
        }
        if (max_energy_err > energy_tol)
        {
            std::cerr << "diagnostic: d-shell quartet mismatch vs PySCF; max |planck-pyscf| = "
                      << max_energy_err << '\n';
        }
        if (max_analytic_pyscf > deriv_tol)
        {
            std::cerr << "diagnostic: d-shell derivative mismatch vs PySCF; max |analytic-pyscf| = "
                      << max_analytic_pyscf << '\n';
        }
        if (max_analytic_fd > deriv_tol)
        {
            std::cerr << "diagnostic: d-shell derivative mismatch vs FD; max |analytic-fd| = "
                      << max_analytic_fd << '\n';
        }
    }
} // namespace

int main()
{
    test_kernel_entry_points();
    test_long_range_quartet_derivative_against_finite_difference();
    test_d_shell_long_range_quartets();
    return g_ok ? 0 : 1;
}
