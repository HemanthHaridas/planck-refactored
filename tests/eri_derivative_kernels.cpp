#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "base/basis.h"
#include "base/types.h"
#include "integrals/hgp.h"
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

    double hgp_contracted_quartet_value(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        return HartreeFock::HeadGordonPople::_contracted_eri_elem(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2],
            kernel, omega);
    }

    double hgp_contracted_quartet_value_native(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        return HartreeFock::HeadGordonPople::_contracted_eri_elem_native_test(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2],
            kernel, omega);
    }

    void print_screened_derivative_term_breakdown(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd,
        const HartreeFock::ERIKernel kernel,
        const double omega)
    {
        const int lAx = sp_ab.A._cartesian[0], lAy = sp_ab.A._cartesian[1], lAz = sp_ab.A._cartesian[2];
        const int lBx = sp_ab.B._cartesian[0], lBy = sp_ab.B._cartesian[1], lBz = sp_ab.B._cartesian[2];
        const int lCx = sp_cd.A._cartesian[0], lCy = sp_cd.A._cartesian[1], lCz = sp_cd.A._cartesian[2];
        const int lDx = sp_cd.B._cartesian[0], lDy = sp_cd.B._cartesian[1], lDz = sp_cd.B._cartesian[2];

        std::cerr << "constituent breakdown for screened quartet (" << sp_ab.A._index << "," << sp_ab.B._index
                  << "|" << sp_cd.A._index << "," << sp_cd.B._index << ")\n";
        for (int center = 0; center < 4; ++center)
        {
            for (int q = 0; q < 3; ++q)
            {
                const int axp = lAx + (q == 0), ayp = lAy + (q == 1), azp = lAz + (q == 2);
                const int bxp = lBx + (q == 0), byp = lBy + (q == 1), bzp = lBz + (q == 2);
                const int cxp = lCx + (q == 0), cyp = lCy + (q == 1), czp = lCz + (q == 2);
                const int dxp = lDx + (q == 0), dyp = lDy + (q == 1), dzp = lDz + (q == 2);

                double os_weighted = 0.0;
                double hgp_weighted = 0.0;
                double os_lower = 0.0;
                double hgp_lower = 0.0;
                int lq = 0;

                if (center == 0)
                {
                    os_weighted = HartreeFock::ObaraSaika::_contracted_eri_elem_weighted_test(
                        sp_ab, sp_cd, axp, ayp, azp, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, center, kernel, omega);
                    hgp_weighted = HartreeFock::HeadGordonPople::_contracted_eri_elem_weighted_native_test(
                        sp_ab, sp_cd, axp, ayp, azp, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz, center, kernel, omega);
                    lq = sp_ab.A._cartesian[q];
                    if (lq > 0)
                    {
                        os_lower = static_cast<double>(lq) * HartreeFock::ObaraSaika::_contracted_eri_elem(
                            sp_ab, sp_cd,
                            lAx - (q == 0), lAy - (q == 1), lAz - (q == 2), lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz,
                            kernel, omega);
                        hgp_lower = static_cast<double>(lq) * HartreeFock::HeadGordonPople::_contracted_eri_elem_native_test(
                            sp_ab, sp_cd,
                            lAx - (q == 0), lAy - (q == 1), lAz - (q == 2), lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz,
                            kernel, omega);
                    }
                }
                else if (center == 1)
                {
                    os_weighted = HartreeFock::ObaraSaika::_contracted_eri_elem_weighted_test(
                        sp_ab, sp_cd, lAx, lAy, lAz, bxp, byp, bzp, lCx, lCy, lCz, lDx, lDy, lDz, center, kernel, omega);
                    hgp_weighted = HartreeFock::HeadGordonPople::_contracted_eri_elem_weighted_native_test(
                        sp_ab, sp_cd, lAx, lAy, lAz, bxp, byp, bzp, lCx, lCy, lCz, lDx, lDy, lDz, center, kernel, omega);
                    lq = sp_ab.B._cartesian[q];
                    if (lq > 0)
                    {
                        os_lower = static_cast<double>(lq) * HartreeFock::ObaraSaika::_contracted_eri_elem(
                            sp_ab, sp_cd,
                            lAx, lAy, lAz, lBx - (q == 0), lBy - (q == 1), lBz - (q == 2), lCx, lCy, lCz, lDx, lDy, lDz,
                            kernel, omega);
                        hgp_lower = static_cast<double>(lq) * HartreeFock::HeadGordonPople::_contracted_eri_elem_native_test(
                            sp_ab, sp_cd,
                            lAx, lAy, lAz, lBx - (q == 0), lBy - (q == 1), lBz - (q == 2), lCx, lCy, lCz, lDx, lDy, lDz,
                            kernel, omega);
                    }
                }
                else if (center == 2)
                {
                    os_weighted = HartreeFock::ObaraSaika::_contracted_eri_elem_weighted_test(
                        sp_ab, sp_cd, lAx, lAy, lAz, lBx, lBy, lBz, cxp, cyp, czp, lDx, lDy, lDz, center, kernel, omega);
                    hgp_weighted = HartreeFock::HeadGordonPople::_contracted_eri_elem_weighted_native_test(
                        sp_ab, sp_cd, lAx, lAy, lAz, lBx, lBy, lBz, cxp, cyp, czp, lDx, lDy, lDz, center, kernel, omega);
                    lq = sp_cd.A._cartesian[q];
                    if (lq > 0)
                    {
                        os_lower = static_cast<double>(lq) * HartreeFock::ObaraSaika::_contracted_eri_elem(
                            sp_ab, sp_cd,
                            lAx, lAy, lAz, lBx, lBy, lBz, lCx - (q == 0), lCy - (q == 1), lCz - (q == 2), lDx, lDy, lDz,
                            kernel, omega);
                        hgp_lower = static_cast<double>(lq) * HartreeFock::HeadGordonPople::_contracted_eri_elem_native_test(
                            sp_ab, sp_cd,
                            lAx, lAy, lAz, lBx, lBy, lBz, lCx - (q == 0), lCy - (q == 1), lCz - (q == 2), lDx, lDy, lDz,
                            kernel, omega);
                    }
                }
                else
                {
                    os_weighted = HartreeFock::ObaraSaika::_contracted_eri_elem_weighted_test(
                        sp_ab, sp_cd, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, dxp, dyp, dzp, center, kernel, omega);
                    hgp_weighted = HartreeFock::HeadGordonPople::_contracted_eri_elem_weighted_native_test(
                        sp_ab, sp_cd, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, dxp, dyp, dzp, center, kernel, omega);
                    lq = sp_cd.B._cartesian[q];
                    if (lq > 0)
                    {
                        os_lower = static_cast<double>(lq) * HartreeFock::ObaraSaika::_contracted_eri_elem(
                            sp_ab, sp_cd,
                            lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx - (q == 0), lDy - (q == 1), lDz - (q == 2),
                            kernel, omega);
                        hgp_lower = static_cast<double>(lq) * HartreeFock::HeadGordonPople::_contracted_eri_elem_native_test(
                            sp_ab, sp_cd,
                            lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx - (q == 0), lDy - (q == 1), lDz - (q == 2),
                            kernel, omega);
                    }
                }

                std::cerr << "  center=" << center << " dir=" << q
                          << " os_weighted=" << std::setprecision(16) << os_weighted
                          << " hgp_weighted=" << hgp_weighted
                          << " os_lower=" << os_lower
                          << " hgp_lower=" << hgp_lower
                          << " os_total=" << (os_weighted - os_lower)
                          << " hgp_total=" << (hgp_weighted - hgp_lower)
                          << '\n';
            }
        }
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
                const auto hgp_coulomb_default = HartreeFock::HeadGordonPople::_compute_eri_deriv_elem(
                    sp_ab, sp_cd);
                const auto hgp_coulomb = HartreeFock::HeadGordonPople::_compute_eri_deriv_elem(
                    sp_ab, sp_cd, HartreeFock::ERIKernel::Coulomb, 0.0);
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
                if (max_abs_diff(hgp_coulomb_default, hgp_coulomb) > 1e-13)
                {
                    fail("HGP default ERI derivative kernel no longer matches the explicit Coulomb path");
                    return;
                }
                if (max_abs_diff(coulomb, hgp_coulomb) > 1e-8)
                {
                    fail("HGP Coulomb ERI derivatives no longer match the OS reference");
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
        const auto analytic_hgp_coulomb = HartreeFock::HeadGordonPople::_compute_eri_deriv_elem(
            sp_ab, sp_cd, HartreeFock::ERIKernel::Coulomb, 0.0);
        const double base_os = contracted_quartet_value(
            sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
        const double base_rys = rys_contracted_quartet_value(
            sp_ab, sp_cd, HartreeFock::ERIKernel::LongRange, omega);
        const double base_coulomb = contracted_quartet_value(
            sp_ab, sp_cd, HartreeFock::ERIKernel::Coulomb, 0.0);
        const double base_hgp_coulomb = HartreeFock::HeadGordonPople::_contracted_eri_elem(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2],
            HartreeFock::ERIKernel::Coulomb, 0.0);

        std::array<double, 12> finite_difference{};
        std::array<double, 12> finite_difference_rys{};
        std::array<double, 12> finite_difference_coulomb{};
        std::array<double, 12> finite_difference_hgp_coulomb{};
        double max_error = 0.0;
        double max_error_coulomb = 0.0;
        double max_error_hgp_coulomb = 0.0;
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
                const double plus_value_hgp_coulomb = HartreeFock::HeadGordonPople::_contracted_eri_elem(
                    *plus_ab, *plus_cd,
                    plus_ab->A._cartesian[0], plus_ab->A._cartesian[1], plus_ab->A._cartesian[2],
                    plus_ab->B._cartesian[0], plus_ab->B._cartesian[1], plus_ab->B._cartesian[2],
                    plus_cd->A._cartesian[0], plus_cd->A._cartesian[1], plus_cd->A._cartesian[2],
                    plus_cd->B._cartesian[0], plus_cd->B._cartesian[1], plus_cd->B._cartesian[2],
                    HartreeFock::ERIKernel::Coulomb, 0.0);
                const double minus_value_hgp_coulomb = HartreeFock::HeadGordonPople::_contracted_eri_elem(
                    *minus_ab, *minus_cd,
                    minus_ab->A._cartesian[0], minus_ab->A._cartesian[1], minus_ab->A._cartesian[2],
                    minus_ab->B._cartesian[0], minus_ab->B._cartesian[1], minus_ab->B._cartesian[2],
                    minus_cd->A._cartesian[0], minus_cd->A._cartesian[1], minus_cd->A._cartesian[2],
                    minus_cd->B._cartesian[0], minus_cd->B._cartesian[1], minus_cd->B._cartesian[2],
                    HartreeFock::ERIKernel::Coulomb, 0.0);
                const double fd_value_hgp_coulomb =
                    (plus_value_hgp_coulomb - minus_value_hgp_coulomb) / (2.0 * step);
                const std::size_t slot = static_cast<std::size_t>(center * 3 + direction);
                finite_difference[slot] = fd_value;
                finite_difference_rys[slot] = fd_value_rys;
                finite_difference_coulomb[slot] = fd_value_coulomb;
                finite_difference_hgp_coulomb[slot] = fd_value_hgp_coulomb;
                max_error = std::max(max_error, std::abs(fd_value - analytic[slot]));
                max_error_coulomb = std::max(max_error_coulomb, std::abs(fd_value_coulomb - analytic_coulomb[slot]));
                max_error_hgp_coulomb = std::max(
                    max_error_hgp_coulomb,
                    std::abs(fd_value_hgp_coulomb - analytic_hgp_coulomb[slot]));
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
                  << " base_hgp_coulomb=" << base_hgp_coulomb
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
        std::cout << "  max |hgp coulomb analytic-fd|=" << max_error_hgp_coulomb << '\n';
        std::cout << "  coulomb center derivatives:\n";
        for (int center = 0; center < 4; ++center)
        {
            std::cout << "    c" << center
                      << " (" << analytic_coulomb[center * 3 + 0]
                      << ", " << analytic_coulomb[center * 3 + 1]
                      << ", " << analytic_coulomb[center * 3 + 2] << ")\n";
        }
        std::cout << "  hgp coulomb center derivatives:\n";
        for (int center = 0; center < 4; ++center)
        {
            std::cout << "    c" << center
                      << " (" << analytic_hgp_coulomb[center * 3 + 0]
                      << ", " << analytic_hgp_coulomb[center * 3 + 1]
                      << ", " << analytic_hgp_coulomb[center * 3 + 2] << ")\n";
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
        if (max_abs_diff(analytic_coulomb, analytic_hgp_coulomb) > 1e-8)
        {
            fail("HGP Coulomb quartet derivatives no longer match the OS reference on the finite-difference diagnostic quartet");
            return;
        }
        if (max_error_hgp_coulomb > tolerance)
        {
            fail("HGP Coulomb quartet derivatives failed direct finite-difference validation");
            return;
        }
    }

    void test_hgp_screened_quartet_derivatives_s_p()
    {
        auto calc_res = make_water_calculator();
        if (!calc_res)
        {
            fail("setup failed: " + calc_res.error());
            return;
        }

        const auto shell_pairs = build_shellpairs(calc_res->_shells);
        constexpr double omega = 0.11;
        constexpr double fd_step = 1e-4;
        constexpr double deriv_tol = 5e-7;
        constexpr double identity_tol = 1e-10;

        const auto *ssss_ab = find_shell_pair(shell_pairs, 1, 5);
        const auto *ssss_cd = find_shell_pair(shell_pairs, 0, 6);
        const auto *psss_ab = find_shell_pair(shell_pairs, 3, 5);
        const auto *psss_cd = find_shell_pair(shell_pairs, 0, 1);
        const auto *ssps_ab = find_shell_pair(shell_pairs, 1, 5);
        const auto *ssps_cd = find_shell_pair(shell_pairs, 3, 6);
        if (!ssss_ab || !ssss_cd || !psss_ab || !psss_cd || !ssps_ab || !ssps_cd)
        {
            fail("failed to locate fixed s/p screened-derivative diagnostic quartets");
            return;
        }

        struct QuartetCase
        {
            const char *name;
            const HartreeFock::ShellPair *ab;
            const HartreeFock::ShellPair *cd;
        };

        const std::array<QuartetCase, 3> cases{{
            {"s-s-s-s", ssss_ab, ssss_cd},
            {"p-s-s-s", psss_ab, psss_cd},
            {"s-s-p-s", ssps_ab, ssps_cd},
        }};

        for (const auto &quartet : cases)
        {
            const auto os_coulomb = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::Coulomb, 0.0);
            const auto os_long_range = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::LongRange, omega);
            const auto os_short_range = HartreeFock::ObaraSaika::_compute_eri_deriv_elem(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::ShortRange, omega);

            const auto hgp_coulomb = HartreeFock::HeadGordonPople::_compute_eri_deriv_elem(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::Coulomb, 0.0);
            const auto hgp_long_range = HartreeFock::HeadGordonPople::_compute_eri_deriv_elem_native_test(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::LongRange, omega);
            const auto hgp_short_range = HartreeFock::HeadGordonPople::_compute_eri_deriv_elem_native_test(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::ShortRange, omega);
            const double os_value_long_range = contracted_quartet_value(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::LongRange, omega);
            const double hgp_value_long_range = hgp_contracted_quartet_value_native(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::LongRange, omega);
            const double os_value_short_range = contracted_quartet_value(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::ShortRange, omega);
            const double hgp_value_short_range = hgp_contracted_quartet_value_native(
                *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::ShortRange, omega);

            if (max_abs_diff(os_coulomb, hgp_coulomb) > 1e-8)
            {
                fail(std::string("HGP Coulomb screened-quartet baseline diverged from OS on ") + quartet.name);
                return;
            }
            if (max_abs_diff(os_long_range, hgp_long_range) > 1e-8)
            {
                if (std::string_view(quartet.name) == "p-s-s-s")
                    print_screened_derivative_term_breakdown(
                        *quartet.ab, *quartet.cd, HartreeFock::ERIKernel::LongRange, omega);
                fail(std::string("HGP long-range ERI derivatives no longer match OS on ") + quartet.name);
                return;
            }
            if (max_abs_diff(os_short_range, hgp_short_range) > 1e-8)
            {
                fail(std::string("HGP short-range ERI derivatives no longer match OS on ") + quartet.name);
                return;
            }

            std::array<double, 12> hgp_reconstructed{};
            for (std::size_t i = 0; i < hgp_reconstructed.size(); ++i)
                hgp_reconstructed[i] = hgp_coulomb[i] - hgp_long_range[i];
            if (max_abs_diff(hgp_reconstructed, hgp_short_range) > identity_tol)
            {
                fail(std::string("HGP short-range derivative no longer equals Coulomb - LongRange on ") + quartet.name);
                return;
            }
            if (std::abs(os_value_long_range - hgp_value_long_range) > 1e-10)
            {
                fail(std::string("HGP long-range quartet value no longer matches OS on ") + quartet.name);
                return;
            }
            if (std::abs(os_value_short_range - hgp_value_short_range) > 1e-10)
            {
                fail(std::string("HGP short-range quartet value no longer matches OS on ") + quartet.name);
                return;
            }

            const std::array<std::size_t, 4> shell_indices{
                shell_index_for_view(calc_res->_shells, quartet.ab->A),
                shell_index_for_view(calc_res->_shells, quartet.ab->B),
                shell_index_for_view(calc_res->_shells, quartet.cd->A),
                shell_index_for_view(calc_res->_shells, quartet.cd->B),
            };

            std::array<double, 12> fd_long_range{};
            std::array<double, 12> fd_short_range{};
            for (int center = 0; center < 4; ++center)
            {
                for (int direction = 0; direction < 3; ++direction)
                {
                    const std::size_t shell_index = shell_indices[center];
                    const auto plus_calc =
                        make_displaced_water_calculator("sto-3g", shell_index, direction, fd_step);
                    const auto minus_calc =
                        make_displaced_water_calculator("sto-3g", shell_index, direction, -fd_step);
                    const auto plus_pairs = build_shellpairs(plus_calc._shells);
                    const auto minus_pairs = build_shellpairs(minus_calc._shells);

                    const auto *plus_ab = find_shell_pair(
                        plus_pairs, quartet.ab->A._index, quartet.ab->B._index);
                    const auto *plus_cd = find_shell_pair(
                        plus_pairs, quartet.cd->A._index, quartet.cd->B._index);
                    const auto *minus_ab = find_shell_pair(
                        minus_pairs, quartet.ab->A._index, quartet.ab->B._index);
                    const auto *minus_cd = find_shell_pair(
                        minus_pairs, quartet.cd->A._index, quartet.cd->B._index);
                    if (!plus_ab || !plus_cd || !minus_ab || !minus_cd)
                    {
                        fail(std::string("failed to recover displaced screened quartet for ") + quartet.name);
                        return;
                    }

                    const std::size_t slot = static_cast<std::size_t>(center * 3 + direction);
                    const double plus_lr = hgp_contracted_quartet_value_native(
                        *plus_ab, *plus_cd, HartreeFock::ERIKernel::LongRange, omega);
                    const double minus_lr = hgp_contracted_quartet_value_native(
                        *minus_ab, *minus_cd, HartreeFock::ERIKernel::LongRange, omega);
                    fd_long_range[slot] = (plus_lr - minus_lr) / (2.0 * fd_step);

                    const double plus_sr = hgp_contracted_quartet_value_native(
                        *plus_ab, *plus_cd, HartreeFock::ERIKernel::ShortRange, omega);
                    const double minus_sr = hgp_contracted_quartet_value_native(
                        *minus_ab, *minus_cd, HartreeFock::ERIKernel::ShortRange, omega);
                    fd_short_range[slot] = (plus_sr - minus_sr) / (2.0 * fd_step);
                }
            }

            if (max_abs_diff(hgp_long_range, fd_long_range) > deriv_tol)
            {
                fail(std::string("HGP long-range ERI derivatives failed finite-difference validation on ") + quartet.name);
                return;
            }
            if (max_abs_diff(hgp_short_range, fd_short_range) > deriv_tol)
            {
                fail(std::string("HGP short-range ERI derivatives failed finite-difference validation on ") + quartet.name);
                return;
            }
        }
    }

    // Exercises the cases the inv_2_delta fix was suspected of affecting on the
    // *unweighted* path (no derivative): screened HGP _contracted_eri_elem must
    // match OS on s-s-s-s, p-s-s-s with ket s/s, and the (p|s|p|s) mixed-bra-ket
    // pattern that the inv_2_delta cross-coupling term governs.
    void test_hgp_unweighted_screened_eri_against_os()
    {
        auto calc_res = make_water_calculator();
        if (!calc_res)
        {
            fail("setup failed: " + calc_res.error());
            return;
        }

        const auto shell_pairs = build_shellpairs(calc_res->_shells);
        constexpr double omega = 0.11;
        constexpr double tol = 1e-10;

        // Cases the user asked to gate explicitly.
        const auto *ssss_ab = find_shell_pair(shell_pairs, 1, 5);
        const auto *ssss_cd = find_shell_pair(shell_pairs, 0, 6);
        const auto *psss_ab = find_shell_pair(shell_pairs, 3, 5);
        const auto *psss_cd = find_shell_pair(shell_pairs, 0, 1);
        // (p,s|p,s): same p-shell on the bra and ket, both ket components are
        // s-shells so the only mixed-AM coupling comes through inv_2_delta.
        const auto *psps_ab = psss_ab;
        const auto *psps_cd = find_shell_pair(shell_pairs, 3, 6);
        if (!ssss_ab || !ssss_cd || !psss_ab || !psss_cd || !psps_ab || !psps_cd)
        {
            fail("unweighted screened-ERI gate: missing fixture shell pairs");
            return;
        }

        struct UnweightedCase
        {
            const char *name;
            const HartreeFock::ShellPair *ab;
            const HartreeFock::ShellPair *cd;
        };
        const std::array<UnweightedCase, 3> targeted{{
            {"s-s-s-s", ssss_ab, ssss_cd},
            {"p-s-s-s (no ket raise)", psss_ab, psss_cd},
            {"p-s-p-s (mixed bra/ket AM)", psps_ab, psps_cd},
        }};

        for (const auto &k : targeted)
        {
            for (const auto kernel : {HartreeFock::ERIKernel::Coulomb,
                                      HartreeFock::ERIKernel::LongRange,
                                      HartreeFock::ERIKernel::ShortRange})
            {
                const double os_val = contracted_quartet_value(*k.ab, *k.cd, kernel, omega);
                const double hgp_val = hgp_contracted_quartet_value_native(*k.ab, *k.cd, kernel, omega);
                const double diff = std::abs(os_val - hgp_val);
                const int k_int = static_cast<int>(kernel);
                std::cout << "unweighted gate " << k.name << " kernel=" << k_int
                          << " os=" << std::setprecision(12) << os_val
                          << " hgp=" << hgp_val << " diff=" << diff << '\n';
                if (diff > tol)
                {
                    fail(std::string("HGP unweighted screened ERI no longer matches OS on ") +
                         k.name + " (kernel=" + std::to_string(k_int) + ")");
                    return;
                }
            }
        }

        // Broader sweep: every quartet of stored shell pairs, three kernels.
        // Catches any AM combination the targeted gate above missed.
        double sweep_max_diff = 0.0;
        std::size_t sweep_count = 0;
        for (const auto &sp_ab : shell_pairs)
        {
            for (const auto &sp_cd : shell_pairs)
            {
                for (const auto kernel : {HartreeFock::ERIKernel::Coulomb,
                                          HartreeFock::ERIKernel::LongRange,
                                          HartreeFock::ERIKernel::ShortRange})
                {
                    const double os_val = contracted_quartet_value(sp_ab, sp_cd, kernel, omega);
                    const double hgp_val = hgp_contracted_quartet_value_native(sp_ab, sp_cd, kernel, omega);
                    const double diff = std::abs(os_val - hgp_val);
                    sweep_max_diff = std::max(sweep_max_diff, diff);
                    ++sweep_count;
                    if (diff > tol)
                    {
                        std::cerr << "sweep mismatch ab=("
                                  << sp_ab.A._index << "," << sp_ab.B._index
                                  << ") cd=(" << sp_cd.A._index << "," << sp_cd.B._index
                                  << ") kernel=" << static_cast<int>(kernel)
                                  << " os=" << std::setprecision(15) << os_val
                                  << " hgp=" << hgp_val << " diff=" << diff << '\n';
                        fail("HGP unweighted screened ERI sweep diverged from OS");
                        return;
                    }
                }
            }
        }
        std::cout << "unweighted screened sweep: " << sweep_count
                  << " quartets, max |OS-HGP| = " << sweep_max_diff << '\n';
    }

} // namespace

int main()
{
    test_kernel_entry_points();
    test_long_range_quartet_derivative_against_finite_difference();
    test_hgp_screened_quartet_derivatives_s_p();
    test_hgp_unweighted_screened_eri_against_os();
    return g_ok ? 0 : 1;
}
