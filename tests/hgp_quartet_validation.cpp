#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/hgp.h"
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

    struct QuartetSpec
    {
        std::string label;
        std::array<int, 4> shell_types;
        double tol = 1e-11;
    };

    std::optional<std::pair<const HartreeFock::ShellPair *, const HartreeFock::ShellPair *>>
    find_quartet_for_shell_types(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const std::array<int, 4> &shell_types)
    {
        for (const auto &sp_ab : shell_pairs)
        {
            const int la = static_cast<int>(sp_ab.A._shell->_shell);
            const int lb = static_cast<int>(sp_ab.B._shell->_shell);
            if (la != shell_types[0] || lb != shell_types[1])
                continue;

            for (const auto &sp_cd : shell_pairs)
            {
                const int lc = static_cast<int>(sp_cd.A._shell->_shell);
                const int ld = static_cast<int>(sp_cd.B._shell->_shell);
                if (lc == shell_types[2] && ld == shell_types[3])
                    return std::make_pair(&sp_ab, &sp_cd);
            }
        }
        return std::nullopt;
    }

    double os_quartet_value(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd)
    {
        return HartreeFock::ObaraSaika::_contracted_eri_elem(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2]);
    }

    double hgp_quartet_value(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd)
    {
        return HartreeFock::HeadGordonPople::_contracted_eri_elem(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2]);
    }

    double rys_quartet_value(
        const HartreeFock::ShellPair &sp_ab,
        const HartreeFock::ShellPair &sp_cd)
    {
        return HartreeFock::RysQuad::_rys_contracted_eri(
            sp_ab, sp_cd,
            sp_ab.A._cartesian[0], sp_ab.A._cartesian[1], sp_ab.A._cartesian[2],
            sp_ab.B._cartesian[0], sp_ab.B._cartesian[1], sp_ab.B._cartesian[2],
            sp_cd.A._cartesian[0], sp_cd.A._cartesian[1], sp_cd.A._cartesian[2],
            sp_cd.B._cartesian[0], sp_cd.B._cartesian[1], sp_cd.B._cartesian[2]);
    }

    void require_near(double actual, double expected, double tol, const std::string &label)
    {
        if (std::abs(actual - expected) > tol)
        {
            std::ostringstream oss;
            oss << label << ": expected " << expected << ", got " << actual
                << " (tol " << tol << ")";
            fail(oss.str());
        }
    }

    void check_basis(const std::string &basis_name, const std::vector<QuartetSpec> &specs)
    {
        auto calc_res = make_water_calculator(basis_name);
        if (!calc_res)
        {
            fail(calc_res.error());
            return;
        }

        const auto shell_pairs = build_shellpairs(calc_res->_shells);
        for (const auto &spec : specs)
        {
            const auto quartet = find_quartet_for_shell_types(shell_pairs, spec.shell_types);
            if (!quartet)
            {
                fail("Missing quartet " + spec.label + " in basis " + basis_name);
                continue;
            }

            const double os = os_quartet_value(*quartet->first, *quartet->second);
            const double hgp = hgp_quartet_value(*quartet->first, *quartet->second);
            const double rys = rys_quartet_value(*quartet->first, *quartet->second);

            require_near(hgp, os, spec.tol, basis_name + " " + spec.label + " HGP vs OS");
            require_near(hgp, rys, spec.tol, basis_name + " " + spec.label + " HGP vs Rys");
        }
    }
} // namespace

int main()
{
    check_basis(
        "sto-3g",
        {
            {"s/s|s/s", {0, 0, 0, 0}, 1e-12},
            {"p/s|p/s", {1, 0, 1, 0}, 1e-11},
        });

    check_basis(
        "6-31g",
        {
            {"s/s|s/s", {0, 0, 0, 0}, 1e-12},
            {"p/s|p/s", {1, 0, 1, 0}, 1e-11},
            {"p/p|p/p", {1, 1, 1, 1}, 1e-10},
        });

    check_basis(
        "6-31g*",
        {
            {"d/s|s/s", {2, 0, 0, 0}, 1e-11},
            {"d/s|p/s", {2, 0, 1, 0}, 1e-10},
            {"d/s|d/s", {2, 0, 2, 0}, 1e-10},
        });

    if (!g_ok)
        return 1;

    std::cout << "hgp_quartet_validation: all checks passed\n";
    return 0;
}
