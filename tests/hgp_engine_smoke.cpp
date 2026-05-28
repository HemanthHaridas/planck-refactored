#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/base.h"
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

    void require_true(bool condition, const std::string &label)
    {
        if (!condition)
            fail(label);
    }

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
} // namespace

int main()
{
    auto calc_res = make_water_calculator("sto-3g");
    if (!calc_res)
    {
        fail(calc_res.error());
        return 1;
    }

    auto &calc = *calc_res;
    const auto shell_pairs = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells.nbasis();

    const auto eri_os = _compute_2e(
        shell_pairs, nb, HartreeFock::IntegralMethod::ObaraSaika);
    const auto eri_hgp = _compute_2e(
        shell_pairs, nb, HartreeFock::IntegralMethod::HeadGordonPople);

    require_true(!eri_hgp.empty(), "HGP ERI tensor should not be empty");
    require_true(eri_hgp.size() == eri_os.size(), "HGP ERI tensor size should match OS");

    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < eri_os.size(); ++i)
        max_abs_diff = std::max(max_abs_diff, std::abs(eri_os[i] - eri_hgp[i]));

    require_near(max_abs_diff, 0.0, 1e-12, "HGP phase-1 ERI tensor must match OS");

    // Touch the parser-facing enum explicitly so the smoke test pins the new
    // engine value all the way through dispatch.
    calc._integral._engine = HartreeFock::IntegralMethod::HeadGordonPople;
    const auto eri_dispatch = _compute_2e(
        shell_pairs, nb, calc._integral._engine);
    double dispatch_diff = 0.0;
    for (std::size_t i = 0; i < eri_os.size(); ++i)
        dispatch_diff = std::max(dispatch_diff, std::abs(eri_dispatch[i] - eri_os[i]));
    require_near(dispatch_diff, 0.0, 1e-12, "Dispatch through engine=hgp must match OS");

    // Quartet-level checks pin the new contracted HGP path directly rather than
    // only observing it through whole-tensor dispatch.
    auto ss_calc = make_water_calculator("sto-3g");
    if (!ss_calc)
    {
        fail(ss_calc.error());
        return 1;
    }
    const auto ss_pairs = build_shellpairs(ss_calc-> _shells);
    const auto ss_quartet = find_quartet_for_shell_types(ss_pairs, {0, 0, 0, 0});
    require_true(ss_quartet.has_value(), "Failed to locate an s/s|s/s quartet");
    if (ss_quartet)
    {
        require_near(
            hgp_quartet_value(*ss_quartet->first, *ss_quartet->second),
            os_quartet_value(*ss_quartet->first, *ss_quartet->second),
            1e-12,
            "HGP s/s|s/s quartet mismatch");
    }

    auto sp_calc = make_water_calculator("6-31g");
    if (!sp_calc)
    {
        fail(sp_calc.error());
        return 1;
    }
    const auto sp_pairs = build_shellpairs(sp_calc-> _shells);
    const auto sp_quartet = find_quartet_for_shell_types(sp_pairs, {1, 0, 1, 0});
    require_true(sp_quartet.has_value(), "Failed to locate a p/s|p/s quartet");
    if (sp_quartet)
    {
        require_near(
            hgp_quartet_value(*sp_quartet->first, *sp_quartet->second),
            os_quartet_value(*sp_quartet->first, *sp_quartet->second),
            1e-11,
            "HGP p/s|p/s quartet mismatch");
    }

    auto d_calc = make_water_calculator("6-31g*");
    if (!d_calc)
    {
        fail(d_calc.error());
        return 1;
    }
    const auto d_pairs = build_shellpairs(d_calc-> _shells);
    const auto d_quartet = find_quartet_for_shell_types(d_pairs, {2, 0, 0, 0});
    require_true(d_quartet.has_value(), "Failed to locate a d/s|s/s quartet");
    if (d_quartet)
    {
        require_near(
            hgp_quartet_value(*d_quartet->first, *d_quartet->second),
            os_quartet_value(*d_quartet->first, *d_quartet->second),
            1e-11,
            "HGP d/s|s/s quartet mismatch");
    }

    if (!g_ok)
        return 1;

    std::cout << "hgp_engine_smoke: all checks passed\n";
    return 0;
}
