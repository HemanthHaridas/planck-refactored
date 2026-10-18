#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "require_boys_library.h"
#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/hgp.h"
#include "integrals/low_l_eri.h"
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

    // Sweep every component quartet with total L <= 2 through the closed-form
    // shell kernel (low_l_eri.h) and compare against per-component HGP. The
    // kernel is fed the first-component pair of each shell and rescaled by
    // each component's norm ratio, exactly as the fused loop does, so the axis
    // readout and the d-shell norms are checked too. Every raise layout must
    // occur: none, one raise on each of A..D, and all ten (c1 <= c2) pairs.
    void check_low_l_kernel(const std::vector<HartreeFock::ShellPair> &pairs, std::size_t nb,
                            const std::string &label, bool expect_d)
    {
        const auto L = [](const HartreeFock::ContractedView &v)
        { return v._cartesian[0] + v._cartesian[1] + v._cartesian[2]; };

        // AO -> first AO of its shell, and (i,j) -> pair.
        std::vector<std::size_t> first(nb, 0);
        std::vector<const HartreeFock::ShellPair *> pair_of(nb * nb, nullptr);
        for (const auto &sp : pairs)
            pair_of[sp.A._index * nb + sp.B._index] = &sp;
        for (std::size_t ao = 1; ao < nb; ++ao)
        {
            const auto *prev = pair_of[(ao - 1) * nb + ao - 1];
            const auto *cur = pair_of[ao * nb + ao];
            first[ao] = (cur->A._shell == prev->A._shell) ? first[ao - 1] : ao;
        }
        const auto norm_of = [&](const HartreeFock::ContractedView &v)
        { return pair_of[v._index * nb + v._index]->A._component_norm; };

        double max_rel = 0.0;
        std::size_t n = 0;
        std::array<std::size_t, 15> layout{}; // [0] none, [1+c] one raise, [5+..] pairs
        const auto pair_slot = [](int a, int b) { // a <= b, 10 pairs
            int s = 0;
            for (int x = 0; x < 4; ++x)
                for (int y = x; y < 4; ++y, ++s)
                    if (x == a && y == b)
                        return s;
            return -1;
        };
        for (const auto &ab : pairs)
            for (const auto &cd : pairs)
            {
                const HartreeFock::ContractedView *v[4] = {&ab.A, &ab.B, &cd.A, &cd.B};
                if (L(*v[0]) + L(*v[1]) + L(*v[2]) + L(*v[3]) > 2)
                    continue;
                int nr = 0, cs[2] = {0, 0}, axes[2] = {0, 0}, na = 0;
                double ratio = 1.0;
                for (int c = 0; c < 4; ++c)
                {
                    for (int u = 0; u < L(*v[c]); ++u)
                        cs[nr++] = c;
                    for (int a = 0; a < 3; ++a)
                        for (int u = 0; u < v[c]->_cartesian[a]; ++u)
                            axes[na++] = a;
                    const auto &f = *pair_of[first[v[c]->_index] * nb + first[v[c]->_index]];
                    ratio *= norm_of(*v[c]) / f.A._component_norm;
                }
                const auto *ab0 = pair_of[first[ab.A._index] * nb + first[ab.B._index]];
                const auto *cd0 = pair_of[first[cd.A._index] * nb + first[cd.B._index]];
                std::array<double, 9> out{};
                HartreeFock::Integrals::low_l_shell_eri(*ab0, *cd0, nr, cs[0], cs[1], out);
                const int slot = nr == 2 ? 3 * axes[0] + axes[1] : axes[0];
                const double got = out[slot] * ratio;
                const double ref = hgp_quartet_value(ab, cd);
                max_rel = std::max(max_rel, std::abs(got - ref) / std::max(1.0, std::abs(ref)));
                ++n;
                ++layout[nr == 0 ? 0 : nr == 1 ? 1 + cs[0] : 5 + pair_slot(cs[0], cs[1])];
            }

        std::cout << "[INFO] low-L closed form vs HGP (" << label << "): " << n
                  << " quartets, max rel diff " << max_rel << '\n';
        require_true(n > 0, label + ": no low-L quartets swept");
        for (int c = 0; c < 15; ++c)
        {
            // Same-center pairs are d shells: only a d basis can produce them.
            const bool d_layout = c == 5 + 0 || c == 5 + 4 || c == 5 + 7 || c == 5 + 9;
            if (d_layout && !expect_d)
                continue;
            require_true(layout[c] > 0, label + ": raise layout " + std::to_string(c) + " never exercised");
        }
        require_near(max_rel, 0.0, 1e-14, label + ": low-L closed form must match HGP");
    }
} // namespace

int main()
{
    if (!require_boys_library())
        return 1;
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

    check_low_l_kernel(sp_pairs, sp_calc->_shells.nbasis(), "water/6-31g", false);
    check_low_l_kernel(d_pairs, d_calc->_shells.nbasis(), "water/6-31g*", true);

    if (!g_ok)
        return 1;

    std::cout << "hgp_engine_smoke: all checks passed\n";
    return 0;
}
