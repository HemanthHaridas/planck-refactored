#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
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
    using Clock = std::chrono::steady_clock;

    double time_ms(int reps, const std::function<void()> &fn)
    {
        fn();
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(reps));
        for (int r = 0; r < reps; ++r)
        {
            const auto t0 = Clock::now();
            fn();
            const auto t1 = Clock::now();
            samples.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2];
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

    void bench_basis(const std::string &basis_name, int reps)
    {
        auto calc_res = make_water_calculator(basis_name);
        if (!calc_res)
        {
            std::printf("[SKIP] %s: %s\n", basis_name.c_str(), calc_res.error().c_str());
            return;
        }

        const auto shell_pairs = build_shellpairs(calc_res->_shells);
        const std::size_t nb = calc_res->_shells.nbasis();

        const auto eri_os = HartreeFock::ObaraSaika::_compute_2e(shell_pairs, nb);
        const auto eri_rys = HartreeFock::RysQuad::_compute_2e(shell_pairs, nb);
        const auto eri_hgp = HartreeFock::HeadGordonPople::_compute_2e(shell_pairs, nb);

        double max_os_hgp = 0.0;
        double max_rys_hgp = 0.0;
        for (std::size_t i = 0; i < eri_os.size(); ++i)
        {
            max_os_hgp = std::max(max_os_hgp, std::abs(eri_os[i] - eri_hgp[i]));
            max_rys_hgp = std::max(max_rys_hgp, std::abs(eri_rys[i] - eri_hgp[i]));
        }

        const double t_os = time_ms(reps, [&]
                                    { volatile double v = HartreeFock::ObaraSaika::_compute_2e(shell_pairs, nb)[0]; (void)v; });
        const double t_rys = time_ms(reps, [&]
                                     { volatile double v = HartreeFock::RysQuad::_compute_2e(shell_pairs, nb)[0]; (void)v; });
        const double t_hgp = time_ms(reps, [&]
                                     { volatile double v = HartreeFock::HeadGordonPople::_compute_2e(shell_pairs, nb)[0]; (void)v; });

        std::printf(
            "%-8s nb=%3zu pairs=%3zu  OS=%9.3f ms  Rys=%9.3f ms  HGP=%9.3f ms  max|OS-HGP|=%8.2e  max|Rys-HGP|=%8.2e\n",
            basis_name.c_str(), nb, shell_pairs.size(), t_os, t_rys, t_hgp,
            max_os_hgp, max_rys_hgp);
    }
} // namespace

int main(int argc, char **argv)
{
    int reps = 5;
    if (argc > 1)
        reps = std::max(1, std::atoi(argv[1]));

    std::printf("HGP AO-ERI benchmark (water, median of %d reps)\n", reps);
    bench_basis("sto-3g", reps);
    bench_basis("6-31g", reps);
    bench_basis("6-31g*", reps);
    return 0;
}
