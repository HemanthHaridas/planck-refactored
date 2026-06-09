// Characterization test for HartreeFock::ObaraSaika::_compute_2e (the one-shot
// full ERI tensor build) ahead of flattening its triangular (p,q) parallel
// loop for load balance.
//
// The loop scatter is store-only (write_eri_permutations uses `omp atomic
// write`, every writer storing the same canonical value), so the output tensor
// is independent of iteration order. A correct flattening must therefore leave
// the tensor *bitwise-identical*. This test pins that with three independent
// checks:
//
//   1. 8-fold permutational symmetry of the output tensor — a structural
//      invariant that a dropped/duplicated quartet in a reindexed loop would
//      break immediately.
//   2. Frozen golden values at a few representative indices.
//   3. A full-tensor checksum (sum and sum-of-squares) frozen from the current
//      build.
//
// Inputs are the fixed water/STO-3G shell set used by the HGP smoke test.

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"

namespace
{
    bool g_ok = true;

    // Frozen baselines, measured from the production OS _compute_2e when this
    // test was added. The test prints the live values so re-pinning is trivial
    // if the inputs ever change. (NaN until pinned: see check_golden.)
    constexpr double GOLDEN_SUM = 75.205770786685136;
    constexpr double GOLDEN_SUMSQ = 66.562378763425713;
    constexpr double GOLDEN_0000 = 4.7850657518157149;
    constexpr double GOLDEN_MIX = 0.13687336735613378; // element (0,1,1,0)

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    HartreeFock::Calculator make_water_sto3g()
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
            std::filesystem::path(get_basis_path()) / "sto-3g";
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
        {
            fail("read_gbs_basis failed: " + basis_res.error());
            return calc;
        }
        calc._shells = std::move(*basis_res);
        return calc;
    }

    // (ij|kl) accessor into the flat row-major tensor.
    inline double at(const std::vector<double> &eri, std::size_t nb,
                     std::size_t i, std::size_t j, std::size_t k, std::size_t l)
    {
        return eri[((i * nb + j) * nb + k) * nb + l];
    }

    // The ERI tensor must obey the full 8-fold permutational symmetry
    // (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (lk|ij) = (kl|ji) = (lk|ji).
    // A flattening bug that skips or double-scatters a quartet breaks this.
    void check_permutational_symmetry(const std::vector<double> &eri, std::size_t nb)
    {
        double max_asym = 0.0;
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                for (std::size_t k = 0; k < nb; ++k)
                    for (std::size_t l = 0; l < nb; ++l)
                    {
                        const double v = at(eri, nb, i, j, k, l);
                        const double perms[7] = {
                            at(eri, nb, j, i, k, l),
                            at(eri, nb, i, j, l, k),
                            at(eri, nb, j, i, l, k),
                            at(eri, nb, k, l, i, j),
                            at(eri, nb, l, k, i, j),
                            at(eri, nb, k, l, j, i),
                            at(eri, nb, l, k, j, i),
                        };
                        for (double p : perms)
                            max_asym = std::max(max_asym, std::abs(v - p));
                    }
        if (max_asym > 1e-13)
            fail("8-fold permutational symmetry violated, max asymmetry = " +
                 std::to_string(max_asym));
    }

    void check_golden(const std::vector<double> &eri, std::size_t nb)
    {
        double sum = 0.0, sumsq = 0.0;
        for (double v : eri)
        {
            sum += v;
            sumsq += v * v;
        }
        const double e0000 = at(eri, nb, 0, 0, 0, 0);
        // A representative nonzero off-diagonal element (i!=j, k!=l,
        // (i,j)!=(k,l)) for water/STO-3G (nb = 7).
        const double emix = at(eri, nb, 0, 1, 1, 0);

        std::cout << std::setprecision(17)
                  << "[INFO] _compute_2e sum   = " << sum << '\n'
                  << "[INFO] _compute_2e sumsq = " << sumsq << '\n'
                  << "[INFO] _compute_2e (0000) = " << e0000 << '\n'
                  << "[INFO] _compute_2e (0110) = " << emix << '\n';

        if (std::isnan(GOLDEN_SUM))
        {
            fail("goldens unpinned — set GOLDEN_SUM/SUMSQ/0000/MIX to the printed "
                 "values above and rebuild");
            return;
        }
        auto near = [](double a, double b)
        { return std::abs(a - b) <= 1e-10 * std::max(1.0, std::abs(b)); };
        if (!near(sum, GOLDEN_SUM))
            fail("sum " + std::to_string(sum) + " != golden " + std::to_string(GOLDEN_SUM));
        if (!near(sumsq, GOLDEN_SUMSQ))
            fail("sumsq " + std::to_string(sumsq) + " != golden " + std::to_string(GOLDEN_SUMSQ));
        if (!near(e0000, GOLDEN_0000))
            fail("(0000) " + std::to_string(e0000) + " != golden " + std::to_string(GOLDEN_0000));
        if (!near(emix, GOLDEN_MIX))
            fail("(0123) " + std::to_string(emix) + " != golden " + std::to_string(GOLDEN_MIX));
    }
} // namespace

int main()
{
    HartreeFock::Calculator calc = make_water_sto3g();
    if (!g_ok)
    {
        std::cout << "compute_2e_unit: FAIL\n";
        return 1;
    }

    const auto shell_pairs = build_shellpairs(calc._shells);
    const std::size_t nb = calc._shells.nbasis();

    const auto eri =
        _compute_2e(shell_pairs, nb, HartreeFock::IntegralMethod::ObaraSaika);

    if (eri.size() != nb * nb * nb * nb)
    {
        fail("ERI tensor size " + std::to_string(eri.size()) +
             " (expected " + std::to_string(nb * nb * nb * nb) + ")");
    }
    else
    {
        check_permutational_symmetry(eri, nb);
        check_golden(eri, nb);
    }

    if (g_ok)
    {
        std::cout << "compute_2e_unit: OK\n";
        return 0;
    }
    std::cout << "compute_2e_unit: FAIL\n";
    return 1;
}
