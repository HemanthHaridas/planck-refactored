// Characterization test for HartreeFock::BasisFunctions::transform_eri_cart_to_sph.
//
// This pins the *current* numerical output of the whole-ERI Cartesian→spherical
// transform so that a later refactor (parallelizing / reordering the four
// single-index contractions) can be proven to preserve behavior. It is a
// safety net, not a derivation: the production function is checked against an
// independent brute-force evaluation of the defining contraction
//
//     (pq|rs)_sph = Σ_{μνλσ} C_pμ C_qν C_rλ C_sσ (μν|λσ)_cart
//
// computed here with its own loops, plus a handful of hardcoded golden values
// so a silent change in both the production code and a re-derived oracle would
// still be caught.
//
// Inputs are deterministic (seeded analytic functions, no RNG) so the same
// tensor is produced on every run and on every platform.

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "basis/spherical.h"

namespace
{
    bool g_ok = true;

    // Frozen baseline for the L=1 golden index (p,q,r,s)=(0,1,2,0), measured
    // from the production transform when this test was added. The test also
    // prints the live value so re-pinning is trivial if inputs ever change.
    constexpr double GOLDEN_SPH_012_0 = 0.23243313099757315;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    // Deterministic dense Cartesian ERI tensor [n_cart^4], flat row-major.
    // No symmetry is imposed: the transform must handle a generic tensor, and
    // using an asymmetric one makes index-order bugs observable.
    std::vector<double> make_eri(std::size_t nc, double seed)
    {
        std::vector<double> eri(nc * nc * nc * nc);
        for (std::size_t mu = 0; mu < nc; ++mu)
            for (std::size_t nu = 0; nu < nc; ++nu)
                for (std::size_t lam = 0; lam < nc; ++lam)
                    for (std::size_t sig = 0; sig < nc; ++sig)
                    {
                        const double v =
                            std::sin(seed + 0.11 * mu + 0.23 * nu +
                                     0.37 * lam + 0.51 * sig) +
                            0.3 * std::cos(seed - 0.7 * mu + 0.13 * sig);
                        eri[((mu * nc + nu) * nc + lam) * nc + sig] = v;
                    }
        return eri;
    }

    // Independent brute-force reference for the four-index transform.
    // Quadruple sum over Cartesian indices — O(n_sph^4 · n_cart^4), only used
    // on tiny cases. Deliberately written differently from the production
    // quarter-transform cascade so a shared bug is unlikely.
    std::vector<double> brute_force(const std::vector<double> &eri_cart,
                                    const Eigen::MatrixXd &C,
                                    std::size_t nc)
    {
        const std::size_t ns = static_cast<std::size_t>(C.rows());
        std::vector<double> out(ns * ns * ns * ns, 0.0);
        for (std::size_t p = 0; p < ns; ++p)
            for (std::size_t q = 0; q < ns; ++q)
                for (std::size_t r = 0; r < ns; ++r)
                    for (std::size_t s = 0; s < ns; ++s)
                    {
                        double acc = 0.0;
                        for (std::size_t mu = 0; mu < nc; ++mu)
                            for (std::size_t nu = 0; nu < nc; ++nu)
                                for (std::size_t lam = 0; lam < nc; ++lam)
                                    for (std::size_t sig = 0; sig < nc; ++sig)
                                        acc += C(p, mu) * C(q, nu) * C(r, lam) * C(s, sig) *
                                               eri_cart[((mu * nc + nu) * nc + lam) * nc + sig];
                        out[((p * ns + q) * ns + r) * ns + s] = acc;
                    }
        return out;
    }

    // Exercise the production transform against the brute-force oracle for one
    // single-shell angular momentum L (so C is the per-shell cart_to_sph_block,
    // [2L+1 × n_cart]).
    void check_for_L(int L)
    {
        auto C_res = HartreeFock::BasisFunctions::cart_to_sph_block(L);
        if (!C_res)
        {
            fail("cart_to_sph_block(L=" + std::to_string(L) + ") failed: " + C_res.error());
            return;
        }
        const Eigen::MatrixXd C = *C_res;
        const std::size_t nc = static_cast<std::size_t>(C.cols());
        const std::size_t ns = static_cast<std::size_t>(C.rows());

        const std::vector<double> eri = make_eri(nc, 0.9 + L);

        auto prod_res = HartreeFock::BasisFunctions::transform_eri_cart_to_sph(eri, C, nc);
        if (!prod_res)
        {
            fail("transform_eri_cart_to_sph(L=" + std::to_string(L) + ") failed: " + prod_res.error());
            return;
        }
        const std::vector<double> &prod = *prod_res;
        const std::vector<double> ref = brute_force(eri, C, nc);

        if (prod.size() != ns * ns * ns * ns)
        {
            fail("L=" + std::to_string(L) + ": output size " + std::to_string(prod.size()) +
                 " (expected " + std::to_string(ns * ns * ns * ns) + ")");
            return;
        }

        double max_abs = 0.0;
        double max_scale = 1.0;
        for (std::size_t i = 0; i < prod.size(); ++i)
        {
            max_abs = std::max(max_abs, std::abs(prod[i] - ref[i]));
            max_scale = std::max(max_scale, std::abs(ref[i]));
        }
        // Tolerance is generous relative to a four-index contraction in double
        // precision; the point is to catch real reordering/indexing errors, not
        // last-bit rounding. ~1e-11 is comfortably tighter than the energies
        // these transforms feed.
        if (max_abs > 1e-11 * max_scale)
            fail("L=" + std::to_string(L) + ": transform disagrees with brute force, max |Δ| = " +
                 std::to_string(max_abs) + " (scale " + std::to_string(max_scale) + ")");
    }

    // Pinned golden value: L=1 (p shell, n_cart = n_sph = 3) with the fixed
    // seed above, index (p,q,r,s) = (0,1,2,0). Frozen from the current
    // production transform so a refactor that changes both the production code
    // and the in-test brute-force oracle in the same wrong way is still caught.
    //
    // The literal below was measured from the production code at the time this
    // test was added (see the [INFO] line this test prints). If the value ever
    // legitimately changes, update it here in the same commit and say why.
    void check_golden_value()
    {
        auto C_res = HartreeFock::BasisFunctions::cart_to_sph_block(1);
        if (!C_res)
        {
            fail("golden: cart_to_sph_block(1) failed: " + C_res.error());
            return;
        }
        const Eigen::MatrixXd C = *C_res;
        const std::size_t nc = static_cast<std::size_t>(C.cols());
        const std::size_t ns = static_cast<std::size_t>(C.rows());
        const std::vector<double> eri = make_eri(nc, 0.9 + 1);
        auto prod_res = HartreeFock::BasisFunctions::transform_eri_cart_to_sph(eri, C, nc);
        if (!prod_res)
        {
            fail("golden: transform failed: " + prod_res.error());
            return;
        }
        const std::vector<double> &prod = *prod_res;
        const std::size_t idx = ((0 * ns + 1) * ns + 2) * ns + 0;
        const double got = prod[idx];
        // Always print the live value so the baseline is visible in the test log
        // and easy to re-pin if the inputs ever change.
        std::cout << "[INFO] transform_eri_cart_to_sph golden (0,1,2,0) = "
                  << std::setprecision(17) << got << '\n';
        constexpr double golden = GOLDEN_SPH_012_0;
        if (std::isnan(golden))
        {
            fail("golden: GOLDEN_SPH_012_0 is unpinned — set it to the printed "
                 "value above and rebuild");
            return;
        }
        if (std::abs(got - golden) > 1e-10 * std::max(1.0, std::abs(golden)))
            fail("golden: (0,1,2,0) = " + std::to_string(got) +
                 " differs from pinned " + std::to_string(golden));
    }
} // namespace

int main()
{
    for (int L = 0; L <= 4; ++L)
        check_for_L(L);
    check_golden_value();

    if (g_ok)
    {
        std::cout << "transform_eri_sph_unit: OK\n";
        return 0;
    }
    std::cout << "transform_eri_sph_unit: FAIL\n";
    return 1;
}
