// Characterization test for HartreeFock::Correlation::transform_eri.
//
// transform_eri is the AO→MO four-index transform used by MP2, coupled
// cluster, CASSCF, RHF/UHF stability, and hybrid-DFT exchange. It is the
// dominant serial hotspot in the post-HF profile. This test pins its current
// numerical output so a later refactor (parallelizing / reordering the four
// quarter transforms) can be proven behavior-preserving.
//
// The production function performs the cascade
//   T1[i,ν,λ,σ] = Σ_μ C1(μ,i) eri[μνλσ]
//   T2[i,a,λ,σ] = Σ_ν C2(ν,a) T1[i,ν,λ,σ]
//   T3[i,a,j,σ] = Σ_λ C3(λ,j) T2[i,a,λ,σ]
//   out[i,a,j,b]= Σ_σ C4(σ,b) T3[i,a,j,σ]
// returning a flat row-major [n1·n2·n3·n4] tensor. We check it against an
// independent brute-force evaluation of the defining contraction, with
// distinct (non-square) leg sizes so an index/stride bug is observable.
//
// Inputs are deterministic (seeded analytic functions, no RNG).

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace HartreeFock::Correlation
{
    // Declared here (not via the heavy post_hf/integrals.h pull-in) so this test
    // links only against the symbol under test. Signature must match
    // src/post_hf/integrals.h exactly.
    std::vector<double> transform_eri(
        const std::vector<double> &eri,
        std::size_t nb,
        const Eigen::MatrixXd &C1,
        const Eigen::MatrixXd &C2,
        const Eigen::MatrixXd &C3,
        const Eigen::MatrixXd &C4);
}

namespace
{
    bool g_ok = true;

    // Frozen baseline at output index (i,a,j,b) = (0,0,0,0) for the asymmetric
    // case below, measured from the production transform when this test was
    // added. The test also prints the live value so re-pinning is trivial.
    constexpr double GOLDEN_AOMO_0000 = -14.484886967550839;

    void fail(const std::string &message)
    {
        std::cerr << "FAIL: " << message << '\n';
        g_ok = false;
    }

    // Deterministic AO ERI tensor [nb^4], flat row-major, no symmetry imposed.
    std::vector<double> make_eri(std::size_t nb, double seed)
    {
        std::vector<double> eri(nb * nb * nb * nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        eri[((mu * nb + nu) * nb + lam) * nb + sig] =
                            std::sin(seed + 0.17 * mu + 0.29 * nu + 0.41 * lam + 0.53 * sig) +
                            0.2 * std::cos(seed - 0.6 * nu + 0.19 * sig);
        return eri;
    }

    // Deterministic [nb × ncol] coefficient block.
    Eigen::MatrixXd make_C(std::size_t nb, std::size_t ncol, double seed)
    {
        Eigen::MatrixXd C(static_cast<Eigen::Index>(nb), static_cast<Eigen::Index>(ncol));
        for (std::size_t r = 0; r < nb; ++r)
            for (std::size_t c = 0; c < ncol; ++c)
                C(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                    std::cos(seed + 0.13 * r - 0.27 * c) + 0.1 * std::sin(seed + 0.5 * r * c);
        return C;
    }

    // Independent brute-force reference: direct quadruple sum over AO indices.
    // Different loop structure from the production quarter-transform cascade so
    // a shared bug is unlikely.
    std::vector<double> brute_force(const std::vector<double> &eri, std::size_t nb,
                                    const Eigen::MatrixXd &C1, const Eigen::MatrixXd &C2,
                                    const Eigen::MatrixXd &C3, const Eigen::MatrixXd &C4)
    {
        const std::size_t n1 = static_cast<std::size_t>(C1.cols());
        const std::size_t n2 = static_cast<std::size_t>(C2.cols());
        const std::size_t n3 = static_cast<std::size_t>(C3.cols());
        const std::size_t n4 = static_cast<std::size_t>(C4.cols());
        std::vector<double> out(n1 * n2 * n3 * n4, 0.0);
        for (std::size_t i = 0; i < n1; ++i)
            for (std::size_t a = 0; a < n2; ++a)
                for (std::size_t j = 0; j < n3; ++j)
                    for (std::size_t b = 0; b < n4; ++b)
                    {
                        double acc = 0.0;
                        for (std::size_t mu = 0; mu < nb; ++mu)
                            for (std::size_t nu = 0; nu < nb; ++nu)
                                for (std::size_t lam = 0; lam < nb; ++lam)
                                    for (std::size_t sig = 0; sig < nb; ++sig)
                                        acc += C1(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(i)) *
                                               C2(static_cast<Eigen::Index>(nu), static_cast<Eigen::Index>(a)) *
                                               C3(static_cast<Eigen::Index>(lam), static_cast<Eigen::Index>(j)) *
                                               C4(static_cast<Eigen::Index>(sig), static_cast<Eigen::Index>(b)) *
                                               eri[((mu * nb + nu) * nb + lam) * nb + sig];
                        out[((i * n2 + a) * n3 + j) * n4 + b] = acc;
                    }
        return out;
    }

    void check_case(std::size_t nb, std::size_t n1, std::size_t n2,
                    std::size_t n3, std::size_t n4, double seed,
                    const char *label)
    {
        const std::vector<double> eri = make_eri(nb, seed);
        const Eigen::MatrixXd C1 = make_C(nb, n1, seed + 0.1);
        const Eigen::MatrixXd C2 = make_C(nb, n2, seed + 0.2);
        const Eigen::MatrixXd C3 = make_C(nb, n3, seed + 0.3);
        const Eigen::MatrixXd C4 = make_C(nb, n4, seed + 0.4);

        const std::vector<double> prod =
            HartreeFock::Correlation::transform_eri(eri, nb, C1, C2, C3, C4);
        const std::vector<double> ref = brute_force(eri, nb, C1, C2, C3, C4);

        if (prod.size() != n1 * n2 * n3 * n4)
        {
            fail(std::string(label) + ": output size " + std::to_string(prod.size()) +
                 " (expected " + std::to_string(n1 * n2 * n3 * n4) + ")");
            return;
        }

        double max_abs = 0.0, max_scale = 1.0;
        for (std::size_t k = 0; k < prod.size(); ++k)
        {
            max_abs = std::max(max_abs, std::abs(prod[k] - ref[k]));
            max_scale = std::max(max_scale, std::abs(ref[k]));
        }
        if (max_abs > 1e-11 * max_scale)
            fail(std::string(label) + ": transform disagrees with brute force, max |Δ| = " +
                 std::to_string(max_abs) + " (scale " + std::to_string(max_scale) + ")");
    }

    // Pins one output element of the asymmetric case (the one most likely to
    // expose a stride bug) against a frozen literal.
    void check_golden_value()
    {
        const std::size_t nb = 5, n1 = 3, n2 = 4, n3 = 2, n4 = 5;
        const std::vector<double> eri = make_eri(nb, 1.7);
        const Eigen::MatrixXd C1 = make_C(nb, n1, 1.7 + 0.1);
        const Eigen::MatrixXd C2 = make_C(nb, n2, 1.7 + 0.2);
        const Eigen::MatrixXd C3 = make_C(nb, n3, 1.7 + 0.3);
        const Eigen::MatrixXd C4 = make_C(nb, n4, 1.7 + 0.4);
        const std::vector<double> prod =
            HartreeFock::Correlation::transform_eri(eri, nb, C1, C2, C3, C4);
        const double got = prod[0]; // (i,a,j,b) = (0,0,0,0)
        std::cout << "[INFO] transform_eri golden (0,0,0,0) = "
                  << std::setprecision(17) << got << '\n';
        constexpr double golden = GOLDEN_AOMO_0000;
        if (std::isnan(golden))
        {
            fail("golden: GOLDEN_AOMO_0000 is unpinned — set it to the printed "
                 "value above and rebuild");
            return;
        }
        if (std::abs(got - golden) > 1e-10 * std::max(1.0, std::abs(golden)))
            fail("golden: (0,0,0,0) = " + std::to_string(got) +
                 " differs from pinned " + std::to_string(golden));
    }
} // namespace

int main()
{
    // Square case (typical full-MO transform).
    check_case(4, 4, 4, 4, 4, 0.5, "square nb=4");
    // Asymmetric case (occ/virt blocks of different sizes — the MP2/CC usage).
    check_case(5, 3, 4, 2, 5, 1.7, "asym nb=5 (3,4,2,5)");
    // Larger square case to stress accumulation.
    check_case(6, 6, 6, 6, 6, 2.9, "square nb=6");
    check_golden_value();

    if (g_ok)
    {
        std::cout << "transform_eri_unit: OK\n";
        return 0;
    }
    std::cout << "transform_eri_unit: FAIL\n";
    return 1;
}
