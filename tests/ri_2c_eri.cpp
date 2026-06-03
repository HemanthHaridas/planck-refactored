// Unit test for the RI 2-center Coulomb metric V_{PQ} = (P|Q).
//
// Three layers of validation:
//
//   1. Analytical: for two normalized uncontracted s-Gaussians at the same
//      center with exponents α, β, the exact value is
//          (s_α | s_β) = N_α N_β · 2π^(5/2) / (αβ √(α+β))   with F_0(0)=1
//      where N_x = (2x/π)^(3/4) is the standard 3D Gaussian normalization.
//      We test this on a hand-crafted single-shell aux basis.
//
//   2. Symmetry + positive-definiteness: for the shipped cc-pVDZ-RIFIT on a
//      water geometry, V must be symmetric and have positive eigenvalues
//      (small min-eig is fine — drops are handled at the Cholesky step).
//
//   3. Determinism: two consecutive computes must give bit-identical output.
//
// PySCF reference values come in once we wire compute_2c_eri into a driver
// path that can be tested against pyscf.scf.df_jk in Step 3+. This test
// stays self-contained.

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <string>

#include "base/types.h"
#include "basis/rifit.h"
#include "post_hf/ri/ri_eri.h"

namespace
{
    bool g_ok = true;
    void fail(const std::string &m)
    {
        std::cerr << "FAIL: " << m << '\n';
        g_ok = false;
    }

    std::string basis_path(const std::string &name)
    {
        if (const char *env = std::getenv("BASIS_PATH"); env && *env)
            return std::string(env) + "/" + name;
        return "basis-sets/" + name;
    }

    // Build a tiny AuxBasis directly (no file I/O) holding one normalized
    // primitive s-Gaussian on a single atom. The contracted coefficient is
    // computed analytically so that the integral matches the closed form.
    HartreeFock::AuxBasis make_single_s(double exponent, const Eigen::Vector3d &center)
    {
        HartreeFock::AuxBasis aux;
        aux.cartesian = true;

        HartreeFock::Shell s;
        s._center = center;
        s._shell = HartreeFock::ShellType::S;
        s._atom_index = 0;
        s._primitives.resize(1);
        s._primitives << exponent;
        s._coefficients.resize(1);
        s._coefficients << 1.0;
        s._normalizations.resize(1);
        // Standard 3D normalized s-Gaussian: N = (2α/π)^(3/4).
        const double N = std::pow(2.0 * exponent / std::numbers::pi, 0.75);
        s._normalizations << N;
        // contracted_normalization would fold the contracted norm into
        // _coefficients; for a single primitive with coefficient 1.0 and
        // primitive norm N, the contracted norm reduces to (∫ N²·g² dr)^(-1/2)
        // = 1.0 because N already makes g unit-normalized. So _coefficients
        // stays at 1.0.

        aux.shells.push_back(std::move(s));
        aux.offsets.push_back(0);
        aux.nfunctions = 1;
        return aux;
    }

    // Two-atom aux basis: same s-Gaussian on two centers, separated by R.
    HartreeFock::AuxBasis make_two_s(double exponent, double R)
    {
        HartreeFock::AuxBasis aux;
        aux.cartesian = true;
        const double N = std::pow(2.0 * exponent / std::numbers::pi, 0.75);
        for (int i = 0; i < 2; ++i)
        {
            HartreeFock::Shell s;
            s._center = Eigen::Vector3d(0.0, 0.0, (i == 0) ? -0.5 * R : +0.5 * R);
            s._shell = HartreeFock::ShellType::S;
            s._atom_index = i;
            s._primitives.resize(1); s._primitives << exponent;
            s._coefficients.resize(1); s._coefficients << 1.0;
            s._normalizations.resize(1); s._normalizations << N;
            aux.shells.push_back(std::move(s));
            aux.offsets.push_back(static_cast<std::size_t>(i));
        }
        aux.nfunctions = 2;
        return aux;
    }
} // namespace

int main()
{
    using HartreeFock::Correlation::RI::compute_2c_eri;
    using HartreeFock::Correlation::RI::ensure_ri_metric_ready;
    using HartreeFock::Correlation::RI::factorize_2c_metric;
    using HartreeFock::Correlation::RI::MetricFactorization;

    // ── Test 1: closed-form (s|s) at coincident centers ─────────────────────
    // For α = β, (s_α | s_α) = N² · 2π^(5/2) / (α² √(2α))
    //                        = (2α/π)^(3/2) · 2π^(5/2) / (α² √(2α))
    {
        const double alpha = 1.3;
        auto aux = make_single_s(alpha, Eigen::Vector3d::Zero());
        auto V_res = compute_2c_eri(aux);
        if (!V_res) { fail("compute_2c_eri (single s): " + V_res.error()); }
        else
        {
            const double V = (*V_res)(0, 0);
            const double N = std::pow(2.0 * alpha / std::numbers::pi, 0.75);
            const double TWO_PI_TO_5_2 =
                2.0 * std::numbers::pi * std::numbers::pi *
                std::sqrt(std::numbers::pi);
            const double expected = N * N * TWO_PI_TO_5_2 /
                                    (alpha * alpha * std::sqrt(2.0 * alpha));
            const double rel = std::abs(V - expected) / std::abs(expected);
            if (rel > 1e-13)
                fail("(s|s) at coincident centers: V=" + std::to_string(V) +
                     " expected=" + std::to_string(expected) +
                     " rel=" + std::to_string(rel));
        }
    }

    // ── Test 2: closed-form (s|s) at separated centers ──────────────────────
    // (s_α(A) | s_α(B)) = N² · 2π^(5/2) / (α² √(2α)) · F_0(T)
    //   with T = (α/2) |A−B|²  (since ρ = α²/2α = α/2)
    // F_0(T) = √(π/T) erf(√T)/2 = √(π)/(2√T) · erf(√T) ; tabulated.
    {
        const double alpha = 0.7;
        const double R = 1.4; // Bohr
        auto aux = make_two_s(alpha, R);
        auto V_res = compute_2c_eri(aux);
        if (!V_res) { fail("compute_2c_eri (two s): " + V_res.error()); }
        else
        {
            const auto &V = *V_res;
            if (std::abs(V(0, 1) - V(1, 0)) > 1e-15)
                fail("(s|s) two-atom: matrix not symmetric");

            const double N = std::pow(2.0 * alpha / std::numbers::pi, 0.75);
            const double TWO_PI_TO_5_2 =
                2.0 * std::numbers::pi * std::numbers::pi *
                std::sqrt(std::numbers::pi);
            const double K = N * N * TWO_PI_TO_5_2 /
                             (alpha * alpha * std::sqrt(2.0 * alpha));
            const double T = (alpha / 2.0) * R * R;
            // F_0(T) = (√π / (2 √T)) erf(√T) for T > 0
            const double F0 =
                (T > 1e-12)
                    ? (std::sqrt(std::numbers::pi) / (2.0 * std::sqrt(T))) * std::erf(std::sqrt(T))
                    : 1.0 - T / 3.0;
            const double expected_off = K * F0;
            const double rel_off =
                std::abs(V(0, 1) - expected_off) / std::abs(expected_off);
            if (rel_off > 1e-12)
                fail("(s|s) two-atom off-diagonal: V01=" + std::to_string(V(0, 1)) +
                     " expected=" + std::to_string(expected_off) +
                     " rel=" + std::to_string(rel_off));

            // Diagonal should equal the same-center value at T=0
            const double rel_diag =
                std::abs(V(0, 0) - K) / std::abs(K);
            if (rel_diag > 1e-13)
                fail("(s|s) two-atom diagonal: V00=" + std::to_string(V(0, 0)) +
                     " expected=" + std::to_string(K));
        }
    }

    // ── Test 3: full cc-pVDZ-RIFIT on water — symmetry + posdef ──────────────
    {
        HartreeFock::Molecule mol;
        mol.natoms = 3;
        mol.atomic_numbers.resize(3); mol.atomic_numbers << 8, 1, 1;
        mol._standard.resize(3, 3);
        mol._standard << 0.0,    0.0,    0.0,
                         0.0,    1.43,   1.11,
                         0.0,   -1.43,   1.11;

        auto aux_res = HartreeFock::BasisFunctions::read_ri_basis(
            basis_path("cc-pVDZ-RIFIT"), mol);
        if (!aux_res) { fail("load cc-pVDZ-RIFIT: " + aux_res.error()); }
        else
        {
            auto V_res = compute_2c_eri(*aux_res);
            if (!V_res) { fail("compute_2c_eri water/RIFIT: " + V_res.error()); }
            else
            {
                const auto &V = *V_res;
                // Symmetry
                const double asym = (V - V.transpose()).cwiseAbs().maxCoeff();
                if (asym > 1e-13)
                    fail("water/RIFIT V not symmetric: ||V - Vᵀ||_∞ = " +
                         std::to_string(asym));
                // Positive-semidefiniteness: allow tiny negative noise, but no
                // materially negative eigenvalues.
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(V);
                if (es.info() != Eigen::Success)
                {
                    fail("water/RIFIT eigensolve failed");
                }
                else
                {
                    const auto &ev = es.eigenvalues();
                    const double eig_tol =
                        1e-10 * std::max(1.0, ev.cwiseAbs().maxCoeff());
                    if (ev.minCoeff() < -eig_tol)
                    {
                        std::printf("[DIAG] eigenvalue range: [%.3e, %.3e], "
                                    "neg count: %lld, smallest few: %.3e %.3e %.3e\n",
                                    ev.minCoeff(), ev.maxCoeff(),
                                    static_cast<long long>((ev.array() < -eig_tol).count()),
                                    ev(0), ev(1), ev(2));
                        std::printf("[DIAG] first 8 diagonal entries:");
                        for (int i = 0; i < std::min<int>(8, V.rows()); ++i)
                            std::printf(" %.3e", V(i, i));
                        std::printf("\n[DIAG] shell L per row: ");
                        {
                            const auto &aux = *aux_res;
                            for (std::size_t k = 0; k < std::min<std::size_t>(8, aux.shells.size()); ++k)
                                std::printf("L=%d@row%zu ",
                                            static_cast<int>(aux.shells[k]._shell),
                                            aux.offsets[k]);
                        }
                        std::printf("\n");
                        fail("water/RIFIT V not positive-semidefinite");
                    }
                }

                // Determinism: re-run and check bit-identical
                auto V2_res = compute_2c_eri(*aux_res);
                if (!V2_res || (V - *V2_res).cwiseAbs().maxCoeff() != 0.0)
                    fail("water/RIFIT V is non-deterministic");

                auto fac_res = factorize_2c_metric(V, 1e-7);
                if (!fac_res)
                    fail("water/RIFIT metric factorization failed: " + fac_res.error());
                else if (fac_res->method != MetricFactorization::Method::Cholesky)
                    fail("water/RIFIT metric should use Cholesky path");

                HartreeFock::Calculator calc;
                calc._molecule = mol;
                calc._basis._basis_path = basis_path("");
                if (!calc._basis._basis_path.empty() && calc._basis._basis_path.back() == '/')
                    calc._basis._basis_path.pop_back();
                calc._mp2.use_ri = true;
                calc._mp2.ri_basis_name = "cc-pVDZ-RIFIT";
                calc._mp2.ri_basis_path = calc._basis._basis_path;
                calc._mp2.ri_lindep = 1e-7;
                auto prep_res = ensure_ri_metric_ready(calc);
                if (!prep_res)
                    fail("ensure_ri_metric_ready failed: " + prep_res.error());
                else if (!calc._ri_aux_basis || !calc._ri_metric_factor)
                    fail("ensure_ri_metric_ready did not populate RI caches");

                std::printf("water/cc-pVDZ-RIFIT: nfunctions=%lld, asym=%.1e, "
                            "min-eig=%.3e\n",
                            static_cast<long long>(V.rows()), asym,
                            es.eigenvalues().minCoeff());
            }
        }
    }

    // ── Test 4: singular metric should fall back to eigen decomposition ──────
    {
        HartreeFock::AuxBasis aux;
        aux.cartesian = true;
        const double alpha = 0.8;
        const double N = std::pow(2.0 * alpha / std::numbers::pi, 0.75);
        for (int i = 0; i < 2; ++i)
        {
            HartreeFock::Shell s;
            s._center = Eigen::Vector3d::Zero();
            s._shell = HartreeFock::ShellType::S;
            s._atom_index = i;
            s._primitives.resize(1); s._primitives << alpha;
            s._coefficients.resize(1); s._coefficients << 1.0;
            s._normalizations.resize(1); s._normalizations << N;
            aux.offsets.push_back(aux.nfunctions);
            aux.nfunctions += 1;
            aux.shells.push_back(std::move(s));
        }

        auto V_res = compute_2c_eri(aux);
        if (!V_res)
        {
            fail("singular metric build failed: " + V_res.error());
        }
        else
        {
            auto fac_res = factorize_2c_metric(*V_res, 1e-7);
            if (!fac_res)
                fail("singular metric factorization failed: " + fac_res.error());
            else if (fac_res->method != MetricFactorization::Method::Eigen)
                fail("singular metric should use eigen fallback");
            else if (fac_res->transform.rows() != 1)
                fail("singular metric should retain exactly one aux mode");
        }
    }

    if (g_ok) std::cout << "PASS: ri_2c_eri\n";
    return g_ok ? 0 : 1;
}
