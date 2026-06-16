// Unit test for src/symmetry/fock_symmetrization.{h,cpp} — Phase 1 of the
// full-symmetry ERI reduction (docs/FULL_SYMMETRY_ERI_DESIGN.md).
//
// symmetrize_matrix implements F = (1/|G|) Σ_R O_R^T M O_R, the projection onto the
// totally-symmetric component of the point group. It is the correctness-critical
// operator the skeleton-Fock reduction depends on, so the design isolates and
// tests it on its own — before any integral-engine change — via the properties it
// must satisfy for the real (non-Abelian) O_R from group_operations:
//
//   - Invariance:  O_S^T · symmetrize(M) · O_S == symmetrize(M) for every S.
//   - Idempotence: symmetrize(symmetrize(M)) == symmetrize(M).
//   - Fixed point: a matrix built group-invariant by construction
//                  (M_sym = Σ_R O_R^T A O_R) is returned unchanged (up to scale).
//   - Symmetry preserved: symmetrize of a symmetric M stays symmetric.
//
// Tested on NH3 (C3v) and CH4 (Td) so the non-Abelian O_R (C3, S4, sigma) are
// exercised, with a fixed random M to make the checks meaningful.

#include "symmetry/fock_symmetrization.h"
#include "symmetry/group_operations.h"
#include "symmetry/symmetry.h"
#include "basis/basis.h"
#include "base/basis.h"
#include "base/types.h"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{
    bool g_ok = true;

    void fail(const std::string &msg)
    {
        std::cerr << "[FAIL] " << msg << '\n';
        g_ok = false;
    }

    std::expected<HartreeFock::Calculator, std::string> make_calculator(
        const std::vector<int> &Z,
        const std::vector<std::array<double, 3>> &xyz,
        int charge, int multiplicity,
        const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;

        const std::size_t n = Z.size();
        mol.natoms = n;
        mol.charge = charge;
        mol.multiplicity = multiplicity;
        mol.atomic_numbers.resize(n);
        mol.atomic_masses.resize(n);
        mol.coordinates.resize(n, 3);
        for (std::size_t i = 0; i < n; ++i)
        {
            mol.atomic_numbers[static_cast<Eigen::Index>(i)] = Z[i];
            mol.atomic_masses[static_cast<Eigen::Index>(i)] = 1.0;
            for (int c = 0; c < 3; ++c)
                mol.coordinates(static_cast<Eigen::Index>(i), c) = xyz[i][static_cast<std::size_t>(c)];
        }

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        if (auto r = HartreeFock::Symmetry::detectSymmetry(mol, HartreeFock::Units::Angstrom); !r)
            return std::unexpected("detectSymmetry: " + r.error());

        const std::filesystem::path gbs = std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis: " + basis_res.error());
        calc._shells = std::move(*basis_res);

        if (auto r = calc.initialize(); !r)
            return std::unexpected("initialize: " + r.error());
        return calc;
    }

    Eigen::MatrixXd random_symmetric(Eigen::Index n, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::MatrixXd A(n, n);
        for (Eigen::Index i = 0; i < n; ++i)
            for (Eigen::Index j = 0; j < n; ++j)
                A(i, j) = dist(rng);
        return 0.5 * (A + A.transpose());
    }

    void check(const std::string &name,
               const std::vector<int> &Z,
               const std::vector<std::array<double, 3>> &xyz,
               int charge, int mult,
               const std::string &basis_name,
               const std::string &expected_pg)
    {
        constexpr double tol = 1e-9;

        auto calc_res = make_calculator(Z, xyz, charge, mult, basis_name);
        if (!calc_res)
        {
            fail(name + ": setup failed: " + calc_res.error());
            return;
        }
        HartreeFock::Calculator calc = std::move(*calc_res);
        if (calc._molecule._point_group != expected_pg)
        {
            fail(name + ": point group " + calc._molecule._point_group + " != " + expected_pg);
            return;
        }

        auto ops_res = HartreeFock::Symmetry::build_group_operations(calc);
        if (!ops_res || !ops_res->valid)
        {
            fail(name + ": build_group_operations failed");
            return;
        }
        const auto &ops = *ops_res;
        const Eigen::Index n = ops.operations.front().matrix.rows();

        const Eigen::MatrixXd M = random_symmetric(n, 12345u);

        auto sym_res = HartreeFock::Symmetry::symmetrize_matrix(M, ops);
        if (!sym_res)
        {
            fail(name + ": symmetrize_matrix error: " + sym_res.error());
            return;
        }
        const Eigen::MatrixXd Fsym = *sym_res;

        // 1. Invariance: O_S^T Fsym O_S == Fsym for every S.
        for (const auto &op : ops.operations)
        {
            const Eigen::MatrixXd t = op.matrix.transpose() * Fsym * op.matrix;
            if ((t - Fsym).cwiseAbs().maxCoeff() > 1e-8)
            {
                fail(name + ": symmetrized matrix not invariant under " + op.label);
                break;
            }
        }

        // 2. Idempotence: symmetrize(Fsym) == Fsym.
        auto twice = HartreeFock::Symmetry::symmetrize_matrix(Fsym, ops);
        if (!twice || (*twice - Fsym).cwiseAbs().maxCoeff() > 1e-9)
            fail(name + ": symmetrize is not idempotent");

        // 3. Symmetry preserved: Fsym stays symmetric (M was symmetric).
        if ((Fsym - Fsym.transpose()).cwiseAbs().maxCoeff() > 1e-9)
            fail(name + ": symmetrized matrix lost its symmetry");

        // 4. Fixed point on a by-construction group-invariant matrix:
        //    M_sym = Σ_R O_R^T A O_R is invariant, so symmetrize(M_sym) == M_sym.
        const Eigen::MatrixXd A = random_symmetric(n, 67890u);
        Eigen::MatrixXd Minv = Eigen::MatrixXd::Zero(n, n);
        for (const auto &op : ops.operations)
            Minv.noalias() += op.matrix.transpose() * A * op.matrix;
        Minv /= static_cast<double>(ops.operations.size());

        auto proj = HartreeFock::Symmetry::symmetrize_matrix(Minv, ops);
        if (!proj || (*proj - Minv).cwiseAbs().maxCoeff() > 1e-8)
            fail(name + ": group-invariant matrix is not a fixed point of symmetrize");

        // 5. Monomial fast path (Item A, docs/FULL_SYMMETRY_PERF_SCOPE.md). For every
        //    op flagged is_monomial, the permute-with-signs form must reproduce the
        //    dense O_Rᵀ M O_R term to ~1e-12 — the gate that the accelerator never
        //    diverges from the source-of-truth matmul. Also confirm classification
        //    actually fired (identity E is always monomial).
        int n_monomial = 0;
        for (const auto &op : ops.operations)
        {
            if (!op.is_monomial)
                continue;
            ++n_monomial;
            if (static_cast<Eigen::Index>(op.mono_map.size()) != n ||
                static_cast<Eigen::Index>(op.mono_sign.size()) != n)
            {
                fail(name + ": monomial op " + op.label + " has wrong-sized map/sign");
                continue;
            }
            // Dense reference term and the monomial-form term, on the same M.
            const Eigen::MatrixXd dense = op.matrix.transpose() * M * op.matrix;
            Eigen::MatrixXd mono = Eigen::MatrixXd::Zero(n, n);
            for (Eigen::Index mu = 0; mu < n; ++mu)
                for (Eigen::Index nu = 0; nu < n; ++nu)
                    mono(mu, nu) =
                        static_cast<double>(op.mono_sign[static_cast<std::size_t>(mu)]) *
                        static_cast<double>(op.mono_sign[static_cast<std::size_t>(nu)]) *
                        M(op.mono_map[static_cast<std::size_t>(mu)],
                          op.mono_map[static_cast<std::size_t>(nu)]);
            const double d = (dense - mono).cwiseAbs().maxCoeff();
            if (d > 1e-12)
                fail(name + ": monomial op " + op.label +
                     " term != dense matmul by " + std::to_string(d));
        }
        if (n_monomial == 0)
            fail(name + ": no operation classified monomial (identity E should be)");
        std::cout << "  " << name << ": " << n_monomial << "/" << ops.operations.size()
                  << " ops monomial\n";

        (void)tol;
    }
} // namespace

int main()
{
    check("NH3/C3v",
          {7, 1, 1, 1},
          {{{0.0000, 0.0000, 0.1173}},
           {{0.0000, 0.9377, -0.2738}},
           {{0.8121, -0.4689, -0.2738}},
           {{-0.8121, -0.4689, -0.2738}}},
          0, 1, "STO-3G", "C3v");

    check("CH4/Td",
          {6, 1, 1, 1, 1},
          {{{0.0000, 0.0000, 0.0000}},
           {{0.6276, 0.6276, 0.6276}},
           {{-0.6276, -0.6276, 0.6276}},
           {{-0.6276, 0.6276, -0.6276}},
           {{0.6276, -0.6276, -0.6276}}},
          0, 1, "STO-3G", "Td");

    // d-shell case: 6-31G* puts Cartesian d on N. Under the non-monomial C3v ops the
    // d-block of O_R is genuinely dense (a reducible 6-function set), so classification
    // must NOT flag those ops monomial — the monomial==dense gate above would catch a
    // false positive, and this case ensures that code path is exercised.
    check("NH3/C3v [6-31G*]",
          {7, 1, 1, 1},
          {{{0.0000, 0.0000, 0.1173}},
           {{0.0000, 0.9377, -0.2738}},
           {{0.8121, -0.4689, -0.2738}},
           {{-0.8121, -0.4689, -0.2738}}},
          0, 1, "6-31g*", "C3v");

    if (g_ok)
        std::cout << "fock_symmetrization: all checks passed\n";
    return g_ok ? 0 : 1;
}
