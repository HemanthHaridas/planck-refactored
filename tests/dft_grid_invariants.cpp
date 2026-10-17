#include <cmath>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "dft/base/angular.h"
#include "dft/base/grid.h"
#include "dft/base/radial.h"

namespace
{
    bool g_ok = true;

    void require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            std::cerr << message << '\n';
            g_ok = false;
        }
    }

    void require_near(double actual, double expected, double tol, const std::string &message)
    {
        if (std::abs(actual - expected) > tol)
        {
            std::ostringstream oss;
            oss << message << ": expected " << expected << ", got " << actual
                << " (tol " << tol << ")";
            std::cerr << oss.str() << '\n';
            g_ok = false;
        }
    }

    template <typename T, typename E>
    void require_unexpected(const std::expected<T, E> &result, const std::string &message)
    {
        if (!result)
            return;
        std::cerr << message << ": call unexpectedly succeeded\n";
        g_ok = false;
    }

    template <typename T, typename E>
    T require_expected(std::expected<T, E> result, const std::string &context)
    {
        if (!result)
        {
            std::cerr << context << ": " << result.error() << '\n';
            g_ok = false;
            return T{};
        }
        return std::move(*result);
    }

    void test_angular_grid_invariants()
    {
        const double four_pi = 4.0 * std::numbers::pi;
        const std::vector<int> supported_sizes = {
            1, 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350,
            434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470,
            3890, 4334, 4802, 5294, 5810};

        for (int n : supported_sizes)
        {
            const Eigen::MatrixXd grid = require_expected(
                DFT::MakeLebedevGrid(n),
                "MakeLebedevGrid(" + std::to_string(n) + ")");
            require(grid.rows() == n, "Lebedev grid returned wrong number of points");
            require(grid.cols() == 4, "Lebedev grid returned wrong number of columns");

            for (Eigen::Index i = 0; i < grid.rows(); ++i)
            {
                const double x = grid(i, 0);
                const double y = grid(i, 1);
                const double z = grid(i, 2);
                const double w = grid(i, 3);
                const double norm = std::sqrt(x * x + y * y + z * z);

                require_near(norm, 1.0, 1e-12, "Lebedev point is not on the unit sphere");
                require(std::isfinite(w), "Lebedev weight is not finite");
            }

            require_near(grid.col(3).sum(), four_pi, 1e-11, "Lebedev weights do not sum to 4pi");
        }

        const Eigen::MatrixXd grid6 = require_expected(DFT::MakeLebedevGrid(6), "MakeLebedevGrid(6)");
        const Eigen::MatrixXd grid14 = require_expected(DFT::MakeLebedevGrid(14), "MakeLebedevGrid(14)");
        const Eigen::MatrixXd grid26 = require_expected(DFT::MakeLebedevGrid(26), "MakeLebedevGrid(26)");

        auto weighted_sum = [](const Eigen::MatrixXd &grid, auto &&fn)
        {
            double sum = 0.0;
            for (Eigen::Index i = 0; i < grid.rows(); ++i)
                sum += grid(i, 3) * fn(grid(i, 0), grid(i, 1), grid(i, 2));
            return sum;
        };

        require_near(
            weighted_sum(grid6, [](double x, double, double)
                         { return x * x; }),
            four_pi / 3.0,
            1e-12,
            "6-point Lebedev grid does not integrate x^2 correctly");

        require_near(
            weighted_sum(grid14, [](double x, double, double)
                         { return x * x * x * x; }),
            four_pi / 5.0,
            1e-12,
            "14-point Lebedev grid does not integrate x^4 correctly");

        require_near(
            weighted_sum(grid26, [](double x, double, double)
                         { return x * x * x * x * x * x; }),
            four_pi / 7.0,
            1e-12,
            "26-point Lebedev grid does not integrate x^6 correctly");

        require_near(
            weighted_sum(grid26, [](double x, double y, double z)
                         { return x * y * z; }),
            0.0,
            1e-12,
            "Lebedev grid should cancel odd moments");

        require_unexpected(
            DFT::MakeLebedevGrid(0),
            "MakeLebedevGrid(0) should reject non-positive sizes");
        require_unexpected(
            DFT::MakeLebedevGrid(-3),
            "MakeLebedevGrid(-3) should reject non-positive sizes");
        require_unexpected(
            DFT::MakeLebedevGrid(7),
            "MakeLebedevGrid(7) should reject unsupported sizes");
    }

    void test_radial_grid_invariants()
    {
        require_near(DFT::treutler_xi(1), 0.80, 0.0, "Wrong Treutler xi for H");
        require_near(DFT::treutler_xi(36), 0.90, 0.0, "Wrong Treutler xi for Kr");
        require_near(DFT::treutler_xi(104), 1.50, 0.0, "Wrong Treutler xi fallback past table end");

        const Eigen::MatrixXd grid = DFT::MakeTreutlerAhlrichsGrid(100, 1.0, 0.6);
        require(grid.rows() == 100, "Radial grid returned wrong number of points");
        require(grid.cols() == 2, "Radial grid returned wrong number of columns");

        double integral_exp = 0.0;
        double integral_r_exp = 0.0;
        for (Eigen::Index i = 0; i < grid.rows(); ++i)
        {
            const double r = grid(i, 0);
            const double w = grid(i, 1);

            require(r > 0.0, "Radial grid point must be positive");
            require(w > 0.0, "Radial grid weight must be positive");

            if (i > 0)
                require(grid(i - 1, 0) > r, "Radial points must be strictly descending");

            integral_exp += w * std::exp(-r);
            integral_r_exp += w * r * std::exp(-r);
        }

        require_near(
            integral_exp,
            2.0,
            1e-6,
            "Treutler-Ahlrichs grid does not integrate exp(-r) * r^2 accurately");
        require_near(
            integral_r_exp,
            6.0,
            1e-5,
            "Treutler-Ahlrichs grid does not integrate exp(-r) * r^3 accurately");
    }

    void test_orca_like_grid_presets()
    {
        const auto coarse = require_expected(
            DFT::grid_preset(DFT::GridLevel::Coarse),
            "grid_preset(Coarse)");
        const auto normal = require_expected(
            DFT::grid_preset(DFT::GridLevel::Normal),
            "grid_preset(Normal)");
        const auto fine = require_expected(
            DFT::grid_preset(DFT::GridLevel::Fine),
            "grid_preset(Fine)");
        const auto ultrafine = require_expected(
            DFT::grid_preset(DFT::GridLevel::UltraFine),
            "grid_preset(UltraFine)");

        require(coarse.angular_scheme == 3, "Coarse grid should map to AngularGrid 3");
        require(normal.angular_scheme == 4, "Normal grid should map to AngularGrid 4");
        require(fine.angular_scheme == 5, "Fine grid should map to AngularGrid 5");
        require(ultrafine.angular_scheme == 6, "UltraFine grid should map to AngularGrid 6");

        require_near(coarse.int_acc, 4.159, 1e-12, "Wrong IntAcc for coarse grid");
        require_near(normal.int_acc, 4.388, 1e-12, "Wrong IntAcc for normal grid");
        require_near(fine.int_acc, 4.629, 1e-12, "Wrong IntAcc for fine grid");
        require_near(ultrafine.int_acc, 4.959, 1e-12, "Wrong IntAcc for ultrafine grid");

        require(require_expected(
                    DFT::effective_angular_scheme(1, DFT::GridLevel::Normal),
                    "effective_angular_scheme(H, Normal)") == 3,
                "Light-atom reduction should lower H/He angular scheme by one");
        require(require_expected(
                    DFT::radial_point_count(8, DFT::GridLevel::Normal),
                    "radial_point_count(O, Normal)") > 0,
                "Normal grid should allocate radial points for oxygen");
    }

    void test_resolved_grid_spec()
    {
        const auto spec = require_expected(
            DFT::resolve_grid_spec(DFT::GridRequest{DFT::GridLevel::Normal, 0}),
            "resolve legacy normal grid spec");
        require(spec.pruning_policy == DFT::GridPruningPolicy::LegacyOrcaIndexV1,
                "Default request must resolve the legacy pruning policy");
        require(spec.preset.level == DFT::GridLevel::Normal,
                "Resolved preset must retain requested quality");
        require(spec.fingerprint().find("legacy-orca-index-v1") != std::string::npos,
                "Resolved grid fingerprint must identify the policy version");
        require(spec.fingerprint().find("angular_override=0") != std::string::npos,
                "Resolved grid fingerprint must identify the angular override");
        require(spec.json().find("\"policy\":\"legacy-orca-index-v1\"") != std::string::npos &&
                    spec.json().find("\"angular_override\":0") != std::string::npos,
                "Resolved grid JSON must report the effective policy and override");

        const Eigen::Vector3d center(.21, -.35, .11);
        const auto direct = require_expected(
            DFT::MakeAtomicGrid(8, center, DFT::GridLevel::Normal),
            "legacy atomic grid");
        const auto resolved = require_expected(
            DFT::MakeAtomicGrid(8, center, spec),
            "resolved atomic grid");
        require(direct.rows() == resolved.rows() && direct.cols() == resolved.cols() &&
                    (direct-resolved).cwiseAbs().maxCoeff() == 0.0,
                "Resolved legacy spec must preserve the atomic grid bit-for-bit");
        require(require_expected(DFT::atomic_point_count(8, spec), "resolved point count") ==
                    require_expected(DFT::atomic_point_count(8, DFT::GridLevel::Normal), "legacy point count"),
                "Resolved legacy spec must preserve the point count");

        const auto refined = require_expected(
            DFT::resolve_grid_spec(DFT::GridRequest{DFT::GridLevel::UltraFine, 1454}),
            "resolve refined grid spec");
        const auto refined_shells = require_expected(
            DFT::angular_shell_sizes(1, refined), "refined shell sizes");
        for (int shell : refined_shells)
            require(shell == 1454, "Resolved angular override must remain unpruned");
        require(refined.fingerprint() != spec.fingerprint(),
                "Different resolved specifications must have different fingerprints");
        require_unexpected(
            DFT::resolve_grid_spec(DFT::GridRequest{DFT::GridLevel::Normal, 591}),
            "unsupported angular override must fail during resolution");
    }

    void test_radial_override_and_oracle()
    {
        constexpr int radial_shells = 96;
        const auto spec = require_expected(
            DFT::resolve_grid_spec(
                DFT::GridRequest{DFT::GridLevel::Normal, 302, radial_shells}),
            "resolve radial override");
        require(spec.radial_override == radial_shells,
                "Resolved grid must retain the requested radial override");
        require(spec.fingerprint().find("radial_override=96") != std::string::npos,
                "Resolved fingerprint must identify the radial override");
        require(require_expected(DFT::radial_point_count(1, spec), "H overridden radial count") == radial_shells &&
                    require_expected(DFT::radial_point_count(16, spec), "S overridden radial count") == radial_shells,
                "A radial override must be uniform across elements");
        require(require_expected(DFT::atomic_point_count(8, spec), "overridden atomic count") ==
                    radial_shells * 302,
                "Uniform radial/angular overrides must have a product point count");
        require_unexpected(
            DFT::resolve_grid_spec(DFT::GridRequest{DFT::GridLevel::Normal, 0, -1}),
            "negative radial override must fail during resolution");

        // Independent Gaussian references at several length scales.  The
        // radial moment is exact analytically, and the deliberately off-center
        // three-dimensional case checks the radial weights together with the
        // angular product construction:
        // int r^2 exp(-alpha r^2) dr = sqrt(pi)/(4 alpha^(3/2));
        // int exp(-alpha |r-d|^2) d^3r = (pi/alpha)^(3/2).
        const auto grid = require_expected(
            DFT::MakeAtomicGrid(8, Eigen::Vector3d::Zero(), spec),
            "overridden Gaussian-integration grid");
        const std::array<std::pair<double, Eigen::Vector3d>, 3> cases{{
            {.35, Eigen::Vector3d(.19, -.11, .23)},
            {.70, Eigen::Vector3d(.37, -.21, .46)},
            {1.40, Eigen::Vector3d(-.31, .27, .16)}}};
        for (const auto &[alpha, displacement] : cases)
        {
            double radial_integral = 0.0;
            double integral = 0.0;
            for (Eigen::Index p = 0; p < grid.rows(); ++p)
            {
                const Eigen::Vector3d point = grid.row(p).head<3>().transpose();
                const double radius = point.norm();
                radial_integral += grid(p, 3) * std::exp(-alpha * radius * radius) /
                    (4.0 * std::numbers::pi);
                integral += grid(p, 3) * std::exp(-alpha * (point-displacement).squaredNorm());
            }
            require_near(radial_integral, std::sqrt(std::numbers::pi) /
                         (4.0 * std::pow(alpha, 1.5)), 2e-7,
                         "Radial Gaussian reference is not integrated accurately");
            require_near(integral, std::pow(std::numbers::pi / alpha, 1.5), 2e-7,
                         "Radial/angular product grid misses off-center Gaussian reference");
        }
    }

    void test_g3_pruning_candidates()
    {
        const auto legacy = require_expected(DFT::resolve_grid_spec(
            DFT::GridRequest{DFT::GridLevel::Normal, 0, 0,
                             DFT::GridPruningPolicy::LegacyOrcaIndexV1}), "legacy candidate");
        const auto no_light = require_expected(DFT::resolve_grid_spec(
            DFT::GridRequest{DFT::GridLevel::Normal, 0, 0,
                             DFT::GridPruningPolicy::LegacyOrcaNoLightReductionV1}), "no-light candidate");
        const auto unpruned = require_expected(DFT::resolve_grid_spec(
            DFT::GridRequest{DFT::GridLevel::Normal, 0, 0,
                             DFT::GridPruningPolicy::LegacyOrcaUnprunedMaxV1}), "unpruned candidate");
        const auto pyscf = require_expected(DFT::resolve_grid_spec(
            DFT::GridRequest{DFT::GridLevel::Normal, 0, 0,
                             DFT::GridPruningPolicy::PySCFNWChem213}), "PySCF candidate");

        require(legacy.policy_name == "legacy-orca-index-v1" &&
                    no_light.policy_name == "legacy-orca-no-light-reduction-v1" &&
                    unpruned.policy_name == "legacy-orca-unpruned-max-v1" &&
                    pyscf.policy_name == "pyscf-nwchem-2.13",
                "Every G3 candidate must have a distinct versioned identity");
        require(require_expected(DFT::atomic_point_count(1, no_light), "H no-light count") >
                    require_expected(DFT::atomic_point_count(1, legacy), "H legacy count"),
                "No-light candidate must remove the legacy H/He reduction");
        require(require_expected(DFT::atomic_point_count(8, unpruned), "O unpruned count") == 75*302,
                "Legacy unpruned-max candidate must retain its maximum angular rule on every O shell");
        // Independent literal totals from PySCF 2.13 gen_grid.nwchem_prune,
        // using its level-3 radial/ANG_ORDER tables and Treutler radii.
        require(require_expected(DFT::atomic_point_count(1, pyscf), "PySCF H count") == 9808 &&
                    require_expected(DFT::atomic_point_count(8, pyscf), "PySCF O count") == 14082 &&
                    require_expected(DFT::atomic_point_count(16, pyscf), "PySCF S count") == 18880,
                "PySCF NWChem candidate must reproduce known PySCF 2.13 atomic point totals");
    }

    void test_unpruned_angular_refinement()
    {
        constexpr auto level = DFT::GridLevel::UltraFine;
        constexpr int angular = 974;
        for (int z : {1, 8})
        {
            const auto shells = require_expected(DFT::angular_shell_sizes(z, level, angular), "unpruned shells");
            for (int count : shells)
                require(count == angular, "Override must bypass pruning AND light-atom reduction");
            const int nr = require_expected(DFT::radial_point_count(z, level), "radial count");
            const auto radial = DFT::MakeTreutlerAhlrichsGrid(nr, DFT::treutler_xi(z));
            const Eigen::Vector3d center(.21, -.35, .11);
            const auto grid = require_expected(DFT::MakeAtomicGrid(z, center, level, angular), "unpruned atomic grid");
            require(grid.rows() == nr*angular, "Unpruned point count must be radial count times angular count");
            require(grid.rows() == require_expected(DFT::atomic_point_count(z, level, angular), "unpruned count"),
                    "Allocation and generation must use the same override");
            for (int ir = 0; ir < nr; ++ir)
            {
                const double expected_weight = 4*std::numbers::pi*radial(ir,1);
                require_near(grid.block(ir*angular,3,angular,1).sum(), expected_weight,
                             1e-11*std::max(1.0,std::abs(expected_weight)), "Angular refinement changed radial weight");
                for (int ia : {0, angular/2, angular-1})
                    require_near((grid.row(ir*angular+ia).head<3>().transpose()-center).norm(), radial(ir,0),
                                 1e-12*std::max(1.0,radial(ir,0)), "Angular refinement changed radial node");
            }
        }
        const auto old = require_expected(DFT::MakeAtomicGrid(1, Eigen::Vector3d::Zero(), level), "default grid");
        const auto zero = require_expected(DFT::MakeAtomicGrid(1, Eigen::Vector3d::Zero(), level, 0), "zero override");
        require(old.rows()==zero.rows() && (old-zero).cwiseAbs().maxCoeff()==0,
                "Omitted/zero angular override must preserve the original grid exactly");
        for (int count : {-1, 1, 591, 5811})
            require_unexpected(DFT::MakeAtomicGrid(8,Eigen::Vector3d::Zero(),level,count), "invalid angular override");

        // Exact spherical moment independent of orientation: integral (u.n)^32
        // = 4*pi/33 for every unit u. The 302-point rule has degree 29; the
        // 974-point rule has degree 53. This checks actual quadrature, not an
        // assumed equality of two rotated finite grids.
        const std::array<Eigen::Vector3d,2> axes{Eigen::Vector3d(1,0,0), Eigen::Vector3d(.8,.48,.36)};
        double coarse_error=0, refined_error=0;
        for (int count : {302, angular})
        {
            const auto grid = require_expected(DFT::MakeLebedevGrid(count), "moment grid");
            for (const auto &axis : axes)
            {
                double value=0;
                for (Eigen::Index p=0;p<grid.rows();++p)
                    value += grid(p,3)*std::pow(grid.row(p).head<3>().dot(axis),32);
                const double error=std::abs(value-4*std::numbers::pi/33);
                if (count==302) coarse_error=std::max(coarse_error,error);
                else refined_error=std::max(refined_error,error);
            }
        }
        require(refined_error<1e-11, "Refined angular grid must integrate rotated degree-32 moment");
        require(refined_error<coarse_error, "Angular refinement must reduce the unresolved moment error");

        HartreeFock::Molecule atom;
        atom.natoms=1; atom.atomic_numbers=Eigen::VectorXi::Constant(1,8);
        atom._coordinates=Eigen::MatrixXd::Zero(1,3);
        const auto molecular=require_expected(DFT::MakeMolecularGrid(atom,level,angular), "refined molecular grid");
        const auto atomic=require_expected(DFT::MakeAtomicGrid(8,Eigen::Vector3d::Zero(),level,angular), "reference atomic grid");
        require(molecular.points.rows()==atomic.rows(), "Molecular overload must forward angular override");
        require((molecular.points-atomic).cwiseAbs().maxCoeff()<1e-12,
                "Single-atom partition must preserve the refined atomic grid");
    }

    void test_atomic_and_molecular_grid_generation()
    {
        const Eigen::Vector3d center = Eigen::Vector3d::Zero();
        const Eigen::MatrixXd atomic_grid = require_expected(
            DFT::MakeAtomicGrid(8, center, DFT::GridLevel::Normal),
            "MakeAtomicGrid(O, Normal)");

        require(atomic_grid.rows() == require_expected(
                                          DFT::atomic_point_count(8, DFT::GridLevel::Normal),
                                          "atomic_point_count(O, Normal)"),
                "Atomic grid point count does not match the preset");
        require(atomic_grid.cols() == 4, "Atomic grid should have 4 columns");

        for (Eigen::Index i = 0; i < atomic_grid.rows(); ++i)
        {
            require(std::isfinite(atomic_grid(i, 3)), "Atomic grid weight should be finite");
            require(atomic_grid(i, 3) > 0.0, "Atomic grid weight should be positive");
        }

        HartreeFock::Molecule mol;
        mol.natoms = 1;
        mol.atomic_numbers.resize(1);
        mol.atomic_numbers << 8;
        mol._coordinates.resize(1, 3);
        mol._coordinates << 0.0, 0.0, 0.0;

        const DFT::MolecularGrid mol_grid = require_expected(
            DFT::MakeMolecularGrid(mol, DFT::GridLevel::Normal),
            "MakeMolecularGrid(single O, Normal)");
        require(mol_grid.points.rows() == atomic_grid.rows(),
                "Single-atom molecular grid should match the atomic grid size");
        require(mol_grid.owner.size() == mol_grid.points.rows(),
                "Owner vector should match molecular grid size");

        for (Eigen::Index i = 0; i < mol_grid.points.rows(); ++i)
        {
            require(mol_grid.owner(i) == 0, "Single-atom molecular grid should be owned by atom 0");
            require_near(mol_grid.points(i, 3), atomic_grid(i, 3), 1e-12,
                         "Single-atom Becke partition should leave atomic weights unchanged");
        }

        Eigen::VectorXi h2_z(2);
        h2_z << 1, 1;
        Eigen::MatrixXd h2_xyz(2, 3);
        h2_xyz << -0.7, 0.0, 0.0,
            0.7, 0.0, 0.0;

        const DFT::MolecularGrid h2_grid = require_expected(
            DFT::MakeMolecularGrid(
                h2_z,
                h2_xyz,
                DFT::GridLevel::Coarse),
            "MakeMolecularGrid(H2, Coarse)");

        require(h2_grid.points.rows() ==
                    2 * require_expected(
                            DFT::atomic_point_count(1, DFT::GridLevel::Coarse),
                            "atomic_point_count(H, Coarse)"),
                "Two-atom molecular grid should concatenate the two atomic grids");

        double total_weight = 0.0;
        for (Eigen::Index i = 0; i < h2_grid.points.rows(); ++i)
        {
            require(std::isfinite(h2_grid.points(i, 3)), "Molecular grid weight should be finite");
            require(h2_grid.points(i, 3) >= 0.0, "Molecular grid weight should be non-negative");
            require(h2_grid.owner(i) == 0 || h2_grid.owner(i) == 1,
                    "Molecular grid owner should be a valid atom index");
            total_weight += h2_grid.points(i, 3);
        }
        require(total_weight > 0.0, "Molecular grid should retain positive total weight");
    }

} // namespace

int main()
{
    test_angular_grid_invariants();
    test_radial_grid_invariants();
    test_orca_like_grid_presets();
    test_resolved_grid_spec();
    test_radial_override_and_oracle();
    test_g3_pruning_candidates();
    test_unpruned_angular_refinement();
    test_atomic_and_molecular_grid_generation();
    return g_ok ? 0 : 1;
}
