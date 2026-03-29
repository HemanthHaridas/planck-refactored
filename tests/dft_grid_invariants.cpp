#include <cmath>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/base/angular.h"
#include "dft/base/grid.h"
#include "dft/base/radial.h"

namespace
{

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(double actual, double expected, double tol, const std::string& message)
{
    if (std::abs(actual - expected) > tol)
    {
        std::ostringstream oss;
        oss << message << ": expected " << expected << ", got " << actual
            << " (tol " << tol << ")";
        throw std::runtime_error(oss.str());
    }
}

template <typename Func>
void require_throws_invalid_argument(Func&& fn, const std::string& message)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument&)
    {
        return;
    }
    throw std::runtime_error(message);
}

void test_angular_grid_invariants()
{
    const double four_pi = 4.0 * std::numbers::pi;
    const std::vector<int> supported_sizes = {
        1, 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350,
        434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470,
        3890, 4334, 4802, 5294, 5810
    };

    for (int n : supported_sizes)
    {
        const Eigen::MatrixXd grid = DFT::MakeLebedevGrid(n);
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

    const Eigen::MatrixXd grid6 = DFT::MakeLebedevGrid(6);
    const Eigen::MatrixXd grid14 = DFT::MakeLebedevGrid(14);
    const Eigen::MatrixXd grid26 = DFT::MakeLebedevGrid(26);

    auto weighted_sum = [](const Eigen::MatrixXd& grid, auto&& fn)
    {
        double sum = 0.0;
        for (Eigen::Index i = 0; i < grid.rows(); ++i)
            sum += grid(i, 3) * fn(grid(i, 0), grid(i, 1), grid(i, 2));
        return sum;
    };

    require_near(
        weighted_sum(grid6, [](double x, double, double) { return x * x; }),
        four_pi / 3.0,
        1e-12,
        "6-point Lebedev grid does not integrate x^2 correctly");

    require_near(
        weighted_sum(grid14, [](double x, double, double) { return x * x * x * x; }),
        four_pi / 5.0,
        1e-12,
        "14-point Lebedev grid does not integrate x^4 correctly");

    require_near(
        weighted_sum(grid26, [](double x, double, double) { return x * x * x * x * x * x; }),
        four_pi / 7.0,
        1e-12,
        "26-point Lebedev grid does not integrate x^6 correctly");

    require_near(
        weighted_sum(grid26, [](double x, double y, double z) { return x * y * z; }),
        0.0,
        1e-12,
        "Lebedev grid should cancel odd moments");

    require_throws_invalid_argument(
        [] { (void)DFT::MakeLebedevGrid(0); },
        "MakeLebedevGrid(0) should reject non-positive sizes");
    require_throws_invalid_argument(
        [] { (void)DFT::MakeLebedevGrid(-3); },
        "MakeLebedevGrid(-3) should reject non-positive sizes");
    require_throws_invalid_argument(
        [] { (void)DFT::MakeLebedevGrid(7); },
        "MakeLebedevGrid(7) should reject unsupported sizes");
}

void test_radial_grid_invariants()
{
    require_near(DFT::treutler_radius(1), 0.80, 0.0, "Wrong Treutler radius for H");
    require_near(DFT::treutler_radius(36), 1.60, 0.0, "Wrong Treutler radius for Kr");
    require_near(DFT::treutler_radius(54), 2.00, 0.0, "Wrong Treutler fallback radius");

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

    require_throws_invalid_argument(
        [] { (void)DFT::MakeTreutlerAhlrichsGrid(0); },
        "MakeTreutlerAhlrichsGrid(0) should reject non-positive sizes");
}

void test_orca_like_grid_presets()
{
    const auto coarse = DFT::grid_preset(DFT::GridLevel::Coarse);
    const auto normal = DFT::grid_preset(DFT::GridLevel::Normal);
    const auto fine = DFT::grid_preset(DFT::GridLevel::Fine);
    const auto ultrafine = DFT::grid_preset(DFT::GridLevel::UltraFine);

    require(coarse.angular_scheme == 3, "Coarse grid should map to AngularGrid 3");
    require(normal.angular_scheme == 4, "Normal grid should map to AngularGrid 4");
    require(fine.angular_scheme == 5, "Fine grid should map to AngularGrid 5");
    require(ultrafine.angular_scheme == 6, "UltraFine grid should map to AngularGrid 6");

    require_near(coarse.int_acc, 4.159, 1e-12, "Wrong IntAcc for coarse grid");
    require_near(normal.int_acc, 4.388, 1e-12, "Wrong IntAcc for normal grid");
    require_near(fine.int_acc, 4.629, 1e-12, "Wrong IntAcc for fine grid");
    require_near(ultrafine.int_acc, 4.959, 1e-12, "Wrong IntAcc for ultrafine grid");

    require(DFT::effective_angular_scheme(1, DFT::GridLevel::Normal) == 3,
            "Light-atom reduction should lower H/He angular scheme by one");
    require(DFT::radial_point_count(8, DFT::GridLevel::Normal) > 0,
            "Normal grid should allocate radial points for oxygen");
}

void test_atomic_and_molecular_grid_generation()
{
    const Eigen::Vector3d center = Eigen::Vector3d::Zero();
    const Eigen::MatrixXd atomic_grid = DFT::MakeAtomicGrid(8, center, DFT::GridLevel::Normal);

    require(atomic_grid.rows() == DFT::atomic_point_count(8, DFT::GridLevel::Normal),
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

    const DFT::MolecularGrid mol_grid = DFT::MakeMolecularGrid(mol, DFT::GridLevel::Normal);
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

    const DFT::MolecularGrid h2_grid = DFT::MakeMolecularGrid(
        h2_z,
        h2_xyz,
        DFT::GridLevel::Coarse);

    require(h2_grid.points.rows() == 2 * DFT::atomic_point_count(1, DFT::GridLevel::Coarse),
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
    try
    {
        test_angular_grid_invariants();
        test_radial_grid_invariants();
        test_orca_like_grid_presets();
        test_atomic_and_molecular_grid_generation();
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
