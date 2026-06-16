#ifndef DFT_GRID_H
#define DFT_GRID_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "angular.h"
#include "base/types.h"
#include "radial.h"

namespace DFT
{

    // ORCA-inspired XC grid quality presets.
    //
    // The angular scheme and IntAcc values follow the ORCA 6 manual tables for
    // AngularGrid / XC IntAcc. The radial rule here still uses the existing
    // Treutler-Ahlrichs M4 mapping from radial.h, and the five-region pruning
    // cutoffs are heuristic but ORCA-like.
    enum class GridLevel
    {
        Coarse,
        Normal,
        Fine,
        UltraFine
    };

    struct GridPreset
    {
        GridLevel level;
        int angular_scheme;
        double int_acc;
        int radial_row_factor;
        bool reduce_light_atoms;
    };

    struct MolecularGrid
    {
        Eigen::MatrixXd points;            // N x 4 -> x, y, z, weight  (Bohr / quadrature weight)
        Eigen::VectorXi owner;             // Generating atom index for each point
        Eigen::VectorXd atomic_weights;    // Unpartitioned atomic-grid weights
        Eigen::VectorXd partition_weights; // Becke partition weight of owner atom
    };

    namespace detail
    {

        inline std::expected<double, std::string> xc_intacc_for_scheme(int angular_scheme)
        {
            switch (angular_scheme)
            {
            case 1:
                return 4.004;
            case 2:
                return 4.004;
            case 3:
                return 4.159;
            case 4:
                return 4.388;
            case 5:
                return 4.629;
            case 6:
                return 4.959;
            case 7:
                return 4.959;
            default:
                return std::unexpected(
                    "xc_intacc_for_scheme: unsupported angular scheme " +
                    std::to_string(angular_scheme));
            }
        }

        inline std::expected<std::array<int, 5>, std::string> angular_shells_for_scheme(int angular_scheme)
        {
            switch (angular_scheme)
            {
            case 1:
                return std::array<int, 5>{14, 26, 50, 50, 26};
            case 2:
                return std::array<int, 5>{14, 26, 50, 110, 50};
            case 3:
                return std::array<int, 5>{26, 50, 110, 194, 110};
            case 4:
                return std::array<int, 5>{26, 110, 194, 302, 194};
            case 5:
                return std::array<int, 5>{26, 194, 302, 434, 302};
            case 6:
                return std::array<int, 5>{50, 302, 434, 590, 434};
            case 7:
                return std::array<int, 5>{110, 434, 590, 770, 590};
            default:
                return std::unexpected(
                    "angular_shells_for_scheme: unsupported angular scheme " +
                    std::to_string(angular_scheme));
            }
        }

        inline std::expected<int, std::string> periodic_row(int Z)
        {
            if (Z <= 0)
                return std::unexpected(
                    "periodic_row: atomic number must be positive, got " + std::to_string(Z));
            if (Z <= 2)
                return 1;
            if (Z <= 10)
                return 2;
            if (Z <= 18)
                return 3;
            if (Z <= 36)
                return 4;
            if (Z <= 54)
                return 5;
            if (Z <= 86)
                return 6;
            return 7;
        }

        inline int effective_angular_scheme(int Z, const GridPreset &preset)
        {
            int scheme = preset.angular_scheme;
            if (preset.reduce_light_atoms && Z <= 2 && scheme > 1)
                --scheme;
            return scheme;
        }

        inline std::expected<int, std::string> radial_point_count(int Z, const GridPreset &preset)
        {
            const int scheme = effective_angular_scheme(Z, preset);
            const auto int_acc = xc_intacc_for_scheme(scheme);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            const auto row = periodic_row(Z);
            if (!row)
                return std::unexpected(row.error());
            const double count = (15.0 * (*int_acc) - 40.0) + static_cast<double>(preset.radial_row_factor * (*row));
            return std::max(1, static_cast<int>(std::lround(count)));
        }

        // ORCA-like five-region pruning. The exact ORCA cutoffs are not published
        // in the manual, so we use a symmetric shell-fraction heuristic.
        inline int pruning_region(int radial_index, int radial_count)
        {
            if (radial_count <= 1)
                return 2;

            const double t = static_cast<double>(radial_index) / static_cast<double>(radial_count - 1);
            if (t < 0.15)
                return 0;
            if (t < 0.40)
                return 1;
            if (t < 0.70)
                return 2;
            if (t < 0.90)
                return 3;
            return 4;
        }

        inline std::expected<Eigen::MatrixXd, std::string> coordinates_bohr(const HartreeFock::Molecule &molecule)
        {
            if (static_cast<std::size_t>(molecule._coordinates.rows()) == molecule.natoms)
                return molecule._coordinates;

            if (static_cast<std::size_t>(molecule.coordinates.rows()) != molecule.natoms)
                return std::unexpected(
                    "coordinates_bohr: molecule coordinates are not initialized");

            if (molecule._is_bohr)
                return molecule.coordinates;

            return molecule.coordinates * ANGSTROM_TO_BOHR;
        }

        inline double treutler_becke_adjustment(int Zi, int Zj)
        {
            const double ri = treutler_radius(Zi);
            const double rj = treutler_radius(Zj);
            const double denom = ri + rj;
            if (denom <= 0.0)
                return 0.0;

            const double u = (ri - rj) / denom;
            if (std::abs(u) < 1e-14)
                return 0.0;

            const double raw = u / (u * u - 1.0);
            return std::clamp(raw, -0.5, 0.5);
        }

        inline double becke_switch(double mu)
        {
            for (int k = 0; k < 3; ++k)
                mu = 1.5 * mu - 0.5 * mu * mu * mu;
            return 0.5 * (1.0 - mu);
        }

        inline double becke_switch_derivative(double mu)
        {
            double derivative = 1.0;
            for (int k = 0; k < 3; ++k)
            {
                derivative *= 1.5 * (1.0 - mu * mu);
                mu = 1.5 * mu - 0.5 * mu * mu * mu;
            }
            return -0.5 * derivative;
        }

        inline double pair_partition(
            const Eigen::Vector3d &point,
            const Eigen::Vector3d &ri,
            const Eigen::Vector3d &rj,
            int Zi,
            int Zj)
        {
            const Eigen::Vector3d rij = ri - rj;
            const double Rij = rij.norm();
            if (Rij < 1e-14)
                return 0.5;

            double mu = (point - ri).norm() - (point - rj).norm();
            mu /= Rij;

            const double aij = treutler_becke_adjustment(Zi, Zj);
            mu += aij * (1.0 - mu * mu);
            mu = std::clamp(mu, -1.0, 1.0);

            return becke_switch(mu);
        }

        inline std::expected<double, std::string> becke_partition_weight(
            int atom_index,
            const Eigen::Vector3d &point,
            const Eigen::VectorXi &atomic_numbers,
            const Eigen::MatrixXd &coordinates)
        {
            const Eigen::Index natoms = atomic_numbers.size();
            std::vector<double> weights(static_cast<std::size_t>(natoms), 1.0);

            for (Eigen::Index i = 0; i < natoms; ++i)
            {
                const Eigen::Vector3d ri = coordinates.row(i).transpose();
                for (Eigen::Index j = i + 1; j < natoms; ++j)
                {
                    const Eigen::Vector3d rj = coordinates.row(j).transpose();
                    const double sij = pair_partition(
                        point,
                        ri,
                        rj,
                        atomic_numbers(i),
                        atomic_numbers(j));
                    weights[static_cast<std::size_t>(i)] *= sij;
                    weights[static_cast<std::size_t>(j)] *= (1.0 - sij);
                }
            }

            double sum = 0.0;
            for (double w : weights)
                sum += w;

            if (sum <= 0.0)
                return std::unexpected("becke_partition_weight: partition weights underflowed");

            return weights[static_cast<std::size_t>(atom_index)] / sum;
        }

    } // namespace detail

    inline std::expected<GridPreset, std::string> grid_preset(GridLevel level)
    {
        switch (level)
        {
        case GridLevel::Coarse:
        {
            auto int_acc = detail::xc_intacc_for_scheme(3);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 3, *int_acc, 5, true};
        }
        case GridLevel::Normal:
        {
            auto int_acc = detail::xc_intacc_for_scheme(4);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 4, *int_acc, 5, true};
        }
        case GridLevel::Fine:
        {
            auto int_acc = detail::xc_intacc_for_scheme(5);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 5, *int_acc, 5, true};
        }
        case GridLevel::UltraFine:
        {
            auto int_acc = detail::xc_intacc_for_scheme(6);
            if (!int_acc)
                return std::unexpected(int_acc.error());
            return GridPreset{level, 6, *int_acc, 5, true};
        }
        }

        return std::unexpected("grid_preset: unsupported grid level");
    }

    inline std::expected<int, std::string> angular_scheme(GridLevel level)
    {
        auto preset = grid_preset(level);
        if (!preset)
            return std::unexpected(preset.error());
        return preset->angular_scheme;
    }

    inline std::expected<int, std::string> effective_angular_scheme(int Z, GridLevel level)
    {
        auto preset = grid_preset(level);
        if (!preset)
            return std::unexpected(preset.error());
        return detail::effective_angular_scheme(Z, *preset);
    }

    inline std::expected<int, std::string> radial_point_count(int Z, GridLevel level)
    {
        auto preset = grid_preset(level);
        if (!preset)
            return std::unexpected(preset.error());
        return detail::radial_point_count(Z, *preset);
    }

    inline std::expected<std::array<int, 5>, std::string> angular_shell_sizes(int Z, GridLevel level)
    {
        auto scheme = effective_angular_scheme(Z, level);
        if (!scheme)
            return std::unexpected(scheme.error());
        return detail::angular_shells_for_scheme(*scheme);
    }

    inline std::expected<int, std::string> atomic_point_count(int Z, GridLevel level)
    {
        const auto nr = radial_point_count(Z, level);
        if (!nr)
            return std::unexpected(nr.error());
        const auto shells = angular_shell_sizes(Z, level);
        if (!shells)
            return std::unexpected(shells.error());

        int total = 0;
        for (int ir = 0; ir < *nr; ++ir)
            total += (*shells)[detail::pruning_region(ir, *nr)];
        return total;
    }

    inline std::expected<Eigen::MatrixXd, std::string> MakeAtomicGrid(
        int Z,
        const Eigen::Vector3d &center,
        GridLevel level)
    {
        if (Z <= 0)
            return std::unexpected(
                "MakeAtomicGrid: atomic number must be positive, got " + std::to_string(Z));

        const auto nr = radial_point_count(Z, level);
        if (!nr)
            return std::unexpected(nr.error());
        const auto shells = angular_shell_sizes(Z, level);
        if (!shells)
            return std::unexpected(shells.error());
        const Eigen::MatrixXd radial = MakeTreutlerAhlrichsGrid(*nr, treutler_radius(Z));

        std::array<Eigen::MatrixXd, 5> angular_cache;
        for (std::size_t shell_index = 0; shell_index < angular_cache.size(); ++shell_index)
        {
            auto angular_grid = MakeLebedevGrid((*shells)[shell_index]);
            if (!angular_grid)
                return std::unexpected(angular_grid.error());
            angular_cache[shell_index] = std::move(*angular_grid);
        }

        const auto total_points = atomic_point_count(Z, level);
        if (!total_points)
            return std::unexpected(total_points.error());
        Eigen::MatrixXd grid(*total_points, 4);
        Eigen::Index row = 0;

        for (int ir = 0; ir < *nr; ++ir)
        {
            const double radius = radial(ir, 0);
            const double wr = radial(ir, 1);
            const Eigen::MatrixXd &angular = angular_cache[detail::pruning_region(ir, *nr)];

            for (Eigen::Index ia = 0; ia < angular.rows(); ++ia, ++row)
            {
                const Eigen::Vector3d direction = angular.row(ia).head<3>().transpose();
                const Eigen::Vector3d point = center + radius * direction;

                grid(row, 0) = point.x();
                grid(row, 1) = point.y();
                grid(row, 2) = point.z();
                grid(row, 3) = wr * angular(ia, 3);
            }
        }

        return grid;
    }

    inline std::expected<MolecularGrid, std::string> MakeMolecularGrid(
        const Eigen::VectorXi &atomic_numbers,
        const Eigen::MatrixXd &coordinates_bohr,
        GridLevel level)
    {
        const Eigen::Index natoms = atomic_numbers.size();
        if (coordinates_bohr.rows() != natoms || coordinates_bohr.cols() != 3)
            return std::unexpected(
                "MakeMolecularGrid: coordinates must have shape natoms x 3");

        std::vector<Eigen::MatrixXd> atomic_grids(static_cast<std::size_t>(natoms));
        Eigen::Index total_points = 0;

        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            auto atomic_grid = MakeAtomicGrid(
                atomic_numbers(atom),
                coordinates_bohr.row(atom).transpose(),
                level);
            if (!atomic_grid)
                return std::unexpected(atomic_grid.error());
            atomic_grids[static_cast<std::size_t>(atom)] = std::move(*atomic_grid);
            total_points += atomic_grids[static_cast<std::size_t>(atom)].rows();
        }

        MolecularGrid result;
        result.points.resize(total_points, 4);
        result.owner.resize(total_points);
        result.atomic_weights.resize(total_points);
        result.partition_weights.resize(total_points);

        Eigen::Index row = 0;
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            const Eigen::MatrixXd &atomic_grid = atomic_grids[static_cast<std::size_t>(atom)];
            for (Eigen::Index i = 0; i < atomic_grid.rows(); ++i, ++row)
            {
                const Eigen::Vector3d point = atomic_grid.row(i).head<3>().transpose();
                const auto partition = detail::becke_partition_weight(
                    static_cast<int>(atom),
                    point,
                    atomic_numbers,
                    coordinates_bohr);
                if (!partition)
                    return std::unexpected(partition.error());

                result.points.row(row).head<3>() = point.transpose();
                result.points(row, 3) = atomic_grid(i, 3) * (*partition);
                result.owner(row) = static_cast<int>(atom);
                result.atomic_weights(row) = atomic_grid(i, 3);
                result.partition_weights(row) = *partition;
            }
        }

        return result;
    }

    inline std::expected<MolecularGrid, std::string> MakeMolecularGrid(
        const HartreeFock::Molecule &molecule,
        GridLevel level)
    {
        auto coords = detail::coordinates_bohr(molecule);
        if (!coords)
            return std::unexpected(coords.error());
        return MakeMolecularGrid(
            molecule.atomic_numbers,
            *coords,
            level);
    }

} // namespace DFT

#endif // DFT_GRID_H
