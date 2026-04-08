#ifndef DFT_GRID_H
#define DFT_GRID_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
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
        Eigen::MatrixXd points; // N x 4 -> x, y, z, weight  (Bohr / quadrature weight)
        Eigen::VectorXi owner;  // Generating atom index for each point
    };

    namespace detail
    {

        inline double xc_intacc_for_scheme(int angular_scheme)
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
                throw std::invalid_argument(
                    "xc_intacc_for_scheme: unsupported angular scheme " +
                    std::to_string(angular_scheme));
            }
        }

        inline std::array<int, 5> angular_shells_for_scheme(int angular_scheme)
        {
            switch (angular_scheme)
            {
            case 1:
                return {14, 26, 50, 50, 26};
            case 2:
                return {14, 26, 50, 110, 50};
            case 3:
                return {26, 50, 110, 194, 110};
            case 4:
                return {26, 110, 194, 302, 194};
            case 5:
                return {26, 194, 302, 434, 302};
            case 6:
                return {50, 302, 434, 590, 434};
            case 7:
                return {110, 434, 590, 770, 590};
            default:
                throw std::invalid_argument(
                    "angular_shells_for_scheme: unsupported angular scheme " +
                    std::to_string(angular_scheme));
            }
        }

        inline int periodic_row(int Z)
        {
            if (Z <= 0)
                throw std::invalid_argument(
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

        inline int radial_point_count(int Z, const GridPreset &preset)
        {
            const int scheme = effective_angular_scheme(Z, preset);
            const double count = (15.0 * xc_intacc_for_scheme(scheme) - 40.0) + static_cast<double>(preset.radial_row_factor * periodic_row(Z));
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

        inline Eigen::MatrixXd coordinates_bohr(const HartreeFock::Molecule &molecule)
        {
            if (static_cast<std::size_t>(molecule._coordinates.rows()) == molecule.natoms)
                return molecule._coordinates;

            if (static_cast<std::size_t>(molecule.coordinates.rows()) != molecule.natoms)
                throw std::invalid_argument(
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

        inline double becke_partition_weight(
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
                throw std::runtime_error("becke_partition_weight: partition weights underflowed");

            return weights[static_cast<std::size_t>(atom_index)] / sum;
        }

    } // namespace detail

    inline GridPreset grid_preset(GridLevel level)
    {
        switch (level)
        {
        case GridLevel::Coarse:
            return {level, 3, detail::xc_intacc_for_scheme(3), 5, true};
        case GridLevel::Normal:
            return {level, 4, detail::xc_intacc_for_scheme(4), 5, true};
        case GridLevel::Fine:
            return {level, 5, detail::xc_intacc_for_scheme(5), 5, true};
        case GridLevel::UltraFine:
            return {level, 6, detail::xc_intacc_for_scheme(6), 5, true};
        }

        throw std::invalid_argument("grid_preset: unsupported grid level");
    }

    inline int angular_scheme(GridLevel level)
    {
        return grid_preset(level).angular_scheme;
    }

    inline int effective_angular_scheme(int Z, GridLevel level)
    {
        return detail::effective_angular_scheme(Z, grid_preset(level));
    }

    inline int radial_point_count(int Z, GridLevel level)
    {
        return detail::radial_point_count(Z, grid_preset(level));
    }

    inline std::array<int, 5> angular_shell_sizes(int Z, GridLevel level)
    {
        return detail::angular_shells_for_scheme(effective_angular_scheme(Z, level));
    }

    inline int atomic_point_count(int Z, GridLevel level)
    {
        const int nr = radial_point_count(Z, level);
        const auto shells = angular_shell_sizes(Z, level);

        int total = 0;
        for (int ir = 0; ir < nr; ++ir)
            total += shells[detail::pruning_region(ir, nr)];
        return total;
    }

    inline Eigen::MatrixXd MakeAtomicGrid(
        int Z,
        const Eigen::Vector3d &center,
        GridLevel level)
    {
        if (Z <= 0)
            throw std::invalid_argument(
                "MakeAtomicGrid: atomic number must be positive, got " + std::to_string(Z));

        const int nr = radial_point_count(Z, level);
        const auto shells = angular_shell_sizes(Z, level);
        const Eigen::MatrixXd radial = MakeTreutlerAhlrichsGrid(nr, treutler_radius(Z));

        std::array<Eigen::MatrixXd, 5> angular_cache = {
            MakeLebedevGrid(shells[0]),
            MakeLebedevGrid(shells[1]),
            MakeLebedevGrid(shells[2]),
            MakeLebedevGrid(shells[3]),
            MakeLebedevGrid(shells[4]),
        };

        Eigen::MatrixXd grid(atomic_point_count(Z, level), 4);
        Eigen::Index row = 0;

        for (int ir = 0; ir < nr; ++ir)
        {
            const double radius = radial(ir, 0);
            const double wr = radial(ir, 1);
            const Eigen::MatrixXd &angular = angular_cache[detail::pruning_region(ir, nr)];

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

    inline MolecularGrid MakeMolecularGrid(
        const Eigen::VectorXi &atomic_numbers,
        const Eigen::MatrixXd &coordinates_bohr,
        GridLevel level)
    {
        const Eigen::Index natoms = atomic_numbers.size();
        if (coordinates_bohr.rows() != natoms || coordinates_bohr.cols() != 3)
            throw std::invalid_argument(
                "MakeMolecularGrid: coordinates must have shape natoms x 3");

        std::vector<Eigen::MatrixXd> atomic_grids(static_cast<std::size_t>(natoms));
        Eigen::Index total_points = 0;

        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            atomic_grids[static_cast<std::size_t>(atom)] = MakeAtomicGrid(
                atomic_numbers(atom),
                coordinates_bohr.row(atom).transpose(),
                level);
            total_points += atomic_grids[static_cast<std::size_t>(atom)].rows();
        }

        MolecularGrid result;
        result.points.resize(total_points, 4);
        result.owner.resize(total_points);

        Eigen::Index row = 0;
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            const Eigen::MatrixXd &atomic_grid = atomic_grids[static_cast<std::size_t>(atom)];
            for (Eigen::Index i = 0; i < atomic_grid.rows(); ++i, ++row)
            {
                const Eigen::Vector3d point = atomic_grid.row(i).head<3>().transpose();
                const double partition = detail::becke_partition_weight(
                    static_cast<int>(atom),
                    point,
                    atomic_numbers,
                    coordinates_bohr);

                result.points.row(row).head<3>() = point.transpose();
                result.points(row, 3) = atomic_grid(i, 3) * partition;
                result.owner(row) = static_cast<int>(atom);
            }
        }

        return result;
    }

    inline MolecularGrid MakeMolecularGrid(
        const HartreeFock::Molecule &molecule,
        GridLevel level)
    {
        return MakeMolecularGrid(
            molecule.atomic_numbers,
            detail::coordinates_bohr(molecule),
            level);
    }

} // namespace DFT

#endif // DFT_GRID_H
