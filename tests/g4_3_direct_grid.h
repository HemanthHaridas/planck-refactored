#ifndef PLANCK_TESTS_G4_3_DIRECT_GRID_H
#define PLANCK_TESTS_G4_3_DIRECT_GRID_H

// Test-only direct realizations of retired and candidate G4.3 grid rules.
// These are not GridPruningPolicy entries: they have no production resolver
// spelling or default.  Each literal mode keeps an independently selected
// rule reproducible for atlas reconciliation before qualification.
#include "dft/base/grid.h"

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace G43Direct
{
    struct Rule
    {
        std::array<double, 4> boundaries;
        std::array<int, 5> lebedev_points;
    };

    enum class Candidate { G437Finalist, G439SecondRowFloorRevision, G439DualEnergyFloorRevision,
                           G43V3ConstrainedCandidate, G43V4RobustCandidate };

    inline const Rule &rule_for_atomic_number(int z,Candidate candidate=Candidate::G437Finalist)
    {
        static const Rule retired_hydrogen_helium{{.4, 1., 2.5, 3.5}, {26, 110, 1454, 770, 434}};
        static const Rule dual_floor_hydrogen_helium{{.4, 1., 2.5, 3.5}, {26, 110, 1454, 770, 590}};
        static const Rule v3_hydrogen_helium{{.4, 1., 3.5, 4.5}, {26, 146, 1454, 434, 434}};
        static const Rule v4_hydrogen_helium{{.25, 1., 2.5, 3.5}, {26, 146, 2030, 974, 590}};
        static const Rule retired_second_row{{.25, 1., 2.5, 4.5}, {50, 434, 1454, 590, 170}};
        static const Rule revised_second_row{{.4, 1., 2.5, 4.5}, {86, 770, 1454, 590, 170}};
        static const Rule dual_floor_second_row{{.4, 1., 2.5, 4.5}, {86, 770, 1730, 590, 170}};
        static const Rule v3_second_row{{.4, 1., 2.5, 4.5}, {110, 434, 1454, 590, 170}};
        static const Rule v4_second_row{{.4, 1., 2.5, 4.5}, {110, 770, 2030, 590, 170}};
        static const Rule third_row{{.1, .5, 3.5, 4.5}, {26, 146, 770, 230, 26}};
        static const Rule v4_third_row{{.4, .8, 2.5, 4.5}, {86, 590, 974, 590, 26}};
        if (z >= 1 && z <= 2)
            return candidate==Candidate::G439DualEnergyFloorRevision?dual_floor_hydrogen_helium:
                candidate==Candidate::G43V4RobustCandidate?v4_hydrogen_helium:
                candidate==Candidate::G43V3ConstrainedCandidate?v3_hydrogen_helium:retired_hydrogen_helium;
        if (z >= 3 && z <= 10)
            return candidate==Candidate::G437Finalist?retired_second_row:
                candidate==Candidate::G439SecondRowFloorRevision?revised_second_row:
                candidate==Candidate::G439DualEnergyFloorRevision?dual_floor_second_row:
                candidate==Candidate::G43V4RobustCandidate?v4_second_row:v3_second_row;
        if (z >= 11 && z <= 18)
            return candidate==Candidate::G43V4RobustCandidate?v4_third_row:third_row;
        throw std::runtime_error("G4.3 direct grid supports atomic numbers 1 through 18 only");
    }

    inline int band(double u, const Rule &rule)
    {
        int result = 0;
        for (const double boundary : rule.boundaries)
            result += u >= boundary;
        return result;
    }

    struct Shell
    {
        int atom;
        int atomic_number;
        int radial_index;
        double radius_bohr;
        double u;
        int lebedev_points;
        Eigen::Index begin;
        Eigen::Index end;
    };

    struct Grid
    {
        DFT::MolecularGrid values;
        std::vector<Shell> shells;
    };

    inline constexpr int radial_points = 135;

    inline std::string fingerprint(Candidate candidate=Candidate::G437Finalist)
    {
        if(candidate==Candidate::G437Finalist)
            return "g4-3-l1-finalist-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg";
        if(candidate==Candidate::G439SecondRowFloorRevision)
            return "g4-3-second-row-band1-floor-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg";
        if(candidate==Candidate::G439DualEnergyFloorRevision)
            return "g4-3-dual-energy-floor-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg";
        if(candidate==Candidate::G43V4RobustCandidate)
            return "g4-3-v4-robust-development-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg";
        return "g4-3-v3-constrained-selection-v1;radial=135;radius=m4-over-bragg;partition=original-becke-treutler-bragg";
    }

    inline Grid make(const HartreeFock::Molecule &molecule,Candidate candidate=Candidate::G437Finalist)
    {
        const auto coordinates = DFT::detail::coordinates_bohr(molecule);
        if (!coordinates) throw std::runtime_error(coordinates.error());
        const Eigen::Index natoms = molecule.atomic_numbers.size();
        if (coordinates->rows() != natoms || coordinates->cols() != 3)
            throw std::runtime_error("G4.3 direct grid has invalid molecular coordinates");

        std::vector<Eigen::MatrixXd> atomic_grids(static_cast<std::size_t>(natoms));
        std::vector<Shell> shells;
        Eigen::Index total_points = 0;
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
        {
            const int z = molecule.atomic_numbers(atom);
            const Rule &rule = rule_for_atomic_number(z,candidate);
            const Eigen::MatrixXd radial = DFT::MakeTreutlerAhlrichsGrid(radial_points, DFT::treutler_xi(z));
            std::map<int, Eigen::MatrixXd> angular;
            std::vector<int> counts(static_cast<std::size_t>(radial_points));
            Eigen::Index atom_points = 0;
            for (int shell = 0; shell < radial_points; ++shell)
            {
                const int count = rule.lebedev_points[band(radial(shell, 0) / DFT::bragg_radius(z), rule)];
                counts[static_cast<std::size_t>(shell)] = count;
                if (!angular.contains(count))
                {
                    auto values = DFT::MakeLebedevGrid(count);
                    if (!values) throw std::runtime_error(values.error());
                    angular.emplace(count, std::move(*values));
                }
                atom_points += count;
            }
            atomic_grids[static_cast<std::size_t>(atom)].resize(atom_points, 4);
            Eigen::Index row = 0;
            for (int shell = 0; shell < radial_points; ++shell)
            {
                const double radius = radial(shell, 0);
                const Eigen::MatrixXd &sphere = angular.at(counts[static_cast<std::size_t>(shell)]);
                for (Eigen::Index point = 0; point < sphere.rows(); ++point, ++row)
                {
                    atomic_grids[static_cast<std::size_t>(atom)].row(row).head<3>() =
                        (coordinates->row(atom).transpose() + radius * sphere.row(point).head<3>().transpose()).transpose();
                    atomic_grids[static_cast<std::size_t>(atom)](row, 3) = radial(shell, 1) * sphere(point, 3);
                }
                shells.push_back({static_cast<int>(atom), z, shell, radius, radius / DFT::bragg_radius(z),
                                  counts[static_cast<std::size_t>(shell)], total_points + row - sphere.rows(),
                                  total_points + row});
            }
            total_points += atom_points;
        }

        DFT::MolecularGrid result;
        result.points.resize(total_points, 4);
        result.owner.resize(total_points);
        result.atomic_weights.resize(total_points);
        result.partition_weights.resize(total_points);
        Eigen::Index row = 0;
        for (Eigen::Index atom = 0; atom < natoms; ++atom)
            for (Eigen::Index point = 0; point < atomic_grids[static_cast<std::size_t>(atom)].rows(); ++point, ++row)
            {
                const Eigen::Vector3d xyz = atomic_grids[static_cast<std::size_t>(atom)].row(point).head<3>().transpose();
                const auto partition = DFT::detail::becke_partition_weight(static_cast<int>(atom), xyz,
                    molecule.atomic_numbers, *coordinates);
                if (!partition) throw std::runtime_error(partition.error());
                result.points.row(row).head<3>() = xyz.transpose();
                result.points(row, 3) = atomic_grids[static_cast<std::size_t>(atom)](point, 3) * *partition;
                result.owner(row) = static_cast<int>(atom);
                result.atomic_weights(row) = atomic_grids[static_cast<std::size_t>(atom)](point, 3);
                result.partition_weights(row) = *partition;
            }
        return {std::move(result), std::move(shells)};
    }
}

#endif
