#ifndef PLANCK_SOLVATION_PCM_H
#define PLANCK_SOLVATION_PCM_H

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Solvation
{
    struct PCMSurfacePoint
    {
        Eigen::Vector3d position = Eigen::Vector3d::Zero(); // Bohr
        double area = 0.0;                                  // Bohr^2
        std::size_t atom_index = 0;
    };

    struct PCMState
    {
        std::vector<PCMSurfacePoint> surface_points;
        std::vector<Eigen::MatrixXd> unit_potential_matrices;
        Eigen::VectorXd nuclear_potential;
        Eigen::MatrixXd influence_matrix;
        double dielectric_factor = 0.0;

        bool enabled() const noexcept
        {
            return !surface_points.empty();
        }
    };

    struct PCMResult
    {
        Eigen::VectorXd total_potential;
        Eigen::VectorXd apparent_charges;
        Eigen::MatrixXd reaction_potential;
        double solvation_energy = 0.0;
    };

    std::expected<PCMState, std::string> build_pcm_state(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    std::expected<PCMResult, std::string> evaluate_pcm_reaction_field(
        const HartreeFock::Calculator &calculator,
        const PCMState &state,
        const Eigen::MatrixXd &total_density);

} // namespace HartreeFock::Solvation

#endif // PLANCK_SOLVATION_PCM_H
