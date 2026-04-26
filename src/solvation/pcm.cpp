#include "pcm.h"

#include <cmath>
#include <format>
#include <numbers>

#include <Eigen/Cholesky>

#include "integrals/base.h"
#include "io/logging.h"
#include "lookup/elements.h"

namespace
{
    constexpr double ISWIG_DIAGONAL_SCALE = 1.07;

    std::vector<Eigen::Vector3d> fibonacci_sphere(int npoints)
    {
        std::vector<Eigen::Vector3d> points;
        points.reserve(static_cast<std::size_t>(npoints));

        const double golden_angle = std::numbers::pi * (3.0 - std::sqrt(5.0));
        for (int i = 0; i < npoints; ++i)
        {
            const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(npoints);
            const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = golden_angle * static_cast<double>(i);
            points.emplace_back(radial * std::cos(phi), radial * std::sin(phi), z);
        }

        return points;
    }
}

std::expected<HartreeFock::Solvation::PCMState, std::string>
HartreeFock::Solvation::build_pcm_state(
    const HartreeFock::Calculator &calculator,
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    PCMState state;
    if (calculator._solvation._model == HartreeFock::SolvationModel::None)
        return state;

    const int points_per_atom = calculator._solvation._surface_points_per_atom;
    if (points_per_atom < 6)
        return std::unexpected("PCM requires at least 6 surface points per atom");

    const auto sphere_directions = fibonacci_sphere(points_per_atom);
    std::vector<double> radii_bohr;
    radii_bohr.reserve(calculator._molecule.natoms);

    for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
    {
        const auto element = element_from_z(static_cast<std::uint64_t>(
            calculator._molecule.atomic_numbers(static_cast<Eigen::Index>(atom))));
        if (!element)
            return std::unexpected("PCM cavity setup failed: " + element.error());

        radii_bohr.push_back(
            calculator._solvation._cavity_scale *
            element->radius *
            ANGSTROM_TO_BOHR);
    }

    for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
    {
        const Eigen::Vector3d center = calculator._molecule._standard.row(static_cast<Eigen::Index>(atom));
        const double radius = radii_bohr[atom];
        const double area_per_point =
            4.0 * std::numbers::pi * radius * radius / static_cast<double>(points_per_atom);

        for (const auto &direction : sphere_directions)
        {
            const Eigen::Vector3d point = center + radius * direction;
            bool buried = false;
            for (std::size_t other = 0; other < calculator._molecule.natoms; ++other)
            {
                if (other == atom)
                    continue;

                const Eigen::Vector3d other_center = calculator._molecule._standard.row(static_cast<Eigen::Index>(other));
                const double other_radius = radii_bohr[other];
                if ((point - other_center).squaredNorm() < other_radius * other_radius)
                {
                    buried = true;
                    break;
                }
            }

            if (!buried)
            {
                state.surface_points.push_back(
                    PCMSurfacePoint{
                        .position = point,
                        .area = area_per_point,
                        .atom_index = atom});
            }
        }
    }

    if (state.surface_points.empty())
        return std::unexpected("PCM cavity generation failed: all surface points were buried");

    const Eigen::Index npoints = static_cast<Eigen::Index>(state.surface_points.size());
    state.nuclear_potential = Eigen::VectorXd::Zero(npoints);
    state.influence_matrix = Eigen::MatrixXd::Zero(npoints, npoints);
    state.unit_potential_matrices.reserve(static_cast<std::size_t>(npoints));

    for (Eigen::Index i = 0; i < npoints; ++i)
    {
        const auto &site = state.surface_points[static_cast<std::size_t>(i)];
        for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
        {
            const Eigen::Vector3d nucleus = calculator._molecule._standard.row(static_cast<Eigen::Index>(atom));
            const double charge = static_cast<double>(calculator._molecule.atomic_numbers(static_cast<Eigen::Index>(atom)));
            state.nuclear_potential(i) += charge / (site.position - nucleus).norm();
        }

        std::vector<HartreeFock::ExternalCharge> unit_charge{
            HartreeFock::ExternalCharge{.position = site.position, .charge = 1.0}};
        state.unit_potential_matrices.push_back(
            _compute_external_charge_attraction(
                shell_pairs,
                calculator._shells.nbasis(),
                unit_charge,
                calculator._integral._engine,
                nullptr));

        for (Eigen::Index j = 0; j < npoints; ++j)
        {
            if (i == j)
            {
                state.influence_matrix(i, j) =
                    ISWIG_DIAGONAL_SCALE *
                    std::sqrt(4.0 * std::numbers::pi / site.area);
            }
            else
            {
                const double distance =
                    (site.position - state.surface_points[static_cast<std::size_t>(j)].position).norm();
                state.influence_matrix(i, j) = 1.0 / distance;
            }
        }
    }

    const double epsilon = calculator._solvation._dielectric;
    state.dielectric_factor = (epsilon - 1.0) / (epsilon + 0.5);

    HartreeFock::Logger::logging(
        HartreeFock::LogLevel::Info,
        "PCM :",
        std::format(
            "C-PCM cavity with {} surface points, epsilon = {:.4f}, scale = {:.3f}",
            state.surface_points.size(),
            epsilon,
            calculator._solvation._cavity_scale));
    if (!calculator._solvation._solvent.empty())
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "PCM Solvent :",
            calculator._solvation._solvent);
    }
    HartreeFock::Logger::blank();

    return state;
}

std::expected<HartreeFock::Solvation::PCMResult, std::string>
HartreeFock::Solvation::evaluate_pcm_reaction_field(
    const HartreeFock::Calculator &calculator,
    const PCMState &state,
    const Eigen::MatrixXd &total_density)
{
    PCMResult result;
    if (!state.enabled())
    {
        result.reaction_potential =
            Eigen::MatrixXd::Zero(calculator._shells.nbasis(), calculator._shells.nbasis());
        return result;
    }

    const Eigen::Index npoints = static_cast<Eigen::Index>(state.surface_points.size());
    result.total_potential = state.nuclear_potential;

    for (Eigen::Index i = 0; i < npoints; ++i)
        result.total_potential(i) +=
            (total_density.array() * state.unit_potential_matrices[static_cast<std::size_t>(i)].array()).sum();

    Eigen::LDLT<Eigen::MatrixXd> solver(state.influence_matrix);
    if (solver.info() != Eigen::Success)
        return std::unexpected("PCM surface linear system factorization failed");

    result.apparent_charges =
        solver.solve(-state.dielectric_factor * result.total_potential);
    if (solver.info() != Eigen::Success)
        return std::unexpected("PCM surface linear system solve failed");

    result.reaction_potential =
        Eigen::MatrixXd::Zero(calculator._shells.nbasis(), calculator._shells.nbasis());
    for (Eigen::Index i = 0; i < npoints; ++i)
        result.reaction_potential +=
            result.apparent_charges(i) * state.unit_potential_matrices[static_cast<std::size_t>(i)];

    result.solvation_energy = 0.5 * result.apparent_charges.dot(result.total_potential);
    return result;
}
