#include "populations/population_detail.h"

namespace HartreeFock::SCF
{
    std::expected<PopulationAnalysis, std::string> mulliken_population_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density)
    {
        // Mulliken uses the original non-orthogonal AO basis, so after input
        // validation the only method-specific work is forming the AO gross
        // populations from P and S.
        if (auto valid = detail::validate_population_inputs(
                molecule, basis, overlap, total_density, spin_density);
            !valid)
        {
            return std::unexpected("Mulliken population analysis " + valid.error());
        }

        auto atom_to_aos = detail::build_atom_ao_indices(molecule, basis);
        if (!atom_to_aos)
            return std::unexpected("Mulliken population analysis " + atom_to_aos.error());

        const Eigen::VectorXd ao_population = detail::gross_ao_population(total_density, overlap);
        Eigen::VectorXd ao_spin_population;
        if (spin_density != nullptr)
            ao_spin_population = detail::gross_ao_population(*spin_density, overlap);

        return detail::accumulate_atomic_populations(
            molecule,
            *atom_to_aos,
            ao_population,
            (spin_density != nullptr) ? &ao_spin_population : nullptr);
    }
} // namespace HartreeFock::SCF
