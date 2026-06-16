#include "populations/population_detail.h"

namespace HartreeFock::SCF
{
    std::expected<PopulationAnalysis, std::string> lowdin_population_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density)
    {
        // Lowdin first moves the density into the symmetrically orthogonalized AO
        // basis. In that representation the AO populations are simply the diagonal
        // of S^(1/2) P S^(1/2), which is why this path differs only in the
        // transformation before the common atom-accumulation step.
        if (auto valid = detail::validate_population_inputs(
                molecule, basis, overlap, total_density, spin_density);
            !valid)
        {
            return std::unexpected("Löwdin population analysis " + valid.error());
        }

        auto atom_to_aos = detail::build_atom_ao_indices(molecule, basis);
        if (!atom_to_aos)
            return std::unexpected("Löwdin population analysis " + atom_to_aos.error());

        auto S_half = detail::symmetric_overlap_sqrt(overlap);
        if (!S_half)
            return std::unexpected("Löwdin population analysis " + S_half.error());

        const Eigen::MatrixXd lowdin_density =
            (*S_half) * total_density * (*S_half);
        const Eigen::VectorXd ao_population = lowdin_density.diagonal();

        Eigen::VectorXd ao_spin_population;
        if (spin_density != nullptr)
        {
            const Eigen::MatrixXd lowdin_spin_density =
                (*S_half) * (*spin_density) * (*S_half);
            ao_spin_population = lowdin_spin_density.diagonal();
        }

        return detail::accumulate_atomic_populations(
            molecule,
            *atom_to_aos,
            ao_population,
            (spin_density != nullptr) ? &ao_spin_population : nullptr);
    }
} // namespace HartreeFock::SCF
