#include "scf/population.h"

#include <format>

namespace HartreeFock::SCF
{
    namespace
    {
        bool matrix_has_shape(const Eigen::MatrixXd &matrix, Eigen::Index rows, Eigen::Index cols)
        {
            return matrix.rows() == rows && matrix.cols() == cols;
        }

        Eigen::VectorXd gross_ao_population(
            const Eigen::MatrixXd &density,
            const Eigen::MatrixXd &overlap)
        {
            // Mulliken AO gross population: q_mu = sum_nu P_mu,nu S_mu,nu.
            return (density.array() * overlap.array()).rowwise().sum();
        }
    } // namespace

    std::expected<PopulationAnalysis, std::string> mulliken_population_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density)
    {
        const Eigen::Index nbasis = static_cast<Eigen::Index>(basis.nbasis());
        if (nbasis == 0)
            return std::unexpected("Mulliken population analysis requires a non-empty AO basis");
        if (molecule.atomic_numbers.size() != static_cast<Eigen::Index>(molecule.natoms))
            return std::unexpected("Mulliken population analysis requires initialized molecular atom data");
        if (!matrix_has_shape(overlap, nbasis, nbasis))
            return std::unexpected(std::format(
                "overlap matrix has shape {}x{} but expected {}x{}",
                overlap.rows(), overlap.cols(), nbasis, nbasis));
        if (!matrix_has_shape(total_density, nbasis, nbasis))
            return std::unexpected(std::format(
                "density matrix has shape {}x{} but expected {}x{}",
                total_density.rows(), total_density.cols(), nbasis, nbasis));
        if (spin_density != nullptr && !matrix_has_shape(*spin_density, nbasis, nbasis))
            return std::unexpected(std::format(
                "spin-density matrix has shape {}x{} but expected {}x{}",
                spin_density->rows(), spin_density->cols(), nbasis, nbasis));

        PopulationAnalysis analysis;
        analysis.atoms.resize(molecule.natoms);
        analysis.ao_population = gross_ao_population(total_density, overlap);
        analysis.has_spin_population = (spin_density != nullptr);

        Eigen::VectorXd ao_spin_population;
        if (analysis.has_spin_population)
            ao_spin_population = gross_ao_population(*spin_density, overlap);

        for (std::size_t atom = 0; atom < molecule.natoms; ++atom)
        {
            analysis.atoms[atom].atom_index = atom;
            analysis.atoms[atom].atomic_number = molecule.atomic_numbers(static_cast<Eigen::Index>(atom));
            analysis.atoms[atom].net_charge = static_cast<double>(analysis.atoms[atom].atomic_number);
        }

        for (std::size_t mu = 0; mu < basis._basis_functions.size(); ++mu)
        {
            const ContractedView &bf = basis._basis_functions[mu];
            if (bf._shell == nullptr)
                return std::unexpected(std::format("basis function {} has no parent shell", mu));
            const std::size_t atom = static_cast<std::size_t>(bf._shell->_atom_index);
            if (atom >= analysis.atoms.size())
                return std::unexpected(std::format(
                    "basis function {} belongs to atom {} but molecule has only {} atom(s)",
                    mu + 1, atom + 1, analysis.atoms.size()));

            analysis.atoms[atom].electron_population += analysis.ao_population(static_cast<Eigen::Index>(mu));
            if (analysis.has_spin_population)
                analysis.atoms[atom].spin_population += ao_spin_population(static_cast<Eigen::Index>(mu));
        }

        analysis.total_electrons = analysis.ao_population.sum();
        analysis.total_charge = 0.0;
        analysis.total_spin_population = analysis.has_spin_population ? ao_spin_population.sum() : 0.0;

        for (auto &atom : analysis.atoms)
        {
            atom.net_charge -= atom.electron_population;
            analysis.total_charge += atom.net_charge;
        }

        return analysis;
    }
} // namespace HartreeFock::SCF
