#include "scf/population.h"

#include <Eigen/Eigenvalues>

#include <format>

namespace HartreeFock::SCF
{
    namespace
    {
        bool matrix_has_shape(const Eigen::MatrixXd &matrix, Eigen::Index rows, Eigen::Index cols)
        {
            return matrix.rows() == rows && matrix.cols() == cols;
        }

        std::expected<void, std::string> validate_population_inputs(
            const Molecule &molecule,
            const Basis &basis,
            const Eigen::MatrixXd &overlap,
            const Eigen::MatrixXd &total_density,
            const Eigen::MatrixXd *spin_density)
        {
            const Eigen::Index nbasis = static_cast<Eigen::Index>(basis.nbasis());
            if (nbasis == 0)
                return std::unexpected("population analysis requires a non-empty AO basis");
            if (molecule.atomic_numbers.size() != static_cast<Eigen::Index>(molecule.natoms))
                return std::unexpected("population analysis requires initialized molecular atom data");
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
            return {};
        }

        std::expected<std::vector<std::vector<int>>, std::string> build_atom_ao_indices(
            const Molecule &molecule,
            const Basis &basis)
        {
            std::vector<std::vector<int>> atom_to_aos(molecule.natoms);
            for (std::size_t mu = 0; mu < basis._basis_functions.size(); ++mu)
            {
                const ContractedView &bf = basis._basis_functions[mu];
                if (bf._shell == nullptr)
                    return std::unexpected(std::format("basis function {} has no parent shell", mu));
                const std::size_t atom = static_cast<std::size_t>(bf._shell->_atom_index);
                if (atom >= molecule.natoms)
                    return std::unexpected(std::format(
                        "basis function {} belongs to atom {} but molecule has only {} atom(s)",
                        mu + 1, atom + 1, molecule.natoms));
                atom_to_aos[atom].push_back(static_cast<int>(mu));
            }
            return atom_to_aos;
        }

        Eigen::VectorXd gross_ao_population(
            const Eigen::MatrixXd &density,
            const Eigen::MatrixXd &overlap)
        {
            // Mulliken AO gross population: q_mu = sum_nu P_mu,nu S_mu,nu.
            return (density.array() * overlap.array()).rowwise().sum();
        }

        std::expected<Eigen::MatrixXd, std::string> symmetric_overlap_sqrt(
            const Eigen::MatrixXd &overlap,
            double threshold = 1e-10)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap);
            if (solver.info() != Eigen::Success)
                return std::unexpected("failed to diagonalize overlap matrix");

            const Eigen::VectorXd &evals = solver.eigenvalues();
            if (evals.minCoeff() < -threshold)
                return std::unexpected(std::format(
                    "overlap matrix is not positive semidefinite (min eigenvalue = {:.3e})",
                    evals.minCoeff()));

            Eigen::VectorXd sqrt_evals(evals.size());
            for (Eigen::Index i = 0; i < evals.size(); ++i)
                sqrt_evals(i) = (evals(i) > threshold) ? std::sqrt(evals(i)) : 0.0;

            return solver.eigenvectors() * sqrt_evals.asDiagonal() * solver.eigenvectors().transpose();
        }

        PopulationAnalysis accumulate_atomic_populations(
            const Molecule &molecule,
            const std::vector<std::vector<int>> &atom_to_aos,
            const Eigen::VectorXd &ao_population,
            const Eigen::VectorXd *ao_spin_population)
        {
            PopulationAnalysis analysis;
            analysis.atoms.resize(molecule.natoms);
            analysis.ao_population = ao_population;
            analysis.has_spin_population = (ao_spin_population != nullptr);

            for (std::size_t atom = 0; atom < molecule.natoms; ++atom)
            {
                analysis.atoms[atom].atom_index = atom;
                analysis.atoms[atom].atomic_number = molecule.atomic_numbers(static_cast<Eigen::Index>(atom));
                analysis.atoms[atom].net_charge = static_cast<double>(analysis.atoms[atom].atomic_number);

                for (const int mu : atom_to_aos[atom])
                {
                    analysis.atoms[atom].electron_population += ao_population(mu);
                    if (ao_spin_population != nullptr)
                        analysis.atoms[atom].spin_population += (*ao_spin_population)(mu);
                }
            }

            analysis.total_electrons = ao_population.sum();
            analysis.total_charge = 0.0;
            analysis.total_spin_population =
                (ao_spin_population != nullptr) ? ao_spin_population->sum() : 0.0;

            for (auto &atom : analysis.atoms)
            {
                atom.net_charge -= atom.electron_population;
                analysis.total_charge += atom.net_charge;
            }

            return analysis;
        }
    } // namespace

    std::expected<PopulationAnalysis, std::string> mulliken_population_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density)
    {
        if (auto valid = validate_population_inputs(
                molecule, basis, overlap, total_density, spin_density);
            !valid)
        {
            return std::unexpected("Mulliken population analysis " + valid.error());
        }

        auto atom_to_aos = build_atom_ao_indices(molecule, basis);
        if (!atom_to_aos)
            return std::unexpected("Mulliken population analysis " + atom_to_aos.error());

        const Eigen::VectorXd ao_population = gross_ao_population(total_density, overlap);
        Eigen::VectorXd ao_spin_population;
        if (spin_density != nullptr)
            ao_spin_population = gross_ao_population(*spin_density, overlap);

        return accumulate_atomic_populations(
            molecule,
            *atom_to_aos,
            ao_population,
            (spin_density != nullptr) ? &ao_spin_population : nullptr);
    }

    std::expected<PopulationAnalysis, std::string> lowdin_population_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density)
    {
        if (auto valid = validate_population_inputs(
                molecule, basis, overlap, total_density, spin_density);
            !valid)
        {
            return std::unexpected("Löwdin population analysis " + valid.error());
        }

        auto atom_to_aos = build_atom_ao_indices(molecule, basis);
        if (!atom_to_aos)
            return std::unexpected("Löwdin population analysis " + atom_to_aos.error());

        auto S_half = symmetric_overlap_sqrt(overlap);
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

        return accumulate_atomic_populations(
            molecule,
            *atom_to_aos,
            ao_population,
            (spin_density != nullptr) ? &ao_spin_population : nullptr);
    }

    std::expected<MayerBondOrderAnalysis, std::string> mayer_bond_order_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *alpha_density,
        const Eigen::MatrixXd *beta_density)
    {
        if (auto valid = validate_population_inputs(
                molecule, basis, overlap, total_density, nullptr);
            !valid)
        {
            return std::unexpected("Mayer bond-order analysis " + valid.error());
        }

        const Eigen::Index nbasis = static_cast<Eigen::Index>(basis.nbasis());
        if (alpha_density != nullptr && !matrix_has_shape(*alpha_density, nbasis, nbasis))
            return std::unexpected(std::format(
                "Mayer bond-order analysis alpha-density matrix has shape {}x{} but expected {}x{}",
                alpha_density->rows(), alpha_density->cols(), nbasis, nbasis));
        if (beta_density != nullptr && !matrix_has_shape(*beta_density, nbasis, nbasis))
            return std::unexpected(std::format(
                "Mayer bond-order analysis beta-density matrix has shape {}x{} but expected {}x{}",
                beta_density->rows(), beta_density->cols(), nbasis, nbasis));

        auto atom_to_aos = build_atom_ao_indices(molecule, basis);
        if (!atom_to_aos)
            return std::unexpected("Mayer bond-order analysis " + atom_to_aos.error());

        MayerBondOrderAnalysis analysis;
        analysis.bond_orders = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(molecule.natoms),
            static_cast<Eigen::Index>(molecule.natoms));

        const Eigen::MatrixXd PS_total = total_density * overlap;
        const Eigen::MatrixXd PS_alpha =
            (alpha_density != nullptr) ? (*alpha_density) * overlap : Eigen::MatrixXd();
        const Eigen::MatrixXd PS_beta =
            (beta_density != nullptr) ? (*beta_density) * overlap : Eigen::MatrixXd();

        for (std::size_t atom_a = 0; atom_a < molecule.natoms; ++atom_a)
        {
            for (std::size_t atom_b = atom_a + 1; atom_b < molecule.natoms; ++atom_b)
            {
                double bond_order = 0.0;
                for (const int mu : (*atom_to_aos)[atom_a])
                    for (const int nu : (*atom_to_aos)[atom_b])
                    {
                        if (alpha_density != nullptr && beta_density != nullptr)
                        {
                            bond_order += PS_alpha(mu, nu) * PS_alpha(nu, mu);
                            bond_order += PS_beta(mu, nu) * PS_beta(nu, mu);
                        }
                        else
                        {
                            bond_order += PS_total(mu, nu) * PS_total(nu, mu);
                        }
                    }

                analysis.bond_orders(
                    static_cast<Eigen::Index>(atom_a),
                    static_cast<Eigen::Index>(atom_b)) = bond_order;
                analysis.bond_orders(
                    static_cast<Eigen::Index>(atom_b),
                    static_cast<Eigen::Index>(atom_a)) = bond_order;
            }
        }

        return analysis;
    }
} // namespace HartreeFock::SCF
