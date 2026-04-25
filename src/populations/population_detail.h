#ifndef HF_POPULATIONS_POPULATION_DETAIL_H
#define HF_POPULATIONS_POPULATION_DETAIL_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <expected>
#include <format>
#include <string>
#include <vector>

#include "populations/population.h"

namespace HartreeFock::SCF::detail
{
    inline bool matrix_has_shape(const Eigen::MatrixXd &matrix, Eigen::Index rows, Eigen::Index cols)
    {
        return matrix.rows() == rows && matrix.cols() == cols;
    }

    inline std::expected<void, std::string> validate_population_inputs(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *spin_density)
    {
        // Every population variant reduces to atom-wise post-processing on AO
        // matrices, so a single validation helper keeps the public entry points
        // consistent about accepted shapes and initialization requirements.
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

    inline std::expected<std::vector<std::vector<int>>, std::string> build_atom_ao_indices(
        const Molecule &molecule,
        const Basis &basis)
    {
        // Population and bond-order reports are ultimately grouped by atom, but
        // the SCF machinery stores everything in AO order. This lookup lets the
        // higher-level routines stay agnostic to the shell/basis-function packing
        // details used by the integral builders.
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

    inline Eigen::VectorXd gross_ao_population(
        const Eigen::MatrixXd &density,
        const Eigen::MatrixXd &overlap)
    {
        // Mulliken gross populations are the AO-wise diagonal of P*S when
        // interpreted with the bra index fixed on the row. The explicit
        // elementwise form avoids an extra temporary matrix multiply when all we
        // need is the row sum.
        return (density.array() * overlap.array()).rowwise().sum();
    }

    inline std::expected<Eigen::MatrixXd, std::string> symmetric_overlap_sqrt(
        const Eigen::MatrixXd &overlap,
        double threshold = 1e-10)
    {
        // Lowdin analysis orthogonalizes the AO basis with S^(1/2). We build
        // that symmetric square root from the eigendecomposition so the
        // transformation stays numerically stable even when tiny overlap
        // eigenvalues are truncated near linear dependence.
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

    inline PopulationAnalysis accumulate_atomic_populations(
        const Molecule &molecule,
        const std::vector<std::vector<int>> &atom_to_aos,
        const Eigen::VectorXd &ao_population,
        const Eigen::VectorXd *ao_spin_population)
    {
        // Once an AO-resolved population vector exists, all analysis flavors
        // share the same last step: sum AO contributions onto each atom, then
        // convert electron populations into net charges. Keeping that logic in
        // one place ensures Mulliken and Lowdin stay consistent about totals and
        // optional spin reporting.
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
} // namespace HartreeFock::SCF::detail

#endif // HF_POPULATIONS_POPULATION_DETAIL_H
