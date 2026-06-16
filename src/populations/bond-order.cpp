#include "populations/population_detail.h"

namespace HartreeFock::SCF
{
    std::expected<MayerBondOrderAnalysis, std::string> mayer_bond_order_analysis(
        const Molecule &molecule,
        const Basis &basis,
        const Eigen::MatrixXd &overlap,
        const Eigen::MatrixXd &total_density,
        const Eigen::MatrixXd *alpha_density,
        const Eigen::MatrixXd *beta_density)
    {
        // Mayer bond orders are built from PS blocks connecting AO subspaces on
        // different atoms. Closed-shell calculations can use the total density
        // directly, while open-shell runs optionally split alpha/beta so the
        // spin-resolved contractions reproduce the usual unrestricted form.
        if (auto valid = detail::validate_population_inputs(
                molecule, basis, overlap, total_density, nullptr);
            !valid)
        {
            return std::unexpected("Mayer bond-order analysis " + valid.error());
        }

        // Spherical mode: densities are in the (2L+1)-per-shell spherical AO basis.
        const Eigen::Index nbasis = static_cast<Eigen::Index>(basis.nbasis_sph());
        if (alpha_density != nullptr && !detail::matrix_has_shape(*alpha_density, nbasis, nbasis))
            return std::unexpected(std::format(
                "Mayer bond-order analysis alpha-density matrix has shape {}x{} but expected {}x{}",
                alpha_density->rows(), alpha_density->cols(), nbasis, nbasis));
        if (beta_density != nullptr && !detail::matrix_has_shape(*beta_density, nbasis, nbasis))
            return std::unexpected(std::format(
                "Mayer bond-order analysis beta-density matrix has shape {}x{} but expected {}x{}",
                beta_density->rows(), beta_density->cols(), nbasis, nbasis));

        auto atom_to_aos = detail::build_atom_ao_indices(molecule, basis);
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
                // The AO-to-atom map lets us extract the atom-block contraction
                // without materializing explicit submatrices. We only fill the
                // upper triangle and mirror it because Mayer bond orders are
                // symmetric by construction.
                // Mayer bond order. The canonical closed-shell form uses the
                // total density directly with no prefactor:
                //   B_AB = Σ_{μ∈A,ν∈B} (P_total S)_μν (P_total S)_νμ,
                // which gives B(H–H) = 1 for H2. The spin-resolved form that
                // reduces to this same value carries an explicit factor of 2:
                //   B_AB = 2·Σ [ (P^α S)_μν (P^α S)_νμ + (P^β S)_μν (P^β S)_νμ ],
                // since for a closed shell P^α = P^β = P_total/2 and
                //   2·[2·(½ P_total S)²] = (P_total S)². The open-shell branch
                // below must keep that factor of 2 — without it every open-shell
                // bond order comes out half its correct value.
                double bond_order = 0.0;
                for (const int mu : (*atom_to_aos)[atom_a])
                    for (const int nu : (*atom_to_aos)[atom_b])
                    {
                        if (alpha_density != nullptr && beta_density != nullptr)
                        {
                            bond_order += 2.0 * PS_alpha(mu, nu) * PS_alpha(nu, mu);
                            bond_order += 2.0 * PS_beta(mu, nu) * PS_beta(nu, mu);
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
