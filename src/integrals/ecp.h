#ifndef HF_ECP_H
#define HF_ECP_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"

namespace HartreeFock
{
    namespace ECP
    {
        // Cartesian AO matrix U_munu = <mu| U_ECP |nu> summed over every ECP on
        // basis._ecps (libecpint). Zero matrix when the basis carries no ECP.
        // Precondition: describe_ecps(basis) succeeded.
        Eigen::MatrixXd _compute_ecp(const HartreeFock::Basis &basis);

        // natoms x 3 nuclear gradient sum_{mu nu} D_mu_nu dU_mu_nu/dR (Ha/Bohr) for a
        // symmetric Cartesian AO density D. shell_atom[s] is the atom of basis._shells[s]
        // (Gradient::detail::build_shell_atom_map). Zero when there is no ECP.
        // Precondition: describe_ecps succeeded with basis L + 1 <= LIBECPINT_MAX_L.
        Eigen::MatrixXd _compute_ecp_gradient(const HartreeFock::Basis &basis,
                                              const std::vector<int> &shell_atom,
                                              std::size_t natoms,
                                              const Eigen::MatrixXd &D);

        // One line per ECP atom (core electrons, Z_eff, channel term counts), or an
        // error when an angular momentum exceeds what libecpint was built for.
        // Empty string when the basis carries no ECP.
        std::expected<std::string, std::string> describe_ecps(const HartreeFock::Basis &basis,
                                                              const HartreeFock::Molecule &molecule);
    } // namespace ECP
} // namespace HartreeFock

#endif
