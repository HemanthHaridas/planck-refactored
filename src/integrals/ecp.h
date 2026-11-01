#ifndef HF_ECP_H
#define HF_ECP_H

#include <Eigen/Core>
#include <expected>
#include <string>

#include "base/types.h"

namespace HartreeFock
{
    namespace ECP
    {
        // Cartesian AO matrix U_munu = <mu| U_ECP |nu> summed over every ECP on
        // basis._ecps (libecpint). Zero matrix when the basis carries no ECP.
        // Precondition: describe_ecps(basis) succeeded.
        Eigen::MatrixXd _compute_ecp(const HartreeFock::Basis &basis);

        // One line per ECP atom (core electrons, Z_eff, channel term counts), or an
        // error when an angular momentum exceeds what libecpint was built for.
        // Empty string when the basis carries no ECP.
        std::expected<std::string, std::string> describe_ecps(const HartreeFock::Basis &basis,
                                                              const HartreeFock::Molecule &molecule);
    } // namespace ECP
} // namespace HartreeFock

#endif
