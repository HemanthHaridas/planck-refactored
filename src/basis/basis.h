#ifndef HF_BASIS_H
#define HF_BASIS_H

#include <Eigen/Core>
#include <string>
#include <vector>

#include "base/types.h"

namespace HartreeFock
{
    namespace BasisFunctions
    {
        HartreeFock::ShellType _map_shell_to_L(const std::string &label);    // Map shell label to angular momentum
        std::vector<Eigen::Vector3i> _cartesian_shell_order(unsigned int L); // Generator for angular momentum tuples

        // Primtive normalizations
        Eigen::VectorXd primitive_normalization(unsigned int L, const Eigen::VectorXd &exponents);

        // Contracted normalization
        double contracted_normalization(unsigned int L, const Eigen::VectorXd &exponents, const Eigen::VectorXd &coefficients, const Eigen::VectorXd &prim_norms);

        // Read Gaussian94 basis sets
        HartreeFock::Basis read_gbs_basis(const std::string file_name, const HartreeFock::Molecule &molecule, const HartreeFock::BasisType &basis_type);
    
    } // namespace BasisFunctions
} // namespace HartreeFock
#endif // !HF_BASIS_H
