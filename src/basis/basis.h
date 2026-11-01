#ifndef HF_BASIS_H
#define HF_BASIS_H

#include <Eigen/Core>
#include <string>
#include <vector>

#include "base/types.h"
#include <expected>

namespace HartreeFock
{
    namespace BasisFunctions
    {
        std::expected<HartreeFock::ShellType, std::string> _map_shell_to_L(const std::string &label); // Map shell label to angular momentum
        std::vector<Eigen::Vector3i> _cartesian_shell_order(unsigned int L);                          // Generator for angular momentum tuples

        // Primtive normalizations
        Eigen::VectorXd primitive_normalization(unsigned int L, const Eigen::VectorXd &exponents);

        // Contracted normalization
        std::expected<double, std::string> contracted_normalization(unsigned int L, const Eigen::VectorXd &exponents, const Eigen::VectorXd &coefficients, const Eigen::VectorXd &prim_norms);

        // (2n-1)!! with the convention (-1)!! = 1.
        int double_factorial(int n);

        // Cartesian component normalization 1/sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)
        double component_norm(int df);

        // Read Gaussian94 basis sets
        std::expected<HartreeFock::Basis, std::string> read_gbs_basis(const std::string file_name, const HartreeFock::Molecule &molecule, const HartreeFock::BasisType &basis_type);

    } // namespace BasisFunctions
} // namespace HartreeFock
#endif // !HF_BASIS_H
