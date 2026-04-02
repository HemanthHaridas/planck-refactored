#include "basis.h"

HartreeFock::ShellType HartreeFock::BasisFunctions::_map_shell_to_L(const std::string &label)
{
    return (label == "S") ? HartreeFock::ShellType::S : (label == "P") ? HartreeFock::ShellType::P
                                                    : (label == "D")   ? HartreeFock::ShellType::D
                                                    : (label == "F")   ? HartreeFock::ShellType::F
                                                    : (label == "G")   ? HartreeFock::ShellType::G
                                                    : (label == "H")   ? HartreeFock::ShellType::H
                                                                       : throw std::runtime_error("Unknown shell label: " + label);
}

std::vector<Eigen::Vector3i> HartreeFock::BasisFunctions::_cartesian_shell_order(unsigned int L)
{
    // Number of Cartesian functions for angular momentum L
    std::size_t nfunc = (L + 1) * (L + 2) / 2;
    std::vector<Eigen::Vector3i> result;
    result.reserve(nfunc); // reserve capacity, don't resize

    for (int lx = L; lx >= 0; --lx)
    {
        for (int ly = L - lx; ly >= 0; --ly)
        {
            int lz = L - lx - ly;
            result.emplace_back(lx, ly, lz); // construct Eigen::Vector3i directly
        }
    }

    return result;
}

Eigen::VectorXd HartreeFock::BasisFunctions::primitive_normalization(unsigned int L, const Eigen::VectorXd &exponents)
{
    constexpr double pi = 3.1415926535897932384626433832795;                   // Value of PI
    const double prefactor = std::pow(2.0, 2.0 * L + 1.5) / std::pow(pi, 1.5); // Prefactor is constant for L

    // Compute primitive normalization factors in one shot
    Eigen::VectorXd normalizations = (exponents.array().pow(L + 1.5) * prefactor).sqrt().matrix();

    return normalizations;
}

double HartreeFock::BasisFunctions::contracted_normalization(unsigned int L, const Eigen::VectorXd &exponents, const Eigen::VectorXd &coefficients, const Eigen::VectorXd &prim_norms)
{
    constexpr double pi = 3.1415926535897932384626433832795;
    const std::size_t n = exponents.size();

    if (coefficients.size() != n || prim_norms.size() != n)
    {
        throw std::runtime_error("contracted_normalization: size mismatch");
    }

    // Pairwise sums ai + aj
    Eigen::MatrixXd aij = exponents.replicate(1, n) + exponents.transpose().replicate(n, 1);

    // Overlap integrals sij = pi^(3/2) / (2^L * (ai+aj)^(L+3/2))
    // This is the primitive pair self-overlap for same-center Cartesian Gaussians
    // x^lx y^ly z^lz exp(-alpha*r^2) where each lx,ly,lz <= 1 (the "off-diagonal" convention).
    const double pi_factor = std::pow(pi, 1.5) / std::pow(2.0, L);
    Eigen::MatrixXd sij = (pi_factor / aij.array().pow(L + 1.5)).matrix();

    // Outer products of contraction coefficients and primitive normalizations
    Eigen::MatrixXd cij = coefficients * coefficients.transpose();
    Eigen::MatrixXd nij = prim_norms * prim_norms.transpose();

    // Double sum over i,j
    double sum = (cij.array() * nij.array() * sij.array()).sum();

    return 1.0 / std::sqrt(sum);
}
