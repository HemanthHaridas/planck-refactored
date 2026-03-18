#ifndef HF_POSTHF_MP2_GRADIENT_H
#define HF_POSTHF_MP2_GRADIENT_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{
    struct RMP2AmplitudeData
    {
        int n_occ  = 0;
        int n_virt = 0;

        Eigen::MatrixXd C_occ;
        Eigen::MatrixXd C_virt;
        Eigen::VectorXd eps_occ;
        Eigen::VectorXd eps_virt;

        // Stored in row-major [i,a,j,b] order.
        std::vector<double> iajb;
        std::vector<double> ibja;
        std::vector<double> t2;
        std::vector<double> tau;
    };

    struct RMP2UnrelaxedDensity
    {
        Eigen::MatrixXd P_occ;   // occupied-occupied block correction
        Eigen::MatrixXd P_virt;  // virtual-virtual block correction
    };

    struct RMP2RelaxedDensity
    {
        Eigen::MatrixXd P_mo;  // full spin-summed MO-space density
        Eigen::MatrixXd P_ao;  // full AO-space density
    };

    std::expected<RMP2AmplitudeData, std::string> build_rmp2_amplitudes(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

    // Build the unrelaxed MP2 one-particle density in MO block form.
    // This is one of the core ingredients of the final analytic gradient.
    std::expected<RMP2UnrelaxedDensity, std::string> build_rmp2_unrelaxed_density(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

    // Build the AO-space unrelaxed density contribution:
    //   P^(2)_AO = C_occ P_occ C_occ^T + C_virt P_virt C_virt^T
    std::expected<Eigen::MatrixXd, std::string> build_rmp2_unrelaxed_density_ao(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

    // Build the closed-shell MP2 orbital-gradient RHS in the RHF occupied-
    // virtual response space. The result is shaped [nvirt x nocc] and can be
    // used as the source term for the RHF CPHF/Z-vector equation.
    std::expected<Eigen::MatrixXd, std::string> build_rmp2_zvector_rhs(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

    // Solve the RHF response equation for the MP2 Z-vector:
    //   A * Z = -RHS_MP2
    // returning Z in [nvirt x nocc] shape.
    std::expected<Eigen::MatrixXd, std::string> solve_rmp2_zvector(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

    // Build the first-order relaxed MP2 one-particle density using the RHF
    // reference occupations, the MP2 unrelaxed oo/vv corrections, and the
    // occupied-virtual Z-vector response block.
    std::expected<RMP2RelaxedDensity, std::string> build_rmp2_relaxed_density(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);
}

#endif
