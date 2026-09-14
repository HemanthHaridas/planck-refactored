#ifndef HF_POPULATIONS_ESP_H
#define HF_POPULATIONS_ESP_H

#include <Eigen/Core>

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::SCF
{
    // Molecular electrostatic potential at an arbitrary list of points, in
    // atomic units (Hartree/e):
    //
    //   phi(r) = sum_A Z_A / |r - R_A|  -  sum_munu P_munu <mu| 1/|r - r'| |nu>
    //
    // Points are in Bohr. `density` is the TOTAL density (P_alpha + P_beta for
    // unrestricted), matching mulliken_population_analysis's convention.
    //
    // The electronic term is the one-electron potential integral the integral
    // engine already provides; see ObaraSaika::_compute_electronic_potential
    // (src/integrals/os.h) for why it is a fused sweep rather than a loop over
    // _compute_external_charge_attraction.
    std::expected<Eigen::VectorXd, std::string> electrostatic_potential(
        const Molecule &molecule,
        const std::vector<ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density,
        const std::vector<Eigen::Vector3d> &points);
} // namespace HartreeFock::SCF

#endif // HF_POPULATIONS_ESP_H
