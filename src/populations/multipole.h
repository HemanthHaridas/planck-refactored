#ifndef HF_POPULATIONS_MULTIPOLE_H
#define HF_POPULATIONS_MULTIPOLE_H

#include <Eigen/Core>

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::ObaraSaika
{
    HartreeFock::MultipoleMatrices _compute_multipole_matrices(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        std::size_t nbasis,
        const Eigen::Vector3d &origin = Eigen::Vector3d::Zero());

    std::expected<HartreeFock::MultipoleMoments, std::string> _compute_multipole_moments(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::Vector3d &origin = Eigen::Vector3d::Zero());
} // namespace HartreeFock::ObaraSaika

#endif // HF_POPULATIONS_MULTIPOLE_H
