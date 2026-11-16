#ifndef HF_POSTHF_RHF_RESPONSE_H
#define HF_POSTHF_RHF_RESPONSE_H

#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{
    // RHF Z-vector: solves A z = -rhs (rhs, z: n_virt x n_occ) with the
    // matrix-free OrbitalHessian (post_hf/orbital_hessian.h), the SCF's own
    // operator, by preconditioned GMRES. PLANCK_DEBUG_RHF_RESPONSE=1 dumps A,
    // rhs and z.
    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &mo_coeff,
        const Eigen::VectorXd &mo_energy,
        const Eigen::MatrixXd &rhs);
} // namespace HartreeFock::Correlation

#endif
