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
    // Build the full RHF CPHF/Z-vector matrix in the occupied-virtual space.
    //
    // The matrix is indexed in (a,i) compound order with a in virtual orbitals
    // and i in occupied orbitals:
    //
    //   A_ai,bj = (eps_a - eps_i) delta_ab delta_ij
    //           + 4 (ai|bj) - (ab|ij) - (aj|bi)
    //
    // This is the canonical RHF response matrix used by the MP2 Z-vector
    // equations and other relaxed-property solvers.
    //
    // The matrix is flattened in compound `(a,i)` order, so row/column
    // `a * nocc + i` always refers to the same occupied-virtual rotation.
    std::expected<Eigen::MatrixXd, std::string> build_rhf_cphf_matrix(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs);

    // Solve A * z = rhs for one or more right-hand sides in the occupied-virtual
    // RHF response space. rhs is shaped [nvirt x nocc], and the solution has the
    // same shape.
    //
    // This wrapper keeps the solver interface aligned with the matrix builder:
    // callers work in the natural virtual-by-occupied layout while the dense
    // solve happens on the flattened vector.
    std::expected<Eigen::MatrixXd, std::string> solve_rhf_cphf(
        HartreeFock::Calculator& calculator,
        const std::vector<HartreeFock::ShellPair>& shell_pairs,
        const Eigen::MatrixXd& rhs);
}

#endif
