#ifndef HF_FOCK_SYMMETRIZATION_H
#define HF_FOCK_SYMMETRIZATION_H

#include "group_operations.h"
#include <Eigen/Core>
#include <expected>
#include <string>

// ─── Skeleton → fully-symmetric Fock projection (Phase 1 of the full-symmetry ERI
//     reduction; see docs/FULL_SYMMETRY_ERI_DESIGN.md) ──────────────────────────
//
// In a symmetry-reduced ERI build only a subset of shell quartets is computed (the
// "skeleton"), giving a skeleton Fock matrix F_skel that is NOT yet the true Fock.
// Averaging F_skel over the point group restores the full operator:
//
//     F = (1/|G|) Σ_R  O_Rᵀ · F_skel · O_R
//
// where O_R are the dense AO operation matrices from group_operations.h. This is a
// projection onto the totally-symmetric (group-invariant) component of the matrix.
//
// This module implements ONLY that linear-algebra projection. It is validated
// independently of any integral reduction by the properties it must satisfy:
//   - Invariance:  O_Sᵀ · symmetrize(M) · O_S == symmetrize(M) for every S ∈ G.
//   - Idempotence: symmetrize(symmetrize(M)) == symmetrize(M).
//   - Fixed point: a matrix that is already group-invariant (e.g. a converged Fock
//                  from a symmetric SCF) is returned unchanged.
// Until Phase 2 wires the skeleton ERI build, nothing calls this in production; it
// is the correctness-critical operator the design isolates and tests on its own.

namespace HartreeFock
{
    namespace Symmetry
    {
        // Project a matrix onto the totally-symmetric component of the point group:
        //   result = (1/|G|) Σ_R O_Rᵀ M O_R.
        // `ops` must be a valid GroupOperations (ops.valid == true) whose O_R are
        // square and match M's dimension. Errors on dimension mismatch or empty/
        // invalid group. With the trivial group {E} the result equals M.
        std::expected<Eigen::MatrixXd, std::string> symmetrize_matrix(
            const Eigen::MatrixXd &matrix,
            const GroupOperations &ops);
    } // namespace Symmetry
} // namespace HartreeFock

#endif // !HF_FOCK_SYMMETRIZATION_H
