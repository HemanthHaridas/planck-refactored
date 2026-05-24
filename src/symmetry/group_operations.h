#ifndef HF_GROUP_OPERATIONS_H
#define HF_GROUP_OPERATIONS_H

#include "base/types.h"
#include <Eigen/Core>
#include <expected>
#include <string>
#include <vector>

// ─── Full point-group AO operation matrices (Phase 0 of the full-symmetry ERI
//     reduction; see docs/FULL_SYMMETRY_ERI_DESIGN.md) ──────────────────────────
//
// This module builds the dense nb×nb matrices O_R that describe how each Cartesian
// AO transforms into a linear combination of AOs under every operation R of the
// molecular point group. Unlike the D2h-only SignedAOSymOp (a monomial map: one AO
// → one AO with a ±1 phase), O_R is fully dense and so represents general
// operations (C3, C4, σ_d, S4, …) that the monomial scheme cannot.
//
// The geometry/angular math is the SAME as src/symmetry/mo_symmetry.cpp's
// build_ao_transform (which drives build_sao_basis and is validated energy-
// transparent in the spherical+symmetry regressions). This module exposes it as a
// standalone, independently-tested unit BEFORE any integral-engine rewrite, so the
// O_R can be trusted on their own (orthogonality, group closure, character sums)
// without perturbing the working SAO path.
//
// Nothing here changes integral computation yet — it only produces the O_R that a
// later skeleton-Fock symmetrization (Phase 1) will consume.

namespace HartreeFock
{
    namespace Symmetry
    {
        // One operation of the point group: its name (for diagnostics) plus the
        // dense AO representation matrix O_R [nb×nb] in the Cartesian AO basis.
        // O_R is orthogonal (O_Rᵀ O_R = I) because the underlying 3×3 spatial
        // operation is orthogonal and the AO basis is component-norm-consistent.
        struct GroupOperation
        {
            std::string label;     // e.g. "E", "C3", "sigma_v", "S4", "i"
            Eigen::MatrixXd matrix; // O_R [nb×nb], dense

            // Shell permutation induced by R: shell_perm[s] = t means shell s maps
            // onto shell t (its atom maps under the nuclear permutation, and the
            // k-th shell of angular type L at the source atom maps to the k-th such
            // shell at the image atom). This is exact even though the within-shell
            // angular mixing in `matrix` is dense. The shell-quartet petite list
            // (os_symm/rys_symm) uses it to pick orbit representatives without
            // inspecting the dense matrix. Indexed by position in Basis::_shells.
            std::vector<int> shell_perm;
        };

        // The full set of AO operation matrices for the molecule's point group.
        struct GroupOperations
        {
            std::vector<GroupOperation> operations; // includes identity at [0]
            std::string point_group;                // Mulliken name of the full group
            int order = 0;                          // |G| == operations.size()
            bool valid = false;                     // false ⇒ symmetry off / C1 / linear
        };

        // Build the dense AO operation matrices for every operation of the FULL
        // molecular point group (not the Abelian subgroup that build_sao_basis
        // selects). Returns valid=false (not an error) when symmetry is off, the
        // group is C1, or the group is linear (C∞v/D∞h) — those have no finite
        // operation set this module handles. Requires the basis to be built and
        // molecule.standard to be the symmetrized Angstrom frame.
        std::expected<GroupOperations, std::string> build_group_operations(
            HartreeFock::Calculator &calculator);
    } // namespace Symmetry
} // namespace HartreeFock

#endif // !HF_GROUP_OPERATIONS_H
