#ifndef HF_SPHERICAL_H
#define HF_SPHERICAL_H

#include <Eigen/Core>
#include <Eigen/QR>
#include <expected>
#include <string>

#include "base/types.h"

namespace HartreeFock
{
    namespace BasisFunctions
    {
        // ── Cartesian → real-spherical transform for one shell of angular momentum L ──
        //
        // Returns the [n_sph × n_cart] matrix T⁺ that maps coefficients expressed in
        // the (component-norm-weighted) Cartesian Gaussian basis to coefficients in the
        // real-spherical-harmonic basis, discarding the r²-contamination subspace for
        // L ≥ 2.
        //
        //   n_cart = (L + 1)(L + 2)/2     n_sph = 2L + 1
        //
        // Index conventions (must match the rest of the basis machinery):
        //   Cartesian source order : _cartesian_shell_order(L) — lx descending, then ly.
        //   Spherical target order : m = −L, −L+1, …, 0, …, +L   (libmsym convention,
        //                            pz=m0, px=m+1, py=m−1; dz2=m0, dxz=m+1, dyz=m−1,
        //                            dx2y2=m+2, dxy=m−2).
        //
        // This is the PRODUCTION entry point. For L = 0…5 it returns hand-verified
        // matrices that have shipped with the symmetry module. L ≥ 6 returns an error
        // for now (no input path reaches it — the GBS parser caps at H/L=5). The
        // closed-form recurrence in spherical_recurrence.h is the independent oracle
        // that validates these matrices; see tests/spherical_transform.cpp.
        std::expected<Eigen::MatrixXd, std::string> cart_to_sph_block(int L);

        // ── Whole-basis Cartesian → spherical transform C ─────────────────────────
        //
        // Assembles the block-diagonal matrix C [nbasis_sph × nbasis] from the per-shell
        // cart_to_sph_block(L), placed at the cumulative spherical/Cartesian offsets in
        // shell order. Mapping Cartesian AO quantities to the spherical basis is then:
        //   S_sph = C · S_cart · Cᵀ ,   c_sph = C · c_cart , etc.
        //
        // Errors if any shell's angular momentum exceeds the supported transform range
        // (see cart_to_sph_block). The result depends only on the shells already loaded
        // into `basis`, so it must be called after the Cartesian shells are built.
        std::expected<Eigen::MatrixXd, std::string> build_cart_to_sph(const HartreeFock::Basis &basis);

    } // namespace BasisFunctions
} // namespace HartreeFock

#endif // !HF_SPHERICAL_H
