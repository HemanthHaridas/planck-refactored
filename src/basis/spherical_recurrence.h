#ifndef HF_SPHERICAL_RECURRENCE_H
#define HF_SPHERICAL_RECURRENCE_H

#include <Eigen/Core>
#include <expected>
#include <string>

namespace HartreeFock
{
    namespace BasisFunctions
    {
        // ── Independent closed-form Cartesian → real-spherical transform ──────────────
        //
        // This is the VERIFICATION ORACLE for cart_to_sph_block (spherical.h). It is a
        // from-the-math implementation of the real-solid-harmonic expansion coefficients
        // c(l, m, lx, ly, lz) of Schlegel & Frisch (Int. J. Quantum Chem. 54, 83 (1995),
        // eqs. 15 & 28), built entirely independently of the hand-coded production
        // matrices. If the two agree for L = 0…5 (up to the shared ordering convention),
        // we have two independent derivations confirming each other; the recurrence then
        // also extends to arbitrary L where no hand-coded oracle exists.
        //
        // It is intentionally NOT wired into production: keeping it separate guarantees a
        // bug here can never silently corrupt real integrals.
        //
        // Output matches cart_to_sph_block exactly:
        //   shape [n_sph × n_cart], n_cart = (L+1)(L+2)/2, n_sph = 2L+1
        //   Cartesian source order : lx descending, then ly  (== _cartesian_shell_order)
        //   Spherical target order : m = −L … +L             (libmsym convention)
        //   Cartesian functions are component-norm-weighted (same basis as the integrals),
        //   so row m of the returned matrix gives the coefficients of the unit-normalized
        //   real spherical harmonic in the unit-normalized Cartesian functions.
        //
        // Valid for L ≥ 0; returns an error only on negative L or factorial overflow.
        std::expected<Eigen::MatrixXd, std::string> cart_to_sph_block_recurrence(int L);

    } // namespace BasisFunctions
} // namespace HartreeFock

#endif // !HF_SPHERICAL_RECURRENCE_H
