#ifndef HF_SPHERICAL_H
#define HF_SPHERICAL_H

#include <Eigen/Core>
#include <Eigen/QR>
#include <expected>
#include <string>
#include <vector>

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

        // ── Whole-ERI Cartesian → spherical transform ─────────────────────────────
        //
        // Transforms a dense Cartesian ERI tensor (chemists' notation, flat row-major
        // [n_cart^4], index p*n³+q*n²+r*n+s) into the spherical AO basis by contracting
        // all four indices with C [n_sph × n_cart]:
        //   (pq|rs)_sph = Σ_{μνλσ} C_pμ C_qν C_rλ C_sσ (μν|λσ)_cart
        // performed as four successive single-index contractions (O(n_cart⁴·n_sph)).
        //
        // `n_cart` must equal C.cols(); the returned tensor is flat [n_sph^4]. To bound
        // memory/time, errors if n_cart exceeds `max_n_cart` (default 150 — the dense
        // n⁴ tensor and this transform are only viable for modest systems anyway).
        std::expected<std::vector<double>, std::string> transform_eri_cart_to_sph(
            const std::vector<double> &eri_cart,
            const Eigen::MatrixXd &C,
            std::size_t n_cart,
            std::size_t max_n_cart = 150);

        // ── Density / energy-weighted density: spherical → Cartesian lift ─────────
        //
        // Given an AO-basis matrix M_sph defined in the spherical (2L+1 per shell)
        // basis (typically a density P_sph or an energy-weighted density W_sph
        // produced by the spherical SCF), returns the equivalent Cartesian-basis
        // representation
        //     M_cart = Cᵀ · M_sph · C            [n_cart × n_cart]
        // so that for any Cartesian AO operator X_cart whose matrix elements are
        // emitted by the integral engine (one-electron operators, ERI derivatives,
        // …) the energy contractions agree:
        //     tr(M_sph · X_sph) = tr(M_sph · C X_cart Cᵀ) = tr(M_cart · X_cart).
        //
        // C is the [n_sph × n_cart] block-diagonal transform from build_cart_to_sph
        // (or Basis::_cart_to_sph). This is the inverse direction of the lowering
        // C · X · Cᵀ used at the energy skin. Because C is a partial isometry
        // (rows span the kept harmonic subspace, dropping the r²-contamination
        // subspace for L ≥ 2), CᵀC is the projector onto that kept subspace, not
        // the identity — so this lift is exact for matrices already living in the
        // kept subspace (true for SCF density and W) but is *not* a round-trip
        // inverse of an arbitrary Cartesian matrix.
        //
        // The shape checks mirror transform_eri_cart_to_sph for consistent error
        // messages at the gradient skin.
        std::expected<Eigen::MatrixXd, std::string> lift_density_sph_to_cart(
            const Eigen::MatrixXd &M_sph,
            const Eigen::MatrixXd &C);

    } // namespace BasisFunctions
} // namespace HartreeFock

#endif // !HF_SPHERICAL_H
