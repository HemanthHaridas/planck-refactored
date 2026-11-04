#include "working_state.h"

#include <Eigen/Core>
#include <cmath>
#include <utility>

#include "basis/spherical.h"
#include "integrals/base.h"
#include "symmetry/integral_symmetry.h"

std::expected<std::vector<HartreeFock::ShellPair>, std::string>
HartreeFock::SCF::rebuild_basis_dependent_state(HartreeFock::Calculator &calc)
{
    // ── Shell pairs ───────────────────────────────────────────────────────────
    std::vector<HartreeFock::ShellPair> shellpairs = build_shellpairs(calc._shells);

    // ── Spherical transform normalization (must precede every consumer) ──────
    // The load-time C produces correct spherical directions but unnormalized
    // rows. Normalize each row m by 1/√((C S_cart Cᵀ)_mm) so diag(S_sph) = 1,
    // using the real Cartesian overlap at the *current* geometry. The row
    // scaling depends on S_cart, so it must re-run on every geometry change;
    // skipping this leaves a stale transform that silently breaks every
    // downstream spherical AO matrix element. This is the exact body of the
    // driver's Step 2.0 normalization, kept in lockstep.
    if (calc._shells._spherical)
    {
        const auto [S_cart, T_cart_unused] =
            _compute_1e(shellpairs, calc._shells.nbasis(),
                        calc._integral._engine, nullptr);
        (void)T_cart_unused;
        Eigen::MatrixXd C = calc._shells._cart_to_sph;
        const Eigen::MatrixXd CS = C * S_cart; // [n_sph × n_cart]
        for (Eigen::Index m = 0; m < C.rows(); ++m)
        {
            const double norm2 = CS.row(m).dot(C.row(m));
            if (norm2 > 0.0)
                C.row(m) /= std::sqrt(norm2);
        }
        calc._shells._cart_to_sph = std::move(C);
    }

    // ── One-electron integrals → working-basis _overlap and _hcore ───────────
    // Refresh integral-symmetry ops against the (possibly new) geometry before
    // requesting the symmetry-aware compute path.
    HartreeFock::Symmetry::update_integral_symmetry(calc);

    auto [S, T] =
        _compute_1e(shellpairs, calc._shells.nbasis(), calc._integral._engine,
                    calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
    Eigen::MatrixXd V = _compute_potential(
        shellpairs, calc._shells, calc._molecule, calc._integral._engine,
        calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);

    // Cartesian engine output → working basis. In spherical mode the lift uses
    // the C that was just normalized above, so diag(_overlap) = 1 holds and
    // SCF works entirely in the spherical basis.
    auto S_w = HartreeFock::BasisFunctions::to_working(calc._shells, S);
    auto H_w = HartreeFock::BasisFunctions::to_working(calc._shells, T + V);
    if (!S_w || !H_w)
        return std::unexpected(!S_w ? S_w.error() : H_w.error());
    calc._overlap = std::move(*S_w);
    calc._hcore = std::move(*H_w);

    return shellpairs;
}
