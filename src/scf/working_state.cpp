#include "working_state.h"

#include <Eigen/Core>
#include <utility>

#include "basis/basis.h"
#include "basis/spherical.h"
#include "integrals/os.h"
#include "integrals/base.h"
#include "symmetry/integral_symmetry.h"

std::expected<void, std::string>
HartreeFock::SCF::compute_working_one_electron(HartreeFock::Calculator &calc,
                                               const std::vector<HartreeFock::ShellPair> &shellpairs)
{
    // Refresh integral-symmetry ops against the (possibly new) geometry before
    // requesting the symmetry-aware compute path.
    HartreeFock::Symmetry::update_integral_symmetry(calc);

    auto [S, T] =
        _compute_1e(shellpairs, calc._shells.nbasis(), calc._integral._engine,
                    calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);
    Eigen::MatrixXd V = _compute_potential(
        shellpairs, calc._shells, calc._molecule, calc._integral._engine,
        calc._use_integral_symmetry ? &calc._integral_symmetry_ops : nullptr);

    // Cartesian engine output -> working basis (C is normalized at basis read,
    // so diag(_overlap) = 1 in spherical mode).
    auto S_w = HartreeFock::BasisFunctions::to_working(calc._shells, S);
    auto H_w = HartreeFock::BasisFunctions::to_working(calc._shells, T + V);
    if (!S_w || !H_w)
        return std::unexpected(!S_w ? S_w.error() : H_w.error());
    calc._overlap = std::move(*S_w);
    calc._hcore = std::move(*H_w);
    return {};
}

std::expected<std::vector<HartreeFock::ShellPair>, std::string>
HartreeFock::SCF::rebuild_basis_dependent_state(HartreeFock::Calculator &calc)
{
    std::vector<HartreeFock::ShellPair> shellpairs = build_shellpairs(calc._shells);
    if (auto r = compute_working_one_electron(calc, shellpairs); !r)
        return std::unexpected(r.error());
    return shellpairs;
}
