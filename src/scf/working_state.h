#ifndef HF_WORKING_STATE_H
#define HF_WORKING_STATE_H

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock
{
    namespace SCF
    {
        // Recomputes the geometry-derived working state that SCF consumes.
        //
        // Assumes calc._shells has already been built from the current
        // _molecule (read_gbs_basis or equivalent), and calc.initialize() has
        // run. Calls calc.recompute_nuclear_repulsion() is NOT done here —
        // the geomopt/freq inner loops already do it explicitly, and the
        // driver wants control over when it fires.
        //
        // What this does:
        //   1. Builds shell pairs from the current shells.
        //   2. In spherical mode, row-normalizes calc._shells._cart_to_sph
        //      so diag(C S_cart Cᵀ) = 1 in the new basis. The transform
        //      directions depend only on L (geometry-independent), but the
        //      row scaling depends on S_cart, which moves with geometry.
        //      Skipping this on a geometry change silently corrupts every
        //      downstream spherical AO matrix element.
        //   3. Computes Cartesian S, T, V using the calc._integral._engine
        //      and the current integral-symmetry ops.
        //   4. Stores the working-basis 1e matrices on calc:
        //        Cartesian mode: _overlap = S,           _hcore = T + V
        //        Spherical mode: _overlap = C·S·Cᵀ,      _hcore = C·(T+V)·Cᵀ
        //
        // Returns the shell pairs that were built — callers use them for the
        // immediately-following SCF and gradient calls.
        //
        // This is the spherical-aware version of the basis-rebuild block
        // that lives inline at the top of src/driver.cpp. The geomopt and
        // freq inner loops (src/opt/geomopt.cpp, src/freq/hessian.cpp) call
        // this helper at every geometry step to stay in sync with the
        // driver's spherical-aware setup; without it, a stale _cart_to_sph
        // and Cartesian-stored _overlap / _hcore would silently break the
        // spherical SCF at displaced geometries.
        std::expected<std::vector<HartreeFock::ShellPair>, std::string>
        rebuild_basis_dependent_state(HartreeFock::Calculator &calc);
    } // namespace SCF
} // namespace HartreeFock

#endif // !HF_WORKING_STATE_H
