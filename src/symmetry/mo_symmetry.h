#ifndef HF_MO_SYMMETRY_H
#define HF_MO_SYMMETRY_H

#include "base/types.h"

namespace HartreeFock
{
    namespace Symmetry
    {
        // Assign irreducible representation labels to all converged MOs.
        // Fills calculator._info._scf.alpha.mo_symmetry (and beta for UHF).
        // No-op when symmetry is off or the point group is C1 or linear.
        void assign_mo_symmetry(HartreeFock::Calculator& calculator);
    }
}

#endif // !HF_MO_SYMMETRY_H
