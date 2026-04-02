#ifndef HF_INTEGRAL_SYMMETRY_H
#define HF_INTEGRAL_SYMMETRY_H

#include <cstddef>

#include "base/types.h"

namespace HartreeFock::Symmetry
{
    // Build signed AO permutation operations for simple axis sign-flip
    // symmetries in the current aligned molecular frame. These operations are
    // used to avoid recomputing equivalent AO integrals.
    //
    // On success, calculator._integral_symmetry_ops always contains at least
    // the identity operation. calculator._use_integral_symmetry is true only
    // when one or more non-identity operations were found.
    std::size_t update_integral_symmetry(HartreeFock::Calculator &calculator);
} // namespace HartreeFock::Symmetry

#endif
