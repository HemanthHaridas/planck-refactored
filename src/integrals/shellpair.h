#ifndef HF_SHELLPAIR_H
#define HF_SHELLPAIR_H

#include "base/types.h"

std::vector<HartreeFock::ShellPair> build_shellpairs(const HartreeFock::Basis &basis);

// Compute the shell pair index for (i,j) given total number of shells
inline std::size_t pair_index(std::size_t i, std::size_t j)
{
    if (i < j)
        std::swap(i, j); // enforce i >= j
    return i * (i + 1) / 2 + j;
}

#endif // !HF_SHELLPAIR_H
