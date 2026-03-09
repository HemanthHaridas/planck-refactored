#ifndef HF_SHELLPAIR_H
#define HF_SHELLPAIR_H

#include "base/types.h"

std::vector <HartreeFock::ShellPair> build_shellpairs(HartreeFock::Basis &basis);

// Compute the shell pair index for (i,j) given total number of shells
inline std::size_t pair_index(std::size_t i, std::size_t j)
{
    if (i < j) std::swap(i, j); // enforce i >= j
    return i * (i + 1) / 2 + j;
}

inline std::pair<std::size_t, std::size_t> invert_pair_index(std::size_t k)
{
    std::size_t i = static_cast<std::size_t>(std::floor((std::sqrt(8.0 * k + 1.0) - 1.0) / 2.0));
    std::size_t j = k - i * (i + 1) / 2;

    return {i, j};
}

#endif // !HF_SHELLPAIR_H
