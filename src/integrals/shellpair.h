#ifndef HF_SHELLPAIR_H
#define HF_SHELLPAIR_H

#include "base/types.h"

std::vector <HartreeFock::ShellPair> build_shellpairs(HartreeFock::Basis &basis);

// Compute the shell pair index for (i,j) given total number of shells
inline std::size_t pair_index(std::size_t i, std::size_t j, std::size_t nshells);

inline std::pair<std::size_t, std::size_t> invert_pair_index(std::size_t p, std::size_t nshells);

#endif // !HF_SHELLPAIR_H
