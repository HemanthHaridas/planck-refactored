#include <cstddef>
#include <cmath>
#include <utility>

#include "shellpair.h"

std::vector <HartreeFock::ShellPair> build_shellpairs(HartreeFock::Basis &basis)
{
    std::size_t n_shells = basis.nbasis();                  // Number of contracted gaussians
    std::vector <HartreeFock::ShellPair> shell_pairs{};     // Shellpair container
    
    for (std::size_t ia = 0; ia < n_shells; ia++)
    {
        for (std::size_t ib = ia; ib < n_shells; ib++)
        {
            shell_pairs.emplace_back(basis._basis_functions[ia], basis._basis_functions[ib]);
        }
    }
    
    return shell_pairs;
}

inline std::size_t pair_index(std::size_t i, std::size_t j)
{
    if (i < j) std::swap(i, j); // enforce i >= j
    return i*(i+1)/2 + j;
}

inline std::pair<std::size_t, std::size_t> invert_pair_index(std::size_t k, std::size_t nshells)
{
    std::size_t i = static_cast<std::size_t>((std::sqrt(8.0 * k + 1) - 1) / 2);
    std::size_t j = k - i * (i + 1) /2;

    return {i, j};
}
