#include <cstddef>
#include <cmath>
#include <utility>

#include "shellpair.h"

std::vector <HartreeFock::ShellPair> build_shellpairs(const HartreeFock::Basis &basis)
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
