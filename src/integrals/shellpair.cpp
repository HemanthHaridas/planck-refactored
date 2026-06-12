#include <cmath>
#include <cstddef>
#include <utility>

#include "shellpair.h"

std::vector<ShellGroup> build_shell_groups(const HartreeFock::Basis &basis)
{
    std::vector<ShellGroup> groups;
    groups.reserve(basis.nshells());

    // _basis_functions is built shell-by-shell, contiguously, so a shell owns a
    // contiguous run of AO components that all share the same Shell*. Walk the
    // flat list and start a new group whenever the underlying Shell* changes.
    const std::size_t nbasis = basis._basis_functions.size();
    const HartreeFock::Shell *current = nullptr;
    std::size_t shell_index = 0;

    for (std::size_t ao = 0; ao < nbasis; ++ao)
    {
        const HartreeFock::Shell *shell = basis._basis_functions[ao]._shell;
        if (shell != current)
        {
            ShellGroup group;
            group.shell = shell;
            group.shell_index = shell_index++;
            group.first_ao = ao;
            group.n_components = 1;
            groups.emplace_back(group);
            current = shell;
        }
        else
        {
            ++groups.back().n_components;
        }
    }

    return groups;
}

std::vector<HartreeFock::ShellPair> expand_shell_groups_to_ao_pairs(
    const HartreeFock::Basis &basis, const std::vector<ShellGroup> &groups)
{
    const std::size_t nbasis = basis._basis_functions.size();
    std::vector<HartreeFock::ShellPair> shell_pairs;
    shell_pairs.reserve(nbasis * (nbasis + 1) / 2);

    // Emit the per-AO upper triangle in the exact order the legacy
    // build_shellpairs produced it: outer AO ia over every component, inner AO
    // ib >= ia. Iterating the groups in order, then their components in order,
    // reproduces the flat ia/ib sweep one-for-one because the groups partition
    // _basis_functions contiguously and in order.
    for (std::size_t ga = 0; ga < groups.size(); ++ga)
    {
        const ShellGroup &A = groups[ga];
        for (std::size_t ca = 0; ca < A.n_components; ++ca)
        {
            const std::size_t ia = A.first_ao + ca;

            // ib runs from ia within group A (upper triangle of the diagonal
            // block), then over every later group in full.
            for (std::size_t cb = ca; cb < A.n_components; ++cb)
            {
                const std::size_t ib = A.first_ao + cb;
                shell_pairs.emplace_back(basis._basis_functions[ia],
                                         basis._basis_functions[ib]);
            }
            for (std::size_t gb = ga + 1; gb < groups.size(); ++gb)
            {
                const ShellGroup &B = groups[gb];
                for (std::size_t cb = 0; cb < B.n_components; ++cb)
                {
                    const std::size_t ib = B.first_ao + cb;
                    shell_pairs.emplace_back(basis._basis_functions[ia],
                                             basis._basis_functions[ib]);
                }
            }
        }
    }

    return shell_pairs;
}

std::vector<HartreeFock::ShellPair> build_shellpairs(const HartreeFock::Basis &basis)
{
    // Route the legacy per-Cartesian-AO shell-pair construction through the new
    // shell-granular layer. The expansion reproduces the previous flat
    // upper-triangle sweep over _basis_functions bitwise, so every integral
    // engine sees an identical ShellPair list (H-10 step 1: no-op adapter).
    const std::vector<ShellGroup> groups = build_shell_groups(basis);
    return expand_shell_groups_to_ao_pairs(basis, groups);
}
