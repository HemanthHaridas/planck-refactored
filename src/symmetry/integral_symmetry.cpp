#include "integral_symmetry.h"

#include <array>
#include <map>
#include <stdexcept>
#include <tuple>

namespace
{
using Key = std::tuple<const HartreeFock::Shell*, int, int, int>;

static int parity_sign(int sx, int sy, int sz, const Eigen::Vector3i& cart)
{
    int sign = 1;
    if (sx < 0 && (cart[0] & 1)) sign = -sign;
    if (sy < 0 && (cart[1] & 1)) sign = -sign;
    if (sz < 0 && (cart[2] & 1)) sign = -sign;
    return sign;
}

static bool build_atom_permutation(const HartreeFock::Molecule& mol,
                                   int sx, int sy, int sz,
                                   std::vector<int>& perm)
{
    const int n = static_cast<int>(mol.natoms);
    perm.assign(n, -1);
    constexpr double tol = 5.0e-4; // Angstrom

    for (int a = 0; a < n; ++a)
    {
        const Eigen::Vector3d src = mol.standard.row(a).transpose();
        const Eigen::Vector3d img(sx * src[0], sy * src[1], sz * src[2]);

        int match = -1;
        for (int b = 0; b < n; ++b)
        {
            if (perm[b] != -1) continue;
            if (mol.atomic_numbers[a] != mol.atomic_numbers[b]) continue;
            const Eigen::Vector3d tgt = mol.standard.row(b).transpose();
            if ((img - tgt).norm() < tol)
            {
                match = b;
                break;
            }
        }

        if (match < 0)
            return false;

        perm[a] = match;
    }

    return true;
}
} // namespace

std::size_t HartreeFock::Symmetry::update_integral_symmetry(
    HartreeFock::Calculator& calculator)
{
    calculator._integral_symmetry_ops.clear();
    calculator._use_integral_symmetry = false;

    const std::size_t nb = calculator._shells.nbasis();
    HartreeFock::SignedAOSymOp identity;
    identity.ao_map.resize(nb);
    identity.ao_sign.resize(nb, 1);
    for (std::size_t mu = 0; mu < nb; ++mu)
        identity.ao_map[mu] = static_cast<int>(mu);
    calculator._integral_symmetry_ops.push_back(std::move(identity));

    if (!calculator._molecule._symmetry ||
        calculator._molecule._point_group == "C1" ||
        calculator._molecule.standard.rows() != static_cast<Eigen::Index>(calculator._molecule.natoms))
    {
        return calculator._integral_symmetry_ops.size();
    }

    const auto& basis  = calculator._shells;
    const auto& bfs    = basis._basis_functions;
    const auto& shells = basis._shells;

    std::map<std::pair<int, int>, std::vector<const HartreeFock::Shell*>> atom_l_shells;
    for (const auto& sh : shells)
    {
        const int atom = static_cast<int>(sh._atom_index);
        const int l    = static_cast<int>(sh._shell);
        atom_l_shells[{atom, l}].push_back(&sh);
    }

    std::map<Key, int> target_index;
    for (const auto& bf : bfs)
    {
        target_index[{bf._shell,
                      bf._cartesian[0],
                      bf._cartesian[1],
                      bf._cartesian[2]}] = static_cast<int>(bf._index);
    }

    const std::array<std::array<int, 3>, 7> candidates = {{
        {{-1,  1,  1}},
        {{ 1, -1,  1}},
        {{ 1,  1, -1}},
        {{-1, -1,  1}},
        {{-1,  1, -1}},
        {{ 1, -1, -1}},
        {{-1, -1, -1}},
    }};

    for (const auto& cand : candidates)
    {
        const int sx = cand[0];
        const int sy = cand[1];
        const int sz = cand[2];

        std::vector<int> atom_perm;
        if (!build_atom_permutation(calculator._molecule, sx, sy, sz, atom_perm))
            continue;

        HartreeFock::SignedAOSymOp op;
        op.ao_map.assign(nb, -1);
        op.ao_sign.assign(nb, 1);

        bool ok = true;
        for (const auto& bf : bfs)
        {
            const int atom_a = static_cast<int>(bf._shell->_atom_index);
            const int atom_b = atom_perm[atom_a];
            const int l      = static_cast<int>(bf._shell->_shell);

            const auto src_it = atom_l_shells.find({atom_a, l});
            const auto tgt_it = atom_l_shells.find({atom_b, l});
            if (src_it == atom_l_shells.end() || tgt_it == atom_l_shells.end() ||
                src_it->second.size() != tgt_it->second.size())
            {
                ok = false;
                break;
            }

            int shell_k = -1;
            for (int k = 0; k < static_cast<int>(src_it->second.size()); ++k)
            {
                if (src_it->second[k] == bf._shell)
                {
                    shell_k = k;
                    break;
                }
            }
            if (shell_k < 0)
            {
                ok = false;
                break;
            }

            const HartreeFock::Shell* tgt_shell = tgt_it->second[shell_k];
            const auto idx_it = target_index.find({tgt_shell,
                                                   bf._cartesian[0],
                                                   bf._cartesian[1],
                                                   bf._cartesian[2]});
            if (idx_it == target_index.end())
            {
                ok = false;
                break;
            }

            op.ao_map[bf._index] = idx_it->second;
            op.ao_sign[bf._index] =
                static_cast<int8_t>(parity_sign(sx, sy, sz, bf._cartesian));
        }

        if (ok)
            calculator._integral_symmetry_ops.push_back(std::move(op));
    }

    calculator._use_integral_symmetry =
        calculator._integral_symmetry_ops.size() > 1;
    return calculator._integral_symmetry_ops.size();
}
