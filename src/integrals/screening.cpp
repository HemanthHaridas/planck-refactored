#include "screening.h"

#include "hgp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

namespace
{
    using SymOps = std::vector<HartreeFock::SignedAOSymOp>;

    struct PairOrbitElem
    {
        std::size_t i = 0;
        std::size_t j = 0;
        int sign = 1;
    };

    bool use_symmetry_ops(const SymOps *sym_ops)
    {
        return sym_ops != nullptr && sym_ops->size() > 1;
    }

    void canonicalize_pair(std::size_t &i, std::size_t &j)
    {
        if (i > j)
            std::swap(i, j);
    }

    // Returns false if a conflicting sign for an already-seen pair would
    // force the Schwarz bound for that pair to zero. The Schwarz bound
    // tolerates phase cancellation between bra and ket (it's a squared-
    // amplitude bound), so we keep the orbit but ignore the conflict flag
    // at the call site — matching the existing behavior in hgp.cpp.
    bool append_pair_orbit(
        std::vector<PairOrbitElem> &orbit,
        std::size_t i, std::size_t j, int sign)
    {
        for (const auto &elem : orbit)
        {
            if (elem.i == i && elem.j == j)
                return elem.sign == sign;
        }
        orbit.push_back({i, j, sign});
        return true;
    }

    std::pair<std::vector<PairOrbitElem>, bool> build_pair_orbit(
        std::size_t i, std::size_t j, const SymOps &sym_ops)
    {
        std::vector<PairOrbitElem> orbit;
        orbit.reserve(sym_ops.size());

        for (const auto &op : sym_ops)
        {
            std::size_t ii = static_cast<std::size_t>(op.ao_map[i]);
            std::size_t jj = static_cast<std::size_t>(op.ao_map[j]);
            const int sign =
                static_cast<int>(op.ao_sign[i]) *
                static_cast<int>(op.ao_sign[j]);
            canonicalize_pair(ii, jj);
            if (!append_pair_orbit(orbit, ii, jj, sign))
                return {orbit, true};
        }

        std::sort(
            orbit.begin(), orbit.end(),
            [](const PairOrbitElem &a, const PairOrbitElem &b)
            { return std::tie(a.i, a.j) < std::tie(b.i, b.j); });
        return {orbit, false};
    }
} // namespace

std::vector<double> HartreeFock::Screening::schwarz_table_hgp(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::size_t nbasis,
    const std::vector<HartreeFock::SignedAOSymOp> *sym_ops)
{
    const std::size_t nb = nbasis;
    std::vector<double> Q(nb * nb, 0.0);
    const bool use_sym = use_symmetry_ops(sym_ops);

    for (const auto &sp : shell_pairs)
    {
        const std::size_t i = sp.A._index;
        const std::size_t j = sp.B._index;
        std::vector<PairOrbitElem> orbit;

        if (use_sym)
        {
            auto [orb, forced_zero] = build_pair_orbit(i, j, *sym_ops);
            orbit = std::move(orb);
            (void)forced_zero;
            if (orbit.front().i != i || orbit.front().j != j)
                continue;
        }

        const double value =
            HartreeFock::HeadGordonPople::_contracted_eri_elem(
                sp, sp,
                sp.A._cartesian[0], sp.A._cartesian[1], sp.A._cartesian[2],
                sp.B._cartesian[0], sp.B._cartesian[1], sp.B._cartesian[2],
                sp.A._cartesian[0], sp.A._cartesian[1], sp.A._cartesian[2],
                sp.B._cartesian[0], sp.B._cartesian[1], sp.B._cartesian[2],
                HartreeFock::ERIKernel::Coulomb, 0.0);
        const double q = std::sqrt(std::abs(value));

        if (!use_sym)
        {
            Q[i * nb + j] = q;
            Q[j * nb + i] = q;
            continue;
        }

        for (const auto &elem : orbit)
        {
            Q[elem.i * nb + elem.j] = q;
            Q[elem.j * nb + elem.i] = q;
        }
    }

    return Q;
}
