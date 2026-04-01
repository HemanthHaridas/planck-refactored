#include "post_hf/casscf/strings.h"

#include "base/types.h"
#include "symmetry/mo_symmetry.h"

#include <algorithm>
#include <bit>
#include <string>

namespace HartreeFock::Correlation::CASSCF
{

using CASSCFInternal::kCIStringBits;
using CASSCFInternal::kMaxSeparateSpinOrbitals;
using CASSCFInternal::low_bit_mask;
using CASSCFInternal::single_bit_mask;

int count_occupied_below(CIString det, int orb)
{
    return std::popcount(det & low_bit_mask(orb));
}

int parity_between(CIString s, int lo, int hi)
{
    if (lo + 1 >= hi) return 1;
    const CIString mask = low_bit_mask(hi - lo - 1) << (lo + 1);
    return (std::popcount(s & mask) % 2 == 0) ? 1 : -1;
}

std::vector<CIString> generate_strings(int n_orb, int n_occ)
{
    std::vector<CIString> result;
    if (n_occ == 0) { result.push_back(0); return result; }
    if (n_occ > n_orb || n_orb > kMaxSeparateSpinOrbitals) return result;

    CIString v = low_bit_mask(n_occ);
    const CIString limit = single_bit_mask(n_orb);
    while (v < limit)
    {
        result.push_back(v);
        const CIString c = v & (-v);
        const CIString r = v + c;
        v = (((r ^ v) >> 2) / c) | r;
    }
    return result;
}

FermionOpResult apply_annihilation(CIString det, int orb)
{
    const CIString bit = single_bit_mask(orb);
    if (!(det & bit)) return {};
    return {det ^ bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
}

FermionOpResult apply_creation(CIString det, int orb)
{
    const CIString bit = single_bit_mask(orb);
    if (det & bit) return {};
    return {det | bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
}

std::vector<int> map_mo_irreps(
    const std::vector<std::string>& mo_sym,
    const std::vector<std::string>& names)
{
    std::vector<int> idx(mo_sym.size(), -1);
    for (std::size_t i = 0; i < mo_sym.size(); ++i)
        for (std::size_t g = 0; g < names.size(); ++g)
            if (mo_sym[i] == names[g]) { idx[i] = static_cast<int>(g); break; }
    return idx;
}

std::optional<SymmetryContext> build_symmetry_context(HartreeFock::Calculator& calc)
{
    auto product_table = HartreeFock::Symmetry::build_abelian_irrep_product_table(calc);
    if (!product_table.valid || product_table.irrep_names.empty())
        return std::nullopt;

    SymmetryContext ctx;
    ctx.names = std::move(product_table.irrep_names);
    ctx.product = std::move(product_table.product);
    ctx.abelian_1d_only = true;

    int identity = -1;
    for (int g = 0; g < static_cast<int>(ctx.product.size()); ++g)
    {
        bool is_identity = true;
        for (int h = 0; h < static_cast<int>(ctx.product[g].size()); ++h)
            if (ctx.product[g][h] != h || ctx.product[h][g] != h)
            {
                is_identity = false;
                break;
            }
        if (is_identity)
        {
            identity = g;
            break;
        }
    }

    if (identity < 0)
        return std::nullopt;

    ctx.totally_symmetric_irrep = identity;
    return ctx;
}

std::optional<int> resolve_target_irrep(
    const std::string& s,
    const SymmetryContext& sym_ctx)
{
    if (s.empty()) return sym_ctx.totally_symmetric_irrep;
    for (std::size_t g = 0; g < sym_ctx.names.size(); ++g)
        if (s == sym_ctx.names[g]) return static_cast<int>(g);
    return std::nullopt;
}

bool point_group_has_only_1d_irreps(const std::string& pg)
{
    if (pg == "C1" || pg == "Ci" || pg == "Cs")
        return true;
    if (pg.find("inf") != std::string::npos || pg.size() < 2)
        return false;

    auto parse_order = [&](std::size_t pos, int& n, std::string& suffix) -> bool
    {
        std::size_t end = pos;
        while (end < pg.size() && pg[end] >= '0' && pg[end] <= '9')
            ++end;
        if (end == pos)
            return false;
        n = std::stoi(pg.substr(pos, end - pos));
        suffix = pg.substr(end);
        return true;
    };

    int n = 0;
    std::string suffix;
    switch (pg[0])
    {
        case 'C':
            if (!parse_order(1, n, suffix)) return false;
            return (suffix.empty() || suffix == "h" || suffix == "v") && n <= 2;
        case 'D':
            if (!parse_order(1, n, suffix)) return false;
            if (suffix.empty() || suffix == "h") return n <= 2;
            if (suffix == "d") return n <= 1;
            return false;
        case 'S':
            if (!parse_order(1, n, suffix)) return false;
            return suffix.empty() && n == 2;
        default:
            return false;
    }
}

void build_spin_strings_unfiltered(
    int n_act,
    int n_alpha,
    int n_beta,
    std::vector<CIString>& a_strs,
    std::vector<CIString>& b_strs)
{
    a_strs = generate_strings(n_act, n_alpha);
    b_strs = generate_strings(n_act, n_beta);
}

std::vector<CIString> build_spin_dets(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act)
{
    std::vector<CIString> sd;
    sd.reserve(dets.size());
    for (auto [ia, ib] : dets)
        sd.push_back(a_strs[ia] | ((n_act >= kCIStringBits) ? 0 : (b_strs[ib] << n_act)));
    return sd;
}

std::unordered_map<CIString, int> build_det_lookup(const std::vector<CIString>& sd)
{
    std::unordered_map<CIString, int> lut;
    lut.reserve(sd.size());
    for (int i = 0; i < static_cast<int>(sd.size()); ++i)
        lut.emplace(sd[i], i);
    return lut;
}

} // namespace HartreeFock::Correlation::CASSCF
