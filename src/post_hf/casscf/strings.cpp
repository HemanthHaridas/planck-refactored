#include "post_hf/casscf/strings.h"

#include "base/types.h"
#include "symmetry/mo_symmetry.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <numeric>
#include <string>

namespace HartreeFock::Correlation::CASSCF
{

    using CASSCFInternal::kCIStringBits;
    using CASSCFInternal::kMaxSeparateSpinOrbitals;
    using CASSCFInternal::low_bit_mask;
    using CASSCFInternal::single_bit_mask;

    namespace
    {
        std::optional<std::vector<int>> normalize_permutation(
            const std::vector<int> &raw,
            int nbasis)
        {
            if (raw.empty())
                return std::vector<int>{};

            if (static_cast<int>(raw.size()) != nbasis)
                return std::nullopt;

            const bool has_zero = std::find(raw.begin(), raw.end(), 0) != raw.end();
            const int offset = has_zero ? 0 : 1;
            std::vector<bool> seen(static_cast<std::size_t>(nbasis), false);
            std::vector<int> permutation;
            permutation.reserve(raw.size());

            for (const int value : raw)
            {
                const int idx = value - offset;
                if (idx < 0 || idx >= nbasis || seen[static_cast<std::size_t>(idx)])
                    return std::nullopt;
                seen[static_cast<std::size_t>(idx)] = true;
                permutation.push_back(idx);
            }
            return permutation;
        }

        std::unordered_map<std::string, int> combine_irrep_counts(
            const std::vector<HartreeFock::IrrepCount> &counts)
        {
            std::unordered_map<std::string, int> result;
            for (const auto &entry : counts)
            {
                if (entry.irrep.empty() || entry.count < 0)
                    continue;
                result[entry.irrep] += entry.count;
            }
            return result;
        }

        std::vector<int> energy_sorted_indices(
            const Eigen::VectorXd &mo_energies,
            int begin,
            int end)
        {
            constexpr double energy_tie_tol = 1e-10;
            std::vector<int> order;
            order.reserve(static_cast<std::size_t>(std::max(0, end - begin)));
            for (int i = begin; i < end; ++i)
                order.push_back(i);
            std::sort(order.begin(), order.end(), [&](int a, int b)
                      {
                const double ea = mo_energies(a);
                const double eb = mo_energies(b);
                if (std::abs(ea - eb) >= energy_tie_tol)
                    return ea < eb;
                return a < b; });
            return order;
        }

        std::unordered_map<std::string, std::vector<int>> group_indices_by_irrep(
            const Eigen::VectorXd &mo_energies,
            const std::vector<std::string> &mo_symmetry)
        {
            std::unordered_map<std::string, std::vector<int>> grouped;
            const std::vector<int> order = energy_sorted_indices(mo_energies, 0, static_cast<int>(mo_energies.size()));
            for (int idx : order)
                grouped[mo_symmetry[static_cast<std::size_t>(idx)]].push_back(idx);
            return grouped;
        }

        std::vector<int> take_first_unmasked(
            const std::vector<int> &order,
            const std::vector<bool> &mask,
            int count)
        {
            std::vector<int> result;
            result.reserve(static_cast<std::size_t>(std::max(0, count)));
            for (int idx : order)
            {
                if (!mask[static_cast<std::size_t>(idx)])
                    continue;
                result.push_back(idx);
                if (static_cast<int>(result.size()) == count)
                    break;
            }
            return result;
        }

        std::unordered_map<std::string, int> count_labels(
            const std::vector<int> &indices,
            const std::vector<std::string> &mo_symmetry)
        {
            std::unordered_map<std::string, int> counts;
            for (int idx : indices)
                ++counts[mo_symmetry[static_cast<std::size_t>(idx)]];
            return counts;
        }

        std::string format_irrep_count_map(const std::unordered_map<std::string, int> &counts)
        {
            std::vector<std::pair<std::string, int>> items(counts.begin(), counts.end());
            std::sort(items.begin(), items.end(), [](const auto &lhs, const auto &rhs)
                      { return lhs.first < rhs.first; });

            std::string out;
            for (const auto &[label, count] : items)
            {
                if (!out.empty())
                    out += " ";
                out += std::format("{}={}", label, count);
            }
            return out.empty() ? std::string("none") : out;
        }

        std::vector<int> contiguous_range(int begin, int count)
        {
            std::vector<int> indices;
            indices.reserve(static_cast<std::size_t>(std::max(0, count)));
            for (int i = 0; i < count; ++i)
                indices.push_back(begin + i);
            return indices;
        }

        std::expected<ActiveOrbitalSelection, std::string> make_identity_selection(
            int nbasis,
            int n_core,
            int n_act)
        {
            if (n_core < 0 || n_act < 0 || n_core + n_act > nbasis)
                return std::unexpected(std::string("invalid active-space dimensions for orbital permutation"));

            ActiveOrbitalSelection selection;
            selection.permutation.resize(static_cast<std::size_t>(nbasis));
            std::iota(selection.permutation.begin(), selection.permutation.end(), 0);
            selection.active_orbitals.resize(static_cast<std::size_t>(n_act));
            for (int i = 0; i < n_act; ++i)
                selection.active_orbitals[static_cast<std::size_t>(i)] = n_core + i;
            return selection;
        }
    } // namespace

    int count_occupied_below(CIString det, int orb)
    {
        // Count occupied orbitals strictly below `orb`; this is the fermionic sign
        // factor used by the creation and annihilation helpers.
        return std::popcount(det & low_bit_mask(orb));
    }

    int parity_between(CIString s, int lo, int hi)
    {
        // Return the parity of the occupied orbitals between two indices.
        if (lo + 1 >= hi)
            return 1;
        const CIString mask = low_bit_mask(hi - lo - 1) << (lo + 1);
        return (std::popcount(s & mask) % 2 == 0) ? 1 : -1;
    }

    std::vector<CIString> generate_strings(int n_orb, int n_occ)
    {
        std::vector<CIString> result;
        if (n_occ == 0)
        {
            result.push_back(0);
            return result;
        }
        if (n_occ > n_orb || n_orb > kMaxSeparateSpinOrbitals)
            return result;

        // Gosper's hack enumerates all fixed-popcount bitstrings in lexicographic
        // order without constructing combinations explicitly.
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
        if (!(det & bit))
            return {};
        // Removing an electron flips the sign when an odd number of occupied
        // orbitals lie below the annihilated orbital.
        return {det ^ bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
    }

    FermionOpResult apply_creation(CIString det, int orb)
    {
        const CIString bit = single_bit_mask(orb);
        if (det & bit)
            return {};
        // Creation uses the same parity convention as annihilation.
        return {det | bit, (count_occupied_below(det, orb) % 2 == 0) ? 1.0 : -1.0, true};
    }

    std::vector<int> map_mo_irreps(
        const std::vector<std::string> &mo_sym,
        const std::vector<std::string> &names)
    {
        // Match MO symmetry labels to the symmetry-table ordering; unmatched labels
        // stay at -1 so downstream symmetry checks can ignore them safely.
        std::vector<int> idx(mo_sym.size(), -1);
        for (std::size_t i = 0; i < mo_sym.size(); ++i)
            for (std::size_t g = 0; g < names.size(); ++g)
                if (mo_sym[i] == names[g])
                {
                    idx[i] = static_cast<int>(g);
                    break;
                }
        return idx;
    }

    std::expected<ActiveOrbitalSelection, std::string> select_active_orbitals(
        const Eigen::VectorXd &mo_energies,
        const std::vector<std::string> &mo_symmetry,
        int n_core,
        int n_act,
        const std::vector<HartreeFock::IrrepCount> &core_irrep_counts,
        const std::vector<HartreeFock::IrrepCount> &active_irrep_counts,
        const std::vector<int> &mo_permutation)
    {
        const int nbasis = static_cast<int>(mo_energies.size());
        if (nbasis == 0)
            return std::unexpected(std::string("empty MO energy vector"));
        if (n_core < 0 || n_act < 0 || n_core + n_act > nbasis)
            return std::unexpected(std::string("invalid active-space dimensions for orbital permutation"));

        if (!mo_permutation.empty())
        {
            auto normalized = normalize_permutation(mo_permutation, nbasis);
            if (!normalized)
                return std::unexpected(std::string("mo_permutation must be a full permutation of the MO basis"));

            ActiveOrbitalSelection selection;
            selection.permutation = std::move(*normalized);
            selection.active_orbitals.reserve(static_cast<std::size_t>(n_act));
            for (int i = 0; i < n_act; ++i)
                selection.active_orbitals.push_back(selection.permutation[static_cast<std::size_t>(n_core + i)]);
            selection.used_symmetry =
                (!core_irrep_counts.empty() || !active_irrep_counts.empty()) && !mo_symmetry.empty();
            return selection;
        }

        if (static_cast<int>(mo_symmetry.size()) != nbasis)
        {
            if (core_irrep_counts.empty() && active_irrep_counts.empty())
                return make_identity_selection(nbasis, n_core, n_act);
            return std::unexpected(std::string("symmetry labels are required for irrep-based active-space selection"));
        }

        auto core_required = combine_irrep_counts(core_irrep_counts);
        auto active_required = combine_irrep_counts(active_irrep_counts);
        const bool inferred_from_current_order =
            core_irrep_counts.empty() && active_irrep_counts.empty();

        const std::vector<int> full_order = energy_sorted_indices(mo_energies, 0, nbasis);
        const auto irrep_indices = group_indices_by_irrep(mo_energies, mo_symmetry);

        if (inferred_from_current_order)
        {
            core_required = count_labels(contiguous_range(0, n_core), mo_symmetry);
            active_required = count_labels(contiguous_range(n_core, n_act), mo_symmetry);
        }

        const int core_requested =
            std::accumulate(core_required.begin(), core_required.end(), 0,
                            [](int sum, const auto &item)
                            { return sum + item.second; });
        const int active_requested =
            std::accumulate(active_required.begin(), active_required.end(), 0,
                            [](int sum, const auto &item)
                            { return sum + item.second; });
        if (core_requested > n_core)
            return std::unexpected(std::string("core_irrep_counts exceed n_core"));
        if (active_requested > n_act)
            return std::unexpected(std::string("active_irrep_counts exceed n_act"));

        if (!inferred_from_current_order && core_required.empty())
            core_required = count_labels(
                energy_sorted_indices(mo_energies, 0, n_core), mo_symmetry);
        else if (!inferred_from_current_order && core_requested < n_core)
        {
            std::vector<bool> mask(static_cast<std::size_t>(nbasis), true);
            for (const auto &[irrep, _] : core_required)
                for (int idx : irrep_indices.contains(irrep) ? irrep_indices.at(irrep) : std::vector<int>{})
                    mask[static_cast<std::size_t>(idx)] = false;

            const std::vector<int> rest = take_first_unmasked(full_order, mask, n_core - core_requested);
            if (static_cast<int>(rest.size()) != n_core - core_requested)
                return std::unexpected(std::string("unable to satisfy core_irrep_counts with the occupied orbitals"));
            const auto rest_counts = count_labels(rest, mo_symmetry);
            for (const auto &[irrep, count] : rest_counts)
                core_required[irrep] += count;
        }

        if (active_requested < n_act)
        {
            std::vector<bool> mask(static_cast<std::size_t>(nbasis), true);
            for (const auto &[irrep, _] : active_required)
                for (int idx : irrep_indices.contains(irrep) ? irrep_indices.at(irrep) : std::vector<int>{})
                    mask[static_cast<std::size_t>(idx)] = false;

            for (const auto &[irrep, core_count] : core_required)
            {
                const auto it = irrep_indices.find(irrep);
                if (it == irrep_indices.end() || static_cast<int>(it->second.size()) < core_count)
                    return std::unexpected(std::string("unable to satisfy core_irrep_counts with the occupied orbitals"));
                for (int k = 0; k < core_count; ++k)
                    mask[static_cast<std::size_t>(it->second[static_cast<std::size_t>(k)])] = false;
            }

            const std::vector<int> rest = take_first_unmasked(full_order, mask, n_act - active_requested);
            if (static_cast<int>(rest.size()) != n_act - active_requested)
                return std::unexpected(std::string("unable to satisfy active_irrep_counts with the available MO symmetry labels"));
            const auto rest_counts = count_labels(rest, mo_symmetry);
            for (const auto &[irrep, count] : rest_counts)
                active_required[irrep] += count;
        }

        std::vector<int> selected_core;
        std::vector<int> selected_active;
        std::vector<bool> used(static_cast<std::size_t>(nbasis), false);
        selected_core.reserve(static_cast<std::size_t>(n_core));
        selected_active.reserve(static_cast<std::size_t>(n_act));

        for (const auto &[irrep, core_count] : core_required)
        {
            if (core_count == 0)
                continue;
            const auto it = irrep_indices.find(irrep);
            if (it == irrep_indices.end() || static_cast<int>(it->second.size()) < core_count)
                return std::unexpected(std::string("unable to satisfy core_irrep_counts with the occupied orbitals"));
            for (int k = 0; k < core_count; ++k)
            {
                const int idx = it->second[static_cast<std::size_t>(k)];
                used[static_cast<std::size_t>(idx)] = true;
                selected_core.push_back(idx);
            }
        }

        for (const auto &[irrep, active_count] : active_required)
        {
            if (active_count == 0)
                continue;
            const int core_count = core_required.contains(irrep) ? core_required[irrep] : 0;
            const auto it = irrep_indices.find(irrep);
            if (it == irrep_indices.end() || static_cast<int>(it->second.size()) < core_count + active_count)
                return std::unexpected(std::string("unable to satisfy active_irrep_counts with the available MO symmetry labels"));
            for (int k = 0; k < active_count; ++k)
            {
                const int idx = it->second[static_cast<std::size_t>(core_count + k)];
                if (used[static_cast<std::size_t>(idx)])
                    return std::unexpected(std::string("core_irrep_counts and active_irrep_counts overlap"));
                used[static_cast<std::size_t>(idx)] = true;
                selected_active.push_back(idx);
            }
        }

        if (static_cast<int>(selected_core.size()) != n_core)
            return std::unexpected(std::format(
                "unable to assemble a full core block from the requested symmetry counts (picked {} of {}, core counts: {}).",
                static_cast<int>(selected_core.size()), n_core, format_irrep_count_map(core_required)));
        if (static_cast<int>(selected_active.size()) != n_act)
            return std::unexpected(std::format(
                "unable to assemble a full active block from the requested symmetry counts (picked {} of {}, core counts: {}, active counts: {}).",
                static_cast<int>(selected_active.size()), n_act,
                format_irrep_count_map(core_required),
                format_irrep_count_map(active_required)));

        std::sort(selected_core.begin(), selected_core.end());
        std::sort(selected_active.begin(), selected_active.end());

        ActiveOrbitalSelection selection;
        selection.permutation.reserve(static_cast<std::size_t>(nbasis));
        selection.permutation.insert(selection.permutation.end(), selected_core.begin(), selected_core.end());
        selection.permutation.insert(selection.permutation.end(), selected_active.begin(), selected_active.end());
        for (int idx = 0; idx < nbasis; ++idx)
        {
            if (used[static_cast<std::size_t>(idx)])
                continue;
            selection.permutation.push_back(idx);
        }
        selection.active_orbitals = std::move(selected_active);
        selection.used_symmetry = true;
        return selection;
    }

    std::expected<Eigen::MatrixXd, std::string> reorder_mo_coefficients(
        const Eigen::MatrixXd &mo_coefficients,
        const std::vector<int> &permutation)
    {
        if (mo_coefficients.cols() == 0 || permutation.empty())
            return mo_coefficients;
        if (static_cast<int>(permutation.size()) != mo_coefficients.cols())
            return std::unexpected(std::string("MO permutation size does not match the coefficient matrix"));

        Eigen::MatrixXd reordered(mo_coefficients.rows(), mo_coefficients.cols());
        for (int i = 0; i < static_cast<int>(permutation.size()); ++i)
        {
            const int src = permutation[static_cast<std::size_t>(i)];
            if (src < 0 || src >= mo_coefficients.cols())
                return std::unexpected(std::string("MO permutation contains an out-of-range index"));
            reordered.col(i) = mo_coefficients.col(src);
        }
        return reordered;
    }

    std::optional<SymmetryContext> build_symmetry_context(HartreeFock::Calculator &calc)
    {
        // Build the minimal symmetry context only when the point group supports a
        // one-dimensional abelian product table.
        auto product_table = HartreeFock::Symmetry::build_abelian_irrep_product_table(calc);
        if (!product_table || !product_table->valid || product_table->irrep_names.empty())
            return std::nullopt;

        SymmetryContext ctx;
        ctx.names = std::move(product_table->irrep_names);
        ctx.product = std::move(product_table->product);
        ctx.abelian_1d_only = true;

        int identity = -1;
        for (int g = 0; g < static_cast<int>(ctx.product.size()); ++g)
        {
            // Find the totally symmetric irrep by testing which row/column acts as
            // a multiplicative identity in the product table.
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
        const std::string &s,
        const SymmetryContext &sym_ctx)
    {
        // An empty target string means "use the totally symmetric irrep".
        if (s.empty())
            return sym_ctx.totally_symmetric_irrep;
        for (std::size_t g = 0; g < sym_ctx.names.size(); ++g)
            if (s == sym_ctx.names[g])
                return static_cast<int>(g);
        return std::nullopt;
    }

    bool point_group_has_only_1d_irreps(const std::string &pg)
    {
        // Only point groups with 1D irreps are accepted here; anything else would
        // need a richer symmetry implementation than this code path assumes.
        if (pg == "C1" || pg == "Ci" || pg == "Cs")
            return true;
        if (pg.find("inf") != std::string::npos || pg.size() < 2)
            return false;

        auto parse_order = [&](std::size_t pos, int &n, std::string &suffix) -> bool
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
            if (!parse_order(1, n, suffix))
                return false;
            return (suffix.empty() || suffix == "h" || suffix == "v") && n <= 2;
        case 'D':
            if (!parse_order(1, n, suffix))
                return false;
            if (suffix.empty() || suffix == "h")
                return n <= 2;
            if (suffix == "d")
                return n <= 1;
            return false;
        case 'S':
            if (!parse_order(1, n, suffix))
                return false;
            return suffix.empty() && n == 2;
        default:
            return false;
        }
    }

    void build_spin_strings_unfiltered(
        int n_act,
        int n_alpha,
        int n_beta,
        std::vector<CIString> &a_strs,
        std::vector<CIString> &b_strs)
    {
        // Build separate alpha and beta string lists without applying symmetry or
        // RAS filtering at this stage.
        a_strs = generate_strings(n_act, n_alpha);
        b_strs = generate_strings(n_act, n_beta);
    }

    std::vector<CIString> build_spin_dets(
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act)
    {
        // Pack each determinant pair into one spin-string key for fast lookup in
        // the CI Hamiltonian application code.
        std::vector<CIString> sd;
        sd.reserve(dets.size());
        for (auto [ia, ib] : dets)
            sd.push_back(a_strs[ia] | ((n_act >= kCIStringBits) ? 0 : (b_strs[ib] << n_act)));
        return sd;
    }

    std::unordered_map<CIString, int> build_det_lookup(const std::vector<CIString> &sd)
    {
        // Precompute the reverse map from packed determinant to CI index.
        std::unordered_map<CIString, int> lut;
        lut.reserve(sd.size());
        for (int i = 0; i < static_cast<int>(sd.size()); ++i)
            lut.emplace(sd[i], i);
        return lut;
    }

} // namespace HartreeFock::Correlation::CASSCF
