#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr int KIND_CREATE = 0;
constexpr int KIND_ANNIHILATE = 1;

constexpr int SPACE_OCC = 0;
constexpr int SPACE_VIR = 1;
constexpr int SPACE_GEN = 2;

constexpr std::uint32_t POS_MASK = (1u << 16) - 1u;
constexpr int KIND_SHIFT = 16;
constexpr int SPACE_SHIFT = 18;
constexpr int BLOCK_SHIFT = 20;
constexpr std::uint32_t EDGE_LOW_MASK = (1u << 20) - 1u;

int word_kind(std::uint32_t word) {
    return static_cast<int>((word >> KIND_SHIFT) & 0x3u);
}

int word_space(std::uint32_t word) {
    return static_cast<int>((word >> SPACE_SHIFT) & 0x3u);
}

int word_block(std::uint32_t word) {
    return static_cast<int>(word >> BLOCK_SHIFT);
}

std::uint32_t pack_edge(int a, int b) {
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (hi << BLOCK_SHIFT) | lo;
}

std::pair<int, int> unpack_edge(std::uint32_t edge) {
    return {
        static_cast<int>(edge & EDGE_LOW_MASK),
        static_cast<int>(edge >> BLOCK_SHIFT),
    };
}

bool can_contract_codes(
    int left_kind,
    int left_space,
    int right_kind,
    int right_space
) {
    if (left_kind == KIND_CREATE && right_kind == KIND_ANNIHILATE) {
        return (left_space == SPACE_OCC || left_space == SPACE_GEN)
            && (right_space == SPACE_OCC || right_space == SPACE_GEN);
    }
    if (left_kind == KIND_ANNIHILATE && right_kind == KIND_CREATE) {
        return (left_space == SPACE_VIR || left_space == SPACE_GEN)
            && (right_space == SPACE_VIR || right_space == SPACE_GEN);
    }
    return false;
}

bool signature_positions_can_contract(
    const std::vector<std::uint32_t> &signature,
    Py_ssize_t left_pos,
    Py_ssize_t right_pos
) {
    const auto lo = std::min(left_pos, right_pos);
    const auto hi = std::max(left_pos, right_pos);
    const auto left = signature[static_cast<std::size_t>(lo)];
    const auto right = signature[static_cast<std::size_t>(hi)];
    return can_contract_codes(
        word_kind(left),
        word_space(left),
        word_kind(right),
        word_space(right)
    );
}

bool can_fully_contract(const std::vector<std::uint32_t> &signature) {
    if (signature.size() % 2 != 0) {
        return false;
    }

    int create_occ = 0;
    int annihilate_occ = 0;
    int annihilate_vir = 0;
    int create_vir = 0;
    int gen_count = 0;

    for (auto word : signature) {
        const auto kind = word_kind(word);
        const auto space = word_space(word);
        if (space == SPACE_GEN) {
            ++gen_count;
        } else if (kind == KIND_CREATE && space == SPACE_OCC) {
            ++create_occ;
        } else if (kind == KIND_ANNIHILATE && space == SPACE_OCC) {
            ++annihilate_occ;
        } else if (kind == KIND_ANNIHILATE && space == SPACE_VIR) {
            ++annihilate_vir;
        } else if (kind == KIND_CREATE && space == SPACE_VIR) {
            ++create_vir;
        }
    }

    const auto occ_deficit = std::abs(create_occ - annihilate_occ);
    const auto vir_deficit = std::abs(annihilate_vir - create_vir);
    return occ_deficit + vir_deficit <= gen_count;
}

struct UnionFind {
    explicit UnionFind(int n) : parent(static_cast<std::size_t>(n)) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    }

    void unite(int a, int b) {
        const auto ra = find(a);
        const auto rb = find(b);
        if (ra != rb) {
            parent[static_cast<std::size_t>(rb)] = ra;
        }
    }

    std::vector<int> parent;
};

bool edges_are_connected(
    int n_blocks,
    const std::vector<std::uint32_t> &edges,
    int ignore_block
) {
    std::vector<int> blocks;
    blocks.reserve(static_cast<std::size_t>(n_blocks));
    for (int block = 0; block < n_blocks; ++block) {
        if (block != ignore_block) {
            blocks.push_back(block);
        }
    }
    if (blocks.size() <= 1) {
        return true;
    }

    std::unordered_map<int, int> remap;
    remap.reserve(blocks.size());
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(blocks.size()); ++i) {
        remap.emplace(blocks[static_cast<std::size_t>(i)], static_cast<int>(i));
    }

    UnionFind uf(static_cast<int>(blocks.size()));
    for (auto edge : edges) {
        auto [a, b] = unpack_edge(edge);
        auto ita = remap.find(a);
        auto itb = remap.find(b);
        if (ita != remap.end() && itb != remap.end()) {
            uf.unite(ita->second, itb->second);
        }
    }

    const auto root = uf.find(0);
    for (int i = 1; i < static_cast<int>(blocks.size()); ++i) {
        if (uf.find(i) != root) {
            return false;
        }
    }
    return true;
}

bool can_still_be_connected(
    const std::vector<std::uint32_t> &signature,
    const std::vector<std::uint32_t> &current_edges,
    int n_blocks,
    int ignore_block
) {
    std::vector<int> blocks;
    for (int block = 0; block < n_blocks; ++block) {
        if (block != ignore_block) {
            blocks.push_back(block);
        }
    }
    if (blocks.size() <= 1) {
        return true;
    }

    std::unordered_map<int, int> block_index;
    block_index.reserve(blocks.size());
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(blocks.size()); ++i) {
        block_index.emplace(blocks[static_cast<std::size_t>(i)], static_cast<int>(i));
    }

    UnionFind current_components(static_cast<int>(blocks.size()));
    for (auto edge : current_edges) {
        auto [a, b] = unpack_edge(edge);
        auto ita = block_index.find(a);
        auto itb = block_index.find(b);
        if (ita != block_index.end() && itb != block_index.end()) {
            current_components.unite(ita->second, itb->second);
        }
    }

    std::unordered_map<int, int> component_remap;
    std::unordered_map<int, int> component_of_block;
    int next_component = 0;
    for (auto block : blocks) {
        const auto root = current_components.find(block_index[block]);
        auto [it, inserted] = component_remap.emplace(root, next_component);
        if (inserted) {
            ++next_component;
        }
        component_of_block.emplace(block, it->second);
    }

    const auto n_components = next_component;
    if (n_components <= 1) {
        return true;
    }

    std::unordered_set<std::uint32_t> candidate_edges(
        current_edges.begin(),
        current_edges.end()
    );
    std::unordered_set<std::uint32_t> component_edges;
    std::vector<int> component_stub_counts(static_cast<std::size_t>(n_components), 0);
    std::vector<char> external_capable(signature.size(), 0);

    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(signature.size()); ++i) {
        for (Py_ssize_t j = i + 1; j < static_cast<Py_ssize_t>(signature.size()); ++j) {
            const auto left = signature[static_cast<std::size_t>(i)];
            const auto right = signature[static_cast<std::size_t>(j)];
            const auto left_block = word_block(left);
            const auto right_block = word_block(right);
            if (left_block == right_block) {
                continue;
            }
            if (!signature_positions_can_contract(signature, i, j)) {
                continue;
            }

            candidate_edges.insert(pack_edge(left_block, right_block));

            const auto left_component = component_of_block.find(left_block);
            const auto right_component = component_of_block.find(right_block);
            if (left_component == component_of_block.end()
                || right_component == component_of_block.end()) {
                continue;
            }
            if (left_component->second != right_component->second) {
                component_edges.insert(
                    pack_edge(left_component->second, right_component->second)
                );
                external_capable[static_cast<std::size_t>(i)] = 1;
                external_capable[static_cast<std::size_t>(j)] = 1;
            }
        }
    }

    if (component_edges.empty()) {
        return false;
    }

    std::vector<std::uint32_t> component_edge_vec(
        component_edges.begin(), component_edges.end()
    );
    if (!edges_are_connected(n_components, component_edge_vec, -1)) {
        return false;
    }

    for (Py_ssize_t pos = 0; pos < static_cast<Py_ssize_t>(signature.size()); ++pos) {
        if (!external_capable[static_cast<std::size_t>(pos)]) {
            continue;
        }
        const auto block = word_block(signature[static_cast<std::size_t>(pos)]);
        auto it = component_of_block.find(block);
        if (it != component_of_block.end()) {
            component_stub_counts[static_cast<std::size_t>(it->second)] += 1;
        }
    }

    int stub_total = 0;
    for (auto count : component_stub_counts) {
        if (count == 0) {
            return false;
        }
        stub_total += count;
    }
    if (stub_total < 2 * (n_components - 1)) {
        return false;
    }

    std::vector<std::uint32_t> candidate_edge_vec(
        candidate_edges.begin(), candidate_edges.end()
    );
    return edges_are_connected(n_blocks, candidate_edge_vec, ignore_block);
}

PyObject *analyze_signature(PyObject *, PyObject *args) {
    PyObject *signature_obj = nullptr;
    PyObject *edges_obj = nullptr;
    int n_blocks = -1;
    int ignore_block = -1;

    if (!PyArg_ParseTuple(
            args, "OOii", &signature_obj, &edges_obj, &n_blocks, &ignore_block
        )) {
        return nullptr;
    }

    PyObject *signature_fast = PySequence_Fast(
        signature_obj, "signature must be a sequence of packed ints"
    );
    if (signature_fast == nullptr) {
        return nullptr;
    }
    PyObject *edges_fast = PySequence_Fast(
        edges_obj, "edges must be a sequence of packed ints"
    );
    if (edges_fast == nullptr) {
        Py_DECREF(signature_fast);
        return nullptr;
    }

    std::vector<std::uint32_t> signature;
    signature.reserve(static_cast<std::size_t>(PySequence_Fast_GET_SIZE(signature_fast)));
    auto **signature_items = PySequence_Fast_ITEMS(signature_fast);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(signature_fast); ++i) {
        const auto value = PyLong_AsUnsignedLong(signature_items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(signature_fast);
            Py_DECREF(edges_fast);
            return nullptr;
        }
        signature.push_back(static_cast<std::uint32_t>(value));
    }

    std::vector<std::uint32_t> edges;
    edges.reserve(static_cast<std::size_t>(PySequence_Fast_GET_SIZE(edges_fast)));
    auto **edge_items = PySequence_Fast_ITEMS(edges_fast);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(edges_fast); ++i) {
        const auto value = PyLong_AsUnsignedLong(edge_items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(signature_fast);
            Py_DECREF(edges_fast);
            return nullptr;
        }
        edges.push_back(static_cast<std::uint32_t>(value));
    }

    Py_DECREF(signature_fast);
    Py_DECREF(edges_fast);

    if (!can_fully_contract(signature)) {
        return Py_BuildValue("(NiN)", Py_False, -1, PyTuple_New(0));
    }

    if (n_blocks >= 0
        && !can_still_be_connected(signature, edges, n_blocks, ignore_block)) {
        return Py_BuildValue("(NiN)", Py_False, -1, PyTuple_New(0));
    }

    int best_pos = -1;
    int best_count = 0;
    int best_span = 0;
    std::vector<Py_ssize_t> best_candidates;

    for (Py_ssize_t pos = 0; pos < static_cast<Py_ssize_t>(signature.size()); ++pos) {
        const auto pivot = signature[static_cast<std::size_t>(pos)];
        std::vector<Py_ssize_t> candidates;
        for (Py_ssize_t partner = 0;
             partner < static_cast<Py_ssize_t>(signature.size());
             ++partner) {
            if (partner == pos) {
                continue;
            }
            const auto other = signature[static_cast<std::size_t>(partner)];
            if (word_block(pivot) == word_block(other)) {
                continue;
            }
            if (signature_positions_can_contract(signature, pos, partner)) {
                candidates.push_back(partner);
            }
        }

        if (candidates.empty()) {
            return Py_BuildValue("(NiN)", Py_False, -1, PyTuple_New(0));
        }

        auto min_gap = static_cast<int>(signature.size());
        for (auto partner : candidates) {
            min_gap = std::min(min_gap, static_cast<int>(std::abs(partner - pos)));
        }

        if (best_pos < 0
            || static_cast<int>(candidates.size()) < best_count
            || (static_cast<int>(candidates.size()) == best_count && min_gap < best_span)
            || (static_cast<int>(candidates.size()) == best_count
                && min_gap == best_span
                && pos < best_pos)) {
            best_pos = static_cast<int>(pos);
            best_count = static_cast<int>(candidates.size());
            best_span = min_gap;
            best_candidates = std::move(candidates);
        }
    }

    PyObject *candidate_tuple = PyTuple_New(static_cast<Py_ssize_t>(best_candidates.size()));
    if (candidate_tuple == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(best_candidates.size()); ++i) {
        PyTuple_SET_ITEM(
            candidate_tuple,
            i,
            PyLong_FromSsize_t(best_candidates[static_cast<std::size_t>(i)])
        );
    }
    return Py_BuildValue("(NiN)", Py_True, best_pos, candidate_tuple);
}

bool lex_less(
    const std::vector<int> &lhs,
    const std::vector<int> &rhs
) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

PyObject *canonicalize_tensor_layout(PyObject *, PyObject *args) {
    PyObject *codes_obj = nullptr;
    PyObject *groups_obj = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &codes_obj, &groups_obj)) {
        return nullptr;
    }

    PyObject *codes_fast = PySequence_Fast(
        codes_obj, "codes must be a sequence of ints"
    );
    if (codes_fast == nullptr) {
        return nullptr;
    }
    PyObject *groups_fast = PySequence_Fast(
        groups_obj, "groups must be a sequence"
    );
    if (groups_fast == nullptr) {
        Py_DECREF(codes_fast);
        return nullptr;
    }

    std::vector<int> codes;
    codes.reserve(static_cast<std::size_t>(PySequence_Fast_GET_SIZE(codes_fast)));
    auto **code_items = PySequence_Fast_ITEMS(codes_fast);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(codes_fast); ++i) {
        const auto value = PyLong_AsLong(code_items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(codes_fast);
            Py_DECREF(groups_fast);
            return nullptr;
        }
        codes.push_back(static_cast<int>(value));
    }

    std::vector<int> order(codes.size());
    std::iota(order.begin(), order.end(), 0);
    int total_sign = 1;

    auto **group_items = PySequence_Fast_ITEMS(groups_fast);
    for (Py_ssize_t g = 0; g < PySequence_Fast_GET_SIZE(groups_fast); ++g) {
        PyObject *group_fast = PySequence_Fast(
            group_items[g], "antisymmetry group must be a sequence"
        );
        if (group_fast == nullptr) {
            Py_DECREF(codes_fast);
            Py_DECREF(groups_fast);
            return nullptr;
        }

        std::vector<int> positions;
        positions.reserve(static_cast<std::size_t>(PySequence_Fast_GET_SIZE(group_fast)));
        auto **pos_items = PySequence_Fast_ITEMS(group_fast);
        std::unordered_set<int> seen_codes;
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(group_fast); ++i) {
            const auto pos = static_cast<int>(PyLong_AsLong(pos_items[i]));
            if (PyErr_Occurred()) {
                Py_DECREF(group_fast);
                Py_DECREF(codes_fast);
                Py_DECREF(groups_fast);
                return nullptr;
            }
            positions.push_back(pos);
            const auto code = codes[static_cast<std::size_t>(pos)];
            if (!seen_codes.insert(code).second) {
                Py_DECREF(group_fast);
                Py_DECREF(codes_fast);
                Py_DECREF(groups_fast);
                return Py_BuildValue("(NiN)", Py_True, 0, PyTuple_New(0));
            }
        }

        std::vector<int> perm(positions.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::vector<int> best_perm = perm;
        std::vector<int> best_codes;
        best_codes.reserve(positions.size());
        for (auto idx : perm) {
            best_codes.push_back(codes[static_cast<std::size_t>(positions[static_cast<std::size_t>(idx)])]);
        }
        int best_sign = 1;

        while (std::next_permutation(perm.begin(), perm.end())) {
            std::vector<int> perm_codes;
            perm_codes.reserve(positions.size());
            for (auto idx : perm) {
                perm_codes.push_back(codes[static_cast<std::size_t>(positions[static_cast<std::size_t>(idx)])]);
            }
            if (lex_less(perm_codes, best_codes)) {
                best_codes = perm_codes;
                best_perm = perm;

                int inversions = 0;
                for (std::size_t i = 0; i < perm.size(); ++i) {
                    for (std::size_t j = i + 1; j < perm.size(); ++j) {
                        if (perm[i] > perm[j]) {
                            ++inversions;
                        }
                    }
                }
                best_sign = (inversions % 2 == 0) ? 1 : -1;
            }
        }

        std::vector<int> reordered_positions(positions.size());
        for (std::size_t i = 0; i < positions.size(); ++i) {
            reordered_positions[i] = order[static_cast<std::size_t>(
                positions[static_cast<std::size_t>(best_perm[i])]
            )];
            codes[static_cast<std::size_t>(positions[i])] =
                best_codes[static_cast<std::size_t>(i)];
        }
        for (std::size_t i = 0; i < positions.size(); ++i) {
            order[static_cast<std::size_t>(positions[i])] =
                reordered_positions[static_cast<std::size_t>(i)];
        }
        total_sign *= best_sign;
        Py_DECREF(group_fast);
    }

    Py_DECREF(codes_fast);
    Py_DECREF(groups_fast);

    PyObject *order_tuple = PyTuple_New(static_cast<Py_ssize_t>(order.size()));
    if (order_tuple == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(order.size()); ++i) {
        PyTuple_SET_ITEM(order_tuple, i, PyLong_FromLong(order[static_cast<std::size_t>(i)]));
    }
    return Py_BuildValue("(NiN)", Py_False, total_sign, order_tuple);
}

PyObject *assign_dummy_ordinals(PyObject *, PyObject *args) {
    PyObject *spaces_obj = nullptr;
    PyObject *free_mask_obj = nullptr;
    PyObject *occ_reserved_obj = nullptr;
    PyObject *vir_reserved_obj = nullptr;
    PyObject *gen_reserved_obj = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "OOOOO",
            &spaces_obj,
            &free_mask_obj,
            &occ_reserved_obj,
            &vir_reserved_obj,
            &gen_reserved_obj
        )) {
        return nullptr;
    }

    auto load_int_sequence = [](PyObject *obj, const char *message) -> PyObject * {
        return PySequence_Fast(obj, message);
    };

    PyObject *spaces_fast = load_int_sequence(spaces_obj, "spaces must be a sequence");
    if (spaces_fast == nullptr) {
        return nullptr;
    }
    PyObject *free_fast = load_int_sequence(free_mask_obj, "free mask must be a sequence");
    if (free_fast == nullptr) {
        Py_DECREF(spaces_fast);
        return nullptr;
    }
    PyObject *occ_fast = load_int_sequence(occ_reserved_obj, "occ reserved must be a sequence");
    if (occ_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(free_fast);
        return nullptr;
    }
    PyObject *vir_fast = load_int_sequence(vir_reserved_obj, "vir reserved must be a sequence");
    if (vir_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(free_fast);
        Py_DECREF(occ_fast);
        return nullptr;
    }
    PyObject *gen_fast = load_int_sequence(gen_reserved_obj, "gen reserved must be a sequence");
    if (gen_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(free_fast);
        Py_DECREF(occ_fast);
        Py_DECREF(vir_fast);
        return nullptr;
    }

    std::unordered_set<int> reserved[3];
    PyObject *reserved_fast[3] = {occ_fast, vir_fast, gen_fast};
    for (int space = 0; space < 3; ++space) {
        auto **items = PySequence_Fast_ITEMS(reserved_fast[space]);
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(reserved_fast[space]); ++i) {
            const auto value = PyLong_AsLong(items[i]);
            if (PyErr_Occurred()) {
                Py_DECREF(spaces_fast);
                Py_DECREF(free_fast);
                Py_DECREF(occ_fast);
                Py_DECREF(vir_fast);
                Py_DECREF(gen_fast);
                return nullptr;
            }
            reserved[space].insert(static_cast<int>(value));
        }
    }

    std::vector<int> next_slot = {0, 0, 0};
    PyObject *result = PyTuple_New(PySequence_Fast_GET_SIZE(spaces_fast));
    if (result == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(free_fast);
        Py_DECREF(occ_fast);
        Py_DECREF(vir_fast);
        Py_DECREF(gen_fast);
        return nullptr;
    }

    auto **space_items = PySequence_Fast_ITEMS(spaces_fast);
    auto **free_items = PySequence_Fast_ITEMS(free_fast);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(spaces_fast); ++i) {
        const auto is_free = PyObject_IsTrue(free_items[i]);
        if (is_free < 0) {
            Py_DECREF(result);
            Py_DECREF(spaces_fast);
            Py_DECREF(free_fast);
            Py_DECREF(occ_fast);
            Py_DECREF(vir_fast);
            Py_DECREF(gen_fast);
            return nullptr;
        }
        if (is_free) {
            PyTuple_SET_ITEM(result, i, PyLong_FromLong(-1));
            continue;
        }

        const auto space = static_cast<int>(PyLong_AsLong(space_items[i]));
        if (PyErr_Occurred()) {
            Py_DECREF(result);
            Py_DECREF(spaces_fast);
            Py_DECREF(free_fast);
            Py_DECREF(occ_fast);
            Py_DECREF(vir_fast);
            Py_DECREF(gen_fast);
            return nullptr;
        }

        while (reserved[space].count(next_slot[space]) != 0) {
            ++next_slot[space];
        }
        const auto assigned = next_slot[space];
        reserved[space].insert(assigned);
        ++next_slot[space];
        PyTuple_SET_ITEM(result, i, PyLong_FromLong(assigned));
    }

    Py_DECREF(spaces_fast);
    Py_DECREF(free_fast);
    Py_DECREF(occ_fast);
    Py_DECREF(vir_fast);
    Py_DECREF(gen_fast);
    return result;
}

PyObject *apply_deltas_layout(PyObject *, PyObject *args) {
    PyObject *spaces_obj = nullptr;
    PyObject *name_order_obj = nullptr;
    PyObject *dummy_mask_obj = nullptr;
    PyObject *protected_rank_obj = nullptr;
    PyObject *pairs_obj = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "OOOOO",
            &spaces_obj,
            &name_order_obj,
            &dummy_mask_obj,
            &protected_rank_obj,
            &pairs_obj
        )) {
        return nullptr;
    }

    auto *spaces_fast = PySequence_Fast(spaces_obj, "spaces must be a sequence");
    if (spaces_fast == nullptr) {
        return nullptr;
    }
    auto *name_fast = PySequence_Fast(
        name_order_obj, "name order must be a sequence"
    );
    if (name_fast == nullptr) {
        Py_DECREF(spaces_fast);
        return nullptr;
    }
    auto *dummy_fast = PySequence_Fast(
        dummy_mask_obj, "dummy mask must be a sequence"
    );
    if (dummy_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(name_fast);
        return nullptr;
    }
    auto *protected_fast = PySequence_Fast(
        protected_rank_obj, "protected ranks must be a sequence"
    );
    if (protected_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(name_fast);
        Py_DECREF(dummy_fast);
        return nullptr;
    }
    auto *pairs_fast = PySequence_Fast(
        pairs_obj, "delta pairs must be a sequence"
    );
    if (pairs_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(name_fast);
        Py_DECREF(dummy_fast);
        Py_DECREF(protected_fast);
        return nullptr;
    }

    const auto n = PySequence_Fast_GET_SIZE(spaces_fast);
    if (PySequence_Fast_GET_SIZE(name_fast) != n
        || PySequence_Fast_GET_SIZE(dummy_fast) != n
        || PySequence_Fast_GET_SIZE(protected_fast) != n) {
        Py_DECREF(spaces_fast);
        Py_DECREF(name_fast);
        Py_DECREF(dummy_fast);
        Py_DECREF(protected_fast);
        Py_DECREF(pairs_fast);
        PyErr_SetString(PyExc_ValueError, "delta layout sequences must match length");
        return nullptr;
    }

    std::vector<int> spaces(static_cast<std::size_t>(n));
    std::vector<int> names(static_cast<std::size_t>(n));
    std::vector<char> dummy_mask(static_cast<std::size_t>(n), 0);
    std::vector<int> protected_rank(static_cast<std::size_t>(n), -1);
    std::vector<int> parent(static_cast<std::size_t>(n));
    std::iota(parent.begin(), parent.end(), 0);

    auto **space_items = PySequence_Fast_ITEMS(spaces_fast);
    auto **name_items = PySequence_Fast_ITEMS(name_fast);
    auto **dummy_items = PySequence_Fast_ITEMS(dummy_fast);
    auto **protected_items = PySequence_Fast_ITEMS(protected_fast);

    for (Py_ssize_t i = 0; i < n; ++i) {
        spaces[static_cast<std::size_t>(i)] = static_cast<int>(
            PyLong_AsLong(space_items[i])
        );
        names[static_cast<std::size_t>(i)] = static_cast<int>(
            PyLong_AsLong(name_items[i])
        );
        protected_rank[static_cast<std::size_t>(i)] = static_cast<int>(
            PyLong_AsLong(protected_items[i])
        );
        const auto is_dummy = PyObject_IsTrue(dummy_items[i]);
        if (PyErr_Occurred() || is_dummy < 0) {
            Py_DECREF(spaces_fast);
            Py_DECREF(name_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(protected_fast);
            Py_DECREF(pairs_fast);
            return nullptr;
        }
        dummy_mask[static_cast<std::size_t>(i)] = static_cast<char>(is_dummy != 0);
        if (PyErr_Occurred()) {
            Py_DECREF(spaces_fast);
            Py_DECREF(name_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(protected_fast);
            Py_DECREF(pairs_fast);
            return nullptr;
        }
    }

    std::function<int(int)> find = [&](int x) {
        while (parent[static_cast<std::size_t>(x)] != x) {
            parent[static_cast<std::size_t>(x)] =
                parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
            x = parent[static_cast<std::size_t>(x)];
        }
        return x;
    };

    auto prefer = [&](int lhs, int rhs) -> std::pair<int, int> {
        const auto lhs_protected = protected_rank[static_cast<std::size_t>(lhs)] >= 0;
        const auto rhs_protected = protected_rank[static_cast<std::size_t>(rhs)] >= 0;

        if (lhs_protected != rhs_protected) {
            return lhs_protected ? std::make_pair(lhs, rhs) : std::make_pair(rhs, lhs);
        }

        if (lhs_protected && rhs_protected) {
            return protected_rank[static_cast<std::size_t>(lhs)]
                    <= protected_rank[static_cast<std::size_t>(rhs)]
                ? std::make_pair(lhs, rhs)
                : std::make_pair(rhs, lhs);
        }

        if (spaces[static_cast<std::size_t>(rhs)] != SPACE_GEN
            && spaces[static_cast<std::size_t>(lhs)] == SPACE_GEN) {
            return {rhs, lhs};
        }
        if (spaces[static_cast<std::size_t>(lhs)] == spaces[static_cast<std::size_t>(rhs)]
            && names[static_cast<std::size_t>(rhs)] < names[static_cast<std::size_t>(lhs)]) {
            return {rhs, lhs};
        }
        return {lhs, rhs};
    };

    auto **pair_items = PySequence_Fast_ITEMS(pairs_fast);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(pairs_fast); ++i) {
        auto *pair_fast = PySequence_Fast(pair_items[i], "delta pair must be a length-2 sequence");
        if (pair_fast == nullptr) {
            Py_DECREF(spaces_fast);
            Py_DECREF(name_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(protected_fast);
            Py_DECREF(pairs_fast);
            return nullptr;
        }
        if (PySequence_Fast_GET_SIZE(pair_fast) != 2) {
            Py_DECREF(pair_fast);
            Py_DECREF(spaces_fast);
            Py_DECREF(name_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(protected_fast);
            Py_DECREF(pairs_fast);
            PyErr_SetString(PyExc_ValueError, "delta pair must have length 2");
            return nullptr;
        }

        auto **items = PySequence_Fast_ITEMS(pair_fast);
        const auto lhs = static_cast<int>(PyLong_AsLong(items[0]));
        const auto rhs = static_cast<int>(PyLong_AsLong(items[1]));
        Py_DECREF(pair_fast);
        if (PyErr_Occurred()) {
            Py_DECREF(spaces_fast);
            Py_DECREF(name_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(protected_fast);
            Py_DECREF(pairs_fast);
            return nullptr;
        }

        auto lhs_root = find(lhs);
        auto rhs_root = find(rhs);
        if (lhs_root == rhs_root) {
            continue;
        }
        const auto lhs_space = spaces[static_cast<std::size_t>(lhs_root)];
        const auto rhs_space = spaces[static_cast<std::size_t>(rhs_root)];
        if (lhs_space != SPACE_GEN && rhs_space != SPACE_GEN && lhs_space != rhs_space) {
            Py_DECREF(spaces_fast);
            Py_DECREF(name_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(protected_fast);
            Py_DECREF(pairs_fast);
            return Py_BuildValue("(NN)", Py_False, PyTuple_New(0));
        }

        auto [keep, drop] = prefer(lhs_root, rhs_root);
        parent[static_cast<std::size_t>(drop)] = keep;
    }

    Py_DECREF(spaces_fast);
    Py_DECREF(name_fast);
    Py_DECREF(dummy_fast);
    Py_DECREF(protected_fast);
    Py_DECREF(pairs_fast);

    PyObject *roots = PyTuple_New(n);
    if (roots == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyTuple_SET_ITEM(roots, i, PyLong_FromLong(find(static_cast<int>(i))));
    }
    return Py_BuildValue("(NN)", Py_True, roots);
}

PyObject *classify_summed_indices(PyObject *, PyObject *args) {
    PyObject *spaces_obj = nullptr;
    PyObject *name_order_obj = nullptr;
    PyObject *dummy_mask_obj = nullptr;
    PyObject *free_mask_obj = nullptr;

    if (!PyArg_ParseTuple(
            args,
            "OOOO",
            &spaces_obj,
            &name_order_obj,
            &dummy_mask_obj,
            &free_mask_obj
        )) {
        return nullptr;
    }

    auto *spaces_fast = PySequence_Fast(spaces_obj, "spaces must be a sequence");
    if (spaces_fast == nullptr) {
        return nullptr;
    }
    auto *names_fast = PySequence_Fast(name_order_obj, "name order must be a sequence");
    if (names_fast == nullptr) {
        Py_DECREF(spaces_fast);
        return nullptr;
    }
    auto *dummy_fast = PySequence_Fast(dummy_mask_obj, "dummy mask must be a sequence");
    if (dummy_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(names_fast);
        return nullptr;
    }
    auto *free_fast = PySequence_Fast(free_mask_obj, "free mask must be a sequence");
    if (free_fast == nullptr) {
        Py_DECREF(spaces_fast);
        Py_DECREF(names_fast);
        Py_DECREF(dummy_fast);
        return nullptr;
    }

    const auto n = PySequence_Fast_GET_SIZE(spaces_fast);
    if (PySequence_Fast_GET_SIZE(names_fast) != n
        || PySequence_Fast_GET_SIZE(dummy_fast) != n
        || PySequence_Fast_GET_SIZE(free_fast) != n) {
        Py_DECREF(spaces_fast);
        Py_DECREF(names_fast);
        Py_DECREF(dummy_fast);
        Py_DECREF(free_fast);
        PyErr_SetString(PyExc_ValueError, "summed-index classifier inputs must match length");
        return nullptr;
    }

    struct Entry {
        int space;
        int name;
        int is_dummy;
        int pos;
    };

    std::unordered_map<std::uint64_t, Entry> unique_entries;
    unique_entries.reserve(static_cast<std::size_t>(n));

    auto **space_items = PySequence_Fast_ITEMS(spaces_fast);
    auto **name_items = PySequence_Fast_ITEMS(names_fast);
    auto **dummy_items = PySequence_Fast_ITEMS(dummy_fast);
    auto **free_items = PySequence_Fast_ITEMS(free_fast);

    for (Py_ssize_t i = 0; i < n; ++i) {
        const auto is_free = PyObject_IsTrue(free_items[i]);
        const auto is_dummy = PyObject_IsTrue(dummy_items[i]);
        if (is_free < 0 || is_dummy < 0) {
            Py_DECREF(spaces_fast);
            Py_DECREF(names_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(free_fast);
            return nullptr;
        }
        if (is_free) {
            continue;
        }

        const auto space = static_cast<int>(PyLong_AsLong(space_items[i]));
        const auto name = static_cast<int>(PyLong_AsLong(name_items[i]));
        if (PyErr_Occurred()) {
            Py_DECREF(spaces_fast);
            Py_DECREF(names_fast);
            Py_DECREF(dummy_fast);
            Py_DECREF(free_fast);
            return nullptr;
        }

        const auto key = (static_cast<std::uint64_t>(space) << 48)
            | (static_cast<std::uint64_t>(name) << 16)
            | static_cast<std::uint64_t>(is_dummy);
        unique_entries.try_emplace(key, Entry{space, name, is_dummy, static_cast<int>(i)});
    }

    Py_DECREF(spaces_fast);
    Py_DECREF(names_fast);
    Py_DECREF(dummy_fast);
    Py_DECREF(free_fast);

    std::vector<Entry> entries;
    entries.reserve(unique_entries.size());
    for (const auto &item : unique_entries) {
        entries.push_back(item.second);
    }
    std::sort(
        entries.begin(),
        entries.end(),
        [](const Entry &lhs, const Entry &rhs) {
            if (lhs.space != rhs.space) {
                return lhs.space < rhs.space;
            }
            if (lhs.name != rhs.name) {
                return lhs.name < rhs.name;
            }
            if (lhs.is_dummy != rhs.is_dummy) {
                return lhs.is_dummy < rhs.is_dummy;
            }
            return lhs.pos < rhs.pos;
        }
    );

    PyObject *result = PyTuple_New(static_cast<Py_ssize_t>(entries.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(entries.size()); ++i) {
        PyTuple_SET_ITEM(result, i, PyLong_FromLong(entries[static_cast<std::size_t>(i)].pos));
    }
    return result;
}

PyMethodDef methods[] = {
    {
        "analyze_signature",
        analyze_signature,
        METH_VARARGS,
        "Analyze a packed Wick signature and return (can_continue, pivot_pos, candidates).",
    },
    {
        "canonicalize_tensor_layout",
        canonicalize_tensor_layout,
        METH_VARARGS,
        "Canonicalize tensor antisymmetry groups and return (is_zero, sign, order).",
    },
    {
        "assign_dummy_ordinals",
        assign_dummy_ordinals,
        METH_VARARGS,
        "Assign canonical dummy pool ordinals for an ordered index list.",
    },
    {
        "apply_deltas_layout",
        apply_deltas_layout,
        METH_VARARGS,
        "Resolve delta unions over local index metadata and return (ok, roots).",
    },
    {
        "classify_summed_indices",
        classify_summed_indices,
        METH_VARARGS,
        "Return sorted unique summed-index positions from local tensor index metadata.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_wickaccel",
    "C++ accelerators for ccgen Wick recursion.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__wickaccel(void) {
    return PyModule_Create(&module);
}
