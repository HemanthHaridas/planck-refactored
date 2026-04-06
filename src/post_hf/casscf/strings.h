#ifndef HF_POSTHF_CASSCF_STRINGS_H
#define HF_POSTHF_CASSCF_STRINGS_H

#include "base/types.h"
#include "post_hf/casscf_internal.h"

#include <Eigen/Core>

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace HartreeFock::Correlation::CASSCF
{

    using HartreeFock::Correlation::CASSCFInternal::CIString;
    using HartreeFock::Correlation::CASSCFInternal::RASParams;
    using HartreeFock::Correlation::CASSCFInternal::SymmetryContext;

    // A symmetry-aware active-space picker returns a full MO permutation so the
    // chosen active block can be made contiguous before the CI/gradient code runs.
    struct ActiveOrbitalSelection
    {
        std::vector<int> permutation;
        std::vector<int> active_orbitals;
        bool used_symmetry = false;
    };

    // Fermionic operators return both the updated determinant and the accumulated
    // sign so the string-level algebra can stay explicit.
    struct FermionOpResult
    {
        CIString det = 0;
        double phase = 0.0;
        bool valid = false;
    };

    // Occupied-orbital and parity helpers are shared by CI, RDM, and response code
    // so every layer uses the same bit ordering convention.
    int count_occupied_below(CIString det, int orb);
    int parity_between(CIString s, int lo, int hi);

    // Enumerate fixed-popcount bitstrings in ascending order.
    std::vector<CIString> generate_strings(int n_orb, int n_occ);

    // Map MO symmetry labels onto the internal product-table ordering used by the
    // determinant symmetry checks.
    std::vector<int> map_mo_irreps(
        const std::vector<std::string> &mo_sym,
        const std::vector<std::string> &names);

    // Pick a reordered MO basis that keeps the requested active orbitals
    // contiguous. If explicit symmetry/permutation metadata is unavailable, the
    // selector falls back to the identity permutation.
    std::expected<ActiveOrbitalSelection, std::string> select_active_orbitals(
        const Eigen::VectorXd &mo_energies,
        const std::vector<std::string> &mo_symmetry,
        int n_core,
        int n_act,
        const std::vector<HartreeFock::IrrepCount> &core_irrep_counts = {},
        const std::vector<HartreeFock::IrrepCount> &active_irrep_counts = {},
        const std::vector<int> &mo_permutation = {});

    // Apply a column permutation to the MO coefficient matrix.
    Eigen::MatrixXd reorder_mo_coefficients(
        const Eigen::MatrixXd &mo_coefficients,
        const std::vector<int> &permutation);

    // Build a minimal symmetry context only when the current point group can be
    // represented by a one-dimensional Abelian product table.
    std::optional<SymmetryContext> build_symmetry_context(HartreeFock::Calculator &calc);

    // Resolve the target CI irrep string into the Abelian symmetry-table index.
    std::optional<int> resolve_target_irrep(
        const std::string &s,
        const SymmetryContext &sym_ctx);

    // This path only supports point groups whose irreps are all one-dimensional.
    bool point_group_has_only_1d_irreps(const std::string &pg);

    // Build the unfiltered alpha/beta string lists before symmetry or RAS
    // screening is applied.
    void build_spin_strings_unfiltered(
        int n_act,
        int n_alpha,
        int n_beta,
        std::vector<CIString> &a_strs,
        std::vector<CIString> &b_strs);

    // Apply a single fermionic creation/annihilation operator with the phase
    // convention used throughout the CI and response code.
    FermionOpResult apply_annihilation(CIString det, int orb);
    FermionOpResult apply_creation(CIString det, int orb);

    // Pack a determinant pair into the bitstring key used by the CI lookup table.
    std::vector<CIString> build_spin_dets(
        const std::vector<CIString> &a_strs,
        const std::vector<CIString> &b_strs,
        const std::vector<std::pair<int, int>> &dets,
        int n_act);

    // Reverse lookup from packed determinant key to CI index.
    std::unordered_map<CIString, int> build_det_lookup(const std::vector<CIString> &sd);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_STRINGS_H
