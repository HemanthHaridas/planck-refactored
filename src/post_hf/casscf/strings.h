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

struct FermionOpResult
{
    CIString det = 0;
    double phase = 0.0;
    bool valid = false;
};

int count_occupied_below(CIString det, int orb);
int parity_between(CIString s, int lo, int hi);

std::vector<CIString> generate_strings(int n_orb, int n_occ);

std::vector<int> map_mo_irreps(
    const std::vector<std::string>& mo_sym,
    const std::vector<std::string>& names);

std::optional<SymmetryContext> build_symmetry_context(HartreeFock::Calculator& calc);

std::optional<int> resolve_target_irrep(
    const std::string& s,
    const SymmetryContext& sym_ctx);

bool point_group_has_only_1d_irreps(const std::string& pg);

void build_spin_strings_unfiltered(
    int n_act,
    int n_alpha,
    int n_beta,
    std::vector<CIString>& a_strs,
    std::vector<CIString>& b_strs);

FermionOpResult apply_annihilation(CIString det, int orb);
FermionOpResult apply_creation(CIString det, int orb);

std::vector<CIString> build_spin_dets(
    const std::vector<CIString>& a_strs,
    const std::vector<CIString>& b_strs,
    const std::vector<std::pair<int, int>>& dets,
    int n_act);

std::unordered_map<CIString, int> build_det_lookup(const std::vector<CIString>& sd);

} // namespace HartreeFock::Correlation::CASSCF

#endif // HF_POSTHF_CASSCF_STRINGS_H
