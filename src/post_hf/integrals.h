#ifndef HF_POSTHF_INTEGRALS_H
#define HF_POSTHF_INTEGRALS_H

#include <string>
#include <vector>
#include <Eigen/Core>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation
{

// ── AO ERI availability ────────────────────────────────────────────────────
//
// Ensure calc._eri is populated.  If it is already non-empty (conventional
// SCF pre-built it), this is a no-op.  Otherwise the full AO ERI tensor is
// computed via ObaraSaika and stored in eri_local; a reference to whichever
// tensor is valid is returned.
//
// Parameters:
//   calc        — Calculator (may have _eri already)
//   shell_pairs — shell pairs for on-the-fly ERI computation
//   eri_local   — caller-owned storage used when _eri is empty
//   tag         — log prefix (e.g. "RMP2 :", "CASSCF :")
const std::vector<double>& ensure_eri(
    HartreeFock::Calculator&                   calc,
    const std::vector<HartreeFock::ShellPair>& shell_pairs,
    std::vector<double>&                       eri_local,
    const std::string&                         tag);


// ── 4-index AO→MO transformation ──────────────────────────────────────────
//
// Performs successive quarter transforms:
//   T1[i,ν,λ,σ]   = Σ_μ  C1(μ,i) * eri[μνλσ]
//   T2[i,a,λ,σ]   = Σ_ν  C2(ν,a) * T1[i,ν,λ,σ]
//   T3[i,a,j,σ]   = Σ_λ  C3(λ,j) * T2[i,a,λ,σ]
//   out[i,a,j,b]  = Σ_σ  C4(σ,b) * T3[i,a,j,σ]
//
// Returns a flat vector of size n1*n2*n3*n4 indexed row-major:
//   out[i*n2*n3*n4 + a*n3*n4 + j*n4 + b]
std::vector<double> transform_eri(
    const std::vector<double>& eri,
    std::size_t                nb,
    const Eigen::MatrixXd&     C1,   // nb × n1
    const Eigen::MatrixXd&     C2,   // nb × n2
    const Eigen::MatrixXd&     C3,   // nb × n3
    const Eigen::MatrixXd&     C4);  // nb × n4


// ── Full internal-space transform (CASSCF) ────────────────────────────────
//
// Convenience wrapper: all four legs use C_int.
// Returns a flat vector of size n_int^4 indexed row-major:
//   out[p*ni3 + q*ni2 + r*ni + s]  =  (pq|rs)  in MO basis
// where p,q,r,s ∈ 0..n_int-1.
std::vector<double> transform_eri_internal(
    const std::vector<double>& eri,
    std::size_t                nb,
    const Eigen::MatrixXd&     C_int);  // nb × n_int

} // namespace HartreeFock::Correlation

#endif // HF_POSTHF_INTEGRALS_H
