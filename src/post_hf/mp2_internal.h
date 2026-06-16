#ifndef HF_POSTHF_MP2_INTERNAL_H
#define HF_POSTHF_MP2_INTERNAL_H

// Private MP2 helpers shared between mp2_rmp2.cpp and mp2_ump2.cpp.
//
// These types mirror PySCF's `_ChemistsERIs` (mp2.py / ump2.py) and the
// frozen-orbital book-keeping in `get_nocc / get_nmo / get_frozen_mask`.
// Nothing in this header is part of the public Planck API; outside of the two
// MP2 implementation files the public surface in `mp2.h` is the only contract.

#include <Eigen/Core>
#include <cstddef>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Correlation::detail
{
    // RMP2 ChemistsERIs: matches PySCF mp2._ChemistsERIs.
    //
    // ovov[i,a,j,b] = (ia|jb) in chemist notation, stored row-major as
    //   index = ((i*nvir + a)*nocc + j)*nvir + b
    // mo_coeff is the active (frozen-removed) MO coefficient block in the AO
    // basis. fock is the active-block MO Fock matrix; for canonical references
    // it is diag(mo_energy).
    struct ChemistsERIs
    {
        Eigen::MatrixXd mo_coeff;     // nb × nmo (active MOs)
        int nocc = 0;
        Eigen::VectorXd mo_energy;    // nmo
        Eigen::MatrixXd fock;         // nmo × nmo
        std::vector<double> ovov;     // (nocc·nvir)² row-major
    };

    // UMP2 ChemistsERIs: matches PySCF ump2._ChemistsERIs.
    struct UChemistsERIs
    {
        Eigen::MatrixXd mo_coeff_a;   // nb × nmoa
        Eigen::MatrixXd mo_coeff_b;   // nb × nmob
        int nocca = 0;
        int noccb = 0;
        Eigen::VectorXd mo_energy_a;  // nmoa
        Eigen::VectorXd mo_energy_b;  // nmob
        Eigen::MatrixXd fock_a;       // nmoa × nmoa
        Eigen::MatrixXd fock_b;       // nmob × nmob
        std::vector<double> ovov;     // αα: (nocca·nvira)² row-major
        std::vector<double> OVOV;     // ββ: (noccb·nvirb)² row-major
        std::vector<double> ovOV;     // αβ: (nocca·nvira)·(noccb·nvirb) row-major
    };

    // RMP2 dimensions after applying the frozen mask. PySCF parity: frozen
    // empty → no orbitals frozen; frozen of length 1 → freeze the lowest N
    // orbitals; otherwise → explicit 0-based MO indices.
    struct RMP2Dims
    {
        int n_occ = 0;          // active occupied
        int n_virt = 0;         // active virtual
        int n_mo = 0;           // active total = n_occ + n_virt
        int n_occ_full = 0;     // total occupied including frozen
        int n_mo_full = 0;      // total MOs
        std::vector<int> active_mo;   // indices into the full MO list (size n_mo)
    };

    struct UMP2Dims
    {
        int nocca = 0;
        int noccb = 0;
        int nvira = 0;
        int nvirb = 0;
        int nmoa = 0;
        int nmob = 0;
        int nocca_full = 0;
        int noccb_full = 0;
        int nmoa_full = 0;
        int nmob_full = 0;
        std::vector<int> active_a;
        std::vector<int> active_b;
    };

    // Resolve PySCF-style frozen-orbital options into concrete active-MO masks.
    std::expected<RMP2Dims, std::string> resolve_rmp2_dims(
        const HartreeFock::Calculator &calculator,
        const HartreeFock::OptionsMP2 &options);

    std::expected<UMP2Dims, std::string> resolve_ump2_dims(
        const HartreeFock::Calculator &calculator,
        const HartreeFock::OptionsMP2 &options);

    // Build the RMP2 ChemistsERIs from the SCF reference, performing the
    // (μν|λσ) → (ia|jb) integral transformation. Mirrors PySCF mp2._make_eris.
    std::expected<ChemistsERIs, std::string> make_eris_rmp2(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Dims &dims);

    std::expected<UChemistsERIs, std::string> make_eris_ump2(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UMP2Dims &dims);

    // Flat-index helpers. Layout: [i, j, a, b] row-major when stored as t2;
    // [i, a, j, b] row-major when stored as ovov (chemist notation).
    inline std::size_t idx_t2(int i, int j, int a, int b, int n_occ, int n_virt) noexcept
    {
        return ((static_cast<std::size_t>(i) * n_occ + j) * n_virt + a) * n_virt + b;
    }
    inline std::size_t idx_ovov(int i, int a, int j, int b, int n_occ, int n_virt) noexcept
    {
        return ((static_cast<std::size_t>(i) * n_virt + a) * n_occ + j) * n_virt + b;
    }
    inline std::size_t idx_t2_ab(int i, int j, int a, int b,
                                 int nocca, int noccb, int nvira, int nvirb) noexcept
    {
        return ((static_cast<std::size_t>(i) * noccb + j) * nvira + a) * nvirb + b;
    }
    inline std::size_t idx_ovOV(int i, int a, int j, int b,
                                int nocca, int noccb, int nvira, int nvirb) noexcept
    {
        return ((static_cast<std::size_t>(i) * nvira + a) * noccb + j) * nvirb + b;
    }

} // namespace HartreeFock::Correlation::detail

#endif // HF_POSTHF_MP2_INTERNAL_H
