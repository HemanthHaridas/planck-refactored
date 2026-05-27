#include "mp2.h"

#include <Eigen/Eigenvalues>
#include <expected>
#include <string>

namespace HartreeFock::Correlation
{
    std::expected<void, std::string> apply_rmp2_result(
        HartreeFock::Calculator &calculator,
        const RMP2Result &result)
    {
        // The driver keeps one canonical MP2 "result slot" on Calculator even
        // though RHF and UHF store amplitudes differently. This helper is the
        // normalization point that copies the RHF view in and clears any stale
        // unrestricted tensors from prior jobs.
        calculator._correlation_energy = result.e_corr;
        calculator._mp2_e_corr_ss = result.e_corr_ss;
        calculator._mp2_e_corr_os = result.e_corr_os;
        calculator._mp2_converged = result.converged;
        calculator._mp2_n_iter = result.n_iter;
        calculator._mp2_nocc = result.n_occ;
        calculator._mp2_nvir = result.n_virt;
        calculator._mp2_active_mo = result.active_mo;
        calculator._mp2_t2 = result.t2;
        calculator._ump2_t2_aa.clear();
        calculator._ump2_t2_ab.clear();
        calculator._ump2_t2_bb.clear();
        return {};
    }

    std::expected<void, std::string> apply_ump2_result(
        HartreeFock::Calculator &calculator,
        const UMP2Result &result)
    {
        // UMP2 mirrors the same Calculator-level bookkeeping, but its amplitudes
        // live in aa/ab/bb spin blocks instead of a single RHF t2 tensor.
        calculator._correlation_energy = result.e_corr;
        calculator._mp2_e_corr_ss = result.e_corr_ss;
        calculator._mp2_e_corr_os = result.e_corr_os;
        calculator._mp2_converged = result.converged;
        calculator._mp2_n_iter = result.n_iter;
        calculator._mp2_nocca = result.nocca;
        calculator._mp2_noccb = result.noccb;
        calculator._mp2_nvira = result.nvira;
        calculator._mp2_nvirb = result.nvirb;
        calculator._mp2_active_mo_alpha = result.active_mo_alpha;
        calculator._mp2_active_mo_beta = result.active_mo_beta;
        calculator._ump2_t2_aa = result.t2_aa;
        calculator._ump2_t2_ab = result.t2_ab;
        calculator._ump2_t2_bb = result.t2_bb;
        calculator._mp2_t2.clear();
        return {};
    }

    std::expected<RMP2NaturalOrbitals, std::string> rmp2_make_natural_orbitals(
        const RMP2Result &result)
    {
        // Natural orbitals are eigenvectors of the correlated one-particle
        // density, expressed first in the active MO space and then rotated back
        // to AO coefficients for downstream printing or export.
        auto rdm1_res = rmp2_make_rdm1(result, false);
        if (!rdm1_res)
            return std::unexpected("rmp2_make_natural_orbitals: " + rdm1_res.error());

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(*rdm1_res);
        if (solver.info() != Eigen::Success)
            return std::unexpected("rmp2_make_natural_orbitals: density diagonalization failed.");

        const int nmo = static_cast<int>(result.mo_coeff.cols());
        RMP2NaturalOrbitals out;
        out.occupations.resize(nmo);
        out.coefficients_mo.resize(nmo, nmo);
        for (int i = 0; i < nmo; ++i)
        {
            const int src = nmo - 1 - i;
            out.occupations(i) = solver.eigenvalues()(src);
            out.coefficients_mo.col(i) = solver.eigenvectors().col(src);
        }
        out.coefficients_ao = result.mo_coeff * out.coefficients_mo;
        return out;
    }
} // namespace HartreeFock::Correlation
