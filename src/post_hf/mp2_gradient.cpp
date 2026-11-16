#include "mp2_gradient.h"

#include <cstdlib>
#include <cstdio>
#include <expected>
#include <format>
#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <tuple>

#include "basis/spherical.h"
#include "scf/scf.h"

#include "gradient/gradient.h"
#include "gradient/two_electron_gradient.h"
#include "integrals/base.h"
#include "post_hf/integrals.h"
#include "post_hf/mp2_internal.h"
#include "post_hf/rhf_response.h"
#include "post_hf/uhf_response.h"
#include "post_hf/ri/ri_eri.h"

namespace
{
    bool rmp2_term_debug_enabled()
    {
        const char *enabled = std::getenv("PLANCK_DEBUG_RMP2_TERMS");
        return enabled != nullptr && std::string_view(enabled) == "1";
    }

    void maybe_print_rmp2_term_matrix(const char *name, const Eigen::MatrixXd &term)
    {
        if (!rmp2_term_debug_enabled())
            return;

        std::cout << "PLANCK_RMP2_TERM " << name << "\n";
        std::cout << std::setprecision(16);
        for (Eigen::Index atom = 0; atom < term.rows(); ++atom)
        {
            std::cout << "PLANCK_RMP2_TERM_ROW "
                      << name << " "
                      << (atom + 1) << " "
                      << term(atom, 0) << " "
                      << term(atom, 1) << " "
                      << term(atom, 2) << "\n";
        }
    }

    void maybe_print_rmp2_matrix(const char *name, const Eigen::MatrixXd &mat)
    {
        const char *enabled = std::getenv("PLANCK_DEBUG_RMP2_MATRICES");
        if (enabled == nullptr || std::string_view(enabled) != "1")
            return;

        std::cout << "PLANCK_RMP2_MATRIX " << name << " " << mat.rows() << " " << mat.cols() << "\n";
        std::cout << std::setprecision(16);
        for (Eigen::Index row = 0; row < mat.rows(); ++row)
            for (Eigen::Index col = 0; col < mat.cols(); ++col)
                std::cout << "PLANCK_RMP2_MATRIX_ELEM "
                          << name << " "
                          << row << " "
                          << col << " "
                          << mat(row, col) << "\n";
    }

    inline std::size_t idx_dm2(int p, int q, int r, int s, int nmo)
    {
        return ((static_cast<std::size_t>(p) * nmo + q) * nmo + r) * nmo + s;
    }

    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // True when the MP2 result correlates only part of the SCF orbitals
    // (mp2_frozen). The gradient builders treat the result's orbitals as the
    // whole space, so they would drop the frozen orbitals' response terms
    // silently. Compared against the SCF MO count, which is right in either basis.
    bool has_frozen_orbitals(std::size_t n_active, const Eigen::MatrixXd &scf_mo_coefficients)
    {
        return n_active != 0 && n_active != static_cast<std::size_t>(scf_mo_coefficients.cols());
    }

    // T2 -> AO half density, the first of the two back-transforms the MP2
    // gradients share: half[i,p,q,j] = (C_l T_ij C_rᵀ)(p,q), where
    // amplitude(i, j) returns the (nv_l x nv_r) virtual block T_ij. One small
    // GEMM per (i,j); each pair writes its own elements.
    template <class Amplitude>
    std::vector<double> pair_half_density(
        int ni, int nj, const Eigen::MatrixXd &C_l, const Eigen::MatrixXd &C_r, Amplitude amplitude)
    {
        const int nao = static_cast<int>(C_l.rows());
        std::vector<double> half(static_cast<std::size_t>(ni) * nao * nao * nj);
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < ni; ++i)
            for (int j = 0; j < nj; ++j)
            {
                const Eigen::MatrixXd P = C_l * amplitude(i, j) * C_r.transpose();
                for (int p = 0; p < nao; ++p)
                    for (int q = 0; q < nao; ++q)
                        half[((static_cast<std::size_t>(i) * nao + p) * nao + q) * nj + j] = P(p, q);
            }
        return half;
    }

    // The second back-transform: B[p,q,r,s] = Σ_ij C_i(p,i) half[i,q,r,j] C_j(s,j),
    // as two GEMMs over reshaped views (nao⁴·nocc, not nao⁴·nocc²).
    std::vector<double> back_transform_half_density(
        const std::vector<double> &half, const Eigen::MatrixXd &C_i, const Eigen::MatrixXd &C_j)
    {
        const Eigen::Index nao = C_i.rows(), ni = C_i.cols(), nj = C_j.cols();
        const Eigen::Index nao3 = nao * nao * nao;
        // X[i,q,r,s] = Σ_j half[i,q,r,j] C_j(s,j)
        const RowMajorMatrix X =
            Eigen::Map<const RowMajorMatrix>(half.data(), ni * nao * nao, nj) * C_j.transpose();
        std::vector<double> B(static_cast<std::size_t>(nao * nao3));
        Eigen::Map<RowMajorMatrix>(B.data(), nao, nao3).noalias() =
            C_i * Eigen::Map<const RowMajorMatrix>(X.data(), ni, nao3);
        return B;
    }

    // dst[p,q,r,s] += Σ over the p<->q (and, with swap_rs, r<->s) images of B,
    // read as B[r,s,p,q] when bra_ket. Threaded over p, each writing its own rows.
    void add_symmetrized(std::vector<double> &dst, const std::vector<double> &B, int nao,
                         bool swap_rs, bool bra_ket)
    {
        const auto b = [&](int p, int q, int r, int s)
        { return bra_ket ? B[idx_dm2(r, s, p, q, nao)] : B[idx_dm2(p, q, r, s, nao)]; };
#pragma omp parallel for schedule(static)
        for (int p = 0; p < nao; ++p)
            for (int q = 0; q < nao; ++q)
                for (int r = 0; r < nao; ++r)
                    for (int s = 0; s < nao; ++s)
                    {
                        double v = b(p, q, r, s) + b(q, p, r, s);
                        if (swap_rs)
                            v += b(p, q, s, r) + b(q, p, s, r);
                        dst[idx_dm2(p, q, r, s, nao)] += v;
                    }
    }

    // Densities and results are in the working basis; SCF::two_electron_fock
    // does the Cartesian round trip (identity in Cartesian mode).
    std::expected<Eigen::MatrixXd, std::string> build_veff_from_density(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density)
    {
        // veff of the relaxed density differentiates the SCF, so it uses the
        // SCF's own operator (fitted only under RI-JK), even for RI-MP2, whose
        // fit covers the correlation energy only
        // (docs/RI_MP2_GRADIENT_RESPONSE.md). Density is symmetric at
        // both call sites.
        return HartreeFock::SCF::two_electron_fock(calculator, shell_pairs, density);
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string> build_veff_from_spin_densities(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density_alpha,
        const Eigen::MatrixXd &density_beta)
    {
        // As build_veff_from_density: the SCF's own operator. Both call sites
        // pass symmetric densities.
        return HartreeFock::SCF::two_electron_fock_uhf(calculator, shell_pairs, density_alpha, density_beta);
    }

    // Working-basis AO matrices -> Cartesian, for the derivative kernels
    // (one_electron_gradient, the overlap-derivative loops, the separable Γ).
    // Each is a finished AO quantity, lifted once; identity in Cartesian mode.
    std::expected<std::vector<Eigen::MatrixXd>, std::string> lift_to_cartesian(
        const HartreeFock::Basis &basis, std::initializer_list<const Eigen::MatrixXd *> mats)
    {
        std::vector<Eigen::MatrixXd> out;
        out.reserve(mats.size());
        for (const Eigen::MatrixXd *m : mats)
        {
            auto c = HartreeFock::BasisFunctions::to_cartesian(basis, *m);
            if (!c)
                return std::unexpected(c.error());
            out.push_back(std::move(*c));
        }
        return out;
    }

    // RMP2 AO pair density dm2[p,q,r,s] = Σ_ij C(p,i) C(s,j) (C_v (4T_ij -
    // 2T_ijᵀ) C_vᵀ)(q,r), symmetrized over p <-> q, in the basis of `C` (the
    // working-basis MO coefficients, or their Cartesian lift Tᵀ C: the build is
    // linear in each coefficient matrix, so the latter IS the 4-index lift).
    std::vector<double> rmp2_pair_density(const HartreeFock::Correlation::RMP2Result &result,
                                          const Eigen::MatrixXd &C)
    {
        const int nocc = result.n_occ, nvirt = result.n_virt;
        const int nao = static_cast<int>(C.rows());
        const auto half = pair_half_density(nocc, nocc, C.middleCols(nocc, nvirt), C.middleCols(nocc, nvirt),
                                            [&](int i, int j)
        {
            const Eigen::Map<const RowMajorMatrix> T(
                result.t2.data() + HartreeFock::Correlation::detail::idx_t2(i, j, 0, 0, nocc, nvirt), nvirt, nvirt);
            return Eigen::MatrixXd(4.0 * T - 2.0 * T.transpose());
        });
        std::vector<double> dm2(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
        add_symmetrized(dm2, back_transform_half_density(half, C.leftCols(nocc), C.leftCols(nocc)),
                        nao, /*swap_rs=*/false, /*bra_ket=*/false);
        return dm2;
    }

    // UMP2 per-spin AO pair densities D_α, D_β (see the caller), in the basis
    // of Ca/Cb (working, or the Cartesian lift Tᵀ C — exact, as above).
    std::pair<std::vector<double>, std::vector<double>> ump2_pair_densities(
        const HartreeFock::Correlation::UMP2Result &result, const Eigen::MatrixXd &Ca, const Eigen::MatrixXd &Cb)
    {
        const int nocca = result.nocca, noccb = result.noccb, nvira = result.nvira, nvirb = result.nvirb;
        const int nao = static_cast<int>(Ca.rows());
        const Eigen::MatrixXd Ca_occ = Ca.leftCols(nocca), Ca_virt = Ca.middleCols(nocca, nvira);
        const Eigen::MatrixXd Cb_occ = Cb.leftCols(noccb), Cb_virt = Cb.middleCols(noccb, nvirb);
        // T_ij virtual block of one spin case, antisymmetrized (½(T - Tᵀ)) for
        // the same-spin amplitudes.
        const auto spin_half = [](int ni, int nj, int nvi, int nvj, const std::vector<double> &t2,
                                  const Eigen::MatrixXd &Cl, const Eigen::MatrixXd &Cr, bool same_spin)
        {
            return pair_half_density(ni, nj, Cl, Cr, [&, same_spin](int i, int j)
            {
                const Eigen::Map<const RowMajorMatrix> T(
                    t2.data() + (static_cast<std::size_t>(i) * nj + j) * nvi * nvj, nvi, nvj);
                return same_spin ? Eigen::MatrixXd(0.5 * (T - T.transpose())) : Eigen::MatrixXd(T);
            });
        };
        // D_α = sym(B_αα) + sym(B_αβ), D_β = sym(B_ββ) + sym(B_αβ) read bra<->ket,
        // where B_στ[p,q,r,s] = Σ_ij C_σ(p,i) half_στ[i,q,r,j] C_τ(s,j) and sym
        // sums the p<->q, r<->s images. One B is alive at a time.
        const std::size_t nao4 = static_cast<std::size_t>(nao) * nao * nao * nao;
        std::vector<double> Da(nao4, 0.0), Db(nao4, 0.0);
        add_symmetrized(Da, back_transform_half_density(
            spin_half(nocca, nocca, nvira, nvira, result.t2_aa, Ca_virt, Ca_virt, true), Ca_occ, Ca_occ),
            nao, /*swap_rs=*/true, /*bra_ket=*/false);
        {
            const auto Bab = back_transform_half_density(
                spin_half(nocca, noccb, nvira, nvirb, result.t2_ab, Ca_virt, Cb_virt, false), Ca_occ, Cb_occ);
            add_symmetrized(Da, Bab, nao, /*swap_rs=*/true, /*bra_ket=*/false);
            add_symmetrized(Db, Bab, nao, /*swap_rs=*/true, /*bra_ket=*/true);
        }
        add_symmetrized(Db, back_transform_half_density(
            spin_half(noccb, noccb, nvirb, nvirb, result.t2_bb, Cb_virt, Cb_virt, true), Cb_occ, Cb_occ),
            nao, /*swap_rs=*/true, /*bra_ket=*/false);
        return {std::move(Da), std::move(Db)};
    }

    Eigen::MatrixXd contract_imat_from_pair_density(
        const std::vector<double> &eri,
        const std::vector<double> &pair_dm2,
        int nb)
    {
        Eigen::MatrixXd imat = Eigen::MatrixXd::Zero(nb, nb);
        for (int p = 0; p < nb; ++p)
            for (int q = 0; q < nb; ++q)
            {
                double val = 0.0;
                for (int i = 0; i < nb; ++i)
                    for (int r = 0; r < nb; ++r)
                        for (int s = 0; s < nb; ++s)
                        {
                            const std::size_t iprs = ((static_cast<std::size_t>(i) * nb + p) * nb + r) * nb + s;
                            const std::size_t iqrs = idx_dm2(i, q, r, s, nb);
                            val += eri[iprs] * pair_dm2[iqrs];
                        }
                imat(p, q) = val;
            }
        return imat;
    }

    std::vector<std::vector<int>> build_atom_ao_lists(const HartreeFock::Calculator &calculator)
    {
        std::vector<std::vector<int>> atom_aos(calculator._molecule.natoms);
        const auto &bfs = calculator._shells._basis_functions;
        for (std::size_t mu = 0; mu < bfs.size(); ++mu)
        {
            const int atom = static_cast<int>(bfs[mu]._shell->_atom_index);
            atom_aos[atom].push_back(static_cast<int>(mu));
        }
        return atom_aos;
    }

    Eigen::MatrixXd nuclear_repulsion_gradient(const HartreeFock::Calculator &calculator)
    {
        const auto &mol = calculator._molecule;
        Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(mol.natoms, 3);
        for (std::size_t a = 0; a < mol.natoms; ++a)
            for (std::size_t b = 0; b < mol.natoms; ++b)
            {
                if (a == b)
                    continue;
                const double Za = mol.nuclear_charge(a);
                const double Zb = mol.nuclear_charge(b);
                const double dx = mol._standard(a, 0) - mol._standard(b, 0);
                const double dy = mol._standard(a, 1) - mol._standard(b, 1);
                const double dz = mol._standard(a, 2) - mol._standard(b, 2);
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double fac = Za * Zb / (r * r * r);
                grad(a, 0) -= fac * dx;
                grad(a, 1) -= fac * dy;
                grad(a, 2) -= fac * dz;
            }
        return grad;
    }

}

namespace HartreeFock::Correlation
{
    std::expected<RMP2Lagrangian, std::string> build_rmp2_lagrangian(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result,
        const KsVeffFn &ks_veff)
    {
        if (result.t2.empty())
            return std::unexpected("build_rmp2_lagrangian: T2 amplitudes are required.");
        if (has_frozen_orbitals(result.active_mo.size(), calculator._info._scf.alpha.mo_coefficients))
            return std::unexpected("build_rmp2_lagrangian: frozen-orbital gradients are not implemented.");

        // The orbital half lives in the working basis (spherical or Cartesian):
        // MO coefficients, the AO ERI tensor (ensure_eri) and _overlap all do.
        const int nao = static_cast<int>(calculator.working_nbasis());
        const int nocc = result.n_occ;
        const int nvirt = result.n_virt;
        const int nmo = nocc + nvirt;

        const Eigen::MatrixXd C_occ = result.mo_coeff.leftCols(nocc);
        const Eigen::MatrixXd C_virt = result.mo_coeff.middleCols(nocc, nvirt);

        auto gamma_res = rmp2_gamma1_intermediates(result);
        if (!gamma_res)
            return std::unexpected(gamma_res.error());
        const auto &[doo, dvv] = *gamma_res;

        RMP2Lagrangian out;
        out.n_occ = nocc;
        out.n_virt = nvirt;
        out.doo = doo;
        out.dvv = dvv;

        out.dm1_corr_mo = Eigen::MatrixXd::Zero(nmo, nmo);
        out.dm1_corr_mo.topLeftCorner(nocc, nocc) = doo + doo.transpose();
        out.dm1_corr_mo.bottomRightCorner(nvirt, nvirt) = dvv + dvv.transpose();
        out.dm1_corr_ao = result.mo_coeff * out.dm1_corr_mo * result.mo_coeff.transpose();

        // Plain nao^4 ERI (skipped entirely under RI). Feeds only the
        // Lagrangian imat below.
        std::vector<double> eri_local;
        static const std::vector<double> empty_eri;
        const std::vector<double> &eri =
            calculator._mp2.use_ri
                ? empty_eri
                : ensure_eri(calculator, shell_pairs, eri_local, "RMP2 Lagrangian :");

        // T2 -> AO 2-RDM buffer in the working basis (rmp2_pair_density).
        out.dm2buf_full = rmp2_pair_density(result, result.mo_coeff);

        // Orbital Lagrangian imat(q,v) += sum_{p,r,s} (pq|rs) dm2buf[p,v,r,s].
        // In build_rmp2_gradient_intermediates this contraction is fused into
        // the derivative-integral loop but uses only the plain ERI; here it is
        // the standalone loop. Under RI it comes from the 3-center factors.
        Eigen::MatrixXd imat_ao = Eigen::MatrixXd::Zero(nao, nao);
        if (calculator._mp2.use_ri)
        {
            imat_ao = HartreeFock::Correlation::RI::build_ri_imat(
                calculator, out.dm2buf_full, nao);
        }
        else
        {
            for (int p = 0; p < nao; ++p)
                for (int q = 0; q < nao; ++q)
                    for (int r = 0; r < nao; ++r)
                        for (int s = 0; s < nao; ++s)
                        {
                            const double eri_pqrs = eri[idx_dm2(p, q, r, s, nao)];
                            for (int v = 0; v < nao; ++v)
                                imat_ao(q, v) += eri_pqrs * out.dm2buf_full[idx_dm2(p, v, r, s, nao)];
                        }
        }

        imat_ao = -imat_ao;
        out.imat_ao = imat_ao;
        out.imat_mo = result.mo_coeff.transpose() * imat_ao * calculator._overlap * result.mo_coeff;

        // veff_corr_ao = 2 * mean-field-response[dm1_corr_ao]. HF (J - 1/2 K)
        // for RMP2; the KS response (J - 1/2 c_x K + V_xc response) for a
        // double hybrid, injected via ks_veff. PySCF's grad/mp2.py does the
        // equivalent with `mp._scf.get_veff(...) * 2`.
        if (ks_veff)
            out.veff_corr_ao = 2.0 * ks_veff(out.dm1_corr_ao);
        else
        {
            auto veff = build_veff_from_density(calculator, shell_pairs, out.dm1_corr_ao);
            if (!veff)
                return std::unexpected("build_rmp2_lagrangian: " + veff.error());
            out.veff_corr_ao = 2.0 * *veff;
        }

        out.Xvo =
            C_virt.transpose() * out.veff_corr_ao * C_occ +
            out.imat_mo.topRightCorner(nocc, nvirt).transpose() -
            out.imat_mo.bottomLeftCorner(nvirt, nocc);

        return out;
    }

    std::expected<RMP2RelaxedDensity, std::string> build_rmp2_energy_weighted_density(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result,
        const RMP2Lagrangian &lag,
        const Eigen::MatrixXd &z,
        const KsVeffFn &ks_veff)
    {
        const int nocc = lag.n_occ;
        const int nvirt = lag.n_virt;
        const int nmo = nocc + nvirt;
        if (z.rows() != nvirt || z.cols() != nocc)
            return std::unexpected(std::format(
                "build_rmp2_energy_weighted_density: Z-vector shape mismatch; "
                "expected {}x{}, got {}x{}.",
                nvirt, nocc, z.rows(), z.cols()));

        const Eigen::MatrixXd C_occ = result.mo_coeff.leftCols(nocc);
        const Eigen::VectorXd eps_occ = result.mo_energy.head(nocc);

        // Verbatim from build_rmp2_gradient_intermediates (was lines 558-590,
        // 677), on whatever orbitals `result` carries.
        Eigen::MatrixXd corr_relaxed_mo = lag.dm1_corr_mo;
        corr_relaxed_mo.bottomLeftCorner(nvirt, nocc) = z;
        corr_relaxed_mo.topRightCorner(nocc, nvirt) = z.transpose();

        RMP2RelaxedDensity out;
        out.P_mo = Eigen::MatrixXd::Zero(nmo, nmo);
        out.P_mo.topLeftCorner(nocc, nocc) = 2.0 * Eigen::MatrixXd::Identity(nocc, nocc);
        out.P_mo += corr_relaxed_mo;
        out.P_ao = result.mo_coeff * out.P_mo * result.mo_coeff.transpose();

        Eigen::MatrixXd zeta_weights = Eigen::MatrixXd::Zero(nmo, nmo);
        for (int p = 0; p < nmo; ++p)
            for (int q = 0; q < nmo; ++q)
                zeta_weights(p, q) = 0.5 * (result.mo_energy(p) + result.mo_energy(q));
        for (int a = 0; a < nvirt; ++a)
            for (int i = 0; i < nocc; ++i)
            {
                zeta_weights(nocc + a, i) = eps_occ(i);
                zeta_weights(i, nocc + a) = eps_occ(i);
            }

        const Eigen::MatrixXd W_ref = 2.0 * C_occ * eps_occ.asDiagonal() * C_occ.transpose();
        out.zeta_ao =
            W_ref + result.mo_coeff * zeta_weights.cwiseProduct(corr_relaxed_mo) * result.mo_coeff.transpose();

        Eigen::MatrixXd imat_mo_sym = lag.imat_mo;
        imat_mo_sym.bottomLeftCorner(nvirt, nocc) = imat_mo_sym.topRightCorner(nocc, nvirt).transpose();
        out.imat_ao = result.mo_coeff * imat_mo_sym * result.mo_coeff.transpose();

        const Eigen::MatrixXd hf_dm1 = 2.0 * C_occ * C_occ.transpose();
        out.dm1_corr_relaxed_ao = out.P_ao - hf_dm1;

        // vhf_s1occ: occ-projected mean-field response of the relaxed
        // correction density (symmetrized). HF (J - 1/2 K) for RMP2; the KS
        // response for a double hybrid via ks_veff. PySCF grad/mp2.py uses
        // `mp._scf.get_veff(mol, dm1+dm1.T)` here, dispatching to KS.
        const Eigen::MatrixXd occ_projector = C_occ * C_occ.transpose();
        const Eigen::MatrixXd relaxed_sym =
            out.dm1_corr_relaxed_ao + out.dm1_corr_relaxed_ao.transpose();
        Eigen::MatrixXd vhf_relaxed;
        if (ks_veff)
            vhf_relaxed = ks_veff(relaxed_sym);
        else
        {
            auto veff = build_veff_from_density(calculator, shell_pairs, relaxed_sym);
            if (!veff)
                return std::unexpected("build_rmp2_energy_weighted_density: " + veff.error());
            vhf_relaxed = std::move(*veff);
        }
        out.vhf_s1occ_ao = occ_projector * vhf_relaxed * occ_projector;

        out.W_ao = 0.5 * (out.zeta_ao + out.zeta_ao.transpose() -
                          out.imat_ao - out.imat_ao.transpose()) +
                   out.vhf_s1occ_ao;
        return out;
    }

    std::expected<RMP2GradientIntermediates, std::string> build_rmp2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result,
        RMP2PreSolved presolved)
    {
        if (result.t2.empty())
            return std::unexpected("build_rmp2_gradient_intermediates: T2 amplitudes are required.");
        if (has_frozen_orbitals(result.active_mo.size(), calculator._info._scf.alpha.mo_coefficients))
            return std::unexpected("build_rmp2_gradient_intermediates: frozen-orbital gradients are not implemented.");
        if ((presolved.lagrangian == nullptr) != (presolved.z == nullptr))
            return std::unexpected(
                "build_rmp2_gradient_intermediates: RMP2PreSolved needs both lagrangian and z, or neither.");

        // nao: the working basis, where the Lagrangian and densities live.
        // nao_cart: the Cartesian basis the derivative kernels contract over.
        const int nao = static_cast<int>(calculator.working_nbasis());
        const int nao_cart = static_cast<int>(calculator._shells.nbasis());
        const int nocc = result.n_occ;
        const int nvirt = result.n_virt;
        const int nmo = nocc + nvirt;

        const Eigen::MatrixXd C_occ = result.mo_coeff.leftCols(nocc);
        const Eigen::MatrixXd C_virt = result.mo_coeff.middleCols(nocc, nvirt);
        const Eigen::VectorXd eps_occ = result.mo_energy.head(nocc);

        // Orbital-basis half: gamma^1, unrelaxed correlation density + veff, the
        // orbital Lagrangian imat (already -1 flipped), and the Z-vector RHS.
        // Shared with the double-hybrid gradient path, which passes it (and the
        // KS-solved z) in via `presolved`.
        RMP2Lagrangian lag_local;
        if (presolved.lagrangian == nullptr)
        {
            auto lag_res = build_rmp2_lagrangian(calculator, shell_pairs, result);
            if (!lag_res)
                return std::unexpected(lag_res.error());
            lag_local = std::move(*lag_res);
        }
        const RMP2Lagrangian &lag =
            presolved.lagrangian ? *presolved.lagrangian : lag_local;
        const Eigen::MatrixXd &doo = lag.doo;
        const Eigen::MatrixXd &dvv = lag.dvv;
        const Eigen::MatrixXd &dm1_corr_mo = lag.dm1_corr_mo;
        const Eigen::MatrixXd &dm1_corr_ao = lag.dm1_corr_ao;
        const std::vector<double> &dm2buf_full = lag.dm2buf_full;

        std::vector<std::vector<int>> atom_aos = build_atom_ao_lists(calculator);

        // Debug output
        if (const char* debug = std::getenv("PLANCK_DEBUG_DM2BUF"))
        {
            if (std::string_view(debug) == "1")
            {
                std::cout << "PLANCK_DM2BUF_FULL ";
                for (double val : dm2buf_full)
                    std::cout << val << " ";
                std::cout << "\n";
            }
        }

        Eigen::MatrixXd electronic = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd two_e_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        // Orbital Lagrangian from build_rmp2_lagrangian (already -1 flipped).
        // imat_ao is rebuilt from imat_mo further down for the overlap term, so
        // it is a mutable local seeded here.
        Eigen::MatrixXd imat_ao = lag.imat_ao;

        const auto &bfs = calculator._shells._basis_functions;
        const Eigen::MatrixXd hf_dm1 = 2.0 * C_occ * C_occ.transpose();

        // RI 2e-gradient term (Step RG3.4): contract the fitted 3-index density
        // against the RG1 derivative tensors instead of the dense 4-center
        // derivative ERIs the dense path uses. dm2buf's bra is non-symmetric, so
        // gamma3 sums both orderings and the contraction runs bra_prefolded.
        if (calculator._mp2.use_ri)
        {
            auto [gamma3, x_proj] =
                HartreeFock::Correlation::RI::build_ri_gamma3_from_ao_dm2(
                    calculator, dm2buf_full, nao);
            auto ri_two_e = HartreeFock::Correlation::RI::build_ri_two_electron_gradient(
                calculator, gamma3, x_proj, /*bra_prefolded=*/true);
            if (!ri_two_e)
                return std::unexpected("RMP2 RI gradient: " + ri_two_e.error());
            two_e_terms += *ri_two_e;
            electronic += *ri_two_e;
        }
        // imat_ao (seeded from lag.imat_ao) is already -1 flipped; imat_mo and
        // veff_corr_ao come straight from the shared Lagrangian. imat_mo is a
        // mutable local -- its bottom-left block is overwritten below for the
        // overlap term.
        Eigen::MatrixXd imat_mo = lag.imat_mo;
        const Eigen::MatrixXd &veff_corr_ao = lag.veff_corr_ao;

        // Debug: print imat details
        if (const char *enabled = std::getenv("PLANCK_DEBUG_RMP2_IMAT"))
            if (std::string_view(enabled) == "1")
            {
                std::cout << "PLANCK_DEBUG_IMAT_MO " << imat_mo.rows() << " " << imat_mo.cols() << "\n";
                std::cout << std::setprecision(16);
                for (int i = 0; i < imat_mo.rows(); ++i)
                    for (int j = 0; j < imat_mo.cols(); ++j)
                        std::cout << "PLANCK_DEBUG_IMAT_MO_ELEM " << i << " " << j << " " << imat_mo(i, j) << "\n";

                auto block_tr = imat_mo.topRightCorner(nocc, nvirt);
                auto block_bl = imat_mo.bottomLeftCorner(nvirt, nocc);
                std::cout << "PLANCK_DEBUG_IMAT_TOP_RIGHT " << block_tr.rows() << " " << block_tr.cols() << "\n";
                for (int i = 0; i < block_tr.rows(); ++i)
                    for (int j = 0; j < block_tr.cols(); ++j)
                        std::cout << "PLANCK_DEBUG_IMAT_TOP_RIGHT_ELEM " << i << " " << j << " " << block_tr(i, j) << "\n";

                std::cout << "PLANCK_DEBUG_IMAT_BOTTOM_LEFT " << block_bl.rows() << " " << block_bl.cols() << "\n";
                for (int i = 0; i < block_bl.rows(); ++i)
                    for (int j = 0; j < block_bl.cols(); ++j)
                        std::cout << "PLANCK_DEBUG_IMAT_BOTTOM_LEFT_ELEM " << i << " " << j << " " << block_bl(i, j) << "\n";
            }

        Eigen::MatrixXd z_local;
        if (presolved.z == nullptr)
        {
            const Eigen::MatrixXd &Xvo = lag.Xvo;
            auto z_res = solve_rhf_cphf(calculator, shell_pairs, result.mo_coeff, result.mo_energy, Xvo);
            if (!z_res)
                return std::unexpected(z_res.error());
            z_local = std::move(*z_res);
        }
        const Eigen::MatrixXd &z = presolved.z ? *presolved.z : z_local;

        // Relaxed density + energy-weighted density: shared with the
        // double-hybrid gradient path (only the Z-vector operator differs,
        // and z is already solved above).
        auto rel_res = build_rmp2_energy_weighted_density(
            calculator, shell_pairs, result, lag, z, presolved.ks_veff);
        if (!rel_res)
            return std::unexpected(rel_res.error());
        const RMP2RelaxedDensity &rel = *rel_res;
        const Eigen::MatrixXd &P_mo = rel.P_mo;
        const Eigen::MatrixXd &P_ao = rel.P_ao;
        const Eigen::MatrixXd &zeta_ao = rel.zeta_ao;
        const Eigen::MatrixXd &dm1_corr_relaxed_ao = rel.dm1_corr_relaxed_ao;
        const Eigen::MatrixXd &vhf_s1occ = rel.vhf_s1occ_ao;
        imat_ao = rel.imat_ao; // symmetrized, used by the s_im1 overlap term below

        const Eigen::MatrixXd dm1_total_ao = hf_dm1 + dm1_corr_relaxed_ao;
        const Eigen::MatrixXd dm1p = hf_dm1 + 2.0 * dm1_corr_relaxed_ao;

        // Skin: everything the Cartesian derivative kernels contract is lifted
        // once (identity in Cartesian mode; exact in spherical, since the
        // transform does not depend on geometry).
        auto cart = lift_to_cartesian(calculator._shells,
                                      {&dm1_total_ao, &imat_ao, &zeta_ao, &vhf_s1occ, &hf_dm1, &dm1p});
        if (!cart)
            return std::unexpected("RMP2 gradient: " + cart.error());
        const Eigen::MatrixXd &dm1_total_c = (*cart)[0];
        const Eigen::MatrixXd &imat_c = (*cart)[1];
        const Eigen::MatrixXd &zeta_c = (*cart)[2];
        const Eigen::MatrixXd &vhf_s1occ_c = (*cart)[3];
        const Eigen::MatrixXd &hf_dm1_c = (*cart)[4];
        const Eigen::MatrixXd &dm1p_c = (*cart)[5];

        const auto one_e_res = HartreeFock::Gradient::one_electron_gradient(calculator, shell_pairs, dm1_total_c);
        if (!one_e_res)
            return std::unexpected(one_e_res.error());
        const auto &one_e_terms = *one_e_res;
        electronic += one_e_terms.core();

        maybe_print_rmp2_matrix("z", z);
        maybe_print_rmp2_matrix("P_mo", P_mo);
        maybe_print_rmp2_matrix("P_ao", P_ao);
        maybe_print_rmp2_matrix("dm1_corr_relaxed_ao", dm1_corr_relaxed_ao);
        maybe_print_rmp2_matrix("dm1p", dm1p);

        Eigen::MatrixXd overlap_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd s_im1_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd s_zeta_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd s_vhf_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        for (std::size_t atom = 0; atom < atom_aos.size(); ++atom)
            for (int p : atom_aos[atom])
                for (int nu = 0; nu < nao_cart; ++nu)
                {
                    const HartreeFock::ShellPair sp(bfs[p], bfs[nu]);
                    const auto dST = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
                    for (int q = 0; q < 3; ++q)
                    {
                        const double s_im1 =
                            dST[q] * imat_c(p, nu) +
                            dST[q] * imat_c(nu, p);
                        const double s_zeta =
                            -dST[q] * zeta_c(p, nu) -
                            dST[q] * zeta_c(nu, p);
                        const double s_vhf =
                            -2.0 * dST[q] * vhf_s1occ_c(p, nu);
                        s_im1_terms(atom, q) += s_im1;
                        s_zeta_terms(atom, q) += s_zeta;
                        s_vhf_terms(atom, q) += s_vhf;
                        overlap_terms(atom, q) += s_im1 + s_zeta + s_vhf;
                    }
                }
        electronic += overlap_terms;

        // Two-electron derivative term: one call to the shared SCF gradient loop
        // (threads, Schwarz screening, Auto's derivative block kernels). Γ is
        // the coefficient the old per-atom sweep contracted with the bra
        // derivative ∂_p(pq|rs): the non-separable 2 dm2buf (dense path only;
        // under RI it comes from the 3-centre factors above) plus the separable
        // HF x relaxed-density term. Symmetrized over r <-> s, both are 8-fold
        // symmetric (T~_ij^ab = 4t - 2t is symmetric under (i,a) <-> (j,b)), so
        // the loop's 1/4 Σ Γ d(pq|rs)/dR equals Σ_{p on A} Γ ∂_p(pq|rs).
        const bool dense_pair = !calculator._mp2.use_ri;
        // Separable Γ: an outer product of two AO matrices, so lifting each
        // factor is exact (every AO index belongs to one factor).
        const auto separable = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double
        {
            return hf_dm1_c(p, q) * dm1p_c(r, s) - 0.5 * hf_dm1_c(p, s) * dm1p_c(r, q) +
                   hf_dm1_c(r, s) * dm1p_c(p, q) - 0.5 * hf_dm1_c(q, r) * dm1p_c(p, s);
        };
        // Pair density over the Cartesian basis: rebuilt from Tᵀ C in spherical
        // mode (exactly the 4-index lift), the Lagrangian's own buffer otherwise.
        std::vector<double> dm2_cart_local;
        if (dense_pair && calculator._shells._spherical)
            dm2_cart_local = rmp2_pair_density(
                result, HartreeFock::BasisFunctions::mos_to_cartesian(calculator._shells, result.mo_coeff));
        const std::vector<double> &dm2_cart = calculator._shells._spherical ? dm2_cart_local : dm2buf_full;
        const auto pair_and_separable = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double
        {
            const double pair = dense_pair
                                    ? 2.0 * dm2_cart[idx_dm2(static_cast<int>(p), static_cast<int>(q),
                                                             static_cast<int>(r), static_cast<int>(s), nao_cart)]
                                    : 0.0;
            return pair + separable(p, q, r, s);
        };
        auto two_e = HartreeFock::Gradient::two_electron_gradient(calculator, shell_pairs, pair_and_separable);
        if (!two_e)
            return std::unexpected("RMP2 gradient: " + two_e.error());
        electronic += *two_e;

        // The non-separable / separable split only feeds the debug print; it
        // costs one more sweep.
        Eigen::MatrixXd vhf1_terms;
        if (rmp2_term_debug_enabled())
        {
            auto sep = HartreeFock::Gradient::two_electron_gradient(calculator, shell_pairs, separable);
            if (!sep)
                return std::unexpected("RMP2 gradient: " + sep.error());
            vhf1_terms = *sep;
            two_e_terms += *two_e - vhf1_terms;
        }

        maybe_print_rmp2_term_matrix("two_e", two_e_terms);
        maybe_print_rmp2_term_matrix("h1", one_e_terms.core());
        maybe_print_rmp2_term_matrix("h1_kinetic", one_e_terms.kinetic);
        maybe_print_rmp2_term_matrix("h1_nuc_a", one_e_terms.nuc_a);
        maybe_print_rmp2_term_matrix("h1_nuc_c", one_e_terms.nuc_c);
        maybe_print_rmp2_term_matrix("s_im1", s_im1_terms);
        maybe_print_rmp2_term_matrix("s_zeta", s_zeta_terms);
        maybe_print_rmp2_term_matrix("s_vhf", s_vhf_terms);
        maybe_print_rmp2_term_matrix("vhf1", vhf1_terms);
        maybe_print_rmp2_term_matrix("electronic", electronic);

        RMP2GradientIntermediates out;
        out.electronic_gradient = std::move(electronic);
        out.P_mo = P_mo;
        out.P_ao = P_ao;
        out.W_ao = rel.W_ao;
        out.P_total_ao = P_ao;
        out.P_gamma_ao = hf_dm1 + 2.0 * dm1_corr_relaxed_ao;
        out.im1_ao = imat_ao; // == rel.imat_ao (symmetrized)
        out.zeta_ao = zeta_ao;
        out.vhf_s1occ_ao = vhf_s1occ;
        out.overlap_gradient = overlap_terms;
        return out;
    }

    std::expected<UMP2GradientIntermediates, std::string> build_ump2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UMP2Result &result)
    {
        if (result.t2_aa.empty() || result.t2_ab.empty() || result.t2_bb.empty())
            return std::unexpected("build_ump2_gradient_intermediates: T2 amplitudes are required.");
        // Without this, mp2_frozen produced a gradient 4.5e-2 Ha/Bohr off the
        // energy FD with no warning (water cation STO-3G, O z).
        if (has_frozen_orbitals(result.active_mo_alpha.size(), calculator._info._scf.alpha.mo_coefficients) ||
            has_frozen_orbitals(result.active_mo_beta.size(), calculator._info._scf.beta.mo_coefficients))
            return std::unexpected("build_ump2_gradient_intermediates: frozen-orbital gradients are not implemented.");

        auto gamma_res = ump2_gamma1_intermediates(result);
        if (!gamma_res)
            return std::unexpected(gamma_res.error());
        const auto &[doo, dvv] = *gamma_res;
        const auto &[dooa, doob] = doo;
        const auto &[dvva, dvvb] = dvv;

        // nao: the working basis (Lagrangian, densities); nao_cart: the
        // Cartesian basis the derivative kernels contract over.
        const int nao = static_cast<int>(calculator.working_nbasis());
        const int nao_cart = static_cast<int>(calculator._shells.nbasis());
        const int nocca = result.nocca;
        const int noccb = result.noccb;
        const int nvira = result.nvira;
        const int nvirb = result.nvirb;

        const Eigen::MatrixXd Ca_occ = result.mo_coeff_alpha.leftCols(nocca);
        const Eigen::MatrixXd Ca_virt = result.mo_coeff_alpha.middleCols(nocca, nvira);
        const Eigen::MatrixXd Cb_occ = result.mo_coeff_beta.leftCols(noccb);
        const Eigen::MatrixXd Cb_virt = result.mo_coeff_beta.middleCols(noccb, nvirb);

        const Eigen::MatrixXd hf_dm1a = calculator._info._scf.alpha.density;
        const Eigen::MatrixXd hf_dm1b = calculator._info._scf.beta.density;
        const Eigen::MatrixXd hf_dm1 = hf_dm1a + hf_dm1b;
        std::vector<std::vector<int>> atom_aos = build_atom_ao_lists(calculator);
        Eigen::MatrixXd electronic = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);

        // Per-spin AO separable densities D_σ[p,v,r,s], built once. They feed
        // the Lagrangians Imat_σ, and their sum D_α + D_β at (p,q,r,s) is the
        // pair coefficient of the 2e derivative term. (They used to be rebuilt
        // inside a per-atom derivative sweep, once per q for every v: N^7.)
        // Built in the working basis here (ump2_pair_densities); the 2e term
        // below uses a Cartesian copy in spherical mode.
        auto pair_densities = ump2_pair_densities(result, result.mo_coeff_alpha, result.mo_coeff_beta);
        std::vector<double> &Da = pair_densities.first;
        std::vector<double> &Db = pair_densities.second;
        const std::size_t nao4 = Da.size();

        // Lagrangians Imat_σ(q,v) = ½ Σ_{p,r,s} (pq|rs) D_σ[p,v,r,s]: the dense
        // nao⁴ ERI, or under RI the 3-center factors (Step RG4.3, the dense
        // tensor is then not built at all).
        Eigen::MatrixXd Imata = Eigen::MatrixXd::Zero(nao, nao);
        Eigen::MatrixXd Imatb = Eigen::MatrixXd::Zero(nao, nao);
        if (calculator._mp2.use_ri)
        {
            Imata = 0.5 * HartreeFock::Correlation::RI::build_ri_imat(calculator, Da, nao);
            Imatb = 0.5 * HartreeFock::Correlation::RI::build_ri_imat(calculator, Db, nao);

            // RI 2e-gradient term (Step RG4.3): contract the fitted 3-index
            // density built from the spin-summed AO separable density (Da + Db)
            // against the RG1 derivative tensors. Same bra_prefolded=true
            // convention as the RHF path -- the packing sums both (μν) and (νμ)
            // orderings.
            //
            // The ½ is a CONVENTION bridge, not physics. build_ri_two_electron_
            // gradient reproduces the RHF dense convention, whose 2e term carries
            // an explicit `2.0 * dm2buf` (PySCF grad/mp2.py: de -= ... * 2). The
            // UMP2 dense term contracts `Da + Db` with NO factor 2, so feeding it
            // straight in double-counts by exactly 2 -- measured RI/dense =
            // 2.0000 on every non-zero component, uniform across atoms and axes.
            // Halving restores the UMP2 convention.
            std::vector<double> pair_dm2_ao(nao4);
            for (std::size_t k = 0; k < nao4; ++k)
                pair_dm2_ao[k] = Da[k] + Db[k];
            auto [gamma3, x_proj] =
                HartreeFock::Correlation::RI::build_ri_gamma3_from_ao_dm2(
                    calculator, pair_dm2_ao, nao);
            auto ri_two_e = HartreeFock::Correlation::RI::build_ri_two_electron_gradient(
                calculator, gamma3, x_proj, /*bra_prefolded=*/true);
            if (!ri_two_e)
                return std::unexpected("UMP2 RI gradient: " + ri_two_e.error());
            electronic += 0.5 * *ri_two_e;
        }
        else
        {
            std::vector<double> eri_local;
            const std::vector<double> &eri =
                ensure_eri(calculator, shell_pairs, eri_local, "UMP2 Gradient :");
            // Each thread owns rows q of Imat_σ.
#pragma omp parallel for schedule(static)
            for (int q = 0; q < nao; ++q)
                for (int p = 0; p < nao; ++p)
                    for (int r = 0; r < nao; ++r)
                        for (int s = 0; s < nao; ++s)
                        {
                            const double e = 0.5 * eri[idx_dm2(p, q, r, s, nao)];
                            for (int v = 0; v < nao; ++v)
                            {
                                Imata(q, v) += e * Da[idx_dm2(p, v, r, s, nao)];
                                Imatb(q, v) += e * Db[idx_dm2(p, v, r, s, nao)];
                            }
                        }
        }
        const auto &bfs = calculator._shells._basis_functions;

        Imata = -result.mo_coeff_alpha.transpose() * Imata * calculator._overlap * result.mo_coeff_alpha;
        Imatb = -result.mo_coeff_beta.transpose() * Imatb * calculator._overlap * result.mo_coeff_beta;

        Eigen::MatrixXd dm1a_mo = Eigen::MatrixXd::Zero(result.mo_coeff_alpha.cols(), result.mo_coeff_alpha.cols());
        Eigen::MatrixXd dm1b_mo = Eigen::MatrixXd::Zero(result.mo_coeff_beta.cols(), result.mo_coeff_beta.cols());
        dm1a_mo.topLeftCorner(nocca, nocca) = 0.5 * (dooa + dooa.transpose());
        dm1a_mo.bottomRightCorner(nvira, nvira) = 0.5 * (dvva + dvva.transpose());
        dm1b_mo.topLeftCorner(noccb, noccb) = 0.5 * (doob + doob.transpose());
        dm1b_mo.bottomRightCorner(nvirb, nvirb) = 0.5 * (dvvb + dvvb.transpose());

        const Eigen::MatrixXd dm1a_ao_seed = result.mo_coeff_alpha * dm1a_mo * result.mo_coeff_alpha.transpose();
        const Eigen::MatrixXd dm1b_ao_seed = result.mo_coeff_beta * dm1b_mo * result.mo_coeff_beta.transpose();
        auto veff_seed = build_veff_from_spin_densities(calculator, shell_pairs, dm1a_ao_seed, dm1b_ao_seed);
        if (!veff_seed)
            return std::unexpected("UMP2 gradient: " + veff_seed.error());
        const auto &[va_ao, vb_ao] = *veff_seed;
        Eigen::MatrixXd Xvo = Ca_virt.transpose() * va_ao * Ca_occ +
                              Imata.topRightCorner(nocca, nvira).transpose() -
                              Imata.bottomLeftCorner(nvira, nocca);
        Eigen::MatrixXd XVO = Cb_virt.transpose() * vb_ao * Cb_occ +
                              Imatb.topRightCorner(noccb, nvirb).transpose() -
                              Imatb.bottomLeftCorner(nvirb, noccb);

        auto resp = solve_uhf_cphf(
            calculator,
            shell_pairs,
            result.mo_coeff_alpha,
            result.mo_coeff_beta,
            result.mo_energy_alpha,
            result.mo_energy_beta,
            nocca,
            noccb,
            Xvo,
            XVO);
        if (!resp)
            return std::unexpected(resp.error());
        dm1a_mo.bottomLeftCorner(nvira, nocca) = resp->alpha;
        dm1a_mo.topRightCorner(nocca, nvira) = resp->alpha.transpose();
        dm1b_mo.bottomLeftCorner(nvirb, noccb) = resp->beta;
        dm1b_mo.topRightCorner(noccb, nvirb) = resp->beta.transpose();

        Imata.bottomLeftCorner(nvira, nocca) = Imata.topRightCorner(nocca, nvira).transpose();
        Imatb.bottomLeftCorner(nvirb, noccb) = Imatb.topRightCorner(noccb, nvirb).transpose();
        const Eigen::MatrixXd im1_ao =
            result.mo_coeff_alpha * Imata * result.mo_coeff_alpha.transpose() +
            result.mo_coeff_beta * Imatb * result.mo_coeff_beta.transpose();

        Eigen::MatrixXd zeta_a_weights = Eigen::MatrixXd::Zero(result.mo_coeff_alpha.cols(), result.mo_coeff_alpha.cols());
        for (int p = 0; p < zeta_a_weights.rows(); ++p)
            for (int q = 0; q < zeta_a_weights.cols(); ++q)
                zeta_a_weights(p, q) = 0.5 * (result.mo_energy_alpha(p) + result.mo_energy_alpha(q));
        for (int a = 0; a < nvira; ++a)
            for (int i = 0; i < nocca; ++i)
            {
                zeta_a_weights(nocca + a, i) = result.mo_energy_alpha(i);
                zeta_a_weights(i, nocca + a) = result.mo_energy_alpha(i);
            }
        Eigen::MatrixXd zeta_b_weights = Eigen::MatrixXd::Zero(result.mo_coeff_beta.cols(), result.mo_coeff_beta.cols());
        for (int p = 0; p < zeta_b_weights.rows(); ++p)
            for (int q = 0; q < zeta_b_weights.cols(); ++q)
                zeta_b_weights(p, q) = 0.5 * (result.mo_energy_beta(p) + result.mo_energy_beta(q));
        for (int a = 0; a < nvirb; ++a)
            for (int i = 0; i < noccb; ++i)
            {
                zeta_b_weights(noccb + a, i) = result.mo_energy_beta(i);
                zeta_b_weights(i, noccb + a) = result.mo_energy_beta(i);
            }

        Eigen::MatrixXd Pa_corr = result.mo_coeff_alpha * dm1a_mo * result.mo_coeff_alpha.transpose();
        Eigen::MatrixXd Pb_corr = result.mo_coeff_beta * dm1b_mo * result.mo_coeff_beta.transpose();
        const Eigen::MatrixXd zeta_a =
            result.mo_coeff_alpha * zeta_a_weights.cwiseProduct(dm1a_mo) * result.mo_coeff_alpha.transpose() +
            Ca_occ * result.mo_energy_alpha.head(nocca).asDiagonal() * Ca_occ.transpose();
        const Eigen::MatrixXd zeta_b =
            result.mo_coeff_beta * zeta_b_weights.cwiseProduct(dm1b_mo) * result.mo_coeff_beta.transpose() +
            Cb_occ * result.mo_energy_beta.head(noccb).asDiagonal() * Cb_occ.transpose();
        const Eigen::MatrixXd zeta = zeta_a + zeta_b;

        auto veff_s1 = build_veff_from_spin_densities(
            calculator, shell_pairs, Pa_corr + Pa_corr.transpose(), Pb_corr + Pb_corr.transpose());
        if (!veff_s1)
            return std::unexpected("UMP2 gradient: " + veff_s1.error());
        const auto &[vs1a, vs1b] = *veff_s1;
        const Eigen::MatrixXd p1a = Ca_occ * Ca_occ.transpose();
        const Eigen::MatrixXd p1b = Cb_occ * Cb_occ.transpose();
        const Eigen::MatrixXd vhf_s1occ =
            0.5 * (p1a * vs1a * p1a + p1b * vs1b * p1b);

        const Eigen::MatrixXd dm1pa = hf_dm1a + 2.0 * Pa_corr;
        const Eigen::MatrixXd dm1pb = hf_dm1b + 2.0 * Pb_corr;
        const Eigen::MatrixXd dm1tot = hf_dm1a + hf_dm1b + Pa_corr + Pb_corr;
        const Eigen::MatrixXd zeta_total = zeta;

        // Skin: everything the Cartesian derivative kernels contract is lifted
        // once (identity in Cartesian mode).
        auto cart = lift_to_cartesian(calculator._shells, {&dm1tot, &im1_ao, &zeta_total, &vhf_s1occ,
                                                           &hf_dm1, &hf_dm1a, &hf_dm1b, &dm1pa, &dm1pb});
        if (!cart)
            return std::unexpected("UMP2 gradient: " + cart.error());
        const Eigen::MatrixXd &dm1tot_c = (*cart)[0];
        const Eigen::MatrixXd &im1_c = (*cart)[1];
        const Eigen::MatrixXd &zeta_c = (*cart)[2];
        const Eigen::MatrixXd &vhf_s1occ_c = (*cart)[3];
        const Eigen::MatrixXd &hf_dm1_c = (*cart)[4];
        const Eigen::MatrixXd &hf_dm1a_c = (*cart)[5];
        const Eigen::MatrixXd &hf_dm1b_c = (*cart)[6];
        const Eigen::MatrixXd &dm1pa_c = (*cart)[7];
        const Eigen::MatrixXd &dm1pb_c = (*cart)[8];

        const auto one_e = HartreeFock::Gradient::one_electron_gradient(calculator, shell_pairs, dm1tot_c);
        if (!one_e)
            return std::unexpected(one_e.error());
        electronic += one_e->core();

        Eigen::MatrixXd overlap_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        for (std::size_t atom = 0; atom < atom_aos.size(); ++atom)
            for (int p : atom_aos[atom])
                for (int nu = 0; nu < nao_cart; ++nu)
                {
                    const HartreeFock::ShellPair sp(bfs[p], bfs[nu]);
                    const auto dST = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
                    for (int q = 0; q < 3; ++q)
                    {
                        overlap_terms(atom, q) += dST[q] * im1_c(p, nu);
                        overlap_terms(atom, q) += dST[q] * im1_c(nu, p);
                        overlap_terms(atom, q) -= dST[q] * zeta_c(p, nu);
                        overlap_terms(atom, q) -= dST[q] * zeta_c(nu, p);
                        overlap_terms(atom, q) -= 2.0 * dST[q] * vhf_s1occ_c(p, nu);
                    }
                }
        electronic += overlap_terms;

        // Two-electron derivative term: one call to the shared SCF gradient loop,
        // as in RHF. Γ is the coefficient the old per-atom sweep contracted with
        // ∂_p(pq|rs): the pair density D_α + D_β (dense path only; RI handled it
        // above) plus the separable HF x relaxed-density term, per spin. Γ is
        // indexed over the Cartesian basis: separable factors lifted (exact for
        // an outer product), pair densities rebuilt from Tᵀ C in spherical mode.
        const bool dense_pair = !calculator._mp2.use_ri;
        std::vector<double> Da_cart_local, Db_cart_local;
        if (dense_pair && calculator._shells._spherical)
        {
            Da = {};
            Db = {}; // the working copies are done (Lagrangians above)
            std::tie(Da_cart_local, Db_cart_local) = ump2_pair_densities(
                result, HartreeFock::BasisFunctions::mos_to_cartesian(calculator._shells, result.mo_coeff_alpha),
                HartreeFock::BasisFunctions::mos_to_cartesian(calculator._shells, result.mo_coeff_beta));
        }
        const std::vector<double> &Da_c = calculator._shells._spherical ? Da_cart_local : Da;
        const std::vector<double> &Db_c = calculator._shells._spherical ? Db_cart_local : Db;
        const auto gamma_u = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double
        {
            double g = hf_dm1_c(p, q) * (dm1pa_c(r, s) + dm1pb_c(r, s)) -
                       hf_dm1a_c(p, s) * dm1pa_c(r, q) - hf_dm1b_c(p, s) * dm1pb_c(r, q) +
                       hf_dm1_c(r, s) * (dm1pa_c(p, q) + dm1pb_c(p, q)) -
                       hf_dm1a_c(q, r) * dm1pa_c(p, s) - hf_dm1b_c(q, r) * dm1pb_c(p, s);
            if (dense_pair)
            {
                const std::size_t k = idx_dm2(static_cast<int>(p), static_cast<int>(q),
                                              static_cast<int>(r), static_cast<int>(s), nao_cart);
                g += Da_c[k] + Db_c[k];
            }
            return g;
        };
        auto two_e = HartreeFock::Gradient::two_electron_gradient(calculator, shell_pairs, gamma_u);
        if (!two_e)
            return std::unexpected("UMP2 gradient: " + two_e.error());
        electronic += *two_e;

        UMP2GradientIntermediates out;
        out.electronic_gradient = std::move(electronic);
        out.P_alpha_corr_ao = Pa_corr;
        out.P_beta_corr_ao = Pb_corr;
        out.P_alpha_ao = hf_dm1a + Pa_corr;
        out.P_beta_ao = hf_dm1b + Pb_corr;
        out.P_total_ao = out.P_alpha_ao + out.P_beta_ao;
        out.W_ao = zeta_total;
        return out;
    }
} // namespace HartreeFock::Correlation
