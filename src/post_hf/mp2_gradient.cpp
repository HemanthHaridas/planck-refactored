#include "mp2_gradient.h"

#include <cstdlib>
#include <expected>
#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>

#include "gradient/gradient.h"
#include "integrals/base.h"
#include "post_hf/integrals.h"
#include "post_hf/mp2_internal.h"
#include "post_hf/rhf_response.h"
#include "post_hf/uhf_response.h"
#include "post_hf/ri/ri_eri.h"

namespace
{
    void maybe_print_rmp2_term_matrix(const char *name, const Eigen::MatrixXd &term)
    {
        const char *enabled = std::getenv("PLANCK_DEBUG_RMP2_TERMS");
        if (enabled == nullptr || std::string_view(enabled) != "1")
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

    Eigen::MatrixXd build_veff_from_density(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density)
    {
        // RI-consistent veff G = J - 1/2 K when MP2 RI is enabled (Step RG3.2).
        // build_ri_fock_rhf returns the same J - 1/2 K as _compute_2e_fock, so
        // the caller factor (2· at veff_corr, projector sandwich at vhf_s1occ)
        // is unchanged. Density is symmetric at both call sites.
        if (calculator._mp2.use_ri)
            return HartreeFock::Correlation::RI::build_ri_fock_rhf(calculator, density);
        return _compute_2e_fock(
            shell_pairs,
            density,
            calculator._shells.nbasis(),
            calculator._integral._engine,
            HartreeFock::ERIKernel::Coulomb,
            0.0,
            calculator._integral._tol_eri,
            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> build_veff_from_spin_densities(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density_alpha,
        const Eigen::MatrixXd &density_beta)
    {
        return _compute_2e_fock_uhf(
            shell_pairs,
            density_alpha,
            density_beta,
            calculator._shells.nbasis(),
            calculator._integral._engine,
            HartreeFock::ERIKernel::Coulomb,
            0.0,
            calculator._integral._tol_eri,
            calculator._use_integral_symmetry ? &calculator._integral_symmetry_ops : nullptr);
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
                const double Za = static_cast<double>(mol.atomic_numbers[a]);
                const double Zb = static_cast<double>(mol.atomic_numbers[b]);
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

    struct OneElectronGradientTerms
    {
        Eigen::MatrixXd total;
        Eigen::MatrixXd kinetic;
        Eigen::MatrixXd nuc_a;
        Eigen::MatrixXd nuc_c;
    };

    OneElectronGradientTerms one_electron_gradient_terms_from_density(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> & /*shell_pairs*/,
        const Eigen::MatrixXd &density)
    {
        OneElectronGradientTerms terms{
            .total = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3),
            .kinetic = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3),
            .nuc_a = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3),
            .nuc_c = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3)};
        const auto &mol = calculator._molecule;
        const int nao = static_cast<int>(calculator._shells.nbasis());
        const auto &bfs = calculator._shells._basis_functions;
        const auto atom_aos = build_atom_ao_lists(calculator);
        for (std::size_t atom = 0; atom < mol.natoms; ++atom)
        {
            std::array<Eigen::MatrixXd, 3> kinetic_ao{
                Eigen::MatrixXd::Zero(nao, nao),
                Eigen::MatrixXd::Zero(nao, nao),
                Eigen::MatrixXd::Zero(nao, nao)};
            std::array<Eigen::MatrixXd, 3> nuc_a_ao{
                Eigen::MatrixXd::Zero(nao, nao),
                Eigen::MatrixXd::Zero(nao, nao),
                Eigen::MatrixXd::Zero(nao, nao)};
            std::array<Eigen::MatrixXd, 3> nuc_c_ao{
                Eigen::MatrixXd::Zero(nao, nao),
                Eigen::MatrixXd::Zero(nao, nao),
                Eigen::MatrixXd::Zero(nao, nao)};

            for (int p : atom_aos[atom])
                for (int nu = 0; nu < nao; ++nu)
                {
                    const HartreeFock::ShellPair sp(bfs[p], bfs[nu]);
                    const auto dST_A = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
                    const auto dV_A = HartreeFock::ObaraSaika::_compute_nuclear_deriv_A_elem(sp, mol);
                    for (int q = 0; q < 3; ++q)
                    {
                        kinetic_ao[q](p, nu) += dST_A[q + 3];
                        nuc_a_ao[q](p, nu) += dV_A[q];
                    }
                }

            const double Z = static_cast<double>(mol.atomic_numbers[atom]);
            const Eigen::Vector3d center(mol._standard(atom, 0), mol._standard(atom, 1), mol._standard(atom, 2));
            for (int q = 0; q < 3; ++q)
            {
                kinetic_ao[q] = kinetic_ao[q] + kinetic_ao[q].transpose().eval();
                nuc_a_ao[q] = nuc_a_ao[q] + nuc_a_ao[q].transpose().eval();
                for (int ii = 0; ii < nao; ++ii)
                    for (int jj = 0; jj <= ii; ++jj)
                    {
                        const HartreeFock::ShellPair sp(bfs[ii], bfs[jj]);
                        const double dv = HartreeFock::ObaraSaika::_compute_nuclear_deriv_C_elem(sp, center, Z, q);
                        nuc_c_ao[q](ii, jj) += dv;
                        if (ii != jj)
                            nuc_c_ao[q](jj, ii) += dv;
                    }
                terms.kinetic(atom, q) = (kinetic_ao[q].cwiseProduct(density.transpose())).sum();
                terms.nuc_a(atom, q) = (nuc_a_ao[q].cwiseProduct(density.transpose())).sum();
                terms.nuc_c(atom, q) = (nuc_c_ao[q].cwiseProduct(density.transpose())).sum();
                terms.total(atom, q) = terms.kinetic(atom, q) + terms.nuc_a(atom, q) + terms.nuc_c(atom, q);
            }
        }
        return terms;
    }

    Eigen::MatrixXd one_electron_gradient_from_density(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density)
    {
        return one_electron_gradient_terms_from_density(calculator, shell_pairs, density).total;
    }

}

namespace HartreeFock::Correlation
{
    std::expected<RMP2GradientIntermediates, std::string> build_rmp2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Result &result)
    {
        if (result.t2.empty())
            return std::unexpected("build_rmp2_gradient_intermediates: T2 amplitudes are required.");
        if (!result.active_mo.empty() &&
            static_cast<int>(result.active_mo.size()) != static_cast<int>(calculator._shells.nbasis()))
            return std::unexpected("build_rmp2_gradient_intermediates: frozen-orbital gradients are not implemented.");

        const int nao = static_cast<int>(calculator._shells.nbasis());
        const int nocc = result.n_occ;
        const int nvirt = result.n_virt;
        const int nmo = nocc + nvirt;

        const Eigen::MatrixXd C_occ = result.mo_coeff.leftCols(nocc);
        const Eigen::MatrixXd C_virt = result.mo_coeff.middleCols(nocc, nvirt);
        const Eigen::VectorXd eps_occ = result.mo_energy.head(nocc);

        auto gamma_res = rmp2_gamma1_intermediates(result);
        if (!gamma_res)
            return std::unexpected(gamma_res.error());
        const auto &[doo, dvv] = *gamma_res;

        Eigen::MatrixXd dm1_corr_mo = Eigen::MatrixXd::Zero(nmo, nmo);
        dm1_corr_mo.topLeftCorner(nocc, nocc) = doo + doo.transpose();
        dm1_corr_mo.bottomRightCorner(nvirt, nvirt) = dvv + dvv.transpose();
        const Eigen::MatrixXd dm1_corr_ao = result.mo_coeff * dm1_corr_mo * result.mo_coeff.transpose();

        std::vector<double> eri_local;
        const std::vector<double> &eri = ensure_eri(
            calculator, shell_pairs, eri_local, "RMP2 Gradient :");

        std::vector<double> pair_dm2_ao(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                for (int la = 0; la < nao; ++la)
                    for (int si = 0; si < nao; ++si)
                    {
                        double val = 0.0;
                        for (int i = 0; i < nocc; ++i)
                            for (int j = 0; j < nocc; ++j)
                                for (int a = 0; a < nvirt; ++a)
                                    for (int b = 0; b < nvirt; ++b)
                                    {
                                        const double tab = result.t2[detail::idx_t2(i, j, a, b, nocc, nvirt)];
                                        const double tba = result.t2[detail::idx_t2(i, j, b, a, nocc, nvirt)];
                                        const double dovov = 4.0 * tab - 2.0 * tba;
                                        val += C_occ(mu, i) * C_virt(nu, a) * C_occ(la, j) * C_virt(si, b) * dovov;
                                        val += C_virt(mu, a) * C_occ(nu, i) * C_virt(la, b) * C_occ(si, j) * dovov;
                                    }
                        pair_dm2_ao[idx_dm2(mu, nu, la, si, nao)] = val;
                    }

        std::vector<std::vector<int>> atom_aos = build_atom_ao_lists(calculator);

        std::vector<double> part_dm2(static_cast<std::size_t>(nocc) * nao * nao * nocc, 0.0);
        auto idx_part = [nao, nocc](int i, int p, int q, int j) -> std::size_t
        {
            return ((static_cast<std::size_t>(i) * nao + p) * nao + q) * nocc + j;
        };
        for (int i = 0; i < nocc; ++i)
            for (int j = 0; j < nocc; ++j)
                for (int p = 0; p < nao; ++p)
                    for (int q = 0; q < nao; ++q)
                    {
                        double val = 0.0;
                        for (int a = 0; a < nvirt; ++a)
                            for (int b = 0; b < nvirt; ++b)
                            {
                                const double tab = result.t2[detail::idx_t2(i, j, a, b, nocc, nvirt)];
                                const double tba = result.t2[detail::idx_t2(i, j, b, a, nocc, nvirt)];
                                val += C_virt(p, a) * C_virt(q, b) * (4.0 * tab - 2.0 * tba);
                            }
                        part_dm2[idx_part(i, p, q, j)] = val;
                    }

        // dm2buf_full = einsum('pi,iqrj,sj->pqrs') + einsum('qi,iprj,sj->pqrs').
        // PySCF additionally symmetrizes r<->s (dm2buf + dm2buf.T(0,1,3,2)) but only
        // because it contracts against s2kl-packed ERIs with a diagonal-halving
        // correction. We contract against full (unpacked) ERIs/derivatives, so the
        // r<->s symmetrization would double-count the pair — it is omitted here.
        std::vector<double> dm2buf_full(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
        for (int p = 0; p < nao; ++p)
            for (int q = 0; q < nao; ++q)
                for (int r = 0; r < nao; ++r)
                    for (int s = 0; s < nao; ++s)
                    {
                        double val = 0.0;
                        for (int i = 0; i < nocc; ++i)
                            for (int j = 0; j < nocc; ++j)
                            {
                                val += C_occ(p, i) * part_dm2[idx_part(i, q, r, j)] * C_occ(s, j);
                                val += C_occ(q, i) * part_dm2[idx_part(i, p, r, j)] * C_occ(s, j);
                            }
                        dm2buf_full[idx_dm2(p, q, r, s, nao)] = val;
                    }

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
        Eigen::MatrixXd vhf1_rs_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd vhf1_rq_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd vhf1_pq_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd vhf1_ps_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd imat_ao = Eigen::MatrixXd::Zero(nao, nao);
        std::vector<std::array<Eigen::MatrixXd, 3>> vhf1(calculator._molecule.natoms);
        for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
            for (int q = 0; q < 3; ++q)
                vhf1[atom][q] = Eigen::MatrixXd::Zero(nao, nao);

        const auto &bfs = calculator._shells._basis_functions;
        const Eigen::MatrixXd hf_dm1 = 2.0 * C_occ * C_occ.transpose();
        for (std::size_t atom = 0; atom < atom_aos.size(); ++atom)
        {
            for (int p : atom_aos[atom])
                for (int q = 0; q < nao; ++q)
                    for (int r = 0; r < nao; ++r)
                        for (int s = 0; s < nao; ++s)
                        {
                            const HartreeFock::ShellPair spAB(bfs[p], bfs[q]);
                            const HartreeFock::ShellPair spCD(bfs[r], bfs[s]);
                            const auto dI = HartreeFock::Gradient::compute_eri_deriv_dispatch(calculator, spAB, spCD);
                            // PySCF contracts the pair density with the ERI derivative
                            // with an overall factor of 2 (grad/mp2.py: de -= ... * 2);
                            // dm2buf_full carries only one bra-derivative permutation here.
                            const double dm2v = 2.0 * dm2buf_full[idx_dm2(p, q, r, s, nao)];
                            for (int comp = 0; comp < 3; ++comp)
                            {
                                two_e_terms(atom, comp) += dI[comp] * dm2v;
                                electronic(atom, comp) += dI[comp] * dm2v;
                            }

                            // Imat(q,v) += sum_{p,r,s} (pq|rs) * dm2buf[p,v,r,s].
                            // dm2buf_full omits the r<->s symmetrization, so the
                            // full-ERI contraction matches PySCF's packed Imat with
                            // factor 1 (verified element-wise: ratio 1.0).
                            const double eri_pqrs = eri[idx_dm2(p, q, r, s, nao)];
                            for (int v = 0; v < nao; ++v)
                                imat_ao(q, v) += eri_pqrs * dm2buf_full[idx_dm2(p, v, r, s, nao)];

                            for (int comp = 0; comp < 3; ++comp)
                            {
                                vhf1[atom][comp](r, s) += dI[comp] * hf_dm1(p, q);
                                vhf1[atom][comp](r, q) -= 0.5 * dI[comp] * hf_dm1(p, s);
                                vhf1[atom][comp](p, q) += dI[comp] * hf_dm1(r, s);
                                vhf1[atom][comp](p, s) -= 0.5 * dI[comp] * hf_dm1(q, r);
                            }
                        }
        }
        imat_ao = -imat_ao;
        Eigen::MatrixXd imat_mo = result.mo_coeff.transpose() * imat_ao * calculator._overlap * result.mo_coeff;
        const Eigen::MatrixXd veff_corr_ao = 2.0 * build_veff_from_density(calculator, shell_pairs, dm1_corr_ao);

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

        Eigen::MatrixXd Xvo =
            C_virt.transpose() * veff_corr_ao * C_occ +
            imat_mo.topRightCorner(nocc, nvirt).transpose() -
            imat_mo.bottomLeftCorner(nvirt, nocc);
        auto z_res = solve_rhf_cphf(calculator, shell_pairs, result.mo_coeff, result.mo_energy, Xvo);
        if (!z_res)
            return std::unexpected(z_res.error());
        const Eigen::MatrixXd &z = *z_res;

        Eigen::MatrixXd corr_relaxed_mo = dm1_corr_mo;
        corr_relaxed_mo.bottomLeftCorner(nvirt, nocc) = z;
        corr_relaxed_mo.topRightCorner(nocc, nvirt) = z.transpose();

        Eigen::MatrixXd P_mo = Eigen::MatrixXd::Zero(nmo, nmo);
        P_mo.topLeftCorner(nocc, nocc) = 2.0 * Eigen::MatrixXd::Identity(nocc, nocc);
        P_mo += corr_relaxed_mo;
        const Eigen::MatrixXd P_ao = result.mo_coeff * P_mo * result.mo_coeff.transpose();

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
        const Eigen::MatrixXd zeta_ao =
            W_ref + result.mo_coeff * zeta_weights.cwiseProduct(corr_relaxed_mo) * result.mo_coeff.transpose();

        imat_mo.bottomLeftCorner(nvirt, nocc) = imat_mo.topRightCorner(nocc, nvirt).transpose();
        imat_ao = result.mo_coeff * imat_mo * result.mo_coeff.transpose();

        const Eigen::MatrixXd occ_projector = C_occ * C_occ.transpose();
        const Eigen::MatrixXd dm1_corr_relaxed_ao = P_ao - hf_dm1;
        const Eigen::MatrixXd vhf_s1occ =
            occ_projector *
            build_veff_from_density(calculator, shell_pairs, dm1_corr_relaxed_ao + dm1_corr_relaxed_ao.transpose()) *
            occ_projector;

        const Eigen::MatrixXd dm1_total_ao = hf_dm1 + dm1_corr_relaxed_ao;
        const auto one_e_terms = one_electron_gradient_terms_from_density(calculator, shell_pairs, dm1_total_ao);
        electronic += one_e_terms.total;

        const Eigen::MatrixXd dm1p = hf_dm1 + 2.0 * dm1_corr_relaxed_ao;

        maybe_print_rmp2_matrix("z", z);
        maybe_print_rmp2_matrix("corr_relaxed_mo", corr_relaxed_mo);
        maybe_print_rmp2_matrix("P_ao", P_ao);
        maybe_print_rmp2_matrix("dm1_corr_relaxed_ao", dm1_corr_relaxed_ao);
        maybe_print_rmp2_matrix("dm1p", dm1p);

        Eigen::MatrixXd overlap_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd s_im1_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd s_zeta_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        Eigen::MatrixXd s_vhf_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        for (std::size_t atom = 0; atom < atom_aos.size(); ++atom)
            for (int p : atom_aos[atom])
                for (int nu = 0; nu < nao; ++nu)
                {
                    const HartreeFock::ShellPair sp(bfs[p], bfs[nu]);
                    const auto dST = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
                    for (int q = 0; q < 3; ++q)
                    {
                        const double s_im1 =
                            dST[q] * imat_ao(p, nu) +
                            dST[q] * imat_ao(nu, p);
                        const double s_zeta =
                            -dST[q] * zeta_ao(p, nu) -
                            dST[q] * zeta_ao(nu, p);
                        const double s_vhf =
                            -2.0 * dST[q] * vhf_s1occ(p, nu);
                        s_im1_terms(atom, q) += s_im1;
                        s_zeta_terms(atom, q) += s_zeta;
                        s_vhf_terms(atom, q) += s_vhf;
                        overlap_terms(atom, q) += s_im1 + s_zeta + s_vhf;
                    }
                }
        electronic += overlap_terms;

        Eigen::MatrixXd vhf1_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
            for (int q = 0; q < 3; ++q)
            {
                vhf1_terms(atom, q) = (vhf1[atom][q].cwiseProduct(dm1p)).sum();
                electronic(atom, q) += vhf1_terms(atom, q);
            }

        for (std::size_t atom = 0; atom < atom_aos.size(); ++atom)
            for (int p : atom_aos[atom])
                for (int q = 0; q < nao; ++q)
                    for (int r = 0; r < nao; ++r)
                        for (int s = 0; s < nao; ++s)
                        {
                            const HartreeFock::ShellPair spAB(bfs[p], bfs[q]);
                            const HartreeFock::ShellPair spCD(bfs[r], bfs[s]);
                            const auto dI = HartreeFock::Gradient::compute_eri_deriv_dispatch(calculator, spAB, spCD);
                            for (int comp = 0; comp < 3; ++comp)
                            {
                                vhf1_rs_terms(atom, comp) += dI[comp] * hf_dm1(p, q) * dm1p(r, s);
                                vhf1_rq_terms(atom, comp) -= 0.5 * dI[comp] * hf_dm1(p, s) * dm1p(r, q);
                                vhf1_pq_terms(atom, comp) += dI[comp] * hf_dm1(r, s) * dm1p(p, q);
                                vhf1_ps_terms(atom, comp) -= 0.5 * dI[comp] * hf_dm1(q, r) * dm1p(p, s);
                            }
                        }

        maybe_print_rmp2_term_matrix("two_e", two_e_terms);
        maybe_print_rmp2_term_matrix("h1", one_e_terms.total);
        maybe_print_rmp2_term_matrix("h1_kinetic", one_e_terms.kinetic);
        maybe_print_rmp2_term_matrix("h1_nuc_a", one_e_terms.nuc_a);
        maybe_print_rmp2_term_matrix("h1_nuc_c", one_e_terms.nuc_c);
        maybe_print_rmp2_term_matrix("s_im1", s_im1_terms);
        maybe_print_rmp2_term_matrix("s_zeta", s_zeta_terms);
        maybe_print_rmp2_term_matrix("s_vhf", s_vhf_terms);
        maybe_print_rmp2_term_matrix("vhf1", vhf1_terms);
        maybe_print_rmp2_term_matrix("vhf1_rs", vhf1_rs_terms);
        maybe_print_rmp2_term_matrix("vhf1_rq", vhf1_rq_terms);
        maybe_print_rmp2_term_matrix("vhf1_pq", vhf1_pq_terms);
        maybe_print_rmp2_term_matrix("vhf1_ps", vhf1_ps_terms);
        maybe_print_rmp2_term_matrix("electronic", electronic);

        RMP2GradientIntermediates out;
        out.electronic_gradient = std::move(electronic);
        out.P_mo = P_mo;
        out.P_ao = P_ao;
        out.W_ao = 0.5 * (zeta_ao + zeta_ao.transpose() - imat_ao - imat_ao.transpose()) + vhf_s1occ;
        out.P_total_ao = P_ao;
        out.P_gamma_ao = hf_dm1 + 2.0 * dm1_corr_relaxed_ao;
        out.im1_ao = std::move(imat_ao);
        out.zeta_ao = zeta_ao;
        out.vhf_s1occ_ao = vhf_s1occ;
        out.Gamma_pair_ao = std::move(pair_dm2_ao);
        return out;
    }

    std::expected<UMP2GradientIntermediates, std::string> build_ump2_gradient_intermediates(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UMP2Result &result)
    {
        if (result.t2_aa.empty() || result.t2_ab.empty() || result.t2_bb.empty())
            return std::unexpected("build_ump2_gradient_intermediates: T2 amplitudes are required.");

        auto gamma_res = ump2_gamma1_intermediates(result);
        if (!gamma_res)
            return std::unexpected(gamma_res.error());
        const auto &[doo, dvv] = *gamma_res;
        const auto &[dooa, doob] = doo;
        const auto &[dvva, dvvb] = dvv;

        const int nao = static_cast<int>(calculator._shells.nbasis());
        const int nocca = result.nocca;
        const int noccb = result.noccb;
        const int nvira = result.nvira;
        const int nvirb = result.nvirb;

        const Eigen::MatrixXd Ca_occ = result.mo_coeff_alpha.leftCols(nocca);
        const Eigen::MatrixXd Ca_virt = result.mo_coeff_alpha.middleCols(nocca, nvira);
        const Eigen::MatrixXd Cb_occ = result.mo_coeff_beta.leftCols(noccb);
        const Eigen::MatrixXd Cb_virt = result.mo_coeff_beta.middleCols(noccb, nvirb);

        auto idx_aa_part = [nao, nocca](int i, int p, int q, int j) -> std::size_t
        {
            return ((static_cast<std::size_t>(i) * nao + p) * nao + q) * nocca + j;
        };
        auto idx_bb_part = [nao, noccb](int i, int p, int q, int j) -> std::size_t
        {
            return ((static_cast<std::size_t>(i) * nao + p) * nao + q) * noccb + j;
        };
        auto idx_ab_part = [nao, noccb](int i, int p, int q, int j) -> std::size_t
        {
            return ((static_cast<std::size_t>(i) * nao + p) * nao + q) * noccb + j;
        };

        std::vector<double> part_dm2aa(static_cast<std::size_t>(nocca) * nao * nao * nocca, 0.0);
        std::vector<double> part_dm2bb(static_cast<std::size_t>(noccb) * nao * nao * noccb, 0.0);
        std::vector<double> part_dm2ab(static_cast<std::size_t>(nocca) * nao * nao * noccb, 0.0);

        for (int i = 0; i < nocca; ++i)
            for (int j = 0; j < nocca; ++j)
                for (int p = 0; p < nao; ++p)
                    for (int q = 0; q < nao; ++q)
                    {
                        double v = 0.0;
                        for (int a = 0; a < nvira; ++a)
                            for (int b = 0; b < nvira; ++b)
                                v += Cb_virt.rows() ? Ca_virt(p, a) * Ca_virt(q, b) *
                                         result.t2_aa[detail::idx_t2(i, j, a, b, nocca, nvira)]
                                                   : 0.0;
                        part_dm2aa[idx_aa_part(i, p, q, j)] = 0.5 * (v - [&]() {
                            double x = 0.0;
                            for (int a = 0; a < nvira; ++a)
                                for (int b = 0; b < nvira; ++b)
                                    x += Ca_virt(p, b) * Ca_virt(q, a) *
                                         result.t2_aa[detail::idx_t2(i, j, a, b, nocca, nvira)];
                            return x;
                        }());
                    }

        for (int i = 0; i < noccb; ++i)
            for (int j = 0; j < noccb; ++j)
                for (int p = 0; p < nao; ++p)
                    for (int q = 0; q < nao; ++q)
                    {
                        double v = 0.0;
                        for (int a = 0; a < nvirb; ++a)
                            for (int b = 0; b < nvirb; ++b)
                                v += Cb_virt(p, a) * Cb_virt(q, b) *
                                     result.t2_bb[detail::idx_t2(i, j, a, b, noccb, nvirb)];
                        part_dm2bb[idx_bb_part(i, p, q, j)] = 0.5 * (v - [&]() {
                            double x = 0.0;
                            for (int a = 0; a < nvirb; ++a)
                                for (int b = 0; b < nvirb; ++b)
                                    x += Cb_virt(p, b) * Cb_virt(q, a) *
                                         result.t2_bb[detail::idx_t2(i, j, a, b, noccb, nvirb)];
                            return x;
                        }());
                    }

        for (int i = 0; i < nocca; ++i)
            for (int j = 0; j < noccb; ++j)
                for (int p = 0; p < nao; ++p)
                    for (int q = 0; q < nao; ++q)
                    {
                        double v = 0.0;
                        for (int a = 0; a < nvira; ++a)
                            for (int b = 0; b < nvirb; ++b)
                                v += Ca_virt(p, a) * Cb_virt(q, b) *
                                     result.t2_ab[detail::idx_t2_ab(i, j, a, b, nocca, noccb, nvira, nvirb)];
                        part_dm2ab[idx_ab_part(i, p, q, j)] = v;
                    }

        const Eigen::MatrixXd hf_dm1a = calculator._info._scf.alpha.density;
        const Eigen::MatrixXd hf_dm1b = calculator._info._scf.beta.density;
        const Eigen::MatrixXd hf_dm1 = hf_dm1a + hf_dm1b;
        std::vector<std::vector<int>> atom_aos = build_atom_ao_lists(calculator);
        std::vector<double> pair_dm2_ao(static_cast<std::size_t>(nao) * nao * nao * nao, 0.0);
        Eigen::MatrixXd Imata = Eigen::MatrixXd::Zero(nao, nao);
        Eigen::MatrixXd Imatb = Eigen::MatrixXd::Zero(nao, nao);
        std::vector<std::array<std::array<Eigen::MatrixXd, 3>, 2>> vhf1(calculator._molecule.natoms);
        for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
            for (int spin = 0; spin < 2; ++spin)
                for (int comp = 0; comp < 3; ++comp)
                    vhf1[atom][spin][comp] = Eigen::MatrixXd::Zero(nao, nao);

        Eigen::MatrixXd electronic = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        std::vector<double> eri_local;
        const std::vector<double> &eri = ensure_eri(
            calculator, shell_pairs, eri_local, "UMP2 Gradient :");
        const auto &bfs = calculator._shells._basis_functions;

        for (std::size_t atom = 0; atom < atom_aos.size(); ++atom)
            for (int p : atom_aos[atom])
                for (int q = 0; q < nao; ++q)
                    for (int r = 0; r < nao; ++r)
                        for (int s = 0; s < nao; ++s)
                        {
                            double dm2a = 0.0;
                            for (int i = 0; i < nocca; ++i)
                                for (int j = 0; j < nocca; ++j)
                                {
                                    dm2a += Ca_occ(p, i) * part_dm2aa[idx_aa_part(i, q, r, j)] * Ca_occ(s, j);
                                    dm2a += Ca_occ(q, i) * part_dm2aa[idx_aa_part(i, p, r, j)] * Ca_occ(s, j);
                                    dm2a += Ca_occ(p, i) * part_dm2aa[idx_aa_part(i, q, s, j)] * Ca_occ(r, j);
                                    dm2a += Ca_occ(q, i) * part_dm2aa[idx_aa_part(i, p, s, j)] * Ca_occ(r, j);
                                }
                            for (int i = 0; i < nocca; ++i)
                                for (int j = 0; j < noccb; ++j)
                                {
                                    dm2a += Ca_occ(p, i) * part_dm2ab[idx_ab_part(i, q, r, j)] * Cb_occ(s, j);
                                    dm2a += Ca_occ(q, i) * part_dm2ab[idx_ab_part(i, p, r, j)] * Cb_occ(s, j);
                                    dm2a += Ca_occ(p, i) * part_dm2ab[idx_ab_part(i, q, s, j)] * Cb_occ(r, j);
                                    dm2a += Ca_occ(q, i) * part_dm2ab[idx_ab_part(i, p, s, j)] * Cb_occ(r, j);
                                }

                            double dm2b = 0.0;
                            for (int i = 0; i < noccb; ++i)
                                for (int j = 0; j < noccb; ++j)
                                {
                                    dm2b += Cb_occ(p, i) * part_dm2bb[idx_bb_part(i, q, r, j)] * Cb_occ(s, j);
                                    dm2b += Cb_occ(q, i) * part_dm2bb[idx_bb_part(i, p, r, j)] * Cb_occ(s, j);
                                    dm2b += Cb_occ(p, i) * part_dm2bb[idx_bb_part(i, q, s, j)] * Cb_occ(r, j);
                                    dm2b += Cb_occ(q, i) * part_dm2bb[idx_bb_part(i, p, s, j)] * Cb_occ(r, j);
                                }
                            for (int i = 0; i < nocca; ++i)
                                for (int j = 0; j < noccb; ++j)
                                {
                                    dm2b += Ca_occ(r, i) * part_dm2ab[idx_ab_part(i, s, q, j)] * Cb_occ(p, j);
                                    dm2b += Ca_occ(r, i) * part_dm2ab[idx_ab_part(i, s, p, j)] * Cb_occ(q, j);
                                    dm2b += Ca_occ(s, i) * part_dm2ab[idx_ab_part(i, r, q, j)] * Cb_occ(p, j);
                                    dm2b += Ca_occ(s, i) * part_dm2ab[idx_ab_part(i, r, p, j)] * Cb_occ(q, j);
                                }

                            pair_dm2_ao[idx_dm2(p, q, r, s, nao)] += dm2a + dm2b;

                            const HartreeFock::ShellPair spAB(bfs[p], bfs[q]);
                            const HartreeFock::ShellPair spCD(bfs[r], bfs[s]);
                            const auto dI = HartreeFock::Gradient::compute_eri_deriv_dispatch(calculator, spAB, spCD);
                            for (int comp = 0; comp < 3; ++comp)
                                electronic(atom, comp) += dI[comp] * (dm2a + dm2b);

                            const double eri_pqrs = eri[idx_dm2(p, q, r, s, nao)];
                            for (int v = 0; v < nao; ++v)
                            {
                                double dm2av = 0.0;
                                for (int i = 0; i < nocca; ++i)
                                    for (int j = 0; j < nocca; ++j)
                                    {
                                        dm2av += Ca_occ(p, i) * part_dm2aa[idx_aa_part(i, v, r, j)] * Ca_occ(s, j);
                                        dm2av += Ca_occ(v, i) * part_dm2aa[idx_aa_part(i, p, r, j)] * Ca_occ(s, j);
                                        dm2av += Ca_occ(p, i) * part_dm2aa[idx_aa_part(i, v, s, j)] * Ca_occ(r, j);
                                        dm2av += Ca_occ(v, i) * part_dm2aa[idx_aa_part(i, p, s, j)] * Ca_occ(r, j);
                                    }
                                for (int i = 0; i < nocca; ++i)
                                    for (int j = 0; j < noccb; ++j)
                                    {
                                        dm2av += Ca_occ(p, i) * part_dm2ab[idx_ab_part(i, v, r, j)] * Cb_occ(s, j);
                                        dm2av += Ca_occ(v, i) * part_dm2ab[idx_ab_part(i, p, r, j)] * Cb_occ(s, j);
                                        dm2av += Ca_occ(p, i) * part_dm2ab[idx_ab_part(i, v, s, j)] * Cb_occ(r, j);
                                        dm2av += Ca_occ(v, i) * part_dm2ab[idx_ab_part(i, p, s, j)] * Cb_occ(r, j);
                                    }

                                double dm2bv = 0.0;
                                for (int i = 0; i < noccb; ++i)
                                    for (int j = 0; j < noccb; ++j)
                                    {
                                        dm2bv += Cb_occ(p, i) * part_dm2bb[idx_bb_part(i, v, r, j)] * Cb_occ(s, j);
                                        dm2bv += Cb_occ(v, i) * part_dm2bb[idx_bb_part(i, p, r, j)] * Cb_occ(s, j);
                                        dm2bv += Cb_occ(p, i) * part_dm2bb[idx_bb_part(i, v, s, j)] * Cb_occ(r, j);
                                        dm2bv += Cb_occ(v, i) * part_dm2bb[idx_bb_part(i, p, s, j)] * Cb_occ(r, j);
                                    }
                                for (int i = 0; i < nocca; ++i)
                                    for (int j = 0; j < noccb; ++j)
                                    {
                                        dm2bv += Ca_occ(r, i) * part_dm2ab[idx_ab_part(i, s, v, j)] * Cb_occ(p, j);
                                        dm2bv += Ca_occ(r, i) * part_dm2ab[idx_ab_part(i, s, p, j)] * Cb_occ(v, j);
                                        dm2bv += Ca_occ(s, i) * part_dm2ab[idx_ab_part(i, r, v, j)] * Cb_occ(p, j);
                                        dm2bv += Ca_occ(s, i) * part_dm2ab[idx_ab_part(i, r, p, j)] * Cb_occ(v, j);
                                    }
                                Imata(q, v) += 0.5 * eri_pqrs * dm2av;
                                Imatb(q, v) += 0.5 * eri_pqrs * dm2bv;
                            }

                            for (int comp = 0; comp < 3; ++comp)
                            {
                                vhf1[atom][0][comp](r, s) += dI[comp] * hf_dm1(p, q);
                                vhf1[atom][1][comp](r, s) += dI[comp] * hf_dm1(p, q);
                                vhf1[atom][0][comp](r, q) -= dI[comp] * hf_dm1a(p, s);
                                vhf1[atom][1][comp](r, q) -= dI[comp] * hf_dm1b(p, s);
                                vhf1[atom][0][comp](p, q) += dI[comp] * hf_dm1(r, s);
                                vhf1[atom][1][comp](p, q) += dI[comp] * hf_dm1(r, s);
                                vhf1[atom][0][comp](p, s) -= dI[comp] * hf_dm1a(q, r);
                                vhf1[atom][1][comp](p, s) -= dI[comp] * hf_dm1b(q, r);
                            }
                        }

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
        const auto [va_ao, vb_ao] = build_veff_from_spin_densities(
            calculator, shell_pairs, dm1a_ao_seed, dm1b_ao_seed);
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

        const auto [vs1a, vs1b] = build_veff_from_spin_densities(
            calculator, shell_pairs, Pa_corr + Pa_corr.transpose(), Pb_corr + Pb_corr.transpose());
        const Eigen::MatrixXd p1a = Ca_occ * Ca_occ.transpose();
        const Eigen::MatrixXd p1b = Cb_occ * Cb_occ.transpose();
        const Eigen::MatrixXd vhf_s1occ =
            0.5 * (p1a * vs1a * p1a + p1b * vs1b * p1b);

        const Eigen::MatrixXd dm1pa = hf_dm1a + 2.0 * Pa_corr;
        const Eigen::MatrixXd dm1pb = hf_dm1b + 2.0 * Pb_corr;
        const Eigen::MatrixXd dm1tot = hf_dm1a + hf_dm1b + Pa_corr + Pb_corr;
        const Eigen::MatrixXd one_e = one_electron_gradient_from_density(calculator, shell_pairs, dm1tot);
        electronic += one_e;

        const Eigen::MatrixXd zeta_total = zeta;
        Eigen::MatrixXd overlap_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        for (std::size_t atom = 0; atom < atom_aos.size(); ++atom)
            for (int p : atom_aos[atom])
                for (int nu = 0; nu < nao; ++nu)
                {
                    const HartreeFock::ShellPair sp(bfs[p], bfs[nu]);
                    const auto dST = HartreeFock::ObaraSaika::_compute_1e_deriv_A(sp);
                    for (int q = 0; q < 3; ++q)
                    {
                        overlap_terms(atom, q) += dST[q] * im1_ao(p, nu);
                        overlap_terms(atom, q) += dST[q] * im1_ao(nu, p);
                        overlap_terms(atom, q) -= dST[q] * zeta_total(p, nu);
                        overlap_terms(atom, q) -= dST[q] * zeta_total(nu, p);
                        overlap_terms(atom, q) -= 2.0 * dST[q] * vhf_s1occ(p, nu);
                    }
                }
        electronic += overlap_terms;

        Eigen::MatrixXd vhf1_terms = Eigen::MatrixXd::Zero(calculator._molecule.natoms, 3);
        for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
            for (int q = 0; q < 3; ++q)
            {
                vhf1_terms(atom, q) = (vhf1[atom][0][q].cwiseProduct(dm1pa)).sum() +
                                      (vhf1[atom][1][q].cwiseProduct(dm1pb)).sum();
                electronic(atom, q) += vhf1_terms(atom, q);
            }

        UMP2GradientIntermediates out;
        out.electronic_gradient = std::move(electronic);
        out.P_alpha_corr_ao = Pa_corr;
        out.P_beta_corr_ao = Pb_corr;
        out.P_alpha_ao = hf_dm1a + Pa_corr;
        out.P_beta_ao = hf_dm1b + Pb_corr;
        out.P_total_ao = out.P_alpha_ao + out.P_beta_ao;
        out.W_ao = zeta_total;
        out.Gamma_pair_ao = std::move(pair_dm2_ao);
        return out;
    }
} // namespace HartreeFock::Correlation
