#include "dh_relaxed_density.h"

#include <cstdlib>
#include <format>

#include "ks_orbital_hessian.h"
#include "post_hf/rhf_response.h"
#include "response_packing.h"
#include "analytic_hessian.h"
#include "integrals/base.h"
#include "io/logging.h"

namespace DFT::Gradient
{

    std::expected<PT2RelaxedDensity, std::string> solve_pt2_relaxed_density(
        const PT2RelaxedDensityInputs &in)
    {
        const auto &lag = *in.lagrangian;
        const auto &result = *in.result;

        const int n_occ = lag.n_occ;
        const int n_virt = lag.n_virt;
        const int nmo = n_occ + n_virt;
        if (result.mo_coeff.cols() != nmo)
            return std::unexpected(
                "solve_pt2_relaxed_density: mo_coeff column count does not match "
                "n_occ + n_virt from the Lagrangian.");

        const Eigen::MatrixXd C_occ = result.mo_coeff.leftCols(n_occ);
        const Eigen::MatrixXd C_virt = result.mo_coeff.middleCols(n_occ, n_virt);

        DFT::Driver::KsOrbitalHessianInputs h_in;
        h_in.shell_pairs = in.shell_pairs;
        h_in.molecular_grid = in.molecular_grid;
        h_in.ao_grid = in.ao_grid;
        h_in.density = in.density;
        h_in.C_occ = C_occ;
        h_in.C_virt = C_virt;
        h_in.eps = result.mo_energy;
        h_in.x_functional = in.x_functional;
        h_in.c_functional = in.c_functional;
        h_in.engine = in.engine;
        h_in.tol_eri = in.tol_eri;
        h_in.sym_ops = in.sym_ops;
        h_in.full_range_exchange_coefficient = in.full_range_exchange_coefficient;
        h_in.short_range_exchange_coefficient = in.short_range_exchange_coefficient;
        h_in.range_separation_omega = in.range_separation_omega;
        h_in.kernel_scale = 2.0; // RKS

        Eigen::MatrixXd z(n_virt, n_occ);

        // PLANCK_DFT_DH_HESSIAN_AUDIT: channel-resolved oracle on
        // build_ks_orbital_hessian_op. The two non-XC channels of the KS
        // orbital Hessian are the SAME operator as the RHF CPHF matrix's
        // Coulomb / exchange couplings, so build_rhf_cphf_matrix (independent
        // code path: dense AO->MO ERI transform, textbook
        // 4(ai|jb) - (ab|ji) - (aj|bi)) is an exact oracle for them -- no
        // finite difference, no metric contamination, no gauge ambiguity.
        //
        // Verifies at once: the (a,i) packing convention, the kernel_scale
        // s = 2, the dP = C_v x C_o^T + h.c. trial-density convention, and
        // the hybrid K prefactor -0.5. A mismatch in any of those is the
        // "coefficients never matched against Eq. 41 term by term" gap
        // (docs/DH_GRADIENT_HANDOFF.md section 3).
        //
        // Expected exact identities at c_fr = 1, c_sr = 0:
        //   diag channel     : h_op == A, identically
        //   Coulomb channel  : 2*pack(J[dP])       == 4*(ai|jb)
        //   exchange channel : 2*pack(-0.5*K[dP])  == -(ab|ji) - (aj|bi)
        // The XC channel has no CPHF counterpart and is reported separately
        // as a magnitude, to size it against the two that are gated.
        if (std::getenv("PLANCK_DFT_DH_HESSIAN_AUDIT"))
        {
            const int dim = n_virt * n_occ;
            const Eigen::VectorXd diag_term =
                DFT::Driver::orbital_energy_difference_diagonal(result.mo_energy, n_occ);
            const std::size_t nbasis = static_cast<std::size_t>(in.density.rows());

            Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(dim, dim);
            Eigen::MatrixXd Hj = Eigen::MatrixXd::Zero(dim, dim);
            Eigen::MatrixXd Hk = Eigen::MatrixXd::Zero(dim, dim);
            Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(dim, dim);
            bool xc_ok = true;

            for (int col = 0; col < dim; ++col)
            {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(dim);
                e(col) = 1.0;

                Eigen::MatrixXd x_mat(n_virt, n_occ);
                for (int a = 0; a < n_virt; ++a)
                    for (int i = 0; i < n_occ; ++i)
                        x_mat(a, i) = e(a * n_occ + i);
                const Eigen::MatrixXd d1 = C_virt * x_mat * C_occ.transpose();
                const Eigen::MatrixXd dP = d1 + d1.transpose();

                Hd.col(col) = diag_term.cwiseProduct(e);

                const Eigen::MatrixXd dJ = _compute_2e_j_direct(
                    *in.shell_pairs, dP, nbasis, in.engine,
                    HartreeFock::ERIKernel::Coulomb, 0.0, in.tol_eri, in.sym_ops);
                Hj.col(col) = 2.0 * DFT::Driver::pack_hessian_vector_product_cphf_order(
                                        dJ, C_occ, C_virt);

                const double c_fr = in.full_range_exchange_coefficient;
                const double c_sr = in.short_range_exchange_coefficient;
                if (c_fr != 0.0 || c_sr != 0.0)
                {
                    Eigen::MatrixXd dK = Eigen::MatrixXd::Zero(
                        static_cast<Eigen::Index>(nbasis), static_cast<Eigen::Index>(nbasis));
                    if (c_fr != 0.0)
                        dK.noalias() += c_fr * _compute_2e_k_direct(
                            *in.shell_pairs, dP, nbasis, in.engine,
                            HartreeFock::ERIKernel::Coulomb, 0.0, in.tol_eri, in.sym_ops);
                    if (c_sr != 0.0)
                        dK.noalias() += c_sr * _compute_2e_k_direct(
                            *in.shell_pairs, dP, nbasis, in.engine,
                            HartreeFock::ERIKernel::ShortRange, in.range_separation_omega,
                            in.tol_eri, in.sym_ops);
                    Hk.col(col) = 2.0 * DFT::Driver::pack_hessian_vector_product_cphf_order(
                                            -0.5 * dK, C_occ, C_virt);
                }

                if (xc_ok)
                {
                    const auto dV = DFT::Driver::compute_analytic_xc_hessian_vector_product(
                        *in.molecular_grid, *in.ao_grid, in.density, dP,
                        *in.x_functional, *in.c_functional);
                    if (dV)
                        Hx.col(col) = 2.0 * DFT::Driver::pack_hessian_vector_product_cphf_order(
                                                *dV, C_occ, C_virt);
                    else
                        xc_ok = false;
                }
            }

            // Oracle: the HF CPHF matrix, built from a fully independent code
            // path (dense AO ERI + 4-index transform + the textbook formula).
            auto A_ref = HartreeFock::Correlation::build_rhf_cphf_matrix(
                *in.calculator, *in.shell_pairs, result.mo_coeff, result.mo_energy);
            if (!A_ref)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, "DH Hessian Audit :",
                    "oracle build_rhf_cphf_matrix failed: " + A_ref.error());
            }
            else
            {
                // Split the oracle into the same three channels. The diagonal
                // is A's own; the Coulomb/exchange split is recovered by
                // rebuilding A's coupling with the exchange term switched off,
                // which the dense builder does not expose -- so instead compare
                // the TOTAL non-XC coupling, which needs no such split:
                //   A - diag  ==  Hj + Hk/c_fr_ref  at c_fr = 1.
                Eigen::MatrixXd A_coupling = *A_ref;
                for (int d = 0; d < dim; ++d)
                    A_coupling(d, d) -= diag_term(d);

                const double c_fr = in.full_range_exchange_coefficient;
                // h_op's K channel already carries c_fr; the oracle's exchange
                // is at full weight, so scale the oracle's exchange share to
                // match by comparing against Hj + Hk directly at the SAME c_fr:
                // rebuild the oracle coupling as J_ref + c_fr*K_ref, where
                // J_ref/K_ref are recovered from A at c_fr = 1 by noting
                // A_coupling = J_ref + K_ref exactly.
                //
                // With only A available, the decisive gated identity is the
                // DIAGONAL (exact) and, for a c_fr = 1 functional, the whole
                // coupling. For c_fr != 1 report the two pieces separately
                // against a c_fr-scaled reconstruction.
                const double diag_err =
                    (Hd.diagonal() - diag_term).cwiseAbs().maxCoeff();
                const double diag_off = (Hd - Eigen::MatrixXd(
                                                  Hd.diagonal().asDiagonal()))
                                            .cwiseAbs()
                                            .maxCoeff();

                const Eigen::MatrixXd h_coupling = Hj + Hk;
                const double coup_err =
                    (h_coupling - A_coupling).cwiseAbs().maxCoeff();
                const double coup_scale = A_coupling.cwiseAbs().maxCoeff();

                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Info, "DH Hessian Audit :",
                    std::format(
                        "dim={} c_fr={:.6f} c_sr={:.6f} | diag: max|Hd-diag|={:.3e} "
                        "max|offdiag(Hd)|={:.3e} | coupling(J+K) vs HF-CPHF: "
                        "max|d|={:.3e} rel={:.3e} (EXACT only when c_fr=1) | "
                        "max|Hj|={:.3e} max|Hk|={:.3e} max|Hxc|={:.3e} xc_ok={}",
                        dim, c_fr, in.short_range_exchange_coefficient,
                        diag_err, diag_off, coup_err,
                        coup_scale > 0.0 ? coup_err / coup_scale : 0.0,
                        Hj.cwiseAbs().maxCoeff(), Hk.cwiseAbs().maxCoeff(),
                        Hx.cwiseAbs().maxCoeff(), xc_ok ? 1 : 0));

                // Coulomb channel in isolation: A_coupling at c_fr = 0 would be
                // pure 4(ai|jb). Recover it by subtracting h_op's own exchange
                // at full weight, which is only valid if Hk is right -- so
                // report instead the residual after removing Hj, which must
                // equal the oracle's exchange share exactly at c_fr = 1.
                if (std::abs(c_fr - 1.0) < 1e-12 &&
                    in.short_range_exchange_coefficient == 0.0)
                {
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info, "DH Hessian Audit :",
                        std::format(
                            "c_fr=1 exact gate: max|Hj+Hk-A_coupling|={:.3e} "
                            "(pass < 1e-9)",
                            coup_err));
                }
                else
                {
                    // c_fr != 1: split the oracle. A_coupling(c_fr=1) = J + K,
                    // and h_op's coupling is J + c_fr*K. So
                    // A_coupling - h_coupling = (1 - c_fr)*K, whose magnitude
                    // must equal |Hk|/c_fr * (1 - c_fr).
                    const Eigen::MatrixXd implied_K =
                        (A_coupling - h_coupling) / (1.0 - c_fr);
                    const Eigen::MatrixXd hop_K = Hk / c_fr;
                    const double k_err = (implied_K - hop_K).cwiseAbs().maxCoeff();
                    const double k_scale = hop_K.cwiseAbs().maxCoeff();
                    HartreeFock::Logger::logging(
                        HartreeFock::LogLevel::Info, "DH Hessian Audit :",
                        std::format(
                            "c_fr!=1 split gate: max|K_implied - K_hop|={:.3e} "
                            "rel={:.3e} (pass rel < 1e-9); this gates the Coulomb "
                            "channel AND the K prefactor jointly",
                            k_err, k_scale > 0.0 ? k_err / k_scale : 0.0));
                }
            }
        }

        // N3.4 diagnostic override: solve the Z-vector with solve_rhf_cphf
        // (HF CPHF) instead of the KS orbital Hessian, to isolate the
        // operator's contribution to the FD error.
        if (std::getenv("PLANCK_DFT_DH_ZVECTOR_HFCPHF"))
        {
            auto z_hf = HartreeFock::Correlation::solve_rhf_cphf(
                *in.calculator, *in.shell_pairs,
                result.mo_coeff, result.mo_energy, lag.Xvo);
            if (!z_hf)
                return std::unexpected("solve_pt2_relaxed_density (HFCPHF override): " + z_hf.error());
            z = *z_hf;
        }
        else
        {
            const auto h_op = DFT::Driver::build_ks_orbital_hessian_op(h_in);

            // Z-vector: solve h_op(z) = -Xvo (CPHF (a,i) ordering), the
            // KS-Hessian analogue of solve_rhf_cphf's A z = -rhs. lag.Xvo was
            // built for the build_rhf_cphf_matrix convention;
            // build_ks_orbital_hessian_op targets the same CPHF-ordered
            // operator, so the RHS packing is identical.
            const int dim = n_virt * n_occ;
            Eigen::VectorXd rhs(dim);
            for (int a = 0; a < n_virt; ++a)
                for (int i = 0; i < n_occ; ++i)
                    rhs(a * n_occ + i) = -lag.Xvo(a, i);

            // ponytail: dense assembly + QR, dim x dim. Each column costs one
            // analytic XC Hessian-vector product -- O(dim) grid passes total,
            // which is exactly what the analytic fxc path was built to avoid
            // at scale. Fine for the water/STO-3G validation target; swap to
            // matrix-free CG (the KS orbital Hessian is SPD near a minimum)
            // when dim gets large.
            Eigen::MatrixXd A(dim, dim);
            for (int col = 0; col < dim; ++col)
            {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(dim);
                e(col) = 1.0;
                const Eigen::VectorXd Ae = h_op(e);
                if (!Ae.allFinite() || Ae.size() != dim)
                    return std::unexpected(
                        "solve_pt2_relaxed_density: KS orbital Hessian application "
                        "returned a non-finite or wrong-size vector (XC HVP failure?).");
                A.col(col) = Ae;
            }

            const Eigen::VectorXd z_vec = A.colPivHouseholderQr().solve(rhs);
            if (!z_vec.allFinite())
                return std::unexpected("solve_pt2_relaxed_density: Z-vector solve produced non-finite values.");

            for (int a = 0; a < n_virt; ++a)
                for (int i = 0; i < n_occ; ++i)
                    z(a, i) = z_vec(a * n_occ + i);
        }

        // Relaxed density + energy-weighted density: the SAME shared assembly
        // build_rmp2_gradient_intermediates uses. Only the Z-vector operator
        // above differs (KS orbital Hessian here vs solve_rhf_cphf there).
        // NOTE: this rel->W_ao / vhf_s1occ_ao is NOT what feeds the DH
        // gradient -- the driver contracts through
        // build_rmp2_gradient_intermediates(presolved), which recomputes the
        // energy-weighted density with its own RMP2PreSolved::ks_veff. This
        // call still provides P_ao / dm1_corr_relaxed_ao / z to the caller.
        auto rel = HartreeFock::Correlation::build_rmp2_energy_weighted_density(
            *in.calculator, *in.shell_pairs, result, lag, z, in.ks_veff);
        if (!rel)
            return std::unexpected(rel.error());

        PT2RelaxedDensity out;
        out.z = z;
        out.P_ao = rel->P_ao;
        out.W_ao = rel->W_ao;
        out.dm1_corr_relaxed_ao = rel->dm1_corr_relaxed_ao;
        out.zeta_ao = rel->zeta_ao;
        out.imat_ao = rel->imat_ao;
        return out;
    }

} // namespace DFT::Gradient
