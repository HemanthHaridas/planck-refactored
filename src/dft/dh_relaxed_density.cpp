#include "dh_relaxed_density.h"

#include <cstdlib>

#include "ks_orbital_hessian.h"
#include "post_hf/rhf_response.h"

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
