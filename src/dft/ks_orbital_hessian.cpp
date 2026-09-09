#include "ks_orbital_hessian.h"

#include "analytic_hessian.h"
#include "response_packing.h"
#include "integrals/base.h"

namespace DFT::Driver
{

    std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
    build_ks_orbital_hessian_op(const KsOrbitalHessianInputs &in)
    {
        const int n_occ = static_cast<int>(in.C_occ.cols());
        const int n_virt = static_cast<int>(in.C_virt.cols());
        const std::size_t nbasis = static_cast<std::size_t>(in.density.rows());

        const Eigen::VectorXd diag_term =
            orbital_energy_difference_diagonal(in.eps, n_occ);

        return [&in, n_occ, n_virt, nbasis, diag_term](
                   const Eigen::VectorXd &x) -> Eigen::VectorXd
        {
            // dP = C_virt * x_mat * C_occ^T + h.c. -- the same half-of-[R,P0]
            // trial density the SOSCF branch and F3's probes use. The s
            // (kernel_scale) factor at the return carries the other half.
            Eigen::MatrixXd x_mat(n_virt, n_occ);
            for (int a = 0; a < n_virt; ++a)
                for (int i = 0; i < n_occ; ++i)
                    x_mat(a, i) = x(a * n_occ + i);
            const Eigen::MatrixXd d1 = in.C_virt * x_mat * in.C_occ.transpose();
            const Eigen::MatrixXd dP = d1 + d1.transpose();

            const Eigen::MatrixXd dJ = _compute_2e_j_direct(
                *in.shell_pairs, dP, nbasis, in.engine,
                HartreeFock::ERIKernel::Coulomb, 0.0, in.tol_eri, in.sym_ops);
            const Eigen::VectorXd J_packed =
                pack_hessian_vector_product_cphf_order(dJ, in.C_occ, in.C_virt);

            const auto dV_xc = compute_analytic_xc_hessian_vector_product(
                *in.molecular_grid, *in.ao_grid, in.density, dP,
                *in.x_functional, *in.c_functional);
            if (!dV_xc)
                return Eigen::VectorXd::Zero(x.size());
            const Eigen::VectorXd xc_packed =
                pack_hessian_vector_product_cphf_order(*dV_xc, in.C_occ, in.C_virt);

            // K response (SOSCF_DFT.md invariant 3): a hybrid's KS Fock carries
            // -0.5*(c_fr*K_Coulomb + c_sr*K_SR); K is linear in the density
            // exactly like J, so the Hessian gains delta of it on dP with the
            // same coeff/sign/kernel split.
            Eigen::VectorXd K_packed = Eigen::VectorXd::Zero(x.size());
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
                K_packed = pack_hessian_vector_product_cphf_order(
                    -0.5 * dK, in.C_occ, in.C_virt);
            }

            return diag_term.cwiseProduct(x) +
                   in.kernel_scale * (J_packed + xc_packed + K_packed);
        };
    }

} // namespace DFT::Driver
