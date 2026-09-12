#include "ks_orbital_hessian.h"
#include "analytic_hessian.h"
#include "response_packing.h"
#include "integrals/base.h"

namespace DFT::Driver
{
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
    build_ks_orbital_hessian_op(const KsOrbitalHessianInputs &in)
    {
        const int nocc = static_cast<int>(in.C_occ.cols());
        const int nvirt = static_cast<int>(in.C_virt.cols());
        const std::size_t nb = static_cast<std::size_t>(in.density.rows());
        const Eigen::VectorXd diag = orbital_energy_difference_diagonal(in.eps, nocc);
        return [&in, nocc, nvirt, nb, diag](const Eigen::VectorXd &x) -> Eigen::VectorXd
        {
            Eigen::MatrixXd xm(nvirt, nocc);
            for (int a = 0; a < nvirt; ++a)
                for (int i = 0; i < nocc; ++i)
                    xm(a, i) = x(a * nocc + i);
            const Eigen::MatrixXd d1 = in.C_virt * xm * in.C_occ.transpose();
            const Eigen::MatrixXd dp = d1 + d1.transpose();
            const Eigen::MatrixXd dj = _compute_2e_j_direct(
                *in.shell_pairs, dp, nb, in.engine, HartreeFock::ERIKernel::Coulomb,
                0.0, in.tol_eri, in.sym_ops);
            const auto dvxc = compute_analytic_xc_hessian_vector_product(
                *in.molecular_grid, *in.ao_grid, in.density, dp,
                *in.x_functional, *in.c_functional);
            if (!dvxc)
                return Eigen::VectorXd::Zero(x.size());
            Eigen::MatrixXd dk = Eigen::MatrixXd::Zero(dj.rows(), dj.cols());
            if (in.full_range_exchange_coefficient != 0.0)
                dk.noalias() += in.full_range_exchange_coefficient * _compute_2e_k_direct(
                    *in.shell_pairs, dp, nb, in.engine, HartreeFock::ERIKernel::Coulomb,
                    0.0, in.tol_eri, in.sym_ops);
            if (in.short_range_exchange_coefficient != 0.0)
                dk.noalias() += in.short_range_exchange_coefficient * _compute_2e_k_direct(
                    *in.shell_pairs, dp, nb, in.engine, HartreeFock::ERIKernel::ShortRange,
                    in.range_separation_omega, in.tol_eri, in.sym_ops);
            return diag.cwiseProduct(x) + in.kernel_scale * (
                pack_hessian_vector_product_cphf_order(dj, in.C_occ, in.C_virt) +
                pack_hessian_vector_product_cphf_order(*dvxc, in.C_occ, in.C_virt) +
                pack_hessian_vector_product_cphf_order(-0.5 * dk, in.C_occ, in.C_virt));
        };
    }
}
