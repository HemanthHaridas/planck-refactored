#include "response_packing.h"

namespace DFT::Driver
{
    Eigen::VectorXd pack_hessian_vector_product_cphf_order(
        const Eigen::Ref<const Eigen::MatrixXd> &delta_v_xc_ao,
        const Eigen::Ref<const Eigen::MatrixXd> &c_occ,
        const Eigen::Ref<const Eigen::MatrixXd> &c_virt)
    {
        const int n_occ = static_cast<int>(c_occ.cols());
        const int n_virt = static_cast<int>(c_virt.cols());
        const Eigen::MatrixXd hx_mo = c_occ.transpose() * delta_v_xc_ao * c_virt; // (n_occ x n_virt)
        Eigen::VectorXd packed(n_occ * n_virt);
        for (int a = 0; a < n_virt; ++a)
            for (int i = 0; i < n_occ; ++i)
                packed(a * n_occ + i) = hx_mo(i, a);
        return packed;
    }
} // namespace DFT::Driver
