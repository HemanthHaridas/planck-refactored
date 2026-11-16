#include "ks_orbital_hessian.h"
#include <memory>
#include "analytic_hessian.h"
#include "response_packing.h"
#include "integrals/base.h"
#include "basis/spherical.h"
#include "post_hf/ri/ri_eri.h"

namespace DFT::Driver
{
    namespace
    {
        bool spherical(const HartreeFock::Basis *basis) { return basis && basis->_spherical; }

        using HartreeFock::BasisFunctions::mos_to_cartesian;

        // Fitted J/K response for one symmetric working-basis trial density
        // (docs/RI_JK_ROUTINE_SCOPE.md step 14). The unpacked factors are built
        // once per operator; the RI builders take and return working-basis
        // matrices, so the caller packs them with the working MOs.
        struct RiResponse
        {
            const HartreeFock::Calculator *calc = nullptr;
            std::shared_ptr<const std::vector<Eigen::MatrixXd>> unpacked;

            static RiResponse make(const HartreeFock::Calculator *calc, double exchange)
            {
                RiResponse r{calc, nullptr};
                if (calc && exchange != 0.0)
                    r.unpacked = std::make_shared<const std::vector<Eigen::MatrixXd>>(
                        HartreeFock::Correlation::RI::build_ri_3index_unpacked(*calc));
                return r;
            }
            Eigen::MatrixXd J(const Eigen::MatrixXd &D) const
            {
                return HartreeFock::Correlation::RI::build_ri_j(*calc, D);
            }
            Eigen::MatrixXd K(const Eigen::MatrixXd &D) const
            {
                return HartreeFock::Correlation::RI::build_ri_k(*calc, D, unpacked.get());
            }
        };

        std::expected<void, std::string> check_ri(const HartreeFock::Calculator *ri, double c_sr)
        {
            if (ri && c_sr != 0.0)
                return std::unexpected("orbital Hessian: RI-JK has no short-range (attenuated) exchange");
            if (ri && ri->_ri_pair_factors.size() == 0)
                return std::unexpected("orbital Hessian: RI-JK cache is not ready");
            return {};
        }

        std::expected<void, std::string> check_mo_rows(const HartreeFock::Basis &basis,
                                                       std::initializer_list<const Eigen::MatrixXd *> mos)
        {
            for (const Eigen::MatrixXd *C : mos)
                if (C->rows() != basis._cart_to_sph.rows())
                    return std::unexpected("orbital Hessian: MO coefficients are not in the working basis");
            return {};
        }
    }

    std::expected<std::function<Eigen::VectorXd(const Eigen::VectorXd &)>, std::string>
    build_ks_orbital_hessian_op(const KsOrbitalHessianInputs &in_working)
    {
        KsOrbitalHessianInputs in = in_working;
        if (auto ok = check_ri(in.ri, in.short_range_exchange_coefficient); !ok)
            return std::unexpected(ok.error());
        // Working-basis MOs for the RI response, before the Cartesian lift.
        const Eigen::MatrixXd Co_w = in.C_occ, Cv_w = in.C_virt;
        const RiResponse ri = RiResponse::make(in.ri, in.full_range_exchange_coefficient);
        if (spherical(in.basis))
        {
            if (auto ok = check_mo_rows(*in.basis, {&in.C_occ, &in.C_virt}); !ok)
                return std::unexpected(ok.error());
            auto density = HartreeFock::BasisFunctions::to_cartesian(*in.basis, in.density);
            if (!density)
                return std::unexpected(density.error());
            in.density = std::move(*density);
            in.C_occ = mos_to_cartesian(*in.basis, in.C_occ);
            in.C_virt = mos_to_cartesian(*in.basis, in.C_virt);
        }
        const int nocc = static_cast<int>(in.C_occ.cols());
        const int nvirt = static_cast<int>(in.C_virt.cols());
        const std::size_t nb = static_cast<std::size_t>(in.density.rows());
        const Eigen::VectorXd diag = orbital_energy_difference_diagonal(in.eps, nocc);
        return std::function<Eigen::VectorXd(const Eigen::VectorXd &)>(
            [in = std::move(in), nocc, nvirt, nb, diag, Co_w, Cv_w, ri](const Eigen::VectorXd &x) -> Eigen::VectorXd
        {
            Eigen::MatrixXd xm(nvirt, nocc);
            for (int a = 0; a < nvirt; ++a)
                for (int i = 0; i < nocc; ++i)
                    xm(a, i) = x(a * nocc + i);
            const Eigen::MatrixXd d1 = in.C_virt * xm * in.C_occ.transpose();
            const Eigen::MatrixXd dp = d1 + d1.transpose();
            const auto dvxc = compute_analytic_xc_hessian_vector_product(
                *in.molecular_grid, *in.ao_grid, in.density, dp,
                *in.x_functional, *in.c_functional);
            if (!dvxc)
                return Eigen::VectorXd::Zero(x.size());
            if (in.ri)
            {
                const Eigen::MatrixXd d1w = Cv_w * xm * Co_w.transpose();
                const Eigen::MatrixXd dpw = d1w + d1w.transpose();
                Eigen::MatrixXd djk = ri.J(dpw);
                if (in.full_range_exchange_coefficient != 0.0)
                    djk.noalias() -= 0.5 * in.full_range_exchange_coefficient * ri.K(dpw);
                return diag.cwiseProduct(x) + in.kernel_scale * (
                    pack_hessian_vector_product_cphf_order(djk, Co_w, Cv_w) +
                    pack_hessian_vector_product_cphf_order(*dvxc, in.C_occ, in.C_virt));
            }
            const Eigen::MatrixXd dj = _compute_2e_j_direct(
                *in.shell_pairs, dp, nb, in.engine, HartreeFock::ERIKernel::Coulomb,
                0.0, in.tol_eri, in.sym_ops);
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
        });
    }

    std::expected<std::function<Eigen::VectorXd(const Eigen::VectorXd &)>, std::string>
    build_uks_orbital_hessian_op(const UksOrbitalHessianInputs &in_working)
    {
        UksOrbitalHessianInputs in = in_working;
        if (auto ok = check_ri(in.ri, in.short_range_exchange_coefficient); !ok)
            return std::unexpected(ok.error());
        const Eigen::MatrixXd Cao_w = in.Ca_occ, Cav_w = in.Ca_virt, Cbo_w = in.Cb_occ, Cbv_w = in.Cb_virt;
        const RiResponse ri = RiResponse::make(in.ri, in.full_range_exchange_coefficient);
        if (spherical(in.basis))
        {
            if (auto ok = check_mo_rows(*in.basis, {&in.Ca_occ, &in.Ca_virt, &in.Cb_occ, &in.Cb_virt}); !ok)
                return std::unexpected(ok.error());
            auto pa = HartreeFock::BasisFunctions::to_cartesian(*in.basis, in.alpha_density);
            auto pb = HartreeFock::BasisFunctions::to_cartesian(*in.basis, in.beta_density);
            if (!pa || !pb)
                return std::unexpected(pa ? pb.error() : pa.error());
            in.alpha_density = std::move(*pa);
            in.beta_density = std::move(*pb);
            for (Eigen::MatrixXd *C : {&in.Ca_occ, &in.Ca_virt, &in.Cb_occ, &in.Cb_virt})
                *C = mos_to_cartesian(*in.basis, *C);
        }
        auto kernel = prepare_uks_xc_kernel(*in.molecular_grid, *in.ao_grid, in.alpha_density, in.beta_density,
                                            *in.x_functional, *in.c_functional);
        if (!kernel)
            return std::unexpected(kernel.error());
        const int na = static_cast<int>(in.Ca_occ.cols()), nva = static_cast<int>(in.Ca_virt.cols());
        const int nb_occ = static_cast<int>(in.Cb_occ.cols()), nvb = static_cast<int>(in.Cb_virt.cols());
        const int nova = nva * na, novb = nvb * nb_occ;
        const std::size_t nb = static_cast<std::size_t>(in.alpha_density.rows());
        const Eigen::VectorXd diag_a = orbital_energy_difference_diagonal(in.eps_a, na);
        const Eigen::VectorXd diag_b = orbital_energy_difference_diagonal(in.eps_b, nb_occ);
        return std::function<Eigen::VectorXd(const Eigen::VectorXd &)>(
            [in = std::move(in), kernel = std::move(*kernel), na, nva, nb_occ, nvb, nova, novb, nb, diag_a,
             diag_b, Cao_w, Cav_w, Cbo_w, Cbv_w, ri](const Eigen::VectorXd &x) -> Eigen::VectorXd
        {
            Eigen::MatrixXd xa(nva, na), xb(nvb, nb_occ);
            for (int a = 0; a < nva; ++a)
                for (int i = 0; i < na; ++i)
                    xa(a, i) = x(a * na + i);
            for (int a = 0; a < nvb; ++a)
                for (int i = 0; i < nb_occ; ++i)
                    xb(a, i) = x(nova + a * nb_occ + i);
            const Eigen::MatrixXd d1a = in.Ca_virt * xa * in.Ca_occ.transpose();
            const Eigen::MatrixXd dPa = d1a + d1a.transpose();
            const Eigen::MatrixXd d1b = in.Cb_virt * xb * in.Cb_occ.transpose();
            const Eigen::MatrixXd dPb = d1b + d1b.transpose();

            if (in.ri)
            {
                const auto dV = kernel.apply(dPa, dPb);
                if (!dV)
                    return Eigen::VectorXd::Zero(x.size());
                const Eigen::MatrixXd ea = Cav_w * xa * Cao_w.transpose(), eb = Cbv_w * xb * Cbo_w.transpose();
                const Eigen::MatrixXd dPa_w = ea + ea.transpose(), dPb_w = eb + eb.transpose();
                Eigen::MatrixXd Ga = ri.J(dPa_w + dPb_w);
                Eigen::MatrixXd Gb = Ga;
                const double c = in.full_range_exchange_coefficient;
                if (c != 0.0)
                {
                    Ga.noalias() -= c * ri.K(dPa_w);
                    Gb.noalias() -= c * ri.K(dPb_w);
                }
                Eigen::VectorXd out(nova + novb);
                out.head(nova) = diag_a.cwiseProduct(x.head(nova)) +
                                 pack_hessian_vector_product_cphf_order(Ga, Cao_w, Cav_w) +
                                 pack_hessian_vector_product_cphf_order(dV->first, in.Ca_occ, in.Ca_virt);
                out.tail(novb) = diag_b.cwiseProduct(x.tail(novb)) +
                                 pack_hessian_vector_product_cphf_order(Gb, Cbo_w, Cbv_w) +
                                 pack_hessian_vector_product_cphf_order(dV->second, in.Cb_occ, in.Cb_virt);
                return out;
            }

            // J from the total trial density, as the UKS Coulomb build.
            const Eigen::MatrixXd dJ = _compute_2e_j_direct(
                *in.shell_pairs, dPa + dPb, nb, in.engine, HartreeFock::ERIKernel::Coulomb, 0.0,
                in.tol_eri, in.sym_ops);
            const auto dV_xc = kernel.apply(dPa, dPb);
            if (!dV_xc)
                return Eigen::VectorXd::Zero(x.size());

            // K per spin, like the UKS Fock (-1 * (c_fr K + c_sr K_SR)); UKS has no
            // RKS-style 2x on the kernel (docs/SOSCF_DFT.md invariant 2).
            Eigen::VectorXd Ka = Eigen::VectorXd::Zero(nova), Kb = Eigen::VectorXd::Zero(novb);
            const double c_fr = in.full_range_exchange_coefficient, c_sr = in.short_range_exchange_coefficient;
            if (c_fr != 0.0 || c_sr != 0.0)
            {
                Eigen::MatrixXd dKa = Eigen::MatrixXd::Zero(nb, nb), dKb = Eigen::MatrixXd::Zero(nb, nb);
                if (c_fr != 0.0)
                {
                    const auto [K_a, K_b] = _compute_2e_k_uhf_direct(
                        *in.shell_pairs, dPa, dPb, nb, in.engine, HartreeFock::ERIKernel::Coulomb, 0.0,
                        in.tol_eri, in.sym_ops);
                    dKa.noalias() += c_fr * K_a;
                    dKb.noalias() += c_fr * K_b;
                }
                if (c_sr != 0.0)
                {
                    const auto [K_a, K_b] = _compute_2e_k_uhf_direct(
                        *in.shell_pairs, dPa, dPb, nb, in.engine, HartreeFock::ERIKernel::ShortRange,
                        in.range_separation_omega, in.tol_eri, in.sym_ops);
                    dKa.noalias() += c_sr * K_a;
                    dKb.noalias() += c_sr * K_b;
                }
                Ka = pack_hessian_vector_product_cphf_order(-1.0 * dKa, in.Ca_occ, in.Ca_virt);
                Kb = pack_hessian_vector_product_cphf_order(-1.0 * dKb, in.Cb_occ, in.Cb_virt);
            }

            Eigen::VectorXd out(nova + novb);
            out.head(nova) = diag_a.cwiseProduct(x.head(nova)) +
                             pack_hessian_vector_product_cphf_order(dJ, in.Ca_occ, in.Ca_virt) +
                             pack_hessian_vector_product_cphf_order(dV_xc->first, in.Ca_occ, in.Ca_virt) + Ka;
            out.tail(novb) = diag_b.cwiseProduct(x.tail(novb)) +
                             pack_hessian_vector_product_cphf_order(dJ, in.Cb_occ, in.Cb_virt) +
                             pack_hessian_vector_product_cphf_order(dV_xc->second, in.Cb_occ, in.Cb_virt) + Kb;
            return out;
        });
    }
}
