#include "mp2_internal.h"

#include <algorithm>

#include "post_hf/integrals.h"
#include "post_hf/ri/ri_eri.h"

namespace
{
    std::vector<bool> build_active_mask(const std::vector<int> &frozen, int n_mo_full)
    {
        // MP2 freezing accepts either "freeze the first n orbitals" or an
        // explicit list of MO indices. Everything else stays in the active
        // transformed ERI space to keep the downstream kernels simple.
        std::vector<bool> active(static_cast<std::size_t>(n_mo_full), true);
        if (frozen.empty())
            return active;

        if (frozen.size() == 1 && frozen.front() >= 0)
        {
            const int n_freeze = std::min(frozen.front(), n_mo_full);
            for (int i = 0; i < n_freeze; ++i)
                active[static_cast<std::size_t>(i)] = false;
            return active;
        }

        for (const int idx : frozen)
            if (idx >= 0 && idx < n_mo_full)
                active[static_cast<std::size_t>(idx)] = false;
        return active;
    }

    // build_ri_pair_factors / build_ri_mo_block now live in ri_eri.{h,cpp}
    // (HartreeFock::Correlation::RI) so the conventional post-HF paths can reuse
    // them. MP2's own ov→ovov packing stays here.

    std::vector<double> ov_block_to_ovov(const Eigen::MatrixXd &b_ov, int nocc, int nvirt)
    {
        const Eigen::MatrixXd gram = b_ov * b_ov.transpose();
        const int nov = nocc * nvirt;
        std::vector<double> ovov(static_cast<std::size_t>(nov) * nov);
        for (int i = 0; i < nocc; ++i)
            for (int a = 0; a < nvirt; ++a)
                for (int j = 0; j < nocc; ++j)
                    for (int b = 0; b < nvirt; ++b)
                        ovov[HartreeFock::Correlation::detail::idx_ovov(
                            i, a, j, b, nocc, nvirt)] =
                            gram(i * nvirt + a, j * nvirt + b);
        return ovov;
    }

    std::vector<double> ov_block_cross_to_ovov(
        const Eigen::MatrixXd &left_b_ov,
        const Eigen::MatrixXd &right_b_ov,
        int left_nocc,
        int right_nocc,
        int left_nvirt,
        int right_nvirt)
    {
        const Eigen::MatrixXd gram = left_b_ov * right_b_ov.transpose();
        const int left_nov = left_nocc * left_nvirt;
        const int right_nov = right_nocc * right_nvirt;

        std::vector<double> ovov(static_cast<std::size_t>(left_nov) * right_nov);
        for (int i = 0; i < left_nocc; ++i)
            for (int a = 0; a < left_nvirt; ++a)
                for (int j = 0; j < right_nocc; ++j)
                    for (int b = 0; b < right_nvirt; ++b)
                        ovov[HartreeFock::Correlation::detail::idx_ovOV(
                            i, a, j, b, left_nocc, right_nocc, left_nvirt, right_nvirt)] =
                            gram(i * left_nvirt + a, j * right_nvirt + b);
        return ovov;
    }
}

namespace HartreeFock::Correlation::detail
{
    std::expected<RMP2Dims, std::string> resolve_rmp2_dims(
        const HartreeFock::Calculator &calculator,
        const HartreeFock::OptionsMP2 &options)
    {
        if (calculator._scf._scf == HartreeFock::SCFType::UHF || calculator._info._scf.is_uhf)
            return std::unexpected("resolve_rmp2_dims: RHF reference required.");
        if (!calculator._info._is_converged)
            return std::unexpected("resolve_rmp2_dims: SCF not converged.");

        const int n_mo_full = static_cast<int>(calculator.working_nbasis());
        int n_electrons = 0;
        for (auto Z : calculator._molecule.atomic_numbers)
            n_electrons += static_cast<int>(Z);
        n_electrons -= calculator._molecule.charge;
        if (n_electrons % 2 != 0)
            return std::unexpected("resolve_rmp2_dims: closed-shell RHF reference required.");

        const int n_occ_full = n_electrons / 2;
        const auto active_mask = build_active_mask(options.frozen, n_mo_full);

        RMP2Dims dims;
        dims.n_mo_full = n_mo_full;
        dims.n_occ_full = n_occ_full;
        // Compress the full RHF orbital list down to the active occupied/virtual
        // blocks that the canonical MP2 formulas expect.
        dims.active_mo.reserve(static_cast<std::size_t>(n_mo_full));
        for (int p = 0; p < n_mo_full; ++p)
        {
            if (!active_mask[static_cast<std::size_t>(p)])
                continue;
            dims.active_mo.push_back(p);
            if (p < n_occ_full)
                ++dims.n_occ;
            else
                ++dims.n_virt;
        }
        dims.n_mo = dims.n_occ + dims.n_virt;
        if (dims.n_occ <= 0 || dims.n_virt <= 0)
            return std::unexpected("resolve_rmp2_dims: no occupied or virtual orbitals after freezing.");
        return dims;
    }

    std::expected<UMP2Dims, std::string> resolve_ump2_dims(
        const HartreeFock::Calculator &calculator,
        const HartreeFock::OptionsMP2 &options)
    {
        if (calculator._scf._scf != HartreeFock::SCFType::UHF || !calculator._info._scf.is_uhf)
            return std::unexpected("resolve_ump2_dims: UHF reference required.");
        if (!calculator._info._is_converged)
            return std::unexpected("resolve_ump2_dims: SCF not converged.");

        const int nb = static_cast<int>(calculator.working_nbasis());
        int n_electrons = 0;
        for (auto Z : calculator._molecule.atomic_numbers)
            n_electrons += static_cast<int>(Z);
        n_electrons -= calculator._molecule.charge;
        const int n_unpaired = static_cast<int>(calculator._molecule.multiplicity) - 1;

        UMP2Dims dims;
        dims.nmoa_full = nb;
        dims.nmob_full = nb;
        dims.nocca_full = (n_electrons + n_unpaired) / 2;
        dims.noccb_full = (n_electrons - n_unpaired) / 2;

        const auto active_mask = build_active_mask(options.frozen, nb);
        // UMP2 tracks alpha and beta active spaces separately even when they
        // share the same frozen-orbital policy, because the occupied counts can
        // differ for open-shell references.
        dims.active_a.reserve(static_cast<std::size_t>(nb));
        dims.active_b.reserve(static_cast<std::size_t>(nb));
        for (int p = 0; p < nb; ++p)
        {
            if (!active_mask[static_cast<std::size_t>(p)])
                continue;
            dims.active_a.push_back(p);
            dims.active_b.push_back(p);
            if (p < dims.nocca_full)
                ++dims.nocca;
            else
                ++dims.nvira;
            if (p < dims.noccb_full)
                ++dims.noccb;
            else
                ++dims.nvirb;
        }
        dims.nmoa = dims.nocca + dims.nvira;
        dims.nmob = dims.noccb + dims.nvirb;
        if (dims.nocca <= 0 || dims.nvira <= 0 || dims.noccb < 0 || dims.nvirb <= 0)
            return std::unexpected("resolve_ump2_dims: empty active block after freezing.");
        return dims;
    }

    std::expected<ChemistsERIs, std::string> make_eris_rmp2(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const RMP2Dims &dims)
    {
        const int nb = static_cast<int>(calculator.working_nbasis());
        const Eigen::MatrixXd &C_full = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::VectorXd &eps_full = calculator._info._scf.alpha.mo_energies;

        Eigen::MatrixXd C_act(nb, dims.n_mo);
        Eigen::VectorXd eps_act(dims.n_mo);
        for (int i = 0; i < dims.n_mo; ++i)
        {
            const int src = dims.active_mo[static_cast<std::size_t>(i)];
            C_act.col(i) = C_full.col(src);
            eps_act(i) = eps_full(src);
        }

        ChemistsERIs out;
        out.mo_coeff = C_act;
        out.nocc = dims.n_occ;
        out.mo_energy = eps_act;
        out.fock = Eigen::MatrixXd(eps_act.asDiagonal());

        const Eigen::MatrixXd C_occ = C_act.leftCols(dims.n_occ);
        const Eigen::MatrixXd C_virt = C_act.middleCols(dims.n_occ, dims.n_virt);

        if (calculator._mp2.use_ri)
        {
            auto ri_ready = HartreeFock::Correlation::RI::ensure_ri_3c_ready(calculator);
            if (!ri_ready)
                return std::unexpected("make_eris_rmp2: " + ri_ready.error());
            if (!calculator._ri_metric_factor)
                return std::unexpected("make_eris_rmp2: RI metric factorization is missing.");

            const Eigen::MatrixXd pair_factors =
                HartreeFock::Correlation::RI::build_ri_pair_factors(calculator);
            const Eigen::MatrixXd b_ov =
                HartreeFock::Correlation::RI::build_ri_mo_block(pair_factors, C_occ, C_virt);
            out.ovov = ov_block_to_ovov(b_ov, dims.n_occ, dims.n_virt);
            return out;
        }

        // Canonical RHF MP2 only needs the occupied-virtual-occupied-virtual
        // block in chemists' notation, so we transform directly into ovov.
        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, "RMP2 :");
        out.ovov = HartreeFock::Correlation::transform_eri(
            eri, static_cast<std::size_t>(nb), C_occ, C_virt, C_occ, C_virt);
        return out;
    }

    std::expected<UChemistsERIs, std::string> make_eris_ump2(
        HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const UMP2Dims &dims)
    {
        const int nb = static_cast<int>(calculator.working_nbasis());
        const Eigen::MatrixXd &Ca_full = calculator._info._scf.alpha.mo_coefficients;
        const Eigen::MatrixXd &Cb_full = calculator._info._scf.beta.mo_coefficients;
        const Eigen::VectorXd &epsa_full = calculator._info._scf.alpha.mo_energies;
        const Eigen::VectorXd &epsb_full = calculator._info._scf.beta.mo_energies;

        Eigen::MatrixXd Ca(nb, dims.nmoa), Cb(nb, dims.nmob);
        Eigen::VectorXd ea(dims.nmoa), eb(dims.nmob);
        for (int i = 0; i < dims.nmoa; ++i)
        {
            const int src = dims.active_a[static_cast<std::size_t>(i)];
            Ca.col(i) = Ca_full.col(src);
            ea(i) = epsa_full(src);
        }
        for (int i = 0; i < dims.nmob; ++i)
        {
            const int src = dims.active_b[static_cast<std::size_t>(i)];
            Cb.col(i) = Cb_full.col(src);
            eb(i) = epsb_full(src);
        }

        UChemistsERIs out;
        out.mo_coeff_a = Ca;
        out.mo_coeff_b = Cb;
        out.nocca = dims.nocca;
        out.noccb = dims.noccb;
        out.mo_energy_a = ea;
        out.mo_energy_b = eb;
        out.fock_a = Eigen::MatrixXd(ea.asDiagonal());
        out.fock_b = Eigen::MatrixXd(eb.asDiagonal());

        const Eigen::MatrixXd Ca_occ = Ca.leftCols(dims.nocca);
        const Eigen::MatrixXd Ca_virt = Ca.middleCols(dims.nocca, dims.nvira);
        const Eigen::MatrixXd Cb_occ = Cb.leftCols(dims.noccb);
        const Eigen::MatrixXd Cb_virt = Cb.middleCols(dims.noccb, dims.nvirb);

        if (calculator._mp2.use_ri)
        {
            auto ri_ready = HartreeFock::Correlation::RI::ensure_ri_3c_ready(calculator);
            if (!ri_ready)
                return std::unexpected("make_eris_ump2: " + ri_ready.error());
            if (!calculator._ri_metric_factor)
                return std::unexpected("make_eris_ump2: RI metric factorization is missing.");

            const Eigen::MatrixXd pair_factors =
                HartreeFock::Correlation::RI::build_ri_pair_factors(calculator);

            // The fitted AO-pair factors are spin-independent; the unrestricted
            // split only enters when we project them into alpha/beta occupied
            // and virtual MO blocks.
            const Eigen::MatrixXd b_ov_a =
                HartreeFock::Correlation::RI::build_ri_mo_block(pair_factors, Ca_occ, Ca_virt);
            const Eigen::MatrixXd b_ov_b =
                HartreeFock::Correlation::RI::build_ri_mo_block(pair_factors, Cb_occ, Cb_virt);
            out.ovov = ov_block_to_ovov(b_ov_a, dims.nocca, dims.nvira);
            out.OVOV = ov_block_to_ovov(b_ov_b, dims.noccb, dims.nvirb);
            out.ovOV = ov_block_cross_to_ovov(
                b_ov_a, b_ov_b, dims.nocca, dims.noccb, dims.nvira, dims.nvirb);
            return out;
        }

        // Unrestricted MP2 needs three spin blocks: alpha-alpha, beta-beta, and
        // alpha-beta. Keeping them separate makes the later spin-resolved energy
        // formulas read much closer to the textbook expressions.
        std::vector<double> eri_local;
        const std::vector<double> &eri = HartreeFock::Correlation::ensure_eri(
            calculator, shell_pairs, eri_local, "UMP2 :");
        out.ovov = HartreeFock::Correlation::transform_eri(
            eri, static_cast<std::size_t>(nb), Ca_occ, Ca_virt, Ca_occ, Ca_virt);
        out.OVOV = HartreeFock::Correlation::transform_eri(
            eri, static_cast<std::size_t>(nb), Cb_occ, Cb_virt, Cb_occ, Cb_virt);
        out.ovOV = HartreeFock::Correlation::transform_eri(
            eri, static_cast<std::size_t>(nb), Ca_occ, Ca_virt, Cb_occ, Cb_virt);
        return out;
    }
} // namespace HartreeFock::Correlation::detail
