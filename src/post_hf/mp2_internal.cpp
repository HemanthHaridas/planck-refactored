#include "mp2_internal.h"

#include <algorithm>

#include "post_hf/integrals.h"

namespace
{
    std::vector<bool> build_active_mask(const std::vector<int> &frozen, int n_mo_full)
    {
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
