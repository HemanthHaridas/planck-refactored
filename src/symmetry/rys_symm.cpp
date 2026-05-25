#include "symmetry/rys_symm.h"

#include "integrals/rys.h"
#include "symmetry/fock_symmetrization.h"
#include "symmetry/skeleton_eri.h"

// Full-symmetry direct Fock (Rys quadrature). Mirror of os_symm.cpp; only the
// contracted-ERI primitive differs (RysQuad::_rys_contracted_eri). Shared skeleton
// machinery in skeleton_eri.h. Separate from rys.cpp for A/B benchmarking.

namespace
{
    inline double rys_eri(const HartreeFock::ShellPair &spAB,
                         const HartreeFock::ShellPair &spCD,
                         int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
                         int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
                         HartreeFock::ERIKernel kernel, double omega)
    {
        return HartreeFock::RysQuad::_rys_contracted_eri(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz,
            kernel, omega);
    }
} // namespace

namespace HartreeFock::RysQuad
{
    std::expected<Eigen::MatrixXd, std::string> _compute_2e_fock_symm(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        const Eigen::MatrixXd &density,
        std::size_t nbasis,
        const HartreeFock::Symmetry::GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri)
    {
        const std::size_t nb = nbasis;
        const bool use_sym = ops.valid && ops.operations.size() > 1;

        auto eri = HartreeFock::Symmetry::build_skeleton_eri(
            shell_pairs, basis, nb, ops, use_sym,
            [&](const HartreeFock::ShellPair &ab, const HartreeFock::ShellPair &cd,
                int ax, int ay, int az, int bx, int by, int bz,
                int cx, int cy, int cz, int dx, int dy, int dz)
            { return rys_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
            tol_eri);
        if (!eri)
            return std::unexpected(eri.error());

        Eigen::MatrixXd G = HartreeFock::Symmetry::contract_fock_rhf(*eri, nb, density);
        if (!use_sym)
            return G;
        return HartreeFock::Symmetry::symmetrize_matrix(G, ops);
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    _compute_2e_fock_uhf_symm(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        const Eigen::MatrixXd &Pa,
        const Eigen::MatrixXd &Pb,
        std::size_t nbasis,
        const HartreeFock::Symmetry::GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri)
    {
        const std::size_t nb = nbasis;
        const bool use_sym = ops.valid && ops.operations.size() > 1;

        auto eri = HartreeFock::Symmetry::build_skeleton_eri(
            shell_pairs, basis, nb, ops, use_sym,
            [&](const HartreeFock::ShellPair &ab, const HartreeFock::ShellPair &cd,
                int ax, int ay, int az, int bx, int by, int bz,
                int cx, int cy, int cz, int dx, int dy, int dz)
            { return rys_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
            tol_eri);
        if (!eri)
            return std::unexpected(eri.error());

        auto [Ga, Gb] = HartreeFock::Symmetry::contract_fock_uhf(*eri, nb, Pa, Pb);
        if (!use_sym)
            return std::make_pair(Ga, Gb);

        auto Ga_s = HartreeFock::Symmetry::symmetrize_matrix(Ga, ops);
        if (!Ga_s)
            return std::unexpected(Ga_s.error());
        auto Gb_s = HartreeFock::Symmetry::symmetrize_matrix(Gb, ops);
        if (!Gb_s)
            return std::unexpected(Gb_s.error());
        return std::make_pair(*Ga_s, *Gb_s);
    }

    // ── Spherical-mode full-symmetry Fock (Step 2) — mirror of os_symm ──────────────
    std::expected<Eigen::MatrixXd, std::string> _compute_2e_fock_symm_spherical(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        const Eigen::MatrixXd &density,
        std::size_t nbasis_cart,
        const Eigen::MatrixXd &cart_to_sph,
        const HartreeFock::Symmetry::GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri)
    {
        const bool use_sym = ops.valid && ops.operations.size() > 1;
        auto eri = HartreeFock::Symmetry::build_skeleton_eri(
            shell_pairs, basis, nbasis_cart, ops, use_sym,
            [&](const HartreeFock::ShellPair &ab, const HartreeFock::ShellPair &cd,
                int ax, int ay, int az, int bx, int by, int bz,
                int cx, int cy, int cz, int dx, int dy, int dz)
            { return rys_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
            tol_eri);
        if (!eri)
            return std::unexpected(eri.error());
        return HartreeFock::Symmetry::spherical_fock_rhf_from_skeleton(
            *eri, nbasis_cart, cart_to_sph, density, ops, use_sym);
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    _compute_2e_fock_uhf_symm_spherical(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        const Eigen::MatrixXd &Pa,
        const Eigen::MatrixXd &Pb,
        std::size_t nbasis_cart,
        const Eigen::MatrixXd &cart_to_sph,
        const HartreeFock::Symmetry::GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri)
    {
        const bool use_sym = ops.valid && ops.operations.size() > 1;
        auto eri = HartreeFock::Symmetry::build_skeleton_eri(
            shell_pairs, basis, nbasis_cart, ops, use_sym,
            [&](const HartreeFock::ShellPair &ab, const HartreeFock::ShellPair &cd,
                int ax, int ay, int az, int bx, int by, int bz,
                int cx, int cy, int cz, int dx, int dy, int dz)
            { return rys_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
            tol_eri);
        if (!eri)
            return std::unexpected(eri.error());
        return HartreeFock::Symmetry::spherical_fock_uhf_from_skeleton(
            *eri, nbasis_cart, cart_to_sph, Pa, Pb, ops, use_sym);
    }
} // namespace HartreeFock::RysQuad
