#include "symmetry/os_symm.h"

#include "integrals/os.h"
#include "symmetry/fock_symmetrization.h"
#include "symmetry/skeleton_eri.h"

// Full-symmetry direct Fock (Obara-Saika). The petite-list / multiplicity /
// scatter / contraction logic is engine-agnostic and lives in skeleton_eri.h; this
// file only supplies the OS contracted-ERI primitive. Kept separate from os.cpp so
// the production D2h path is untouched and the two can be A/B benchmarked.

namespace
{
    // OS contracted ERI for one Cartesian-component shell quartet.
    inline double os_eri(const HartreeFock::ShellPair &spAB,
                         const HartreeFock::ShellPair &spCD,
                         int lAx, int lAy, int lAz, int lBx, int lBy, int lBz,
                         int lCx, int lCy, int lCz, int lDx, int lDy, int lDz,
                         HartreeFock::ERIKernel kernel, double omega)
    {
        return HartreeFock::ObaraSaika::_contracted_eri_elem(
            spAB, spCD, lAx, lAy, lAz, lBx, lBy, lBz, lCx, lCy, lCz, lDx, lDy, lDz,
            kernel, omega);
    }
} // namespace

namespace HartreeFock::ObaraSaika
{
    std::expected<std::vector<double>, std::string> _build_skeleton_eri_symm(
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        std::size_t nbasis,
        const HartreeFock::Symmetry::GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri)
    {
        const bool use_sym = ops.valid && ops.operations.size() > 1;
        return HartreeFock::Symmetry::build_skeleton_eri(
            shell_pairs, basis, nbasis, ops, use_sym,
            [&](const HartreeFock::ShellPair &ab, const HartreeFock::ShellPair &cd,
                int ax, int ay, int az, int bx, int by, int bz,
                int cx, int cy, int cz, int dx, int dy, int dz)
            { return os_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
            tol_eri);
    }

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
            { return os_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
            tol_eri);
        if (!eri)
            return std::unexpected(eri.error());

        Eigen::MatrixXd G = HartreeFock::Symmetry::contract_fock_rhf(*eri, nb, density);
        if (!use_sym)
            return G; // skeleton == full Fock when no reduction is applied
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
            { return os_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
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

    // ── Spherical-mode full-symmetry Fock (Step 2) ─────────────────────────────────
    // Cartesian skeleton ERI (petite list over Cartesian shell quartets), then the
    // tensor is transformed to spherical, contracted with the spherical density, and
    // symmetrized with the spherical O_R. `cart_to_sph` is C [nb_sph × nb_cart];
    // `nbasis_cart` is the Cartesian AO count. `density`/result are spherical-sized.
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
            { return os_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
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
            { return os_eri(ab, cd, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, kernel, omega); },
            tol_eri);
        if (!eri)
            return std::unexpected(eri.error());
        return HartreeFock::Symmetry::spherical_fock_uhf_from_skeleton(
            *eri, nbasis_cart, cart_to_sph, Pa, Pb, ops, use_sym);
    }
} // namespace HartreeFock::ObaraSaika
