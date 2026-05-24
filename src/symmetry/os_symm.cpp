#include "symmetry/os_symm.h"

#include <array>
#include <cmath>
#include <map>
#include <tuple>

#include "integrals/os.h"
#include "symmetry/fock_symmetrization.h"

// Full-symmetry direct Fock (Obara-Saika). See os_symm.h for the method. This file
// is deliberately separate from os.cpp so the production D2h-only path is untouched
// and the two can be A/B benchmarked and cross-validated.

namespace
{
    using HartreeFock::ShellPair;
    using HartreeFock::Symmetry::GroupOperations;

    // Canonicalize a shell quartet (sa,sb|sc,sd) under the 8-fold permutational ERI
    // symmetry to its lexicographic-minimum key, for orbit-representative tests.
    static std::array<int, 4> canon_quartet(int sa, int sb, int sc, int sd)
    {
        if (sa < sb)
            std::swap(sa, sb);
        if (sc < sd)
            std::swap(sc, sd);
        if (std::tie(sa, sb) < std::tie(sc, sd))
        {
            std::swap(sa, sc);
            std::swap(sb, sd);
        }
        return {sa, sb, sc, sd};
    }

    // Is (sa,sb|sc,sd) the lexicographic representative of its orbit under the full
    // group's shell permutations (combined with the 8-fold ERI symmetry already in
    // canon_quartet)? Representative iff no group image gives a strictly smaller key.
    static bool is_quartet_representative(int sa, int sb, int sc, int sd,
                                          const GroupOperations &ops)
    {
        const std::array<int, 4> self = canon_quartet(sa, sb, sc, sd);
        for (const auto &op : ops.operations)
        {
            const auto &P = op.shell_perm;
            if (canon_quartet(P[sa], P[sb], P[sc], P[sd]) < self)
                return false;
        }
        return true;
    }

    // Number of DISTINCT shell quartets in the orbit of (sa,sb|sc,sd) under the
    // group. The skeleton contracts only the representative; weighting it by the
    // orbit size lets symmetrize() redistribute the contribution exactly across the
    // orbit so the projected Fock equals the full Fock. (See os_symm.h derivation.)
    static int orbit_multiplicity(int sa, int sb, int sc, int sd,
                                  const GroupOperations &ops)
    {
        std::vector<std::array<int, 4>> seen;
        seen.reserve(ops.operations.size());
        for (const auto &op : ops.operations)
        {
            const auto &P = op.shell_perm;
            const std::array<int, 4> img = canon_quartet(P[sa], P[sb], P[sc], P[sd]);
            bool found = false;
            for (const auto &e : seen)
                if (e == img)
                {
                    found = true;
                    break;
                }
            if (!found)
                seen.push_back(img);
        }
        return static_cast<int>(seen.size());
    }

    // Store-only 8-fold permutational scatter of one (i,j|k,l) AO-block value into
    // the dense ERI tensor (same convention as os.cpp::write_eri_permutations).
    static void scatter8(std::vector<double> &eri, std::size_t nb,
                         std::size_t i, std::size_t j, std::size_t k, std::size_t l,
                         double val)
    {
        const std::size_t nb2 = nb * nb, nb3 = nb * nb * nb;
        const std::array<std::array<std::size_t, 4>, 8> perms = {{{i, j, k, l},
                                                                  {j, i, k, l},
                                                                  {i, j, l, k},
                                                                  {j, i, l, k},
                                                                  {k, l, i, j},
                                                                  {l, k, i, j},
                                                                  {k, l, j, i},
                                                                  {l, k, j, i}}};
        for (const auto &pm : perms)
            eri[pm[0] * nb3 + pm[1] * nb2 + pm[2] * nb + pm[3]] = val;
    }

    // Build the (orbit-weighted) skeleton ERI tensor over the petite list. When
    // use_sym is false this computes every quartet once → skeleton == full tensor.
    static std::expected<std::vector<double>, std::string> build_skeleton_eri(
        const std::vector<ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        std::size_t nb,
        const GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        bool use_sym)
    {
        // Shell pointer → index (in Basis::_shells order, which ops.shell_perm uses).
        std::map<const HartreeFock::Shell *, int> shell_id;
        for (int s = 0; s < static_cast<int>(basis._shells.size()); ++s)
            shell_id[&basis._shells[s]] = s;

        std::vector<double> eri(nb * nb * nb * nb, 0.0);
        const std::size_t npairs = shell_pairs.size();

        for (std::size_t p = 0; p < npairs; ++p)
        {
            const auto &spAB = shell_pairs[p];
            const int sa = shell_id.at(spAB.A._shell);
            const int sb = shell_id.at(spAB.B._shell);
            const std::size_t i = spAB.A._index;
            const std::size_t j = spAB.B._index;
            const int lAx = spAB.A._cartesian[0], lAy = spAB.A._cartesian[1], lAz = spAB.A._cartesian[2];
            const int lBx = spAB.B._cartesian[0], lBy = spAB.B._cartesian[1], lBz = spAB.B._cartesian[2];

            for (std::size_t q = p; q < npairs; ++q)
            {
                const auto &spCD = shell_pairs[q];
                const int sc = shell_id.at(spCD.A._shell);
                const int sd = shell_id.at(spCD.B._shell);
                const std::size_t k = spCD.A._index;
                const std::size_t l = spCD.B._index;

                if (use_sym && !is_quartet_representative(sa, sb, sc, sd, ops))
                    continue;

                const int lCx = spCD.A._cartesian[0], lCy = spCD.A._cartesian[1], lCz = spCD.A._cartesian[2];
                const int lDx = spCD.B._cartesian[0], lDy = spCD.B._cartesian[1], lDz = spCD.B._cartesian[2];

                double val = HartreeFock::ObaraSaika::_contracted_eri_elem(
                    spAB, spCD,
                    lAx, lAy, lAz, lBx, lBy, lBz,
                    lCx, lCy, lCz, lDx, lDy, lDz,
                    kernel, omega);

                if (use_sym)
                    val *= static_cast<double>(orbit_multiplicity(sa, sb, sc, sd, ops));

                scatter8(eri, nb, i, j, k, l, val);
            }
        }
        return eri;
    }
} // namespace

namespace HartreeFock::ObaraSaika
{
    std::expected<Eigen::MatrixXd, std::string> _compute_2e_fock_symm(
        const std::vector<ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        const Eigen::MatrixXd &density,
        std::size_t nbasis,
        const GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri)
    {
        (void)tol_eri; // screening to be added with the petite list later
        const std::size_t nb = nbasis;
        const std::size_t nb2 = nb * nb, nb3 = nb * nb * nb;
        const bool use_sym = ops.valid && ops.operations.size() > 1;

        auto eri_res = build_skeleton_eri(shell_pairs, basis, nb, ops, kernel, omega, use_sym);
        if (!eri_res)
            return std::unexpected(eri_res.error());
        const std::vector<double> &eri = *eri_res;

        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        G(mu, nu) += density(lam, sig) *
                                     (eri[mu * nb3 + nu * nb2 + lam * nb + sig] -
                                      0.5 * eri[mu * nb3 + lam * nb2 + nu * nb + sig]);

        if (!use_sym)
            return G; // skeleton == full Fock when no reduction is applied

        return HartreeFock::Symmetry::symmetrize_matrix(G, ops);
    }

    std::expected<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>, std::string>
    _compute_2e_fock_uhf_symm(
        const std::vector<ShellPair> &shell_pairs,
        const HartreeFock::Basis &basis,
        const Eigen::MatrixXd &Pa,
        const Eigen::MatrixXd &Pb,
        std::size_t nbasis,
        const GroupOperations &ops,
        HartreeFock::ERIKernel kernel,
        double omega,
        double tol_eri)
    {
        (void)tol_eri;
        const std::size_t nb = nbasis;
        const std::size_t nb2 = nb * nb, nb3 = nb * nb * nb;
        const bool use_sym = ops.valid && ops.operations.size() > 1;

        auto eri_res = build_skeleton_eri(shell_pairs, basis, nb, ops, kernel, omega, use_sym);
        if (!eri_res)
            return std::unexpected(eri_res.error());
        const std::vector<double> &eri = *eri_res;

        const Eigen::MatrixXd Pt = Pa + Pb;
        Eigen::MatrixXd Ga = Eigen::MatrixXd::Zero(nb, nb);
        Eigen::MatrixXd Gb = Eigen::MatrixXd::Zero(nb, nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                    {
                        const double coulomb = eri[mu * nb3 + nu * nb2 + lam * nb + sig];
                        const double exch = eri[mu * nb3 + lam * nb2 + nu * nb + sig];
                        Ga(mu, nu) += Pt(lam, sig) * coulomb - Pa(lam, sig) * exch;
                        Gb(mu, nu) += Pt(lam, sig) * coulomb - Pb(lam, sig) * exch;
                    }

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
} // namespace HartreeFock::ObaraSaika
