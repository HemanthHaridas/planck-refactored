#include "ecp.h"

#include <format>
#include <vector>

#include <libecpint.hpp>

#include "lookup/elements.h"

namespace
{
    // Planck's shells and ECPs in libecpint's form, shared by U and its derivative.
    struct LibecpintInput
    {
        std::vector<libecpint::GaussianShell> shells;
        std::vector<std::size_t> first; // first basis function of each shell
        std::vector<libecpint::ECP> ecps;
        int max_lb = 0, max_lu = 0;
    };

    LibecpintInput to_libecpint(const HartreeFock::Basis &basis)
    {
        // libecpint integrates c_p x^i y^j z^k exp(-a r^2) with the coefficients it is
        // given. Planck's function is component_norm * sum_p (c_p Nc) N_p x^i y^j z^k
        // exp(-a r^2) with N_p the x^L primitive norm, so pass (c_p Nc) N_p and apply
        // component_norm per element. Both order Cartesian components xx, xy, xz, yy, ...
        LibecpintInput in;
        std::size_t offset = 0;
        for (const HartreeFock::Shell &sh : basis._shells)
        {
            const int l = static_cast<int>(sh._shell);
            in.max_lb = std::max(in.max_lb, l);
            const std::array<double, 3> center = {sh._center[0], sh._center[1], sh._center[2]};
            libecpint::GaussianShell g(center, l);
            for (Eigen::Index p = 0; p < sh._primitives.size(); ++p)
                g.addPrim(sh._primitives[p], sh._coefficients[p] * sh._normalizations[p]);
            in.shells.push_back(std::move(g));
            in.first.push_back(offset);
            offset += (l + 1) * (l + 2) / 2;
        }

        // libecpint takes n as written in the Gaussian94 file (r^(n-2)) and treats the
        // channel with the largest l as the local part, so U_L goes in at l = L.
        for (const HartreeFock::ECPPotential &e : basis._ecps)
        {
            const int L = static_cast<int>(e._semilocal.size());
            in.max_lu = std::max(in.max_lu, L);
            libecpint::ECP u(e._center.data());
            for (const HartreeFock::ECPTerm &t : e._local)
                u.addPrimitive(t.n, L, t.exponent, t.coefficient, false);
            for (int l = 0; l < L; ++l)
                for (const HartreeFock::ECPTerm &t : e._semilocal[l])
                    u.addPrimitive(t.n, l, t.exponent, t.coefficient, false);
            u.sort();
            in.ecps.push_back(std::move(u));
        }
        return in;
    }
} // namespace

Eigen::MatrixXd HartreeFock::ECP::_compute_ecp(const HartreeFock::Basis &basis)
{
    const std::size_t nbasis = basis.nbasis();
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(nbasis, nbasis);
    if (basis._ecps.empty())
        return U;

    const auto [shells, first, ecps, max_lb, max_lu] = to_libecpint(basis);
    const libecpint::ECPIntegral engine(max_lb, max_lu);
    const int nshell = static_cast<int>(shells.size());

    // Each (a, b >= a) block is owned by one iteration: no shared writes, and the
    // ECP sum inside runs in fixed order, so U is thread-count invariant.
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int a = 0; a < nshell; ++a)
    {
        libecpint::TwoIndex<double> block;
        for (int b = a; b < nshell; ++b)
        {
            const int na = shells[a].ncartesian();
            const int nb = shells[b].ncartesian();
            Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(na, nb);
            for (const libecpint::ECP &u : ecps)
            {
                engine.compute_shell_pair(u, shells[a], shells[b], block);
                for (int i = 0; i < na; ++i)
                    for (int j = 0; j < nb; ++j)
                        sum(i, j) += block(i, j);
            }
            for (int i = 0; i < na; ++i)
                for (int j = 0; j < nb; ++j)
                {
                    const std::size_t mu = first[a] + i;
                    const std::size_t nu = first[b] + j;
                    const double v = sum(i, j) * basis._basis_functions[mu]._component_norm *
                                     basis._basis_functions[nu]._component_norm;
                    U(mu, nu) = v;
                    U(nu, mu) = v;
                }
        }
    }
    return U;
}

Eigen::MatrixXd HartreeFock::ECP::_compute_ecp_gradient(const HartreeFock::Basis &basis,
                                                       const std::vector<int> &shell_atom,
                                                       std::size_t natoms,
                                                       const Eigen::MatrixXd &D)
{
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(natoms), 3);
    if (basis._ecps.empty())
        return grad;

    const auto [shells, first, ecps, max_lb, max_lu] = to_libecpint(basis);
    const libecpint::ECPIntegral engine(max_lb, max_lu, 1);
    const int nshell = static_cast<int>(shells.size());

    // dE/dR = sum_{mu nu} D_mu_nu dU_mu_nu/dR over the (a, b >= a) triangle, weight 2
    // off the diagonal block. libecpint returns [Ax..Az, Bx..Bz, Cx..Cz], each meant to
    // be added to its own centre's gradient. Shell pairs share atom rows, so pairs are
    // binned by a and the bins summed in fixed order: thread-count invariant.
    constexpr int kBins = 64;
    std::vector<Eigen::MatrixXd> bins(kBins, grad);
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int bin = 0; bin < kBins; ++bin)
    {
        std::array<libecpint::TwoIndex<double>, 9> d;
        for (int a = bin; a < nshell; a += kBins)
            for (int b = a; b < nshell; ++b)
            {
                const int na = shells[a].ncartesian();
                const int nb = shells[b].ncartesian();
                const double w = (a == b) ? 1.0 : 2.0;
                for (std::size_t c = 0; c < ecps.size(); ++c)
                {
                    engine.compute_shell_pair_derivative(ecps[c], shells[a], shells[b], d);
                    const int atom[3] = {shell_atom[a], shell_atom[b],
                                         static_cast<int>(basis._ecps[c]._atom_index)};
                    for (int k = 0; k < 9; ++k)
                    {
                        double s = 0.0;
                        for (int i = 0; i < na; ++i)
                            for (int j = 0; j < nb; ++j)
                            {
                                const std::size_t mu = first[a] + i;
                                const std::size_t nu = first[b] + j;
                                s += D(mu, nu) * d[k](i, j) *
                                     basis._basis_functions[mu]._component_norm *
                                     basis._basis_functions[nu]._component_norm;
                            }
                        bins[bin](atom[k / 3], k % 3) += w * s;
                    }
                }
            }
    }
    for (const Eigen::MatrixXd &b : bins)
        grad += b;
    return grad;
}

std::expected<std::string, std::string> HartreeFock::ECP::describe_ecps(
    const HartreeFock::Basis &basis, const HartreeFock::Molecule &molecule, int deriv)
{
    if (basis._ecps.empty())
        return std::string{};
    int max_lb = 0;
    for (const HartreeFock::Shell &sh : basis._shells)
        max_lb = std::max(max_lb, static_cast<int>(sh._shell));
    std::string msg;
    for (const HartreeFock::ECPPotential &ecp : basis._ecps)
    {
        const int L = static_cast<int>(ecp._semilocal.size());
        // libecpint only asserts these limits, and its asserts are compiled out in Release.
        if (max_lb + deriv > LIBECPINT_MAX_L || L > LIBECPINT_MAX_L)
            return std::unexpected(std::format(
                "ECP integrals: basis L {} (+{} for derivatives) / ECP L {} exceed libecpint's "
                "LIBECPINT_MAX_L {}",
                max_lb, deriv, L, LIBECPINT_MAX_L));
        auto element = element_from_z(molecule.atomic_numbers[ecp._atom_index]);
        std::string terms = std::to_string(ecp._local.size());
        for (const auto &channel : ecp._semilocal)
            terms += "/" + std::to_string(channel.size());
        msg += std::format("\n  atom {} {}: {} core electrons, Z_eff {}, L {}, terms {}",
                           ecp._atom_index + 1, element ? element->symbol : "?", ecp._n_core,
                           molecule.nuclear_charge(ecp._atom_index), L, terms);
    }
    return msg;
}
