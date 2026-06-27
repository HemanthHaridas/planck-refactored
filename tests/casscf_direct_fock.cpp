// Phase A0 gate: the direct (tensor-free) MCSCF Fock builders
// (build_inactive_fock_mo_direct / build_active_fock_mo_direct) reproduce the
// tensor-contracting builders (build_inactive_fock_mo / build_active_fock_mo)
// to ‖F_direct − F_tensor‖ < 1e-12.
//
// Both compute the same operator — the closed-shell J − ½K driven by a core
// (or active-1-RDM) density, transformed into the current MO basis. The tensor
// version contracts the full materialized n_AO⁴ ERI tensor via
// ObaraSaika::_compute_fock_rhf; the direct version builds the same AO Fock from
// shell-pair ERIs via the screened, engine-dispatched direct kernel
// (HeadGordonPople), with no dense buffer. This test pins the equivalence A1/A2
// rely on before routing the production builders through the direct path.
//
// Exercised on water/6-31g* (d-shells) with several MO bases C and a couple of
// active 1-RDMs, so the comparison covers non-trivial core + active densities.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/basis.h"
#include "base/types.h"
#include "basis/basis.h"
#include "integrals/base.h"
#include "integrals/shellpair.h"
#include "post_hf/casscf/orbital.h"

namespace
{
    using namespace HartreeFock::Correlation::CASSCF;

    bool g_ok = true;

    void check(const std::string &what, double diff, double tol = 1e-12)
    {
        if (!(diff < tol) || !std::isfinite(diff))
        {
            std::cerr << "FAIL " << what << ": ‖F_direct − F_tensor‖ = " << diff
                      << " (tol " << tol << ")\n";
            g_ok = false;
        }
        else
        {
            std::cout << "OK   " << what << ": ‖Δ‖ = " << diff << '\n';
        }
    }

    std::expected<HartreeFock::Calculator, std::string> make_water(const std::string &basis_name)
    {
        HartreeFock::Calculator calc;
        HartreeFock::Molecule &mol = calc._molecule;
        mol.natoms = 3;
        mol.charge = 0;
        mol.multiplicity = 1;
        mol.atomic_numbers.resize(3);
        mol.atomic_numbers << 8, 1, 1;
        mol.atomic_masses = Eigen::VectorXd::Ones(3);
        Eigen::MatrixXd c(3, 3);
        c << 0.000000, 0.000000, 0.117176,
            0.000000, 0.757200, -0.468704,
            0.000000, -0.757200, -0.468704;
        mol.coordinates = c;

        calc._basis._basis = HartreeFock::BasisType::Cartesian;
        calc.prepare_coordinates();
        mol.set_standard_from_bohr(mol._coordinates);

        const std::filesystem::path gbs =
            std::filesystem::path(get_basis_path()) / basis_name;
        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            gbs.string(), mol, calc._basis._basis);
        if (!basis_res)
            return std::unexpected("read_gbs_basis failed: " + basis_res.error());
        calc._shells = std::move(*basis_res);
        return calc;
    }

    // A deterministic "MO-like" coefficient matrix: orthonormalize a fixed
    // pseudo-random nb×nb matrix. The Fock builders do not require C to be the
    // SCF MOs — only that the densities they form are valid — so any well-
    // conditioned C exercises the contraction equally. `seed` varies the basis.
    Eigen::MatrixXd make_C(int nb, unsigned seed)
    {
        Eigen::MatrixXd M(nb, nb);
        unsigned s = seed * 2654435761u + 1u;
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j)
            {
                s = s * 1103515245u + 12345u;
                M(i, j) = (static_cast<double>((s >> 9) & 0x7fffff) / 8388608.0) - 0.5;
            }
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
        return qr.householderQ();
    }
}

int main()
{
    auto water = make_water("6-31g*");
    if (!water)
    {
        std::cerr << water.error() << '\n';
        return 1;
    }
    HartreeFock::Calculator &calc = *water;
    const int nb = static_cast<int>(calc._shells._basis_functions.size());
    const std::vector<HartreeFock::ShellPair> shell_pairs =
        build_shellpairs(calc._shells);

    // Full AO ERI tensor (the tensor-path input), built at a tight screening tol
    // so the comparison isolates the contraction *algebra*, not screening (the
    // screening tolerance trade-off is A3's concern, not A0's). The direct
    // builders below use tol_eri=1e-14 to match.
    const double tol = 1e-14;
    const std::vector<double> eri = HartreeFock::ObaraSaika::_compute_2e(
        shell_pairs, static_cast<std::size_t>(nb),
        HartreeFock::ERIKernel::Coulomb, 0.0, tol);

    // H_core: any symmetric matrix works for the inactive-Fock comparison (it is
    // added identically on both sides and then C^T(...)C). Use the calculator's
    // if present, else identity — the J−½K contribution is what differs.
    Eigen::MatrixXd H_core = calc._hcore.size() == nb * nb
                                 ? calc._hcore
                                 : Eigen::MatrixXd::Identity(nb, nb);

    const int n_core = 4; // water-like closed core size for the test
    const int n_act = 4;

    for (unsigned seed : {1u, 7u, 42u})
    {
        const Eigen::MatrixXd C = make_C(nb, seed);

        const Eigen::MatrixXd Fi_tensor =
            build_inactive_fock_mo(C, H_core, eri, n_core, nb);
        const Eigen::MatrixXd Fi_direct =
            build_inactive_fock_mo_direct(C, H_core, shell_pairs, n_core, nb,
                                          HartreeFock::IntegralMethod::HeadGordonPople, tol);
        check("inactive seed=" + std::to_string(seed),
              (Fi_direct - Fi_tensor).cwiseAbs().maxCoeff());

        // A symmetric positive active 1-RDM (gamma): identity is a valid density.
        Eigen::MatrixXd gamma = Eigen::MatrixXd::Identity(n_act, n_act);
        gamma(0, 1) = gamma(1, 0) = 0.3; // off-diagonal to exercise mixing
        const Eigen::MatrixXd Fa_tensor =
            build_active_fock_mo(C, gamma, eri, n_core, n_act, nb);
        const Eigen::MatrixXd Fa_direct =
            build_active_fock_mo_direct(C, gamma, shell_pairs, n_core, n_act, nb,
                                        HartreeFock::IntegralMethod::HeadGordonPople, tol);
        check("active   seed=" + std::to_string(seed),
              (Fa_direct - Fa_tensor).cwiseAbs().maxCoeff());
    }

    if (!g_ok)
    {
        std::cerr << "planck-casscf-direct-fock: FAIL\n";
        return 1;
    }
    std::cout << "planck-casscf-direct-fock: OK\n";
    return 0;
}
