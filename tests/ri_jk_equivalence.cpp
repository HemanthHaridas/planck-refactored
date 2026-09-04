// RI-JK builder harness (Step J0).
//
// Stands up the fixture the RI Coulomb/exchange steps (J1/J3) check against:
//   - a Calculator with the RI cache ready (metric + 3-center) on water/cc-pVDZ,
//   - a fixed symmetric test density D,
//   - the dense reference G = J - 1/2 K from ObaraSaika::_compute_fock_rhf(eri,D).
//
// At J0 there is no RI J/K code yet, so this only exercises the oracle: it
// asserts the dense G is symmetric (it must be, for a symmetric D and the 8-fold
// ERI symmetry). J1 adds the RI-Coulomb vs dense-J assertion; J3 adds the full
// RI-G vs dense-G assertion. Keeping the fixture in one place means those steps
// are a few added lines here, each independently revertible.

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "base/types.h"
#include "basis/basis.h"
#include "basis/rifit.h"
#include "integrals/base.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"
#include "post_hf/ri/ri_eri.h"

namespace
{
    bool g_ok = true;
    void fail(const std::string &m)
    {
        std::cerr << "FAIL: " << m << '\n';
        g_ok = false;
    }

    std::filesystem::path repo_root()
    {
        if (const char *env = std::getenv("BASIS_PATH"); env && *env)
            return std::filesystem::path(env).parent_path();
        return std::filesystem::current_path();
    }
}

int main()
{
    using HartreeFock::BasisFunctions::read_gbs_basis;
    using HartreeFock::BasisFunctions::read_ri_basis;
    using HartreeFock::Correlation::RI::ensure_ri_3c_ready;
    using HartreeFock::Correlation::RI::ensure_ri_metric_ready;

    const auto root = repo_root();
    const auto basis_file = root / "basis-sets" / "cc-pVDZ";
    const auto aux_file = root / "basis-sets" / "cc-pVDZ-RIFIT";

    HartreeFock::Molecule mol;
    mol.natoms = 3;
    mol.atomic_numbers.resize(3);
    mol.atomic_numbers << 8, 1, 1;
    mol._standard.resize(3, 3);
    mol._standard << 0.0, 0.0, 0.0,
        0.0, 1.43, 1.11,
        0.0, -1.43, 1.11;
    mol._standard_is_bohr = true;

    HartreeFock::Calculator calc;
    calc._molecule = mol;
    calc._basis._basis_name = "cc-pVDZ";
    calc._basis._basis_path = (root / "basis-sets").string();
    calc._integral._engine = HartreeFock::IntegralMethod::HeadGordonPople;
    calc._mp2.use_ri = true;
    calc._mp2.ri_basis_name = "cc-pVDZ-RIFIT";
    calc._mp2.ri_basis_path = (root / "basis-sets").string();
    calc._mp2.ri_lindep = 1e-7;

    auto basis_res = read_gbs_basis(basis_file.string(), mol, HartreeFock::BasisType::Cartesian);
    if (!basis_res)
        fail("read_gbs_basis failed: " + basis_res.error());
    else
        calc._shells = std::move(*basis_res);

    auto aux_res = read_ri_basis(aux_file.string(), mol);
    if (!aux_res)
        fail("read_ri_basis failed: " + aux_res.error());
    else
        calc._ri_aux_basis = std::make_shared<HartreeFock::AuxBasis>(std::move(*aux_res));

    if (g_ok)
    {
        auto prep = ensure_ri_metric_ready(calc);
        if (!prep)
            fail("ensure_ri_metric_ready failed: " + prep.error());
    }
    if (g_ok)
    {
        auto prep = ensure_ri_3c_ready(calc);
        if (!prep)
            fail("ensure_ri_3c_ready failed: " + prep.error());
    }
    if (!g_ok)
        return 1;

    const std::size_t nb = calc._shells.nbasis();

    // Dense AO ERI tensor for the reference Fock build.
    const auto shell_pairs = build_shellpairs(calc._shells);
    const std::vector<double> dense_eri =
        _compute_2e(shell_pairs, nb, calc._integral._engine,
                    HartreeFock::ERIKernel::Coulomb, 0.0, 1e-12, nullptr);

    // A fixed, symmetric, deterministic test density. Not a real SCF density —
    // only symmetry and reproducibility matter for the equivalence checks.
    Eigen::MatrixXd D(nb, nb);
    for (std::size_t mu = 0; mu < nb; ++mu)
        for (std::size_t nu = 0; nu < nb; ++nu)
            D(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) =
                std::cos(0.4 * static_cast<double>(mu) - 0.6 * static_cast<double>(nu));
    D = 0.5 * (D + D.transpose()).eval(); // symmetrize

    // Dense oracle: G = J - 1/2 K.
    const Eigen::MatrixXd G_dense =
        HartreeFock::ObaraSaika::_compute_fock_rhf(dense_eri, D, nb);

    // J0 assertion: the oracle is symmetric for a symmetric density.
    const double asym = (G_dense - G_dense.transpose()).cwiseAbs().maxCoeff();
    std::cout << "dense G: nb=" << nb << "  ‖G‖_max=" << G_dense.cwiseAbs().maxCoeff()
              << "  max|G-Gᵀ|=" << asym << '\n';
    if (asym > 1e-10)
        fail("dense reference Fock is not symmetric for a symmetric density");

    // J1: RI Coulomb vs dense J. Build the dense J alone (not J - 1/2 K) so we
    // isolate the Coulomb half: J_{μν} = Σ_{λσ} (μν|λσ) D_{λσ}.
    Eigen::MatrixXd J_dense = Eigen::MatrixXd::Zero(nb, nb);
    {
        const std::size_t nb2 = nb * nb, nb3 = nb * nb * nb;
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
            {
                double v = 0.0;
                for (std::size_t lam = 0; lam < nb; ++lam)
                    for (std::size_t sig = 0; sig < nb; ++sig)
                        v += D(static_cast<Eigen::Index>(lam), static_cast<Eigen::Index>(sig)) *
                             dense_eri[mu * nb3 + nu * nb2 + lam * nb + sig];
                J_dense(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) = v;
            }
    }

    const Eigen::MatrixXd J_ri = HartreeFock::Correlation::RI::build_ri_j(calc, D);
    const double j_rel =
        (J_ri - J_dense).norm() / std::max(J_dense.norm(), 1e-300);
    std::cout << "J: ‖RI-dense‖/‖dense‖=" << j_rel
              << "  max|Δ|=" << (J_ri - J_dense).cwiseAbs().maxCoeff() << '\n';
    if (j_rel > 2e-2)
        fail("RI Coulomb J disagrees with dense beyond fitting accuracy (>2e-2) "
             "— indicates a packing/charge-accumulation bug, not fitting error");

    // J2: the unpacked B[Q](μ,ν) must round-trip the packed pair factors it
    // came from, and be symmetric in (μ,ν).
    {
        const Eigen::MatrixXd pf =
            HartreeFock::Correlation::RI::build_ri_pair_factors(calc);
        const auto B = HartreeFock::Correlation::RI::build_ri_3index_unpacked(calc);
        const Eigen::Index naux = pf.cols();
        double max_rt = 0.0, max_asym = 0.0;
        std::size_t pair_row = 0;
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu <= mu; ++nu, ++pair_row)
                for (Eigen::Index Q = 0; Q < naux; ++Q)
                {
                    const double packed = pf(static_cast<Eigen::Index>(pair_row), Q);
                    const auto &BQ = B[static_cast<std::size_t>(Q)];
                    max_rt = std::max(max_rt,
                                      std::abs(BQ(static_cast<Eigen::Index>(mu),
                                                  static_cast<Eigen::Index>(nu)) -
                                               packed));
                    max_asym = std::max(max_asym,
                                        std::abs(BQ(static_cast<Eigen::Index>(mu),
                                                    static_cast<Eigen::Index>(nu)) -
                                                 BQ(static_cast<Eigen::Index>(nu),
                                                    static_cast<Eigen::Index>(mu))));
                }
        std::cout << "unpacked B: naux=" << naux
                  << "  max|round-trip|=" << max_rt
                  << "  max|B-Bᵀ|=" << max_asym << '\n';
        if (max_rt > 1e-14)
            fail("unpacked B does not match the packed pair factors");
        if (max_asym > 1e-14)
            fail("unpacked B is not symmetric in (μ,ν)");
    }

    // J3: full RI Fock G = J - 1/2 K vs the dense _compute_fock_rhf oracle.
    // This is the load-bearing gate for the whole RI-JK builder.
    const Eigen::MatrixXd G_ri =
        HartreeFock::Correlation::RI::build_ri_fock_rhf(calc, D);
    const double g_rel =
        (G_ri - G_dense).norm() / std::max(G_dense.norm(), 1e-300);
    const double g_asym = (G_ri - G_ri.transpose()).cwiseAbs().maxCoeff();
    std::cout << "G=J-½K: ‖RI-dense‖/‖dense‖=" << g_rel
              << "  max|Δ|=" << (G_ri - G_dense).cwiseAbs().maxCoeff()
              << "  max|G-Gᵀ|=" << g_asym << '\n';
    if (g_rel > 2e-2)
        fail("full RI Fock G disagrees with dense beyond fitting accuracy "
             "(>2e-2) — indicates an exchange-contraction bug, not fitting error");
    if (g_asym > 1e-10)
        fail("RI Fock G is not symmetric for a symmetric density");

    // J4 (Step RG4.1): unrestricted RI Fock {Ga, Gb} = {J(Pa+Pb) - K(Pa),
    // J(Pa+Pb) - K(Pb)} vs the dense _compute_fock_uhf oracle. Pa and Pb are
    // deliberately DIFFERENT — with Pa == Pb a wrong exchange factor (e.g. the
    // closed-shell ½ carried over) can still land close, and the spin-resolved
    // structure goes untested.
    {
        Eigen::MatrixXd Pa(nb, nb), Pb(nb, nb);
        for (std::size_t mu = 0; mu < nb; ++mu)
            for (std::size_t nu = 0; nu < nb; ++nu)
            {
                Pa(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) =
                    std::sin(0.4 * mu + 0.7 * nu + 0.3);
                Pb(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) =
                    std::cos(0.6 * mu - 0.2 * nu + 1.1);
            }
        Pa = (0.5 * (Pa + Pa.transpose())).eval();
        Pb = (0.5 * (Pb + Pb.transpose())).eval();

        const auto [Ga_dense, Gb_dense] =
            HartreeFock::ObaraSaika::_compute_fock_uhf(dense_eri, Pa, Pb, nb);
        const auto [Ga_ri, Gb_ri] =
            HartreeFock::Correlation::RI::build_ri_fock_uhf(calc, Pa, Pb);

        const double a_rel =
            (Ga_ri - Ga_dense).norm() / std::max(Ga_dense.norm(), 1e-300);
        const double b_rel =
            (Gb_ri - Gb_dense).norm() / std::max(Gb_dense.norm(), 1e-300);
        std::cout << "UHF G: ‖RI-dense‖/‖dense‖  alpha=" << a_rel
                  << "  beta=" << b_rel << '\n';
        if (a_rel > 2e-2 || b_rel > 2e-2)
            fail("RI UHF Fock disagrees with dense beyond fitting accuracy "
                 "(>2e-2) — wrong Coulomb/exchange split or a stray ½ on K");

        // Sanity: with Pa == Pb == D/2 the UHF form must reduce to the RHF one,
        // G_sigma = J(D) - K(D/2) = J(D) - ½K(D). Pins the missing-½ convention.
        const auto [Gh_a, Gh_b] =
            HartreeFock::Correlation::RI::build_ri_fock_uhf(calc, 0.5 * D, 0.5 * D);
        const double closed_shell_rel =
            (Gh_a - G_ri).norm() / std::max(G_ri.norm(), 1e-300);
        std::cout << "UHF G(D/2,D/2) vs RHF G(D): rel=" << closed_shell_rel << '\n';
        if (closed_shell_rel > 1e-12)
            fail("RI UHF Fock at Pa=Pb=D/2 does not reduce to the RHF J-½K form");
        if ((Gh_a - Gh_b).cwiseAbs().maxCoeff() > 1e-12)
            fail("RI UHF Fock gives different alpha/beta for Pa == Pb");
    }

    if (g_ok)
        std::cout << "PASS: ri_jk_equivalence (J0 fixture)\n";
    return g_ok ? 0 : 1;
}
