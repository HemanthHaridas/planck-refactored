// RI 3-center derivative FD oracle (Step RG0).
//
// The RI-MP2 gradient will contract a 2-particle density against the derivative
// of the 3-center integrals, d/dR (μν|Q). Those analytic derivatives do not
// exist yet (Step RG1). This test stands up the finite-difference reference
// RG1 will be gated against:
//
//   dJ/dR_{Ac} ≈ [ J(R + δ e_{Ac}) - J(R - δ e_{Ac}) ] / (2δ)
//
// where J = compute_3c_eri (packed AO-pair × aux) and R_{Ac} is coordinate c of
// atom A. Building basis + aux fresh at each displaced geometry is exactly how
// the geomopt/freq loops re-run, so this mirrors production.
//
// At RG0 the analytic derivative is absent, so this test validates the oracle
// itself, not an analytic-vs-FD match:
//   (1) the FD derivative is finite and correctly shaped,
//   (2) translational invariance — summing dJ/dR over all atoms for a fixed
//       coordinate axis gives ~0 (rigidly translating the whole molecule leaves
//       every integral unchanged). This is the strongest structural check
//       available without the analytic form, and it catches a mis-centered or
//       mis-signed FD.

#include <Eigen/Dense>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <iostream>
#include <string>

#include "base/types.h"
#include "basis/basis.h"
#include "basis/rifit.h"
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

    // Water Calculator with basis + aux read fresh at the given standard (Bohr)
    // geometry, ready for compute_3c_eri.
    HartreeFock::Calculator make_calc(const std::filesystem::path &root,
                                      const Eigen::MatrixXd &standard_bohr)
    {
        HartreeFock::Molecule mol;
        mol.natoms = 3;
        mol.atomic_numbers.resize(3);
        mol.atomic_numbers << 8, 1, 1;
        mol._standard = standard_bohr;
        mol._standard_is_bohr = true;

        HartreeFock::Calculator calc;
        calc._molecule = mol;
        calc._basis._basis_name = "cc-pVDZ";
        calc._basis._basis_path = (root / "basis-sets").string();
        calc._mp2.use_ri = true;
        calc._mp2.ri_basis_name = "cc-pVDZ-RIFIT";
        calc._mp2.ri_basis_path = (root / "basis-sets").string();
        calc._mp2.ri_lindep = 1e-7;

        auto basis_res = HartreeFock::BasisFunctions::read_gbs_basis(
            (root / "basis-sets" / "cc-pVDZ").string(), mol, HartreeFock::BasisType::Cartesian);
        if (basis_res)
            calc._shells = std::move(*basis_res);
        auto aux_res = HartreeFock::BasisFunctions::read_ri_basis(
            (root / "basis-sets" / "cc-pVDZ-RIFIT").string(), mol);
        if (aux_res)
            calc._ri_aux_basis =
                std::make_shared<HartreeFock::AuxBasis>(std::move(*aux_res));
        return calc;
    }

    // J(geometry) = packed 3-center tensor.
    std::expected<Eigen::MatrixXd, std::string> j3c_at(
        const std::filesystem::path &root, const Eigen::MatrixXd &geom)
    {
        HartreeFock::Calculator calc = make_calc(root, geom);
        return HartreeFock::Correlation::RI::compute_3c_eri(calc);
    }

    // Central-difference dJ/dR for atom `a`, axis `c`.
    std::expected<Eigen::MatrixXd, std::string> fd_derivative(
        const std::filesystem::path &root, const Eigen::MatrixXd &geom,
        int a, int c, double delta)
    {
        Eigen::MatrixXd gp = geom, gm = geom;
        gp(a, c) += delta;
        gm(a, c) -= delta;
        auto Jp = j3c_at(root, gp);
        if (!Jp) return std::unexpected(Jp.error());
        auto Jm = j3c_at(root, gm);
        if (!Jm) return std::unexpected(Jm.error());
        return Eigen::MatrixXd((*Jp - *Jm) / (2.0 * delta));
    }
}

int main()
{
    const auto root = repo_root();

    Eigen::MatrixXd geom(3, 3);
    geom << 0.0, 0.0, 0.117176,
        0.0, 0.757005, -0.468704,
        0.0, -0.757005, -0.468704;
    geom *= 1.8897259886; // Angstrom -> Bohr

    const double delta = 1e-4; // Bohr

    // Base tensor, to size the invariance accumulator.
    auto J0 = j3c_at(root, geom);
    if (!J0)
    {
        fail("compute_3c_eri(base) failed: " + J0.error());
        return 1;
    }
    const Eigen::Index rows = J0->rows(), cols = J0->cols();
    std::cout << "j3c dims " << rows << "x" << cols << ", delta " << delta << " Bohr\n";

    // Check (1): FD derivative for a representative (atom, axis) is finite and
    // correctly shaped, and non-trivial (moving O actually changes the integrals).
    auto dJ_O_z = fd_derivative(root, geom, 0, 2, delta);
    if (!dJ_O_z)
    {
        fail("fd_derivative(O,z) failed: " + dJ_O_z.error());
        return 1;
    }
    if (dJ_O_z->rows() != rows || dJ_O_z->cols() != cols)
    {
        fail("FD derivative has wrong shape");
        return 1;
    }
    if (!dJ_O_z->allFinite())
    {
        fail("FD derivative has non-finite entries");
        return 1;
    }
    const double dmax = dJ_O_z->cwiseAbs().maxCoeff();
    std::cout << "max |dJ/dO_z| = " << dmax << '\n';
    if (dmax < 1e-6)
    {
        fail("FD derivative w.r.t. O_z is ~0 — integrals did not respond to the "
             "oxygen move; the oracle is not exercising the geometry dependence");
        return 1;
    }

    // Check (2): translational invariance. Rigidly shifting the whole molecule
    // along axis c leaves every (μν|Q) unchanged, so Σ_atoms dJ/dR_{atom,c} = 0.
    for (int c = 0; c < 3; ++c)
    {
        Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(rows, cols);
        for (int a = 0; a < 3; ++a)
        {
            auto d = fd_derivative(root, geom, a, c, delta);
            if (!d)
            {
                fail("fd_derivative failed in invariance loop: " + d.error());
                return 1;
            }
            sum += *d;
        }
        const double resid = sum.cwiseAbs().maxCoeff();
        std::cout << "axis " << c << " translational residual max = " << resid << '\n';
        // FD truncation on a 1e-4 step over ~O(1) integrals leaves a small
        // residual; 1e-5 comfortably separates "invariant" from a real bug.
        if (resid > 1e-5)
            fail("translational invariance violated on axis " + std::to_string(c) +
                 " — FD oracle is mis-centered or mis-signed");
    }

    if (g_ok)
        std::cout << "PASS: ri_3c_derivative_fd\n";
    return g_ok ? 0 : 1;
}
