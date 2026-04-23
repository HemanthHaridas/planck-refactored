#include <cmath>
#include <iostream>
#include <sstream>

#include "base/types.h"
#include "integrals/os.h"
#include "integrals/shellpair.h"

namespace
{
    bool g_ok = true;

    void require_near(double actual, double expected, double tol, const std::string &message)
    {
        if (std::abs(actual - expected) > tol)
        {
            std::ostringstream oss;
            oss << message << ": expected " << expected << ", got " << actual
                << " (tol " << tol << ")";
            std::cerr << oss.str() << '\n';
            g_ok = false;
        }
    }

    double primitive_s_normalization(double exponent)
    {
        return std::pow(2.0 * exponent / M_PI, 0.75);
    }

    HartreeFock::Basis make_single_s_basis(const Eigen::Vector3d &center, double exponent)
    {
        HartreeFock::Basis basis;
        basis._shells.emplace_back();

        auto &shell = basis._shells.back();
        shell._center = center;
        shell._shell = HartreeFock::ShellType::S;
        shell._atom_index = 0;
        shell._primitives = Eigen::VectorXd::Constant(1, exponent);
        shell._coefficients = Eigen::VectorXd::Ones(1);
        shell._normalizations = Eigen::VectorXd::Constant(1, primitive_s_normalization(exponent));

        basis._basis_functions.emplace_back();
        auto &basis_function = basis._basis_functions.back();
        basis_function._shell = &shell;
        basis_function._index = 0;
        basis_function._component_norm = 1.0;
        basis_function._cartesian = Eigen::Vector3i::Zero();

        return basis;
    }

    void test_centered_s_orbital()
    {
        const auto basis = make_single_s_basis(Eigen::Vector3d::Zero(), 1.0);
        const auto shell_pairs = build_shellpairs(basis);
        const auto matrices = HartreeFock::ObaraSaika::_compute_multipole_matrices(
            shell_pairs,
            basis.nbasis(),
            Eigen::Vector3d::Zero());

        require_near(matrices.dipole[0](0, 0), 0.0, 1e-12, "Centered s orbital should have zero x dipole");
        require_near(matrices.dipole[1](0, 0), 0.0, 1e-12, "Centered s orbital should have zero y dipole");
        require_near(matrices.dipole[2](0, 0), 0.0, 1e-12, "Centered s orbital should have zero z dipole");

        require_near(matrices.quadrupole[0](0, 0), 0.25, 1e-12, "Centered s orbital x^2 moment mismatch");
        require_near(matrices.quadrupole[3](0, 0), 0.25, 1e-12, "Centered s orbital y^2 moment mismatch");
        require_near(matrices.quadrupole[5](0, 0), 0.25, 1e-12, "Centered s orbital z^2 moment mismatch");
        require_near(matrices.quadrupole[1](0, 0), 0.0, 1e-12, "Centered s orbital xy moment mismatch");
        require_near(matrices.quadrupole[2](0, 0), 0.0, 1e-12, "Centered s orbital xz moment mismatch");
        require_near(matrices.quadrupole[4](0, 0), 0.0, 1e-12, "Centered s orbital yz moment mismatch");
    }

    void test_shifted_s_orbital()
    {
        const Eigen::Vector3d center(1.0, 0.0, 0.0);
        const auto basis = make_single_s_basis(center, 1.0);
        const auto shell_pairs = build_shellpairs(basis);
        const auto matrices = HartreeFock::ObaraSaika::_compute_multipole_matrices(
            shell_pairs,
            basis.nbasis(),
            Eigen::Vector3d::Zero());

        require_near(matrices.dipole[0](0, 0), 1.0, 1e-12, "Shifted s orbital x dipole mismatch");
        require_near(matrices.dipole[1](0, 0), 0.0, 1e-12, "Shifted s orbital y dipole should vanish");
        require_near(matrices.dipole[2](0, 0), 0.0, 1e-12, "Shifted s orbital z dipole should vanish");

        require_near(matrices.quadrupole[0](0, 0), 1.25, 1e-12, "Shifted s orbital x^2 moment mismatch");
        require_near(matrices.quadrupole[3](0, 0), 0.25, 1e-12, "Shifted s orbital y^2 moment mismatch");
        require_near(matrices.quadrupole[5](0, 0), 0.25, 1e-12, "Shifted s orbital z^2 moment mismatch");
    }

} // namespace

int main()
{
    test_centered_s_orbital();
    test_shifted_s_orbital();
    return g_ok ? 0 : 1;
}
