#include "populations/multipole.h"

#include <array>

#include "integrals/os.h"

namespace
{
    std::array<double, 3> os_1d_moments(
        const double gamma,
        const double distPA,
        const double distPB,
        const double distPC,
        const int lA,
        const int lB)
    {
        // This is the 1D Obara-Saika recurrence specialized for the zeroth,
        // first, and second Cartesian moments needed by dipole/quadrupole
        // integrals. Returning all three orders together keeps the shell-pair
        // loop below compact and avoids recomputing the same recurrence tables
        // three separate times for x, y, and z.
        double M[MAX_L + 1][MAX_L + 1][3] = {};

        M[0][0][0] = 1.0;
        M[0][0][1] = distPC;
        M[0][0][2] = distPC * distPC + gamma;

        for (int i = 1; i <= lA; ++i)
        {
            for (int n = 0; n <= 2; ++n)
            {
                M[i][0][n] = distPA * M[i - 1][0][n];
                if (i > 1)
                    M[i][0][n] += (i - 1) * gamma * M[i - 2][0][n];
                if (n > 0)
                    M[i][0][n] += n * gamma * M[i - 1][0][n - 1];
            }
        }

        for (int j = 1; j <= lB; ++j)
        {
            for (int n = 0; n <= 2; ++n)
            {
                M[0][j][n] = distPB * M[0][j - 1][n];
                if (j > 1)
                    M[0][j][n] += (j - 1) * gamma * M[0][j - 2][n];
                if (n > 0)
                    M[0][j][n] += n * gamma * M[0][j - 1][n - 1];
            }
        }

        for (int i = 1; i <= lA; ++i)
        {
            for (int j = 1; j <= lB; ++j)
            {
                for (int n = 0; n <= 2; ++n)
                {
                    M[i][j][n] = distPA * M[i - 1][j][n];
                    if (i > 1)
                        M[i][j][n] += (i - 1) * gamma * M[i - 2][j][n];
                    M[i][j][n] += j * gamma * M[i - 1][j - 1][n];
                    if (n > 0)
                        M[i][j][n] += n * gamma * M[i - 1][j][n - 1];
                }
            }
        }

        return {M[lA][lB][0], M[lA][lB][1], M[lA][lB][2]};
    }
} // namespace

HartreeFock::MultipoleMatrices HartreeFock::ObaraSaika::_compute_multipole_matrices(
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    std::size_t nbasis,
    const Eigen::Vector3d &origin)
{
    // The multipole API mirrors the one-electron integral builders: we assemble
    // full AO matrices over the shell-pair list, but here each pair produces the
    // three dipole components and six unique quadrupole components referenced to
    // the caller-supplied origin.
    HartreeFock::MultipoleMatrices matrices{};
    for (auto &component : matrices.dipole)
        component = Eigen::MatrixXd::Zero(nbasis, nbasis);
    for (auto &component : matrices.quadrupole)
        component = Eigen::MatrixXd::Zero(nbasis, nbasis);

    const std::size_t npairs = shell_pairs.size();

#pragma omp parallel for schedule(dynamic)
    for (std::size_t p = 0; p < npairs; ++p)
    {
        const auto &sp = shell_pairs[p];
        const std::size_t ii = sp.A._index;
        const std::size_t jj = sp.B._index;

        const auto &cartA = sp.A._cartesian;
        const auto &cartB = sp.B._cartesian;

        double dipole_x = 0.0;
        double dipole_y = 0.0;
        double dipole_z = 0.0;
        double quad_xx = 0.0;
        double quad_xy = 0.0;
        double quad_xz = 0.0;
        double quad_yy = 0.0;
        double quad_yz = 0.0;
        double quad_zz = 0.0;

        for (const auto &pp : sp.primitive_pairs)
        {
            const double gamma = 0.5 * pp.inv_zeta;
            const double scale = pp.prefactor * pp.coeff_product;

            // Each Cartesian direction is separable for Gaussian moments, so the
            // 3D multipole integral becomes products of 1D moment recurrences.
            // `pp.center - origin` shifts the primitive-product center into the
            // requested multipole frame before the contractions are accumulated.
            const auto mx = os_1d_moments(
                gamma,
                pp.pA[0],
                pp.pB[0],
                pp.center[0] - origin[0],
                cartA[0],
                cartB[0]);
            const auto my = os_1d_moments(
                gamma,
                pp.pA[1],
                pp.pB[1],
                pp.center[1] - origin[1],
                cartA[1],
                cartB[1]);
            const auto mz = os_1d_moments(
                gamma,
                pp.pA[2],
                pp.pB[2],
                pp.center[2] - origin[2],
                cartA[2],
                cartB[2]);

            dipole_x += mx[1] * my[0] * mz[0] * scale;
            dipole_y += mx[0] * my[1] * mz[0] * scale;
            dipole_z += mx[0] * my[0] * mz[1] * scale;

            quad_xx += mx[2] * my[0] * mz[0] * scale;
            quad_xy += mx[1] * my[1] * mz[0] * scale;
            quad_xz += mx[1] * my[0] * mz[1] * scale;
            quad_yy += mx[0] * my[2] * mz[0] * scale;
            quad_yz += mx[0] * my[1] * mz[1] * scale;
            quad_zz += mx[0] * my[0] * mz[2] * scale;
        }

        // The real AO basis used here yields Hermitian one-electron multipole
        // matrices, so the shell-pair contribution can be mirrored directly.
        matrices.dipole[0](ii, jj) = dipole_x;
        matrices.dipole[1](ii, jj) = dipole_y;
        matrices.dipole[2](ii, jj) = dipole_z;
        matrices.dipole[0](jj, ii) = dipole_x;
        matrices.dipole[1](jj, ii) = dipole_y;
        matrices.dipole[2](jj, ii) = dipole_z;

        matrices.quadrupole[0](ii, jj) = quad_xx;
        matrices.quadrupole[1](ii, jj) = quad_xy;
        matrices.quadrupole[2](ii, jj) = quad_xz;
        matrices.quadrupole[3](ii, jj) = quad_yy;
        matrices.quadrupole[4](ii, jj) = quad_yz;
        matrices.quadrupole[5](ii, jj) = quad_zz;
        matrices.quadrupole[0](jj, ii) = quad_xx;
        matrices.quadrupole[1](jj, ii) = quad_xy;
        matrices.quadrupole[2](jj, ii) = quad_xz;
        matrices.quadrupole[3](jj, ii) = quad_yy;
        matrices.quadrupole[4](jj, ii) = quad_yz;
        matrices.quadrupole[5](jj, ii) = quad_zz;
    }

    return matrices;
}

std::expected<HartreeFock::MultipoleMoments, std::string>
HartreeFock::ObaraSaika::_compute_multipole_moments(
    const HartreeFock::Calculator &calculator,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const Eigen::Vector3d &origin)
{
    // The reporting path wants physical moments, not raw AO integrals, so this
    // routine contracts the AO multipole matrices with the converged density and
    // then adds the nuclear point-charge contribution about the same origin.
    const std::size_t nbasis = calculator._shells.nbasis();
    const Eigen::Index nbasis_idx = static_cast<Eigen::Index>(nbasis);
    const auto &alpha_density = calculator._info._scf.alpha.density;

    if (alpha_density.rows() != nbasis_idx || alpha_density.cols() != nbasis_idx)
        return std::unexpected("alpha density matrix is not initialized for multipole analysis");

    Eigen::MatrixXd total_density = alpha_density;
    if (calculator._info._scf.is_uhf)
    {
        // Multipole moments depend on the total charge density, so unrestricted
        // calculations simply add alpha and beta channels before the AO
        // contraction.
        const auto &beta_density = calculator._info._scf.beta.density;
        if (beta_density.rows() != nbasis_idx || beta_density.cols() != nbasis_idx)
            return std::unexpected("beta density matrix is not initialized for multipole analysis");
        total_density += beta_density;
    }

    const HartreeFock::MultipoleMatrices matrices =
        _compute_multipole_matrices(shell_pairs, nbasis, origin);

    HartreeFock::MultipoleMoments moments{};
    moments.origin = origin;

    for (int axis = 0; axis < 3; ++axis)
        moments.electronic_dipole[axis] =
            -(total_density.array() * matrices.dipole[axis].array()).sum();

    for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
    {
        const double charge = static_cast<double>(calculator._molecule.atomic_numbers[atom]);
        const Eigen::Vector3d position =
            calculator._molecule._standard.row(static_cast<Eigen::Index>(atom)).transpose() - origin;
        moments.nuclear_dipole += charge * position;
    }
    moments.total_dipole = moments.nuclear_dipole + moments.electronic_dipole;

    Eigen::Matrix3d raw_electronic = Eigen::Matrix3d::Zero();
    raw_electronic(0, 0) = -(total_density.array() * matrices.quadrupole[0].array()).sum();
    raw_electronic(0, 1) = -(total_density.array() * matrices.quadrupole[1].array()).sum();
    raw_electronic(0, 2) = -(total_density.array() * matrices.quadrupole[2].array()).sum();
    raw_electronic(1, 1) = -(total_density.array() * matrices.quadrupole[3].array()).sum();
    raw_electronic(1, 2) = -(total_density.array() * matrices.quadrupole[4].array()).sum();
    raw_electronic(2, 2) = -(total_density.array() * matrices.quadrupole[5].array()).sum();
    raw_electronic(1, 0) = raw_electronic(0, 1);
    raw_electronic(2, 0) = raw_electronic(0, 2);
    raw_electronic(2, 1) = raw_electronic(1, 2);

    Eigen::Matrix3d raw_nuclear = Eigen::Matrix3d::Zero();
    for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
    {
        const double charge = static_cast<double>(calculator._molecule.atomic_numbers[atom]);
        const Eigen::Vector3d position =
            calculator._molecule._standard.row(static_cast<Eigen::Index>(atom)).transpose() - origin;
        raw_nuclear += charge * (position * position.transpose());
    }

    auto to_traceless = [](const Eigen::Matrix3d &raw)
    {
        // Planck reports the conventional traceless quadrupole tensor:
        // Q = 3 R R^T - Tr(R R^T) I. The AO contraction above produces the raw
        // second moment tensor, and this helper converts both electronic and
        // nuclear pieces into the final reported form.
        Eigen::Matrix3d quadrupole = 3.0 * raw;
        quadrupole.diagonal().array() -= raw.trace();
        return quadrupole;
    };

    moments.electronic_quadrupole = to_traceless(raw_electronic);
    moments.nuclear_quadrupole = to_traceless(raw_nuclear);
    moments.total_quadrupole = moments.electronic_quadrupole + moments.nuclear_quadrupole;

    return moments;
}
