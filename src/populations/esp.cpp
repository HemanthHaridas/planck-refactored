#include "populations/esp.h"

#include "integrals/os.h"

#include <cstddef>

std::expected<Eigen::VectorXd, std::string>
HartreeFock::SCF::electrostatic_potential(
    const Molecule &molecule,
    const std::vector<ShellPair> &shell_pairs,
    const Eigen::MatrixXd &density,
    const std::vector<Eigen::Vector3d> &points)
{
    if (density.rows() != density.cols())
        return std::unexpected("ESP: density matrix must be square");

    if (molecule._standard.rows() != static_cast<Eigen::Index>(molecule.natoms))
        return std::unexpected(
            "ESP: molecule._standard has no row per atom -- coordinates were "
            "not prepared (see the Coordinate Units gotcha: _standard must be "
            "in Bohr before any potential is evaluated)");

    if (points.empty())
        return Eigen::VectorXd();

    // ── Nuclear term: sum_A Z_A / |r - R_A| ──────────────────────────────────
    // _standard is the Bohr frame basis centers and nuclear repulsion use, so
    // the potential is evaluated in the same frame the integrals were built in.
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(points.size()));

    for (std::size_t k = 0; k < points.size(); ++k)
    {
        double acc = 0.0;
        for (std::size_t a = 0; a < molecule.natoms; ++a)
        {
            const double Z = molecule.nuclear_charge(a);
            if (Z == 0.0) // ghost atom (BSSE counterpoise): no nucleus
                continue;

            const Eigen::Vector3d R = molecule._standard.row(static_cast<Eigen::Index>(a));
            const double distance = (points[k] - R).norm();
            if (distance < 1e-12)
                return std::unexpected(
                    "ESP: evaluation point coincides with a nucleus -- the "
                    "potential diverges there");

            acc += Z / distance;
        }
        phi(static_cast<Eigen::Index>(k)) = acc;
    }

    // ── Electronic term ──────────────────────────────────────────────────────
    // _compute_electronic_potential returns the RAW positive contraction
    // P : <mu|1/r|nu>; the electron's negative charge is applied here, which is
    // what makes this the physical potential rather than the AO integral.
    const Eigen::VectorXd phi_el =
        HartreeFock::ObaraSaika::_compute_electronic_potential(shell_pairs, density, points);

    if (phi_el.size() != phi.size())
        return std::unexpected("ESP: electronic potential length mismatch");

    phi -= phi_el;
    return phi;
}
