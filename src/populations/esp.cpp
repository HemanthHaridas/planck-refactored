#include "populations/esp.h"

#include "integrals/os.h"
#include "lookup/elements.h"

#include <algorithm>
#include <cmath>
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

std::expected<std::vector<Eigen::Vector3d>, std::string>
HartreeFock::SCF::chelpg_grid(
    const Molecule &molecule,
    double spacing,
    double headspace,
    double radius_scale)
{
    if (spacing <= 0.0)
        return std::unexpected("CHELPG grid: spacing must be positive");
    if (headspace <= 0.0)
        return std::unexpected("CHELPG grid: headspace must be positive");
    if (radius_scale <= 0.0)
        return std::unexpected("CHELPG grid: radius_scale must be positive");
    if (molecule._standard.rows() != static_cast<Eigen::Index>(molecule.natoms))
        return std::unexpected(
            "CHELPG grid: molecule._standard has no row per atom -- coordinates "
            "were not prepared (see the Coordinate Units gotcha)");

    // Per-atom exclusion radii, in Bohr. Ghost atoms carry basis functions but
    // no nucleus, so they neither exclude points nor anchor the box: a point
    // near a ghost is a legitimate sampling point for the real atoms.
    std::vector<double> radii;
    std::vector<Eigen::Index> real_atoms;
    radii.reserve(molecule.natoms);
    for (std::size_t a = 0; a < molecule.natoms; ++a)
    {
        if (molecule.nuclear_charge(a) == 0.0)
        {
            radii.push_back(0.0);
            continue;
        }
        const auto element = element_from_z(static_cast<std::uint64_t>(
            molecule.atomic_numbers(static_cast<Eigen::Index>(a))));
        if (!element)
            return std::unexpected("CHELPG grid: " + element.error());

        radii.push_back(radius_scale * element->radius * ANGSTROM_TO_BOHR);
        real_atoms.push_back(static_cast<Eigen::Index>(a));
    }

    if (real_atoms.empty())
        return std::unexpected("CHELPG grid: molecule has no real (non-ghost) atoms");

    // Bounding box over the real atoms, padded by headspace.
    Eigen::Vector3d lo = molecule._standard.row(real_atoms.front());
    Eigen::Vector3d hi = lo;
    for (Eigen::Index a : real_atoms)
    {
        const Eigen::Vector3d r = molecule._standard.row(a);
        lo = lo.cwiseMin(r);
        hi = hi.cwiseMax(r);
    }
    lo.array() -= headspace;
    hi.array() += headspace;

    std::vector<Eigen::Vector3d> points;
    const auto steps = [&](int axis) {
        return static_cast<long>(std::floor((hi[axis] - lo[axis]) / spacing)) + 1;
    };

    for (long ix = 0; ix < steps(0); ++ix)
        for (long iy = 0; iy < steps(1); ++iy)
            for (long iz = 0; iz < steps(2); ++iz)
            {
                const Eigen::Vector3d p(lo[0] + static_cast<double>(ix) * spacing,
                                        lo[1] + static_cast<double>(iy) * spacing,
                                        lo[2] + static_cast<double>(iz) * spacing);

                bool inside_an_atom = false;
                bool within_headspace = false;
                for (Eigen::Index a : real_atoms)
                {
                    const double d_sq =
                        (p - molecule._standard.row(a).transpose()).squaredNorm();
                    const double r = radii[static_cast<std::size_t>(a)];
                    if (d_sq < r * r)
                    {
                        inside_an_atom = true;
                        break;
                    }
                    // Measured from the vdW surface, not the nucleus, so the
                    // shell thickness is uniform rather than varying by element.
                    const double reach = r + headspace;
                    if (d_sq <= reach * reach)
                        within_headspace = true;
                }

                if (!inside_an_atom && within_headspace)
                    points.push_back(p);
            }

    if (points.empty())
        return std::unexpected(
            "CHELPG grid: no points survived the exclusion test -- spacing may be "
            "larger than the sampling shell, or radius_scale too large");

    return points;
}

std::expected<HartreeFock::SCF::ESPChargeFit, std::string>
HartreeFock::SCF::fit_esp_charges(
    const Molecule &molecule,
    const std::vector<Eigen::Vector3d> &points,
    const Eigen::VectorXd &potential,
    double total_charge)
{
    const std::size_t natoms = molecule.natoms;
    if (natoms == 0)
        return std::unexpected("ESP fit: molecule has no atoms");
    if (points.size() != static_cast<std::size_t>(potential.size()))
        return std::unexpected("ESP fit: point count does not match potential length");
    if (points.size() < natoms)
        return std::unexpected(
            "ESP fit: fewer grid points than atoms -- the fit is underdetermined");
    if (molecule._standard.rows() != static_cast<Eigen::Index>(natoms))
        return std::unexpected("ESP fit: molecule._standard has no row per atom");

    const Eigen::Index n = static_cast<Eigen::Index>(natoms);
    const Eigen::Index m = static_cast<Eigen::Index>(points.size());

    // Design matrix: A(k,a) = 1 / |r_k - R_a|, so phi_model = A q.
    Eigen::MatrixXd A(m, n);
    for (Eigen::Index k = 0; k < m; ++k)
        for (Eigen::Index a = 0; a < n; ++a)
        {
            const double d =
                (points[static_cast<std::size_t>(k)] -
                 molecule._standard.row(a).transpose())
                    .norm();
            if (d < 1e-12)
                return std::unexpected(
                    "ESP fit: a grid point coincides with a nucleus");
            A(k, a) = 1.0 / d;
        }

    // Normal equations with one Lagrange multiplier for sum(q) = total_charge:
    //
    //   [ 2 A^T A   1 ] [ q      ]   [ 2 A^T phi   ]
    //   [ 1^T       0 ] [ lambda ] = [ total_charge ]
    //
    // Symmetric but INDEFINITE (the zero block guarantees a negative
    // eigenvalue), so LLT would fail here -- LDL^T is required, not merely
    // preferred.
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n + 1, n + 1);
    M.topLeftCorner(n, n) = 2.0 * A.transpose() * A;
    M.topRightCorner(n, 1).setOnes();
    M.bottomLeftCorner(1, n).setOnes();

    Eigen::VectorXd rhs(n + 1);
    rhs.head(n) = 2.0 * A.transpose() * potential;
    rhs(n) = total_charge;

    const Eigen::VectorXd solution = M.ldlt().solve(rhs);
    if (!solution.allFinite())
        return std::unexpected(
            "ESP fit: the constrained least-squares solve did not produce a "
            "finite result -- the grid may be degenerate (e.g. all points "
            "collinear with the atoms)");

    ESPChargeFit fit;
    fit.charges = solution.head(n);
    fit.n_points = points.size();

    const Eigen::VectorXd residual = A * fit.charges - potential;
    fit.rms = std::sqrt(residual.squaredNorm() / static_cast<double>(m));

    const double phi_rms =
        std::sqrt(potential.squaredNorm() / static_cast<double>(m));
    // A uniformly zero potential has no scale to be relative to; report the
    // absolute RMS rather than dividing by zero.
    fit.rrms = (phi_rms > 0.0) ? (fit.rms / phi_rms) : fit.rms;

    return fit;
}
