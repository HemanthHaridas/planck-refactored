#include "populations/esp.h"

#include "base/sphere.h"
#include "integrals/os.h"
#include "io/logging.h"
#include "lookup/elements.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

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

std::expected<Eigen::VectorXd, std::string>
HartreeFock::SCF::reaction_field_potential(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<Solvation::PCMSurfacePoint> &surface_points,
    const Eigen::VectorXd &charges)
{
    if (static_cast<Eigen::Index>(surface_points.size()) != charges.size())
        return std::unexpected(
            "ESP reaction field: " + std::to_string(surface_points.size()) +
            " tesserae but " + std::to_string(charges.size()) +
            " apparent charges -- the cavity and the PCM solve disagree");

    if (points.empty())
        return Eigen::VectorXd();

    Eigen::VectorXd phi = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(points.size()));

    for (std::size_t k = 0; k < points.size(); ++k)
    {
        double acc = 0.0;
        for (std::size_t i = 0; i < surface_points.size(); ++i)
        {
            const double distance = (points[k] - surface_points[i].position).norm();

            // An ESP point sitting on a tessera diverges the same way one
            // sitting on a nucleus does, and for the same reason. Reject it
            // rather than returning an enormous finite number: the sampling
            // shells and the cavity are separately constructed surfaces with
            // different radii conventions, so a collision means the two were
            // configured to overlap, which is worth saying out loud.
            if (distance < 1e-12)
                return std::unexpected(
                    "ESP reaction field: evaluation point coincides with PCM "
                    "tessera " + std::to_string(i) + " -- the potential "
                    "diverges there (the ESP shells and the PCM cavity "
                    "overlap; check radius_scale against cavity_scale)");

            acc += charges(static_cast<Eigen::Index>(i)) / distance;
        }
        phi(static_cast<Eigen::Index>(k)) = acc;
    }

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

std::expected<std::vector<Eigen::Vector3d>, std::string>
HartreeFock::SCF::connolly_grid(
    const Molecule &molecule,
    const std::vector<double> &shell_scales,
    int points_per_shell,
    double radius_scale)
{
    if (shell_scales.empty())
        return std::unexpected("Connolly grid: no shell scales given");
    if (points_per_shell <= 0)
        return std::unexpected("Connolly grid: points_per_shell must be positive");
    if (radius_scale <= 0.0)
        return std::unexpected("Connolly grid: radius_scale must be positive");
    for (double s : shell_scales)
        if (s <= 0.0)
            return std::unexpected("Connolly grid: shell scales must be positive");
    if (molecule._standard.rows() != static_cast<Eigen::Index>(molecule.natoms))
        return std::unexpected(
            "Connolly grid: molecule._standard has no row per atom -- coordinates "
            "were not prepared (see the Coordinate Units gotcha)");

    // Base vdW radii in Bohr. Ghosts carry basis functions but no nucleus, so
    // they neither host shells nor bury anyone else's points.
    std::vector<double> radii(molecule.natoms, 0.0);
    std::vector<Eigen::Index> real_atoms;
    for (std::size_t a = 0; a < molecule.natoms; ++a)
    {
        if (molecule.nuclear_charge(a) == 0.0)
            continue;

        const auto element = element_from_z(static_cast<std::uint64_t>(
            molecule.atomic_numbers(static_cast<Eigen::Index>(a))));
        if (!element)
            return std::unexpected("Connolly grid: " + element.error());

        radii[a] = radius_scale * element->radius * ANGSTROM_TO_BOHR;
        real_atoms.push_back(static_cast<Eigen::Index>(a));
    }

    if (real_atoms.empty())
        return std::unexpected("Connolly grid: molecule has no real (non-ghost) atoms");

    // One direction set, reused for every atom and every shell -- but expressed
    // in a MOLECULE-DERIVED frame, not the lab frame.
    //
    // This is the whole point of E4a and it is easy to get wrong. A sphere is
    // rotationally symmetric; a fixed DISCRETE SAMPLING of one is not. Using
    // fibonacci_sphere() directly pins the sample pattern to the lab axes, so
    // rotating the molecule slides the points across each atom's surface and
    // the fitted charges move -- measured at 3.3e-02 (200 points/shell) and
    // 8.1e-03 (800), i.e. barely better than the cubic lattice it replaced, and
    // NOT converging away with density. Rotating the same directions with the
    // molecule instead gives 7.1e-15 / 1.0e-14: solver noise, density
    // independent.
    //
    // Since this function receives no rotation, the frame has to come from the
    // geometry itself. Any orthonormal frame that is a fixed function of the
    // atom positions works, because under a rigid motion R the frame rotates to
    // R*frame and the sample points follow the molecule exactly. Gram-Schmidt
    // on displacements from the centroid is sufficient and avoids the
    // degenerate-eigenvector discontinuity that rules out an inertia-tensor
    // frame (see docs/ESP_CHARGES.md, section 3, on why principal axes were
    // rejected).
    Eigen::Matrix3d frame = Eigen::Matrix3d::Identity();
    {
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (Eigen::Index a : real_atoms)
            centroid += molecule._standard.row(a).transpose();
        centroid /= static_cast<double>(real_atoms.size());

        // Pick the farthest atom from the centroid as the first axis: a
        // rotation-invariant choice, so the same atom is picked in any frame.
        // Ties are broken by atom index, which is also frame independent.
        auto axis_from_farthest = [&](const Eigen::Vector3d &reject_onto,
                                      bool do_reject) -> Eigen::Vector3d {
            Eigen::Vector3d best = Eigen::Vector3d::Zero();
            double best_norm = 0.0;
            for (Eigen::Index a : real_atoms)
            {
                Eigen::Vector3d v =
                    molecule._standard.row(a).transpose() - centroid;
                if (do_reject)
                    v -= reject_onto * reject_onto.dot(v);
                const double n = v.norm();
                if (n > best_norm + 1e-10)
                {
                    best_norm = n;
                    best = v;
                }
            }
            return (best_norm > 1e-8) ? best.normalized() : Eigen::Vector3d::Zero();
        };

        Eigen::Vector3d e1 = axis_from_farthest(Eigen::Vector3d::Zero(), false);
        if (e1.squaredNorm() < 0.5) // single atom, or all atoms coincident
            e1 = Eigen::Vector3d::UnitX();

        Eigen::Vector3d e2 = axis_from_farthest(e1, true);
        if (e2.squaredNorm() < 0.5)
        {
            // Linear molecule: no second axis is defined by the geometry, and
            // that is fine -- a linear arrangement is rotationally symmetric
            // about e1, so any perpendicular completion samples equivalently.
            const Eigen::Vector3d seed =
                (std::abs(e1[0]) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
            e2 = (seed - e1 * e1.dot(seed)).normalized();
        }

        frame.col(0) = e1;
        frame.col(1) = e2;
        frame.col(2) = e1.cross(e2);
    }

    std::vector<Eigen::Vector3d> directions = fibonacci_sphere(points_per_shell);
    for (Eigen::Vector3d &d : directions)
        d = frame * d;

    std::vector<Eigen::Vector3d> points;
    points.reserve(real_atoms.size() * shell_scales.size() * directions.size());

    for (double shell : shell_scales)
        for (Eigen::Index a : real_atoms)
        {
            const Eigen::Vector3d center = molecule._standard.row(a);
            const double r = shell * radii[static_cast<std::size_t>(a)];

            for (const Eigen::Vector3d &dir : directions)
            {
                const Eigen::Vector3d p = center + r * dir;

                // Buried if it falls inside any OTHER atom's sphere at this
                // same shell scale -- the Connolly "solvent-accessible surface"
                // test. Comparing at the same scale keeps the surface a single
                // smooth envelope instead of a union of mismatched spheres.
                bool buried = false;
                for (Eigen::Index b : real_atoms)
                {
                    if (b == a)
                        continue;
                    const double rb = shell * radii[static_cast<std::size_t>(b)];
                    if ((p - molecule._standard.row(b).transpose()).squaredNorm() <
                        rb * rb)
                    {
                        buried = true;
                        break;
                    }
                }

                if (!buried)
                    points.push_back(p);
            }
        }

    if (points.empty())
        return std::unexpected(
            "Connolly grid: every candidate point was buried inside another "
            "atom's sphere");

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

std::expected<HartreeFock::SCF::RESPChargeFit, std::string>
HartreeFock::SCF::fit_resp_charges(
    const Molecule &molecule,
    const std::vector<Eigen::Vector3d> &points,
    const Eigen::VectorXd &potential,
    double total_charge,
    const RESPOptions &options)
{
    const std::size_t natoms = molecule.natoms;
    if (natoms == 0)
        return std::unexpected("RESP fit: molecule has no atoms");
    if (points.size() != static_cast<std::size_t>(potential.size()))
        return std::unexpected("RESP fit: point count does not match potential length");
    if (points.size() < natoms)
        return std::unexpected(
            "RESP fit: fewer grid points than atoms -- the fit is underdetermined");
    if (molecule._standard.rows() != static_cast<Eigen::Index>(natoms))
        return std::unexpected("RESP fit: molecule._standard has no row per atom");
    if (options.strength < 0.0)
        return std::unexpected("RESP fit: restraint strength must not be negative");
    if (options.tightness <= 0.0)
        return std::unexpected("RESP fit: restraint tightness must be positive");
    if (options.max_iterations < 1)
        return std::unexpected("RESP fit: max_iterations must be at least 1");

    for (const auto &group : options.equivalence_groups)
        for (std::size_t a : group)
            if (a >= natoms)
                return std::unexpected(
                    "RESP fit: equivalence group names atom " + std::to_string(a + 1) +
                    ", which is out of range");

    const Eigen::Index n = static_cast<Eigen::Index>(natoms);
    const Eigen::Index m = static_cast<Eigen::Index>(points.size());

    // Design matrix, identical to the unrestrained fit.
    Eigen::MatrixXd A(m, n);
    for (Eigen::Index k = 0; k < m; ++k)
        for (Eigen::Index a = 0; a < n; ++a)
        {
            const double d =
                (points[static_cast<std::size_t>(k)] -
                 molecule._standard.row(a).transpose())
                    .norm();
            if (d < 1e-12)
                return std::unexpected("RESP fit: a grid point coincides with a nucleus");
            A(k, a) = 1.0 / d;
        }

    const Eigen::MatrixXd AtA = 2.0 * A.transpose() * A;
    const Eigen::VectorXd Atb = 2.0 * A.transpose() * potential;

    // Which atoms feel the restraint. Hydrogens are conventionally exempt.
    std::vector<bool> restrained(natoms, options.strength > 0.0);
    if (options.exempt_hydrogen)
        for (std::size_t a = 0; a < natoms; ++a)
            if (molecule.atomic_numbers(static_cast<Eigen::Index>(a)) == 1)
                restrained[a] = false;

    // Constraint rows: one for the total charge, plus one per equivalence pair.
    // Each group of size g contributes g-1 rows of the form q_i - q_j = 0,
    // chaining consecutive members rather than pairing all against the first,
    // which keeps the rows independent.
    std::size_t n_equiv_rows = 0;
    for (const auto &group : options.equivalence_groups)
        if (group.size() > 1)
            n_equiv_rows += group.size() - 1;

    const Eigen::Index n_con = 1 + static_cast<Eigen::Index>(n_equiv_rows);
    const Eigen::Index dim = n + n_con;

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_con, n);
    Eigen::VectorXd d_con = Eigen::VectorXd::Zero(n_con);
    C.row(0).setOnes();
    d_con(0) = total_charge;
    {
        Eigen::Index row = 1;
        for (const auto &group : options.equivalence_groups)
            for (std::size_t i = 1; i < group.size(); ++i)
            {
                C(row, static_cast<Eigen::Index>(group[i - 1])) = 1.0;
                C(row, static_cast<Eigen::Index>(group[i])) = -1.0;
                d_con(row) = 0.0;
                ++row;
            }
    }

    // Iterate: the restraint's contribution to the normal matrix is the
    // diagonal a / sqrt(q^2 + b^2), which depends on the current charges.
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
    RESPChargeFit result;
    const double b_sq = options.tightness * options.tightness;

    for (int iter = 1; iter <= options.max_iterations; ++iter)
    {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dim, dim);
        M.topLeftCorner(n, n) = AtA;
        for (Eigen::Index a = 0; a < n; ++a)
            if (restrained[static_cast<std::size_t>(a)])
                M(a, a) += options.strength / std::sqrt(q(a) * q(a) + b_sq);

        M.topRightCorner(n, n_con) = C.transpose();
        M.bottomLeftCorner(n_con, n) = C;

        Eigen::VectorXd rhs(dim);
        rhs.head(n) = Atb;
        rhs.tail(n_con) = d_con;

        const Eigen::VectorXd solution = M.ldlt().solve(rhs);
        if (!solution.allFinite())
            return std::unexpected(
                "RESP fit: the restrained solve did not produce a finite result -- "
                "the grid may be degenerate or the equivalence constraints "
                "may be contradictory");

        const Eigen::VectorXd q_new = solution.head(n);
        const double shift = (q_new - q).cwiseAbs().maxCoeff();
        q = q_new;
        result.iterations = iter;

        if (shift < options.convergence)
        {
            result.converged = true;
            break;
        }
    }

    result.fit.charges = q;
    result.fit.n_points = points.size();

    // Report the fit quality against the POTENTIAL, not against the restrained
    // objective: the restraint deliberately trades a little fit quality for
    // better-behaved charges, and hiding that trade in the reported number
    // would defeat the purpose of reporting it.
    const Eigen::VectorXd residual = A * q - potential;
    result.fit.rms = std::sqrt(residual.squaredNorm() / static_cast<double>(m));
    const double phi_rms =
        std::sqrt(potential.squaredNorm() / static_cast<double>(m));
    result.fit.rrms = (phi_rms > 0.0) ? (result.fit.rms / phi_rms) : result.fit.rms;

    return result;
}

void HartreeFock::SCF::log_esp_report(
    const HartreeFock::Calculator &calculator,
    const std::vector<HartreeFock::ShellPair> &shell_pairs,
    const std::vector<HartreeFock::Solvation::PCMSurfacePoint> *surface_points,
    const Eigen::VectorXd *apparent_charges)
{
    if (!calculator._esp._enabled)
        return;

    // Under PCM the fitted density IS solvent-polarized (the reaction field
    // enters the Fock every SCF iteration), and there are then TWO legitimate
    // potentials to fit, differing by the reaction field's own contribution
    // sum_i q_i / |r - s_i|:
    //
    //   solute (default) -- nuc + el only. What force-field charge derivation
    //     wants: the AMBER/RESP convention fits the polarized solute WITHOUT
    //     the continuum, because an MD run supplies the solvent explicitly and
    //     would otherwise count it twice.
    //   total in-solvent  -- nuc + el + rxn. What another code prints, and what
    //     continuum interaction energies need.
    //
    // Neither was identified before, so a user comparing against another code
    // could not tell which one they had. `reaction_field_label` below is that
    // identification, and it is printed in BOTH cases.
    const bool solvated =
        calculator._solvation._model != HartreeFock::SolvationModel::None;
    const bool want_reaction_field =
        solvated && calculator._esp._include_reaction_field;

    if (calculator._esp._include_reaction_field && !solvated)
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Warning,
            "ESP Reaction Field :",
            "'include_reaction_field' has no effect without a solvation "
            "model; there is no reaction field to include");

    if (want_reaction_field && !(surface_points && apparent_charges &&
                                 apparent_charges->size() > 0))
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Warning,
            "ESP Reaction Field :",
            "PCM is active but its apparent charges are unavailable here, so "
            "the reaction field cannot be added -- reporting the SOLUTE "
            "potential instead");

    // Reflects what was actually assembled, not what was asked for: the caller
    // degrades to the solute potential on any PCM failure, and the label has to
    // follow it or it would state the opposite of the truth.
    const bool has_reaction_field =
        want_reaction_field && surface_points && apparent_charges &&
        apparent_charges->size() > 0;
    const char *reaction_field_label =
        !solvated ? nullptr
                  : (has_reaction_field
                         ? "TOTAL in-solvent potential (nuclear + electronic + "
                           "PCM reaction field)"
                         : "SOLUTE potential (nuclear + electronic) of the "
                           "solvent-polarized density; the PCM reaction field "
                           "is excluded. This is the force-field convention, "
                           "not the total in-solvent potential");

    if (solvated)
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Warning,
            "ESP + PCM :",
            std::string("the fit is to the ") + reaction_field_label);

    // The grid is just another way to fill the point list, so everything below
    // this block is identical for both modes -- there is no second code path.
    std::vector<Eigen::Vector3d> generated;
    if (calculator._esp.wants_grid())
    {
        auto grid =
            (calculator._esp._grid == HartreeFock::OptionsESP::Grid::Connolly)
                ? HartreeFock::SCF::connolly_grid(
                      calculator._molecule,
                      calculator._esp._shell_scales,
                      calculator._esp._points_per_shell,
                      calculator._esp._radius_scale)
                : HartreeFock::SCF::chelpg_grid(
                      calculator._molecule,
                      calculator._esp._grid_spacing * ANGSTROM_TO_BOHR,
                      calculator._esp._grid_headspace * ANGSTROM_TO_BOHR,
                      calculator._esp._radius_scale);
        if (!grid)
        {
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Warning,
                "Electrostatic Potential :",
                "Unavailable: " + grid.error());
            HartreeFock::Logger::blank();
            return;
        }
        generated = std::move(*grid);
    }

    const std::vector<Eigen::Vector3d> &points =
        calculator._esp.wants_grid() ? generated : calculator._esp._points;

    if (points.empty())
        return;

    // Total density: alpha + beta for an unrestricted reference, and for RHF the
    // alpha channel already holds the full P_total (the same convention
    // log_population_report uses just above).
    const bool has_spin_channels =
        calculator._scf._scf != HartreeFock::SCFType::RHF &&
        calculator._info._scf.beta.density.rows() == calculator._info._scf.alpha.density.rows();
    Eigen::MatrixXd total_density = calculator._info._scf.alpha.density;
    if (has_spin_channels)
        total_density += calculator._info._scf.beta.density;

    auto phi = HartreeFock::SCF::electrostatic_potential(
        calculator._molecule, shell_pairs, total_density, points);

    if (!phi)
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Warning,
            "Electrostatic Potential :",
            "Unavailable: " + phi.error());
        HartreeFock::Logger::blank();
        return;
    }

    // The two quantities differ by exactly this term, added here rather than
    // inside electrostatic_potential so the sum is visible at the call site.
    if (has_reaction_field)
    {
        auto phi_rxn = HartreeFock::SCF::reaction_field_potential(
            points, *surface_points, *apparent_charges);
        if (!phi_rxn)
        {
            // A tessera collision is a misconfiguration, not a reason to
            // silently print the other quantity under the label promising
            // this one -- the label was already emitted above.
            HartreeFock::Logger::logging(
                HartreeFock::LogLevel::Error,
                "ESP Reaction Field :",
                phi_rxn.error());
            HartreeFock::Logger::blank();
            return;
        }
        *phi += *phi_rxn;
    }

    // Grid mode: the point table would be thousands of rows and tells nobody
    // anything, so report the fitted charges instead -- that is what the grid
    // was generated for.
    if (calculator._esp.wants_grid())
    {
        const double total_charge = static_cast<double>(calculator._molecule.charge);

        // RESP is the same fit plus a restraint, so both routes produce the
        // same ESPChargeFit and everything below is shared. Only the iteration
        // count is RESP-specific, and it is reported because an unconverged
        // restraint is otherwise invisible in the charges.
        Eigen::VectorXd charges;
        double rrms = 0.0;
        double rms = 0.0;
        int resp_iterations = 0;
        bool resp_converged = true;

        if (calculator._esp._resp)
        {
            HartreeFock::SCF::RESPOptions options;
            options.strength = calculator._esp._resp_strength;
            options.tightness = calculator._esp._resp_tightness;
            options.exempt_hydrogen = calculator._esp._resp_exempt_hydrogen;
            options.equivalence_groups = calculator._esp._resp_equivalence;

            auto fit = HartreeFock::SCF::fit_resp_charges(
                calculator._molecule, points, *phi, total_charge, options);
            if (!fit)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "ESP Charges :",
                    "Unavailable: " + fit.error());
                HartreeFock::Logger::blank();
                return;
            }
            charges = fit->fit.charges;
            rrms = fit->fit.rrms;
            rms = fit->fit.rms;
            resp_iterations = fit->iterations;
            resp_converged = fit->converged;
        }
        else
        {
            auto fit = HartreeFock::SCF::fit_esp_charges(
                calculator._molecule, points, *phi, total_charge);
            if (!fit)
            {
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "ESP Charges :",
                    "Unavailable: " + fit.error());
                HartreeFock::Logger::blank();
                return;
            }
            charges = fit->charges;
            rrms = fit->rrms;
            rms = fit->rms;
        }

        const bool connolly =
            (calculator._esp._grid == HartreeFock::OptionsESP::Grid::Connolly);
        const char *grid_label =
            calculator._esp._resp
                ? (connolly ? "RESP Charges (Connolly) :" : "RESP Charges (CHELPG) :")
                : (connolly ? "ESP Charges (Connolly) :" : "ESP Charges (CHELPG) :");
        HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, grid_label, "");

        constexpr int charge_width = 62;
        std::cout << std::string(charge_width, '-') << "\n"
                  << std::setw(6) << std::right << "Atom"
                  << std::setw(8) << std::right << "Elem"
                  << std::setw(8) << std::right << "Z"
                  << std::setw(24) << std::right << "Charge (e)" << "\n"
                  << std::string(charge_width, '-') << "\n";

        for (std::size_t a = 0; a < calculator._molecule.natoms; ++a)
        {
            const int Z = calculator._molecule.atomic_numbers(static_cast<Eigen::Index>(a));
            const auto element = element_from_z(static_cast<std::uint64_t>(Z));
            const std::string symbol = element ? std::string(element->symbol) : "?";
            std::cout << std::setw(6) << std::right << (a + 1)
                      << std::setw(8) << std::right << symbol
                      << std::setw(8) << std::right << Z
                      << std::setw(24) << std::right << std::fixed << std::setprecision(8)
                      << charges(static_cast<Eigen::Index>(a)) << "\n";
        }

        std::cout << std::string(charge_width, '-') << "\n"
                  << std::setw(22) << std::left << "  Total"
                  << std::setw(24) << std::right << std::fixed << std::setprecision(8)
                  << charges.sum() << "\n"
                  << std::string(charge_width, '-') << "\n";

        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info, "ESP Grid Points :", points.size());
        std::cout << "  ESP Fit RRMS " << std::scientific << std::setprecision(6)
                  << rrms << "\n";
        std::cout << "  ESP Fit RMS  " << std::scientific << std::setprecision(6)
                  << rms << "\n";

        if (calculator._esp._resp)
        {
            std::cout << "  RESP Iterations " << resp_iterations << "\n";
            if (!resp_converged)
                HartreeFock::Logger::logging(
                    HartreeFock::LogLevel::Warning,
                    "RESP :",
                    "the restraint iteration did not converge -- the charges "
                    "below are the last iterate, not a converged fit");
        }

        HartreeFock::Logger::blank();
        return;
    }

    HartreeFock::Logger::logging(HartreeFock::LogLevel::Info, "Electrostatic Potential :", "");

    constexpr int line_width = 78;
    std::cout << std::string(line_width, '-') << "\n"
              << std::setw(6) << std::right << "Point"
              << std::setw(14) << std::right << "X (Bohr)"
              << std::setw(14) << std::right << "Y (Bohr)"
              << std::setw(14) << std::right << "Z (Bohr)"
              << std::setw(24) << std::right << "Potential (a.u.)" << "\n"
              << std::string(line_width, '-') << "\n";

    // `points`, not calculator._esp._points: the two coincide here because this
    // branch is only reached when no grid was requested, but reading the option
    // directly would silently print an empty table if the grid branch above
    // ever stopped returning early.
    for (std::size_t k = 0; k < points.size(); ++k)
    {
        const Eigen::Vector3d &p = points[k];
        std::cout << std::setw(6) << std::right << (k + 1)
                  << std::setw(14) << std::right << std::fixed << std::setprecision(6) << p[0]
                  << std::setw(14) << std::right << std::fixed << std::setprecision(6) << p[1]
                  << std::setw(14) << std::right << std::fixed << std::setprecision(6) << p[2]
                  << std::setw(24) << std::right << std::fixed << std::setprecision(12)
                  << (*phi)(static_cast<Eigen::Index>(k)) << "\n";
    }

    std::cout << std::string(line_width, '-') << "\n";
    HartreeFock::Logger::blank();
}
