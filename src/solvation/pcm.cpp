// ─────────────────────────────────────────────────────────────────────────────
// C-PCM implementation. See pcm.h for the physics summary.
//
// Two entry points:
//   build_pcm_state            — one-shot cavity + density-independent setup
//   evaluate_pcm_reaction_field — per-SCF-iteration density-dependent solve
//
// The split is deliberate: every quantity that depends only on the geometry
// (tesserae positions, areas, influence matrix, nuclear potential at the
// surface, AO matrices for unit point charges at each tessera) is built once
// in build_pcm_state. The hot-path evaluator only does:
//   1. one Frobenius product per tessera against the current density,
//   2. one Cholesky-class linear solve on the (npts x npts) influence matrix,
//   3. one weighted sum of the cached unit-charge matrices to assemble V_rxn.
// No shell-pair integrals are recomputed inside the SCF loop.
// ─────────────────────────────────────────────────────────────────────────────

#include "pcm.h"

#include <cmath>
#include <format>
#include <numbers>

#include <Eigen/Cholesky>

#include "integrals/base.h"
#include "io/logging.h"
#include "lookup/elements.h"

namespace
{
    // ISWIG diagonal correction factor for the C-PCM influence matrix
    // (Pascual-Ahuir, Silla, Tunon). The bare D_ii = sqrt(4 pi / a_i) form
    // approximates the self-energy of a uniformly charged disc of area a_i;
    // multiplying by 1.07 was found empirically to give accurate solvation
    // free energies across a wide range of solvents and cavity tessellations.
    constexpr double ISWIG_DIAGONAL_SCALE = 1.07;

    // Quasi-uniform tiling of the unit sphere by the golden-angle (Fibonacci)
    // construction. Compared to a lat/long grid this avoids the pole pile-up,
    // and compared to a Lebedev grid it gives a single-parameter density that
    // can be set arbitrarily by the user. Adequate for PCM-energy purposes;
    // would not be appropriate for high-accuracy XC integration.
    std::vector<Eigen::Vector3d> fibonacci_sphere(int npoints)
    {
        std::vector<Eigen::Vector3d> points;
        points.reserve(static_cast<std::size_t>(npoints));

        // pi * (3 - sqrt(5)) is the golden angle in radians.
        const double golden_angle = std::numbers::pi * (3.0 - std::sqrt(5.0));
        for (int i = 0; i < npoints; ++i)
        {
            // z is uniformly distributed on (-1, 1) with the +0.5 shift
            // keeping endpoints away from the poles, so each point sits at
            // the centroid of its strip rather than on a pole.
            const double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(npoints);
            const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double phi = golden_angle * static_cast<double>(i);
            points.emplace_back(radial * std::cos(phi), radial * std::sin(phi), z);
        }

        return points;
    }
}

// One-shot cavity + density-independent precomputation. After this returns,
// PCMState contains everything the per-iteration evaluator needs.
std::expected<HartreeFock::Solvation::PCMState, std::string>
HartreeFock::Solvation::build_pcm_state(
    const HartreeFock::Calculator &calculator,
    const std::vector<HartreeFock::ShellPair> &shell_pairs)
{
    PCMState state;
    // Solvation off → return an "enabled() == false" state. Callers can pass
    // this around freely; evaluate_pcm_reaction_field short-circuits to a zero
    // operator.
    if (calculator._solvation._model == HartreeFock::SolvationModel::None)
        return state;

    const int points_per_atom = calculator._solvation._surface_points_per_atom;
    // Six is the minimum that gives a non-degenerate octahedral-ish coverage
    // of a sphere; any fewer and the cavity is essentially undefined.
    if (points_per_atom < 6)
        return std::unexpected("PCM requires at least 6 surface points per atom");

    // ── Atomic sphere radii ──────────────────────────────────────────────────
    // Each atom A is wrapped in a sphere of radius R_A = s * R_A^vdW, where
    // s = cavity_scale (default 1.2 — the standard Bondi-radius PCM choice)
    // and R_A^vdW comes from the periodic-table lookup table in Angstrom.
    const auto sphere_directions = fibonacci_sphere(points_per_atom);
    std::vector<double> radii_bohr;
    radii_bohr.reserve(calculator._molecule.natoms);

    for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
    {
        const auto element = element_from_z(static_cast<std::uint64_t>(
            calculator._molecule.atomic_numbers(static_cast<Eigen::Index>(atom))));
        if (!element)
            return std::unexpected("PCM cavity setup failed: " + element.error());

        radii_bohr.push_back(
            calculator._solvation._cavity_scale *
            element->radius *
            ANGSTROM_TO_BOHR);
    }

    // ── Tessellation with bury-test pruning ──────────────────────────────────
    // For each atom: place points_per_atom Fibonacci-sphere points on the
    // atom's sphere, then discard any that fall inside another atom's sphere
    // (that point is "buried" — it's inside the cavity, not on its surface).
    // What survives is an approximation to the solvent-accessible surface.
    //
    // No inter-sphere smoothing is applied. This is simpler than GEPOL/SES
    // schemes but can introduce small ripples in the energy as a function of
    // geometry — fine for single-point energies but a reason analytic
    // gradients are not implemented.
    for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
    {
        const Eigen::Vector3d center = calculator._molecule._standard.row(static_cast<Eigen::Index>(atom));
        const double radius = radii_bohr[atom];
        // Equal-area weights: 4 pi R^2 / N_points. This is exact for the
        // pre-pruning sphere; surviving points inherit the same area, which
        // slightly under-counts area near sphere intersections. Acceptable
        // for typical organic chemistry use cases.
        const double area_per_point =
            4.0 * std::numbers::pi * radius * radius / static_cast<double>(points_per_atom);

        for (const auto &direction : sphere_directions)
        {
            const Eigen::Vector3d point = center + radius * direction;
            bool buried = false;
            for (std::size_t other = 0; other < calculator._molecule.natoms; ++other)
            {
                if (other == atom)
                    continue;

                const Eigen::Vector3d other_center = calculator._molecule._standard.row(static_cast<Eigen::Index>(other));
                const double other_radius = radii_bohr[other];
                // squaredNorm avoids the sqrt — strict interior test.
                if ((point - other_center).squaredNorm() < other_radius * other_radius)
                {
                    buried = true;
                    break;
                }
            }

            if (!buried)
            {
                state.surface_points.push_back(
                    PCMSurfacePoint{
                        .position = point,
                        .area = area_per_point,
                        .atom_index = atom});
            }
        }
    }

    // Pathological-but-possible case: points_per_atom too small or atoms too
    // close together to expose any surface. Fail loudly rather than letting
    // a zero-tessera cavity propagate (which would produce a degenerate D
    // and a NaN reaction field).
    if (state.surface_points.empty())
        return std::unexpected("PCM cavity generation failed: all surface points were buried");

    // ── Density-independent precomputation ───────────────────────────────────
    // For each tessera i:
    //   - phi_nuc_i: the nuclei's contribution to the total potential at s_i
    //   - V^(i): the AO matrix of a unit point charge at s_i — this is what
    //     the SCF loop will weight by the (density-dependent) apparent charge
    //     q_i to assemble V_rxn
    //   - influence-matrix row D_i*: 1/|s_i - s_j| off-diagonal, ISWIG diagonal
    const Eigen::Index npoints = static_cast<Eigen::Index>(state.surface_points.size());
    state.nuclear_potential = Eigen::VectorXd::Zero(npoints);
    state.influence_matrix = Eigen::MatrixXd::Zero(npoints, npoints);
    state.unit_potential_matrices.reserve(static_cast<std::size_t>(npoints));

    for (Eigen::Index i = 0; i < npoints; ++i)
    {
        const auto &site = state.surface_points[static_cast<std::size_t>(i)];

        // Nuclear potential at this tessera. Coordinates are in Bohr (the
        // _standard frame) so this gives Hartree atomic units directly.
        for (std::size_t atom = 0; atom < calculator._molecule.natoms; ++atom)
        {
            const Eigen::Vector3d nucleus = calculator._molecule._standard.row(static_cast<Eigen::Index>(atom));
            const double charge = static_cast<double>(calculator._molecule.atomic_numbers(static_cast<Eigen::Index>(atom)));
            state.nuclear_potential(i) += charge / (site.position - nucleus).norm();
        }

        // V^(i) = AO matrix of a single +1 charge at s_i. We re-use the same
        // Obara-Saika nuclear-attraction kernel that builds V_ne for the
        // molecule, just with a one-element charge list. No symmetry ops are
        // passed: the reaction-field operator is built in the full AO basis,
        // and SAO blocking happens elsewhere when symmetry is enabled.
        std::vector<HartreeFock::ExternalCharge> unit_charge{
            HartreeFock::ExternalCharge{.position = site.position, .charge = 1.0}};
        state.unit_potential_matrices.push_back(
            _compute_external_charge_attraction(
                shell_pairs,
                calculator._shells.nbasis(),
                unit_charge,
                calculator._integral._engine,
                nullptr));

        // Influence-matrix row. The diagonal element is the self-interaction
        // of a smeared surface charge with itself (the bare 1/r form would
        // diverge for a true point charge); see the ISWIG_DIAGONAL_SCALE
        // comment at the top of this file for the empirical 1.07 factor.
        for (Eigen::Index j = 0; j < npoints; ++j)
        {
            if (i == j)
            {
                state.influence_matrix(i, j) =
                    ISWIG_DIAGONAL_SCALE *
                    std::sqrt(4.0 * std::numbers::pi / site.area);
            }
            else
            {
                const double distance =
                    (site.position - state.surface_points[static_cast<std::size_t>(j)].position).norm();
                state.influence_matrix(i, j) = 1.0 / distance;
            }
        }
    }

    // f(eps) = (eps - 1) / (eps + 1/2). C-PCM (Klamt-Schuurmann) form:
    // interpolates between vacuum (eps=1 → f=0, no polarization) and a
    // perfect conductor (eps→∞ → f=1). The "+ 1/2" is what makes this the
    // conductor-like flavour rather than the strict-conductor (COSMO) form
    // (which uses "+ 0").
    const double epsilon = calculator._solvation._dielectric;
    state.dielectric_factor = (epsilon - 1.0) / (epsilon + 0.5);

    HartreeFock::Logger::logging(
        HartreeFock::LogLevel::Info,
        "PCM :",
        std::format(
            "C-PCM cavity with {} surface points, epsilon = {:.4f}, scale = {:.3f}",
            state.surface_points.size(),
            epsilon,
            calculator._solvation._cavity_scale));
    if (!calculator._solvation._solvent.empty())
    {
        HartreeFock::Logger::logging(
            HartreeFock::LogLevel::Info,
            "PCM Solvent :",
            calculator._solvation._solvent);
    }
    HartreeFock::Logger::blank();

    return state;
}

// Per-SCF-iteration density-dependent step. Called from run_rhf, run_uhf, and
// the KS-DFT driver inside the SCF loop. The total_density argument is the
// AO-basis density that drives the electronic potential — for UHF/UKS this
// must be P_alpha + P_beta (the spin densities cancel in the reaction field
// because V_rxn is built from the total electron density, not the spin density).
std::expected<HartreeFock::Solvation::PCMResult, std::string>
HartreeFock::Solvation::evaluate_pcm_reaction_field(
    const HartreeFock::Calculator &calculator,
    const PCMState &state,
    const Eigen::MatrixXd &total_density)
{
    PCMResult result;
    // Solvation off → hand back a zero reaction field so the caller can
    // unconditionally do F = F_gas + V_rxn without branching.
    if (!state.enabled())
    {
        result.reaction_potential =
            Eigen::MatrixXd::Zero(calculator._shells.nbasis(), calculator._shells.nbasis());
        return result;
    }

    const Eigen::Index npoints = static_cast<Eigen::Index>(state.surface_points.size());

    // ── 1. phi_tot at each tessera = phi_nuc + phi_elec ──────────────────────
    // The electronic part is sum_munu P_munu * V^(i)_munu — a single Frobenius
    // product per tessera against the current density. The nuclear part is
    // already cached in state.nuclear_potential.
    //
    // Sign convention: _compute_external_charge_attraction returns a matrix
    // with the electron sign already folded in (it is built like a nuclear-
    // attraction integral, which is conventionally negative). So the
    // contraction P : V^(i) directly gives the electronic contribution to
    // phi_tot with the correct sign.
    result.total_potential = state.nuclear_potential;
    for (Eigen::Index i = 0; i < npoints; ++i)
        result.total_potential(i) +=
            (total_density.array() * state.unit_potential_matrices[static_cast<std::size_t>(i)].array()).sum();

    // ── 2. Solve D q = -f(eps) phi_tot ───────────────────────────────────────
    // D is symmetric and positive-definite (off-diagonals are 1/r couplings
    // between distinct tesserae; diagonals are the dominating ISWIG self-
    // energies), so an LDLT factorization is the natural choice.
    //
    // We re-factor D every iteration. It only depends on the cavity (not on
    // the density), so this could be cached across SCF iterations; we trade a
    // small amount of repeated work for code clarity.
    Eigen::LDLT<Eigen::MatrixXd> solver(state.influence_matrix);
    if (solver.info() != Eigen::Success)
        return std::unexpected("PCM surface linear system factorization failed");

    result.apparent_charges =
        solver.solve(-state.dielectric_factor * result.total_potential);
    if (solver.info() != Eigen::Success)
        return std::unexpected("PCM surface linear system solve failed");

    // ── 3. V_rxn_munu = sum_i q_i V^(i)_munu ─────────────────────────────────
    // A single weighted sum of the cached unit-charge matrices. This is the
    // operator that gets added to the Fock / Kohn-Sham matrix; it is what
    // makes the dielectric polarization "feed back" on the solute orbitals
    // at every SCF iteration.
    result.reaction_potential =
        Eigen::MatrixXd::Zero(calculator._shells.nbasis(), calculator._shells.nbasis());
    for (Eigen::Index i = 0; i < npoints; ++i)
        result.reaction_potential +=
            result.apparent_charges(i) * state.unit_potential_matrices[static_cast<std::size_t>(i)];

    // ── 4. Solvation energy contribution G_rxn = (1/2) q . phi_tot ───────────
    // The 1/2 is what prevents double counting when V_rxn is also added to F:
    // tr[P V_rxn] = q . phi_el (twice the electronic interaction with the
    // surface), so adding the full q . phi_tot to E would over-count. The
    // factor of 1/2 makes the total energy stationary at convergence.
    result.solvation_energy = 0.5 * result.apparent_charges.dot(result.total_potential);
    return result;
}
