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
#include <map>
#include <numbers>

#include <Eigen/Cholesky>

#include "dft/base/angular.h"
#include "integrals/base.h"
#include "io/logging.h"
#include "lookup/elements.h"

namespace
{
    // zeta_N of the SWIG Gaussian width xi_i = zeta_N / (R sqrt(w_i)), per
    // Lebedev size (York and Karplus, J. Phys. Chem. A 103, 11060 (1999),
    // Table 1, as tabulated in PySCF solvent/pcm.py). A Lebedev size missing
    // here (74, 230, 266) has no published zeta and is rejected.
    const std::map<int, double> SWIG_ZETA = {
        {6, 4.84566077868}, {14, 4.86458714334}, {26, 4.85478226219},
        {38, 4.90105812685}, {50, 4.89250673295}, {86, 4.89741372580},
        {110, 4.90101060987}, {146, 4.89825187392}, {170, 4.90685517725},
        {194, 4.90337644248}, {302, 4.90498088169}, {350, 4.86879474832},
        {434, 4.90567349080}, {590, 4.90624071359}, {770, 4.90656435779},
        {974, 4.90685167998}, {1202, 4.90704098216}, {1454, 4.90721023869},
        {1730, 4.90733270691}, {2030, 4.90744499142}, {2354, 4.90753082825},
        {2702, 4.90760972766}, {3074, 4.90767282394}, {3470, 4.90773141371},
        {3890, 4.90777965981}, {4334, 4.90782469526}, {4802, 4.90749125553},
        {5294, 4.90762073452}, {5810, 4.90792902522}};

    // Smoothstep h(x) = x^3 (10 - 15x + 6x^2), clamped to [0, 1]
    // (Lange and Herbert, J. Chem. Phys. 133, 244111 (2010), Eq. 3.19).
    double switch_h(double x)
    {
        if (x <= 0.0)
            return 0.0;
        if (x >= 1.0)
            return 1.0;
        return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x);
    }
} // namespace

// One-shot cavity + density-independent precomputation. After this returns,
// PCMState contains everything the per-iteration evaluator needs.
//
// SWIG discretization (Lange and Herbert 2010), the scheme PySCF's C-PCM uses
// by default, so the two agree tightly:
//   - each atom J carries a sphere of radius R_J = cavity_scale * R_J^vdW
//     (modified Bondi: Bondi with H = 1.1 A), sampled by an N-point Lebedev
//     grid with weights w_i (sum 4 pi);
//   - a point i on atom A's sphere is faded out as it is buried in another
//     sphere J by F_i = prod_{J != A} h((r_iJ - R_in,J) / R_sw,J), with
//     R_sw = R sqrt(14/N), alpha = 1/2 + R/R_sw - sqrt((R/R_sw)^2 - 1/28),
//     R_in = R - alpha R_sw. F is smooth in the geometry, so tesserae fade
//     in and out instead of appearing and vanishing;
//   - each tessera carries a normalized Gaussian charge of width
//     xi_i = zeta_N / (R sqrt(w_i)), which makes every Coulomb element finite
//     (erf(xi_ij r)/r between tesserae, erf(xi r)/r against the solute).
// The self term S_ii = xi_i sqrt(2/pi) / F_i diverges as F_i -> 0, so a
// fading tessera's charge goes smoothly to zero.
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

    const int ngrid = calculator._solvation._surface_points_per_atom;
    const auto zeta = SWIG_ZETA.find(ngrid);
    if (zeta == SWIG_ZETA.end())
        return std::unexpected(std::format(
            "PCM surface_points_per_atom {} is not a supported Lebedev size "
            "(6, 14, 26, 38, 50, 86, 110, 146, 170, 194, 302, 350, 434, 590, ...)",
            ngrid));
    const auto lebedev = DFT::MakeLebedevGrid(ngrid);
    if (!lebedev)
        return std::unexpected("PCM cavity setup failed: " + lebedev.error());

    const auto &molecule = calculator._molecule;
    const std::size_t natoms = molecule.natoms;

    std::vector<double> radius(natoms), r_sw(natoms), r_in(natoms);
    for (std::size_t atom = 0; atom < natoms; ++atom)
    {
        const auto element = element_from_z(static_cast<std::uint64_t>(
            molecule.atomic_numbers(static_cast<Eigen::Index>(atom))));
        if (!element)
            return std::unexpected("PCM cavity setup failed: " + element.error());

        // Modified Bondi: H is 1.1 A for continuum cavities (Rowland and
        // Taylor), not the 1.2 A of the general vdW table.
        const double vdw = element->Z == 1 ? 1.1 : element->radius;
        radius[atom] = calculator._solvation._cavity_scale * vdw * ANGSTROM_TO_BOHR;
        r_sw[atom] = radius[atom] * std::sqrt(14.0 / ngrid);
        const double ratio = radius[atom] / r_sw[atom];
        r_in[atom] = radius[atom] - (0.5 + ratio - std::sqrt(ratio * ratio - 1.0 / 28.0)) * r_sw[atom];
    }

    for (std::size_t atom = 0; atom < natoms; ++atom)
    {
        const Eigen::Vector3d center = molecule._standard.row(static_cast<Eigen::Index>(atom));
        for (Eigen::Index g = 0; g < lebedev->rows(); ++g)
        {
            const Eigen::Vector3d direction = lebedev->row(g).head<3>().transpose();
            const double weight = (*lebedev)(g, 3);
            const Eigen::Vector3d point = center + radius[atom] * direction;

            double switching = 1.0;
            for (std::size_t other = 0; other < natoms && switching > 0.0; ++other)
            {
                if (other == atom)
                    continue;
                const Eigen::Vector3d other_center = molecule._standard.row(static_cast<Eigen::Index>(other));
                double x = ((point - other_center).norm() - r_in[other]) / r_sw[other];
                if (x < 1e-8) // PySCF's cutoff, kept so both prune the same points
                    x = 0.0;
                switching *= switch_h(x);
            }

            // Same pruning threshold as PySCF: a tessera whose weight has
            // faded below 1e-16 contributes nothing but a near-singular row.
            if (weight * switching <= 1e-16)
                continue;

            state.surface_points.push_back(PCMSurfacePoint{
                .position = point,
                .area = weight * radius[atom] * radius[atom] * switching,
                .exponent = zeta->second / (radius[atom] * std::sqrt(weight)),
                .switching = switching,
                .atom_index = atom});
        }
    }

    if (state.surface_points.empty())
        return std::unexpected("PCM cavity generation failed: all surface points were buried");

    // ── Density-independent precomputation ───────────────────────────────────
    // For each tessera i:
    //   - phi_nuc_i = sum_A Z_A erf(xi_i r_iA) / r_iA
    //   - V^(i): AO matrix of a unit Gaussian charge at s_i, which the SCF
    //     loop weights by the (density-dependent) apparent charge q_i
    //   - S_ij = erf(xi_ij r_ij) / r_ij, xi_ij = xi_i xi_j / sqrt(xi_i^2 + xi_j^2);
    //     S_ii = xi_i sqrt(2/pi) / F_i
    // ponytail: one nbasis^2 matrix per tessera; a fused sweep (as
    // _compute_electronic_potential) if PCM memory ever matters.
    const Eigen::Index npoints = static_cast<Eigen::Index>(state.surface_points.size());
    state.nuclear_potential = Eigen::VectorXd::Zero(npoints);
    state.influence_matrix = Eigen::MatrixXd::Zero(npoints, npoints);
    state.unit_potential_matrices.reserve(static_cast<std::size_t>(npoints));

    for (Eigen::Index i = 0; i < npoints; ++i)
    {
        const auto &site = state.surface_points[static_cast<std::size_t>(i)];

        for (std::size_t atom = 0; atom < natoms; ++atom)
        {
            const Eigen::Vector3d nucleus = molecule._standard.row(static_cast<Eigen::Index>(atom));
            const double distance = (site.position - nucleus).norm();
            state.nuclear_potential(i) +=
                molecule.nuclear_charge(atom) * std::erf(site.exponent * distance) / distance;
        }

        std::vector<HartreeFock::ExternalCharge> unit_charge{HartreeFock::ExternalCharge{
            .position = site.position, .charge = 1.0, .exponent = site.exponent * site.exponent}};
        state.unit_potential_matrices.push_back(
            _compute_external_charge_attraction(
                shell_pairs,
                calculator._shells.nbasis(),
                unit_charge,
                calculator._integral._engine,
                nullptr));

        state.influence_matrix(i, i) =
            site.exponent * std::sqrt(2.0 / std::numbers::pi) / site.switching;
        for (Eigen::Index j = 0; j < i; ++j)
        {
            const auto &other = state.surface_points[static_cast<std::size_t>(j)];
            const double xi_ij = site.exponent * other.exponent /
                                 std::hypot(site.exponent, other.exponent);
            const double distance = (site.position - other.position).norm();
            state.influence_matrix(i, j) = state.influence_matrix(j, i) =
                std::erf(xi_ij * distance) / distance;
        }
    }

    // C-PCM: f(eps) = (eps - 1) / eps (Barone and Cossi 1998; PySCF's
    // 'C-PCM'). Klamt's COSMO uses (eps - 1) / (eps + 1/2); this code used
    // that factor under the C-PCM name until the SWIG cavity landed.
    const double epsilon = calculator._solvation._dielectric;
    state.dielectric_factor = (epsilon - 1.0) / epsilon;

    HartreeFock::Logger::logging(
        HartreeFock::LogLevel::Info,
        "PCM :",
        std::format(
            "C-PCM (SWIG) cavity with {} surface points, epsilon = {:.4f}, scale = {:.3f}",
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
    // D is the symmetric positive-definite Coulomb matrix of the Gaussian
    // surface charges, so an LDLT factorization is the natural choice.
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
