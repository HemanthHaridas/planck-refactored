#ifndef PLANCK_SOLVATION_PCM_H
#define PLANCK_SOLVATION_PCM_H

// ─────────────────────────────────────────────────────────────────────────────
// Conductor-like polarizable continuum model (C-PCM).
//
// The solute lives inside a molecule-shaped cavity carved out of a structureless
// dielectric of relative permittivity epsilon. The dielectric polarizes in
// response to the solute charge density, and that polarization is represented
// by an apparent surface charge layer q_i living on a tessellated cavity
// boundary. Each tessera i has position s_i and area a_i.
//
// In the C-PCM (conductor-like) approximation the surface integral equation
// reduces to a single dense linear system
//
//      D q = -f(eps) phi_tot,        f(eps) = (eps - 1) / (eps + 1/2),
//
// where D is the symmetric "surface influence matrix" of Coulomb couplings
// between tesserae and phi_tot is the total electrostatic potential the solute
// produces at each tessera (nuclei + electrons). The reaction-field potential
// felt by the solute electrons is then
//
//      phi_rxn(r) = sum_i q_i / |r - s_i|
//
// and the matching one-electron operator added to the Fock / Kohn-Sham matrix
// is V_rxn_munu = sum_i q_i V^(i)_munu, where V^(i) is the AO matrix of a
// unit point charge at s_i. The reaction-field contribution to the energy is
//
//      G_rxn = (1/2) q . phi_tot,
//
// where the 1/2 prevents double counting against V_rxn already living inside
// the Fock matrix.
//
// Only the C-PCM dielectric form is implemented here. IEF-PCM, SS(V)PE,
// analytic gradients, and PCM coupling to post-HF correlation are not.
// ─────────────────────────────────────────────────────────────────────────────

#include <expected>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Solvation
{
    // One tessera of the cavity surface. The cavity is built once at the start
    // of the calculation and never re-tessellated (PCM is single-point only).
    struct PCMSurfacePoint
    {
        Eigen::Vector3d position = Eigen::Vector3d::Zero(); // s_i, in Bohr
        double area = 0.0;                                  // a_i, in Bohr^2
        std::size_t atom_index = 0;                         // owning atom (for diagnostics)
    };

    // All quantities that depend on the cavity geometry but NOT on the density.
    // Built once by build_pcm_state(); re-used at every SCF iteration.
    struct PCMState
    {
        std::vector<PCMSurfacePoint> surface_points;

        // unit_potential_matrices[i] is the AO matrix of a unit point charge
        // placed at surface_points[i].position. Storing one matrix per tessera
        // up front means the only density-dependent surface work per SCF
        // iteration is a Frobenius product against the current density and a
        // linear combination to assemble V_rxn — no shell-pair integrals are
        // re-evaluated inside the SCF loop.
        std::vector<Eigen::MatrixXd> unit_potential_matrices;

        // Nuclear contribution to phi_tot at each tessera: phi_nuc_i = sum_A Z_A / |s_i - R_A|.
        // Density-independent, cached up front.
        Eigen::VectorXd nuclear_potential;

        // Symmetric, positive-definite Coulomb interaction matrix between tesserae
        // (off-diagonals 1/|s_i - s_j|, diagonals via the ISWIG self-energy
        // prescription). Solved every SCF iteration to obtain the apparent
        // charges; could in principle be factored once and re-solved, but the
        // current implementation re-factors it for clarity.
        Eigen::MatrixXd influence_matrix;

        // f(epsilon) = (epsilon - 1) / (epsilon + 1/2). Klamt-Schuurmann
        // C-PCM scaling — interpolates between vacuum (eps=1, f=0) and a
        // perfect conductor (eps -> inf, f -> 1).
        double dielectric_factor = 0.0;

        // True iff the cavity actually has tesserae (i.e. PCM is requested
        // and not all surface points were buried). Used by callers to skip
        // the reaction-field branch entirely when solvation is off.
        bool enabled() const noexcept
        {
            return !surface_points.empty();
        }
    };

    // Per-iteration outputs: the apparent charges that solve the C-PCM linear
    // system for the current density, the reaction-field operator that gets
    // added to the Fock / Kohn-Sham matrix, and the matching solvation energy.
    struct PCMResult
    {
        Eigen::VectorXd total_potential;    // phi_tot at each tessera (nuc + el)
        Eigen::VectorXd apparent_charges;   // q_i solving D q = -f(eps) phi_tot
        Eigen::MatrixXd reaction_potential; // V_rxn in the AO basis
        double solvation_energy = 0.0;      // G_rxn = (1/2) q . phi_tot
    };

    // One-shot setup: build the cavity, precompute the influence matrix, the
    // nuclear potential at each tessera, and one unit-charge AO matrix per
    // tessera. Returns an empty (disabled) state when SolvationModel::None is
    // requested. Errors out if surface_points_per_atom < 6 or if every
    // candidate surface point is buried inside another atom's sphere.
    std::expected<PCMState, std::string> build_pcm_state(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs);

    // Per-iteration evaluation: given the current total density (P_alpha + P_beta
    // for unrestricted, or P for restricted), assemble phi_tot, solve the C-PCM
    // linear system for the apparent charges, build V_rxn, and return G_rxn.
    // When the state is disabled (no PCM), returns a zero reaction-field matrix
    // and zero energy so the SCF caller does not need to special-case the
    // gas-phase branch.
    std::expected<PCMResult, std::string> evaluate_pcm_reaction_field(
        const HartreeFock::Calculator &calculator,
        const PCMState &state,
        const Eigen::MatrixXd &total_density);

} // namespace HartreeFock::Solvation

#endif // PLANCK_SOLVATION_PCM_H
