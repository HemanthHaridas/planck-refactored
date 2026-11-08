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
//      D q = -f(eps) phi_tot,        f(eps) = (eps - 1) / eps,
//
// where D is the symmetric "surface influence matrix" of Coulomb couplings
// between tesserae and phi_tot is the total electrostatic potential the solute
// produces at each tessera (nuclei + electrons). The reaction-field potential
// felt by the solute electrons is then
//
//      phi_rxn(r) = sum_i q_i erf(xi_i |r - s_i|) / |r - s_i|
//
// (each q_i is a normalized Gaussian of width xi_i: the SWIG cavity, see
// pcm.cpp) and the matching one-electron operator added to the Fock /
// Kohn-Sham matrix is V_rxn_munu = sum_i q_i V^(i)_munu, where V^(i) is the
// AO matrix of a unit Gaussian charge at s_i. The reaction-field contribution to the energy is
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

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::Solvation
{
    // One tessera of the cavity surface. The cavity is built once per geometry;
    // every tessera property is smooth in the nuclear coordinates (SWIG).
    struct PCMSurfacePoint
    {
        Eigen::Vector3d position = Eigen::Vector3d::Zero(); // s_i, in Bohr
        double area = 0.0;                                  // a_i = w_i R^2 F_i, in Bohr^2
        double exponent = 0.0;                              // Gaussian width xi_i, in 1/Bohr
        double switching = 1.0;                             // SWIG switching value F_i in (0, 1]
        std::size_t atom_index = 0;                         // owning atom (for diagnostics)
    };

    // All quantities that depend on the cavity geometry but NOT on the density.
    // Built once by build_pcm_state(); re-used at every SCF iteration.
    struct PCMState
    {
        std::vector<PCMSurfacePoint> surface_points;

        // Nuclear contribution to phi_tot at each tessera: phi_nuc_i = sum_A Z_A / |s_i - R_A|.
        // Density-independent, cached up front.
        Eigen::VectorXd nuclear_potential;

        // LDLT factor of S, the symmetric positive-definite Coulomb matrix of
        // the Gaussian tessera charges (Lange-Herbert: erf(xi_ij r_ij)/r_ij,
        // diagonal xi_i sqrt(2/pi) / F_i). Factored once per geometry; each
        // SCF iteration only back-substitutes.
        Eigen::LDLT<Eigen::MatrixXd> influence_factor;

        // f(epsilon) = (epsilon - 1) / epsilon, the C-PCM scaling —
        // interpolates between vacuum (eps=1, f=0) and a perfect conductor
        // (eps -> inf, f -> 1).
        double dielectric_factor = 0.0;

        // Per-atom SWIG switching radii R_in,J and R_sw,J (Bohr). The cavity
        // build uses them for F_i; the gradient needs them again for dF_i/dR.
        std::vector<double> switch_inner;
        std::vector<double> switch_width;

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

    // One-shot setup: build the cavity, factor the influence matrix and
    // compute the nuclear potential at each tessera. Returns an empty (disabled) state when SolvationModel::None is
    // requested. Errors out if surface_points_per_atom is not a supported
    // Lebedev size or if every candidate surface point is buried.
    std::expected<PCMState, std::string> build_pcm_state(
        const HartreeFock::Calculator &calculator);

    // Per-iteration evaluation: given the current total density (P_alpha + P_beta
    // for unrestricted, or P for restricted), assemble phi_tot, solve the C-PCM
    // linear system for the apparent charges, build V_rxn, and return G_rxn.
    // phi_el and V_rxn are each one fused sweep over the shell pairs, so no
    // per-tessera AO matrix is stored: memory is O(n_tess^2) for the factor.
    // When the state is disabled (no PCM), returns a zero reaction-field matrix
    // and zero energy so the SCF caller does not need to special-case the
    // gas-phase branch.
    std::expected<PCMResult, std::string> evaluate_pcm_reaction_field(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const PCMState &state,
        const Eigen::MatrixXd &total_density);

    // S_ij = erf(xi_ij r_ij)/r_ij, S_ii = xi_i sqrt(2/pi)/F_i for the given
    // tesserae: the matrix build_pcm_state factors.
    Eigen::MatrixXd pcm_influence_matrix(const std::vector<PCMSurfacePoint> &points);

    // The two cavity-geometry terms of dG_rxn/dR at fixed apparent charges q
    // (natoms x 3, Hartree/Bohr; docs/PCM_GEOMETRY_SCOPE.md step 4). Every
    // tessera moves rigidly with its owning atom, and xi_i is geometry-free.
    //   grad_nuc:    q . dphi_nuc, phi_nuc_i = sum_J Z_J erf(xi_i r_iJ)/r_iJ
    //   grad_solver: q^T dS q / (2 f), through the tessera positions off the
    //                diagonal and through F_i on it
    // The state must have been built at the molecule's current geometry.
    Eigen::MatrixXd pcm_grad_nuc(const HartreeFock::Molecule &molecule,
                                 const PCMState &state, const Eigen::VectorXd &q);
    Eigen::MatrixXd pcm_grad_solver(const HartreeFock::Molecule &molecule,
                                    const PCMState &state, const Eigen::VectorXd &q);

    // The integral term of dG_rxn/dR: q . dphi_el at fixed density and q,
    // through the AO centres (ao_atom[mu], Cartesian AO index) and the
    // tessera centres. total_density is in the working basis, as for
    // evaluate_pcm_reaction_field.
    std::expected<Eigen::MatrixXd, std::string> pcm_grad_qv(
        const HartreeFock::Calculator &calculator,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const PCMState &state,
        const Eigen::MatrixXd &total_density,
        const Eigen::VectorXd &q,
        const std::vector<int> &ao_atom);

} // namespace HartreeFock::Solvation

#endif // PLANCK_SOLVATION_PCM_H
