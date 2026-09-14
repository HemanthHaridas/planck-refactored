#ifndef HF_POPULATIONS_ESP_H
#define HF_POPULATIONS_ESP_H

#include <Eigen/Core>

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::SCF
{
    // Molecular electrostatic potential at an arbitrary list of points, in
    // atomic units (Hartree/e):
    //
    //   phi(r) = sum_A Z_A / |r - R_A|  -  sum_munu P_munu <mu| 1/|r - r'| |nu>
    //
    // Points are in Bohr. `density` is the TOTAL density (P_alpha + P_beta for
    // unrestricted), matching mulliken_population_analysis's convention.
    //
    // The electronic term is the one-electron potential integral the integral
    // engine already provides; see ObaraSaika::_compute_electronic_potential
    // (src/integrals/os.h) for why it is a fused sweep rather than a loop over
    // _compute_external_charge_attraction.
    std::expected<Eigen::VectorXd, std::string> electrostatic_potential(
        const Molecule &molecule,
        const std::vector<ShellPair> &shell_pairs,
        const Eigen::MatrixXd &density,
        const std::vector<Eigen::Vector3d> &points);

    // CHELPG evaluation grid (Breneman & Wiberg 1990): a cubic lattice over the
    // molecule's bounding box plus `headspace`, keeping only points that lie
    // OUTSIDE every atom's (scaled) van der Waals radius but within `headspace`
    // of at least one atom. The first test excludes the region where the
    // potential is dominated by the density itself and a point-charge model is
    // meaningless; the second stops the box corners contributing points so far
    // out that they carry no information about the molecule.
    //
    // Returns points in BOHR, in the same frame as molecule._standard, so the
    // result feeds electrostatic_potential directly.
    //
    // `radius_scale` multiplies the tabulated vdW radius, mirroring
    // OptionsSolvation::_cavity_scale. Note ElementData::radius is a
    // cutoff-sense vdW radius, not the leading-edge-slope kind (see
    // docs/ESP_CHARGES.md section 4), so reproducing another code's
    // published CHELPG numbers may need this knob.
    std::expected<std::vector<Eigen::Vector3d>, std::string> chelpg_grid(
        const Molecule &molecule,
        double spacing,
        double headspace,
        double radius_scale = 1.0);

    // Merz-Kollman / Connolly sampling shells: for each atom, `points_per_shell`
    // Fibonacci-sphere points on a sphere of radius `scale * r_vdW` for every
    // scale in `shell_scales`, discarding any point buried inside another
    // atom's sphere at that same scale.
    //
    // Unlike the CHELPG cubic lattice, this construction is ROTATIONALLY
    // SYMMETRIC: rotating the molecule rotates the sample set with it rather
    // than resampling a fixed lattice, so the fitted charges do not depend on
    // the molecule's orientation in the lab frame. chelpg_grid has a measured
    // 3-7% orientation dependence that does not converge with spacing (see
    // docs/ESP_CHARGES.md); this is the fix, and it is also what RESP
    // conventionally samples on.
    //
    // Returns points in BOHR, in the molecule._standard frame.
    std::expected<std::vector<Eigen::Vector3d>, std::string> connolly_grid(
        const Molecule &molecule,
        const std::vector<double> &shell_scales,
        int points_per_shell,
        double radius_scale = 1.0);

    struct ESPChargeFit
    {
        Eigen::VectorXd charges;  // one per atom, in e
        double rrms = 0.0;        // relative root-mean-square fit error
        double rms = 0.0;         // absolute RMS residual, a.u.
        std::size_t n_points = 0; // grid points actually used
    };

    // Least-squares atomic charges reproducing `potential` at `points`, subject
    // to sum(q) == total_charge exactly.
    //
    // The constraint enters as a Lagrange multiplier rather than a penalty, so
    // it is satisfied to solver precision instead of approximately. That makes
    // the system symmetric-indefinite of size (natoms+1), hence LDL^T -- the
    // same choice geomopt.cpp:662 makes for its own indefinite system.
    //
    // `rrms` is the fit error relative to the RMS of the potential being fitted,
    // which is the quantity worth gating: the charges alone always look
    // plausible because the constraint forces them to sum correctly, whether or
    // not they reproduce anything.
    std::expected<ESPChargeFit, std::string> fit_esp_charges(
        const Molecule &molecule,
        const std::vector<Eigen::Vector3d> &points,
        const Eigen::VectorXd &potential,
        double total_charge);

    struct RESPOptions
    {
        // Bayly et al. (1993) stage-1 values. `a` sets how hard poorly
        // determined charges are pulled toward zero; `b` is the width of the
        // flat-bottomed region, below which the restraint is effectively
        // quadratic and above which it is effectively linear.
        double strength = 0.0005; // a, atomic units
        double tightness = 0.1;   // b, atomic units

        // Atoms exempt from the restraint. Hydrogens are conventionally NOT
        // restrained, because the restraint exists to tame buried heavy atoms
        // whose charges the ESP barely constrains, and hydrogens are always on
        // the surface where the data is good.
        bool exempt_hydrogen = true;

        // Groups of atom indices forced to share one charge (0-based). Used for
        // symmetry-equivalent atoms that the fit would otherwise give slightly
        // different charges to purely because the grid samples them unevenly.
        std::vector<std::vector<std::size_t>> equivalence_groups;

        int max_iterations = 50;
        double convergence = 1e-10; // max |dq| between iterations
    };

    struct RESPChargeFit
    {
        ESPChargeFit fit;
        int iterations = 0;
        bool converged = false;
    };

    // RESP: the same constrained least-squares problem as fit_esp_charges, plus
    // a hyperbolic restraint a*sum(sqrt(q^2 + b^2) - b) pulling charges toward
    // zero.
    //
    // The restraint is solved by iterating the SAME (natoms+1) LDL^T system,
    // not by a new solver: its derivative contributes a diagonal
    // a/sqrt(q_k^2 + b^2) to the normal matrix, which depends on q and so is
    // refreshed each pass until the charges stop moving. Typically ~10-25
    // iterations.
    //
    // Why it exists: an atom with little grid nearby (a buried carbon) has
    // almost no leverage on the potential, so the unrestrained fit is free to
    // give it a large charge cancelled by its neighbours. Those charges fit the
    // ESP but transfer badly and behave poorly in dynamics. The restraint
    // removes that freedom without meaningfully degrading the fit.
    //
    // Stage 2 (methyl/methylene refitting) is deliberately not implemented: it
    // exists for AMBER compatibility specifically, and `equivalence_groups`
    // already covers the general case of forcing atoms to share a charge.
    std::expected<RESPChargeFit, std::string> fit_resp_charges(
        const Molecule &molecule,
        const std::vector<Eigen::Vector3d> &points,
        const Eigen::VectorXd &potential,
        double total_charge,
        const RESPOptions &options = {});
} // namespace HartreeFock::SCF

#endif // HF_POPULATIONS_ESP_H
