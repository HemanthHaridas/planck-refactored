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
    // docs/ESP_CHARGES_SCOPE.md section 3), so reproducing another code's
    // published CHELPG numbers may need this knob.
    std::expected<std::vector<Eigen::Vector3d>, std::string> chelpg_grid(
        const Molecule &molecule,
        double spacing,
        double headspace,
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
} // namespace HartreeFock::SCF

#endif // HF_POPULATIONS_ESP_H
