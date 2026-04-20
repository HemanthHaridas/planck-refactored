#ifndef HF_CHECKPOINT_H
#define HF_CHECKPOINT_H

#include <expected>
#include <string>

#include "base/types.h"

// ─── Checkpoint I/O ───────────────────────────────────────────────────────────
//
// Binary checkpoint format (.hfchk).  Layout (in order, all little-endian):
//
//  [8]  magic: "PLNKCHK\0"
//  [4]  version: uint32 = 6
//  [8]  nbasis: uint64
//  [1]  is_uhf: uint8   (1 when the checkpoint carries separate alpha/beta
//                       spin channels; used for both UHF and ROHF)
//  [1]  is_converged: uint8
//  [4]  last_iter: uint32
//  [8]  total_energy: double
//  [8]  nuclear_repulsion: double
//  [8]  natoms: uint64
//  [4]  charge: int32
//  [4]  multiplicity: uint32
//  [natoms × 4]     atomic_numbers: int32[]
//  [natoms × 3 × 8] coordinates_bohr: double[] (row-major: x0y0z0 x1y1z1 ...)
//                    → always the final (possibly optimized) geometry in standard frame
//  [4 + len]        basis name: uint32 length + chars (no null terminator)
//  [1]              has_opt_coords: uint8 (1 if coordinates came from a converged geomopt)
//
//  Then for each spin channel (alpha always, beta only when is_uhf):
//    density matrix, fock matrix, mo_energies (as n×1), mo_coefficients
//
//  Each matrix block: [8 rows][8 cols][rows×cols×8 data in column-major order]
//
//  Version 3 appends:
//    [1]              has_casscf_mos: uint8
//    [matrix]         casscf_mo_coefficients (nbasis×nbasis) when flag is 1
//
//  Version 4 further appends (basis shell data for cube file generation):
//    [1]              has_basis: uint8 (always 1 in new saves)
//    [8]              nshells: uint64
//    For each shell:
//      [4]            shell_type: int32 (0=S 1=P 2=D 3=F 4=G 5=H)
//      [4]            nprim: uint32
//      [8×3]          center_x, center_y, center_z: double (Bohr, standard frame)
//      [nprim×8]      primitives: double[]     (exponents alpha_k)
//      [nprim×8]      coefficients: double[]   (c_k × Nc, contracted norm folded in)
//      [nprim×8]      normalizations: double[] (per-primitive N_k)
//    [8]              nbf: uint64 (= nbasis)
//    For each basis function (ContractedView):
//      [8]            shell_index: uint64 (index into the shells array above)
//      [4]            lx: int32
//      [4]            ly: int32
//      [4]            lz: int32
//      [8]            component_norm: double (1/sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!))
//
//  Version 5 further appends:
//    [1]              has_casscf_active_densities: uint8
//    [vector]         casscf_active_densities (active natural occupations) when flag is 1
//
//  Version 6 further appends:
//    [1]              has_casscf_active_orbitals: uint8
//    [4]              casscf_active_start: int32 (0-based MO index) when flag is 1
//    [4]              casscf_active_count: int32 when flag is 1
//
// Restart semantics:
//   guess density — load() fills _overlap, _hcore, _info._scf.{alpha,beta},
//     and _total_energy.  Geometry comes from the input file.
//   guess full    — load_geometry() is called first (before basis/symmetry setup)
//     to restore coordinates, charge, and multiplicity from the checkpoint.
//     load() is then called as normal to supply the density.

namespace HartreeFock
{
    namespace Checkpoint
    {
        // Write all post-convergence SCF data to a binary checkpoint file.
        std::expected<void, std::string> save(const HartreeFock::Calculator &calc,
                                              const std::string &path);

        // Read checkpoint data into an already-parsed Calculator.
        //
        // load_1e_matrices controls whether the stored overlap and core-Hamiltonian
        // matrices are applied to the calculator:
        //   true  (guess full)    — the geometry came from the checkpoint so the
        //                           stored 1e matrices are valid; loading them skips
        //                           integral recomputation.
        //   false (guess density) — geometry may differ from the checkpoint, so
        //                           1e matrices are skipped; the caller must compute
        //                           fresh integrals.
        //
        // On success, always fills: _info._scf (density/fock/MOs), _total_energy,
        //   _nuclear_repulsion.  Also fills _overlap and _hcore when load_1e_matrices.
        //
        // When restarting across restricted/open-shell references, the loader adapts
        // the stored density to the current target:
        //   RHF checkpoint -> UHF/ROHF : split the stored MO set into alpha/beta
        //                                spin densities using the current occupations.
        //   UHF/ROHF checkpoint -> RHF : combine alpha+beta densities into the
        //                                restricted density matrix.
        // Returns an error string if the file is missing, corrupt, or nbasis mismatches.
        std::expected<void, std::string> load(HartreeFock::Calculator &calc,
                                              const std::string &path,
                                              bool load_1e_matrices = true);

        // Geometry + molecule data extracted from a checkpoint.
        // Used by guess full before the basis/symmetry setup so that the
        // optimized geometry is used for all downstream steps.
        struct GeometryData
        {
            std::size_t natoms;
            int charge;
            unsigned int multiplicity;
            Eigen::VectorXi atomic_numbers;
            Eigen::MatrixXd coords_bohr; // natoms × 3, standard frame, Bohr
            bool has_opt_coords;         // true if from a converged geomopt
        };

        // Read only the molecule geometry block from the checkpoint.
        // Does NOT validate nbasis — intended for use before basis construction.
        // Returns an error if the file is missing, has bad magic/version, or the
        // stored atomic numbers differ from those in calc._molecule.
        std::expected<GeometryData, std::string>
        load_geometry(const std::string &path);

        // MO data extracted from a checkpoint, without nbasis validation.
        // Used for basis-set projection when the checkpoint was made in a smaller basis.
        struct MOData
        {
            std::size_t nbasis; // nbasis of the checkpoint's basis
            bool is_uhf;
            std::string basis_name;   // basis name stored in the checkpoint
            Eigen::MatrixXd C_alpha;  // all alpha MO columns (nbasis × nbasis)
            Eigen::MatrixXd C_beta;   // beta MOs if is_uhf
            Eigen::MatrixXd C_casscf; // converged CASSCF MOs if present
        };

        // Read MO coefficients from checkpoint without enforcing nbasis match.
        // Returns an error only if the file is missing or has bad magic/version.
        std::expected<MOData, std::string> load_mos(const std::string &path);

        // Project occupied MOs from a small basis onto the large basis.
        //
        //   X_large   : large-basis orthogonalizer X = S^{-1/2}  (nb_large × nb_large)
        //   S_cross   : cross-overlap S(μ^large, ν^small)         (nb_large × nb_small)
        //   C_occ     : occupied MOs in the small basis            (nb_small × n_occ)
        //   factor    : density-matrix occupation factor (2.0 for RHF, 1.0 per spin for UHF)
        //
        // Returns the projected density matrix P (nb_large × nb_large).
        // Uses SVD Löwdin projection to guarantee orthonormality in S_large metric.
        Eigen::MatrixXd project_density(const Eigen::MatrixXd &X_large,
                                        const Eigen::MatrixXd &S_cross,
                                        const Eigen::MatrixXd &C_occ,
                                        double factor);
    } // namespace Checkpoint
} // namespace HartreeFock

#endif // !HF_CHECKPOINT_H
