#ifndef HF_IO_FCIDUMP_H
#define HF_IO_FCIDUMP_H

#include <expected>
#include <string>
#include <vector>

#include "base/types.h"
#include "integrals/shellpair.h"

namespace HartreeFock::IO
{

    // ── FCIDUMP export ──────────────────────────────────────────────────────────
    //
    // Write the molecular-orbital electronic Hamiltonian to a file in the standard
    // MOLPRO FCIDUMP format. This is the universal interchange format read by
    // external FCI / DMRG / selected-CI / FCIQMC solvers (PySCF, Block2, CheMPS2,
    // Dice/SHCI, NECI, ...), so an FCIDUMP lets those codes run a much larger
    // (near-)FCI on the Hamiltonian that Planck's SCF + integral engine produced
    // than Planck's own bit-packed determinant CI can reach.
    //
    // The file contains, in MO basis and Chemists' notation `(ij|kl)`:
    //   * a &FCI header (NORB, NELEC, MS2, ORBSYM, ISYM),
    //   * the unique two-electron integrals (8-fold permutational symmetry),
    //   * the unique one-electron core-Hamiltonian integrals h(i,j),
    //   * the scalar nuclear repulsion energy (indices 0 0 0 0).
    //
    // Requires a converged RHF or ROHF reference — both store one common spatial
    // orbital set in the alpha channel, which is what the single-orbital-set
    // FCIDUMP layout assumes.
    //
    // ORBSYM: when symmetry is active and the point group has only one-dimensional
    // (Abelian) irreps, each orbital's Mulliken irrep label is mapped to its
    // MOLPRO irrep number (the same numbering PySCF's fcidump tool uses). When
    // symmetry is unavailable or the group is not 1-D Abelian, every orbital is
    // labelled with irrep 1 (no symmetry), which every downstream solver accepts.
    //
    std::expected<void, std::string> write_fcidump(
        HartreeFock::Calculator &calc,
        const std::vector<HartreeFock::ShellPair> &shell_pairs,
        const std::string &path);

    // Map a list of Mulliken irrep labels (e.g. "A1", "B1g") to MOLPRO ORBSYM
    // numbers for the given point group. Returns an empty vector when the point
    // group is not a supported 1-D Abelian group, signalling the caller to fall
    // back to all-ones ORBSYM. Exposed for unit testing.
    std::vector<int> molpro_orbsym(
        const std::string &point_group,
        const std::vector<std::string> &mo_labels);

} // namespace HartreeFock::IO

#endif // HF_IO_FCIDUMP_H
