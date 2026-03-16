#ifndef HF_INTCOORDS_H
#define HF_INTCOORDS_H

#include <vector>
#include <array>
#include <Eigen/Core>
#include "base/types.h"

namespace HartreeFock
{
    namespace Opt
    {
        enum class ICType { Stretch, Bend, Torsion };

        // ── Single primitive internal coordinate ─────────────────────────────
        //
        // atoms[0..3] hold the atom indices (unused slots = -1).
        //   Stretch:  atoms[0], atoms[1]
        //   Bend:     atoms[0] (A), atoms[1] (B, central), atoms[2] (C)
        //   Torsion:  atoms[0] (A), atoms[1] (B), atoms[2] (C), atoms[3] (D)
        //
        // Values: Bohr for stretches, radians for bends/torsions.
        struct InternalCoord
        {
            ICType            type;
            std::array<int,4> atoms = {-1,-1,-1,-1};

            double          value(const Eigen::MatrixXd& xyz) const;
            Eigen::VectorXd brow (const Eigen::MatrixXd& xyz, int natoms) const;
        };

        // ── Redundant IC system ───────────────────────────────────────────────
        struct IntCoordSystem
        {
            std::vector<InternalCoord> coords;
            int natoms = 0;

            int nics() const noexcept { return static_cast<int>(coords.size()); }

            // Build a redundant GIC set from Bohr Cartesians + atomic numbers.
            // Adds all bonds, all valence bends, and all proper torsions,
            // skipping angles > 175° (nearly linear).
            static IntCoordSystem build(const Eigen::MatrixXd& xyz_bohr,
                                        const Eigen::VectorXi& Z);

            // Insert ic into the system if an equivalent coordinate is not already
            // present (checks both forward and reverse orderings).  Returns the
            // 0-based index of the (possibly pre-existing) coordinate.
            int add_coord(const InternalCoord& ic);

            // Evaluate all IC values.
            Eigen::VectorXd values(const Eigen::MatrixXd& xyz) const;

            // Wilson B-matrix  (nics × 3*natoms).
            Eigen::MatrixXd bmatrix(const Eigen::MatrixXd& xyz) const;

            // IC gradient from Cartesian gradient:  g_q = G⁺ B g_x
            // where G = B Bᵀ (metric) and G⁺ is its Moore-Penrose inverse.
            Eigen::VectorXd cart_to_ic_grad(const Eigen::MatrixXd& xyz,
                                             const Eigen::VectorXd& g_cart) const;

            // Back-transform an IC displacement Δq to a Cartesian step via
            // microiterations (Schlegel 1984).  Returns the new xyz matrix.
            Eigen::MatrixXd ic_to_cart_step(const Eigen::MatrixXd& xyz0,
                                             const Eigen::VectorXd& dq,
                                             int max_iter = 25) const;
        };

    } // namespace Opt
} // namespace HartreeFock

#endif // HF_INTCOORDS_H
