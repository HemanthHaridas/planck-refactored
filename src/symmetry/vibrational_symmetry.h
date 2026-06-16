#ifndef HF_VIBRATIONAL_SYMMETRY_H
#define HF_VIBRATIONAL_SYMMETRY_H

#include <Eigen/Dense>
#include <expected>
#include <string>
#include <vector>

#include "base/types.h"

namespace HartreeFock
{
    namespace Symmetry
    {
        // Assign Mulliken irrep labels to vibrational normal modes.
        // Returns one label per column of normal_modes, or an empty vector when
        // symmetry analysis is unavailable/unsupported for the current molecule.
        //
        // Important: for non-Abelian groups this routine intentionally classifies
        // modes in the largest Abelian all-1D subgroup, not the full point group.
        // That mirrors MO labeling: the returned labels are subgroup labels in
        // cases such as Td/Oh/D3d, even though other symmetry layers may use the
        // full group.
        std::expected<std::vector<std::string>, std::string> assign_vibrational_symmetry(
            const HartreeFock::Calculator &calc,
            const Eigen::MatrixXd &normal_modes);
    } // namespace Symmetry
} // namespace HartreeFock

#endif // HF_VIBRATIONAL_SYMMETRY_H
