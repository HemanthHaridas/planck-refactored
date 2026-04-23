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
        std::expected<std::vector<std::string>, std::string> assign_vibrational_symmetry(
            const HartreeFock::Calculator &calc,
            const Eigen::MatrixXd &normal_modes);
    } // namespace Symmetry
} // namespace HartreeFock

#endif // HF_VIBRATIONAL_SYMMETRY_H
