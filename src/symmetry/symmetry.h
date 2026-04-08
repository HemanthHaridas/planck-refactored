#ifndef HF_SYMMETRY_H
#define HF_SYMMETRY_H

#include <cstdlib>
#include <expected>

#include "base/types.h"
#include "external/libmsym/install/include/libmsym/msym.h"
#include "external/libmsym/install/include/libmsym/msym_EXPORTS.h"
#include "external/libmsym/install/include/libmsym/msym_error.h"

namespace HartreeFock
{
    namespace Symmetry
    {
        std::expected<void, std::string> detectSymmetry(HartreeFock::Molecule &molecule);
    }
} // namespace HartreeFock

#endif // !HF_SYMMETRY_H
