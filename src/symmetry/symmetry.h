#ifndef HF_SYMMETRY_H
#define HF_SYMMETRY_H

#include <expected>
#include <cstdlib>

#include "external/libmsym/install/include/libmsym/msym.h"
#include "external/libmsym/install/include/libmsym/msym_EXPORTS.h"
#include "external/libmsym/install/include/libmsym/msym_error.h"
#include "base/types.h"

namespace HartreeFock
{
    namespace Symmetry
    {
        std::expected<void, std::string> detectSymmetry(HartreeFock::Molecule &molecule);
    }
}

#endif // !HF_SYMMETRY_H
