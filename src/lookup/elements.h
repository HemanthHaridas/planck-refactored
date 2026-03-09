#ifndef HF_ELEMENTS_H
#define HF_ELEMENTS_H

#include <array>
#include <string_view>
#include <cstdint>

struct ElementData
{
    std::string_view symbol;   // Chemical symbol (e.g. "C")
    std::uint64_t    Z;        // Atomic number
    double           mass;     // Atomic mass (amu)
    double           radius;   // Covalent radius (Angstrom)
};

extern const std::array<ElementData, 99> periodic_table;

const ElementData& element_from_symbol(std::string_view symbol);
const ElementData& element_from_z(std::uint64_t Z);

#endif // !HF_ELEMENTS_H
