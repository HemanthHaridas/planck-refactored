#ifndef HF_ELEMENTS_H
#define HF_ELEMENTS_H

#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

struct ElementData
{
    std::string_view symbol; // Chemical symbol (e.g. "C")
    std::uint64_t Z;         // Atomic number
    double mass;             // Atomic mass (amu)
    // Van der Waals radius (Angstrom) -- NOT covalent. Verified against
    // PySCF's Bondi table: H 1.20, C 1.70, O 1.52 agree exactly, where the
    // covalent values would be ~0.31 / 0.73 / 0.66. Used by the PCM cavity
    // (src/solvation/pcm.cpp), which scales it by cavity_scale, and by any
    // ESP-derived charge fitting that needs an exclusion shell.
    double radius;
};

extern const std::array<ElementData, 99> planck_periodic_table;

std::expected<ElementData, std::string> element_from_symbol(std::string_view symbol);
std::expected<ElementData, std::string> element_from_z(std::uint64_t Z);

#endif // !HF_ELEMENTS_H
