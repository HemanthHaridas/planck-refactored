#ifndef HF_TABLES_H
#define HF_TABLES_H

#include <array>

// Length conversion factors
const double BOHR_TO_ANGSTROM = 0.529177210903;
const double ANGSTROM_TO_BOHR = 1.8897259886;
const double AU_TO_DEBYE = 2.541746473;

// Energy conversion factors
const double HARTREE_TO_EV = 27.211386245988;
const double EV_TO_HARTREE = (1.0 / 27.211386245988);
const double HARTREE_TO_KCALMOL = 627.5094740631;
const double HARTREE_TO_KJMOL = 2625.4996394799;

inline constexpr double GEOMOPT_LINE_SEARCH_MIN_ALPHA = 1.0e-6;
inline constexpr std::array<double, 4> CASSCF_PROBE_STEP_SCALES = {
    1.0, 0.5, 0.25, 0.125};
inline constexpr std::array<double, 7> CASSCF_MACRO_STEP_SCALES = {
    1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625};

#endif // !HF_TABLES_H
