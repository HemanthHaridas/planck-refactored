#ifndef HF_BSSE_COUNTERPOISE_H
#define HF_BSSE_COUNTERPOISE_H

#include <expected>
#include <string>

#include "base/types.h"

namespace HartreeFock
{
    namespace BSSE
    {
        // Result of a Boys–Bernardi counterpoise calculation for a dimer A···B.
        // All energies are SCF total energies in Hartree.
        struct CounterpoiseResult
        {
            double e_dimer = 0.0;     // E(AB) in the dimer basis
            double e_mono_a = 0.0;    // E(A) in A's own (monomer) basis
            double e_mono_b = 0.0;    // E(B) in B's own (monomer) basis
            double e_mono_a_cp = 0.0; // E(A)* in the dimer basis (B ghosted)
            double e_mono_b_cp = 0.0; // E(B)* in the dimer basis (A ghosted)

            // Derived quantities (filled by run_counterpoise).
            double bsse = 0.0;            // [E(A)*-E(A)] + [E(B)*-E(B)]  (<= 0)
            double interaction_raw = 0.0; // E(AB) - E(A) - E(B)
            double interaction_cp = 0.0;  // E(AB) - E(A)* - E(B)*  (= raw - bsse)
        };

        // Run the full counterpoise procedure described by calculator._bsse.
        // The parent calculator must already be parsed (molecule, basis, scf
        // options); this function builds and runs the five SCF sub-calculations
        // internally, prints a report, and returns the energy decomposition.
        // SCF-level only (RHF/UHF/ROHF); errors propagate as strings.
        std::expected<CounterpoiseResult, std::string>
        run_counterpoise(const HartreeFock::Calculator &parent);
    } // namespace BSSE
} // namespace HartreeFock

#endif // !HF_BSSE_COUNTERPOISE_H
