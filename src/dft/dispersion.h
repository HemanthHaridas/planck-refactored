#ifndef PLANCK_DFT_DISPERSION_H
#define PLANCK_DFT_DISPERSION_H

#include <expected>
#include <string>

#include <Eigen/Dense>

#include "base/types.h"

// Grimme D3 dispersion through simple-dftd3 (docs/DISPERSION_D3.md). A function
// of the nuclear geometry only, so it enters exactly where nuclear repulsion
// does: one energy in the KS-SCF total and one gradient in the KS gradient.
namespace DFT::Dispersion
{
    struct Result
    {
        double energy = 0.0;      // Eh
        Eigen::MatrixXd gradient; // natoms x 3, Eh/Bohr, frame of _molecule._standard
    };

    // Zero energy and gradient when `dispersion none`. Ghost atoms carry no
    // dispersion (zero gradient rows). Needs resolved _exchange_id/_correlation_id.
    std::expected<Result, std::string> compute(const HartreeFock::Calculator &calculator);

    // simple-dftd3 parameter-set name: dispersion_functional if set, else
    // derived from the resolved libxc functionals.
    std::expected<std::string, std::string> parameter_set_name(const HartreeFock::OptionsDFT &dft);
} // namespace DFT::Dispersion

#endif
