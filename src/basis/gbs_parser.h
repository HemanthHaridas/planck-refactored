#ifndef HF_BASIS_GBS_PARSER_H
#define HF_BASIS_GBS_PARSER_H

// Private header — Gaussian94 (.gbs) parser shared between the orbital basis
// loader (gaussian.cpp) and the RI auxiliary basis loader (rifit.cpp).
// The on-disk format is identical for orbital and aux basis sets, so the
// element→shells→primitives reader is the only thing that needs to be shared.
// Everything downstream (Shell construction, normalization, AO indexing,
// spherical transform) differs between the two roles and stays in the
// respective .cpp files.

#include <expected>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace HartreeFock::BasisFunctions::detail
{
    struct GbsPrimitive
    {
        double exponent;
        double coefficient;
    };

    struct GbsShell
    {
        std::string label; // "S", "P", "D", ...
        std::vector<GbsPrimitive> primitives;
    };

    using GbsBasisSet = std::unordered_map<std::string, std::vector<GbsShell>>;

    // Parse a Gaussian94 (.gbs) stream into element-keyed shell descriptors.
    // Handles the SP "fused" shell type by splitting into separate S and P shells.
    // Returns the parsed element→shells map on success or a human-readable error.
    std::expected<GbsBasisSet, std::string> read_gbs(std::ifstream &input);
} // namespace HartreeFock::BasisFunctions::detail

#endif // HF_BASIS_GBS_PARSER_H
